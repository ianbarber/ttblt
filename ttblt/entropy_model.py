"""Byte-level entropy model for dynamic patching.

A small causal transformer that predicts the next byte, used to compute
per-byte entropies for placing patch boundaries at high-entropy positions
(word/morpheme boundaries).
"""

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import nn
import torch.nn.functional as F

from torchtune.modules import MultiHeadAttention, RMSNorm, TransformerSelfAttentionLayer
from torchtune.models.qwen2._component_builders import qwen2_mlp

from ttblt.bltqwen import VOCAB_SIZE, PAD_ID


# ---- Entropy Model ----------------------------------------------------------

class ByteEntropyModel(nn.Module):
    """Small causal byte-level transformer for computing per-byte entropies.

    ~14M params with default config.  Uses sliding window attention.
    Weight tying between input embedding and output projection.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = 384,
        num_layers: int = 6,
        num_heads: int = 6,
        head_dim: int = 64,
        hidden_dim: int = 1536,
        max_seq_len: int = 4096,
        sliding_window: int = 512,
        norm_eps: float = 1e-5,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.sliding_window = sliding_window
        self.vocab_size = vocab_size

        self.tok_embeddings = nn.Embedding(vocab_size, embed_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self_attn = MultiHeadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_kv_heads=num_heads,
                head_dim=head_dim,
                q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
                k_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
                v_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
                output_proj=nn.Linear(num_heads * head_dim, embed_dim, bias=False),
                max_seq_len=max_seq_len,
                attn_dropout=attn_dropout,
                is_causal=False,  # We pass our own sliding window mask
            )
            mlp = qwen2_mlp(dim=embed_dim, hidden_dim=hidden_dim)
            layer = TransformerSelfAttentionLayer(
                attn=self_attn,
                mlp=mlp,
                sa_norm=RMSNorm(embed_dim, eps=norm_eps),
                mlp_norm=RMSNorm(embed_dim, eps=norm_eps),
            )
            self.layers.append(layer)

        self.norm = RMSNorm(embed_dim, eps=norm_eps)

        # Output projection -- weight-tied with tok_embeddings
        self.output = nn.Linear(embed_dim, vocab_size, bias=False)
        self.output.weight = self.tok_embeddings.weight

        self._init_weights()

    def _init_weights(self, init_std: float = 0.02):
        nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=init_std)
        for layer_idx, layer in enumerate(self.layers):
            factor = 1.0 / math.sqrt(2 * (layer_idx + 1))
            std = init_std * factor
            with torch.no_grad():
                for p in layer.parameters():
                    nn.init.trunc_normal_(p, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def _make_sliding_window_mask(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """Causal sliding window mask.  [1, seq_len, seq_len] bool, True = attend.

        Shape [1, q, k] because torchtune adds head dim via mask[:, None, :, :].
        """
        positions = torch.arange(seq_len, device=device)
        diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # [i, j] = i - j
        mask = (diff >= 0) & (diff < self.sliding_window)
        return mask.unsqueeze(0)  # [1, seq_len, seq_len] — broadcasts over batch

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [batch, seq_len] byte IDs
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        seq_len = tokens.size(1)
        x = self.tok_embeddings(tokens)
        mask = self._make_sliding_window_mask(seq_len, tokens.device)

        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.norm(x)
        return self.output(x)


# ---- Entropy Computation ----------------------------------------------------

@torch.no_grad()
def compute_byte_entropies(
    model: ByteEntropyModel,
    byte_tokens: torch.Tensor,
) -> torch.Tensor:
    """Compute per-byte entropies from the entropy model's predictive distribution.

    Args:
        model: Trained ByteEntropyModel (should be in eval mode)
        byte_tokens: [batch, seq_len] byte token IDs

    Returns:
        entropies: [batch, seq_len] per-byte entropy in nats.
            Position 0 gets max entropy (always starts a new patch).
    """
    logits = model(byte_tokens)  # [batch, seq_len, vocab_size]

    # Entropy of the predictive distribution at each position.
    # logits[i] predicts token[i+1], so entropy[i] measures surprise of token[i+1].
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropies = -(probs * log_probs).sum(dim=-1)  # [batch, seq_len]

    # Position 0: always start a new patch
    entropies[:, 0] = entropies.max().item() + 1.0

    return entropies


# ---- Entropy -> Patch IDs ---------------------------------------------------

def entropy_to_patch_ids(
    entropies: torch.Tensor,
    threshold: float,
    max_patch_size: int = 0,
    pad_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Convert per-byte entropies to patch IDs.

    Following the BLT paper: a high-entropy position *ends* the current patch,
    and the *next* byte starts a new patch. Specifically, entropies[i] measures
    uncertainty about byte i+1, so when entropies[i] > threshold, byte i+1
    starts a new patch.

    Args:
        entropies: [batch, seq_len] per-position entropies from the entropy model
        threshold: entropy threshold for patch boundary (nats)
        max_patch_size: safety cap on patch size (0 = no cap, matching BLT paper default)
        pad_mask: [batch, seq_len] bool, True for padding. Padding continues last patch.

    Returns:
        patch_ids: [batch, seq_len] patch ID per byte
    """
    batch_size, seq_len = entropies.shape
    device = entropies.device

    patch_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    current_patch = torch.zeros(batch_size, dtype=torch.long, device=device)
    patch_len = torch.zeros(batch_size, dtype=torch.long, device=device)

    for pos in range(seq_len):
        # Check if we should start a new patch at this position.
        # A new patch starts when:
        #   1. The *previous* position had high entropy (entropies[pos-1] > threshold),
        #      meaning this byte was hard to predict → patch boundary.
        #   2. Current patch has hit max_patch_size (if set).
        if pos > 0:
            trigger = entropies[:, pos - 1] > threshold
            if max_patch_size > 0:
                trigger = trigger | (patch_len >= max_patch_size)
            current_patch += trigger.long()
            patch_len = torch.where(trigger, torch.zeros_like(patch_len), patch_len)
        patch_len += 1
        patch_ids[:, pos] = current_patch

    # Padding bytes continue last real patch ID
    if pad_mask is not None:
        for b in range(batch_size):
            pad_positions = pad_mask[b].nonzero(as_tuple=True)[0]
            if len(pad_positions) > 0:
                first_pad = pad_positions[0].item()
                if first_pad > 0:
                    patch_ids[b, first_pad:] = patch_ids[b, first_pad - 1]

    return patch_ids


def calibrate_threshold(
    entropies: torch.Tensor,
    target_avg_patch_size: float = 4.5,
    min_threshold: float = 0.1,
    max_threshold: float = 5.0,
    num_steps: int = 50,
    max_patch_size: int = 16,
    pad_mask: Optional[torch.Tensor] = None,
) -> float:
    """Binary search for threshold achieving target average patch size."""
    lo, hi = min_threshold, max_threshold

    for _ in range(num_steps):
        mid = (lo + hi) / 2
        patch_ids = entropy_to_patch_ids(
            entropies, threshold=mid, max_patch_size=max_patch_size, pad_mask=pad_mask
        )

        # Count total patches and real bytes across batch
        total_patches = 0
        total_real_bytes = 0
        for b in range(entropies.size(0)):
            if pad_mask is not None:
                real_len = (~pad_mask[b]).sum().item()
            else:
                real_len = entropies.size(1)
            total_real_bytes += real_len
            total_patches += patch_ids[b, :real_len].max().item() + 1

        avg_patch_size = total_real_bytes / max(total_patches, 1)

        if avg_patch_size < target_avg_patch_size:
            lo = mid  # Patches too small, raise threshold
        else:
            hi = mid  # Patches too large, lower threshold

    return (lo + hi) / 2


# ---- Pre-computed Patch ID Store ---------------------------------------------

class PatchIdStore:
    """Loads sharded .pt files of pre-computed patch_ids.

    Directory structure:
        patch_ids_dir/
            metadata.json
            patch_ids_0000.pt  -- list of 1D patch_id tensors
            patch_ids_0001.pt
            ...
    """

    def __init__(self, patch_ids_dir: str):
        self.patch_ids_dir = Path(os.path.expanduser(patch_ids_dir))
        with open(self.patch_ids_dir / "metadata.json") as f:
            self.metadata = json.load(f)
        self.shard_size = self.metadata["shard_size"]
        self.num_examples = self.metadata["num_examples"]
        self._cache = {}

    def _load_shard(self, shard_idx: int) -> list:
        if shard_idx not in self._cache:
            path = self.patch_ids_dir / f"patch_ids_{shard_idx:04d}.pt"
            self._cache[shard_idx] = torch.load(path, weights_only=True)
        return self._cache[shard_idx]

    def __getitem__(self, idx: int) -> torch.Tensor:
        shard_idx = idx // self.shard_size
        local_idx = idx % self.shard_size
        shard = self._load_shard(shard_idx)
        return shard[local_idx]

    def __len__(self) -> int:
        return self.num_examples


class DatasetWithPatchIds(torch.utils.data.Dataset):
    """Wraps a torchtune dataset, adding pre-computed patch_ids to each sample."""

    def __init__(self, dataset, patch_id_store: PatchIdStore):
        self.dataset = dataset
        self.patch_id_store = patch_id_store

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        sample["patch_ids"] = self.patch_id_store[idx]
        return sample


def padded_collate_sft_with_patches(
    batch: List[Dict],
    padding_idx: int = PAD_ID,
    ignore_idx: int = -100,
) -> Dict[str, torch.Tensor]:
    """Collate that handles patch_ids alongside tokens/labels.

    Pads patch_ids so padding bytes continue the last real patch ID.
    """
    from torchtune.data import padded_collate_sft

    has_patches = "patch_ids" in batch[0]

    if has_patches:
        patch_ids_list = [sample.pop("patch_ids") for sample in batch]

    collated = padded_collate_sft(batch, padding_idx=padding_idx, ignore_idx=ignore_idx)

    if has_patches:
        max_len = collated["tokens"].size(1)
        padded_patch_ids = []
        for pids in patch_ids_list:
            seq_len = len(pids)
            if seq_len < max_len:
                last_id = pids[-1].item()
                padding = torch.full((max_len - seq_len,), last_id, dtype=pids.dtype)
                pids = torch.cat([pids, padding])
            else:
                pids = pids[:max_len]
            padded_patch_ids.append(pids)
        collated["patch_ids"] = torch.stack(padded_patch_ids)

    return collated
