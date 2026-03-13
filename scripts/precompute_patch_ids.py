"""Pre-compute entropy-based patch_ids for the training dataset.

Runs the trained entropy model over all examples and saves sharded .pt files
of patch_id tensors. These are loaded during training via PatchIdStore.

Usage:
    conda run -n qwen python scripts/precompute_patch_ids.py \
        --model_path ~/models/entropy_model/best.pt \
        --output_dir ~/models/entropy_patches/slimorca \
        --target_patch_size 6

    # Or with a fixed threshold:
    conda run -n qwen python scripts/precompute_patch_ids.py \
        --model_path ~/models/entropy_model/best.pt \
        --output_dir ~/models/entropy_patches/slimorca \
        --threshold 1.335
"""

import argparse
import json
import os

import torch
from tqdm import tqdm

from torchtune.datasets import slimorca_dataset
from ttblt.bltqwen import blt_tokenizer, PAD_ID
from ttblt.entropy_model import (
    ByteEntropyModel,
    compute_byte_entropies,
    entropy_to_patch_ids,
    calibrate_threshold,
)


def main():
    parser = argparse.ArgumentParser(description="Pre-compute entropy-based patch_ids")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--shard_size", type=int, default=10000)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fixed threshold. If not set, auto-calibrate.",
    )
    parser.add_argument("--target_patch_size", type=float, default=4.5)
    parser.add_argument("--max_patch_size", type=int, default=0, help="0 = no cap (BLT paper default)")
    parser.add_argument("--calibration_samples", type=int, default=1000)
    args = parser.parse_args()

    args.model_path = os.path.expanduser(args.model_path)
    args.output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load entropy model
    print("Loading entropy model...")
    model = ByteEntropyModel(max_seq_len=args.max_seq_len).to(device).to(torch.bfloat16)
    state_dict = torch.load(args.model_path, weights_only=True, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    print("Entropy model loaded.")

    # Load dataset
    print("Loading SlimOrca dataset...")
    tokenizer = blt_tokenizer(max_seq_len=args.max_seq_len)
    dataset = slimorca_dataset(tokenizer)
    num_examples = len(dataset)
    print(f"Dataset size: {num_examples}")

    # Calibrate threshold if not fixed
    threshold = args.threshold
    if threshold is None:
        print(
            f"Calibrating threshold for target avg patch size "
            f"{args.target_patch_size}..."
        )
        # Collect a sample of entropies for calibration
        cal_indices = torch.randperm(num_examples)[: args.calibration_samples]
        cal_tokens = []
        for idx in cal_indices:
            tokens = dataset[idx.item()]["tokens"][: args.max_seq_len]
            cal_tokens.append(torch.tensor(tokens, dtype=torch.long))

        # Pad and batch
        max_len = max(t.size(0) for t in cal_tokens)
        cal_padded = torch.full(
            (len(cal_tokens), max_len), PAD_ID, dtype=torch.long, device=device
        )
        cal_pad_mask = torch.ones(
            (len(cal_tokens), max_len), dtype=torch.bool, device=device
        )
        for i, t in enumerate(cal_tokens):
            cal_padded[i, : t.size(0)] = t.to(device)
            cal_pad_mask[i, : t.size(0)] = False

        # Compute entropies in batches
        all_entropies = []
        for start in range(0, len(cal_tokens), args.batch_size):
            end = min(start + args.batch_size, len(cal_tokens))
            batch_tokens = cal_padded[start:end]
            entropies = compute_byte_entropies(model, batch_tokens)
            all_entropies.append(entropies.cpu())
        all_entropies = torch.cat(all_entropies, dim=0)
        cal_pad_mask_cpu = cal_pad_mask.cpu()

        threshold = calibrate_threshold(
            all_entropies,
            target_avg_patch_size=args.target_patch_size,
            max_patch_size=args.max_patch_size,
            pad_mask=cal_pad_mask_cpu,
        )
        print(f"Calibrated threshold: {threshold:.4f}")
    else:
        print(f"Using fixed threshold: {threshold}")

    # Pre-compute patch_ids in batches
    print("Pre-computing patch_ids...")
    shard_idx = 0
    shard_data = []
    total_patches = 0
    total_bytes = 0

    pbar = tqdm(total=num_examples, desc="Computing patch_ids")
    i = 0
    while i < num_examples:
        batch_end = min(i + args.batch_size, num_examples)
        batch_tokens = []
        batch_lengths = []

        for j in range(i, batch_end):
            tokens = dataset[j]["tokens"][: args.max_seq_len]
            batch_tokens.append(torch.tensor(tokens, dtype=torch.long))
            batch_lengths.append(len(tokens))

        # Pad batch
        max_len = max(t.size(0) for t in batch_tokens)
        padded = torch.full(
            (len(batch_tokens), max_len), PAD_ID, dtype=torch.long, device=device
        )
        pad_mask = torch.ones(
            (len(batch_tokens), max_len), dtype=torch.bool, device=device
        )
        for k, t in enumerate(batch_tokens):
            padded[k, : t.size(0)] = t.to(device)
            pad_mask[k, : t.size(0)] = False

        entropies = compute_byte_entropies(model, padded)
        patch_ids = entropy_to_patch_ids(
            entropies,
            threshold=threshold,
            max_patch_size=args.max_patch_size,
            pad_mask=pad_mask,
        )

        for k in range(len(batch_tokens)):
            real_len = batch_lengths[k]
            pids = patch_ids[k, :real_len].cpu()
            shard_data.append(pids)
            total_patches += pids.max().item() + 1
            total_bytes += real_len

        # Save shard when full
        while len(shard_data) >= args.shard_size:
            to_save = shard_data[: args.shard_size]
            shard_path = os.path.join(
                args.output_dir, f"patch_ids_{shard_idx:04d}.pt"
            )
            torch.save(to_save, shard_path)
            shard_data = shard_data[args.shard_size :]
            shard_idx += 1

        pbar.update(batch_end - i)
        pbar.set_postfix(
            shard=shard_idx,
            avg_ps=f"{total_bytes / max(total_patches, 1):.1f}",
        )
        i = batch_end

    pbar.close()

    # Save remaining
    if shard_data:
        shard_path = os.path.join(args.output_dir, f"patch_ids_{shard_idx:04d}.pt")
        torch.save(shard_data, shard_path)
        shard_idx += 1

    # Save metadata
    avg_patch_size = total_bytes / max(total_patches, 1)
    metadata = {
        "threshold": threshold,
        "avg_patch_size": avg_patch_size,
        "max_patch_size": args.max_patch_size,
        "shard_size": args.shard_size,
        "num_examples": num_examples,
        "num_shards": shard_idx,
        "max_seq_len": args.max_seq_len,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone! Saved {shard_idx} shards to {args.output_dir}")
    print(f"Average patch size: {avg_patch_size:.2f} bytes")
    print(f"Threshold: {threshold:.4f}")


if __name__ == "__main__":
    main()
