"""Train the byte-level entropy model on SlimOrca.

Usage:
    conda run -n qwen python scripts/train_entropy_model.py
    conda run -n qwen python scripts/train_entropy_model.py --epochs 2 --batch_size 16
"""

import argparse
import math
import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from torchtune.datasets import slimorca_dataset
from ttblt.bltqwen import blt_tokenizer, PAD_ID
from ttblt.entropy_model import ByteEntropyModel


def collate_bytes(batch, padding_idx=PAD_ID, max_seq_len=4096):
    """Pad byte sequences to uniform length for next-byte prediction."""
    tokens_list = [
        torch.tensor(sample["tokens"][:max_seq_len], dtype=torch.long)
        for sample in batch
    ]
    max_len = max(t.size(0) for t in tokens_list)
    padded = torch.full((len(tokens_list), max_len), padding_idx, dtype=torch.long)
    pad_mask = torch.ones(len(tokens_list), max_len, dtype=torch.bool)
    for i, t in enumerate(tokens_list):
        padded[i, : t.size(0)] = t
        pad_mask[i, : t.size(0)] = False
    return {"tokens": padded, "pad_mask": pad_mask}


def get_cosine_lr(step, warmup_steps, total_steps, lr, min_lr=1e-6):
    """Cosine LR schedule with linear warmup."""
    if step < warmup_steps:
        return lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


def main():
    parser = argparse.ArgumentParser(description="Train byte-level entropy model")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--output_dir", type=str, default="~/models/entropy_model")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    args.output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    print("Loading SlimOrca dataset...")
    tokenizer = blt_tokenizer(max_seq_len=args.max_seq_len)
    dataset = slimorca_dataset(tokenizer)
    print(f"Dataset size: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_bytes(
            batch, max_seq_len=args.max_seq_len
        ),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Create model
    model = ByteEntropyModel(max_seq_len=args.max_seq_len).to(device).to(torch.bfloat16)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Entropy model: {param_count / 1e6:.1f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(dataloader) * args.epochs

    print(f"Training for {args.epochs} epoch(s), {total_steps} steps")
    print(f"Output dir: {args.output_dir}")

    best_loss = float("inf")
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        epoch_loss = 0.0
        epoch_count = 0
        log_loss = 0.0
        log_count = 0

        for batch in pbar:
            tokens = batch["tokens"].to(device)
            pad_mask = batch["pad_mask"].to(device)

            # Next-byte prediction: input = tokens[:, :-1], target = tokens[:, 1:]
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]
            target_mask = ~pad_mask[:, 1:]  # Only compute loss on real bytes

            # Update LR
            lr = get_cosine_lr(global_step, args.warmup_steps, total_steps, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            logits = model(inputs)  # [batch, seq_len-1, vocab_size]

            # Masked cross-entropy loss
            logits_flat = logits.reshape(-1, logits.size(-1)).float()
            targets_flat = targets.reshape(-1)
            mask_flat = target_mask.reshape(-1)

            loss_all = nn.functional.cross_entropy(
                logits_flat, targets_flat, reduction="none"
            )
            loss = (loss_all * mask_flat).sum() / mask_flat.sum().clamp(min=1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_tokens = mask_flat.sum().item()
            epoch_loss += loss.item() * batch_tokens
            epoch_count += batch_tokens
            log_loss += loss.item() * batch_tokens
            log_count += batch_tokens
            global_step += 1

            if global_step % args.log_every == 0:
                avg_loss = log_loss / max(log_count, 1)
                pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")
                log_loss = 0.0
                log_count = 0

        # End of epoch
        avg_epoch_loss = epoch_loss / max(epoch_count, 1)
        print(f"Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")

        # Save epoch checkpoint
        save_path = os.path.join(args.output_dir, f"epoch_{epoch}.pt")
        torch.save(model.state_dict(), save_path)
        print(f"Saved checkpoint: {save_path}")

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_path = os.path.join(args.output_dir, "best.pt")
            torch.save(model.state_dict(), best_path)
            print(f"New best model saved: {best_path}")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
