"""Evaluate BPB across multiple passages and checkpoints.

Usage:
    conda run -n qwen python scripts/eval_bpb_multi.py \
        --checkpoint_dir ~/models/ttblt_v3/full_single_device/epoch_20 \
        [--all_checkpoints ~/models/ttblt_v3/full_single_device/]
"""

import argparse
import os
import glob
import torch
from torchtune import config, training
from ttblt.bltqwen import qwen2_5_blt, blt_tokenizer

EVAL_PASSAGES = {
    # Original eval passage (plain prose, out-of-distribution)
    "photosynthesis": (
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request.\n\n### Instruction:\n"
        "Explain how photosynthesis works in simple terms.\n\n### Response:\n\n"
        "Photosynthesis is the process by which plants convert sunlight into energy. "
        "Plants absorb light through chlorophyll in their leaves, which captures the "
        "sun's energy. They then use this energy to convert carbon dioxide from the air "
        "and water from the soil into glucose, a type of sugar that serves as food for "
        "the plant. Oxygen is released as a byproduct of this process."
    ),
    # SlimOrca-style instruction/response (in-distribution format)
    "slimorca_math": (
        "system\nYou are an AI assistant that helps people find information.\n"
        "human\nWhat is the sum of the first 10 prime numbers?\n"
        "assistant\nThe first 10 prime numbers are 2, 3, 5, 7, 11, 13, 17, 19, 23, and 29. "
        "Their sum is 2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 + 23 + 29 = 129."
    ),
    # SlimOrca-style with longer response
    "slimorca_explain": (
        "system\nYou are an AI assistant. Provide a detailed answer.\n"
        "human\nExplain the difference between a stack and a queue in computer science.\n"
        "assistant\nA stack and a queue are both abstract data types used to store "
        "collections of elements, but they differ in how elements are added and removed. "
        "A stack follows the Last-In-First-Out (LIFO) principle, meaning the most "
        "recently added element is the first one to be removed. Think of a stack of "
        "plates: you add plates to the top and remove them from the top. Common "
        "operations are push (add to top) and pop (remove from top). A queue follows "
        "the First-In-First-Out (FIFO) principle, meaning the first element added is "
        "the first one removed. Think of a line at a store: the first person in line "
        "is the first person served. Common operations are enqueue (add to back) and "
        "dequeue (remove from front)."
    ),
    # Plain English prose (different domain)
    "history": (
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request.\n\n### Instruction:\n"
        "Briefly describe the causes of World War I.\n\n### Response:\n\n"
        "World War I was caused by a complex web of factors including rising "
        "nationalism across Europe, imperial competition for colonies and resources, "
        "a tangled system of military alliances, and an arms race between major powers. "
        "The immediate trigger was the assassination of Archduke Franz Ferdinand of "
        "Austria-Hungary in Sarajevo in June 1914. This set off a chain reaction as "
        "alliance obligations drew one nation after another into the conflict."
    ),
    # Short factual (tests byte-level precision)
    "factual_short": (
        "system\nYou are a helpful assistant.\n"
        "human\nWhat is the capital of France?\n"
        "assistant\nThe capital of France is Paris."
    ),
}


def eval_bpb(model, tokenizer, text, device):
    """Compute BPB on a single passage."""
    model.eval()
    tokens = tokenizer.encode(text, add_bos=True, add_eos=True)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids[:, :-1])
        if isinstance(logits, list):
            logits = torch.cat(logits, dim=1)
        targets = input_ids[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="mean",
        )
        bpb = loss.item() / 0.6931  # nats to bits
    return bpb


def load_model(checkpoint_dir, device):
    """Load model from a checkpoint directory."""
    model_cfg = dict(
        dropout=0.1,
        entropy_model_path=os.path.expanduser("~/models/entropy_model/best.pt"),
        entropy_threshold=1.335,
        encoder_num_layers=3,
        encoder_num_cross_layers=1,
        decoder_num_layers=9,
        decoder_num_cross_layers=3,
        use_hash_ngrams=1,
        patch_size=8,
    )

    # Load base Qwen weights
    from torchtune.training import FullModelHFCheckpointer
    base_dir = os.path.expanduser("~/models/Qwen2_5-3B-Instruct")
    checkpointer = FullModelHFCheckpointer(
        checkpoint_dir=base_dir,
        checkpoint_files=[
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ],
        model_type="QWEN2",
        output_dir="/tmp/eval_dummy",
    )
    ckpt = checkpointer.load_checkpoint()

    with training.set_default_dtype(torch.bfloat16), torch.device(device):
        model = qwen2_5_blt(**model_cfg)

    # Remove tok_embeddings from state dict
    keys_to_remove = [k for k in ckpt[training.MODEL_KEY] if "tok_embeddings.weight" in k]
    for k in keys_to_remove:
        del ckpt[training.MODEL_KEY][k]
    model.load_state_dict(ckpt[training.MODEL_KEY], strict=False)

    # Load adapter checkpoint on top
    adapter_path = os.path.join(checkpoint_dir, "adapter_model.pt")
    if os.path.exists(adapter_path):
        warm_state = torch.load(adapter_path, weights_only=True, map_location=device)
        model.load_state_dict(warm_state, strict=False)

    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--all_checkpoints", default=None,
                        help="Base dir to eval all epoch_* subdirs")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    tokenizer = blt_tokenizer(max_seq_len=4096)

    if args.all_checkpoints:
        base = os.path.expanduser(args.all_checkpoints)
        dirs = sorted(glob.glob(os.path.join(base, "epoch_*")),
                      key=lambda x: int(x.split("_")[-1]))
    else:
        dirs = [os.path.expanduser(args.checkpoint_dir)]

    print(f"{'Step':>6} | {'photo':>6} | {'orca_m':>6} | {'orca_e':>6} | {'hist':>6} | {'fact':>6} | {'AVG':>6}")
    print("-" * 58)

    for ckpt_dir in dirs:
        epoch_num = int(ckpt_dir.split("_")[-1])
        step = epoch_num * 250

        model = load_model(ckpt_dir, args.device)

        results = {}
        for name, text in EVAL_PASSAGES.items():
            results[name] = eval_bpb(model, tokenizer, text, args.device)

        avg = sum(results.values()) / len(results)
        print(f"{step:>6} | {results['photosynthesis']:>6.3f} | {results['slimorca_math']:>6.3f} | "
              f"{results['slimorca_explain']:>6.3f} | {results['history']:>6.3f} | "
              f"{results['factual_short']:>6.3f} | {avg:>6.3f}")

        # Free GPU memory
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
