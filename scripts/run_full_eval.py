"""Comprehensive BLT evaluation: 6 evaluations comparing BLT vs base Qwen 2.5 3B.

Usage:
    conda run --cwd /home/ianbarber/Projects/ttblt -n qwen python scripts/run_full_eval.py

Implements all evaluations from scripts/EVAL_PLAN.md:
1. BPB comparison across passage types
2. Robustness to noisy input
3. Character-level tasks
4. Morphological tasks
5. Cross-script / multilingual
6. Adversarial tokenization
"""

import gc
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# ============================================================
# Configuration
# ============================================================

BLT_CHECKPOINT_DIR = os.path.expanduser("~/models/ttblt_v3/full_single_device/epoch_29")
QWEN_BASE_DIR = os.path.expanduser("~/models/Qwen2_5-3B-Instruct")
ENTROPY_MODEL_PATH = os.path.expanduser("~/models/entropy_model/best.pt")
DEVICE = "cuda"
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "eval_results.md")

# BLT model config (must match training)
BLT_MODEL_CFG = dict(
    dropout=0.1,
    entropy_model_path=ENTROPY_MODEL_PATH,
    entropy_threshold=1.335,
    encoder_num_layers=3,
    encoder_num_cross_layers=1,
    decoder_num_layers=9,
    decoder_num_cross_layers=3,
    use_hash_ngrams=1,
    patch_size=8,
)

# Generation settings (from eval plan)
BLT_GEN_KWARGS = dict(
    temperature=0.3,
    top_k=50,
    max_new_tokens=200,
    greedy=True,
)


# ============================================================
# Passages and prompts from EVAL_PLAN.md
# ============================================================

# Section 1: BPB passages (original 5 from eval_bpb_multi + 3 new)
BPB_PASSAGES = {
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
    "slimorca_math": (
        "system\nYou are an AI assistant that helps people find information.\n"
        "human\nWhat is the sum of the first 10 prime numbers?\n"
        "assistant\nThe first 10 prime numbers are 2, 3, 5, 7, 11, 13, 17, 19, 23, and 29. "
        "Their sum is 2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 + 23 + 29 = 129."
    ),
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
    "factual_short": (
        "system\nYou are a helpful assistant.\n"
        "human\nWhat is the capital of France?\n"
        "assistant\nThe capital of France is Paris."
    ),
    # New passages from eval plan
    "wikipedia": (
        "The Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical "
        "rainforest in the Amazon biome that covers most of the Amazon basin of South "
        "America. This basin encompasses 7,000,000 km2, of which 5,500,000 km2 are "
        "covered by the rainforest. This region includes territory belonging to nine "
        "nations and 3,344 formally acknowledged indigenous territories. The majority "
        "of the forest, about 60%, is in Brazil, followed by Peru with 13%, Colombia "
        "with 10%, and with minor amounts in Bolivia, Ecuador, French Guiana, Guyana, "
        "Suriname, and Venezuela."
    ),
    "code_snippet": (
        "def fibonacci(n: int) -> list[int]:\n"
        '    """Return the first n Fibonacci numbers.\n'
        "    \n"
        "    Args:\n"
        "        n: Number of Fibonacci numbers to generate.\n"
        "    \n"
        "    Returns:\n"
        "        A list of the first n Fibonacci numbers.\n"
        '    """\n'
        "    if n <= 0:\n"
        "        return []\n"
        "    elif n == 1:\n"
        "        return [0]\n"
        "    \n"
        "    fib = [0, 1]\n"
        "    for i in range(2, n):\n"
        "        fib.append(fib[i-1] + fib[i-2])\n"
        "    return fib\n"
        "\n"
        "# Example usage\n"
        "result = fibonacci(10)\n"
        "print(f'First 10 Fibonacci numbers: {result}')\n"
    ),
    "mixed_language": (
        "In Japanese, the word for 'beautiful' is \u7f8e\u3057\u3044 (utsukushii). "
        "The Chinese character \u7f8e means beauty and is shared between Japanese kanji "
        "and Chinese hanzi. In Korean, beauty is written as \uc544\ub984\ub2f5\ub2e4 "
        "(areumdapda). These East Asian languages share many cultural concepts but express "
        "them through very different writing systems and phonologies."
    ),
}

# Section 2: Robustness prompts
ROBUSTNESS_PROMPTS = {
    "exercise": (
        "system\nYou are a helpful assistant.\nhuman\n"
        "What are three benefits of regular exercise?\nassistant\n"
    ),
    "neural_net": (
        "system\nYou are a helpful assistant.\nhuman\n"
        "Explain what a neural network is in simple terms.\nassistant\n"
    ),
    "poem": (
        "system\nYou are a helpful assistant.\nhuman\n"
        "Write a short poem about the ocean.\nassistant\n"
    ),
}

# Section 3: Character-level tasks
CHARACTER_TASKS = {
    "count_r_strawberry": {
        "prompt": "system\nYou are a helpful assistant.\nhuman\nHow many times does the letter 'r' appear in the word 'strawberry'?\nassistant\n",
        "answer": "3",
        "explanation": "s-t-r-a-w-b-e-r-r-y has 3 r's",
    },
    "5th_letter_elephant": {
        "prompt": "system\nYou are a helpful assistant.\nhuman\nWhat is the 5th letter of the word 'elephant'?\nassistant\n",
        "answer": "h",
        "explanation": "e-l-e-p-h, 5th letter is h",
    },
    "spell_backwards_banana": {
        "prompt": "system\nYou are a helpful assistant.\nhuman\nSpell the word 'banana' backwards.\nassistant\n",
        "answer": "ananab",
        "explanation": "banana reversed is ananab",
    },
    "count_letters_mississippi": {
        "prompt": "system\nYou are a helpful assistant.\nhuman\nHow many letters are in the word 'mississippi'?\nassistant\n",
        "answer": "11",
        "explanation": "m-i-s-s-i-s-s-i-p-p-i = 11 letters",
    },
    "letter_after_q": {
        "prompt": "system\nYou are a helpful assistant.\nhuman\nWhat letter comes after 'q' in the English alphabet?\nassistant\n",
        "answer": "r",
        "explanation": "q is followed by r",
    },
    "i_before_e_receive": {
        "prompt": "system\nYou are a helpful assistant.\nhuman\nDoes the word 'receive' follow the 'i before e except after c' rule?\nassistant\n",
        "answer": "yes",
        "explanation": "receive has 'cei' - e before i after c, following the rule",
    },
}

# Section 4: Morphological tasks
MORPHOLOGICAL_TASKS = {
    "root_unbelievably": {
        "prompt": "system\nYou are a helpful assistant.\nhuman\nWhat is the root word of 'unbelievably'?\nassistant\n",
        "expected_keywords": ["believe"],
    },
    "morphemes_internationalization": {
        "prompt": "system\nYou are a helpful assistant.\nhuman\nBreak the word 'internationalization' into its morphemes (prefixes, root, suffixes).\nassistant\n",
        "expected_keywords": ["inter", "nation", "ize", "ation"],
    },
    "prefix_uncomfortable": {
        "prompt": "system\nYou are a helpful assistant.\nhuman\nAdd the prefix 'un-' to the word 'comfortable' and use it in a sentence.\nassistant\n",
        "expected_keywords": ["uncomfortable"],
    },
    "suffix_happy": {
        "prompt": "system\nYou are a helpful assistant.\nhuman\nWhat suffix would you add to 'happy' to make it mean 'the state of being happy'?\nassistant\n",
        "expected_keywords": ["ness", "happiness"],
    },
}

# Section 5: Cross-script BPB passages
CROSS_SCRIPT_BPB = {
    "japanese_chinese": "The Japanese word for cat is \u732b (neko). In Chinese, it is also written as \u732b but pronounced m\u0101o.",
    "german_compound": "The German word Donaudampfschifffahrtsgesellschaftskapit\u00e4n is one of the longest compound words.",
    "math_symbols": "In mathematics, we write \u03c0 \u2248 3.14159 and e \u2248 2.71828.",
}

CROSS_SCRIPT_GEN = {
    "french_phrase": "system\nYou are a helpful assistant.\nhuman\nWhat does the French phrase 'c'est la vie' mean?\nassistant\n",
    "transliterate_hello": "system\nYou are a helpful assistant.\nhuman\nTransliterate 'hello' into Japanese hiragana.\nassistant\n",
}

# Section 6: Adversarial tokenization passages
ADVERSARIAL_PASSAGES = {
    "irregular_spacing": 'The    quick     brown    fox',
    "dot_separated": "H.e.l.l.o. .W.o.r.l.d.",
    "naming_conventions": "CamelCaseVariableName = snake_case_variable_name",
    "char_runs": "aaaaaaaaaaabbbbbbbbbccccccccc",
    "emoji_heavy": "\U0001f331 + \u2600\ufe0f \u2192 \U0001f34e (photosynthesis simplified)",
}


# ============================================================
# Noise functions for Section 2
# ============================================================

def apply_typos(text: str, rate: float = 0.2) -> str:
    """Swap adjacent characters in ~rate of words."""
    random.seed(42)
    words = text.split()
    result = []
    for w in words:
        if random.random() < rate and len(w) > 2:
            idx = random.randint(0, len(w) - 2)
            w = w[:idx] + w[idx + 1] + w[idx] + w[idx + 2:]
        result.append(w)
    return " ".join(result)


def apply_missing_chars(text: str, rate: float = 0.2) -> str:
    """Drop a random character from ~rate of words."""
    random.seed(42)
    words = text.split()
    result = []
    for w in words:
        if random.random() < rate and len(w) > 2:
            idx = random.randint(0, len(w) - 1)
            w = w[:idx] + w[idx + 1:]
        result.append(w)
    return " ".join(result)


def apply_leetspeak(text: str) -> str:
    """Replace some letters with numbers."""
    mapping = {'a': '4', 'e': '3', 'i': '1', 'o': '0'}
    result = []
    for ch in text:
        if ch.lower() in mapping:
            result.append(mapping[ch.lower()])
        else:
            result.append(ch)
    return "".join(result)


# ============================================================
# BLT model loading and evaluation
# ============================================================

def load_blt_model(checkpoint_dir: str, device: str = "cuda"):
    """Load the BLT model from checkpoint (same pattern as eval_bpb_multi.py)."""
    from torchtune import training
    from torchtune.training import FullModelHFCheckpointer
    from ttblt.bltqwen import qwen2_5_blt

    # Load base Qwen weights
    base_dir = QWEN_BASE_DIR
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
        model = qwen2_5_blt(**BLT_MODEL_CFG)

    # Remove tok_embeddings from Qwen state dict (replaced by byte embeddings)
    keys_to_remove = [k for k in ckpt[training.MODEL_KEY] if "tok_embeddings.weight" in k]
    for k in keys_to_remove:
        del ckpt[training.MODEL_KEY][k]
    model.load_state_dict(ckpt[training.MODEL_KEY], strict=False)

    # Load adapter (fine-tuned) weights on top
    adapter_path = os.path.join(checkpoint_dir, "adapter_model.pt")
    if os.path.exists(adapter_path):
        warm_state = torch.load(adapter_path, weights_only=True, map_location=device)
        model.load_state_dict(warm_state, strict=False)

    model.to(device)
    model.eval()
    return model


def blt_eval_bpb(model, tokenizer, text: str, device: str = "cuda") -> float:
    """Compute BPB for the BLT model on a text passage."""
    model.eval()
    tokens = tokenizer.encode(text, add_bos=True, add_eos=True)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids[:, :-1])
        if isinstance(logits, list):
            logits = torch.cat(logits, dim=1)
        targets = input_ids[:, 1:]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="mean",
        )
        bpb = loss.item() / math.log(2)
    return bpb


def blt_generate(model, tokenizer, prompt_text: str, device: str = "cuda") -> str:
    """Generate text from the BLT model given a prompt."""
    tokens = tokenizer.encode(prompt_text, add_bos=True, add_eos=False)
    input_ids = torch.tensor(tokens, dtype=torch.long, device=device)
    with torch.no_grad():
        output = model.unified_generate(input_ids, **BLT_GEN_KWARGS)
    return tokenizer.decode(output[0].tolist())


# ============================================================
# Qwen model loading and evaluation (via torchtune)
# ============================================================

def load_qwen_model(device: str = "cuda"):
    """Load base Qwen 2.5 3B Instruct via torchtune."""
    from torchtune import training
    from torchtune.training import FullModelHFCheckpointer
    from torchtune.models.qwen2_5 import qwen2_5_3b, qwen2_5_tokenizer

    print("Loading base Qwen 2.5 3B Instruct via torchtune...")

    # Load tokenizer
    tokenizer = qwen2_5_tokenizer(
        path=os.path.join(QWEN_BASE_DIR, "vocab.json"),
        merges_file=os.path.join(QWEN_BASE_DIR, "merges.txt"),
        max_seq_len=4096,
    )

    # Load checkpoint
    checkpointer = FullModelHFCheckpointer(
        checkpoint_dir=QWEN_BASE_DIR,
        checkpoint_files=[
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ],
        model_type="QWEN2",
        output_dir="/tmp/eval_dummy_qwen",
    )
    ckpt = checkpointer.load_checkpoint()

    with training.set_default_dtype(torch.bfloat16), torch.device(device):
        model = qwen2_5_3b()

    model.load_state_dict(ckpt[training.MODEL_KEY], strict=True)
    model.to(device)
    model.eval()
    return model, tokenizer


def qwen_eval_bpb(model, tokenizer, text: str, device: str = "cuda") -> float:
    """Compute BPB for the base Qwen model.

    BPB = (nats_per_token / ln(2)) * (num_tokens / num_bytes)
    """
    tokens = tokenizer.encode(text, add_bos=False, add_eos=True)
    num_tokens = len(tokens)
    num_bytes = len(text.encode("utf-8"))
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(input_ids[:, :-1])
        if isinstance(logits, list):
            logits = torch.cat(logits, dim=1)
        targets = input_ids[:, 1:]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="mean",
        )
        loss_nats = loss.item()

    bpb = (loss_nats / math.log(2)) * ((num_tokens - 1) / num_bytes)
    return bpb


def qwen_generate(model, tokenizer, prompt_text: str, device: str = "cuda") -> str:
    """Generate text from the base Qwen model using simple autoregressive loop."""
    tokens = tokenizer.encode(prompt_text, add_bos=False, add_eos=False)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

    eos_id = tokenizer.eos_id if hasattr(tokenizer, 'eos_id') else 151645
    max_new = 200

    for _ in range(max_new):
        with torch.no_grad():
            logits = model(input_ids)
            if isinstance(logits, list):
                logits = torch.cat(logits, dim=1)
            next_logits = logits[0, -1, :]
            next_token = torch.argmax(next_logits).item()

        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_token]], device=device)],
            dim=1,
        )

        if next_token == eos_id:
            break

    # Decode only the generated part
    generated_tokens = input_ids[0, len(tokens):].tolist()
    return tokenizer.decode(generated_tokens)


# ============================================================
# Free GPU memory helper
# ============================================================

def free_model(*models):
    """Delete models and free GPU memory."""
    for m in models:
        if m is not None:
            del m
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# Main evaluation loop
# ============================================================

def main():
    print("=" * 70)
    print("BLT Full Evaluation Suite")
    print(f"BLT checkpoint: {BLT_CHECKPOINT_DIR}")
    print(f"Qwen base: {QWEN_BASE_DIR}")
    print("=" * 70)

    results = {}

    # ------------------------------------------------------------------
    # Phase 1: BLT evaluations (BPB + generation)
    # ------------------------------------------------------------------
    print("\n>>> Loading BLT model...")
    from ttblt.bltqwen import blt_tokenizer
    blt_tok = blt_tokenizer(max_seq_len=4096)
    blt_model = load_blt_model(BLT_CHECKPOINT_DIR, DEVICE)

    # --- Eval 1: BPB comparison ---
    print("\n--- Eval 1: BPB Comparison ---")
    blt_bpb = {}
    for name, text in BPB_PASSAGES.items():
        bpb = blt_eval_bpb(blt_model, blt_tok, text, DEVICE)
        blt_bpb[name] = bpb
        print(f"  BLT BPB [{name}]: {bpb:.4f}")
    results["bpb_blt"] = blt_bpb

    # --- Eval 2: Robustness BPB (BLT) ---
    print("\n--- Eval 2: Robustness (BLT BPB) ---")
    blt_robustness = {}
    for name, prompt in ROBUSTNESS_PROMPTS.items():
        clean_bpb = blt_eval_bpb(blt_model, blt_tok, prompt, DEVICE)
        typo_bpb = blt_eval_bpb(blt_model, blt_tok, apply_typos(prompt), DEVICE)
        missing_bpb = blt_eval_bpb(blt_model, blt_tok, apply_missing_chars(prompt), DEVICE)
        leet_bpb = blt_eval_bpb(blt_model, blt_tok, apply_leetspeak(prompt), DEVICE)
        blt_robustness[name] = {
            "clean": clean_bpb,
            "typos": typo_bpb,
            "missing": missing_bpb,
            "leet": leet_bpb,
            "ratio_typos": typo_bpb / clean_bpb if clean_bpb > 0 else float("inf"),
            "ratio_missing": missing_bpb / clean_bpb if clean_bpb > 0 else float("inf"),
            "ratio_leet": leet_bpb / clean_bpb if clean_bpb > 0 else float("inf"),
        }
        print(f"  BLT [{name}] clean={clean_bpb:.4f} typos={typo_bpb:.4f} miss={missing_bpb:.4f} leet={leet_bpb:.4f}")
    results["robustness_blt"] = blt_robustness

    # --- Eval 2 (cont): Robustness generation (BLT) ---
    print("\n--- Eval 2 (cont): Robustness Generation (BLT) ---")
    blt_robust_gen = {}
    for name, prompt in ROBUSTNESS_PROMPTS.items():
        clean_gen = blt_generate(blt_model, blt_tok, prompt, DEVICE)
        typo_gen = blt_generate(blt_model, blt_tok, apply_typos(prompt), DEVICE)
        blt_robust_gen[name] = {"clean": clean_gen, "typos": typo_gen}
        print(f"  BLT [{name}] clean gen: {clean_gen[:100]}...")
        print(f"  BLT [{name}] typo gen:  {typo_gen[:100]}...")
    results["robustness_gen_blt"] = blt_robust_gen

    # --- Eval 3: Character-level tasks (BLT) ---
    print("\n--- Eval 3: Character-Level Tasks (BLT) ---")
    blt_char_results = {}
    for name, task in CHARACTER_TASKS.items():
        gen = blt_generate(blt_model, blt_tok, task["prompt"], DEVICE)
        blt_char_results[name] = gen
        print(f"  BLT [{name}]: {gen[:150]}")
    results["char_tasks_blt"] = blt_char_results

    # --- Eval 4: Morphological tasks (BLT) ---
    print("\n--- Eval 4: Morphological Tasks (BLT) ---")
    blt_morph_results = {}
    for name, task in MORPHOLOGICAL_TASKS.items():
        gen = blt_generate(blt_model, blt_tok, task["prompt"], DEVICE)
        blt_morph_results[name] = gen
        print(f"  BLT [{name}]: {gen[:150]}")
    results["morph_tasks_blt"] = blt_morph_results

    # --- Eval 5: Cross-script BPB + generation (BLT) ---
    print("\n--- Eval 5: Cross-Script (BLT) ---")
    blt_cross_bpb = {}
    for name, text in CROSS_SCRIPT_BPB.items():
        bpb = blt_eval_bpb(blt_model, blt_tok, text, DEVICE)
        blt_cross_bpb[name] = bpb
        print(f"  BLT BPB [{name}]: {bpb:.4f}")
    results["cross_script_bpb_blt"] = blt_cross_bpb

    blt_cross_gen = {}
    for name, prompt in CROSS_SCRIPT_GEN.items():
        gen = blt_generate(blt_model, blt_tok, prompt, DEVICE)
        blt_cross_gen[name] = gen
        print(f"  BLT gen [{name}]: {gen[:150]}")
    results["cross_script_gen_blt"] = blt_cross_gen

    # --- Eval 6: Adversarial tokenization BPB (BLT) ---
    print("\n--- Eval 6: Adversarial Tokenization (BLT) ---")
    blt_adv_bpb = {}
    for name, text in ADVERSARIAL_PASSAGES.items():
        bpb = blt_eval_bpb(blt_model, blt_tok, text, DEVICE)
        blt_adv_bpb[name] = bpb
        print(f"  BLT BPB [{name}]: {bpb:.4f}")
    results["adversarial_bpb_blt"] = blt_adv_bpb

    # Free BLT model
    print("\n>>> Freeing BLT model...")
    free_model(blt_model)
    del blt_model
    time.sleep(2)

    # ------------------------------------------------------------------
    # Phase 2: Qwen evaluations (BPB + generation)
    # ------------------------------------------------------------------
    print("\n>>> Loading base Qwen model...")
    qwen_model, qwen_tok = load_qwen_model(DEVICE)

    # --- Eval 1: BPB comparison (Qwen) ---
    print("\n--- Eval 1: BPB Comparison (Qwen) ---")
    qwen_bpb = {}
    for name, text in BPB_PASSAGES.items():
        bpb = qwen_eval_bpb(qwen_model, qwen_tok, text, DEVICE)
        qwen_bpb[name] = bpb
        print(f"  Qwen BPB [{name}]: {bpb:.4f}")
    results["bpb_qwen"] = qwen_bpb

    # --- Eval 2: Robustness BPB (Qwen) ---
    print("\n--- Eval 2: Robustness (Qwen BPB) ---")
    qwen_robustness = {}
    for name, prompt in ROBUSTNESS_PROMPTS.items():
        clean_bpb = qwen_eval_bpb(qwen_model, qwen_tok, prompt, DEVICE)
        typo_bpb = qwen_eval_bpb(qwen_model, qwen_tok, apply_typos(prompt), DEVICE)
        missing_bpb = qwen_eval_bpb(qwen_model, qwen_tok, apply_missing_chars(prompt), DEVICE)
        leet_bpb = qwen_eval_bpb(qwen_model, qwen_tok, apply_leetspeak(prompt), DEVICE)
        qwen_robustness[name] = {
            "clean": clean_bpb,
            "typos": typo_bpb,
            "missing": missing_bpb,
            "leet": leet_bpb,
            "ratio_typos": typo_bpb / clean_bpb if clean_bpb > 0 else float("inf"),
            "ratio_missing": missing_bpb / clean_bpb if clean_bpb > 0 else float("inf"),
            "ratio_leet": leet_bpb / clean_bpb if clean_bpb > 0 else float("inf"),
        }
        print(f"  Qwen [{name}] clean={clean_bpb:.4f} typos={typo_bpb:.4f} miss={missing_bpb:.4f} leet={leet_bpb:.4f}")
    results["robustness_qwen"] = qwen_robustness

    # --- Eval 2 (cont): Robustness generation (Qwen) ---
    print("\n--- Eval 2 (cont): Robustness Generation (Qwen) ---")
    qwen_robust_gen = {}
    for name, prompt in ROBUSTNESS_PROMPTS.items():
        clean_gen = qwen_generate(qwen_model, qwen_tok, prompt, DEVICE)
        typo_gen = qwen_generate(qwen_model, qwen_tok, apply_typos(prompt), DEVICE)
        qwen_robust_gen[name] = {"clean": clean_gen, "typos": typo_gen}
        print(f"  Qwen [{name}] clean gen: {clean_gen[:100]}...")
        print(f"  Qwen [{name}] typo gen:  {typo_gen[:100]}...")
    results["robustness_gen_qwen"] = qwen_robust_gen

    # --- Eval 3: Character-level tasks (Qwen) ---
    print("\n--- Eval 3: Character-Level Tasks (Qwen) ---")
    qwen_char_results = {}
    for name, task in CHARACTER_TASKS.items():
        gen = qwen_generate(qwen_model, qwen_tok, task["prompt"], DEVICE)
        qwen_char_results[name] = gen
        print(f"  Qwen [{name}]: {gen[:150]}")
    results["char_tasks_qwen"] = qwen_char_results

    # --- Eval 4: Morphological tasks (Qwen) ---
    print("\n--- Eval 4: Morphological Tasks (Qwen) ---")
    qwen_morph_results = {}
    for name, task in MORPHOLOGICAL_TASKS.items():
        gen = qwen_generate(qwen_model, qwen_tok, task["prompt"], DEVICE)
        qwen_morph_results[name] = gen
        print(f"  Qwen [{name}]: {gen[:150]}")
    results["morph_tasks_qwen"] = qwen_morph_results

    # --- Eval 5: Cross-script BPB + generation (Qwen) ---
    print("\n--- Eval 5: Cross-Script (Qwen) ---")
    qwen_cross_bpb = {}
    for name, text in CROSS_SCRIPT_BPB.items():
        bpb = qwen_eval_bpb(qwen_model, qwen_tok, text, DEVICE)
        qwen_cross_bpb[name] = bpb
        print(f"  Qwen BPB [{name}]: {bpb:.4f}")
    results["cross_script_bpb_qwen"] = qwen_cross_bpb

    qwen_cross_gen = {}
    for name, prompt in CROSS_SCRIPT_GEN.items():
        gen = qwen_generate(qwen_model, qwen_tok, prompt, DEVICE)
        qwen_cross_gen[name] = gen
        print(f"  Qwen gen [{name}]: {gen[:150]}")
    results["cross_script_gen_qwen"] = qwen_cross_gen

    # --- Eval 6: Adversarial tokenization BPB (Qwen) ---
    print("\n--- Eval 6: Adversarial Tokenization (Qwen) ---")
    qwen_adv_bpb = {}
    for name, text in ADVERSARIAL_PASSAGES.items():
        bpb = qwen_eval_bpb(qwen_model, qwen_tok, text, DEVICE)
        qwen_adv_bpb[name] = bpb
        print(f"  Qwen BPB [{name}]: {bpb:.4f}")
    results["adversarial_bpb_qwen"] = qwen_adv_bpb

    # Free Qwen model
    print("\n>>> Freeing Qwen model...")
    free_model(qwen_model)

    # ------------------------------------------------------------------
    # Write results to markdown
    # ------------------------------------------------------------------
    print(f"\n>>> Writing results to {OUTPUT_FILE}")
    write_results_md(results)
    print("Done!")

    # Also save raw results as JSON for later use
    json_path = OUTPUT_FILE.replace(".md", ".json")
    # Convert any non-serializable values
    def make_serializable(obj):
        if isinstance(obj, float):
            if math.isinf(obj) or math.isnan(obj):
                return str(obj)
            return obj
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    with open(json_path, "w") as f:
        json.dump(make_serializable(results), f, indent=2, ensure_ascii=False)
    print(f"Raw results saved to {json_path}")


# ============================================================
# Markdown report generation
# ============================================================

def write_results_md(results: dict):
    """Write comprehensive evaluation results as markdown."""
    lines = []

    def add(text=""):
        lines.append(text)

    add("# BLT vs Qwen 2.5 3B Instruct: Full Evaluation Results")
    add()
    add(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}")
    add(f"**BLT Checkpoint:** `{BLT_CHECKPOINT_DIR}` (epoch 29)")
    add(f"**Base Model:** Qwen 2.5 3B Instruct (`{QWEN_BASE_DIR}`)")
    add(f"**BLT Config:** encoder 3+1, decoder 9+3, hash n-grams, patch_size=8, entropy threshold=1.335")
    add()

    # ------------------------------------------------------------------
    # 1. BPB Comparison
    # ------------------------------------------------------------------
    add("## 1. Bits-Per-Byte (BPB) Comparison")
    add()
    add("Lower BPB = better compression = better modeling of the text.")
    add()
    add("| Passage | BLT BPB | Qwen BPB | Difference | Winner |")
    add("|---------|---------|----------|------------|--------|")

    blt_bpb = results.get("bpb_blt", {})
    qwen_bpb = results.get("bpb_qwen", {})
    blt_total, qwen_total, count = 0, 0, 0
    for name in BPB_PASSAGES:
        b = blt_bpb.get(name, float("nan"))
        q = qwen_bpb.get(name, float("nan"))
        diff = b - q
        winner = "BLT" if b < q else "Qwen"
        add(f"| {name} | {b:.4f} | {q:.4f} | {diff:+.4f} | {winner} |")
        if not (math.isnan(b) or math.isnan(q)):
            blt_total += b
            qwen_total += q
            count += 1

    if count > 0:
        avg_blt = blt_total / count
        avg_qwen = qwen_total / count
        avg_diff = avg_blt - avg_qwen
        avg_winner = "BLT" if avg_blt < avg_qwen else "Qwen"
        add(f"| **AVERAGE** | **{avg_blt:.4f}** | **{avg_qwen:.4f}** | **{avg_diff:+.4f}** | **{avg_winner}** |")
    add()

    # ------------------------------------------------------------------
    # 2. Robustness to Noisy Input
    # ------------------------------------------------------------------
    add("## 2. Robustness to Noisy Input")
    add()
    add("### 2a. BPB Degradation Ratios")
    add()
    add("Ratio = corrupted_bpb / clean_bpb. Lower ratio = more robust to noise.")
    add()
    add("| Prompt | Model | Clean BPB | Typos BPB | Missing BPB | Leet BPB | Typo Ratio | Missing Ratio | Leet Ratio |")
    add("|--------|-------|-----------|-----------|-------------|----------|------------|---------------|------------|")

    for name in ROBUSTNESS_PROMPTS:
        blt_r = results.get("robustness_blt", {}).get(name, {})
        qwen_r = results.get("robustness_qwen", {}).get(name, {})
        if blt_r:
            add(f"| {name} | BLT | {blt_r.get('clean',0):.4f} | {blt_r.get('typos',0):.4f} | {blt_r.get('missing',0):.4f} | {blt_r.get('leet',0):.4f} | {blt_r.get('ratio_typos',0):.3f} | {blt_r.get('ratio_missing',0):.3f} | {blt_r.get('ratio_leet',0):.3f} |")
        if qwen_r:
            add(f"| {name} | Qwen | {qwen_r.get('clean',0):.4f} | {qwen_r.get('typos',0):.4f} | {qwen_r.get('missing',0):.4f} | {qwen_r.get('leet',0):.4f} | {qwen_r.get('ratio_typos',0):.3f} | {qwen_r.get('ratio_missing',0):.3f} | {qwen_r.get('ratio_leet',0):.3f} |")
    add()

    # Average degradation ratios
    add("**Average Degradation Ratios:**")
    add()
    for model_name, key in [("BLT", "robustness_blt"), ("Qwen", "robustness_qwen")]:
        ratios = {"typos": [], "missing": [], "leet": []}
        for name in ROBUSTNESS_PROMPTS:
            r = results.get(key, {}).get(name, {})
            if r:
                ratios["typos"].append(r.get("ratio_typos", 0))
                ratios["missing"].append(r.get("ratio_missing", 0))
                ratios["leet"].append(r.get("ratio_leet", 0))
        if ratios["typos"]:
            avg_t = sum(ratios["typos"]) / len(ratios["typos"])
            avg_m = sum(ratios["missing"]) / len(ratios["missing"])
            avg_l = sum(ratios["leet"]) / len(ratios["leet"])
            add(f"- **{model_name}:** typos={avg_t:.3f}, missing={avg_m:.3f}, leet={avg_l:.3f}")
    add()

    # Robustness generation samples
    add("### 2b. Generation from Corrupted Prompts")
    add()
    for name in ROBUSTNESS_PROMPTS:
        add(f"#### Prompt: {name}")
        add()
        add(f"**Original prompt:** `{ROBUSTNESS_PROMPTS[name][:80]}...`")
        add(f"**Corrupted (typos):** `{apply_typos(ROBUSTNESS_PROMPTS[name])[:80]}...`")
        add()

        blt_gen = results.get("robustness_gen_blt", {}).get(name, {})
        qwen_gen = results.get("robustness_gen_qwen", {}).get(name, {})

        add("| | Clean | Typos |")
        add("|---|---|---|")
        # Extract just the generated part (after prompt)
        blt_clean = blt_gen.get("clean", "N/A")
        blt_typo = blt_gen.get("typos", "N/A")
        qwen_clean = qwen_gen.get("clean", "N/A")
        qwen_typo = qwen_gen.get("typos", "N/A")

        # Truncate for table display
        def trunc(s, n=300):
            s = s.replace("\n", " ").replace("|", "\\|")
            return s[:n] + ("..." if len(s) > n else "")

        add(f"| **BLT** | {trunc(blt_clean)} | {trunc(blt_typo)} |")
        add(f"| **Qwen** | {trunc(qwen_clean)} | {trunc(qwen_typo)} |")
        add()

    # ------------------------------------------------------------------
    # 3. Character-Level Tasks
    # ------------------------------------------------------------------
    add("## 3. Character-Level Task Scorecard")
    add()
    add("| Task | Expected | BLT Response | BLT Correct? | Qwen Response | Qwen Correct? |")
    add("|------|----------|-------------|---------------|---------------|----------------|")

    blt_char_correct = 0
    qwen_char_correct = 0
    total_char = len(CHARACTER_TASKS)

    for name, task in CHARACTER_TASKS.items():
        blt_resp = results.get("char_tasks_blt", {}).get(name, "N/A")
        qwen_resp = results.get("char_tasks_qwen", {}).get(name, "N/A")
        expected = task["answer"]

        # Simple check: does the expected answer appear in the response?
        blt_correct = expected.lower() in blt_resp.lower()
        qwen_correct = expected.lower() in qwen_resp.lower()
        if blt_correct:
            blt_char_correct += 1
        if qwen_correct:
            qwen_char_correct += 1

        # Truncate response for table
        def trunc_resp(s, n=120):
            s = s.replace("\n", " ").replace("|", "\\|")
            return s[:n] + ("..." if len(s) > n else "")

        blt_mark = "Yes" if blt_correct else "No"
        qwen_mark = "Yes" if qwen_correct else "No"
        add(f"| {name} | {expected} | {trunc_resp(blt_resp)} | {blt_mark} | {trunc_resp(qwen_resp)} | {qwen_mark} |")

    add()
    add(f"**BLT Score:** {blt_char_correct}/{total_char} correct")
    add(f"**Qwen Score:** {qwen_char_correct}/{total_char} correct")
    add()

    # ------------------------------------------------------------------
    # 4. Morphological Tasks
    # ------------------------------------------------------------------
    add("## 4. Morphological Tasks")
    add()
    for name, task in MORPHOLOGICAL_TASKS.items():
        blt_resp = results.get("morph_tasks_blt", {}).get(name, "N/A")
        qwen_resp = results.get("morph_tasks_qwen", {}).get(name, "N/A")

        add(f"### {name}")
        add()
        add(f"**Expected keywords:** {task['expected_keywords']}")
        add()
        add(f"**BLT:**")
        add(f"```")
        add(blt_resp[:500])
        add(f"```")
        add()
        add(f"**Qwen:**")
        add(f"```")
        add(qwen_resp[:500])
        add(f"```")
        add()

    # ------------------------------------------------------------------
    # 5. Cross-Script / Multilingual
    # ------------------------------------------------------------------
    add("## 5. Cross-Script / Multilingual")
    add()
    add("### 5a. BPB on Mixed-Script Text")
    add()
    add("| Passage | BLT BPB | Qwen BPB | Difference | Winner |")
    add("|---------|---------|----------|------------|--------|")

    blt_cross = results.get("cross_script_bpb_blt", {})
    qwen_cross = results.get("cross_script_bpb_qwen", {})
    for name in CROSS_SCRIPT_BPB:
        b = blt_cross.get(name, float("nan"))
        q = qwen_cross.get(name, float("nan"))
        diff = b - q
        winner = "BLT" if b < q else "Qwen"
        add(f"| {name} | {b:.4f} | {q:.4f} | {diff:+.4f} | {winner} |")
    add()

    add("### 5b. Cross-Script Generation")
    add()
    for name in CROSS_SCRIPT_GEN:
        blt_gen = results.get("cross_script_gen_blt", {}).get(name, "N/A")
        qwen_gen = results.get("cross_script_gen_qwen", {}).get(name, "N/A")
        add(f"#### {name}")
        add()
        add(f"**BLT:**")
        add(f"```")
        add(blt_gen[:500])
        add(f"```")
        add()
        add(f"**Qwen:**")
        add(f"```")
        add(qwen_gen[:500])
        add(f"```")
        add()

    # ------------------------------------------------------------------
    # 6. Adversarial Tokenization
    # ------------------------------------------------------------------
    add("## 6. Adversarial Tokenization BPB")
    add()
    add("Text patterns that challenge traditional BPE tokenizers.")
    add()
    add("| Passage | Text | BLT BPB | Qwen BPB | Difference | Winner |")
    add("|---------|------|---------|----------|------------|--------|")

    blt_adv = results.get("adversarial_bpb_blt", {})
    qwen_adv = results.get("adversarial_bpb_qwen", {})
    for name, text in ADVERSARIAL_PASSAGES.items():
        b = blt_adv.get(name, float("nan"))
        q = qwen_adv.get(name, float("nan"))
        diff = b - q
        winner = "BLT" if b < q else "Qwen"
        display_text = text[:40].replace("|", "\\|") + ("..." if len(text) > 40 else "")
        add(f"| {name} | `{display_text}` | {b:.4f} | {q:.4f} | {diff:+.4f} | {winner} |")
    add()

    # ------------------------------------------------------------------
    # 7. Overall Assessment
    # ------------------------------------------------------------------
    add("## 7. Overall Assessment")
    add()

    # Compute summary stats
    add("### Summary Statistics")
    add()

    # BPB wins
    bpb_blt_wins = sum(1 for n in BPB_PASSAGES if blt_bpb.get(n, 999) < qwen_bpb.get(n, 999))
    bpb_qwen_wins = len(BPB_PASSAGES) - bpb_blt_wins
    add(f"- **Standard BPB:** BLT wins {bpb_blt_wins}/{len(BPB_PASSAGES)}, Qwen wins {bpb_qwen_wins}/{len(BPB_PASSAGES)}")

    # Adversarial wins
    adv_blt_wins = sum(1 for n in ADVERSARIAL_PASSAGES if blt_adv.get(n, 999) < qwen_adv.get(n, 999))
    adv_qwen_wins = len(ADVERSARIAL_PASSAGES) - adv_blt_wins
    add(f"- **Adversarial BPB:** BLT wins {adv_blt_wins}/{len(ADVERSARIAL_PASSAGES)}, Qwen wins {adv_qwen_wins}/{len(ADVERSARIAL_PASSAGES)}")

    # Cross-script wins
    cs_blt_wins = sum(1 for n in CROSS_SCRIPT_BPB if blt_cross.get(n, 999) < qwen_cross.get(n, 999))
    cs_qwen_wins = len(CROSS_SCRIPT_BPB) - cs_blt_wins
    add(f"- **Cross-Script BPB:** BLT wins {cs_blt_wins}/{len(CROSS_SCRIPT_BPB)}, Qwen wins {cs_qwen_wins}/{len(CROSS_SCRIPT_BPB)}")

    # Character tasks
    add(f"- **Character Tasks:** BLT {blt_char_correct}/{total_char}, Qwen {qwen_char_correct}/{total_char}")

    # Average robustness ratios
    for model_name, key in [("BLT", "robustness_blt"), ("Qwen", "robustness_qwen")]:
        avg_ratios = []
        for name in ROBUSTNESS_PROMPTS:
            r = results.get(key, {}).get(name, {})
            if r:
                for rk in ["ratio_typos", "ratio_missing", "ratio_leet"]:
                    val = r.get(rk, 0)
                    if not (math.isinf(val) or math.isnan(val)):
                        avg_ratios.append(val)
        if avg_ratios:
            overall = sum(avg_ratios) / len(avg_ratios)
            add(f"- **{model_name} Average Robustness Degradation Ratio:** {overall:.3f}")
    add()

    add("### Key Findings")
    add()
    add("*(Populated after running the evaluation)*")
    add()

    # Write to file
    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(lines))
    print(f"Results written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
