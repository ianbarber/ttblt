# BLT Byte-Level Evaluation Plan

## Objective

Evaluate the strengths and weaknesses of the byte-level BLT model compared to the base token-level Qwen 2.5 3B, focusing on tasks where byte-level operation should provide advantages.

## Checkpoint

Use the latest available checkpoint from `~/models/ttblt_v3/full_single_device/` (highest epoch number). Load using the same pattern as `scripts/eval_bpb_multi.py`.

## Environment

All commands use `conda run --cwd /home/ianbarber/Projects/ttblt -n qwen python ...`

## Evaluations

### 1. BPB Comparison vs Base Qwen (Quantitative)

Compare bits-per-byte between BLT and the base Qwen model on the same text passages. This gives an apples-to-apples compression comparison.

**Method:**
- For BLT: use the existing `eval_bpb()` function pattern from `scripts/eval_bpb_multi.py`
- For Qwen: load the base Qwen 2.5 3B Instruct model via torchtune, compute cross-entropy loss on the same passages, convert to BPB by dividing nats-per-token loss by the tokens-to-bytes ratio (total_bytes / total_tokens for each passage gives the conversion factor: `bpb = loss_nats / ln(2) * (num_tokens / num_bytes)`)
- Test passages: use the 5 from `EVAL_PASSAGES` in `eval_bpb_multi.py`, plus a few additional ones below

**Additional passages:**
- A Wikipedia paragraph (general knowledge prose)
- A code snippet (Python function with comments)
- Mixed-language text (English with Japanese/Chinese characters)

### 2. Robustness to Noisy Input (Quantitative + Qualitative)

**Method:**
- Take 3 clean SlimOrca-format prompts and create corrupted versions:
  - `typos`: swap adjacent characters in ~20% of words (e.g., "the" -> "teh")
  - `missing_chars`: drop random characters from ~20% of words
  - `leetspeak`: replace some letters with numbers (a->4, e->3, i->1, o->0)
- Measure BPB on corrupted input for both BLT and Qwen
- Also generate responses from both models given corrupted prompts and compare quality
- Key metric: BPB degradation ratio (corrupted_bpb / clean_bpb) — lower is more robust

**Prompts:**
```
system\nYou are a helpful assistant.\nhuman\nWhat are three benefits of regular exercise?\nassistant\n
system\nYou are a helpful assistant.\nhuman\nExplain what a neural network is in simple terms.\nassistant\n
system\nYou are a helpful assistant.\nhuman\nWrite a short poem about the ocean.\nassistant\n
```

### 3. Character-Level Tasks (Qualitative)

**Method:**
- Generate responses to character-level questions using BLT's `unified_generate()`
- Also generate from base Qwen for comparison
- Score: correct/incorrect for each task

**Prompts (SlimOrca format for in-distribution):**
```
system\nYou are a helpful assistant.\nhuman\nHow many times does the letter 'r' appear in the word 'strawberry'?\nassistant\n
system\nYou are a helpful assistant.\nhuman\nWhat is the 5th letter of the word 'elephant'?\nassistant\n
system\nYou are a helpful assistant.\nhuman\nSpell the word 'banana' backwards.\nassistant\n
system\nYou are a helpful assistant.\nhuman\nHow many letters are in the word 'mississippi'?\nassistant\n
system\nYou are a helpful assistant.\nhuman\nWhat letter comes after 'q' in the English alphabet?\nassistant\n
system\nYou are a helpful assistant.\nhuman\nDoes the word 'receive' follow the 'i before e except after c' rule?\nassistant\n
```

### 4. Morphological Tasks (Qualitative)

**Prompts:**
```
system\nYou are a helpful assistant.\nhuman\nWhat is the root word of 'unbelievably'?\nassistant\n
system\nYou are a helpful assistant.\nhuman\nBreak the word 'internationalization' into its morphemes (prefixes, root, suffixes).\nassistant\n
system\nYou are a helpful assistant.\nhuman\nAdd the prefix 'un-' to the word 'comfortable' and use it in a sentence.\nassistant\n
system\nYou are a helpful assistant.\nhuman\nWhat suffix would you add to 'happy' to make it mean 'the state of being happy'?\nassistant\n
```

### 5. Cross-Script / Multilingual (Quantitative + Qualitative)

**BPB eval on mixed-script text:**
```
"The Japanese word for cat is 猫 (neko). In Chinese, it is also written as 猫 but pronounced māo."
"The German word Donaudampfschifffahrtsgesellschaftskapitän is one of the longest compound words."
"In mathematics, we write π ≈ 3.14159 and e ≈ 2.71828."
```

**Generation prompts:**
```
system\nYou are a helpful assistant.\nhuman\nWhat does the French phrase 'c'est la vie' mean?\nassistant\n
system\nYou are a helpful assistant.\nhuman\nTransliterate 'hello' into Japanese hiragana.\nassistant\n
```

### 6. Adversarial Tokenization (Quantitative)

**BPB eval on text that challenges tokenizers:**
```
"The    quick     brown    fox" (irregular spacing)
"H.e.l.l.o. .W.o.r.l.d." (dot-separated)
"CamelCaseVariableName = snake_case_variable_name" (code naming conventions)
"aaaaaaaaaaabbbbbbbbbccccccccc" (character runs)
"🌱 + ☀️ → 🍎 (photosynthesis simplified)" (emoji-heavy)
```

## Output

Write results to `scripts/eval_results.md` with:
1. Summary table of BPB comparisons (BLT vs Qwen) across all passage types
2. Robustness degradation ratios
3. Character-level task scorecard (correct/incorrect for each model)
4. Morphological task responses (side-by-side)
5. Cross-script BPB and generation samples
6. Adversarial tokenization BPB comparison
7. Overall assessment of byte-level advantages demonstrated

## Implementation Notes

- For loading base Qwen for comparison: use `torchtune.models.qwen2.qwen2_5` with the HF checkpointer pointing to `~/models/Qwen2_5-3B-Instruct/`. Use the Qwen tokenizer from `torchtune.models.qwen2` for tokenization.
- For BLT generation: use `model.unified_generate()` with `temperature=0.3, top_k=50, max_new_tokens=200`
- For Qwen generation: use standard torchtune generate or a simple autoregressive loop
- All generation should use greedy decoding for reproducibility (set `greedy=True` or `temperature=0.01`)
- The BLT model config is: `dropout=0.1, entropy_model_path=~/models/entropy_model/best.pt, entropy_threshold=1.335, encoder_num_layers=3, encoder_num_cross_layers=1, decoder_num_layers=9, decoder_num_cross_layers=3, use_hash_ngrams=1, patch_size=8`
- Base Qwen checkpoint: `~/models/Qwen2_5-3B-Instruct/`
- BLT checkpoint: latest `epoch_*` in `~/models/ttblt_v3/full_single_device/`
