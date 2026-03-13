# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ttblt (Tiny Byte Latent Transformers) is a simplified implementation of [Byte Latent Transformers](https://github.com/facebookresearch/blt) as a TorchTune recipe, focused on fine-tuning Qwen 2.5 3B. Instead of traditional tokenization, it operates on raw UTF-8 bytes with a local encoder-decoder that compresses byte sequences into patches processed by the pre-trained model.

## Commands

### Environment
All commands use the `qwen` conda environment (PyTorch 2.10.0+cu130, torchtune 0.6.1, torchao 0.7.0):
```bash
conda run -n qwen <command>
```

### Setup
```bash
conda run -n qwen tune download Qwen/Qwen2.5-3B-Instruct --output-dir /tmp/Qwen2_5-3B-Instruct
```

### Training
```bash
conda run -n qwen tune run full_finetune_single_device.py --config qwen2_5_3B_blt_full_single_device.yaml
```
Override config values via CLI: `--config ... checkpointer.checkpoint_dir=<DIR>`

### Generation/Inference
```bash
conda run -n qwen tune run local_generate.py --config custom_generation_config.yaml
```
Override checkpoint: `checkpointer.checkpoint_dir=/tmp/torchtune/qwen2_5_3B_blt/full_single_device/epoch_0`

### Decoder Mask Test
```bash
conda run -n qwen python -m ttblt.bltqwen
```

## Architecture

The system has three main files plus config:

### `ttblt/bltqwen.py` — Core Model Architecture

All model components live in this single file:

- **`ByteLatentModelTokenizer`**: UTF-8 byte tokenizer (vocab 259: 256 bytes + BOS=257, EOS=258, PAD=256). No BPE — just raw bytes.
- **`HashNGramEmbedder`**: Optional n-gram (3-8) hash embeddings added to byte embeddings. Can use separate tables per n-gram size or a shared table. Controlled by `use_hash_ngrams` config (0=off, 1=separate, 2=shared).
- **`compute_local_entropy()` / `dynamic_patch()`**: Entropy-based dynamic patching — determines patch boundaries by local sequence complexity. `patch_reduce()` aggregates byte embeddings into patch embeddings via mean-pooling.
- **`LocalEncoderWithPooling`**: Small transformer (1 self-attn + 1 cross-attn layer, 8 heads, 2048 dim) that processes bytes. Encoder cross-attn: patches (query) attend to bytes (key/value). Outputs patch embeddings projected to global model dimension.
- **`LocalDecoder`**: Predicts byte logits from byte embeddings + patch embeddings via self-attention (9 layers) and cross-attention (1 layer) to patches. Decoder cross-attn: bytes (query) attend to patches (key/value).
- **`ByteLatentQwen2p5Decoder`**: Main model combining local encoder/decoder with Qwen 2.5 3B. Cross-attention is added to 1/3 of the Qwen layers. Supports freezing global (Qwen) parameters for N steps to warm up the local encoder/decoder first.
- **`qwen2_5_blt()`** and **`blt_tokenizer()`**: Factory functions referenced by config YAML.

### `full_finetune_single_device.py` — Training Recipe

TorchTune `FTRecipeInterface` implementation. Key modifications from standard TorchTune recipe:
- Non-strict checkpoint loading (new BLT parameters don't exist in Qwen checkpoint)
- Filters out token embeddings from Qwen (replaced by byte embeddings)
- Uses `FullModelHFCheckpointer` for loading but `FullModelMetaCheckpointer` format for saving
- Supports optimizer-in-backward for memory efficiency (requires `gradient_accumulation_steps=1`)
- Split learning rates: 4e-4 for new BLT params, 4e-5 for pretrained Qwen (10x ratio)
- Cosine LR schedule with 2000-step linear warmup
- Gradient clipping at 1.0, weight decay 0.1

### `local_generate.py` — Inference Recipe

Loads trained checkpoints and runs `unified_generate()` with configurable sampling (temperature, top-k, top-p, frequency/repetition penalties). KV cache is disabled due to cross-attention incompatibility.

## Key Design Decisions

- **Byte-level operation**: Sequences are ~4x longer than token-based (4096 bytes ≈ 1024 tokens), affecting memory and speed.
- **Cross-attention on 1/3 of layers**: Memory constraint — training fits in 24GB VRAM with batch_size=1.
- **Split LR replaces freezing**: Instead of `freeze_global_for_n_steps`, uses split learning rates (4e-4 new / 4e-5 Qwen) with cosine warmup.
- **Depth-dependent init**: New layers use `std = 0.02 / sqrt(2*(layer_idx+1))` for stable early training.
- **No KV cache**: Cross-attention layers make standard KV caching incompatible.
- **Single file model**: All model components are in `bltqwen.py` for simplicity.

## Current Status

Training converges well (loss 0.73 after 1 epoch on Alpaca, 19h on DGX Spark GB10). Generation produces partially coherent English — recognizable words and structure but garbled byte-level spelling. See `EXPERIMENTS.md` for full results.

**Latest checkpoint:** `/tmp/torchtune/qwen2_5_3B_blt/full_single_device/epoch_0`
**WandB:** https://wandb.ai/ian-barber/ttqwblt/runs/rkydz7ol

**Active areas:** Byte-level generation quality, longer training, larger datasets, dynamic patching.
