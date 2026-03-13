    # Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import math
import sys
import time
from typing import Any, Dict, List

import torch
from omegaconf import DictConfig
from torch import nn

from torchtune import config, generation, training, utils
from torchtune.data import Message, Role
from torchtune.training import FullModelTorchTuneCheckpointer
from torchtune.training.checkpointing._utils import safe_torch_load

logger = utils.get_logger("DEBUG")


class InferenceRecipe:
    """
    Recipe for generating tokens from a dense Transformer-based LLM.

    Currently this recipe supports single-GPU generation only. Speculative
    decoding is not supported.

    For more details on how to use this recipe for generation, please see our
    tutorial: https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#generation

    For using this recipe with a quantized model, please the following section of
    the above tutorial:
    https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#speeding-up-generation-using-quantization
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(dtype=cfg.dtype, device=self._device)
        self._quantizer = config.instantiate(cfg.quantizer)
        self._quantization_mode = training.get_quantizer_mode(self._quantizer)

        training.set_seed(seed=cfg.seed)

    def setup(self, cfg: DictConfig) -> None:
        checkpointer = config.instantiate(cfg.checkpointer)

        if self._quantization_mode is not None:
            if not isinstance(checkpointer, FullModelTorchTuneCheckpointer):
                raise ValueError(
                    "Quantization is only supported for models quantized and saved with the "
                    "FullModelTorchTuneCheckpointer - please ensure you have quantized your "
                    "model and are using the quantized weights!"
                )
            if "qat" in self._quantization_mode:
                raise ValueError(
                    "You have specified a quantizer with 'QAT' - "
                    "QAT quantizers should only be used during quantization aware training "
                    "and when quantizing models. Please use the corresponding post-training "
                    "quantizer e.g. Int8DynActInt4WeightQuantizer for Int8DynActInt4WeightQATQuantizer."
                )

        # if self._quantization_mode is None:
        #     ckpt_dict = checkpointer.load_checkpoint()
        # else:
        #     # weights_only needs to be False when loading a quantized model
        #     ckpt_dict = checkpointer.load_checkpoint(weights_only=False)
        # TODO: hack loading. 
        model_state_dict = safe_torch_load(checkpointer._checkpoint_path)
        
        self._model = self._setup_model(
            model_cfg=cfg.model,
            model_state_dict=model_state_dict,
        )
        self._tokenizer = config.instantiate(cfg.tokenizer)

    def _setup_model(
        self,
        model_cfg: DictConfig,
        model_state_dict: Dict[str, Any],
    ) -> nn.Module:
        with training.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(model_cfg)

        if self._quantization_mode is not None:
            model = self._quantizer.quantize(model)
            model = model.to(device=self._device, dtype=self._dtype)
            for k, v in model_state_dict.items():
                model_state_dict[k] = v.to(self._device)
            model.load_state_dict(model_state_dict, assign=True)
        else:
            # strict=False: entropy model keys may be missing from checkpoint
            result = model.load_state_dict(model_state_dict, strict=False)
            if result.missing_keys:
                logger.info(f"Missing keys in checkpoint (expected for entropy model): {len(result.missing_keys)} keys")
            if result.unexpected_keys:
                logger.warning(f"Unexpected keys in checkpoint: {result.unexpected_keys}")

        # Validate model was loaded in with the expected dtype.
        training.validate_expected_param_dtype(
            model.named_parameters(), dtype=self._dtype
        )
        logger.info(f"Model is initialized with precision {self._dtype}.")

        return model

    def convert_prompt_to_tokens(
        self,
        prompt: Dict[Role, str],
    ) -> List[int]:
        """
        Convert the prompt string to a user message with optional system messages
        and tokenize using the prompt template defined on the tokenizer.
        """
        messages = []
        if "system" in prompt and prompt["system"] is not None:
            messages.append(Message(role="system", content=prompt["system"]))
        messages.extend(
            [
                Message(role="user", content=prompt["user"], eot=True),
                # Empty assistant message to kick-start generation
                Message(role="assistant", content=""),
            ]
        )
        return self._tokenizer({"messages": messages}, inference=True)["tokens"]

    @torch.inference_mode()
    def diagnostic_loss(self):
        """Run a forward pass on a training-format prompt and print cross-entropy loss.
        Uses the same tokenization path as training (role-prefixed messages, no BOS)."""
        from torchtune.data import Message
        messages = [
            Message(role="user", content=(
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\nWhat is the capital of France?\n\n"
                "### Response:\n"
            ), eot=True),
            Message(role="assistant", content="The capital of France is Paris."),
        ]
        result = self._tokenizer({"messages": messages}, inference=False)
        diag_tokens = result["tokens"]
        diag_input = torch.tensor([diag_tokens], dtype=torch.long, device=self._device)

        logits = self._model(diag_input[:, :-1])
        targets = diag_input[:, 1:]
        if isinstance(logits, list):
            logits = torch.cat(logits, dim=1)
        loss = nn.CrossEntropyLoss()(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        bpb = loss.item() / math.log(2)  # convert nats to bits-per-byte
        logger.info(f"Diagnostic loss (training format): {loss.item():.4f} nats, {bpb:.4f} bits-per-byte")
        return loss.item()

    @torch.inference_mode()
    def generate(self, cfg: DictConfig):
        self._model.eval()

        # Run diagnostic forward pass first
        self.diagnostic_loss()

        tokens = self.convert_prompt_to_tokens(
            cfg.prompt,
        )
        #tokens = [1] + list("Hello ".encode('utf-8'))  # Starting prompt
        custom_generate_next_token = None

        # Ensure the cache is setup on the right device, with only as many tokens as we need
        if cfg.enable_kv_cache:
            with self._device:
                self._model.setup_caches(
                    batch_size=1,
                    dtype=self._dtype,
                    decoder_max_seq_len=prompt.numel() + cfg.max_new_tokens,
                )
        # Use patch based model decoding.
        t0 = time.perf_counter()
        prompt = torch.tensor(tokens, dtype=torch.long, device=self._device) # Changed to long.

        max_new = cfg.max_new_tokens

        generated_tokens = self._model.unified_generate(prompt, greedy=True, max_new_tokens=max_new)
        decoded = self._tokenizer.decode(generated_tokens[0])
        print("Generated text (greedy):", decoded)

        generated_tokens = self._model.unified_generate(prompt, greedy=False, max_new_tokens=max_new)
        decoded = self._tokenizer.decode(generated_tokens[0])
        print("Generated text (sampling):", decoded)

        generated_tokens = self._model.unified_generate(prompt, greedy=False, max_new_tokens=max_new, top_k=150)
        decoded = self._tokenizer.decode(generated_tokens[0])
        print("Generated text (sampling, tk150):", decoded)

        generated_tokens = self._model.unified_generate(prompt, greedy=False, max_new_tokens=max_new, top_k=50)
        decoded = self._tokenizer.decode(generated_tokens[0])
        print("Generated text (sampling, tk50):", decoded)

        generated_tokens = self._model.unified_generate(prompt, greedy=False, max_new_tokens=max_new, top_k=50, temperature=1.0)
        decoded = self._tokenizer.decode(generated_tokens[0])
        print("Generated text (sampling, tk50, t1):", decoded)

        generated_tokens = self._model.unified_generate(prompt, greedy=False, max_new_tokens=max_new, top_k=50, temperature=0.6)
        decoded = self._tokenizer.decode(generated_tokens[0])
        print("Generated text (sampling, tk50, t0.6):", decoded)

        generated_tokens = self._model.unified_generate(prompt, greedy=False, max_new_tokens=max_new, top_k=50, temperature=0.3)
        decoded = self._tokenizer.decode(generated_tokens[0])
        print("Generated text (sampling, tk50, t0.3):", decoded)

        generated_tokens = self._model.unified_generate(prompt, greedy=False, max_new_tokens=max_new, top_k=50, temperature=0.3, repetition_penalty=1.0)
        decoded = self._tokenizer.decode(generated_tokens[0])
        print("Generated text (sampling, tk50, t0.3, rep1):", decoded)

        generated_tokens = self._model.unified_generate(prompt, greedy=False, max_new_tokens=max_new, top_k=50, temperature=0.3, repetition_penalty=0.6)
        decoded = self._tokenizer.decode(generated_tokens[0])
        print("Generated text (sampling, tk50, t0.3, rep0.6):", decoded)

        logits = self._model(generated_tokens[:, :-1])  # Input all but last token
        targets = generated_tokens[:, 1:]  # Predict next tokens
        loss = nn.CrossEntropyLoss()(logits.reshape(-1, logits.size(-1)), targets.view(-1))        
        print("Loss on generated (patches):", loss.item())
        t = time.perf_counter() - t0

        model_size = sum(
            [
                p.numel() * p.dtype.itemsize
                for p in itertools.chain(
                    self._model.parameters(), self._model.buffers()
                )
            ]
        )

        tokens_generated = len(generated_tokens[0]) - prompt.size(0)
        tokens_sec = tokens_generated / t
        logger.info(
            f"Time for inference: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec"
        )
        logger.info(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")
        if self._device.type != "cpu":
            torch_device = utils.get_torch_device_namespace()
            logger.info(
                f"Memory used: {torch_device.max_memory_allocated() / 1e9:.02f} GB"
            )


@config.parse
def main(cfg: DictConfig) -> None:
    config.log_config(recipe_name="InferenceRecipe", cfg=cfg)
    recipe = InferenceRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.generate(cfg=cfg)


if __name__ == "__main__":
    sys.exit(main())
