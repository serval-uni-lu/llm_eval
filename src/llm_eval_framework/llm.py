import os
import re
import json
import torch
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LLMOutput:
    """Output from LLM generation."""

    content: str
    logprobs: Optional[Dict[str, Any]] = None

    def extract_json(self) -> Optional[Dict[str, Any]]:
        """Extract JSON object from the output content.

        Returns:
            Dictionary if valid JSON found, None otherwise
        """
        # Try to find JSON in the content
        json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        matches = re.findall(json_pattern, self.content, re.DOTALL)

        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        # If no JSON found, try to extract from markdown code blocks
        code_block_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        matches = re.findall(code_block_pattern, self.content, re.DOTALL)

        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        return None


class LLM:
    """LLM using HuggingFace transformers."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        enable_compile: bool = True,
    ):
        """Initialize LLM with a transformers model.

        Args:
            model_name: HuggingFace model name or path
            device: Device to run model on (auto-detected if None)
            load_in_4bit: Load model in 4-bit quantization
            load_in_8bit: Load model in 8-bit quantization
            enable_compile: Enable torch.compile optimizations (default True)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_compile = enable_compile

        # Configure torch._dynamo to prevent recompilation limit errors
        if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "config"):
            # Increase cache size limit from default (8) to handle varying input sizes
            torch._dynamo.config.cache_size_limit = 256

            # Disable compilation if requested (via parameter or environment variable)
            if (
                not self.enable_compile
                or os.environ.get("DISABLE_TORCH_COMPILE", "0") == "1"
            ):
                torch._dynamo.config.suppress_errors = True

        print(f"Loading model: {model_name} on {self.device}...")

        # Setup quantization config if needed
        quantization_kwargs = {}
        if load_in_4bit or load_in_8bit:
            from transformers import BitsAndBytesConfig

            quantization_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
            )
            quantization_kwargs["device_map"] = "auto"

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Configure tokenizer padding
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        is_gemma3 = "gemma-3" in model_name.lower()
        if self.device == "cuda" and not is_gemma3:
            dtype = torch.float16
        else:
            dtype = torch.float32

        model_kwargs = {"dtype": dtype, **quantization_kwargs}

        # Only use device_map for quantized models (required by bitsandbytes)
        # Regular models use .to(device) for cleaner memory management
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # Move to device if not quantized
        if not quantization_kwargs:
            self.model = self.model.to(self.device)

        print("Model loaded successfully!")

    def generate(
        self,
        prompts: str | List[str],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        topk_logprobs: Optional[int] = None,
    ) -> List[LLMOutput]:
        """Generate responses for a batch of prompts.

        Args:
            prompts: either a single input (str) prompt or a List of input prompts
            max_new_tokens: Maximum tokens to generate (may be capped to fit context)
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            topk_logprobs: Number of top log probabilities to return

        Returns:
            List of LLMOutput objects (one per prompt)
        """
        if not prompts:
            return []

        if isinstance(prompts, str):
            prompts = [prompts]

        return_logprobs = topk_logprobs is not None
        requested_tokens = max_new_tokens if max_new_tokens is not None else 512

        # Get model's max context length
        max_context = getattr(self.model.config, "max_position_embeddings", None)
        if max_context is None:
            max_context = getattr(
                self.model.config,
                "n_positions",
                getattr(self.model.config, "max_sequence_length", 8192),
            )

        # Tokenize batch
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        input_lengths = inputs.input_ids.ne(self.tokenizer.pad_token_id).sum(dim=1)
        max_input_length = int(input_lengths.max())

        # Cap max_new_tokens to fit context window (batch-wide)
        available_tokens = max_context - max_input_length
        if available_tokens <= 0:
            raise ValueError(
                f"Longest input length ({max_input_length}) exceeds model context "
                f"({max_context}). Cannot generate any tokens."
            )

        actual_max_tokens = min(requested_tokens, available_tokens)
        if actual_max_tokens < requested_tokens:
            print(
                f"Capping max_new_tokens from {requested_tokens} to "
                f"{actual_max_tokens} to fit context"
            )

        # Build generation kwargs
        gen_kwargs = {
            "return_dict_in_generate": True,
            "output_scores": return_logprobs,
            "pad_token_id": self.tokenizer.pad_token_id,
            "max_new_tokens": actual_max_tokens,
            "max_length": None,
        }

        if temperature is not None:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["do_sample"] = temperature > 0
        if top_p is not None:
            gen_kwargs["top_p"] = top_p

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        sequences = outputs.sequences
        results: List[LLMOutput] = []

        # Process each example independently
        for i, input_len in enumerate(input_lengths.tolist()):
            generated_tokens = sequences[i, input_len:]
            generated_text = self.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            )

            logprobs_dict = None
            if return_logprobs and outputs.scores:
                logprobs_dict = self._compute_logprobs(
                    generated_tokens,
                    [score[i] for score in outputs.scores],
                    topk_logprobs,
                )

            results.append(
                LLMOutput(
                    content=generated_text,
                    logprobs=logprobs_dict,
                )
            )

        return results

    def _compute_logprobs(
        self, generated_tokens: torch.Tensor, scores: tuple, topk: int
    ) -> Dict[str, Any]:
        """Compute logprobs from generation scores.

        Args:
            generated_tokens: Generated token IDs
            scores: Tuple of score tensors from generation
            topk: Number of top alternative tokens to include

        Returns:
            Dictionary with logprobs information
        """
        # Compute transition scores (log probabilities)
        transition_scores = self.model.compute_transition_scores(
            sequences=generated_tokens.unsqueeze(0),
            scores=scores,
            normalize_logits=True,
        )

        # Build content logprobs structure
        content_logprobs = []

        for i, (token_id, score) in enumerate(
            zip(generated_tokens, transition_scores[0])
        ):
            token = self.tokenizer.decode(token_id)
            logprob = score.cpu().item()

            # Get top logprobs for this position
            top_logprobs = []
            if i < len(scores) and topk > 0:
                # Get top k+1 tokens (including the selected one)
                log_probs = torch.log_softmax(scores[i][0], dim=-1)
                top_k_values, top_k_indices = torch.topk(
                    log_probs, k=min(topk + 1, log_probs.shape[-1])
                )

                for val, idx in zip(top_k_values, top_k_indices):
                    if idx != token_id:
                        top_logprobs.append(
                            {
                                "token": self.tokenizer.decode(idx),
                                "logprob": val.cpu().item(),
                            }
                        )
                        if len(top_logprobs) >= topk:
                            break

            content_logprobs.append(
                {"token": token, "logprob": logprob, "top_logprobs": top_logprobs}
            )

        return {"content": content_logprobs}

    @staticmethod
    def list_cached_models(
        sort_by: str = "size", reverse: bool = True, model_filter: Optional[str] = None
    ) -> List[Dict[str, any]]:
        """List HuggingFace models cached locally.

        Args:
            sort_by: Sort key - "size", "name", "accessed", or "modified"
            reverse: Sort in descending order (default True)
            model_filter: Optional substring to filter model names

        Returns:
            List of dicts with model info
        """
        from .model_cache import list_cached_models

        return list_cached_models(sort_by, reverse, model_filter)

    @staticmethod
    def print_cached_models(
        sort_by: str = "size", reverse: bool = True, model_filter: Optional[str] = None
    ) -> None:
        """Print cached HuggingFace models in a formatted table.

        Args:
            sort_by: Sort key - "size", "name", "accessed", or "modified"
            reverse: Sort in descending order (default True)
            model_filter: Optional substring to filter model names
        """
        from .model_cache import print_cached_models

        print_cached_models(sort_by, reverse, model_filter)

    @staticmethod
    def get_cache_size() -> float:
        """Get total size of HuggingFace cache in GB.

        Returns:
            Total cache size in gigabytes
        """
        from .model_cache import get_cache_size

        return get_cache_size()

    def unload(self) -> None:
        """Unload the model and free GPU/CPU memory."""
        import gc
        import ctypes
        import platform

        # Step 1: Delete model and tokenizer
        if hasattr(self, "model"):
            del self.model
            self.model = None

        if hasattr(self, "tokenizer"):
            del self.tokenizer
            self.tokenizer = None

        # Step 2: Run garbage collection
        gc.collect()

        # Step 3: Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Step 4: Release CPU memory (Linux/macOS only)
        try:
            if platform.system() == "Linux":
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            elif platform.system() == "Darwin":
                libc = ctypes.CDLL("libc.dylib")
                libc.malloc_zone_pressure_relief(0, 0)
        except Exception:
            pass

        print("Model unloaded and memory freed")
