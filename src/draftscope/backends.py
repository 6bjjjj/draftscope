"""Lazy adapters. The portable NVIDIA path is Transformers; vLLM is optional."""

from dataclasses import dataclass, replace
from importlib.metadata import version
from importlib.util import find_spec
import time
import shutil
import warnings

from .config import GenerationConfig, ModelSpec, QWEN_DRAFT, QWEN_INT4
from .errors import CompatibilityError
from .hardware import choose_dtype, detect_gpu, prefix_ineligibility, validate_backend
from .metrics import Timing


@dataclass
class Output:
    text: str
    token_ids: list[int]
    timing: Timing
    finish_reason: str | None = None


def engine_kwargs(model: ModelSpec, config: GenerationConfig, dtype: str, draft_path=None):
    """Pure, testable construction of vLLM arguments. No downloads or CUDA calls."""
    out = dict(
        model=model.model,
        revision=model.revision,
        tokenizer=model.tokenizer or model.model,
        tokenizer_revision=model.tokenizer_revision or model.revision,
        dtype=dtype,
        kv_cache_dtype=dtype,
        tensor_parallel_size=1,
        seed=config.seed,
        max_model_len=config.max_model_len,
        max_num_seqs=1,
        max_num_batched_tokens=config.max_model_len,
        gpu_memory_utilization=config.memory_fraction,
        language_model_only=True,
        enable_prefix_caching=False,
        enforce_eager=False,
        disable_log_stats=True,
        trust_remote_code=False,
        cpu_offload_gb=0,
    )
    if config.speculation != "none":
        out.update(spec_method=config.speculation, spec_tokens=config.draft_tokens)
    if config.speculation == "dflash":
        if not draft_path:
            raise ValueError("A pinned DFlash checkpoint path is required")
        out["spec_model"] = str(draft_path)
    size = 1 + config.draft_tokens if config.speculation != "none" else 1
    out["compilation_config"] = {
        "cudagraph_capture_sizes": [size],
        "max_cudagraph_capture_size": size,
    }
    if config.experimental_prefix:
        if model.model != QWEN_INT4[0] or model.revision != QWEN_INT4[1] or dtype != "bfloat16":
            raise CompatibilityError(
                "Prefix V8 requires the pinned Qwen3.8 INT4 target and BF16 activations"
            )
        out["worker_extension_cls"] = "draftscope.experimental.prefix_v8.PrefixHeadWorker"
    return out


class TransformersBackend:
    def __init__(self, model, config):
        import torch
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoTokenizer,
        )

        self.torch, self.config = torch, config
        self.gpu = detect_gpu(config.device_index)
        self.dtype = choose_dtype(self.gpu, config.dtype)
        self.device = f"cuda:{config.device_index}"
        torch.cuda.set_device(config.device_index)
        torch.cuda.set_per_process_memory_fraction(config.memory_fraction, config.device_index)
        torch.manual_seed(config.seed)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model.tokenizer or model.model,
            revision=model.tokenizer_revision or model.revision,
            trust_remote_code=False,
        )
        hf_config = AutoConfig.from_pretrained(
            model.model, revision=model.revision, trust_remote_code=False
        )
        cls = (
            AutoModelForImageTextToText
            if hf_config.model_type == "qwen3_5"
            else AutoModelForCausalLM
        )
        self.model = cls.from_pretrained(
            model.model,
            revision=model.revision,
            dtype=getattr(torch, self.dtype),
            device_map={"": self.device},
            attn_implementation="sdpa",
            trust_remote_code=False,
        ).eval()

    def generate(self, prompt):
        if self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=not bool(self.tokenizer.chat_template)
        ).to(self.device)
        length = inputs["input_ids"].shape[1]
        if length + self.config.max_new_tokens > self.config.max_model_len:
            raise ValueError(
                "Prompt plus output cap exceeds max_model_len; input was not truncated"
            )
        args = dict(max_new_tokens=self.config.max_new_tokens, do_sample=False)
        if self.config.fixed_tokens:
            args.update(
                min_new_tokens=self.config.max_new_tokens,
                eos_token_id=None,
                forced_eos_token_id=None,
            )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            args["pad_token_id"] = self.tokenizer.eos_token_id
        self.torch.cuda.synchronize(self.config.device_index)
        start = time.perf_counter()
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, **args)
        self.torch.cuda.synchronize(self.config.device_index)
        elapsed = time.perf_counter() - start
        ids = generated[0, length:].tolist()
        # No per-token timestamps are available in this non-streaming adapter.
        # Returning null is more accurate than inventing a decode-only rate.
        return Output(
            self.tokenizer.decode(ids, skip_special_tokens=True),
            ids,
            Timing(len(ids), elapsed),
            None,
        )

    def close(self):
        del self.model
        self.torch.cuda.empty_cache()


class VLLMBackend:
    def __init__(self, model, config):
        import os

        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        if version("vllm") != "0.31.0":
            raise CompatibilityError("This adapter is pinned to vLLM 0.31.0")
        self.gpu = detect_gpu(config.device_index)
        validate_backend(self.gpu, "vllm")
        if config.device_index != 0:
            raise CompatibilityError(
                "For vLLM select the physical GPU with CUDA_VISIBLE_DEVICES, then use index 0"
            )
        from vllm import LLM, SamplingParams

        self.config, self.SamplingParams = config, SamplingParams
        self.dtype = choose_dtype(self.gpu, config.dtype)
        self.acceleration = {
            "requested": config.experimental_prefix,
            "enabled": False,
            "reason": None,
        }
        if config.experimental_prefix:
            reason = prefix_ineligibility(
                self.gpu, self.dtype, nvcc_available=bool(shutil.which("nvcc"))
            )
            if reason is None:
                try:
                    bindings_present = find_spec("cuda.bindings") is not None
                except (ImportError, ValueError):
                    bindings_present = False
                if not bindings_present:
                    reason = "CUDA Python bindings unavailable; install the prefix extra"
            if reason:
                self.acceleration["reason"] = reason
                warnings.warn(f"Prefix V8 inactive; retaining stock engine: {reason}", stacklevel=2)
                config = replace(config, experimental_prefix=False)
                self.config = config
        draft = None
        if config.speculation == "dflash":
            if model.model != QWEN_INT4[0] or model.revision != QWEN_INT4[1]:
                raise CompatibilityError(
                    "The bundled DFlash preset supports only the pinned Qwen3.8 INT4 target"
                )
            from huggingface_hub import snapshot_download

            draft = snapshot_download(
                QWEN_DRAFT[0],
                revision=QWEN_DRAFT[1],
                allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"],
                token=False,
            )
        self.llm = LLM(**engine_kwargs(model, config, self.dtype, draft))
        self.tokenizer = self.llm.get_tokenizer()
        if config.experimental_prefix:
            installation = self.llm.collective_rpc("research_prefix_install")
            self.acceleration["installation"] = installation
            ready = bool(installation) and all(x.get("self_check_passed") for x in installation)
            if ready:
                self.llm.collective_rpc("research_prefix_enable", args=(True,))
            self.acceleration["enabled"] = ready
            if not ready:
                self.config = replace(config, experimental_prefix=False)
                self.acceleration["reason"] = (
                    "Prefix startup check did not pass; stock head retained"
                )
                warnings.warn(self.acceleration["reason"], stacklevel=2)

    def generate(self, prompt):
        if self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        ids = self.tokenizer.encode(
            prompt, add_special_tokens=not bool(self.tokenizer.chat_template)
        )
        if len(ids) + self.config.max_new_tokens > self.config.max_model_len:
            raise ValueError(
                "Prompt plus output cap exceeds max_model_len; input was not truncated"
            )
        params = self.SamplingParams(
            temperature=0,
            seed=self.config.seed,
            max_tokens=self.config.max_new_tokens,
            min_tokens=self.config.max_new_tokens if self.config.fixed_tokens else 0,
            ignore_eos=self.config.fixed_tokens,
            detokenize=False,
        )
        start = time.perf_counter()
        result = self.llm.generate([{"prompt_token_ids": ids}], params, use_tqdm=False)[0]
        elapsed = time.perf_counter() - start
        out = result.outputs[0]
        metrics = result.metrics
        if metrics is not None and getattr(metrics, "is_corrupted", False):
            raise RuntimeError("Engine reported nonfinite logits; result rejected")
        first, last = (
            getattr(metrics, "first_token_ts", None),
            getattr(metrics, "last_token_ts", None),
        )
        if not first or not last:
            first = last = None
        tokens = list(out.token_ids)
        return Output(
            self.tokenizer.decode(tokens, skip_special_tokens=True),
            tokens,
            Timing(len(tokens), elapsed, first, last),
            str(out.finish_reason),
        )

    def close(self):
        # vLLM lifecycle differs across versions; CLI runs this adapter in its own
        # process group so the supervisor also terminates workers on exit/timeout.
        shutdown = getattr(self.llm, "shutdown", None)
        if callable(shutdown):
            shutdown()


def create_backend(model, config):
    if config.backend == "transformers":
        return TransformersBackend(model, config)
    return VLLMBackend(model, config)
