"""Validated settings; optimized presets keep checkpoint revisions explicit."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
import re

QWEN_BF16 = ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
QWEN_INT4 = ("RedHatAI/Qwen3.8-27B-INT4", "91bd022d5b49442a868bc35008f6c21e1860edfa")
QWEN_DRAFT = ("incoai/Qwen3.8-27B-DFlash2", "015e795645c74b1a0eeef3b570031fb62e769bc5")


@dataclass(frozen=True)
class ModelSpec:
    model: str = QWEN_BF16[0]
    revision: str = QWEN_BF16[1]
    tokenizer: str | None = None
    tokenizer_revision: str | None = None

    def __post_init__(self):
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a local directory or Hugging Face repository ID")
        if not Path(self.model).is_dir() and not re.fullmatch(r"[0-9a-f]{40}", self.revision or ""):
            raise ValueError("Remote models require a pinned 40-character commit revision")
        if self.tokenizer and not Path(self.tokenizer).is_dir():
            if not re.fullmatch(r"[0-9a-f]{40}", self.tokenizer_revision or ""):
                raise ValueError("A separate remote tokenizer requires its own pinned revision")


@dataclass(frozen=True)
class GenerationConfig:
    backend: str = "transformers"
    device_index: int = 0
    dtype: str = "auto"
    max_new_tokens: int = 128
    max_model_len: int = 4096
    memory_fraction: float = 0.88
    seed: int = 39001
    speculation: str = "none"
    draft_tokens: int = 7
    experimental_prefix: bool = False
    fixed_tokens: bool = False

    def __post_init__(self):
        if self.backend not in {"transformers", "vllm"}:
            raise ValueError("backend must be transformers or vllm")
        if self.dtype not in {"auto", "float16", "bfloat16", "float32"}:
            raise ValueError("Unsupported dtype")
        if type(self.device_index) is not int or self.device_index < 0:
            raise ValueError("device_index must be a nonnegative integer")
        if type(self.max_model_len) is not int or self.max_model_len < 2:
            raise ValueError("max_model_len must be an integer >=2")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if type(self.fixed_tokens) is not bool or type(self.experimental_prefix) is not bool:
            raise ValueError("fixed_tokens and experimental_prefix must be booleans")
        if (
            type(self.max_new_tokens) is not int
            or not 1 <= self.max_new_tokens < self.max_model_len
        ):
            raise ValueError("max_new_tokens must be positive and smaller than max_model_len")
        if not isfinite(self.memory_fraction) or not 0 < self.memory_fraction <= 0.95:
            raise ValueError("memory_fraction must be in (0, 0.95]")
        if self.speculation not in {"none", "mtp", "dflash"}:
            raise ValueError("Unknown speculation method")
        if self.speculation != "none" and self.backend != "vllm":
            raise ValueError("Speculative presets require the vllm backend")
        if type(self.draft_tokens) is not int or not 1 <= self.draft_tokens <= 16:
            raise ValueError("draft_tokens must be between 1 and 16")
        if self.experimental_prefix and not (
            self.backend == "vllm" and self.speculation == "dflash" and self.draft_tokens == 7
        ):
            raise ValueError("Experimental prefix requires vllm, DFlash and exactly 7 draft tokens")
