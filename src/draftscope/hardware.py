"""NVIDIA-only device detection. No model download and no silent CPU fallback."""

from dataclasses import asdict, dataclass
import importlib
import platform

from .errors import CompatibilityError


@dataclass(frozen=True)
class GPU:
    index: int
    name: str
    capability: tuple[int, int]
    memory_bytes: int
    bf16: bool
    torch_version: str
    cuda_version: str
    system: str

    def to_dict(self):
        return asdict(self)


def detect_gpu(index=0, *, torch_module=None) -> GPU:
    try:
        torch = torch_module or importlib.import_module("torch")
    except ImportError as exc:
        raise CompatibilityError(
            "Install a CUDA-enabled PyTorch build before GPU inference"
        ) from exc
    if getattr(torch.version, "hip", None):
        raise CompatibilityError(
            "DraftScope currently supports NVIDIA CUDA GPUs only; ROCm is unsupported"
        )
    if not torch.cuda.is_available():
        raise CompatibilityError(
            "No usable NVIDIA CUDA device. Check the NVIDIA driver and PyTorch build"
        )
    count = torch.cuda.device_count()
    if type(index) is not int or not 0 <= index < count:
        raise CompatibilityError(f"GPU index {index} is unavailable; detected {count} device(s)")
    props = torch.cuda.get_device_properties(index)
    cap = tuple(torch.cuda.get_device_capability(index))
    # BF16 must be a native device capability, not an emulation claim.
    bf16 = cap[0] >= 8
    return GPU(
        index,
        props.name,
        cap,
        int(props.total_memory),
        bf16,
        str(torch.__version__),
        str(torch.version.cuda),
        platform.system(),
    )


def choose_dtype(gpu: GPU, requested="auto") -> str:
    if requested == "auto":
        return "bfloat16" if gpu.bf16 else "float16"
    if requested == "bfloat16" and not gpu.bf16:
        raise CompatibilityError(
            "Native BF16 requires an Ampere-or-newer compatible NVIDIA GPU; use float16"
        )
    if requested not in {"float16", "bfloat16", "float32"}:
        raise ValueError("Unsupported dtype")
    return requested


def validate_backend(gpu: GPU, backend: str, experimental_prefix=False):
    if backend == "vllm":
        if gpu.system != "Linux":
            raise CompatibilityError(
                "The pinned vLLM backend requires Linux; use transformers on Windows"
            )
        if gpu.capability < (7, 5):
            raise CompatibilityError(
                "The pinned vLLM path requires compute capability >=7.5; try transformers"
            )
    if experimental_prefix:
        reason = prefix_ineligibility(gpu, "bfloat16")
        if reason:
            raise CompatibilityError(reason)


def prefix_ineligibility(gpu: GPU, dtype: str, *, nvcc_available=True):
    """Capability checks, not a GPU-name whitelist or a validation claim."""
    if gpu.system != "Linux":
        return "Prefix V8 requires the Linux vLLM backend"
    if gpu.capability < (8, 0) or not gpu.bf16 or dtype != "bfloat16":
        return "Prefix V8 requires compute capability >=8.0 and BF16 activations"
    if gpu.torch_version.split("+")[0] != "2.13.0" or gpu.cuda_version != "13.0":
        return "Prefix V8 private APIs require PyTorch 2.13.0 / CUDA 13.0"
    if not nvcc_available:
        return "NVCC is unavailable; prefix V8 cannot compile its CUDA setter"
    return None


def cuda_architecture(capability):
    if len(capability) != 2 or any(type(x) is not int or x < 0 for x in capability):
        raise ValueError("Invalid CUDA compute capability")
    return f"sm_{capability[0]}{capability[1]}"
