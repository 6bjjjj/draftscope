"""A bounded benchmark writer with recoverable partial outputs."""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import time

from .backends import create_backend
from .config import GenerationConfig, ModelSpec
from .metrics import summarize
from .storage import create_run, write_json


def run(
    model: ModelSpec,
    config: GenerationConfig,
    prompts,
    output,
    *,
    warmups=2,
    repeats=5,
    max_seconds=600,
    backend_factory=create_backend,
):
    if (
        not isinstance(prompts, list)
        or not prompts
        or not all(isinstance(x, str) and x.strip() for x in prompts)
    ):
        raise ValueError("prompts must be a nonempty list of nonempty strings")
    if type(repeats) is not int or repeats < 1 or type(warmups) is not int or warmups < 0:
        raise ValueError("repeats must be positive and warmups nonnegative")
    if not 0 < max_seconds <= 86400:
        raise ValueError("max_seconds must be in (0, 86400]")
    folder = create_run(output)
    started = time.monotonic()
    records, backend = [], None
    protocol = {
        "schema": 1,
        "model": asdict(model),
        "config": asdict(config),
        "batch": 1,
        "prompts": prompts,
        "input_sha256": hashlib.sha256(
            json.dumps(prompts, ensure_ascii=False).encode()
        ).hexdigest(),
        "warmups": warmups,
        "repeats": repeats,
        "max_seconds": max_seconds,
        "utc": datetime.now(timezone.utc).isoformat(),
        "quality_accepted": None,
        "timing_scope": "Pretokenized generation wall time. Includes prefill and backend overhead; excludes model load, warmups, tokenization and detokenization.",
    }
    write_json(folder / "protocol.json", protocol)
    status, error = "failed", None

    def budget():
        if time.monotonic() - started >= max_seconds:
            raise TimeoutError("Run time budget exhausted; partial outputs preserved")

    try:
        budget()
        backend = backend_factory(model, config)
        write_json(folder / "hardware.json", backend.gpu.to_dict())
        write_json(
            folder / "startup.json",
            {
                "seconds": time.monotonic() - started,
                "dtype": backend.dtype,
                "effective_config": asdict(getattr(backend, "config", config)),
                "acceleration": getattr(
                    backend, "acceleration", {"requested": False, "enabled": False}
                ),
            },
        )
        for index, prompt in enumerate(prompts):
            for _ in range(warmups):
                budget()
                backend.generate(prompt)
            reference = None
            for repeat in range(repeats):
                budget()
                result = backend.generate(prompt)
                if len(result.token_ids) != result.timing.generated_tokens:
                    raise RuntimeError("Timing token count differs from generated token IDs")
                if config.fixed_tokens and len(result.token_ids) != config.max_new_tokens:
                    raise RuntimeError("Fixed-length run returned an unexpected token count")
                if reference is None:
                    reference = result.token_ids
                row = {
                    "prompt_index": index,
                    "repeat": repeat,
                    "text": result.text,
                    "token_ids": result.token_ids,
                    "finish_reason": result.finish_reason,
                    "own_sequence_equal": result.token_ids == reference,
                    **result.timing.to_dict(),
                }
                records.append(row)
                write_json(folder / "timing.json", records)
                if not row["own_sequence_equal"]:
                    raise RuntimeError(
                        "Repeated greedy outputs changed; speed admission is inconclusive"
                    )
        budget()
        write_json(folder / "summary.json", summarize(records))
        status = "complete"
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        write_json(folder / "error.json", error)
        raise
    finally:
        cleanup_error = None
        try:
            if backend is not None:
                backend.close()
        except BaseException as exc:
            cleanup_error = {"type": type(exc).__name__, "message": str(exc)}
            if error is None:
                status = "failed"
                write_json(folder / "error.json", cleanup_error)
                raise
        finally:
            write_json(
                folder / "completion.json",
                {
                    "status": status,
                    "records": len(records),
                    "seconds": time.monotonic() - started,
                    "quality_accepted": None,
                    "error": error,
                    "cleanup_error": cleanup_error,
                },
            )
    return folder
