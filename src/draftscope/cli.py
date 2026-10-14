"""Only an explicit benchmark command may load/download model weights."""

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

from . import __version__
from .config import GenerationConfig, ModelSpec, QWEN_BF16, QWEN_INT4
from .errors import DraftScopeError
from .hardware import choose_dtype, detect_gpu, validate_backend
from .metrics import summarize
from .storage import write_json


def stop_process(process):
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    elif process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name != "posix":
            process.kill()
    finally:
        if os.name == "posix":
            # The group can outlive its leader. Reap GPU children even when the
            # main process has exited already or handled SIGTERM immediately.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.poll() is None:
            process.wait(timeout=10)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="draftscope")
    parser.add_argument("--version", action="version", version=__version__)
    subs = parser.add_subparsers(dest="command", required=True)
    doctor = subs.add_parser("doctor", help="Check the selected NVIDIA device; no model download")
    doctor.add_argument("--device-index", type=int, default=0)
    plan = subs.add_parser("plan", help="Print settings without CUDA or network access")
    bench = subs.add_parser("benchmark", help="Run a bounded batch-one inference experiment")
    for p in (plan, bench):
        p.add_argument("--backend", choices=["transformers", "vllm"], default="transformers")
        p.add_argument("--model", default=QWEN_BF16[0])
        p.add_argument("--revision", default=None)
        p.add_argument("--int4-dflash", action="store_true")
        p.add_argument("--device-index", type=int, default=0)
        p.add_argument(
            "--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"]
        )
        p.add_argument("--max-new-tokens", type=int, default=128)
        p.add_argument("--max-model-len", type=int, default=4096)
        p.add_argument("--memory-fraction", type=float, default=0.88)
        p.add_argument("--fixed-tokens", action="store_true")
        p.add_argument("--experimental-prefix", action="store_true")
    bench.add_argument("--prompts", type=Path, required=True, help="UTF-8 JSON list of prompts")
    bench.add_argument(
        "--output",
        type=Path,
        required=True,
        help="A new directory; existing runs are never overwritten",
    )
    bench.add_argument("--max-seconds", type=int, default=600)
    bench.add_argument("--warmups", type=int, default=2)
    bench.add_argument("--repeats", type=int, default=5)
    analyze = subs.add_parser("analyze", help="Recompute timing arithmetic on CPU")
    analyze.add_argument("results", type=Path)
    worker = subs.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("settings", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            gpu = detect_gpu(args.device_index)
            print(
                json.dumps(
                    {
                        **gpu.to_dict(),
                        "auto_dtype": choose_dtype(gpu),
                        "note": "Detection is not proof of model fit or GPU integration correctness.",
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "analyze":
            records = json.loads((args.results / "timing.json").read_text(encoding="utf-8"))
            completion_path = args.results / "completion.json"
            completion = (
                json.loads(completion_path.read_text(encoding="utf-8"))
                if completion_path.exists()
                else {"status": "incomplete"}
            )
            supervisor = args.results / "supervisor.json"
            if supervisor.exists():
                state = json.loads(supervisor.read_text(encoding="utf-8"))
                if state.get("status") == "timeout":
                    completion["status"] = "timeout"
            print(json.dumps({**summarize(records), "run_status": completion["status"]}, indent=2))
            return 0
        if args.command == "_worker":
            from .runner import run

            value = json.loads(args.settings.read_text(encoding="utf-8"))
            run(
                ModelSpec(**value["model"]),
                GenerationConfig(**value["config"]),
                value["prompts"],
                value["output"],
                warmups=value["warmups"],
                repeats=value["repeats"],
                max_seconds=value["max_seconds"],
            )
            return 0
        if args.model != QWEN_BF16[0] and not Path(args.model).is_dir() and args.revision is None:
            raise ValueError("A custom remote --model requires its explicit --revision commit")
        model = ModelSpec(args.model, args.revision or QWEN_BF16[1])
        if args.int4_dflash:
            if args.model != QWEN_BF16[0] or args.revision is not None:
                raise ValueError(
                    "--int4-dflash is a pinned preset; do not combine it with custom model/revision"
                )
            if args.backend != "vllm":
                raise ValueError("--int4-dflash requires --backend vllm")
            model = ModelSpec(*QWEN_INT4, tokenizer=QWEN_BF16[0], tokenizer_revision=QWEN_BF16[1])
        config = GenerationConfig(
            backend=args.backend,
            device_index=args.device_index,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            max_model_len=args.max_model_len,
            memory_fraction=args.memory_fraction,
            speculation="dflash" if args.int4_dflash else "none",
            experimental_prefix=args.experimental_prefix,
            fixed_tokens=args.fixed_tokens,
        )
        if args.command == "plan":
            print(
                json.dumps(
                    {
                        "model": asdict(model),
                        "config": asdict(config),
                        "gpu_tested": False,
                        "quality_accepted": None,
                    },
                    indent=2,
                )
            )
            return 0
        if not 0 < args.max_seconds <= 86400:
            raise ValueError("max-seconds must be in (0, 86400]")
        if args.warmups < 0 or args.repeats < 1:
            raise ValueError("warmups must be nonnegative and repeats positive")
        if args.output.exists():
            raise FileExistsError("Output directory already exists; choose a new run")
        prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
        if (
            not isinstance(prompts, list)
            or not prompts
            or not all(isinstance(p, str) and p.strip() for p in prompts)
        ):
            raise ValueError("Prompt file must contain a nonempty JSON list of nonempty strings")
        gpu = detect_gpu(config.device_index)
        validate_backend(gpu, config.backend)
        choose_dtype(gpu, config.dtype)
        settings = {
            "model": asdict(model),
            "config": asdict(config),
            "prompts": prompts,
            "output": str(args.output.resolve()),
            "warmups": args.warmups,
            "repeats": args.repeats,
            "max_seconds": args.max_seconds,
        }
        env = os.environ.copy()
        env.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
        env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        env["DRAFTSCOPE_DEADLINE_UTC"] = (
            datetime.now(timezone.utc) + timedelta(seconds=args.max_seconds)
        ).isoformat()
        env["DRAFTSCOPE_CACHE_DIR"] = str(args.output.resolve() / "cuda_build")
        with tempfile.TemporaryDirectory(prefix="draftscope-") as temp:
            path = Path(temp) / "settings.json"
            write_json(path, settings)
            process = subprocess.Popen(
                [sys.executable, "-m", "draftscope", "_worker", str(path)],
                env=env,
                start_new_session=os.name == "posix",
            )
            try:
                return process.wait(timeout=args.max_seconds)
            except subprocess.TimeoutExpired:
                stop_process(process)
                write_json(
                    args.output / "supervisor.json",
                    {"status": "timeout", "max_seconds": args.max_seconds},
                )
                print("Time budget exhausted; existing partial files retained.", file=sys.stderr)
                return 124
            finally:
                # Also reap Linux GPU workers if the main worker exited first.
                stop_process(process)
    except (DraftScopeError, ValueError, OSError, ImportError) as exc:
        print(f"draftscope: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
