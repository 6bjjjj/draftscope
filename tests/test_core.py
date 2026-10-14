from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
import contextlib
import io
import json
import math
import tempfile
import unittest
from unittest.mock import patch

from draftscope.backends import engine_kwargs
from draftscope.cli import main, stop_process
from draftscope.config import GenerationConfig, ModelSpec, QWEN_BF16, QWEN_INT4
from draftscope.errors import CompatibilityError
from draftscope.experimental.guards import accepted_prefix, runtime_settings
from draftscope.hardware import (
    GPU,
    choose_dtype,
    cuda_architecture,
    detect_gpu,
    prefix_ineligibility,
    validate_backend,
)
from draftscope.metrics import Timing, summarize
from draftscope.storage import create_run, sha256, verify_manifest, write_json


def gpu(**kwargs):
    value = GPU(
        0, "NVIDIA A100-SXM4-80GB", (8, 0), 80 * 1024**3, True, "2.13.0+cu130", "13.0", "Linux"
    )
    return replace(value, **kwargs)


class ConfigurationTests(unittest.TestCase):
    def test_remote_revisions_are_immutable(self):
        with self.assertRaises(ValueError):
            ModelSpec("owner/model", "main")
        with self.assertRaises(ValueError):
            ModelSpec(tokenizer="owner/tokenizer")
        self.assertEqual(ModelSpec().revision, QWEN_BF16[1])

    def test_local_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(ModelSpec(folder, "local").model, folder)

    def test_invalid_generation_parameters(self):
        cases = [
            dict(device_index=-1),
            dict(device_index=True),
            dict(max_new_tokens=0),
            dict(max_model_len=0),
            dict(max_model_len=1.5),
            dict(max_new_tokens=4096),
            dict(memory_fraction=math.nan),
            dict(memory_fraction=1),
            dict(seed=-1),
            dict(draft_tokens=17),
            dict(fixed_tokens=1),
            dict(speculation="dflash"),
            dict(experimental_prefix=True),
            dict(backend="cpu"),
        ]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                GenerationConfig(**case)

    def test_pinned_dflash_arguments(self):
        config = GenerationConfig(backend="vllm", speculation="dflash")
        model = ModelSpec(*QWEN_INT4, tokenizer=QWEN_BF16[0], tokenizer_revision=QWEN_BF16[1])
        kwargs = engine_kwargs(model, config, "bfloat16", "/pinned/draft")
        self.assertEqual(kwargs["max_num_seqs"], 1)
        self.assertEqual(kwargs["cpu_offload_gb"], 0)
        self.assertFalse(kwargs["enable_prefix_caching"])
        self.assertFalse(kwargs["trust_remote_code"])
        self.assertEqual(kwargs["spec_tokens"], 7)
        self.assertEqual(kwargs["tokenizer_revision"], QWEN_BF16[1])
        self.assertNotIn("worker_extension_cls", kwargs)
        with self.assertRaises(ValueError):
            engine_kwargs(model, config, "bfloat16")

    def test_prefix_not_allowed_on_unvalidated_target(self):
        config = GenerationConfig(backend="vllm", speculation="dflash", experimental_prefix=True)
        with self.assertRaises(CompatibilityError):
            engine_kwargs(ModelSpec(), config, "bfloat16", "/draft")
        with self.assertRaises(CompatibilityError):
            engine_kwargs(ModelSpec(*QWEN_INT4), config, "float16", "/draft")


class HardwareTests(unittest.TestCase):
    def fake_torch(self, cap=(8, 0), available=True, hip=None):
        cuda = SimpleNamespace(
            is_available=lambda: available,
            device_count=lambda: 1,
            get_device_properties=lambda _: SimpleNamespace(
                name="NVIDIA test", total_memory=8 * 1024**3
            ),
            get_device_capability=lambda _: cap,
        )
        return SimpleNamespace(
            cuda=cuda, version=SimpleNamespace(hip=hip, cuda="13.0"), __version__="2.13.0"
        )

    def test_architecture_routing(self):
        for cap, dtype in [
            ((7, 5), "float16"),
            ((8, 0), "bfloat16"),
            ((8, 9), "bfloat16"),
            ((9, 0), "bfloat16"),
            ((12, 0), "bfloat16"),
        ]:
            with self.subTest(cap=cap):
                detected = detect_gpu(torch_module=self.fake_torch(cap))
                self.assertEqual(choose_dtype(detected), dtype)

    def test_no_cpu_or_rocm_fallback(self):
        for fake in [self.fake_torch(available=False), self.fake_torch(hip="6.0")]:
            with self.assertRaises(CompatibilityError):
                detect_gpu(torch_module=fake)
        with self.assertRaises(CompatibilityError):
            detect_gpu(1, torch_module=self.fake_torch())

    def test_old_card_bf16_rejected(self):
        with self.assertRaises(CompatibilityError):
            choose_dtype(gpu(capability=(7, 5), bf16=False), "bfloat16")

    def test_vllm_platform_and_capability(self):
        for value in [gpu(system="Windows"), gpu(capability=(7, 0))]:
            with self.assertRaises(CompatibilityError):
                validate_backend(value, "vllm")
        validate_backend(gpu(system="Windows"), "transformers")

    def test_prefix_guard(self):
        validate_backend(gpu(), "vllm", True)
        for name, cap in [
            ("RTX 3090", (8, 6)),
            ("RTX 4090", (8, 9)),
            ("H100", (9, 0)),
            ("B200", (10, 0)),
            ("RTX 5090", (12, 0)),
        ]:
            validate_backend(
                gpu(name=name, capability=cap, memory_bytes=24 * 1024**3), "vllm", True
            )
            self.assertEqual(cuda_architecture(cap), f"sm_{cap[0]}{cap[1]}")
        for value in [
            gpu(name="T4", capability=(7, 5), bf16=False),
            gpu(cuda_version="12.9"),
            gpu(torch_version="2.12.0"),
        ]:
            with self.assertRaises(CompatibilityError):
                validate_backend(value, "vllm", True)
        self.assertIsNotNone(prefix_ineligibility(gpu(), "float16"))
        self.assertIsNotNone(prefix_ineligibility(gpu(), "bfloat16", nvcc_available=False))

    def test_speculative_prefix_stops_at_first_rejection(self):
        self.assertEqual(accepted_prefix([1, 2, 3], [1, 2]), [1, 2, 3])
        self.assertEqual(accepted_prefix([1, 8, 3], [1, 2]), [1, 8])
        self.assertEqual(accepted_prefix([9, 8, 3], [1, 2]), [9])
        with self.assertRaises(ValueError):
            accepted_prefix([1], [1])


class MetricTests(unittest.TestCase):
    def test_request_and_decode_have_distinct_denominators(self):
        timing = Timing(101, 2, 10, 11)
        self.assertEqual(timing.request_tokens_per_second, 50.5)
        self.assertEqual(timing.decode_tokens_per_second, 100)
        self.assertIsNone(Timing(1, 1, 10, 10).decode_tokens_per_second)
        self.assertIsNone(Timing(50, 2).decode_tokens_per_second)

    def test_invalid_timings_rejected(self):
        cases = [
            (-1, 1),
            (True, 1),
            (1, 0),
            (1, math.nan),
            (1, math.inf),
            (2, 1, 10, 10),
            (2, 1, 10, 9),
            (1, 1, 10, None),
        ]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                Timing(*case)

    def test_summary_recomputes_not_trusts_saved_rate(self):
        row = Timing(10, 2).to_dict()
        row["request_tokens_per_second"] = 500
        summary = summarize([row, Timing(30, 3).to_dict()])
        self.assertEqual(summary["pooled_request_tokens_per_second"], 8)
        self.assertIsNone(summary["quality_accepted"])
        self.assertIsNone(summary["median_decode_tokens_per_second"])


class StorageTests(unittest.TestCase):
    def test_utf8_atomic_write_and_no_nan(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.json"
            write_json(path, {"text": "اختبار"})
            before = path.read_bytes()
            self.assertEqual(json.loads(before)["text"], "اختبار")
            with self.assertRaises(ValueError):
                write_json(path, {"value": math.nan})
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(folder).glob("*.tmp")), [])

    def test_exclusive_runs_and_integrity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = create_run(Path(folder) / "run")
            with self.assertRaises(FileExistsError):
                create_run(root)
            write_json(root / "x.json", {"x": 1})
            self.assertEqual(verify_manifest(root, {"x.json": sha256(root / "x.json")}), 1)
            with self.assertRaises(ValueError):
                verify_manifest(root, {"x.json": "bad"})
            with self.assertRaises(ValueError):
                verify_manifest(root, {"../outside": "bad"})


class DeadlineTests(unittest.TestCase):
    def test_extension_requires_explicit_finite_budget(self):
        now = datetime(2026, 10, 14, tzinfo=timezone.utc)
        env = {
            "DRAFTSCOPE_CACHE_DIR": str(Path.cwd() / "build"),
            "DRAFTSCOPE_DEADLINE_UTC": (now + timedelta(minutes=1)).isoformat(),
        }
        self.assertEqual(runtime_settings(env, now=now)[1], now + timedelta(minutes=1))
        for changes in [
            {"DRAFTSCOPE_DEADLINE_UTC": now.isoformat()},
            {"DRAFTSCOPE_DEADLINE_UTC": "2026-10-14T00:01:00"},
            {"DRAFTSCOPE_CACHE_DIR": "relative"},
            {"DRAFTSCOPE_DEADLINE_UTC": (now + timedelta(days=2)).isoformat()},
        ]:
            with self.assertRaises(CompatibilityError):
                runtime_settings({**env, **changes}, now=now)
        with self.assertRaises(CompatibilityError):
            runtime_settings({}, now=now)
        with self.assertRaises(CompatibilityError):
            runtime_settings(env, now=now, optimized=True)

    def test_group_children_killed_after_leader_exits(self):
        process = SimpleNamespace(pid=123, wait=lambda **kw: 0, poll=lambda: 0)
        with (
            patch("draftscope.cli.os.name", "posix"),
            patch("draftscope.cli.os.killpg", create=True) as kill,
            patch("draftscope.cli.signal.SIGKILL", 9, create=True),
        ):
            stop_process(process)
        self.assertEqual(kill.call_count, 2)


class CLITests(unittest.TestCase):
    def test_partial_and_supervisor_timeout_are_not_reported_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_json(root / "timing.json", [Timing(2, 1).to_dict()])
            for expected in ["incomplete", "timeout"]:
                if expected == "timeout":
                    write_json(root / "completion.json", {"status": "complete"})
                    write_json(root / "supervisor.json", {"status": "timeout"})
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(main(["analyze", folder]), 0)
                self.assertEqual(json.loads(output.getvalue())["run_status"], expected)

    def test_plan_never_detects_gpu(self):
        output = io.StringIO()
        with (
            patch("draftscope.cli.detect_gpu", side_effect=AssertionError("GPU touched")),
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(main(["plan"]), 0)
        self.assertFalse(json.loads(output.getvalue())["gpu_tested"])

    def test_custom_model_requires_own_revision(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["plan", "--model", "other/model"]), 2)
            self.assertEqual(
                main(["plan", "--backend", "vllm", "--int4-dflash", "--revision", "a" * 40]), 2
            )


if __name__ == "__main__":
    unittest.main()
