from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch

from draftscope.backends import Output
from draftscope.config import GenerationConfig, ModelSpec
from draftscope.metrics import Timing
from draftscope.runner import run


class FakeBackend:
    def __init__(self, *, divergent=False, broken=False, close_error=False, wrong_count=False):
        self.gpu = SimpleNamespace(to_dict=lambda: {"name": "CPU mock, not GPU evidence"})
        self.dtype, self.closed, self.calls = "bfloat16", False, 0
        self.divergent, self.broken = divergent, broken
        self.close_error, self.wrong_count = close_error, wrong_count

    def generate(self, prompt):
        self.calls += 1
        if self.broken and self.calls == 2:
            raise RuntimeError("simulated generation failure")
        ids = [1, self.calls if self.divergent else 2]
        return Output("اختبار", ids, Timing(3 if self.wrong_count else 2, 0.2))

    def close(self):
        self.closed = True
        if self.close_error:
            raise RuntimeError("cleanup failure")


class RunnerTests(unittest.TestCase):
    def invoke(self, folder, backend, **kwargs):
        return run(
            ModelSpec(),
            GenerationConfig(max_new_tokens=2),
            ["مرحبا"],
            folder,
            warmups=0,
            repeats=2,
            backend_factory=lambda *_: backend,
            **kwargs,
        )

    def test_complete_recoverable_run(self):
        with tempfile.TemporaryDirectory() as temp:
            backend = FakeBackend()
            folder = self.invoke(Path(temp) / "run", backend)
            completion = json.loads((folder / "completion.json").read_text())
            records = json.loads((folder / "timing.json").read_text(encoding="utf-8"))
            self.assertEqual(completion["status"], "complete")
            self.assertEqual(completion["records"], 2)
            self.assertIsNone(completion["quality_accepted"])
            self.assertEqual(records[0]["text"], "اختبار")
            self.assertTrue(backend.closed)

    def test_divergence_and_generation_error_preserve_partial(self):
        for backend, count in [(FakeBackend(divergent=True), 2), (FakeBackend(broken=True), 1)]:
            with self.subTest(count=count), tempfile.TemporaryDirectory() as temp:
                folder = Path(temp) / "run"
                with self.assertRaises(RuntimeError):
                    self.invoke(folder, backend)
                self.assertEqual(
                    len(json.loads((folder / "timing.json").read_text(encoding="utf-8"))), count
                )
                self.assertEqual(
                    json.loads((folder / "completion.json").read_text())["status"], "failed"
                )
                self.assertTrue(backend.closed)

    def test_cleanup_failure_not_reported_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "run"
            with self.assertRaisesRegex(RuntimeError, "cleanup failure"):
                self.invoke(folder, FakeBackend(close_error=True))
            self.assertEqual(
                json.loads((folder / "completion.json").read_text())["status"], "failed"
            )

    def test_cleanup_failure_does_not_mask_generation_error(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError, "generation failure"):
                self.invoke(Path(temp) / "run", FakeBackend(broken=True, close_error=True))

    def test_budget_exhaustion_before_model_load(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "run"
            with patch("draftscope.runner.time.monotonic", side_effect=[0, 2, 2]):
                with self.assertRaises(TimeoutError):
                    self.invoke(folder, FakeBackend(), max_seconds=1)
            self.assertEqual(json.loads((folder / "completion.json").read_text())["records"], 0)

    def test_wrong_token_count_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError, "token count"):
                self.invoke(Path(temp) / "run", FakeBackend(wrong_count=True))

    def test_string_is_not_a_prompt_list(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "not-created"
            with self.assertRaises(ValueError):
                run(ModelSpec(), GenerationConfig(), "not a list", folder)
            self.assertFalse(folder.exists())


if __name__ == "__main__":
    unittest.main()
