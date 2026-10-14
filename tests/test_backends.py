from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import warnings

from draftscope.backends import TransformersBackend, VLLMBackend
from draftscope.config import GenerationConfig, ModelSpec, QWEN_BF16, QWEN_INT4
from draftscope.hardware import GPU


class BackendTests(unittest.TestCase):
    def test_transformers_chat_does_not_duplicate_special_tokens(self):
        backend = object.__new__(TransformersBackend)
        backend.config = GenerationConfig(max_new_tokens=2)
        backend.device = "cuda:0"
        backend.torch = SimpleNamespace(
            cuda=SimpleNamespace(synchronize=Mock()), inference_mode=nullcontext
        )

        class Inputs(dict):
            def to(self, _):
                return self

        inputs = Inputs(input_ids=SimpleNamespace(shape=(1, 3)))
        backend.tokenizer = Mock(chat_template="template", pad_token_id=0, eos_token_id=1)
        backend.tokenizer.return_value = inputs
        backend.tokenizer.decode.return_value = "done"
        generated = Mock()
        generated.__getitem__ = Mock(return_value=SimpleNamespace(tolist=lambda: [10, 11]))
        backend.model = SimpleNamespace(generate=Mock(return_value=generated))
        result = backend.generate("test")
        self.assertFalse(backend.tokenizer.call_args.kwargs["add_special_tokens"])
        self.assertEqual(result.token_ids, [10, 11])
        self.assertIsNone(result.timing.decode_tokens_per_second)
        self.assertEqual(backend.torch.cuda.synchronize.call_count, 2)
        backend.config = GenerationConfig(max_new_tokens=2, max_model_len=4)
        with self.assertRaisesRegex(ValueError, "not truncated"):
            backend.generate("too long")
        self.assertEqual(backend.model.generate.call_count, 1)

    def vllm(self, gpu, *, nvcc=True, passed=True):
        llm = Mock()
        llm.collective_rpc.return_value = [{"installed": passed, "self_check_passed": passed}]
        factory = Mock(return_value=llm)
        modules = {
            "vllm": SimpleNamespace(LLM=factory, SamplingParams=Mock()),
            "huggingface_hub": SimpleNamespace(
                snapshot_download=Mock(return_value="/pinned/draft")
            ),
        }
        config = GenerationConfig(backend="vllm", speculation="dflash", experimental_prefix=True)
        model = ModelSpec(*QWEN_INT4, tokenizer=QWEN_BF16[0], tokenizer_revision=QWEN_BF16[1])
        with (
            patch.dict("sys.modules", modules),
            patch("draftscope.backends.version", return_value="0.31.0"),
            patch("draftscope.backends.detect_gpu", return_value=gpu),
            patch("draftscope.backends.shutil.which", return_value="/nvcc" if nvcc else None),
            patch("draftscope.backends.find_spec", return_value=object()),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore")
            backend = VLLMBackend(model, config)
        return backend, factory, llm

    def test_unsupported_prefix_retains_same_target_and_stock_engine(self):
        device = GPU(0, "T4", (7, 5), 16 * 1024**3, False, "2.13.0", "13.0", "Linux")
        backend, factory, llm = self.vllm(device)
        self.assertFalse(backend.acceleration["enabled"])
        self.assertFalse(backend.config.experimental_prefix)
        self.assertEqual(factory.call_args.kwargs["model"], QWEN_INT4[0])
        self.assertNotIn("worker_extension_cls", factory.call_args.kwargs)
        llm.collective_rpc.assert_not_called()

    def test_hopper_can_qualify_without_a100_name(self):
        device = GPU(0, "H100", (9, 0), 80 * 1024**3, True, "2.13.0", "13.0", "Linux")
        backend, factory, llm = self.vllm(device)
        self.assertTrue(backend.acceleration["enabled"])
        self.assertIn("worker_extension_cls", factory.call_args.kwargs)
        self.assertEqual(llm.collective_rpc.call_args.args, ("research_prefix_enable",))
        backend, _, llm = self.vllm(
            replace(device, name="RTX 4090", capability=(8, 9)), passed=False
        )
        self.assertFalse(backend.acceleration["enabled"])
        self.assertEqual(llm.collective_rpc.call_count, 1)
        self.assertFalse(backend.config.experimental_prefix)

    def test_missing_nvcc_does_not_break_stock_inference(self):
        device = GPU(0, "H100", (9, 0), 80 * 1024**3, True, "2.13.0", "13.0", "Linux")
        backend, factory, _ = self.vllm(device, nvcc=False)
        self.assertFalse(backend.acceleration["enabled"])
        self.assertIn("NVCC", backend.acceleration["reason"])
        self.assertNotIn("worker_extension_cls", factory.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
