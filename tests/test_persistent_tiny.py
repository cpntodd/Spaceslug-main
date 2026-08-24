import json
import tempfile
import unittest

from spaceslug.gpu_lora_training import PersistentTinyTrainer, persistent_tiny_capability


class _Backend:
    def __init__(self):
        self.calls = []
        self.handle = object()
        self.adapters = [[float(i)] * size for i, size in enumerate([256, 256, 256, 256, 256, 256, 256, 256])]

    def create_tiny_persistent_full(self, model, adapter):
        self.calls.append("create")
        return self.handle

    def begin_tiny_accumulation(self, handle): self.calls.append("begin")

    def accumulate_tiny_backward(self, handle, token, position, target, mask):
        self.calls.append(("backward", token, position, target, mask))
        return {"loss": [float(position)]}

    def finalize_tiny_sgd(self, handle, learning_rate, normalizer):
        self.calls.append(("finalize", learning_rate, normalizer))

    def accumulate_tiny_windows(self, handle, tokens, targets, mask, window_length):
        self.calls.append(("windows", tokens, targets, mask, window_length))
        return [float(i) for i in range(len(tokens))]

    def readback_tiny_adapters(self, handle): return self.adapters
    def update_tiny_adapters(self, handle, values): self.adapters = values; self.calls.append("restore")
    def close_tiny_persistent(self, handle): self.calls.append("close")
    def execute_tiny_fixed_retained_forward(self, handle, tokens):
        self.calls.append(("fixed_forward", len(tokens)))
        class Result:
            status, operation = "ok", "tiny_forward_fixed_retained"
            output, metrics = {"logits": [0.0] * (128 * 259)}, {"fixed_tokens": 128}
        return Result()
    def begin_tiny_adamw(self, handle): self.calls.append("adamw_begin")
    def accumulate_tiny_adamw(self, handle, token, position, target, mask):
        self.calls.append(("adamw_backward", token, position, target, mask)); return {"loss": [float(position)]}
    def finalize_tiny_adamw(self, handle, learning_rate, beta1, beta2, epsilon, weight_decay, normalizer):
        self.calls.append(("adamw_finalize", learning_rate, beta1, beta2, epsilon, weight_decay, normalizer))
    def readback_tiny_adamw_state(self, handle): return {"adapters": [0.0] * 2048, "m": [0.0] * 2048, "v": [0.0] * 2048, "step": 1}
    def restore_tiny_adamw_state(self, handle, state): self.calls.append("adamw_restore")


class PersistentTinyTrainerTest(unittest.TestCase):
    def test_fixed_contract_accumulates_then_finalizes(self):
        class Model: hidden_size, vocab_size = 64, 259
        class Adapter: hidden_size, rank = 64, 4
        backend = _Backend()
        trainer = PersistentTinyTrainer(backend, Model(), Adapter(), 0.25)
        report = trainer.train_tokens([3, 4], [5, 6], [1, 0])
        self.assertEqual(report["step"], 1)
        self.assertEqual(backend.calls[1], "begin")
        self.assertEqual(backend.calls[-1], ("finalize", 0.25, 1.0))
        self.assertEqual(report["loss"], [0.0, 1.0])
        trainer.close()

    def test_native_adamw_selection_and_state_roundtrip(self):
        class Model: hidden_size, vocab_size = 64, 259
        class Adapter: hidden_size, rank = 64, 4
        backend = _Backend(); trainer = PersistentTinyTrainer(backend, Model(), Adapter(), 0.25, optimizer="adamw", weight_decay=0.01)
        report = trainer.train_tokens_adamw([1, 2], [3, 4], [1, 1])
        self.assertEqual(report["optimizer"], "adamw")
        self.assertIn(("adamw_finalize", 0.25, 0.9, 0.999, 1e-8, 0.01, 2.0), backend.calls)
        trainer.restore_optimizer_state(trainer.readback_optimizer_state())
        self.assertIn("adamw_restore", backend.calls)
        trainer.close()

    def test_fixed_windows_use_one_backend_batch_call(self):
        class Model: hidden_size, vocab_size = 64, 259
        class Adapter: hidden_size, rank = 64, 4
        backend = _Backend()
        backend.accumulate_tiny_windows = lambda handle, tokens, targets, mask, window_length: (backend.calls.append(("windows", tokens, targets, mask, window_length)) or [float(i) for i in range(len(tokens))])
        trainer = PersistentTinyTrainer(backend, Model(), Adapter(), 0.25)
        report = trainer.train_windows([1, 2, 3, 4], [5, 6, 7, 8], 2, [1, 1, 0, 1])
        self.assertEqual(report["loss"], [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(report["windows"], 2)
        self.assertIn(("windows", [1, 2, 3, 4], [5, 6, 7, 8], [1, 1, 0, 1], 2), backend.calls)
        trainer.close()

    def test_deterministic_bounded_streaming_and_resume_metadata(self):
        class Model: hidden_size, vocab_size = 64, 259
        class Adapter: hidden_size, rank = 64, 4
        backend = _Backend(); trainer = PersistentTinyTrainer(backend, Model(), Adapter())
        report = trainer.train_windows(list(range(8)), list(range(10, 18)), 2, batch_windows=1, max_windows=2)
        self.assertEqual(report["window_position"], 2)
        self.assertFalse(report["dataset_device_resident"])
        with tempfile.TemporaryDirectory() as root:
            path = root + "/checkpoint.json"; trainer.checkpoint(path)
            with open(path) as checkpoint_file:
                payload = json.load(checkpoint_file)
            self.assertEqual(payload["training"]["sample_position"], 4)
            resumed = PersistentTinyTrainer.resume(_Backend(), Model(), Adapter(), path)
            self.assertEqual((resumed.sample_position, resumed.window_position), (4, 2))
            resumed.close()
        trainer.close()

    def test_adamw_window_positions_reset_for_each_window(self):
        class Model: hidden_size, vocab_size = 64, 259
        class Adapter: hidden_size, rank = 64, 4
        backend = _Backend()
        trainer = PersistentTinyTrainer(backend, Model(), Adapter(), optimizer="adamw")
        trainer.train_windows([1, 2, 3, 4], [5, 6, 7, 8], 2, batch_windows=2)
        positions = [call[2] for call in backend.calls if isinstance(call, tuple) and call[0] == "adamw_backward"]
        self.assertEqual(positions, [0, 1, 0, 1])
        trainer.close()

    def test_adamw_windows_stage_tokens_and_checkpoint_state(self):
        class Model: hidden_size, vocab_size = 64, 259
        class Adapter: hidden_size, rank = 64, 4
        backend = _Backend(); trainer = PersistentTinyTrainer(backend, Model(), Adapter(), 0.25, optimizer="adamw", weight_decay=0.01)
        report = trainer.train_windows([1, 2, 3, 4], [5, 6, 7, 8], 2, [1, 0, 1, 1], batch_windows=1)
        self.assertEqual(report["optimizer"], "adamw")
        self.assertTrue(report["host_staging"])
        self.assertEqual(report["window_position"], 2)
        self.assertIn(("adamw_finalize", 0.25, 0.9, 0.999, 1e-8, 0.01, 3.0), backend.calls)
        with tempfile.TemporaryDirectory() as root:
            path = root + "/adamw.json"; trainer.checkpoint(path)
            with open(path) as checkpoint_file:
                payload = json.load(checkpoint_file)
            self.assertIn("optimizer_state", payload)
            resumed = PersistentTinyTrainer.resume(_Backend(), Model(), Adapter(), path)
            self.assertEqual((resumed.step_index, resumed.window_position, resumed.sample_position), (1, 2, 4))
            self.assertIn("adamw_restore", resumed.backend.calls)
            resumed.close()
        trainer.close()

    def test_adamw_stream_resume_matches_uninterrupted_window_positions(self):
        class Model: hidden_size, vocab_size = 64, 259
        class Adapter: hidden_size, rank = 64, 4
        data = (list(range(8)), list(range(10, 18)), [1] * 8)
        full_backend = _Backend(); full = PersistentTinyTrainer(full_backend, Model(), Adapter(), optimizer="adamw")
        full.train_windows(data[0], data[1], 2, data[2], batch_windows=1)
        split_backend = _Backend(); split = PersistentTinyTrainer(split_backend, Model(), Adapter(), optimizer="adamw")
        split.train_windows(data[0], data[1], 2, data[2], batch_windows=1, max_windows=2)
        with tempfile.TemporaryDirectory() as root:
            path = root + "/stream.json"; split.checkpoint(path)
            resumed = PersistentTinyTrainer.resume(_Backend(), Model(), Adapter(), path)
            resumed.train_windows(data[0], data[1], 2, data[2], batch_windows=1)
            self.assertEqual(resumed.window_position, full.window_position)
            self.assertEqual(resumed.sample_position, full.sample_position)
            resumed.close()
        split.close(); full.close()

    def test_fixed_forward_requires_exactly_128_tokens_and_reports_metadata(self):
        class Model: hidden_size, vocab_size = 64, 259
        class Adapter: hidden_size, rank = 64, 4
        backend = _Backend()
        trainer = PersistentTinyTrainer(backend, Model(), Adapter())
        with self.assertRaises(ValueError): trainer.fixed_forward([1] * 127)
        with self.assertRaises(ValueError): trainer.fixed_forward([1] * 129)
        report = trainer.fixed_forward([1] * 128)
        self.assertTrue(report["fixed_forward_retention"])
        self.assertFalse(report["production_training"])
        self.assertIn(("fixed_forward", 128), backend.calls)
        trainer.close()

    def test_persistent_tiny_capability_metadata(self):
        capability = persistent_tiny_capability()
        self.assertTrue(capability["fixed_forward_retention"])
        self.assertEqual(capability["fixed_forward_tokens"], 128)
        self.assertFalse(capability["production_training"])
        self.assertEqual(capability["optimizers"], ["sgd", "adamw"])
        self.assertEqual(capability["production_status"], "bounded")
        self.assertTrue(capability["immutable_command_buffer_reuse_prototype"])
        self.assertTrue(capability["native_adamw_state_checkpoint"])
        self.assertFalse(capability["dataset_device_resident"])

    def test_streaming_capability_boundaries_are_explicit(self):
        class Model: hidden_size, vocab_size = 64, 259
        class Adapter: hidden_size, rank = 64, 4
        with self.assertRaises(ValueError):
            list(PersistentTinyTrainer.iter_window_batches([1, 2], [3, 4], 2, max_windows=-1))
        trainer = PersistentTinyTrainer(_Backend(), Model(), Adapter(), optimizer="adamw")
        report = trainer.train_windows([1, 2], [3, 4], 2)
        self.assertEqual(report["optimizer"], "adamw")
        trainer.close()

    def test_unsupported_shape_is_explicit(self):
        class Model: hidden_size, vocab_size = 32, 259
        class Adapter: hidden_size, rank = 32, 4
        with self.assertRaises(ValueError): PersistentTinyTrainer(_Backend(), Model(), Adapter())

    def test_restore_and_readback_are_exposed(self):
        class Model: hidden_size, vocab_size = 64, 259
        class Adapter: hidden_size, rank = 64, 4
        backend = _Backend(); trainer = PersistentTinyTrainer(backend, Model(), Adapter())
        values = trainer.readback_adapter(); trainer.restore_adapter(values)
        self.assertIn("restore", backend.calls)
        trainer.close()


if __name__ == "__main__": unittest.main()
