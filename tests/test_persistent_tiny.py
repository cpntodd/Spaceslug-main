import unittest

from spaceslug.gpu_lora_training import PersistentTinyTrainer


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

    def readback_tiny_adapters(self, handle): return self.adapters
    def update_tiny_adapters(self, handle, values): self.adapters = values; self.calls.append("restore")
    def close_tiny_persistent(self, handle): self.calls.append("close")
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
