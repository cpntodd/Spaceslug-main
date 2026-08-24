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
