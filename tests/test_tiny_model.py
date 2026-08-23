import tempfile
from pathlib import Path
import unittest

from spaceslug.tiny_model import TinyBigramModel


class TinyModelTest(unittest.TestCase):
    def test_training_is_deterministic_with_resumable_adamw(self):
        sequences = [[0, 1, 0, 1], [1, 0, 1, 0]]
        uninterrupted = TinyBigramModel.create(2)
        uninterrupted_state = {}
        for _ in range(12):
            uninterrupted.train_step(sequences, learning_rate=0.1, optimizer_state=uninterrupted_state)

        resumed = TinyBigramModel.create(2)
        resumed_state = {}
        for _ in range(5):
            resumed.train_step(sequences, learning_rate=0.1, optimizer_state=resumed_state)
        checkpoint_state = {
            "step": resumed_state["step"],
            "first": [row[:] for row in resumed_state["first"]],
            "second": [row[:] for row in resumed_state["second"]],
        }
        for _ in range(7):
            resumed.train_step(sequences, learning_rate=0.1, optimizer_state=checkpoint_state)

        self.assertEqual(resumed.weights, uninterrupted.weights)
        self.assertEqual(checkpoint_state, uninterrupted_state)

    def test_loss_decreases_and_checkpoint_reloads(self):
        sequences = [[0, 1, 0, 1], [0, 1, 0, 1]]
        model = TinyBigramModel.create(2)
        before, _ = model.loss_and_gradients(sequences)
        optimizer_state = {}
        for _ in range(20):
            model.train_step(sequences, learning_rate=0.2, optimizer_state=optimizer_state)
        after, _ = model.loss_and_gradients(sequences)
        self.assertLess(after, before)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "tiny.json"
            model.save(checkpoint)
            restored = TinyBigramModel.load(checkpoint)
            self.assertEqual(restored.weights, model.weights)
            self.assertEqual(restored.loss_and_gradients(sequences)[0], after)


if __name__ == "__main__":
    unittest.main()
