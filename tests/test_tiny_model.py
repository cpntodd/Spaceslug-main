import tempfile
from pathlib import Path
import unittest

from spaceslug.tiny_model import TinyBigramModel


class TinyModelTest(unittest.TestCase):
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
