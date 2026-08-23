import unittest

from spaceslug.batching import target_only_batches
from spaceslug.tiny_attention_model import TinyAttentionModel
from spaceslug.tokenizer import default_tokenizer


class TinyAttentionModelTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = default_tokenizer()
        self.batch = target_only_batches([
            {"prompt": "Q: ", "target": "a"}, {"prompt": "Q: ", "target": "b"},
        ], self.tokenizer, batch_size=2)[0]

    def test_attention_scale_gradient_matches_centered_difference(self):
        model = TinyAttentionModel(self.tokenizer.vocab_size, hidden_size=3)
        epsilon = 1e-5
        _, analytic = model.loss_and_gradient(self.batch)
        original = model.attention_scale
        model.attention_scale = original + epsilon
        plus, _ = model.loss_and_gradient(self.batch)
        model.attention_scale = original - epsilon
        minus, _ = model.loss_and_gradient(self.batch)
        model.attention_scale = original
        self.assertAlmostEqual(analytic, (plus - minus) / (2.0 * epsilon), delta=1e-7)

    def test_masked_targets_train_attention_parameter(self):
        model = TinyAttentionModel(self.tokenizer.vocab_size, hidden_size=3)
        before, _ = model.loss_and_gradient(self.batch)
        for _ in range(100):
            model.train_step(self.batch, 0.5)
        after, _ = model.loss_and_gradient(self.batch)
        self.assertLess(after, before)


if __name__ == "__main__":
    unittest.main()
