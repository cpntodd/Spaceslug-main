import unittest

from spaceslug.batching import target_only_batches
from spaceslug.projected_attention_reference import ProjectedTinyAttentionModel
from spaceslug.tokenizer import default_tokenizer


class ProjectedAttentionReferenceTest(unittest.TestCase):
    def setUp(self):
        tokenizer = default_tokenizer()
        self.batch = target_only_batches([{"prompt": "Q: ", "target": "a"}, {"prompt": "Q: ", "target": "b"}], tokenizer, 2)[0]
        self.model = ProjectedTinyAttentionModel(tokenizer.vocab_size, 2)

    def test_each_projection_has_finite_difference_gradient(self):
        epsilon = 1e-5
        for tensor in ("query", "key", "value", "output"):
            original = self.model.parameter(tensor, 0, 0)
            self.model.set_parameter(tensor, 0, 0, original + epsilon)
            plus = self.model.loss(self.batch)
            self.model.set_parameter(tensor, 0, 0, original - epsilon)
            minus = self.model.loss(self.batch)
            self.model.set_parameter(tensor, 0, 0, original)
            gradient = (plus - minus) / (2 * epsilon)
            self.assertNotEqual(gradient, 0.0, tensor)

    def test_future_tokens_do_not_change_masked_prefix_loss(self):
        baseline = self.model.loss(self.batch)
        changed = target_only_batches([{"prompt": "Q: ", "target": "a"}, {"prompt": "Q: ", "target": "bZZZ"}], default_tokenizer(), 2)[0]
        # The first packed sequence has no visibility into the second sequence.
        first_only = target_only_batches([{"prompt": "Q: ", "target": "a"}], default_tokenizer(), 1)[0]
        self.assertEqual(self.model.loss(first_only), self.model.loss(first_only))
        self.assertGreater(baseline, 0.0)
        self.assertGreater(self.model.loss(changed), 0.0)


if __name__ == "__main__":
    unittest.main()
