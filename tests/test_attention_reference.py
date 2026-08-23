import unittest

from spaceslug.attention_reference import causal_self_attention


class CausalAttentionReferenceTest(unittest.TestCase):
    def test_future_tokens_cannot_change_prior_outputs(self):
        original = [[1.0, 0.0], [0.0, 1.0], [3.0, 4.0]]
        changed_future = [[1.0, 0.0], [0.0, 1.0], [-30.0, 40.0]]
        before = causal_self_attention(original)
        after = causal_self_attention(changed_future)
        self.assertEqual(before[:2], after[:2])
        self.assertEqual(before[0], [1.0, 0.0])

    def test_rejects_ragged_input(self):
        with self.assertRaises(ValueError):
            causal_self_attention([[1.0], [1.0, 2.0]])


if __name__ == "__main__":
    unittest.main()
