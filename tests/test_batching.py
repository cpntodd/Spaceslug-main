import unittest

from spaceslug.batching import target_only_batches
from spaceslug.tokenizer import default_tokenizer


class TargetOnlyBatchTest(unittest.TestCase):
    def test_prompt_targets_are_masked_and_padded_deterministically(self):
        tokenizer = default_tokenizer()
        batches = target_only_batches([
            {"prompt": "Q: ", "target": "a"},
            {"prompt": "Q: longer ", "target": "b"},
        ], tokenizer, batch_size=2)
        self.assertEqual(len(batches), 1)
        batch = batches[0]
        self.assertEqual(len(batch.input_tokens[0]), len(batch.input_tokens[1]))
        self.assertEqual(batch.token_count, 4)  # UTF-8 target byte plus EOS for each record.
        first_mask = batch.loss_mask[0]
        self.assertTrue(all(not value for value in first_mask[:len(tokenizer.encode("Q: ", add_eos=False)) - 1]))
        self.assertTrue(any(first_mask))
        self.assertFalse(first_mask[-1])  # padding does not affect loss.


if __name__ == "__main__":
    unittest.main()
