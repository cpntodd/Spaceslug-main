import tempfile
from pathlib import Path
import unittest

from spaceslug.tiny_artifact import load_tiny_artifact, write_tiny_artifact
from spaceslug.tiny_model import TinyBigramModel
from spaceslug.tokenizer import default_tokenizer


class TokenizerArtifactTest(unittest.TestCase):
    def test_byte_tokenizer_is_utf8_deterministic(self):
        tokenizer = default_tokenizer()
        tokens = tokenizer.encode("hello, λ")
        self.assertEqual(tokens, tokenizer.encode("hello, λ"))
        self.assertEqual(tokenizer.decode(tokens), "hello, λ")
        self.assertEqual(tokenizer.vocab_size, 259)

    def test_artifact_round_trip_and_checksum_rejection(self):
        tokenizer = default_tokenizer()
        model = TinyBigramModel.create(tokenizer.vocab_size)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "tiny.spaceslug"
            manifest = write_tiny_artifact(root, model, tokenizer)
            loaded_model, loaded_tokenizer, loaded_manifest = load_tiny_artifact(root)
            self.assertEqual(loaded_manifest["revision"], manifest["revision"])
            self.assertEqual(loaded_tokenizer.fingerprint(), tokenizer.fingerprint())
            self.assertEqual(loaded_model.weights, model.weights)
            (root / "tensors" / "weights.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_tiny_artifact(root)


if __name__ == "__main__":
    unittest.main()
