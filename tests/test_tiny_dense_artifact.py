import tempfile
from pathlib import Path
import unittest

from spaceslug.tiny_dense_artifact import load_dense_tiny_artifact, write_dense_tiny_artifact
from spaceslug.tiny_dense_model import TinyDenseCausalModel
from spaceslug.tokenizer import default_tokenizer


class DenseTinyArtifactTest(unittest.TestCase):
    def test_dense_artifact_round_trip_and_checksum_rejection(self):
        tokenizer = default_tokenizer()
        model = TinyDenseCausalModel.create(tokenizer.vocab_size, 4)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dense.spaceslug"
            manifest = write_dense_tiny_artifact(root, model, tokenizer)
            restored, restored_tokenizer, restored_manifest = load_dense_tiny_artifact(root)
            self.assertEqual(manifest["revision"], restored_manifest["revision"])
            self.assertEqual(restored.embedding, model.embedding)
            self.assertEqual(restored_tokenizer.fingerprint(), tokenizer.fingerprint())
            (root / "model.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_dense_tiny_artifact(root)


if __name__ == "__main__":
    unittest.main()
