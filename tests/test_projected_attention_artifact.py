import tempfile
from pathlib import Path
import unittest

from spaceslug.projected_attention_artifact import load_projected_artifact, write_projected_artifact
from spaceslug.projected_attention_reference import ProjectedTinyAttentionModel
from spaceslug.tokenizer import default_tokenizer


class ProjectedAttentionArtifactTest(unittest.TestCase):
    def test_artifact_is_created_with_checksummed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_projected_artifact(Path(directory) / "tiny.spaceslug", ProjectedTinyAttentionModel(259, 2), default_tokenizer())
            self.assertEqual(manifest["architecture"], "spaceslug-tiny-projected-attention-cpu-reference")
            self.assertTrue(manifest["files"])
            self.assertTrue(manifest["revision"].startswith("sha256:"))
            restored, restored_tokenizer, restored_manifest = load_projected_artifact(Path(directory) / "tiny.spaceslug")
            self.assertEqual(restored.query, ProjectedTinyAttentionModel(259, 2).query)
            self.assertEqual(restored_tokenizer.fingerprint(), default_tokenizer().fingerprint())
            self.assertEqual(restored_manifest["revision"], manifest["revision"])
            (Path(directory) / "tiny.spaceslug" / "model.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_projected_artifact(Path(directory) / "tiny.spaceslug")


if __name__ == "__main__":
    unittest.main()
