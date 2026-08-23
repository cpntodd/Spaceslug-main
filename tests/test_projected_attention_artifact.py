import tempfile
from pathlib import Path
import unittest

from spaceslug.projected_attention_artifact import write_projected_artifact
from spaceslug.projected_attention_reference import ProjectedTinyAttentionModel
from spaceslug.tokenizer import default_tokenizer


class ProjectedAttentionArtifactTest(unittest.TestCase):
    def test_artifact_is_created_with_checksummed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_projected_artifact(Path(directory) / "tiny.spaceslug", ProjectedTinyAttentionModel(259, 2), default_tokenizer())
            self.assertEqual(manifest["architecture"], "spaceslug-tiny-projected-attention-cpu-reference")
            self.assertTrue(manifest["files"])
            self.assertTrue(manifest["revision"].startswith("sha256:"))


if __name__ == "__main__":
    unittest.main()
