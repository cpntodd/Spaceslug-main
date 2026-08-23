import json
import tempfile
from pathlib import Path
import unittest

from spaceslug.projected_attention_artifact import load_projected_artifact, write_projected_artifact
from spaceslug.projected_attention_inference import next_token
from spaceslug.projected_attention_reference import ProjectedTinyAttentionModel
from spaceslug.tokenizer import default_tokenizer


class ProjectedAttentionInferenceTest(unittest.TestCase):
    def test_loaded_artifact_produces_deterministic_next_token(self):
        tokenizer = default_tokenizer()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "tiny.spaceslug"
            write_projected_artifact(root, ProjectedTinyAttentionModel(tokenizer.vocab_size, 2), tokenizer)
            model, loaded_tokenizer, _ = load_projected_artifact(root)
            token = next_token(model, loaded_tokenizer, "Q: ")
            self.assertGreaterEqual(token, 0)
            self.assertLess(token, tokenizer.vocab_size)
            self.assertEqual(token, next_token(model, loaded_tokenizer, "Q: "))

    def test_loaded_artifact_matches_recorded_inference_result(self):
        tokenizer = default_tokenizer()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "tiny.spaceslug"
            write_projected_artifact(root, ProjectedTinyAttentionModel(tokenizer.vocab_size, 2), tokenizer)
            model, loaded_tokenizer, manifest = load_projected_artifact(root)
            recorded = {"artifact_revision": manifest["revision"], "inference": {"prompt": "Q: ", "next_token": next_token(model, loaded_tokenizer, "Q: ")}}
            report = Path(directory) / "report.json"
            report.write_text(json.dumps(recorded, sort_keys=True) + "\n", encoding="utf-8")
            loaded_report = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(loaded_report["artifact_revision"], manifest["revision"])
            self.assertEqual(loaded_report["inference"]["next_token"], next_token(model, loaded_tokenizer, "Q: "))


if __name__ == "__main__":
    unittest.main()
