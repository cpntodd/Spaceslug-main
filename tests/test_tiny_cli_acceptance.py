import json
import tempfile
from pathlib import Path
import unittest

from spaceslug.projected_attention_artifact import load_projected_artifact
from spaceslug.projected_attention_inference import next_token
from spaceslug.projected_attention_training import ProjectedAttentionConfig, run_projected_training
from spaceslug.tokenizer import default_tokenizer
from tests.fixtures.tiny_acceptance import create_tiny_acceptance_bundle


class TinyCliAcceptanceTest(unittest.TestCase):
    def test_complete_training_report_has_held_out_loss_and_loadable_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = create_tiny_acceptance_bundle(root / "dataset.dts")
            result = run_projected_training(bundle, ProjectedAttentionConfig(steps=2, learning_rate=0.1, batch_size=2), tokenizer=default_tokenizer(), checkpoint=root / "checkpoint.json", artifact=root / "artifact.spaceslug", experiment=root / "run", code_revision="acceptance-test")
            report = json.loads(Path(result["experiment"]).read_text(encoding="utf-8"))
            model, tokenizer, manifest = load_projected_artifact(root / "artifact.spaceslug")
            self.assertEqual(report["metrics"]["artifact_revision"], manifest["revision"])
            self.assertGreater(report["metrics"]["test_loss"], 0.0)
            self.assertEqual(report["metrics"]["inference"]["next_token"], next_token(model, tokenizer, "Q: "))
            self.assertEqual(report["code_revision"], "acceptance-test")


if __name__ == "__main__":
    unittest.main()
