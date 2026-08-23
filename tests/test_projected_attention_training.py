import json
import tempfile
from pathlib import Path
import unittest

from spaceslug.dataset import create_bundle, verify_bundle
from spaceslug.projected_attention_experiment import write_projected_experiment
from spaceslug.projected_attention_training import ProjectedAttentionConfig, load_projected_checkpoint, save_projected_checkpoint, train_projected_attention
from spaceslug.tokenizer import default_tokenizer


class ProjectedAttentionTrainingTest(unittest.TestCase):
    def test_dataset_training_reduces_loss_and_checkpoint_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = verify_bundle(create_bundle(Path(directory) / "fixture.dts", "attention-fixture", {
                "train": [{"record_id": "a", "prompt": "Q: ", "target": "a"}, {"record_id": "b", "prompt": "Q: ", "target": "b"}],
                "validation": [{"record_id": "v", "prompt": "Q: ", "target": "a"}], "test": [],
            }).root)
            model, metrics = train_projected_attention(bundle, ProjectedAttentionConfig(steps=10, learning_rate=0.2, batch_size=2), tokenizer=default_tokenizer())
            self.assertLess(metrics["final_train_loss"], metrics["initial_train_loss"])
            self.assertGreater(metrics["validation_loss"], 0.0)
            checkpoint = Path(directory) / "attention.json"
            save_projected_checkpoint(checkpoint, model, metrics)
            restored, restored_metrics = load_projected_checkpoint(checkpoint)
            self.assertEqual(restored.query, model.query)
            self.assertEqual(restored.value, model.value)
            self.assertEqual(restored_metrics, metrics)
            record = json.loads(write_projected_experiment(Path(directory) / "run", "attention-run", metrics, code_revision="test").read_text(encoding="utf-8"))
            self.assertEqual(record["dataset_revision"], bundle.manifest["revision"])
            self.assertEqual(record["status"], "kept")


if __name__ == "__main__":
    unittest.main()
