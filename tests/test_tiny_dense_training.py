import json
import tempfile
from pathlib import Path
import unittest

from spaceslug.dataset import create_bundle, verify_bundle
from spaceslug.tiny_dense_training import DenseTinyTrainingConfig, load_dense_checkpoint, save_dense_checkpoint, train_dense_tiny, write_experiment_record
from spaceslug.tokenizer import default_tokenizer


class DenseTinyTrainingTest(unittest.TestCase):
    def test_dataset_training_checkpoint_and_experiment_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = verify_bundle(create_bundle(root / "fixture.dts", "dense-fixture", {
                "train": [{"record_id": "a", "text": "abababab"}],
                "validation": [{"record_id": "v", "text": "abab"}], "test": [],
            }, tokenizer_id="spaceslug-byte", tokenizer_revision="v1").root)
            config = DenseTinyTrainingConfig(steps=30, learning_rate=0.5, hidden_size=4)
            model, metrics = train_dense_tiny(bundle, config, tokenizer=default_tokenizer())
            self.assertLess(metrics["final_train_loss"], metrics["initial_train_loss"])
            checkpoint = root / "dense-checkpoint.json"
            save_dense_checkpoint(checkpoint, model, metrics)
            restored, restored_metrics = load_dense_checkpoint(checkpoint)
            self.assertEqual(restored.embedding, model.embedding)
            self.assertEqual(restored_metrics, metrics)
            record_path = write_experiment_record(root / "run", "dense-fixture-run", metrics, code_revision="test")
            record = json.loads(record_path.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "kept")
            self.assertEqual(record["dataset_revision"], bundle.manifest["revision"])
            self.assertEqual(record["budget"]["steps"], 30)

    def test_resumed_training_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = verify_bundle(create_bundle(root / "fixture.dts", "resume-fixture", {
                "train": [{"record_id": "a", "text": "abababab"}], "validation": [], "test": [],
            }, tokenizer_id="spaceslug-byte", tokenizer_revision="v1").root)
            tokenizer = default_tokenizer()
            first, first_metrics = train_dense_tiny(bundle, DenseTinyTrainingConfig(steps=10, learning_rate=0.5, hidden_size=4), tokenizer=tokenizer)
            checkpoint = root / "first.json"
            save_dense_checkpoint(checkpoint, first, first_metrics)
            restored, restored_metrics = load_dense_checkpoint(checkpoint)
            resumed, resumed_metrics = train_dense_tiny(bundle, DenseTinyTrainingConfig(steps=20, learning_rate=0.5, hidden_size=4), tokenizer=tokenizer, model=restored, prior_steps=restored_metrics["optimizer_step"])
            uninterrupted, _ = train_dense_tiny(bundle, DenseTinyTrainingConfig(steps=30, learning_rate=0.5, hidden_size=4), tokenizer=tokenizer)
            self.assertEqual(resumed.embedding, uninterrupted.embedding)
            self.assertEqual(resumed.output, uninterrupted.output)
            self.assertEqual(resumed.output_bias, uninterrupted.output_bias)
            self.assertEqual(resumed_metrics["optimizer_step"], 30)


if __name__ == "__main__":
    unittest.main()
