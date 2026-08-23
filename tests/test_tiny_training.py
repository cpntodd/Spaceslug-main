import tempfile
from pathlib import Path
import unittest

from spaceslug.dataset import create_bundle, verify_bundle
from spaceslug.tiny_training import TinyTrainingConfig, load_training_checkpoint, save_training_checkpoint, train_tiny
from spaceslug.tokenizer import default_tokenizer


class TinyTrainingTest(unittest.TestCase):
    def test_dataset_training_reduces_loss_and_checkpoint_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = verify_bundle(create_bundle(
                Path(directory) / "fixture.dts", "tiny-fixture",
                {"train": [{"record_id": "a", "text": "abababab"}],
                 "validation": [{"record_id": "v", "text": "abab"}], "test": []},
                tokenizer_id="spaceslug-byte", tokenizer_revision="v1",
            ).root)
            tokenizer = default_tokenizer()
            first_config = TinyTrainingConfig(steps=8, learning_rate=0.2, seed=0)
            model, state, metrics = train_tiny(bundle, first_config, tokenizer=tokenizer)
            self.assertLess(metrics["final_train_loss"], metrics["initial_train_loss"])
            self.assertEqual(metrics["optimizer_step"], 8)
            self.assertEqual(metrics["dataset_revision"], bundle.manifest["revision"])

            checkpoint = Path(directory) / "checkpoint.json"
            save_training_checkpoint(checkpoint, model, state, metrics)
            restored_model, restored_state, restored_metrics = load_training_checkpoint(checkpoint)
            self.assertEqual(restored_model.weights, model.weights)
            self.assertEqual(restored_state, state)
            self.assertEqual(restored_metrics, metrics)

            resumed, resumed_state, resumed_metrics = train_tiny(
                bundle, TinyTrainingConfig(steps=4, learning_rate=0.2), tokenizer=tokenizer,
                model=restored_model, optimizer_state=restored_state,
            )
            uninterrupted, uninterrupted_state, _ = train_tiny(
                bundle, TinyTrainingConfig(steps=12, learning_rate=0.2), tokenizer=tokenizer,
            )
            self.assertEqual(resumed.weights, uninterrupted.weights)
            self.assertEqual(resumed_state, uninterrupted_state)
            self.assertEqual(resumed_metrics["optimizer_step"], 12)


if __name__ == "__main__":
    unittest.main()
