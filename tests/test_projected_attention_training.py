import json
import tempfile
from pathlib import Path
import unittest

from spaceslug.dataset import create_bundle, verify_bundle
from spaceslug.projected_attention_experiment import write_projected_experiment
from spaceslug.projected_attention_training import ProjectedAttentionConfig, load_projected_checkpoint, run_projected_training, save_projected_checkpoint, train_projected_attention
from spaceslug.tokenizer import default_tokenizer


class ProjectedAttentionTrainingTest(unittest.TestCase):
    def test_dataset_training_reduces_loss_and_checkpoint_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = verify_bundle(create_bundle(Path(directory) / "fixture.dts", "attention-fixture", {
                "train": [{"record_id": "a", "prompt": "Q: ", "target": "a"}, {"record_id": "b", "prompt": "Q: ", "target": "b"}],
                "validation": [{"record_id": "v", "prompt": "Q: ", "target": "a"}], "test": [{"record_id": "t", "prompt": "Q: ", "target": "b"}],
            }).root)
            model, optimizer_state, metrics = train_projected_attention(bundle, ProjectedAttentionConfig(steps=10, learning_rate=0.2, batch_size=2), tokenizer=default_tokenizer())
            self.assertLess(metrics["final_train_loss"], metrics["initial_train_loss"])
            self.assertGreater(metrics["validation_loss"], 0.0)
            self.assertGreater(metrics["test_loss"], 0.0)
            self.assertGreaterEqual(metrics["test_token_accuracy"], 0.0)
            self.assertLessEqual(metrics["test_token_accuracy"], 1.0)
            checkpoint = Path(directory) / "attention.json"
            save_projected_checkpoint(checkpoint, model, optimizer_state, metrics)
            restored, restored_metrics, restored_optimizer = load_projected_checkpoint(checkpoint)
            self.assertEqual(restored.query, model.query)
            self.assertEqual(restored.value, model.value)
            self.assertEqual(restored_metrics, metrics)
            self.assertEqual(restored_optimizer, optimizer_state)
            record = json.loads(write_projected_experiment(Path(directory) / "run", "attention-run", metrics, code_revision="test").read_text(encoding="utf-8"))
            self.assertEqual(record["dataset_revision"], bundle.manifest["revision"])
            self.assertEqual(record["status"], "kept")
            self.assertEqual(record["runtime_revision"], "not-used")
            self.assertEqual(record["command"], "")

    def test_integrated_run_produces_checkpoint_artifact_and_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = verify_bundle(create_bundle(root / "fixture.dts", "integrated-fixture", {
                "train": [{"record_id": "a", "prompt": "Q: ", "target": "a"}], "validation": [], "test": [],
            }).root)
            result = run_projected_training(bundle, ProjectedAttentionConfig(steps=2, learning_rate=0.1), tokenizer=default_tokenizer(), checkpoint=root / "checkpoint.json", artifact=root / "artifact.spaceslug", experiment=root / "run", code_revision="test")
            self.assertTrue((root / "checkpoint.json").is_file())
            self.assertTrue((root / "artifact.spaceslug" / "manifest.json").is_file())
            self.assertEqual(result["metrics"]["optimizer_step"], 2)
            self.assertTrue(Path(result["experiment"]).is_file())
            experiment = json.loads(Path(result["experiment"]).read_text(encoding="utf-8"))
            self.assertEqual(experiment["metrics"]["inference"]["prompt"], "Q: ")
            self.assertIsInstance(experiment["metrics"]["inference"]["next_token"], int)
            self.assertTrue(experiment["metrics"]["artifact_revision"].startswith("sha256:"))
            self.assertEqual(experiment["metrics"]["checkpoint_identity"]["schema_version"], 2)

    def test_step_callback_receives_loss_series(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = verify_bundle(create_bundle(Path(directory) / "fixture.dts", "callback", {"train": [{"record_id": "a", "prompt": "Q: ", "target": "a"}], "validation": [], "test": []}).root)
            observed = []
            train_projected_attention(bundle, ProjectedAttentionConfig(steps=3, learning_rate=0.1), tokenizer=default_tokenizer(), on_step=lambda step, loss: observed.append((step, loss)))
            self.assertEqual([step for step, _ in observed], [1, 2, 3])
            self.assertTrue(all(loss > 0.0 for _, loss in observed))

    def test_time_budget_and_early_stop_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = verify_bundle(create_bundle(root / "fixture.dts", "budget", {
                "train": [{"record_id": "a", "prompt": "Q: ", "target": "a"}], "validation": [], "test": [],
            }).root)
            model, _, metrics = train_projected_attention(bundle, ProjectedAttentionConfig(steps=100, learning_rate=0.1, max_seconds=0.000001), tokenizer=default_tokenizer())
            self.assertEqual(metrics["completed_steps"], 1)
            self.assertEqual(metrics["stopped_reason"], "time_budget")
            _, _, early_metrics = train_projected_attention(bundle, ProjectedAttentionConfig(steps=100, learning_rate=0.1, early_stop_patience=1), tokenizer=default_tokenizer(), model=model)
            self.assertIn(early_metrics["stopped_reason"], ("early_stop", "steps"))

    def test_resume_rejects_different_dataset_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_bundle = verify_bundle(create_bundle(root / "first.dts", "first", {"train": [{"record_id": "a", "prompt": "Q: ", "target": "a"}], "validation": [], "test": []}).root)
            second_bundle = verify_bundle(create_bundle(root / "second.dts", "second", {"train": [{"record_id": "a", "prompt": "Q: ", "target": "b"}], "validation": [], "test": []}).root)
            tokenizer = default_tokenizer()
            first, optimizer, metrics = train_projected_attention(first_bundle, ProjectedAttentionConfig(steps=1, learning_rate=0.1), tokenizer=tokenizer)
            checkpoint = root / "checkpoint.json"
            save_projected_checkpoint(checkpoint, first, optimizer, metrics)
            with self.assertRaisesRegex(ValueError, "dataset revision"):
                run_projected_training(second_bundle, ProjectedAttentionConfig(steps=1, learning_rate=0.1), tokenizer=tokenizer, checkpoint=root / "next.json", artifact=root / "next.spaceslug", experiment=root / "next-run", resume=checkpoint)

    def test_should_stop_reports_cancelled_and_stops_early(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = verify_bundle(create_bundle(Path(directory) / "fixture.dts", "cancel", {
                "train": [{"record_id": "a", "prompt": "Q: ", "target": "a"}], "validation": [], "test": [],
            }).root)
            calls = []
            def stop_after_two():
                calls.append(True)
                return len(calls) >= 2
            _, _, metrics = train_projected_attention(bundle, ProjectedAttentionConfig(steps=10, learning_rate=0.1), tokenizer=default_tokenizer(), should_stop=stop_after_two)
            self.assertEqual(metrics["stopped_reason"], "cancelled")
            self.assertEqual(metrics["completed_steps"], 2)

    def test_resume_matches_uninterrupted_projected_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = verify_bundle(create_bundle(root / "fixture.dts", "resume-attention", {
                "train": [{"record_id": "a", "prompt": "Q: ", "target": "a"}, {"record_id": "b", "prompt": "Q: ", "target": "b"}], "validation": [], "test": [],
            }).root)
            tokenizer = default_tokenizer()
            first, first_optimizer, first_metrics = train_projected_attention(bundle, ProjectedAttentionConfig(steps=2, learning_rate=0.1), tokenizer=tokenizer)
            checkpoint = root / "first.json"
            save_projected_checkpoint(checkpoint, first, first_optimizer, first_metrics)
            restored, prior, restored_optimizer = load_projected_checkpoint(checkpoint)
            resumed, resumed_optimizer, resumed_metrics = train_projected_attention(bundle, ProjectedAttentionConfig(steps=3, learning_rate=0.1), tokenizer=tokenizer, model=restored, prior_steps=prior["optimizer_step"], optimizer_state=restored_optimizer)
            uninterrupted, uninterrupted_optimizer, uninterrupted_metrics = train_projected_attention(bundle, ProjectedAttentionConfig(steps=5, learning_rate=0.1), tokenizer=tokenizer)
            self.assertEqual(resumed.query, uninterrupted.query)
            self.assertEqual(resumed.output, uninterrupted.output)
            self.assertEqual(resumed_metrics["optimizer_step"], uninterrupted_metrics["optimizer_step"])
            self.assertEqual(resumed_optimizer, uninterrupted_optimizer)


if __name__ == "__main__":
    unittest.main()
