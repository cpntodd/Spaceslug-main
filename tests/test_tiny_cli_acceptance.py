import json
import tempfile
from pathlib import Path
import unittest

from spaceslug.projected_attention_artifact import load_projected_artifact
from spaceslug.regression import compare_reports
from spaceslug.regression_series import load_regression_series, write_regression_series
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
            self.assertGreaterEqual(report["metrics"]["test_token_accuracy"], 0.0)
            self.assertLessEqual(report["metrics"]["test_token_accuracy"], 1.0)
            self.assertEqual(report["metrics"]["inference"]["next_token"], next_token(model, tokenizer, "Q: "))
            self.assertEqual(report["code_revision"], "acceptance-test")
            self.assertEqual(report["runtime_revision"], "not-used")
            self.assertEqual(report["command"], "spaceslug tiny-attention-train")
            self.assertEqual(report["metrics"]["acceptance"]["reproducibility"], "pass")
            self.assertEqual(report["metrics"]["acceptance"]["quality"], "not-established")
            self.assertTrue(report["metrics"]["acceptance"]["ready_for_bounded_testing"])
            comparison = compare_reports([report, report])
            self.assertTrue(comparison["pass"])
            self.assertEqual(comparison["runs"], 2)
            self.assertEqual(comparison["min_test_token_accuracy"], report["metrics"]["test_token_accuracy"])
            series_path = write_regression_series(root / "series", "tiny-series", [report, report])
            series = load_regression_series(series_path)
            self.assertEqual(series["series_id"], "tiny-series")
            self.assertTrue(series["comparison"]["pass"])
            self.assertEqual(series["runs"][0]["config"], report["config"])

    def test_two_bounded_configurations_are_recorded_as_a_series(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = create_tiny_acceptance_bundle(root / "dataset.dts")
            reports = []
            for index, learning_rate in enumerate((0.1, 0.05)):
                run = run_projected_training(bundle, ProjectedAttentionConfig(steps=1, learning_rate=learning_rate, batch_size=2), tokenizer=default_tokenizer(), checkpoint=root / f"checkpoint-{index}.json", artifact=root / f"artifact-{index}.spaceslug", experiment=root / f"run-{index}", code_revision=f"config-{index}")
                reports.append(json.loads(Path(run["experiment"]).read_text(encoding="utf-8")))
            series = load_regression_series(write_regression_series(root / "config-series", "tiny-config-series", reports, max_loss_increase=1.0))
            self.assertEqual(series["series_id"], "tiny-config-series")
            self.assertEqual(len(series["runs"]), 2)
            self.assertEqual({run["config"]["learning_rate"] for run in series["runs"]}, {0.1, 0.05})


if __name__ == "__main__":
    unittest.main()
