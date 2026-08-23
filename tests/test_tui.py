import tempfile
from pathlib import Path
import unittest

from spaceslug.tui import SpaceslugTui
from tests.fixtures.tiny_acceptance import create_tiny_acceptance_bundle


class TuiTest(unittest.TestCase):
    def test_headless_workflow_and_worm_graph(self):
        tui = SpaceslugTui()
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "notes.md").write_text("hello", encoding="utf-8")
            selection = tui.select_directory(directory)
            self.assertEqual(len(selection.files), 1)
        config = tui.set_model("Spaceslug-0.1B", training_mode="lora")
        self.assertEqual(config["training_mode"], "lora")
        tui.set_training(steps=3, epochs=2)
        tui.add_loss(2.0)
        tui.add_loss(1.0)
        self.assertEqual(len(tui.worm_graph()), 8)
        rendered = tui.render()
        self.assertIn("worm graph", rendered)
        self.assertIn("training", rendered)
        self.assertEqual(tui.state.model_id, "Spaceslug-0.1B")

    def test_tui_reports_structured_gpu_forward_parity(self):
        from spaceslug.backend import ExecutionResult
        tui = SpaceslugTui()
        cpu = tui.run_cpu_inference([1, 2, 3])
        gpu = ExecutionResult("ok", "tiny_projected_attention_forward", "vulkan-radv", "runtime", "RX580", False, {}, {"logits": cpu["logits"]})
        report = tui.compare_gpu_forward_result([1, 2, 3], gpu)
        self.assertEqual(report["parity"]["status"], "pass")
        self.assertEqual(tui.state.backend, "vulkan-radv")

    def test_cpu_inference_updates_backend_and_output(self):
        tui = SpaceslugTui()
        record = tui.run_cpu_inference([1, 2, 3])
        self.assertEqual(record["backend"], "cpu-reference")
        self.assertFalse(record["gpu_execution"])
        self.assertIn("next token", tui.state.status)

    def test_gpu_forward_plan_exposes_resolved_contract(self):
        tui = SpaceslugTui()
        plan = tui.plan_gpu_forward("/mnt/Data/Projects/Cpntodd_Cactus/vulkan-runtime", "runtime")
        self.assertEqual(plan["operation"], "tiny_projected_attention_forward")
        self.assertEqual(plan["status"], "planned-not-implemented")
        self.assertEqual(plan["parity_gate"], "CPU logits vs RADV logits")

    def test_gpu_gemm_requires_cpu_gate_and_reports_parity(self):
        tui = SpaceslugTui()
        with self.assertRaises(ValueError):
            tui.verify_gpu_gemm("/missing/runtime", "runtime")
        tui.state.cpu_verified = True
        result = tui.verify_gpu_gemm("/mnt/Data/Projects/Cpntodd_Cactus/vulkan-runtime", "a195bc6d50bb16528fe8970d74254a855264a35c")
        self.assertTrue(result["gpu_passed"])
        self.assertEqual(result["parity"], "cpu-reference")

    def test_cpu_verify_and_training_stream_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = create_tiny_acceptance_bundle(Path(directory) / "dataset.dts")
            tui = SpaceslugTui()
            tui.select_bundle(bundle)
            verification = tui.verify_cpu()
            self.assertTrue(verification["passed"])
            metrics = tui.train_cpu()
            self.assertLess(metrics["final_train_loss"], metrics["initial_train_loss"])
            self.assertEqual(len(tui.state.loss_history), tui.state.steps * tui.state.epochs)
            self.assertIn("CPU training complete", tui.state.status)

    def test_invalid_training_controls_fail(self):
        with self.assertRaises(ValueError):
            SpaceslugTui().set_training(steps=0)


if __name__ == "__main__":
    unittest.main()
