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
        self.assertIn("CPU inference", rendered)
        self.assertIn("GPU chain", rendered)
        self.assertIn("training", rendered)
        self.assertEqual(tui.state.model_id, "Spaceslug-0.1B")

    def test_tui_reports_runtime_capabilities(self):
        tui = SpaceslugTui()
        capabilities = tui.runtime_capabilities()
        self.assertIn("sgemm", capabilities["operations"])
        self.assertIn("attention_kernel", capabilities["operations"])
        self.assertTrue(capabilities["device"])

    def test_tui_attention_gate_reports_radv(self):
        tui = SpaceslugTui()
        report = tui.run_attention_gate()
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["parity"], "cpu-reference")
        self.assertIn("RX 580", report["device"])

    def test_tui_runs_gpu_lora_train_step(self):
        tui = SpaceslugTui()
        report = tui.run_lora_train_step([0.01] * 64, [0.02] * 64, [0.01] * 256, [0.02] * 256, learning_rate=0.05)
        self.assertEqual(report["status"], "ok")
        self.assertTrue(report["gpu_execution"])
        self.assertIn("GPU LoRA train: ok", tui.state.status)

    def test_tui_lora_plan_is_ready_without_execution_claim(self):
        tui = SpaceslugTui()
        report = tui.run_lora_plan()
        self.assertEqual(report["status"], "ready")
        self.assertFalse(report["gpu_execution"])
        self.assertIn("GPU LoRA: ready", tui.state.status)

    def test_tui_gpu_chain_plan_displays_steps(self):
        tui = SpaceslugTui()
        report = tui.run_gpu_chain_plan([1, 2, 3])
        self.assertEqual(report["status"], "not-run")
        self.assertIn("causal_softmax", report["steps"])
        self.assertIn("GPU chain: not-run", tui.state.status)

    def test_tui_gpu_forward_plan_is_not_run(self):
        tui = SpaceslugTui()
        report = tui.run_gpu_forward_plan([1, 2, 3])
        self.assertEqual(report["parity"]["status"], "not-run")
        self.assertIn("GPU forward: not-run", tui.state.status)

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
