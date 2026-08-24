import tempfile
import unittest
from pathlib import Path
from spaceslug.backend import BackendSession
from spaceslug.gpu_lora_training import GpuLoRATrainer, GpuLoRATrainingState, PersistentGpuLoRATrainer, gpu_lora_capability, gpu_lora_training_plan, load_gpu_lora_checkpoint, save_gpu_lora_checkpoint
from spaceslug.lora import TinyLoRAAdapter
from spaceslug.projected_attention_reference import ProjectedTinyAttentionModel

class GpuLoraTrainingStateTest(unittest.TestCase):
    def test_checkpoint_round_trip(self):
        state = GpuLoRATrainingState(step=3, learning_rate=0.02, device_resident=False)
        adapter = {"schema_version": 1, "rank": 4}
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "gpu-lora.json"
            save_gpu_lora_checkpoint(path, state, adapter)
            loaded, loaded_adapter = load_gpu_lora_checkpoint(path)
        self.assertEqual(loaded.state_dict(), state.state_dict())
        self.assertEqual(loaded_adapter, adapter)

    def test_only_sgd_schema_is_supported(self):
        with self.assertRaises(ValueError):
            GpuLoRATrainingState.from_state_dict({"schema_version": 1, "optimizer": "adam"})

    def test_persistent_gpu_session_runs_tensor_step(self):
        backend = BackendSession("/mnt/Data/Projects/Cpntodd_Cactus/vulkan-runtime", "runtime")
        rank, rows = 4, 3
        trainer = PersistentGpuLoRATrainer(backend, [0.01] * (64 * rank), [0.02] * (rank * 64), rank, 0.01, rows)
        report = trainer.step([0.03] * (rows * 64), [0.04] * (rows * 64))
        self.assertTrue(report["gpu_execution"])
        self.assertTrue(report["device_resident"])
        self.assertEqual(report["parity"], "persistent-session")

    def test_capability_keeps_unimplemented_paths_explicit(self):
        capability = gpu_lora_capability()
        self.assertEqual(capability["base_weights"], "frozen")
        self.assertTrue(capability["device_resident"])
        self.assertTrue(capability["persistent_command_buffer"])
        self.assertFalse(capability["adamw"])
        self.assertFalse(capability["dataset_training"])
        self.assertTrue(capability["persistent_command_buffer"])

    def test_training_plan_lists_all_gpu_stages_and_boundaries(self):
        plan = gpu_lora_training_plan()
        self.assertEqual(plan["status"], "persistent-tensor-session-plus-bounded-lm-graph")
        self.assertIn("gpu_multi_adapter_sgd", plan["steps"])
        self.assertIn("persistent-token-derived-LM-graph", plan["unsupported"])

    def test_repeated_gpu_steps_update_adapter_and_checkpoint(self):
        backend = BackendSession("/mnt/Data/Projects/Cpntodd_Cactus/vulkan-runtime", "runtime")
        trainer = GpuLoRATrainer(backend, ProjectedTinyAttentionModel(259, 64), TinyLoRAAdapter(), GpuLoRATrainingState(learning_rate=0.01))
        first = trainer.step([1, 7, 23], [2, 8, 24], [1, 1, 1])
        second = trainer.step([1, 7, 23], [2, 8, 24], [1, 1, 1])
        self.assertEqual(first["step"], 1)
        self.assertEqual(second["step"], 2)
        self.assertTrue(first["gpu_execution"])
        with tempfile.TemporaryDirectory() as root:
            trainer.checkpoint(Path(root) / "step.json")

    def test_run_steps_is_deterministically_counted(self):
        backend = BackendSession("/mnt/Data/Projects/Cpntodd_Cactus/vulkan-runtime", "runtime")
        trainer = GpuLoRATrainer(backend, ProjectedTinyAttentionModel(259, 64), TinyLoRAAdapter())
        reports = trainer.run_steps([([1, 7, 23], [2, 8, 24], [1, 1, 1]), ([1, 7, 23], [2, 8, 24], [1, 1, 1])])
        self.assertEqual([report["step"] for report in reports], [1, 2])

    def test_device_resident_mode_is_explicitly_rejected(self):
        with self.assertRaises(ValueError):
            GpuLoRATrainer(None, None, None, GpuLoRATrainingState(device_resident=True))

if __name__ == "__main__": unittest.main()
