import tempfile
import unittest
from pathlib import Path
from spaceslug.gpu_lora_training import GpuLoRATrainingState, gpu_lora_capability, load_gpu_lora_checkpoint, save_gpu_lora_checkpoint

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

    def test_capability_keeps_unimplemented_paths_explicit(self):
        capability = gpu_lora_capability()
        self.assertEqual(capability["base_weights"], "frozen")
        self.assertFalse(capability["device_resident"])
        self.assertFalse(capability["adamw"])
        self.assertFalse(capability["dataset_training"])

    def test_only_sgd_schema_is_supported(self):
        with self.assertRaises(ValueError):
            GpuLoRATrainingState.from_state_dict({"schema_version": 1, "optimizer": "adam"})

if __name__ == "__main__": unittest.main()
