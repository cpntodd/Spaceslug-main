import tempfile
import unittest
from pathlib import Path

from spaceslug.dataset import create_bundle
from spaceslug.native_desktop_training import readiness, run_native_training


class _Caps:
    device = "AMD Radeon RX 580 Series (RADV POLARIS10)"
    software_vulkan = False
    operations = ("tiny_forward_abi", "lora_train_step_abi", "attention_causal_backward_abi", "lora_gradients_multi_abi", "lora_sgd_multi_abi")


class _Backend:
    def __init__(self, *_): pass
    def capabilities(self): return _Caps()


class _Trainer:
    instances = []
    def __init__(self, backend, model, adapter, learning_rate, optimizer):
        self.rank = adapter.rank; self.steps = 0; self.closed = False
        self.__class__.instances.append(self)
    def train_tokens(self, tokens, targets, mask):
        self.steps += 1
        return {"loss": [float(self.steps)] * len(tokens), "gpu_execution": True}
    def checkpoint(self, path):
        Path(path).write_text('{"native": true}\n', encoding="utf-8")
    def close(self): self.closed = True


class NativeReadinessTest(unittest.TestCase):
    def test_fail_closed_without_probe_and_for_rank8(self):
        self.assertFalse(readiness(None, 4)["ready"])
        probe = {"device": _Caps.device, "software_vulkan": False, "operations": list(_Caps.operations)}
        self.assertTrue(readiness(probe, 4)["ready"])
        self.assertFalse(readiness(probe, 8)["ready"])

    def test_rejects_lavapipe_and_missing_operations(self):
        probe = {"device": "llvmpipe", "software_vulkan": True, "operations": []}
        gate = readiness(probe, 4)
        self.assertFalse(gate["ready"])
        self.assertIn("lavapipe", gate["reason"])
        self.assertTrue(gate["missing"])


class NativeRunnerTest(unittest.TestCase):
    def test_runs_windows_reports_loss_and_writes_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = create_bundle(root / "train.dts", "test", {"train": [{"record_id": "r1", "prompt": "", "target": "abcdefghijk"}], "validation": [], "test": []}, tokenizer_id="spaceslug-byte", tokenizer_revision="v1", preprocessing_pipeline="test", preprocessing_revision="1", seed=0)
            seen = []
            result = run_native_training(bundle.root, runtime_root=root, runtime_revision="test", rank=4, steps=2, learning_rate=.01, checkpoint=root/"checkpoint.json", artifact=root/"artifact", experiment=root/"experiment", on_step=lambda step, loss: seen.append((step, loss)), backend_factory=_Backend, trainer_factory=_Trainer)
            self.assertEqual(seen, [(1, 1.0), (2, 2.0)])
            self.assertTrue(result["gpu_execution"])
            self.assertEqual(result["backend"], "vulkan-radv")
            self.assertTrue((root/"checkpoint.json").is_file())
            self.assertTrue((root/"artifact"/"native-training.json").is_file())
            self.assertTrue((root/"experiment"/"experiment.json").is_file())
            self.assertTrue(_Trainer.instances[-1].closed)

    def test_cancellation_stops_before_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = create_bundle(root / "train.dts", "test", {"train": [{"record_id": "r1", "target": "abc"}], "validation": [], "test": []}, tokenizer_id="spaceslug-byte", tokenizer_revision="v1", preprocessing_pipeline="test", preprocessing_revision="1", seed=0)
            result = run_native_training(bundle.root, runtime_root=root, runtime_revision="test", rank=4, steps=5, learning_rate=.01, checkpoint=root/"checkpoint.json", artifact=root/"artifact", experiment=root/"experiment", should_stop=lambda: True, backend_factory=_Backend, trainer_factory=_Trainer)
            self.assertEqual(result["metrics"]["stopped_reason"], "cancelled")
            self.assertEqual(result["metrics"]["steps"], 0)


if __name__ == "__main__": unittest.main()
