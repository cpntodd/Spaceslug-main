import unittest
from spaceslug.backend import BackendSession

class LoraGpuTest(unittest.TestCase):
    def run_train_step(self, software_vulkan=False):
        backend = BackendSession("/mnt/Data/Projects/Cpntodd_Cactus/vulkan-runtime", "runtime", software_vulkan=software_vulkan)
        x = [0.01 * ((i % 13) - 6) for i in range(3 * 64)]
        dy = [0.02 * ((i % 11) - 5) for i in range(3 * 64)]
        a = [0.01 * ((i % 7) - 3) for i in range(64 * 4)]
        b = [0.01 * ((i % 5) - 2) for i in range(4 * 64)]
        return backend.execute_tiny_lora(x, a, b, dy, learning_rate=0.05)

    def test_native_lora_train_step_matches_cpu_reference(self):
        result = self.run_train_step()
        self.assertEqual(result.status, "ok")
        self.assertTrue(result.metrics["gpu_execution"])
        self.assertEqual(result.metrics["forward"]["status"], "pass")
        self.assertEqual(result.metrics["dA_update"]["status"], "pass")
        self.assertEqual(result.metrics["dB_update"]["status"], "pass")

    def test_native_lora_train_step_matches_lavapipe_reference(self):
        result = self.run_train_step(software_vulkan=True)
        self.assertEqual(result.status, "ok")
        self.assertTrue(result.metrics["gpu_execution"])
        self.assertEqual(result.metrics["forward"]["status"], "pass")

    def test_lora_train_requires_dy(self):
        backend = BackendSession("/mnt/Data/Projects/Cpntodd_Cactus/vulkan-runtime", "runtime")
        result = backend.execute_tiny_lora([1.0] * 64, [0.1] * 256, [0.1] * 256)
        self.assertEqual(result.status, "not-run")
        self.assertIn("dY", result.metrics["reason"])

if __name__ == "__main__": unittest.main()
