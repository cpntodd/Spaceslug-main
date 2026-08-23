import unittest
from spaceslug.backend import BackendSession

class CausalLossGpuTest(unittest.TestCase):
    def test_loss_and_dlogits_radv(self):
        backend = BackendSession("/mnt/Data/Projects/Cpntodd_Cactus/vulkan-runtime", "runtime")
        result = backend.execute_causal_loss([0.1 * ((i % 7) - 3) for i in range(3 * 8)], [2, 7, 1], [1, 0, 1], 8)
        self.assertEqual(result.status, "ok")
        self.assertTrue(result.metrics["gpu_execution"])
        self.assertEqual(result.metrics["loss"]["status"], "pass")
        self.assertEqual(result.metrics["dlogits"]["status"], "pass")

    def test_loss_and_dlogits_lavapipe(self):
        backend = BackendSession("/mnt/Data/Projects/Cpntodd_Cactus/vulkan-runtime", "runtime", software_vulkan=True)
        result = backend.execute_causal_loss([0.1 * ((i % 7) - 3) for i in range(3 * 8)], [2, 7, 1], [1, 0, 1], 8)
        self.assertEqual(result.status, "ok")
        self.assertTrue(result.metrics["gpu_execution"])

if __name__ == "__main__": unittest.main()
