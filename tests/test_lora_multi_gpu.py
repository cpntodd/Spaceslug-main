import unittest

from spaceslug.backend import BackendSession


class LoraMultiGpuTest(unittest.TestCase):
    def test_all_four_targets_have_native_gradient_outputs(self):
        backend = BackendSession("/mnt/Data/Projects/Cpntodd_Cactus/vulkan-runtime", "runtime")
        rows, rank = 3, 4
        x = [0.01 * ((i % 13) - 6) for i in range(rows * 64)]
        dy = [0.02 * ((i % 11) - 5) for i in range(rows * 64)]
        a = [0.01 * ((i % 7) - 3) for i in range(4 * 64 * rank)]
        b = [0.01 * ((i % 5) - 2) for i in range(4 * rank * 64)]
        for target in range(4):
            result = backend.execute_lora_gradients_multi(x, dy, a, b, rank, target)
            self.assertEqual(result.status, "ok")
            self.assertTrue(result.metrics["gpu_execution"])
            self.assertEqual(len(result.output["dA"]), 4 * 64 * rank)
            self.assertEqual(len(result.output["dB"]), 4 * rank * 64)

    def test_multi_gradient_contract_rejects_invalid_target(self):
        backend = BackendSession("/mnt/Data/Projects/Cpntodd_Cactus/vulkan-runtime", "runtime")
        result = backend.execute_lora_gradients_multi([0.0] * 64, [0.0] * 64, [0.0] * (4 * 64 * 4), [0.0] * (4 * 4 * 64), 4, 4)
        self.assertEqual(result.status, "not-run")
        self.assertIn("target", result.metrics["reason"])


if __name__ == "__main__":
    unittest.main()
