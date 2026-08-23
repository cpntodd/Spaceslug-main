import unittest

from spaceslug.backend import BackendSession


class NativeSgemmTest(unittest.TestCase):
    def setUp(self):
        self.session = BackendSession("/mnt/Data/Projects/Cpntodd_Cactus/vulkan-runtime", "a195bc6d50bb16528fe8970d74254a855264a35c")

    def test_native_sgemm_returns_gpu_result(self):
        result = self.session.execute_sgemm_native([1.0] * (64 * 32), [2.0] * (32 * 64), 64, 64, 32)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.operation, "sgemm")
        self.assertEqual(result.output["first"], 64.0)
        self.assertEqual(result.metrics["parity"], "cpu-reference")
        self.assertLessEqual(result.metrics["max_relative_error"], 1e-3)
        self.assertEqual(result.metrics["status"], "pass")
        self.assertEqual(len(result.output["values"]), 64 * 64)

    def test_projected_qkv_uses_native_sgemm_for_padded_shape(self):
        result = self.session.execute_projected_qkv([1.0] * (64 * 64), [2.0] * (64 * 64), 64, 64, cpu_reference=[128.0] * (64 * 64))
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.operation, "qkv_projection_sgemm")
        self.assertEqual(result.metrics["parity"], "cpu-projection-output")
        self.assertLessEqual(result.metrics["cpu_projection_parity"]["max_relative_error"], 1e-3)
        self.assertEqual(result.metrics["cpu_projection_parity"]["status"], "pass")

    def test_projected_qkv_reports_unpadded_shape_without_false_gpu_claim(self):
        result = self.session.execute_projected_qkv([1.0] * (4 * 2), [2.0] * 4, 4, 2)
        self.assertEqual(result.status, "not-run")
        self.assertEqual(result.metrics["parity"], "not-run")


if __name__ == "__main__":
    unittest.main()
