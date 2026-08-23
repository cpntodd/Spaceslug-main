import unittest

from spaceslug.backend import BackendSession


class NativeSgemmTest(unittest.TestCase):
    def test_native_sgemm_returns_gpu_result(self):
        session = BackendSession("/mnt/Data/Projects/Cpntodd_Cactus/vulkan-runtime", "a195bc6d50bb16528fe8970d74254a855264a35c")
        result = session.execute_sgemm_native([1.0] * (64 * 32), [2.0] * (32 * 64), 64, 64, 32)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.operation, "sgemm")
        self.assertEqual(result.output["first"], 64.0)
        self.assertEqual(result.metrics["parity"], "cpu-reference")


if __name__ == "__main__":
    unittest.main()
