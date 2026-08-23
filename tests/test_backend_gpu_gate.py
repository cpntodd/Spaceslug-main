import unittest
from pathlib import Path

from spaceslug.backend import BackendSession


class BackendGpuGateTest(unittest.TestCase):
    def test_gpu_prerequisites_are_explicit(self):
        session = BackendSession(Path("/mnt/Data/Projects/Cpntodd_Cactus/vulkan-runtime"), "a195bc6d50bb16528fe8970d74254a855264a35c")
        result = session.verify_tiny_gpu_prerequisites()
        self.assertFalse(result["tiny_gpu_inference"])
        self.assertFalse(result["tiny_gpu_training"])
        self.assertTrue(result["cpu_gate_required"])
        self.assertEqual(result["next_operation"], "tiny_projected_attention_forward")
        self.assertTrue(result["gemm_parity_available"])
        parity = session.execute_sgemm_parity()
        self.assertEqual(parity.operation, "sgemm")
        self.assertEqual(parity.metrics["parity"], "cpu-reference")


if __name__ == "__main__":
    unittest.main()
