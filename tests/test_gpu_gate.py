import tempfile
from pathlib import Path
import unittest

from spaceslug.cpu_verification import CpuVerification
from spaceslug.gpu_gate import run_tiny_gemm_gate


class GpuGateTest(unittest.TestCase):
    def test_gpu_gate_refuses_without_cpu_verification(self):
        cpu = CpuVerification(False, "cpu-reference", "dataset", 1.0, 2.0, "", 0, "failed")
        result = run_tiny_gemm_gate(cpu, "/missing/runtime", "runtime")
        self.assertFalse(result.gpu_passed)
        self.assertEqual(result.reason, "CPU verification is required")

    def test_gpu_gate_runs_after_cpu_verification(self):
        cpu = CpuVerification(True, "cpu-reference", "dataset", 2.0, 1.0, "sha256:x", 10, "ok")
        result = run_tiny_gemm_gate(cpu, Path("/mnt/Data/Projects/Cpntodd_Cactus/vulkan-runtime"), "a195bc6d50bb16528fe8970d74254a855264a35c")
        self.assertTrue(result.gpu_passed)
        self.assertEqual(result.parity, "cpu-reference")


if __name__ == "__main__":
    unittest.main()
