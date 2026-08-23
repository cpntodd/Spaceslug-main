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
        from spaceslug.projected_attention_reference import ProjectedTinyAttentionModel
        cpu_forward = session.execute_projected_attention_cpu_fallback([1, 2, 3], ProjectedTinyAttentionModel(259))
        self.assertEqual(cpu_forward.status, "ok")
        self.assertTrue(cpu_forward.fallback_used)
        self.assertFalse(cpu_forward.metrics["gpu_execution"])
        self.assertEqual(len(cpu_forward.output["logits"]), 259)
        plan = session.projected_attention_forward_plan(hidden_size=2, sequence_length=4, vocab_size=259)
        attention = session.execute_attention_kernel_parity([1.0] * (128 * 64), [1.0] * (128 * 64), [1.0] * (128 * 64), 128, 64)
        self.assertEqual(attention.status, "ok")
        self.assertEqual(attention.operation, "attention")
        self.assertEqual(attention.metrics["parity"], "cpu-reference")
        self.assertEqual(plan["status"], "planned-not-implemented")
        self.assertFalse(plan["gpu_execution"])
        self.assertEqual(plan["next_kernel"], "qkv_projection_sgemm")
        parity = session.execute_sgemm_parity()
        self.assertEqual(parity.operation, "sgemm")
        self.assertEqual(parity.metrics["parity"], "cpu-reference")


if __name__ == "__main__":
    unittest.main()
