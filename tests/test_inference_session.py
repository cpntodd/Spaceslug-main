import unittest

from spaceslug.backend import BackendSession
from spaceslug.inference_session import InferenceSession
from spaceslug.projected_attention_reference import ProjectedTinyAttentionModel


class InferenceSessionTest(unittest.TestCase):
    def test_attention_gate_reports_runtime_parity(self):
        session = InferenceSession(BackendSession("/mnt/Data/Projects/Cpntodd_Cactus/vulkan-runtime", "runtime"), ProjectedTinyAttentionModel(259))
        result = session.run_attention_gate([1.0] * (128 * 64), [1.0] * (128 * 64), [1.0] * (128 * 64), 128, 64)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["parity"], "cpu-reference")

    def test_gpu_chain_plan_lists_forward_steps(self):
        session = InferenceSession(BackendSession("/mnt/Data/Projects/Cpntodd_Cactus/vulkan-runtime", "runtime"), ProjectedTinyAttentionModel(259))
        result = session.run_gpu_chain_plan([1, 2, 3])
        self.assertEqual(result["status"], "not-run")
        self.assertIn("qkv_projection_sgemm", result["steps"])
        self.assertIn("causal_softmax", result["steps"])
        self.assertFalse(result["gpu_execution"])

    def test_gpu_plan_runs_tiny_forward_parity(self):
        session = InferenceSession(BackendSession("/mnt/Data/Projects/Cpntodd_Cactus/vulkan-runtime", "runtime"), ProjectedTinyAttentionModel(259, 64))
        result = session.run_gpu_plan([1, 2, 3])
        self.assertTrue(result["gpu_execution"])
        self.assertEqual(result["parity"]["status"], "pass")
        self.assertEqual(result["device"].startswith("AMD Radeon RX 580"), True)

    def test_gpu_plan_is_explicitly_not_run(self):
        session = InferenceSession(BackendSession("/mnt/Data/Projects/Cpntodd_Cactus/vulkan-runtime", "runtime"), ProjectedTinyAttentionModel(259))
        result = session.run_gpu_plan([1, 2, 3])
        self.assertEqual(result["parity"]["status"], "not-run")
        self.assertIn("not implemented", result["parity"]["reason"])
        self.assertFalse(result["gpu_execution"])

    def test_cpu_session_records_backend_and_deterministic_token(self):
        session = InferenceSession(BackendSession("/mnt/Data/Projects/Cpntodd_Cactus/vulkan-runtime", "runtime"), ProjectedTinyAttentionModel(259))
        first = session.next_token([1, 2, 3])
        second = session.next_token([1, 2, 3])
        self.assertEqual(first, second)
        self.assertEqual(session.last_record["backend"], "cpu-reference")
        self.assertFalse(session.last_record["gpu_execution"])


if __name__ == "__main__":
    unittest.main()
