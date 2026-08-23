import unittest

from spaceslug.backend import BackendSession
from spaceslug.inference_session import InferenceSession
from spaceslug.projected_attention_reference import ProjectedTinyAttentionModel


class LoraGateTest(unittest.TestCase):
    def test_lora_gpu_status_is_explicitly_not_run(self):
        session = InferenceSession(BackendSession("/mnt/Data/Projects/Cpntodd_Cactus/vulkan-runtime", "runtime"), ProjectedTinyAttentionModel(259))
        result = session.run_lora_plan()
        self.assertEqual(result["status"], "not-run")
        self.assertFalse(result["gpu_execution"])
        self.assertIn("LoRA", result["reason"])
        self.assertEqual(result["rank"], 4)

    def test_lora_tensor_api_has_structured_not_run_result(self):
        backend = BackendSession("/mnt/Data/Projects/Cpntodd_Cactus/vulkan-runtime", "runtime")
        result = backend.execute_tiny_lora([1.0] * 64, [0.1] * 16, [0.2] * 256)
        self.assertEqual(result.status, "not-run")
        self.assertEqual(result.metrics["rank"], 4)
        self.assertEqual(result.output["rows"], 1)


if __name__ == "__main__":
    unittest.main()
