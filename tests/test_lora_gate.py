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


if __name__ == "__main__":
    unittest.main()
