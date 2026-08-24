import unittest
from pathlib import Path

from spaceslug.backend import BackendSession
from spaceslug.native_training import native_fp32_lm_head_capability, native_fp32_lm_head_training_plan


class NativeFP32LMHeadBoundaryTest(unittest.TestCase):
    def test_capability_is_lm_head_only_and_does_not_claim_full_base(self):
        capability = native_fp32_lm_head_capability()
        self.assertEqual(capability["status"], "planned-not-implemented")
        self.assertFalse(capability["native_binding"])
        self.assertEqual(capability["dtype"], "fp32")
        self.assertEqual(capability["trainable_parameter_groups"], ["lm_head"])
        self.assertFalse(capability["full_base_training"])
        self.assertFalse(capability["dataset_training"])
        self.assertIn("attention_qkv", capability["unsupported_full_base_groups"])
        self.assertIn("embeddings", capability["frozen_parameter_groups"])

    def test_plan_keeps_full_base_groups_unsupported(self):
        plan = native_fp32_lm_head_training_plan(hidden_size=64, vocab_size=259)
        self.assertEqual(plan["dimensions"], {"hidden_size": 64, "vocab_size": 259})
        self.assertIn("native_lm_head_backward", plan["steps"])
        self.assertIn("native_full_base_backward", plan["unsupported_steps"])
        self.assertFalse(plan["full_base_training"])

    def test_backend_exposes_same_metadata_without_native_claim(self):
        runtime = Path(__file__).parents[2] / "vulkan-runtime"
        metadata = BackendSession(runtime, "test").capabilities().metadata
        capability = metadata["native_fp32_lm_head_only_base_training"]
        self.assertFalse(capability["native_binding"])
        self.assertEqual(capability["trainable_parameter_groups"], ["lm_head"])
        self.assertFalse(capability["full_base_training"])


if __name__ == "__main__":
    unittest.main()
