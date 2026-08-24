import unittest
from pathlib import Path

from spaceslug.backend import BackendSession
from spaceslug.native_training import integrated_tiny_group_adamw_capability, integrated_tiny_lm_head_adamw_capability, integrated_tiny_lm_head_capability, native_fp32_lm_head_capability, native_fp32_lm_head_training_plan


class NativeFP32LMHeadBoundaryTest(unittest.TestCase):
    def test_capability_is_lm_head_only_and_does_not_claim_full_base(self):
        capability = native_fp32_lm_head_capability()
        self.assertEqual(capability["status"], "implemented-standalone")
        self.assertTrue(capability["native_binding"])
        self.assertTrue(capability["standalone_api"])
        self.assertFalse(capability["graph_owned_lm_head"])
        self.assertFalse(capability["forward_integration"])
        self.assertFalse(capability["tiny_graph_integration"])
        self.assertFalse(capability["dataset_integration"])
        self.assertEqual(capability["dtype"], "fp32")
        self.assertEqual(capability["optimizer"], "sgd")
        self.assertEqual(capability["implemented_subsets"], ["lm_head", "output_projection", "combined_qkv"])
        self.assertEqual(capability["trainable_parameter_groups"], ["lm_head", "output_projection", "combined_qkv"])
        self.assertFalse(capability["full_base_training"])
        self.assertFalse(capability["dataset_training"])
        self.assertIn("attention_qkv", capability["unsupported_full_base_groups"])
        self.assertIn("embeddings", capability["frozen_parameter_groups"])

    def test_plan_keeps_full_base_groups_unsupported(self):
        plan = native_fp32_lm_head_training_plan(hidden_size=64, vocab_size=259)
        self.assertEqual(plan["dimensions"], {"hidden_size": 64, "vocab_size": 259})
        self.assertIn("native_lm_head_backward", plan["steps"])
        self.assertIn("caller_supplied_projected_activations", plan["steps"])
        self.assertEqual(plan["optimizer"], "sgd")
        self.assertFalse(plan["tiny_graph_integration"])
        self.assertEqual(plan["implemented_subsets"], ["lm_head", "output_projection", "combined_qkv"])
        self.assertIn("native_full_base_backward", plan["unsupported_steps"])
        self.assertFalse(plan["full_base_training"])

    def test_integrated_contract_is_distinct_and_adamw_is_minus_four(self):
        capability = integrated_tiny_lm_head_capability(available=True, runtime_capability="mock-runtime")
        self.assertTrue(capability["integrated_graph_sgd"])
        self.assertTrue(capability["graph_owned_lm_head"])
        self.assertFalse(capability["standalone_api"])
        self.assertEqual(capability["optimizer"], "sgd")
        self.assertFalse(capability["adamw"])
        self.assertEqual(capability["adamw_return_code"], -4)

    def test_backend_exposes_same_metadata_without_native_claim(self):
        runtime = Path(__file__).parents[2] / "vulkan-runtime"
        metadata = BackendSession(runtime, "test").capabilities().metadata
        capability = metadata["native_fp32_base_training_subsets"]
        self.assertIs(capability, metadata["native_fp32_lm_head_only_base_training"])
        self.assertTrue(capability["native_binding"])
        self.assertFalse(capability["forward_integration"])
        self.assertFalse(capability["tiny_graph_integration"])
        self.assertEqual(capability["trainable_parameter_groups"], ["lm_head", "output_projection", "combined_qkv"])
        self.assertFalse(capability["full_base_training"])

    def test_integrated_adamw_metadata_distinguishes_graph_state_and_boundaries(self):
        capability = integrated_tiny_lm_head_adamw_capability(available=True, runtime_capability="mock-runtime")
        self.assertTrue(capability["integrated_graph_adamw"])
        self.assertEqual(capability["optimizer_state"], "graph-owned-m-v-step")
        self.assertFalse(capability["dataset_integration"])
        self.assertFalse(capability["retained_training"])
        self.assertFalse(capability["standalone_api"])

    def test_backend_reports_conditional_integrated_sgd_and_adamw_boundary(self):
        runtime = Path(__file__).parents[2] / "vulkan-runtime"
        metadata = BackendSession(runtime, "test").capabilities().metadata
        integrated = metadata["tiny_graph_integrated_lm_head_sgd"]
        self.assertTrue(integrated["integrated_graph_sgd"])
        self.assertFalse(integrated["standalone_api"])
        adamw = metadata["tiny_graph_integrated_lm_head_adamw"]
        self.assertEqual(adamw["return_code"], 0 if adamw["status"] == "available" else -4)
        self.assertEqual(adamw["optimizer_state"], "graph-owned-m-v-step")
        output_adamw = metadata["tiny_graph_integrated_output_adamw"]
        self.assertEqual(output_adamw["return_code"], 0 if output_adamw["status"] == "available" else -4)
        self.assertEqual(output_adamw["parameter_group"], "output")
        self.assertFalse(metadata["tiny_graph_integrated_qkv_sgd"]["integrated_graph_adamw"])

    def test_mocked_ctypes_trainer_binds_integrated_sgd(self):
        session = BackendSession("/tmp/runtime", "test")
        calls = []
        class Native:
            def spaceslug_tiny_forward_train_lm_head_sgd(self, handle, tokens, targets, masks, rows, rate):
                calls.append((handle, list(tokens), list(targets), list(masks), rows, rate))
                return 0
        session._native = lambda: Native()
        session.train_tiny_graph_lm_head_sgd(123, [1, 2], [3, 4], [1, 1], 0.25)
        self.assertEqual(calls, [(123, [1, 2], [3, 4], [1, 1], 2, 0.25)])

    def test_integrated_output_and_qkv_sgd_contracts_are_distinct(self):
        output = integrated_tiny_lm_head_capability(group="output", available=True)
        qkv = integrated_tiny_lm_head_capability(group="qkv", available=True)
        self.assertEqual(output["parameter_group"], "output")
        self.assertEqual(qkv["parameter_group"], "qkv")
        self.assertTrue(output["adamw_unsupported"])
        self.assertTrue(qkv["adamw_unsupported"])
        self.assertFalse(output["integrated_graph_adamw"])
        self.assertFalse(qkv["integrated_graph_adamw"])
        self.assertEqual(output["adamw_return_code"], -4)
        self.assertEqual(qkv["adamw_return_code"], -4)
        self.assertFalse(output["standalone_api"])

    def test_mocked_ctypes_trainer_binds_output_and_qkv_sgd_and_caps_rows(self):
        session = BackendSession("/tmp/runtime", "test")
        calls = []
        class Native:
            def spaceslug_tiny_forward_train_output_sgd(self, *args): calls.append(("output", args)); return 0
            def spaceslug_tiny_forward_train_qkv_sgd(self, *args): calls.append(("qkv", args)); return 0
        session._native = lambda: Native()
        session.train_tiny_graph_output_sgd(123, [1, 2], [3, 4], [1, 1], 0.25)
        session.train_tiny_graph_qkv_sgd(123, [1], [2], [1], 0.5)
        self.assertEqual([call[0] for call in calls], ["output", "qkv"])
        with self.assertRaises(ValueError):
            session.train_tiny_graph_output_sgd(123, [1] * 129, [2] * 129, [1] * 129, 0.1)

    def test_output_adamw_metadata_and_mocked_binding(self):
        capability = integrated_tiny_group_adamw_capability(group="output", available=True)
        self.assertEqual(capability["parameter_group"], "output")
        self.assertTrue(capability["integrated_graph_adamw"])
        self.assertFalse(capability["dataset_integration"])
        self.assertFalse(capability["retained_training"])
        session = BackendSession("/tmp/runtime", "test")
        calls = []
        class Native:
            def spaceslug_tiny_forward_train_output_adamw(self, *args): calls.append(("train", args)); return 0
            def spaceslug_tiny_forward_readback_base_train_output_adamw_state(self, handle, w, m, v, step): calls.append(("read", handle)); step._obj.value = 3; return 0
            def spaceslug_tiny_forward_update_base_train_output_adamw_state(self, *args): calls.append(("update", args[-1])); return 0
        session._native = lambda: Native()
        session.train_tiny_graph_output_adamw(123, [1], [2], [1], 0.01)
        state = session.readback_tiny_graph_output_adamw_state(123)
        session.update_tiny_graph_output_adamw_state(123, state)
        self.assertEqual(state["step"], 3)
        self.assertEqual(calls[-1], ("update", 3))

    def test_mocked_ctypes_trainer_binds_graph_adamw_and_state(self):
        session = BackendSession("/tmp/runtime", "test")
        calls = []
        class Native:
            def spaceslug_tiny_forward_train_lm_head_adamw(self, *args): calls.append(("train", args)); return 0
            def spaceslug_tiny_forward_readback_base_train_lm_head_adamw_state(self, handle, w, m, v, step):
                calls.append(("read", handle)); step._obj.value = 7; return 0
            def spaceslug_tiny_forward_update_base_train_lm_head_adamw_state(self, *args): calls.append(("update", args[-1])); return 0
        session._native = lambda: Native()
        session.train_tiny_graph_lm_head_adamw(123, [1], [2], [1], 0.01)
        state = session.readback_tiny_graph_lm_head_adamw_state(123)
        session.update_tiny_graph_lm_head_adamw_state(123, state)
        self.assertEqual(calls[0][0], "train")
        self.assertEqual(state["step"], 7)
        self.assertEqual(calls[-1], ("update", 7))

    def test_runtime_graph_owned_lm_head_boundary_is_metadata_only(self):
        runtime = Path(__file__).parents[2] / "vulkan-runtime"
        metadata = BackendSession(runtime, "test").capabilities().metadata
        self.assertIn("tiny_graph_base_train_lm_head", metadata)
        self.assertIn("tiny_graph_base_train_lm_head_capability", metadata)
        self.assertFalse(metadata["tiny_graph_base_train_lm_head_training"])
        self.assertEqual(metadata["tiny_graph_base_train_lm_head_training_methods"], metadata["tiny_graph_base_train_lm_head"])
        self.assertEqual(metadata["tiny_graph_base_train_lm_head_training_return_code"], -4 if metadata["tiny_graph_base_train_lm_head"] else None)
        if metadata["tiny_graph_base_train_lm_head"]:
            self.assertTrue(metadata["tiny_graph_base_train_lm_head_group_supported"])
            self.assertIn("base_train_group_lm_head_owned_fp32_fixed_window_sgd", metadata["tiny_graph_base_train_lm_head_capability"])
        else:
            self.assertIsNone(metadata["tiny_graph_base_train_lm_head_capability"])


if __name__ == "__main__":
    unittest.main()
