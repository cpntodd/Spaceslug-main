import unittest
from pathlib import Path

from spaceslug.backend import BackendSession
from spaceslug.native_training import integrated_tiny_embedding_sgd_capability, integrated_tiny_group_adamw_capability, integrated_tiny_lm_head_adamw_capability, integrated_tiny_lm_head_capability, native_fp32_lm_head_capability, native_fp32_lm_head_training_plan


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

    def test_integrated_embedding_contract_is_distinct_and_bounded(self):
        capability = integrated_tiny_embedding_sgd_capability(available=True, runtime_capability="mock-runtime")
        self.assertTrue(capability["integrated_graph_sgd"])
        self.assertTrue(capability["graph_owned_embedding"])
        self.assertFalse(capability["standalone_api"])
        self.assertEqual(capability["rows_max"], 128)
        self.assertFalse(capability["positions_supported"])
        self.assertFalse(capability["ffn_supported"])
        self.assertFalse(capability["normalization_supported"])
        self.assertFalse(capability["dataset_integration"])

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

    def test_adamw_capability_matrix_exposes_only_lm_head_and_output(self):
        lm_head = integrated_tiny_group_adamw_capability(group="lm_head", available=True)
        output = integrated_tiny_group_adamw_capability(group="output", available=True)
        self.assertEqual(lm_head["status"], "available")
        self.assertEqual(output["status"], "available")
        self.assertTrue(lm_head["integrated_graph_adamw"])
        self.assertTrue(output["integrated_graph_adamw"])
        qkv = integrated_tiny_group_adamw_capability(group="qkv", available=True)
        self.assertTrue(qkv["integrated_graph_adamw"])
        self.assertTrue(qkv["from_existing_gradients"])

    def test_qkv_adamw_contract_consumes_existing_gradients(self):
         capability = integrated_tiny_group_adamw_capability(group="qkv", available=True)
         self.assertTrue(capability["integrated_graph_adamw"])
         self.assertTrue(capability["from_existing_gradients"])
         self.assertEqual(capability["gradient_source"], "existing-qkv-gradients")
         self.assertFalse(capability["dataset_integration"])
         self.assertFalse(capability["retained_training"])

    def test_qkv_adamw_is_explicitly_unavailable_even_when_qkv_sgd_is_available(self):
        qkv_sgd = integrated_tiny_lm_head_capability(group="qkv", available=True)
        self.assertEqual(qkv_sgd["status"], "available")
        self.assertFalse(qkv_sgd["integrated_graph_adamw"])
        self.assertTrue(qkv_sgd["adamw_unsupported"])
        self.assertEqual(qkv_sgd["adamw_return_code"], -4)
        runtime = Path(__file__).parents[2] / "vulkan-runtime"
        self.assertTrue(qkv_sgd["adamw_unsupported"])

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

    def test_mocked_ctypes_trainer_binds_graph_embedding_sgd_and_caps_rows(self):
        session = BackendSession("/tmp/runtime", "test")
        calls = []
        class Native:
            def spaceslug_tiny_forward_train_embeddings_sgd(self, *args): calls.append(args); return 0
        session._native = lambda: Native()
        session.train_tiny_graph_embeddings_sgd(123, [1, 2], [3, 4], [1, 1], 0.25)
        self.assertEqual(calls[0][0], 123)
        self.assertEqual(calls[0][4:], (2, 0.25))
        with self.assertRaises(ValueError):
            session.train_tiny_graph_embeddings_sgd(123, [1] * 129, [2] * 129, [1] * 129, 0.1)

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

    def test_qkv_adamw_mocked_binding_uses_existing_gradients(self):
        session = BackendSession("/tmp/runtime", "test")
        calls = []
        class Native:
            def spaceslug_tiny_forward_train_qkv_adamw_from_gradients(self, *args): calls.append(args); return 0
        session._native = lambda: Native()
        session.train_tiny_graph_qkv_adamw(123, 0.01, weight_decay=0.02)
        self.assertEqual(calls, [(123, 0.01, 0.9, 0.999, 1e-8, 0.02)])

    def test_qkv_adamw_state_readback_and_restore(self):
        session = BackendSession("/tmp/runtime", "test")
        calls = []
        class Native:
            def spaceslug_tiny_forward_readback_base_train_qkv_adamw_state(self, handle, *args):
                calls.append(("read", handle, len(args))); args[-1]._obj.value = 11; return 0
            def spaceslug_tiny_forward_update_base_train_qkv_adamw_state(self, handle, *args):
                calls.append(("update", handle, len(args), args[-1])); return 0
        session._native = lambda: Native()
        state = session.readback_tiny_graph_qkv_adamw_state(123, 2)
        self.assertEqual((state["step"], len(state["weight"]), len(state["m"]), len(state["v"])), (11, 3, 3, 3))
        session.update_tiny_graph_qkv_adamw_state(123, state, 2)
        self.assertEqual(calls[0], ("read", 123, 10))
        self.assertEqual(calls[1][0:3], ("update", 123, 10))

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

    def test_mocked_bounded_scalar_metrics_binding(self):
        session = BackendSession("/tmp/runtime", "test")
        session._device = "mock-device"
        class Native:
            def spaceslug_tiny_forward_loss_fixed_metrics(self, handle, tokens, targets, masks, loss, count):
                loss._obj.value = 1.25
                count._obj.value = 3
                return 0
        session._native = lambda: Native()
        result = session.execute_tiny_fixed_loss_metrics(123, [1] * 128, [2] * 128, [1] * 128)
        self.assertEqual(result.output, {"loss": 1.25, "count": 3})
        with self.assertRaises(ValueError):
            session.execute_tiny_fixed_loss_metrics(123, [1], [2], [1])

    def test_mocked_graph_dstate_readback_binds_exact_rows_and_metadata(self):
        session = BackendSession("/tmp/runtime", "test")
        calls = []
        class Native:
            def spaceslug_tiny_forward_readback_graph_dstate(self, handle, tokens, targets, masks, rows, output):
                calls.append((handle, rows, len(tokens), len(targets), len(masks), len(output)))
                for index in range(rows * 64): output[index] = float(index)
                return 0
        session._native = lambda: Native()
        report = session.readback_tiny_graph_dstate(123, [1, 2, 3], [4, 5, 6], [1, 1, 0])
        self.assertEqual((report["rows"], report["hidden"], report["capacity"]), (3, 64, 128))
        self.assertEqual(len(report["dstate"]), 3)
        self.assertFalse(report["embedding_update"])
        self.assertEqual(calls, [(123, 3, 3, 3, 3, 8192)])
        for bad in (0, 129):
            with self.assertRaises(ValueError):
                session.readback_tiny_graph_dstate(123, [1], [2], [1], bad)
        with self.assertRaises(ValueError):
            session.readback_tiny_graph_dstate(123, [1], [], [1])

    def test_runtime_graph_owned_lm_head_boundary_is_metadata_only(self):
        runtime = Path(__file__).parents[2] / "vulkan-runtime"
        metadata = BackendSession(runtime, "test").capabilities().metadata
        self.assertIn("tiny_graph_base_train_lm_head", metadata)
        self.assertIn("tiny_graph_base_train_lm_head_capability", metadata)
        self.assertTrue(metadata["tiny_graph_dstate_readback_abi"])
        self.assertIn("tiny_graph_embedding_dstate_", metadata["tiny_graph_dstate_readback_capability"])
        self.assertTrue(metadata["tiny_graph_embedding_training"])
        self.assertTrue(metadata["tiny_graph_integrated_embedding_sgd"]["integrated_graph_sgd"])
        self.assertEqual(metadata["tiny_graph_dstate_readback_status"], 0)
        self.assertEqual((metadata["tiny_graph_dstate_hidden"], metadata["tiny_graph_dstate_token_capacity"], metadata["tiny_graph_dstate_float_count"]), (64, 128, 8192))
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
