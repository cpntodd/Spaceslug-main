import ctypes
import unittest
from unittest.mock import Mock

from spaceslug.backend import BackendSession


class NativeSchemaV4GroupsTests(unittest.TestCase):
    def _backend(self, mask=32 | 64):
        backend = BackendSession.__new__(BackendSession)
        native = Mock()
        native.spaceslug_tiny_forward_readback_base_checkpoint.return_value = 0
        native.spaceslug_tiny_forward_import_base_train_embeddings.return_value = 0
        native.spaceslug_tiny_forward_import_base_train_positions.return_value = 0
        native.spaceslug_tiny_forward_update_gamma_state.return_value = 0
        native.spaceslug_tiny_forward_update_ffn_state.return_value = 0
        native.spaceslug_tiny_base_checkpoint_group_mask.return_value = mask
        native.spaceslug_tiny_base_checkpoint_adamw_step.return_value = 9
        native.spaceslug_tiny_base_checkpoint_profile_rank.return_value = 4
        native.spaceslug_tiny_base_checkpoint_embeddings_float_count.return_value = 259 * 64
        native.spaceslug_tiny_base_checkpoint_positions_float_count.return_value = 128 * 64
        native.spaceslug_tiny_base_checkpoint_embeddings.return_value = (ctypes.c_float * (259 * 64))()
        native.spaceslug_tiny_base_checkpoint_positions.return_value = (ctypes.c_float * (128 * 64))()
        native.spaceslug_tiny_base_checkpoint_float_count.side_effect = lambda _, bit: 64 if bit == 32 else 3 * (64 * 256 + 256 + 256 * 64 + 64)
        native.spaceslug_tiny_base_checkpoint_weights.side_effect = lambda _, bit: (ctypes.c_float * (64 if bit == 32 else 3 * (64 * 256 + 256 + 256 * 64 + 64)))()
        backend._native = lambda: native
        return backend, native

    def test_readback_exposes_normalization_and_ffn_groups(self):
        backend, _ = self._backend()
        result = backend.readback_tiny_graph_base_checkpoint(ctypes.c_void_p(1))
        self.assertEqual(len(result["normalization"]), 64)
        self.assertEqual(len(result["ffn"]), 3 * (64 * 256 + 256 + 256 * 64 + 64))

    def test_restore_passes_new_payloads_to_native_abi(self):
        backend, native = self._backend()
        checkpoint = {"version": 3, "group_mask": 32 | 64, "embeddings": [0.0] * (259 * 64), "positions": [0.0] * (128 * 64), "normalization": [0.1] * 64, "ffn": [0.2] * (3 * (64 * 256 + 256 + 256 * 64 + 64)) , "adamw_step": 9}
        backend.update_tiny_graph_base_checkpoint(ctypes.c_void_p(1), checkpoint)
        native.spaceslug_tiny_forward_update_gamma_state.assert_called_once()
        native.spaceslug_tiny_forward_update_ffn_state.assert_called_once()


if __name__ == "__main__":
    unittest.main()
