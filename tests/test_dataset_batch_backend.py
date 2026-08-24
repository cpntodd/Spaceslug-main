import ctypes
import unittest
from unittest.mock import Mock

from spaceslug.backend import BackendSession


class DatasetBatchBackendTest(unittest.TestCase):
    def setUp(self):
        self.session = BackendSession("/tmp/runtime", "test")
        native = Mock()
        native.vulkan_runtime_dataset_batch_capability.return_value = (
            b"standalone_device_resident_fixed_window_dataset_batch_one_staged_upload_gpu_control_readback"
        )
        native.vulkan_runtime_dataset_batch_create.return_value = 1234
        native.vulkan_runtime_dataset_batch_process.return_value = 0
        self.session._library = native
        self.session._library_path = self.session.runtime_root / "libvulkan_runtime_api.so"
        self.session._device = "test-device"

    def test_capability_is_explicitly_standalone_and_not_training(self):
        self.assertIn("standalone", self.session.dataset_batch_buffer_capability())
        self.assertFalse(self.session.capabilities().metadata["dataset_batch_buffer_training"])

    def test_process_binds_rectangular_inputs_and_outputs(self):
        handle = self.session.create_dataset_batch_buffer(2, 3)
        result = self.session.process_dataset_batch_buffer(
            handle, [1, 2, 3, 4, 5, 6], [0] * 6, [1] * 6, [7, 8], 2, 3
        )
        self.assertEqual(result, [0.0] * 4)
        self.session._library.vulkan_runtime_dataset_batch_process.assert_called_once()
        self.session.close_dataset_batch_buffer(handle)
        self.session._library.vulkan_runtime_dataset_batch_destroy.assert_called_once_with(handle)

    def test_invalid_shapes_are_rejected_before_native_call(self):
        with self.assertRaises(ValueError):
            self.session.process_dataset_batch_buffer(ctypes.c_void_p(1), [1], [1], [1], [1], 1, 2)
        with self.assertRaises(ValueError):
            self.session.create_dataset_batch_buffer(0, 2)


if __name__ == "__main__":
    unittest.main()
