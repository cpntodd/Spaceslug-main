from pathlib import Path
import unittest

from spaceslug.backend import BackendSession
from spaceslug.reference import vector_add


RUNTIME = Path(__file__).parents[2] / "vulkan-runtime"
RUNTIME_REVISION = "a195bc6"


class BackendSmokeTest(unittest.TestCase):
    def test_vector_add_runtime_smoke_and_cpu_contract(self):
        session = BackendSession(RUNTIME, RUNTIME_REVISION)
        capabilities = session.capabilities()
        self.assertEqual(capabilities.backend, "spaceslug")
        self.assertEqual(capabilities.runtime_revision, RUNTIME_REVISION)
        self.assertIn("vector_add", capabilities.operations)
        self.assertIsNotNone(capabilities.device)
        self.assertIn("RADV", capabilities.device)

        result = session.execute_vector_add()
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.operation, "vector_add")
        self.assertFalse(result.fallback_used)
        if (RUNTIME / "build/debug/libvulkan_runtime_api.so").is_file():
            self.assertEqual(result.metrics["execution"], "native-shared-library")
            self.assertEqual(result.output["first"], 3.0)
            self.assertEqual(result.output["last"], 3.0)
        else:
            self.assertTrue(result.output["runtime_report"].endswith("PASS"))
        self.assertEqual(vector_add([0.25, 1.5], [0.75, 2.5]), [1.0, 4.0])
        self.assertGreater(result.metrics["host_elapsed_seconds"], 0.0)
        self.assertFalse(result.metrics["software_vulkan"])
        self.assertEqual(result.metrics["parity"], "cpu-reference")
        self.assertEqual(result.metrics["max_abs_error"], 0.0)

        lavapipe = BackendSession(RUNTIME, RUNTIME_REVISION, software_vulkan=True)
        lava_result = lavapipe.execute_vector_add(
            [0.25] * (1 << 20), [0.75] * (1 << 20)
        )
        self.assertEqual(lava_result.status, "ok")
        self.assertTrue(lava_result.metrics["software_vulkan"])
        self.assertEqual(lava_result.metrics["parity"], "cpu-reference")
        self.assertEqual(lava_result.metrics["max_abs_error"], 0.0)
        self.assertEqual(lava_result.output["first"], 1.0)
        self.assertEqual(lava_result.output["last"], 1.0)

        # The runtime owns the Vulkan dispatch. Each device result is compared
        # element-for-element with the host CPU reference; no fallback output is
        # accepted as runtime evidence.


if __name__ == "__main__":
    unittest.main()
