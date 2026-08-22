from pathlib import Path
import unittest

from spaceslug.backend import BackendSession
from spaceslug.reference import vector_add


RUNTIME = Path(__file__).parents[2] / "vulkan-runtime"
RUNTIME_REVISION = "3e2b6f0"


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
        self.assertTrue(result.output["runtime_report"].endswith("PASS"))
        self.assertEqual(vector_add([0.25, 1.5], [0.75, 2.5]), [1.0, 4.0])
        self.assertGreater(result.metrics["host_elapsed_seconds"], 0.0)
        self.assertFalse(result.metrics["software_vulkan"])

        lavapipe = BackendSession(RUNTIME, RUNTIME_REVISION, software_vulkan=True)
        lava_result = lavapipe.execute_vector_add()
        self.assertEqual(lava_result.status, "ok")
        self.assertTrue(lava_result.metrics["software_vulkan"])

        # The runtime's vector_add executable compares its deterministic output
        # against its CPU reference. This host test verifies that passed evidence
        # is surfaced without silently substituting a fallback implementation.


if __name__ == "__main__":
    unittest.main()
