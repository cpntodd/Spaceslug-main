import tempfile
from pathlib import Path
import unittest

from spaceslug.cpu_verification import verify_cpu_training
from tests.fixtures.tiny_acceptance import create_tiny_acceptance_bundle


class CpuVerificationTest(unittest.TestCase):
    def test_cpu_gate_passes_and_reports_reference_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            result = verify_cpu_training(create_tiny_acceptance_bundle(Path(directory) / "dataset.dts"))
            self.assertTrue(result.passed)
            self.assertEqual(result.backend, "cpu-reference")
            self.assertLessEqual(result.final_loss, result.initial_loss)
            self.assertTrue(result.artifact_revision.startswith("sha256:"))


if __name__ == "__main__":
    unittest.main()
