import json
import subprocess
import sys
import unittest


class CliCommandsTest(unittest.TestCase):
    def run_gate(self, *extra):
        return json.loads(subprocess.run([sys.executable, "-m", "spaceslug.cli", "tiny-attention-gate", "--tokens", "128", "--hidden-size", "64", *extra], check=True, capture_output=True, text=True).stdout)

    def test_attention_gate_cli_reports_radv(self):
        result = self.run_gate()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["parity"], "cpu-reference")
        self.assertIn("PASS", result["report"]["runtime_report"])

    def test_attention_gate_cli_reports_lavapipe(self):
        result = self.run_gate("--software-vulkan")
        self.assertEqual(result["status"], "ok")
        self.assertIn("llvmpipe", result["device"])
        self.assertEqual(result["parity"], "cpu-reference")


if __name__ == "__main__":
    unittest.main()
