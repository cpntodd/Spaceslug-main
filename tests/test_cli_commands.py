import json
import subprocess
import sys
import unittest


class CliCommandsTest(unittest.TestCase):
    def test_attention_gate_cli_reports_radv(self):
        completed = subprocess.run([sys.executable, "-m", "spaceslug.cli", "tiny-attention-gate", "--tokens", "128", "--hidden-size", "64"], check=True, capture_output=True, text=True)
        result = json.loads(completed.stdout)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["parity"], "cpu-reference")
        self.assertIn("PASS", result["report"]["runtime_report"])


if __name__ == "__main__":
    unittest.main()
