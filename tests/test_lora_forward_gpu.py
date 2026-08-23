import unittest

from spaceslug.backend import BackendSession
from spaceslug.lora import TinyLoRAAdapter
from spaceslug.projected_attention_reference import ProjectedTinyAttentionModel


class LoraForwardGpuTest(unittest.TestCase):
    def run_case(self, software_vulkan=False, nonzero=False, tokens=None):
        model = ProjectedTinyAttentionModel(259, 64)
        adapter = TinyLoRAAdapter()
        if nonzero:
            for matrix in adapter.matrices.values():
                matrix.B[0][0] = 0.02
                matrix.B[1][1] = -0.015
        backend = BackendSession("/mnt/Data/Projects/Cpntodd_Cactus/vulkan-runtime", "runtime", software_vulkan=software_vulkan)
        return backend.execute_tiny_lora_forward(tokens or [1, 7, 23], model, adapter)

    def test_zero_adapter_matches_cpu_lora_reference(self):
        result = self.run_case()
        self.assertEqual(result.status, "ok")
        self.assertTrue(result.metrics["gpu_execution"])
        self.assertEqual(result.metrics["parity"], "cpu-lora-reference")

    def test_nonzero_adapter_matches_cpu_lora_reference(self):
        result = self.run_case(nonzero=True)
        self.assertEqual(result.status, "ok")
        self.assertTrue(result.metrics["gpu_execution"])

    def test_multiple_sequence_lengths_match_reference(self):
        for tokens in ([3], [1, 7, 23], list(range(9))):
            result = self.run_case(nonzero=True, tokens=tokens)
            self.assertEqual(result.status, "ok", tokens)
            self.assertTrue(result.metrics["gpu_execution"], tokens)

    def test_nonzero_adapter_matches_lavapipe_reference(self):
        result = self.run_case(software_vulkan=True, nonzero=True)
        self.assertEqual(result.status, "ok")
        self.assertTrue(result.metrics["gpu_execution"])


if __name__ == "__main__":
    unittest.main()
