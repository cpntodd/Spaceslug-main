import unittest

from spaceslug.backend import BackendSession
from spaceslug.forward_parity import build_gpu_forward_report, compare_gpu_logits, record_cpu_forward
from spaceslug.backend import ExecutionResult
from spaceslug.projected_attention_reference import ProjectedTinyAttentionModel


class ForwardParityTest(unittest.TestCase):
    def test_cpu_baseline_can_be_compared_to_future_gpu_logits(self):
        session = BackendSession("/mnt/Data/Projects/Cpntodd_Cactus/vulkan-runtime", "runtime")
        record = record_cpu_forward(session, [1, 2, 3], ProjectedTinyAttentionModel(259))
        result = compare_gpu_logits(record, record["logits"])
        self.assertEqual(result["parity"]["status"], "pass")
        self.assertTrue(result["gpu_execution"])
        self.assertEqual(result["gpu_backend"], "vulkan-radv")
        gpu = ExecutionResult("ok", "tiny_projected_attention_forward", "vulkan-radv", "runtime", "RX580", False, {}, {"logits": record["logits"]})
        structured = build_gpu_forward_report(record, gpu)
        self.assertEqual(structured["parity"]["status"], "pass")
        self.assertEqual(structured["device"], "RX580")


if __name__ == "__main__":
    unittest.main()
