import unittest

from spaceslug.backend import BackendSession
from spaceslug.forward_parity import compare_gpu_logits, record_cpu_forward
from spaceslug.projected_attention_reference import ProjectedTinyAttentionModel


class ForwardParityTest(unittest.TestCase):
    def test_cpu_baseline_can_be_compared_to_future_gpu_logits(self):
        session = BackendSession("/mnt/Data/Projects/Cpntodd_Cactus/vulkan-runtime", "runtime")
        record = record_cpu_forward(session, [1, 2, 3], ProjectedTinyAttentionModel(259))
        result = compare_gpu_logits(record, record["logits"])
        self.assertEqual(result["parity"]["status"], "pass")
        self.assertTrue(result["gpu_execution"])
        self.assertEqual(result["gpu_backend"], "vulkan-radv")


if __name__ == "__main__":
    unittest.main()
