import unittest
from spaceslug.backend import BackendSession
from spaceslug.lora import TinyLoRAAdapter
from spaceslug.projected_attention_reference import ProjectedTinyAttentionModel

class LoraTrainGraphTest(unittest.TestCase):
    def test_graph_reaches_gpu_loss_and_lm_backward(self):
        model = ProjectedTinyAttentionModel(259, 64)
        adapter = TinyLoRAAdapter()
        rank = adapter.rank
        backend = BackendSession("/mnt/Data/Projects/Cpntodd_Cactus/vulkan-runtime", "runtime")
        result = backend.execute_tiny_lora_train_graph([1, 7, 23], [2, 8, 24], [1, 1, 1], model, adapter, learning_rate=0.01)
        self.assertEqual(result.status, "ok")
        self.assertTrue(result.metrics["gpu_execution"])
        self.assertEqual(result.metrics["parity"], "gradient-update-partial")
        self.assertEqual(len(result.output["dprojected"]), 128 * 64)
        self.assertEqual(set(result.output["lora_gradients"]), {"query", "key", "value", "output"})
        self.assertEqual(len(result.output["updated_a"]), 4 * 64 * rank)
        self.assertEqual(len(result.output["updated_b"]), 4 * rank * 64)
        self.assertNotEqual(result.output["updated_a"], [0.01] * (4 * 64 * rank))

if __name__ == "__main__": unittest.main()
