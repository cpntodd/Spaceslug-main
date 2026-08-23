import unittest
from spaceslug.backend import BackendSession

class LoraBackwardChainTest(unittest.TestCase):
    def test_composed_backward_chain_produces_all_targets(self):
        backend = BackendSession("/mnt/Data/Projects/Cpntodd_Cactus/vulkan-runtime", "runtime")
        tokens, rank = 3, 4
        q = [0.01 * ((i % 13) - 6) for i in range(tokens * 64)]
        k = [0.02 * ((i % 11) - 5) for i in range(tokens * 64)]
        v = [0.03 * ((i % 7) - 3) for i in range(tokens * 64)]
        dc = [0.04 * ((i % 5) - 2) for i in range(tokens * 64)]
        factors_a = [0.01] * (4 * 64 * rank)
        factors_b = [0.02] * (4 * rank * 64)
        inputs = {name: q[:] for name in ("query", "key", "value", "output")}
        result = backend.execute_tiny_lora_backward_chain(q, k, v, dc, inputs, factors_a, factors_b, rank)
        self.assertEqual(result.status, "ok")
        self.assertTrue(result.metrics["gpu_execution"])
        self.assertEqual(set(result.output["lora_gradients"]), {"query", "key", "value", "output"})

if __name__ == "__main__": unittest.main()
