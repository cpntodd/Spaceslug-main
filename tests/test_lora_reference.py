import unittest
from spaceslug.lora import TinyLoRAAdapter, LoRAProjectedTinyAttention, lora_gradients_from_weight_gradient, sgd_update
from spaceslug.projected_attention_reference import ProjectedTinyAttentionModel

class LoraReferenceTest(unittest.TestCase):
    def test_zero_adapter_matches_base(self):
        base=ProjectedTinyAttentionModel(259,64); adapter=TinyLoRAAdapter(); wrapped=LoRAProjectedTinyAttention(base,adapter)
        self.assertEqual(wrapped.logits_for_tokens([1,2,3]), base.logits_for_tokens([1,2,3]))
    def test_state_and_shapes_round_trip(self):
        adapter=TinyLoRAAdapter(); state=adapter.state_dict(); restored=TinyLoRAAdapter.from_state_dict(state)
        self.assertEqual(state["schema_version"],1); self.assertEqual(restored.state_dict(),state)
        self.assertEqual(len(state["matrices"]["query"]["A"]),64); self.assertEqual(len(state["matrices"]["query"]["B"]),4)
    def test_gradient_chain_and_sgd_update(self):
        adapter=TinyLoRAAdapter(); matrix=adapter.matrices["query"]
        matrix.B[0][0] = 0.02
        gradient=[[0.01 * (i+j+1) for j in range(64)] for i in range(64)]
        d_a,d_b=lora_gradients_from_weight_gradient(gradient,matrix)
        before=matrix.A[0][0],matrix.B[0][0]
        sgd_update(matrix,d_a,d_b,0.1)
        self.assertNotEqual(matrix.A[0][0],before[0])
        self.assertNotEqual(matrix.B[0][0],before[1])

if __name__ == "__main__": unittest.main()
