import unittest
from spaceslug.lora import TinyLoRAAdapter, LoRAProjectedTinyAttention
from spaceslug.projected_attention_reference import ProjectedTinyAttentionModel

class LoraReferenceTest(unittest.TestCase):
    def test_zero_adapter_matches_base(self):
        base=ProjectedTinyAttentionModel(259,64); adapter=TinyLoRAAdapter(); wrapped=LoRAProjectedTinyAttention(base,adapter)
        self.assertEqual(wrapped.logits_for_tokens([1,2,3]), base.logits_for_tokens([1,2,3]))
    def test_state_and_shapes_round_trip(self):
        adapter=TinyLoRAAdapter(); state=adapter.state_dict(); restored=TinyLoRAAdapter.from_state_dict(state)
        self.assertEqual(state["schema_version"],1); self.assertEqual(restored.state_dict(),state)
        self.assertEqual(len(state["matrices"]["query"]["A"]),64); self.assertEqual(len(state["matrices"]["query"]["B"]),4)

if __name__ == "__main__": unittest.main()
