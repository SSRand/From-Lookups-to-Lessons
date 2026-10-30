"""Real small-model CPU checks; never allocates CUDA."""
import unittest
import torch
from transformers import MistralConfig, MistralForCausalLM
from resmem.memory import MistralMLPModel
from precision.runtime import Runtime, generate, parameter_inventory


class RuntimeTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(72)
        self.config = MistralConfig(vocab_size=31,hidden_size=16,intermediate_size=32,
            num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=128,
            pad_token_id=0,bos_token_id=1,eos_token_id=2)
        self.base = MistralForCausalLM(self.config).eval()
        self.mem = MistralMLPModel(self.config,input_dim=16,output_dim=16).eval()

    def test_cached_generation_matches_full_prefix_all_three_readouts(self):
        # Detects wrong position_ids, hook positions, stale capture, or ignored cache.
        ids = torch.tensor([[0,1,4,7],[1,3,4,8]])
        mask = (ids != 0).long()
        for arm in ('base','mlp','res'):
            runtime = Runtime(self.base, self.mem if arm!='base' else None, arm)
            actual = generate(runtime, ids, mask, max_new_tokens=4, fixed_tokens=True, eos_token_ids=[2])
            expected=[]; prefix=ids.clone(); attention=mask.clone()
            for _ in range(4):
                logits,_ = runtime.forward(prefix,attention,None)
                next_token=logits.argmax(-1)
                expected.append(next_token.tolist())
                prefix=torch.cat([prefix,next_token[:,None]],-1)
                attention=torch.cat([attention,torch.ones_like(next_token[:,None])],-1)
            self.assertEqual(actual['token_ids'],torch.tensor(expected).T.tolist())
            self.assertEqual(actual['metrics']['generated_lengths'],[4,4])
            runtime.close()

    def test_eos_first_step_stops_and_counts_it_once(self):
        runtime=Runtime(self.base,None,'base')
        ids=torch.tensor([[1,4,7]])
        logits,_=runtime.forward(ids,torch.ones_like(ids),None)
        eos=logits.argmax(-1).item()
        result=generate(runtime,ids,torch.ones_like(ids),max_new_tokens=4,fixed_tokens=False,eos_token_ids=[eos])
        self.assertEqual(result['metrics']['generated_lengths'],[1])
        self.assertEqual(result['stopped_on_eos'],[True])
        self.assertIsNone(result['metrics']['decode_tokens_per_second'])
        runtime.close()

    def test_dtype_and_weight_bytes_are_measured_not_assumed(self):
        model=torch.nn.Sequential(torch.nn.Linear(2,3,bias=False).to(torch.bfloat16),torch.nn.LayerNorm(3))
        result=parameter_inventory(model)
        self.assertEqual(result['parameter_count'],12)
        self.assertEqual(result['parameter_payload_bytes'],36)
        self.assertEqual(result['parameter_bytes_by_dtype'],{'torch.bfloat16':12,'torch.float32':24})

    def test_mixed_eos_batch_does_not_count_finished_row_padding(self):
        runtime=Runtime(self.base,None,'base')
        first=torch.tensor([[1,4,7]])
        logits,_=runtime.forward(first,torch.ones_like(first),None)
        eos=logits.argmax(-1).item()
        second=None
        for token in range(3,31):
            candidate=torch.tensor([[1,5,token]])
            logits,_=runtime.forward(candidate,torch.ones_like(candidate),None)
            if logits.argmax(-1).item()!=eos:
                second=candidate;break
        self.assertIsNotNone(second)
        ids=torch.cat([first,second])
        result=generate(runtime,ids,torch.ones_like(ids),max_new_tokens=4,fixed_tokens=False,eos_token_ids=[eos])
        self.assertEqual(result['metrics']['generated_lengths'][0],1)
        self.assertGreater(result['metrics']['generated_lengths'][1],1)
        self.assertEqual(result['metrics']['generated_tokens'],sum(map(len,result['token_ids'])))
        self.assertEqual(result['metrics']['decode_generated_tokens'],len(result['token_ids'][1])-1)
        runtime.close()

    def test_bf16_base_fp32_memory_has_explicit_cast(self):
        runtime=Runtime(self.base.to(torch.bfloat16),self.mem,'res')
        ids=torch.tensor([[1,4,7]])
        result=generate(runtime,ids,torch.ones_like(ids),max_new_tokens=2,fixed_tokens=True,eos_token_ids=[2])
        self.assertEqual(result['metrics']['generated_lengths'],[2])
        runtime.close()


if __name__ == '__main__':unittest.main()
