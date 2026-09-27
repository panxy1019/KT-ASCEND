"""Opt-in torch tests for the eager four-branch GR reference adapter."""
import importlib.util
import pathlib
import sys
import unittest
import torch

PATH = pathlib.Path(__file__).parents[1] / "runtime" / "hyperconnection.py"
spec = importlib.util.spec_from_file_location("round9_hyperconnection", PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

class GRTests(unittest.TestCase):
    def make(self, combine=True):
        return mod.GatedResidual(mod.HyperConnectionConfig(hc_count=4, hidden_size=8,
            hc_lowrank=3, params_dtype=torch.float32, hc_per_branch_norm=True),
            device="cpu", use_combine=combine)

    def test_zero_gates_preserve_all_four_distinct_residual_branches(self):
        m=self.make()
        with torch.no_grad():
            for p in m.parameters(): p.zero_()
        x=torch.arange(1,65,dtype=torch.float32).reshape(2,32)
        block=torch.arange(16,dtype=torch.float32).reshape(2,8)
        norm=torch.stack([x[:,8*b:8*(b+1)]/torch.sqrt(
            (x[:,8*b:8*(b+1)].double()**2).mean(-1,keepdim=True)+1e-6)
            for b in range(4)],dim=1).float()
        mixed,state=m.mix(x)
        torch.testing.assert_close(mixed,norm.mean(1)*0.5,rtol=1e-6,atol=1e-6)
        y=m.combine(block,state).reshape(2,4,8)
        for b in range(4):
            torch.testing.assert_close(y[:,b],x[:,8*b:8*(b+1)]+block,rtol=0,atol=0)

    def test_nonzero_gates_match_independent_double_equations(self):
        torch.manual_seed(901)
        m=self.make()
        with torch.no_grad():
            for p in m.parameters(): p.normal_(0,0.2)
        x=torch.randn(3,32); block=torch.randn(3,8)
        parts=[]
        for b in range(4):
            v=x[:,8*b:8*(b+1)].double()
            parts.append(v/torch.sqrt((v*v).mean(-1,keepdim=True)+1e-6)*
                         (1+m.hc_norm.weight[8*b:8*(b+1)].double()))
        norm=torch.cat(parts,-1)
        low=norm @ m.input_mix_weight_down.weight.double().T / 4
        activated=low/(1+torch.exp(-low))
        gates=1/(1+torch.exp(-(activated @ m.input_mix_weight_up.weight.double().T)))
        want_mix=sum(parts[b]*gates[:,8*b:8*(b+1)] for b in range(4))/4
        inject=2/(1+torch.exp(-(norm @ m.block_inject_weight.weight.double().T)/4))
        want_combine=torch.cat([x[:,8*b:8*(b+1)].double()+block.double()*inject[:,b:b+1]
                                for b in range(4)],-1)
        mixed,state=m.mix(x)
        torch.testing.assert_close(mixed.double(),want_mix,rtol=1e-5,atol=2e-6)
        torch.testing.assert_close(m.combine(block,state).double(),want_combine,rtol=1e-5,atol=2e-6)

    def test_empty_batch_mix_only_and_bad_width(self):
        m=self.make(combine=False)
        mixed,state=m.mix(torch.empty(0,32))
        self.assertEqual(tuple(mixed.shape),(0,8))
        self.assertFalse(hasattr(m,"block_inject_weight"))
        with self.assertRaises(ValueError): m.mix(torch.empty(2,31))

    def test_batch_split_equivalence_and_no_cross_request_state(self):
        m=self.make();x=torch.randn(5,32);block=torch.randn(5,8)
        whole=m.combine(block,m.mix(x)[1])
        pieces=torch.cat([m.combine(block[i:i+1],m.mix(x[i:i+1])[1]) for i in range(5)])
        torch.testing.assert_close(whole,pieces,rtol=1e-5,atol=1e-6)

if __name__=="__main__": unittest.main()
