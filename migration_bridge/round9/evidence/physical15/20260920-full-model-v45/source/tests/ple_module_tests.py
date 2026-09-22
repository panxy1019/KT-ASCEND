"""Opt-in Torch tests for synchronous PLE equations and explicit state."""
import pathlib
import sys
import unittest
import torch
sys.path.insert(0,str(pathlib.Path(__file__).parents[1]))
from runtime.ple_sync import SynchronousPLE

class PLETests(unittest.TestCase):
    def make(self):
        torch.manual_seed(92)
        return SynchronousPLE(hidden_size=8,embedding_size=4,kernel_size=3,dilation=2,dtype=torch.float32)
    def test_zero_projections_and_state(self):
        m=self.make()
        with torch.no_grad():
            for p in m.parameters():p.zero_()
        out,state=m(torch.randn(5,32),torch.randn(5,4))
        self.assertTrue(torch.equal(out,torch.zeros_like(out)))
        self.assertEqual(tuple(state.shape),(32,4))
        self.assertTrue(torch.equal(state,torch.zeros_like(state)))
    def test_chunk_decode_repeat_reset_and_input_state_unchanged(self):
        m=self.make();x=torch.randn(9,32);e=torch.randn(9,4)
        whole,state=m(x,e);s=None;pieces=[]
        for start,end in ((0,2),(2,3),(3,7),(7,9)):
            old=s.clone() if s is not None else None
            out,new=m(x[start:end],e[start:end],s)
            if old is not None:torch.testing.assert_close(s,old,rtol=0,atol=0)
            pieces.append(out);s=new
        torch.testing.assert_close(torch.cat(pieces),whole,rtol=1e-5,atol=2e-6)
        torch.testing.assert_close(s,state,rtol=1e-5,atol=2e-6)
        again,_=m(x,e);torch.testing.assert_close(again,whole,rtol=0,atol=0)
    def test_explicit_reference_gate_and_dilated_convolution(self):
        m=self.make();x=torch.randn(5,32);e=torch.randn(5,4)
        def norm(x,w):
            x=x.double().reshape(5,4,8)
            return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)*(1+w.double().reshape(4,8))
        key=e.double()@m.key_proj.weight.double().T
        value=e.double()@m.value_proj.weight.double().T
        score=(norm(key,m.norm_key.weight)*norm(x,m.norm_query.weight)).sum(-1,keepdim=True)/(8**0.5)
        gate=torch.sigmoid(torch.sign(score)*torch.sqrt(torch.abs(score).clamp_min(1e-6)))
        gated=(gate*value.unsqueeze(1)).flatten(1)
        normalized=norm(gated,m.norm_conv.weight).flatten(1)
        conv=[]
        for t in range(5):
            y=torch.zeros(32,dtype=torch.float64)
            for k in range(3):
                source=t-4+2*k
                if source>=0:y+=normalized[source]*m.conv1d.weight[:,0,k].double()
            conv.append(y*torch.sigmoid(y))
        out,_=m(x,e)
        torch.testing.assert_close(out.double(),gated+torch.stack(conv),rtol=1e-4,atol=2e-6)
    def test_empty_and_invalid_state(self):
        m=self.make();out,state=m(torch.empty(0,32),torch.empty(0,4))
        self.assertEqual(tuple(out.shape),(0,32));self.assertEqual(tuple(state.shape),(32,4))
        with self.assertRaises(ValueError):m(torch.zeros(1,32),torch.zeros(1,4),torch.zeros(32,3))
        with self.assertRaises(ValueError):m(torch.zeros(1,31),torch.zeros(1,4))
if __name__=='__main__':unittest.main()
