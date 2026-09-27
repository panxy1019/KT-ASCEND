"""Opt-in exact compressed-QSA reference tests, independent small examples."""
import pathlib,sys,unittest
import torch
sys.path.insert(0,str(pathlib.Path(__file__).parents[1]))
from runtime.qsa import compress_index_keys,select_tokens,sparse_gqa,text_rope
class QSATests(unittest.TestCase):
    def test_compress_raw_before_norm_and_first_position(self):
        raw=torch.tensor([[1.,0.],[0.,3.],[2.,1.],[1.,4.],[9.,1.]])
        positions=torch.tensor([10,11,12,13,14]);seen=[]
        def rope(x,p):seen.extend(p.tolist());return x
        out=compress_index_keys(raw,torch.tensor([0.,0.]),positions,rope,ratio=4)
        avg=raw[:4].double().mean(0);want=avg/torch.sqrt(avg.square().mean()+1e-6)
        torch.testing.assert_close(out[0].double(),want,rtol=1e-6,atol=1e-6)
        self.assertEqual(seen,[10]);self.assertEqual(tuple(out.shape),(1,2))
    def test_causal_boundaries_and_exact_tail(self):
        q=torch.ones(7,2,2);k=torch.tensor([[1.,0.],[2.,0.]])
        positions=torch.tensor([0,2,3,4,6,7,8])
        ids=select_tokens(q,k,positions,ratio=4,topk_blocks=1)
        wanted=[[0],[0,1,2],[0,1,2,3],[0,1,2,3,4],list(range(7)),[4,5,6,7],[4,5,6,7,8]]
        for got,want in zip(ids,wanted):self.assertEqual(got[got>=0].tolist(),want)
        self.assertEqual(tuple(ids.shape),(7,7))
    def test_relu_per_head_not_after_head_sum(self):
        q=torch.tensor([[[1.,0.],[0.,1.]]]);keys=torch.tensor([[4.,-3.],[2.,1.]])
        # Correct scores4 vs3 choose group0; ReLU(sum heads) gives1 vs3.
        ids=select_tokens(q,keys,torch.tensor([7]),ratio=4,topk_blocks=1)
        self.assertEqual(ids[0,:4].tolist(),[0,1,2,3])
    def test_sparse_gqa_and_padding_independent_of_unselected_kv(self):
        q=torch.tensor([[[1.,0.],[0.,1.],[2.,0.],[0.,2.]]])
        k=torch.arange(12,dtype=torch.float32).reshape(3,2,2)/10
        v=torch.arange(12,dtype=torch.float32).reshape(3,2,2)
        chosen=torch.tensor([[0,2,-1]])
        out=sparse_gqa(q,k,v,chosen)
        want=[]
        for head in range(4):
            kv=head//2;scores=torch.stack([(q[0,head].double()*k[i,kv].double()).sum()/(2**0.5) for i in (0,2)])
            want.append(sum(torch.softmax(scores,0)[j]*v[i,kv].double() for j,i in enumerate((0,2))))
        torch.testing.assert_close(out[0].double(),torch.stack(want),rtol=1e-6,atol=1e-6)
        v[1]=9999;k[1]=-9999
        torch.testing.assert_close(sparse_gqa(q,k,v,chosen),out,rtol=0,atol=0)
        self.assertTrue(torch.equal(sparse_gqa(q,k,v,torch.full((1,3),-1)),torch.zeros_like(q)))
    def test_text_rope_partial_neox_and_reject_multiaxis(self):
        x=torch.arange(24,dtype=torch.float32).reshape(2,2,6)/10
        positions=torch.tensor([0,3]);out=text_rope(x,positions,rotary_dim=4,theta=100.)
        torch.testing.assert_close(out[0],x[0],rtol=0,atol=0)
        torch.testing.assert_close(out[...,4:],x[...,4:],rtol=0,atol=0)
        for head in range(2):
            for j in range(2):
                angle=3/(100**(2*j/4));c=torch.cos(torch.tensor(angle));v=torch.sin(torch.tensor(angle))
                self.assertAlmostEqual(out[1,head,j].item(),(x[1,head,j]*c-x[1,head,j+2]*v).item(),places=6)
        with self.assertRaises(ValueError):text_rope(x,positions.expand(3,-1),rotary_dim=4)
    def test_invalid_and_more_than_budget(self):
        q=torch.ones(1,4,128);keys=torch.arange(513,dtype=torch.float32).unsqueeze(1).expand(-1,128)
        ids=select_tokens(q,keys,torch.tensor([2051]))
        self.assertEqual(tuple(ids.shape),(1,2051));self.assertEqual((ids>=0).sum().item(),2048)
        self.assertNotIn(0,ids[0].tolist());self.assertIn(2051,ids[0].tolist())
        with self.assertRaises(ValueError):select_tokens(q,keys,torch.tensor([-1]))
        with self.assertRaises(ValueError):sparse_gqa(torch.ones(1,3,2),torch.ones(2,2,2),torch.ones(2,2,2),torch.tensor([[0]]))
if __name__=='__main__':unittest.main()
