"""CPU-only tests for acceptance boundaries and manifest invariants."""
import copy
import sys
import unittest
import tempfile
import json
import socket
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'round6_5/tools'))
from build_ratio_profiles import build
from common import product_gate, score_letters, validate_answer, validate_profile
from monitor_serial_host import npu0_hbm
from serial_session import validate_launch, info, check_port_available


class ContractTests(unittest.TestCase):
    def test_port_reuse_accepts_timewait_rejects_live_listener(self):
        server=socket.socket();server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        server.bind(('127.0.0.1',0));port=server.getsockname()[1];server.listen()
        with self.assertRaises(OSError):check_port_available(port)
        client=socket.create_connection(('127.0.0.1',port))
        peer,_=server.accept();peer.shutdown(socket.SHUT_WR);client.recv(1)
        client.close();peer.close();server.close()
        check_port_available(port)

    def test_optional_rpc_is_not_fabricated_memory(self):
        self.assertEqual(info()['status'],'UNAVAILABLE')
        self.assertNotIn('internal_states',info())

    def test_launch_kv_graph_batch_identity(self):
        args=['--max-total-tokens','4096','--max-running-requests','1','--tp-size','1',
              '--dtype','bfloat16','--device','npu','--disable-cuda-graph']
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'resolved-command.json'
            path.write_text(json.dumps(dict(argv=args)))
            self.assertEqual(validate_launch(Path(directory),False)['batch_limit'],1)
            args[3]='2';path.write_text(json.dumps(dict(argv=args)))
            with self.assertRaises(ValueError):validate_launch(Path(directory),False)

    def test_910b3_hbm_format(self):
        text='| 0     910B3               | OK |\n| 0 | 0000:C1:00.0 | 0  0 / 0   3471 / 65536         |'
        self.assertEqual(npu0_hbm(text),dict(used_mib=3471,capacity_mib=65536))

    def test_5pp_integer_boundary(self):
        self.assertTrue(product_gate(162, 174)['pass_5pp'])
        self.assertFalse(product_gate(161, 174)['pass_5pp'])

    def test_no_fixed_baseline_assumption(self):
        self.assertTrue(product_gate(88,100)['pass_5pp'])
        self.assertFalse(product_gate(87,100)['pass_5pp'])

    def test_equal_and_improved(self):
        self.assertTrue(product_gate(174,174)['pass_5pp'])
        self.assertTrue(product_gate(180,174)['pass_5pp'])

    def test_wrong_sample_size(self):
        with self.assertRaises(ValueError): product_gate(162,174,255)

    def test_letter_order_tie(self):
        self.assertEqual(score_letters([(0,4),(0,3),(0,2),(0,1)], [1,2,3,4])[0],'A')

    def test_candidate_identity(self):
        with self.assertRaises(ValueError): score_letters([(0,1)]*4, [1,2,3,4])

    def test_candidate_nan(self):
        with self.assertRaises(ValueError): score_letters([(float('nan'),1),(0,2),(0,3),(0,4)], [1,2,3,4])

    def test_all_layer_nested_masks(self):
        cfg=dict(n_routed_experts=64,num_hidden_layers=27,first_k_dense_replace=1,moe_layer_freq=1)
        profiles=[build(cfg,list(range(1,27)),n) for n in (8,16,32,64)]
        for p in profiles: validate_profile(p)
        for left,right in zip(profiles,profiles[1:]):
            self.assertLessEqual(set(left['rows'][0]['npu_ids']),set(right['rows'][0]['npu_ids']))

    def test_mask_change_rejected(self):
        cfg=dict(n_routed_experts=64,num_hidden_layers=27,first_k_dense_replace=1,moe_layer_freq=1)
        p=build(cfg,list(range(1,27)),32); p['rows'][0]['npu_ids'][0]=1
        with self.assertRaises(ValueError): validate_profile(p)

    def test_wrong_layer_scope(self):
        cfg=dict(n_routed_experts=64,num_hidden_layers=27,first_k_dense_replace=1,moe_layer_freq=1)
        with self.assertRaises(ValueError): validate_profile(build(cfg,[1,9,17,26],32))

    def test_output_budget_correlation_finite(self):
        answer=dict(output_ids=[82],meta_info=dict(id='rid',prompt_tokens=2,
            output_token_logprobs=[[-.2,82]],finish_reason=dict(type='length')))
        self.assertEqual(validate_answer(answer,'rid',[1,2],1,True),[82])
        for field,value in [('id','wrong'),('prompt_tokens',3),('output_token_logprobs',[[float('inf'),82]])]:
            broken=copy.deepcopy(answer); broken['meta_info'][field]=value
            with self.assertRaises(ValueError): validate_answer(broken,'rid',[1,2],1,True)


if __name__=='__main__': unittest.main()
