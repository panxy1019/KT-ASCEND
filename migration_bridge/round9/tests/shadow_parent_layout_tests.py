"""Rejected NZ view experiment must remain fail-closed without device access."""
import pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from runtime.shadow_parent_layout import update_parent
class Parent:
 def __init__(self):self.calls=[]
 def index_copy_(self,dim,index,source):self.calls.append((dim,index,source));return self
 def __getattr__(self,name):raise AssertionError('rejected candidate touched tensor metadata: '+name)
class Tests(unittest.TestCase):
 def test_default_off_is_exact_original_call(self):
  d=Parent();i=object();s=object();result=update_parent(d,i,s,get_format=lambda _:self.fail('OFF format lookup'))
  self.assertIs(result,d);self.assertEqual(d.calls,[(0,i,s)])
 def test_rejected_candidate_fails_before_tensor_access(self):
  d=Parent()
  with self.assertRaisesRegex(RuntimeError,'REJECTED_NUMERIC'):
   update_parent(d,object(),object(),enabled=True,get_format=lambda _:self.fail('unsafe format lookup'))
  self.assertFalse(d.calls)
 def test_nonboolean_flag_rejected(self):
  with self.assertRaises(ValueError):update_parent(Parent(),object(),object(),enabled='on',get_format=lambda _:29)
if __name__=='__main__':unittest.main()
