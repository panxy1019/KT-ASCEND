import ast
from pathlib import Path
import unittest

class TrialResultTests(unittest.TestCase):
    def test_only_captured_finite_content_errors_are_separable(self):
        path=Path(__file__).parents[1]/'tools/finalize_candidate_trial.py'
        tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='content_only_failure']
        ns={};exec(compile(ast.Module(nodes,type_ignores=[]),str(path),'exec'),ns)
        row=dict(numeric_finite=True,response=dict(text='```json\n{}\n```'),passed=False,parse_error='JSONDecodeError(test)')
        report=dict(status='FAIL',rows=[row],error=row['parse_error'])
        self.assertTrue(ns['content_only_failure'](report))
        row['numeric_finite']=False;self.assertFalse(ns['content_only_failure'](report))
        row['numeric_finite']=True;row.pop('parse_error');self.assertFalse(ns['content_only_failure'](report))
        report['rows']=[];self.assertFalse(ns['content_only_failure'](report))

    def classify(self,failed):
        path=Path(__file__).parents[1]/'tools/finalize_candidate_trial.py'
        tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='classify_failures']
        ns={};exec(compile(ast.Module(nodes,type_ignores=[]),str(path),'exec'),ns)
        return ns['classify_failures'](failed)
    def test_content_failure_cannot_become_overall_pass(self):
        overall,resource=self.classify(['canaries.json'])
        self.assertEqual(overall,'FAIL_CANDIDATE_CONTENT_CANARY')
        self.assertEqual(resource,'PASS_CANDIDATE_RESOURCE_SYSTEM')
    def test_resource_or_guard_failure_still_blocks(self):
        for failures in (['context4096.json'],['guard_stopped_run'],['canaries.json','container_OOMKilled']):
            overall,resource=self.classify(failures)
            self.assertEqual(overall,'FAIL_CANDIDATE_RESOURCE_TRIAL')
            self.assertEqual(resource,'FAIL_CANDIDATE_RESOURCE_SYSTEM')
    def test_both_axes_pass(self):
        self.assertEqual(self.classify([]),('PASS_CANDIDATE_SYSTEM_RESOURCE_ONLY','PASS_CANDIDATE_RESOURCE_SYSTEM'))
