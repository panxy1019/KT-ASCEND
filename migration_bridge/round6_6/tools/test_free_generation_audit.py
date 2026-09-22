import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class FreeGenerationAuditTests(unittest.TestCase):
    def run_case(self, mismatch=False):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);capture=root/'capture'
            (capture/'npu').mkdir(parents=True);(capture/'hybrid').mkdir()
            baseline=[1]*64;tokens=[1]*64;tokens[5]=2
            report=dict(status='CAPTURE_INTEGRITY_PASS',inputs_sha256='fixture',
                        rows=[dict(rid='fixture',baseline_output_ids=baseline)])
            for mode in ('npu','hybrid'):
                (capture/mode/'capture-result.json').write_text(json.dumps(report))
            choices=[dict(rid='fixture',step=i,natural_token=2 if i==5 else 1) for i in range(65)]
            if mismatch:choices[4]['natural_token']=3
            (capture/'hybrid/forcing.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in choices))
            generation=root/'generation.json'
            generation.write_text(json.dumps(dict(status='GENERATION_PASS',input_sha256='fixture',
                rows=[dict(rid='fixture',reference64=tokens)])))
            diagnostics=root/'diagnostics.json'
            diagnostics.write_text(json.dumps(dict(rows=[dict(rid='fixture',token_index=5)])))
            output=root/'result.json'
            proc=subprocess.run([sys.executable,str(Path(__file__).with_name('analyze_free_generation.py')),
                '--capture',str(capture),'--generation',str(generation),'--diagnostics',str(diagnostics),
                '--output',str(output)],capture_output=True,text=True)
            return proc.returncode,json.loads(output.read_text()) if output.exists() else None,proc.stderr

    def test_accepts_matching_history_through_first_divergence(self):
        code,result,_=self.run_case()
        self.assertEqual(code,0)
        self.assertEqual(result['rows'][0]['first_divergence'],5)

    def test_rejects_observer_mismatch_before_first_divergence(self):
        code,result,error=self.run_case(mismatch=True)
        self.assertNotEqual(code,0)
        self.assertIsNone(result)
        self.assertIn('OBSERVER_COMMON_HISTORY_MISMATCH',error)


if __name__=='__main__':unittest.main()
