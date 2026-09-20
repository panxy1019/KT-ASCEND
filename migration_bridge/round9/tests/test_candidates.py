import dataclasses
import unittest
from migration_bridge.round9.runtime.candidate_config import Candidate, validate_topology, runtime_arguments

class CandidateTests(unittest.TestCase):
    def candidate(self):
        return Candidate.from_dict(dict(workers=128, storage='BF16', placement='static', numa_nodes=[2,3,6,7]))

    def topology(self):
        return [dict(cpu=n*100+i, core=f'{n}:{i}', numa=n) for n in (2,3,6,7) for i in range(40)]

    def test_four_equal_slices_and_arguments(self):
        c=self.candidate()
        self.assertEqual(c.partitions(), [(0,160),(160,160),(320,160),(480,160)])
        args=runtime_arguments(c)
        self.assertEqual((args['kt_cpuinfer'], args['kt_threadpool_count']), (128,4))
        self.assertFalse(args['kt_enable_dynamic_expert_update'])

    def test_topology_uses_physical_cores_and_auxiliary(self):
        allocation=validate_topology(self.candidate(),self.topology(),136)
        self.assertEqual([len(p) for p in allocation['workers']], [32]*4)
        self.assertEqual(len(allocation['auxiliary']),8)
        with self.assertRaises(ValueError):validate_topology(self.candidate(),self.topology(),128)
        for quota in (float('nan'),float('inf')):
            with self.assertRaises(ValueError):validate_topology(self.candidate(),self.topology(),quota)
        rows=self.topology()[:80]
        rows += [dict(r,cpu=r['cpu']+1000) for r in rows]
        with self.assertRaises(ValueError):validate_topology(self.candidate(),rows,136)

    def test_invalid_profiles_fail_closed(self):
        for update in [dict(workers=True),dict(workers=96),dict(numa_nodes=[2,2,6,7]),dict(storage='FP16'),dict(placement='token'),dict(unknown=1)]:
            with self.subTest(update=update),self.assertRaises(ValueError):
                Candidate.from_dict(dict(dict(workers=128,storage='BF16',placement='static',numa_nodes=[2,3,6,7]),**update))

    def test_lowbit_is_not_a_native_support_claim(self):
        c=dataclasses.replace(self.candidate(),storage='MXFP4')
        self.assertEqual(c.native_status(), 'LOWBIT_CAPABILITY_CHECK_REQUIRED_FULL_MODEL_PENDING')
        with self.assertRaisesRegex(RuntimeError,'LOWBIT_CAPABILITY'):runtime_arguments(c,require_native=True)

    def test_native_and_scheduler_capabilities_are_independent(self):
        c=dataclasses.replace(self.candidate(),storage='Q8_0',placement='request_boundary')
        with self.assertRaisesRegex(RuntimeError,'DYNAMIC_SCHEDULER'):
            runtime_arguments(c,require_native=True,native_block32_tail=True)
        args=runtime_arguments(c,require_native=True,native_block32_tail=True,dynamic_boundary_installed=True)
        self.assertFalse(args['kt_enable_dynamic_expert_update'])
        with self.assertRaises(RuntimeError):runtime_arguments(c,require_native=True,dynamic_boundary_installed=True)

    def test_model_has_explicit_candidate_seam(self):
        from pathlib import Path
        from migration_bridge.round9.tools.install_candidate_overlay import candidate_model_source
        model=Path(__file__).parents[1]/'sglang_overlay/python/sglang/srt/models/qwen4_exp.py'
        source=model.read_text(); result=candidate_model_source(source)
        self.assertTrue('validate_kt_config' in result)
        self.assertIn('cap_bytes = 43008 * 1024**2',result)
        self.assertIn('ROUND9_RESOURCE_MODE',result)
        self.assertTrue('Round9 requires the frozen Q64/BF16/C64 static profile' in result)
        compile(result,'candidate_model','exec')
        with self.assertRaises(ValueError):candidate_model_source(result)

    def test_overlay_install_isolated_and_binds_whole_tree(self):
        import hashlib,json,tempfile
        from pathlib import Path
        from migration_bridge.round9.tools.install_candidate_overlay import install,tree_files,tree_digest
        original=Path(__file__).parents[1]/'sglang_overlay/python/sglang/srt/models/qwen4_exp.py'
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base=root/'baseline'
            model=base/'sglang/srt/models/qwen4_exp.py';model.parent.mkdir(parents=True)
            model.write_bytes(original.read_bytes())
            (base/'sglang/srt/layers/round9').mkdir(parents=True)
            (base/'scheduler.py').write_text('BASELINE_MARKER=1\n')
            sched=base/'sglang/srt/managers/scheduler.py';sched.parent.mkdir(parents=True)
            sched.write_text('''class Scheduler:
    def event_loop_normal(self):
        while True:
            if batch:
                result = self.run_batch(batch)
            self.last_batch = batch
    @DynamicGradMode()
    def event_loop_overlap(self):pass
''')
            report=install(base,root/'candidate')
            self.assertEqual(model.read_bytes(),original.read_bytes())
            self.assertIn('scheduler.py',report['baseline_files'])
            self.assertIn('sglang/srt/layers/round9/dynamic_adapter.py',report['candidate_files'])
            self.assertNotEqual(report['baseline_tree_sha256'],report['candidate_tree_sha256'])
            self.assertEqual(tree_digest(tree_files(root/'candidate')),report['candidate_tree_sha256'])
            with self.assertRaises(FileExistsError):install(base,root/'candidate')
            (root/'alias').symlink_to(base,target_is_directory=True)
            with self.assertRaises(ValueError):install(root/'baseline',root/'alias/nested')

    def test_native_four_pool_partition_matches_python(self):
        import pathlib, subprocess, tempfile
        root=pathlib.Path(__file__).resolve().parents[3]
        code='#include "kt-kernel/operators/llamafile/tp_slices.hpp"\n#include <iostream>\nint main(){for(auto s:make_llamafile_tp_slices(640,4,true))std::cout<<s.offset<<" "<<s.length<<"\\n";}\n'
        with tempfile.TemporaryDirectory() as d:
            source=pathlib.Path(d)/'check.cpp';source.write_text(code)
            exe=pathlib.Path(d)/'check'
            subprocess.run(['g++','-std=c++17','-I',str(root),str(source),'-o',str(exe)],check=True,capture_output=True)
            actual=[tuple(map(int,x.split())) for x in subprocess.check_output([str(exe)],text=True).splitlines()]
            self.assertEqual(actual,self.candidate().partitions())

if __name__=='__main__':unittest.main()
