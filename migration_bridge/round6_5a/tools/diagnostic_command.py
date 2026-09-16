"""Reuse frozen launch construction; append an explicit debug-only hook."""
import json
import os
from pathlib import Path
import runpy
import sys

sys.path.insert(0,'/workspace/r65-tools/migration_bridge/round6_5/tools')

original=os.execv
def diagnostic_exec(path,argv):
    root=Path(os.environ['R65A_INTERVENTION_ROOT'])
    mode=os.environ['R65A_INTERVENTION_MODE']
    spec=dict(name='round6.5a-counterfactual-not-production',
        target_modules=['model.layers.*.mlp.topk'],hook_factory='diagnostic_hooks:make_hook',
        config=dict(mode=mode,output=str(root),baseline='/workspace/results/r65-four-R32-localization-v1/npu'))
    if '--forward-hooks' in argv:
        i=argv.index('--forward-hooks')+1
        argv[i]=json.dumps(json.loads(argv[i])+[spec])
    else:argv.extend(['--forward-hooks',json.dumps([spec])])
    os.environ['PYTHONPATH']=str(Path(__file__).parent)+':'+os.environ.get('PYTHONPATH','')
    root.mkdir(parents=True,exist_ok=False)
    (root/'diagnostic-command.json').write_text(json.dumps(dict(argv=argv,mode=mode),indent=2)+'\n')
    original(path,argv)
os.execv=diagnostic_exec
runpy.run_path('/workspace/r65-tools/migration_bridge/round6_5/tools/static_command.py',run_name='__main__')
