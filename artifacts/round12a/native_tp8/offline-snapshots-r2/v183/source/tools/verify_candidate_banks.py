"""CPU-only full catalog/type/byte verification, separate from model admission."""
import argparse,json,pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[3]))
from migration_bridge.round9.runtime.candidate_bank import verify_catalog
p=argparse.ArgumentParser()
for name in ('profile','banks','model','receipt','output'):p.add_argument('--'+name,type=pathlib.Path,required=True)
a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
report=verify_catalog(a.profile,a.banks,a.model,a.receipt)
a.output.write_text(json.dumps(report,indent=2)+'\n');print(report['status'],flush=True)
