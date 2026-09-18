"""Read-only base-stack config probe, writes only Round8-owned evidence."""
import importlib.metadata as md
import json,platform,sys,traceback
from pathlib import Path
root=Path('/workspace/round8')
report={'python':sys.version,'machine':platform.machine(),'scope':'CONFIG_ONLY_NO_NPU_TENSOR_OR_MODEL_LOAD'}
for name in ('torch','torch-npu','transformers','sglang','sgl-kernel-npu'):
    try:report[name]=md.version(name)
    except md.PackageNotFoundError:report[name]='NOT_INSTALLED'
try:
    from transformers import AutoConfig
    cfg=AutoConfig.from_pretrained(root/'metadata',local_files_only=True,trust_remote_code=False)
    text=cfg.text_config
    report.update(config_status='PASS',architecture=cfg.architectures,
        layers=text.num_hidden_layers,experts=text.num_experts,topk=text.num_experts_per_tok)
except Exception:
    report.update(config_status='FAIL',traceback=traceback.format_exc())
(root/'entry-probe.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
