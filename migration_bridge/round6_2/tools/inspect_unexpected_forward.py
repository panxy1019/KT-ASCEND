"""Read-only first-failure metadata inspection; never repairs qualification."""
import json
from pathlib import Path
import torch

root = Path('/workspace/results/r62-qd2-defer1')
captures = Path('/workspace/results/r62-qd2-defer1-server/tensors')
expected = json.loads((root / 'run_spec.json').read_text())['provenance']
requests = {json.loads(p.read_text())['meta_info']['id'] for p in (root / 'responses').glob('*.json')}
paths = sorted(captures.rglob('Pass*.pt'))
print(json.dumps({'requests': len(requests), 'capture_files': len(paths)}), flush=True)
for index, path in enumerate(paths):
    tx = torch.load(path, map_location='cpu', weights_only=False)['capture.transaction']
    rids = tx['batch']['request_ids']
    if len(rids) != 1 or rids[0] not in requests or tx['provenance'] != expected:
        print(json.dumps({'first_failure_index': index, 'path': str(path),
                          'request_ids': rids, 'unknown_request': any(r not in requests for r in rids),
                          'provenance_differing_keys': [k for k in set(expected) | set(tx['provenance'])
                                                       if expected.get(k) != tx['provenance'].get(k)],
                          'transaction': tx}, default=str), flush=True)
        break
else:
    print('NO_UNEXPECTED_FORWARD_REPRODUCED', flush=True)
