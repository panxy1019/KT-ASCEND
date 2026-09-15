"""Verify original deployed bytes, explicitly update only changed test tools."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('refuse provenance overwrite')
    data = json.loads(args.source.read_text())
    toolroot = Path('/workspace/results/round6_2-prequalification/tools')
    changed = str(toolroot / 'run_qualification_capture.py')
    for path, expected in data['file_sha256'].items():
        actual = sha(path)
        if path != changed and actual != expected:
            raise ValueError('UNAUTHORIZED_SOURCE_DRIFT: ' + path)
        if path == changed:
            data['file_sha256'][path] = actual
    for name in ('capture_session_boundary.py', 'wait_capture_ready.py', 'prepare_capture_retry_metadata.py'):
        path = toolroot / name
        data['file_sha256'][str(path)] = sha(path)
    data['capture_retry_incident'] = dict(original_metadata_sha256=sha(args.source),
        reason='STARTUP_HEALTH_GENERATION_CONTAMINATED_CAPTURE_SESSION',
        original_run_status='INVALID', readiness_endpoint='/model_info',
        model_semantics_changed=False, frozen_inputs_changed=False,
        numerical_rules_changed=False)
    args.output.write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'sha256': sha(args.output)}), flush=True)


if __name__ == '__main__':
    main()
