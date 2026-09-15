"""Non-generating startup probe plus strict empty-session boundary."""
import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
from capture_session_boundary import require_empty_capture_session


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--log', type=Path, required=True)
    p.add_argument('--captures', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--url', default='http://127.0.0.1:31000')
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('refuse readiness evidence overwrite')
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if args.log.exists() and 'The server is fired up and ready to roll!' in args.log.read_text(errors='replace'):
            try:
                with urllib.request.urlopen(args.url + '/model_info', timeout=10) as response:
                    info = json.load(response)
                if not info.get('is_generation') or not info.get('model_path'):
                    raise ValueError('MODEL_INFO_NOT_GENERATION_SERVER')
                require_empty_capture_session(args.captures)
                report = dict(status='PASS', endpoint='/model_info', generating_probe=False,
                              capture_session_empty=True, model_info=info, time_unix=time.time())
                args.output.write_text(json.dumps(report, indent=2) + '\n')
                print(json.dumps(report), flush=True)
                return
            except (urllib.error.URLError, TimeoutError):
                pass
        time.sleep(2)
    raise TimeoutError('non-generating startup readiness timeout')


if __name__ == '__main__':
    main()
