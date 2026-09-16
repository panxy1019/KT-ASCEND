"""One bounded continuation of an existing capture; never launches a model."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    deadline = time.monotonic()+14400
    while time.monotonic() < deadline:
        state = json.loads((args.root/'progress.json').read_text())
        if state['status'] == 'FAILED':
            raise RuntimeError('CAPTURE_FAILED_NO_ANALYSIS')
        if state['status'] == 'DESCRIPTIVE_CAPTURE_COMPLETE_PENDING_ANALYSIS':
            subprocess.run([sys.executable, str(Path(__file__).with_name('analyze_descriptive_holdout.py')),
                            '--root', str(args.root)], check=True)
            return
        if state['status'] != 'RUNNING':
            raise ValueError('UNEXPECTED_CAPTURE_STATE:'+state['status'])
        time.sleep(15)
    raise TimeoutError('CAPTURE_NOT_COMPLETE_WITHIN_FOUR_HOURS')


if __name__ == '__main__':
    main()
