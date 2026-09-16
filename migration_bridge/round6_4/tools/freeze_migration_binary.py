"""Freeze a target-specific extension only after its CPU regression passes."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--regression', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = a.regression/'guard/result.json'
    guard = json.loads(result.read_text())
    assert guard['reason'] == 'EXIT_ZERO', guard
    junit = a.regression/'junit.xml'
    suites = list(ET.parse(junit).getroot().iter('testsuite'))
    assert sum(int(s.get('tests', 0)) for s in suites) >= 18
    assert all(int(s.get('errors', 0)) == int(s.get('failures', 0)) == 0 for s in suites)
    name = 'kt_kernel_ext.cpython-311-aarch64-linux-gnu.so'
    target = Path('/workspace/results/venv-r6/lib/python3.11/site-packages/kt_kernel')/name
    assert sha(target) == sha(a.build/'lib'/name)
    original = sha(a.build/'original'/name)
    assert original == '3fbb95c00c5906b905e8fd689346f4e78f86714fddb6372fa0e481e0da7ce43f'
    record = dict(version='r64-target-cpu-rebuild-v1', binary_path=str(target),
                  original_sha256=original, target_sha256=sha(target),
                  cpu_regression_status='PASS', regression_root=str(a.regression),
                  regression_result_sha256=sha(result), junit_sha256=sha(junit),
                  cmake_cache_sha256=sha(a.build/'build/CMakeCache.txt'),
                  build_log_sha256=sha(a.build/'build.log'),
                  scope='TARGET_CPU_ISA_COMPATIBILITY_NOT_FULL_MODEL_QUALIFICATION')
    with a.output.open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
