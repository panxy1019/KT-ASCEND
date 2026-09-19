"""Create or audit the isolated .16 Q32 stability/Dynamic container."""
import hashlib
import json
import socket
import subprocess
import time
from pathlib import Path

ROOT=Path('/home/admin/kt-round8-qwen35')
SOURCE=Path('/home/admin/kt-round8-src')
NAME='kt-round8-qwen35-stability'
IMAGE='kt-r67-frozen-runtime:20260917'


def command(*args):return subprocess.check_output(args,text=True,timeout=120)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert socket.gethostname()=='gpu-server-02'
    assert (ROOT/'.transfer-complete').exists() and SOURCE.exists()
    image=json.loads(command('docker','image','inspect',IMAGE))[0]
    assert image['Id']=='sha256:b81e664642aecd5f4cc8ec38b233aae38f65b9cf7feea40e54e13c08cd319347'
    manifest=json.loads((ROOT/'cpu-experts-f32-v1/manifest.json').read_text())
    assert manifest['exact_projection_instances']==30720 and len(manifest['layers'])==40
    model=json.loads((ROOT/'01_MODEL/QWEN35_MODEL_MANIFEST.json').read_text())
    assert model['revision']=='59d61f3ce65a6d9863b86d2e96597125219dc754'
    existing=subprocess.run(['docker','inspect',NAME],capture_output=True,text=True)
    if existing.returncode:
        command('docker','create','--name',NAME,'--cpuset-cpus','128-191',
            '--cpuset-mems','4,5','--memory','384g','--memory-swap','384g',
            '--shm-size','16g','--network','none',
            '--device','/dev/davinci2','--device','/dev/davinci_manager',
            '--device','/dev/devmm_svm','--device','/dev/hisi_hdc',
            '-v',f'{ROOT}:/workspace/round8:rw','-v',f'{SOURCE}:/workspace/kt-src:ro',
            '-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro',
            '-v','/usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro',
            '-e','ASCEND_RT_VISIBLE_DEVICES=0',IMAGE,'infinity')
    inspect=json.loads(command('docker','inspect',NAME))[0]
    assert inspect['HostConfig']['CpusetCpus']=='128-191'
    assert inspect['HostConfig']['CpusetMems']=='4,5'
    assert inspect['HostConfig']['Memory']==inspect['HostConfig']['MemorySwap']==384*2**30
    record=dict(status='SECONDARY_CONTAINER_READY',timestamp=time.time(),host='110.129.0.16',
        hostname=socket.gethostname(),container=NAME,image=image['Id'],npu=2,cpus='128-191',mems='4,5',
        model_revision=model['revision'],cpu_manifest_sha256=sha(ROOT/'cpu-experts-f32-v1/manifest.json'),
        q32_profile_sha256=sha(ROOT/'q32_profile.json'),
        q32_command_sha256=sha(ROOT/'q32-perf-launch-v2/resolved-command.json'))
    (ROOT/'secondary-host-provenance.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
