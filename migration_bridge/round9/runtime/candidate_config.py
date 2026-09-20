"""Opt-in candidate contracts. Planning never grants device/resource admission."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Candidate:
    workers: int
    storage: str
    placement: str
    numa_nodes: tuple

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {'workers','storage','placement','numa_nodes'}:
            raise ValueError('candidate requires exactly workers/storage/placement/numa_nodes')
        workers=value['workers']; nodes=value['numa_nodes']
        if type(workers) is not int or workers not in (64,128):
            raise ValueError('workers must be64 or128 physical workers')
        if not isinstance(nodes,list) or len(nodes)!=workers//32 or any(type(n) is not int or n<0 for n in nodes) or len(set(nodes))!=len(nodes):
            raise ValueError('one distinct NUMA node per32-worker pool is required')
        if value['storage'] not in ('BF16','MXFP4','Q8_0') or value['placement'] not in ('static','request_boundary'):
            raise ValueError('unsupported candidate storage/placement')
        return cls(workers,value['storage'],value['placement'],tuple(nodes))

    def partitions(self):
        width=640//len(self.numa_nodes)
        return [(i*width,width) for i in range(len(self.numa_nodes))]

    def native_status(self):
        if self.storage!='BF16':
            return 'LOWBIT_CAPABILITY_CHECK_REQUIRED_FULL_MODEL_PENDING'
        return 'BF16_NATIVE_CAPABILITY_CHECK_REQUIRED_DEVICE_TEST_PENDING'


def validate_topology(candidate, rows, quota):
    """Choose distinct physical workers and8 auxiliaries from caller's allowed CPUs.

    Caller must collect fresh host/cgroup allowed-CPU intersection. This pure
    function cannot infer idle capacity, pin threads or reserve the selected CPUs.
    """
    if type(quota) not in (int,float) or not math.isfinite(quota) or quota < candidate.workers+8:
        raise ValueError('insufficient CPU quota for workers+8aux')
    seen_cpu=set(); physical={}; nodes={n:[] for n in candidate.numa_nodes}
    for row in rows:
        if set(row)!={'cpu','core','numa'} or type(row['cpu']) is not int or row['cpu']<0 or type(row['numa']) is not int or not isinstance(row['core'],str) or not row['core']:
            raise ValueError('malformed CPU topology')
        if row['cpu'] in seen_cpu:raise ValueError('duplicate logical CPU')
        seen_cpu.add(row['cpu'])
        key=row['core']
        if key in physical:
            if physical[key]!=row['numa']:raise ValueError('physical core spans NUMA nodes')
            continue
        physical[key]=row['numa']
        if row['numa'] in nodes:nodes[row['numa']].append(row['cpu'])
    if any(len(v)<32 for v in nodes.values()):raise ValueError('fewer than32 distinct physical cores in a selected NUMA node')
    workers=[sorted(nodes[n])[:32] for n in candidate.numa_nodes]
    spare=[cpu for n in candidate.numa_nodes for cpu in sorted(nodes[n])[32:]]
    if len(spare)<8:raise ValueError('insufficient distinct auxiliary cores')
    return dict(workers=workers,auxiliary=spare[:8],quota=candidate.workers+8,
                scope='TOPOLOGY_CAPACITY_ONLY_NOT_RESERVATION')


def runtime_arguments(candidate, require_native=False, *, native_block32_tail=False, dynamic_boundary_installed=False):
    if require_native and candidate.storage!='BF16' and native_block32_tail is not True:raise RuntimeError(candidate.native_status())
    if require_native and candidate.placement!='static' and dynamic_boundary_installed is not True:
        raise RuntimeError('DYNAMIC_SCHEDULER_BOUNDARY_INSTALLATION_AND_DEVICE_GATE_REQUIRED')
    return dict(kt_cpuinfer=candidate.workers,kt_threadpool_count=len(candidate.numa_nodes),
                kt_numa_nodes=list(candidate.numa_nodes),kt_method='LLAMAFILE',
                kt_num_gpu_experts=64,kt_enable_dynamic_expert_update=False,
                kt_max_deferred_experts_per_token=0,kt_gpu_prefill_token_threshold=0)


def validate_kt_config(cfg, document, native_bf16_tail=False, native_block32_tail=False, dynamic_boundary_installed=False):
    """Called from the actual model opt-in seam; baseline guard remains separate."""
    candidate=Candidate.from_dict(document)
    expected=runtime_arguments(candidate,require_native=True,native_block32_tail=native_block32_tail,dynamic_boundary_installed=dynamic_boundary_installed)
    if candidate.storage=='BF16' and not native_bf16_tail:raise RuntimeError('UNSUPPORTED_NATIVE_BF16_F32_TAIL')
    if (cfg is None or int(cfg.gpu_experts_mask.sum())!=64 or
        cfg.cpuinfer_threads!=expected['kt_cpuinfer'] or cfg.threadpool_count!=expected['kt_threadpool_count'] or
        list(cfg.numa_nodes)!=expected['kt_numa_nodes'] or cfg.method!='LLAMAFILE' or
        (cfg.max_deferred_experts_per_token or 0)!=0 or cfg.kt_enable_dynamic_expert_update or
        cfg.gpu_prefill_token_threshold):
        raise RuntimeError('CANDIDATE_RUNTIME_CONFIG_MISMATCH')
    return candidate


# Process-local binding established by every real model layer before scheduling.
_MODEL_BINDING = None
_MODEL_LAYERS = set()


def load_candidate_profile(path):
    import hashlib,json
    from pathlib import Path
    if not path:raise RuntimeError('candidate profile is required')
    document=json.loads(Path(path).read_text())
    candidate=Candidate.from_dict(document)
    sha=hashlib.sha256(json.dumps(document,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return candidate,document,sha


def bind_model_profile(path,layer,bank_identity):
    global _MODEL_BINDING
    candidate,document,sha=load_candidate_profile(path)
    binding=dict(profile_sha256=sha,bank_identity=bank_identity)
    if _MODEL_BINDING is not None and _MODEL_BINDING!=binding:
        raise RuntimeError('candidate profile or banks changed during model load')
    if type(layer) is not int or not 0<=layer<48:raise RuntimeError('invalid candidate layer')
    _MODEL_BINDING=binding;_MODEL_LAYERS.add(layer)
    return binding


def require_dynamic_model_profile(path):
    candidate,document,sha=load_candidate_profile(path)
    if candidate.placement!='request_boundary':raise RuntimeError('Dynamic requires request_boundary profile')
    if _MODEL_BINDING is None or _MODEL_LAYERS!=set(range(48)) or _MODEL_BINDING['profile_sha256']!=sha:
        raise RuntimeError('Dynamic profile is not bound to all48 loaded model layers')
    return _MODEL_BINDING
