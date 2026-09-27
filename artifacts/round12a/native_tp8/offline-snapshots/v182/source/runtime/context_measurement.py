"""Read and validate the eight physical Qwen4 TP1 pool tensors.

This helper performs no allocation and imports no device runtime.  Its object
paths are the concrete Round9 SGLang pool paths; a changed or missing path fails
closed instead of being inferred from a similar tensor elsewhere.
"""
from dataclasses import dataclass

from .context_profile import (
    Qwen4TP1Footprint,
    TensorMeasurement,
    validate_qwen4_tp1_measurements,
)


TENSOR_ATTRIBUTE_CHAINS = {
    'ordinary_kv': 'runner.token_to_kv_pool.full_kv_pool.kv_buffer',
    'qsa_compressed': 'runner.token_to_kv_pool.qsa_compressed_keys',
    'qsa_raw': 'runner.token_to_kv_pool.qsa_raw_keys',
    'qsa_positions': 'runner.token_to_kv_pool.qsa_rope_positions',
    'gdn_conv': 'runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0]',
    'gdn_temporal': 'runner.req_to_token_pool.mamba_pool.mamba_cache.temporal',
    'ple_conv': 'runner.req_to_token_pool.short_conv_pool.conv_state',
    'ple_history': 'runner.req_to_token_pool.ngram_pool.context',
}


@dataclass(frozen=True)
class RuntimeTensorMeasurement:
    name: str
    attribute_chain: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    device: str
    data_ptr: int
    storage_data_ptr: int
    storage_nbytes: int


@dataclass(frozen=True)
class RuntimePoolMeasurement:
    tensors: tuple[RuntimeTensorMeasurement, ...]
    total_bytes: int
    expected_floor_bytes: int
    profile_scope: str
    devices: tuple[str, ...]
    required_device: str | None
    scope: str

    def by_name(self):
        return {tensor.name: tensor for tensor in self.tensors}


def _attribute(owner, name, chain):
    try:
        value = getattr(owner, name)
    except AttributeError as exc:
        raise ValueError(f'missing runtime pool attribute: {chain}') from exc
    if value is None:
        raise ValueError(f'runtime pool attribute is disabled: {chain}')
    return value


def _positive_exact_int(value, description):
    if type(value) is not int or value < 1:
        raise ValueError(f'{description} must be a positive exact int')
    return value


def _measure(name, tensor):
    chain = TENSOR_ATTRIBUTE_CHAINS[name]
    try:
        shape = tuple(tensor.shape)
        dtype = str(tensor.dtype)
        numel = tensor.numel()
        element_size = tensor.element_size()
        reported_nbytes = tensor.nbytes
        device = str(tensor.device)
        data_ptr = tensor.data_ptr()
        storage_offset = tensor.storage_offset()
        contiguous = tensor.is_contiguous()
        storage = tensor.untyped_storage()
        storage_data_ptr = storage.data_ptr()
        storage_nbytes = storage.nbytes()
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f'{name} is not a measurable tensor at {chain}') from exc
    if (not shape or any(type(value) is not int or value < 1 for value in shape) or
            not dtype or not device):
        raise ValueError(f'{name} has invalid runtime shape or dtype')
    numel = _positive_exact_int(numel, f'{name}.numel()')
    element_size = _positive_exact_int(element_size, f'{name}.element_size()')
    if type(reported_nbytes) is not int or reported_nbytes < 1:
        raise ValueError(f'{name}.nbytes must be a positive exact int')
    shape_numel = 1
    for dimension in shape:
        shape_numel *= dimension
    calculated = numel * element_size
    if numel != shape_numel or reported_nbytes != calculated:
        raise ValueError(f'{name} runtime byte metadata is internally inconsistent')
    data_ptr = _positive_exact_int(data_ptr, f'{name}.data_ptr()')
    storage_data_ptr = _positive_exact_int(
        storage_data_ptr, f'{name}.untyped_storage().data_ptr()')
    storage_nbytes = _positive_exact_int(
        storage_nbytes, f'{name}.untyped_storage().nbytes()')
    if (type(storage_offset) is not int or storage_offset != 0 or
            contiguous is not True or data_ptr != storage_data_ptr or
            storage_nbytes != reported_nbytes):
        raise ValueError(f'{name} must be a contiguous zero-offset owning tensor')
    return RuntimeTensorMeasurement(
        name, chain, shape, dtype, reported_nbytes, device, data_ptr,
        storage_data_ptr, storage_nbytes)


def measure_and_validate_qwen4_tp1_pools(runner, footprint, *, require_device=None):
    """Measure the concrete live pools and compare all facts with ``footprint``."""
    if not isinstance(footprint, Qwen4TP1Footprint):
        raise TypeError('Qwen4TP1Footprint is required')
    if require_device not in (None, 'npu'):
        raise ValueError('require_device must be None or npu')
    req_pool = _attribute(runner, 'req_to_token_pool', 'runner.req_to_token_pool')
    kv_pool = _attribute(runner, 'token_to_kv_pool', 'runner.token_to_kv_pool')
    full_pool = _attribute(kv_pool, 'full_kv_pool',
                           'runner.token_to_kv_pool.full_kv_pool')
    req_mamba = _attribute(req_pool, 'mamba_pool',
                           'runner.req_to_token_pool.mamba_pool')
    kv_mamba = _attribute(kv_pool, 'mamba_pool',
                          'runner.token_to_kv_pool.mamba_pool')
    if req_mamba is not kv_mamba:
        raise ValueError('request and QSA pools do not share the same MambaPool')
    expected_tensors = {item.name: item for item in footprint.expected_tensors}
    expected_qsa_layers = expected_tensors['ordinary_kv'].shape[1]
    full_layer_nums = _positive_exact_int(
        _attribute(kv_pool, 'full_layer_nums',
                   'runner.token_to_kv_pool.full_layer_nums'),
        'QSA full-layer count')
    if full_layer_nums != expected_qsa_layers:
        raise ValueError('QSA full-layer count differs from TP1 footprint')
    request_capacity = _positive_exact_int(
        _attribute(kv_pool, 'request_capacity',
                   'runner.token_to_kv_pool.request_capacity'),
        'QSA request capacity')
    if request_capacity != footprint.raw_request_slots:
        raise ValueError('QSA request capacity differs from TP1 footprint')
    mamba_size = _positive_exact_int(
        _attribute(req_mamba, 'size',
                   'runner.req_to_token_pool.mamba_pool.size'),
        'Mamba request capacity')
    if mamba_size != footprint.recurrent_request_slots:
        raise ValueError('Mamba request capacity differs from TP1 footprint')

    mamba_cache = _attribute(req_mamba, 'mamba_cache',
                             'runner.req_to_token_pool.mamba_pool.mamba_cache')
    conv = _attribute(mamba_cache, 'conv',
                      'runner.req_to_token_pool.mamba_pool.mamba_cache.conv')
    if not isinstance(conv, (list, tuple)) or len(conv) != 1 or conv[0] is None:
        raise ValueError('admitted Qwen4 TP1 profile requires exactly one GDN conv tensor')
    short_conv_pool = _attribute(req_pool, 'short_conv_pool',
                                 'runner.req_to_token_pool.short_conv_pool')
    ngram_pool = _attribute(req_pool, 'ngram_pool',
                            'runner.req_to_token_pool.ngram_pool')
    for label, pool in (('PLE conv', short_conv_pool), ('PLE history', ngram_pool)):
        size = _positive_exact_int(
            _attribute(pool, 'size', f'{label} pool.size'),
            f'{label} request capacity')
        if size != footprint.recurrent_request_slots:
            raise ValueError(f'{label} request capacity differs from TP1 footprint')

    tensors = {
        'ordinary_kv': _attribute(full_pool, 'kv_buffer',
                                  TENSOR_ATTRIBUTE_CHAINS['ordinary_kv']),
        'qsa_compressed': _attribute(kv_pool, 'qsa_compressed_keys',
                                     TENSOR_ATTRIBUTE_CHAINS['qsa_compressed']),
        'qsa_raw': _attribute(kv_pool, 'qsa_raw_keys',
                              TENSOR_ATTRIBUTE_CHAINS['qsa_raw']),
        'qsa_positions': _attribute(kv_pool, 'qsa_rope_positions',
                                    TENSOR_ATTRIBUTE_CHAINS['qsa_positions']),
        'gdn_conv': conv[0],
        'gdn_temporal': _attribute(mamba_cache, 'temporal',
                                   TENSOR_ATTRIBUTE_CHAINS['gdn_temporal']),
        'ple_conv': _attribute(short_conv_pool, 'conv_state',
                               TENSOR_ATTRIBUTE_CHAINS['ple_conv']),
        'ple_history': _attribute(ngram_pool, 'context',
                                  TENSOR_ATTRIBUTE_CHAINS['ple_history']),
    }
    measured = tuple(_measure(name, tensors[name]) for name in TENSOR_ATTRIBUTE_CHAINS)
    if len({item.storage_data_ptr for item in measured}) != len(measured):
        raise ValueError('runtime pool tensor categories share underlying storage')
    devices = tuple(sorted({item.device for item in measured}))
    if len(devices) != 1:
        raise ValueError('runtime pool tensor categories span multiple devices')
    if (require_device is not None and
            any(item.device.split(':', 1)[0] != require_device for item in measured)):
        raise ValueError(f'runtime pool tensors are not all on required {require_device} device')
    profile_measurements = {
        item.name: TensorMeasurement(item.shape, item.dtype, item.nbytes)
        for item in measured
    }
    summary = validate_qwen4_tp1_measurements(footprint, profile_measurements)
    scope = ('LIVE_POOL_TENSORS_MEASURED_TP1_PROFILE_AND_REQUIRED_DEVICE_VALIDATED'
             if require_device is not None else
             'POOL_OBJECT_TENSORS_MEASURED_TP1_PROFILE_VALIDATED_DEVICE_ONLY_RECORDED')
    return RuntimePoolMeasurement(
        measured, summary.total_bytes, footprint.physical_floor_bytes,
        footprint.scope, devices, require_device, scope)


__all__ = (
    'TENSOR_ATTRIBUTE_CHAINS', 'RuntimeTensorMeasurement',
    'RuntimePoolMeasurement', 'measure_and_validate_qwen4_tp1_pools',
)
