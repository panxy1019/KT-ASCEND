"""Guarded R2 parent update for a private NZ29 shadow.

The default path is the existing ``index_copy_`` operation.  The candidate
path is deliberately dependency-injectable so its host-side contract can be
tested without importing torch or accessing an NPU.  These tests do not
qualify device dispatch or numerical behavior.
"""

from time import perf_counter_ns


def _add_cost(costs, name, elapsed):
    if costs is not None:
        costs[name] = costs.get(name, 0) + elapsed


def _validate_host_slots(host_slots, slot_count, update_count):
    if not isinstance(host_slots, tuple):
        raise ValueError("host_slots must be an immutable host tuple")
    if type(slot_count) is not int or slot_count <= 0:
        raise ValueError("slot_count must be a positive host integer")
    if type(update_count) is not int or update_count < 0:
        raise ValueError("update_count must be a nonnegative host integer")
    if len(host_slots) != update_count:
        raise ValueError("host slot count does not match update count")
    if any(type(slot) is not int for slot in host_slots):
        raise ValueError("host slots must contain integers")
    if len(set(host_slots)) != len(host_slots):
        raise ValueError("host slots must be unique")
    if any(slot < 0 or slot >= slot_count for slot in host_slots):
        raise ValueError("host slot is out of range")


def _provider_functions(get_format, format_cast, scatter_nd_update):
    if get_format is None or format_cast is None or scatter_nd_update is None:
        import torch_npu

        get_format = get_format or torch_npu.get_npu_format
        format_cast = format_cast or torch_npu.npu_format_cast
        scatter_nd_update = (
            scatter_nd_update or torch_npu.npu_scatter_nd_update_
        )
    return get_format, format_cast, scatter_nd_update


def update_parent(
    destination,
    index,
    staged,
    *,
    enabled=False,
    private_shadow_only=False,
    host_slots=None,
    slot_count=None,
    update_count=None,
    get_format=None,
    format_cast=None,
    scatter_nd_update=None,
    host_phase_costs=None,
    clock_ns=perf_counter_ns,
):
    """Update one resident parent while preserving its object and storage.

    ``host_phase_costs`` contains host call/enqueue durations only.  This
    helper never synchronizes, retries, falls back, or retains the temporary
    ND representation.
    """
    if type(enabled) is not bool:
        raise ValueError("enabled must be an explicit boolean")
    if not enabled:
        return destination.index_copy_(0, index, staged)
    if private_shadow_only is not True:
        raise RuntimeError("R2 is permitted only for the private shadow")

    started = clock_ns()
    _validate_host_slots(host_slots, slot_count, update_count)
    _add_cost(host_phase_costs, "validate_host_ns", clock_ns() - started)

    get_format, format_cast, scatter_nd_update = _provider_functions(
        get_format, format_cast, scatter_nd_update
    )
    if int(get_format(destination)) != 29:
        raise RuntimeError("R2 requires an NZ29 destination")
    if int(get_format(index)) != 2:
        raise RuntimeError("R2 requires an ND index")
    if int(get_format(staged)) != 2:
        raise RuntimeError("R2 requires ND staged updates")

    original_pointer = destination.data_ptr()
    original_shape = tuple(destination.shape)
    original_stride = tuple(destination.stride())

    started = clock_ns()
    nd_parent = format_cast(destination, 2)
    if not nd_parent.is_contiguous():
        nd_parent = nd_parent.contiguous()
    if (
        nd_parent is destination
        or nd_parent.data_ptr() == original_pointer
        or getattr(nd_parent, "_base", None) is not None
    ):
        nd_parent = nd_parent.clone()
    _add_cost(host_phase_costs, "format_cast_host_ns", clock_ns() - started)

    if int(get_format(nd_parent)) != 2:
        raise RuntimeError("format cast did not produce ND")
    if tuple(nd_parent.shape) != original_shape:
        raise RuntimeError("ND parent shape does not match NZ29 destination")
    if nd_parent is destination or nd_parent.data_ptr() == original_pointer:
        raise RuntimeError("ND parent must not alias the NZ29 destination")
    if getattr(nd_parent, "_base", None) is not None:
        raise RuntimeError("ND parent must be an owning tensor, not a view")
    if not nd_parent.is_contiguous():
        raise RuntimeError("ND parent must be contiguous")

    scatter_index = index.reshape(-1, 1)
    started = clock_ns()
    scatter_nd_update(nd_parent, scatter_index, staged)
    _add_cost(host_phase_costs, "scatter_host_ns", clock_ns() - started)

    started = clock_ns()
    destination.copy_(nd_parent)
    _add_cost(host_phase_costs, "copy_host_ns", clock_ns() - started)

    if destination.data_ptr() != original_pointer:
        raise RuntimeError("R2 changed destination storage identity")
    if tuple(destination.shape) != original_shape:
        raise RuntimeError("R2 changed destination shape")
    if tuple(destination.stride()) != original_stride:
        raise RuntimeError("R2 changed destination stride")
    if int(get_format(destination)) != 29:
        raise RuntimeError("R2 changed destination NZ29 format")
    return destination
