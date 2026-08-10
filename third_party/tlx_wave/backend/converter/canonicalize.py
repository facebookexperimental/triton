"""Structural target-program cleanup for the TLX Wave converter."""

from dataclasses import replace

from . import target_ir


def eliminate_redundant_compiler_membar_barriers(target_program):
    """Drop membars that do not delimit a closed memory-issue epoch.

    The generic membar analysis can insert a local barrier within one async
    group, or immediately before an async-wait-synchronized local load. Those
    cases have no source issue boundary to preserve. A barrier between closed
    async groups is different: it is an explicit compiler ordering assumption
    and must reach the completion-free barrier-order model.
    """
    synchronous_lds_regions = _synchronous_lds_region_ids(target_program)
    regions = tuple(
        _eliminate_redundant_compiler_membar_barriers_in_region(
            target_program,
            region,
            synchronous_lds_regions,
        ) for region in target_program.regions)
    if regions == target_program.regions:
        return target_program
    return replace(target_program, regions=regions)


def _eliminate_redundant_compiler_membar_barriers_in_region(
    target_program,
    region,
    synchronous_lds_regions,
):
    # A nested region may observe a synchronous LDS frontier from a loop
    # backedge or branch predecessor. An explicit wait establishes a new,
    # known epoch.
    has_synchronous_lds_predecessor = region.target_region_id != 0
    has_committed_dma_epoch = False
    has_open_dma_group = False
    retained_op_ids = []
    region_op_ids = tuple(int(op_id) for op_id in region.op_ids)

    for position, op_id in enumerate(region_op_ids):
        op = target_program.ops[op_id]
        attrs = target_ir.attrs_dict(op)

        if op.kind == "barrier":
            next_op = (target_program.ops[region_op_ids[position + 1]] if position + 1 < len(region_op_ids) else None)
            next_is_wait_synchronized_load = (next_op is not None and next_op.kind in {
                "local_load",
            } and bool(target_ir.attrs_dict(next_op).get(
                "synced_via_async_wait",
                False,
            )))
            delimits_closed_dma_epoch = (has_committed_dma_epoch and not has_open_dma_group)
            is_redundant_dma_membar = (bool(attrs.get("compiler_membar_barrier", False))
                                       and int(attrs.get("address_space", 0)) == 1 and not op.operands
                                       and not has_synchronous_lds_predecessor
                                       and (next_is_wait_synchronized_load or not delimits_closed_dma_epoch))
            if not is_redundant_dma_membar:
                retained_op_ids.append(op_id)
                has_synchronous_lds_predecessor = False
                has_committed_dma_epoch = False
                has_open_dma_group = False
            continue

        retained_op_ids.append(op_id)
        if (op.kind == "async_wait" and attrs.get("publication_mode") == "workgroup"):
            has_synchronous_lds_predecessor = False
            has_committed_dma_epoch = False
            has_open_dma_group = False
            continue
        if (op.kind == "buffer_load_to_local" and attrs.get("mode") == "symbolic_copy"):
            has_open_dma_group = True
            continue
        if op.kind == "async_commit_group":
            has_committed_dma_epoch = True
            has_open_dma_group = False
            continue
        if _is_synchronous_lds_op(op):
            has_synchronous_lds_predecessor = True
            continue
        if op.kind == "cond_barrier" or any(int(region_id) in synchronous_lds_regions for region_id in op.region_ids):
            has_synchronous_lds_predecessor = True

    return target_ir.TargetRegion(
        region.target_region_id,
        tuple(retained_op_ids),
        region.block_arg_ids,
        region.yield_value_ids,
    )


def _synchronous_lds_region_ids(target_program):
    synchronous = set()
    changed = True
    while changed:
        changed = False
        for region in target_program.regions:
            if region.target_region_id in synchronous:
                continue
            if any(
                    _is_synchronous_lds_op(target_program.ops[int(op_id)]) or any(
                        int(nested_region_id) in synchronous
                        for nested_region_id in target_program.ops[int(op_id)].region_ids)
                    for op_id in region.op_ids):
                synchronous.add(region.target_region_id)
                changed = True
    return frozenset(synchronous)


def _is_synchronous_lds_op(op):
    return op.kind in {
        "local_load",
        "local_store",
    }
