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
        )
        for region in target_program.regions
    )
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
            next_op = (
                target_program.ops[region_op_ids[position + 1]]
                if position + 1 < len(region_op_ids)
                else None
            )
            next_is_wait_synchronized_load = (
                next_op is not None
                and next_op.kind in {
                    "local_load",
                    "local_load_mma_payload",
                }
                and bool(
                    target_ir.attrs_dict(next_op).get(
                        "synced_via_async_wait",
                        False,
                    )
                )
            )
            delimits_closed_dma_epoch = (
                has_committed_dma_epoch and not has_open_dma_group
            )
            is_redundant_dma_membar = (
                bool(attrs.get("compiler_membar_barrier", False))
                and int(attrs.get("address_space", 0)) == 1
                and not op.operands
                and not has_synchronous_lds_predecessor
                and (
                    next_is_wait_synchronized_load
                    or not delimits_closed_dma_epoch
                )
            )
            if not is_redundant_dma_membar:
                retained_op_ids.append(op_id)
                has_synchronous_lds_predecessor = False
                has_committed_dma_epoch = False
                has_open_dma_group = False
            continue

        retained_op_ids.append(op_id)
        if (
            op.kind == "async_wait"
            and attrs.get("publication_mode") == "workgroup"
        ):
            has_synchronous_lds_predecessor = False
            has_committed_dma_epoch = False
            has_open_dma_group = False
            continue
        if (
            op.kind == "buffer_load_to_local"
            and attrs.get("mode") == "symbolic_copy"
        ):
            has_open_dma_group = True
            continue
        if op.kind == "async_commit_group":
            has_committed_dma_epoch = True
            has_open_dma_group = False
            continue
        if _is_synchronous_lds_op(op):
            has_synchronous_lds_predecessor = True
            continue
        if op.kind == "cond_barrier" or any(
            int(region_id) in synchronous_lds_regions
            for region_id in op.region_ids
        ):
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
                _is_synchronous_lds_op(target_program.ops[int(op_id)])
                or any(
                    int(nested_region_id) in synchronous
                    for nested_region_id in
                    target_program.ops[int(op_id)].region_ids
                )
                for op_id in region.op_ids
            ):
                synchronous.add(region.target_region_id)
                changed = True
    return frozenset(synchronous)


def _is_synchronous_lds_op(op):
    if op.kind in {
        "local_load",
        "local_load_mma_payload",
        "local_store",
    }:
        return True
    return (
        op.kind == "buffer_load_to_local"
        and target_ir.attrs_dict(op).get("mode") == "scalarized_load_store"
    )


def eliminate_dead_target_ops(target_program):
    producer_by_result = {}
    for op in target_program.ops:
        for result in op.results:
            producer_by_result[int(result)] = op.target_op_id

    provenance_slice_ops = _provenance_slice_op_ids(
        target_program,
        producer_by_result,
    )
    live_ops = set()
    worklist = []
    for region in target_program.regions:
        for op_id in region.op_ids:
            op = target_program.ops[int(op_id)]
            if not _is_dead_eliminable(op, provenance_slice_ops):
                live_ops.add(op.target_op_id)
                worklist.append(op.target_op_id)
        for value_id in region.yield_value_ids:
            producer_id = producer_by_result.get(int(value_id))
            if producer_id is not None and producer_id not in live_ops:
                live_ops.add(producer_id)
                worklist.append(producer_id)

    while worklist:
        op = target_program.ops[worklist.pop()]
        for operand in _live_operands(op):
            producer_id = producer_by_result.get(int(operand))
            if producer_id is not None and producer_id not in live_ops:
                live_ops.add(producer_id)
                worklist.append(producer_id)

    regions = tuple(
        target_ir.TargetRegion(
            region.target_region_id,
            tuple(op_id for op_id in region.op_ids if int(op_id) in live_ops),
            region.block_arg_ids,
            region.yield_value_ids,
        ) for region in target_program.regions)
    if regions == target_program.regions:
        return target_program
    return replace(target_program, regions=regions)


def _is_dead_eliminable(op, provenance_slice_ops):
    return (
        op.kind in {
            "affine_materialize",
            "component_join",
            "component_split",
            "layout_convert",
            "make_buffer",
            "type_convert",
        }
        or op.target_op_id in provenance_slice_ops
    )


_PROVENANCE_SLICE_PURE_OPS = frozenset({
    "addptr",
    "binary",
    "broadcast",
    "cmpi",
    "component_join",
    "component_split",
    "constant",
    "expand_dims",
    "layout_convert",
    "make_range",
    "make_buffer",
    "maxsi",
    "minsi",
    "program_id",
    "select",
    "splat",
})


def _provenance_slice_op_ids(target_program, producer_by_result):
    worklist = [
        int(value_id)
        for op in target_program.ops
        for value_id in target_ir.attrs_dict(op).get(
            target_ir.PROVENANCE_ONLY_TARGET_IDS_ATTR,
            (),
        )
    ]
    slice_ops = set()
    while worklist:
        producer_id = producer_by_result.get(worklist.pop())
        if producer_id is None or producer_id in slice_ops:
            continue
        producer = target_program.ops[int(producer_id)]
        if producer.kind not in _PROVENANCE_SLICE_PURE_OPS:
            continue
        slice_ops.add(producer_id)
        worklist.extend(int(operand) for operand in producer.operands)
    return frozenset(slice_ops)


def _live_operands(op):
    attrs = target_ir.attrs_dict(op)
    provenance_only = frozenset(
        int(value_id)
        for value_id in attrs.get(
            target_ir.PROVENANCE_ONLY_TARGET_IDS_ATTR,
            (),
        )
    )
    if provenance_only:
        return tuple(
            operand for operand in op.operands
            if int(operand) not in provenance_only
        )
    return op.operands
