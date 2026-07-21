"""Target-program canonicalization for the TLX Wave converter."""

from . import target_ir


def canonicalize_target_program(target_program):
    target_program = _hoist_async_waits_before_mma_payload_loads(target_program)
    target_program = _fuse_single_use_cmpi_selects(target_program)
    if len(target_program.regions) != 1:
        return target_program
    return _share_div_rem_pairs(target_program)


def eliminate_redundant_compiler_membar_barriers(target_program):
    """Remove compiler LDS rendezvous that follow only async DMA issues.

    AMD membar sees storage aliases before the bridge resolves structural
    memdesc views, so it can conservatively put a local barrier between
    independent direct-to-LDS issues.  Such a barrier cannot complete a DMA;
    readiness remains exclusive to an explicit async wait.  Keep every
    pre-existing source barrier and every membar barrier that follows a
    synchronous LDS access, while dropping only compiler-created barriers in
    DMA-only epochs.
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
    return target_ir.TargetProgram(
        target_program.values,
        target_program.ops,
        regions,
        dict(target_program.source_value_targets),
        dict(target_program.erased_source_values),
        target_program.kernel,
    )


def _eliminate_redundant_compiler_membar_barriers_in_region(
    target_program,
    region,
    synchronous_lds_regions,
):
    # A nested region may observe a synchronous LDS frontier from a loop
    # backedge or branch predecessor.  Start conservatively; a workgroup wait
    # or retained barrier establishes a new, known rendezvous epoch.
    has_synchronous_lds_predecessor = region.target_region_id != 0
    retained_op_ids = []
    for op_id in region.op_ids:
        op = target_program.ops[int(op_id)]
        attrs = target_ir.attrs_dict(op)

        if op.kind == "barrier":
            is_redundant_dma_only_membar = (
                bool(attrs.get("compiler_membar_barrier", False))
                and int(attrs.get("address_space", 0)) == 1
                and not op.operands
                and not has_synchronous_lds_predecessor
            )
            if not is_redundant_dma_only_membar:
                retained_op_ids.append(op_id)
                has_synchronous_lds_predecessor = False
            continue

        retained_op_ids.append(op_id)
        if (
            op.kind == "async_wait"
            and attrs.get("publication_mode") == "workgroup"
        ):
            has_synchronous_lds_predecessor = False
            continue
        if op.kind in {
            "local_load",
            "local_load_mma_payload",
            "local_store",
        }:
            has_synchronous_lds_predecessor = True
            continue
        if (
            op.kind == "buffer_load_to_local"
            and attrs.get("mode") == "scalarized_load_store"
        ):
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
    return target_ir.TargetProgram(
        target_program.values,
        target_program.ops,
        regions,
        dict(target_program.source_value_targets),
        dict(target_program.erased_source_values),
        target_program.kernel,
    )


def _is_dead_eliminable(op, provenance_slice_ops):
    return (
        op.kind in {
            "affine_materialize",
            "layout_convert",
            "type_convert",
        }
        or op.target_op_id in provenance_slice_ops
    )


_PROVENANCE_SLICE_PURE_OPS = frozenset({
    "addptr",
    "binary",
    "broadcast",
    "cmpi",
    "cmpi_select",
    "constant",
    "expand_dims",
    "layout_convert",
    "make_range",
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


_ASYNC_WAIT_HOIST_PURE_OPS = frozenset({
    "addptr",
    "assume",
    "binary",
    "broadcast",
    "cmpi",
    "cmpi_select",
    "constant",
    "expand_dims",
    "float_binary",
    "float_cast",
    "mma_packet_truncf",
    "make_range",
    "memdesc_index",
    "memdesc_view",
    "maxsi",
    "minsi",
    "program_id",
    "select",
    "splat",
    "token",
})


def _hoist_async_waits_before_mma_payload_loads(target_program):
    regions = []
    changed = False
    for region in target_program.regions:
        op_ids = list(region.op_ids)
        index = 0
        while index < len(op_ids):
            op = target_program.ops[int(op_ids[index])]
            if op.kind != "async_wait":
                index += 1
                continue
            insert = index
            while insert > 0:
                previous = target_program.ops[int(op_ids[insert - 1])]
                if not _can_hoist_async_wait_over(target_program, op, previous):
                    break
                insert -= 1
            if insert == index:
                index += 1
                continue
            op_id = op_ids.pop(index)
            op_ids.insert(insert, op_id)
            changed = True
            index = insert + 1
        regions.append(
            target_ir.TargetRegion(
                region.target_region_id,
                tuple(op_ids),
                region.block_arg_ids,
                region.yield_value_ids,
            ))
    if not changed:
        return target_program
    return target_ir.TargetProgram(
        target_program.values,
        target_program.ops,
        tuple(regions),
        dict(target_program.source_value_targets),
        dict(target_program.erased_source_values),
        target_program.kernel,
    )


def _can_hoist_async_wait_over(target_program, wait_op, previous_op):
    if previous_op.region_ids:
        return False
    if any(int(result) in wait_op.operands for result in previous_op.results):
        return False
    if previous_op.kind == "local_load_mma_payload":
        return True
    if previous_op.kind not in _ASYNC_WAIT_HOIST_PURE_OPS:
        return False
    return all(
        target_program.values[int(result)].type.representation != "token"
        for result in previous_op.results
    )


def _fuse_single_use_cmpi_selects(target_program):
    """Fuse a compare used only by a data select into one target operation.

    A distributed compare result is a tuple of physical Wave masks.  Emitting
    that tuple at the producer boundary makes every SGPR mask live until the
    following select.  The fused target operation preserves ordinary target
    SSA semantics while allowing mechanical emission in register-bounded
    compare/select batches, just like a scalar backend instruction selector.
    """
    producer_by_result = {}
    use_count = {}
    region_by_op = {}
    for region in target_program.regions:
        for op_id in region.op_ids:
            region_by_op[int(op_id)] = int(region.target_region_id)
    for op in target_program.ops:
        for result in op.results:
            producer_by_result[int(result)] = op
        for operand in op.operands:
            use_count[int(operand)] = use_count.get(int(operand), 0) + 1

    fused_by_select = {}
    removed = set()
    for select in target_program.ops:
        if select.kind != "select" or len(select.operands) != 3:
            continue
        condition_id = int(select.operands[0])
        compare = producer_by_result.get(condition_id)
        if (
            compare is None
            or compare.kind != "cmpi"
            or len(compare.operands) != 2
            or len(compare.results) != 1
            or use_count.get(condition_id, 0) != 1
            or region_by_op.get(compare.target_op_id)
            != region_by_op.get(select.target_op_id)
        ):
            continue
        result_type = target_program.values[int(select.results[0])].type
        condition_type = target_program.values[condition_id].type
        if (
            result_type.representation in {"mask", "mask_tuple"}
            or int(condition_type.component_count)
            != int(result_type.component_count)
        ):
            continue
        compare_attrs = target_ir.attrs_dict(compare)
        fused_by_select[select.target_op_id] = target_ir.TargetOp(
            select.target_op_id,
            "cmpi_select",
            (
                int(compare.operands[0]),
                int(compare.operands[1]),
                int(select.operands[1]),
                int(select.operands[2]),
            ),
            select.results,
            _attrs_tuple({
                "predicate": compare_attrs["predicate"],
                "source_width": compare_attrs["source_width"],
            }),
            select.fact_ids,
            select.fact_target_ids,
            select.layout_map_ids,
            select.region_ids,
            select.source_op_index,
        )
        removed.add(compare.target_op_id)

    if not fused_by_select:
        return target_program

    ops = []
    old_to_new = {}
    for old_op in target_program.ops:
        if old_op.target_op_id in removed:
            continue
        op = fused_by_select.get(old_op.target_op_id, old_op)
        new_id = len(ops)
        old_to_new[old_op.target_op_id] = new_id
        ops.append(target_ir.TargetOp(
            new_id,
            op.kind,
            op.operands,
            op.results,
            op.attrs,
            op.fact_ids,
            op.fact_target_ids,
            op.layout_map_ids,
            op.region_ids,
            op.source_op_index,
        ))

    regions = tuple(
        target_ir.TargetRegion(
            region.target_region_id,
            tuple(
                old_to_new[int(op_id)]
                for op_id in region.op_ids
                if int(op_id) not in removed
            ),
            region.block_arg_ids,
            region.yield_value_ids,
        )
        for region in target_program.regions
    )
    return target_ir.TargetProgram(
        target_program.values,
        tuple(ops),
        regions,
        dict(target_program.source_value_targets),
        dict(target_program.erased_source_values),
        target_program.kernel,
    )


def _share_div_rem_pairs(target_program):
    divs_by_key = {}
    for op in target_program.ops:
        operation = _binary_operation(op)
        if operation in {"divsi", "divui"}:
            divs_by_key.setdefault((_div_rem_flavor(operation), op.operands), op)

    if not any(
            _binary_operation(op) in {"remsi", "remui"} and (_div_rem_flavor(_binary_operation(op)),
                                                             op.operands) in divs_by_key for op in target_program.ops):
        return target_program

    values = list(target_program.values)
    ops = []
    skipped_op_ids = set()
    emitted_div_ids = set()

    def append_op(op, **updates):
        new_op = target_ir.TargetOp(
            len(ops),
            updates.get("kind", op.kind),
            tuple(updates.get("operands", op.operands)),
            tuple(updates.get("results", op.results)),
            _attrs_tuple(updates.get("attrs", target_ir.attrs_dict(op))),
            tuple(updates.get("fact_ids", op.fact_ids)),
            tuple(updates.get("fact_target_ids", op.fact_target_ids)),
            tuple(updates.get("layout_map_ids", op.layout_map_ids)),
            tuple(updates.get("region_ids", op.region_ids)),
            updates.get("source_op_index", op.source_op_index),
        )
        ops.append(new_op)
        return new_op

    for op in target_program.ops:
        if op.target_op_id in skipped_op_ids:
            continue

        operation = _binary_operation(op)
        if operation not in {"remsi", "remui"}:
            append_op(op)
            if operation in {"divsi", "divui"}:
                emitted_div_ids.add(op.target_op_id)
            continue

        div_op = divs_by_key.get((_div_rem_flavor(operation), op.operands))
        if div_op is None:
            append_op(op)
            continue

        if div_op.target_op_id not in emitted_div_ids:
            append_op(div_op)
            emitted_div_ids.add(div_op.target_op_id)
            skipped_op_ids.add(div_op.target_op_id)

        lhs, rhs = op.operands
        rem_result = _single_result(op)
        product_value_id = len(values)
        values.append(
            target_ir.TargetValue(
                product_value_id,
                target_program.values[rem_result].type,
                debug_name=f"rem_product_{rem_result}",
            ))
        attrs = target_ir.attrs_dict(op)
        source_width = attrs.get("source_width")
        binary_attrs = {}
        if source_width is not None:
            binary_attrs["source_width"] = source_width
        append_op(
            op,
            operands=(div_op.results[0], rhs),
            results=(product_value_id, ),
            attrs={**binary_attrs, "operation": "muli"},
            fact_ids=(),
            fact_target_ids=(),
            layout_map_ids=(),
        )
        append_op(
            op,
            operands=(lhs, product_value_id),
            attrs={**binary_attrs, "operation": "subi"},
        )

    return target_ir.TargetProgram(
        tuple(values),
        tuple(ops),
        _renumber_regions(target_program, ops),
        dict(target_program.source_value_targets),
        dict(target_program.erased_source_values),
        target_program.kernel,
    )


def _renumber_regions(target_program, ops):
    if not target_program.regions:
        return ()
    regions = list(target_program.regions)
    first = regions[0]
    regions[0] = target_ir.TargetRegion(
        first.target_region_id,
        tuple(op.target_op_id for op in ops),
        first.block_arg_ids,
        first.yield_value_ids,
    )
    return tuple(regions)


def _binary_operation(op):
    if op.kind != "binary":
        return None
    return target_ir.attrs_dict(op).get("operation")


def _div_rem_flavor(operation):
    if operation.endswith("si"):
        return "si"
    if operation.endswith("ui"):
        return "ui"
    return None


def _single_result(op):
    if len(op.results) != 1:
        raise AssertionError("binary target ops must have one result")
    return op.results[0]


def _attrs_tuple(attrs):
    return tuple(target_ir.TargetAttr(str(name), value) for name, value in sorted(attrs.items()))
