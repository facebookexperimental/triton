"""Explicit memory ordering around ordering barriers."""

from dataclasses import replace

from . import target_ir
from .diagnostics import fail

STAGE = "barrier_order"

_PRE_BARRIER_PROVENANCE = "memory_barrier_predecessors"
_POST_BARRIER_PROVENANCE = "memory_barrier_successors"


def thread_barrier_issue_order(target_program):
    """Thread sparse completion and issue frontiers through each target region.

    An ordering barrier orders real memory issue on either side, but it must
    not turn global memory or direct-to-LDS DMA completion into an implicit
    wait. Their tokens are projected through ``issue_token`` before the
    barrier. LDS reads must complete before a workgroup barrier can release a
    peer wave to overwrite the storage, so their completion remains an
    ordinary SSA dependency. The barrier result is projected before reaching
    following memory issuers.

    Structured operations carry only the LDS-read completion frontier needed
    by a nested source barrier. DMA completion is never part of that frontier.
    """
    values = list(target_program.values)
    ops = list(target_program.ops)
    regions = []

    def resource_targets(target_value_ids):
        return tuple(
            dict.fromkeys(resource_target_id for target_value_id in target_value_ids
                          for resource_target_id in values[int(target_value_id)].resource_target_ids))

    def add_token_value(domain, debug_name, resource_target_ids=()):
        value_id = len(values)
        values.append(
            target_ir.TargetValue(
                value_id,
                target_ir.TargetType("token", "token"),
                debug_name=str(debug_name),
                event_domain=str(domain),
                resource_target_ids=tuple(resource_target_ids),
            ))
        return value_id

    def add_issue_token(operands, domain, provenance, source_op_index):
        operands = tuple(dict.fromkeys(int(value_id) for value_id in operands))
        if not operands:
            fail(
                "TLXW_BARRIER_ORDER_EMPTY_PROJECTION",
                STAGE,
                "issue-order projection requires at least one input token",
                source_op_index=source_op_index,
            )
        result_id = add_token_value(
            domain,
            f"{domain}_{source_op_index}_{len(ops)}",
            resource_targets(operands),
        )
        op_id = len(ops)
        ops.append(
            target_ir.TargetOp(
                op_id,
                "issue_token",
                operands,
                (result_id, ),
                target_ir._attrs_tuple(
                    {
                        "input_count": len(operands),
                        "projection_domain": str(domain),
                        "projection_provenance": str(provenance),
                    }, op_id),
                source_op_index=source_op_index,
            ))
        return op_id, result_id

    for region in target_program.regions:
        original_op_ids = tuple(int(op_id) for op_id in region.op_ids)
        has_later_issue_barrier = _suffix_matches(
            original_op_ids,
            ops,
            _orders_memory_issue,
        )
        has_later_completion_barrier = _suffix_matches(
            original_op_ids,
            ops,
            _completes_lds_reads,
        )
        has_later_ordered_op = _suffix_matches(
            original_op_ids,
            ops,
            lambda op: (op.kind in target_ir.MEMORY_ISSUER_OP_KINDS or _orders_memory_issue(op)),
        )
        ordered_op_ids = []
        preceding_memory_tokens = []
        preceding_completion_tokens = []
        barrier_epoch = None

        for position, op_id in enumerate(original_op_ids):
            op = ops[op_id]
            if op.kind in target_ir.MEMORY_ISSUER_OP_KINDS:
                if barrier_epoch is not None:
                    op = _append_barrier_dependency(op, barrier_epoch)
                    ops[op_id] = op
                requires_completion = _requires_barrier_completion(op)
                needs_frontier_result = (has_later_completion_barrier[position]
                                         if requires_completion else has_later_issue_barrier[position])
                if needs_frontier_result:
                    op, completion_id = _ensure_memory_completion_result(
                        op,
                        values,
                    )
                    ops[op_id] = op
                    if requires_completion:
                        preceding_completion_tokens.append(completion_id)
                    else:
                        preceding_memory_tokens.append(completion_id)
                ordered_op_ids.append(op_id)
                continue

            completes_lds_reads = _completes_lds_reads(op)
            orders_memory_issue = _orders_memory_issue(op)
            if not completes_lds_reads and not orders_memory_issue:
                ordered_op_ids.append(op_id)
                continue

            if completes_lds_reads and preceding_completion_tokens:
                op = _append_barrier_lds_read_dependencies(
                    op,
                    preceding_completion_tokens,
                )
                preceding_completion_tokens = []
                ops[op_id] = op

            if not orders_memory_issue:
                ordered_op_ids.append(op_id)
                continue

            issue_dependency = None
            if preceding_memory_tokens:
                issue_op_id, issue_dependency = add_issue_token(
                    preceding_memory_tokens,
                    target_ir.EVENT_DOMAIN_MEMORY_ISSUE,
                    _PRE_BARRIER_PROVENANCE,
                    op.source_op_index,
                )
                ordered_op_ids.append(issue_op_id)
            elif barrier_epoch is not None:
                # Consecutive ordering barriers need no extra projection: the
                # previous epoch is already completion-free.
                issue_dependency = barrier_epoch

            if issue_dependency is not None:
                op = _append_barrier_dependency(op, issue_dependency)
            ops[op_id] = op
            ordered_op_ids.append(op_id)

            preceding_memory_tokens = []
            preceding_completion_tokens = []
            barrier_epoch = None
            if not has_later_ordered_op[position]:
                continue

            op, barrier_result_id = _ensure_ordering_barrier_result(op, values)
            ops[op_id] = op
            epoch_op_id, barrier_epoch = add_issue_token(
                (barrier_result_id, ),
                target_ir.EVENT_DOMAIN_BARRIER_ISSUE,
                _POST_BARRIER_PROVENANCE,
                op.source_op_index,
            )
            ordered_op_ids.append(epoch_op_id)

        regions.append(
            target_ir.TargetRegion(
                region.target_region_id,
                tuple(ordered_op_ids),
                region.block_arg_ids,
                region.yield_value_ids,
            ))

    ordered_program = replace(
        target_program,
        values=tuple(values),
        ops=tuple(ops),
        regions=tuple(regions),
    )
    return _thread_structured_lds_read_completion(ordered_program)


def _thread_structured_lds_read_completion(target_program):
    """Carry the explicit LDS-read frontier through structured target SSA.

    A source workgroup barrier inside a loop or branch still orders reads from
    the enclosing structured scope.  Represent that source ordering directly:
    local-load completion results form one sparse frontier, ``scf.if`` merges
    it as an ordinary result, and ``scf.for`` carries it as an ordinary
    iter_arg.  No allocation identity, alias relation, or memory destination is
    inspected.
    """
    if not any(_completes_lds_reads(op) for op in target_program.ops):
        return target_program

    values = list(target_program.values)
    ops = list(target_program.ops)
    regions = list(target_program.regions)

    def resource_targets(target_value_ids):
        return tuple(
            dict.fromkeys(resource_target_id for target_value_id in target_value_ids
                          for resource_target_id in values[int(target_value_id)].resource_target_ids))

    def add_value(domain, debug_name, resource_target_ids=()):
        target_value_id = len(values)
        values.append(
            target_ir.TargetValue(
                target_value_id,
                target_ir.TargetType("token", "token"),
                debug_name=str(debug_name),
                event_domain=str(domain),
                resource_target_ids=tuple(resource_target_ids),
            ))
        return target_value_id

    def ensure_completion_barrier_result(op):
        if len(op.results) > 1:
            fail(
                "TLXW_BARRIER_ORDER_BARRIER_RESULT",
                STAGE,
                "LDS completion barrier may expose at most one token result",
                target_op_id=op.target_op_id,
            )
        attrs = target_ir.attrs_dict(op)
        attrs["lds_completion_result_count"] = 1
        if op.results:
            return replace(
                op,
                attrs=target_ir._attrs_tuple(
                    attrs,
                    op.target_op_id,
                ),
            ), int(op.results[0])
        result_id = add_value(
            target_ir.EVENT_DOMAIN_MEMORY_COMPLETION,
            f"lds_completion_barrier_{op.target_op_id}",
            resource_targets(op.operands),
        )
        return replace(
            op,
            results=(result_id, ),
            attrs=target_ir._attrs_tuple(
                attrs,
                op.target_op_id,
            ),
        ), result_id

    def add_empty_token(op_ids, source_op_index, debug_name):
        result_id = add_value(
            target_ir.EVENT_DOMAIN_EMPTY,
            debug_name,
        )
        op_id = len(ops)
        ops.append(
            target_ir.TargetOp(
                op_id,
                "token",
                (),
                (result_id, ),
                target_ir._attrs_tuple({
                    "event_domain": target_ir.EVENT_DOMAIN_EMPTY,
                }, op_id),
                source_op_index=source_op_index,
            ))
        op_ids.append(op_id)
        return result_id

    def join_frontier(
        op_ids,
        target_value_ids,
        source_op_index,
        debug_name,
    ):
        target_value_ids = tuple(dict.fromkeys(int(value_id) for value_id in target_value_ids))
        if len(target_value_ids) == 1:
            return target_value_ids[0]
        if not target_value_ids:
            return add_empty_token(
                op_ids,
                source_op_index,
                debug_name,
            )
        result_id = add_value(
            target_ir.EVENT_DOMAIN_MEMORY_COMPLETION,
            debug_name,
            resource_targets(target_value_ids),
        )
        op_id = len(ops)
        ops.append(
            target_ir.TargetOp(
                op_id,
                "token_join",
                target_value_ids,
                (result_id, ),
                target_ir._attrs_tuple(
                    {
                        "event_domain": target_ir.EVENT_DOMAIN_MEMORY_COMPLETION,
                        "input_count": len(target_value_ids),
                    }, op_id),
                source_op_index=source_op_index,
            ))
        op_ids.append(op_id)
        return result_id

    region_has_local_read_cache = {}

    def region_has_local_read(region_id):
        region_id = int(region_id)
        cached = region_has_local_read_cache.get(region_id)
        if cached is not None:
            return cached
        # Break impossible malformed recursive-region cycles defensively.
        region_has_local_read_cache[region_id] = False
        result = any(op.kind == "local_load" or any(
            region_has_local_read(child_region_id)
            for child_region_id in op.region_ids)
                     for op_id in regions[region_id].op_ids
                     for op in (ops[int(op_id)], ))
        region_has_local_read_cache[region_id] = result
        return result

    processed_regions = set()

    def process_region(region_id, incoming_completion_ids=()):
        region_id = int(region_id)
        if region_id in processed_regions:
            fail(
                "TLXW_BARRIER_ORDER_REGION_REENTRY",
                STAGE,
                "structured completion region was reached more than once",
                target_region_id=region_id,
            )
        processed_regions.add(region_id)
        region = regions[region_id]
        ordered_op_ids = []
        completion_frontier = list(dict.fromkeys(int(value_id) for value_id in incoming_completion_ids))

        for op_id in region.op_ids:
            op_id = int(op_id)
            op = ops[op_id]

            if op.kind == "local_load":
                op, completion_id = _ensure_memory_completion_result(
                    op,
                    values,
                )
                ops[op_id] = op
                completion_frontier.append(completion_id)
                completion_frontier = list(dict.fromkeys(completion_frontier))
                ordered_op_ids.append(op_id)
                continue

            if _completes_lds_reads(op):
                if completion_frontier:
                    op = _append_barrier_lds_read_dependencies(
                        op,
                        completion_frontier,
                    )
                    op, completion_id = ensure_completion_barrier_result(op)
                    completion_frontier = [completion_id]
                else:
                    completion_frontier = []
                ops[op_id] = op
                ordered_op_ids.append(op_id)
                continue

            if op.kind == "if" and len(op.region_ids) == 2:
                branch_ids = tuple(int(value) for value in op.region_ids)
                needs_carry = bool(completion_frontier) or any(
                    region_has_local_read(branch_id) for branch_id in branch_ids)
                if not needs_carry:
                    for branch_id in branch_ids:
                        process_region(branch_id)
                    ordered_op_ids.append(op_id)
                    continue

                branch_yield_ids = []
                branch_resource_ids = []
                for branch_index, branch_id in enumerate(branch_ids):
                    outgoing_ids = process_region(
                        branch_id,
                        completion_frontier,
                    )
                    branch_op_ids = list(regions[branch_id].op_ids)
                    yield_id = join_frontier(
                        branch_op_ids,
                        outgoing_ids,
                        op.source_op_index,
                        (f"if_lds_completion_{op.target_op_id}_"
                         f"{branch_index}"),
                    )
                    branch = regions[branch_id]
                    regions[branch_id] = replace(
                        branch,
                        op_ids=tuple(branch_op_ids),
                        yield_value_ids=(
                            *branch.yield_value_ids,
                            yield_id,
                        ),
                    )
                    branch_yield_ids.append(yield_id)
                    branch_resource_ids.extend(values[yield_id].resource_target_ids)

                result_id = add_value(
                    target_ir.EVENT_DOMAIN_MEMORY_COMPLETION,
                    f"if_lds_completion_result_{op.target_op_id}",
                    tuple(dict.fromkeys(branch_resource_ids)),
                )
                attrs = target_ir.attrs_dict(op)
                attrs["lds_completion_result_count"] = 1
                ops[op_id] = replace(
                    op,
                    results=(*op.results, result_id),
                    attrs=target_ir._attrs_tuple(
                        attrs,
                        op.target_op_id,
                    ),
                )
                completion_frontier = [result_id]
                ordered_op_ids.append(op_id)
                continue

            if op.kind == "for_loop" and len(op.region_ids) == 1:
                body_region_id = int(op.region_ids[0])
                needs_carry = (bool(completion_frontier) or region_has_local_read(body_region_id))
                if not needs_carry:
                    process_region(body_region_id)
                    ordered_op_ids.append(op_id)
                    continue

                init_id = join_frontier(
                    ordered_op_ids,
                    completion_frontier,
                    op.source_op_index,
                    f"loop_lds_completion_init_{op.target_op_id}",
                )
                block_arg_id = add_value(
                    target_ir.EVENT_DOMAIN_MEMORY_COMPLETION,
                    f"loop_lds_completion_arg_{op.target_op_id}",
                    values[init_id].resource_target_ids,
                )
                result_id = add_value(
                    target_ir.EVENT_DOMAIN_MEMORY_COMPLETION,
                    f"loop_lds_completion_result_{op.target_op_id}",
                    values[init_id].resource_target_ids,
                )
                body_region = regions[body_region_id]
                regions[body_region_id] = replace(
                    body_region,
                    block_arg_ids=(
                        *body_region.block_arg_ids,
                        block_arg_id,
                    ),
                )
                outgoing_ids = process_region(
                    body_region_id,
                    (block_arg_id, ),
                )
                body_op_ids = list(regions[body_region_id].op_ids)
                yield_id = join_frontier(
                    body_op_ids,
                    outgoing_ids,
                    op.source_op_index,
                    f"loop_lds_completion_yield_{op.target_op_id}",
                )
                body_region = regions[body_region_id]
                regions[body_region_id] = replace(
                    body_region,
                    op_ids=tuple(body_op_ids),
                    yield_value_ids=(
                        *body_region.yield_value_ids,
                        yield_id,
                    ),
                )
                carry_resources = resource_targets((init_id, yield_id))
                values[block_arg_id] = replace(
                    values[block_arg_id],
                    resource_target_ids=carry_resources,
                )
                values[result_id] = replace(
                    values[result_id],
                    resource_target_ids=carry_resources,
                )
                attrs = target_ir.attrs_dict(op)
                attrs["init_arg_count"] = (int(attrs.get("init_arg_count", 0)) + 1)
                attrs["lds_completion_carry_count"] = 1
                ops[op_id] = replace(
                    op,
                    operands=(*op.operands, init_id),
                    results=(*op.results, result_id),
                    attrs=target_ir._attrs_tuple(
                        attrs,
                        op.target_op_id,
                    ),
                )
                completion_frontier = [result_id]
                ordered_op_ids.append(op_id)
                continue

            for child_region_id in op.region_ids:
                process_region(child_region_id)
            ordered_op_ids.append(op_id)

        regions[region_id] = replace(
            regions[region_id],
            op_ids=tuple(ordered_op_ids),
        )
        return tuple(dict.fromkeys(completion_frontier))

    process_region(0)
    for region in regions:
        if int(region.target_region_id) not in processed_regions:
            process_region(region.target_region_id)

    return replace(
        target_program,
        values=tuple(values),
        ops=tuple(ops),
        regions=tuple(regions),
    )


def _suffix_matches(op_ids, ops, predicate):
    result = [False] * len(op_ids)
    seen = False
    for position in range(len(op_ids) - 1, -1, -1):
        result[position] = seen
        if predicate(ops[int(op_ids[position])]):
            seen = True
    return tuple(result)


def _orders_memory_issue(op):
    return (op.kind == "barrier" and bool(target_ir.attrs_dict(op).get("orders_memory_issue", False)))


def _completes_lds_reads(op):
    # Every source workgroup barrier is a literal completion point for
    # preceding LDS reads. This is structural source semantics, not an
    # allocation-identity or alias inference.
    return op.kind == "barrier"


def _requires_barrier_completion(op):
    # An LDS read must finish before a workgroup barrier can allow another
    # wave to overwrite the storage. Other memory operations need only remain
    # issued before the barrier; in particular, direct DMA completion is owned
    # exclusively by the explicit async_wait protocol.
    return op.kind == "local_load"


def _append_barrier_lds_read_dependencies(op, target_value_ids):
    attrs = target_ir.attrs_dict(op)
    count = int(attrs.get("lds_read_dependency_count", 0))
    issue_count = int(attrs.get("barrier_order_dependency_count", 0))
    if count < 0 or issue_count < 0 or count + issue_count > len(op.operands):
        fail(
            "TLXW_BARRIER_ORDER_COMPLETION_SEGMENT",
            STAGE,
            "target barrier has malformed completion or issue segments",
            target_op_id=op.target_op_id,
        )
    issue_begin = len(op.operands) - issue_count
    completion_begin = issue_begin - count
    ordinary_operands = op.operands[:completion_begin]
    existing_completion_operands = op.operands[completion_begin:issue_begin]
    issue_operands = op.operands[issue_begin:]
    target_value_ids = tuple(target_value_id for target_value_id in dict.fromkeys(
        int(target_value_id) for target_value_id in target_value_ids)
                             if target_value_id not in existing_completion_operands)
    completion_operands = (
        *existing_completion_operands,
        *target_value_ids,
    )
    attrs["lds_read_dependency_count"] = len(completion_operands)
    return replace(
        op,
        operands=(
            *ordinary_operands,
            *completion_operands,
            *issue_operands,
        ),
        attrs=target_ir._attrs_tuple(attrs, op.target_op_id),
    )


def _append_barrier_dependency(op, target_value_id):
    attrs = target_ir.attrs_dict(op)
    count = int(attrs.get("barrier_order_dependency_count", 0))
    if count:
        fail(
            "TLXW_BARRIER_ORDER_DUPLICATE_DEPENDENCY",
            STAGE,
            f"target {op.kind} already has a barrier-order dependency",
            target_op_id=op.target_op_id,
        )
    attrs["barrier_order_dependency_count"] = 1
    return replace(
        op,
        operands=(*op.operands, int(target_value_id)),
        attrs=target_ir._attrs_tuple(attrs, op.target_op_id),
    )


def _ensure_memory_completion_result(op, values):
    existing = _existing_memory_completion_result(op, values)
    if existing is not None:
        return op, existing

    attrs = target_ir.attrs_dict(op)
    synthetic_count = int(attrs.get("issue_order_result_count", 0))
    if synthetic_count:
        fail(
            "TLXW_BARRIER_ORDER_RESULT_SEGMENT",
            STAGE,
            f"target {op.kind} has a malformed issue-order result segment",
            target_op_id=op.target_op_id,
        )
    result_id = len(values)
    values.append(
        target_ir.TargetValue(
            result_id,
            target_ir.TargetType("token", "token"),
            debug_name=f"memory_completion_{op.target_op_id}",
            event_domain=target_ir.EVENT_DOMAIN_MEMORY_COMPLETION,
            resource_target_ids=tuple(
                dict.fromkeys(resource_target_id for target_value_id in op.operands
                              for resource_target_id in values[int(target_value_id)].resource_target_ids)),
        ))
    attrs["issue_order_result_count"] = 1
    if op.kind in {
            "local_load",
            "local_store",
    }:
        attrs["completion_result_count"] = (int(attrs.get("completion_result_count", 0)) + 1)
    return replace(
        op,
        results=(*op.results, result_id),
        attrs=target_ir._attrs_tuple(attrs, op.target_op_id),
    ), result_id


def _existing_memory_completion_result(op, values):
    if op.kind == "buffer_load_to_local":
        if len(op.results) != 1:
            fail(
                "TLXW_BARRIER_ORDER_DMA_RESULT",
                STAGE,
                "direct DMA must expose exactly one completion token",
                target_op_id=op.target_op_id,
            )
        return int(op.results[0])

    if op.kind not in {
            "local_load",
            "local_store",
    }:
        return None
    completion_count = int(target_ir.attrs_dict(op).get("completion_result_count", 0))
    if completion_count == 0:
        return None
    if completion_count != 1 or not op.results:
        fail(
            "TLXW_BARRIER_ORDER_LOCAL_RESULT",
            STAGE,
            "local memory completion segment must contain exactly one token",
            target_op_id=op.target_op_id,
        )
    result_id = int(op.results[-1])
    if values[result_id].type.representation != "token":
        fail(
            "TLXW_BARRIER_ORDER_LOCAL_RESULT",
            STAGE,
            "local memory completion result must be a token",
            target_op_id=op.target_op_id,
            target_value_id=result_id,
        )
    return result_id


def _ensure_ordering_barrier_result(op, values):
    if len(op.results) > 1:
        fail(
            "TLXW_BARRIER_ORDER_BARRIER_RESULT",
            STAGE,
            "issue-ordering barrier may expose at most one token result",
            target_op_id=op.target_op_id,
        )
    if op.results:
        result_id = int(op.results[0])
        value = values[result_id]
        values[result_id] = replace(
            value,
            event_domain=target_ir.EVENT_DOMAIN_MEMORY_BARRIER,
        )
        return op, result_id

    result_id = len(values)
    values.append(
        target_ir.TargetValue(
            result_id,
            target_ir.TargetType("token", "token"),
            debug_name=f"memory_barrier_{op.target_op_id}",
            event_domain=target_ir.EVENT_DOMAIN_MEMORY_BARRIER,
            resource_target_ids=tuple(
                dict.fromkeys(resource_target_id for target_value_id in op.operands
                              for resource_target_id in values[int(target_value_id)].resource_target_ids)),
        ))
    return replace(op, results=(result_id, )), result_id
