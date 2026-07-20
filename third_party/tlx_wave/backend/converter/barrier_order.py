"""Completion-free memory issue ordering around full-memory barriers."""

from dataclasses import replace

from . import target_ir
from .diagnostics import fail

STAGE = "barrier_order"

FULL_MEMORY_ADDRESS_SPACE = 31

_PRE_BARRIER_PROVENANCE = "full_barrier_predecessors"
_POST_BARRIER_PROVENANCE = "full_barrier_successors"


def thread_full_barrier_issue_order(target_program):
    """Thread sparse issue-only frontiers through each target region.

    A full-memory source barrier orders real memory issue on either side, but
    it must not turn an async DMA completion into an implicit wait.  Raw memory
    tokens are therefore projected through ``issue_token`` before the barrier.
    The barrier result is projected a second time before being shared by every
    following memory issuer.  Pure operations receive no dependency.

    Structured operations already delimit WaveAMDMachine scheduling regions,
    so the frontier is intentionally region-local and does not become an
    implicit loop/branch carry.
    """
    values = list(target_program.values)
    ops = list(target_program.ops)
    regions = []

    def add_token_value(domain, debug_name):
        value_id = len(values)
        values.append(target_ir.TargetValue(
            value_id,
            target_ir.TargetType("token", "token"),
            debug_name=str(debug_name),
            event_domain=str(domain),
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
        )
        op_id = len(ops)
        ops.append(target_ir.TargetOp(
            op_id,
            "issue_token",
            operands,
            (result_id, ),
            target_ir._attrs_tuple({
                "input_count": len(operands),
                "projection_domain": str(domain),
                "projection_provenance": str(provenance),
            }, op_id),
            source_op_index=source_op_index,
        ))
        return op_id, result_id

    for region in target_program.regions:
        original_op_ids = tuple(int(op_id) for op_id in region.op_ids)
        has_later_full_barrier = _suffix_matches(
            original_op_ids,
            ops,
            _is_full_memory_barrier,
        )
        has_later_ordered_op = _suffix_matches(
            original_op_ids,
            ops,
            lambda op: (
                op.kind in target_ir.MEMORY_ISSUER_OP_KINDS
                or _is_full_memory_barrier(op)
            ),
        )
        ordered_op_ids = []
        preceding_memory_tokens = []
        barrier_epoch = None

        for position, op_id in enumerate(original_op_ids):
            op = ops[op_id]
            if op.kind in target_ir.MEMORY_ISSUER_OP_KINDS:
                if barrier_epoch is not None:
                    op = _append_barrier_dependency(op, barrier_epoch)
                    ops[op_id] = op
                if has_later_full_barrier[position]:
                    op, completion_id = _ensure_memory_completion_result(
                        op,
                        values,
                    )
                    ops[op_id] = op
                    preceding_memory_tokens.append(completion_id)
                ordered_op_ids.append(op_id)
                continue

            if not _is_full_memory_barrier(op):
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
                # Consecutive full barriers need no extra projection: the
                # previous epoch is already completion-free.
                issue_dependency = barrier_epoch

            if issue_dependency is not None:
                op = _append_barrier_dependency(op, issue_dependency)
                ops[op_id] = op
            ordered_op_ids.append(op_id)

            preceding_memory_tokens = []
            barrier_epoch = None
            if not has_later_ordered_op[position]:
                continue

            op, barrier_result_id = _ensure_full_barrier_result(op, values)
            ops[op_id] = op
            epoch_op_id, barrier_epoch = add_issue_token(
                (barrier_result_id, ),
                target_ir.EVENT_DOMAIN_BARRIER_ISSUE,
                _POST_BARRIER_PROVENANCE,
                op.source_op_index,
            )
            ordered_op_ids.append(epoch_op_id)

        regions.append(target_ir.TargetRegion(
            region.target_region_id,
            tuple(ordered_op_ids),
            region.block_arg_ids,
            region.yield_value_ids,
        ))

    return target_ir.TargetProgram(
        tuple(values),
        tuple(ops),
        tuple(regions),
        dict(target_program.source_value_targets),
        dict(target_program.erased_source_values),
        target_program.kernel,
    )


def _suffix_matches(op_ids, ops, predicate):
    result = [False] * len(op_ids)
    seen = False
    for position in range(len(op_ids) - 1, -1, -1):
        result[position] = seen
        if predicate(ops[int(op_ids[position])]):
            seen = True
    return tuple(result)


def _is_full_memory_barrier(op):
    return (
        op.kind == "barrier"
        and int(target_ir.attrs_dict(op).get("address_space", 0))
        == FULL_MEMORY_ADDRESS_SPACE
    )


def _append_barrier_dependency(op, target_value_id):
    attrs = target_ir.attrs_dict(op)
    count = int(attrs.get("barrier_order_dependency_count", 0))
    if count:
        fail(
            "TLXW_BARRIER_ORDER_DUPLICATE_DEPENDENCY",
            STAGE,
            f"target {op.kind} already has a full-barrier order dependency",
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
    values.append(target_ir.TargetValue(
        result_id,
        target_ir.TargetType("token", "token"),
        debug_name=f"memory_completion_{op.target_op_id}",
        event_domain=target_ir.EVENT_DOMAIN_MEMORY_COMPLETION,
    ))
    attrs["issue_order_result_count"] = 1
    if op.kind in {
        "local_load",
        "local_load_mma_payload",
        "local_store",
    }:
        attrs["completion_result_count"] = (
            int(attrs.get("completion_result_count", 0)) + 1
        )
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
        "local_load_mma_payload",
        "local_store",
    }:
        return None
    completion_count = int(
        target_ir.attrs_dict(op).get("completion_result_count", 0)
    )
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


def _ensure_full_barrier_result(op, values):
    if len(op.results) > 1:
        fail(
            "TLXW_BARRIER_ORDER_BARRIER_RESULT",
            STAGE,
            "full-memory barrier may expose at most one token result",
            target_op_id=op.target_op_id,
        )
    if op.results:
        result_id = int(op.results[0])
        value = values[result_id]
        values[result_id] = replace(
            value,
            event_domain=target_ir.EVENT_DOMAIN_FULL_BARRIER,
        )
        return op, result_id

    result_id = len(values)
    values.append(target_ir.TargetValue(
        result_id,
        target_ir.TargetType("token", "token"),
        debug_name=f"full_barrier_{op.target_op_id}",
        event_domain=target_ir.EVENT_DOMAIN_FULL_BARRIER,
    ))
    return replace(op, results=(result_id, )), result_id
