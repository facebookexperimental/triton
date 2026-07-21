"""Stateless source-op to target-program conversion."""

from dataclasses import dataclass, replace
import math
import re
import struct

from .diagnostics import Diagnostic, fail
from . import domains
from . import layouts
from . import coordinates
from . import layout_remap
from . import target_ir

STAGE = "op_conversion"

_BINARY_OPS = {
    "arith.addi": "addi",
    "arith.subi": "subi",
    "arith.muli": "muli",
    "arith.andi": "andi",
    "arith.ori": "ori",
    "arith.xori": "xori",
    "arith.divsi": "divsi",
    "arith.divui": "divui",
    "arith.remsi": "remsi",
    "arith.remui": "remui",
}

_FLOAT_BINARY_OPS = {
    "arith.addf": "addf",
    "arith.divf": "divf",
    "arith.maximumf": "maximumf",
    "arith.maxnumf": "maxnumf",
    "arith.subf": "subf",
    "arith.mulf": "mulf",
}

_FLOAT_UNARY_OPS = {
    "math.exp2": "exp2",
}

_FASTMATH_FLAG_ORDER = (
    "reassoc",
    "nnan",
    "ninf",
    "nsz",
    "arcp",
    "contract",
    "afn",
    "fast",
)

_FLOAT_CAST_OPS = {
    "arith.extf": "fp_convert",
    "arith.truncf": "fp_convert",
}

_MMA_PACKET_REPRESENTATIONS = frozenset({
    "simd_packet",
    "simd_packet_tuple",
})

_LAYOUT_PRESERVING_SIMPLE_OPS = frozenset(
    (*_BINARY_OPS, *_FLOAT_BINARY_OPS, *_FLOAT_UNARY_OPS, *_FLOAT_CAST_OPS, "arith.cmpi", "arith.maxsi",
     "arith.minsi", "arith.select", "tt.addptr"))

_MMA_PACKET_RESULT_SOURCE_OPS = frozenset({
    "arith.constant",
    "arith.addf",
    "arith.divf",
    "arith.extf",
    "arith.maximumf",
    "arith.maxnumf",
    "arith.mulf",
    "arith.select",
    "arith.subf",
    "arith.truncf",
    "amdg.buffer_load",
    "tt.broadcast",
    "tt.reshape",
    "tt.trans",
    "ttg.local_load",
    "ttg.convert_layout",
    "tt.expand_dims",
    "tt.dot",
    "tt.dot_scaled",
    "math.exp2",
    "scf.if",
    "scf.for",
})


@dataclass(frozen=True)
class OpConversionView:
    op_index: int
    op_name: str
    attrs: dict
    operand_target_ids: tuple[int, ...]
    result_target_ids: tuple[int, ...]
    result_layout_map_ids: tuple[int, ...]
    result_source_value_ids: tuple[int, ...]
    layout_address_value_ids: frozenset[int]
    fact_ids: tuple[int, ...]
    fact_target_ids: tuple[int, ...]
    operand_fact_ids: tuple[int, ...]
    operand_fact_target_ids: tuple[int, ...]
    operand_ranges: tuple[tuple[int | None, int | None], ...]


@dataclass(frozen=True)
class MemdescInfo:
    value_id: int
    element_type: str | None
    element_byte_width: int | None
    shape: tuple[int, ...]
    alloc_shape: tuple[int, ...]
    allocation_bytes: int


@dataclass(frozen=True)
class MemdescViewInfo:
    value_id: int
    logical_origin: tuple[int, ...]
    physical_shape: tuple[int, ...]


@dataclass(frozen=True)
class ConversionInput:
    kernel: target_ir.TargetKernel
    kernel_arg_ids: tuple[int, ...]
    top_region_id: int
    ops: tuple
    regions: tuple
    num_warps: int
    threads_per_warp: int
    value_element_byte_widths: dict[int, int | None]
    value_divisibilities: dict[int, int | None]
    memdescs: dict[int, MemdescInfo]
    memdesc_views: dict[int, MemdescViewInfo | None]
    memdesc_physical_allocation_bytes: dict[int, int]
    memdesc_index_slot_stride_bytes: dict[int, int]
    local_alloc_allocation_bytes: dict[int, int]
    constant_ints: dict[int, int]
    fact_ids_by_op: dict[int, tuple[int, ...]]
    token_nodes_by_op: dict[int, object]
    token_groups_by_commit: dict[int, object]
    token_groups_by_id: dict[int, object]
    loop_token_carries_by_op: dict[int, tuple[object, ...]]
    if_token_carries_by_op: dict[int, tuple[object, ...]]
    async_protocol_dependency_value_ids_by_op: dict[int, tuple[int, ...]]
    wait_publication_barrier_by_op: dict[int, int]
    async_issue_dependency_target_ids_by_op: dict[int, tuple[int, ...]]
    static_memdesc_byte_offsets: dict[int, int]
    layout_address_value_ids: frozenset[int]


def convert_ops(source_program, type_layout_program, fact_program, token_program):
    conversion_input = _build_conversion_input(
        source_program,
        type_layout_program,
        fact_program,
        token_program,
    )
    builder = target_ir.TargetBuilder(conversion_input.kernel)
    _seed_kernel_arguments(builder, conversion_input, type_layout_program)

    _convert_region(
        builder,
        conversion_input,
        type_layout_program,
        fact_program,
        conversion_input.top_region_id,
        allow_yield=False,
    )

    return builder.build()


def _build_conversion_input(source_program, type_layout_program, fact_program, token_program):
    memdescs = _memdesc_infos(source_program)
    memdesc_views = _compute_memdesc_view_infos(
        source_program.values,
        source_program.ops,
        source_program.kernel.arg_ids,
        memdescs,
    )
    constant_ints = _constant_ints(source_program)
    memdesc_physical_allocation_bytes = _compute_memdesc_physical_allocation_bytes(
        source_program.values,
        source_program.ops,
        type_layout_program,
        memdescs,
    )
    memdesc_index_slot_stride_bytes = _compute_memdesc_index_slot_stride_bytes(
        source_program.ops,
        type_layout_program,
        memdescs,
        memdesc_physical_allocation_bytes,
    )
    local_alloc_allocation_bytes = _compute_local_alloc_allocation_bytes(
        source_program.ops,
        type_layout_program,
        memdescs,
        memdesc_physical_allocation_bytes,
        memdesc_index_slot_stride_bytes,
    )
    static_memdesc_byte_offsets = _compute_static_memdesc_byte_offsets(
        source_program.ops,
        memdescs,
        memdesc_index_slot_stride_bytes,
        constant_ints,
    )
    async_protocol_dependencies = (
        token_program.async_protocol_dependency_value_ids_by_op
    )
    wait_publication_barriers = _wait_publication_barrier_by_op(
        source_program,
        async_protocol_dependencies,
    )
    kernel = target_ir.TargetKernel(
        source_program.kernel.name,
        source_program.kernel.target,
        source_program.kernel.num_ctas,
        source_program.kernel.num_warps,
        source_program.kernel.threads_per_warp,
        source_program.kernel.noinline,
    )
    return ConversionInput(
        kernel,
        tuple(source_program.kernel.arg_ids),
        int(source_program.top_region_id),
        tuple(source_program.ops),
        tuple(source_program.regions),
        int(source_program.kernel.num_warps or 1),
        int(source_program.kernel.threads_per_warp or 64),
        {value_id: value.type.element_byte_width
         for value_id, value in source_program.values.items()},
        {value_id: value.type.divisibility
         for value_id, value in source_program.values.items()},
        memdescs,
        memdesc_views,
        memdesc_physical_allocation_bytes,
        memdesc_index_slot_stride_bytes,
        local_alloc_allocation_bytes,
        constant_ints,
        _fact_ids_by_source_op(fact_program),
        {node.op_index: node
         for node in token_program.nodes},
        {group.commit_op_index: group
         for group in token_program.groups},
        {group.group_id: group
         for group in token_program.groups},
        token_program.loop_token_carries_by_op,
        token_program.if_token_carries_by_op,
        async_protocol_dependencies,
        wait_publication_barriers,
        {},
        static_memdesc_byte_offsets,
        _layout_address_value_ids(source_program),
    )


def _wait_publication_barrier_by_op(
    source_program,
    async_protocol_dependencies,
):
    """Find a source CTA barrier immediately following an explicit wait.

    AMD membar materializes this pair as ``async_wait; ttg.barrier local``.
    The bridge's workgroup wait-ready operation already lowers to the same
    physical barrier, so preserving both would duplicate the rendezvous.  The
    relation is purely structural: it does not inspect DMA destinations,
    allocation identity, or memory aliases.
    """
    result = {}
    for region in source_program.regions:
        for wait_op_index, barrier_op_index in zip(
            region.op_indices,
            region.op_indices[1:],
        ):
            wait_op = source_program.ops[int(wait_op_index)]
            barrier_op = source_program.ops[int(barrier_op_index)]
            if wait_op.name != "ttg.async_wait":
                continue
            wait_result = None if not wait_op.results else int(wait_op.results[0])
            has_local_consumer = (
                wait_result is not None
                and any(
                    wait_result in tuple(int(value_id) for value_id in value_ids)
                    for value_ids in async_protocol_dependencies.values()
                )
            )
            has_release_dependencies = bool(
                async_protocol_dependencies.get(int(wait_op.index), ())
            )
            if (
                int(source_program.kernel.num_warps or 1) <= 1
                or not (has_local_consumer or has_release_dependencies)
            ):
                continue
            if barrier_op.name == "rocdl.s.barrier":
                result[int(wait_op.index)] = int(barrier_op.index)
                continue
            if (
                barrier_op.name == "ttg.barrier"
                and int(barrier_op.attrs.get("addrSpace", -1)) == 1
            ):
                result[int(wait_op.index)] = int(barrier_op.index)
    return result


def _layout_address_value_ids(source_program):
    deps = {int(value_id): set() for value_id in source_program.values}
    roots = set()

    for op in source_program.ops:
        roots.update(_layout_address_root_value_ids(op))
        if op.name == "scf.for":
            _record_for_address_deps(deps, source_program, op)
            continue
        if op.name == "scf.if":
            _record_if_address_deps(deps, source_program, op)
            continue
        for result_value_id in op.results:
            _add_address_deps(deps, result_value_id, op.operands)

    address_value_ids = set()
    worklist = list(roots)
    while worklist:
        value_id = int(worklist.pop())
        if value_id in address_value_ids:
            continue
        address_value_ids.add(value_id)
        worklist.extend(
            int(dep_value_id) for dep_value_id in deps.get(value_id, ()) if int(dep_value_id) not in address_value_ids)
    return frozenset(address_value_ids)


def _layout_address_root_value_ids(op):
    if op.name == "amdg.buffer_load_to_local":
        fields = _buffer_load_to_local_fields(op)
        roots = [fields["offset_value_id"]]
        if fields["stride_value_id"] is not None:
            roots.append(fields["stride_value_id"])
        return tuple(roots)
    if op.name == "amdg.buffer_load":
        fields = _buffer_load_fields(op)
        roots = [fields["offset_value_id"]]
        if fields["stride_value_id"] is not None:
            roots.append(fields["stride_value_id"])
        return tuple(roots)
    if op.name == "amdg.buffer_store":
        fields = _buffer_store_fields(op)
        roots = [fields["offset_value_id"]]
        if fields["boundary_check_value_id"] is not None:
            roots.append(fields["boundary_check_value_id"])
        return tuple(roots)
    if op.name == "tt.load":
        return (_load_fields(op)["pointer_value_id"], )
    if op.name == "tt.store":
        return (_store_fields(op)["pointer_value_id"], )
    if op.name == "ttg.memdesc_index" and len(op.operands) >= 2:
        return (op.operands[1], )
    return ()


def _record_for_address_deps(deps, source_program, op):
    if not op.region_ids:
        return
    yielded_value_ids = _region_yield_value_ids(source_program, op.region_ids[0])
    region = source_program.regions[op.region_ids[0]]
    if region.block_arg_ids:
        _add_address_deps(deps, region.block_arg_ids[0], op.operands[:3])
        for index, block_arg_value_id in enumerate(region.block_arg_ids[1:]):
            iter_arg_value_ids = []
            init_operand_index = 3 + index
            if init_operand_index < len(op.operands):
                iter_arg_value_ids.append(op.operands[init_operand_index])
            if index < len(yielded_value_ids):
                iter_arg_value_ids.append(yielded_value_ids[index])
            _add_address_deps(deps, block_arg_value_id, iter_arg_value_ids)
    for index, result_value_id in enumerate(op.results):
        if index < len(yielded_value_ids):
            _add_address_deps(deps, result_value_id, (yielded_value_ids[index], ))


def _record_if_address_deps(deps, source_program, op):
    yielded_by_region = tuple(_region_yield_value_ids(source_program, region_id) for region_id in op.region_ids)
    for index, result_value_id in enumerate(op.results):
        _add_address_deps(
            deps,
            result_value_id,
            tuple(yielded_value_ids[index]
                  for yielded_value_ids in yielded_by_region
                  if index < len(yielded_value_ids)),
        )


def _region_yield_value_ids(source_program, region_id):
    region = source_program.regions[region_id]
    for op_index in reversed(region.op_indices):
        op = source_program.ops[op_index]
        if op.name == "scf.yield":
            return tuple(op.operands)
    return ()


def _add_address_deps(deps, value_id, dep_value_ids):
    deps.setdefault(int(value_id), set()).update(int(dep_value_id) for dep_value_id in dep_value_ids)


def _convert_region(
    builder,
    conversion_input,
    type_layout_program,
    fact_program,
    region_id,
    *,
    allow_yield,
):
    for op_index in conversion_input.regions[region_id].op_indices:
        op = conversion_input.ops[op_index]
        if op.name == "scf.yield":
            if not allow_yield:
                fail(
                    "TLXW_OP_UNEXPECTED_YIELD",
                    STAGE,
                    "scf.yield is only valid inside a converted region",
                    source_op_index=op.index,
                )
            return op.operands
        _convert_source_op(
            builder,
            conversion_input,
            type_layout_program,
            fact_program,
            op,
        )
    # Result-free SCF regions can use an implicit terminator.  The importer sees
    # a genuinely empty region for the absent else arm of a one-sided scf.if.
    # Callers validate the returned count, so a missing value yield still gets
    # rejected as a for/if yield mismatch.
    return ()


def _convert_source_op(
    builder,
    conversion_input,
    type_layout_program,
    fact_program,
    op,
):
    _require_allowed_mma_packet_results(type_layout_program, op)
    if op.name == "scf.for":
        _convert_for(
            builder,
            conversion_input,
            type_layout_program,
            fact_program,
            op,
        )
        return
    if op.name == "scf.if":
        _convert_if(
            builder,
            conversion_input,
            type_layout_program,
            fact_program,
            op,
        )
        return
    if op.name == "tt.reduce":
        _convert_reduce(
            builder,
            conversion_input,
            type_layout_program,
            op,
        )
        return
    if op.name == "arith.constant" and _has_mma_packet_result(type_layout_program, op):
        _convert_mma_packet_constant(builder, type_layout_program, op)
        return
    if op.name == "arith.extf" and _has_mma_packet_result(type_layout_program, op):
        _convert_mma_packet_float_cast(builder, type_layout_program, op)
        return
    if op.name == "arith.extf" and _has_register_vector_payload_result(builder, type_layout_program, op):
        _convert_register_vector_payload_float_cast(builder, type_layout_program, op)
        return
    if op.name == "arith.truncf" and _has_mma_packet_result(type_layout_program, op):
        _convert_mma_packet_truncf(builder, type_layout_program, op)
        return
    if op.name == "arith.truncf" and _has_register_vector_payload_result(builder, type_layout_program, op):
        _convert_register_vector_payload_float_cast(builder, type_layout_program, op)
        return
    if op.name == "ttg.local_alloc":
        _convert_local_alloc(
            builder,
            conversion_input,
            type_layout_program,
            op,
        )
        return
    if op.name == "ttg.memdesc_index":
        _convert_memdesc_index(
            builder,
            conversion_input,
            type_layout_program,
            op,
        )
        return
    if op.name == "ttg.memdesc_reshape":
        _convert_memdesc_reshape(builder, type_layout_program, op)
        return
    if op.name == "ttg.memdesc_reinterpret":
        _convert_memdesc_reinterpret(builder, type_layout_program, op)
        return
    if op.name == "ttg.memdesc_subslice":
        _convert_memdesc_subslice(
            builder,
            conversion_input,
            type_layout_program,
            op,
        )
        return
    if op.name == "ttg.memdesc_trans":
        _convert_memdesc_trans(builder, type_layout_program, op)
        return
    if op.name == "ttg.local_load":
        _convert_local_load(builder, conversion_input, type_layout_program, op)
        return
    if op.name == "ttg.local_store":
        _convert_local_store(builder, conversion_input, type_layout_program, op)
        return
    if op.name == "ttg.convert_layout":
        _convert_layout(builder, conversion_input, type_layout_program, op)
        return
    if op.name in {"tt.reshape", "tt.trans"}:
        _convert_structural_tensor_view(builder, type_layout_program, op)
        return
    if op.name == "tt.dot":
        _convert_dot(builder, conversion_input, type_layout_program, op)
        return
    if op.name == "tt.dot_scaled":
        _convert_dot_scaled(builder, conversion_input, type_layout_program, op)
        return
    if op.name == "amdg.buffer_load_to_local":
        _convert_buffer_load_to_local(
            builder,
            conversion_input,
            type_layout_program,
            fact_program,
            op,
        )
        return
    if op.name == "amdg.buffer_load":
        _convert_buffer_load(
            builder,
            conversion_input,
            type_layout_program,
            fact_program,
            op,
        )
        return
    if op.name == "amdg.buffer_store":
        _convert_buffer_store(
            builder,
            conversion_input,
            type_layout_program,
            fact_program,
            op,
        )
        return
    if op.name == "tt.load":
        _convert_load(builder, conversion_input, type_layout_program, op)
        return
    if op.name == "tt.store":
        _convert_store(builder, conversion_input, type_layout_program, op)
        return
    if op.name == "ttg.async_commit_group":
        _convert_async_commit_group(
            builder,
            type_layout_program,
            conversion_input.token_groups_by_commit,
            op,
        )
        return
    if op.name == "ttg.async_wait":
        _convert_async_wait(
            builder,
            conversion_input,
            type_layout_program,
            op,
        )
        return
    if op.name == "rocdl.sched.barrier":
        _convert_sched_barrier(builder, op)
        return
    if op.name in {"ttg.barrier", "rocdl.s.barrier"}:
        _convert_barrier(builder, conversion_input, op)
        return
    if op.name == "amdg.cond_barrier":
        _convert_cond_barrier(builder, op)
        return
    if op.name == "rocdl.s.setprio":
        _convert_set_priority(builder, op)
        return
    if op.name == "tt.make_range":
        _convert_make_range(builder, type_layout_program, op)
        return
    if op.name == "tt.expand_dims":
        _convert_expand_dims(builder, type_layout_program, op)
        return
    if op.name == "tt.broadcast":
        _convert_broadcast(builder, type_layout_program, op)
        return
    if op.name == "arith.cmpi":
        _convert_cmpi(
            builder,
            conversion_input,
            type_layout_program,
            fact_program,
            op,
        )
        return
    if op.name == "arith.select":
        _convert_select(
            builder,
            conversion_input,
            type_layout_program,
            fact_program,
            op,
        )
        return
    converter = _converter_for_op(op.name)
    if converter is None:
        fail(
            "TLXW_OP_UNSUPPORTED",
            STAGE,
            f"no op conversion for {op.name}",
            source_op_index=op.index,
        )
    _require_simple_op_layout_contract(type_layout_program, op)
    operand_target_ids = _operand_target_ids(builder, op)
    result_target_ids, result_layout_map_ids = _declare_results(
        builder,
        op,
        type_layout_program,
    )
    fact_ids = conversion_input.fact_ids_by_op.get(op.index, ())
    operand_fact_ids = _operand_assume_fact_ids(conversion_input, fact_program, op)
    view = OpConversionView(
        op.index,
        op.name,
        dict(op.attrs),
        operand_target_ids,
        result_target_ids,
        result_layout_map_ids,
        tuple(op.results),
        conversion_input.layout_address_value_ids,
        fact_ids,
        _fact_target_ids(builder, fact_program, fact_ids, op),
        operand_fact_ids,
        _fact_target_ids(builder, fact_program, operand_fact_ids, op),
        _operand_ranges(conversion_input, fact_program, op),
    )
    converter(builder, view)


def _seed_kernel_arguments(builder, conversion_input, type_layout_program):
    arg_target_ids = []
    for source_value_id in conversion_input.kernel_arg_ids:
        converted = type_layout_program.values[source_value_id]
        if converted.type.representation in _MMA_PACKET_REPRESENTATIONS:
            fail(
                "TLXW_OP_MMA_PACKET_PRODUCER",
                STAGE,
                "MMA packet layouts cannot be kernel arguments; packet values "
                "must be materialized inside the kernel",
                source_value_id=source_value_id,
            )
        arg_target_ids.append(
            builder.add_value(
                target_ir.target_type_from_converted(converted.type),
                source_value_id=source_value_id,
                debug_name=f"arg{source_value_id}",
            ))
    builder.set_kernel_arg_targets(tuple(arg_target_ids))


def _declare_results(builder, op, type_layout_program):
    result_target_ids = []
    result_layout_map_ids = []
    for source_value_id in op.results:
        converted = type_layout_program.values[source_value_id]
        result_target_ids.append(
            builder.add_value(
                target_ir.target_type_from_converted(converted.type),
                source_value_id=source_value_id,
                debug_name=f"v{source_value_id}",
            ))
        if converted.layout_map_id is not None:
            result_layout_map_ids.append(converted.layout_map_id)
    return tuple(result_target_ids), tuple(result_layout_map_ids)


def _protocol_token_type():
    return target_ir.TargetType("token", "token")


def _declare_protocol_token(
    builder,
    *,
    event_domain,
    debug_name,
    source_value_id=None,
):
    target_id = builder.add_value(
        _protocol_token_type(),
        source_value_id=source_value_id,
        debug_name=debug_name,
        event_domain=event_domain,
    )
    return target_id


def _join_protocol_tokens(
    builder,
    target_ids,
    op,
    *,
    debug_name,
    event_domain=target_ir.EVENT_DOMAIN_LDS_FRONTIER,
):
    target_ids = tuple(dict.fromkeys(int(target_id) for target_id in target_ids))
    result_id = _declare_protocol_token(
        builder,
        event_domain=(
            target_ir.EVENT_DOMAIN_EMPTY
            if not target_ids else event_domain
        ),
        debug_name=debug_name,
    )
    if target_ids:
        builder.add_op(
            "token_join",
            operands=target_ids,
            results=(result_id, ),
            attrs={
                "event_domain": str(event_domain),
                "input_count": len(target_ids),
            },
            source_op_index=op.index,
        )
    else:
        builder.add_op(
            "token",
            results=(result_id, ),
            attrs={"event_domain": target_ir.EVENT_DOMAIN_EMPTY},
            source_op_index=op.index,
        )
    return result_id


def _operand_target_ids(builder, op):
    operand_target_ids = []
    for source_value_id in op.operands:
        targets = builder.source_value_targets.get(source_value_id)
        if not targets:
            fail(
                "TLXW_OP_UNCONVERTED_OPERAND",
                STAGE,
                f"operand {source_value_id} has no converted target value",
                source_op_index=op.index,
                source_value_id=source_value_id,
            )
        if len(targets) != 1:
            fail(
                "TLXW_OP_MULTI_VALUE_OPERAND",
                STAGE,
                f"operand {source_value_id} maps to multiple target values {targets}",
                source_op_index=op.index,
                source_value_id=source_value_id,
            )
        operand_target_ids.append(targets[0])
    return tuple(operand_target_ids)


def _converter_for_op(op_name):
    return _SIMPLE_OP_CONVERTERS.get(op_name)


def _convert_constant(builder, view):
    (result_target_id, ) = view.result_target_ids
    builder.add_op(
        "constant",
        results=view.result_target_ids,
        attrs={"value": _constant_literal(
            view.attrs.get("value"),
            source_op_index=view.op_index,
            element_type=builder.values[result_target_id].type.element_type,
        )},
        source_op_index=view.op_index,
    )


def _convert_binary(builder, view):
    operation = _BINARY_OPS[view.op_name]
    if operation in {"divsi", "remsi"} and _can_use_unsigned_div_rem(view):
        operation = "divui" if operation == "divsi" else "remui"
    operand_target_ids = view.operand_target_ids
    if operation in {"divsi", "divui", "remsi", "remui"}:
        operand_target_ids = (
            operand_target_ids[0],
            _materialize_proven_exact_positive_scalar(
                builder,
                operand_target_ids[1],
                view.operand_ranges[1],
                view,
            ),
        )
    source_width = _target_int_width(builder, view.result_target_ids)
    attrs = {
        "operation": operation,
        "source_width": source_width,
    }
    nsw, nuw = _arith_overflow_flags(view)
    if not nsw and _layout_address_binary_no_signed_wrap(view, operation, source_width):
        nsw = True
    if not nsw and _range_proves_no_signed_wrap(view, operation, source_width):
        nsw = True
    if not nuw and _range_proves_no_unsigned_wrap(view, operation, source_width):
        nuw = True
    if nsw:
        attrs["nsw"] = True
    if nuw:
        attrs["nuw"] = True
    builder.add_op(
        "binary",
        operands=operand_target_ids,
        results=view.result_target_ids,
        attrs=attrs,
        fact_ids=view.operand_fact_ids,
        fact_target_ids=view.operand_fact_target_ids,
        layout_map_ids=view.result_layout_map_ids,
        source_op_index=view.op_index,
    )


def _materialize_proven_exact_positive_scalar(
    builder,
    target_value_id,
    value_range,
    view,
):
    """Make a dominating exact-range proof structural at a scalar use edge.

    Wave symbolic div/rem requires a positive static divisor. Source range
    analysis can prove that a dynamically spelled scalar is exact after a
    dominating ``llvm.intr.assume`` (for example, a boundary tile extent).
    Materialize that proven value at the consuming edge instead of asking Wave
    to rediscover source dominance and range propagation.
    """
    lower, upper = value_range
    if lower is None or upper is None or int(lower) != int(upper) or int(lower) <= 0:
        return int(target_value_id)
    target_type = builder.values[int(target_value_id)].type
    if target_type.kind != "scalar" or target_type.representation != "scalar":
        return int(target_value_id)
    if target_type.element_type not in {"index", "i8", "i16", "i32", "i64"}:
        return int(target_value_id)
    exact_target_id = builder.add_value(
        target_type,
        debug_name=f"exact_operand_{view.op_index}_{target_value_id}",
    )
    builder.add_op(
        "constant",
        results=(exact_target_id, ),
        attrs={"value": int(lower)},
        source_op_index=view.op_index,
    )
    return exact_target_id


def _layout_address_binary_no_signed_wrap(view, operation, source_width):
    if operation not in {"addi", "subi", "muli"}:
        return False
    if source_width is None or int(source_width) <= 0:
        return False
    if view.result_layout_map_ids:
        # Integer tensors with layout maps are layout-address values in the TLX
        # Wave target contract. Their add/sub/mul overflow is UB, so the bridge
        # records the no-wrap provenance where the layout association is still
        # explicit.
        return True
    if any(int(value_id) in view.layout_address_value_ids for value_id in view.result_source_value_ids):
        # Scalar layout bases often lose the explicit layout map before they are
        # splatted or attached as affine bindings.  Keep the same address
        # no-overflow provenance for arithmetic in the transitive backward slice
        # from load/store/DMA offsets and memdesc indices.
        return True
    return False


def _can_use_unsigned_div_rem(view):
    if len(view.operand_ranges) != 2:
        return False
    lhs_range, rhs_range = view.operand_ranges
    lhs_lower = lhs_range[0]
    rhs_lower = rhs_range[0]
    return (lhs_lower is not None and lhs_lower >= 0 and rhs_lower is not None and rhs_lower > 0)


def _range_proves_no_signed_wrap(view, operation, source_width):
    if operation not in {"addi", "subi", "muli"}:
        return False
    if source_width is None or int(source_width) <= 0:
        return False
    if len(view.operand_ranges) != 2:
        return False
    lhs_range, rhs_range = view.operand_ranges
    if any(bound is None for bound in (*lhs_range, *rhs_range)):
        return False
    lhs_lower, lhs_upper = (int(lhs_range[0]), int(lhs_range[1]))
    rhs_lower, rhs_upper = (int(rhs_range[0]), int(rhs_range[1]))
    if operation == "addi":
        lower = lhs_lower + rhs_lower
        upper = lhs_upper + rhs_upper
    elif operation == "subi":
        lower = lhs_lower - rhs_upper
        upper = lhs_upper - rhs_lower
    else:
        products = (
            lhs_lower * rhs_lower,
            lhs_lower * rhs_upper,
            lhs_upper * rhs_lower,
            lhs_upper * rhs_upper,
        )
        lower = min(products)
        upper = max(products)
    signed_min = -(1 << (int(source_width) - 1))
    signed_max = (1 << (int(source_width) - 1)) - 1
    return signed_min <= lower and upper <= signed_max


def _range_proves_no_unsigned_wrap(view, operation, source_width):
    if operation not in {"addi", "subi", "muli"}:
        return False
    if source_width is None or int(source_width) <= 0:
        return False
    if len(view.operand_ranges) != 2:
        return False
    lhs_range, rhs_range = view.operand_ranges
    if any(bound is None for bound in (*lhs_range, *rhs_range)):
        return False
    lhs_lower, lhs_upper = (int(lhs_range[0]), int(lhs_range[1]))
    rhs_lower, rhs_upper = (int(rhs_range[0]), int(rhs_range[1]))
    if operation == "addi":
        lower = lhs_lower + rhs_lower
        upper = lhs_upper + rhs_upper
    elif operation == "subi":
        lower = lhs_lower - rhs_upper
        upper = lhs_upper - rhs_lower
    else:
        products = (
            lhs_lower * rhs_lower,
            lhs_lower * rhs_upper,
            lhs_upper * rhs_lower,
            lhs_upper * rhs_upper,
        )
        lower = min(products)
        upper = max(products)
    return 0 <= lower and upper < (1 << int(source_width))


def _convert_float_binary(builder, view):
    result_type = builder.values[view.result_target_ids[0]].type
    attrs = _float_binary_attrs(view)
    if result_type.representation in _MMA_PACKET_REPRESENTATIONS:
        _require_mma_packet_float_binary_type(builder, view, result_type)
        builder.add_op(
            "float_binary",
            operands=view.operand_target_ids,
            results=view.result_target_ids,
            attrs=attrs,
            layout_map_ids=view.result_layout_map_ids,
            source_op_index=view.op_index,
        )
        return
    for target_value_id in (*view.operand_target_ids, *view.result_target_ids):
        target_type = builder.values[target_value_id].type
        if not _supports_float_binary_type(view.op_name, target_type, result_type):
            fail(
                "TLXW_OP_UNSUPPORTED_FLOAT_BINARY",
                STAGE,
                f"{view.op_name} requires supported Wave SIMD float operands",
                source_op_index=view.op_index,
                target_value_id=target_value_id,
            )
    builder.add_op(
        "float_binary",
        operands=view.operand_target_ids,
        results=view.result_target_ids,
        attrs=attrs,
        layout_map_ids=view.result_layout_map_ids,
        source_op_index=view.op_index,
    )


def _convert_float_unary(builder, view):
    if len(view.operand_target_ids) != 1 or len(view.result_target_ids) != 1:
        fail(
            "TLXW_OP_UNSUPPORTED_FLOAT_UNARY",
            STAGE,
            f"{view.op_name} requires one operand and one result",
            source_op_index=view.op_index,
        )
    operand_type = builder.values[view.operand_target_ids[0]].type
    result_type = builder.values[view.result_target_ids[0]].type
    supported_representations = {
        "simd",
        "simd_tuple",
        *_MMA_PACKET_REPRESENTATIONS,
    }
    if (operand_type.representation not in supported_representations
            or result_type.representation not in supported_representations
            or operand_type.representation != result_type.representation
            or operand_type.element_type != "f32" or result_type.element_type != "f32"
            or int(operand_type.component_count) != int(result_type.component_count)):
        fail(
            "TLXW_OP_UNSUPPORTED_FLOAT_UNARY",
            STAGE,
            f"{view.op_name} requires matching f32 SIMD or MMA packet payloads",
            source_op_index=view.op_index,
        )
    attrs = {"operation": _FLOAT_UNARY_OPS[view.op_name]}
    if "fastmath" in view.attrs:
        fastmath = _normalize_fastmath_flags(
            view.attrs["fastmath"],
            source_op_index=view.op_index,
        )
        if fastmath:
            attrs["fastmath"] = fastmath
    builder.add_op(
        "float_unary",
        operands=view.operand_target_ids,
        results=view.result_target_ids,
        attrs=attrs,
        layout_map_ids=view.result_layout_map_ids,
        source_op_index=view.op_index,
    )


def _float_binary_attrs(view):
    attrs = {"operation": _FLOAT_BINARY_OPS[view.op_name]}
    if "fastmath" in view.attrs:
        flags = _normalize_fastmath_flags(
            view.attrs["fastmath"],
            source_op_index=view.op_index,
        )
        if not flags:
            flags = _default_float_binary_fastmath_flags(view.op_name)
    else:
        flags = _default_float_binary_fastmath_flags(view.op_name)
    if flags:
        attrs["fastmath"] = flags
    return attrs


def _default_float_binary_fastmath_flags(op_name):
    # AMD LLVM lowers plain fadd/fmul and relies on AMDGPU instruction
    # selection to contract them where legal for Triton. Wave's local fma
    # combine is gated by arith fastmath, so the bridge records that same
    # contraction permission explicitly on Wave fadd/fmul ops.
    if op_name in {"arith.addf", "arith.mulf"}:
        return ("contract",)
    return ()


def _normalize_fastmath_flags(value, *, source_op_index):
    text = str(value)
    match = re.search(r"fastmath<([^>]*)>", text)
    if match is None:
        fail(
            "TLXW_OP_UNSUPPORTED_FASTMATH",
            STAGE,
            f"unsupported arith fastmath attribute {text}",
            source_op_index=source_op_index,
        )
    flag_text = match.group(1).strip()
    if flag_text in {"", "none"}:
        return ()
    flags = tuple(part.strip() for part in flag_text.split(",") if part.strip())
    unknown = sorted(set(flags) - set(_FASTMATH_FLAG_ORDER))
    if unknown:
        fail(
            "TLXW_OP_UNSUPPORTED_FASTMATH",
            STAGE,
            f"unsupported arith fastmath flags {', '.join(unknown)}",
            source_op_index=source_op_index,
        )
    if "fast" in flags:
        return ("fast",)
    return tuple(flag for flag in _FASTMATH_FLAG_ORDER if flag in flags)


def _supports_float_binary_type(op_name, target_type, result_type):
    if target_type.representation not in {"simd", "simd_tuple"}:
        return False
    if target_type.element_type != result_type.element_type:
        return False
    if op_name in {"arith.divf", "arith.maximumf", "arith.maxnumf", "arith.subf"}:
        return target_type.element_type == "f32"
    return target_type.element_type in {"f16", "f32"}


def _require_mma_packet_float_binary_type(builder, view, result_type):
    if result_type.element_type != "f32":
        fail(
            "TLXW_OP_UNSUPPORTED_FLOAT_BINARY",
            STAGE,
            "fragment float binary ops are only supported for f32 MFMA payloads",
            source_op_index=view.op_index,
        )
    for target_value_id in (*view.operand_target_ids, *view.result_target_ids):
        target_type = builder.values[target_value_id].type
        if target_type.representation not in _MMA_PACKET_REPRESENTATIONS:
            fail(
                "TLXW_OP_UNSUPPORTED_FLOAT_BINARY",
                STAGE,
                f"{view.op_name} MMA packet arithmetic requires packet operands and result",
                source_op_index=view.op_index,
                target_value_id=target_value_id,
            )
        if target_type.element_type != result_type.element_type:
            fail(
                "TLXW_OP_UNSUPPORTED_FLOAT_BINARY",
                STAGE,
                f"{view.op_name} MMA packet arithmetic requires matching element types",
                source_op_index=view.op_index,
                target_value_id=target_value_id,
            )


def _convert_float_cast(builder, view):
    result_type = builder.values[view.result_target_ids[0]].type
    operand_type = builder.values[view.operand_target_ids[0]].type
    for target_value_id in (*view.operand_target_ids, *view.result_target_ids):
        target_type = builder.values[target_value_id].type
        if target_type.representation not in {"simd", "simd_tuple"}:
            fail(
                "TLXW_OP_UNSUPPORTED_FLOAT_CAST",
                STAGE,
                f"{view.op_name} requires Wave SIMD float values",
                source_op_index=view.op_index,
                target_value_id=target_value_id,
            )
    if view.op_name == "arith.extf":
        supported = operand_type.element_type in {"f16", "bf16"} and result_type.element_type == "f32"
        description = "f16/bf16 to f32"
    else:
        supported = operand_type.element_type == "f32" and result_type.element_type in {"f16", "bf16"}
        description = "f32 to f16/bf16"
    if not supported:
        fail(
            "TLXW_OP_UNSUPPORTED_FLOAT_CAST",
            STAGE,
            f"only {description} {view.op_name} is converted yet",
            source_op_index=view.op_index,
        )
    builder.add_op(
        "float_cast",
        operands=view.operand_target_ids,
        results=view.result_target_ids,
        attrs={"operation": _FLOAT_CAST_OPS[view.op_name]},
        layout_map_ids=view.result_layout_map_ids,
        source_op_index=view.op_index,
    )


def _convert_register_vector_payload_float_cast(builder, type_layout_program, op):
    if len(op.operands) != 1 or len(op.results) != 1:
        fail(
            "TLXW_OP_UNSUPPORTED_FLOAT_CAST",
            STAGE,
            f"register-payload {op.name} requires one operand and one result",
            source_op_index=op.index,
        )
    operand = type_layout_program.values[op.operands[0]]
    result = type_layout_program.values[op.results[0]]
    for value, label in ((operand, "operand"), (result, "result")):
        if value.type.representation not in {"simd", "simd_tuple"}:
            fail(
                "TLXW_OP_UNSUPPORTED_FLOAT_CAST",
                STAGE,
                f"register-payload {op.name} requires a SIMD {label}",
                source_op_index=op.index,
                source_value_id=value.value_id,
            )
    if op.name == "arith.extf":
        supported = operand.type.element_type in {"f16", "bf16"} and result.type.element_type == "f32"
        description = "f16/bf16 to f32"
    else:
        supported = operand.type.element_type == "f32" and result.type.element_type in {"f16", "bf16"}
        description = "f32 to f16/bf16"
    if not supported:
        fail(
            "TLXW_OP_UNSUPPORTED_FLOAT_CAST",
            STAGE,
            f"only {description} register-payload {op.name} is converted yet",
            source_op_index=op.index,
        )
    if int(operand.type.component_count) != int(result.type.component_count):
        fail(
            "TLXW_OP_UNSUPPORTED_FLOAT_CAST",
            STAGE,
            f"register-payload {op.name} component counts must match",
            source_op_index=op.index,
        )
    _require_same_layout_except_element_type(
        type_layout_program,
        operand,
        result,
        f"register-payload {op.name} operand and result",
        op,
    )
    result_target_ids, result_layout_map_ids = _declare_results(
        builder,
        op,
        type_layout_program,
    )
    builder.add_op(
        "float_cast",
        operands=_operand_target_ids(builder, op),
        results=result_target_ids,
        attrs={
            "operation": _FLOAT_CAST_OPS[op.name],
            "registers": _register_vector_payload_registers(type_layout_program, result, op),
            "result_value_mode": "register_vector_payload",
        },
        layout_map_ids=result_layout_map_ids,
        source_op_index=op.index,
    )


def _convert_mma_packet_float_cast(builder, type_layout_program, op):
    if len(op.operands) != 1 or len(op.results) != 1:
        fail(
            "TLXW_OP_UNSUPPORTED_FLOAT_CAST",
            STAGE,
            "MMA packet arith.extf requires one operand and one result",
            source_op_index=op.index,
        )
    operand = type_layout_program.values[op.operands[0]]
    result = type_layout_program.values[op.results[0]]
    if operand.type.representation not in _MMA_PACKET_REPRESENTATIONS:
        fail(
            "TLXW_OP_UNSUPPORTED_FLOAT_CAST",
            STAGE,
            "MMA packet arith.extf requires a packet operand",
            source_op_index=op.index,
            source_value_id=operand.value_id,
        )
    if operand.type.element_type not in {"f16", "bf16"} or result.type.element_type != "f32":
        fail(
            "TLXW_OP_UNSUPPORTED_FLOAT_CAST",
            STAGE,
            "only f16/bf16 to f32 MMA packet arith.extf is converted yet",
            source_op_index=op.index,
        )
    if int(operand.type.component_count) != int(result.type.component_count):
        fail(
            "TLXW_OP_UNSUPPORTED_FLOAT_CAST",
            STAGE,
            "MMA packet arith.extf component counts must match",
            source_op_index=op.index,
        )
    _require_same_layout_except_element_type(
        type_layout_program,
        operand,
        result,
        "MMA packet arith.extf operand and result",
        op,
    )
    result_target_ids, result_layout_map_ids = _declare_results(
        builder,
        op,
        type_layout_program,
    )
    builder.add_op(
        "float_cast",
        operands=_operand_target_ids(builder, op),
        results=result_target_ids,
        attrs={
            "operation": _FLOAT_CAST_OPS[op.name],
            "result_value_mode": "mma_packet_payload",
            "registers": _mma_packet_registers(type_layout_program, result, op),
        },
        layout_map_ids=result_layout_map_ids,
        source_op_index=op.index,
    )


def _convert_cmpi(
    builder,
    conversion_input,
    type_layout_program,
    fact_program,
    op,
):
    _require_simple_op_layout_contract(type_layout_program, op)
    if len(op.operands) != 2 or len(op.results) != 1:
        fail(
            "TLXW_OP_CMPI",
            STAGE,
            "arith.cmpi requires two operands and one result",
            source_op_index=op.index,
        )
    operand_target_ids = tuple(
        _materialize_affine_edge_or_original(
            builder,
            conversion_input,
            type_layout_program,
            fact_program,
            int(source_value_id),
            op,
            no_signed_wrap=False,
        )
        for source_value_id in op.operands
    )
    result_target_ids, result_layout_map_ids = _declare_results(
        builder,
        op,
        type_layout_program,
    )
    builder.add_op(
        "cmpi",
        operands=operand_target_ids,
        results=result_target_ids,
        attrs={
            "predicate": _cmpi_predicate(op.attrs.get("predicate")),
            "source_width": _target_int_width(builder, operand_target_ids),
        },
        layout_map_ids=result_layout_map_ids,
        source_op_index=op.index,
    )


def _convert_minsi(builder, view):
    builder.add_op(
        "minsi",
        operands=view.operand_target_ids,
        results=view.result_target_ids,
        attrs={"source_width": _target_int_width(builder, view.result_target_ids)},
        layout_map_ids=view.result_layout_map_ids,
        source_op_index=view.op_index,
    )


def _convert_maxsi(builder, view):
    builder.add_op(
        "maxsi",
        operands=view.operand_target_ids,
        results=view.result_target_ids,
        attrs={"source_width": _target_int_width(builder, view.result_target_ids)},
        layout_map_ids=view.result_layout_map_ids,
        source_op_index=view.op_index,
    )


def _convert_assume(builder, view):
    if not view.fact_ids:
        return
    builder.add_op(
        "assume",
        operands=view.operand_target_ids,
        fact_ids=view.fact_ids,
        fact_target_ids=view.fact_target_ids,
        source_op_index=view.op_index,
    )


def _arith_overflow_flags(view):
    if view.op_name not in {"arith.addi", "arith.muli", "arith.subi"}:
        return False, False
    attr = view.attrs.get("overflowFlags")
    if attr is None:
        return False, False
    text = str(attr)
    return "nsw" in text, "nuw" in text


def _convert_make_range(builder, type_layout_program, op):
    result_target_ids, result_layout_map_ids = _declare_results(
        builder,
        op,
        type_layout_program,
    )
    attrs = {
        "start": _int_attr(op.attrs, "start"),
        "end": _int_attr(op.attrs, "end"),
    }
    attrs.update(_make_range_coordinate_attrs(type_layout_program, op))
    builder.add_op(
        "make_range",
        results=result_target_ids,
        attrs=attrs,
        layout_map_ids=result_layout_map_ids,
        source_op_index=op.index,
    )


def _make_range_coordinate_attrs(type_layout_program, op):
    if len(op.results) != 1:
        fail(
            "TLXW_OP_MAKE_RANGE",
            STAGE,
            "tt.make_range requires one result",
            source_op_index=op.index,
        )
    result = type_layout_program.values[op.results[0]]
    if result.layout_map_id is None:
        return {}
    layout = type_layout_program.layouts[int(result.layout_map_id)]
    if layout.kind not in {
            "blocked",
            "linear",
            "generic_linear",
            "slice",
            "amd_mfma",
    }:
        return {}
    lane_width = int(result.type.lane_width or layout.lane_width)
    warp_count = _layout_warp_count(layout)
    plan = coordinates.layout_coordinate_plan(
        layout,
        int(result.type.component_count),
        lane_width,
        warp_count,
        op,
        result.value_id,
    )
    if coordinates.is_default_flat_make_range(plan, lane_width):
        return {}
    affine = coordinates.is_flat_affine_make_range(plan, lane_width, warp_count)
    if affine is not None:
        bases, stride = affine
        return {
            "coordinate_mode": "affine_workitem",
            "component_bases": tuple(int(base) for base in bases),
            "workitem_stride": int(stride),
        }
    bit_affine = coordinates.is_flat_bit_affine_make_range(plan)
    if bit_affine is not None:
        bases, coefficients = bit_affine
        return {
            "coordinate_mode": "bit_affine_workitem",
            "component_bases": tuple(int(base) for base in bases),
            "workitem_coefficients": tuple(int(coefficient) for coefficient in coefficients),
        }
    return {
        "coordinate_mode":
        "layout_coordinates",
        "coordinate_shape":
        tuple(int(dim) for dim in plan.shape),
        "component_coordinate_bases":
        tuple(tuple(int(value) for value in bases) for bases in plan.component_bases),
        "workitem_coordinate_coefficients":
        tuple(tuple(int(value) for value in coefficients) for coefficients in plan.workitem_coefficients),
    }


def _layout_warp_count(layout):
    return layouts.layout_warp_count(layout)


def _convert_splat(builder, view):
    builder.add_op(
        "splat",
        operands=view.operand_target_ids,
        results=view.result_target_ids,
        layout_map_ids=view.result_layout_map_ids,
        source_op_index=view.op_index,
    )


def _convert_addptr(builder, view):
    builder.add_op(
        "addptr",
        operands=view.operand_target_ids,
        results=view.result_target_ids,
        layout_map_ids=view.result_layout_map_ids,
        source_op_index=view.op_index,
    )


def _convert_expand_dims(builder, type_layout_program, op):
    if len(op.operands) != 1 or len(op.results) != 1:
        fail(
            "TLXW_OP_UNSUPPORTED_REMAP",
            STAGE,
            "tt.expand_dims requires one operand and one result",
            source_op_index=op.index,
        )
    result_target_ids, result_layout_map_ids = _declare_results(
        builder,
        op,
        type_layout_program,
    )
    operand = type_layout_program.values[op.operands[0]]
    result = type_layout_program.values[op.results[0]]
    axis = _int_attr(op.attrs, "axis")
    attrs = {"axis": axis}
    operand_type = operand.type
    result_type = result.type
    if result_type.representation in _MMA_PACKET_REPRESENTATIONS:
        if operand_type.representation not in {"simd", "simd_tuple"}:
            fail(
                "TLXW_OP_UNSUPPORTED_REMAP",
                STAGE,
                "tt.expand_dims to an MMA packet requires SIMD input",
                source_op_index=op.index,
                source_value_id=operand.value_id,
            )
        if operand_type.element_type != result_type.element_type:
            fail(
                "TLXW_OP_UNSUPPORTED_REMAP",
                STAGE,
                "tt.expand_dims to an MMA packet requires matching element types",
                source_op_index=op.index,
                source_value_id=result.value_id,
            )
        registers = _mma_packet_registers(type_layout_program, result, op)
        source_indices = _expand_dims_mma_packet_source_indices(
            type_layout_program,
            operand,
            result,
            axis,
            registers,
            op,
        )
        attrs.update({
            "packet_source_indices": source_indices,
            "registers": registers,
            "result_value_mode": "mma_packet_remap",
            "source_component_count": int(operand_type.component_count),
        })
    builder.add_op(
        "expand_dims",
        operands=_operand_target_ids(builder, op),
        results=result_target_ids,
        attrs=attrs,
        layout_map_ids=result_layout_map_ids,
        source_op_index=op.index,
    )


def _expand_dims_mma_packet_source_indices(
    type_layout_program,
    operand,
    result,
    axis,
    registers,
    op,
):
    operand_layout = _require_layout(type_layout_program, operand.layout_map_id, op)
    result_layout = _require_layout(type_layout_program, result.layout_map_id, op)
    if result_layout.kind != "amd_mfma":
        fail(
            "TLXW_OP_UNSUPPORTED_REMAP",
            STAGE,
            "tt.expand_dims MMA packet result requires an amd_mfma layout",
            source_op_index=op.index,
            source_value_id=result.value_id,
        )
    source_linear = layouts.distributed_linear_layout(
        operand_layout,
        stage=STAGE,
        source_op_index=op.index,
    )
    result_linear = layouts.distributed_linear_layout(
        result_layout,
        stage=STAGE,
        source_op_index=op.index,
    )
    source_register_count = layouts.linear_layout_in_dim_size(source_linear, "register")
    result_register_count = layouts.linear_layout_in_dim_size(result_linear, "register")
    if source_register_count != int(operand.type.component_count):
        fail(
            "TLXW_OP_UNSUPPORTED_REMAP",
            STAGE,
            "tt.expand_dims source payload does not match its distributed register layout",
            source_op_index=op.index,
            source_value_id=operand.value_id,
        )
    if result_register_count != int(result.type.component_count) * int(registers):
        fail(
            "TLXW_OP_UNSUPPORTED_REMAP",
            STAGE,
            "tt.expand_dims MMA packet does not match its distributed register layout",
            source_op_index=op.index,
            source_value_id=result.value_id,
        )

    lane_width = int(result.type.lane_width or operand.type.lane_width or 64)
    warp_count = max(_layout_warp_count(operand_layout), _layout_warp_count(result_layout))
    source_by_coordinate = {}
    for warp in range(warp_count):
        for lane in range(lane_width):
            for source_register in range(source_register_count):
                coordinate = layouts.linear_layout_coords(
                    source_linear,
                    source_register,
                    lane,
                    warp=warp,
                )
                source_by_coordinate.setdefault(coordinate, []).append(
                    (warp, lane, source_register)
                )

    source_indices = []
    for result_register in range(result_register_count):
        source_registers = set()
        for warp in range(warp_count):
            for lane in range(lane_width):
                coordinate = layouts.linear_layout_coords(
                    result_linear,
                    result_register,
                    lane,
                    warp=warp,
                )
                if axis < 0 or axis >= len(coordinate):
                    fail(
                        "TLXW_OP_UNSUPPORTED_REMAP",
                        STAGE,
                        "tt.expand_dims axis is outside the result layout rank",
                        source_op_index=op.index,
                        source_value_id=result.value_id,
                    )
                source_coordinate = coordinate[:axis] + coordinate[axis + 1:]
                candidates = source_by_coordinate.get(source_coordinate, ())
                same_item = [
                    source_register
                    for source_warp, source_lane, source_register in candidates
                    if source_warp == warp and source_lane == lane
                ]
                if not same_item:
                    fail(
                        "TLXW_OP_UNSUPPORTED_REMAP",
                        STAGE,
                        "tt.expand_dims MMA packet remap requires cross-lane movement",
                        source_op_index=op.index,
                        source_value_id=result.value_id,
                    )
                source_registers.add(int(same_item[0]))
        if len(source_registers) != 1:
            fail(
                "TLXW_OP_UNSUPPORTED_REMAP",
                STAGE,
                "tt.expand_dims MMA packet remap has a lane-varying source register",
                source_op_index=op.index,
                source_value_id=result.value_id,
            )
        source_indices.append(source_registers.pop())
    return tuple(source_indices)


def _convert_broadcast(builder, type_layout_program, op):
    result_target_ids, result_layout_map_ids = _declare_results(
        builder,
        op,
        type_layout_program,
    )
    attrs = {}
    register_payload_remap = _broadcast_register_payload_remap(
        type_layout_program,
        op,
    )
    if register_payload_remap is not None:
        attrs.update(register_payload_remap)
    else:
        component_sources = _broadcast_component_sources(type_layout_program, op)
        if component_sources is not None:
            attrs["component_sources"] = component_sources
    builder.add_op(
        "broadcast",
        operands=_operand_target_ids(builder, op),
        results=result_target_ids,
        attrs=attrs,
        layout_map_ids=result_layout_map_ids,
        source_op_index=op.index,
    )


def _broadcast_register_payload_remap(type_layout_program, op):
    operand = type_layout_program.values[op.operands[0]]
    result = type_layout_program.values[op.results[0]]
    if operand.layout_map_id is None or result.layout_map_id is None:
        return None
    if operand.type.representation in {
            "mask",
            "mask_tuple",
            "per_lane_pointer",
            "pointer_tuple",
    }:
        return None
    operand_layout = type_layout_program.layouts[int(operand.layout_map_id)]
    result_layout = type_layout_program.layouts[int(result.layout_map_id)]
    supported = {"amd_mfma", "blocked", "generic_linear", "linear", "slice"}
    if operand_layout.kind not in supported or result_layout.kind not in supported:
        return None
    source_linear = layouts.distributed_linear_layout(
        operand_layout,
        stage=STAGE,
        source_op_index=op.index,
    )
    result_linear = layouts.distributed_linear_layout(
        result_layout,
        stage=STAGE,
        source_op_index=op.index,
    )
    source_slot_count = layouts.linear_layout_in_dim_size(
        source_linear,
        "register",
    )
    result_slot_count = layouts.linear_layout_in_dim_size(
        result_linear,
        "register",
    )
    source_component_count = int(operand.type.component_count)
    result_component_count = int(result.type.component_count)
    if (source_slot_count % source_component_count
            or result_slot_count % result_component_count):
        fail(
            "TLXW_OP_BROADCAST",
            STAGE,
            "tt.broadcast register slots must evenly partition components",
            source_op_index=op.index,
            source_value_id=result.value_id,
        )
    source_registers = source_slot_count // source_component_count
    result_registers = result_slot_count // result_component_count
    if source_registers == 1 and result_registers == 1:
        return None
    source_bases = layouts.linear_layout_component_registers(
        source_linear,
        operand_layout,
        source_component_count,
        stage=STAGE,
        source_op_index=op.index,
        source_value_id=operand.value_id,
    )
    result_bases = layouts.linear_layout_component_registers(
        result_linear,
        result_layout,
        result_component_count,
        stage=STAGE,
        source_op_index=op.index,
        source_value_id=result.value_id,
    )
    if source_bases != tuple(
            component * source_registers
            for component in range(source_component_count)):
        fail(
            "TLXW_OP_BROADCAST",
            STAGE,
            "tt.broadcast source register packets are not contiguous",
            source_op_index=op.index,
            source_value_id=operand.value_id,
        )
    if result_bases != tuple(
            component * result_registers
            for component in range(result_component_count)):
        fail(
            "TLXW_OP_BROADCAST",
            STAGE,
            "tt.broadcast result register packets are not contiguous",
            source_op_index=op.index,
            source_value_id=result.value_id,
        )
    lane_width = int(result.type.lane_width or operand.type.lane_width or 64)
    source_warps = layouts.layout_warp_count(operand_layout)
    result_warps = layouts.layout_warp_count(result_layout)
    if source_warps != result_warps:
        fail(
            "TLXW_OP_BROADCAST",
            STAGE,
            "tt.broadcast source and result warp counts must match",
            source_op_index=op.index,
        )
    source_by_coordinate = {}
    for warp in range(source_warps):
        for lane in range(lane_width):
            for source_slot in range(source_slot_count):
                coordinate = layouts.linear_layout_coords(
                    source_linear,
                    source_slot,
                    lane,
                    warp=warp,
                )
                source_by_coordinate.setdefault(
                    (warp, lane, coordinate),
                    [],
                ).append(source_slot)

    source_slots = []
    for result_slot in range(result_slot_count):
        slots = set()
        for warp in range(result_warps):
            for lane in range(lane_width):
                result_coordinate = layouts.linear_layout_coords(
                    result_linear,
                    result_slot,
                    lane,
                    warp=warp,
                )
                source_coordinate = tuple(
                    0 if int(source_extent) == 1 else int(coordinate)
                    for source_extent, coordinate in zip(
                        operand_layout.shape,
                        result_coordinate,
                    )
                )
                candidates = source_by_coordinate.get(
                    (warp, lane, source_coordinate),
                    (),
                )
                if not candidates:
                    fail(
                        "TLXW_OP_BROADCAST",
                        STAGE,
                        "tt.broadcast result register is not covered by the source layout",
                        source_op_index=op.index,
                        source_value_id=result.value_id,
                    )
                slots.add(min(int(candidate) for candidate in candidates))
        if len(slots) != 1:
            fail(
                "TLXW_OP_BROADCAST",
                STAGE,
                "tt.broadcast register mapping varies across threads",
                source_op_index=op.index,
                source_value_id=result.value_id,
            )
        source_slots.append(slots.pop())
    return {
        "register_payload_source_slots": tuple(source_slots),
        "result_registers_per_component": int(result_registers),
        "source_component_count": int(source_component_count),
        "source_registers_per_component": int(source_registers),
    }


def _broadcast_component_sources(type_layout_program, op):
    if len(op.operands) != 1 or len(op.results) != 1:
        fail(
            "TLXW_OP_BROADCAST",
            STAGE,
            "tt.broadcast requires one operand and one result",
            source_op_index=op.index,
        )
    operand = type_layout_program.values[op.operands[0]]
    result = type_layout_program.values[op.results[0]]
    if operand.layout_map_id is None or result.layout_map_id is None:
        return None
    operand_layout = type_layout_program.layouts[int(operand.layout_map_id)]
    result_layout = type_layout_program.layouts[int(result.layout_map_id)]
    if len(operand_layout.shape) != len(result_layout.shape):
        fail(
            "TLXW_OP_BROADCAST",
            STAGE,
            "tt.broadcast requires rank-matched source and result layouts",
            source_op_index=op.index,
        )
    if operand_layout.kind not in {
            "blocked",
            "linear",
            "generic_linear",
            "slice",
            "amd_mfma",
    }:
        return None
    if result_layout.kind not in {
            "blocked",
            "linear",
            "generic_linear",
            "slice",
            "amd_mfma",
    }:
        return None
    if int(operand.type.component_count) == int(result.type.component_count):
        return tuple(range(int(result.type.component_count)))
    for source_extent, result_extent in zip(operand_layout.shape, result_layout.shape):
        if int(source_extent) not in {1, int(result_extent)}:
            fail(
                "TLXW_OP_BROADCAST",
                STAGE,
                "tt.broadcast source dimensions must either match the result "
                "or have extent one",
                source_op_index=op.index,
            )

    lane_width = int(result.type.lane_width or operand.type.lane_width or result_layout.lane_width
                     or operand_layout.lane_width or 64)
    warp_count = max(
        layouts.layout_warp_count(operand_layout),
        layouts.layout_warp_count(result_layout),
    )
    source_linear = layouts.distributed_linear_layout(
        operand_layout,
        stage=STAGE,
        source_op_index=op.index,
    )
    result_linear = layouts.distributed_linear_layout(
        result_layout,
        stage=STAGE,
        source_op_index=op.index,
    )
    source_registers = layouts.linear_layout_component_registers(
        source_linear,
        operand_layout,
        operand.type.component_count,
        stage=STAGE,
        source_op_index=op.index,
        source_value_id=operand.value_id,
    )
    result_registers = layouts.linear_layout_component_registers(
        result_linear,
        result_layout,
        result.type.component_count,
        stage=STAGE,
        source_op_index=op.index,
        source_value_id=result.value_id,
    )

    source_by_thread_coord = {}
    for warp in range(int(warp_count)):
        for source_component, source_register in enumerate(source_registers):
            for lane in range(int(lane_width)):
                coords = layouts.linear_layout_coords(
                    source_linear,
                    source_register,
                    lane,
                    warp=warp,
                )
                key = (int(warp), int(lane), tuple(int(coord) for coord in coords))
                source_by_thread_coord.setdefault(key, []).append(int(source_component))

    component_sources = []
    for result_register in result_registers:
        source_registers = set()
        for warp in range(int(warp_count)):
            for lane in range(int(lane_width)):
                result_coords = layouts.linear_layout_coords(
                    result_linear,
                    result_register,
                    lane,
                    warp=warp,
                )
                source_coords = tuple(0 if int(source_extent) == 1 else int(coord)
                                      for source_extent, coord in zip(operand_layout.shape, result_coords))
                source_components = source_by_thread_coord.get((int(warp), int(lane), source_coords))
                if not source_components:
                    fail(
                        "TLXW_OP_BROADCAST",
                        STAGE,
                        "tt.broadcast result coordinate is not covered by the "
                        "source layout",
                        source_op_index=op.index,
                        source_value_id=result.value_id,
                    )
                # Shape application can leave replicated register slots in an
                # explicit linear layout.  They name the same logical tensor
                # element, so use the canonical lowest component just as the
                # register-payload broadcast path does above.
                source_registers.add(min(int(component) for component in source_components))
        if len(source_registers) != 1:
            fail(
                "TLXW_OP_BROADCAST",
                STAGE,
                "tt.broadcast requires a component-invariant source mapping",
                source_op_index=op.index,
                source_value_id=result.value_id,
            )
        component_sources.append(next(iter(source_registers)))
    return tuple(int(source) for source in component_sources)


def _convert_program_id(builder, view):
    builder.add_op(
        "program_id",
        results=view.result_target_ids,
        attrs={"axis": _int_attr(view.attrs, "axis")},
        source_op_index=view.op_index,
    )


def _convert_thread_id(builder, view):
    builder.add_op(
        "thread_id",
        results=view.result_target_ids,
        attrs={"axis": _int_attr(view.attrs, "axis")},
        source_op_index=view.op_index,
    )


def _convert_workitem_id_x(builder, view):
    builder.add_op(
        "thread_id",
        results=view.result_target_ids,
        attrs={"axis": 0},
        source_op_index=view.op_index,
    )


def _convert_index_cast(builder, view):
    if len(view.operand_target_ids) != 1 or len(view.result_target_ids) != 1:
        fail(
            "TLXW_OP_INDEX_CAST",
            STAGE,
            "arith.index_cast requires one operand and one result",
            source_op_index=view.op_index,
        )
    operand_type = builder.values[view.operand_target_ids[0]].type
    result_type = builder.values[view.result_target_ids[0]].type
    if (
        operand_type.representation != result_type.representation
        or operand_type.component_count != result_type.component_count
        or operand_type.lane_width != result_type.lane_width
        or {operand_type.element_type, result_type.element_type}
        not in ({"index", "i32"}, {"index", "i64"})
    ):
        fail(
            "TLXW_OP_INDEX_CAST",
            STAGE,
            "arith.index_cast requires a structural index/integer cast with "
            "unchanged value distribution",
            source_op_index=view.op_index,
        )
    builder.add_op(
        "type_convert",
        operands=view.operand_target_ids,
        results=view.result_target_ids,
        attrs={"mode": "index_cast"},
        source_op_index=view.op_index,
    )


def _convert_if(
    builder,
    conversion_input,
    type_layout_program,
    fact_program,
    op,
):
    if len(op.operands) != 1 or len(op.region_ids) != 2:
        fail(
            "TLXW_OP_UNSUPPORTED_IF",
            STAGE,
            "scf.if conversion requires one condition and then/else regions",
            source_op_index=op.index,
        )
    token_carries = conversion_input.if_token_carries_by_op.get(op.index, ())
    outer_protocol_state = builder.snapshot_protocol_state()
    condition_targets = _operand_target_ids(builder, op)
    data_result_target_ids, result_layout_map_ids = _declare_results(
        builder,
        op,
        type_layout_program,
    )
    token_result_target_ids = tuple(
        builder.add_value(
            target_ir.target_type_from_converted(
                type_layout_program.values[_if_token_carry_type_source_value_id(carry)].type),
            debug_name=f"if_token_result_{op.index}_{index}",
        ) for index, carry in enumerate(token_carries))
    result_target_ids = (*data_result_target_ids, *token_result_target_ids)
    then_region_id = builder.add_region()
    else_region_id = builder.add_region()
    builder.restore_protocol_state(outer_protocol_state)
    with builder.insertion_region(then_region_id):
        then_yields = _convert_region(
            builder,
            conversion_input,
            type_layout_program,
            fact_program,
            op.region_ids[0],
            allow_yield=True,
        )
    then_protocol_state = builder.snapshot_protocol_state()
    builder.restore_protocol_state(outer_protocol_state)
    with builder.insertion_region(else_region_id):
        else_yields = _convert_region(
            builder,
            conversion_input,
            type_layout_program,
            fact_program,
            op.region_ids[1],
            allow_yield=True,
        )
    else_protocol_state = builder.snapshot_protocol_state()
    if (len(then_yields) != len(data_result_target_ids)
            or len(else_yields) != len(data_result_target_ids)):
        fail(
            "TLXW_OP_IF_YIELD_MISMATCH",
            STAGE,
            "scf.if yield counts must match result count",
            source_op_index=op.index,
        )
    _require_yield_layouts(
        type_layout_program,
        then_yields,
        op.results,
        "scf.if then yield and result",
        op,
    )
    _require_yield_layouts(
        type_layout_program,
        else_yields,
        op.results,
        "scf.if else yield and result",
        op,
    )
    with builder.insertion_region(then_region_id):
        then_token_yields = tuple(
            _if_token_yield_target_id(
                builder,
                type_layout_program,
                op,
                carry,
                carry.then_source_value_id,
                "then",
            ) for carry in token_carries)
    with builder.insertion_region(else_region_id):
        else_token_yields = tuple(
            _if_token_yield_target_id(
                builder,
                type_layout_program,
                op,
                carry,
                carry.else_source_value_id,
                "else",
            ) for carry in token_carries)
    for _carry, result_target_id, then_target_id, else_target_id in zip(
        token_carries,
        token_result_target_ids,
        then_token_yields,
        else_token_yields,
    ):
        branch_domains = tuple(dict.fromkeys(
            builder.values[int(target_id)].event_domain
            for target_id in (then_target_id, else_target_id)
            if builder.values[int(target_id)].event_domain
            not in {None, target_ir.EVENT_DOMAIN_EMPTY}
        ))
        if len(branch_domains) == 1:
            builder.set_value_event_domain(result_target_id, branch_domains[0])
    protocol_carry_specs = _if_protocol_carry_specs(
        token_carries,
        outer_protocol_state,
        then_protocol_state,
        else_protocol_state,
    )
    protocol_result_target_ids = tuple(
        _declare_protocol_token(
            builder,
            event_domain=target_ir.EVENT_DOMAIN_LDS_FRONTIER,
            debug_name=f"if_lds_frontier_result_{op.index}_{index}",
        )
        for index, (_keys, then_target_ids, else_target_ids) in enumerate(
            protocol_carry_specs
        )
    )
    with builder.insertion_region(then_region_id):
        then_protocol_yields = tuple(
            _join_protocol_tokens(
                builder,
                then_target_ids,
                op,
                debug_name=f"if_then_lds_frontier_{op.index}_{index}",
            )
            for index, (_keys, then_target_ids, _else_target_ids) in enumerate(
                protocol_carry_specs
            )
        )
    with builder.insertion_region(else_region_id):
        else_protocol_yields = tuple(
            _join_protocol_tokens(
                builder,
                else_target_ids,
                op,
                debug_name=f"if_else_lds_frontier_{op.index}_{index}",
            )
            for index, (_keys, _then_target_ids, else_target_ids) in enumerate(
                protocol_carry_specs
            )
        )
    builder.set_region_yields(
        then_region_id,
        (*tuple(_single_source_target(builder, source_value_id, op) for source_value_id in then_yields),
         *then_token_yields,
         *then_protocol_yields),
    )
    builder.set_region_yields(
        else_region_id,
        (*tuple(_single_source_target(builder, source_value_id, op) for source_value_id in else_yields),
         *else_token_yields,
         *else_protocol_yields),
    )
    data_result_packet_registers = tuple(
        _mma_packet_registers(type_layout_program, type_layout_program.values[source_value_id], op)
        if type_layout_program.values[source_value_id].type.representation in _MMA_PACKET_REPRESENTATIONS
        else 0
        for source_value_id in op.results
    )
    result_packet_registers = (
        (*data_result_packet_registers, *((0, ) * (
            len(token_carries) + len(protocol_result_target_ids)
        )))
        if data_result_packet_registers else ()
    )
    result_target_ids = (
        *result_target_ids,
        *protocol_result_target_ids,
    )
    builder.add_op(
        "if",
        operands=condition_targets,
        results=result_target_ids,
        attrs={
            "result_packet_registers": result_packet_registers,
            "protocol_frontier_result_count": len(
                protocol_result_target_ids
            ),
            "protocol_frontier_key_mappings": tuple(
                (
                    tuple(int(key) for key in keys),
                    int(result_target_id),
                )
                for (keys, _then_ids, _else_ids), result_target_id in zip(
                    protocol_carry_specs,
                    protocol_result_target_ids,
                )
            ),
            "token_carry_target_mappings": tuple(
                (
                    -1
                    if carry.then_source_value_id is None
                    else _single_source_target(
                        builder,
                        carry.then_source_value_id,
                        op,
                    ),
                    -1
                    if carry.else_source_value_id is None
                    else _single_source_target(
                        builder,
                        carry.else_source_value_id,
                        op,
                    ),
                    token_result_target_id,
                )
                for carry, token_result_target_id in zip(
                    token_carries,
                    token_result_target_ids,
                )
            ),
        },
        layout_map_ids=result_layout_map_ids,
        region_ids=(then_region_id, else_region_id),
        source_op_index=op.index,
    )
    _replace_source_targets(
        builder,
        tuple(
            (source_value_id, token_result_target_id)
            for carry, token_result_target_id in zip(token_carries, token_result_target_ids)
            for source_value_id in (carry.then_source_value_id, carry.else_source_value_id)
            if source_value_id is not None
        ),
    )
    builder.restore_protocol_state(outer_protocol_state)
    for (
        keys,
        _then_target_ids,
        _else_target_ids,
    ), protocol_result_target_id in zip(
        protocol_carry_specs,
        protocol_result_target_ids,
    ):
        for key in keys:
            builder.set_protocol_frontier(key, (protocol_result_target_id, ))


def _if_protocol_carry_specs(
    token_carries,
    outer_state,
    then_state,
    else_state,
):
    keys = set(outer_state) | set(then_state) | set(else_state)
    parent = {int(key): int(key) for key in keys}

    def find(key):
        key = int(key)
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(lhs, rhs):
        lhs_root = find(lhs)
        rhs_root = find(rhs)
        if lhs_root != rhs_root:
            parent[max(lhs_root, rhs_root)] = min(lhs_root, rhs_root)

    for carry in token_carries:
        source_ids = tuple(
            int(source_value_id)
            for source_value_id in (
                carry.then_source_value_id,
                carry.else_source_value_id,
            )
            if source_value_id is not None
        )
        keys.update(source_ids)
        for source_value_id in source_ids:
            parent.setdefault(source_value_id, source_value_id)
        if len(source_ids) == 2:
            union(*source_ids)

    key_groups = {}
    for key in sorted(keys):
        key_groups.setdefault(find(key), []).append(int(key))

    specs = []
    for group_keys in key_groups.values():
        then_target_ids = tuple(dict.fromkeys(
            target_id
            for key in group_keys
            for target_id in then_state.get(int(key), ())
        ))
        else_target_ids = tuple(dict.fromkeys(
            target_id
            for key in group_keys
            for target_id in else_state.get(int(key), ())
        ))
        if then_target_ids == else_target_ids:
            continue
        if not then_target_ids and not else_target_ids:
            continue
        specs.append((
            tuple(sorted(int(key) for key in group_keys)),
            then_target_ids,
            else_target_ids,
        ))
    return tuple(specs)


def _if_token_yield_target_id(
    builder,
    type_layout_program,
    op,
    carry,
    source_value_id,
    branch_name,
):
    if source_value_id is not None:
        return _single_source_target(builder, source_value_id, op)
    token_target_id = builder.add_value(
        target_ir.target_type_from_converted(
            type_layout_program.values[_if_token_carry_type_source_value_id(carry)].type),
        debug_name=f"if_token_{branch_name}_yield_{op.index}",
        event_domain=target_ir.EVENT_DOMAIN_EMPTY,
    )
    builder.add_op(
        "token",
        results=(token_target_id, ),
        attrs={"event_domain": target_ir.EVENT_DOMAIN_EMPTY},
        source_op_index=op.index,
    )
    return token_target_id


def _if_token_carry_type_source_value_id(carry):
    if carry.then_source_value_id is not None:
        return int(carry.then_source_value_id)
    if carry.else_source_value_id is not None:
        return int(carry.else_source_value_id)
    raise AssertionError("if token carry requires a token from at least one branch")


def _convert_for(
    builder,
    conversion_input,
    type_layout_program,
    fact_program,
    op,
):
    if len(op.region_ids) != 1 or len(op.operands) < 3:
        fail(
            "TLXW_OP_UNSUPPORTED_FOR",
            STAGE,
            "scf.for conversion requires lower, upper, step, and one body region",
            source_op_index=op.index,
        )
    data_init_arg_count = len(op.operands) - 3
    if len(op.results) != data_init_arg_count:
        fail(
            "TLXW_OP_FOR_RESULT_MISMATCH",
            STAGE,
            "scf.for result count must match iter_args count",
            source_op_index=op.index,
        )
    source_region = conversion_input.regions[op.region_ids[0]]
    if len(source_region.block_arg_ids) != 1 + data_init_arg_count:
        fail(
            "TLXW_OP_FOR_REGION_ARGS",
            STAGE,
            "scf.for body must have induction variable plus iter_arg block args",
            source_op_index=op.index,
        )
    _require_for_iter_layouts(
        type_layout_program,
        op.operands[3:],
        op.results,
        source_region.block_arg_ids[1:],
        op,
    )
    source_region_op_names = frozenset(
        conversion_input.ops[int(op_index)].name
        for op_index in source_region.op_indices
    )
    explicit_warp_pipeline_protocol = (
        "rocdl.sched.barrier" in source_region_op_names
        and "rocdl.s.setprio" in source_region_op_names
    )

    token_carries = conversion_input.loop_token_carries_by_op.get(op.index, ())
    outer_protocol_state = builder.snapshot_protocol_state()
    protocol_carry_specs = _loop_protocol_carry_specs(
        conversion_input,
        op,
        token_carries,
        outer_protocol_state,
    )
    source_loop_operands = _operand_target_ids(builder, op)
    token_init_target_ids = tuple(
        _loop_token_init_target_id(
            builder,
            type_layout_program,
            op,
            carry,
        ) for carry in token_carries)
    protocol_init_target_ids = tuple(
        _join_protocol_tokens(
            builder,
            init_target_ids,
            op,
            debug_name=f"loop_lds_frontier_init_{op.index}_{index}",
        )
        for index, (
            _keys,
            _init_key,
            _yield_key,
            init_target_ids,
        ) in enumerate(protocol_carry_specs)
    )
    loop_operands = (
        *source_loop_operands,
        *token_init_target_ids,
        *protocol_init_target_ids,
    )
    result_target_ids, result_layout_map_ids = _declare_results(
        builder,
        op,
        type_layout_program,
    )
    token_result_target_ids = tuple(
        builder.add_value(
            target_ir.target_type_from_converted(
                type_layout_program.values[_loop_token_carry_type_source_value_id(carry)].type),
            debug_name=f"loop_token_result_{op.index}_{index}",
            event_domain=_loop_token_carry_event_domain(builder, carry),
        ) for index, carry in enumerate(token_carries))
    protocol_result_target_ids = tuple(
        _declare_protocol_token(
            builder,
            event_domain=target_ir.EVENT_DOMAIN_LDS_FRONTIER,
            debug_name=f"loop_lds_frontier_result_{op.index}_{index}",
        )
        for index, (
            _keys,
            _init_key,
            _yield_key,
            init_target_ids,
        ) in enumerate(protocol_carry_specs)
    )
    result_target_ids = (
        *result_target_ids,
        *token_result_target_ids,
        *protocol_result_target_ids,
    )
    block_arg_target_ids = tuple(
        builder.add_value(
            target_ir.target_type_from_converted(type_layout_program.values[source_value_id].type),
            source_value_id=source_value_id,
            debug_name=f"r{op.region_ids[0]}_arg{index}",
        ) for index, source_value_id in enumerate(source_region.block_arg_ids))
    token_block_arg_target_ids = tuple(
        builder.add_value(
            target_ir.target_type_from_converted(
                type_layout_program.values[_loop_token_carry_type_source_value_id(carry)].type),
            debug_name=f"loop_token_arg_{op.index}_{index}",
            event_domain=_loop_token_carry_event_domain(builder, carry),
        ) for index, carry in enumerate(token_carries))
    protocol_block_arg_target_ids = tuple(
        _declare_protocol_token(
            builder,
            event_domain=target_ir.EVENT_DOMAIN_LDS_FRONTIER,
            debug_name=f"loop_lds_frontier_arg_{op.index}_{index}",
        )
        for index, (
            _keys,
            _init_key,
            _yield_key,
            init_target_ids,
        ) in enumerate(protocol_carry_specs)
    )
    block_arg_target_ids = (
        *block_arg_target_ids,
        *token_block_arg_target_ids,
        *protocol_block_arg_target_ids,
    )
    target_region_id = builder.add_region(block_arg_ids=block_arg_target_ids)
    token_issue_dependency_pairs = _loop_token_carry_issue_dependencies(
        token_carries,
        token_block_arg_target_ids,
    )
    issue_dependencies = _loop_async_issue_dependencies(
        tuple(carry for carry, _token_block_arg_target_id in token_issue_dependency_pairs),
        tuple(token_block_arg_target_id for _carry, token_block_arg_target_id in token_issue_dependency_pairs),
    )
    body_conversion_input = replace(
        conversion_input,
        async_issue_dependency_target_ids_by_op={
            **conversion_input.async_issue_dependency_target_ids_by_op,
            **issue_dependencies,
        },
    )
    saved_token_targets = _replace_source_targets(
        builder,
        tuple((carry.init_source_value_id, token_block_arg_target_id) for carry, token_block_arg_target_id in zip(
            token_carries,
            token_block_arg_target_ids,
        ) if carry.init_source_value_id is not None),
    )
    builder.restore_protocol_state(outer_protocol_state)
    for (
        keys,
        init_key,
        _yield_key,
        _init_target_ids,
    ), protocol_block_arg_target_id in zip(
        protocol_carry_specs,
        protocol_block_arg_target_ids,
    ):
        builder.set_protocol_frontier(
            init_key,
            (protocol_block_arg_target_id, ),
        )
        for key in keys:
            if int(key) in outer_protocol_state:
                builder.set_protocol_frontier(
                    key,
                    (protocol_block_arg_target_id, ),
                )
            elif key != init_key:
                builder.protocol_frontiers.pop(int(key), None)
    with builder.insertion_region(target_region_id):
        try:
            yielded_source_values = _convert_region(
                builder,
                body_conversion_input,
                type_layout_program,
                fact_program,
                op.region_ids[0],
                allow_yield=True,
            )
        finally:
            _restore_source_targets(builder, saved_token_targets)
    body_protocol_state = builder.snapshot_protocol_state()
    if len(yielded_source_values) != data_init_arg_count:
        fail(
            "TLXW_OP_FOR_YIELD_MISMATCH",
            STAGE,
            "scf.for yield count must match iter_args count",
            source_op_index=op.index,
        )
    _require_yield_layouts(
        type_layout_program,
        yielded_source_values,
        op.results,
        "scf.for yield and result",
        op,
    )
    yielded_target_ids = tuple(
        _single_source_target(builder, source_value_id, op) for source_value_id in yielded_source_values)
    with builder.insertion_region(target_region_id):
        yielded_token_target_ids = tuple(
            _loop_token_yield_target_id(
                builder,
                type_layout_program,
                op,
                carry,
            ) for carry in token_carries)
        yielded_protocol_target_ids = tuple(
            _join_protocol_tokens(
                builder,
                body_protocol_state.get(int(yield_key), ()),
                op,
                debug_name=f"loop_lds_frontier_yield_{op.index}_{index}",
            )
            for index, (
                _keys,
                _init_key,
                yield_key,
                _init_target_ids,
            ) in enumerate(protocol_carry_specs)
        )
    builder.set_region_yields(
        target_region_id,
        (
            *yielded_target_ids,
            *yielded_token_target_ids,
            *yielded_protocol_target_ids,
        ),
    )
    builder.add_op(
        "for_loop",
        operands=loop_operands,
        results=result_target_ids,
        attrs={
            "init_arg_count": (
                data_init_arg_count
                + len(token_carries)
                + len(protocol_carry_specs)
            ),
            "protocol_frontier_init_arg_indices": tuple(
                data_init_arg_count + len(token_carries) + index
                for index in range(len(protocol_carry_specs))
            ),
            "protocol_frontier_key_mappings": tuple(
                (
                    tuple(int(key) for key in keys),
                    int(block_arg_target_id),
                    int(result_target_id),
                )
                for (
                    keys,
                    _init_key,
                    _yield_key,
                    _init_target_ids,
                ), block_arg_target_id, result_target_id in zip(
                    protocol_carry_specs,
                    protocol_block_arg_target_ids,
                    protocol_result_target_ids,
                )
            ),
            "source_result_count": data_init_arg_count,
            "explicit_warp_pipeline_protocol": explicit_warp_pipeline_protocol,
        },
        layout_map_ids=result_layout_map_ids,
        region_ids=(target_region_id, ),
        source_op_index=op.index,
    )
    _replace_source_targets(
        builder,
        tuple((carry.yield_source_value_id, token_result_target_id) for carry, token_result_target_id in zip(
            token_carries,
            token_result_target_ids,
        ) if carry.yield_source_value_id is not None),
    )
    builder.restore_protocol_state(outer_protocol_state)
    for (
        keys,
        _init_key,
        _yield_key,
        _init_target_ids,
    ), protocol_result_target_id in zip(
        protocol_carry_specs,
        protocol_result_target_ids,
    ):
        for key in keys:
            builder.set_protocol_frontier(key, (protocol_result_target_id, ))


def _loop_token_init_target_id(builder, type_layout_program, op, carry):
    init_source_value_id = carry.init_source_value_id
    if init_source_value_id is not None:
        return _single_source_target(builder, init_source_value_id, op)
    yield_source_value_id = carry.yield_source_value_id
    token_target_id = builder.add_value(
        target_ir.target_type_from_converted(type_layout_program.values[yield_source_value_id].type),
        debug_name=f"loop_token_init_{op.index}",
        event_domain=target_ir.EVENT_DOMAIN_EMPTY,
    )
    builder.add_op(
        "token",
        results=(token_target_id, ),
        attrs={"event_domain": target_ir.EVENT_DOMAIN_EMPTY},
        source_op_index=op.index,
    )
    return token_target_id


def _loop_token_yield_target_id(builder, type_layout_program, op, carry):
    yield_source_value_id = carry.yield_source_value_id
    if yield_source_value_id is not None:
        return _single_source_target(builder, yield_source_value_id, op)
    init_source_value_id = carry.init_source_value_id
    if init_source_value_id is None:
        fail(
            "TLXW_OP_UNSUPPORTED_FOR_TOKENS",
            STAGE,
            "scf.for async token carry has neither an initial nor yielded token",
            source_op_index=op.index,
        )
    token_target_id = builder.add_value(
        target_ir.target_type_from_converted(type_layout_program.values[init_source_value_id].type),
        debug_name=f"loop_token_yield_{op.index}",
        event_domain=target_ir.EVENT_DOMAIN_EMPTY,
    )
    builder.add_op(
        "token",
        results=(token_target_id, ),
        attrs={"event_domain": target_ir.EVENT_DOMAIN_EMPTY},
        source_op_index=op.index,
    )
    return token_target_id


def _loop_token_carry_type_source_value_id(carry):
    if carry.yield_source_value_id is not None:
        return int(carry.yield_source_value_id)
    if carry.init_source_value_id is not None:
        return int(carry.init_source_value_id)
    raise AssertionError("loop token carry requires an initial or yielded source token")


def _loop_token_carry_event_domain(builder, carry):
    for source_value_id in (
        carry.init_source_value_id,
        carry.yield_source_value_id,
    ):
        if source_value_id is None:
            continue
        targets = builder.source_value_targets.get(int(source_value_id), ())
        if len(targets) != 1:
            continue
        domain = builder.values[int(targets[0])].event_domain
        if domain is not None:
            return domain
    return None


def _loop_token_carry_issue_dependencies(token_carries, token_block_arg_target_ids):
    return tuple((carry, token_block_arg_target_id) for carry, token_block_arg_target_id in zip(
        token_carries,
        token_block_arg_target_ids,
    ) if carry.add_issue_dependency)


def _loop_protocol_carry_specs(
    conversion_input,
    op,
    token_carries,
    outer_protocol_state,
):
    specs = []
    covered_keys = set()
    for carry in token_carries:
        if not carry.readiness_carry:
            continue
        init_key = carry.init_source_value_id
        yield_key = carry.yield_source_value_id
        if init_key is None and yield_key is None:
            continue
        init_key = int(yield_key if init_key is None else init_key)
        yield_key = int(init_key if yield_key is None else yield_key)
        keys = tuple(sorted({init_key, yield_key}))
        covered_keys.update(keys)
        specs.append((
            keys,
            init_key,
            yield_key,
            tuple(outer_protocol_state.get(init_key, ())),
        ))

    body_dependency_keys = _region_async_protocol_dependency_keys(
        conversion_input,
        op.region_ids[0],
    )
    body_wait_keys = tuple(
        int(result_value_id)
        for op_index in sorted(
            _region_op_indices_recursive(
                conversion_input,
                op.region_ids[0],
            )
        )
        for source_op in (conversion_input.ops[int(op_index)], )
        if source_op.name == "ttg.async_wait"
        for result_value_id in source_op.results[:1]
        if int(result_value_id) in body_dependency_keys
    )
    outer_frontier_keys = tuple(
        int(key)
        for key, target_ids in sorted(outer_protocol_state.items())
        if target_ids and int(key) not in covered_keys
    )
    uncovered_body_wait_keys = tuple(
        key for key in body_wait_keys if key not in covered_keys
    )
    if outer_frontier_keys and uncovered_body_wait_keys:
        # One collective LDS frontier is sufficient even when a staged loop
        # rotates several logical wait epochs.  All DMA packets consume the
        # same frontier, and the body yields the join of the epoch(s) produced
        # on that path.
        keys = tuple(sorted({
            *outer_frontier_keys,
            *uncovered_body_wait_keys,
        }))
        specs.append((
            keys,
            outer_frontier_keys[0],
            uncovered_body_wait_keys[-1],
            tuple(dict.fromkeys(
                target_id
                for key in outer_frontier_keys
                for target_id in outer_protocol_state.get(key, ())
            )),
        ))
        covered_keys.update(keys)
    for key in sorted(set(outer_protocol_state).intersection(body_dependency_keys)):
        key = int(key)
        if key in covered_keys:
            continue
        specs.append((
            (key, ),
            key,
            key,
            tuple(outer_protocol_state.get(key, ())),
        ))
    return tuple(specs)


def _region_async_protocol_dependency_keys(conversion_input, region_id):
    keys = set()
    for op_index in conversion_input.regions[int(region_id)].op_indices:
        keys.update(
            int(source_value_id)
            for source_value_id in (
                conversion_input.async_protocol_dependency_value_ids_by_op.get(
                    int(op_index), ()
                )
            )
        )
        for child_region_id in conversion_input.ops[int(op_index)].region_ids:
            keys.update(
                _region_async_protocol_dependency_keys(
                    conversion_input,
                    child_region_id,
                )
            )
    return frozenset(keys)


def _loop_async_issue_dependencies(token_carries, token_block_arg_target_ids):
    dependencies_by_op = {}
    for carry, token_block_arg_target_id in zip(token_carries, token_block_arg_target_ids):
        for op_index in carry.issue_dependency_op_indices:
            existing = dependencies_by_op.setdefault(op_index, tuple())
            dependencies_by_op[op_index] = (
                *existing,
                int(token_block_arg_target_id),
            )
    return dependencies_by_op


def _region_op_indices_recursive(conversion_input, region_id):
    result = []
    for op_index in conversion_input.regions[region_id].op_indices:
        result.append(op_index)
        for child_region_id in conversion_input.ops[op_index].region_ids:
            result.extend(_region_op_indices_recursive(conversion_input, child_region_id))
    return frozenset(result)


def _replace_source_targets(builder, replacements):
    saved = {}
    for source_value_id, target_value_id in replacements:
        source_value_id = int(source_value_id)
        saved[source_value_id] = builder.source_value_targets.get(source_value_id)
        builder.source_value_targets[source_value_id] = (int(target_value_id), )
    return saved


def _restore_source_targets(builder, saved):
    for source_value_id, targets in saved.items():
        if targets is None:
            builder.source_value_targets.pop(source_value_id, None)
        else:
            builder.source_value_targets[source_value_id] = targets


def _convert_local_alloc(
    builder,
    conversion_input,
    type_layout_program,
    op,
):
    if len(op.results) != 1:
        fail(
            "TLXW_OP_LOCAL_ALLOC_RESULT",
            STAGE,
            "ttg.local_alloc must produce one memdesc result",
            source_op_index=op.index,
        )
    result_target_ids, result_layout_map_ids = _declare_results(
        builder,
        op,
        type_layout_program,
    )
    memdesc = _memdesc_info(conversion_input, op.results[0], op)
    shape = tuple(memdesc.shape or memdesc.alloc_shape)
    builder.add_op(
        "local_alloc",
        results=result_target_ids,
        attrs={
            "allocation_bytes":
            int(conversion_input.local_alloc_allocation_bytes.get(
                op.results[0],
                memdesc.allocation_bytes,
            )),
            "align":
            16,
            "element_type":
            memdesc.element_type,
            "shape":
            tuple(int(dim) for dim in shape),
        },
        layout_map_ids=result_layout_map_ids,
        source_op_index=op.index,
    )


def _convert_memdesc_index(
    builder,
    conversion_input,
    type_layout_program,
    op,
):
    if len(op.operands) != 2 or len(op.results) != 1:
        fail(
            "TLXW_OP_MEMDESC_INDEX",
            STAGE,
            "ttg.memdesc_index requires memdesc/index operands and one result",
            source_op_index=op.index,
        )
    result_target_ids, result_layout_map_ids = _declare_results(
        builder,
        op,
        type_layout_program,
    )
    parent_memdesc = _memdesc_info(conversion_input, op.operands[0], op)
    memdesc = _memdesc_info(conversion_input, op.results[0], op)
    slot_size_bytes = int(
        conversion_input.memdesc_index_slot_stride_bytes.get(
            op.results[0],
            conversion_input.memdesc_physical_allocation_bytes.get(
                op.results[0],
                memdesc.allocation_bytes,
            ),
        ))
    element_byte_width = memdesc.element_byte_width
    if element_byte_width is None or slot_size_bytes % int(element_byte_width):
        fail(
            "TLXW_OP_MEMDESC_INDEX",
            STAGE,
            "ttg.memdesc_index slot size is not element aligned",
            source_op_index=op.index,
            source_value_id=op.results[0],
        )
    element_count = slot_size_bytes // int(element_byte_width)
    static_byte_offset = conversion_input.static_memdesc_byte_offsets.get(op.results[0])
    builder.add_op(
        "memdesc_index",
        operands=_operand_target_ids(builder, op),
        results=result_target_ids,
        attrs={
            "element_byte_width": memdesc.element_byte_width,
            "elements_per_slot": element_count,
            "slot_count": _memdesc_index_slot_count(parent_memdesc, memdesc),
            "static_byte_offset": static_byte_offset,
        },
        layout_map_ids=result_layout_map_ids,
        source_op_index=op.index,
    )


def _convert_memdesc_trans(builder, type_layout_program, op):
    if len(op.operands) != 1 or len(op.results) != 1:
        fail(
            "TLXW_OP_MEMDESC_TRANS",
            STAGE,
            "ttg.memdesc_trans requires one memdesc operand and one result",
            source_op_index=op.index,
        )
    operand = type_layout_program.values[op.operands[0]]
    result = type_layout_program.values[op.results[0]]
    if operand.type.representation != "memdesc" or result.type.representation != "memdesc":
        fail(
            "TLXW_OP_MEMDESC_TRANS",
            STAGE,
            "ttg.memdesc_trans requires memdesc operand and result types",
            source_op_index=op.index,
        )
    result_target_ids, result_layout_map_ids = _declare_results(
        builder,
        op,
        type_layout_program,
    )
    builder.add_op(
        "memdesc_view",
        operands=_operand_target_ids(builder, op),
        results=result_target_ids,
        attrs={"view": "transpose"},
        layout_map_ids=result_layout_map_ids,
        source_op_index=op.index,
    )


def _convert_memdesc_reshape(builder, type_layout_program, op):
    if len(op.operands) != 1 or len(op.results) != 1:
        fail(
            "TLXW_OP_MEMDESC_RESHAPE",
            STAGE,
            "ttg.memdesc_reshape requires one memdesc operand and one result",
            source_op_index=op.index,
        )
    operand = type_layout_program.values[op.operands[0]]
    result = type_layout_program.values[op.results[0]]
    if operand.type.representation != "memdesc" or result.type.representation != "memdesc":
        fail(
            "TLXW_OP_MEMDESC_RESHAPE",
            STAGE,
            "ttg.memdesc_reshape requires memdesc operand and result types",
            source_op_index=op.index,
        )
    result_target_ids, result_layout_map_ids = _declare_results(
        builder,
        op,
        type_layout_program,
    )
    builder.add_op(
        "memdesc_view",
        operands=_operand_target_ids(builder, op),
        results=result_target_ids,
        attrs={"view": "reshape"},
        layout_map_ids=result_layout_map_ids,
        source_op_index=op.index,
    )


def _convert_memdesc_reinterpret(builder, type_layout_program, op):
    if len(op.operands) != 1 or len(op.results) != 1:
        fail(
            "TLXW_OP_MEMDESC_REINTERPRET",
            STAGE,
            "ttg.memdesc_reinterpret requires one memdesc operand and one result",
            source_op_index=op.index,
        )
    operand = type_layout_program.values[op.operands[0]]
    result = type_layout_program.values[op.results[0]]
    if operand.type.representation != "memdesc" or result.type.representation != "memdesc":
        fail(
            "TLXW_OP_MEMDESC_REINTERPRET",
            STAGE,
            "ttg.memdesc_reinterpret requires memdesc operand and result types",
            source_op_index=op.index,
        )
    result_target_ids, result_layout_map_ids = _declare_results(
        builder,
        op,
        type_layout_program,
    )
    builder.add_op(
        "memdesc_view",
        operands=_operand_target_ids(builder, op),
        results=result_target_ids,
        attrs={"view": "reinterpret"},
        layout_map_ids=result_layout_map_ids,
        source_op_index=op.index,
    )


def _convert_memdesc_subslice(
    builder,
    conversion_input,
    type_layout_program,
    op,
):
    if len(op.operands) != 1 or len(op.results) != 1:
        fail(
            "TLXW_OP_MEMDESC_SUBSLICE",
            STAGE,
            "ttg.memdesc_subslice requires one memdesc operand and one result",
            source_op_index=op.index,
        )
    view = _memdesc_view_info(conversion_input, op.results[0], op)
    result_target_ids, result_layout_map_ids = _declare_results(
        builder,
        op,
        type_layout_program,
    )
    builder.add_op(
        "memdesc_view",
        operands=_operand_target_ids(builder, op),
        results=result_target_ids,
        attrs={
            "view": "subslice",
            "logical_origin": tuple(int(value) for value in view.logical_origin),
            "physical_shape": tuple(int(value) for value in view.physical_shape),
        },
        layout_map_ids=result_layout_map_ids,
        source_op_index=op.index,
    )


def _convert_buffer_load_to_local(
    builder,
    conversion_input,
    type_layout_program,
    fact_program,
    op,
):
    fields = _buffer_load_to_local_fields(op)
    if fields["other_value_id"] is not None:
        fail(
            "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
            STAGE,
            "amdg.buffer_load_to_local other fallback is not converted yet",
            source_op_index=op.index,
        )
    _require_default_cache(fields["cache"], op)
    token_node = conversion_input.token_nodes_by_op.get(op.index)
    if token_node is None or token_node.value_id not in op.results:
        fail(
            "TLXW_OP_BUFFER_ASYNC_TOKEN",
            STAGE,
            "amdg.buffer_load_to_local requires a token graph node",
            source_op_index=op.index,
        )
    async_group_ids = tuple(
        int(group.group_id)
        for group in conversion_input.token_groups_by_id.values()
        if token_node.value_id in group.member_token_ids
    )
    if len(async_group_ids) > 1:
        fail(
            "TLXW_OP_BUFFER_ASYNC_GROUP",
            STAGE,
            "amdg.buffer_load_to_local token belongs to multiple commit groups",
            source_op_index=op.index,
        )
    async_group_id = async_group_ids[0] if async_group_ids else -1
    range_fact = _pointer_byte_range_fact(
        fact_program,
        fields["base_value_id"],
        op,
    )
    result_target_ids, result_layout_map_ids = _declare_results(
        builder,
        op,
        type_layout_program,
    )
    for result_target_id in result_target_ids:
        builder.set_value_event_domain(
            result_target_id,
            target_ir.EVENT_DOMAIN_DMA_COMPLETION,
        )
    base_target_id = _single_source_target(builder, fields["base_value_id"], op)
    source_issue_dependency_target_ids = tuple(dict.fromkeys(
        conversion_input.async_issue_dependency_target_ids_by_op.get(
            op.index,
            (),
        )
    ))
    issue_dependency_target_ids = source_issue_dependency_target_ids
    destination_target_id = _single_source_target(
        builder,
        fields["memdesc_value_id"],
        op,
    )
    memdesc = _memdesc_info(conversion_input, fields["memdesc_value_id"], op)
    if memdesc.element_byte_width is None:
        fail(
            "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
            STAGE,
            "amdg.buffer_load_to_local DMA requires known element byte width",
            source_op_index=op.index,
            source_value_id=fields["memdesc_value_id"],
        )
    offset_type = type_layout_program.values[fields["offset_value_id"]].type
    if offset_type.representation not in {"simd", "simd_tuple"}:
        fail(
            "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
            STAGE,
            "amdg.buffer_load_to_local requires SIMD offset components",
            source_op_index=op.index,
            source_value_id=fields["offset_value_id"],
        )
    has_mask = fields["mask_value_id"] is not None
    mask_component_count = 0
    mask_alignment = 1
    if has_mask:
        mask = type_layout_program.values[fields["mask_value_id"]]
        mask_component_count = int(mask.type.component_count)
        if int(mask.type.component_count) not in (1, int(offset_type.component_count)):
            fail(
                "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
                STAGE,
                "amdg.buffer_load_to_local mask must be scalar or match "
                "offset components",
                source_op_index=op.index,
                source_value_id=fields["mask_value_id"],
            )
        _require_mask_layout_compatible(
            type_layout_program,
            mask,
            type_layout_program.values[fields["offset_value_id"]],
            "amdg.buffer_load_to_local mask",
            op,
        )
    packet_plan = _buffer_load_to_local_packet_plan(
        conversion_input,
        type_layout_program,
        fact_program,
        fields["memdesc_value_id"],
        fields["offset_value_id"],
        memdesc,
        int(offset_type.lane_width or conversion_input.threads_per_warp),
        op,
    )
    if packet_plan is not None and has_mask:
        packet_elements = int(packet_plan["packet_elements"])
        if _buffer_load_to_local_packet_mask_is_supported(
                conversion_input,
                type_layout_program,
                fact_program,
                fields["mask_value_id"],
                packet_plan,
                memdesc,
                op,
        ):
            mask_alignment = packet_elements
        else:
            packet_plan = None
    if packet_plan is not None:
        scalar_component_sources = _buffer_load_to_local_packet_scalar_component_sources(
            conversion_input,
            type_layout_program,
            packet_plan,
            packet_plan["scalar_value_ids"],
            tuple(int(dim) for dim in memdesc.shape),
            int(offset_type.lane_width or conversion_input.threads_per_warp),
            op,
        )
        if scalar_component_sources is None:
            packet_plan = None
    if packet_plan is not None:
        packet_elements = int(packet_plan["packet_elements"])
        mask_source_indices = (
            tuple(0 for _ in range(int(packet_plan["component_count"])))
            if mask_component_count == 1
            else tuple(
                component * packet_elements
                for component in range(int(packet_plan["component_count"]))
            )
        )
        source_offset_upper = _buffer_source_offset_upper(
            range_fact.upper,
            packet_plan["packet_bytes"],
            memdesc.element_byte_width,
            op,
        )
        source_offset_no_signed_wrap = _affine_source_offset_no_signed_wrap(
            conversion_input,
            fact_program,
            packet_plan["source_affine"],
            op,
            source_offset_upper,
        )
        runtime_offset_target_id = _materialize_packet_affine_edge(
            builder,
            type_layout_program,
            int(fields["offset_value_id"]),
            packet_plan,
            scalar_component_sources,
            op,
            no_signed_wrap=bool(source_offset_no_signed_wrap),
            offset_range=(0, int(source_offset_upper)),
        )
        runtime_mask_target_id = None
        if has_mask:
            runtime_mask_target_id = _component_remap_edge(
                builder,
                _single_source_target(
                    builder,
                    int(fields["mask_value_id"]),
                    op,
                ),
                mask_source_indices,
                op,
            )
        packet_operands = [
            destination_target_id,
            base_target_id,
            runtime_offset_target_id,
        ]
        if runtime_mask_target_id is not None:
            packet_operands.append(runtime_mask_target_id)
        packet_operands.extend(issue_dependency_target_ids)
        builder.add_op(
            "buffer_load_to_local",
            operands=tuple(packet_operands),
            results=result_target_ids,
            attrs={
                "cache_modifier": int(fields["cache"] or 1),
                "async_group_id": int(async_group_id),
                "component_count": int(packet_plan["component_count"]),
                "component_thread_count": int(packet_plan["component_thread_count"]),
                "destination_component_offsets": tuple(packet_plan["destination_component_offsets"]),
                "destination_wave_count": int(packet_plan["destination_wave_count"]),
                "destination_wave_offset_coefficients_dwords": tuple(
                    int(value) for value in packet_plan["destination_wave_offset_coefficients_dwords"]),
                "destination_wave_stride_dwords": int(packet_plan["destination_wave_stride_dwords"]),
                "element_byte_width": int(memdesc.element_byte_width),
                "element_type": memdesc.element_type,
                "has_mask": has_mask,
                "has_stride_operand": fields["stride_value_id"] is not None,
                "lane_width": int(offset_type.lane_width or conversion_input.threads_per_warp),
                "mask_alignment": int(mask_alignment),
                "mask_component_count": (
                    int(packet_plan["component_count"]) if has_mask else 0
                ),
                "mask_mode": "zero_fill_inactive" if has_mask else "none",
                "mode": "dma_packet_lds",
                "packet_bytes": int(packet_plan["packet_bytes"]),
                "packet_elements": int(packet_plan["packet_elements"]),
                "range_bytes": int(range_fact.upper),
                "issue_dependency_count": len(issue_dependency_target_ids),
                "source_issue_dependency_count": len(
                    source_issue_dependency_target_ids
                ),
            },
            fact_ids=(range_fact.fact_id, ),
            fact_target_ids=(base_target_id, ),
            layout_map_ids=result_layout_map_ids,
            source_op_index=op.index,
        )
        return
    destination_plan = _local_component_store_plan(
        conversion_input,
        type_layout_program,
        fields["memdesc_value_id"],
        fields["offset_value_id"],
        int(offset_type.component_count),
        int(offset_type.lane_width or conversion_input.threads_per_warp),
        op,
    )
    scalar_offset_upper = _buffer_source_offset_upper(
        range_fact.upper,
        memdesc.element_byte_width,
        memdesc.element_byte_width,
        op,
    )
    source_affine_plan = _buffer_affine_offset_plan(
        conversion_input,
        type_layout_program,
        fact_program,
        fields["offset_value_id"],
        int(offset_type.component_count),
        int(offset_type.lane_width or conversion_input.threads_per_warp),
        op,
    )
    if has_mask and mask_alignment == 1:
        mask_alignment = _buffer_mask_alignment(
            conversion_input,
            type_layout_program,
            fact_program,
            fields["mask_value_id"],
            int(offset_type.component_count),
            int(offset_type.lane_width or conversion_input.threads_per_warp),
            int(memdesc.element_byte_width),
            op,
        )
    original_component_count = int(offset_type.component_count)
    active_components = tuple(range(original_component_count))
    if has_mask and mask_component_count == original_component_count:
        maybe_active_components = _mask_maybe_active_components(
            conversion_input,
            type_layout_program,
            fact_program,
            int(fields["mask_value_id"]),
            original_component_count,
            int(offset_type.lane_width or conversion_input.threads_per_warp),
            op,
        )
        # An empty masked memory operation still needs a structural token
        # representation.  Until target IR has one, retain the conservative
        # scalarized form rather than inventing an emitter-only no-op.
        if maybe_active_components:
            active_components = maybe_active_components
    if active_components != tuple(range(original_component_count)):
        destination_plan = _select_local_component_store_plan_components(
            destination_plan,
            active_components,
        )
    mask_source_indices = (
        ()
        if not has_mask
        else tuple(
            0
            if mask_component_count == 1
            else component // int(mask_alignment) * int(mask_alignment)
            for component in active_components
        )
    )
    source_offset_no_signed_wrap = (
        _affine_source_offset_no_signed_wrap(
            conversion_input,
            fact_program,
            source_affine_plan["source_affine"],
            op,
            scalar_offset_upper,
        )
        if source_affine_plan is not None
        else False
    )
    runtime_offset_target_id = _materialize_affine_edge_or_original(
        builder,
        conversion_input,
        type_layout_program,
        fact_program,
        int(fields["offset_value_id"]),
        op,
        no_signed_wrap=bool(source_offset_no_signed_wrap),
        result_element_type="index",
        value_range=(0, int(scalar_offset_upper)),
    )
    runtime_offset_target_id = _component_remap_edge(
        builder,
        runtime_offset_target_id,
        active_components,
        op,
    )
    runtime_mask_target_id = None
    if has_mask:
        runtime_mask_target_id = _component_remap_edge(
            builder,
            _single_source_target(
                builder,
                int(fields["mask_value_id"]),
                op,
            ),
            mask_source_indices,
            op,
        )
    scalarized_operands = [
        destination_target_id,
        base_target_id,
        runtime_offset_target_id,
    ]
    if runtime_mask_target_id is not None:
        scalarized_operands.append(runtime_mask_target_id)
    scalarized_operands.extend(issue_dependency_target_ids)
    builder.add_op(
        "buffer_load_to_local",
        operands=tuple(scalarized_operands),
        results=result_target_ids,
        attrs={
            "cache_modifier": int(fields["cache"] or 1),
            "async_group_id": int(async_group_id),
            "component_count": len(active_components),
            **_local_component_store_plan_attrs(destination_plan),
            "element_byte_width": int(memdesc.element_byte_width),
            "element_type": memdesc.element_type,
            "has_mask": has_mask,
            "has_stride_operand": fields["stride_value_id"] is not None,
            "lane_width": int(offset_type.lane_width or conversion_input.threads_per_warp),
            "mask_mode": "exec_where" if has_mask else "none",
            "mask_alignment": int(mask_alignment),
            "mask_component_count": (
                len(active_components) if has_mask else 0
            ),
            "mode": "scalarized_load_store",
            "range_bytes": int(range_fact.upper),
            "issue_dependency_count": len(issue_dependency_target_ids),
            "source_issue_dependency_count": len(
                source_issue_dependency_target_ids
            ),
        },
        fact_ids=(range_fact.fact_id, ),
        fact_target_ids=(base_target_id, ),
        layout_map_ids=result_layout_map_ids,
        source_op_index=op.index,
    )


def _local_component_store_plan_attrs(destination_plan):
    offset_mode = destination_plan["offset_mode"]
    if offset_mode == "affine":
        return {
            "destination_offset_mode": "affine",
            "destination_component_offsets": tuple(destination_plan["component_offsets"]),
            "destination_lane_stride_elements": int(destination_plan["lane_stride_elements"]),
            "destination_wave_stride_elements": int(destination_plan["wave_stride_elements"]),
        }
    if offset_mode == "layout_coordinates":
        return {
            "destination_offset_mode":
            "layout_coordinates",
            "destination_coordinate_shape":
            tuple(int(dim) for dim in destination_plan["coordinate_shape"]),
            "destination_component_coordinate_bases":
            tuple(tuple(int(value) for value in bases) for bases in destination_plan["component_coordinate_bases"]),
            "destination_workitem_coordinate_coefficients":
            tuple(
                tuple(int(value)
                      for value in coefficients)
                for coefficients in destination_plan["workitem_coordinate_coefficients"]),
            **destination_plan["shared_layout_attrs"],
        }
    fail(
        "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
        STAGE,
        f"unsupported scalarized destination offset mode {offset_mode}",
    )


def _select_local_component_store_plan_components(destination_plan, components):
    components = tuple(int(component) for component in components)
    selected = dict(destination_plan)
    if destination_plan["offset_mode"] == "affine":
        selected["component_offsets"] = tuple(
            int(destination_plan["component_offsets"][component])
            for component in components
        )
        return selected
    if destination_plan["offset_mode"] == "layout_coordinates":
        selected["component_coordinate_bases"] = tuple(
            tuple(int(value) for value in destination_plan["component_coordinate_bases"][component])
            for component in components
        )
        return selected
    fail(
        "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
        STAGE,
        f"unsupported scalarized destination offset mode {destination_plan['offset_mode']}",
    )


def _local_tensor_access_attrs(
    conversion_input,
    type_layout_program,
    memdesc_value_id,
    tensor_value,
    description,
    op,
):
    memdesc = _memdesc_info(conversion_input, memdesc_value_id, op)
    if memdesc.element_byte_width is None:
        fail(
            "TLXW_OP_LOCAL_MEMORY",
            STAGE,
            "ttg local memory access requires known memdesc element byte width",
            source_op_index=op.index,
            source_value_id=memdesc_value_id,
        )
    if tensor_value.type.element_type != memdesc.element_type:
        fail(
            "TLXW_OP_LOCAL_MEMORY",
            STAGE,
            f"{description} element type must match the memdesc element type",
            source_op_index=op.index,
            source_value_id=tensor_value.value_id,
        )
    tensor_layout = _require_layout(type_layout_program, tensor_value.layout_map_id, op)
    if tensor_layout.kind not in {
            "blocked",
            "linear",
            "generic_linear",
            "slice",
            "amd_mfma",
    }:
        fail(
            "TLXW_OP_LOCAL_MEMORY",
            STAGE,
            f"{description} layout {tensor_layout.kind} is not a supported "
            "distributed local-memory access layout",
            source_op_index=op.index,
            source_value_id=tensor_value.value_id,
        )
    shape = tuple(int(dim) for dim in (memdesc.shape or memdesc.alloc_shape))
    if tuple(int(dim) for dim in tensor_layout.shape) != shape:
        fail(
            "TLXW_OP_LOCAL_MEMORY",
            STAGE,
            f"{description} shape must match the memdesc shape",
            source_op_index=op.index,
            source_value_id=tensor_value.value_id,
        )
    memdesc_layout_id = type_layout_program.values[memdesc_value_id].layout_map_id
    memdesc_layout = (None if memdesc_layout_id is None else type_layout_program.layouts[int(memdesc_layout_id)])
    view = _memdesc_view_info(conversion_input, memdesc_value_id, op)
    lane_width = int(tensor_value.type.lane_width or tensor_layout.lane_width or conversion_input.threads_per_warp)
    warp_count = _layout_warp_count(tensor_layout)
    plan = coordinates.layout_coordinate_plan(
        tensor_layout,
        int(tensor_value.type.component_count),
        int(lane_width),
        int(warp_count),
        op,
        tensor_value.value_id,
    )
    return {
        "component_count":
        int(tensor_value.type.component_count),
        "destination_component_coordinate_bases":
        tuple(tuple(int(value) for value in bases) for bases in plan.component_bases),
        "destination_coordinate_shape":
        tuple(int(dim) for dim in plan.shape),
        "destination_offset_mode":
        "layout_coordinates",
        "destination_workitem_coordinate_coefficients":
        tuple(tuple(int(value) for value in coefficients) for coefficients in plan.workitem_coefficients),
        "element_byte_width":
        int(memdesc.element_byte_width),
        "element_type":
        memdesc.element_type,
        "lane_width":
        int(lane_width),
        **_scalarized_shared_layout_attrs(
            memdesc_layout,
            view.physical_shape,
            memdesc.element_byte_width,
            op,
            logical_origin=view.logical_origin,
        ),
    }


def _buffer_load_result_value_attrs(loaded, has_mask, has_other, access_element_count, element_byte_width):
    if has_mask or has_other:
        return {}
    if loaded.type.element_type != "i8":
        return {}
    packet_width = _vector_packet_width(
        access_element_count,
        element_byte_width,
        int(loaded.type.component_count),
    )
    if packet_width is None:
        return {}
    return {
        "result_packet_width": int(packet_width),
        "result_value_mode": "vector_packets",
    }


def _buffer_load_layout_packet_result_value_attrs(
    conversion_input,
    type_layout_program,
    loaded,
    has_mask,
    has_other,
    access_element_count,
    element_byte_width,
    mask_alignment,
):
    if has_other or loaded.type.representation not in {"simd", "simd_tuple"}:
        return {}
    packet_width = _vector_packet_width(
        access_element_count,
        element_byte_width,
        int(loaded.type.component_count),
    )
    if packet_width is None:
        return {}
    if has_mask and int(mask_alignment) < int(packet_width):
        return {}

    users = [
        candidate
        for candidate in conversion_input.ops
        if int(loaded.value_id) in (int(operand) for operand in candidate.operands)
    ]
    if not users:
        return {}
    for user in users:
        if user.name != "ttg.convert_layout" or len(user.operands) != 1 or len(user.results) != 1:
            return {}
        result = type_layout_program.values[user.results[0]]
        result_layout = _require_layout(type_layout_program, result.layout_map_id, user)
        if result_layout.kind != "dot_operand":
            return {}

    return {
        "result_packet_width": int(packet_width),
        "result_value_mode": "vector_packets",
    }


def _buffer_load_register_vector_result_value_attrs(
    type_layout_program,
    loaded,
    offsets,
    has_other,
    op,
):
    registers = _register_vector_payload_registers_or_none(type_layout_program, loaded, op)
    if registers is None:
        return {}
    scalar_payload_count = int(loaded.type.component_count) * int(registers)
    if int(offsets.type.component_count) != scalar_payload_count:
        return {}
    if has_other:
        fail(
            "TLXW_OP_UNSUPPORTED_BUFFER_LOAD",
            STAGE,
            "register-payload amdg.buffer_load does not support an other operand",
            source_op_index=op.index,
        )
    if loaded.type.element_type not in {"f16", "bf16", "f32"}:
        fail(
            "TLXW_OP_UNSUPPORTED_BUFFER_LOAD",
            STAGE,
            "register-payload amdg.buffer_load requires f16, bf16, or f32 results",
            source_op_index=op.index,
            source_value_id=loaded.value_id,
        )
    return {
        "registers": int(registers),
        "result_packet_width": int(registers),
        "result_value_mode": "register_vector_payload",
    }


def _buffer_load_mma_packet_result_value_attrs(type_layout_program, loaded, has_other, op):
    if loaded.type.representation not in _MMA_PACKET_REPRESENTATIONS:
        return {}
    if has_other:
        fail(
            "TLXW_OP_UNSUPPORTED_BUFFER_LOAD",
            STAGE,
            "MMA packet amdg.buffer_load does not support an other operand",
            source_op_index=op.index,
        )
    if loaded.type.element_type not in {"f16", "bf16", "f32"}:
        fail(
            "TLXW_OP_UNSUPPORTED_BUFFER_LOAD",
            STAGE,
            "MMA packet amdg.buffer_load requires f16, bf16, or f32 results",
            source_op_index=op.index,
            source_value_id=loaded.value_id,
        )
    registers = _mma_packet_registers(type_layout_program, loaded, op)
    return {
        "result_packet_width": int(registers),
        "result_value_mode": "mma_packet_payload",
        "registers": int(registers),
    }


def _vector_packet_width(access_element_count, element_byte_width, component_count):
    access_element_count = int(access_element_count)
    element_byte_width = int(element_byte_width)
    component_count = int(component_count)
    if access_element_count <= 1 or component_count <= 0:
        return None
    packet_width = min(access_element_count, _buffer_max_packet_elements(element_byte_width))
    while packet_width > 1:
        if (component_count % packet_width == 0 and access_element_count % packet_width == 0
                and _buffer_packet_payload_is_legal(packet_width, element_byte_width)):
            return int(packet_width)
        packet_width -= 1
    return None


def _local_load_result_value_attrs(attrs, result):
    if result.type.element_type != "i8":
        return {}
    if int(attrs.get("element_byte_width", 0)) != 1:
        return {}
    if _local_load_transpose_vector_packet_indices(attrs, int(result.type.component_count)) is None:
        return {}
    return {
        "result_packet_width": 4,
        "result_transpose_packet_width": 8,
        "result_value_mode": "transpose_vector_packets",
    }


def _local_load_mma_packet_result_value_attrs(type_layout_program, result, op):
    if result.type.representation not in _MMA_PACKET_REPRESENTATIONS:
        return {}
    if result.type.element_type not in {"f16", "bf16", "f32"}:
        fail(
            "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            "MMA packet ttg.local_load requires f16, bf16, or f32 results",
            source_op_index=op.index,
            source_value_id=result.value_id,
        )
    registers = _mma_packet_registers(type_layout_program, result, op)
    return {
        "registers": int(registers),
        "result_packet_width": int(registers),
        "result_value_mode": "mma_packet_payload",
    }


def _local_load_structural_packet_result_value_attrs(
    conversion_input,
    type_layout_program,
    memdesc_value_id,
    result,
    op,
):
    """Keep a contiguous LDS packet raw through zero-cost tensor views.

    A structural reshape or transpose may regroup scalar bridge components
    into one ordinary SIMD packet consumed by an MMA.  Materializing the
    source components as typed sub-dword values would turn one vector LDS read
    into scalar reads plus register permutations.  Follow only a single-use
    chain of proven structural views, then independently prove that every
    resulting packet is byte-contiguous and naturally aligned in the shared
    allocation for every workitem.  WaveAMD fragment types remain confined to
    the MMA emitter; this contract carries only raw register bits.
    """
    if result.type.representation not in {"simd", "simd_tuple"}:
        return {}

    current = result
    first_alias_plan = None
    final_alias_plan = None
    visited = set()
    while current.type.representation not in _MMA_PACKET_REPRESENTATIONS:
        if int(current.value_id) in visited:
            return {}
        visited.add(int(current.value_id))
        users = tuple(
            candidate
            for candidate in conversion_input.ops
            if int(current.value_id) in tuple(int(value) for value in candidate.operands)
        )
        if len(users) != 1:
            return {}
        user = users[0]
        if len(user.operands) != 1 or len(user.results) != 1:
            return {}
        successor = type_layout_program.values[int(user.results[0])]
        current_layout = _require_layout(
            type_layout_program,
            current.layout_map_id,
            user,
        )
        successor_layout = _require_layout(
            type_layout_program,
            successor.layout_map_id,
            user,
        )
        if user.name in {"tt.reshape", "tt.trans"}:
            alias_plan = layout_remap.structural_view_alias_plan(
                current,
                successor,
                current_layout,
                successor_layout,
                user,
            )
        elif user.name == "ttg.convert_layout":
            alias_plan = layout_remap.redistribution_plan(
                current,
                successor,
                current_layout,
                successor_layout,
                user,
            )
            if alias_plan is None or alias_plan.get("mode") != "alias":
                return {}
        else:
            return {}
        if first_alias_plan is None:
            first_alias_plan = alias_plan
        final_alias_plan = alias_plan
        current = successor

    if first_alias_plan is None or final_alias_plan is None:
        return {}
    source_components = int(result.type.component_count)
    if (
        int(first_alias_plan["source_component_count"]) != source_components
        or int(first_alias_plan["source_packet_width"]) != 1
        or int(first_alias_plan["source_slot_count"]) != source_components
    ):
        return {}
    packet_width = int(final_alias_plan["result_packet_width"])
    packet_count = int(final_alias_plan["result_component_count"])
    if (
        packet_width <= 1
        or packet_count * packet_width != source_components
        or int(final_alias_plan["result_slot_count"]) != source_components
    ):
        return {}

    element_byte_width = conversion_input.value_element_byte_widths.get(
        int(result.value_id)
    )
    if element_byte_width is None:
        return {}
    packet_bits = packet_width * int(element_byte_width) * 8
    element_bit_width = int(element_byte_width) * 8
    if (
        packet_bits <= 0
        or packet_bits > 128
        or packet_bits % 32
        or 32 % element_bit_width
    ):
        return {}

    packet_component_indices = _contiguous_local_load_packet_indices(
        conversion_input,
        type_layout_program,
        memdesc_value_id,
        result,
        packet_width,
        op,
    )
    if packet_component_indices is None:
        return {}
    return {
        "raw_packet_component_indices": tuple(
            int(index) for index in packet_component_indices
        ),
        "result_element_bit_width": int(element_bit_width),
        "result_packet_width": int(packet_width),
        "result_value_mode": "raw_layout_vector_packets",
    }


def _contiguous_local_load_packet_indices(
    conversion_input,
    type_layout_program,
    memdesc_value_id,
    result,
    packet_width,
    op,
):
    """Return packet starts after exhaustively proving LDS byte adjacency."""
    component_count = int(result.type.component_count)
    packet_width = int(packet_width)
    if packet_width <= 1 or component_count % packet_width:
        return None
    memdesc = _memdesc_info(conversion_input, memdesc_value_id, op)
    if memdesc.element_byte_width is None:
        return None
    memdesc_layout_id = type_layout_program.values[memdesc_value_id].layout_map_id
    memdesc_layout = (
        None
        if memdesc_layout_id is None
        else type_layout_program.layouts[int(memdesc_layout_id)]
    )
    result_layout = _require_layout(
        type_layout_program,
        result.layout_map_id,
        op,
    )
    lane_width = int(
        result.type.lane_width
        or result_layout.lane_width
        or conversion_input.threads_per_warp
    )
    warp_count = int(_layout_warp_count(result_layout))
    coordinate_plan = coordinates.layout_coordinate_plan(
        result_layout,
        component_count,
        lane_width,
        warp_count,
        op,
        result.value_id,
    )
    view = _memdesc_view_info(conversion_input, memdesc_value_id, op)
    packet_bytes = packet_width * int(memdesc.element_byte_width)
    packet_starts = tuple(range(0, component_count, packet_width))
    for workitem in range(lane_width * warp_count):
        for packet_start in packet_starts:
            first_byte_offset = None
            for element in range(packet_width):
                component = packet_start + element
                logical_coords = _static_coordinate_plan_coords(
                    coordinate_plan.component_bases[component],
                    coordinate_plan.workitem_coefficients,
                    workitem,
                )
                physical_coords = tuple(
                    int(origin) + int(coord)
                    for origin, coord in zip(
                        view.logical_origin,
                        logical_coords,
                    )
                )
                record = layouts.shared_physical_offset(
                    memdesc_layout,
                    view.physical_shape,
                    physical_coords,
                    int(memdesc.element_byte_width),
                    stage=STAGE,
                    diagnostic="TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
                    source_op_index=op.index,
                    source_value_id=memdesc_value_id,
                )
                byte_offset = int(record.byte_offset)
                if first_byte_offset is None:
                    first_byte_offset = byte_offset
                    if first_byte_offset % packet_bytes:
                        return None
                    continue
                if byte_offset != (
                    first_byte_offset
                    + element * int(memdesc.element_byte_width)
                ):
                    return None
    return packet_starts


def _static_coordinate_plan_coords(
    component_base,
    workitem_coefficients,
    workitem,
):
    coords = [int(value) for value in component_base]
    for bit, coefficients in enumerate(workitem_coefficients):
        if not (int(workitem) & (1 << bit)):
            continue
        for dim, coefficient in enumerate(coefficients):
            coords[dim] ^= int(coefficient)
    return tuple(coords)


def _local_load_transpose_vector_packet_indices(attrs, component_count):
    indices = []
    index = 0
    while index < int(component_count):
        transpose_packet = _local_load_transpose_packet_elements(attrs, index, component_count)
        if transpose_packet != 8:
            return None
        indices.append(index)
        index += int(transpose_packet)
    return tuple(indices)


def _local_load_transpose_packet_elements(attrs, index, component_count):
    if int(index) + 8 > int(component_count):
        return 1
    if attrs.get("destination_offset_mode") != "layout_coordinates":
        return 1
    if attrs.get("element_type") != "i8":
        return 1
    if int(attrs.get("element_byte_width", 0)) != 1:
        return 1
    if int(attrs.get("lane_width", 0)) != 64:
        return 1
    if attrs.get("destination_physical_offset_plan") != "swizzled_xor":
        return 1
    if attrs.get("destination_physical_offset_unit") != "element":
        return 1
    if int(attrs.get("destination_physical_element_byte_width", 0)) != 1:
        return 1
    if tuple(int(value) for value in attrs.get("destination_physical_order", ())) != (0, 1):
        return 1
    if int(attrs.get("destination_physical_swizzled_vec", 0)) != 1:
        return 1
    if int(attrs.get("destination_physical_swizzled_per_phase", 0)) != 1:
        return 1
    if int(attrs.get("destination_physical_swizzled_max_phase", 0)) != 1:
        return 1
    shape = tuple(int(value) for value in attrs["destination_coordinate_shape"])
    if len(shape) != 2:
        return 1
    component_bases = tuple(
        tuple(int(value) for value in bases) for bases in attrs["destination_component_coordinate_bases"])
    if int(index) + 8 > len(component_bases):
        return 1
    workitem_coefficients = tuple(
        tuple(int(value)
              for value in coefficients)
        for coefficients in attrs["destination_workitem_coordinate_coefficients"])
    if any(len(coefficients) != len(shape) for coefficients in workitem_coefficients):
        return 1
    packet_bases = component_bases[int(index):int(index) + 8]
    first_row, first_col = packet_bases[0]
    if int(first_row) % 32 or int(first_col) % 4:
        return 1
    expected = []
    for row_group in range(4):
        for col_group in range(2):
            expected.append((int(first_row) + row_group * 32, int(first_col) + col_group * 4))
    if tuple(expected) != tuple(packet_bases):
        return 1
    if int(first_row) + 96 >= int(shape[0]) or int(first_col) + 4 >= int(shape[1]):
        return 1
    return 8


def _convert_buffer_load(builder, conversion_input, type_layout_program, fact_program, op):
    fields = _buffer_load_fields(op)
    _require_supported_direct_buffer_cache(fields["cache"], op, is_store=False)
    if fields["other_value_id"] is not None and fields["mask_value_id"] is None:
        fail(
            "TLXW_OP_UNSUPPORTED_BUFFER_LOAD",
            STAGE,
            "amdg.buffer_load other operand requires a mask operand",
            source_op_index=op.index,
        )
    range_fact = _pointer_byte_range_fact(fact_program, fields["base_value_id"], op)
    loaded = type_layout_program.values[op.results[0]]
    offsets = type_layout_program.values[fields["offset_value_id"]]
    has_other = fields["other_value_id"] is not None
    fragment_result_value_attrs = _buffer_load_mma_packet_result_value_attrs(
        type_layout_program,
        loaded,
        has_other,
        op,
    )
    register_result_value_attrs = _buffer_load_register_vector_result_value_attrs(
        type_layout_program,
        loaded,
        offsets,
        has_other,
        op,
    )
    result_payload_width = max(
        int(fragment_result_value_attrs.get("registers", 1)),
        int(register_result_value_attrs.get("registers", 1)),
    )
    access_component_count = (
        int(loaded.type.component_count) * result_payload_width
    )
    if int(offsets.type.component_count) != access_component_count:
        fail(
            "TLXW_OP_BUFFER_LOAD",
            STAGE,
            "amdg.buffer_load offsets must match the scalar result payload",
            source_op_index=op.index,
        )
    _require_same_layout_except_element_type(
        type_layout_program,
        offsets,
        loaded,
        "amdg.buffer_load result and offsets",
        op,
    )
    base_target_id = _single_source_target(builder, fields["base_value_id"], op)
    if fields["mask_value_id"] is not None:
        mask = type_layout_program.values[fields["mask_value_id"]]
        if int(mask.type.component_count) != access_component_count:
            fail(
                "TLXW_OP_BUFFER_LOAD",
                STAGE,
                "amdg.buffer_load mask must match the scalar result payload",
                source_op_index=op.index,
            )
        _require_mask_layout_compatible(
            type_layout_program,
            mask,
            loaded,
            "amdg.buffer_load mask",
            op,
        )
    other_target_id = None
    if fields["other_value_id"] is not None:
        other = type_layout_program.values[fields["other_value_id"]]
        if int(other.type.component_count) not in (
                1,
                int(loaded.type.component_count),
        ):
            fail(
                "TLXW_OP_BUFFER_LOAD",
                STAGE,
                "amdg.buffer_load other must be scalar or match result components",
                source_op_index=op.index,
            )
        if other.layout_map_id is not None:
            _require_same_layout_except_element_type(
                type_layout_program,
                other,
                loaded,
                "amdg.buffer_load other and result",
                op,
            )
        other_target_id = _single_source_target(
            builder,
            fields["other_value_id"],
            op,
        )
    element_byte_width = conversion_input.value_element_byte_widths.get(op.results[0])
    if element_byte_width is None:
        fail(
            "TLXW_OP_BUFFER_LOAD",
            STAGE,
            "amdg.buffer_load requires known result element byte width",
            source_op_index=op.index,
            source_value_id=op.results[0],
        )
    source_access_element_count = int(fields["contiguity"] or 1)
    if source_access_element_count <= 0:
        fail(
            "TLXW_OP_BUFFER_LOAD",
            STAGE,
            "amdg.buffer_load contiguity must be positive",
            source_op_index=op.index,
        )
    has_mask = fields["mask_value_id"] is not None
    mask_alignment = 1
    if has_mask:
        mask_alignment = _buffer_mask_alignment(
            conversion_input,
            type_layout_program,
            fact_program,
            fields["mask_value_id"],
            int(offsets.type.component_count),
            int(loaded.type.lane_width or offsets.type.lane_width or 64),
            int(element_byte_width),
            op,
        )
    affine_plan = _buffer_affine_offset_plan(
        conversion_input,
        type_layout_program,
        fact_program,
        fields["offset_value_id"],
        int(offsets.type.component_count),
        int(loaded.type.lane_width or offsets.type.lane_width or 64),
        op,
    )
    inferred_access_element_count = (
        _buffer_load_affine_access_element_count(
            affine_plan,
            int(offsets.type.component_count),
            int(element_byte_width),
        ) if source_access_element_count == 1 else 1)
    result_packet_width = max(
        int(fragment_result_value_attrs.get("result_packet_width", 1)),
        int(register_result_value_attrs.get("result_packet_width", 1)),
    )
    access_element_count = max(
        source_access_element_count,
        inferred_access_element_count,
        result_packet_width,
    )
    access_bytes = int(element_byte_width) * access_element_count
    offset_upper = _buffer_source_offset_upper(
        range_fact.upper,
        access_bytes,
        element_byte_width,
        op,
    )
    offset_no_signed_wrap = (
        _affine_source_offset_no_signed_wrap(
            conversion_input,
            fact_program,
            affine_plan["source_affine"],
            op,
            offset_upper,
        )
        if affine_plan is not None
        else False
    )
    runtime_offset_target_id = _materialize_affine_edge_or_original(
        builder,
        conversion_input,
        type_layout_program,
        fact_program,
        int(fields["offset_value_id"]),
        op,
        no_signed_wrap=bool(offset_no_signed_wrap),
        result_element_type="index",
        value_range=(0, int(offset_upper)),
    )
    mask_source_indices = (
        ()
        if not has_mask
        else tuple(
            component // int(mask_alignment) * int(mask_alignment)
            for component in range(access_component_count)
        )
    )
    runtime_mask_target_id = None
    if has_mask:
        runtime_mask_target_id = _component_remap_edge(
            builder,
            _single_source_target(
                builder,
                int(fields["mask_value_id"]),
                op,
            ),
            mask_source_indices,
            op,
        )
    result_target_ids, result_layout_map_ids = _declare_results(
        builder,
        op,
        type_layout_program,
    )
    result_value_attrs = _buffer_load_result_value_attrs(
        loaded,
        has_mask,
        has_other,
        access_element_count,
        element_byte_width,
    )
    layout_packet_result_value_attrs = _buffer_load_layout_packet_result_value_attrs(
        conversion_input,
        type_layout_program,
        loaded,
        has_mask,
        has_other,
        access_element_count,
        element_byte_width,
        mask_alignment,
    )
    result_value_attrs = {
        **result_value_attrs,
        **register_result_value_attrs,
        **fragment_result_value_attrs,
        **layout_packet_result_value_attrs,
    }
    runtime_operands = [base_target_id, runtime_offset_target_id]
    if runtime_mask_target_id is not None:
        runtime_operands.append(runtime_mask_target_id)
    if other_target_id is not None:
        runtime_operands.append(other_target_id)
    builder.add_op(
        "buffer_load",
        operands=tuple(runtime_operands),
        results=result_target_ids,
        attrs={
            "access_element_count": access_element_count,
            "access_component_count": access_component_count,
            "cache_modifier": int(fields["cache"] or 1),
            "component_count": int(loaded.type.component_count),
            "element_byte_width": int(element_byte_width),
            "element_type": loaded.type.element_type,
            "has_mask": has_mask,
            "has_other": has_other,
            "has_stride_operand": fields["stride_value_id"] is not None,
            "lane_width": int(loaded.type.lane_width or offsets.type.lane_width or 64),
            "mask_alignment": int(mask_alignment),
            "mask_mode": "exec_where" if has_mask else "none",
            "source_access_element_count": int(source_access_element_count),
            "range_bytes": int(range_fact.upper),
            **result_value_attrs,
        },
        fact_ids=(range_fact.fact_id, ),
        fact_target_ids=(base_target_id, ),
        layout_map_ids=result_layout_map_ids,
        source_op_index=op.index,
    )


def _convert_buffer_store(builder, conversion_input, type_layout_program, fact_program, op):
    fields = _buffer_store_fields(op)
    _require_supported_direct_buffer_cache(fields["cache"], op, is_store=True)
    range_fact = _pointer_byte_range_fact(fact_program, fields["base_value_id"], op)
    value = type_layout_program.values[fields["value_value_id"]]
    offsets = type_layout_program.values[fields["offset_value_id"]]
    value_payload_width = 1
    if value.type.representation in _MMA_PACKET_REPRESENTATIONS:
        value_payload_width = _mma_packet_registers(
            type_layout_program,
            value,
            op,
        )
    else:
        register_payload_width = _register_vector_payload_registers_or_none(
            type_layout_program,
            value,
            op,
        )
        if register_payload_width is not None:
            value_payload_width = int(register_payload_width)
    access_component_count = (
        int(value.type.component_count) * int(value_payload_width)
    )
    if int(offsets.type.component_count) != access_component_count:
        fail(
            "TLXW_OP_BUFFER_STORE",
            STAGE,
            "amdg.buffer_store offsets must match the scalar value payload",
            source_op_index=op.index,
        )
    _require_same_layout_except_element_type(
        type_layout_program,
        value,
        offsets,
        "amdg.buffer_store value and offsets",
        op,
    )
    store_component_count = int(value.type.component_count)
    store_lane_width = int(value.type.lane_width or offsets.type.lane_width or 64)
    base_target_id = _single_source_target(builder, fields["base_value_id"], op)
    value_target_id = _single_source_target(
        builder,
        fields["value_value_id"],
        op,
    )
    if fields["mask_value_id"] is not None:
        mask = type_layout_program.values[fields["mask_value_id"]]
        if int(mask.type.component_count) != access_component_count:
            fail(
                "TLXW_OP_BUFFER_STORE",
                STAGE,
                "amdg.buffer_store mask must match the scalar value payload",
                source_op_index=op.index,
            )
        _require_mask_layout_compatible(
            type_layout_program,
            mask,
            offsets,
            "amdg.buffer_store mask",
            op,
        )
    element_byte_width = conversion_input.value_element_byte_widths.get(fields["value_value_id"])
    if element_byte_width is None:
        fail(
            "TLXW_OP_BUFFER_STORE",
            STAGE,
            "amdg.buffer_store requires known value element byte width",
            source_op_index=op.index,
            source_value_id=fields["value_value_id"],
        )
    source_access_element_count = int(fields["contiguity"] or 1)
    if source_access_element_count <= 0:
        fail(
            "TLXW_OP_BUFFER_STORE",
            STAGE,
            "amdg.buffer_store contiguity must be positive",
            source_op_index=op.index,
        )
    has_mask = fields["mask_value_id"] is not None
    affine_plan = _buffer_affine_offset_plan(
        conversion_input,
        type_layout_program,
        fact_program,
        fields["offset_value_id"],
        int(offsets.type.component_count),
        int(store_lane_width),
        op,
    )
    inferred_access_element_count = (
        _buffer_affine_access_element_count(
            affine_plan,
            access_component_count,
            int(element_byte_width),
        ) if source_access_element_count == 1 else 1)
    access_element_count = max(
        source_access_element_count,
        inferred_access_element_count,
    )
    access_bytes = int(element_byte_width) * access_element_count
    offset_upper = _buffer_source_offset_upper(
        range_fact.upper,
        access_bytes,
        element_byte_width,
        op,
    )
    mask_alignment = 1
    if has_mask:
        mask_alignment = _buffer_mask_alignment(
            conversion_input,
            type_layout_program,
            fact_program,
            fields["mask_value_id"],
            int(offsets.type.component_count),
            int(offsets.type.lane_width or value.type.lane_width or 64),
            int(element_byte_width),
            op,
        )
    offset_no_signed_wrap = (
        _affine_source_offset_no_signed_wrap(
            conversion_input,
            fact_program,
            affine_plan["source_affine"],
            op,
            offset_upper,
        )
        if affine_plan is not None
        else False
    )
    runtime_offset_target_id = _materialize_affine_edge_or_original(
        builder,
        conversion_input,
        type_layout_program,
        fact_program,
        int(fields["offset_value_id"]),
        op,
        no_signed_wrap=bool(offset_no_signed_wrap),
        result_element_type="index",
        value_range=(0, int(offset_upper)),
    )
    mask_source_indices = (
        ()
        if not has_mask
        else tuple(
            component // int(mask_alignment) * int(mask_alignment)
            for component in range(access_component_count)
        )
    )
    runtime_mask_target_id = None
    if has_mask:
        runtime_mask_target_id = _component_remap_edge(
            builder,
            _single_source_target(
                builder,
                int(fields["mask_value_id"]),
                op,
            ),
            mask_source_indices,
            op,
        )
    runtime_operands = [
        value_target_id,
        base_target_id,
        runtime_offset_target_id,
    ]
    if runtime_mask_target_id is not None:
        runtime_operands.append(runtime_mask_target_id)
    builder.add_op(
        "buffer_store",
        operands=tuple(runtime_operands),
        attrs={
            "access_element_count": access_element_count,
            "access_component_count": access_component_count,
            "cache_modifier": int(fields["cache"] or 1),
            "component_count": int(store_component_count),
            "element_byte_width": int(element_byte_width),
            "element_type": value.type.element_type,
            "has_boundary_check_operand": fields["boundary_check_value_id"] is not None,
            "has_mask": has_mask,
            "lane_width": int(store_lane_width),
            "mask_alignment": int(mask_alignment),
            "mask_mode": "exec_where" if has_mask else "none",
            "value_payload_width": int(value_payload_width),
            "range_bytes": int(range_fact.upper),
        },
        fact_ids=(range_fact.fact_id, ),
        fact_target_ids=(base_target_id, ),
        source_op_index=op.index,
    )


def _convert_load(builder, conversion_input, type_layout_program, op):
    del conversion_input
    fields = _load_fields(op)
    _require_default_tt_memory_attrs(op)
    pointer = type_layout_program.values[fields["pointer_value_id"]]
    loaded = type_layout_program.values[op.results[0]]
    if pointer.type.representation not in {"per_lane_pointer", "pointer_tuple"}:
        fail(
            "TLXW_OP_LOAD",
            STAGE,
            "tt.load requires a tensor pointer operand",
            source_op_index=op.index,
            source_value_id=fields["pointer_value_id"],
        )
    if loaded.type.representation not in {"simd", "simd_tuple"}:
        fail(
            "TLXW_OP_LOAD",
            STAGE,
            "tt.load requires a tensor result",
            source_op_index=op.index,
            source_value_id=op.results[0],
        )
    if pointer.type.element_type != loaded.type.element_type:
        fail(
            "TLXW_OP_LOAD",
            STAGE,
            "tt.load pointer/result element types must match",
            source_op_index=op.index,
            source_value_id=op.results[0],
        )
    component_count = int(loaded.type.component_count)
    if int(pointer.type.component_count) != component_count:
        fail(
            "TLXW_OP_LOAD",
            STAGE,
            "tt.load pointer/result component counts must match",
            source_op_index=op.index,
        )
    _require_same_layout_except_element_type(
        type_layout_program,
        pointer,
        loaded,
        "tt.load pointer and result",
        op,
    )
    operands = [_single_source_target(builder, fields["pointer_value_id"], op)]
    if fields["mask_value_id"] is not None:
        mask = type_layout_program.values[fields["mask_value_id"]]
        if int(mask.type.component_count) not in (1, component_count):
            fail(
                "TLXW_OP_LOAD",
                STAGE,
                "tt.load mask must be scalar or match result components",
                source_op_index=op.index,
                source_value_id=fields["mask_value_id"],
            )
        _require_mask_layout_compatible(
            type_layout_program,
            mask,
            loaded,
            "tt.load mask",
            op,
        )
        operands.append(_single_source_target(builder, fields["mask_value_id"], op))
    if fields["other_value_id"] is not None:
        if fields["mask_value_id"] is None:
            fail(
                "TLXW_OP_LOAD",
                STAGE,
                "tt.load other operand requires a mask operand",
                source_op_index=op.index,
                source_value_id=fields["other_value_id"],
            )
        other = type_layout_program.values[fields["other_value_id"]]
        if other.type.representation not in {"scalar", "simd", "simd_tuple"}:
            fail(
                "TLXW_OP_LOAD",
                STAGE,
                "tt.load other requires a scalar or tensor value operand",
                source_op_index=op.index,
                source_value_id=fields["other_value_id"],
            )
        if other.type.element_type != loaded.type.element_type:
            fail(
                "TLXW_OP_LOAD",
                STAGE,
                "tt.load other/result element types must match",
                source_op_index=op.index,
                source_value_id=fields["other_value_id"],
            )
        if int(other.type.component_count) not in (1, component_count):
            fail(
                "TLXW_OP_LOAD",
                STAGE,
                "tt.load other must be scalar or match result components",
                source_op_index=op.index,
                source_value_id=fields["other_value_id"],
            )
        if other.layout_map_id is not None:
            _require_same_layout_except_element_type(
                type_layout_program,
                other,
                loaded,
                "tt.load other and result",
                op,
            )
        operands.append(_single_source_target(builder, fields["other_value_id"], op))
    result_target_ids, result_layout_map_ids = _declare_results(
        builder,
        op,
        type_layout_program,
    )
    builder.add_op(
        "load",
        operands=tuple(operands),
        results=result_target_ids,
        attrs={
            "component_count": component_count,
            "element_type": loaded.type.element_type,
            "has_mask": fields["mask_value_id"] is not None,
            "has_other": fields["other_value_id"] is not None,
            "lane_width": int(loaded.type.lane_width or pointer.type.lane_width or 64),
            "mask_mode": "exec_where" if fields["mask_value_id"] is not None else "none",
        },
        layout_map_ids=result_layout_map_ids,
        source_op_index=op.index,
    )


def _convert_store(builder, conversion_input, type_layout_program, op):
    del conversion_input
    fields = _store_fields(op)
    _require_default_tt_memory_attrs(op)
    pointer = type_layout_program.values[fields["pointer_value_id"]]
    value = type_layout_program.values[fields["value_value_id"]]
    if pointer.type.representation not in {"per_lane_pointer", "pointer_tuple"}:
        fail(
            "TLXW_OP_STORE",
            STAGE,
            "tt.store requires a tensor pointer operand",
            source_op_index=op.index,
            source_value_id=fields["pointer_value_id"],
        )
    if value.type.representation not in {"scalar", "simd", "simd_tuple"}:
        fail(
            "TLXW_OP_STORE",
            STAGE,
            "tt.store requires a scalar or tensor value operand",
            source_op_index=op.index,
            source_value_id=fields["value_value_id"],
        )
    if pointer.type.element_type != value.type.element_type:
        fail(
            "TLXW_OP_STORE",
            STAGE,
            "tt.store pointer/value element types must match",
            source_op_index=op.index,
            source_value_id=fields["value_value_id"],
        )
    component_count = int(pointer.type.component_count)
    if int(value.type.component_count) not in (1, component_count):
        fail(
            "TLXW_OP_STORE",
            STAGE,
            "tt.store value must be scalar or match pointer components",
            source_op_index=op.index,
            source_value_id=fields["value_value_id"],
        )
    if value.layout_map_id is not None:
        _require_same_layout_except_element_type(
            type_layout_program,
            value,
            pointer,
            "tt.store value and pointer",
            op,
        )
    operands = [
        _single_source_target(builder, fields["pointer_value_id"], op),
        _single_source_target(builder, fields["value_value_id"], op),
    ]
    if fields["mask_value_id"] is not None:
        mask = type_layout_program.values[fields["mask_value_id"]]
        if int(mask.type.component_count) not in (1, component_count):
            fail(
                "TLXW_OP_STORE",
                STAGE,
                "tt.store mask must be scalar or match pointer components",
                source_op_index=op.index,
                source_value_id=fields["mask_value_id"],
            )
        _require_mask_layout_compatible(
            type_layout_program,
            mask,
            pointer,
            "tt.store mask",
            op,
        )
        operands.append(_single_source_target(builder, fields["mask_value_id"], op))
    builder.add_op(
        "store",
        operands=tuple(operands),
        attrs={
            "component_count": component_count,
            "element_type": pointer.type.element_type,
            "has_mask": fields["mask_value_id"] is not None,
            "lane_width": int(pointer.type.lane_width or value.type.lane_width or 64),
            "mask_mode": "exec_where" if fields["mask_value_id"] is not None else "none",
        },
        source_op_index=op.index,
    )


def _declare_lds_completion(
    builder,
    op,
    dominating_wait_value_ids,
):
    if not dominating_wait_value_ids:
        return ()
    completion_target_id = _declare_protocol_token(
        builder,
        event_domain=target_ir.EVENT_DOMAIN_LDS_COMPLETION,
        debug_name=f"lds_completion_{op.index}",
    )
    builder.append_protocol_frontier(
        dominating_wait_value_ids,
        completion_target_id,
    )
    return (completion_target_id, )


def _convert_local_load(builder, conversion_input, type_layout_program, op):
    if len(op.operands) not in {1, 2} or len(op.results) != 1:
        fail(
            "TLXW_OP_LOCAL_LOAD",
            STAGE,
            "ttg.local_load requires one memdesc operand, an optional token, and one result",
            source_op_index=op.index,
        )
    result_value_id = op.results[0]
    result = type_layout_program.values[result_value_id]
    result_layout = (None if result.layout_map_id is None else type_layout_program.layouts[int(result.layout_map_id)])
    memdesc_value_id = op.operands[0]
    memdesc = _memdesc_info(conversion_input, memdesc_value_id, op)
    token_value_id = None if len(op.operands) == 1 else op.operands[1]
    if token_value_id is not None and type_layout_program.values[token_value_id].type.representation != "token":
        fail(
            "TLXW_OP_LOCAL_LOAD",
            STAGE,
            "ttg.local_load optional second operand must be an async token",
            source_op_index=op.index,
            source_value_id=token_value_id,
        )
    target_operands = [_single_source_target(builder, memdesc_value_id, op)]
    if token_value_id is not None:
        target_operands.append(_single_source_target(builder, token_value_id, op))
    dominating_wait_value_ids = (
        conversion_input.async_protocol_dependency_value_ids_by_op.get(
            op.index,
            (),
        )
    )
    readiness_target_ids = tuple(dict.fromkeys(
        _single_source_target(builder, source_value_id, op)
        for source_value_id in dominating_wait_value_ids
    ))
    target_operands.extend(readiness_target_ids)
    target_operands = tuple(dict.fromkeys(target_operands))
    # A dominating wait is a structural dependency for every following DS
    # read, but it does not by itself make an ordinary local load "relaxed".
    # Only the source load's explicit protocol marker permits the emitter to
    # bypass synchronous LDS access state (for example a preceding local_store).
    synced_via_async_wait = bool(
        token_value_id is not None
        or _attr_bool(op.attrs.get("ttg.amdg.syncedViaAsyncWait"))
    )
    if result_layout is None:
        fail(
            "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            "ttg.local_load requires a structural result layout",
            source_op_index=op.index,
            source_value_id=result_value_id,
        )
    if result_layout.kind != "dot_operand":
        result_target_ids, result_layout_map_ids = _declare_results(
            builder,
            op,
            type_layout_program,
        )
        completion_target_ids = _declare_lds_completion(
            builder,
            op,
            dominating_wait_value_ids,
        )
        attrs = _local_tensor_access_attrs(
            conversion_input,
            type_layout_program,
            memdesc_value_id,
            result,
            "ttg.local_load result",
            op,
        )
        attrs["synced_via_async_wait"] = synced_via_async_wait
        attrs["readiness_dependency_count"] = len(readiness_target_ids)
        attrs["protocol_tracked"] = bool(completion_target_ids)
        attrs["data_result_count"] = len(result_target_ids)
        attrs["completion_result_count"] = len(completion_target_ids)
        attrs.update(_local_load_result_value_attrs(attrs, result))
        attrs.update(
            _local_load_structural_packet_result_value_attrs(
                conversion_input,
                type_layout_program,
                memdesc_value_id,
                result,
                op,
            )
        )
        attrs.update(_local_load_mma_packet_result_value_attrs(type_layout_program, result, op))
        builder.add_op(
            "local_load",
            operands=target_operands,
            results=(*result_target_ids, *completion_target_ids),
            attrs=attrs,
            layout_map_ids=result_layout_map_ids,
            source_op_index=op.index,
        )
        return
    registers = _fragment_registers(memdesc.element_type, result_layout, op)
    parent = result_layout.properties.get("parent_properties", {})
    instr_shape = tuple(parent.get("instr_shape", ()))
    fragment_rows, fragment_columns = _operand_fragment_shape(instr_shape, op)
    result_target_ids, result_layout_map_ids = _declare_results(
        builder,
        op,
        type_layout_program,
    )
    completion_target_ids = _declare_lds_completion(
        builder,
        op,
        dominating_wait_value_ids,
    )
    load_plan = _fragment_local_load_plan(
        conversion_input,
        type_layout_program,
        memdesc_value_id,
        result_layout,
        int(result.type.component_count),
        registers,
        op,
    )
    builder.add_op(
        "local_load_mma_payload",
        operands=target_operands,
        results=(*result_target_ids, *completion_target_ids),
        attrs={
            "columns": int(fragment_columns),
            "component_count": int(result.type.component_count),
            "element_type": memdesc.element_type,
            "lane_width": int(result.type.lane_width or 64),
            "registers": int(registers),
            "role": int(result_layout.properties["op_idx"]),
            "rows": int(fragment_rows),
            "synced_via_async_wait": synced_via_async_wait,
            "readiness_dependency_count": len(readiness_target_ids),
            "protocol_tracked": bool(completion_target_ids),
            "data_result_count": len(result_target_ids),
            "completion_result_count": len(completion_target_ids),
            **load_plan,
        },
        layout_map_ids=result_layout_map_ids,
        source_op_index=op.index,
    )


def _convert_local_store(builder, conversion_input, type_layout_program, op):
    if len(op.operands) != 2 or op.results:
        fail(
            "TLXW_OP_LOCAL_STORE",
            STAGE,
            "ttg.local_store requires value and memdesc operands and no results",
            source_op_index=op.index,
        )
    value_id = op.operands[0]
    memdesc_value_id = op.operands[1]
    value = type_layout_program.values[value_id]
    if value.type.representation not in {
            "simd",
            "simd_tuple",
            "mask",
            "mask_tuple",
            *_MMA_PACKET_REPRESENTATIONS,
    }:
        fail(
            "TLXW_OP_LOCAL_STORE",
            STAGE,
            f"ttg.local_store cannot store {value.type.representation} values",
            source_op_index=op.index,
            source_value_id=value_id,
        )
    dominating_wait_value_ids = (
        conversion_input.async_protocol_dependency_value_ids_by_op.get(
            op.index,
            (),
        )
    )
    readiness_target_ids = tuple(dict.fromkeys(
        _single_source_target(builder, source_value_id, op)
        for source_value_id in dominating_wait_value_ids
    ))
    memdesc_target_id = _single_source_target(builder, memdesc_value_id, op)
    completion_target_ids = _declare_lds_completion(
        builder,
        op,
        dominating_wait_value_ids,
    )
    attrs = _local_tensor_access_attrs(
        conversion_input,
        type_layout_program,
        memdesc_value_id,
        value,
        "ttg.local_store value",
        op,
    )
    attrs["readiness_dependency_count"] = len(readiness_target_ids)
    attrs["protocol_tracked"] = bool(completion_target_ids)
    attrs["data_result_count"] = 0
    attrs["completion_result_count"] = len(completion_target_ids)
    builder.add_op(
        "local_store",
        operands=(
            _single_source_target(builder, value_id, op),
            memdesc_target_id,
            *readiness_target_ids,
        ),
        results=completion_target_ids,
        attrs=attrs,
        source_op_index=op.index,
    )


def _convert_mma_packet_constant(builder, type_layout_program, op):
    if len(op.results) != 1:
        fail(
            "TLXW_OP_MMA_PACKET_CONSTANT",
            STAGE,
            "MMA packet constants must have one result",
            source_op_index=op.index,
        )
    result = type_layout_program.values[op.results[0]]
    result_layout = type_layout_program.layouts[int(result.layout_map_id)]
    if result_layout.kind != "amd_mfma":
        fail(
            "TLXW_OP_MMA_PACKET_CONSTANT",
            STAGE,
            "only amd_mfma accumulator constants are converted as MMA packets",
            source_op_index=op.index,
            source_value_id=op.results[0],
        )
    value = _constant_literal(
        op.attrs.get("value"),
        source_op_index=op.index,
        element_type=result.type.element_type,
    )
    result_target_ids, result_layout_map_ids = _declare_results(
        builder,
        op,
        type_layout_program,
    )
    registers = _acc_fragment_registers(result_layout, op)
    instr_shape = tuple(result_layout.properties.get("instr_shape", ()))
    fragment_rows, fragment_columns = _acc_fragment_shape(instr_shape, op)
    builder.add_op(
        "mma_packet_constant",
        results=result_target_ids,
        attrs={
            "columns": int(fragment_columns),
            "component_count": int(result.type.component_count),
            "element_type": result_layout.element_type,
            "value": value,
            "lane_width": int(result.type.lane_width or 64),
            "registers": int(registers),
            "role": 2,
            "rows": int(fragment_rows),
        },
        layout_map_ids=result_layout_map_ids,
        source_op_index=op.index,
    )


def _convert_dot(builder, conversion_input, type_layout_program, op):
    if len(op.operands) != 3 or len(op.results) != 1:
        fail(
            "TLXW_OP_DOT",
            STAGE,
            "tt.dot requires lhs, rhs, accumulator, and one result",
            source_op_index=op.index,
        )
    lhs = type_layout_program.values[op.operands[0]]
    rhs = type_layout_program.values[op.operands[1]]
    acc = type_layout_program.values[op.operands[2]]
    result = type_layout_program.values[op.results[0]]
    result_layout = type_layout_program.layouts[int(result.layout_map_id)]
    if result_layout.kind != "amd_mfma":
        fail(
            "TLXW_OP_DOT",
            STAGE,
            "tt.dot result must use an amd_mfma layout",
            source_op_index=op.index,
            source_value_id=op.results[0],
        )
    instr_shape = tuple(result_layout.properties.get("instr_shape", ()))
    if instr_shape not in {(16, 16, 32), (32, 32, 16)}:
        fail(
            "TLXW_OP_DOT",
            STAGE,
            f"unsupported MFMA instruction shape {instr_shape}",
            source_op_index=op.index,
            source_value_id=op.results[0],
        )
    if lhs.type.element_type != rhs.type.element_type:
        fail(
            "TLXW_OP_DOT",
            STAGE,
            "tt.dot lhs/rhs element types must match",
            source_op_index=op.index,
        )
    kind = _mma_kind(lhs.type.element_type, instr_shape, op)
    is_transposed = bool(result_layout.properties.get("is_transposed", False))
    if is_transposed and int(instr_shape[0]) != int(instr_shape[1]):
        fail(
            "TLXW_OP_DOT",
            STAGE,
            "transposed MFMA dot lowering requires a symmetric instruction "
            f"shape, got {instr_shape}",
            source_op_index=op.index,
            source_value_id=op.results[0],
        )
    warps_per_cta = tuple(result_layout.properties.get("warps_per_cta", ()))
    m_tiles, n_tiles = _mfma_per_wave_tiles(result_layout, instr_shape, warps_per_cta, op)
    acc_layout = _require_layout(type_layout_program, acc.layout_map_id, op)
    if not _same_layout_alias(acc, result, acc_layout, result_layout):
        fail(
            "TLXW_OP_DOT",
            STAGE,
            "tt.dot accumulator layout must match the result layout",
            source_op_index=op.index,
            source_value_id=op.operands[2],
        )
    lhs_layout = _require_layout(type_layout_program, lhs.layout_map_id, op)
    rhs_layout = _require_layout(type_layout_program, rhs.layout_map_id, op)
    _require_dot_operand_layout(lhs_layout, 0, op)
    _require_dot_operand_layout(rhs_layout, 1, op)
    _require_dot_operand_parent_layout(lhs_layout, result_layout, 0, op)
    _require_dot_operand_parent_layout(rhs_layout, result_layout, 1, op)
    lhs_k_tiles = _dot_operand_k_tiles(lhs_layout, instr_shape, op)
    rhs_k_tiles = _dot_operand_k_tiles(rhs_layout, instr_shape, op)
    if lhs_k_tiles != rhs_k_tiles:
        fail(
            "TLXW_OP_DOT",
            STAGE,
            "tt.dot lhs/rhs K tile counts do not match",
            source_op_index=op.index,
        )
    k_tiles = lhs_k_tiles
    if (int(lhs.type.component_count) != m_tiles * k_tiles or int(rhs.type.component_count) != n_tiles * k_tiles):
        fail(
            "TLXW_OP_DOT",
            STAGE,
            "tt.dot operand fragment component counts do not match "
            "the result MFMA tile grid",
            source_op_index=op.index,
        )
    if int(acc.type.component_count) != m_tiles * n_tiles:
        fail(
            "TLXW_OP_DOT",
            STAGE,
            "tt.dot accumulator component count does not match "
            "the result MFMA tile grid",
            source_op_index=op.index,
        )
    operand_rows, operand_columns = _operand_fragment_shape(instr_shape, op)
    acc_rows, acc_columns = _acc_fragment_shape(instr_shape, op)
    lhs_registers = _fragment_registers(lhs.type.element_type, lhs_layout, op)
    rhs_registers = _fragment_registers(rhs.type.element_type, rhs_layout, op)
    acc_registers = _acc_fragment_registers(result_layout, op)
    lane_width = int(result.type.lane_width or lhs.type.lane_width or rhs.type.lane_width or acc.type.lane_width or 64)
    result_target_ids, result_layout_map_ids = _declare_results(
        builder,
        op,
        type_layout_program,
    )
    mma_operand_target_ids = list(_operand_target_ids(builder, op))
    builder.add_op(
        "mma",
        operands=tuple(mma_operand_target_ids),
        results=result_target_ids,
        attrs={
            "acc_columns": int(acc_columns),
            "acc_element_type": acc.type.element_type,
            "acc_registers": int(acc_registers),
            "acc_role": 2,
            "acc_rows": int(acc_rows),
            "kind": kind,
            "k_tiles": int(k_tiles),
            "lane_width": int(lane_width),
            "lhs_columns": int(operand_columns),
            "lhs_element_type": lhs.type.element_type,
            "lhs_registers": int(lhs_registers),
            "lhs_role": 0,
            "lhs_rows": int(operand_rows),
            "m_tiles": int(m_tiles),
            "n_tiles": int(n_tiles),
            "rhs_columns": int(operand_columns),
            "rhs_element_type": rhs.type.element_type,
            "rhs_registers": int(rhs_registers),
            "rhs_role": 1,
            "rhs_rows": int(operand_rows),
            "swap_operands_for_transposed_result": bool(is_transposed),
        },
        layout_map_ids=result_layout_map_ids,
        source_op_index=op.index,
    )


def _convert_dot_scaled(builder, conversion_input, type_layout_program, op):
    if len(op.results) != 1 or len(op.operands) not in (3, 5):
        fail(
            "TLXW_OP_DOT_SCALED",
            STAGE,
            "tt.dot_scaled requires lhs, rhs, accumulator, optional lhs/rhs "
            "scales, and one result",
            source_op_index=op.index,
        )
    lhs = type_layout_program.values[op.operands[0]]
    rhs = type_layout_program.values[op.operands[1]]
    acc = type_layout_program.values[op.operands[2]]
    lhs_scale = type_layout_program.values[op.operands[3]] if len(op.operands) == 5 else None
    rhs_scale = type_layout_program.values[op.operands[4]] if len(op.operands) == 5 else None
    result = type_layout_program.values[op.results[0]]
    result_layout = type_layout_program.layouts[int(result.layout_map_id)]
    if result_layout.kind != "amd_mfma":
        fail(
            "TLXW_OP_DOT_SCALED",
            STAGE,
            "tt.dot_scaled result must use an amd_mfma layout",
            source_op_index=op.index,
            source_value_id=op.results[0],
        )
    instr_shape = tuple(result_layout.properties.get("instr_shape", ()))
    if instr_shape != (16, 16, 128):
        fail(
            "TLXW_OP_DOT_SCALED",
            STAGE,
            f"unsupported scaled MFMA instruction shape {instr_shape}",
            source_op_index=op.index,
            source_value_id=op.results[0],
        )
    a_elem_type = _scale_dot_elem_type(op.attrs.get("a_elem_type"), op)
    b_elem_type = _scale_dot_elem_type(op.attrs.get("b_elem_type"), op)
    kind = _scaled_mma_kind(a_elem_type, b_elem_type, instr_shape, op)
    if (lhs_scale is None) != (rhs_scale is None):
        fail(
            "TLXW_OP_DOT_SCALED",
            STAGE,
            "single-scale tt.dot_scaled lowering is not supported; upstream "
            "must materialize the missing constant scale",
            source_op_index=op.index,
        )
    is_transposed = bool(result_layout.properties.get("is_transposed", False))
    if is_transposed and int(instr_shape[0]) != int(instr_shape[1]):
        fail(
            "TLXW_OP_DOT_SCALED",
            STAGE,
            "transposed scaled MFMA lowering requires a symmetric instruction "
            f"shape, got {instr_shape}",
            source_op_index=op.index,
            source_value_id=op.results[0],
        )
    warps_per_cta = tuple(result_layout.properties.get("warps_per_cta", ()))
    m_tiles, n_tiles = _mfma_per_wave_tiles(result_layout, instr_shape, warps_per_cta, op)
    acc_layout = _require_layout(type_layout_program, acc.layout_map_id, op)
    if not _same_layout_alias(acc, result, acc_layout, result_layout):
        fail(
            "TLXW_OP_DOT_SCALED",
            STAGE,
            "tt.dot_scaled accumulator layout must match the result layout",
            source_op_index=op.index,
            source_value_id=op.operands[2],
        )
    lhs_layout = _require_layout(type_layout_program, lhs.layout_map_id, op)
    rhs_layout = _require_layout(type_layout_program, rhs.layout_map_id, op)
    _require_dot_operand_layout(lhs_layout, 0, op)
    _require_dot_operand_layout(rhs_layout, 1, op)
    _require_dot_operand_parent_layout(lhs_layout, result_layout, 0, op)
    _require_dot_operand_parent_layout(rhs_layout, result_layout, 1, op)
    lhs_k_tiles = _dot_operand_k_tiles(lhs_layout, instr_shape, op)
    rhs_k_tiles = _dot_operand_k_tiles(rhs_layout, instr_shape, op)
    if lhs_k_tiles != rhs_k_tiles:
        fail(
            "TLXW_OP_DOT_SCALED",
            STAGE,
            "tt.dot_scaled lhs/rhs K tile counts do not match",
            source_op_index=op.index,
        )
    k_tiles = lhs_k_tiles
    if (int(lhs.type.component_count) != m_tiles * k_tiles or int(rhs.type.component_count) != n_tiles * k_tiles):
        fail(
            "TLXW_OP_DOT_SCALED",
            STAGE,
            "tt.dot_scaled operand fragment component counts do not match "
            "the result MFMA tile grid",
            source_op_index=op.index,
        )
    if int(acc.type.component_count) != m_tiles * n_tiles:
        fail(
            "TLXW_OP_DOT_SCALED",
            STAGE,
            "tt.dot_scaled accumulator component count does not match "
            "the result MFMA tile grid",
            source_op_index=op.index,
        )
    if lhs.type.element_type != "i8" or rhs.type.element_type != "i8":
        fail(
            "TLXW_OP_DOT_SCALED",
            STAGE,
            "native scaled MFMA lowering expects packed i8 dot operands",
            source_op_index=op.index,
        )
    scale_attrs = {"has_scales": lhs_scale is not None}
    scale_operands = ()
    if lhs_scale is not None and rhs_scale is not None:
        _require_scaled_mma_scale(type_layout_program, lhs_scale, 0, op)
        _require_scaled_mma_scale(type_layout_program, rhs_scale, 1, op)
        scale_attrs.update(
            _scaled_mma_scale_pack_attrs(
                m_tiles,
                n_tiles,
                k_tiles,
                int(lhs_scale.type.component_count),
                int(rhs_scale.type.component_count),
                op,
            ))
        scale_operands = (
            _single_source_target(builder, op.operands[3], op),
            _single_source_target(builder, op.operands[4], op),
        )
    operand_rows, operand_columns = _operand_fragment_shape(instr_shape, op)
    acc_rows, acc_columns = _acc_fragment_shape(instr_shape, op)
    lhs_registers = _fragment_registers(lhs.type.element_type, lhs_layout, op)
    rhs_registers = _fragment_registers(rhs.type.element_type, rhs_layout, op)
    acc_registers = _acc_fragment_registers(result_layout, op)
    lane_width = int(result.type.lane_width or lhs.type.lane_width or rhs.type.lane_width or acc.type.lane_width or 64)
    result_target_ids, result_layout_map_ids = _declare_results(
        builder,
        op,
        type_layout_program,
    )
    builder.add_op(
        "mma_scaled",
        operands=(
            _single_source_target(builder, op.operands[0], op),
            _single_source_target(builder, op.operands[1], op),
            _single_source_target(builder, op.operands[2], op),
            *scale_operands,
        ),
        results=result_target_ids,
        attrs={
            "acc_columns": int(acc_columns),
            "acc_element_type": acc.type.element_type,
            "acc_registers": int(acc_registers),
            "acc_role": 2,
            "acc_rows": int(acc_rows),
            "kind": kind,
            "k_tiles": int(k_tiles),
            "lane_width": int(lane_width),
            "lhs_columns": int(operand_columns),
            "lhs_element_type": lhs.type.element_type,
            "lhs_registers": int(lhs_registers),
            "lhs_role": 0,
            "lhs_rows": int(operand_rows),
            "m_tiles": int(m_tiles),
            "n_tiles": int(n_tiles),
            "rhs_columns": int(operand_columns),
            "rhs_element_type": rhs.type.element_type,
            "rhs_registers": int(rhs_registers),
            "rhs_role": 1,
            "rhs_rows": int(operand_rows),
            "swap_operands_for_transposed_result": bool(is_transposed),
            **scale_attrs,
        },
        layout_map_ids=result_layout_map_ids,
        source_op_index=op.index,
    )


def _convert_mma_packet_truncf(builder, type_layout_program, op):
    if len(op.operands) != 1 or len(op.results) != 1:
        fail(
            "TLXW_OP_MMA_PACKET_TRUNCF",
            STAGE,
            "MMA packet arith.truncf requires one operand and one result",
            source_op_index=op.index,
        )
    operand = type_layout_program.values[op.operands[0]]
    result = type_layout_program.values[op.results[0]]
    if operand.type.element_type != "f32" or result.type.element_type not in {"f16", "bf16"}:
        fail(
            "TLXW_OP_MMA_PACKET_TRUNCF",
            STAGE,
            "only f32 to f16/bf16 MMA packet truncf is converted yet",
            source_op_index=op.index,
        )
    if int(operand.type.component_count) != int(result.type.component_count):
        fail(
            "TLXW_OP_MMA_PACKET_TRUNCF",
            STAGE,
            "MMA packet truncf component counts must match",
            source_op_index=op.index,
        )
    operand_layout = type_layout_program.layouts[int(operand.layout_map_id)]
    _require_same_layout_except_element_type(
        type_layout_program,
        operand,
        result,
        "MMA packet truncf operand and result",
        op,
    )
    registers = _acc_fragment_registers(operand_layout, op)
    result_target_ids, result_layout_map_ids = _declare_results(
        builder,
        op,
        type_layout_program,
    )
    builder.add_op(
        "mma_packet_truncf",
        operands=_operand_target_ids(builder, op),
        results=result_target_ids,
        attrs={
            "component_count": int(result.type.component_count),
            "lane_width": int(result.type.lane_width or 64),
            "registers": int(registers),
            "result_element_type": result.type.element_type,
        },
        layout_map_ids=result_layout_map_ids,
        source_op_index=op.index,
    )


def _convert_layout(builder, conversion_input, type_layout_program, op):
    if len(op.operands) != 1 or len(op.results) != 1:
        fail(
            "TLXW_OP_CONVERT_LAYOUT",
            STAGE,
            "ttg.convert_layout requires one operand and one result",
            source_op_index=op.index,
        )
    operand = type_layout_program.values[op.operands[0]]
    result = type_layout_program.values[op.results[0]]
    operand_layout = (None
                      if operand.layout_map_id is None else type_layout_program.layouts[int(operand.layout_map_id)])
    result_layout = (None if result.layout_map_id is None else type_layout_program.layouts[int(result.layout_map_id)])
    result_target_ids, result_layout_map_ids = _declare_results(
        builder,
        op,
        type_layout_program,
    )
    same_layout = _same_layout_alias(operand, result, operand_layout, result_layout)
    if same_layout:
        attrs = {
            "fact_policy": "preserve_equivalent",
            "group_size": 1,
            "mode": "alias",
            "result_component_count": int(result.type.component_count),
        }
    else:
        redistribution_remap = layout_remap.redistribution_plan(
            operand,
            result,
            operand_layout,
            result_layout,
            op,
        )
        if redistribution_remap is None:
            layout_remap.reject_unsupported_pair(
                operand_layout,
                result_layout,
                op,
            )
        fact_policy = (
            "preserve_equivalent"
            if redistribution_remap["mode"] == "alias"
            else "invalidate_layout_sensitive"
        )
        attrs = {
            "fact_policy": fact_policy,
            **redistribution_remap,
        }
    builder.add_op(
        "layout_convert",
        operands=_operand_target_ids(builder, op),
        results=result_target_ids,
        attrs=attrs,
        layout_map_ids=result_layout_map_ids,
        source_op_index=op.index,
    )


def _convert_structural_tensor_view(builder, type_layout_program, op):
    """Preserve Triton's zero-cost tensor view as a packet alias.

    TritonGPU infers the result encoding of ``tt.reshape`` and ``tt.trans`` so
    that the physical register payload is unchanged.  Their LLVM lowering
    consequently only repacks, or directly forwards, the source values.  Keep
    that contract structural in the bridge instead of turning the view into a
    semantic redistribution.
    """
    if len(op.operands) != 1 or len(op.results) != 1:
        fail(
            "TLXW_OP_STRUCTURAL_VIEW",
            STAGE,
            f"{op.name} requires one operand and one result",
            source_op_index=op.index,
        )
    operand = type_layout_program.values[op.operands[0]]
    result = type_layout_program.values[op.results[0]]
    operand_layout = _require_layout(
        type_layout_program,
        operand.layout_map_id,
        op,
    )
    result_layout = _require_layout(
        type_layout_program,
        result.layout_map_id,
        op,
    )
    alias_plan = layout_remap.structural_view_alias_plan(
        operand,
        result,
        operand_layout,
        result_layout,
        op,
    )
    result_target_ids, result_layout_map_ids = _declare_results(
        builder,
        op,
        type_layout_program,
    )
    builder.add_op(
        "layout_convert",
        operands=_operand_target_ids(builder, op),
        results=result_target_ids,
        attrs={
            "fact_policy": "preserve_equivalent",
            **alias_plan,
            "view_kind": op.name,
        },
        layout_map_ids=result_layout_map_ids,
        source_op_index=op.index,
    )


def _add_layout_remap_scratch_attrs(attrs, conversion_input, result, op):
    if attrs.get("mode") not in {
            "cta_exchange_register_remap",
            "dot_operand_vector_payload",
            "mfma_vector_register_remap",
    }:
        return attrs
    if "scratch_element_count" not in attrs:
        return attrs
    element_byte_width = conversion_input.value_element_byte_widths.get(result.value_id)
    if result.type.representation in {"mask", "mask_tuple"}:
        element_byte_width = 4
    if element_byte_width is None:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "CTA exchange layout remap requires a known element byte width",
            source_op_index=op.index,
            source_value_id=result.value_id,
        )
    scratch_elements = int(attrs["scratch_element_count"])
    if scratch_elements <= 0:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "CTA exchange layout remap produced an empty scratch allocation",
            source_op_index=op.index,
            source_value_id=result.value_id,
        )
    packet_elements = 1
    has_physical_plan = attrs.get("scratch_physical_plan") == "optimal_swizzling_ldst"
    if (not has_physical_plan and attrs.get("mode") == "cta_exchange_register_remap"
            and result.type.representation not in {
            "mask",
            "mask_tuple",
    }):
        packet_elements = _cta_exchange_packet_elements(attrs, element_byte_width)
    source_store_packet_elements = 1
    if (attrs.get("mode") == "dot_operand_vector_payload"
            and not has_physical_plan):
        source_store_packet_elements = _dot_operand_source_store_packet_elements(
            attrs,
            element_byte_width,
        )
        payload_transpose_load_elements = _dot_operand_payload_transpose_load_elements(
            attrs,
            element_byte_width,
        )
    else:
        payload_transpose_load_elements = 1
    scratch_bytes = _align_to(
        scratch_elements * int(packet_elements) * int(element_byte_width),
        16,
    )
    result_attrs = {
        **attrs,
        "scratch_allocation_bytes": int(scratch_bytes),
        "scratch_align": 16,
    }
    if packet_elements > 1:
        result_attrs["cta_exchange_packet_elements"] = int(packet_elements)
    if source_store_packet_elements > 1:
        result_attrs["source_store_packet_elements"] = int(source_store_packet_elements)
    if payload_transpose_load_elements > 1:
        result_attrs["payload_transpose_load_elements"] = int(payload_transpose_load_elements)
    return result_attrs


def _cta_exchange_packet_elements(attrs, element_byte_width):
    exchange_groups = tuple(attrs.get("exchange_groups", ()))
    if len(exchange_groups) < 2:
        return 1
    max_packet_elements = min(_buffer_max_packet_elements(element_byte_width), len(exchange_groups))
    for packet_elements in range(max_packet_elements, 1, -1):
        if not _buffer_packet_payload_is_legal(packet_elements, element_byte_width):
            continue
        if layout_remap.cta_exchange_has_packet_group(exchange_groups, packet_elements):
            return int(packet_elements)
    return 1


def _dot_operand_source_store_packet_elements(attrs, element_byte_width):
    source_store_bases = tuple(int(value) for value in attrs.get("source_store_bases", ()))
    source_store_coefficients = tuple(
        tuple(int(value) for value in coefficients)
        for coefficients in attrs.get("source_store_coefficients", ())
    )
    component_count = int(attrs.get("source_component_count", len(source_store_bases)))
    if (component_count <= 1 or len(source_store_bases) != component_count
            or len(source_store_coefficients) != component_count):
        return 1
    max_packet_elements = min(
        component_count,
        _buffer_max_packet_elements(element_byte_width),
    )
    for packet_elements in range(max_packet_elements, 1, -1):
        if component_count % packet_elements:
            continue
        if not _buffer_packet_payload_is_legal(packet_elements, element_byte_width):
            continue
        if _dot_operand_source_store_has_packet_tiling(
                source_store_bases,
                source_store_coefficients,
                packet_elements,
        ):
            return int(packet_elements)
    return 1


def _dot_operand_source_store_has_packet_tiling(
    source_store_bases,
    source_store_coefficients,
    packet_elements,
):
    packet_elements = int(packet_elements)
    for index in range(0, len(source_store_bases), packet_elements):
        base = int(source_store_bases[index])
        coefficients = tuple(int(value) for value in source_store_coefficients[index])
        for element in range(packet_elements):
            component = index + element
            if tuple(int(value) for value in source_store_coefficients[component]) != coefficients:
                return False
            if int(source_store_bases[component]) != base + element:
                return False
    return True


def _dot_operand_payload_transpose_load_elements(attrs, element_byte_width):
    if bool(attrs.get("payload_vector_contiguous", True)):
        return 1
    if int(attrs.get("role", -1)) != 1:
        return 1
    if int(element_byte_width) != 2:
        return 1
    element_type = attrs.get("element_type")
    if element_type not in {"bf16", "f16"}:
        return 1
    chunk_elements = 4
    elements_per_lane = int(attrs.get("elements_per_lane", 0))
    result_count = int(attrs.get("result_component_count", 0))
    if elements_per_lane <= 0 or elements_per_lane % chunk_elements or result_count <= 0:
        return 1
    transpose_bases = tuple(
        tuple(int(value) for value in bases) for bases in attrs.get("payload_transpose_load_bases", ()))
    transpose_coefficients = tuple(
        tuple(tuple(int(value) for value in coefficients) for coefficients in component_coefficients)
        for component_coefficients in attrs.get("payload_transpose_load_coefficients", ())
    )
    chunks_per_component = elements_per_lane // chunk_elements
    if len(transpose_bases) != result_count or len(transpose_coefficients) != result_count:
        return 1
    for component_bases, component_coefficients in zip(transpose_bases, transpose_coefficients):
        if len(component_bases) != chunks_per_component or len(component_coefficients) != chunks_per_component:
            return 1
    scalar_bases = tuple(
        tuple(int(value) for value in bases) for bases in attrs.get("payload_scalar_load_bases", ()))
    scalar_coefficients = tuple(
        tuple(tuple(int(value) for value in coefficients) for coefficients in component_coefficients)
        for component_coefficients in attrs.get("payload_scalar_load_coefficients", ())
    )
    if len(scalar_bases) != result_count or len(scalar_coefficients) != result_count:
        return 1
    stride = None
    for component_bases, component_coefficients in zip(scalar_bases, scalar_coefficients):
        if len(component_bases) != elements_per_lane or len(component_coefficients) != elements_per_lane:
            return 1
        for chunk_start in range(0, elements_per_lane, chunk_elements):
            coefficients = component_coefficients[chunk_start]
            if any(component_coefficients[chunk_start + element] != coefficients for element in range(chunk_elements)):
                return 1
            chunk_stride = component_bases[chunk_start + 1] - component_bases[chunk_start]
            if chunk_stride <= 0:
                return 1
            if stride is None:
                stride = int(chunk_stride)
            elif int(chunk_stride) != int(stride):
                return 1
            for element in range(chunk_elements):
                if component_bases[chunk_start + element] != component_bases[chunk_start] + element * int(stride):
                    return 1
    return int(chunk_elements)


def _same_layout_alias(operand, result, operand_layout, result_layout):
    if int(operand.type.component_count) != int(result.type.component_count):
        return False
    if operand.type.element_type != result.type.element_type:
        return False
    if operand_layout is None or result_layout is None:
        return operand_layout is result_layout
    return (operand_layout.kind == result_layout.kind and tuple(operand_layout.shape) == tuple(result_layout.shape)
            and operand_layout.element_type == result_layout.element_type
            and operand_layout.properties == result_layout.properties)


def _same_layout_except_element_type(operand_layout, result_layout):
    if operand_layout is None or result_layout is None:
        return operand_layout is result_layout
    return (operand_layout.kind == result_layout.kind and tuple(operand_layout.shape) == tuple(result_layout.shape)
            and int(operand_layout.component_count) == int(result_layout.component_count)
            and int(operand_layout.lane_width) == int(result_layout.lane_width)
            and operand_layout.properties == result_layout.properties)


def _layout_for_converted_value(type_layout_program, value):
    if value.layout_map_id is None:
        return None
    return type_layout_program.layouts[int(value.layout_map_id)]


def _require_same_layout_except_element_type(
    type_layout_program,
    operand,
    result,
    description,
    op,
):
    operand_layout = _layout_for_converted_value(type_layout_program, operand)
    result_layout = _layout_for_converted_value(type_layout_program, result)
    if _same_layout_except_element_type(operand_layout, result_layout):
        return
    fail(
        "TLXW_OP_LAYOUT_MISMATCH",
        STAGE,
        f"{description} layouts must match; use ttg.convert_layout for layout changes",
        source_op_index=op.index,
        source_value_id=result.value_id,
    )


def _require_simple_op_layout_contract(type_layout_program, op):
    if op.name not in _LAYOUT_PRESERVING_SIMPLE_OPS or not op.results:
        return
    result = type_layout_program.values[op.results[0]]
    for operand_id in op.operands:
        operand = type_layout_program.values[operand_id]
        _require_same_layout_except_element_type(
            type_layout_program,
            operand,
            result,
            f"{op.name} operand and result",
            op,
        )


def _require_yield_layouts(
    type_layout_program,
    yielded_source_value_ids,
    result_value_ids,
    description,
    op,
):
    for index, (yielded_source_value_id, result_value_id) in enumerate(zip(yielded_source_value_ids, result_value_ids)):
        _require_same_layout_except_element_type(
            type_layout_program,
            type_layout_program.values[yielded_source_value_id],
            type_layout_program.values[result_value_id],
            f"{description} {index}",
            op,
        )


def _require_for_iter_layouts(
    type_layout_program,
    init_value_ids,
    result_value_ids,
    block_arg_value_ids,
    op,
):
    for index, (init_value_id, result_value_id) in enumerate(zip(init_value_ids, result_value_ids)):
        _require_same_layout_except_element_type(
            type_layout_program,
            type_layout_program.values[init_value_id],
            type_layout_program.values[result_value_id],
            f"scf.for iter_arg and result {index}",
            op,
        )
    for index, (block_arg_value_id, result_value_id) in enumerate(zip(block_arg_value_ids, result_value_ids)):
        _require_same_layout_except_element_type(
            type_layout_program,
            type_layout_program.values[block_arg_value_id],
            type_layout_program.values[result_value_id],
            f"scf.for block argument and result {index}",
            op,
        )


def _require_mask_layout_compatible(
    type_layout_program,
    mask,
    reference,
    description,
    op,
):
    if mask.layout_map_id is None:
        return
    _require_same_layout_except_element_type(
        type_layout_program,
        mask,
        reference,
        description,
        op,
    )


def _convert_async_commit_group(builder, type_layout_program, token_groups_by_commit, op):
    group = token_groups_by_commit.get(op.index)
    if group is None:
        fail(
            "TLXW_OP_ASYNC_COMMIT_TOKEN",
            STAGE,
            "ttg.async_commit_group requires a token group",
            source_op_index=op.index,
        )
    result_target_ids, _ = _declare_results(builder, op, type_layout_program)
    for result_target_id in result_target_ids:
        builder.set_value_event_domain(
            result_target_id,
            target_ir.EVENT_DOMAIN_DMA_GROUP,
        )
    operands = tuple(_single_source_target(builder, token_value_id, op) for token_value_id in group.member_token_ids)
    issue_group_size = _int_attr_or_default(op.attrs, "tlx.async_issue_group_size", 0)
    issue_delay_cycles = _int_attr_or_default(op.attrs, "tlx.async_issue_delay_cycles", 0)
    issue_delay_overlap_cycles = _int_attr_or_default(
        op.attrs, "tlx.async_issue_delay_overlap_cycles", 0)
    issue_delay_skip_thread_threshold = _int_attr_or_default(
        op.attrs, "tlx.async_issue_delay_skip_thread_threshold", 0)
    if bool(issue_group_size) != bool(issue_delay_cycles):
        fail(
            "TLXW_OP_ASYNC_COMMIT_TOKEN",
            STAGE,
            "async commit issue delay requires both a group size and delay cycles",
            source_op_index=op.index,
        )
    if issue_group_size < 0 or issue_delay_cycles < 0:
        fail(
            "TLXW_OP_ASYNC_COMMIT_TOKEN",
            STAGE,
            "async commit issue delay fields must be non-negative",
            source_op_index=op.index,
        )
    if not 0 <= issue_delay_overlap_cycles <= issue_delay_cycles:
        fail(
            "TLXW_OP_ASYNC_COMMIT_TOKEN",
            STAGE,
            "async commit issue delay overlap exceeds the delay",
            source_op_index=op.index,
        )
    if issue_delay_skip_thread_threshold < 0:
        fail(
            "TLXW_OP_ASYNC_COMMIT_TOKEN",
            STAGE,
            "async commit issue delay thread threshold must be non-negative",
            source_op_index=op.index,
        )
    builder.add_op(
        "async_commit_group",
        operands=operands,
        results=result_target_ids,
        attrs={
            "group_id": int(group.group_id),
            "member_count": len(group.member_token_ids),
            "issue_group_size": int(issue_group_size),
            "issue_delay_cycles": int(issue_delay_cycles),
            "issue_delay_overlap_cycles": int(issue_delay_overlap_cycles),
            "issue_delay_skip_thread_threshold": int(issue_delay_skip_thread_threshold),
        },
        source_op_index=op.index,
    )


def _convert_async_wait(
    builder,
    conversion_input,
    type_layout_program,
    op,
):
    node = conversion_input.token_nodes_by_op.get(op.index)
    if node is None:
        fail(
            "TLXW_OP_ASYNC_WAIT_TOKEN",
            STAGE,
            "ttg.async_wait requires a token graph node",
            source_op_index=op.index,
        )
    result_target_ids, _ = _declare_results(builder, op, type_layout_program)
    wait_source_value_id = None if not op.results else int(op.results[0])
    if node.input_token_ids:
        wait_token_ids = node.input_token_ids
    else:
        wait_token_ids = _implicit_wait_token_ids(conversion_input, node, op)
    group_operand_ids = tuple(dict.fromkeys(
        _single_source_target(builder, token_value_id, op)
        for token_value_id in wait_token_ids
    ))
    release_value_ids = (
        conversion_input.async_protocol_dependency_value_ids_by_op.get(
            op.index,
            (),
        )
    )
    live_release_value_ids = tuple(
        int(source_value_id)
        for source_value_id, target_ids in sorted(
            builder.protocol_frontiers.items()
        )
        if target_ids
    )
    release_value_ids = tuple(dict.fromkeys((
        *release_value_ids,
        *live_release_value_ids,
    )))
    release_operand_ids = tuple(dict.fromkeys(
        target_id
        for source_value_id in release_value_ids
        for target_id in builder.protocol_frontiers.get(
            int(source_value_id), ()
        )
    ))
    retained_group_operand_ids = tuple(dict.fromkeys(
        _single_source_target(
            builder,
            conversion_input.token_groups_by_id[group_id].token_value_id,
            op,
        )
        for group_id in node.retained_group_ids
        if conversion_input.token_groups_by_id[group_id].token_value_id
        is not None
    ))
    retained_issue_operand_ids = ()
    if retained_group_operand_ids:
        retained_issue_target_id = _declare_protocol_token(
            builder,
            event_domain=target_ir.EVENT_DOMAIN_DMA_ISSUE,
            debug_name=f"retained_issue_{op.index}",
        )
        builder.add_op(
            "issue_token",
            operands=retained_group_operand_ids,
            results=(retained_issue_target_id, ),
            attrs={
                "input_count": len(retained_group_operand_ids),
                "projection_domain": target_ir.EVENT_DOMAIN_DMA_ISSUE,
                "projection_provenance": "partial_wait_retained_group",
                "retained_group_ids": tuple(
                    int(group_id) for group_id in node.retained_group_ids
                ),
            },
            source_op_index=op.index,
        )
        retained_issue_operand_ids = (retained_issue_target_id, )
    publication_mode = _async_wait_publication_mode(
        conversion_input,
        wait_source_value_id,
        bool(release_operand_ids),
    )
    coalesced_source_barrier_op_index = (
        int(conversion_input.wait_publication_barrier_by_op[op.index])
        if (
            publication_mode == "workgroup"
            and op.index in conversion_input.wait_publication_barrier_by_op
        )
        else -1
    )
    ready_domain = (
        target_ir.EVENT_DOMAIN_WORKGROUP_READY
        if publication_mode == "workgroup"
        else target_ir.EVENT_DOMAIN_WAVE_LOCAL_READY
    )
    for result_target_id in result_target_ids:
        builder.set_value_event_domain(result_target_id, ready_domain)
    operands = (
        *group_operand_ids,
        *retained_issue_operand_ids,
        *release_operand_ids,
    )
    builder.add_op(
        "async_wait",
        operands=operands,
        results=result_target_ids,
        attrs={
            "wait_group": -1 if node.wait_group is None else int(node.wait_group),
            "waited_group_ids": tuple(int(group_id) for group_id in node.waited_group_ids),
            "retained_group_ids": tuple(
                int(group_id) for group_id in node.retained_group_ids
            ),
            "completed_group_dependency_count": len(group_operand_ids),
            "retained_issue_dependency_count": len(
                retained_issue_operand_ids
            ),
            "lds_release_dependency_count": len(release_operand_ids),
            "publication_mode": publication_mode,
            "publication_provenance": (
                "amd_membar_compatibility"
                if publication_mode == "workgroup"
                else "single_wave_ownership"
            ),
            "coalesced_source_barrier_op_index": (
                coalesced_source_barrier_op_index
            ),
        },
        source_op_index=op.index,
    )
    for release_value_id in release_value_ids:
        builder.protocol_frontiers.pop(int(release_value_id), None)
    if wait_source_value_id is not None:
        builder.set_protocol_frontier(wait_source_value_id, ())


def _async_wait_publication_mode(
    conversion_input,
    wait_source_value_id,
    has_release_dependencies,
):
    wait_has_local_consumer = (
        wait_source_value_id is not None
        and any(
            int(wait_source_value_id) in tuple(int(value_id) for value_id in value_ids)
            for value_ids in conversion_input.async_protocol_dependency_value_ids_by_op.values()
        )
    )
    if (
        conversion_input.num_warps > 1
        and (wait_has_local_consumer or has_release_dependencies)
    ):
        return "workgroup"
    return "wave_local"


def _implicit_wait_token_ids(conversion_input, node, op):
    if int(node.wait_group or 0) > 0 and _waited_tokens_cross_if_merge(
        conversion_input,
        node,
    ):
        # The source token graph is linearized across sibling regions.  A
        # partial implicit wait needs a path-sensitive queue length to decide
        # which branch-local groups remain live; treating neutral branch tokens
        # as real queue entries would wait the wrong group.  Full drains are
        # path independent and are merged structurally by _convert_if.
        fail(
            "TLXW_OP_UNSUPPORTED_IF_TOKENS",
            STAGE,
            "a partial implicit async wait after scf.if requires "
            "path-sensitive branch queue lengths",
            source_op_index=op.index,
        )
    wait_token_ids = []
    for group_id in node.waited_group_ids:
        group = conversion_input.token_groups_by_id[group_id]
        if group.token_value_id is None:
            continue
        if (_source_token_crosses_if_branch_path(
                conversion_input,
                group.commit_op_index,
                op.index,
        ) and not _source_token_has_if_merge(conversion_input, group.token_value_id)):
            fail(
                "TLXW_OP_UNSUPPORTED_IF_TOKENS",
                STAGE,
                "implicit ttg.async_wait cannot wait an async group from a "
                "different scf.if branch path",
                source_op_index=op.index,
            )
        wait_token_ids.append(group.token_value_id)
    return tuple(wait_token_ids)


def _waited_tokens_cross_if_merge(conversion_input, node):
    waited_token_ids = {
        conversion_input.token_groups_by_id[group_id].token_value_id
        for group_id in node.waited_group_ids
    }
    return any(
        source_value_id in waited_token_ids
        for carries in conversion_input.if_token_carries_by_op.values()
        for carry in carries
        for source_value_id in (
            carry.then_source_value_id,
            carry.else_source_value_id,
        )
        if source_value_id is not None
    )


def _source_token_has_if_merge(conversion_input, source_value_id):
    return any(
        source_value_id in (carry.then_source_value_id, carry.else_source_value_id)
        for carries in conversion_input.if_token_carries_by_op.values()
        for carry in carries
    )


def _convert_return(builder, view):
    builder.add_op(
        "return",
        operands=view.operand_target_ids,
        source_op_index=view.op_index,
    )


def _convert_reduce(builder, conversion_input, type_layout_program, op):
    if len(op.operands) != 1 or len(op.results) != 1 or len(op.region_ids) != 1:
        fail(
            "TLXW_OP_REDUCTION",
            STAGE,
            "tt.reduce currently requires one input, one result, and one combiner region",
            source_op_index=op.index,
        )
    operand = type_layout_program.values[op.operands[0]]
    result = type_layout_program.values[op.results[0]]
    if operand.type.representation not in _MMA_PACKET_REPRESENTATIONS:
        fail(
            "TLXW_OP_REDUCTION",
            STAGE,
            "tt.reduce input must be an MFMA packet payload",
            source_op_index=op.index,
            source_value_id=operand.value_id,
        )
    if result.type.representation not in {"simd", "simd_tuple"}:
        fail(
            "TLXW_OP_REDUCTION",
            STAGE,
            "tt.reduce result must be a distributed SIMD payload",
            source_op_index=op.index,
            source_value_id=result.value_id,
        )
    if operand.type.element_type != result.type.element_type:
        fail(
            "TLXW_OP_REDUCTION",
            STAGE,
            "tt.reduce input and result element types must match",
            source_op_index=op.index,
        )
    if result.type.element_type != "f32":
        fail(
            "TLXW_OP_REDUCTION",
            STAGE,
            "MFMA fragment reductions currently require f32 elements",
            source_op_index=op.index,
        )
    axis = _int_attr(op.attrs, "axis")
    operand_layout = _require_layout(type_layout_program, operand.layout_map_id, op)
    result_layout = _require_layout(type_layout_program, result.layout_map_id, op)
    _require_mfma_slice_reduction_layouts(operand_layout, result_layout, axis, op)
    combiner = _reduction_combiner(conversion_input, op)
    operations = {
        "arith.addf": "addf",
        "arith.maximumf": "maximumf",
        "arith.maxnumf": "maxnumf",
        "arith.mulf": "mulf",
    }
    operation = operations.get(combiner.name)
    if operation is None:
        fail(
            "TLXW_OP_REDUCTION",
            STAGE,
            f"unsupported tt.reduce combiner {combiner.name}",
            source_op_index=op.index,
        )
    builder.erase_source_value(
        combiner.results[0],
        f"folded {combiner.name} into tt.reduce target operation",
    )
    attrs = {
        "axis": int(axis),
        "component_terms": _within_wave_reduction_terms(
            operand,
            result,
            operand_layout,
            result_layout,
            axis,
            op,
        ),
        "lane_width": int(result.type.lane_width or operand.type.lane_width or 64),
        "operation": operation,
        "source_registers": int(layouts.mfma_registers_per_component(
            operand_layout,
            stage=STAGE,
            source_op_index=op.index,
        )),
    }
    if combiner.name in {"arith.addf", "arith.mulf"}:
        if "fastmath" in combiner.attrs:
            fastmath = _normalize_fastmath_flags(
                combiner.attrs["fastmath"],
                source_op_index=combiner.index,
            )
        else:
            fastmath = _default_float_binary_fastmath_flags(combiner.name)
        if fastmath:
            attrs["fastmath"] = fastmath
    result_target_ids, result_layout_map_ids = _declare_results(
        builder,
        op,
        type_layout_program,
    )
    builder.add_op(
        "reduction",
        operands=(_single_source_target(builder, operand.value_id, op), ),
        results=result_target_ids,
        attrs=attrs,
        layout_map_ids=result_layout_map_ids,
        source_op_index=op.index,
    )


def _reduction_combiner(conversion_input, op):
    region = conversion_input.regions[op.region_ids[0]]
    region_ops = tuple(conversion_input.ops[index] for index in region.op_indices)
    if len(region.block_arg_ids) != 2 or len(region_ops) != 2:
        fail(
            "TLXW_OP_REDUCTION",
            STAGE,
            "tt.reduce combiner must contain two block arguments, one binary op, and tt.reduce.return",
            source_op_index=op.index,
        )
    combiner, terminator = region_ops
    if (len(combiner.operands) != 2 or len(combiner.results) != 1
            or tuple(combiner.operands) != tuple(region.block_arg_ids)):
        fail(
            "TLXW_OP_REDUCTION",
            STAGE,
            "tt.reduce combiner must apply one binary op directly to its block arguments",
            source_op_index=op.index,
        )
    if (terminator.name != "tt.reduce.return" or tuple(terminator.operands) != tuple(combiner.results)):
        fail(
            "TLXW_OP_REDUCTION",
            STAGE,
            "tt.reduce combiner must return its binary result",
            source_op_index=op.index,
        )
    return combiner


def _require_mfma_slice_reduction_layouts(operand_layout, result_layout, axis, op):
    if operand_layout.kind != "amd_mfma":
        fail(
            "TLXW_OP_REDUCTION",
            STAGE,
            "tt.reduce fragment input must use an amd_mfma layout",
            source_op_index=op.index,
            source_value_id=operand_layout.value_id,
        )
    if (result_layout.kind != "slice"
            or result_layout.properties.get("parent_kind") != "amd_mfma"):
        fail(
            "TLXW_OP_REDUCTION",
            STAGE,
            "tt.reduce result must use an MFMA slice layout",
            source_op_index=op.index,
            source_value_id=result_layout.value_id,
        )
    if int(result_layout.properties.get("dim", -1)) != int(axis):
        fail(
            "TLXW_OP_REDUCTION",
            STAGE,
            "tt.reduce axis must match the result slice dimension",
            source_op_index=op.index,
            source_value_id=result_layout.value_id,
        )
    if result_layout.properties.get("parent_properties", {}) != operand_layout.properties:
        fail(
            "TLXW_OP_REDUCTION",
            STAGE,
            "tt.reduce result slice parent must match the input MFMA layout",
            source_op_index=op.index,
            source_value_id=result_layout.value_id,
        )


def _within_wave_reduction_terms(operand, result, operand_layout, result_layout, axis, op):
    lane_width = int(result.type.lane_width or operand.type.lane_width or 64)
    warp_count = layouts.layout_warp_count(operand_layout)
    if layouts.layout_warp_count(result_layout) != warp_count:
        fail(
            "TLXW_OP_REDUCTION",
            STAGE,
            "tt.reduce input and result layouts must have the same warp count",
            source_op_index=op.index,
        )
    source_linear = layouts.distributed_linear_layout(
        operand_layout,
        stage=STAGE,
        source_op_index=op.index,
    )
    result_linear = layouts.distributed_linear_layout(
        result_layout,
        stage=STAGE,
        source_op_index=op.index,
    )
    source_component_registers = layouts.linear_layout_component_registers(
        source_linear,
        operand_layout,
        operand.type.component_count,
        stage=STAGE,
        source_op_index=op.index,
        source_value_id=operand.value_id,
    )
    result_component_registers = layouts.linear_layout_component_registers(
        result_linear,
        result_layout,
        result.type.component_count,
        stage=STAGE,
        source_op_index=op.index,
        source_value_id=result.value_id,
    )
    registers_per_component = layouts.mfma_registers_per_component(
        operand_layout,
        stage=STAGE,
        source_op_index=op.index,
    )
    source_slots = {}
    for warp in range(int(warp_count)):
        for component, component_register in enumerate(source_component_registers):
            for register in range(int(registers_per_component)):
                for lane in range(lane_width):
                    coords = layouts.linear_layout_coords(
                        source_linear,
                        int(component_register) + int(register),
                        lane,
                        warp=warp,
                    )
                    key = (int(warp), tuple(int(coord) for coord in coords))
                    source_slots.setdefault(key, set()).add((int(component), int(register), int(lane)))

    component_terms = []
    reduction_extent = int(operand_layout.shape[int(axis)])
    for result_register in result_component_registers:
        terms = []
        for reduction_coord in range(reduction_extent):
            source_component_registers_for_term = set()
            lane_maps = []
            for warp in range(int(warp_count)):
                lane_map = []
                for lane in range(lane_width):
                    result_coords = list(layouts.linear_layout_coords(
                        result_linear,
                        result_register,
                        lane,
                        warp=warp,
                    ))
                    result_coords.insert(int(axis), int(reduction_coord))
                    candidates = source_slots.get((int(warp), tuple(result_coords)))
                    if not candidates:
                        fail(
                            "TLXW_OP_REDUCTION",
                            STAGE,
                            "tt.reduce axis crosses waves or is not covered by the input layout",
                            source_op_index=op.index,
                            source_value_id=operand.value_id,
                        )
                    source_component, source_register, source_lane = min(candidates)
                    source_component_registers_for_term.add((source_component, source_register))
                    lane_map.append(source_lane)
                lane_maps.append(tuple(lane_map))
            if len(source_component_registers_for_term) != 1 or len(set(lane_maps)) != 1:
                fail(
                    "TLXW_OP_REDUCTION",
                    STAGE,
                    "tt.reduce requires a uniform within-wave source register map",
                    source_op_index=op.index,
                    source_value_id=operand.value_id,
                )
            source_component, source_register = next(iter(source_component_registers_for_term))
            lane_base, lane_coefficients = _bit_linear_lane_map(lane_maps[0], lane_width, op)
            terms.append((
                int(source_component),
                int(source_register),
                int(lane_base),
                tuple(int(value) for value in lane_coefficients),
            ))
        component_terms.append(tuple(terms))
    return tuple(component_terms)


def _bit_linear_lane_map(lane_map, lane_width, op):
    lane_width = int(lane_width)
    if lane_width <= 0 or lane_width & (lane_width - 1):
        fail(
            "TLXW_OP_REDUCTION",
            STAGE,
            f"tt.reduce requires a power-of-two lane width, got {lane_width}",
            source_op_index=op.index,
        )
    lane_map = tuple(int(value) for value in lane_map)
    if len(lane_map) != lane_width:
        fail(
            "TLXW_OP_REDUCTION",
            STAGE,
            "tt.reduce source lane map width does not match the target wave",
            source_op_index=op.index,
        )
    base = lane_map[0]
    bit_count = lane_width.bit_length() - 1
    coefficients = tuple(base ^ lane_map[1 << bit] for bit in range(bit_count))
    for lane, source_lane in enumerate(lane_map):
        expected = base
        for bit, coefficient in enumerate(coefficients):
            if lane & (1 << bit):
                expected ^= coefficient
        if expected != source_lane:
            fail(
                "TLXW_OP_REDUCTION",
                STAGE,
                "tt.reduce source lane map is not bit-linear",
                source_op_index=op.index,
            )
    return int(base), tuple(int(value) for value in coefficients)


def _convert_select(
    builder,
    conversion_input,
    type_layout_program,
    fact_program,
    op,
):
    _require_simple_op_layout_contract(type_layout_program, op)
    if len(op.operands) != 3 or len(op.results) != 1:
        fail(
            "TLXW_OP_SELECT",
            STAGE,
            "arith.select requires one condition, two values, and one result",
            source_op_index=op.index,
        )
    condition = type_layout_program.values[int(op.operands[0])]
    if condition.type.element_type != "i1":
        fail(
            "TLXW_OP_SELECT",
            STAGE,
            "arith.select condition must have i1 element type",
            source_op_index=op.index,
            source_value_id=int(op.operands[0]),
        )
    condition_target_id = _single_source_target(
        builder,
        int(op.operands[0]),
        op,
    )
    true_target_id = _single_source_target(builder, int(op.operands[1]), op)
    false_target_id = _single_source_target(builder, int(op.operands[2]), op)
    true_value = type_layout_program.values[int(op.operands[1])]
    false_value = type_layout_program.values[int(op.operands[2])]
    result = type_layout_program.values[int(op.results[0])]
    result_target_ids, result_layout_map_ids = _declare_results(
        builder,
        op,
        type_layout_program,
    )
    result_target_id = result_target_ids[0]

    if result.type.representation in _MMA_PACKET_REPRESENTATIONS:
        true_target_id, scalar_type = _unpack_mma_packet_edge(
            builder,
            type_layout_program,
            true_value,
            true_target_id,
            op,
        )
        false_target_id, false_scalar_type = _unpack_mma_packet_edge(
            builder,
            type_layout_program,
            false_value,
            false_target_id,
            op,
        )
        if scalar_type != false_scalar_type:
            fail(
                "TLXW_OP_SELECT",
                STAGE,
                "arith.select packet operands require identical structural "
                "scalar edge types",
                source_op_index=op.index,
            )
        if int(condition.type.component_count) not in {
            1,
            int(scalar_type.component_count),
        }:
            fail(
                "TLXW_OP_SELECT",
                STAGE,
                "arith.select condition components must match the unpacked "
                "packet payload",
                source_op_index=op.index,
                source_value_id=int(op.operands[0]),
            )
        scalar_result_target_id = builder.add_value(
            scalar_type,
            debug_name=f"select_scalar_result_{op.index}",
        )
        builder.add_op(
            "select",
            operands=(condition_target_id, true_target_id, false_target_id),
            results=(scalar_result_target_id, ),
            source_op_index=op.index,
        )
        _pack_mma_packet_edge(
            builder,
            type_layout_program,
            result,
            scalar_result_target_id,
            result_target_id,
            op,
        )
        return

    builder.add_op(
        "select",
        operands=(condition_target_id, true_target_id, false_target_id),
        results=result_target_ids,
        layout_map_ids=result_layout_map_ids,
        source_op_index=op.index,
    )


def _unpack_mma_packet_edge(
    builder,
    type_layout_program,
    value,
    target_value_id,
    op,
):
    if value.type.representation not in _MMA_PACKET_REPRESENTATIONS:
        fail(
            "TLXW_OP_SELECT",
            STAGE,
            "packet select requires packet operands on both value edges",
            source_op_index=op.index,
            source_value_id=value.value_id,
        )
    packet_width = _mma_packet_registers(type_layout_program, value, op)
    scalar_component_count = int(value.type.component_count) * int(packet_width)
    scalar_type = target_ir.TargetType(
        "tensor",
        "simd" if scalar_component_count == 1 else "simd_tuple",
        value.type.element_type,
        int(value.type.lane_width or 64),
        scalar_component_count,
    )
    result_target_id = builder.add_value(
        scalar_type,
        debug_name=f"packet_unpack_{op.index}_{value.value_id}",
    )
    builder.add_op(
        "type_convert",
        operands=(int(target_value_id), ),
        results=(result_target_id, ),
        attrs={
            "mode": "packet_to_scalar_components",
            "packet_component_count": int(value.type.component_count),
            "packet_width": int(packet_width),
        },
        layout_map_ids=(() if value.layout_map_id is None else (value.layout_map_id, )),
        source_op_index=op.index,
    )
    return result_target_id, scalar_type


def _pack_mma_packet_edge(
    builder,
    type_layout_program,
    result,
    scalar_target_id,
    result_target_id,
    op,
):
    packet_width = _mma_packet_registers(type_layout_program, result, op)
    builder.add_op(
        "type_convert",
        operands=(int(scalar_target_id), ),
        results=(int(result_target_id), ),
        attrs={
            "mode": "scalar_components_to_packet",
            "packet_component_count": int(result.type.component_count),
            "packet_width": int(packet_width),
        },
        layout_map_ids=(() if result.layout_map_id is None else (result.layout_map_id, )),
        source_op_index=op.index,
    )


def _component_remap_edge(
    builder,
    target_value_id,
    component_sources,
    op,
):
    """Create an explicit component-only conversion on a target SSA edge."""
    component_sources = tuple(int(source) for source in component_sources)
    source_type = builder.values[int(target_value_id)].type
    source_count = int(source_type.component_count)
    if not component_sources or any(
        source < 0 or source >= source_count
        for source in component_sources
    ):
        fail(
            "TLXW_OP_TYPE_CONVERT",
            STAGE,
            "component remap references an invalid source component",
            source_op_index=op.index,
            target_value_id=int(target_value_id),
        )
    if component_sources == tuple(range(source_count)):
        return int(target_value_id)
    if source_type.representation in {"mask", "mask_tuple"}:
        representation = "mask" if len(component_sources) == 1 else "mask_tuple"
    elif source_type.representation in {"simd", "simd_tuple"}:
        representation = "simd" if len(component_sources) == 1 else "simd_tuple"
    else:
        fail(
            "TLXW_OP_TYPE_CONVERT",
            STAGE,
            "component remap requires a mask or scalar-SIMD value",
            source_op_index=op.index,
            target_value_id=int(target_value_id),
        )
    result_type = target_ir.TargetType(
        source_type.kind,
        representation,
        source_type.element_type,
        source_type.lane_width,
        len(component_sources),
    )
    result_target_id = builder.add_value(
        result_type,
        debug_name=f"component_edge_{op.index}_{target_value_id}",
    )
    builder.add_op(
        "type_convert",
        operands=(int(target_value_id), ),
        results=(result_target_id, ),
        attrs={
            "component_sources": component_sources,
            "mode": "component_remap",
        },
        source_op_index=op.index,
    )
    return result_target_id


_SIMPLE_OP_CONVERTERS = {
    "arith.constant": _convert_constant,
    **{op_name: _convert_binary
       for op_name in _BINARY_OPS},
    **{op_name: _convert_float_binary
       for op_name in _FLOAT_BINARY_OPS},
    **{op_name: _convert_float_unary
       for op_name in _FLOAT_UNARY_OPS},
    **{op_name: _convert_float_cast
       for op_name in _FLOAT_CAST_OPS},
    "arith.maxsi": _convert_maxsi,
    "arith.minsi": _convert_minsi,
    "llvm.intr.assume": _convert_assume,
    "tt.splat": _convert_splat,
    "tt.addptr": _convert_addptr,
    "tt.get_program_id": _convert_program_id,
    "gpu.thread_id": _convert_thread_id,
    "rocdl.workitem.id.x": _convert_workitem_id_x,
    "arith.index_cast": _convert_index_cast,
    "tt.return": _convert_return,
}

_SPECIALIZED_SOURCE_OPS = frozenset({
    "arith.cmpi",
    "arith.select",
    "arith.truncf",
    "amdg.cond_barrier",
    "rocdl.s.barrier",
    "rocdl.s.setprio",
    "rocdl.sched.barrier",
    "ttg.barrier",
    "scf.for",
    "scf.if",
    "tt.broadcast",
    "tt.expand_dims",
    "tt.reshape",
    "tt.trans",
    "tt.make_range",
    "tt.reduce",
    "ttg.local_alloc",
    "ttg.memdesc_index",
    "ttg.memdesc_reinterpret",
    "ttg.memdesc_reshape",
    "ttg.memdesc_subslice",
    "ttg.memdesc_trans",
    "amdg.buffer_load_to_local",
    "amdg.buffer_load",
    "amdg.buffer_store",
    "tt.load",
    "tt.store",
    "ttg.local_load",
    "ttg.local_store",
    "ttg.convert_layout",
    "tt.dot",
    "tt.dot_scaled",
    "ttg.async_commit_group",
    "ttg.async_wait",
})

_SUPPORTED_SOURCE_OPS = frozenset(_SIMPLE_OP_CONVERTERS) | _SPECIALIZED_SOURCE_OPS
_UNOWNED_SOURCE_OPS = _SUPPORTED_SOURCE_OPS - domains.all_source_ops()
if _UNOWNED_SOURCE_OPS:
    raise RuntimeError(f"unsupported source op domains: {sorted(_UNOWNED_SOURCE_OPS)}")


def _convert_sched_barrier(builder, op):
    if op.results:
        fail(
            "TLXW_OP_UNEXPECTED_RESULT",
            STAGE,
            "rocdl.sched.barrier must not produce values",
            source_op_index=op.index,
        )
    border = op.attrs.get("triton.warp_pipeline.border")
    builder.add_op(
        "sched_barrier",
        attrs={
            "border": "" if border is None else str(border),
            "mask": int(op.attrs.get("mask", 0) or 0),
        },
        source_op_index=op.index,
    )


def _convert_barrier(builder, conversion_input, op):
    if op.operands or op.results:
        fail(
            "TLXW_OP_UNSUPPORTED_BARRIER",
            STAGE,
            f"{op.name} must not have operands or results",
            source_op_index=op.index,
        )
    if int(op.index) in {
        int(barrier_op_index)
        for barrier_op_index in
        conversion_input.wait_publication_barrier_by_op.values()
    }:
        return
    dependency_target_ids = builder.protocol_frontier_target_ids()
    result_target_ids = ()
    if dependency_target_ids:
        result_target_ids = (
            _declare_protocol_token(
                builder,
                event_domain=target_ir.EVENT_DOMAIN_LDS_RELEASED,
                debug_name=f"barrier_lds_release_{op.index}",
            ),
        )
    attrs = {
        "address_space": int(op.attrs.get("addrSpace", 0)),
        "dependency_count": len(dependency_target_ids),
    }
    if bool(op.attrs.get("tlx.compiler_membar_barrier", False)):
        attrs["compiler_membar_barrier"] = True
    builder.add_op(
        "barrier",
        operands=dependency_target_ids,
        results=result_target_ids,
        attrs=attrs,
        source_op_index=op.index,
    )
    if not result_target_ids:
        return
    for source_value_id, target_ids in tuple(
        builder.protocol_frontiers.items()
    ):
        if target_ids:
            builder.set_protocol_frontier(
                source_value_id,
                result_target_ids,
            )


def _convert_cond_barrier(builder, op):
    if len(op.operands) != 1 or op.results:
        fail(
            "TLXW_OP_UNSUPPORTED_COND_BARRIER",
            STAGE,
            "amdg.cond_barrier requires one predicate and no results",
            source_op_index=op.index,
        )
    builder.add_op(
        "cond_barrier",
        operands=_operand_target_ids(builder, op),
        source_op_index=op.index,
    )


def _convert_set_priority(builder, op):
    if op.operands or op.results:
        fail(
            "TLXW_OP_UNSUPPORTED_SET_PRIORITY",
            STAGE,
            "rocdl.s.setprio must not have operands or results",
            source_op_index=op.index,
        )
    priority = int(op.attrs.get("priority", -1))
    if priority < 0 or priority > 3:
        fail(
            "TLXW_OP_UNSUPPORTED_SET_PRIORITY",
            STAGE,
            f"rocdl.s.setprio priority must be in [0, 3], got {priority}",
            source_op_index=op.index,
        )
    builder.add_op(
        "set_priority",
        attrs={"priority": priority},
        source_op_index=op.index,
    )


def _fact_ids_by_source_op(fact_program):
    result = {}
    for fact in fact_program.facts:
        if fact.source_op_index is None:
            continue
        result.setdefault(fact.source_op_index, tuple())
        result[fact.source_op_index] = (*result[fact.source_op_index], fact.fact_id)
    return result


def _fact_target_ids(builder, fact_program, fact_ids, op):
    target_ids = []
    for fact_id in fact_ids:
        try:
            fact = fact_program.facts[fact_id]
        except IndexError:
            fail(
                "TLXW_OP_UNKNOWN_FACT",
                STAGE,
                f"op references missing fact {fact_id}",
                source_op_index=op.index,
                fact_id=fact_id,
            )
        target_ids.append(_fact_target_id(builder, fact, op))
    return tuple(target_ids)


def _fact_target_id(builder, fact, op):
    targets = builder.source_value_targets.get(fact.subject_value_id)
    if not targets:
        fail(
            "TLXW_OP_FACT_TARGET",
            STAGE,
            f"fact {fact.fact_id} subject has no converted target value",
            source_op_index=op.index,
            source_value_id=fact.subject_value_id,
            fact_id=fact.fact_id,
        )
    if len(targets) != 1:
        fail(
            "TLXW_OP_FACT_TARGET",
            STAGE,
            f"fact {fact.fact_id} subject maps to multiple target values {targets}",
            source_op_index=op.index,
            source_value_id=fact.subject_value_id,
            fact_id=fact.fact_id,
        )
    return targets[0]


def _operand_assume_fact_ids(conversion_input, fact_program, op):
    fact_ids = []
    seen = set()
    for source_value_id in op.operands:
        for fact_id in fact_program.by_value.get(source_value_id, ()):
            fact = fact_program.facts[fact_id]
            if fact.kind != "range" or fact.provenance != "llvm.intr.assume":
                continue
            if not _source_fact_is_in_scope(
                    conversion_input,
                    fact.source_op_index,
                    op.index,
            ):
                continue
            if fact_id in seen:
                continue
            seen.add(fact_id)
            fact_ids.append(fact_id)
    return tuple(fact_ids)


def _source_fact_is_in_scope(conversion_input, fact_op_index, user_op_index):
    if fact_op_index is None:
        return False
    if fact_op_index == user_op_index:
        return True
    try:
        fact_op = conversion_input.ops[fact_op_index]
    except IndexError:
        return False
    fact_region_id = fact_op.parent_region_id
    if fact_region_id is None:
        return False
    user_anchor = _op_anchor_in_region(
        conversion_input,
        user_op_index,
        fact_region_id,
    )
    if user_anchor is None:
        return False
    if user_anchor == user_op_index:
        return True
    if user_anchor == fact_op_index:
        return True
    return _op_precedes_in_region(
        conversion_input,
        fact_region_id,
        fact_op_index,
        user_anchor,
    )


def _source_token_crosses_if_branch_path(conversion_input, commit_op_index, wait_op_index):
    commit_branches = _enclosing_if_branch_regions(conversion_input, commit_op_index)
    if not commit_branches:
        return False
    wait_branches = _enclosing_if_branch_regions(conversion_input, wait_op_index)
    for if_op_index, commit_branch_region_id in commit_branches.items():
        wait_branch_region_id = wait_branches.get(if_op_index)
        if wait_branch_region_id is None:
            return True
        if wait_branch_region_id != commit_branch_region_id:
            return True
    return False


def _enclosing_if_branch_regions(conversion_input, op_index):
    try:
        region_id = conversion_input.ops[op_index].parent_region_id
    except IndexError:
        return {}
    result = {}
    while region_id is not None:
        try:
            region = conversion_input.regions[region_id]
        except IndexError:
            break
        parent_op_index = region.parent_op_index
        if parent_op_index is None:
            break
        try:
            parent_op = conversion_input.ops[parent_op_index]
        except IndexError:
            break
        if parent_op.name == "scf.if":
            result[parent_op_index] = region_id
        region_id = parent_op.parent_region_id
    return result


def _op_anchor_in_region(conversion_input, op_index, region_id):
    current_op_index = op_index
    while True:
        try:
            current_op = conversion_input.ops[current_op_index]
        except IndexError:
            return None
        current_region_id = current_op.parent_region_id
        if current_region_id == region_id:
            return current_op_index
        if current_region_id is None:
            return None
        parent_op_index = conversion_input.regions[current_region_id].parent_op_index
        if parent_op_index is None:
            return None
        current_op_index = parent_op_index


def _op_precedes_in_region(
    conversion_input,
    region_id,
    lhs_op_index,
    rhs_op_index,
):
    try:
        region_ops = conversion_input.regions[region_id].op_indices
        return region_ops.index(lhs_op_index) < region_ops.index(rhs_op_index)
    except (IndexError, ValueError):
        return False


def _operand_ranges(conversion_input, fact_program, op):
    return tuple(
        _combined_range_for_value(
            conversion_input,
            fact_program,
            source_value_id,
            op.index,
        ) for source_value_id in op.operands)


def _combined_range_for_value(conversion_input, fact_program, value_id, user_op_index):
    lower = None
    upper = None
    for fact_id in fact_program.by_value.get(value_id, ()):
        fact = fact_program.facts[fact_id]
        if fact.kind != "range":
            continue
        if not _range_fact_is_in_scope(conversion_input, fact, user_op_index):
            continue
        if fact.lower is not None:
            lower = fact.lower if lower is None else max(lower, fact.lower)
        if fact.upper is not None:
            upper = fact.upper if upper is None else min(upper, fact.upper)
    return lower, upper


def _range_fact_is_in_scope(conversion_input, fact, user_op_index):
    if fact.source_op_index is None:
        return fact.provenance != "llvm.intr.assume"
    return _source_fact_is_in_scope(
        conversion_input,
        fact.source_op_index,
        user_op_index,
    )


def _pointer_byte_range_fact(fact_program, value_id, op):
    for fact_id in fact_program.by_value.get(value_id, ()):
        fact = fact_program.facts[fact_id]
        if fact.kind == "pointer_byte_range" and fact.upper is not None:
            return fact
    fail(
        "TLXW_OP_MISSING_POINTER_RANGE_FACT",
        STAGE,
        "amdg.buffer_load_to_local requires a pointer byte-range fact",
        source_op_index=op.index,
        source_value_id=value_id,
    )


def _memdesc_infos(source_program):
    result = {}
    for value_id, value in source_program.values.items():
        if value.type.kind != "memdesc":
            continue
        result[value_id] = MemdescInfo(
            value_id,
            value.type.element_type,
            value.type.element_byte_width,
            tuple(value.type.shape),
            tuple(value.type.alloc_shape),
            _memdesc_size_bytes(value.type, value.owner_op_index, value_id),
        )
    return result


def _compute_memdesc_view_infos(source_values, ops, kernel_arg_ids, memdescs):
    result = {}

    def base_view(value_id):
        memdesc = _memdesc_info_from_table(memdescs, value_id, None)
        physical_shape = tuple(int(dim) for dim in (memdesc.alloc_shape or memdesc.shape))
        return MemdescViewInfo(
            int(value_id),
            tuple(0 for _ in physical_shape),
            physical_shape,
        )

    for value_id in kernel_arg_ids:
        if value_id in memdescs:
            result[int(value_id)] = base_view(value_id)

    for op in ops:
        memdesc_results = tuple(result_id for result_id in op.results if result_id in memdescs)
        if not memdesc_results:
            continue
        if op.name in {"ttg.local_alloc", "ttg.memdesc_index"}:
            for result_id in memdesc_results:
                result[int(result_id)] = base_view(result_id)
            continue
        if op.name == "ttg.memdesc_reshape":
            if len(op.operands) != 1 or len(op.results) != 1:
                fail(
                    "TLXW_OP_MEMDESC_RESHAPE",
                    STAGE,
                    "ttg.memdesc_reshape requires one memdesc operand and one result",
                    source_op_index=op.index,
                )
            parent_id = int(op.operands[0])
            result_id = int(op.results[0])
            parent_view = result.get(parent_id)
            if parent_view is None:
                fail(
                    "TLXW_OP_MEMDESC_RESHAPE",
                    STAGE,
                    "ttg.memdesc_reshape parent view is not structurally resolvable",
                    source_op_index=op.index,
                    source_value_id=parent_id,
                )
            parent = _memdesc_info_from_table(memdescs, parent_id, op)
            child = _memdesc_info_from_table(memdescs, result_id, op)
            parent_type = source_values[parent_id].type
            child_type = source_values[result_id].type
            if (parent.element_type != child.element_type
                    or parent.element_byte_width != child.element_byte_width
                    or parent_type.memory_space != child_type.memory_space
                    or parent_type.mutable != child_type.mutable):
                fail(
                    "TLXW_OP_MEMDESC_RESHAPE",
                    STAGE,
                    "ttg.memdesc_reshape must preserve element type, memory "
                    "space, and mutability",
                    source_op_index=op.index,
                    source_value_id=result_id,
                )
            if any(int(origin) for origin in parent_view.logical_origin):
                fail(
                    "TLXW_OP_MEMDESC_RESHAPE",
                    STAGE,
                    "ttg.memdesc_reshape of a nonzero-origin view requires an "
                    "explicit structural origin map",
                    source_op_index=op.index,
                    source_value_id=result_id,
                )
            child_physical_shape = tuple(int(dim) for dim in (child.alloc_shape or child.shape))
            if (_product(parent.shape) != _product(child.shape)
                    or _product(parent_view.physical_shape) != _product(child_physical_shape)):
                fail(
                    "TLXW_OP_MEMDESC_RESHAPE",
                    STAGE,
                    "ttg.memdesc_reshape must preserve logical and physical "
                    "element counts",
                    source_op_index=op.index,
                    source_value_id=result_id,
                )
            result[result_id] = MemdescViewInfo(
                result_id,
                tuple(0 for _ in child_physical_shape),
                child_physical_shape,
            )
            continue
        if op.name == "ttg.memdesc_subslice":
            if len(op.operands) != 1 or len(op.results) != 1:
                fail(
                    "TLXW_OP_MEMDESC_SUBSLICE",
                    STAGE,
                    "ttg.memdesc_subslice requires one memdesc operand and one result",
                    source_op_index=op.index,
                )
            parent_id = int(op.operands[0])
            result_id = int(op.results[0])
            parent_view = result.get(parent_id)
            if parent_view is None:
                fail(
                    "TLXW_OP_MEMDESC_SUBSLICE",
                    STAGE,
                    "ttg.memdesc_subslice parent view is not structurally resolvable",
                    source_op_index=op.index,
                    source_value_id=parent_id,
                )
            parent = _memdesc_info_from_table(memdescs, parent_id, op)
            child = _memdesc_info_from_table(memdescs, result_id, op)
            offsets = op.attrs.get("offsets")
            if not isinstance(offsets, (tuple, list)):
                fail(
                    "TLXW_OP_MEMDESC_SUBSLICE",
                    STAGE,
                    "ttg.memdesc_subslice requires structural integer offsets",
                    source_op_index=op.index,
                    source_value_id=result_id,
                )
            offsets = tuple(int(offset) for offset in offsets)
            parent_shape = tuple(int(dim) for dim in parent.shape)
            child_shape = tuple(int(dim) for dim in child.shape)
            rank = len(parent_shape)
            if len(offsets) != rank or len(child_shape) != rank:
                fail(
                    "TLXW_OP_MEMDESC_SUBSLICE",
                    STAGE,
                    "ttg.memdesc_subslice offset/result ranks must match the parent rank",
                    source_op_index=op.index,
                    source_value_id=result_id,
                )
            if (parent.element_type != child.element_type
                    or parent.element_byte_width != child.element_byte_width):
                fail(
                    "TLXW_OP_MEMDESC_SUBSLICE",
                    STAGE,
                    "ttg.memdesc_subslice must preserve the memdesc element type",
                    source_op_index=op.index,
                    source_value_id=result_id,
                )
            parent_type = source_values[parent_id].type
            child_type = source_values[result_id].type
            if (parent_type.encoding != child_type.encoding
                    or parent_type.memory_space != child_type.memory_space
                    or parent_type.mutable != child_type.mutable):
                fail(
                    "TLXW_OP_MEMDESC_SUBSLICE",
                    STAGE,
                    "ttg.memdesc_subslice must preserve layout, memory space, and mutability",
                    source_op_index=op.index,
                    source_value_id=result_id,
                )
            for offset, child_extent, parent_extent in zip(offsets, child_shape, parent_shape):
                if int(offset) < 0 or int(child_extent) <= 0 or int(offset) + int(child_extent) > int(parent_extent):
                    fail(
                        "TLXW_OP_MEMDESC_SUBSLICE",
                        STAGE,
                        "ttg.memdesc_subslice lies outside the parent logical shape",
                        source_op_index=op.index,
                        source_value_id=result_id,
                    )
            physical_shape = tuple(int(dim) for dim in parent_view.physical_shape)
            child_alloc_shape = tuple(int(dim) for dim in child.alloc_shape)
            if child_alloc_shape and child_alloc_shape != physical_shape:
                fail(
                    "TLXW_OP_MEMDESC_SUBSLICE",
                    STAGE,
                    "ttg.memdesc_subslice allocation shape must preserve the parent allocation",
                    source_op_index=op.index,
                    source_value_id=result_id,
                )
            logical_origin = tuple(
                int(origin) + int(offset)
                for origin, offset in zip(parent_view.logical_origin, offsets)
            )
            result[result_id] = MemdescViewInfo(
                result_id,
                logical_origin,
                physical_shape,
            )
            continue
        if op.name == "ttg.memdesc_trans" and len(op.operands) == 1 and len(op.results) == 1:
            parent_view = result.get(int(op.operands[0]))
            if parent_view is None:
                result[int(op.results[0])] = None
                continue
            result_id = int(op.results[0])
            child = _memdesc_info_from_table(memdescs, result_id, op)
            physical_shape = tuple(reversed(parent_view.physical_shape))
            logical_origin = tuple(reversed(parent_view.logical_origin))
            child_alloc_shape = tuple(int(dim) for dim in child.alloc_shape)
            if child_alloc_shape and child_alloc_shape != physical_shape:
                fail(
                    "TLXW_OP_MEMDESC_TRANS",
                    STAGE,
                    "ttg.memdesc_trans allocation shape is not the reversed parent allocation",
                    source_op_index=op.index,
                    source_value_id=result_id,
                )
            result[result_id] = MemdescViewInfo(
                result_id,
                logical_origin,
                physical_shape,
            )
            continue
        if op.name == "arith.select" and len(op.operands) == 3 and len(op.results) == 1:
            lhs = result.get(int(op.operands[1]))
            rhs = result.get(int(op.operands[2]))
            if (lhs is not None and rhs is not None
                    and lhs.logical_origin == rhs.logical_origin
                    and lhs.physical_shape == rhs.physical_shape):
                result[int(op.results[0])] = MemdescViewInfo(
                    int(op.results[0]),
                    lhs.logical_origin,
                    lhs.physical_shape,
                )
            else:
                result[int(op.results[0])] = None
            continue
        for result_id in memdesc_results:
            result[int(result_id)] = None
    return result


def _constant_ints(source_program):
    result = {}
    for op in source_program.ops:
        if op.name != "arith.constant" or len(op.results) != 1:
            continue
        literal = _constant_literal(
            op.attrs.get("value"),
            source_op_index=op.index,
            element_type=source_program.values[op.results[0]].type.element_type,
        )
        if type(literal) is int:
            result[op.results[0]] = literal
    return result


def _compute_memdesc_physical_allocation_bytes(
    source_values,
    ops,
    type_layout_program,
    memdescs,
):
    ops_by_index = {op.index: op for op in ops}
    indexed_children_by_parent = _indexed_memdesc_children_by_parent(ops)
    result = {}
    visiting = set()

    def compute(value_id):
        value_id = int(value_id)
        if value_id in result:
            return result[value_id]
        if value_id in visiting:
            fail(
                "TLXW_OP_MEMDESC_INDEX",
                STAGE,
                "cyclic ttg.memdesc_index relationship while sizing local memory",
                source_value_id=value_id,
            )
        visiting.add(value_id)
        memdesc = _memdesc_info_from_table(memdescs, value_id, None)
        value = source_values.get(value_id)
        op = (None if value is None or value.owner_op_index is None else ops_by_index.get(int(value.owner_op_index)))
        children = indexed_children_by_parent.get(value_id)
        if children:
            size = _indexed_memdesc_parent_allocation_bytes(
                memdesc,
                _layout_for_value(type_layout_program, value_id),
                children,
                memdescs,
                lambda child_value_id, _child_memdesc: compute(child_value_id),
            )
        else:
            size = _memdesc_physical_allocation_bytes(
                memdesc,
                _layout_for_value(type_layout_program, value_id),
                op,
            )
        visiting.remove(value_id)
        result[value_id] = int(size)
        return result[value_id]

    for value_id in memdescs:
        compute(value_id)
    return result


def _indexed_memdesc_children_by_parent(ops):
    indexed_children_by_parent = {}
    for op in ops:
        if op.name != "ttg.memdesc_index" or len(op.operands) != 2 or len(op.results) != 1:
            continue
        indexed_children_by_parent.setdefault(op.operands[0], []).append((op, op.results[0]))
    return indexed_children_by_parent


def _layout_for_value(type_layout_program, value_id):
    converted = type_layout_program.values.get(value_id)
    if converted is None or converted.layout_map_id is None:
        return None
    return type_layout_program.layouts[int(converted.layout_map_id)]


def _memdesc_physical_allocation_bytes(memdesc, layout, op):
    dense_size = int(memdesc.allocation_bytes)
    if layout is None or layout.kind in {
            "none",
            "linear",
            "shared_linear",
            "swizzled_shared",
    }:
        return dense_size
    if layout.kind != "padded_shared":
        return dense_size
    element_byte_width = memdesc.element_byte_width
    shape = tuple(int(dim) for dim in (memdesc.alloc_shape or memdesc.shape or ()))
    if element_byte_width is None or not shape:
        return dense_size
    element_count = _product(shape)
    if element_count <= 0:
        return dense_size
    last_offset = _static_shared_byte_offset_from_linear(
        layout,
        shape,
        element_count - 1,
        int(element_byte_width),
        op,
    )
    if last_offset is None:
        return dense_size
    return _align_to(max(dense_size, int(last_offset) + int(element_byte_width)), 16)


def _compute_memdesc_index_slot_stride_bytes(
    ops,
    type_layout_program,
    memdescs,
    memdesc_physical_allocation_bytes,
):
    result = {}
    for op in ops:
        if op.name != "ttg.memdesc_index" or len(op.operands) != 2 or len(op.results) != 1:
            continue
        child_value_id = op.results[0]
        child_memdesc = _memdesc_info_from_table(memdescs, child_value_id, op)
        child_size = int(
            memdesc_physical_allocation_bytes.get(
                child_value_id,
                child_memdesc.allocation_bytes,
            ))
        stride = _memdesc_index_parent_slot_stride_bytes(
            _memdesc_info_from_table(memdescs, op.operands[0], op),
            child_memdesc,
            _layout_for_value(type_layout_program, op.operands[0]),
            op,
        )
        result[child_value_id] = int(child_size if stride is None else stride)
    return result


def _memdesc_index_parent_slot_stride_bytes(parent_memdesc, child_memdesc, parent_layout, op):
    offsets = _memdesc_index_parent_slot_offsets(parent_memdesc, child_memdesc, parent_layout, op)
    if offsets is None or len(offsets) <= 1:
        return None
    stride = int(offsets[1]) - int(offsets[0])
    if stride <= 0:
        return None
    previous = int(offsets[0])
    for offset in offsets[1:]:
        offset = int(offset)
        if offset - previous != stride:
            return None
        previous = offset
    return int(stride)


def _memdesc_index_slot_count(parent_memdesc, child_memdesc):
    parent_shape = tuple(int(dim) for dim in (parent_memdesc.alloc_shape or parent_memdesc.shape or ()))
    child_shape = tuple(int(dim) for dim in (child_memdesc.alloc_shape or child_memdesc.shape or ()))
    if not parent_shape or not child_shape:
        return None
    parent_elements = _product(parent_shape)
    child_elements = _product(child_shape)
    if child_elements <= 0 or parent_elements <= 0 or parent_elements % child_elements:
        return None
    slot_count = parent_elements // child_elements
    return int(slot_count) if slot_count > 0 else None


def _memdesc_index_parent_slot_offsets(parent_memdesc, child_memdesc, parent_layout, op):
    element_byte_width = child_memdesc.element_byte_width
    if element_byte_width is None:
        return None
    parent_shape = tuple(int(dim) for dim in (parent_memdesc.alloc_shape or parent_memdesc.shape or ()))
    child_shape = tuple(int(dim) for dim in (child_memdesc.alloc_shape or child_memdesc.shape or ()))
    if not parent_shape or not child_shape:
        return None
    parent_elements = _product(parent_shape)
    child_elements = _product(child_shape)
    if child_elements <= 0 or parent_elements <= 0 or parent_elements % child_elements:
        return None
    slot_count = parent_elements // child_elements
    offsets = []
    for slot in range(int(slot_count)):
        linear = int(slot) * int(child_elements)
        offset = _memdesc_index_parent_slot_byte_offset(
            parent_layout,
            parent_shape,
            linear,
            int(element_byte_width),
            op,
        )
        if offset is None:
            return None
        offsets.append(int(offset))
    return tuple(offsets)


def _memdesc_index_parent_slot_byte_offset(parent_layout, parent_shape, linear, element_byte_width, op):
    if parent_layout is not None and parent_layout.kind == "padded_shared":
        try:
            intervals, paddings = layouts.padded_shared_parameters(
                parent_layout,
                stage=STAGE,
                diagnostic="TLXW_OP_MEMDESC_INDEX",
                source_op_index=op.index,
                source_value_id=parent_layout.value_id,
            )
        except Diagnostic:
            return None
        element_offset = int(linear)
        for interval, padding in zip(intervals, paddings):
            element_offset += (int(linear) // int(interval)) * int(padding)
        return int(element_offset) * int(element_byte_width)
    try:
        return _static_shared_byte_offset_from_linear(
            parent_layout,
            parent_shape,
            int(linear),
            int(element_byte_width),
            op,
            diagnostic="TLXW_OP_MEMDESC_INDEX",
        )
    except Diagnostic:
        return None


def _compute_local_alloc_allocation_bytes(
    ops,
    type_layout_program,
    memdescs,
    memdesc_physical_allocation_bytes,
    memdesc_index_slot_stride_bytes,
):
    indexed_children_by_parent = _indexed_memdesc_children_by_parent(ops)

    result = {}
    for op in ops:
        if op.name != "ttg.local_alloc" or not op.results:
            continue
        value_id = op.results[0]
        memdesc = _memdesc_info_from_table(memdescs, value_id, op)
        children = indexed_children_by_parent.get(value_id)
        if not children:
            result[value_id] = int(memdesc_physical_allocation_bytes.get(
                value_id,
                memdesc.allocation_bytes,
            ))
            continue

        result[value_id] = _indexed_memdesc_parent_allocation_bytes(
            memdesc,
            _layout_for_value(type_layout_program, value_id),
            children,
            memdescs,
            lambda child_value_id, child_memdesc: memdesc_physical_allocation_bytes.get(
                child_value_id,
                child_memdesc.allocation_bytes,
            ),
            lambda child_value_id, child_memdesc: memdesc_index_slot_stride_bytes.get(
                child_value_id,
                child_memdesc.allocation_bytes,
            ),
        )
    return result


def _indexed_memdesc_parent_allocation_bytes(
    parent_memdesc,
    parent_layout,
    children,
    memdescs,
    child_size_fn,
    child_stride_fn=None,
):
    parent_elements = _product(parent_memdesc.alloc_shape or parent_memdesc.shape or (1, ))
    child_slot_elements = None
    child_slot_bytes = None
    child_slot_stride_bytes = None
    first_child = None
    for child_op, child_value_id in children:
        child_memdesc = _memdesc_info_from_table(memdescs, child_value_id, child_op)
        child_elements = _product(child_memdesc.alloc_shape or child_memdesc.shape or (1, ))
        if child_elements <= 0 or parent_elements % child_elements:
            fail(
                "TLXW_OP_MEMDESC_INDEX",
                STAGE,
                "ttg.memdesc_index child shape does not evenly tile the "
                "local allocation",
                source_op_index=child_op.index,
                source_value_id=child_value_id,
            )
        if child_slot_elements is None:
            child_slot_elements = int(child_elements)
        elif child_slot_elements != int(child_elements):
            fail(
                "TLXW_OP_MEMDESC_INDEX",
                STAGE,
                "ttg.memdesc_index children for a local allocation must "
                "have matching slot sizes",
                source_op_index=child_op.index,
                source_value_id=child_value_id,
            )
        child_size = int(child_size_fn(child_value_id, child_memdesc))
        child_slot_bytes = child_size if child_slot_bytes is None else max(int(child_slot_bytes), child_size)
        if first_child is None:
            first_child = (child_op, child_memdesc)
        if child_stride_fn is not None:
            child_stride = int(child_stride_fn(child_value_id, child_memdesc))
            child_slot_stride_bytes = (child_stride if child_slot_stride_bytes is None else max(
                int(child_slot_stride_bytes),
                child_stride,
            ))

    if child_slot_elements is None or child_slot_bytes is None:
        return int(parent_memdesc.allocation_bytes)
    slot_count = parent_elements // int(child_slot_elements)
    if first_child is not None:
        child_op, child_memdesc = first_child
        slot_offsets = _memdesc_index_parent_slot_offsets(parent_memdesc, child_memdesc, parent_layout, child_op)
        if slot_offsets is not None and len(slot_offsets) == int(slot_count):
            return _align_to(max(int(offset) + int(child_slot_bytes) for offset in slot_offsets), 16)
    if child_slot_stride_bytes is not None:
        return _align_to((max(1, int(slot_count)) - 1) * int(child_slot_stride_bytes) + int(child_slot_bytes), 16)
    return _align_to(slot_count * int(child_slot_bytes), 16)


def _compute_static_memdesc_byte_offsets(
    ops,
    memdescs,
    memdesc_index_slot_stride_bytes,
    constant_ints,
):
    cumulative_offsets = {
        op.results[0]: 0
        for op in ops
        if op.name == "ttg.local_alloc" and op.results
    }
    result_offsets = {}
    for op in ops:
        if op.name != "ttg.memdesc_index" or len(op.operands) != 2 or len(op.results) != 1:
            continue
        base_offset = cumulative_offsets.get(op.operands[0])
        static_index = constant_ints.get(op.operands[1])
        if base_offset is None or static_index is None:
            continue
        slot_size = _memdesc_info_from_table(
            memdescs,
            op.results[0],
            op,
        ).allocation_bytes
        slot_size = memdesc_index_slot_stride_bytes.get(op.results[0], slot_size)
        relative_offset = int(static_index) * int(slot_size)
        result_offsets[op.results[0]] = relative_offset
        cumulative_offsets[op.results[0]] = int(base_offset) + int(relative_offset)
    return result_offsets


def _memdesc_size_bytes(source_type, source_op_index, source_value_id):
    element_byte_width = source_type.element_byte_width
    if element_byte_width is None:
        fail(
            "TLXW_OP_MEMDESC_ELEMENT_SIZE",
            STAGE,
            f"cannot size LDS allocation {source_type.raw}: unknown element byte width",
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    return _product(source_type.alloc_shape or source_type.shape or (1, )) * int(element_byte_width)


def _memdesc_info(conversion_input, value_id, op):
    return _memdesc_info_from_table(conversion_input.memdescs, value_id, op)


def _memdesc_view_info(conversion_input, value_id, op):
    view = conversion_input.memdesc_views.get(int(value_id))
    if view is not None:
        return view
    fail(
        "TLXW_OP_MEMDESC_VIEW_INFO",
        STAGE,
        f"expected structurally resolvable memdesc view metadata for value {value_id}",
        source_op_index=op.index if op is not None else None,
        source_value_id=value_id,
    )


def _memdesc_info_from_table(memdescs, value_id, op):
    memdesc = memdescs.get(value_id)
    if memdesc is not None:
        return memdesc
    fail(
        "TLXW_OP_MEMDESC_INFO",
        STAGE,
        f"expected memdesc metadata for value {value_id}",
        source_op_index=op.index if op is not None else None,
        source_value_id=value_id,
    )


def _align_to(value, alignment):
    value = int(value)
    alignment = int(alignment)
    return ((value + alignment - 1) // alignment) * alignment


def _single_source_target(builder, source_value_id, op):
    targets = builder.source_value_targets.get(source_value_id)
    if not targets:
        fail(
            "TLXW_OP_UNCONVERTED_OPERAND",
            STAGE,
            f"yielded value {source_value_id} has no converted target value",
            source_op_index=op.index,
            source_value_id=source_value_id,
        )
    if len(targets) != 1:
        fail(
            "TLXW_OP_MULTI_VALUE_OPERAND",
            STAGE,
            f"yielded value {source_value_id} maps to multiple target values {targets}",
            source_op_index=op.index,
            source_value_id=source_value_id,
        )
    return targets[0]


def _buffer_load_to_local_fields(op):
    segments = _operand_segments(op, 6, (1, 1, 1, 0, 0, 0))
    if int(segments[0]) != 1 or int(segments[1]) != 1 or int(segments[2]) != 1:
        fail(
            "TLXW_OP_MALFORMED_BUFFER_ASYNC",
            STAGE,
            "amdg.buffer_load_to_local requires destination, base, and offsets",
            source_op_index=op.index,
        )
    if any(int(segment) > 1 for segment in segments[3:]):
        fail(
            "TLXW_OP_MALFORMED_BUFFER_ASYNC",
            STAGE,
            "amdg.buffer_load_to_local optional segments must be scalar",
            source_op_index=op.index,
        )
    _require_operand_count(op, segments)
    base_index = int(segments[0])
    offset_index = base_index + int(segments[1])
    mask_index = offset_index + int(segments[2])
    other_index = mask_index + int(segments[3])
    stride_index = other_index + int(segments[4])
    return {
        "memdesc_value_id": op.operands[0],
        "base_value_id": op.operands[base_index],
        "offset_value_id": op.operands[offset_index],
        "mask_value_id": op.operands[mask_index] if int(segments[3]) else None,
        "other_value_id": op.operands[other_index] if int(segments[4]) else None,
        "stride_value_id": op.operands[stride_index] if int(segments[5]) else None,
        "cache": _int_attr_or_default(op.attrs, "cache", 1),
    }


def _load_fields(op):
    if len(op.results) != 1:
        fail(
            "TLXW_OP_MALFORMED_LOAD",
            STAGE,
            "tt.load requires one result",
            source_op_index=op.index,
        )
    if len(op.operands) not in (1, 2, 3):
        fail(
            "TLXW_OP_MALFORMED_LOAD",
            STAGE,
            "tt.load requires pointer plus optional mask/other operands",
            source_op_index=op.index,
        )
    return {
        "pointer_value_id": op.operands[0],
        "mask_value_id": op.operands[1] if len(op.operands) >= 2 else None,
        "other_value_id": op.operands[2] if len(op.operands) >= 3 else None,
    }


def _store_fields(op):
    if op.results:
        fail(
            "TLXW_OP_MALFORMED_STORE",
            STAGE,
            "tt.store must not produce results",
            source_op_index=op.index,
        )
    if len(op.operands) not in (2, 3):
        fail(
            "TLXW_OP_MALFORMED_STORE",
            STAGE,
            "tt.store requires pointer, value, and optional mask operands",
            source_op_index=op.index,
        )
    return {
        "pointer_value_id": op.operands[0],
        "value_value_id": op.operands[1],
        "mask_value_id": op.operands[2] if len(op.operands) == 3 else None,
    }


def _buffer_load_fields(op):
    segments = _operand_segments(op, 5, None)
    if int(segments[0]) != 1 or int(segments[1]) != 1:
        fail(
            "TLXW_OP_MALFORMED_BUFFER_LOAD",
            STAGE,
            "amdg.buffer_load requires base pointer and offsets operands",
            source_op_index=op.index,
        )
    if int(segments[2]) not in (0, 1):
        fail(
            "TLXW_OP_MALFORMED_BUFFER_LOAD",
            STAGE,
            "amdg.buffer_load supports at most one stride operand",
            source_op_index=op.index,
        )
    if int(segments[3]) not in (0, 1) or int(segments[4]) not in (0, 1):
        fail(
            "TLXW_OP_MALFORMED_BUFFER_LOAD",
            STAGE,
            "amdg.buffer_load supports at most one mask and one other operand",
            source_op_index=op.index,
        )
    _require_operand_count(op, segments)
    offset_index = int(segments[0])
    stride_index = offset_index + int(segments[1])
    mask_index = stride_index + int(segments[2])
    other_index = mask_index + int(segments[3])
    return {
        "base_value_id": op.operands[0],
        "offset_value_id": op.operands[offset_index],
        "stride_value_id": op.operands[stride_index] if int(segments[2]) else None,
        "mask_value_id": op.operands[mask_index] if int(segments[3]) else None,
        "other_value_id": op.operands[other_index] if int(segments[4]) else None,
        "cache": _int_attr_or_default(op.attrs, "cache", 1),
        "contiguity": _int_attr_or_default(op.attrs, "contiguity", 1),
    }


def _buffer_store_fields(op):
    segments = _operand_segments(op, 5, None)
    if int(segments[0]) != 1 or int(segments[1]) != 1 or int(segments[2]) != 1:
        fail(
            "TLXW_OP_MALFORMED_BUFFER_STORE",
            STAGE,
            "amdg.buffer_store requires value, base pointer, and offsets",
            source_op_index=op.index,
        )
    if int(segments[3]) not in (0, 1):
        fail(
            "TLXW_OP_MALFORMED_BUFFER_STORE",
            STAGE,
            "amdg.buffer_store supports at most one boundary-check operand",
            source_op_index=op.index,
        )
    if int(segments[4]) not in (0, 1):
        fail(
            "TLXW_OP_MALFORMED_BUFFER_STORE",
            STAGE,
            "amdg.buffer_store supports at most one mask operand",
            source_op_index=op.index,
        )
    _require_operand_count(op, segments)
    base_index = int(segments[0])
    offset_index = base_index + int(segments[1])
    mask_index = offset_index + int(segments[2]) + int(segments[3])
    boundary_index = offset_index + int(segments[2])
    return {
        "value_value_id": op.operands[0],
        "base_value_id": op.operands[base_index],
        "offset_value_id": op.operands[offset_index],
        "boundary_check_value_id": op.operands[boundary_index] if int(segments[3]) else None,
        "mask_value_id": op.operands[mask_index] if int(segments[4]) else None,
        "cache": _int_attr_or_default(op.attrs, "cache", 1),
        "contiguity": _int_attr_or_default(op.attrs, "contiguity", 1),
    }


def _operand_segments(op, expected_len, default):
    segments = op.attrs.get("operandSegmentSizes")
    if segments is None:
        segments = default
    segments = tuple(int(segment) for segment in segments)
    if len(segments) != int(expected_len):
        fail(
            "TLXW_OP_MALFORMED_OPERAND_SEGMENTS",
            STAGE,
            f"{op.name} expected {expected_len} operand segments, got {segments}",
            source_op_index=op.index,
        )
    if any(segment < 0 for segment in segments):
        fail(
            "TLXW_OP_MALFORMED_OPERAND_SEGMENTS",
            STAGE,
            f"{op.name} operand segments must be nonnegative, got {segments}",
            source_op_index=op.index,
        )
    return segments


def _require_operand_count(op, segments):
    if sum(int(segment) for segment in segments) != len(op.operands):
        fail(
            "TLXW_OP_MALFORMED_OPERAND_SEGMENTS",
            STAGE,
            f"{op.name} operand segments {segments} do not match "
            f"{len(op.operands)} operands",
            source_op_index=op.index,
        )


def _require_default_cache(cache, op):
    if cache in (None, 1):
        return
    fail(
        "TLXW_OP_UNSUPPORTED_CACHE_MODIFIER",
        STAGE,
        f"Wave lowering does not support {op.name} cacheModifier={cache}",
        source_op_index=op.index,
    )


def _require_supported_direct_buffer_cache(cache, op, *, is_store):
    supported = {1, 3, 4, 5, 6} if is_store else {1, 2, 3, 5, 7}
    if cache is None or cache in supported:
        return
    fail(
        "TLXW_OP_UNSUPPORTED_CACHE_MODIFIER",
        STAGE,
        f"Wave lowering does not support {op.name} cacheModifier={cache}",
        source_op_index=op.index,
    )


def _require_default_tt_memory_attrs(op):
    cache = op.attrs.get("cache")
    if cache is not None and int(cache) != 1:
        fail(
            "TLXW_OP_UNSUPPORTED_CACHE_MODIFIER",
            STAGE,
            f"Wave lowering does not support {op.name} cache={cache}",
            source_op_index=op.index,
        )
    cache_modifier = _attr_text(op.attrs.get("cacheModifier"))
    if cache_modifier not in {"", "none", "#tt.cache_modifier<none>"}:
        fail(
            "TLXW_OP_UNSUPPORTED_CACHE_MODIFIER",
            STAGE,
            f"Wave lowering does not support {op.name} cacheModifier={cache_modifier}",
            source_op_index=op.index,
        )
    evict = op.attrs.get("evict")
    if evict is not None and int(evict) != 1:
        fail(
            "TLXW_OP_UNSUPPORTED_EVICTION_POLICY",
            STAGE,
            f"Wave lowering does not support {op.name} evict={evict}",
            source_op_index=op.index,
        )
    eviction_policy = _attr_text(op.attrs.get("evictionPolicy"))
    if eviction_policy not in {"", "none", "evict_normal", "#tt.eviction_policy<normal>"}:
        fail(
            "TLXW_OP_UNSUPPORTED_EVICTION_POLICY",
            STAGE,
            f"Wave lowering does not support {op.name} evictionPolicy={eviction_policy}",
            source_op_index=op.index,
        )
    if _attr_bool(op.attrs.get("isVolatile")):
        fail(
            "TLXW_OP_UNSUPPORTED_VOLATILE",
            STAGE,
            f"Wave lowering does not support volatile {op.name}",
            source_op_index=op.index,
        )


def _attr_text(value):
    if value is None:
        return ""
    return str(value).strip().strip('"')


def _attr_bool(value):
    text = _attr_text(value).lower()
    return text in {"true", "1"}


def _local_component_store_plan(
    conversion_input,
    type_layout_program,
    memdesc_value_id,
    offset_value_id,
    component_count,
    lane_width,
    op,
):
    memdesc = _memdesc_info(conversion_input, memdesc_value_id, op)
    view = _memdesc_view_info(conversion_input, memdesc_value_id, op)
    shape = tuple(int(dim) for dim in (memdesc.shape or memdesc.alloc_shape))
    total_elements = _product(shape)
    wave_count = max(1, int(conversion_input.num_warps))
    if int(component_count) * int(lane_width) * wave_count != total_elements:
        fail(
            "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
            STAGE,
            "scalarized amdg.buffer_load_to_local currently requires "
            "all-active full per-wave components",
            source_op_index=op.index,
            source_value_id=memdesc_value_id,
        )
    memdesc_layout_id = type_layout_program.values[memdesc_value_id].layout_map_id
    memdesc_layout = (None if memdesc_layout_id is None else type_layout_program.layouts[int(memdesc_layout_id)])
    offset_layout_id = type_layout_program.values[offset_value_id].layout_map_id
    offset_layout = (None if offset_layout_id is None else type_layout_program.layouts[int(offset_layout_id)])
    if (offset_layout is None or offset_layout.kind not in {"blocked", "linear", "generic_linear"}
            or len(offset_layout.shape) != len(shape)):
        fail(
            "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
            STAGE,
            "scalarized amdg.buffer_load_to_local requires a structural "
            "distributed offset layout for local destination mapping",
            source_op_index=op.index,
            source_value_id=offset_value_id,
        )
    linear = layouts.distributed_linear_layout(
        offset_layout,
        stage=STAGE,
        source_op_index=op.index,
    )
    if not linear.is_injective():
        fail(
            "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
            STAGE,
            "scalarized amdg.buffer_load_to_local requires an injective "
            "offset layout for local destination mapping",
            source_op_index=op.index,
            source_value_id=offset_value_id,
        )
    component_wave_offsets = []
    for component in range(int(component_count)):
        component_offsets = []
        for wave in range(wave_count):
            component_offsets.append(
                tuple(
                    _local_physical_offset_for_distributed_slot(
                        memdesc_layout,
                        shape,
                        view.physical_shape,
                        view.logical_origin,
                        memdesc.element_byte_width,
                        linear,
                        component,
                        lane,
                        wave,
                        op,
                        offset_value_id,
                    ) for lane in range(int(lane_width))))
        component_wave_offsets.append(tuple(component_offsets))
    affine_plan = _try_affine_local_component_store_plan(
        component_wave_offsets,
        int(lane_width),
        wave_count,
    )
    if affine_plan is not None:
        return affine_plan
    return _coordinate_local_component_store_plan(
        offset_layout,
        memdesc_layout,
        shape,
        view.physical_shape,
        view.logical_origin,
        memdesc.element_byte_width,
        int(component_count),
        int(lane_width),
        wave_count,
        op,
        offset_value_id,
    )


def _try_affine_local_component_store_plan(
    component_wave_offsets,
    lane_width,
    wave_count,
):
    component_offsets = []
    lane_stride = None
    wave_stride = None
    for wave_offsets_by_lane in component_wave_offsets:
        wave_offsets = []
        for lane_offsets in wave_offsets_by_lane:
            base = int(lane_offsets[0])
            current_lane_stride = (0 if int(lane_width) == 1 else int(lane_offsets[1]) - base)
            if any(int(offset) != base + lane * current_lane_stride for lane, offset in enumerate(lane_offsets)):
                return None
            if lane_stride is None:
                lane_stride = current_lane_stride
            elif lane_stride != current_lane_stride:
                return None
            wave_offsets.append(base)
        component_offsets.append(wave_offsets[0])
        current_wave_stride = (0 if wave_count == 1 else int(wave_offsets[1]) - int(wave_offsets[0]))
        if any(
                int(offset) != int(wave_offsets[0]) + wave * current_wave_stride
                for wave, offset in enumerate(wave_offsets)):
            return None
        if wave_stride is None:
            wave_stride = current_wave_stride
        elif wave_stride != current_wave_stride:
            return None
    return {
        "offset_mode": "affine",
        "component_offsets": tuple(int(offset) for offset in component_offsets),
        "lane_stride_elements": int(lane_stride if lane_stride is not None else 1),
        "wave_stride_elements": int(wave_stride or 0),
    }


def _coordinate_local_component_store_plan(
    offset_layout,
    memdesc_layout,
    shape,
    physical_shape,
    logical_origin,
    element_byte_width,
    component_count,
    lane_width,
    wave_count,
    op,
    offset_value_id,
):
    plan = coordinates.layout_coordinate_plan(
        offset_layout,
        int(component_count),
        int(lane_width),
        int(wave_count),
        op,
        offset_value_id,
    )
    return {
        "offset_mode":
        "layout_coordinates",
        "coordinate_shape":
        tuple(int(dim) for dim in plan.shape),
        "component_coordinate_bases":
        tuple(tuple(int(value) for value in bases) for bases in plan.component_bases),
        "workitem_coordinate_coefficients":
        tuple(tuple(int(value) for value in coefficients) for coefficients in plan.workitem_coefficients),
        "shared_layout_attrs":
        _scalarized_shared_layout_attrs(
            memdesc_layout,
            physical_shape,
            element_byte_width,
            op,
            logical_origin=logical_origin,
        ),
    }


def _scalarized_shared_layout_attrs(
    layout,
    physical_shape,
    element_byte_width,
    op,
    *,
    logical_origin=(),
):
    physical_shape = tuple(int(dim) for dim in physical_shape)
    logical_origin = tuple(int(value) for value in logical_origin)
    if not logical_origin:
        logical_origin = tuple(0 for _ in physical_shape)
    if len(logical_origin) != len(physical_shape):
        fail(
            "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            "shared logical origin rank does not match the physical allocation rank",
            source_op_index=op.index,
        )
    plan = layouts.shared_physical_offset_expression_plan(
        layout,
        physical_shape,
        element_byte_width,
        stage=STAGE,
        diagnostic="TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
        source_op_index=op.index,
        source_value_id=None if layout is None else layout.value_id,
    )
    return {
        **layouts.physical_offset_expression_plan_attrs(plan, "destination"),
        "destination_logical_origin": logical_origin,
        "destination_physical_shape": physical_shape,
    }


def _local_physical_offset_for_distributed_slot(
    memdesc_layout,
    shape,
    physical_shape,
    logical_origin,
    element_byte_width,
    distributed_layout,
    component,
    lane,
    wave,
    op,
    source_value_id,
):
    coords = layouts.linear_layout_coords(
        distributed_layout,
        int(component),
        int(lane),
        warp=int(wave),
    )
    if len(coords) != len(shape):
        fail(
            "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
            STAGE,
            "distributed offset layout rank does not match local memdesc rank",
            source_op_index=op.index,
            source_value_id=source_value_id,
        )
    for coord, extent in zip(coords, shape):
        if int(coord) < 0 or int(coord) >= int(extent):
            fail(
                "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
                STAGE,
                "distributed offset layout maps a component outside the "
                "local memdesc shape",
                source_op_index=op.index,
                source_value_id=source_value_id,
            )
    physical_coords = tuple(
        int(origin) + int(coord)
        for origin, coord in zip(logical_origin, coords)
    )
    byte_offset = _static_shared_byte_offset(
        memdesc_layout,
        physical_shape,
        physical_coords,
        int(element_byte_width),
        op,
        diagnostic="TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
    )
    if int(byte_offset) % int(element_byte_width):
        fail(
            "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
            STAGE,
            "local destination physical byte offset is not element aligned",
            source_op_index=op.index,
            source_value_id=source_value_id,
        )
    return int(byte_offset) // int(element_byte_width)


def _buffer_load_to_local_packet_plan(
    conversion_input,
    type_layout_program,
    fact_program,
    memdesc_value_id,
    offset_value_id,
    memdesc,
    lane_width,
    op,
):
    affine = fact_program.tensor_affine.get(offset_value_id)
    if affine is None:
        return None
    shape = tuple(int(dim) for dim in memdesc.shape)
    view = _memdesc_view_info(conversion_input, memdesc_value_id, op)
    physical_shape = tuple(int(dim) for dim in view.physical_shape)
    logical_origin = tuple(int(origin) for origin in view.logical_origin)
    if (len(shape) != len(physical_shape)
            or len(shape) != len(logical_origin)
            or any(int(origin) < 0 or int(origin) + int(extent) > int(physical_extent)
                   for origin, extent, physical_extent in zip(
                       logical_origin, shape, physical_shape))):
        return None
    if tuple(affine.shape) != shape or not shape:
        return None
    packet_byte_candidates = _dma_packet_byte_candidates(
        memdesc.element_byte_width,
        include_narrow=True,
    )
    if not packet_byte_candidates:
        return None
    total_elements = _product(shape)
    wave_count = int(conversion_input.num_warps)
    if wave_count <= 0:
        return None
    layout_id = type_layout_program.values[memdesc_value_id].layout_map_id
    layout = (None if layout_id is None else type_layout_program.layouts[int(layout_id)])
    packet_order = _shared_layout_physical_order(
        layout,
        physical_shape,
        op,
        diagnostic="TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
    )
    identity_view = (
        physical_shape == shape and not any(int(origin) for origin in logical_origin)
    )
    for packet_bytes in packet_byte_candidates:
        packet_elements = int(packet_bytes) // int(memdesc.element_byte_width)
        elements_per_wave_packet = int(lane_width) * int(packet_elements)
        elements_per_cta_packet = int(wave_count) * elements_per_wave_packet
        if total_elements % elements_per_cta_packet:
            continue
        component_count = total_elements // elements_per_cta_packet
        try:
            physical_linear_bases = _packet_physical_linear_component_bases(
                layout,
                physical_shape,
                op,
            )
            if physical_linear_bases is not None:
                identity_bases = layouts.identity_offset_bases(
                    physical_shape,
                    packet_order,
                )
                if (
                    not identity_view
                    and tuple(physical_linear_bases) == tuple(identity_bases)
                ):
                    # Padded encodings expose their order shorthand as a
                    # synthesized linearComponent.  Prove that it is exactly
                    # the ordered identity map before restricting a view with
                    # the structural coordinate path below.  Keep complete
                    # memdescs on the physical-linear path so an op-level
                    # contiguity contract can justify narrow packets even when
                    # affine axis facts do not recover source contiguity.
                    physical_linear_bases = None
            if physical_linear_bases is not None:
                # A restricted view of a non-identity linear component is not
                # generally a contiguous physical-linear interval.  Keep that
                # case on the proven fallback until the restriction itself is
                # represented as a linear relation.
                if not identity_view:
                    continue
                if _packet_physical_linear_source_is_contiguous(
                        affine,
                        shape,
                        physical_linear_bases,
                        packet_elements,
                        component_count,
                        wave_count,
                        lane_width,
                ):
                    source_contiguity_mode = "affine_proven"
                elif _int_attr_or_default(op.attrs, "contiguity", 1) >= packet_elements:
                    # Match the AMD LLVM lowering contract for
                    # amdg.buffer_load_to_local: the op-level contiguity hint
                    # may justify direct-to-LDS vectorization even when IR
                    # axis facts cannot reconstruct the contiguous source
                    # expression.
                    source_contiguity_mode = "op_contiguity"
                else:
                    continue
                destination_offsets, destination_wave_stride_dwords, destination_wave_offset_coefficients_dwords = (
                    _packet_physical_linear_destination_offsets(
                        layout,
                        shape,
                        component_count,
                        elements_per_wave_packet,
                        memdesc.element_byte_width,
                        wave_count,
                        op,
                    ))
                source_coordinate_mode = "physical_linear_component"
                source_linear_component_bases = physical_linear_bases
            else:
                if not _packet_source_is_contiguous(
                        affine,
                        shape,
                        packet_order,
                        packet_elements,
                        component_count,
                        wave_count,
                        lane_width,
                ):
                    continue
                destination_offsets, destination_wave_stride_dwords, destination_wave_offset_coefficients_dwords = (
                    _packet_destination_offsets(
                        layout,
                        shape,
                        physical_shape,
                        logical_origin,
                        packet_order,
                        component_count,
                        elements_per_cta_packet,
                        elements_per_wave_packet,
                        memdesc.element_byte_width,
                        wave_count,
                        op,
                    ))
                source_coordinate_mode = "ordered_linear"
                source_linear_component_bases = ()
                source_contiguity_mode = "affine_proven"
        except Diagnostic as diagnostic:
            if diagnostic.code != "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC":
                raise
            continue
        scalar_value_ids, terms = _packet_affine_terms(affine)
        return {
            "component_thread_count": int(wave_count) * int(lane_width),
            "component_count": int(component_count),
            "destination_component_offsets": destination_offsets,
            "destination_wave_count": int(wave_count),
            "destination_wave_offset_coefficients_dwords": tuple(destination_wave_offset_coefficients_dwords),
            "destination_wave_stride_dwords": int(destination_wave_stride_dwords),
            "packet_bytes": int(packet_bytes),
            "packet_elements": int(packet_elements),
            "packet_order": tuple(int(dim) for dim in packet_order),
            "source_affine": affine,
            "source_coordinate_mode": source_coordinate_mode,
            "source_contiguity_mode": source_contiguity_mode,
            "source_linear_component_bases": tuple(
                tuple(int(value) for value in basis) for basis in source_linear_component_bases),
            "scalar_value_ids": tuple(scalar_value_ids),
            "source_offset_terms": tuple(terms),
        }
    return None


def _buffer_load_to_local_packet_mask_is_supported(
    conversion_input,
    type_layout_program,
    fact_program,
    mask_value_id,
    packet_plan,
    memdesc,
    op,
):
    memdesc_layout_id = type_layout_program.values[memdesc.value_id].layout_map_id
    memdesc_layout = None if memdesc_layout_id is None else type_layout_program.layouts[int(memdesc_layout_id)]
    # Rank-1 padded shared layouts rely on scalarized stores preserving the
    # physical padding gaps under masks; keep that path until masked packet DMA
    # has an equivalent proof.
    if len(memdesc.shape) == 1 and memdesc_layout is not None and memdesc_layout.kind == "padded_shared":
        return False
    mask_value = type_layout_program.values[int(mask_value_id)]
    if int(mask_value.type.component_count) == 1:
        return True
    packet_elements = int(packet_plan["packet_elements"])
    if int(mask_value.type.component_count) != int(packet_plan["component_count"]) * packet_elements:
        return False
    producer_by_result = _producer_by_result(conversion_input)
    divisibility_memo = {}
    packet_memo = {}
    packet_coords = _buffer_load_to_local_packet_coordinate_groups(
        packet_plan,
        tuple(int(dim) for dim in memdesc.shape),
        op,
    )
    return _mask_packet_coordinate_predicate_is_safe(
        conversion_input,
        type_layout_program,
        fact_program,
        producer_by_result,
        divisibility_memo,
        packet_memo,
        int(mask_value_id),
        packet_coords,
        packet_elements,
    )


def _buffer_load_to_local_packet_scalar_component_sources(
    conversion_input,
    type_layout_program,
    packet_plan,
    scalar_value_ids,
    shape,
    lane_width,
    op,
):
    component_count = int(packet_plan["component_count"])
    component_thread_count = int(packet_plan["component_thread_count"])
    scalar_sources = []
    for source_value_id in scalar_value_ids:
        value = type_layout_program.values[int(source_value_id)]
        if int(value.type.component_count) == 1:
            scalar_sources.append(tuple(0 for _ in range(component_count)))
            continue
        if value.layout_map_id is None:
            return None
        component_sources = _buffer_load_to_local_packet_component_sources_from_value(
            conversion_input,
            type_layout_program,
            int(source_value_id),
            packet_plan,
            tuple(int(dim) for dim in shape),
            int(lane_width),
            component_count,
            component_thread_count,
            op,
        )
        if component_sources is None:
            return None
        scalar_sources.append(component_sources)
    return tuple(scalar_sources)


def _buffer_load_to_local_packet_component_sources_from_value(
    conversion_input,
    type_layout_program,
    source_value_id,
    packet_plan,
    shape,
    lane_width,
    component_count,
    component_thread_count,
    op,
):
    if int(lane_width) <= 0 or int(component_thread_count) % int(lane_width):
        return None
    packet_wave_count = int(component_thread_count) // int(lane_width)
    producer_by_result = _producer_by_result(conversion_input)
    coordinate_lookups = {}
    broadcast_component_sources = {}

    packet_elements = int(packet_plan["packet_elements"])
    coordinate_mode = packet_plan.get("source_coordinate_mode", "ordered_linear")
    component_sources = []
    for component in range(int(component_count)):
        sources = set()
        for workitem in range(int(component_thread_count)):
            linear = component * int(component_thread_count) * packet_elements + workitem * packet_elements
            coords = _buffer_load_to_local_packet_coordinate(
                packet_plan,
                shape,
                coordinate_mode,
                linear,
                op,
            )
            source_component = _packet_scalar_component_at_coordinate(
                type_layout_program,
                producer_by_result,
                int(source_value_id),
                int(workitem),
                coords,
                int(lane_width),
                int(packet_wave_count),
                op,
                coordinate_lookups,
                broadcast_component_sources,
                frozenset(),
            )
            if source_component is None:
                return None
            sources.add(int(source_component))
        if len(sources) != 1:
            return None
        component_sources.append(next(iter(sources)))
    return tuple(int(source) for source in component_sources)


def _packet_scalar_component_at_coordinate(
    type_layout_program,
    producer_by_result,
    source_value_id,
    workitem,
    coordinate,
    lane_width,
    wave_count,
    op,
    coordinate_lookups,
    broadcast_component_sources,
    visiting,
):
    """Resolve a packet affine leaf through structural broadcast replication."""
    source_value_id = int(source_value_id)
    if source_value_id in visiting:
        return None
    value = type_layout_program.values[int(source_value_id)]
    if int(value.type.component_count) == 1:
        return 0
    if value.layout_map_id is None:
        return None
    layout = type_layout_program.layouts[int(value.layout_map_id)]
    lookup = coordinate_lookups.get(source_value_id)
    if lookup is None:
        lookup = _layout_component_coordinate_lookup(
            layout,
            int(value.type.component_count),
            int(lane_width),
            int(wave_count),
            op,
            source_value_id,
        )
        if lookup is None:
            return None
        coordinate_lookups[source_value_id] = lookup
    projected = _project_coordinate_to_layout(layout, coordinate)
    candidates = () if projected is None else lookup.get(
        (int(workitem), projected),
        (),
    )

    producer = producer_by_result.get(source_value_id)
    component_sources = None
    if producer is not None and producer.name == "tt.broadcast":
        if source_value_id not in broadcast_component_sources:
            broadcast_component_sources[source_value_id] = (
                _broadcast_component_sources(
                    type_layout_program,
                    producer,
                )
            )
        component_sources = broadcast_component_sources[source_value_id]
    if candidates:
        candidate = min(int(component) for component in candidates)
        if component_sources is None:
            return candidate
        source_component = int(component_sources[candidate])
        equivalent = tuple(
            int(component) for component, mapped in enumerate(component_sources)
            if int(mapped) == source_component
        )
        return min(equivalent) if equivalent else None

    if component_sources is None or len(producer.operands) != 1:
        return None
    operand_component = _packet_scalar_component_at_coordinate(
        type_layout_program,
        producer_by_result,
        int(producer.operands[0]),
        int(workitem),
        coordinate,
        int(lane_width),
        int(wave_count),
        op,
        coordinate_lookups,
        broadcast_component_sources,
        visiting | {source_value_id},
    )
    if operand_component is None:
        return None
    equivalent = tuple(
        int(component) for component, mapped in enumerate(component_sources)
        if int(mapped) == int(operand_component)
    )
    return min(equivalent) if equivalent else None


def _layout_component_coordinate_lookup(
    layout,
    component_count,
    lane_width,
    wave_count,
    op,
    source_value_id,
):
    if layout.kind not in {
            "blocked",
            "linear",
            "generic_linear",
            "slice",
            "amd_mfma",
    }:
        return None
    if int(wave_count) > layouts.layout_warp_count(layout):
        return None
    try:
        linear = layouts.distributed_linear_layout(
            layout,
            stage=STAGE,
            source_op_index=op.index,
        )
        registers = layouts.linear_layout_component_registers(
            linear,
            layout,
            int(component_count),
            stage=STAGE,
            source_op_index=op.index,
            source_value_id=int(source_value_id),
        )
    except Diagnostic:
        return None
    result = {}
    for warp in range(int(wave_count)):
        for source_component, register in enumerate(registers):
            for lane in range(int(lane_width)):
                coords = tuple(int(coord) for coord in layouts.linear_layout_coords(
                    linear,
                    int(register),
                    int(lane),
                    warp=int(warp),
                ))
                result.setdefault(
                    (int(warp) * int(lane_width) + int(lane), coords),
                    [],
                ).append(int(source_component))
    return result


def _project_coordinate_to_layout(layout, coordinate):
    """Project a surrounding tensor coordinate into a broadcast leaf layout.

    Tensor-affine analysis deliberately retains scalar tensor leaves instead
    of expanding their broadcasted values.  A leaf may therefore have
    singleton dimensions, or a slice layout whose removed singleton dimension
    is still present in the surrounding affine tensor.  Recover that purely
    structural projection from the layout hierarchy.
    """
    return _project_coordinate_to_layout_parts(
        str(layout.kind),
        layout.properties,
        tuple(int(dim) for dim in layout.shape),
        tuple(int(value) for value in coordinate),
    )


def _project_coordinate_to_layout_parts(kind, properties, shape, coordinate):
    if len(coordinate) == len(shape):
        projected = []
        for extent, value in zip(shape, coordinate):
            if int(extent) == 1:
                projected.append(0)
            elif 0 <= int(value) < int(extent):
                projected.append(int(value))
            else:
                return None
        return tuple(projected)
    if kind != "slice" or len(coordinate) <= len(shape):
        return None
    dim = int(properties.get("dim", -1))
    if dim < 0 or dim > len(shape):
        return None
    parent_shape = list(int(extent) for extent in shape)
    parent_shape.insert(dim, 1)
    parent_coordinate = _project_coordinate_to_layout_parts(
        str(properties.get("parent_kind", "")),
        properties.get("parent_properties", {}),
        tuple(parent_shape),
        coordinate,
    )
    if parent_coordinate is None:
        return None
    return tuple(
        int(value) for index, value in enumerate(parent_coordinate)
        if int(index) != dim
    )


def _layout_coordinate_component_sources(
    result_layout,
    result_component_count,
    source_layout,
    source_component_count,
    lane_width,
    op,
    source_value_id,
):
    """Map result components to leaf components at identical logical points."""
    supported = {
        "amd_mfma",
        "blocked",
        "generic_linear",
        "linear",
        "slice",
    }
    if result_layout.kind not in supported or source_layout.kind not in supported:
        return None
    result_warps = layouts.layout_warp_count(result_layout)
    source_warps = layouts.layout_warp_count(source_layout)
    if int(result_warps) != int(source_warps):
        return None
    try:
        result_linear = layouts.distributed_linear_layout(
            result_layout,
            stage=STAGE,
            source_op_index=op.index,
        )
        source_linear = layouts.distributed_linear_layout(
            source_layout,
            stage=STAGE,
            source_op_index=op.index,
        )
        result_registers = layouts.linear_layout_component_registers(
            result_linear,
            result_layout,
            int(result_component_count),
            stage=STAGE,
            source_op_index=op.index,
        )
        source_registers = layouts.linear_layout_component_registers(
            source_linear,
            source_layout,
            int(source_component_count),
            stage=STAGE,
            source_op_index=op.index,
            source_value_id=int(source_value_id),
        )
    except Diagnostic:
        return None

    source_by_thread_coord = {}
    for warp in range(int(source_warps)):
        for lane in range(int(lane_width)):
            for source_component, register in enumerate(source_registers):
                source_coord = tuple(int(value) for value in layouts.linear_layout_coords(
                    source_linear,
                    int(register),
                    int(lane),
                    warp=int(warp),
                ))
                source_by_thread_coord.setdefault(
                    (int(warp), int(lane), source_coord),
                    [],
                ).append(int(source_component))

    component_sources = []
    for result_register in result_registers:
        sources = set()
        for warp in range(int(result_warps)):
            for lane in range(int(lane_width)):
                result_coord = layouts.linear_layout_coords(
                    result_linear,
                    int(result_register),
                    int(lane),
                    warp=int(warp),
                )
                source_coord = _project_coordinate_to_layout(
                    source_layout,
                    result_coord,
                )
                if source_coord is None:
                    return None
                candidates = source_by_thread_coord.get(
                    (int(warp), int(lane), source_coord),
                    (),
                )
                if not candidates:
                    return None
                sources.add(min(int(candidate) for candidate in candidates))
        if len(sources) != 1:
            return None
        component_sources.append(sources.pop())
    return tuple(int(source) for source in component_sources)


def _layout_affine_scalar_component_sources(
    type_layout_program,
    result_value,
    scalar_value_ids,
    lane_width,
    op,
):
    if result_value.layout_map_id is None:
        return None
    result_layout = type_layout_program.layouts[int(result_value.layout_map_id)]
    result_component_count = int(result_value.type.component_count)
    scalar_sources = []
    for source_value_id in scalar_value_ids:
        source_value = type_layout_program.values[int(source_value_id)]
        source_component_count = int(source_value.type.component_count)
        if source_component_count == 1:
            scalar_sources.append(tuple(0 for _ in range(result_component_count)))
            continue
        if source_value.layout_map_id is None:
            return None
        source_layout = type_layout_program.layouts[int(source_value.layout_map_id)]
        component_sources = _layout_coordinate_component_sources(
            result_layout,
            result_component_count,
            source_layout,
            source_component_count,
            int(lane_width),
            op,
            int(source_value_id),
        )
        if component_sources is None:
            return None
        scalar_sources.append(component_sources)
    return tuple(scalar_sources)


def _buffer_load_to_local_packet_coordinate_groups(packet_plan, shape, op):
    packet_elements = int(packet_plan["packet_elements"])
    component_count = int(packet_plan["component_count"])
    component_thread_count = int(packet_plan["component_thread_count"])
    coordinate_mode = packet_plan.get("source_coordinate_mode", "ordered_linear")
    groups = []
    for component in range(component_count):
        for workitem in range(component_thread_count):
            linear_start = component * component_thread_count * packet_elements + workitem * packet_elements
            groups.append(
                tuple(
                    _buffer_load_to_local_packet_coordinate(
                        packet_plan,
                        shape,
                        coordinate_mode,
                        linear_start + element,
                        op,
                    ) for element in range(packet_elements)))
    return tuple(groups)


def _buffer_load_to_local_packet_coordinate(packet_plan, shape, coordinate_mode, linear, op):
    if coordinate_mode == "ordered_linear":
        return _ordered_coords_from_linear(
            int(linear),
            shape,
            tuple(int(dim) for dim in packet_plan["packet_order"]),
        )
    if coordinate_mode == "physical_linear_component":
        return _coords_from_linear_component_offset(
            int(linear),
            tuple(tuple(int(value) for value in basis) for basis in packet_plan["source_linear_component_bases"]),
            len(shape),
        )
    fail(
        "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
        STAGE,
        f"unsupported packet mask coordinate mode {coordinate_mode}",
        source_op_index=op.index,
    )


def _mask_packet_coordinate_predicate_is_safe(
    conversion_input,
    type_layout_program,
    fact_program,
    producer_by_result,
    divisibility_memo,
    packet_memo,
    mask_value_id,
    packet_coords,
    packet_elements,
):
    key = (int(mask_value_id), id(packet_coords), int(packet_elements))
    if key in packet_memo:
        return packet_memo[key]
    source_op = producer_by_result.get(int(mask_value_id))
    result = False
    if source_op is None:
        result = False
    elif source_op.name == "arith.constant":
        result = True
    elif source_op.name == "arith.andi" and len(source_op.operands) == 2:
        result = all(
            _mask_packet_coordinate_predicate_is_safe(
                conversion_input,
                type_layout_program,
                fact_program,
                producer_by_result,
                divisibility_memo,
                packet_memo,
                operand,
                packet_coords,
                packet_elements,
            ) for operand in source_op.operands)
    elif source_op.name == "tt.broadcast":
        result = _broadcast_mask_packet_coordinate_predicate_is_safe(
            conversion_input,
            type_layout_program,
            fact_program,
            producer_by_result,
            divisibility_memo,
            packet_memo,
            source_op,
            packet_coords,
            packet_elements,
        )
    elif source_op.name == "arith.cmpi":
        result = _cmpi_mask_packet_coordinate_predicate_is_safe(
            conversion_input,
            fact_program,
            divisibility_memo,
            source_op,
            packet_coords,
            packet_elements,
        )
    packet_memo[key] = bool(result)
    return bool(result)


def _broadcast_mask_packet_coordinate_predicate_is_safe(
    conversion_input,
    type_layout_program,
    fact_program,
    producer_by_result,
    divisibility_memo,
    packet_memo,
    source_op,
    packet_coords,
    packet_elements,
):
    if len(source_op.operands) != 1 or len(source_op.results) != 1:
        return False
    operand = type_layout_program.values[source_op.operands[0]]
    result = type_layout_program.values[source_op.results[0]]
    if operand.layout_map_id is None or result.layout_map_id is None:
        return False
    operand_layout = type_layout_program.layouts[int(operand.layout_map_id)]
    result_layout = type_layout_program.layouts[int(result.layout_map_id)]
    if len(operand_layout.shape) != len(result_layout.shape):
        return False
    for source_extent, result_extent in zip(operand_layout.shape, result_layout.shape):
        if int(source_extent) not in {1, int(result_extent)}:
            return False
    projected_groups = tuple(
        tuple(
            tuple(
                0 if int(source_extent) == 1 else int(coord)
                for source_extent, coord in zip(operand_layout.shape, coords))
            for coords in group) for group in packet_coords)
    return _mask_packet_coordinate_predicate_is_safe(
        conversion_input,
        type_layout_program,
        fact_program,
        producer_by_result,
        divisibility_memo,
        packet_memo,
        source_op.operands[0],
        projected_groups,
        packet_elements,
    )


def _cmpi_mask_packet_coordinate_predicate_is_safe(
    conversion_input,
    fact_program,
    divisibility_memo,
    source_op,
    packet_coords,
    packet_elements,
):
    if len(source_op.operands) != 2:
        return False
    predicate = _cmpi_predicate(source_op.attrs.get("predicate"))
    lhs_affine = fact_program.tensor_affine.get(source_op.operands[0])
    rhs_affine = fact_program.tensor_affine.get(source_op.operands[1])
    if lhs_affine is None or rhs_affine is None:
        return False
    if tuple(lhs_affine.shape) != tuple(rhs_affine.shape):
        return False
    rank = len(lhs_affine.shape)
    packet_elements = int(packet_elements)
    for group in packet_coords:
        if len(group) != packet_elements or any(len(coords) != rank for coords in group):
            return False
        base_coords = group[0]
        lhs_deltas = tuple(_affine_static_delta(lhs_affine, base_coords, coords) for coords in group)
        rhs_deltas = tuple(_affine_static_delta(rhs_affine, base_coords, coords) for coords in group)
        if any(delta is None for delta in (*lhs_deltas, *rhs_deltas)):
            return False
        if all(int(delta) == 0 for delta in lhs_deltas) and all(int(delta) == 0 for delta in rhs_deltas):
            continue
        if predicate not in {"slt", "ult"}:
            return False
        if not all(int(delta) == 0 for delta in rhs_deltas):
            return False
        if not all(0 <= int(delta) < packet_elements for delta in lhs_deltas):
            return False
        if _affine_value_mod(
                conversion_input,
                lhs_affine,
                base_coords,
                packet_elements,
                divisibility_memo,
        ) != 0:
            return False
        if _affine_value_mod(
                conversion_input,
                rhs_affine,
                base_coords,
                packet_elements,
                divisibility_memo,
        ) != 0:
            return False
    return True


def _buffer_affine_offset_plan(
    conversion_input,
    type_layout_program,
    fact_program,
    offset_value_id,
    component_count,
    lane_width,
    op,
    *,
    coordinate_layout=None,
    coordinate_source_value_id=None,
):
    affine = fact_program.tensor_affine.get(offset_value_id)
    if affine is None:
        return None
    offset_value = type_layout_program.values[offset_value_id]
    if coordinate_layout is None:
        if offset_value.layout_map_id is None:
            return None
        layout = type_layout_program.layouts[int(offset_value.layout_map_id)]
        source_value_id = offset_value_id
    else:
        layout = coordinate_layout
        source_value_id = offset_value_id if coordinate_source_value_id is None else coordinate_source_value_id
    if layout.kind not in {
            "blocked",
            "linear",
            "generic_linear",
            "slice",
            "amd_mfma",
    }:
        return None
    if tuple(affine.shape) != tuple(int(dim) for dim in layout.shape):
        return None
    try:
        plan = coordinates.layout_coordinate_plan(
            layout,
            int(component_count),
            int(lane_width),
            _layout_warp_count(layout),
            op,
            source_value_id,
        )
    except Diagnostic:
        return None
    if plan is None:
        return None
    scalar_value_ids, terms = _packet_affine_terms(affine)
    return {
        "component_coordinate_bases":
        tuple(tuple(int(value) for value in bases) for bases in plan.component_bases),
        "coordinate_shape":
        tuple(int(dim) for dim in plan.shape),
        "offset_terms":
        tuple(terms),
        "scalar_value_ids":
        tuple(scalar_value_ids),
        "source_affine":
        affine,
        "workitem_coordinate_coefficients":
        tuple(tuple(int(value) for value in coefficients) for coefficients in plan.workitem_coefficients),
    }


def _buffer_load_affine_access_element_count(affine_plan, component_count, element_byte_width):
    return _buffer_affine_access_element_count(affine_plan, component_count, element_byte_width)


def _buffer_affine_access_element_count(affine_plan, component_count, element_byte_width):
    if affine_plan is None:
        return 1
    max_packet_elements = min(
        int(component_count),
        _buffer_max_packet_elements(element_byte_width),
    )
    for packet_elements in range(max_packet_elements, 1, -1):
        if not _buffer_packet_payload_is_legal(packet_elements, element_byte_width):
            continue
        if _buffer_load_affine_packets_are_contiguous(
                affine_plan,
                int(component_count),
                int(packet_elements),
        ):
            return int(packet_elements)
    return 1


def _buffer_load_affine_packets_are_contiguous(affine_plan, component_count, packet_elements):
    expected_deltas = tuple(range(int(packet_elements)))
    for packet_start in range(0, int(component_count), int(packet_elements)):
        if int(packet_start) + int(packet_elements) > int(component_count):
            continue
        deltas = _affine_packet_static_deltas(
            affine_plan,
            int(packet_start),
            int(packet_elements),
        )
        if deltas != expected_deltas:
            return False
    return True


def _buffer_mask_alignment(
    conversion_input,
    type_layout_program,
    fact_program,
    mask_value_id,
    component_count,
    lane_width,
    element_byte_width,
    op,
):
    max_packet_elements = _buffer_max_packet_elements(element_byte_width)
    if max_packet_elements <= 1:
        return 1
    producer_by_result = _producer_by_result(conversion_input)
    divisibility_memo = {}
    packet_memo = {}
    for packet_elements in range(max_packet_elements, 1, -1):
        if not _buffer_packet_payload_is_legal(packet_elements, element_byte_width):
            continue
        if int(component_count) % int(packet_elements):
            continue
        if all(
                _mask_packet_leading_predicate_is_safe(
                    conversion_input,
                    type_layout_program,
                    fact_program,
                    producer_by_result,
                    divisibility_memo,
                    packet_memo,
                    mask_value_id,
                    packet_start,
                    packet_elements,
                    lane_width,
                    op,
                ) for packet_start in range(0, int(component_count), int(packet_elements))):
            return int(packet_elements)
    return 1


def _mask_maybe_active_components(
    conversion_input,
    type_layout_program,
    fact_program,
    mask_value_id,
    component_count,
    lane_width,
    op,
):
    producer_by_result = _producer_by_result(conversion_input)
    memo = {}
    return tuple(
        component
        for component in range(int(component_count))
        if not _mask_component_is_provably_false(
            conversion_input,
            type_layout_program,
            fact_program,
            producer_by_result,
            int(mask_value_id),
            component,
            int(lane_width),
            op,
            memo,
            frozenset(),
        )
    )


def _mask_component_is_provably_false(
    conversion_input,
    type_layout_program,
    fact_program,
    producer_by_result,
    mask_value_id,
    component,
    lane_width,
    user_op,
    memo,
    visiting,
):
    key = (int(mask_value_id), int(component))
    if key in memo:
        return memo[key]
    if key in visiting:
        return False
    source_op = producer_by_result.get(int(mask_value_id))
    result = False
    if source_op is None:
        result = False
    elif source_op.name == "arith.constant":
        value = type_layout_program.values[int(mask_value_id)]
        literal = _constant_literal(
            source_op.attrs.get("value"),
            source_op_index=source_op.index,
            element_type=value.type.element_type,
        )
        result = value.type.element_type == "i1" and literal in {False, 0}
    elif source_op.name == "arith.cmpi":
        result = _cmpi_mask_component_is_provably_false(
            conversion_input,
            type_layout_program,
            fact_program,
            source_op,
            int(component),
            int(lane_width),
            user_op,
        )
    elif source_op.name in {"arith.andi", "arith.ori"} and len(source_op.operands) == 2:
        operand_results = []
        result_count = int(type_layout_program.values[int(mask_value_id)].type.component_count)
        for operand in source_op.operands:
            operand_count = int(type_layout_program.values[int(operand)].type.component_count)
            if operand_count == 1:
                operand_component = 0
            elif operand_count == result_count:
                operand_component = int(component)
            else:
                operand_results.append(False)
                continue
            operand_results.append(
                _mask_component_is_provably_false(
                    conversion_input,
                    type_layout_program,
                    fact_program,
                    producer_by_result,
                    int(operand),
                    operand_component,
                    int(lane_width),
                    user_op,
                    memo,
                    visiting | {key},
                )
            )
        result = (
            any(operand_results)
            if source_op.name == "arith.andi"
            else all(operand_results)
        )
    elif source_op.name == "tt.broadcast":
        component_sources = _broadcast_component_sources(
            type_layout_program,
            source_op,
        )
        if component_sources is not None and int(component) < len(component_sources):
            result = _mask_component_is_provably_false(
                conversion_input,
                type_layout_program,
                fact_program,
                producer_by_result,
                int(source_op.operands[0]),
                int(component_sources[int(component)]),
                int(lane_width),
                user_op,
                memo,
                visiting | {key},
            )
    memo[key] = bool(result)
    return bool(result)


def _cmpi_mask_component_is_provably_false(
    conversion_input,
    type_layout_program,
    fact_program,
    source_op,
    component,
    lane_width,
    user_op,
):
    if len(source_op.operands) != 2 or len(source_op.results) != 1:
        return False
    lhs, rhs = (int(value_id) for value_id in source_op.operands)
    lhs_value = type_layout_program.values[lhs]
    rhs_value = type_layout_program.values[rhs]
    if (
        int(lhs_value.type.component_count) != int(rhs_value.type.component_count)
        or int(component) >= int(lhs_value.type.component_count)
        or lhs_value.type.element_type != rhs_value.type.element_type
    ):
        return False
    lhs_plan = _tensor_affine_component_plan(
        conversion_input,
        type_layout_program,
        fact_program,
        lhs,
        int(lane_width),
        user_op,
    )
    rhs_plan = _tensor_affine_component_plan(
        conversion_input,
        type_layout_program,
        fact_program,
        rhs,
        int(lane_width),
        user_op,
    )
    if lhs_plan is None or rhs_plan is None:
        return False
    lhs_coefficients = tuple(lhs_plan["workitem_coordinate_coefficients"])
    rhs_coefficients = tuple(rhs_plan["workitem_coordinate_coefficients"])
    if len(lhs_coefficients) != len(rhs_coefficients):
        return False
    lhs_bases = tuple(lhs_plan["component_coordinate_bases"])
    rhs_bases = tuple(rhs_plan["component_coordinate_bases"])
    if int(component) >= len(lhs_bases) or int(component) >= len(rhs_bases):
        return False
    predicate = _cmpi_predicate(source_op.attrs.get("predicate"))
    for workitem in range(1 << len(lhs_coefficients)):
        lhs_coords = coordinates._coords_from_plan(
            lhs_bases[int(component)],
            lhs_coefficients,
            workitem,
        )
        rhs_coords = coordinates._coords_from_plan(
            rhs_bases[int(component)],
            rhs_coefficients,
            workitem,
        )
        lhs_static = _static_affine_value(lhs_plan["source_affine"], lhs_coords)
        rhs_static = _static_affine_value(rhs_plan["source_affine"], rhs_coords)
        if lhs_static is None or rhs_static is None:
            return False
        compared = _static_integer_compare(
            predicate,
            lhs_static,
            rhs_static,
            lhs_value.type.element_type,
        )
        if compared is None or compared:
            return False
    return True


def _static_affine_value(affine, coords):
    value = 0
    for term in affine.terms:
        if term.kind == "const":
            value += int(term.coefficient)
            continue
        if term.kind == "dim":
            value += int(term.coefficient) * int(coords[int(term.dim)])
            continue
        return None
    return int(value)


def _static_integer_compare(predicate, lhs, rhs, element_type):
    width = _int_width(element_type)
    if width is None or width <= 0:
        return None
    modulus = 1 << int(width)
    mask = modulus - 1
    lhs_unsigned = int(lhs) & mask
    rhs_unsigned = int(rhs) & mask
    if predicate == "eq":
        return lhs_unsigned == rhs_unsigned
    if predicate == "ne":
        return lhs_unsigned != rhs_unsigned
    if predicate.startswith("u"):
        lhs_value, rhs_value = lhs_unsigned, rhs_unsigned
    elif predicate.startswith("s"):
        sign = 1 << (int(width) - 1)
        lhs_value = lhs_unsigned - modulus if lhs_unsigned & sign else lhs_unsigned
        rhs_value = rhs_unsigned - modulus if rhs_unsigned & sign else rhs_unsigned
    else:
        return None
    relation = predicate[1:]
    if relation == "lt":
        return lhs_value < rhs_value
    if relation == "le":
        return lhs_value <= rhs_value
    if relation == "gt":
        return lhs_value > rhs_value
    if relation == "ge":
        return lhs_value >= rhs_value
    return None


def _producer_by_result(conversion_input):
    producer = {}
    for op in conversion_input.ops:
        for result in op.results:
            producer[int(result)] = op
    return producer


def _mask_packet_leading_predicate_is_safe(
    conversion_input,
    type_layout_program,
    fact_program,
    producer_by_result,
    divisibility_memo,
    packet_memo,
    mask_value_id,
    packet_start,
    packet_elements,
    lane_width,
    user_op,
):
    key = (int(mask_value_id), int(packet_start), int(packet_elements))
    if key in packet_memo:
        return packet_memo[key]
    source_op = producer_by_result.get(int(mask_value_id))
    result = False
    if source_op is None:
        result = False
    elif source_op.name == "arith.constant":
        result = True
    elif source_op.name == "arith.andi" and len(source_op.operands) == 2:
        result = all(
            _mask_packet_leading_predicate_is_safe(
                conversion_input,
                type_layout_program,
                fact_program,
                producer_by_result,
                divisibility_memo,
                packet_memo,
                operand,
                packet_start,
                packet_elements,
                lane_width,
                user_op,
            ) for operand in source_op.operands)
    elif source_op.name == "tt.broadcast":
        result = _broadcast_mask_packet_leading_predicate_is_safe(
            conversion_input,
            type_layout_program,
            fact_program,
            producer_by_result,
            divisibility_memo,
            packet_memo,
            source_op,
            packet_start,
            packet_elements,
            lane_width,
            user_op,
        )
    elif source_op.name == "arith.cmpi":
        result = _cmpi_mask_packet_leading_predicate_is_safe(
            conversion_input,
            type_layout_program,
            fact_program,
            producer_by_result,
            divisibility_memo,
            source_op,
            packet_start,
            packet_elements,
            lane_width,
            user_op,
        )
    packet_memo[key] = bool(result)
    return bool(result)


def _broadcast_mask_packet_leading_predicate_is_safe(
    conversion_input,
    type_layout_program,
    fact_program,
    producer_by_result,
    divisibility_memo,
    packet_memo,
    source_op,
    packet_start,
    packet_elements,
    lane_width,
    user_op,
):
    component_sources = _broadcast_component_sources(type_layout_program, source_op)
    if component_sources is None:
        return False
    packet_sources = tuple(
        int(component_sources[int(packet_start) + element]) for element in range(int(packet_elements)))
    if all(source == packet_sources[0] for source in packet_sources):
        return True
    if packet_sources != tuple(range(packet_sources[0], packet_sources[0] + int(packet_elements))):
        return False
    if packet_sources[0] % int(packet_elements):
        return False
    return _mask_packet_leading_predicate_is_safe(
        conversion_input,
        type_layout_program,
        fact_program,
        producer_by_result,
        divisibility_memo,
        packet_memo,
        source_op.operands[0],
        packet_sources[0],
        packet_elements,
        lane_width,
        user_op,
    )


def _cmpi_mask_packet_leading_predicate_is_safe(
    conversion_input,
    type_layout_program,
    fact_program,
    producer_by_result,
    divisibility_memo,
    source_op,
    packet_start,
    packet_elements,
    lane_width,
    user_op,
):
    if len(source_op.operands) != 2 or len(source_op.results) != 1:
        return False
    predicate = _cmpi_predicate(source_op.attrs.get("predicate"))
    if predicate not in {"slt", "ult", "sle", "ule"}:
        return False
    lhs, rhs = source_op.operands
    lhs_plan = _tensor_affine_component_plan(
        conversion_input,
        type_layout_program,
        fact_program,
        lhs,
        lane_width,
        user_op,
    )
    rhs_plan = _tensor_affine_component_plan(
        conversion_input,
        type_layout_program,
        fact_program,
        rhs,
        lane_width,
        user_op,
    )
    if lhs_plan is None or rhs_plan is None:
        return False
    lhs_deltas = _affine_packet_static_deltas(lhs_plan, packet_start, packet_elements)
    rhs_deltas = _affine_packet_static_deltas(rhs_plan, packet_start, packet_elements)
    if lhs_deltas is None or rhs_deltas is None:
        return False
    if all(delta == 0 for delta in lhs_deltas) and all(delta == 0 for delta in rhs_deltas):
        return True
    if not all(delta == 0 for delta in rhs_deltas):
        return False
    if not all(0 <= int(delta) < int(packet_elements) for delta in lhs_deltas):
        return False
    if not _affine_packet_base_is_divisible(
            conversion_input,
            lhs_plan,
            packet_start,
            packet_elements,
            divisibility_memo,
    ):
        return False
    return _affine_packet_base_is_divisible(
        conversion_input,
        rhs_plan,
        packet_start,
        packet_elements,
        divisibility_memo,
    )


def _tensor_affine_component_plan(
    conversion_input,
    type_layout_program,
    fact_program,
    value_id,
    lane_width,
    op,
):
    value = type_layout_program.values[int(value_id)]
    return _buffer_affine_offset_plan(
        conversion_input,
        type_layout_program,
        fact_program,
        int(value_id),
        int(value.type.component_count),
        int(lane_width),
        op,
    )


def _materialize_affine_edge_or_original(
    builder,
    conversion_input,
    type_layout_program,
    fact_program,
    source_value_id,
    op,
    *,
    no_signed_wrap,
    result_element_type="i32",
    value_range=None,
):
    """Materialize an affine tensor through an explicit target-IR edge op.

    Affine analysis may compact a distributed i32 tensor into layout
    coordinates plus scalar bindings.  The compact form is never attached to
    the consuming semantic op and is never carried as an emitter-only object:
    an ``affine_materialize`` target op reconstructs the ordinary SIMD value
    immediately on the edge where it is required.
    """
    source_value_id = int(source_value_id)
    value = type_layout_program.values[source_value_id]
    if (
        value.type.element_type != "i32"
        or value.type.representation not in {"simd", "simd_tuple"}
    ):
        return _single_source_target(builder, source_value_id, op)
    result_element_type = str(result_element_type)
    if result_element_type not in {"i32", "index"}:
        fail(
            "TLXW_OP_AFFINE_EDGE",
            STAGE,
            "affine edge results must use i32 or index elements",
            source_op_index=op.index,
            source_value_id=source_value_id,
        )
    if result_element_type == "index":
        if (
            not isinstance(value_range, tuple)
            or len(value_range) != 2
            or not all(isinstance(bound, int) for bound in value_range)
        ):
            fail(
                "TLXW_OP_AFFINE_EDGE",
                STAGE,
                "index affine edges require a closed integer value range",
                source_op_index=op.index,
                source_value_id=source_value_id,
            )
    lane_width = int(value.type.lane_width or conversion_input.threads_per_warp)
    plan = _tensor_affine_component_plan(
        conversion_input,
        type_layout_program,
        fact_program,
        source_value_id,
        lane_width,
        op,
    )
    if plan is None:
        source_target_id = _single_source_target(
            builder,
            source_value_id,
            op,
        )
        if result_element_type == "i32":
            return source_target_id
        return _bounded_i32_to_index_edge(
            builder,
            source_target_id,
            value_range,
            op,
        )
    scalar_component_sources = _layout_affine_scalar_component_sources(
        type_layout_program,
        value,
        plan["scalar_value_ids"],
        lane_width,
        op,
    )
    if scalar_component_sources is None:
        source_target_id = _single_source_target(
            builder,
            source_value_id,
            op,
        )
        if result_element_type == "i32":
            return source_target_id
        return _bounded_i32_to_index_edge(
            builder,
            source_target_id,
            value_range,
            op,
        )
    scalar_target_ids = tuple(
        _single_source_target(builder, int(scalar_value_id), op)
        for scalar_value_id in plan["scalar_value_ids"]
    )
    source_target_type = target_ir.target_type_from_converted(value.type)
    result_target_id = builder.add_value(
        target_ir.TargetType(
            source_target_type.kind,
            source_target_type.representation,
            result_element_type,
            source_target_type.lane_width,
            source_target_type.component_count,
        ),
        debug_name=f"affine_edge_{op.index}_{source_value_id}",
    )
    replaced_target_id = _single_source_target(
        builder,
        source_value_id,
        op,
    )
    builder.add_op(
        "affine_materialize",
        operands=scalar_target_ids,
        results=(result_target_id, ),
        attrs={
            "mode": "layout_coordinates",
            "component_coordinate_bases": tuple(
                tuple(int(component) for component in bases)
                for bases in plan["component_coordinate_bases"]
            ),
            "coordinate_shape": tuple(
                int(dim) for dim in plan["coordinate_shape"]
            ),
            "no_signed_wrap": bool(no_signed_wrap),
            "scalar_count": len(scalar_target_ids),
            "scalar_component_sources": tuple(
                tuple(int(component) for component in sources)
                for sources in scalar_component_sources
            ),
            "terms": tuple(plan["offset_terms"]),
            **(
                {"value_range": tuple(int(bound) for bound in value_range)}
                if result_element_type == "index" else {}
            ),
            "workitem_coordinate_coefficients": tuple(
                tuple(int(component) for component in coefficients)
                for coefficients in plan["workitem_coordinate_coefficients"]
            ),
            target_ir.PROVENANCE_ONLY_TARGET_IDS_ATTR: (
                int(replaced_target_id),
            ),
        },
        layout_map_ids=(
            () if value.layout_map_id is None else (int(value.layout_map_id), )
        ),
        source_op_index=op.index,
    )
    return result_target_id


def _bounded_i32_to_index_edge(
    builder,
    source_target_id,
    value_range,
    op,
):
    """Convert an ordinary bounded i32 value to the index representation.

    This fallback keeps the buffer-offset representation contract explicit
    even when affine analysis cannot replace the source computation with a
    symbolic coordinate expression.
    """
    source_target_id = int(source_target_id)
    source_type = builder.values[source_target_id].type
    result_target_id = builder.add_value(
        target_ir.TargetType(
            source_type.kind,
            source_type.representation,
            "index",
            source_type.lane_width,
            source_type.component_count,
        ),
        debug_name=f"bounded_index_edge_{op.index}_{source_target_id}",
    )
    builder.add_op(
        "type_convert",
        operands=(source_target_id, ),
        results=(result_target_id, ),
        attrs={
            "mode": "bounded_i32_to_index",
            "value_range": tuple(int(bound) for bound in value_range),
        },
        source_op_index=op.index,
    )
    return result_target_id


def _materialize_packet_affine_edge(
    builder,
    type_layout_program,
    source_value_id,
    packet_plan,
    scalar_component_sources,
    op,
    *,
    no_signed_wrap,
    offset_range,
):
    """Materialize packet-leading affine offsets as an explicit edge value.

    Packet DMA consumes one source offset per packet rather than every scalar
    element offset represented by the source tensor.  That is a structural
    representation change, so it is expressed by ``affine_materialize`` and
    not encoded in the DMA operation or replayed by its emitter.
    """
    source_value_id = int(source_value_id)
    value = type_layout_program.values[source_value_id]
    component_count = int(packet_plan["component_count"])
    lane_width = int(value.type.lane_width or 64)
    scalar_value_ids = tuple(
        int(value_id) for value_id in packet_plan["scalar_value_ids"]
    )
    scalar_target_ids = tuple(
        _single_source_target(builder, value_id, op)
        for value_id in scalar_value_ids
    )
    result_type = target_ir.TargetType(
        value.type.kind,
        "simd" if component_count == 1 else "simd_tuple",
        "index",
        lane_width,
        component_count,
    )
    result_target_id = builder.add_value(
        result_type,
        debug_name=f"packet_affine_edge_{op.index}_{source_value_id}",
    )
    replaced_target_id = _single_source_target(
        builder,
        source_value_id,
        op,
    )
    builder.add_op(
        "affine_materialize",
        operands=scalar_target_ids,
        results=(result_target_id, ),
        attrs={
            "component_thread_count": int(
                packet_plan["component_thread_count"]
            ),
            "coordinate_mode": str(packet_plan["source_coordinate_mode"]),
            "coordinate_order": tuple(
                int(dim) for dim in packet_plan["packet_order"]
            ),
            "coordinate_shape": tuple(
                int(dim) for dim in packet_plan["source_affine"].shape
            ),
            "linear_component_bases": tuple(
                tuple(int(component) for component in basis)
                for basis in packet_plan["source_linear_component_bases"]
            ),
            "mode": "packet_coordinates",
            "no_signed_wrap": bool(no_signed_wrap),
            "value_range": tuple(int(bound) for bound in offset_range),
            "packet_elements": int(packet_plan["packet_elements"]),
            "scalar_component_sources": tuple(
                tuple(int(component) for component in sources)
                for sources in scalar_component_sources
            ),
            "scalar_count": len(scalar_target_ids),
            "terms": tuple(packet_plan["source_offset_terms"]),
            target_ir.PROVENANCE_ONLY_TARGET_IDS_ATTR: (
                int(replaced_target_id),
            ),
        },
        source_op_index=op.index,
    )
    return result_target_id


def _affine_packet_static_deltas(plan, packet_start, packet_elements):
    component_bases = tuple(plan["component_coordinate_bases"])
    workitem_coefficients = tuple(plan["workitem_coordinate_coefficients"])
    if int(packet_start) + int(packet_elements) > len(component_bases):
        return None
    workitem_count = 1 << len(workitem_coefficients)
    deltas = [None] * int(packet_elements)
    for workitem in range(workitem_count):
        base_coords = coordinates._coords_from_plan(component_bases[int(packet_start)], workitem_coefficients, workitem)
        for element in range(int(packet_elements)):
            coords = coordinates._coords_from_plan(
                component_bases[int(packet_start) + element],
                workitem_coefficients,
                workitem,
            )
            delta = _affine_static_delta(plan["source_affine"], base_coords, coords)
            if delta is None:
                return None
            if deltas[element] is None:
                deltas[element] = int(delta)
            elif deltas[element] != int(delta):
                return None
    return tuple(int(delta) for delta in deltas)


def _affine_static_delta(affine, base_coords, coords):
    delta = 0
    for term in affine.terms:
        coefficient = int(term.coefficient)
        if term.kind in {"const", "scalar", "scalar_product"}:
            continue
        if term.kind == "dim":
            dim = int(term.dim)
            delta += coefficient * (int(coords[dim]) - int(base_coords[dim]))
            continue
        if term.kind == "dim_scalar":
            dim = int(term.dim)
            if int(coords[dim]) != int(base_coords[dim]):
                return None
            continue
        return None
    return int(delta)


def _affine_packet_base_is_divisible(
    conversion_input,
    plan,
    packet_start,
    packet_elements,
    divisibility_memo,
):
    component_bases = tuple(plan["component_coordinate_bases"])
    workitem_coefficients = tuple(plan["workitem_coordinate_coefficients"])
    if int(packet_start) >= len(component_bases):
        return False
    workitem_count = 1 << len(workitem_coefficients)
    for workitem in range(workitem_count):
        coords = coordinates._coords_from_plan(component_bases[int(packet_start)], workitem_coefficients, workitem)
        if _affine_value_mod(
                conversion_input,
                plan["source_affine"],
                coords,
                int(packet_elements),
                divisibility_memo,
        ) != 0:
            return False
    return True


def _affine_value_mod(conversion_input, affine, coords, modulus, divisibility_memo):
    modulus = int(modulus)
    residue = 0
    for term in affine.terms:
        coefficient = int(term.coefficient)
        if term.kind == "const":
            residue = (residue + coefficient) % modulus
            continue
        if term.kind == "dim":
            residue = (residue + coefficient * int(coords[int(term.dim)])) % modulus
            continue
        if term.kind == "scalar":
            if not _term_with_scalar_divisible(conversion_input, coefficient, term.scalar_value_ids, modulus,
                                               divisibility_memo):
                return None
            continue
        if term.kind == "dim_scalar":
            dim_factor = coefficient * int(coords[int(term.dim)])
            if dim_factor % modulus and not _term_with_scalar_divisible(
                    conversion_input,
                    dim_factor,
                    term.scalar_value_ids,
                    modulus,
                    divisibility_memo,
            ):
                return None
            continue
        if term.kind == "scalar_product":
            if not _term_with_scalar_divisible(conversion_input, coefficient, term.scalar_value_ids, modulus,
                                               divisibility_memo):
                return None
            continue
        return None
    return residue % modulus


def _term_with_scalar_divisible(conversion_input, coefficient, scalar_value_ids, modulus, divisibility_memo):
    divisor = abs(int(coefficient))
    for scalar_value_id in scalar_value_ids:
        scalar_divisibility = _source_value_divisibility(conversion_input, scalar_value_id, divisibility_memo)
        if scalar_divisibility is None:
            continue
        divisor *= int(scalar_divisibility)
    return divisor % int(modulus) == 0


def _source_value_divisibility(conversion_input, value_id, memo):
    value_id = int(value_id)
    if value_id in memo:
        return memo[value_id]
    declared = conversion_input.value_divisibilities.get(value_id)
    if declared is not None and int(declared) > 0:
        memo[value_id] = int(declared)
        return int(declared)
    if value_id in conversion_input.constant_ints:
        value = abs(int(conversion_input.constant_ints[value_id]))
        memo[value_id] = (1 << 30) if value == 0 else value & -value
        return memo[value_id]
    producer = None
    for op in conversion_input.ops:
        if value_id in op.results:
            producer = op
            break
    result = None
    if producer is not None and producer.name in {"arith.addi", "arith.subi"} and len(producer.operands) == 2:
        lhs = _source_value_divisibility(conversion_input, producer.operands[0], memo)
        rhs = _source_value_divisibility(conversion_input, producer.operands[1], memo)
        if lhs is not None and rhs is not None:
            result = math.gcd(int(lhs), int(rhs))
    elif producer is not None and producer.name == "arith.muli" and len(producer.operands) == 2:
        lhs = _source_value_divisibility(conversion_input, producer.operands[0], memo)
        rhs = _source_value_divisibility(conversion_input, producer.operands[1], memo)
        if lhs is not None and rhs is not None:
            result = int(lhs) * int(rhs)
        elif lhs is not None:
            result = int(lhs)
        elif rhs is not None:
            result = int(rhs)
    memo[value_id] = result
    return result


def _buffer_max_packet_elements(element_byte_width):
    element_byte_width = int(element_byte_width)
    if element_byte_width <= 0:
        return 1
    return max(1, 16 // element_byte_width)


def _buffer_packet_payload_is_legal(packet_elements, element_byte_width):
    packet_elements = int(packet_elements)
    element_byte_width = int(element_byte_width)
    if packet_elements <= 1 or element_byte_width <= 0:
        return False
    payload_bits = packet_elements * element_byte_width * 8
    return payload_bits <= 128 and (payload_bits == 16 or payload_bits % 32 == 0)


def _affine_source_offset_no_signed_wrap(
    conversion_input,
    fact_program,
    affine,
    op,
    offset_upper,
):
    del conversion_input, fact_program, affine, op
    if int(offset_upper) > 0x7FFFFFFF:
        return False
    # Affine DMA/buffer offsets are layout-address expressions.  The target IR
    # is only defined for executions where that address math does not overflow;
    # dynamic scalar leaf ranges are not the no-wrap provenance.  The packet
    # offset range limits the final reconstructed i32 offset, and overflow of
    # any intermediate add/mul on the way there is outside the target semantics.
    return True


def _packet_source_is_contiguous(
    affine,
    shape,
    packet_order,
    packet_elements,
    component_count,
    wave_count,
    lane_width,
):
    if not packet_order:
        return False
    packet_dim = int(packet_order[0])
    delta = 0
    for term in affine.terms:
        coefficient = int(term.coefficient)
        if term.kind in {"const", "scalar", "scalar_product"}:
            continue
        if term.kind == "dim":
            if int(term.dim) == packet_dim:
                delta += coefficient
            continue
        if term.kind == "dim_scalar":
            if int(term.dim) == packet_dim and coefficient:
                return False
            continue
        return False
    if delta != 1:
        return False

    component_thread_count = int(wave_count) * int(lane_width)
    for component in range(int(component_count)):
        component_base = int(component) * component_thread_count * int(packet_elements)
        for workitem in range(component_thread_count):
            linear_start = component_base + workitem * int(packet_elements)
            linear_end = linear_start + int(packet_elements) - 1
            if linear_end >= _product(shape):
                return False
            start = _ordered_coords_from_linear(linear_start, shape, packet_order)
            end = _ordered_coords_from_linear(linear_end, shape, packet_order)
            for dim in range(len(shape)):
                if dim == packet_dim:
                    if int(end[dim]) - int(start[dim]) != int(packet_elements) - 1:
                        return False
                elif int(end[dim]) != int(start[dim]):
                    return False
    return True


def _packet_physical_linear_component_bases(layout, shape, op):
    if layout is None:
        return None
    if layout.kind == "padded_shared":
        if layout.properties.get("linear_component") is None:
            return None
        return layouts.padded_shared_linear_component_bases(
            layout,
            shape,
            stage=STAGE,
            diagnostic="TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
            source_op_index=op.index,
            source_value_id=layout.value_id,
        )
    if layout.kind == "swizzled_shared":
        if layouts.is_identity_swizzled_shared(layout, shape):
            return None
        return layouts.swizzled_shared_linear_component_bases(
            layout,
            shape,
            stage=STAGE,
            diagnostic="TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
            source_op_index=op.index,
            source_value_id=layout.value_id,
        )
    if layout.kind == "shared_linear":
        layouts.shared_linear_inverse_offset_bases(
            layout,
            shape,
            stage=STAGE,
            diagnostic="TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
            source_op_index=op.index,
            source_value_id=layout.value_id,
        )
        return layouts.linear_layout_bases(
            layout.properties["linear_component"],
            "offset",
        )
    return None


def _packet_physical_linear_source_is_contiguous(
    affine,
    shape,
    bases,
    packet_elements,
    component_count,
    wave_count,
    lane_width,
):
    component_thread_count = int(wave_count) * int(lane_width)
    total_elements = _product(shape)
    for component in range(int(component_count)):
        component_base = int(component) * component_thread_count * int(packet_elements)
        for workitem in range(component_thread_count):
            linear_start = component_base + workitem * int(packet_elements)
            linear_end = linear_start + int(packet_elements) - 1
            if linear_end >= total_elements:
                return False
            start_signature = _affine_static_signature(
                affine,
                _coords_from_linear_component_offset(linear_start, bases, len(shape)),
            )
            for element in range(1, int(packet_elements)):
                signature = _affine_static_signature(
                    affine,
                    _coords_from_linear_component_offset(linear_start + element, bases, len(shape)),
                )
                if not _affine_static_signature_delta_is_unit(start_signature, signature, element):
                    return False
    return True


def _affine_static_signature(affine, coords):
    constant = 0
    scalar_coefficients = {}
    scalar_product_coefficients = {}
    for term in affine.terms:
        coefficient = int(term.coefficient)
        if term.kind == "const":
            constant += coefficient
            continue
        if term.kind == "dim":
            constant += coefficient * int(coords[int(term.dim)])
            continue
        if term.kind == "scalar":
            key = tuple(int(value) for value in term.scalar_value_ids)
            scalar_coefficients[key] = scalar_coefficients.get(key, 0) + coefficient
            continue
        if term.kind == "dim_scalar":
            key = tuple(int(value) for value in term.scalar_value_ids)
            scalar_coefficients[key] = scalar_coefficients.get(key, 0) + coefficient * int(coords[int(term.dim)])
            continue
        if term.kind == "scalar_product":
            key = tuple(int(value) for value in term.scalar_value_ids)
            scalar_product_coefficients[key] = scalar_product_coefficients.get(key, 0) + coefficient
            continue
        return None
    return (
        int(constant),
        tuple(sorted((key, int(value)) for key, value in scalar_coefficients.items() if int(value))),
        tuple(sorted((key, int(value)) for key, value in scalar_product_coefficients.items() if int(value))),
    )


def _affine_static_signature_delta_is_unit(start, current, expected_delta):
    if start is None or current is None:
        return False
    return (int(current[0]) - int(start[0]) == int(expected_delta) and current[1] == start[1]
            and current[2] == start[2])


def _packet_destination_offsets(
    layout,
    shape,
    physical_shape,
    logical_origin,
    packet_order,
    component_count,
    elements_per_cta_packet,
    elements_per_wave_packet,
    element_byte_width,
    wave_count,
    op,
):
    destination_offsets = []
    wave_stride_dwords = None
    wave_offset_coefficients_dwords = None
    for component in range(int(component_count)):
        component_linear = int(component) * int(elements_per_cta_packet)
        wave_offsets = tuple(
            _packet_physical_component_offset(
                layout,
                shape,
                physical_shape,
                logical_origin,
                packet_order,
                component_linear + wave * int(elements_per_wave_packet),
                elements_per_wave_packet,
                op,
            ) for wave in range(int(wave_count)))
        destination_offsets.append(wave_offsets[0])
        component_deltas = []
        for wave_offset in wave_offsets:
            byte_delta = (int(wave_offset) - int(wave_offsets[0])) * int(element_byte_width)
            if byte_delta % 4:
                fail(
                    "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
                    STAGE,
                    "packet DMA destination wave offset must be dword aligned",
                    source_op_index=op.index,
                    source_value_id=layout.value_id if layout is not None else None,
                )
            component_deltas.append(byte_delta // 4)
        component_deltas = tuple(int(delta) for delta in component_deltas)
        component_stride = _uniform_wave_stride(component_deltas)
        component_coefficients = ()
        if component_stride is None:
            component_coefficients = _bit_linear_wave_offset_coefficients(component_deltas)
            if component_coefficients is None:
                fail(
                    "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
                    STAGE,
                    "packet DMA destination wave offsets must be uniform or bit-linear",
                    source_op_index=op.index,
                    source_value_id=layout.value_id if layout is not None else None,
                )
            component_stride = 0
        if component_stride != 0:
            component_coefficients = ()
        if wave_stride_dwords is None:
            wave_stride_dwords = int(component_stride)
        elif wave_stride_dwords != component_stride:
            fail(
                "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
                STAGE,
                "packet DMA destination wave stride must be identical for all components",
                source_op_index=op.index,
                source_value_id=layout.value_id if layout is not None else None,
            )
        if wave_offset_coefficients_dwords is None:
            wave_offset_coefficients_dwords = tuple(int(value) for value in component_coefficients)
        elif wave_offset_coefficients_dwords != tuple(int(value) for value in component_coefficients):
            fail(
                "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
                STAGE,
                "packet DMA destination wave offset coefficients must be identical for all components",
                source_op_index=op.index,
                source_value_id=layout.value_id if layout is not None else None,
            )
    return (
        tuple(destination_offsets),
        int(wave_stride_dwords or 0),
        tuple(wave_offset_coefficients_dwords or ()),
    )


def _packet_physical_linear_destination_offsets(
    layout,
    shape,
    component_count,
    elements_per_wave_packet,
    element_byte_width,
    wave_count,
    op,
):
    if layout.kind == "padded_shared":
        intervals, paddings = layouts.padded_shared_parameters(
            layout,
            stage=STAGE,
            diagnostic="TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
            source_op_index=op.index,
            source_value_id=layout.value_id,
        )
    elif layout.kind in {"shared_linear", "swizzled_shared"}:
        intervals, paddings = (), ()
    else:
        fail(
            "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
            STAGE,
            "physical-linear packet DMA requires an imported shared layout map",
            source_op_index=op.index,
            source_value_id=layout.value_id,
        )
    destination_offsets = []
    wave_stride_dwords = None
    wave_offset_coefficients_dwords = None
    for component in range(int(component_count)):
        component_physical = int(component) * int(wave_count) * int(elements_per_wave_packet)
        wave_offsets = []
        for wave in range(int(wave_count)):
            physical = component_physical + int(wave) * int(elements_per_wave_packet)
            physical_end = physical + int(elements_per_wave_packet) - 1
            for interval in intervals:
                if physical // int(interval) != physical_end // int(interval):
                    fail(
                        "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
                        STAGE,
                        "packet DMA physical-linear destination crosses a padded LDS interval",
                        source_op_index=op.index,
                        source_value_id=layout.value_id,
                    )
            wave_offsets.append(_padded_physical_element_offset(physical, intervals, paddings))
        destination_offsets.append(wave_offsets[0])
        component_deltas = []
        for wave_offset in wave_offsets:
            byte_delta = (int(wave_offset) - int(wave_offsets[0])) * int(element_byte_width)
            if byte_delta % 4:
                fail(
                    "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
                    STAGE,
                    "packet DMA physical-linear wave offset must be dword aligned",
                    source_op_index=op.index,
                    source_value_id=layout.value_id,
                )
            component_deltas.append(byte_delta // 4)
        component_deltas = tuple(int(delta) for delta in component_deltas)
        component_stride = _uniform_wave_stride(component_deltas)
        component_coefficients = ()
        if component_stride is None:
            component_coefficients = _bit_linear_wave_offset_coefficients(component_deltas)
            if component_coefficients is None:
                fail(
                    "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
                    STAGE,
                    "packet DMA physical-linear wave offsets must be uniform or bit-linear",
                    source_op_index=op.index,
                    source_value_id=layout.value_id,
                )
            component_stride = 0
        if component_stride != 0:
            component_coefficients = ()
        if wave_stride_dwords is None:
            wave_stride_dwords = int(component_stride)
        elif wave_stride_dwords != int(component_stride):
            fail(
                "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
                STAGE,
                "packet DMA physical-linear wave stride must be identical for all components",
                source_op_index=op.index,
                source_value_id=layout.value_id,
            )
        if wave_offset_coefficients_dwords is None:
            wave_offset_coefficients_dwords = tuple(int(value) for value in component_coefficients)
        elif wave_offset_coefficients_dwords != tuple(int(value) for value in component_coefficients):
            fail(
                "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
                STAGE,
                "packet DMA physical-linear wave offset coefficients must be identical for all components",
                source_op_index=op.index,
                source_value_id=layout.value_id,
            )
    return (
        tuple(destination_offsets),
        int(wave_stride_dwords or 0),
        tuple(wave_offset_coefficients_dwords or ()),
    )


def _padded_physical_element_offset(physical, intervals, paddings):
    offset = int(physical)
    for interval, padding in zip(intervals, paddings):
        offset += (int(physical) // int(interval)) * int(padding)
    return int(offset)


def _coords_from_linear_component_offset(linear, bases, rank):
    coords = [0] * int(rank)
    for bit, basis in enumerate(tuple(bases)):
        bit_value = (int(linear) >> int(bit)) & 1
        if not bit_value:
            continue
        for dim, value in enumerate(tuple(int(value) for value in basis)):
            if int(value):
                coords[int(dim)] ^= int(value)
    return tuple(int(value) for value in coords)


def _uniform_wave_stride(deltas):
    deltas = tuple(int(delta) for delta in deltas)
    if not deltas or deltas[0] != 0:
        return None
    if len(deltas) == 1:
        return 0
    stride = int(deltas[1])
    for wave, delta in enumerate(deltas):
        if int(delta) != stride * int(wave):
            return None
    return int(stride)


def _bit_linear_wave_offset_coefficients(deltas):
    deltas = tuple(int(delta) for delta in deltas)
    if not deltas or deltas[0] != 0:
        return None
    if len(deltas) == 1:
        return ()
    if len(deltas) & (len(deltas) - 1):
        return None
    coefficients = tuple(int(deltas[1 << bit]) for bit in range(len(deltas).bit_length() - 1))
    for wave, delta in enumerate(deltas):
        expected = 0
        for bit, coefficient in enumerate(coefficients):
            if int(wave) & (1 << bit):
                expected += int(coefficient)
        if int(delta) != expected:
            return None
    return coefficients


def _packet_physical_component_offset(
    layout,
    shape,
    physical_shape,
    logical_origin,
    packet_order,
    ordered_linear_start,
    lane_width,
    op,
):
    shape = tuple(int(dim) for dim in shape)
    physical_shape = tuple(int(dim) for dim in physical_shape)
    logical_origin = tuple(int(value) for value in logical_origin)
    linear_start = int(ordered_linear_start)
    linear_end = linear_start + int(lane_width) - 1
    start_coords = _ordered_coords_from_linear(linear_start, shape, packet_order)
    end_coords = _ordered_coords_from_linear(linear_end, shape, packet_order)
    physical_start_coords = tuple(
        int(origin) + int(coord)
        for origin, coord in zip(logical_origin, start_coords)
    )
    physical_end_coords = tuple(
        int(origin) + int(coord)
        for origin, coord in zip(logical_origin, end_coords)
    )
    source_value_id = None if layout is None else layout.value_id
    start_record = layouts.shared_physical_offset(
        layout,
        physical_shape,
        physical_start_coords,
        1,
        stage=STAGE,
        diagnostic="TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
        source_op_index=op.index,
        source_value_id=source_value_id,
    )
    end_record = layouts.shared_physical_offset(
        layout,
        physical_shape,
        physical_end_coords,
        1,
        stage=STAGE,
        diagnostic="TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
        source_op_index=op.index,
        source_value_id=source_value_id,
    )
    if int(end_record.element_offset) - int(start_record.element_offset) != int(lane_width) - 1:
        fail(
            "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
            STAGE,
            "packet DMA destination is not physically contiguous across the component",
            source_op_index=op.index,
            source_value_id=source_value_id,
        )
    # memdesc_subslice is an address-preserving view in target IR.  Its
    # logical origin therefore remains part of the component offset from the
    # indexed/allocation base rather than being subtracted here.
    return int(start_record.element_offset)


def _dma_packet_byte_candidates(element_byte_width, *, include_narrow=True):
    if element_byte_width is None:
        return ()
    element_byte_width = int(element_byte_width)
    if element_byte_width <= 0:
        return ()
    candidates = []
    for packet_bytes in ((16, 4) if include_narrow else (16, )):
        if packet_bytes % element_byte_width:
            continue
        packet_elements = packet_bytes // element_byte_width
        if _buffer_packet_payload_is_legal(packet_elements, element_byte_width):
            candidates.append(int(packet_bytes))
    return tuple(candidates)


def _buffer_source_offset_upper(range_upper_bytes, packet_bytes, element_byte_width, op):
    range_upper_bytes = int(range_upper_bytes)
    packet_bytes = int(packet_bytes)
    element_byte_width = int(element_byte_width)
    if packet_bytes <= 0 or element_byte_width <= 0:
        fail(
            "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
            STAGE,
            f"{op.name} source offset range requires positive access and "
            "element byte widths",
            source_op_index=op.index,
        )
    if range_upper_bytes < packet_bytes - 1:
        fail(
            "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
            STAGE,
            f"{op.name} access exceeds source pointer range",
            source_op_index=op.index,
        )
    return (range_upper_bytes - packet_bytes + 1) // element_byte_width


def _buffer_inactive_byte_offset():
    return 1 << 31


def _buffer_inactive_element_offset(element_byte_width, op):
    element_byte_width = int(element_byte_width)
    if element_byte_width <= 0:
        fail(
            "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
            STAGE,
            f"{op.name} inactive source offset requires positive element byte width",
            source_op_index=op.index,
        )
    inactive_byte_offset = _buffer_inactive_byte_offset()
    if inactive_byte_offset % element_byte_width:
        fail(
            "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
            STAGE,
            f"{op.name} inactive byte offset is not aligned to element size",
            source_op_index=op.index,
        )
    return inactive_byte_offset // element_byte_width


def _packet_affine_terms(affine):
    scalar_value_ids = []
    scalar_slots = {}
    terms = []
    for term in affine.terms:
        slots = tuple(_scalar_slot(value_id, scalar_value_ids, scalar_slots) for value_id in term.scalar_value_ids)
        terms.append((
            term.kind,
            int(term.coefficient),
            -1 if term.dim is None else int(term.dim),
            slots,
        ))
    return tuple(scalar_value_ids), tuple(terms)


def _scalar_slot(value_id, scalar_value_ids, scalar_slots):
    if value_id not in scalar_slots:
        scalar_slots[value_id] = len(scalar_value_ids)
        scalar_value_ids.append(value_id)
    return scalar_slots[value_id]


def _fragment_registers(element_type, result_layout, op):
    parent = result_layout.properties.get("parent_properties", {})
    instr_shape = tuple(parent.get("instr_shape", ()))
    if (element_type in {"f16", "bf16"} and instr_shape in {(16, 16, 32), (32, 32, 16)}):
        return 4
    if element_type == "i8" and instr_shape == (16, 16, 128):
        return 4
    fail(
        "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
        STAGE,
        "ttg.local_load fragment registers are not known for "
        f"element_type={element_type}, instr_shape={instr_shape}",
        source_op_index=op.index,
        source_value_id=result_layout.value_id,
    )


def _mma_packet_registers(type_layout_program, converted, op):
    layout = _require_layout(type_layout_program, converted.layout_map_id, op)
    if layout.kind != "amd_mfma":
        fail(
            "TLXW_OP_UNSUPPORTED_MMA_PACKET",
            STAGE,
            "MFMA packets require an amd_mfma layout",
            source_op_index=op.index,
            source_value_id=converted.value_id,
        )
    linear = layouts.distributed_linear_layout(
        layout,
        stage=STAGE,
        source_op_index=op.index,
    )
    register_count = layouts.linear_layout_in_dim_size(linear, "register")
    component_count = int(layout.component_count)
    if component_count <= 0 or register_count % component_count:
        fail(
            "TLXW_OP_UNSUPPORTED_MMA_PACKET",
            STAGE,
            "MFMA payload registers must evenly partition layout components",
            source_op_index=op.index,
            source_value_id=converted.value_id,
        )
    return register_count // component_count


def _register_vector_payload_registers(type_layout_program, converted, op):
    registers = _register_vector_payload_registers_or_none(
        type_layout_program,
        converted,
        op,
    )
    if registers is None:
        fail(
            "TLXW_OP_UNSUPPORTED_REGISTER_PAYLOAD",
            STAGE,
            "register vector payloads require an MFMA layout or MFMA slice layout",
            source_op_index=op.index,
            source_value_id=converted.value_id,
        )
    return int(registers)


def _register_vector_payload_registers_or_none(type_layout_program, converted, op):
    if converted.layout_map_id is None:
        return None
    if converted.type.representation not in {"simd", "simd_tuple"}:
        return None
    layout = _require_layout(type_layout_program, converted.layout_map_id, op)
    if layout.kind == "amd_mfma":
        return layouts.mfma_registers_per_component(
            layout,
            stage=STAGE,
            source_op_index=op.index,
        )
    if layout.kind != "slice" or layout.properties.get("parent_kind") != "amd_mfma":
        return None
    return _mfma_registers_per_component_from_properties(
        layout.properties.get("parent_properties", {}),
        layout,
        op,
    )


def _mfma_registers_per_component_from_properties(properties, layout, op):
    instr_shape = tuple(int(value) for value in properties.get("instr_shape", ()))
    if instr_shape in {(16, 16, 32), (16, 16, 128)}:
        return 4
    if instr_shape == (32, 32, 16):
        return 16
    fail(
        "TLXW_OP_UNSUPPORTED_REGISTER_PAYLOAD",
        STAGE,
        f"unsupported MFMA register payload count for instrShape={instr_shape}",
        source_op_index=op.index,
        source_value_id=layout.value_id,
    )


def _acc_fragment_registers(result_layout, op):
    instr_shape = tuple(result_layout.properties.get("instr_shape", ()))
    if result_layout.element_type == "f32" and instr_shape == (16, 16, 32):
        return 4
    if result_layout.element_type == "f32" and instr_shape == (16, 16, 128):
        return 4
    if result_layout.element_type == "f32" and instr_shape == (32, 32, 16):
        return 16
    fail(
        "TLXW_OP_FRAGMENT_CONSTANT",
        STAGE,
        "accumulator fragment registers are not known for "
        f"element_type={result_layout.element_type}, instr_shape={instr_shape}",
        source_op_index=op.index,
        source_value_id=result_layout.value_id,
    )


def _mma_kind(element_type, instr_shape, op):
    if instr_shape == (16, 16, 32) and element_type == "f16":
        return "mfma.f32.16x16x32.f16"
    if instr_shape == (16, 16, 32) and element_type == "bf16":
        return "mfma.f32.16x16x32.bf16"
    if instr_shape == (32, 32, 16) and element_type == "f16":
        return "mfma.f32.32x32x16.f16"
    if instr_shape == (32, 32, 16) and element_type == "bf16":
        return "mfma.f32.32x32x16.bf16"
    fail(
        "TLXW_OP_DOT",
        STAGE,
        f"unsupported MFMA element type {element_type} for {instr_shape}",
        source_op_index=op.index,
    )


def _scaled_mma_kind(a_elem_type, b_elem_type, instr_shape, op):
    if instr_shape == (16, 16, 128) and a_elem_type == "e2m1" and b_elem_type == "e2m1":
        return "mfma.scale.f32.16x16x128.f4.f4"
    fail(
        "TLXW_OP_DOT_SCALED",
        STAGE,
        "unsupported native scaled MFMA element types "
        f"lhs={a_elem_type}, rhs={b_elem_type}, instr_shape={instr_shape}",
        source_op_index=op.index,
    )


def _scale_dot_elem_type(attr, op):
    mapping = {
        0: "e4m3",
        1: "e5m2",
        2: "e2m3",
        3: "e3m2",
        4: "e2m1",
        5: "bf16",
        6: "f16",
    }
    try:
        return mapping[int(attr)]
    except (TypeError, ValueError, KeyError):
        fail(
            "TLXW_OP_DOT_SCALED",
            STAGE,
            f"unknown tt.dot_scaled element type attribute {attr}",
            source_op_index=op.index,
        )


def _require_scaled_mma_scale(type_layout_program, scale, op_idx, op):
    if scale.type.element_type != "i8":
        fail(
            "TLXW_OP_DOT_SCALED",
            STAGE,
            "native scaled MFMA lowering expects i8 scale operands",
            source_op_index=op.index,
            source_value_id=scale.value_id,
        )
    if scale.type.representation not in {"simd", "simd_tuple"}:
        fail(
            "TLXW_OP_DOT_SCALED",
            STAGE,
            "native scaled MFMA lowering expects SIMD scale operands",
            source_op_index=op.index,
            source_value_id=scale.value_id,
        )
    if scale.layout_map_id is None:
        fail(
            "TLXW_OP_DOT_SCALED",
            STAGE,
            "native scaled MFMA lowering requires scale layout metadata",
            source_op_index=op.index,
            source_value_id=scale.value_id,
        )
    scale_layout = type_layout_program.layouts[int(scale.layout_map_id)]
    # The existing AMD lowering requires linear scale layouts. The bridge also
    # accepts generic_linear because Triton may preserve the same linear map
    # under the generic spelling when the map is only required to be surjective.
    if scale_layout.kind not in {"linear", "generic_linear"}:
        fail(
            "TLXW_OP_DOT_SCALED",
            STAGE,
            "native scaled MFMA lowering expects linear scale layouts",
            source_op_index=op.index,
            source_value_id=scale.value_id,
        )
    del op_idx


def _scaled_mma_scale_pack_attrs(
    m_tiles,
    n_tiles,
    k_tiles,
    lhs_scale_component_count,
    rhs_scale_component_count,
    op,
):
    lhs = _scaled_mma_one_scale_pack_attrs(
        "lhs_scale",
        m_tiles,
        k_tiles,
        lhs_scale_component_count,
        op,
    )
    rhs = _scaled_mma_one_scale_pack_attrs(
        "rhs_scale",
        n_tiles,
        k_tiles,
        rhs_scale_component_count,
        op,
    )
    return {**lhs, **rhs}


def _scaled_mma_one_scale_pack_attrs(prefix, non_k_tiles, k_tiles, component_count, op):
    scale_pack_width = min(4, int(non_k_tiles) * int(k_tiles))
    k_packed_vals = min(4, int(k_tiles))
    if scale_pack_width <= 0 or k_packed_vals <= 0 or scale_pack_width % k_packed_vals:
        fail(
            "TLXW_OP_DOT_SCALED",
            STAGE,
            "scaled MFMA scale packing requires K packing to evenly divide "
            f"the scale pack width; non_k_tiles={non_k_tiles}, "
            f"k_tiles={k_tiles}",
            source_op_index=op.index,
        )
    non_k_packed_vals = scale_pack_width // k_packed_vals
    k_groups = _ceil_div(int(k_tiles), int(k_packed_vals))
    non_k_groups = _ceil_div(int(non_k_tiles), int(non_k_packed_vals))
    group_count = int(k_groups) * int(non_k_groups)
    required_components = int(group_count) * int(scale_pack_width)
    if int(component_count) < required_components:
        fail(
            "TLXW_OP_DOT_SCALED",
            STAGE,
            "scaled MFMA scale operand does not provide enough components "
            f"for packed scale groups; need {required_components}, got "
            f"{component_count}",
            source_op_index=op.index,
        )
    return {
        f"{prefix}_group_count": int(group_count),
        f"{prefix}_k_groups": int(k_groups),
        f"{prefix}_k_packed_vals": int(k_packed_vals),
        f"{prefix}_non_k_groups": int(non_k_groups),
        f"{prefix}_non_k_packed_vals": int(non_k_packed_vals),
        f"{prefix}_pack_width": int(scale_pack_width),
    }


def _operand_fragment_shape(instr_shape, op):
    if instr_shape in {(16, 16, 32), (16, 16, 128)}:
        return 16, 16
    if instr_shape == (32, 32, 16):
        return 32, 32
    fail(
        "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
        STAGE,
        f"unsupported local_load MFMA instruction shape {instr_shape}",
        source_op_index=op.index,
    )


def _acc_fragment_shape(instr_shape, op):
    if instr_shape in {(16, 16, 32), (16, 16, 128)}:
        return 16, 16
    if instr_shape == (32, 32, 16):
        return 32, 32
    fail(
        "TLXW_OP_FRAGMENT_CONSTANT",
        STAGE,
        f"unsupported accumulator MFMA instruction shape {instr_shape}",
        source_op_index=op.index,
    )


def _mfma_output_tile_shape(instr_shape, op):
    if instr_shape in {(16, 16, 32), (16, 16, 128), (32, 32, 16)}:
        return 32, 32
    fail(
        "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
        STAGE,
        f"unsupported MFMA instruction shape {instr_shape}",
        source_op_index=op.index,
    )


def _mfma_k_dim(instr_shape, op):
    if instr_shape == (16, 16, 32):
        return 32
    if instr_shape == (16, 16, 128):
        return 128
    if instr_shape == (32, 32, 16):
        return 16
    fail(
        "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
        STAGE,
        f"unsupported MFMA K dimension for instruction shape {instr_shape}",
        source_op_index=op.index,
    )


def _has_mma_packet_result(type_layout_program, op):
    for value_id in op.results:
        converted = type_layout_program.values[value_id]
        if converted.type.representation in _MMA_PACKET_REPRESENTATIONS:
            return True
    return False


def _has_register_vector_payload_result(builder, type_layout_program, op):
    if len(op.operands) != 1:
        return False
    source_target_ids = builder.source_value_targets.get(int(op.operands[0]), ())
    if len(source_target_ids) != 1:
        return False
    source_target_id = int(source_target_ids[0])
    source_mode = None
    for target_op in reversed(builder.ops):
        if source_target_id not in target_op.results:
            continue
        source_mode = target_ir.attrs_dict(target_op).get("result_value_mode")
        break
    if source_mode != "register_vector_payload":
        return False
    for value_id in op.results:
        converted = type_layout_program.values[value_id]
        if _register_vector_payload_registers_or_none(
                type_layout_program,
                converted,
                op,
        ) is not None:
            return True
    return False


def _require_allowed_mma_packet_results(type_layout_program, op):
    if op.name in _MMA_PACKET_RESULT_SOURCE_OPS:
        return
    for value_id in op.results:
        converted = type_layout_program.values[value_id]
        if converted.type.representation not in _MMA_PACKET_REPRESENTATIONS:
            continue
        fail(
            "TLXW_OP_MMA_PACKET_PRODUCER",
            STAGE,
            "MMA SIMD packets may only be produced by packet-aware ops, "
            "control-flow carries, or explicit ttg.convert_layout; "
            f"got {op.name}",
            source_op_index=op.index,
            source_value_id=value_id,
        )


def _require_layout(type_layout_program, layout_map_id, op):
    if layout_map_id is None:
        fail(
            "TLXW_OP_MISSING_LAYOUT",
            STAGE,
            "operation requires a converted layout map",
            source_op_index=op.index,
        )
    return type_layout_program.layouts[int(layout_map_id)]


def _require_dot_operand_layout(layout, op_idx, op):
    if layout.kind != "dot_operand" or int(layout.properties.get("op_idx", -1)) != int(op_idx):
        fail(
            "TLXW_OP_DOT",
            STAGE,
            f"tt.dot operand {op_idx} must use matching dot_operand layout",
            source_op_index=op.index,
            source_value_id=layout.value_id,
        )


def _require_dot_operand_parent_layout(operand_layout, result_layout, op_idx, op):
    parent_kind = operand_layout.properties.get("parent_kind")
    parent_properties = operand_layout.properties.get("parent_properties", {})
    if parent_kind == result_layout.kind and parent_properties == result_layout.properties:
        return
    fail(
        "TLXW_OP_DOT",
        STAGE,
        f"tt.dot operand {op_idx} parent MFMA layout must match the result layout",
        source_op_index=op.index,
        source_value_id=operand_layout.value_id,
    )


def _mfma_per_wave_tiles(result_layout, instr_shape, warps_per_cta, op):
    if len(instr_shape) < 3 or len(warps_per_cta) < 2:
        fail(
            "TLXW_OP_DOT",
            STAGE,
            "tt.dot requires MFMA instrShape and warpsPerCTA metadata",
            source_op_index=op.index,
            source_value_id=result_layout.value_id,
        )
    warps_m = int(warps_per_cta[0])
    warps_n = int(warps_per_cta[1])
    if warps_m <= 0 or warps_n <= 0:
        fail(
            "TLXW_OP_DOT",
            STAGE,
            f"invalid MFMA warpsPerCTA {warps_per_cta}",
            source_op_index=op.index,
            source_value_id=result_layout.value_id,
        )
    total_m_tiles = _ceil_div(int(result_layout.shape[0]), int(instr_shape[0]))
    total_n_tiles = _ceil_div(int(result_layout.shape[1]), int(instr_shape[1]))
    return _ceil_div(total_m_tiles, warps_m), _ceil_div(total_n_tiles, warps_n)


def _dot_operand_k_tiles(layout, instr_shape, op):
    k_tile_extent = layouts.dot_operand_storage_k_tile_extent(
        layout.element_type,
        layout.properties,
    )
    op_idx = int(layout.properties["op_idx"])
    if op_idx == 0:
        if len(layout.shape) < 2:
            fail(
                "TLXW_OP_DOT",
                STAGE,
                "A dot operand requires rank-2 shape",
                source_op_index=op.index,
                source_value_id=layout.value_id,
            )
        return _ceil_div(int(layout.shape[1]), k_tile_extent)
    if op_idx == 1:
        if len(layout.shape) < 2:
            fail(
                "TLXW_OP_DOT",
                STAGE,
                "B dot operand requires rank-2 shape",
                source_op_index=op.index,
                source_value_id=layout.value_id,
            )
        return _ceil_div(int(layout.shape[0]), k_tile_extent)
    fail(
        "TLXW_OP_DOT",
        STAGE,
        f"unsupported dot operand index {op_idx}",
        source_op_index=op.index,
        source_value_id=layout.value_id,
    )


def _fragment_local_load_plan(
    conversion_input,
    type_layout_program,
    memdesc_value_id,
    result_layout,
    component_count,
    registers,
    op,
):
    memdesc = _memdesc_info(conversion_input, memdesc_value_id, op)
    view = _memdesc_view_info(conversion_input, memdesc_value_id, op)
    layout_id = type_layout_program.values[memdesc_value_id].layout_map_id
    layout = (None if layout_id is None else type_layout_program.layouts[int(layout_id)])
    transpose_plan = _transpose_mma_payload_load_plan(
        memdesc,
        view,
        layout,
        result_layout,
        component_count,
        registers,
        op,
    )
    if transpose_plan is not None:
        return transpose_plan
    indexed_plan = _indexed_mma_payload_load_plan(
        memdesc,
        view,
        layout,
        result_layout,
        component_count,
        registers,
        op,
    )
    if indexed_plan is not None:
        return indexed_plan
    offset_plan = _fragment_component_dword_offsets(
        conversion_input,
        type_layout_program,
        memdesc_value_id,
        result_layout,
        component_count,
        registers,
        op,
    )
    return {
        "component_dword_offsets": tuple(offset_plan["component_dword_offsets"]),
        "load_mode": "mma_payload_load",
        "warps_per_cta": tuple(offset_plan["warps_per_cta"]),
        "wave_tile_axis": offset_plan["wave_tile_axis"],
        "wave_tile_stride_dwords": int(offset_plan["wave_tile_stride_dwords"]),
    }


def _transpose_mma_payload_load_plan(
    memdesc,
    view,
    layout,
    result_layout,
    component_count,
    registers,
    op,
):
    parent = result_layout.properties.get("parent_properties", {})
    instr_shape = tuple(parent.get("instr_shape", ()))
    op_idx = int(result_layout.properties.get("op_idx", -1))
    element_byte_width = int(memdesc.element_byte_width or 0)
    packet = _transpose_load_packet_shape(
        result_layout.element_type,
        element_byte_width,
        instr_shape,
        op_idx,
        int(result_layout.lane_width),
        int(registers),
    )
    if packet is None:
        return None
    if memdesc.element_type != result_layout.element_type:
        return None
    if not _is_supported_transpose_layout(layout):
        return None
    chunk_elements, packet_elements = packet
    tile_plan = _fragment_component_tile_offsets(
        memdesc,
        result_layout,
        component_count,
        op,
    )
    source_shape = _dot_operand_source_shape(result_layout, instr_shape, op)
    elements_per_lane = int(registers) * (4 // int(element_byte_width))
    if int(elements_per_lane) % int(chunk_elements):
        fail(
            "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            "transpose local_load requires a whole number of transpose-load "
            f"chunks per lane, got elements_per_lane={elements_per_lane} "
            f"and chunk_elements={chunk_elements}",
            source_op_index=op.index,
            source_value_id=result_layout.value_id,
        )
    lane_layout = _mma_access_lane_layout(
        result_layout,
        instr_shape,
        int(elements_per_lane),
        transpose_load=True,
    )
    for tile_offsets in tile_plan["component_tile_offsets"]:
        _validate_transpose_load_packets(
            layout,
            tuple(int(dim) for dim in memdesc.shape),
            view.physical_shape,
            view.logical_origin,
            source_shape,
            tile_offsets,
            int(element_byte_width),
            int(result_layout.lane_width),
            elements_per_lane,
            (0, ),
            int(chunk_elements),
            int(packet_elements),
            lane_layout,
            op,
        )
    chunks_per_component = int(elements_per_lane) // int(chunk_elements)
    chunk_element_deltas = _transpose_load_chunk_element_deltas(
        layout,
        tuple(int(dim) for dim in memdesc.shape),
        view.physical_shape,
        view.logical_origin,
        source_shape,
        tuple(tile_plan["component_tile_offsets"]),
        int(element_byte_width),
        int(result_layout.lane_width),
        elements_per_lane,
        chunks_per_component,
        int(chunk_elements),
        tile_plan["wave_tile_axis"],
        tuple(tile_plan["warps_per_cta"]),
        int(tile_plan["wave_tile_stride_elements"]),
        lane_layout,
        op,
    )
    attrs = _encoded_shared_layout_attrs(
        layout,
        view.physical_shape,
        element_byte_width,
        op,
    )
    result = {
        **attrs,
        "chunk_elements": int(chunk_elements),
        "chunks_per_component": int(chunks_per_component),
        "component_tile_offsets": tuple(tile_plan["component_tile_offsets"]),
        "elements_per_lane": int(elements_per_lane),
        "load_mode": "transpose_mma_payload_load",
        "memdesc_shape": tuple(int(dim) for dim in memdesc.shape),
        "memdesc_logical_origin": tuple(int(value) for value in view.logical_origin),
        "memdesc_physical_shape": tuple(int(dim) for dim in view.physical_shape),
        "packet_elements": int(packet_elements),
        "source_shape": tuple(source_shape),
        "warps_per_cta": tuple(tile_plan["warps_per_cta"]),
        "wave_tile_axis": tile_plan["wave_tile_axis"],
        "wave_tile_stride_elements": int(tile_plan["wave_tile_stride_elements"]),
        **_mma_access_attrs(
            result_layout,
            instr_shape,
            elements_per_lane,
            lane_layout,
            transpose_load=True,
        ),
    }
    if chunk_element_deltas is not None:
        result["chunk_element_deltas"] = tuple(chunk_element_deltas)
    return result


def _transpose_load_packet_shape(
    element_type,
    element_byte_width,
    instr_shape,
    op_idx,
    lane_width,
    registers,
):
    if not (int(op_idx) == 1 and int(lane_width) == 64 and int(registers) == 4):
        return None
    instr_shape = tuple(int(value) for value in instr_shape)
    element_type = str(element_type)
    if (element_type in {"f16", "bf16", "i16"} and int(element_byte_width) == 2
            and instr_shape in {(16, 16, 32), (32, 32, 16)}):
        return (4, 4)
    return None


def _indexed_mma_payload_load_plan(
    memdesc,
    view,
    layout,
    result_layout,
    component_count,
    registers,
    op,
):
    parent = result_layout.properties.get("parent_properties", {})
    instr_shape = tuple(parent.get("instr_shape", ()))
    supported_element_widths = {
        "bf16": 2,
        "f16": 2,
        "i8": 1,
    }
    if result_layout.element_type not in supported_element_widths:
        return None
    if memdesc.element_type != result_layout.element_type:
        return None
    element_byte_width = int(memdesc.element_byte_width or 0)
    if element_byte_width != supported_element_widths[result_layout.element_type]:
        return None
    requires_indexed = False
    if (tuple(int(dim) for dim in view.physical_shape) != tuple(int(dim) for dim in memdesc.shape)
            or any(int(value) for value in view.logical_origin)):
        requires_indexed = True
    if layout is None or layout.kind == "none":
        pass
    elif layout.kind in {"linear", "generic_linear"}:
        return None
    elif layout.kind == "swizzled_shared":
        if _is_identity_swizzled_layout(layout):
            pass
        else:
            layouts.swizzled_shared_parameters(
                layout,
                view.physical_shape,
                stage=STAGE,
                diagnostic="TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
                source_op_index=op.index,
                source_value_id=layout.value_id,
            )
            requires_indexed = True
    elif layout.kind == "padded_shared":
        _padded_shared_parameters(layout, op)
        requires_indexed = True
    elif layout.kind == "shared_linear":
        layouts.shared_linear_inverse_offset_bases(
            layout,
            view.physical_shape,
            stage=STAGE,
            diagnostic="TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
            source_op_index=op.index,
            source_value_id=layout.value_id,
        )
        requires_indexed = True
    else:
        return None
    tile_plan = _fragment_component_tile_offsets(
        memdesc,
        result_layout,
        component_count,
        op,
    )
    source_shape = _dot_operand_source_shape(result_layout, instr_shape, op)
    if len(memdesc.shape) != len(source_shape):
        return None
    if tuple(int(dim) for dim in memdesc.shape) != tuple(int(dim) for dim in source_shape):
        requires_indexed = True
    if not requires_indexed:
        return None
    layout_attrs = _encoded_shared_layout_attrs(
        layout,
        view.physical_shape,
        memdesc.element_byte_width,
        op,
    )
    elements_per_lane = int(registers) * (4 // element_byte_width)
    lane_layout = _indexed_mma_payload_lane_layout(
        layout,
        result_layout,
        instr_shape,
        int(elements_per_lane),
    )
    load_mode = ("swizzled_mma_payload_load" if layout is not None and layout.kind == "swizzled_shared"
                 and not _is_identity_swizzled_layout(layout) else "indexed_mma_payload_load")
    wave_offsets = ((0, ) if load_mode == "swizzled_mma_payload_load" else _possible_wave_tile_element_offsets(
        tile_plan["wave_tile_axis"],
        tuple(tile_plan["warps_per_cta"]),
        int(tile_plan["wave_tile_stride_elements"]),
        op,
    ))
    for tile_offsets in tile_plan["component_tile_offsets"]:
        _validate_mma_payload_load_packets(
            layout,
            tuple(int(dim) for dim in memdesc.shape),
            view.physical_shape,
            view.logical_origin,
            source_shape,
            tile_offsets,
            element_byte_width,
            int(result_layout.lane_width),
            elements_per_lane,
            wave_offsets,
            op,
            lane_layout=lane_layout,
        )
    result = {
        **layout_attrs,
        "component_tile_offsets": tuple(tile_plan["component_tile_offsets"]),
        "elements_per_lane": int(elements_per_lane),
        "load_mode": load_mode,
        "memdesc_shape": tuple(int(dim) for dim in memdesc.shape),
        "memdesc_logical_origin": tuple(int(value) for value in view.logical_origin),
        "memdesc_physical_shape": tuple(int(dim) for dim in view.physical_shape),
        "source_shape": tuple(source_shape),
        "warps_per_cta": tuple(tile_plan["warps_per_cta"]),
        "wave_tile_axis": tile_plan["wave_tile_axis"],
        "wave_tile_stride_elements": int(tile_plan["wave_tile_stride_elements"]),
        **_mma_access_attrs(
            result_layout,
            instr_shape,
            elements_per_lane,
            lane_layout,
        ),
    }
    physical_deltas = _mma_payload_canonical_physical_element_deltas(
        layout,
        tuple(int(dim) for dim in memdesc.shape),
        tuple(int(dim) for dim in view.physical_shape),
        tuple(int(value) for value in view.logical_origin),
        tuple(int(dim) for dim in source_shape),
        tuple(tile_plan["component_tile_offsets"]),
        int(element_byte_width),
        int(result_layout.lane_width),
        int(elements_per_lane),
        tuple(wave_offsets),
        str(lane_layout),
        op,
    )
    if physical_deltas is not None:
        result.update({
            "component_physical_element_deltas": tuple(physical_deltas),
            "physical_base_logical_origin": (0, ) * len(view.logical_origin),
            "physical_base_tile_offsets": (0, ) * len(source_shape),
        })
    return result


def _mma_payload_canonical_physical_element_deltas(
    layout,
    memdesc_shape,
    memdesc_physical_shape,
    memdesc_logical_origin,
    source_shape,
    component_tile_offsets,
    element_byte_width,
    lane_width,
    elements_per_lane,
    wave_offsets,
    lane_layout,
    op,
):
    """Prove packet bases are a canonical base plus static physical deltas.

    Indexed fragment coordinates commonly differ only in layout bits that are
    invariant over the lane and wave coordinates.  Keeping those differences
    inside a fully expanded symbolic layout expression prevents the target
    from selecting LDS immediate offsets.  Prove the additive form directly
    from the imported layout by exhaustively checking the finite lane/warp
    domain.  Returning ``None`` preserves the general symbolic path whenever
    the relation is not exact.
    """
    memdesc_shape = tuple(int(dim) for dim in memdesc_shape)
    memdesc_physical_shape = tuple(int(dim) for dim in memdesc_physical_shape)
    memdesc_logical_origin = tuple(int(value) for value in memdesc_logical_origin)
    source_shape = tuple(int(dim) for dim in source_shape)
    component_tile_offsets = tuple(
        tuple(int(value) for value in offsets)
        for offsets in component_tile_offsets
    )
    wave_offsets = tuple(int(value) for value in wave_offsets)
    element_byte_width = int(element_byte_width)
    if (
        element_byte_width <= 0
        or not component_tile_offsets
        or len(memdesc_shape) != len(memdesc_physical_shape)
        or len(memdesc_shape) != len(memdesc_logical_origin)
        or len(memdesc_shape) != len(source_shape)
        or any(len(offsets) != len(source_shape) for offsets in component_tile_offsets)
    ):
        return None

    canonical_tile_offsets = (0, ) * len(source_shape)
    canonical_logical_origin = (0, ) * len(memdesc_logical_origin)
    deltas = []
    for tile_offsets in component_tile_offsets:
        expected_delta = None
        for wave_offset in wave_offsets:
            for lane in range(int(lane_width)):
                canonical_linear = _static_local_fragment_lane_offset(
                    memdesc_shape,
                    source_shape,
                    canonical_tile_offsets,
                    lane,
                    int(elements_per_lane),
                    0,
                    wave_offset,
                    lane_layout,
                    op,
                    fail_on_oob=False,
                )
                component_linear = _static_local_fragment_lane_offset(
                    memdesc_shape,
                    source_shape,
                    tile_offsets,
                    lane,
                    int(elements_per_lane),
                    0,
                    wave_offset,
                    lane_layout,
                    op,
                    fail_on_oob=False,
                )
                if canonical_linear is None or component_linear is None:
                    return None
                canonical_byte_offset = _static_memdesc_view_byte_offset_from_linear(
                    layout,
                    memdesc_shape,
                    memdesc_physical_shape,
                    canonical_logical_origin,
                    canonical_linear,
                    element_byte_width,
                    op,
                )
                component_byte_offset = _static_memdesc_view_byte_offset_from_linear(
                    layout,
                    memdesc_shape,
                    memdesc_physical_shape,
                    memdesc_logical_origin,
                    component_linear,
                    element_byte_width,
                    op,
                )
                if canonical_byte_offset is None or component_byte_offset is None:
                    return None
                byte_delta = int(component_byte_offset) - int(canonical_byte_offset)
                if byte_delta % element_byte_width:
                    return None
                element_delta = byte_delta // element_byte_width
                if expected_delta is None:
                    expected_delta = int(element_delta)
                elif int(element_delta) != int(expected_delta):
                    return None
        if expected_delta is None:
            return None
        deltas.append(int(expected_delta))
    return tuple(deltas)


def _fragment_component_tile_offsets(memdesc, result_layout, component_count, op):
    shape = tuple(int(dim) for dim in memdesc.shape)
    op_idx = int(result_layout.properties["op_idx"])
    parent = result_layout.properties.get("parent_properties", {})
    instr_shape = tuple(parent.get("instr_shape", ()))
    warps_per_cta = tuple(parent.get("warps_per_cta", ()))
    tiles_per_warp = tuple(parent.get("tiles_per_warp", (1, 1)))
    if len(tiles_per_warp) < 2:
        tiles_per_warp = (1, 1)
    k_tile_extent = layouts.dot_operand_storage_k_tile_extent(
        result_layout.element_type,
        result_layout.properties,
    )
    if len(instr_shape) < 3 or len(warps_per_cta) < 2:
        fail(
            "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            "dot-operand local_load requires MFMA instrShape and warpsPerCTA",
            source_op_index=op.index,
            source_value_id=result_layout.value_id,
        )
    tile_offsets = []
    wave_tile_axis = "none"
    wave_tile_stride_elements = 0
    for component in range(int(component_count)):
        if op_idx == 0:
            if len(shape) < 2:
                fail(
                    "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
                    STAGE,
                    "A-fragment local_load requires rank-2 memdesc shape",
                    source_op_index=op.index,
                    source_value_id=result_layout.value_id,
                )
            k_tiles = _ceil_div(shape[1], k_tile_extent)
            if component_count % k_tiles:
                fail(
                    "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
                    STAGE,
                    "A-fragment component count is not divisible by K tiles",
                    source_op_index=op.index,
                    source_value_id=result_layout.value_id,
                )
            m_tile = component // k_tiles
            k_tile = component % k_tiles
            warps_m = max(1, int(warps_per_cta[0]))
            tiles_m = max(1, int(tiles_per_warp[0]))
            cta_m_tile = layouts.mfma_cta_tile_coordinate(
                m_tile, 0, warps_m, tiles_m)
            tile_offsets.append((
                cta_m_tile * instr_shape[0],
                k_tile * k_tile_extent,
            ))
            wave_tile_axis = "m"
            wave_tile_stride_elements = tiles_m * instr_shape[0] * shape[1]
        elif op_idx == 1:
            if len(shape) < 2:
                fail(
                    "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
                    STAGE,
                    "B-fragment local_load requires rank-2 memdesc shape",
                    source_op_index=op.index,
                    source_value_id=result_layout.value_id,
                )
            k_tiles = _ceil_div(shape[0], k_tile_extent)
            if component_count % k_tiles:
                fail(
                    "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
                    STAGE,
                    "B-fragment component count is not divisible by K tiles",
                    source_op_index=op.index,
                    source_value_id=result_layout.value_id,
                )
            n_tile = component // k_tiles
            k_tile = component % k_tiles
            warps_n = max(1, int(warps_per_cta[1]))
            tiles_n = max(1, int(tiles_per_warp[1]))
            cta_n_tile = layouts.mfma_cta_tile_coordinate(
                n_tile, 0, warps_n, tiles_n)
            tile_offsets.append((
                k_tile * k_tile_extent,
                cta_n_tile * instr_shape[1],
            ))
            wave_tile_axis = "n"
            wave_tile_stride_elements = tiles_n * instr_shape[1]
        else:
            fail(
                "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
                STAGE,
                f"unsupported dot operand index {op_idx}",
                source_op_index=op.index,
                source_value_id=result_layout.value_id,
            )
    return {
        "component_tile_offsets": tuple(tuple(int(coord) for coord in offsets) for offsets in tile_offsets),
        "warps_per_cta": tuple(int(value) for value in warps_per_cta),
        "wave_tile_axis": wave_tile_axis,
        "wave_tile_stride_elements": int(wave_tile_stride_elements),
    }


def _dot_operand_source_shape(result_layout, instr_shape, op):
    k_dim = layouts.dot_operand_storage_k_tile_extent(
        result_layout.element_type,
        result_layout.properties,
    )
    op_idx = int(result_layout.properties.get("op_idx", -1))
    if op_idx == 0:
        return (int(instr_shape[0]), k_dim)
    if op_idx == 1:
        return (k_dim, int(instr_shape[1]))
    fail(
        "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
        STAGE,
        f"unsupported dot operand index {op_idx}",
        source_op_index=op.index,
        source_value_id=result_layout.value_id,
    )


def _mma_access_lane_layout(
    result_layout,
    instr_shape,
    elements_per_lane,
    *,
    transpose_load=False,
):
    op_idx = int(result_layout.properties.get("op_idx", -1))
    if (tuple(int(value) for value in instr_shape) == (16, 16, 128) and result_layout.element_type == "i8"
            and int(result_layout.lane_width) == 64 and int(elements_per_lane) == 16):
        if op_idx == 0:
            return "gfx950_mfma_a"
        if op_idx == 1:
            return "gfx950_mfma_b"
    if (op_idx == 0 and tuple(int(value) for value in instr_shape) in {(16, 16, 32), (32, 32, 16)}
            and int(result_layout.lane_width) == 64 and int(elements_per_lane) == 8):
        return "gfx950_mfma_a"
    if (op_idx == 1 and bool(transpose_load) and tuple(int(value) for value in instr_shape) in {(16, 16, 32),
                                                                                                (32, 32, 16)}
            and int(result_layout.lane_width) == 64 and int(elements_per_lane) == 8):
        return "gfx950_mfma_b_transpose"
    return "row_major_linear"


def _indexed_mma_payload_lane_layout(layout, result_layout, instr_shape, elements_per_lane):
    # A B operand stored with the K dimension as the physical minor dimension
    # is already arranged in the register order consumed by the gfx950 MFMA.
    # This is a property of the physical order, not of the particular shared
    # layout encoding: padded and swizzled layouts use the same lane map once
    # their structural offset plan proves each per-lane packet contiguous.
    if (layout is not None and tuple(layout.properties.get("order", ())) == (0, 1)
            and int(result_layout.properties.get("op_idx", -1)) == 1
            and tuple(int(value) for value in instr_shape) in {(16, 16, 32), (32, 32, 16)}
            and int(result_layout.lane_width) == 64 and int(elements_per_lane) == 8):
        return "gfx950_mfma_b"
    return _mma_access_lane_layout(
        result_layout,
        instr_shape,
        int(elements_per_lane),
    )


def _is_supported_transpose_layout(layout):
    if layout is None:
        return False
    if layout.kind == "padded_shared":
        return tuple(layout.properties.get("order", ())) == (1, 0)
    return _is_supported_swizzled_layout(layout)


def _is_supported_swizzled_layout(layout):
    if layout is None or layout.kind != "swizzled_shared":
        return False
    per_phase = int(layout.properties.get("per_phase", 0))
    max_phase = int(layout.properties.get("max_phase", 0))
    return (int(layout.properties.get("vec", 0)) == 8 and tuple(layout.properties.get("order", ())) == (1, 0)
            and (per_phase, max_phase) in {(1, 16), (2, 8), (4, 4)})


def _is_identity_swizzled_layout(layout):
    order = tuple(layout.properties.get("order", ())) if layout is not None else ()
    return (layout is not None and layout.kind == "swizzled_shared" and int(layout.properties.get("vec", 0)) == 1
            and int(layout.properties.get("per_phase", 0)) == 1 and int(layout.properties.get("max_phase", 0)) == 1
            and order in {(1, 0), (0, ), ()})


def _encoded_shared_layout_attrs(layout, shape, element_byte_width, op):
    plan = layouts.shared_physical_offset_expression_plan(
        layout,
        shape,
        element_byte_width,
        stage=STAGE,
        diagnostic="TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
        source_op_index=op.index,
        source_value_id=None if layout is None else layout.value_id,
    )
    return layouts.physical_offset_expression_plan_attrs(plan, "shared")


def _validate_transpose_load_packets(
    layout,
    memdesc_shape,
    memdesc_physical_shape,
    memdesc_logical_origin,
    source_shape,
    tile_offsets,
    element_byte_width,
    lane_width,
    elements_per_lane,
    wave_offsets,
    chunk_elements,
    packet_elements,
    lane_layout,
    op,
):
    required_chunk_alignment = 8 if int(element_byte_width) == 2 else 1
    packet_elements = int(packet_elements)
    chunk_elements = int(chunk_elements)
    if chunk_elements % packet_elements:
        fail(
            "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            "transpose load chunk size must be divisible by its contiguous "
            f"packet size, got chunk_elements={chunk_elements} and "
            f"packet_elements={packet_elements}",
            source_op_index=op.index,
        )
    chunks_per_component = int(elements_per_lane) // int(chunk_elements)
    for chunk in range(chunks_per_component):
        chunk_extra = int(chunk_elements) * chunk
        _validate_transpose_chunk_base_alignment(
            layout,
            memdesc_shape,
            memdesc_physical_shape,
            memdesc_logical_origin,
            source_shape,
            tile_offsets,
            element_byte_width,
            lane_width,
            elements_per_lane,
            wave_offsets,
            int(chunk_extra),
            required_chunk_alignment,
            lane_layout,
            op,
        )
        if int(element_byte_width) == 1:
            _validate_transpose_load_element_deltas(
                layout,
                memdesc_shape,
                memdesc_physical_shape,
                memdesc_logical_origin,
                source_shape,
                tile_offsets,
                element_byte_width,
                lane_width,
                elements_per_lane,
                wave_offsets,
                int(chunk_extra),
                int(chunk_elements),
                lane_layout,
                op,
            )
            continue
        for packet_start in range(0, int(chunk_elements), int(packet_elements)):
            packet_extra = int(chunk_extra) + int(packet_start)
            packet_alignment = int(packet_elements) * int(element_byte_width)
            _validate_mma_payload_load_packets(
                layout,
                memdesc_shape,
                memdesc_physical_shape,
                memdesc_logical_origin,
                source_shape,
                tile_offsets,
                element_byte_width,
                lane_width,
                elements_per_lane,
                wave_offsets,
                op,
                local_extra_elements=packet_extra,
                packet_elements=packet_elements,
                required_alignment=packet_alignment,
                packet_description="transpose load",
                lane_layout=lane_layout,
            )


def _validate_transpose_load_element_deltas(
    layout,
    memdesc_shape,
    memdesc_physical_shape,
    memdesc_logical_origin,
    source_shape,
    tile_offsets,
    element_byte_width,
    lane_width,
    elements_per_lane,
    wave_offsets,
    local_extra_elements,
    chunk_elements,
    lane_layout,
    op,
):
    expected_deltas = None
    for wave_offset in wave_offsets:
        for lane in range(int(lane_width)):
            first = None
            deltas = []
            for element in range(int(chunk_elements)):
                linear = _static_local_fragment_lane_offset(
                    memdesc_shape,
                    source_shape,
                    tile_offsets,
                    int(lane),
                    int(elements_per_lane),
                    int(local_extra_elements) + int(element),
                    int(wave_offset),
                    lane_layout,
                    op,
                )
                byte_offset = _static_memdesc_view_byte_offset_from_linear(
                    layout,
                    memdesc_shape,
                    memdesc_physical_shape,
                    memdesc_logical_origin,
                    linear,
                    int(element_byte_width),
                    op,
                )
                if byte_offset is None:
                    fail(
                        "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
                        STAGE,
                        "transpose load coordinate exceeds memdesc shape "
                        f"{memdesc_shape}",
                        source_op_index=op.index,
                    )
                if first is None:
                    first = int(byte_offset)
                deltas.append(int(byte_offset) - int(first))
            deltas = tuple(deltas)
            if expected_deltas is None:
                expected_deltas = deltas
            elif deltas != expected_deltas:
                fail(
                    "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
                    STAGE,
                    "transpose load element byte deltas are not lane-invariant",
                    source_op_index=op.index,
                )


def _validate_transpose_chunk_base_alignment(
    layout,
    memdesc_shape,
    memdesc_physical_shape,
    memdesc_logical_origin,
    source_shape,
    tile_offsets,
    element_byte_width,
    lane_width,
    elements_per_lane,
    wave_offsets,
    local_extra_elements,
    required_alignment,
    lane_layout,
    op,
):
    for wave_offset in wave_offsets:
        for lane in range(int(lane_width)):
            linear = _static_local_fragment_lane_offset(
                memdesc_shape,
                source_shape,
                tile_offsets,
                int(lane),
                int(elements_per_lane),
                int(local_extra_elements),
                int(wave_offset),
                lane_layout,
                op,
            )
            byte_offset = _static_memdesc_view_byte_offset_from_linear(
                layout,
                memdesc_shape,
                memdesc_physical_shape,
                memdesc_logical_origin,
                linear,
                int(element_byte_width),
                op,
            )
            if byte_offset is None:
                fail(
                    "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
                    STAGE,
                    "transpose load coordinate exceeds memdesc shape "
                    f"{memdesc_shape}",
                    source_op_index=op.index,
                )
            if byte_offset % int(required_alignment):
                fail(
                    "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
                    STAGE,
                    "transpose load packet physical byte offset "
                    f"{byte_offset} is not {required_alignment}-byte aligned",
                    source_op_index=op.index,
                )


def _transpose_load_chunk_element_deltas(
    layout,
    memdesc_shape,
    memdesc_physical_shape,
    memdesc_logical_origin,
    source_shape,
    component_tile_offsets,
    element_byte_width,
    lane_width,
    elements_per_lane,
    chunks_per_component,
    chunk_elements,
    wave_tile_axis,
    warps_per_cta,
    wave_tile_stride_elements,
    lane_layout,
    op,
):
    wave_offsets = _possible_wave_tile_element_offsets(
        wave_tile_axis,
        warps_per_cta,
        wave_tile_stride_elements,
        op,
    )
    result = []
    for tile_offsets in component_tile_offsets:
        component_deltas = []
        for chunk in range(int(chunks_per_component)):
            extra_elements = int(chunk_elements) * chunk
            byte_deltas = set()
            for wave_offset in wave_offsets:
                for lane in range(int(lane_width)):
                    base_linear = _static_local_fragment_lane_offset(
                        memdesc_shape,
                        source_shape,
                        tile_offsets,
                        int(lane),
                        int(elements_per_lane),
                        0,
                        int(wave_offset),
                        lane_layout,
                        op,
                        fail_on_oob=False,
                    )
                    if base_linear is None:
                        return None
                    chunk_linear = _static_local_fragment_lane_offset(
                        memdesc_shape,
                        source_shape,
                        tile_offsets,
                        int(lane),
                        int(elements_per_lane),
                        int(extra_elements),
                        int(wave_offset),
                        lane_layout,
                        op,
                        fail_on_oob=False,
                    )
                    if chunk_linear is None:
                        return None
                    base_byte = _static_memdesc_view_byte_offset_from_linear(
                        layout,
                        memdesc_shape,
                        memdesc_physical_shape,
                        memdesc_logical_origin,
                        base_linear,
                        int(element_byte_width),
                        op,
                    )
                    chunk_byte = _static_memdesc_view_byte_offset_from_linear(
                        layout,
                        memdesc_shape,
                        memdesc_physical_shape,
                        memdesc_logical_origin,
                        chunk_linear,
                        int(element_byte_width),
                        op,
                    )
                    if base_byte is None or chunk_byte is None:
                        return None
                    byte_deltas.add(chunk_byte - base_byte)
            if len(byte_deltas) != 1:
                return None
            byte_delta = next(iter(byte_deltas))
            if byte_delta % int(element_byte_width):
                return None
            component_deltas.append(byte_delta // int(element_byte_width))
        result.append(tuple(component_deltas))
    return tuple(result)


def _possible_wave_tile_element_offsets(
    wave_tile_axis,
    warps_per_cta,
    wave_tile_stride_elements,
    op,
):
    stride = int(wave_tile_stride_elements)
    if wave_tile_axis == "none" or stride == 0:
        return (0, )
    if len(warps_per_cta) < 2:
        fail(
            "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            "transpose load requires warpsPerCTA for wave-tile delta proof",
            source_op_index=op.index,
        )
    if wave_tile_axis == "m":
        count = int(warps_per_cta[0])
    elif wave_tile_axis == "n":
        count = int(warps_per_cta[1])
    else:
        fail(
            "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            f"unsupported transpose load wave axis {wave_tile_axis}",
            source_op_index=op.index,
        )
    if count <= 0:
        fail(
            "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            "transpose load wave-tile delta proof requires positive warp count",
            source_op_index=op.index,
        )
    return tuple(index * stride for index in range(count))


def _validate_mma_payload_load_packets(
    layout,
    memdesc_shape,
    memdesc_physical_shape,
    memdesc_logical_origin,
    source_shape,
    tile_offsets,
    element_byte_width,
    lane_width,
    elements_per_lane,
    wave_offsets,
    op,
    *,
    local_extra_elements=0,
    packet_elements=None,
    required_alignment=4,
    packet_description="fragment load",
    lane_layout="row_major_linear",
):
    packet_elements = (int(elements_per_lane) if packet_elements is None else int(packet_elements))
    for wave_offset in wave_offsets:
        for lane in range(int(lane_width)):
            first = None
            for element in range(int(packet_elements)):
                linear = _static_local_fragment_lane_offset(
                    memdesc_shape,
                    source_shape,
                    tile_offsets,
                    int(lane),
                    int(elements_per_lane),
                    int(local_extra_elements) + int(element),
                    int(wave_offset),
                    lane_layout,
                    op,
                )
                byte_offset = _static_memdesc_view_byte_offset_from_linear(
                    layout,
                    memdesc_shape,
                    memdesc_physical_shape,
                    memdesc_logical_origin,
                    linear,
                    int(element_byte_width),
                    op,
                )
                if byte_offset is None:
                    fail(
                        "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
                        STAGE,
                        "fragment load coordinate exceeds memdesc shape "
                        f"{memdesc_shape}",
                        source_op_index=op.index,
                    )
                if first is None:
                    first = byte_offset
                    if first % int(required_alignment):
                        fail(
                            "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
                            STAGE,
                            f"{packet_description} packet physical byte offset "
                            f"{first} is not {required_alignment}-byte aligned",
                            source_op_index=op.index,
                        )
                    continue
                expected = first + element * int(element_byte_width)
                if byte_offset != expected:
                    fail(
                        "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
                        STAGE,
                        f"{packet_description} packet is not physically "
                        f"contiguous at linear offset {linear}",
                        source_op_index=op.index,
                    )


def _static_local_fragment_lane_offset(
    memdesc_shape,
    source_shape,
    tile_offsets,
    lane,
    elements_per_lane,
    extra_elements,
    wave_offset,
    lane_layout,
    op,
    *,
    fail_on_oob=True,
):
    if lane_layout == "row_major_linear":
        return _static_local_fragment_linear_offset(
            memdesc_shape,
            source_shape,
            tile_offsets,
            int(lane) * int(elements_per_lane) + int(extra_elements),
            int(wave_offset),
            op,
            fail_on_oob=fail_on_oob,
        )
    if lane_layout == "gfx950_mfma_a":
        if len(memdesc_shape) != 2 or len(source_shape) != 2 or len(tile_offsets) != 2:
            fail(
                "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
                STAGE,
                "gfx950 MFMA A fragment load requires rank-2 source and memdesc shapes",
                source_op_index=op.index,
            )
        row = int(lane) % int(source_shape[0])
        col = ((int(lane) // int(source_shape[0])) * int(elements_per_lane) + int(extra_elements))
        coords = (int(tile_offsets[0]) + row, int(tile_offsets[1]) + col)
        for dim, coord in enumerate(coords):
            if int(coord) < 0 or int(coord) >= int(memdesc_shape[dim]):
                if not fail_on_oob:
                    return None
                _check_coords_in_bounds(coords, memdesc_shape, op)
        linear = _static_linear_offset(memdesc_shape, coords) + int(wave_offset)
        if int(linear) < 0 or int(linear) >= _product(memdesc_shape):
            if not fail_on_oob:
                return None
        return linear
    if lane_layout == "gfx950_mfma_b_transpose":
        if len(memdesc_shape) != 2 or len(source_shape) != 2 or len(tile_offsets) != 2:
            fail(
                "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
                STAGE,
                "gfx950 MFMA B transpose load requires rank-2 source and memdesc shapes",
                source_op_index=op.index,
            )
        non_k_dim = int(source_shape[1])
        if non_k_dim % 16:
            fail(
                "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
                STAGE,
                "gfx950 MFMA B transpose load requires the non-K dimension "
                "to be a multiple of the ds_read_tr group width",
                source_op_index=op.index,
            )
        lane_in_group = int(lane) % 16
        non_k_group = (int(lane) % non_k_dim) // 16
        k_group = int(lane) // non_k_dim
        chunk_k = (int(extra_elements) // 4) * 4
        packet_col = int(extra_elements) % 4
        row = (k_group * int(elements_per_lane) + chunk_k + lane_in_group // 4)
        col = non_k_group * 16 + 4 * (lane_in_group % 4) + packet_col
        coords = (int(tile_offsets[0]) + row, int(tile_offsets[1]) + col)
        for dim, coord in enumerate(coords):
            if int(coord) < 0 or int(coord) >= int(memdesc_shape[dim]):
                if not fail_on_oob:
                    return None
                _check_coords_in_bounds(coords, memdesc_shape, op)
        linear = _static_linear_offset(memdesc_shape, coords) + int(wave_offset)
        if int(linear) < 0 or int(linear) >= _product(memdesc_shape):
            if not fail_on_oob:
                return None
        return linear
    if lane_layout == "gfx950_mfma_b":
        if len(memdesc_shape) != 2 or len(source_shape) != 2 or len(tile_offsets) != 2:
            fail(
                "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
                STAGE,
                "gfx950 MFMA B fragment load requires rank-2 source and memdesc shapes",
                source_op_index=op.index,
            )
        non_k_dim = int(source_shape[1])
        col = int(lane) % non_k_dim
        row = (int(lane) // non_k_dim) * int(elements_per_lane) + int(extra_elements)
        coords = (int(tile_offsets[0]) + row, int(tile_offsets[1]) + col)
        for dim, coord in enumerate(coords):
            if int(coord) < 0 or int(coord) >= int(memdesc_shape[dim]):
                if not fail_on_oob:
                    return None
                _check_coords_in_bounds(coords, memdesc_shape, op)
        linear = _static_linear_offset(memdesc_shape, coords) + int(wave_offset)
        if int(linear) < 0 or int(linear) >= _product(memdesc_shape):
            if not fail_on_oob:
                return None
        return linear
    fail(
        "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
        STAGE,
        f"unsupported fragment lane layout {lane_layout}",
        source_op_index=op.index,
    )


def _mma_access_attrs(
    result_layout,
    instr_shape,
    elements_per_lane,
    lane_layout,
    *,
    transpose_load=False,
):
    return {
        "mma_access_element_type": result_layout.element_type,
        "mma_access_instr_shape": tuple(int(value) for value in instr_shape),
        "mma_access_lane_layout": str(lane_layout),
        "mma_access_lane_width": int(result_layout.lane_width or 64),
        "mma_access_role": int(result_layout.properties.get("op_idx", -1)),
        "mma_access_transpose_load": bool(transpose_load),
        "mma_access_vector_payload_width": int(elements_per_lane),
    }


def _static_local_fragment_linear_offset(
    memdesc_shape,
    source_shape,
    tile_offsets,
    local_linear,
    wave_offset,
    op,
    *,
    fail_on_oob=True,
):
    if len(memdesc_shape) != len(source_shape) or len(tile_offsets) != len(source_shape):
        fail(
            "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            "fragment load coordinate remap requires matching ranks",
            source_op_index=op.index,
        )
    local_coords = _static_delinearize_row_major(
        int(local_linear),
        source_shape,
        op,
    )
    coords = tuple(int(tile_offsets[dim]) + int(local_coords[dim]) for dim in range(len(source_shape)))
    for dim, coord in enumerate(coords):
        if int(coord) < 0 or int(coord) >= int(memdesc_shape[dim]):
            if not fail_on_oob:
                return None
            _check_coords_in_bounds(coords, memdesc_shape, op)
    linear = _static_linear_offset(memdesc_shape, coords) + int(wave_offset)
    if int(linear) < 0 or int(linear) >= _product(memdesc_shape):
        if not fail_on_oob:
            return None
    return linear


def _check_coords_in_bounds(coords, shape, op):
    for dim, coord in enumerate(coords):
        if int(coord) < 0 or int(coord) >= int(shape[dim]):
            fail(
                "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
                STAGE,
                f"fragment load coordinate {tuple(coords)} exceeds memdesc shape {shape}",
                source_op_index=op.index,
            )


def _static_shared_byte_offset(
    layout,
    shape,
    coords,
    element_byte_width,
    op,
    *,
    diagnostic="TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
):
    record = layouts.shared_physical_offset(
        layout,
        shape,
        coords,
        int(element_byte_width),
        stage=STAGE,
        diagnostic=diagnostic,
        source_op_index=None if op is None else op.index,
        source_value_id=None if layout is None else layout.value_id,
    )
    return int(record.byte_offset)


def _static_shared_byte_offset_from_linear(
    layout,
    shape,
    linear,
    element_byte_width,
    op,
    *,
    diagnostic="TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
):
    record = layouts.shared_physical_offset_from_linear(
        layout,
        shape,
        linear,
        int(element_byte_width),
        stage=STAGE,
        diagnostic=diagnostic,
        source_op_index=None if op is None else op.index,
        source_value_id=None if layout is None else layout.value_id,
    )
    return None if record is None else int(record.byte_offset)


def _static_memdesc_view_byte_offset_from_linear(
    layout,
    logical_shape,
    physical_shape,
    logical_origin,
    linear,
    element_byte_width,
    op,
):
    logical_shape = tuple(int(dim) for dim in logical_shape)
    physical_shape = tuple(int(dim) for dim in physical_shape)
    logical_origin = tuple(int(value) for value in logical_origin)
    if (len(logical_shape) != len(physical_shape)
            or len(logical_origin) != len(logical_shape)):
        fail(
            "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            "memdesc view ranks do not match for physical offset materialization",
            source_op_index=op.index,
        )
    if int(linear) < 0 or int(linear) >= _product(logical_shape):
        return None
    logical_coords = _static_delinearize_row_major(
        int(linear),
        logical_shape,
        op,
    )
    physical_coords = tuple(
        int(origin) + int(coord)
        for origin, coord in zip(logical_origin, logical_coords)
    )
    if any(
            int(coord) < 0 or int(coord) >= int(extent)
            for coord, extent in zip(physical_coords, physical_shape)):
        return None
    return _static_shared_byte_offset(
        layout,
        physical_shape,
        physical_coords,
        int(element_byte_width),
        op,
    )


def _static_linear_offset(shape, coords):
    return layouts.static_linear_offset(shape, coords)


def _default_physical_order(shape):
    return layouts.default_physical_order(shape)


def _expand_physical_order(order, rank, layout, op, diagnostic):
    return layouts.expand_physical_order(
        order,
        rank,
        layout=layout,
        stage=STAGE,
        diagnostic=diagnostic,
        source_op_index=None if op is None else op.index,
        source_value_id=None if layout is None else layout.value_id,
    )


def _shared_layout_physical_order(
    layout,
    shape,
    op,
    *,
    diagnostic="TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
):
    return layouts.shared_layout_physical_order(
        layout,
        shape,
        stage=STAGE,
        diagnostic=diagnostic,
        source_op_index=None if op is None else op.index,
        source_value_id=None if layout is None else layout.value_id,
    )


def _ordered_linear_offset(shape, coords, order):
    return layouts.ordered_linear_offset(shape, coords, order)


def _ordered_coords_from_linear(linear, shape, order):
    return layouts.ordered_coords_from_linear(linear, shape, order)


def _static_swizzled_byte_offset(layout, shape, coords, element_byte_width, op):
    return _static_shared_byte_offset(
        layout,
        shape,
        coords,
        element_byte_width,
        op,
    )


def _static_padded_byte_offset(layout, shape, coords, element_byte_width, op):
    return _static_shared_byte_offset(
        layout,
        shape,
        coords,
        element_byte_width,
        op,
    )


def _swizzled_shared_parameters(layout, shape, op):
    return layouts.swizzled_shared_parameters(
        layout,
        shape,
        stage=STAGE,
        diagnostic="TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
        source_op_index=None if op is None else op.index,
        source_value_id=layout.value_id,
    )


def _padded_shared_parameters(layout, op):
    return layouts.padded_shared_parameters(
        layout,
        stage=STAGE,
        diagnostic="TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
        source_op_index=None if op is None else op.index,
        source_value_id=layout.value_id,
    )


def _swizzled_shared_description(layout):
    return layouts.swizzled_shared_description(layout)


def _padded_shared_description(layout):
    return layouts.padded_shared_description(layout)


def _static_delinearize_row_major(linear, shape, op):
    return layouts.static_delinearize_row_major(
        linear,
        shape,
        stage=STAGE,
        diagnostic="TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
        source_op_index=None if op is None else op.index,
    )


def _fragment_component_dword_offsets(
    conversion_input,
    type_layout_program,
    memdesc_value_id,
    result_layout,
    component_count,
    registers,
    op,
):
    memdesc = _memdesc_info(conversion_input, memdesc_value_id, op)
    view = _memdesc_view_info(conversion_input, memdesc_value_id, op)
    if memdesc.element_byte_width is None:
        fail(
            "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            "ttg.local_load requires a known memdesc element width",
            source_op_index=op.index,
            source_value_id=memdesc_value_id,
        )
    layout_id = type_layout_program.values[memdesc_value_id].layout_map_id
    layout = (None if layout_id is None else type_layout_program.layouts[int(layout_id)])
    shape = tuple(int(dim) for dim in memdesc.shape)
    op_idx = int(result_layout.properties["op_idx"])
    parent = result_layout.properties.get("parent_properties", {})
    instr_shape = tuple(parent.get("instr_shape", ()))
    warps_per_cta = tuple(parent.get("warps_per_cta", ()))
    tiles_per_warp = tuple(parent.get("tiles_per_warp", (1, 1)))
    if len(tiles_per_warp) < 2:
        tiles_per_warp = (1, 1)
    k_tile_extent = layouts.dot_operand_storage_k_tile_extent(
        result_layout.element_type,
        result_layout.properties,
    )
    if len(instr_shape) < 3 or len(warps_per_cta) < 2:
        fail(
            "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            "dot-operand local_load requires MFMA instrShape and warpsPerCTA",
            source_op_index=op.index,
            source_value_id=result_layout.value_id,
        )
    offsets = []
    linear_offsets = []
    wave_tile_axis = "none"
    wave_tile_stride_elements = 0
    for component in range(int(component_count)):
        if op_idx == 0:
            if len(shape) < 2:
                fail(
                    "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
                    STAGE,
                    "A-fragment local_load requires rank-2 memdesc shape",
                    source_op_index=op.index,
                    source_value_id=memdesc_value_id,
                )
            k_tiles = _ceil_div(shape[1], k_tile_extent)
            if component_count % k_tiles:
                fail(
                    "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
                    STAGE,
                    "A-fragment component count is not divisible by K tiles",
                    source_op_index=op.index,
                    source_value_id=result_layout.value_id,
                )
            m_tile = component // k_tiles
            k_tile = component % k_tiles
            warps_m = max(1, int(warps_per_cta[0]))
            tiles_m = max(1, int(tiles_per_warp[0]))
            cta_m_tile = layouts.mfma_cta_tile_coordinate(
                m_tile, 0, warps_m, tiles_m)
            linear = (cta_m_tile * instr_shape[0] * shape[1] + k_tile * k_tile_extent)
            wave_tile_axis = "m"
            wave_tile_stride_elements = tiles_m * instr_shape[0] * shape[1]
        elif op_idx == 1:
            if len(shape) < 2:
                fail(
                    "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
                    STAGE,
                    "B-fragment local_load requires rank-2 memdesc shape",
                    source_op_index=op.index,
                    source_value_id=memdesc_value_id,
                )
            k_tiles = _ceil_div(shape[0], k_tile_extent)
            if component_count % k_tiles:
                fail(
                    "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
                    STAGE,
                    "B-fragment component count is not divisible by K tiles",
                    source_op_index=op.index,
                    source_value_id=result_layout.value_id,
                )
            n_tile = component // k_tiles
            k_tile = component % k_tiles
            warps_n = max(1, int(warps_per_cta[1]))
            tiles_n = max(1, int(tiles_per_warp[1]))
            cta_n_tile = layouts.mfma_cta_tile_coordinate(
                n_tile, 0, warps_n, tiles_n)
            linear = (k_tile * k_tile_extent * shape[1] + cta_n_tile * instr_shape[1])
            wave_tile_axis = "n"
            wave_tile_stride_elements = tiles_n * instr_shape[1]
        else:
            fail(
                "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
                STAGE,
                f"unsupported dot operand index {op_idx}",
                source_op_index=op.index,
                source_value_id=result_layout.value_id,
            )
        linear_offsets.append(int(linear))
        byte_offset = _static_memdesc_view_byte_offset_from_linear(
            layout,
            shape,
            view.physical_shape,
            view.logical_origin,
            linear,
            int(memdesc.element_byte_width),
            op,
        )
        if byte_offset is None:
            fail(
                "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
                STAGE,
                "fragment local_load base lies outside its memdesc view",
                source_op_index=op.index,
                source_value_id=memdesc_value_id,
            )
        if byte_offset % 4:
            fail(
                "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
                STAGE,
                "fragment local_load base must be dword aligned",
                source_op_index=op.index,
                source_value_id=memdesc_value_id,
            )
        offsets.append(byte_offset // 4)
    wave_tile_stride_dwords = _element_stride_to_dwords(
        layout,
        shape,
        view.physical_shape,
        view.logical_origin,
        wave_tile_stride_elements,
        int(memdesc.element_byte_width),
        tuple(linear_offsets),
        op,
    )
    return {
        "component_dword_offsets": tuple(offsets),
        "warps_per_cta": tuple(int(value) for value in warps_per_cta),
        "wave_tile_axis": wave_tile_axis,
        "wave_tile_stride_dwords": int(wave_tile_stride_dwords),
    }


def _physical_element_offset(layout, shape, linear, op):
    return _physical_component_offset(layout, shape, int(linear), 1, op)


def _element_stride_to_dwords(
    layout,
    logical_shape,
    physical_shape,
    logical_origin,
    element_stride,
    element_byte_width,
    linear_offsets,
    op,
):
    if int(element_stride) == 0:
        return 0
    byte_strides = set()
    for linear in linear_offsets:
        base = _static_memdesc_view_byte_offset_from_linear(
            layout,
            logical_shape,
            physical_shape,
            logical_origin,
            int(linear),
            int(element_byte_width),
            op,
        )
        shifted = _static_memdesc_view_byte_offset_from_linear(
            layout,
            logical_shape,
            physical_shape,
            logical_origin,
            int(linear) + int(element_stride),
            int(element_byte_width),
            op,
        )
        if base is not None and shifted is not None:
            byte_strides.add(int(shifted) - int(base))
    if not byte_strides:
        return 0
    if len(byte_strides) != 1:
        fail(
            "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            "shared MFMA local_load needs per-component wave stride remaps",
            source_op_index=op.index,
            source_value_id=None if layout is None else layout.value_id,
        )
    byte_stride = next(iter(byte_strides))
    if byte_stride % 4:
        fail(
            "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            "fragment local_load wave tile stride must be dword aligned",
            source_op_index=op.index,
            source_value_id=layout.value_id if layout is not None else None,
        )
    return byte_stride // 4


def _physical_component_offset(layout, shape, linear_start, lane_width, op):
    linear_end = int(linear_start) + int(lane_width) - 1
    if layout is None or layout.kind in {"none", "swizzled_shared"}:
        if layout is not None and layout.kind == "swizzled_shared":
            _require_identity_swizzled(layout, op)
        return int(linear_start)
    if layout.kind == "padded_shared":
        intervals = layout.properties.get("intervals", ())
        paddings = layout.properties.get("paddings", ())
        if len(intervals) != 1 or len(paddings) != 1:
            fail(
                "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
                STAGE,
                "scalarized amdg.buffer_load_to_local supports one "
                "padded interval",
                source_op_index=op.index,
                source_value_id=layout.value_id,
            )
        interval = int(intervals[0])
        padding = int(paddings[0])
        if interval <= 0:
            fail(
                "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
                STAGE,
                "padded shared interval must be positive",
                source_op_index=op.index,
                source_value_id=layout.value_id,
            )
        physical_start = _static_shared_byte_offset_from_linear(
            layout,
            shape,
            int(linear_start),
            1,
            op,
        )
        physical_end = _static_shared_byte_offset_from_linear(
            layout,
            shape,
            int(linear_end),
            1,
            op,
        )
        if physical_start is None or physical_end is None:
            fail(
                "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
                STAGE,
                "scalarized amdg.buffer_load_to_local component exceeds "
                "local memdesc shape",
                source_op_index=op.index,
                source_value_id=layout.value_id,
            )
        unpadded_start = _ordered_linear_offset(
            shape,
            _static_delinearize_row_major(int(linear_start), shape, op),
            _shared_layout_physical_order(layout, shape, op),
        )
        unpadded_end = _ordered_linear_offset(
            shape,
            _static_delinearize_row_major(int(linear_end), shape, op),
            _shared_layout_physical_order(layout, shape, op),
        )
        if unpadded_start // interval != unpadded_end // interval:
            fail(
                "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
                STAGE,
                "scalarized amdg.buffer_load_to_local component crosses "
                "a padded LDS interval",
                source_op_index=op.index,
                source_value_id=layout.value_id,
            )
        return int(physical_start)
    fail(
        "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
        STAGE,
        f"amdg.buffer_load_to_local destination layout {layout.kind} "
        "is not converted yet",
        source_op_index=op.index,
        source_value_id=layout.value_id,
    )


def _require_identity_swizzled(layout, op):
    props = layout.properties
    if (int(props.get("vec", 0)) == 1 and int(props.get("per_phase", 0)) == 1 and int(props.get("max_phase", 0)) == 1):
        return
    fail(
        "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
        STAGE,
        "amdg.buffer_load_to_local swizzled destination requires an "
        "explicit remap target op",
        source_op_index=op.index,
        source_value_id=layout.value_id,
    )


def _constant_literal(value, *, source_op_index=None, element_type=None):
    if value is None:
        return None
    text = str(value).strip()
    dense = re.fullmatch(r"dense<([^>]*)>\s*(?::.*)?", text, re.DOTALL)
    if dense is not None:
        payload = dense.group(1).strip()
        if payload.startswith("[") or "," in payload:
            fail(
                "TLXW_OP_UNSUPPORTED_CONSTANT",
                STAGE,
                "tensor constants are converted only for dense splats",
                source_op_index=source_op_index,
            )
        text = payload
    else:
        text = text.split(":", 1)[0].strip()
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    match = re.fullmatch(r"([+-]?0[xX][0-9a-fA-F]+)", text)
    if match is not None and _is_float_element_type(element_type):
        return _float_bit_pattern_literal(match.group(1), element_type, source_op_index)
    match = re.fullmatch(r"([+-]?\d+)", text)
    if match is not None:
        return int(match.group(1), 0)
    match = re.fullmatch(
        r"([+-]?(?:(?:\d+\.\d*)|(?:\.\d+)|(?:\d+))(?:[eE][+-]?\d+)?)",
        text,
    )
    if match is not None:
        return float(match.group(1))
    return text


def _is_float_element_type(element_type):
    return element_type in {"f16", "bf16", "f32", "f64"}


def _float_bit_pattern_literal(text, element_type, source_op_index):
    widths = {"f16": 16, "bf16": 16, "f32": 32, "f64": 64}
    width = widths[element_type]
    bits = int(text, 0)
    if bits < 0 or bits >= 1 << width:
        fail(
            "TLXW_OP_UNSUPPORTED_CONSTANT",
            STAGE,
            f"floating-point bit pattern {text!r} does not fit {element_type}",
            source_op_index=source_op_index,
        )
    if element_type == "f16":
        return struct.unpack("<e", struct.pack("<H", bits))[0]
    if element_type == "bf16":
        return struct.unpack("<f", struct.pack("<I", bits << 16))[0]
    if element_type == "f32":
        return struct.unpack("<f", struct.pack("<I", bits))[0]
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def _cmpi_predicate(value):
    predicates = {
        0: "eq",
        1: "ne",
        2: "slt",
        3: "sle",
        4: "sgt",
        5: "sge",
        6: "ult",
        7: "ule",
        8: "ugt",
        9: "uge",
    }
    if value is None:
        return "unknown"
    if isinstance(value, str) and value in predicates.values():
        return value
    return predicates.get(int(value), str(value))


def _target_int_width(builder, target_value_ids):
    for target_value_id in target_value_ids:
        target_type = builder.values[target_value_id].type
        width = _int_width(target_type.element_type)
        if width is not None:
            return width
    return None


def _int_width(raw_type):
    match = re.fullmatch(r"i([0-9]+)", str(raw_type))
    return None if match is None else int(match.group(1))


def _product(values):
    result = 1
    for value in values:
        result *= int(value)
    return result


def _ceil_div(lhs, rhs):
    return (int(lhs) + int(rhs) - 1) // int(rhs)


def _int_attr(attrs, name):
    value = attrs.get(name)
    if value is None:
        fail(
            "TLXW_OP_MISSING_ATTR",
            STAGE,
            f"required attr {name} is missing",
        )
    return int(value)


def _int_attr_or_default(attrs, name, default):
    value = attrs.get(name)
    return int(default) if value is None else int(value)
