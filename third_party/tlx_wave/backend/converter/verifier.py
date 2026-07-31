"""Target-program verifier for the TLX Wave converter."""

from . import domains
from . import target_ir
from .diagnostics import fail

STAGE = "verification"

_PROOF_DEPENDENT_OPS = frozenset({"assume", "buffer_load_to_local", "buffer_load", "buffer_store"})
_ASSUMPTION_KINDS = frozenset({"divisible", "pointer_byte_range", "range"})
_TARGET_REPRESENTATIONS = frozenset({
    "buffer_pointer",
    "buffer_pointer_tuple",
    "uniform_buffer_pointer",
    "mask",
    "mask_tuple",
    "memdesc",
    "per_lane_pointer",
    "pointer_tuple",
    "scalar",
    "simd",
    "simd_packet",
    "simd_packet_tuple",
    "simd_tuple",
    "token",
    "uniform_pointer",
})


def verify_target_program(
    target_program,
    *,
    source_program=None,
    token_program=None,
):
    _verify_target_contract(target_program)
    _verify_target_layouts(target_program)
    _verify_target_assumptions(target_program)
    _verify_target_value_ids(target_program)
    _verify_target_value_types(target_program)
    _verify_ops(target_program, source_program)
    if source_program is not None:
        _verify_source_results_covered(source_program, target_program)
    if token_program is not None and source_program is not None:
        _verify_memory_effects_tokenized(source_program, token_program)
    return True


def _verify_target_contract(target_program):
    contract = target_program.contract
    if int(contract.schema_version) != target_ir.TARGET_SCHEMA_VERSION:
        fail(
            "TLXW_VERIFY_SCHEMA_VERSION",
            STAGE,
            f"unsupported target schema version {contract.schema_version}",
        )
    if contract.address_arithmetic != target_ir.ADDRESS_ARITHMETIC_NO_OVERFLOW:
        fail(
            "TLXW_VERIFY_ADDRESS_ARITHMETIC",
            STAGE,
            "target contract must state that address arithmetic does not overflow",
        )
    if type(contract.enable_fp_fusion) is not bool:
        fail(
            "TLXW_VERIFY_FP_FUSION",
            STAGE,
            "target FP-fusion contract must be boolean",
        )


def _verify_target_layouts(target_program):
    for expected_id, layout in enumerate(target_program.layouts):
        if layout.layout_map_id != expected_id:
            fail(
                "TLXW_VERIFY_LAYOUT_ID",
                STAGE,
                f"target layout id {layout.layout_map_id} does not match "
                f"position {expected_id}",
            )
        if not layout.kind or any(int(dim) <= 0 for dim in layout.shape):
            fail(
                "TLXW_VERIFY_LAYOUT_SCHEMA",
                STAGE,
                f"target layout {layout.layout_map_id} has an incomplete shape or kind",
            )
        if int(layout.component_count) <= 0 or int(layout.lane_width) <= 0:
            fail(
                "TLXW_VERIFY_LAYOUT_SCHEMA",
                STAGE,
                f"target layout {layout.layout_map_id} has an invalid physical domain",
            )
        property_names = set()
        for prop in layout.properties:
            if (not isinstance(prop, target_ir.TargetAttr) or not prop.name or prop.name in property_names
                    or not _is_layout_schema_value(prop.value)):
                fail(
                    "TLXW_VERIFY_LAYOUT_SCHEMA",
                    STAGE,
                    f"target layout {layout.layout_map_id} has malformed "
                    "symbolic properties",
                )
            property_names.add(prop.name)
        if layout.linear_layout is not None and not _is_layout_schema_value(layout.linear_layout):
            fail(
                "TLXW_VERIFY_LAYOUT_SCHEMA",
                STAGE,
                f"target layout {layout.layout_map_id} has a malformed "
                "symbolic linear layout",
            )


def _verify_target_assumptions(target_program):
    value_count = len(target_program.values)
    for expected_id, assumption in enumerate(target_program.assumptions):
        if assumption.assumption_id != expected_id:
            fail(
                "TLXW_VERIFY_ASSUMPTION_ID",
                STAGE,
                f"target assumption id {assumption.assumption_id} does not "
                f"match position {expected_id}",
                fact_id=assumption.assumption_id,
            )
        if assumption.kind not in _ASSUMPTION_KINDS:
            fail(
                "TLXW_VERIFY_ASSUMPTION_KIND",
                STAGE,
                f"unsupported target assumption kind {assumption.kind}",
                fact_id=assumption.assumption_id,
            )
        if not assumption.predicate or not assumption.subject_target_ids:
            fail(
                "TLXW_VERIFY_ASSUMPTION_SCHEMA",
                STAGE,
                f"target assumption {assumption.assumption_id} is not bound "
                "to a predicate and target value",
                fact_id=assumption.assumption_id,
            )
        if len(set(assumption.subject_target_ids)) != len(assumption.subject_target_ids):
            fail(
                "TLXW_VERIFY_ASSUMPTION_SCHEMA",
                STAGE,
                f"target assumption {assumption.assumption_id} has duplicate "
                "target subjects",
                fact_id=assumption.assumption_id,
            )
        if assumption.kind == "divisible":
            if assumption.divisor is None or int(assumption.divisor) <= 1:
                fail(
                    "TLXW_VERIFY_ASSUMPTION_SCHEMA",
                    STAGE,
                    f"target assumption {assumption.assumption_id} has an "
                    "invalid divisor",
                    fact_id=assumption.assumption_id,
                )
        elif assumption.lower is None and assumption.upper is None:
            fail(
                "TLXW_VERIFY_ASSUMPTION_SCHEMA",
                STAGE,
                f"target assumption {assumption.assumption_id} has no bounds",
                fact_id=assumption.assumption_id,
            )
        if (assumption.lower is not None and assumption.upper is not None
                and int(assumption.lower) > int(assumption.upper)):
            fail(
                "TLXW_VERIFY_ASSUMPTION_SCHEMA",
                STAGE,
                f"target assumption {assumption.assumption_id} has an empty "
                "range",
                fact_id=assumption.assumption_id,
            )
        if assumption.width is not None and int(assumption.width) <= 0:
            fail(
                "TLXW_VERIFY_ASSUMPTION_SCHEMA",
                STAGE,
                f"target assumption {assumption.assumption_id} has an invalid "
                "bit width",
                fact_id=assumption.assumption_id,
            )
        for target_value_id in assumption.subject_target_ids:
            if target_value_id < 0 or target_value_id >= value_count:
                fail(
                    "TLXW_VERIFY_ASSUMPTION_TARGET",
                    STAGE,
                    f"target assumption {assumption.assumption_id} references "
                    f"missing value {target_value_id}",
                    target_value_id=target_value_id,
                    fact_id=assumption.assumption_id,
                )


def _verify_target_value_ids(target_program):
    for expected_id, value in enumerate(target_program.values):
        if value.target_value_id != expected_id:
            fail(
                "TLXW_VERIFY_VALUE_ID",
                STAGE,
                f"target value id {value.target_value_id} does not match "
                f"position {expected_id}",
                target_value_id=value.target_value_id,
            )


def _verify_target_value_types(target_program):
    layouts = {layout.layout_map_id: layout for layout in target_program.layouts}
    for value in target_program.values:
        representation = str(value.type.representation)
        resources = tuple(int(target_id) for target_id in value.resource_target_ids)
        if len(resources) != len(set(resources)):
            fail(
                "TLXW_VERIFY_RESOURCE_TARGETS",
                STAGE,
                "target value resource identities must be unique",
                target_value_id=value.target_value_id,
            )
        if resources and representation not in {"memdesc", "token"}:
            fail(
                "TLXW_VERIFY_RESOURCE_REPRESENTATION",
                STAGE,
                "only memdesc and token values may carry resource identities",
                target_value_id=value.target_value_id,
            )
        for resource_target_id in resources:
            if not 0 <= resource_target_id < len(target_program.values):
                fail(
                    "TLXW_VERIFY_RESOURCE_TARGET",
                    STAGE,
                    f"target value references missing resource {resource_target_id}",
                    target_value_id=value.target_value_id,
                )
            if target_program.values[resource_target_id].type.representation != "memdesc":
                fail(
                    "TLXW_VERIFY_RESOURCE_TARGET",
                    STAGE,
                    "resource identity must name a memdesc value",
                    target_value_id=value.target_value_id,
                )
        if value.event_domain is not None:
            if value.event_domain not in target_ir.EVENT_DOMAINS:
                fail(
                    "TLXW_VERIFY_EVENT_DOMAIN",
                    STAGE,
                    f"unknown target event domain {value.event_domain}",
                    target_value_id=value.target_value_id,
                )
            if representation != "token":
                fail(
                    "TLXW_VERIFY_EVENT_REPRESENTATION",
                    STAGE,
                    "target event domains require token representation",
                    target_value_id=value.target_value_id,
                )
        if representation in {"fragment", "fragment_tuple"}:
            fail(
                "TLXW_VERIFY_FRAGMENT_BOUNDARY",
                STAGE,
                "WaveAMD fragment representations are MMA-lowering details "
                "and cannot appear in target-program values",
                target_value_id=value.target_value_id,
            )
        if representation not in _TARGET_REPRESENTATIONS:
            fail(
                "TLXW_VERIFY_TARGET_REPRESENTATION",
                STAGE,
                f"unsupported target representation {representation}",
                target_value_id=value.target_value_id,
            )
        if int(value.type.component_count) <= 0:
            fail(
                "TLXW_VERIFY_TARGET_COMPONENT_COUNT",
                STAGE,
                "target values require a positive component count",
                target_value_id=value.target_value_id,
            )
        if value.layout_map_id is not None:
            layout = layouts.get(int(value.layout_map_id))
            if layout is None:
                fail(
                    "TLXW_VERIFY_UNKNOWN_LAYOUT",
                    STAGE,
                    f"target value {value.target_value_id} references missing "
                    f"layout {value.layout_map_id}",
                    target_value_id=value.target_value_id,
                )
            if (value.type.lane_width is not None and int(value.type.lane_width) != int(layout.lane_width)):
                fail(
                    "TLXW_VERIFY_LAYOUT_TYPE",
                    STAGE,
                    f"target value {value.target_value_id} lane width does not "
                    f"match layout {value.layout_map_id}",
                    target_value_id=value.target_value_id,
                )


def _verify_ops(target_program, source_program):
    value_count = len(target_program.values)
    op_count = len(target_program.ops)
    facts_by_id = {assumption.assumption_id: assumption for assumption in target_program.assumptions}
    layout_ids = frozenset(layout.layout_map_id for layout in target_program.layouts)
    for expected_id, op in enumerate(target_program.ops):
        if op.target_op_id != expected_id:
            fail(
                "TLXW_VERIFY_OP_ID",
                STAGE,
                f"target op id {op.target_op_id} does not match position {expected_id}",
                target_op_id=op.target_op_id,
            )
        if op.kind not in domains.all_target_ops():
            fail(
                "TLXW_VERIFY_UNKNOWN_TARGET_OP",
                STAGE,
                f"target op {op.target_op_id} has unknown kind {op.kind}",
                target_op_id=op.target_op_id,
            )
        for target_value_id in (*op.operands, *op.results):
            if target_value_id < 0 or target_value_id >= value_count:
                fail(
                    "TLXW_VERIFY_UNKNOWN_TARGET_VALUE",
                    STAGE,
                    f"target op {op.target_op_id} references missing "
                    f"value {target_value_id}",
                    target_op_id=op.target_op_id,
                    target_value_id=target_value_id,
                )
        for layout_map_id in op.layout_map_ids:
            if layout_map_id not in layout_ids:
                fail(
                    "TLXW_VERIFY_UNKNOWN_LAYOUT",
                    STAGE,
                    f"target op {op.target_op_id} references missing layout "
                    f"{layout_map_id}",
                    target_op_id=op.target_op_id,
                )
        _verify_attrs(op)
        if op.kind in {"buffer_load_to_local", "buffer_load", "buffer_store"}:
            _verify_memory_edges(op, target_program)
        if op.kind in target_ir.MEMORY_ISSUER_OP_KINDS:
            _verify_memory_issue_order(op, target_program)
        if op.kind in {
                "token",
                "issue_token",
                "barrier",
                "buffer_load_to_local",
                "async_commit_group",
                "async_wait",
                "local_load",
                "local_store",
        }:
            _verify_async_protocol_op(op, target_program, source_program)
        if op.kind == "type_convert":
            _verify_type_convert(op, target_program)
        if op.kind == "make_buffer":
            _verify_make_buffer(op, target_program)
        if op.kind == "layout_convert":
            _verify_layout_convert_fact_policy(op)
        if op.kind in {
                "broadcast",
                "expand_dims",
                "join",
                "layout_convert",
                "split",
        }:
            _verify_layout_transform(op, target_program)
        if op.kind == "reduction":
            _verify_reduction(op, target_program)
        if len(op.fact_target_ids) != len(op.fact_ids):
            fail(
                "TLXW_VERIFY_FACT_TARGET_COUNT",
                STAGE,
                f"target op {op.target_op_id} has {len(op.fact_ids)} facts "
                f"but {len(op.fact_target_ids)} fact targets",
                target_op_id=op.target_op_id,
            )
        for target_value_id in op.fact_target_ids:
            if target_value_id < 0 or target_value_id >= value_count:
                fail(
                    "TLXW_VERIFY_UNKNOWN_FACT_TARGET",
                    STAGE,
                    f"target op {op.target_op_id} references missing fact "
                    f"target value {target_value_id}",
                    target_op_id=op.target_op_id,
                    target_value_id=target_value_id,
                )
        for fact_id in op.fact_ids:
            if fact_id not in facts_by_id:
                fail(
                    "TLXW_VERIFY_UNKNOWN_FACT",
                    STAGE,
                    f"target op {op.target_op_id} references missing fact {fact_id}",
                    target_op_id=op.target_op_id,
                    fact_id=fact_id,
                )
        for fact_id, target_value_id in zip(op.fact_ids, op.fact_target_ids):
            _verify_fact_target_compatible(
                target_program,
                op,
                facts_by_id[fact_id],
                target_value_id,
            )
        if op.kind == "layout_convert":
            if source_program is not None:
                _verify_layout_convert_source_op(op, source_program)
        if op.kind in _PROOF_DEPENDENT_OPS and not op.fact_ids:
            fail(
                "TLXW_VERIFY_MISSING_FACT",
                STAGE,
                f"target op {op.target_op_id} ({op.kind}) requires fact provenance",
                target_op_id=op.target_op_id,
            )
    _verify_region_op_ids(target_program, op_count)


def _verify_memory_issue_order(op, target_program):
    attrs = _attrs_dict(op)
    dependency_count = int(attrs.get("barrier_order_dependency_count", 0))
    if dependency_count not in {0, 1} or dependency_count > len(op.operands):
        fail(
            "TLXW_VERIFY_BARRIER_ORDER_SEGMENT",
            STAGE,
            "memory operation has a malformed barrier-order operand segment",
            target_op_id=op.target_op_id,
        )
    barrier_issue_operands = tuple(int(operand)
                                   for operand in op.operands[-dependency_count:]) if dependency_count else ()
    for operand in barrier_issue_operands:
        value = target_program.values[operand]
        if (value.type.representation != "token" or value.event_domain != target_ir.EVENT_DOMAIN_BARRIER_ISSUE):
            fail(
                "TLXW_VERIFY_BARRIER_ORDER_DOMAIN",
                STAGE,
                "memory barrier-order dependency must be a completion-free "
                "barrier-issue token",
                target_op_id=op.target_op_id,
                target_value_id=operand,
            )
        producer = _target_value_producer(target_program, operand)
        _require_precedes_in_same_region(target_program, producer, op)

    ordinary_operands = (op.operands[:-dependency_count] if dependency_count else op.operands)
    hidden_barrier_operands = tuple(
        int(operand)
        for operand in ordinary_operands
        if target_program.values[int(operand)].event_domain == target_ir.EVENT_DOMAIN_BARRIER_ISSUE)
    if hidden_barrier_operands:
        fail(
            "TLXW_VERIFY_BARRIER_ORDER_SEGMENT",
            STAGE,
            "barrier-issue tokens must use the dedicated final operand segment",
            target_op_id=op.target_op_id,
            target_value_id=hidden_barrier_operands[0],
        )

    result_count = int(attrs.get("issue_order_result_count", 0))
    if result_count not in {0, 1} or result_count > len(op.results):
        fail(
            "TLXW_VERIFY_BARRIER_ORDER_RESULT",
            STAGE,
            "memory operation has a malformed issue-order result segment",
            target_op_id=op.target_op_id,
        )
    if op.kind == "buffer_load_to_local" and result_count:
        fail(
            "TLXW_VERIFY_BARRIER_ORDER_RESULT",
            STAGE,
            "direct DMA must reuse its explicit completion token for issue projection",
            target_op_id=op.target_op_id,
        )
    if not result_count:
        return
    result_id = int(op.results[-1])
    result = target_program.values[result_id]
    if (result.type.representation != "token" or result.event_domain != target_ir.EVENT_DOMAIN_MEMORY_COMPLETION):
        fail(
            "TLXW_VERIFY_BARRIER_ORDER_RESULT",
            STAGE,
            "synthetic memory issue-order result must carry raw memory completion",
            target_op_id=op.target_op_id,
            target_value_id=result_id,
        )
    if op.kind in {"local_load", "local_store"}:
        completion_count = int(attrs.get("completion_result_count", 0))
        if completion_count < result_count:
            fail(
                "TLXW_VERIFY_BARRIER_ORDER_RESULT",
                STAGE,
                "local issue-order result must belong to the completion segment",
                target_op_id=op.target_op_id,
            )


def _verify_issue_projection_structure(op, target_program, projection_domain):
    if projection_domain == target_ir.EVENT_DOMAIN_DMA_ISSUE:
        return
    if projection_domain == target_ir.EVENT_DOMAIN_MEMORY_ISSUE:
        for operand in op.operands:
            producer = _target_value_producer(target_program, operand)
            if producer.kind not in target_ir.MEMORY_ISSUER_OP_KINDS:
                fail(
                    "TLXW_VERIFY_BARRIER_ORDER_PROVENANCE",
                    STAGE,
                    "pre-barrier issue projection must come from memory issuers",
                    target_op_id=op.target_op_id,
                    target_value_id=int(operand),
                )
            _require_precedes_in_same_region(target_program, producer, op)
        return
    if projection_domain == target_ir.EVENT_DOMAIN_BARRIER_ISSUE:
        if len(op.operands) != 1:
            fail(
                "TLXW_VERIFY_BARRIER_ORDER_PROVENANCE",
                STAGE,
                "post-barrier issue projection requires one ordering-barrier result",
                target_op_id=op.target_op_id,
            )
        producer = _target_value_producer(target_program, op.operands[0])
        attrs = _attrs_dict(producer)
        if (producer.kind != "barrier" or not bool(attrs.get("orders_memory_issue", False))):
            fail(
                "TLXW_VERIFY_BARRIER_ORDER_PROVENANCE",
                STAGE,
                "post-barrier issue projection must come from an explicit "
                "memory-issue-ordering barrier",
                target_op_id=op.target_op_id,
                target_value_id=int(op.operands[0]),
            )
        _require_precedes_in_same_region(target_program, producer, op)


def _target_value_producer(target_program, target_value_id):
    producers = tuple(op for op in target_program.ops if int(target_value_id) in op.results)
    if len(producers) != 1:
        fail(
            "TLXW_VERIFY_BARRIER_ORDER_PROVENANCE",
            STAGE,
            "barrier-order token must have exactly one target producer",
            target_value_id=int(target_value_id),
        )
    return producers[0]


def _require_precedes_in_same_region(target_program, producer, consumer):
    producer_position = _target_op_region_position(
        target_program,
        producer.target_op_id,
    )
    consumer_position = _target_op_region_position(
        target_program,
        consumer.target_op_id,
    )
    if (producer_position[0] != consumer_position[0] or producer_position[1] >= consumer_position[1]):
        fail(
            "TLXW_VERIFY_BARRIER_ORDER_DOMINANCE",
            STAGE,
            "barrier-order dependency must be produced earlier in the same region",
            target_op_id=consumer.target_op_id,
        )


def _target_op_region_position(target_program, target_op_id):
    positions = tuple((int(region.target_region_id), position)
                      for region in target_program.regions
                      for position, op_id in enumerate(region.op_ids)
                      if int(op_id) == int(target_op_id))
    if len(positions) != 1:
        fail(
            "TLXW_VERIFY_BARRIER_ORDER_DOMINANCE",
            STAGE,
            "barrier-order operation must belong to exactly one target region",
            target_op_id=int(target_op_id),
        )
    return positions[0]


def _require_value_dominates_op(target_program, target_value_id, consumer):
    """Verify ordinary structured-SSA availability for a token operand."""
    target_value_id = int(target_value_id)
    producers = tuple(op for op in target_program.ops if target_value_id in op.results)
    block_arg_regions = tuple(region for region in target_program.regions if target_value_id in region.block_arg_ids)
    if len(producers) + len(block_arg_regions) != 1:
        fail(
            "TLXW_VERIFY_BARRIER_ORDER_PROVENANCE",
            STAGE,
            "barrier source-order token must have exactly one target "
            "producer or block argument",
            target_op_id=consumer.target_op_id,
            target_value_id=target_value_id,
        )

    consumer_region_id, _ = _target_op_region_position(
        target_program,
        consumer.target_op_id,
    )
    if block_arg_regions:
        block_arg_region_id = int(block_arg_regions[0].target_region_id)
        if _region_is_ancestor_of(
                target_program,
                block_arg_region_id,
                consumer_region_id,
        ):
            return
        fail(
            "TLXW_VERIFY_BARRIER_ORDER_DOMINANCE",
            STAGE,
            "barrier source-order block argument must belong to the "
            "consumer region or an ancestor",
            target_op_id=consumer.target_op_id,
            target_value_id=target_value_id,
        )

    producer = producers[0]
    producer_region_id, producer_position = _target_op_region_position(
        target_program,
        producer.target_op_id,
    )
    if producer_region_id == consumer_region_id:
        _, consumer_position = _target_op_region_position(
            target_program,
            consumer.target_op_id,
        )
        if producer_position < consumer_position:
            return
    else:
        containing_op = _containing_op_in_ancestor_region(
            target_program,
            consumer_region_id,
            producer_region_id,
        )
        if containing_op is not None:
            _, containing_position = _target_op_region_position(
                target_program,
                containing_op.target_op_id,
            )
            if producer_position < containing_position:
                return
    fail(
        "TLXW_VERIFY_BARRIER_ORDER_DOMINANCE",
        STAGE,
        "barrier source-order token must dominate its consumer",
        target_op_id=consumer.target_op_id,
        target_value_id=target_value_id,
    )


def _region_is_ancestor_of(target_program, ancestor_region_id, region_id):
    ancestor_region_id = int(ancestor_region_id)
    region_id = int(region_id)
    while True:
        if region_id == ancestor_region_id:
            return True
        parent = _target_region_parent(target_program, region_id)
        if parent is None:
            return False
        _, region_id = parent


def _containing_op_in_ancestor_region(
    target_program,
    region_id,
    ancestor_region_id,
):
    region_id = int(region_id)
    ancestor_region_id = int(ancestor_region_id)
    while region_id != ancestor_region_id:
        parent = _target_region_parent(target_program, region_id)
        if parent is None:
            return None
        parent_op, parent_region_id = parent
        if parent_region_id == ancestor_region_id:
            return parent_op
        region_id = parent_region_id
    return None


def _target_region_parent(target_program, region_id):
    parents = tuple((op, parent_region_id) for op in target_program.ops if int(region_id) in op.region_ids
                    for parent_region_id, _ in ((_target_op_region_position(target_program, op.target_op_id), )))
    if len(parents) > 1:
        fail(
            "TLXW_VERIFY_BARRIER_ORDER_DOMINANCE",
            STAGE,
            "target region must have at most one containing operation",
        )
    return None if not parents else parents[0]


def _verify_async_protocol_op(op, target_program, source_program=None):
    attrs = _attrs_dict(op)

    def require_token(value_id, description, allowed_domains=None):
        value = target_program.values[int(value_id)]
        if value.type.representation != "token":
            fail(
                "TLXW_VERIFY_ASYNC_PROTOCOL_TYPE",
                STAGE,
                f"{description} must be a token",
                target_op_id=op.target_op_id,
                target_value_id=int(value_id),
            )
        if (allowed_domains is not None and value.event_domain not in allowed_domains):
            fail(
                "TLXW_VERIFY_ASYNC_PROTOCOL_DOMAIN",
                STAGE,
                f"{description} has event domain {value.event_domain!r}; "
                f"expected one of {sorted(str(domain) for domain in allowed_domains)}",
                target_op_id=op.target_op_id,
                target_value_id=int(value_id),
            )

    if op.kind == "token":
        if op.operands or len(op.results) != 1:
            fail(
                "TLXW_VERIFY_ASYNC_PROTOCOL_SHAPE",
                STAGE,
                "empty token requires no operands and one result",
                target_op_id=op.target_op_id,
            )
        require_token(
            op.results[0],
            "empty token result",
            {target_ir.EVENT_DOMAIN_EMPTY, None},
        )
        return

    if op.kind == "issue_token":
        input_count = int(attrs.get("input_count", -1))
        if (len(op.results) != 1 or input_count <= 0 or input_count != len(op.operands)):
            fail(
                "TLXW_VERIFY_ASYNC_PROTOCOL_SHAPE",
                STAGE,
                f"{op.kind} requires an exact input count and one result",
                target_op_id=op.target_op_id,
            )
        for operand in op.operands:
            require_token(operand, f"{op.kind} operand")
        projection_domain = attrs.get("projection_domain")
        expected_provenance = {
            target_ir.EVENT_DOMAIN_DMA_ISSUE: "partial_wait_retained_group",
            target_ir.EVENT_DOMAIN_MEMORY_ISSUE: "memory_barrier_predecessors",
            target_ir.EVENT_DOMAIN_BARRIER_ISSUE: "memory_barrier_successors",
        }.get(projection_domain)
        if expected_provenance is None:
            fail(
                "TLXW_VERIFY_ASYNC_PROTOCOL_DOMAIN",
                STAGE,
                "issue token requires an explicit supported projection domain",
                target_op_id=op.target_op_id,
            )
        if attrs.get("projection_provenance") != expected_provenance:
            fail(
                "TLXW_VERIFY_ASYNC_PROTOCOL_PROVENANCE",
                STAGE,
                f"{projection_domain} projection requires provenance "
                f"{expected_provenance!r}",
                target_op_id=op.target_op_id,
            )
        allowed_input_domains = {
            target_ir.EVENT_DOMAIN_DMA_ISSUE: {
                target_ir.EVENT_DOMAIN_DMA_COMPLETION,
                target_ir.EVENT_DOMAIN_DMA_GROUP,
                target_ir.EVENT_DOMAIN_EMPTY,
                None,
            },
            target_ir.EVENT_DOMAIN_MEMORY_ISSUE: {
                target_ir.EVENT_DOMAIN_MEMORY_COMPLETION,
                target_ir.EVENT_DOMAIN_DMA_COMPLETION,
            },
            target_ir.EVENT_DOMAIN_BARRIER_ISSUE: {
                target_ir.EVENT_DOMAIN_MEMORY_BARRIER,
            },
        }[projection_domain]
        for operand in op.operands:
            require_token(
                operand,
                f"{projection_domain} projection operand",
                allowed_input_domains,
            )
        _verify_issue_projection_structure(
            op,
            target_program,
            projection_domain,
        )
        require_token(
            op.results[0],
            "issue token result",
            {projection_domain},
        )
        return

    if op.kind == "barrier":
        dependency_count = int(attrs.get("dependency_count", -1))
        lds_read_dependency_count = int(attrs.get("lds_read_dependency_count", 0))
        issue_dependency_count = int(attrs.get("barrier_order_dependency_count", 0))
        if (dependency_count not in {0, 1} or lds_read_dependency_count < 0 or issue_dependency_count not in {0, 1}
                or dependency_count + lds_read_dependency_count + issue_dependency_count != len(op.operands)):
            fail(
                "TLXW_VERIFY_ASYNC_PROTOCOL_SEGMENTS",
                STAGE,
                "barrier source-order, LDS-read, and memory-issue segments "
                "are malformed",
                target_op_id=op.target_op_id,
            )
        readiness_operands = op.operands[:dependency_count]
        lds_read_begin = dependency_count
        lds_read_end = lds_read_begin + lds_read_dependency_count
        lds_read_operands = op.operands[lds_read_begin:lds_read_end]
        issue_operands = op.operands[lds_read_end:]
        orders_memory_issue = bool(attrs.get("orders_memory_issue", False))
        lds_completion_result_count = int(attrs.get("lds_completion_result_count", 0))
        requires_issue_order = (int(attrs.get("address_space", 0)) == 31
                                or bool(attrs.get("compiler_membar_barrier", False)))
        if requires_issue_order and not orders_memory_issue:
            fail(
                "TLXW_VERIFY_BARRIER_ORDER_ASSUMPTION",
                STAGE,
                "full-memory and compiler membar barriers must order memory issue",
                target_op_id=op.target_op_id,
            )
        if len(op.results) > 1:
            fail(
                "TLXW_VERIFY_ASYNC_PROTOCOL_SHAPE",
                STAGE,
                "memory-issue-ordering barrier may expose at most one token",
                target_op_id=op.target_op_id,
            )
        if (lds_completion_result_count not in {0, 1} or lds_completion_result_count > len(op.results)):
            fail(
                "TLXW_VERIFY_ASYNC_PROTOCOL_SEGMENTS",
                STAGE,
                "barrier LDS-completion result segment is malformed",
                target_op_id=op.target_op_id,
            )
        for operand in readiness_operands:
            require_token(
                operand,
                "barrier source wait-order dependency",
                {target_ir.EVENT_DOMAIN_WAVE_LOCAL_READY},
            )
            _require_value_dominates_op(target_program, operand, op)
        for operand in lds_read_operands:
            require_token(
                operand,
                "barrier LDS-read completion dependency",
                {
                    target_ir.EVENT_DOMAIN_MEMORY_COMPLETION,
                    target_ir.EVENT_DOMAIN_MEMORY_BARRIER,
                },
            )
            producers = tuple(producer for producer in target_program.ops if int(operand) in producer.results)
            block_arg_regions = tuple(region for region in target_program.regions
                                      if int(operand) in region.block_arg_ids)
            is_local_read = (len(producers) == 1 and producers[0].kind in {
                "local_load",
            })
            is_structured_completion = (len(producers) == 1 and producers[0].kind in {
                "barrier",
                "for_loop",
                "if",
                "token_join",
            }) or (not producers and len(block_arg_regions) == 1)
            if not is_local_read and not is_structured_completion:
                fail(
                    "TLXW_VERIFY_BARRIER_ORDER_PROVENANCE",
                    STAGE,
                    "barrier LDS-read completion must come from a local load "
                    "or an explicit structured SSA carry",
                    target_op_id=op.target_op_id,
                    target_value_id=int(operand),
                )
            _require_value_dominates_op(
                target_program,
                operand,
                op,
            )
        for operand in issue_operands:
            require_token(
                operand,
                "barrier issue-order dependency",
                {
                    target_ir.EVENT_DOMAIN_MEMORY_ISSUE,
                    target_ir.EVENT_DOMAIN_BARRIER_ISSUE,
                },
            )
            producer = _target_value_producer(target_program, operand)
            _require_precedes_in_same_region(target_program, producer, op)
        if op.results:
            allowed_result_domains = set()
            if orders_memory_issue:
                allowed_result_domains.add(target_ir.EVENT_DOMAIN_MEMORY_BARRIER)
            if lds_completion_result_count:
                allowed_result_domains.add(target_ir.EVENT_DOMAIN_MEMORY_COMPLETION)
            require_token(
                op.results[0],
                "barrier memory-order result",
                allowed_result_domains,
            )
        return

    if op.kind == "buffer_load_to_local":
        if len(op.results) != 1:
            fail(
                "TLXW_VERIFY_ASYNC_PROTOCOL_SHAPE",
                STAGE,
                "direct-to-LDS DMA requires one completion result",
                target_op_id=op.target_op_id,
            )
        for result in op.results:
            require_token(
                result,
                "direct-to-LDS completion result",
                {target_ir.EVENT_DOMAIN_DMA_COMPLETION},
            )
        if "lds_release_dependency_count" in attrs:
            fail(
                "TLXW_VERIFY_ASYNC_PROTOCOL_SEGMENTS",
                STAGE,
                "direct-to-LDS DMA must not carry an LDS-release segment",
                target_op_id=op.target_op_id,
            )
        total_issue_count = int(attrs.get("issue_dependency_count", -1))
        source_count = int(attrs.get("source_issue_dependency_count", -1))
        barrier_count = int(attrs.get("barrier_order_dependency_count", 0))
        if (min(total_issue_count, source_count) < 0 or source_count != total_issue_count
                or barrier_count not in {0, 1} or total_issue_count + barrier_count > len(op.operands)):
            fail(
                "TLXW_VERIFY_ASYNC_PROTOCOL_SEGMENTS",
                STAGE,
                "direct-to-LDS issue segments are malformed",
                target_op_id=op.target_op_id,
            )
        dependency_begin = (len(op.operands) - total_issue_count - barrier_count)
        source_operands = op.operands[dependency_begin:dependency_begin + total_issue_count]
        for operand in source_operands:
            require_token(
                operand,
                "direct-to-LDS source issue dependency",
                {
                    target_ir.EVENT_DOMAIN_DMA_ISSUE,
                    target_ir.EVENT_DOMAIN_EMPTY,
                    None,
                },
            )
        return

    if op.kind == "async_commit_group":
        for operand in op.operands:
            require_token(
                operand,
                "async commit member",
                {
                    target_ir.EVENT_DOMAIN_DMA_COMPLETION,
                    target_ir.EVENT_DOMAIN_EMPTY,
                    None,
                },
            )
        if len(op.results) != 1:
            fail(
                "TLXW_VERIFY_ASYNC_PROTOCOL_SHAPE",
                STAGE,
                "async commit group requires one result",
                target_op_id=op.target_op_id,
            )
        for result in op.results:
            require_token(
                result,
                "async commit group result",
                {target_ir.EVENT_DOMAIN_DMA_GROUP},
            )
        return

    if op.kind == "async_wait":
        completed_count = int(attrs.get("completed_group_dependency_count", -1))
        retained_count = int(attrs.get("retained_issue_dependency_count", -1))
        if (min(completed_count, retained_count) < 0 or completed_count + retained_count != len(op.operands)):
            fail(
                "TLXW_VERIFY_ASYNC_PROTOCOL_SEGMENTS",
                STAGE,
                "async wait requires exact completed and retained-issue "
                "operand segments",
                target_op_id=op.target_op_id,
            )
        completed_end = completed_count
        for operand in op.operands[:completed_end]:
            require_token(
                operand,
                "async wait completed-group dependency",
                {
                    target_ir.EVENT_DOMAIN_DMA_COMPLETION,
                    target_ir.EVENT_DOMAIN_DMA_GROUP,
                    target_ir.EVENT_DOMAIN_EMPTY,
                    None,
                },
            )
        for operand in op.operands[completed_end:]:
            require_token(
                operand,
                "async wait retained issue dependency",
                {target_ir.EVENT_DOMAIN_DMA_ISSUE},
            )
        if len(op.results) != 1:
            fail(
                "TLXW_VERIFY_ASYNC_PROTOCOL_SHAPE",
                STAGE,
                "async wait requires one ready result",
                target_op_id=op.target_op_id,
            )
        for result in op.results:
            require_token(
                result,
                "async wait ready result",
                {target_ir.EVENT_DOMAIN_WAVE_LOCAL_READY},
            )
        return

    if op.kind not in {
            "local_load",
            "local_store",
    }:
        return

    explicit_count = int(attrs.get("explicit_dependency_count", -1))
    data_count = int(attrs.get("data_result_count", -1))
    completion_count = int(attrs.get("completion_result_count", -1))
    barrier_count = int(attrs.get("barrier_order_dependency_count", 0))
    base_operand_count = 2 if op.kind == "local_store" else 1
    if (explicit_count not in {0, 1} or data_count < 0 or completion_count not in {0, 1}
            or data_count + completion_count != len(op.results) or barrier_count not in {0, 1}
            or base_operand_count + explicit_count + barrier_count != len(op.operands)):
        fail(
            "TLXW_VERIFY_ASYNC_PROTOCOL_LOCAL_SEGMENTS",
            STAGE,
            "local access has malformed explicit dependency, barrier-order, "
            "or result segments",
            target_op_id=op.target_op_id,
        )
    if op.kind == "local_store" and explicit_count:
        fail(
            "TLXW_VERIFY_ASYNC_PROTOCOL_SEGMENTS",
            STAGE,
            "ttg.local_store has no source token operand",
            target_op_id=op.target_op_id,
        )
    if explicit_count:
        require_token(
            op.operands[base_operand_count],
            "local-load explicit wait dependency",
            {target_ir.EVENT_DOMAIN_WAVE_LOCAL_READY},
        )
    if completion_count:
        require_token(
            op.results[-1],
            "local access memory-issue completion result",
            {target_ir.EVENT_DOMAIN_MEMORY_COMPLETION},
        )


def _verify_layout_convert_fact_policy(op):
    attrs = _attrs_dict(op)
    policy = attrs.get("fact_policy")
    if policy != "invalidate_layout_sensitive":
        fail(
            "TLXW_VERIFY_LAYOUT_FACT_POLICY",
            STAGE,
            "layout_convert must invalidate layout-sensitive facts",
            target_op_id=op.target_op_id,
        )
    if op.fact_ids:
        fail(
            "TLXW_VERIFY_LAYOUT_FACT_POLICY",
            STAGE,
            "layout_convert that invalidates layout-sensitive facts must not "
            "carry fact ids",
            target_op_id=op.target_op_id,
        )


def _verify_layout_convert_source_op(op, source_program):
    allowed_source_ops = {
        "tt.reshape",
        "tt.trans",
        "ttg.convert_layout",
    }
    if op.source_op_index is None:
        fail(
            "TLXW_VERIFY_LAYOUT_CONVERT_SOURCE",
            STAGE,
            "layout_convert target op must come from a source layout operation",
            target_op_id=op.target_op_id,
        )
    try:
        source_op = source_program.ops[int(op.source_op_index)]
    except IndexError:
        fail(
            "TLXW_VERIFY_LAYOUT_CONVERT_SOURCE",
            STAGE,
            "layout_convert target op references an unknown source op",
            target_op_id=op.target_op_id,
            source_op_index=op.source_op_index,
        )
    if source_op.name not in allowed_source_ops:
        fail(
            "TLXW_VERIFY_LAYOUT_CONVERT_SOURCE",
            STAGE,
            "layout_convert target op must come from source ttg.convert_layout "
            f"or a structural tensor view, not {source_op.name}",
            target_op_id=op.target_op_id,
            source_op_index=op.source_op_index,
        )


def _verify_fact_target_compatible(target_program, op, fact, target_value_id):
    del target_program
    if target_value_id in fact.subject_target_ids:
        return
    fail(
        "TLXW_VERIFY_FACT_TARGET",
        STAGE,
        f"assumption {fact.assumption_id} does not apply to target value "
        f"{target_value_id}",
        target_op_id=op.target_op_id,
        target_value_id=target_value_id,
        fact_id=fact.assumption_id,
    )


def _verify_attrs(op):
    names = set()
    for attr in op.attrs:
        if attr.name in names:
            fail(
                "TLXW_VERIFY_DUPLICATE_ATTR",
                STAGE,
                f"target op {op.target_op_id} has duplicate attr {attr.name}",
                target_op_id=op.target_op_id,
            )
        names.add(attr.name)
        if not _is_schema_value(attr.value):
            fail(
                "TLXW_VERIFY_NON_SCHEMA_ATTR",
                STAGE,
                f"target op {op.target_op_id} attr {attr.name} is not schema data",
                target_op_id=op.target_op_id,
            )


def _verify_memory_edges(op, target_program):
    attrs = _attrs_dict(op)
    semantic_edge_attrs = frozenset({
        "offset_component_coordinate_bases",
        "offset_range",
        "offset_scalar_count",
        "offset_shape",
        "offset_terms",
        "offset_workitem_coordinate_coefficients",
        "source_coordinate_mode",
        "source_linear_component_bases",
        "source_offset_no_signed_wrap",
        "source_offset_range",
        "source_offset_terms",
        "source_scalar_component_sources",
        "source_scalar_count",
        "source_shape",
    })
    leaked_attrs = tuple(sorted(semantic_edge_attrs.intersection(attrs)))
    if attrs.get("offset_mode", "operand") != "operand" or leaked_attrs:
        fail(
            "TLXW_VERIFY_MEMORY_EDGE",
            STAGE,
            "memory operations must consume a typed offset operand; affine "
            f"edge attrs are forbidden: {leaked_attrs}",
            target_op_id=op.target_op_id,
        )
    source_pointer_mode = attrs.get("source_pointer_mode", "base_offset")
    if source_pointer_mode != "base_offset":
        fail(
            "TLXW_VERIFY_MEMORY_EDGE",
            STAGE,
            f"unsupported memory source pointer mode {source_pointer_mode!r}",
            target_op_id=op.target_op_id,
        )
    offset_operand = {
        "buffer_load": 1,
        "buffer_load_to_local": 2,
        "buffer_store": 2,
    }[op.kind]
    if len(op.operands) <= offset_operand:
        fail(
            "TLXW_VERIFY_MEMORY_EDGE",
            STAGE,
            "memory operation is missing its typed offset operand",
            target_op_id=op.target_op_id,
        )
    offset_type = target_program.values[int(op.operands[offset_operand])].type
    if (offset_type.element_type not in {"i32", "index"} or offset_type.representation not in {"simd", "simd_tuple"}):
        fail(
            "TLXW_VERIFY_MEMORY_EDGE",
            STAGE,
            "memory offset operands must use a SIMD i32 or index "
            "representation",
            target_op_id=op.target_op_id,
            target_value_id=int(op.operands[offset_operand]),
        )
    has_mask = bool(attrs.get("has_mask", False))
    redundant_register_mask = attrs.get("redundant_register_mask", 0)
    redundant_lane_mask = attrs.get("redundant_lane_mask", 0)
    redundant_wave_mask = attrs.get("redundant_wave_mask", 0)
    ownership_masks = (
        redundant_register_mask,
        redundant_lane_mask,
        redundant_wave_mask,
    )
    if (any(not isinstance(mask, int) or mask < 0 for mask in ownership_masks)
            or (any(ownership_masks) and (op.kind != "buffer_store"))):
        fail(
            "TLXW_VERIFY_MEMORY_EDGE",
            STAGE,
            "invalid canonical ownership masks on memory operation",
            target_op_id=op.target_op_id,
        )
    wave_count = attrs.get("wave_count")
    if (redundant_wave_mask
            and (not isinstance(wave_count, int) or wave_count <= 1 or wave_count &
                 (wave_count - 1) or redundant_wave_mask >= wave_count or redundant_wave_mask & ~(wave_count - 1))):
        fail(
            "TLXW_VERIFY_MEMORY_EDGE",
            STAGE,
            "redundant-wave mask is incompatible with the memory wave count",
            target_op_id=op.target_op_id,
        )
    lane_width = attrs.get("lane_width")
    if (redundant_lane_mask
            and (not isinstance(lane_width, int) or lane_width <= 1 or lane_width &
                 (lane_width - 1) or redundant_lane_mask >= lane_width or redundant_lane_mask & ~(lane_width - 1))):
        fail(
            "TLXW_VERIFY_MEMORY_EDGE",
            STAGE,
            "redundant-lane mask is incompatible with the memory lane width",
            target_op_id=op.target_op_id,
        )
    access_component_count = attrs.get("access_component_count")
    if (redundant_register_mask
            and (op.kind != "buffer_store" or not isinstance(access_component_count, int) or access_component_count <= 1
                 or redundant_register_mask >= access_component_count)):
        fail(
            "TLXW_VERIFY_MEMORY_EDGE",
            STAGE,
            "redundant-register mask is incompatible with store components",
            target_op_id=op.target_op_id,
        )
    mode = attrs.get(
        "mask_operand_mode",
        "operand" if has_mask else "none",
    )
    if mode not in {"none", "operand"}:
        fail(
            "TLXW_VERIFY_MASK_EDGE",
            STAGE,
            f"unsupported memory mask operand mode {mode!r}",
            target_op_id=op.target_op_id,
        )
    if (mode == "none") != (not has_mask):
        fail(
            "TLXW_VERIFY_MASK_EDGE",
            STAGE,
            "memory mask operand mode does not match has_mask",
            target_op_id=op.target_op_id,
        )
    forbidden = tuple(name for name in attrs if name.startswith("mask_predicate_") or name == "mask_scalar_count")
    if forbidden:
        fail(
            "TLXW_VERIFY_MASK_EDGE",
            STAGE,
            "memory operations must consume a typed mask operand; semantic "
            f"predicate attrs are forbidden: {forbidden}",
            target_op_id=op.target_op_id,
        )


def _verify_make_buffer(op, target_program):
    attrs = _attrs_dict(op)
    if len(op.operands) != 1 or len(op.results) != 1:
        fail(
            "TLXW_VERIFY_MAKE_BUFFER",
            STAGE,
            "make_buffer requires one base pointer and one result",
            target_op_id=op.target_op_id,
        )
    base_type = target_program.values[int(op.operands[0])].type
    result_type = target_program.values[int(op.results[0])].type
    range_bytes = attrs.get("range_bytes")
    if (base_type.representation != "uniform_pointer" or result_type.representation != "uniform_buffer_pointer"
            or int(result_type.component_count) != 1 or base_type.element_type != result_type.element_type
            or attrs.get("element_type") != result_type.element_type or not isinstance(range_bytes, int)
            or range_bytes <= 0 or range_bytes > (1 << 31) - 1):
        fail(
            "TLXW_VERIFY_MAKE_BUFFER",
            STAGE,
            "make_buffer base, result, element type, or byte range is invalid",
            target_op_id=op.target_op_id,
        )


def _verify_type_convert(op, target_program):
    attrs = _attrs_dict(op)
    if len(op.operands) != 1 or len(op.results) != 1:
        fail(
            "TLXW_VERIFY_TYPE_CONVERT",
            STAGE,
            "type_convert requires one operand and one result",
            target_op_id=op.target_op_id,
        )
    mode = attrs.get("mode")
    if mode not in {
            "index_cast",
            "packet_to_scalar_components",
            "scalar_components_to_packet",
    }:
        fail(
            "TLXW_VERIFY_TYPE_CONVERT",
            STAGE,
            f"unsupported structural type conversion mode {mode!r}",
            target_op_id=op.target_op_id,
        )
    operand_type = target_program.values[int(op.operands[0])].type
    result_type = target_program.values[int(op.results[0])].type
    if mode == "index_cast":
        if (operand_type.kind != result_type.kind or operand_type.representation != result_type.representation
                or operand_type.lane_width != result_type.lane_width
                or operand_type.component_count != result_type.component_count
                or {operand_type.element_type, result_type.element_type} not in ({"index", "i32"}, {"index", "i64"})):
            fail(
                "TLXW_VERIFY_TYPE_CONVERT",
                STAGE,
                "index cast has inconsistent types or value distribution",
                target_op_id=op.target_op_id,
            )
        return
    packet_count = attrs.get("packet_component_count")
    packet_width = attrs.get("packet_width")
    if (not isinstance(packet_count, int) or packet_count <= 0 or not isinstance(packet_width, int)
            or packet_width <= 0):
        fail(
            "TLXW_VERIFY_TYPE_CONVERT",
            STAGE,
            "packet structural conversion requires positive dimensions",
            target_op_id=op.target_op_id,
        )
    packet_representations = {"simd_packet", "simd_packet_tuple"}
    scalar_representations = {"simd", "simd_tuple"}
    source_type, destination_type = ((operand_type, result_type) if mode == "packet_to_scalar_components" else
                                     (result_type, operand_type))
    if (source_type.representation not in packet_representations
            or destination_type.representation not in scalar_representations
            or int(source_type.component_count) != packet_count
            or int(destination_type.component_count) != packet_count * packet_width
            or source_type.element_type != destination_type.element_type
            or source_type.lane_width != destination_type.lane_width):
        fail(
            "TLXW_VERIFY_TYPE_CONVERT",
            STAGE,
            "packet structural conversion types do not match its dimensions",
            target_op_id=op.target_op_id,
        )


def _verify_layout_transform(op, target_program):
    attrs = _attrs_dict(op)
    allowed_attrs = {
        "broadcast": frozenset(),
        "expand_dims": frozenset({"axis"}),
        "join": frozenset(),
        "split": frozenset(),
    }
    if op.kind == "layout_convert":
        allowed = {"fact_policy", "transform"}
        if attrs.get("transform") == "trans":
            allowed.add("order")
        allowed_attrs["layout_convert"] = frozenset(allowed)
    leaked_attrs = tuple(sorted(set(attrs) - allowed_attrs[op.kind]))
    if leaked_attrs:
        fail(
            "TLXW_VERIFY_LAYOUT_ATTRS",
            STAGE,
            f"{op.kind} carries non-structural layout attrs {leaked_attrs}",
            target_op_id=op.target_op_id,
        )
    expected_counts = {
        "broadcast": (1, 1),
        "expand_dims": (1, 1),
        "join": (2, 1),
        "layout_convert": (1, 1),
        "split": (1, 2),
    }
    expected_operands, expected_results = expected_counts[op.kind]
    if (len(op.operands) != expected_operands or len(op.results) != expected_results):
        fail(
            "TLXW_VERIFY_LAYOUT_TRANSFORM",
            STAGE,
            f"{op.kind} has an invalid operand or result count",
            target_op_id=op.target_op_id,
        )
    value_ids = tuple(int(value_id) for value_id in (*op.operands, *op.results))
    values = tuple(target_program.values[value_id] for value_id in value_ids)
    types = tuple(value.type for value in values)
    if (any(value.layout_map_id is None for value in values)
            or any(type_.element_type != types[0].element_type for type_ in types)
            or any(type_.lane_width != types[0].lane_width for type_ in types)):
        fail(
            "TLXW_VERIFY_LAYOUT_TRANSFORM",
            STAGE,
            f"{op.kind} requires layout-bearing tensor values with matching "
            "element and lane types",
            target_op_id=op.target_op_id,
        )
    layouts = tuple(target_program.layouts[int(value.layout_map_id)] for value in values)
    if any(layout.linear_layout is None for layout in layouts):
        fail(
            "TLXW_VERIFY_LAYOUT_TRANSFORM",
            STAGE,
            f"{op.kind} requires symbolic linear layouts",
            target_op_id=op.target_op_id,
        )
    shapes = tuple(tuple(int(dim) for dim in layout.shape) for layout in layouts)
    valid = True
    if op.kind == "broadcast":
        source_shape, result_shape = shapes
        valid = len(source_shape) == len(result_shape) and all(source in {1, result}
                                                               for source, result in zip(source_shape, result_shape))
    elif op.kind == "expand_dims":
        source_shape, result_shape = shapes
        axis = attrs.get("axis")
        valid = (isinstance(axis, int) and 0 <= axis < len(result_shape) and result_shape[axis] == 1
                 and result_shape[:axis] + result_shape[axis + 1:] == source_shape)
    elif op.kind == "join":
        first_shape, second_shape, result_shape = shapes
        valid = first_shape == second_shape and result_shape == first_shape + (2, )
    elif op.kind == "split":
        source_shape, first_shape, second_shape = shapes
        valid = (bool(source_shape) and source_shape[-1] == 2 and first_shape == second_shape == source_shape[:-1])
    else:
        source_shape, result_shape = shapes
        transform = attrs.get("transform")
        if transform == "identity":
            valid = source_shape == result_shape
        elif transform == "reshape":
            valid = _shape_product(source_shape) == _shape_product(result_shape)
        elif transform == "trans":
            order = attrs.get("order")
            valid = (isinstance(order, tuple) and sorted(order) == list(range(len(source_shape)))
                     and tuple(source_shape[dim] for dim in order) == result_shape)
        else:
            valid = False
    if not valid:
        fail(
            "TLXW_VERIFY_LAYOUT_TRANSFORM",
            STAGE,
            f"{op.kind} has incompatible structural layout semantics",
            target_op_id=op.target_op_id,
        )


def _verify_reduction(op, target_program):
    attrs = _attrs_dict(op)
    allowed = {"axis"}
    leaked_attrs = tuple(sorted(set(attrs) - allowed))
    if leaked_attrs:
        fail(
            "TLXW_VERIFY_REDUCTION_ATTRS",
            STAGE,
            f"reduction carries non-structural attrs {leaked_attrs}",
            target_op_id=op.target_op_id,
        )
    if len(op.operands) != 1 or len(op.results) != 1 or len(op.region_ids) != 1:
        fail(
            "TLXW_VERIFY_REDUCTION",
            STAGE,
            "reduction requires one operand, one result, and one combiner region",
            target_op_id=op.target_op_id,
        )
    operand = target_program.values[int(op.operands[0])]
    result = target_program.values[int(op.results[0])]
    types = (operand.type, result.type)
    if (any(value.layout_map_id is None for value in (operand, result)) or any(type_.representation not in {
            "simd",
            "simd_tuple",
            "simd_packet",
            "simd_packet_tuple",
    } for type_ in types) or types[0].element_type != types[1].element_type
            or types[0].lane_width != types[1].lane_width):
        fail(
            "TLXW_VERIFY_REDUCTION",
            STAGE,
            "reduction requires layout-bearing SIMD packets with matching element types and lane widths",
            target_op_id=op.target_op_id,
        )
    region_id = int(op.region_ids[0])
    if region_id <= 0 or region_id >= len(target_program.regions):
        fail(
            "TLXW_VERIFY_REDUCTION",
            STAGE,
            "reduction references an invalid combiner region",
            target_op_id=op.target_op_id,
        )
    region = target_program.regions[region_id]
    if len(region.block_arg_ids) != 2 or len(region.yield_value_ids) != 1:
        fail(
            "TLXW_VERIFY_REDUCTION",
            STAGE,
            "reduction combiner requires two arguments and one yielded value",
            target_op_id=op.target_op_id,
        )
    combiner_values = tuple(target_program.values[int(target_value_id)] for target_value_id in (
        *region.block_arg_ids,
        *region.yield_value_ids,
    ))
    if any(value.layout_map_id is not None or value.type.representation != "simd" or value.type.element_type !=
           types[0].element_type or value.type.lane_width != types[0].lane_width or int(value.type.component_count) != 1
           for value in combiner_values):
        fail(
            "TLXW_VERIFY_REDUCTION",
            STAGE,
            "reduction combiner values must be scalar-packet SIMD values "
            "matching the reduced packet element type",
            target_op_id=op.target_op_id,
        )
    axis = attrs.get("axis")
    if not isinstance(axis, int):
        fail(
            "TLXW_VERIFY_REDUCTION",
            STAGE,
            "reduction axis must be an integer",
            target_op_id=op.target_op_id,
        )


def _shape_product(shape):
    result = 1
    for dim in shape:
        result *= int(dim)
    return result


def _attrs_dict(op):
    return {attr.name: attr.value for attr in op.attrs}


def _verify_region_op_ids(target_program, op_count):
    for region in target_program.regions:
        for target_op_id in region.op_ids:
            if target_op_id < 0 or target_op_id >= op_count:
                fail(
                    "TLXW_VERIFY_UNKNOWN_REGION_OP",
                    STAGE,
                    f"target region {region.target_region_id} references "
                    f"missing op {target_op_id}",
                    target_op_id=target_op_id,
                )


def _verify_source_results_covered(source_program, target_program):
    erased = set(target_program.erased_source_values)
    for op in source_program.ops:
        for source_value_id in op.results:
            targets = target_program.source_value_targets.get(source_value_id, ())
            if source_value_id in erased:
                continue
            if len(targets) != 1:
                fail(
                    "TLXW_VERIFY_SOURCE_RESULT_COVERAGE",
                    STAGE,
                    f"source result {source_value_id} has {len(targets)} "
                    "target values",
                    source_op_index=op.index,
                    source_value_id=source_value_id,
                )


def _verify_memory_effects_tokenized(source_program, token_program):
    effect_op_indices = {effect.op_index for effect in token_program.memory_effects}
    for op in source_program.ops:
        if op.name in {
                "tt.load",
                "tt.store",
                "ttg.async_copy_global_to_local",
                "amdg.buffer_load",
                "amdg.buffer_load_to_local",
                "amdg.buffer_store",
                "ttg.local_load",
                "ttg.local_store",
        }:
            if op.index not in effect_op_indices:
                fail(
                    "TLXW_VERIFY_UNTOKENIZED_MEMORY_EFFECT",
                    STAGE,
                    f"memory op {op.name} has no memory effect",
                    source_op_index=op.index,
                )


def _is_schema_value(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, tuple):
        return all(_is_schema_value(item) for item in value)
    if isinstance(value, frozenset):
        return all(_is_schema_value(item) for item in value)
    return False


def _is_layout_schema_value(value):
    if _is_schema_value(value):
        return True
    if isinstance(value, target_ir.TargetAttr):
        return bool(value.name) and _is_layout_schema_value(value.value)
    if isinstance(value, target_ir.TargetLinearLayout):
        if (not value.in_dims or len(set(value.in_dims)) != len(value.in_dims) or not value.out_dims
                or len({name
                        for name, _size in value.out_dims}) != len(value.out_dims)
                or any(not name or int(size) <= 0 for name, size in value.out_dims)):
            return False
        out_rank = len(value.out_dims)
        if tuple(name for name, _bases in value.bases) != value.in_dims:
            return False
        return all(len(basis) == out_rank for _name, bases in value.bases for basis in bases)
    if isinstance(value, tuple):
        return all(_is_layout_schema_value(item) for item in value)
    return False
