"""Target-program verifier for the TLX Wave converter."""

from . import domains
from . import target_ir
from .diagnostics import fail

STAGE = "verification"

_PROOF_DEPENDENT_OPS = frozenset({"assume", "buffer_load_to_local", "buffer_load", "buffer_store"})
_TARGET_REPRESENTATIONS = frozenset({
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
    fact_program=None,
    token_program=None,
):
    _verify_target_value_ids(target_program)
    _verify_target_value_types(target_program)
    _verify_ops(target_program, fact_program, source_program)
    if source_program is not None:
        _verify_source_results_covered(source_program, target_program)
    if token_program is not None and source_program is not None:
        _verify_memory_effects_tokenized(source_program, token_program)
    return True


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
    for value in target_program.values:
        representation = str(value.type.representation)
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


def _verify_ops(target_program, fact_program, source_program):
    value_count = len(target_program.values)
    op_count = len(target_program.ops)
    facts_by_id = _facts_by_id(fact_program)
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
        _verify_attrs(op)
        _verify_provenance_only_target_ids(op, value_count)
        if op.kind in {"buffer_load_to_local", "buffer_load", "buffer_store"}:
            _verify_memory_edges(op, target_program)
        if op.kind in {
            "token",
            "token_join",
            "issue_token",
            "barrier",
            "buffer_load_to_local",
            "async_commit_group",
            "async_wait",
            "local_load",
            "local_load_mma_payload",
            "local_store",
        }:
            _verify_async_protocol_op(op, target_program, source_program)
        if op.kind == "affine_materialize":
            _verify_affine_materialize(op, target_program)
        if op.kind == "type_convert":
            _verify_type_convert(op, target_program)
        if op.kind == "cmpi_select":
            _verify_cmpi_select(op, target_program)
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
            _verify_layout_convert_mode(op)
            _verify_layout_convert_fact_policy(op)
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
        if (
            allowed_domains is not None
            and value.event_domain not in allowed_domains
        ):
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

    if op.kind in {"token_join", "issue_token"}:
        input_count = int(attrs.get("input_count", -1))
        if (
            len(op.results) != 1
            or input_count <= 0
            or input_count != len(op.operands)
        ):
            fail(
                "TLXW_VERIFY_ASYNC_PROTOCOL_SHAPE",
                STAGE,
                f"{op.kind} requires an exact input count and one result",
                target_op_id=op.target_op_id,
            )
        for operand in op.operands:
            require_token(operand, f"{op.kind} operand")
        expected = {target_ir.EVENT_DOMAIN_LDS_FRONTIER}
        if op.kind == "issue_token":
            projection_domain = attrs.get("projection_domain")
            if projection_domain != target_ir.EVENT_DOMAIN_DMA_ISSUE:
                fail(
                    "TLXW_VERIFY_ASYNC_PROTOCOL_DOMAIN",
                    STAGE,
                    "issue token requires an explicit DMA-issue projection domain",
                    target_op_id=op.target_op_id,
                )
            expected_provenance = "partial_wait_retained_group"
            if attrs.get("projection_provenance") != expected_provenance:
                fail(
                    "TLXW_VERIFY_ASYNC_PROTOCOL_PROVENANCE",
                    STAGE,
                    f"{projection_domain} projection requires provenance "
                    f"{expected_provenance!r}",
                    target_op_id=op.target_op_id,
                )
            expected = {projection_domain}
            allowed_input_domains = {
                target_ir.EVENT_DOMAIN_DMA_COMPLETION,
                target_ir.EVENT_DOMAIN_DMA_GROUP,
                target_ir.EVENT_DOMAIN_EMPTY,
                None,
            }
            for operand in op.operands:
                require_token(
                    operand,
                    f"{projection_domain} projection operand",
                    allowed_input_domains,
                )
        else:
            result_domain = attrs.get("event_domain")
            if result_domain != target_ir.EVENT_DOMAIN_LDS_FRONTIER:
                fail(
                    "TLXW_VERIFY_ASYNC_PROTOCOL_DOMAIN",
                    STAGE,
                    "token join requires an explicit LDS-frontier domain",
                    target_op_id=op.target_op_id,
                )
            for operand in op.operands:
                require_token(
                    operand,
                    "LDS-frontier join operand",
                    {
                        target_ir.EVENT_DOMAIN_LDS_COMPLETION,
                        target_ir.EVENT_DOMAIN_LDS_FRONTIER,
                        target_ir.EVENT_DOMAIN_LDS_RELEASED,
                        target_ir.EVENT_DOMAIN_EMPTY,
                    },
                )
        require_token(op.results[0], f"{op.kind} result", expected)
        return

    if op.kind == "barrier":
        dependency_count = int(attrs.get("dependency_count", -1))
        if dependency_count != len(op.operands):
            fail(
                "TLXW_VERIFY_ASYNC_PROTOCOL_SEGMENTS",
                STAGE,
                "barrier dependency count does not match its operands",
                target_op_id=op.target_op_id,
            )
        if not op.operands:
            if op.results:
                fail(
                    "TLXW_VERIFY_ASYNC_PROTOCOL_SHAPE",
                    STAGE,
                    "dependency-free barrier must not publish an LDS result",
                    target_op_id=op.target_op_id,
                )
            return
        if len(op.results) != 1:
            fail(
                "TLXW_VERIFY_ASYNC_PROTOCOL_SHAPE",
                STAGE,
                "LDS-dependent barrier requires one publication result",
                target_op_id=op.target_op_id,
            )
        for operand in op.operands:
            require_token(
                operand,
                "barrier LDS dependency",
                {
                    target_ir.EVENT_DOMAIN_LDS_COMPLETION,
                    target_ir.EVENT_DOMAIN_LDS_FRONTIER,
                    target_ir.EVENT_DOMAIN_LDS_RELEASED,
                    target_ir.EVENT_DOMAIN_EMPTY,
                },
            )
        require_token(
            op.results[0],
            "barrier LDS publication result",
            {target_ir.EVENT_DOMAIN_LDS_RELEASED},
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
        if (
            min(total_issue_count, source_count) < 0
            or source_count != total_issue_count
            or total_issue_count > len(op.operands)
        ):
            fail(
                "TLXW_VERIFY_ASYNC_PROTOCOL_SEGMENTS",
                STAGE,
                "direct-to-LDS issue segments are malformed",
                target_op_id=op.target_op_id,
            )
        dependency_begin = len(op.operands) - total_issue_count
        source_operands = op.operands[dependency_begin:]
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
        release_count = int(attrs.get("lds_release_dependency_count", -1))
        if (
            min(completed_count, retained_count, release_count) < 0
            or completed_count + retained_count + release_count
            != len(op.operands)
        ):
            fail(
                "TLXW_VERIFY_ASYNC_PROTOCOL_SEGMENTS",
                STAGE,
                "async wait requires completed, retained-issue, and "
                "LDS-release operand segments",
                target_op_id=op.target_op_id,
            )
        completed_end = completed_count
        retained_end = completed_end + retained_count
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
        for operand in op.operands[completed_end:retained_end]:
            require_token(
                operand,
                "async wait retained issue dependency",
                {target_ir.EVENT_DOMAIN_DMA_ISSUE},
            )
        for operand in op.operands[retained_end:]:
            require_token(
                operand,
                "async wait LDS-release dependency",
                {
                    target_ir.EVENT_DOMAIN_LDS_COMPLETION,
                    target_ir.EVENT_DOMAIN_LDS_FRONTIER,
                    target_ir.EVENT_DOMAIN_LDS_RELEASED,
                    target_ir.EVENT_DOMAIN_EMPTY,
                },
            )
        mode = attrs.get("publication_mode")
        result_domain = {
            "workgroup": target_ir.EVENT_DOMAIN_WORKGROUP_READY,
            "wave_local": target_ir.EVENT_DOMAIN_WAVE_LOCAL_READY,
        }.get(mode)
        if result_domain is None:
            fail(
                "TLXW_VERIFY_ASYNC_PROTOCOL_PUBLICATION",
                STAGE,
                f"unsupported async wait publication mode {mode!r}",
                target_op_id=op.target_op_id,
            )
        expected_provenance = {
            "workgroup": "amd_membar_compatibility",
            "wave_local": "single_wave_ownership",
        }[mode]
        if attrs.get("publication_provenance") != expected_provenance:
            fail(
                "TLXW_VERIFY_ASYNC_PROTOCOL_PROVENANCE",
                STAGE,
                f"async wait {mode} publication requires provenance "
                f"{expected_provenance!r}",
                target_op_id=op.target_op_id,
            )
        coalesced_barrier_op_index = int(
            attrs.get("coalesced_source_barrier_op_index", -1)
        )
        if coalesced_barrier_op_index < -1:
            fail(
                "TLXW_VERIFY_ASYNC_PROTOCOL_PUBLICATION",
                STAGE,
                "coalesced source barrier index must be -1 or nonnegative",
                target_op_id=op.target_op_id,
            )
        if coalesced_barrier_op_index >= 0:
            _verify_coalesced_wait_publication_barrier(
                op,
                target_program,
                source_program,
                coalesced_barrier_op_index,
                mode,
            )
        if len(op.results) != 1:
            fail(
                "TLXW_VERIFY_ASYNC_PROTOCOL_SHAPE",
                STAGE,
                "async wait requires one ready result",
                target_op_id=op.target_op_id,
            )
        for result in op.results:
            require_token(result, "async wait ready result", {result_domain})
        return

    if not bool(attrs.get("protocol_tracked", False)):
        return
    data_count = int(attrs.get("data_result_count", -1))
    completion_count = int(attrs.get("completion_result_count", -1))
    if (
        data_count < 0
        or completion_count != 1
        or data_count + completion_count != len(op.results)
    ):
        fail(
            "TLXW_VERIFY_ASYNC_PROTOCOL_LOCAL_RESULTS",
            STAGE,
            "tracked LDS access requires one explicit completion result",
            target_op_id=op.target_op_id,
        )
    require_token(
        op.results[-1],
        "tracked LDS completion result",
        {target_ir.EVENT_DOMAIN_LDS_COMPLETION},
    )
    readiness_count = int(attrs.get("readiness_dependency_count", -1))
    if readiness_count <= 0 or readiness_count > len(op.operands):
        fail(
            "TLXW_VERIFY_ASYNC_PROTOCOL_SEGMENTS",
            STAGE,
            "tracked LDS access requires an explicit readiness segment",
            target_op_id=op.target_op_id,
        )
    for operand in op.operands[-readiness_count:]:
        require_token(
            operand,
            "tracked LDS readiness dependency",
            {
                target_ir.EVENT_DOMAIN_WORKGROUP_READY,
                target_ir.EVENT_DOMAIN_WAVE_LOCAL_READY,
            },
        )


def _verify_coalesced_wait_publication_barrier(
    op,
    target_program,
    source_program,
    barrier_op_index,
    publication_mode,
):
    if publication_mode != "workgroup":
        fail(
            "TLXW_VERIFY_ASYNC_PROTOCOL_PUBLICATION",
            STAGE,
            "only a workgroup wait-ready point may coalesce a source CTA barrier",
            target_op_id=op.target_op_id,
        )
    if source_program is None or op.source_op_index is None:
        fail(
            "TLXW_VERIFY_ASYNC_PROTOCOL_PROVENANCE",
            STAGE,
            "coalesced wait publication requires source-program provenance",
            target_op_id=op.target_op_id,
        )
    wait_op_index = int(op.source_op_index)
    if not (0 <= wait_op_index < len(source_program.ops)):
        fail(
            "TLXW_VERIFY_ASYNC_PROTOCOL_PROVENANCE",
            STAGE,
            f"async wait source op index {wait_op_index} is out of range",
            target_op_id=op.target_op_id,
        )
    if not (0 <= barrier_op_index < len(source_program.ops)):
        fail(
            "TLXW_VERIFY_ASYNC_PROTOCOL_PROVENANCE",
            STAGE,
            f"coalesced barrier source op index {barrier_op_index} is out of range",
            target_op_id=op.target_op_id,
        )
    wait_op = source_program.ops[wait_op_index]
    barrier_op = source_program.ops[barrier_op_index]
    is_cta_barrier = (
        barrier_op.name == "rocdl.s.barrier"
        or (
            barrier_op.name == "ttg.barrier"
            and int(barrier_op.attrs.get("addrSpace", -1)) == 1
        )
    )
    if wait_op.name != "ttg.async_wait" or not is_cta_barrier:
        fail(
            "TLXW_VERIFY_ASYNC_PROTOCOL_PROVENANCE",
            STAGE,
            "coalesced publication must pair ttg.async_wait with a source CTA barrier",
            target_op_id=op.target_op_id,
        )
    if wait_op.parent_region_id != barrier_op.parent_region_id:
        fail(
            "TLXW_VERIFY_ASYNC_PROTOCOL_PROVENANCE",
            STAGE,
            "coalesced wait and barrier must belong to the same source region",
            target_op_id=op.target_op_id,
        )
    source_region = source_program.regions[int(wait_op.parent_region_id)]
    try:
        wait_position = source_region.op_indices.index(wait_op_index)
    except ValueError:
        wait_position = -1
    if (
        wait_position < 0
        or wait_position + 1 >= len(source_region.op_indices)
        or int(source_region.op_indices[wait_position + 1]) != barrier_op_index
    ):
        fail(
            "TLXW_VERIFY_ASYNC_PROTOCOL_PROVENANCE",
            STAGE,
            "coalesced source CTA barrier must immediately follow its async wait",
            target_op_id=op.target_op_id,
        )
    if any(
        candidate.kind == "barrier"
        and candidate.source_op_index == barrier_op_index
        for candidate in target_program.ops
    ):
        fail(
            "TLXW_VERIFY_ASYNC_PROTOCOL_PUBLICATION",
            STAGE,
            "a coalesced source CTA barrier must not also be emitted separately",
            target_op_id=op.target_op_id,
        )


def _verify_layout_convert_mode(op):
    mode = _attrs_dict(op).get("mode")
    if mode not in {"alias", "redistribute"}:
        fail(
            "TLXW_VERIFY_LAYOUT_MODE",
            STAGE,
            "layout_convert mode must be alias or redistribute",
            target_op_id=op.target_op_id,
        )


def _verify_layout_convert_fact_policy(op):
    attrs = _attrs_dict(op)
    policy = attrs.get("fact_policy")
    if policy not in {"preserve_equivalent", "invalidate_layout_sensitive"}:
        fail(
            "TLXW_VERIFY_LAYOUT_FACT_POLICY",
            STAGE,
            "layout_convert requires an explicit fact_policy",
            target_op_id=op.target_op_id,
        )
    if policy == "invalidate_layout_sensitive" and op.fact_ids:
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
    value = target_program.values[target_value_id]
    if value.source_value_id is not None:
        if value.source_value_id == fact.subject_value_id:
            return
        fail(
            "TLXW_VERIFY_FACT_TARGET",
            STAGE,
            f"fact {fact.fact_id} applies to source value "
            f"{fact.subject_value_id}, not target value {target_value_id}",
            target_op_id=op.target_op_id,
            target_value_id=target_value_id,
            fact_id=fact.fact_id,
        )
    source_targets = target_program.source_value_targets.get(fact.subject_value_id)
    if source_targets is None or target_value_id in source_targets:
        return
    fail(
        "TLXW_VERIFY_FACT_TARGET",
        STAGE,
        f"fact {fact.fact_id} target value {target_value_id} is not mapped "
        f"from source value {fact.subject_value_id}",
        target_op_id=op.target_op_id,
        target_value_id=target_value_id,
        fact_id=fact.fact_id,
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


def _verify_provenance_only_target_ids(op, value_count):
    value_ids = _attrs_dict(op).get(
        target_ir.PROVENANCE_ONLY_TARGET_IDS_ATTR,
        (),
    )
    if not isinstance(value_ids, tuple):
        fail(
            "TLXW_VERIFY_PROVENANCE_TARGETS",
            STAGE,
            "provenance-only target IDs must be a tuple",
            target_op_id=op.target_op_id,
        )
    for target_value_id in value_ids:
        if (not isinstance(target_value_id, int)
                or target_value_id < 0
                or target_value_id >= value_count):
            fail(
                "TLXW_VERIFY_PROVENANCE_TARGETS",
                STAGE,
                "target op references an unknown provenance-only value",
                target_op_id=op.target_op_id,
                target_value_id=(
                    target_value_id
                    if isinstance(target_value_id, int)
                    else None
                ),
            )
        if int(target_value_id) in op.operands:
            fail(
                "TLXW_VERIFY_PROVENANCE_TARGETS",
                STAGE,
                "provenance-only values must not be runtime operands",
                target_op_id=op.target_op_id,
                target_value_id=int(target_value_id),
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
    offset_type = target_program.values[
        int(op.operands[offset_operand])
    ].type
    if (
        offset_type.element_type != "index"
        or offset_type.representation not in {"simd", "simd_tuple"}
    ):
        fail(
            "TLXW_VERIFY_MEMORY_EDGE",
            STAGE,
            "memory offset operands must use the SIMD index representation",
            target_op_id=op.target_op_id,
            target_value_id=int(op.operands[offset_operand]),
        )
    has_mask = bool(attrs.get("has_mask", False))
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
    forbidden = tuple(
        name for name in attrs
        if name.startswith("mask_predicate_") or name == "mask_scalar_count"
    )
    if forbidden:
        fail(
            "TLXW_VERIFY_MASK_EDGE",
            STAGE,
            "memory operations must consume a typed mask operand; semantic "
            f"predicate attrs are forbidden: {forbidden}",
            target_op_id=op.target_op_id,
        )


def _verify_affine_materialize(op, target_program):
    attrs = _attrs_dict(op)
    if len(op.results) != 1:
        fail(
            "TLXW_VERIFY_AFFINE_EDGE",
            STAGE,
            "affine_materialize requires exactly one result",
            target_op_id=op.target_op_id,
        )
    result_type = target_program.values[int(op.results[0])].type
    if (
        result_type.element_type not in {"i32", "index"}
        or result_type.representation not in {"simd", "simd_tuple"}
    ):
        fail(
            "TLXW_VERIFY_AFFINE_EDGE",
            STAGE,
            "affine_materialize result must be an i32 or index SIMD value",
            target_op_id=op.target_op_id,
            target_value_id=int(op.results[0]),
        )
    scalar_count = attrs.get("scalar_count")
    if not isinstance(scalar_count, int) or scalar_count != len(op.operands):
        fail(
            "TLXW_VERIFY_AFFINE_EDGE",
            STAGE,
            "affine_materialize scalar count must match its operands",
            target_op_id=op.target_op_id,
        )
    terms = attrs.get("terms")
    if not isinstance(terms, tuple):
        fail(
            "TLXW_VERIFY_AFFINE_EDGE",
            STAGE,
            "affine_materialize requires tuple term attrs",
            target_op_id=op.target_op_id,
        )
    mode = attrs.get("mode")
    if mode == "packet_coordinates":
        if result_type.element_type != "index":
            fail(
                "TLXW_VERIFY_AFFINE_EDGE",
                STAGE,
                "packet-coordinate affine materialization must produce index values",
                target_op_id=op.target_op_id,
            )
        _verify_packet_affine_materialize(
            op,
            target_program,
            attrs,
            result_type,
        )
        return
    if mode != "layout_coordinates":
        fail(
            "TLXW_VERIFY_AFFINE_EDGE",
            STAGE,
            f"unsupported affine materialization mode {mode!r}",
            target_op_id=op.target_op_id,
        )
    value_range = attrs.get("value_range")
    if result_type.element_type == "index":
        if (
            not isinstance(value_range, tuple)
            or len(value_range) != 2
            or not all(isinstance(bound, int) for bound in value_range)
            or int(value_range[0]) < 0
            or int(value_range[0]) > int(value_range[1])
        ):
            fail(
                "TLXW_VERIFY_AFFINE_EDGE",
                STAGE,
                "index affine materialization requires a nonnegative closed range",
                target_op_id=op.target_op_id,
            )
    elif value_range is not None:
        fail(
            "TLXW_VERIFY_AFFINE_EDGE",
            STAGE,
            "i32 affine materialization must not carry an index value range",
            target_op_id=op.target_op_id,
        )
    shape = attrs.get("coordinate_shape")
    bases = attrs.get("component_coordinate_bases")
    coefficients = attrs.get("workitem_coordinate_coefficients")
    if not all(isinstance(value, tuple) for value in (shape, bases, coefficients)):
        fail(
            "TLXW_VERIFY_AFFINE_EDGE",
            STAGE,
            "layout-coordinate affine materialization requires tuple "
            "coordinate attrs",
            target_op_id=op.target_op_id,
        )
    rank = len(shape)
    if len(bases) != int(result_type.component_count) or any(
        not isinstance(values, tuple) or len(values) != rank
        for values in (*bases, *coefficients)
    ):
        fail(
            "TLXW_VERIFY_AFFINE_EDGE",
            STAGE,
            "affine_materialize coordinate ranks do not match its result",
            target_op_id=op.target_op_id,
        )
    scalar_sources = attrs.get("scalar_component_sources")
    if scalar_sources is not None:
        _verify_affine_scalar_component_sources(
            op,
            target_program,
            scalar_sources,
            int(result_type.component_count),
        )


def _verify_packet_affine_materialize(
    op,
    target_program,
    attrs,
    result_type,
):
    shape = attrs.get("coordinate_shape")
    order = attrs.get("coordinate_order")
    coordinate_mode = attrs.get("coordinate_mode")
    linear_bases = attrs.get("linear_component_bases")
    scalar_sources = attrs.get("scalar_component_sources")
    component_thread_count = attrs.get("component_thread_count")
    packet_elements = attrs.get("packet_elements")
    value_range = attrs.get("value_range")
    component_count = int(result_type.component_count)
    rank = len(shape) if isinstance(shape, tuple) else -1
    if (
        rank <= 0
        or not isinstance(order, tuple)
        or tuple(sorted(order)) != tuple(range(rank))
        or not isinstance(component_thread_count, int)
        or component_thread_count <= 0
        or not isinstance(packet_elements, int)
        or packet_elements <= 0
        or not isinstance(value_range, tuple)
        or len(value_range) != 2
        or not all(isinstance(bound, int) for bound in value_range)
        or int(value_range[0]) < 0
        or int(value_range[0]) > int(value_range[1])
    ):
        fail(
            "TLXW_VERIFY_AFFINE_EDGE",
            STAGE,
            "packet-coordinate affine materialization has an invalid "
            "coordinate or range schema",
            target_op_id=op.target_op_id,
        )
    if coordinate_mode == "ordered_linear":
        if linear_bases not in {None, ()}:
            fail(
                "TLXW_VERIFY_AFFINE_EDGE",
                STAGE,
                "ordered packet coordinates must not carry linear bases",
                target_op_id=op.target_op_id,
            )
    elif coordinate_mode == "physical_linear_component":
        if (
            not isinstance(linear_bases, tuple)
            or not linear_bases
            or any(
                not isinstance(basis, tuple) or len(basis) != rank
                for basis in linear_bases
            )
        ):
            fail(
                "TLXW_VERIFY_AFFINE_EDGE",
                STAGE,
                "physical packet coordinates require ranked linear bases",
                target_op_id=op.target_op_id,
            )
    else:
        fail(
            "TLXW_VERIFY_AFFINE_EDGE",
            STAGE,
            f"unsupported packet coordinate mode {coordinate_mode!r}",
            target_op_id=op.target_op_id,
        )
    if not isinstance(scalar_sources, tuple):
        fail(
            "TLXW_VERIFY_AFFINE_EDGE",
            STAGE,
            "packet affine scalar maps must match its operands",
            target_op_id=op.target_op_id,
        )
    _verify_affine_scalar_component_sources(
        op,
        target_program,
        scalar_sources,
        component_count,
    )


def _verify_affine_scalar_component_sources(
    op,
    target_program,
    scalar_sources,
    component_count,
):
    if len(scalar_sources) != len(op.operands):
        fail(
            "TLXW_VERIFY_AFFINE_EDGE",
            STAGE,
            "affine scalar maps must match its operands",
            target_op_id=op.target_op_id,
        )
    for operand_id, sources in zip(op.operands, scalar_sources):
        operand_type = target_program.values[int(operand_id)].type
        if (
            not isinstance(sources, tuple)
            or len(sources) != component_count
            or any(
                not isinstance(source, int)
                or source < 0
                or source >= int(operand_type.component_count)
                for source in sources
            )
        ):
            fail(
                "TLXW_VERIFY_AFFINE_EDGE",
                STAGE,
                "packet affine scalar component map is inconsistent with "
                "its operand type",
                target_op_id=op.target_op_id,
                target_value_id=int(operand_id),
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
        "bounded_i32_to_index",
        "component_remap",
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
        if (
            operand_type.kind != result_type.kind
            or operand_type.representation != result_type.representation
            or operand_type.lane_width != result_type.lane_width
            or operand_type.component_count != result_type.component_count
            or {operand_type.element_type, result_type.element_type}
            not in ({"index", "i32"}, {"index", "i64"})
        ):
            fail(
                "TLXW_VERIFY_TYPE_CONVERT",
                STAGE,
                "index cast has inconsistent types or value distribution",
                target_op_id=op.target_op_id,
            )
        return
    if mode == "bounded_i32_to_index":
        value_range = attrs.get("value_range")
        if (
            operand_type.element_type != "i32"
            or result_type.element_type != "index"
            or operand_type.kind != result_type.kind
            or operand_type.representation != result_type.representation
            or operand_type.representation not in {"scalar", "simd", "simd_tuple"}
            or operand_type.lane_width != result_type.lane_width
            or operand_type.component_count != result_type.component_count
            or not isinstance(value_range, tuple)
            or len(value_range) != 2
            or not all(isinstance(bound, int) for bound in value_range)
            or int(value_range[0]) < 0
            or int(value_range[0]) > int(value_range[1])
        ):
            fail(
                "TLXW_VERIFY_TYPE_CONVERT",
                STAGE,
                "bounded i32-to-index conversion has inconsistent types or range",
                target_op_id=op.target_op_id,
            )
        return
    if mode == "component_remap":
        component_sources = attrs.get("component_sources")
        if (
            not isinstance(component_sources, tuple)
            or not component_sources
            or len(component_sources) != int(result_type.component_count)
            or any(
                not isinstance(component, int)
                or component < 0
                or component >= int(operand_type.component_count)
                for component in component_sources
            )
            or operand_type.kind != result_type.kind
            or operand_type.element_type != result_type.element_type
            or operand_type.lane_width != result_type.lane_width
        ):
            fail(
                "TLXW_VERIFY_TYPE_CONVERT",
                STAGE,
                "component remap types or source map are inconsistent",
                target_op_id=op.target_op_id,
            )
        return
    packet_count = attrs.get("packet_component_count")
    packet_width = attrs.get("packet_width")
    if (
        not isinstance(packet_count, int)
        or packet_count <= 0
        or not isinstance(packet_width, int)
        or packet_width <= 0
    ):
        fail(
            "TLXW_VERIFY_TYPE_CONVERT",
            STAGE,
            "packet structural conversion requires positive dimensions",
            target_op_id=op.target_op_id,
        )
    packet_representations = {"simd_packet", "simd_packet_tuple"}
    scalar_representations = {"simd", "simd_tuple"}
    source_type, destination_type = (
        (operand_type, result_type)
        if mode == "packet_to_scalar_components"
        else (result_type, operand_type)
    )
    if (
        source_type.representation not in packet_representations
        or destination_type.representation not in scalar_representations
        or int(source_type.component_count) != packet_count
        or int(destination_type.component_count) != packet_count * packet_width
        or source_type.element_type != destination_type.element_type
        or source_type.lane_width != destination_type.lane_width
    ):
        fail(
            "TLXW_VERIFY_TYPE_CONVERT",
            STAGE,
            "packet structural conversion types do not match its dimensions",
            target_op_id=op.target_op_id,
        )


def _verify_cmpi_select(op, target_program):
    attrs = _attrs_dict(op)
    if len(op.operands) != 4 or len(op.results) != 1:
        fail(
            "TLXW_VERIFY_CMPI_SELECT",
            STAGE,
            "cmpi_select requires compare lhs/rhs, true/false values, and "
            "one result",
            target_op_id=op.target_op_id,
        )
    if (
        set(attrs) != {"predicate", "source_width"}
        or attrs.get("predicate") not in {
            "eq",
            "ne",
            "slt",
            "sle",
            "sgt",
            "sge",
            "ult",
            "ule",
            "ugt",
            "uge",
        }
        or attrs.get("source_width") != 32
    ):
        fail(
            "TLXW_VERIFY_CMPI_SELECT",
            STAGE,
            "cmpi_select requires a supported i32 predicate",
            target_op_id=op.target_op_id,
        )
    lhs_type, rhs_type, true_type, false_type = (
        target_program.values[int(operand)].type
        for operand in op.operands
    )
    result_type = target_program.values[int(op.results[0])].type
    result_components = int(result_type.component_count)
    if (
        lhs_type.element_type != "i32"
        or rhs_type.element_type != "i32"
        or lhs_type.lane_width != rhs_type.lane_width
        or int(lhs_type.component_count) not in {1, result_components}
        or int(rhs_type.component_count) not in {1, result_components}
        or true_type.kind != result_type.kind
        or false_type.kind != result_type.kind
        or true_type.element_type != result_type.element_type
        or false_type.element_type != result_type.element_type
        or true_type.lane_width != result_type.lane_width
        or false_type.lane_width != result_type.lane_width
        or int(true_type.component_count) not in {1, result_components}
        or int(false_type.component_count) not in {1, result_components}
        or result_type.representation in {"mask", "mask_tuple"}
    ):
        fail(
            "TLXW_VERIFY_CMPI_SELECT",
            STAGE,
            "cmpi_select operand and result types are inconsistent",
            target_op_id=op.target_op_id,
        )


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


def _facts_by_id(fact_program):
    if fact_program is None:
        return {}
    return {fact.fact_id: fact for fact in fact_program.facts}


def _is_schema_value(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, tuple):
        return all(_is_schema_value(item) for item in value)
    if isinstance(value, frozenset):
        return all(_is_schema_value(item) for item in value)
    return False
