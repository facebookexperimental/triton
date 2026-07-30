"""Fail-closed audits for expert-authored manual-CUDA plans."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._schema import (
    ArtifactSource,
    async_obligation_kind,
    AuditError,
    AUTHORING_MODE,
    AUTHORING_SCHEMA,
    B200_CAPACITIES,
    cross_dependency_map,
    dependency_map,
    HANDOFF_SCHEMA,
    instruction_map,
    issue_ordinals,
    load_handoff,
    load_json_artifact,
    load_pipelined_ir,
    MAPPING_SCHEMA,
    MEMORY_PLAN_SCHEMA,
    SYNC_PLAN_SCHEMA,
    target_reference,
    validate_plan_header,
)


@dataclass(frozen=True)
class AuditResult:
    kind: str
    checked: int
    ok: bool = True


def _required_list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise AuditError(f"{key} must be a list")
    return value


def _required_object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AuditError(f"{name} must be an object")
    return value


def _required_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise AuditError(f"{name} must be a non-empty string")
    return value


def _required_sha256(value: object, name: str) -> str:
    value = _required_string(value, name)
    if len(value) != 64 or value != value.lower():
        raise AuditError(f"{name} must be a lowercase SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise AuditError(f"{name} must be a lowercase SHA-256") from error
    return value


def _required_non_negative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise AuditError(f"{name} must be a non-negative integer")
    return value


def _required_positive_int(value: object, name: str) -> int:
    value = _required_non_negative_int(value, name)
    if value == 0:
        raise AuditError(f"{name} must be positive")
    return value


def _required_string_list(value: object, name: str, *, allow_empty: bool = False):
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise AuditError(f"{name} must be a list of non-empty strings")
    if not allow_empty and not value:
        raise AuditError(f"{name} must not be empty")
    if len(value) != len(set(value)):
        raise AuditError(f"{name} must not contain duplicates")
    return value


def _require_approved(payload: dict[str, Any], name: str) -> None:
    if payload.get("status") != "approved":
        raise AuditError(f"{name} status must be approved")
    review = _required_object(payload.get("review"), f"{name}.review")
    if review.get("status") != "approved":
        raise AuditError(f"{name} review status must be approved")
    _required_string(review.get("reviewer"), f"{name}.review.reviewer")


def _artifact_relative_path(artifact, relative: str, name: str) -> Path:
    if artifact.path is None:
        raise AuditError(f"{name} needs a file-backed manifest")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise AuditError(f"{name} path must stay beside the manifest")
    root = artifact.path.parent.resolve()
    resolved = (root / candidate).resolve()
    if resolved != root and root not in resolved.parents:
        raise AuditError(f"{name} path escapes the manifest directory")
    return resolved


def _source_text(mapping_artifact, plan: dict[str, Any]) -> str:
    source = _required_object(plan.get("cuda_source"), "mapping.cuda_source")
    path = _artifact_relative_path(
        mapping_artifact,
        _required_string(source.get("path"), "mapping.cuda_source.path"),
        "CUDA source",
    )
    try:
        content = path.read_bytes()
    except OSError as error:
        raise AuditError(f"cannot read CUDA source: {error}") from error
    expected = _required_sha256(source.get("sha256"), "mapping.cuda_source.sha256")
    if hashlib.sha256(content).hexdigest() != expected:
        raise AuditError("CUDA source sha256 mismatch")
    try:
        return content.decode()
    except UnicodeDecodeError as error:
        raise AuditError("CUDA source must be UTF-8") from error


def _evidence_bytes(artifact, reference: object, name: str) -> bytes:
    record = _required_object(reference, name)
    path = _artifact_relative_path(
        artifact,
        _required_string(record.get("path"), f"{name}.path"),
        name,
    )
    try:
        content = path.read_bytes()
    except OSError as error:
        raise AuditError(f"cannot read {name}: {error}") from error
    expected = _required_sha256(record.get("sha256"), f"{name}.sha256")
    if hashlib.sha256(content).hexdigest() != expected:
        raise AuditError(f"{name} sha256 mismatch")
    if not content:
        raise AuditError(f"{name} must not be empty")
    return content


def _build_evidence(mapping_artifact, plan: dict[str, Any]) -> str:
    evidence = _required_object(plan.get("build_evidence"), "mapping.build_evidence")
    if evidence.get("status") != "verified":
        raise AuditError("mapping build evidence status must be verified")
    if evidence.get("cuda_arch") != "sm_100a":
        raise AuditError("mapping build evidence must target sm_100a")
    if evidence.get("source_sha256") != plan["cuda_source"]["sha256"]:
        raise AuditError("mapping build evidence source sha256 mismatch")
    for field in ("command", "toolchain", "reviewer"):
        _required_string(evidence.get(field), f"mapping.build_evidence.{field}")
    binary = _evidence_bytes(mapping_artifact, evidence.get("binary"), "CUDA binary")
    if (
        len(binary) < 20
        or not binary.startswith(b"\x7fELF")
        or binary[4] != 2
        or binary[5] != 1
        or int.from_bytes(binary[18:20], "little") != 190
    ):
        raise AuditError("CUDA binary is not a 64-bit NVIDIA ELF cubin")
    disassembly = _evidence_bytes(
        mapping_artifact,
        evidence.get("disassembly"),
        "CUDA disassembly",
    )
    try:
        return disassembly.decode()
    except UnicodeDecodeError as error:
        raise AuditError("CUDA disassembly must be UTF-8") from error


def _source_anchor(source: str, anchor: object, name: str) -> str:
    anchor = _required_string(anchor, name)
    occurrences = source.count(anchor)
    if occurrences != 1:
        raise AuditError(f"{name} must occur exactly once in CUDA source")
    return anchor


def _physical_groups(ir, plan: dict[str, Any], source: str) -> dict[int, list[int]]:
    expected = {group["id"]: group for group in ir.payload["warp_groups"]}
    result: dict[int, list[int]] = {}
    used_warps: set[int] = set()
    for position, raw in enumerate(_required_list(plan, "physical_groups")):
        group = _required_object(raw, f"physical_groups[{position}]")
        group_id = _required_non_negative_int(
            group.get("group"), f"physical_groups[{position}].group"
        )
        if group_id in result or group_id not in expected:
            raise AuditError(f"invalid or duplicate physical group {group_id}")
        width = _required_positive_int(group.get("width"), f"group {group_id}.width")
        if width != expected[group_id]["width"]:
            raise AuditError(f"physical group {group_id} width mismatch")
        warps = group.get("physical_warps")
        if not isinstance(warps, list) or any(
            not isinstance(warp, int)
            or isinstance(warp, bool)
            or warp < 0
            or warp >= B200_CAPACITIES["physical_warps"]
            for warp in warps
        ):
            raise AuditError(f"physical group {group_id} has invalid warp IDs")
        if len(warps) != width or len(set(warps)) != width:
            raise AuditError(
                f"physical group {group_id} must map exactly {width} warps"
            )
        if used_warps.intersection(warps):
            raise AuditError("physical warp IDs may not be shared by groups")
        used_warps.update(warps)
        for field in (
            "thread_mapping",
            "entry_predicate",
            "regions",
            "reviewer",
        ):
            _required_string(group.get(field), f"physical group {group_id}.{field}")
        _source_anchor(
            source,
            group.get("dispatch_anchor"),
            f"physical group {group_id}.dispatch_anchor",
        )
        result[group_id] = warps
    if set(result) != set(expected):
        raise AuditError("physical_groups must cover every Twill group")
    return result


def _mapping_regions(ir, plan: dict[str, Any], source: str) -> None:
    regions = _required_list(plan, "regions")
    expected = ir.payload["pipeline"]["regions"]
    if len(regions) != len(expected):
        raise AuditError("mapping regions must cover every pipeline region")
    for position, (raw, expected_region) in enumerate(zip(regions, expected)):
        region = _required_object(raw, f"regions[{position}]")
        for field, value in expected_region.items():
            if region.get(field) != value:
                raise AuditError(f"mapping region {position} {field} mismatch")
        _source_anchor(
            source,
            region.get("source_anchor"),
            f"mapping region {position}.source_anchor",
        )
        _required_string(
            region.get("realization"), f"mapping region {position}.realization"
        )
        _required_string(region.get("reviewer"), f"mapping region {position}.reviewer")


def _mapping_nodes(
    ir,
    plan: dict[str, Any],
    source: str,
    disassembly: str,
    physical_groups: dict[int, list[int]],
) -> tuple[set[str], set[str]]:
    instructions = instruction_map(ir)
    ordinals = issue_ordinals(ir)
    seen: set[int] = set()
    allocation_ids: set[str] = set()
    sync_ids: set[str] = set()
    source_anchors: set[str] = set()
    disassembly_anchors: dict[str, tuple[str, str | None]] = {}
    facts = (
        "op_ref",
        "op_kind",
        "pipeline",
        "cycle",
        "stage",
        "offset",
        "group",
        "group_width",
        "lanes",
    )
    for position, raw in enumerate(_required_list(plan, "nodes")):
        node = _required_object(raw, f"nodes[{position}]")
        node_id = _required_non_negative_int(
            node.get("instruction_id"), f"nodes[{position}].instruction_id"
        )
        if node_id in seen:
            raise AuditError(f"instruction {node_id} is mapped more than once")
        instruction = instructions.get(node_id)
        if instruction is None:
            raise AuditError(f"mapping references unknown instruction {node_id}")
        seen.add(node_id)
        for field in facts:
            if node.get(field) != instruction[field]:
                raise AuditError(f"instruction {node_id} {field} mismatch")
        if node.get("issue_order") != ordinals[node_id]:
            raise AuditError(f"instruction {node_id} issue_order mismatch")
        manual = _required_object(
            node.get("manual_cuda"), f"instruction {node_id}.manual_cuda"
        )
        if manual.get("physical_group") != instruction["group"]:
            raise AuditError(f"instruction {node_id} physical group mismatch")
        expected_warps = [
            physical_groups[instruction["group"]][lane] for lane in instruction["lanes"]
        ]
        if manual.get("physical_warps") != expected_warps:
            raise AuditError(f"instruction {node_id} physical warp mapping mismatch")
        source_anchor = _source_anchor(
            source,
            manual.get("source_anchor"),
            f"instruction {node_id}.manual_cuda.source_anchor",
        )
        if source_anchor in source_anchors:
            raise AuditError("mapped instructions must use unique source anchors")
        source_anchors.add(source_anchor)
        realization = manual.get("machine_realization")
        equivalence_proof = manual.get("equivalence_proof")
        if realization == "zero_cost_alias":
            if instruction["pipeline"] != "NONE":
                raise AuditError(
                    f"instruction {node_id} is not eligible for a zero-cost alias"
                )
            if manual.get("disassembly_anchor") is not None:
                raise AuditError(
                    f"zero-cost instruction {node_id} must not name disassembly"
                )
            _required_string(
                equivalence_proof,
                f"instruction {node_id}.manual_cuda.equivalence_proof",
            )
            if manual.get("disassembly_status") != "not_applicable":
                raise AuditError(
                    f"instruction {node_id} disassembly_status must be not_applicable"
                )
        elif realization in {"instruction", "shared_instruction"}:
            disassembly_anchor = _source_anchor(
                disassembly,
                manual.get("disassembly_anchor"),
                f"instruction {node_id}.manual_cuda.disassembly_anchor",
            )
            if realization == "instruction":
                if equivalence_proof is not None:
                    raise AuditError(
                        f"instruction {node_id} must not declare an equivalence proof"
                    )
                if disassembly_anchor in disassembly_anchors:
                    raise AuditError(
                        "exclusive machine instructions need unique disassembly anchors"
                    )
            else:
                proof = _required_string(
                    equivalence_proof,
                    f"instruction {node_id}.manual_cuda.equivalence_proof",
                )
                previous = disassembly_anchors.get(disassembly_anchor)
                if previous is not None and previous != (realization, proof):
                    raise AuditError(
                        "shared disassembly anchors need one equivalence proof"
                    )
            disassembly_anchors[disassembly_anchor] = (
                realization,
                equivalence_proof,
            )
            if manual.get("disassembly_status") != "verified":
                raise AuditError(
                    f"instruction {node_id} disassembly_status must be verified"
                )
            disassembly_line = next(
                line for line in disassembly.splitlines() if disassembly_anchor in line
            )
            primitive = _required_string(
                manual.get("primitive"),
                f"instruction {node_id}.manual_cuda.primitive",
            )
            if primitive not in disassembly_line:
                raise AuditError(
                    f"instruction {node_id} primitive is absent from disassembly"
                )
        else:
            raise AuditError(f"instruction {node_id} has invalid machine_realization")
        for field in (
            "lane_predicate",
            "iteration_expression",
            "region_realization",
            "stage_slot_expression",
            "semantic_lowering",
            "primitive",
            "predicate",
            "semantic_completion",
            "reviewer",
        ):
            _required_string(
                manual.get(field), f"instruction {node_id}.manual_cuda.{field}"
            )
        if manual.get("source_status") != "verified":
            raise AuditError(f"instruction {node_id} source_status must be verified")
        for field, destination in (
            ("operand_allocations", allocation_ids),
            ("result_allocations", allocation_ids),
            ("sync_channels", sync_ids),
        ):
            destination.update(
                _required_string_list(
                    manual.get(field),
                    f"instruction {node_id}.manual_cuda.{field}",
                    allow_empty=True,
                )
            )
    missing = sorted(set(instructions) - seen)
    if missing:
        raise AuditError(f"instructions missing from mapping: {missing}")
    return allocation_ids, sync_ids


def _mapping_dependencies(ir, plan: dict[str, Any]) -> set[str]:
    expected = dependency_map(ir)
    cross = cross_dependency_map(ir)
    seen: set[int] = set()
    sync_ids: set[str] = set()
    facts = (
        "src",
        "dst",
        "distance",
        "latency",
        "producer_group",
        "consumer_group",
    )
    for position, raw in enumerate(_required_list(plan, "dependencies")):
        edge = _required_object(raw, f"dependencies[{position}]")
        edge_index = _required_non_negative_int(
            edge.get("index"), f"dependencies[{position}].index"
        )
        if edge_index in seen or edge_index not in expected:
            raise AuditError(f"invalid or duplicate mapped dependency {edge_index}")
        seen.add(edge_index)
        for field in facts:
            if edge.get(field) != expected[edge_index][field]:
                raise AuditError(f"dependency {edge_index} {field} mismatch")
        for field in (
            "dynamic_iteration_relation",
            "transport",
            "completion_witness",
            "prologue_availability",
            "epilogue_drain",
            "reviewer",
        ):
            _required_string(edge.get(field), f"dependency {edge_index}.{field}")
        channel = edge.get("sync_channel")
        order_proof = edge.get("program_order_proof")
        if edge_index in cross:
            sync_ids.add(
                _required_string(channel, f"dependency {edge_index}.sync_channel")
            )
            if order_proof is not None:
                raise AuditError(
                    f"cross-warp dependency {edge_index} cannot use program order"
                )
        elif channel is None:
            _required_string(
                order_proof, f"dependency {edge_index}.program_order_proof"
            )
        else:
            sync_ids.add(
                _required_string(channel, f"dependency {edge_index}.sync_channel")
            )
    if set(expected) != seen:
        raise AuditError("mapping dependencies must cover every IR edge")
    return sync_ids


def audit_mapping(
    ir_path: ArtifactSource,
    mapping_path: ArtifactSource,
) -> AuditResult:
    """Audit complete node/edge/source coverage and physical placement."""
    ir = load_pipelined_ir(ir_path)
    artifact = load_json_artifact(mapping_path, "mapping manifest")
    plan = artifact.payload
    validate_plan_header(plan, MAPPING_SCHEMA, ir)
    _require_approved(plan, "mapping")
    source = _source_text(artifact, plan)
    if (
        "__global__" not in source
        or '#error "Manual CUDA lowering is required' in source
    ):
        raise AuditError("CUDA source does not contain an executable kernel")
    disassembly = _build_evidence(artifact, plan)
    physical_groups = _physical_groups(ir, plan, source)
    _mapping_regions(ir, plan, source)
    _mapping_nodes(ir, plan, source, disassembly, physical_groups)
    _mapping_dependencies(ir, plan)
    coverage = _required_object(plan.get("coverage"), "mapping.coverage")
    expected_counts = {
        "ir_node_count": len(ir.payload["instructions"]),
        "mapped_node_count": len(ir.payload["instructions"]),
        "ir_edge_count": len(ir.payload["dependencies"]),
        "mapped_edge_count": len(ir.payload["dependencies"]),
    }
    if coverage != expected_counts:
        raise AuditError("mapping coverage is not complete")
    if plan.get("deviations") != []:
        raise AuditError("faithful mapping requires an empty deviations list")
    return AuditResult(
        kind="mapping",
        checked=len(ir.payload["instructions"]) + len(ir.payload["dependencies"]),
    )


def _allocation_space(storage_kind: str) -> tuple[str, str]:
    spaces = {
        "register": ("register_words", "register_words"),
        "shared": ("shared_bytes", "shared_bytes"),
        "barrier": ("shared_bytes", "shared_bytes"),
        "descriptor": ("shared_bytes", "shared_bytes"),
        "tmem": ("tmem_columns", "tmem_columns"),
    }
    try:
        return spaces[storage_kind]
    except KeyError as error:
        raise AuditError(f"unsupported storage kind {storage_kind!r}") from error


def _allocation(
    raw: Any,
    position: int,
    capacities: dict[str, int],
    horizon: int,
    instruction_ids: set[int],
) -> tuple[str, str, int, int, int, int, str | None]:
    allocation = _required_object(raw, f"allocations[{position}]")
    allocation_id = _required_string(
        allocation.get("id"), f"allocations[{position}].id"
    )
    _required_string(
        allocation.get("semantic_value"), f"allocation {allocation_id}.semantic_value"
    )
    storage_kind = _required_string(
        allocation.get("storage_kind"), f"allocation {allocation_id}.storage_kind"
    )
    space, expected_unit = _allocation_space(storage_kind)
    if allocation.get("unit") != expected_unit:
        raise AuditError(f"allocation {allocation_id} has the wrong unit")
    offset = _required_non_negative_int(
        allocation.get("offset"), f"allocation {allocation_id}.offset"
    )
    size = _required_positive_int(
        allocation.get("size"), f"allocation {allocation_id}.size"
    )
    slot_size = _required_positive_int(
        allocation.get("slot_size"), f"allocation {allocation_id}.slot_size"
    )
    alignment = _required_positive_int(
        allocation.get("alignment"), f"allocation {allocation_id}.alignment"
    )
    if alignment & (alignment - 1):
        raise AuditError(f"allocation {allocation_id} alignment must be a power of two")
    if offset % alignment:
        raise AuditError(f"allocation {allocation_id} offset violates alignment")
    if offset + size > capacities[space]:
        raise AuditError(f"allocation {allocation_id} exceeds B200 {space} capacity")
    for field in (
        "element_type",
        "logical_shape",
        "physical_shape",
        "layout",
        "owner",
        "slot_expression",
        "phase_expression",
        "source_anchor",
        "reviewer",
    ):
        _required_string(allocation.get(field), f"allocation {allocation_id}.{field}")
    slots = _required_positive_int(
        allocation.get("slots"), f"allocation {allocation_id}.slots"
    )
    if size != slot_size * slots:
        raise AuditError(f"allocation {allocation_id} size must include every slot")
    producers = allocation.get("producer_nodes")
    consumers = allocation.get("consumer_nodes")
    for values, field in ((producers, "producer_nodes"), (consumers, "consumer_nodes")):
        if not isinstance(values, list) or any(
            not isinstance(value, int) or value not in instruction_ids
            for value in values
        ):
            raise AuditError(f"allocation {allocation_id}.{field} has invalid nodes")
    if not producers and not consumers:
        raise AuditError(f"allocation {allocation_id} is not connected to an IR node")
    lifetime = _required_object(
        allocation.get("lifetime"), f"allocation {allocation_id}.lifetime"
    )
    start = _required_non_negative_int(
        lifetime.get("start"), f"allocation {allocation_id}.lifetime.start"
    )
    end = _required_non_negative_int(
        lifetime.get("end"), f"allocation {allocation_id}.lifetime.end"
    )
    if start >= end or end > horizon:
        raise AuditError(
            f"allocation {allocation_id} lifetime must satisfy 0 <= start < end <= {horizon}"
        )
    if (
        _required_non_negative_int(
            lifetime.get("wraps_checked"),
            f"allocation {allocation_id}.lifetime.wraps_checked",
        )
        < 2
    ):
        raise AuditError(f"allocation {allocation_id} must audit at least two wraps")
    for field in ("first_write", "publication", "last_completion", "release"):
        _required_string(
            lifetime.get(field), f"allocation {allocation_id}.lifetime.{field}"
        )
    release_channel = lifetime.get("release_channel")
    if release_channel is not None:
        _required_string(
            release_channel, f"allocation {allocation_id}.lifetime.release_channel"
        )
    alias_set = allocation.get("alias_set")
    if alias_set is not None:
        _required_string(alias_set, f"allocation {allocation_id}.alias_set")
    return allocation_id, space, offset, offset + size, start, end, alias_set


def _reject_overlaps(allocations: list[tuple]) -> None:
    for index, left in enumerate(allocations):
        for right in allocations[index + 1 :]:
            if left[1] != right[1]:
                continue
            address_overlap = left[2] < right[3] and right[2] < left[3]
            lifetime_overlap = left[4] < right[5] and right[4] < left[5]
            if address_overlap and lifetime_overlap:
                raise AuditError(
                    f"allocations {left[0]!r} and {right[0]!r} overlap while live"
                )


def _alias_sets(plan: dict[str, Any], allocations: list[tuple]) -> None:
    expected: dict[str, set[str]] = {}
    for allocation in allocations:
        if allocation[6] is not None:
            expected.setdefault(allocation[6], set()).add(allocation[0])
    actual: dict[str, set[str]] = {}
    for position, raw in enumerate(_required_list(plan, "alias_sets")):
        alias = _required_object(raw, f"alias_sets[{position}]")
        alias_id = _required_string(alias.get("id"), f"alias_sets[{position}].id")
        if alias_id in actual:
            raise AuditError(f"duplicate alias set {alias_id}")
        actual[alias_id] = set(
            _required_string_list(alias.get("members"), f"alias set {alias_id}.members")
        )
        _required_string(
            alias.get("non_overlap_proof"), f"alias set {alias_id}.non_overlap_proof"
        )
        _required_string(
            alias.get("transition_sync"), f"alias set {alias_id}.transition_sync"
        )
        _required_string(alias.get("reviewer"), f"alias set {alias_id}.reviewer")
    if actual != expected:
        raise AuditError("alias_sets do not exactly describe allocation aliases")


def _memory_budget(plan: dict[str, Any], allocations: list[tuple], ir) -> None:
    budget = _required_object(plan.get("resource_budget"), "memory.resource_budget")
    planned = _required_object(budget.get("planned"), "memory.resource_budget.planned")
    expected_warps = sum(group["width"] for group in ir.payload["warp_groups"])
    fields = {
        "register_words": B200_CAPACITIES["register_words"],
        "dynamic_smem_bytes": B200_CAPACITIES["shared_bytes"],
        "barrier_smem_bytes": B200_CAPACITIES["shared_bytes"],
        "tmem_columns": B200_CAPACITIES["tmem_columns"],
        "physical_warps": B200_CAPACITIES["physical_warps"],
    }
    for field, limit in fields.items():
        value = _required_non_negative_int(planned.get(field), f"planned {field}")
        if value > limit:
            raise AuditError(f"planned {field} exceeds the B200 limit")
    if planned["physical_warps"] != expected_warps:
        raise AuditError("planned physical_warps does not match the Twill groups")
    per_warp = planned.get("per_warp_register_words")
    if not isinstance(per_warp, dict) or len(per_warp) != expected_warps:
        raise AuditError(
            "planned per_warp_register_words must cover every physical warp"
        )
    machine = ir.payload["provenance"]["machine"]
    per_warp_values = []
    for warp, words in per_warp.items():
        try:
            warp_id = int(warp)
        except (TypeError, ValueError) as error:
            raise AuditError(
                "per-warp register keys must be integer warp IDs"
            ) from error
        if (
            str(warp_id) != str(warp)
            or not 0 <= warp_id < B200_CAPACITIES["physical_warps"]
        ):
            raise AuditError("per-warp register keys must be B200 warp IDs")
        words = _required_non_negative_int(words, f"register words for warp {warp_id}")
        if words > machine["regs_per_warp"]:
            raise AuditError(f"warp {warp_id} exceeds the per-warp register limit")
        per_warp_values.append(words)
    if sum(per_warp_values) != planned["register_words"]:
        raise AuditError("per-warp register words do not sum to the planned total")
    fixed_limits = (
        (
            "register_words",
            "regs_fixed_overhead",
            B200_CAPACITIES["register_words"],
        ),
        (
            "dynamic_smem_bytes",
            "smem_fixed_overhead",
            B200_CAPACITIES["shared_bytes"],
        ),
        ("tmem_columns", "tmem_fixed_cols", B200_CAPACITIES["tmem_columns"]),
        ("physical_warps", "warp_fixed_overhead", B200_CAPACITIES["physical_warps"]),
    )
    for planned_field, overhead_field, limit in fixed_limits:
        if planned[planned_field] + machine[overhead_field] > limit:
            raise AuditError(
                f"planned {planned_field} plus fixed overhead exceeds B200"
            )
    maxima = {name: 0 for name in ("register_words", "shared_bytes", "tmem_columns")}
    for allocation in allocations:
        maxima[allocation[1]] = max(maxima[allocation[1]], allocation[3])
    if planned["register_words"] < maxima["register_words"]:
        raise AuditError("planned register_words is smaller than allocation extent")
    if planned["dynamic_smem_bytes"] < maxima["shared_bytes"]:
        raise AuditError("planned dynamic_smem_bytes is smaller than allocation extent")
    barrier_bytes = sum(
        allocation["size"]
        for allocation in plan["allocations"]
        if allocation["storage_kind"] == "barrier"
    )
    if planned["barrier_smem_bytes"] < barrier_bytes:
        raise AuditError("planned barrier_smem_bytes is smaller than barrier storage")
    if planned["tmem_columns"] < maxima["tmem_columns"]:
        raise AuditError("planned tmem_columns is smaller than allocation extent")
    reported = _required_object(
        budget.get("compiler_reported"), "memory.resource_budget.compiler_reported"
    )
    reported_registers = _required_non_negative_int(
        reported.get("register_words"), "compiler reported register_words"
    )
    reported_smem = _required_non_negative_int(
        reported.get("dynamic_smem_bytes"), "compiler reported dynamic_smem_bytes"
    )
    spill_bytes = _required_non_negative_int(
        reported.get("spill_bytes"), "compiler reported spill_bytes"
    )
    if reported_registers > B200_CAPACITIES["register_words"]:
        raise AuditError("compiler reported registers exceed the B200 limit")
    if reported_smem > B200_CAPACITIES["shared_bytes"]:
        raise AuditError("compiler reported shared memory exceeds the B200 limit")
    if spill_bytes != 0:
        raise AuditError("faithful memory plan requires zero compiler spills")
    if reported_registers != planned["register_words"]:
        raise AuditError("compiler register use disagrees with the reviewed plan")
    if reported_smem != planned["dynamic_smem_bytes"]:
        raise AuditError("compiler shared-memory use disagrees with the reviewed plan")


def _memory_descriptors(
    plan: dict[str, Any], allocations: dict[str, dict[str, Any]], ir
) -> None:
    descriptors = _required_list(plan, "descriptors")
    needs_descriptor = any(
        instruction["op_kind"] == "tt.descriptor_load"
        for instruction in ir.payload["instructions"]
    )
    if needs_descriptor and not descriptors:
        raise AuditError("TMA instructions require descriptor declarations")
    seen: set[str] = set()
    for position, raw in enumerate(descriptors):
        descriptor = _required_object(raw, f"descriptors[{position}]")
        descriptor_id = _required_string(
            descriptor.get("id"), f"descriptors[{position}].id"
        )
        if descriptor_id in seen:
            raise AuditError(f"duplicate descriptor {descriptor_id}")
        seen.add(descriptor_id)
        allocation_id = descriptor.get("allocation_id")
        if allocation_id not in allocations:
            raise AuditError(
                f"descriptor {descriptor_id} references unknown allocation"
            )
        if allocations[allocation_id]["storage_kind"] != "descriptor":
            raise AuditError(f"descriptor {descriptor_id} must use descriptor storage")
        _required_positive_int(
            descriptor.get("rank"), f"descriptor {descriptor_id}.rank"
        )
        for field in (
            "element_type",
            "global_shape",
            "global_strides",
            "box_shape",
            "swizzle",
            "interleave",
            "source_anchor",
            "reviewer",
        ):
            _required_string(
                descriptor.get(field), f"descriptor {descriptor_id}.{field}"
            )


def _memory_lifetime_audit(plan: dict[str, Any], allocation_ids: set[str]) -> None:
    seen: set[str] = set()
    for position, raw in enumerate(_required_list(plan, "lifetime_audit")):
        audit = _required_object(raw, f"lifetime_audit[{position}]")
        allocation_id = _required_string(
            audit.get("allocation_id"), f"lifetime_audit[{position}].allocation_id"
        )
        if allocation_id in seen or allocation_id not in allocation_ids:
            raise AuditError(f"invalid or duplicate lifetime audit {allocation_id}")
        seen.add(allocation_id)
        if (
            _required_non_negative_int(
                audit.get("wraps_checked"),
                f"lifetime audit {allocation_id}.wraps_checked",
            )
            < 2
        ):
            raise AuditError(f"lifetime audit {allocation_id} needs two wraps")
        for field in ("prologue", "steady_state", "epilogue", "reviewer"):
            _required_string(
                audit.get(field), f"lifetime audit {allocation_id}.{field}"
            )
    if seen != allocation_ids:
        raise AuditError("lifetime_audit must cover every allocation")


def audit_memory(
    ir_path: ArtifactSource,
    memory_plan_path: ArtifactSource,
) -> AuditResult:
    """Audit B200 capacities, semantic allocations, lifetimes, and budgets."""
    ir = load_pipelined_ir(ir_path)
    plan = load_json_artifact(memory_plan_path, "memory plan").payload
    validate_plan_header(plan, MEMORY_PLAN_SCHEMA, ir)
    _require_approved(plan, "memory")
    if plan.get("capacities") != B200_CAPACITIES:
        raise AuditError("memory capacities must be the fixed B200 capacities")
    raw_allocations = _required_list(plan, "allocations")
    if not raw_allocations:
        raise AuditError("memory plan must declare at least one allocation")
    instruction_ids = set(instruction_map(ir))
    allocations = [
        _allocation(
            raw,
            position,
            B200_CAPACITIES,
            ir.payload["pipeline"]["horizon"],
            instruction_ids,
        )
        for position, raw in enumerate(raw_allocations)
    ]
    allocation_ids = [allocation[0] for allocation in allocations]
    if len(allocation_ids) != len(set(allocation_ids)):
        raise AuditError("allocation IDs must be unique")
    _reject_overlaps(allocations)
    _alias_sets(plan, allocations)
    _memory_budget(plan, allocations, ir)
    _memory_descriptors(
        plan,
        {allocation["id"]: allocation for allocation in raw_allocations},
        ir,
    )
    _memory_lifetime_audit(plan, set(allocation_ids))
    return AuditResult(kind="memory", checked=len(allocations))


def _channel_map(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for position, raw in enumerate(_required_list(plan, "channels")):
        channel = _required_object(raw, f"channels[{position}]")
        channel_id = _required_string(channel.get("id"), f"channels[{position}].id")
        if channel_id in result:
            raise AuditError(f"duplicate synchronization channel {channel_id}")
        for field in (
            "barrier_id",
            "allocation_id",
            "payload_allocation",
            "slot_expression",
            "producer_witness",
            "phase_recurrence",
            "prologue",
            "steady_state",
            "epilogue",
            "predicate_behavior",
            "specification",
            "reviewer",
        ):
            _required_string(channel.get(field), f"channel {channel_id}.{field}")
        if channel.get("scope") not in {"cta", "cluster"}:
            raise AuditError(f"channel {channel_id} has invalid scope")
        _required_positive_int(
            channel.get("slot_count"), f"channel {channel_id}.slot_count"
        )
        for field in (
            "producer_groups",
            "consumer_groups",
            "producer_warps",
            "consumer_warps",
        ):
            values = channel.get(field)
            if (
                not isinstance(values, list)
                or not values
                or any(
                    not isinstance(value, int) or isinstance(value, bool) or value < 0
                    for value in values
                )
            ):
                raise AuditError(
                    f"channel {channel_id}.{field} has invalid participants"
                )
        initialization = _required_object(
            channel.get("initialization"), f"channel {channel_id}.initialization"
        )
        _required_string(
            initialization.get("actor"), f"channel {channel_id}.initialization.actor"
        )
        _required_positive_int(
            initialization.get("arrival_count"),
            f"channel {channel_id}.initialization.arrival_count",
        )
        arrival_actors = _required_string_list(
            initialization.get("arrival_actors"),
            f"channel {channel_id}.initialization.arrival_actors",
        )
        if initialization["arrival_count"] != len(arrival_actors):
            raise AuditError(
                f"channel {channel_id} arrival_count does not match actors"
            )
        _required_non_negative_int(
            initialization.get("transaction_bytes"),
            f"channel {channel_id}.initialization.transaction_bytes",
        )
        wait = _required_object(
            channel.get("consumer_wait"), f"channel {channel_id}.consumer_wait"
        )
        release = _required_object(
            channel.get("release"), f"channel {channel_id}.release"
        )
        for owner, record in (("consumer_wait", wait), ("release", release)):
            for field in ("primitive", "phase_expression", "predicate"):
                _required_string(
                    record.get(field), f"channel {channel_id}.{owner}.{field}"
                )
        initial_phase = _required_non_negative_int(
            channel.get("initial_phase"), f"channel {channel_id}.initial_phase"
        )
        if initial_phase not in (0, 1):
            raise AuditError(f"channel {channel_id}.initial_phase must be 0 or 1")
        result[channel_id] = channel
    return result


def _sync_dependencies(ir, plan: dict[str, Any], channels: dict[str, dict[str, Any]]):
    expected = dependency_map(ir)
    cross = cross_dependency_map(ir)
    seen: set[int] = set()
    used_channels: set[str] = set()
    facts = (
        "src",
        "dst",
        "distance",
        "latency",
        "producer_group",
        "consumer_group",
    )
    wait_edges: list[dict[str, Any]] = []
    for position, raw in enumerate(_required_list(plan, "dependencies")):
        binding = _required_object(raw, f"dependencies[{position}]")
        edge_index = _required_non_negative_int(
            binding.get("edge_index"), f"dependencies[{position}].edge_index"
        )
        if edge_index in seen or edge_index not in expected:
            raise AuditError(f"invalid or duplicate sync dependency {edge_index}")
        seen.add(edge_index)
        for field in facts:
            if binding.get(field) != expected[edge_index][field]:
                raise AuditError(f"sync dependency {edge_index} {field} mismatch")
        mechanism = binding.get("mechanism")
        if mechanism not in {"program_order", "channel"}:
            raise AuditError(f"sync dependency {edge_index} has invalid mechanism")
        channel = binding.get("channel")
        if mechanism == "program_order":
            if edge_index in cross:
                raise AuditError(f"cross-warp dependency {edge_index} needs a channel")
            if channel is not None:
                raise AuditError(
                    f"program-order dependency {edge_index} names a channel"
                )
            source_kind = instruction_map(ir)[expected[edge_index]["src"]]["op_kind"]
            if async_obligation_kind(source_kind) is not None:
                raise AuditError(
                    f"async dependency {edge_index} cannot rely on program order"
                )
            _required_string(
                binding.get("program_order_proof"),
                f"sync dependency {edge_index}.program_order_proof",
            )
        else:
            channel = _required_string(channel, f"sync dependency {edge_index}.channel")
            if channel not in channels:
                raise AuditError(
                    f"sync dependency {edge_index} references unknown channel"
                )
            used_channels.add(channel)
        wait_edges.append(
            {
                "src": binding["src"],
                "dst": binding["dst"],
                "distance": binding["distance"],
            }
        )
        for field in (
            "completion_witness",
            "consumer_wait",
            "predicate",
            "prologue",
            "epilogue",
            "reviewer",
        ):
            _required_string(
                binding.get(field), f"sync dependency {edge_index}.{field}"
            )
    if seen != set(expected):
        raise AuditError("sync dependencies must cover every IR edge")
    return used_channels, wait_edges


def _expected_async_obligations(ir) -> dict[str, tuple[int, str]]:
    return {
        f"node-{instruction['id']}-{kind}": (instruction["id"], kind)
        for instruction in ir.payload["instructions"]
        if (kind := async_obligation_kind(instruction["op_kind"])) is not None
    }


def _sync_obligations(ir, plan: dict[str, Any], channels: dict[str, dict[str, Any]]):
    expected = _expected_async_obligations(ir)
    seen: set[str] = set()
    used_channels: set[str] = set()
    wait_edges: list[dict[str, Any]] = []
    instruction_ids = set(instruction_map(ir))
    for position, raw in enumerate(_required_list(plan, "additional_obligations")):
        obligation = _required_object(raw, f"additional_obligations[{position}]")
        obligation_id = _required_string(
            obligation.get("id"), f"additional_obligations[{position}].id"
        )
        if obligation_id in seen or obligation_id not in expected:
            raise AuditError(f"invalid or duplicate async obligation {obligation_id}")
        seen.add(obligation_id)
        node_id, kind = expected[obligation_id]
        if (
            obligation.get("instruction_id") != node_id
            or obligation.get("kind") != kind
        ):
            raise AuditError(f"async obligation {obligation_id} does not match the IR")
        channel = _required_string(
            obligation.get("channel"), f"async obligation {obligation_id}.channel"
        )
        if channel not in channels:
            raise AuditError(
                f"async obligation {obligation_id} references unknown channel"
            )
        used_channels.add(channel)
        wait_kind = obligation.get("wait_kind")
        if wait_kind == "kernel_exit":
            if kind != "tma_store_completion":
                raise AuditError(
                    f"async obligation {obligation_id} cannot wait at kernel exit"
                )
            if obligation.get("wait_node") is not None:
                raise AuditError(
                    f"kernel-exit obligation {obligation_id} must not name a node"
                )
            _required_string(
                obligation.get("kernel_exit_wait"),
                f"async obligation {obligation_id}.kernel_exit_wait",
            )
        elif wait_kind == "instruction":
            if obligation.get("kernel_exit_wait") is not None:
                raise AuditError(
                    f"instruction obligation {obligation_id} names a kernel-exit wait"
                )
            wait_node = _required_non_negative_int(
                obligation.get("wait_node"),
                f"async obligation {obligation_id}.wait_node",
            )
            if wait_node not in instruction_ids:
                raise AuditError(
                    f"async obligation {obligation_id} has unknown wait node"
                )
            distance = _required_non_negative_int(
                obligation.get("distance"),
                f"async obligation {obligation_id}.distance",
            )
            matching_edges = [
                edge
                for edge in ir.payload["dependencies"]
                if edge["src"] == node_id
                and edge["dst"] == wait_node
                and edge["distance"] == distance
            ]
            if not matching_edges:
                raise AuditError(
                    f"async obligation {obligation_id} waiter is not an IR consumer"
                )
            wait_edges.append({"src": node_id, "dst": wait_node, "distance": distance})
        else:
            raise AuditError(f"async obligation {obligation_id} has invalid wait_kind")
        for field in ("completion_witness", "predicate", "epilogue_drain", "reviewer"):
            _required_string(
                obligation.get(field), f"async obligation {obligation_id}.{field}"
            )
    if seen != set(expected):
        raise AuditError("additional_obligations do not cover every async operation")
    return used_channels, wait_edges


def _ordering_edges(ir, plan: dict[str, Any], channels: dict[str, dict[str, Any]]):
    instructions = instruction_map(ir)
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for position, raw in enumerate(_required_list(plan, "ordering_edges")):
        edge = _required_object(raw, f"ordering_edges[{position}]")
        edge_id = _required_string(edge.get("id"), f"ordering_edges[{position}].id")
        if edge_id in seen:
            raise AuditError(f"duplicate ordering edge {edge_id}")
        seen.add(edge_id)
        _required_string(
            edge.get("allocation_id"), f"ordering edge {edge_id}.allocation_id"
        )
        src = _required_non_negative_int(
            edge.get("src"), f"ordering edge {edge_id}.src"
        )
        dst = _required_non_negative_int(
            edge.get("dst"), f"ordering edge {edge_id}.dst"
        )
        if src not in instructions or dst not in instructions:
            raise AuditError(f"ordering edge {edge_id} references an unknown node")
        _required_non_negative_int(
            edge.get("distance"), f"ordering edge {edge_id}.distance"
        )
        channel = _required_string(
            edge.get("channel"), f"ordering edge {edge_id}.channel"
        )
        if channel not in channels:
            raise AuditError(f"ordering edge {edge_id} references unknown channel")
        _required_string(edge.get("reason"), f"ordering edge {edge_id}.reason")
        _required_string(edge.get("reviewer"), f"ordering edge {edge_id}.reviewer")
        result.append(edge)
    return result


def _phase_traces(plan: dict[str, Any], channels: dict[str, dict[str, Any]]) -> None:
    seen: set[str] = set()
    for position, raw in enumerate(_required_list(plan, "phase_traces")):
        trace = _required_object(raw, f"phase_traces[{position}]")
        channel_id = _required_string(
            trace.get("channel"), f"phase_traces[{position}].channel"
        )
        if channel_id in seen or channel_id not in channels:
            raise AuditError(f"invalid or duplicate phase trace {channel_id}")
        seen.add(channel_id)
        channel = channels[channel_id]
        if trace.get("initial_phase") != channel["initial_phase"]:
            raise AuditError(f"phase trace {channel_id} initial phase mismatch")
        if (
            _required_non_negative_int(
                trace.get("wraps_checked"), f"phase trace {channel_id}.wraps_checked"
            )
            < 2
        ):
            raise AuditError(f"phase trace {channel_id} must cover two wraps")
        regions = _required_string_list(
            trace.get("regions"), f"phase trace {channel_id}.regions"
        )
        if set(regions) != {"prologue", "steady_state", "epilogue"}:
            raise AuditError(
                f"phase trace {channel_id} must cover every pipeline region"
            )
        events = trace.get("events")
        minimum = 2 * channel["slot_count"]
        if not isinstance(events, list) or len(events) < minimum:
            raise AuditError(
                f"phase trace {channel_id} needs at least {minimum} events"
            )
        iterations: set[int] = set()
        for event_position, raw_event in enumerate(events):
            event = _required_object(
                raw_event, f"phase trace {channel_id} event {event_position}"
            )
            iteration = _required_non_negative_int(
                event.get("iteration"), f"phase trace {channel_id} event iteration"
            )
            if iteration in iterations:
                raise AuditError(
                    f"phase trace {channel_id} repeats iteration {iteration}"
                )
            iterations.add(iteration)
            expected_slot = iteration % channel["slot_count"]
            expected_phase = (
                channel["initial_phase"] + iteration // channel["slot_count"]
            ) % 2
            if (
                event.get("slot") != expected_slot
                or event.get("phase") != expected_phase
            ):
                raise AuditError(
                    f"phase trace {channel_id} slot/phase recurrence mismatch"
                )
            for field in ("producer", "completion", "wait", "release"):
                _required_string(
                    event.get(field), f"phase trace {channel_id} event.{field}"
                )
        if not set(range(minimum)).issubset(iterations):
            raise AuditError(
                f"phase trace {channel_id} does not cover two consecutive wraps"
            )
        for field in ("zero_trip", "partial_iteration", "epilogue_drain", "reviewer"):
            _required_string(trace.get(field), f"phase trace {channel_id}.{field}")
    if seen != set(channels):
        raise AuditError("phase_traces must cover every synchronization channel")


def _expanded_edges(ir, wait_edges, iterations):
    instructions = instruction_map(ir)
    nodes = {
        (node_id, phase) for node_id in instructions for phase in range(iterations)
    }
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for dependency in wait_edges:
        distance = dependency["distance"]
        for phase in range(iterations):
            consumer_phase = phase + distance
            if consumer_phase < iterations:
                edges.append(
                    ((dependency["src"], phase), (dependency["dst"], consumer_phase))
                )
    for group in ir.payload["warp_groups"]:
        trace = [entry["node"] for entry in group["issue_trace"]]
        for phase in range(iterations):
            edges.extend(
                ((src, phase), (dst, phase)) for src, dst in zip(trace, trace[1:])
            )
            if trace and phase + 1 < iterations:
                edges.append(((trace[-1], phase), (trace[0], phase + 1)))
    return nodes, edges


def _reject_cycle(ir, wait_edges, iterations) -> None:
    nodes, edges = _expanded_edges(ir, wait_edges, iterations)
    successors = {node: [] for node in nodes}
    indegree = {node: 0 for node in nodes}
    for src, dst in edges:
        successors[src].append(dst)
        indegree[dst] += 1
    ready = sorted(node for node, count in indegree.items() if count == 0)
    visited = 0
    while ready:
        node = ready.pop()
        visited += 1
        for successor in successors[node]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
    if visited != len(nodes):
        cyclic = sorted(node for node, count in indegree.items() if count > 0)
        raise AuditError(f"phase-expanded synchronization graph has a cycle: {cyclic}")


def audit_sync(
    ir_path: ArtifactSource,
    sync_plan_path: ArtifactSource,
) -> AuditResult:
    """Audit dependency coverage, completion witnesses, phases, and deadlocks."""
    ir = load_pipelined_ir(ir_path)
    plan = load_json_artifact(sync_plan_path, "synchronization manifest").payload
    validate_plan_header(plan, SYNC_PLAN_SCHEMA, ir)
    _require_approved(plan, "synchronization")
    channels = _channel_map(plan)
    used_channels, dependency_wait_edges = _sync_dependencies(ir, plan, channels)
    obligation_channels, obligation_wait_edges = _sync_obligations(ir, plan, channels)
    used_channels.update(obligation_channels)
    ordering_edges = _ordering_edges(ir, plan, channels)
    used_channels.update(edge["channel"] for edge in ordering_edges)
    if set(channels) != used_channels:
        raise AuditError("every synchronization channel must discharge an obligation")
    _phase_traces(plan, channels)
    wait_edges = dependency_wait_edges + obligation_wait_edges + ordering_edges
    maximum_distance = max(
        (edge["distance"] for edge in wait_edges),
        default=0,
    )
    maximum_slots = max(
        (channel["slot_count"] for channel in channels.values()),
        default=1,
    )
    iterations = max(
        ir.payload["pipeline"]["copies"],
        2 * maximum_slots + maximum_distance + 1,
    )
    _reject_cycle(
        ir,
        wait_edges,
        iterations,
    )
    coverage = _required_object(plan.get("coverage"), "synchronization.coverage")
    expected_async = len(_expected_async_obligations(ir))
    expected_coverage = {
        "ir_edges": len(ir.payload["dependencies"]),
        "mapped_ir_edges": len(ir.payload["dependencies"]),
        "cross_warp_edges": len(ir.payload["cross_warp_dependencies"]),
        "mapped_cross_warp_edges": len(ir.payload["cross_warp_dependencies"]),
        "async_completion_obligations": expected_async,
        "mapped_async_completion_obligations": expected_async,
        "reuse_obligations": len(ordering_edges),
        "mapped_reuse_obligations": len(ordering_edges),
    }
    if coverage != expected_coverage:
        raise AuditError("synchronization coverage is not complete")
    return AuditResult(
        kind="synchronization",
        checked=len(ir.payload["dependencies"]) + expected_async + len(ordering_edges),
    )


def _authoring_manifest_reference(authoring, entry: object, name: str):
    reference = _required_object(entry, f"authoring.manifests.{name}")
    path = _artifact_relative_path(
        authoring,
        _required_string(reference.get("path"), f"authoring.manifests.{name}.path"),
        f"{name} manifest",
    )
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise AuditError(f"cannot read {name} manifest: {error}") from error
    if digest != _required_sha256(
        reference.get("sha256"), f"authoring.manifests.{name}.sha256"
    ):
        raise AuditError(f"{name} manifest sha256 mismatch")
    return path, digest


def audit_authoring(
    ir_path: ArtifactSource,
    handoff_path: ArtifactSource,
    authoring_path: ArtifactSource,
) -> AuditResult:
    """Audit provenance, clean-room declaration, reviews, and manifest hashes."""
    ir = load_pipelined_ir(ir_path)
    handoff = load_handoff(handoff_path, ir)
    authoring = load_json_artifact(authoring_path, "manual CUDA authoring record")
    plan = authoring.payload
    validate_plan_header(plan, AUTHORING_SCHEMA, ir)
    if plan.get("status") != "approved":
        raise AuditError("authoring status must be approved")
    handoff_ref = _required_object(plan.get("handoff"), "authoring.handoff")
    if handoff_ref.get("schema_version") != HANDOFF_SCHEMA:
        raise AuditError("authoring record references the wrong handoff schema")
    if handoff_ref.get("sha256") != handoff.sha256:
        raise AuditError("authoring handoff sha256 mismatch")
    if plan.get("solution_sha256") != ir.payload["solution_sha256"]:
        raise AuditError("authoring solution sha256 mismatch")
    if plan.get("source_program_sha256") != ir.payload["source_program"]["sha256"]:
        raise AuditError("authoring source-program sha256 mismatch")
    if plan.get("solver_provenance") != ir.payload["provenance"]:
        raise AuditError("authoring solver provenance mismatch")
    clean_room = _required_object(plan.get("clean_room"), "authoring.clean_room")
    if clean_room.get("fa4_fa3_use") != "black-box-baselines-only":
        raise AuditError("FA4/FA3 use must be black-box-baselines-only")
    if clean_room.get("prohibited_reuse") != [
        "warp_roles",
        "mma_order",
        "barrier_protocol",
    ]:
        raise AuditError("clean-room prohibited-reuse declaration is incomplete")
    _required_string(
        clean_room.get("declaration_reviewed_by"),
        "clean_room.declaration_reviewed_by",
    )
    wrappers = clean_room.get("reviewed_leaf_wrappers")
    if not isinstance(wrappers, list):
        raise AuditError("reviewed_leaf_wrappers must be a list")
    for position, raw in enumerate(wrappers):
        wrapper = _required_object(raw, f"reviewed_leaf_wrappers[{position}]")
        for field in ("file", "symbol", "revision", "reviewer", "primitive", "reason"):
            _required_string(
                wrapper.get(field), f"reviewed leaf wrapper {position}.{field}"
            )
    manifests = _required_object(plan.get("manifests"), "authoring.manifests")
    digests = {
        name: _authoring_manifest_reference(authoring, manifests.get(name), name)[1]
        for name in ("mapping", "memory", "synchronization")
    }
    reviews = _required_list(plan, "human_reviews")
    reviewed: dict[str, str] = {}
    for position, raw in enumerate(reviews):
        review = _required_object(raw, f"human_reviews[{position}]")
        kind = _required_string(review.get("kind"), f"human_reviews[{position}].kind")
        if kind in reviewed or kind not in {
            "mapping",
            "memory",
            "synchronization",
            "clean_room",
        }:
            raise AuditError(f"invalid or duplicate human review {kind}")
        _required_string(review.get("reviewer"), f"human review {kind}.reviewer")
        if review.get("status") != "approved":
            raise AuditError(f"human review {kind} must be approved")
        artifact_sha256 = _required_sha256(
            review.get("artifact_sha256"), f"human review {kind}.artifact_sha256"
        )
        reviewed[kind] = artifact_sha256
    if set(reviewed) != {"mapping", "memory", "synchronization", "clean_room"}:
        raise AuditError(
            "human reviews must cover mapping, memory, sync, and clean room"
        )
    for kind in ("mapping", "memory", "synchronization"):
        if reviewed[kind] != digests[kind]:
            raise AuditError(f"human review {kind} hash mismatch")
    clean_room_sha256 = hashlib.sha256(
        json.dumps(
            clean_room,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    if reviewed["clean_room"] != clean_room_sha256:
        raise AuditError("human review clean_room hash mismatch")
    return AuditResult(kind="authoring", checked=4 + len(wrappers))


def audit_bundle(
    ir_path: ArtifactSource,
    handoff_path: ArtifactSource,
    authoring_path: ArtifactSource,
    mapping_path: ArtifactSource,
    memory_plan_path: ArtifactSource,
    sync_plan_path: ArtifactSource,
) -> AuditResult:
    """Run all audits and check mapping/memory/synchronization cross-references."""
    ir = load_pipelined_ir(ir_path)
    audit_authoring(ir_path, handoff_path, authoring_path)
    authoring_artifact = load_json_artifact(
        authoring_path, "manual CUDA authoring record"
    )
    explicit_artifacts = {
        "mapping": load_json_artifact(mapping_path, "mapping manifest"),
        "memory": load_json_artifact(memory_plan_path, "memory plan"),
        "synchronization": load_json_artifact(
            sync_plan_path, "synchronization manifest"
        ),
    }
    signed_manifests = authoring_artifact.payload["manifests"]
    for kind, artifact in explicit_artifacts.items():
        signed = signed_manifests[kind]
        signed_path = _artifact_relative_path(
            authoring_artifact,
            signed["path"],
            f"signed {kind} manifest",
        )
        if artifact.path is None or artifact.path.resolve() != signed_path:
            raise AuditError(f"bundle {kind} manifest is not the authoring-signed file")
        if artifact.sha256 != signed["sha256"]:
            raise AuditError(f"bundle {kind} manifest hash is not authoring-signed")
    mapping_result = audit_mapping(ir_path, mapping_path)
    memory_result = audit_memory(ir_path, memory_plan_path)
    sync_result = audit_sync(ir_path, sync_plan_path)

    mapping = explicit_artifacts["mapping"].payload
    memory = explicit_artifacts["memory"].payload
    synchronization = explicit_artifacts["synchronization"].payload
    allocations = {allocation["id"]: allocation for allocation in memory["allocations"]}
    allocation_ids = set(allocations)
    channel_ids = {channel["id"] for channel in synchronization["channels"]}
    physical_groups = {
        group["group"]: group["physical_warps"] for group in mapping["physical_groups"]
    }
    mapped_allocations, mapped_channels = _mapping_nodes(
        ir,
        mapping,
        _source_text(explicit_artifacts["mapping"], mapping),
        _build_evidence(explicit_artifacts["mapping"], mapping),
        physical_groups,
    )
    mapped_channels.update(_mapping_dependencies(ir, mapping))
    if not mapped_allocations.issubset(allocation_ids):
        raise AuditError("mapping references unknown memory allocations")
    if not mapped_channels.issubset(channel_ids):
        raise AuditError("mapping references unknown synchronization channels")
    memory_channels = {
        allocation["lifetime"]["release_channel"]
        for allocation in memory["allocations"]
        if allocation["lifetime"].get("release_channel") is not None
    }
    if not memory_channels.issubset(channel_ids):
        raise AuditError("memory plan references unknown release channels")
    mapping_nodes = {node["instruction_id"]: node for node in mapping["nodes"]}
    for node_id, node in mapping_nodes.items():
        manual = node["manual_cuda"]
        for allocation_id in manual["operand_allocations"]:
            if node_id not in allocations[allocation_id]["consumer_nodes"]:
                raise AuditError(
                    f"instruction {node_id} operand ownership disagrees with memory plan"
                )
        for allocation_id in manual["result_allocations"]:
            if node_id not in allocations[allocation_id]["producer_nodes"]:
                raise AuditError(
                    f"instruction {node_id} result ownership disagrees with memory plan"
                )
        if node["op_kind"] in {"tt.descriptor_load", "tt.descriptor_store"}:
            if not any(
                allocations[allocation_id]["storage_kind"] == "descriptor"
                for allocation_id in manual["operand_allocations"]
            ):
                raise AuditError(
                    f"TMA instruction {node_id} lacks a descriptor allocation"
                )
    mapped_warps = {warp for warps in physical_groups.values() for warp in warps}
    planned_warps = {
        int(warp)
        for warp in memory["resource_budget"]["planned"]["per_warp_register_words"]
    }
    if planned_warps != mapped_warps:
        raise AuditError("per-warp register plan disagrees with physical mapping")
    ordering_edges = synchronization["ordering_edges"]
    for allocation in memory["allocations"]:
        release_channel = allocation["lifetime"].get("release_channel")
        requires_release = (
            allocation["storage_kind"] in {"shared", "tmem", "barrier"}
            and allocation["producer_nodes"]
            and allocation["consumer_nodes"]
        )
        if requires_release and release_channel is None:
            raise AuditError(
                f"mutable allocation {allocation['id']} lacks a release channel"
            )
        if release_channel is None:
            continue
        reuse_edges = [
            edge
            for edge in ordering_edges
            if edge["allocation_id"] == allocation["id"]
            and edge["channel"] == release_channel
            and edge["src"] in allocation["consumer_nodes"]
            and edge["dst"] in allocation["producer_nodes"]
            and edge["distance"] == allocation["slots"]
        ]
        if not reuse_edges:
            raise AuditError(
                f"allocation {allocation['id']} lacks a derived reuse edge"
            )
    for channel in synchronization["channels"]:
        if channel["allocation_id"] not in allocation_ids:
            raise AuditError("synchronization channel references unknown storage")
        if allocations[channel["allocation_id"]]["storage_kind"] != "barrier":
            raise AuditError("synchronization channel must use barrier storage")
        if allocations[channel["allocation_id"]]["slot_size"] != 16:
            raise AuditError("B200 barrier slots must reserve a 16-byte pair")
        if channel["payload_allocation"] not in allocation_ids:
            raise AuditError("synchronization channel references unknown payload")
        if allocations[channel["allocation_id"]]["slots"] != channel["slot_count"]:
            raise AuditError("channel slot count disagrees with barrier storage")
        if allocations[channel["payload_allocation"]]["slots"] != channel["slot_count"]:
            raise AuditError("channel slot count disagrees with payload storage")
        for role in ("producer", "consumer"):
            groups = channel[f"{role}_groups"]
            if any(group not in physical_groups for group in groups):
                raise AuditError(
                    f"synchronization channel references unknown {role} group"
                )
            allowed_warps = {
                warp for group in groups for warp in physical_groups[group]
            }
            role_warps = channel[f"{role}_warps"]
            if len(role_warps) != len(set(role_warps)) or not set(role_warps).issubset(
                allowed_warps
            ):
                raise AuditError(
                    f"synchronization channel {role} warps disagree with mapping"
                )
    for alias_set in memory["alias_sets"]:
        if alias_set["transition_sync"] not in channel_ids:
            raise AuditError("alias-set transition_sync references unknown channel")
    channels = {channel["id"]: channel for channel in synchronization["channels"]}
    for edge in ordering_edges:
        channel = channels[edge["channel"]]
        release_mapping = mapping_nodes[edge["src"]]["manual_cuda"]
        producer_mapping = mapping_nodes[edge["dst"]]["manual_cuda"]
        if not set(release_mapping["physical_warps"]).issubset(
            channel["consumer_warps"]
        ):
            raise AuditError("release edge omits consumer lanes")
        if not set(producer_mapping["physical_warps"]).issubset(
            channel["producer_warps"]
        ):
            raise AuditError("reuse edge omits producer lanes")
    mapping_edges = {edge["index"]: edge for edge in mapping["dependencies"]}
    for binding in synchronization["dependencies"]:
        mapped_channel = mapping_edges[binding["edge_index"]]["sync_channel"]
        if binding["mechanism"] == "channel" and mapped_channel != binding["channel"]:
            raise AuditError("mapping and synchronization dependency channels disagree")
        if binding["mechanism"] == "program_order" and mapped_channel is not None:
            raise AuditError("mapping and synchronization mechanisms disagree")
        if binding["mechanism"] == "channel":
            channel = next(
                item
                for item in synchronization["channels"]
                if item["id"] == binding["channel"]
            )
            if binding["producer_group"] not in channel["producer_groups"]:
                raise AuditError("channel producer group disagrees with dependency")
            if binding["consumer_group"] not in channel["consumer_groups"]:
                raise AuditError("channel consumer group disagrees with dependency")
            producer = mapping_nodes[binding["src"]]["manual_cuda"]
            consumer = mapping_nodes[binding["dst"]]["manual_cuda"]
            if not set(producer["physical_warps"]).issubset(channel["producer_warps"]):
                raise AuditError("channel omits dependency producer lanes")
            if not set(consumer["physical_warps"]).issubset(channel["consumer_warps"]):
                raise AuditError("channel omits dependency consumer lanes")
    instructions = instruction_map(ir)
    for obligation in synchronization["additional_obligations"]:
        channel = channels[obligation["channel"]]
        producer_group = instructions[obligation["instruction_id"]]["group"]
        if producer_group not in channel["producer_groups"]:
            raise AuditError("async obligation producer disagrees with channel")
        producer_mapping = mapping_nodes[obligation["instruction_id"]]["manual_cuda"]
        if obligation["channel"] not in producer_mapping["sync_channels"]:
            raise AuditError("async producer mapping omits its channel")
        if not set(producer_mapping["physical_warps"]).issubset(
            channel["producer_warps"]
        ):
            raise AuditError("channel omits async producer lanes")
        if obligation["wait_kind"] == "instruction":
            consumer_group = instructions[obligation["wait_node"]]["group"]
            if consumer_group not in channel["consumer_groups"]:
                raise AuditError("async obligation waiter disagrees with channel")
            waiter_mapping = mapping_nodes[obligation["wait_node"]]["manual_cuda"]
            if obligation["channel"] not in waiter_mapping["sync_channels"]:
                raise AuditError("async waiter mapping omits its channel")
            if not set(waiter_mapping["physical_warps"]).issubset(
                channel["consumer_warps"]
            ):
                raise AuditError("channel omits async waiter lanes")
        if obligation["kind"].startswith("tma_"):
            if channel["initialization"]["transaction_bytes"] <= 0:
                raise AuditError("TMA channel must account for transaction bytes")
    checked = (
        mapping_result.checked
        + memory_result.checked
        + sync_result.checked
        + len(mapped_allocations)
        + len(mapped_channels)
    )
    return AuditResult(kind="bundle", checked=checked)
