"""Textual TTGIR normalization and PlanBundle extraction.

The parser intentionally consumes final textual TTGIR rather than compiler C++
objects. That makes M1.1--M1.3 usable on archived artifacts and keeps the first
implementation independent of the existing modulo-scheduler implementation.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from .model import BaselineManifest, PlanBundle, PlanError


_SSA = re.compile(r"%[A-Za-z0-9_.$-]+(?:#\d+)?")
_RESULT = re.compile(r"^\s*(?P<lhs>%[A-Za-z0-9_.$-]+(?::\d+)?)\s*=\s*(?P<rhs>.*)$")
_OP_NAME = re.compile(r"(?:\"([^\"]+)\"|([A-Za-z_][A-Za-z0-9_.]*))")
_LOC_USE = re.compile(r"\s+loc\(#loc[A-Za-z0-9_.$-]*\)")
_LOC_DEF = re.compile(r"^\s*#loc\d*\s*=\s*loc\(.*$", re.MULTILINE)
_LAYOUT = re.compile(r"^\s*(#[A-Za-z0-9_.$-]+)\s*=\s*(#ttg\..+)$")
_FUNC = re.compile(r"tt\.func\s+(?:public\s+)?@([A-Za-z0-9_.$-]+)")


@dataclass
class ParsedOp:
    ordinal: int
    result: str | None
    kind: str
    operands: list[str]
    text: str
    scope: str


def _without_locations(text: str) -> str:
    text = _LOC_DEF.sub("", text)
    text = _LOC_USE.sub("", text)
    return text


def normalize_ttgir(text: str) -> str:
    """Remove debug locations and deterministically alpha-rename SSA values."""
    text = _without_locations(text)
    names: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        value = match.group(0)
        if value not in names:
            names[value] = f"%v{len(names)}"
        return names[value]

    lines: list[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line.strip())
        if not line:
            continue
        lines.append(_SSA.sub(replace, line))
    return "\n".join(lines) + "\n"


def _op_name(rhs: str) -> str | None:
    match = _OP_NAME.match(rhs.lstrip())
    if not match:
        return None
    return match.group(1) or match.group(2)


def _result_names(lhs: str) -> list[str]:
    if ":" not in lhs:
        return [lhs]
    base, count_text = lhs.rsplit(":", 1)
    if not count_text.isdigit():
        return [lhs]
    return [f"{base}#{index}" for index in range(int(count_text))]


def _parse_ops(text: str) -> tuple[list[ParsedOp], dict[str, ParsedOp]]:
    ops: list[ParsedOp] = []
    definitions: dict[str, ParsedOp] = {}
    loop_scopes: list[tuple[int, str]] = []
    for line in _without_locations(text).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        indent = len(line) - len(line.lstrip())
        if stripped.startswith("}"):
            while loop_scopes and indent <= loop_scopes[-1][0]:
                loop_scopes.pop()
            continue
        if stripped.startswith("^bb"):
            continue
        result_match = _RESULT.match(line)
        result = result_match.group("lhs") if result_match else None
        rhs = result_match.group("rhs") if result_match else stripped
        kind = _op_name(rhs)
        if kind is None or kind in {"module", "tt.func"}:
            continue
        if "." not in kind and kind not in {"scf.for", "scf.if", "scf.yield", "return"}:
            continue
        operand_region = rhs.split(" : ", 1)[0]
        operands = _SSA.findall(operand_region)
        op = ParsedOp(
            ordinal=len(ops),
            result=result,
            kind=kind,
            operands=operands,
            text=re.sub(r"\s+", " ", stripped),
            scope=loop_scopes[-1][1] if loop_scopes else "function",
        )
        ops.append(op)
        if result:
            for name in _result_names(result):
                definitions[name] = op
        if kind == "scf.for":
            loop_scopes.append((indent, f"loop:{result or op.ordinal}"))
    return ops, definitions


def _layout_aliases(text: str) -> dict[str, str]:
    layouts: dict[str, str] = {}
    for line in text.splitlines():
        match = _LAYOUT.match(line)
        if match:
            layouts[match.group(1)] = re.sub(r"\s+", " ", match.group(2).strip())
    return layouts


def _type_section(text: str) -> str:
    return text.split(" : ", 1)[1] if " : " in text else ""


def _result_layout(text: str) -> str | None:
    result_type = _type_section(text).split(" -> ")[-1]
    aliases = re.findall(r"#[A-Za-z0-9_.$-]+", result_type)
    return aliases[-1] if aliases else None


def _tensor_shapes(text: str) -> list[list[int]]:
    values: list[list[int]] = []
    for shape in re.findall(r"tensor<([0-9x]+)x(?:bf16|f16|f32|i\d+)", text):
        values.append([int(value) for value in shape.split("x")])
    return values


def _attributes(text: str) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    for key in ("resident", "accumulator", "register_class"):
        match = re.search(rf'{key}\s+"([^"]+)"', text)
        if match:
            attributes[key] = match.group(1)
    initialize = re.search(r"initialize\s+(true|false)", text)
    if initialize:
        attributes["initialize"] = initialize.group(1) == "true"
    return attributes


def _trace_text(value: str, definitions: dict[str, ParsedOp], depth: int = 24) -> str:
    seen: set[str] = set()
    work = [value]
    fragments: list[str] = []
    while work and len(seen) < depth:
        current = work.pop()
        if current in seen:
            continue
        seen.add(current)
        op = definitions.get(current)
        if op is None:
            fragments.append(current)
            continue
        fragments.append(op.text)
        work.extend(op.operands)
    return " ".join(fragments).lower()


def _dot_role(op: ParsedOp, definitions: dict[str, ParsedOp]) -> str:
    attributes = _attributes(op.text)
    if attributes.get("accumulator") == "transient" and attributes.get("resident") == "rhs":
        return "dq"
    result = (op.result or "").lower()
    if "dq" in result:
        return "dq"
    if "score" in result:
        return "qk"
    if result.startswith("%dp") or "_dp" in result:
        return "dp"
    accumulator = op.operands[2] if len(op.operands) >= 3 else ""
    trace = _trace_text(accumulator, definitions)
    if re.search(r"(?:^|[^a-z])dv(?:_|[^a-z]|$)", trace):
        return "dv"
    if re.search(r"(?:^|[^a-z])dk(?:_|[^a-z]|$)", trace):
        return "dk"
    return "unknown"


def _slice_offsets(value: str, definitions: dict[str, ParsedOp]) -> list[int] | None:
    """Follow an accumulator chain to its statically sliced output tile."""
    seen: set[str] = set()
    current = value
    while current not in seen and len(seen) < 32:
        seen.add(current)
        op = definitions.get(current)
        if op is None:
            return None
        if "extract_slice" in op.kind:
            match = re.search(r"(?:offsets\s*=\s*)?\[([^]]+)\]\s*(?:sizes|strides|:)", op.text)
            if not match:
                return None
            values = [item.strip() for item in match.group(1).split(",")]
            return [int(item) for item in values] if all(item.lstrip("-").isdigit() for item in values) else None
        if op.kind in {"amdg.scheduled_mfma", "tt.dot"} and len(op.operands) >= 3:
            current = op.operands[2]
            continue
        return None
    return None


def _stable_id(prefix: str, signature: str, occurrence: int = 0) -> str:
    suffix = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}/{suffix}/{occurrence}"


def _operation_signature(op: ParsedOp) -> str:
    value = _SSA.sub("%ssa", op.text)
    return f"{op.scope}|{op.kind}|{value}"


def extract_plan(
    text: str,
    *,
    manifest: BaselineManifest | None = None,
    source_name: str = "",
    native_value_graph: dict[str, Any] | None = None,
) -> PlanBundle:
    ops, definitions = _parse_ops(text)
    layouts = _layout_aliases(text)
    normalized = normalize_ttgir(text)
    function_match = _FUNC.search(text)
    kernel = function_match.group(1) if function_match else "unknown"

    native_function: dict[str, Any] | None = None
    native_operations: dict[int, dict[str, Any]] = {}
    if native_value_graph is not None:
        if native_value_graph.get("schema_version") not in {
            "plan-value-graph/0.1",
            "plan-value-graph/0.2",
            "plan-value-graph/0.3",
        }:
            raise PlanError("unsupported native plan-value-graph schema")
        native_function = next(
            (
                function
                for function in native_value_graph.get("functions", [])
                if function.get("function") == kernel
            ),
            None,
        )
        if native_function is None:
            raise PlanError(f"native value graph does not contain kernel {kernel!r}")
        native_operations = {operation["ordinal"]: operation for operation in native_function.get("operations", [])}
        if len(native_operations) != len(ops):
            raise PlanError(
                f"native value graph has {len(native_operations)} operations; textual TTGIR has {len(ops)}"
            )

    operation_entries: list[dict[str, Any]] = []
    schedule: list[str] = []
    operation_ids: dict[int, str] = {}
    signature_counts: dict[str, int] = {}
    for op in ops:
        signature = _operation_signature(op)
        count = signature_counts.get(signature, 0)
        signature_counts[signature] = count + 1
        native_operation = native_operations.get(op.ordinal)
        if native_operation is not None and native_operation.get("kind") != op.kind:
            raise PlanError(
                f"native/text operation mismatch at ordinal {op.ordinal}: "
                f"{native_operation.get('kind')!r} != {op.kind!r}"
            )
        op_id = native_operation["id"] if native_operation is not None else _stable_id("op", signature, count)
        operation_ids[op.ordinal] = op_id
        entry = {
            "id": op_id,
            "kind": op.kind,
            "scope": op.scope,
            "ordinal": op.ordinal,
            "result_arity": len(_result_names(op.result)) if op.result else 0,
            "operand_count": len(op.operands),
        }
        if native_operation is not None:
            entry["identity_quality"] = native_operation.get("identity_quality", "semantic")
            entry["stable_locator"] = native_operation.get("locator", "")
        operation_entries.append(entry)
        schedule.append(op_id)

    fragments: list[dict[str, Any]] = []
    role_counts: dict[str, int] = {}
    for op in ops:
        if op.kind not in {"amdg.scheduled_mfma", "tt.dot", "ttng.tc_gen5_mma"}:
            continue
        role = _dot_role(op, definitions)
        role_index = role_counts.get(role, 0)
        role_counts[role] = role_index + 1
        shapes = _tensor_shapes(op.text)
        accumulator = op.operands[2] if len(op.operands) >= 3 else ""
        offsets = _slice_offsets(accumulator, definitions)
        attributes = _attributes(op.text)
        layout = _result_layout(op.text)
        semantic_signature = "|".join(
            (
                role,
                op.kind,
                str(shapes),
                str(offsets),
                str(layouts.get(layout or "", layout)),
                str(attributes),
                str(role_index),
            )
        )
        fragments.append(
            {
                "id": _stable_id(f"dot/{role}", semantic_signature),
                "operation": operation_ids[op.ordinal],
                "role": role,
                "role_ordinal": role_index,
                "kind": op.kind,
                "operand_shapes": shapes[:-1] if len(shapes) > 1 else shapes,
                "result_shape": shapes[-1] if shapes else [],
                "accumulator_slice_offsets": offsets,
                "result_layout": layout,
                "mma_layout": layouts.get(layout or "", ""),
                **attributes,
            }
        )

    storage: list[dict[str, Any]] = []
    for op in ops:
        if op.kind not in {"ttg.local_alloc", "ttg.local_dealloc"}:
            continue
        storage.append(
            {
                "operation": operation_ids[op.ordinal],
                "kind": op.kind,
                "name": op.result,
                "shapes": _tensor_shapes(op.text),
                "type": _type_section(op.text),
            }
        )

    sync_kinds = {
        "ttg.async_commit_group",
        "ttg.async_wait",
        "ttg.barrier",
        "amdg.mfma_commit",
        "amdg.mfma_wait",
    }
    synchronization = [
        {
            "operation": operation_ids[op.ordinal],
            "kind": op.kind,
            "ordinal": op.ordinal,
            "attributes": _attributes(op.text),
        }
        for op in ops
        if op.kind in sync_kinds
    ]

    diagnostics: list[str] = []
    unknown = sum(fragment["role"] == "unknown" for fragment in fragments)
    if unknown:
        diagnostics.append(f"{unknown} dot fragments have an unresolved output role")
    if not fragments:
        diagnostics.append("no dot or scheduled MFMA operations were found")
    if native_function is not None:
        diagnostics.extend(
            f"native:{item.get('severity', 'note')}:{item.get('code', 'unknown')}:"
            f"{item.get('message', '')}"
            for item in native_function.get("diagnostics", [])
        )

    provenance: dict[str, Any] = {"source_name": source_name}
    case: dict[str, Any] = {}
    if manifest:
        provenance.update(
            {
                "source_revision": manifest.source_revision,
                "compiler_revision": manifest.compiler_revision,
                "schedule": manifest.schedule.__dict__,
                "artifacts": manifest.artifacts,
            }
        )
        case = manifest.case.__dict__
    bundle = PlanBundle(
        kernel=kernel,
        case=case,
        provenance=provenance,
        operations=operation_entries,
        dot_fragments=fragments,
        storage=storage,
        synchronization=synchronization,
        schedule=schedule,
        layouts=layouts,
        normalized_ir_hash="sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        values=native_function.get("values", []) if native_function is not None else [],
        lineage_edges=native_function.get("lineage_edges", []) if native_function is not None else [],
        blocks=native_function.get("blocks", []) if native_function is not None else [],
        live_segments=(
            native_function.get("live_segments", []) if native_function is not None else []
        ),
        lds_aliases=native_function.get("lds_aliases", []) if native_function is not None else [],
        memory_accesses=(
            native_function.get("memory_accesses", []) if native_function is not None else []
        ),
        lds_allocations=(
            native_function.get("lds_allocations", []) if native_function is not None else []
        ),
        async_transactions=(
            native_function.get("async_transactions", []) if native_function is not None else []
        ),
        async_groups=native_function.get("async_groups", []) if native_function is not None else [],
        async_waits=native_function.get("async_waits", []) if native_function is not None else [],
        lds_reuse_hazards=(
            native_function.get("lds_reuse_hazards", []) if native_function is not None else []
        ),
        value_graph_fingerprint=(
            native_function.get("semantic_fingerprint", "") if native_function is not None else ""
        ),
        diagnostics=diagnostics,
    )
    bundle.refresh_hashes()
    bundle.validate()
    return bundle
