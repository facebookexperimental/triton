"""Diagnose narrow AMD memory instructions and possible adjacent merges."""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
from typing import Any

_MEMORY_INSTRUCTION = re.compile(
    r"^\s*(?P<opcode>(?P<family>buffer|global|flat)_(?P<direction>load|store)_[A-Za-z0-9_]+)\b(?P<operands>.*)$")
_BUFFER_DWORD = re.compile(
    r"^\s*(?P<opcode>buffer_(?P<direction>load|store)_dword)\s+(?P<data>v(?P<register>\d+))\s*,\s*(?P<rest>.+)$")
_VGPR_OPERAND = re.compile(r"\bv(?:(?P<single>\d+)|\[(?P<start>\d+):(?P<end>\d+)\])\b")
_IMMEDIATE_OFFSET = re.compile(r"\boffset:(?P<offset>0x[0-9A-Fa-f]+|\d+)\b")
_SOURCE_LOCATION = re.compile(r";\s*(?P<file>.+?):(?P<line>\d+):(?P<column>\d+)\s*$")
_FORMAT_WIDTH = re.compile(r"_format_(?P<d16>d16(?:_hi)?_)?(?P<channels>x|xy|xyz|xyzw)$")
_TTGIR_BLOCKED_LAYOUT = re.compile(r"^(?P<name>#\w+) = #ttg\.blocked<\{sizePerThread = \[(?P<sizes>[\d, ]+)\].*"
                                   r"order = \[(?P<order>[\d, ]+)\]")
_TTGIR_LINEAR_LAYOUT = re.compile(r"^(?P<name>#\w+) = #ttg\.linear<\{register = \[(?P<bases>.*?)\], "
                                  r"lane = \[(?P<lane_bases>.*?)\], warp = ")
_TTGIR_LOCATION = re.compile(r'^\s*(?P<name>#loc\d+) = loc\("(?P<file>[^"]+)":(?P<line>\d+):(?P<column>\d+)\)')
_TTGIR_CALLSITE_LOCATION = re.compile(
    r"^\s*(?P<name>#loc\d+) = loc\(callsite\((?P<callee>#loc\d+) at (?P<caller>#loc\d+)\)\)")
_TTGIR_POINTER_LAYOUT = re.compile(r": !tt\.ptr<(?P<dtype>[^>]+)>\[tensor<.*?,\s*(?P<layout>#\w+)>\]")
_TTGIR_BUFFER_LOAD_RESULT = re.compile(
    r": !tt\.ptr<(?P<dtype>[^>]+)>\s*->\s*tensor<(?P<tensor>[^,>]+),\s*(?P<layout>#\w+)>")
_TTGIR_BUFFER_STORE_RESULT = re.compile(
    r"amdg\.buffer_store\b.*?: !tt\.ptr<(?P<dtype>[^>]+)>\s*->\s*tensor<[^,>]+,\s*(?P<layout>#\w+)>")
_TTGIR_OPERATION_LOCATION = re.compile(r"loc\((?P<location>#loc\d+)\)\s*$")
_TTGIR_CONTIGUITY = re.compile(r"\bcontiguity = (?P<count>\d+) : i32\b")
_OBJDUMP_INSTRUCTION = re.compile(
    r"^\s*(?P<opcode>[A-Za-z0-9_.]+)\b.*//\s*(?P<address>[0-9A-Fa-f]+):\s*(?P<encoding>[0-9A-Fa-f]+)")
_RAW_INSTRUCTION = re.compile(r"^\s+(?P<opcode>[A-Za-z][A-Za-z0-9_.]+)\b")
_RAW_LABEL = re.compile(r"^(?P<label>\.L[A-Za-z0-9_.$]+):")
_RAW_BRANCH_TARGET = re.compile(r"\s(?P<label>\.L[A-Za-z0-9_.$]+)(?:\s|$)")
_WIDTH_TOKENS = (
    ("dwordx4", 16),
    ("dwordx3", 12),
    ("dwordx2", 8),
    ("dword", 4),
    ("short", 2),
    ("ushort", 2),
    ("sshort", 2),
    ("byte", 1),
    ("ubyte", 1),
    ("sbyte", 1),
)
_SKIPPABLE_DIRECTIVE = re.compile(r"^\s*\.(?:file|loc)\b")


def _strip_comment(line: str) -> str:
    return line.split(";", 1)[0].split("//", 1)[0].rstrip()


def _memory_width_bytes(opcode: str) -> int | None:
    format_match = _FORMAT_WIDTH.search(opcode)
    if format_match:
        component_bytes = 2 if format_match.group("d16") else 4
        return len(format_match.group("channels")) * component_bytes
    for token, width in _WIDTH_TOKENS:
        if re.search(rf"(?:^|_){token}(?:_|$)", opcode):
            return width
    return None


def _normalize_address(rest: str) -> tuple[str, int]:
    offset_match = _IMMEDIATE_OFFSET.search(rest)
    offset = int(offset_match.group("offset"), 0) if offset_match else 0
    normalized = _IMMEDIATE_OFFSET.sub("offset:<imm>", rest)
    return " ".join(normalized.split()), offset


def _source_location(line: str) -> str | None:
    match = _SOURCE_LOCATION.search(line)
    if not match:
        return None
    path = pathlib.PurePath(match.group("file")).name
    return f"{path}:{match.group('line')}:{match.group('column')}"


def _vgpr_operand_contains(operands: str, register: int) -> bool:
    for match in _VGPR_OPERAND.finditer(operands):
        if match.group("single") is not None:
            if int(match.group("single")) == register:
                return True
            continue
        if int(match.group("start")) <= register <= int(match.group("end")):
            return True
    return False


def _parse_merge_instruction(line: str, line_number: int, source: str | None) -> dict[str, Any] | None:
    match = _BUFFER_DWORD.match(_strip_comment(line))
    if not match:
        return None
    rest = match.group("rest")
    if _vgpr_operand_contains(rest, int(match.group("register"))):
        return None
    address, offset = _normalize_address(rest)
    return {
        "direction": match.group("direction"),
        "opcode": match.group("opcode"),
        "register": int(match.group("register")),
        "address_signature": address,
        "offset": offset,
        "line": line_number,
        "source_location": source,
    }


def _can_extend_merge_run(run: list[dict[str, Any]], instruction: dict[str, Any]) -> bool:
    previous = run[-1]
    return (instruction["opcode"] == previous["opcode"]
            and instruction["address_signature"] == previous["address_signature"]
            and instruction["register"] == previous["register"] + 1 and instruction["offset"] == previous["offset"] + 4)


def _merge_opportunity(group: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(group)
    first = group[0]
    target_opcode = f"buffer_{first['direction']}_dwordx{count}"
    target_width = count * 4
    immediate_aligned = first["offset"] % target_width == 0
    return {
        "direction": first["direction"],
        "from_opcode": first["opcode"],
        "target_opcode": target_opcode,
        "instruction_count": count,
        "bytes_per_lane": target_width,
        "line_start": first["line"],
        "line_end": group[-1]["line"],
        "register_start": first["register"],
        "register_end": group[-1]["register"],
        "offset_start": first["offset"],
        "offset_end": group[-1]["offset"],
        "address_signature": first["address_signature"],
        "source_location": first["source_location"],
        "estimated_issue_reduction": count - 1,
        "confidence": "needs_ir_proof",
        "immediate_offset_aligned": immediate_aligned,
        "unproven_requirements": ["base_alignment", "bounds", "mask_equivalence"],
    }


def _finish_merge_run(run: list[dict[str, Any]], output: list[dict[str, Any]]) -> None:
    start = 0
    remaining = len(run)
    while remaining >= 2:
        count = min(4, remaining)
        output.append(_merge_opportunity(run[start:start + count]))
        start += count
        remaining -= count


def _find_merge_opportunities(lines: list[str]) -> list[dict[str, Any]]:
    opportunities: list[dict[str, Any]] = []
    run: list[dict[str, Any]] = []
    current_source: str | None = None

    for line_number, line in enumerate(lines, start=1):
        location = _source_location(line)
        if location is not None:
            current_source = location

        instruction = _parse_merge_instruction(line, line_number, current_source)
        if instruction is not None:
            if run and not _can_extend_merge_run(run, instruction):
                _finish_merge_run(run, opportunities)
                run = []
            run.append(instruction)
            continue

        stripped = line.strip()
        if not stripped or stripped.startswith((";", "//")) or _SKIPPABLE_DIRECTIVE.match(line):
            continue
        _finish_merge_run(run, opportunities)
        run = []

    _finish_merge_run(run, opportunities)
    return opportunities


def _find_direct_to_lds_opportunities(lines: list[str]) -> list[dict[str, Any]]:
    opportunities = []
    current_source: str | None = None
    for line_number, line in enumerate(lines, start=1):
        location = _source_location(line)
        if location is not None:
            current_source = location
        match = _MEMORY_INSTRUCTION.match(_strip_comment(line))
        if not match or match.group("family") != "buffer" or match.group("direction") != "load":
            continue
        opcode = match.group("opcode")
        width = _memory_width_bytes(opcode)
        if width is None or width >= 16 or not re.search(r"\blds\b", match.group("operands")):
            continue
        format_match = _FORMAT_WIDTH.search(opcode)
        if format_match:
            target_opcode = opcode[:format_match.start("channels")] + "xyzw"
            target_width = (2 if format_match.group("d16") else 4) * 4
        else:
            target_opcode = "buffer_load_dwordx4"
            target_width = 16
        opportunities.append({
            "opcode":
            opcode,
            "target_opcode":
            target_opcode,
            "bytes_per_lane":
            width,
            "target_bytes_per_lane":
            target_width,
            "line":
            line_number,
            "source_location":
            current_source,
            "confidence":
            "needs_ir_proof",
            "unproven_requirements": [
                "source_contiguity",
                "destination_contiguity",
                "base_alignment",
                "bounds",
                "useful_bytes",
            ],
        })
    return opportunities


def _instruction_stream(assembly: str) -> list[dict[str, Any]]:
    instructions = []
    labels = {}
    for line in assembly.splitlines():
        label = _RAW_LABEL.match(line)
        if label:
            labels[label.group("label")] = len(instructions) * 4
        objdump = _OBJDUMP_INSTRUCTION.match(line)
        if objdump:
            instructions.append({
                "line": line,
                "opcode": objdump.group("opcode"),
                "address": int(objdump.group("address"), 16),
                "encoding": int(objdump.group("encoding"), 16),
            })
            continue
        raw = _RAW_INSTRUCTION.match(line)
        if raw and not raw.group("opcode").startswith(("loc", "file", "cfi_", "p2align")):
            branch = _RAW_BRANCH_TARGET.search(line)
            instructions.append({
                "line": line,
                "opcode": raw.group("opcode"),
                "address": len(instructions) * 4,
                "encoding": None,
                "branch_label": branch.group("label") if branch else None,
            })
    for instruction in instructions:
        label = instruction.get("branch_label")
        if label in labels:
            instruction["branch_target"] = labels[label]
    return instructions


def _backward_branch_target(instruction: dict[str, Any]) -> int | None:
    if "branch_target" in instruction:
        return instruction["branch_target"]
    if instruction["encoding"] is None:
        return None
    offset = instruction["encoding"] & 0xFFFF
    offset = offset - 0x10000 if offset & 0x8000 else offset
    return instruction["address"] + 4 + offset * 4


def _load_bearing_hot_loop(instructions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    for instruction in instructions:
        if not instruction["opcode"].startswith(("s_branch", "s_cbranch")):
            continue
        target = _backward_branch_target(instruction)
        if target is None or target >= instruction["address"]:
            continue
        window = [item for item in instructions if target <= item["address"] <= instruction["address"]]
        mfma_count = sum(item["opcode"].startswith("v_mfma") for item in window)
        load_count = sum(item["opcode"].startswith(("buffer_load", "global_load", "flat_load")) for item in window)
        if mfma_count and load_count:
            candidates.append((mfma_count, load_count, instruction["address"] - target, window))
    return max(candidates, key=lambda item: (item[0], item[1], item[2]))[3] if candidates else []


def _hot_loop_load_width_report(assembly: str) -> dict[str, Any]:
    loop = _load_bearing_hot_loop(_instruction_stream(assembly))
    widths: dict[str, int] = {}
    mfma_count = sum(item["opcode"].startswith("v_mfma") for item in loop)
    for instruction in loop:
        match = _MEMORY_INSTRUCTION.match(_strip_comment(instruction["line"]))
        if not match or match.group("direction") != "load":
            continue
        width = _memory_width_bytes(match.group("opcode"))
        if width is not None:
            widths[str(width)] = widths.get(str(width), 0) + 1
    issue_count = sum(widths.values())
    useful_bytes = sum(int(width) * count for width, count in widths.items())
    minimum_issues = math.ceil(useful_bytes / 16) if useful_bytes else 0
    excess_issues = issue_count - minimum_issues
    return {
        "found": bool(loop),
        "mfma_count": mfma_count,
        "load_width_histogram_bytes": dict(sorted(widths.items(), key=lambda item: int(item[0]))),
        "load_issue_count": issue_count,
        "useful_bytes_per_lane": useful_bytes,
        "minimum_16byte_issue_count": minimum_issues,
        "excess_issue_count": excess_issues,
        "issue_efficiency": minimum_issues / issue_count if issue_count else None,
        "requires_layout_redistribution": excess_issues > 0,
        "scope": "static steady-state loop; the minimum assumes legal 16-byte regrouping across lane-owned bytes",
    }


def _ttgir_element_bytes(dtype: str) -> int | None:
    if dtype.startswith(("f8", "i8", "ui8")):
        return 1
    if dtype in ("bf16", "f16", "i16", "ui16"):
        return 2
    if dtype in ("f32", "i32", "ui32"):
        return 4
    if dtype in ("f64", "i64", "ui64"):
        return 8
    return None


def _linear_layout_contiguity_by_dimension(bases: list[list[int]]) -> list[int]:
    rank = len(bases[0]) if bases else 0
    result = []
    for dimension in range(rank):
        contiguous = 1
        for basis in bases:
            if not any(basis):
                continue
            if any(value for index, value in enumerate(basis) if index != dimension) or basis[dimension] != contiguous:
                break
            contiguous *= 2
        result.append(contiguous)
    return result


def _linear_layout_owned_extent_by_dimension(bases: list[list[int]]) -> list[int]:
    rank = len(bases[0]) if bases else 0
    result = []
    for dimension in range(rank):
        values = {
            basis[dimension]
            for basis in bases
            if basis[dimension] and not any(value for index, value in enumerate(basis) if index != dimension)
        }
        extent = 1
        while extent in values:
            extent *= 2
        result.append(extent)
    return result


def _ttgir_layout_info(lines: list[str]) -> dict[str, dict[str, Any]]:
    layouts = {}
    for line in lines:
        blocked = _TTGIR_BLOCKED_LAYOUT.match(line)
        if blocked:
            sizes = [int(value.strip()) for value in blocked.group("sizes").split(",")]
            order = [int(value.strip()) for value in blocked.group("order").split(",")]
            if order and order[0] < len(sizes):
                layouts[blocked.group("name")] = {
                    "kind": "blocked",
                    "contiguous_elements_per_lane": sizes[order[0]],
                    "last_dimension_contiguous_elements_per_lane": sizes[-1],
                }
            continue
        linear = _TTGIR_LINEAR_LAYOUT.match(line)
        if linear:
            bases = ast.literal_eval(f"[{linear.group('bases')}]")
            lane_bases = ast.literal_eval(f"[{linear.group('lane_bases')}]")
            by_dimension = _linear_layout_contiguity_by_dimension(bases)
            lane_by_dimension = _linear_layout_contiguity_by_dimension(lane_bases)
            layouts[linear.group("name")] = {
                "kind": "linear",
                "contiguous_elements_per_lane": max(by_dimension, default=1),
                "last_dimension_contiguous_elements_per_lane": (by_dimension[-1] if by_dimension else 1),
                "register_contiguity_by_dimension": by_dimension,
                "lane_contiguity_by_dimension": lane_by_dimension,
                "register_owned_extent_by_dimension": _linear_layout_owned_extent_by_dimension(bases),
                "lane_owned_extent_by_dimension": _linear_layout_owned_extent_by_dimension(lane_bases),
            }
    return layouts


def _ttgir_source_locations(lines: list[str]) -> dict[str, str]:
    direct = {}
    callsites = {}
    for line in lines:
        match = _TTGIR_LOCATION.match(line)
        if match:
            path = pathlib.PurePath(match.group("file")).name
            direct[match.group("name")] = f"{path}:{match.group('line')}:{match.group('column')}"
            continue
        callsite = _TTGIR_CALLSITE_LOCATION.match(line)
        if callsite:
            callsites[callsite.group("name")] = (callsite.group("callee"), callsite.group("caller"))

    locations = dict(direct)

    def resolve(name: str, active: set[str]) -> str | None:
        if name in direct:
            return direct[name]
        if name in active or name not in callsites:
            return None
        active.add(name)
        callee, caller = callsites[name]
        resolved = resolve(callee, active) or resolve(caller, active)
        active.remove(name)
        return resolved

    for name in callsites:
        resolved = resolve(name, set())
        if resolved:
            locations[name] = resolved
    return locations


def _ttgir_explicit_contiguity_cap(line_number: int, line: str, layouts: dict[str, int],
                                   locations: dict[str, str]) -> dict[str, Any] | None:
    if "amdg.buffer_load" not in line or "amdg.buffer_load_to_local" in line or "tlx.preserve_layout" not in line:
        return None
    result = _TTGIR_BUFFER_LOAD_RESULT.search(line)
    contiguity = _TTGIR_CONTIGUITY.search(line)
    location = _TTGIR_OPERATION_LOCATION.search(line)
    if not result or not contiguity or result.group("layout") not in layouts:
        return None
    element_bytes = _ttgir_element_bytes(result.group("dtype"))
    if element_bytes is None or element_bytes > 16:
        return None
    elements_per_lane = layouts[result.group("layout")]["contiguous_elements_per_lane"]
    explicit_contiguity = int(contiguity.group("count"))
    candidate_contiguity = min(elements_per_lane, 16 // element_bytes)
    if explicit_contiguity >= candidate_contiguity:
        return None
    return {
        "line": line_number,
        "source_location": locations.get(location.group("location")) if location else None,
        "dtype": result.group("dtype"),
        "layout": result.group("layout"),
        "elements_per_lane": elements_per_lane,
        "explicit_contiguity": explicit_contiguity,
        "candidate_contiguity": candidate_contiguity,
        "explicit_bytes_per_lane": explicit_contiguity * element_bytes,
        "candidate_bytes_per_lane": candidate_contiguity * element_bytes,
        "static_occurrences": 1,
        "classification": "explicit_contiguity_cap",
        "confidence": "needs_address_proof",
        "unproven_requirements": ["consecutive_addresses", "base_alignment", "bounds", "mask_equivalence"],
    }


def _ttgir_explicit_contiguity_caps(ttgir: str) -> list[dict[str, Any]]:
    lines = ttgir.splitlines()
    layouts = _ttgir_layout_info(lines)
    locations = _ttgir_source_locations(lines)
    candidates: dict[tuple[Any, ...], dict[str, Any]] = {}
    for line_number, line in enumerate(lines, start=1):
        candidate = _ttgir_explicit_contiguity_cap(line_number, line, layouts, locations)
        if candidate is None:
            continue
        key = tuple(candidate[field] for field in (
            "source_location",
            "dtype",
            "layout",
            "elements_per_lane",
            "explicit_contiguity",
            "candidate_contiguity",
        ))
        if key in candidates:
            candidates[key]["static_occurrences"] += 1
        else:
            candidates[key] = candidate
    return list(candidates.values())


def _ttgir_packed_byte_load_candidates(ttgir: str) -> list[dict[str, Any]]:
    lines = ttgir.splitlines()
    layouts = _ttgir_layout_info(lines)
    locations = _ttgir_source_locations(lines)
    candidates: dict[tuple[Any, ...], dict[str, Any]] = {}
    for line_number, line in enumerate(lines, start=1):
        if "amdg.buffer_load" not in line or "amdg.buffer_load_to_local" in line:
            continue
        result = _TTGIR_BUFFER_LOAD_RESULT.search(line)
        location = _TTGIR_OPERATION_LOCATION.search(line)
        if not result or result.group("dtype") not in ("i8", "ui8"):
            continue
        layout = layouts.get(result.group("layout"))
        if not layout or layout["kind"] != "linear":
            continue
        tensor_shape = result.group("tensor").rsplit("x", 1)[0]
        dimensions = [int(value) for value in tensor_shape.split("x")]
        register_owned = layout["register_owned_extent_by_dimension"]
        lane_owned = layout["lane_owned_extent_by_dimension"]
        if (not dimensions or dimensions[-1] != 4 or not register_owned or not lane_owned or register_owned[-1] != 1
                or lane_owned[-1] < 4):
            continue
        source_location = locations.get(location.group("location")) if location else None
        key = (source_location, result.group("layout"))
        if key in candidates:
            candidates[key]["static_occurrences"] += 1
            continue
        candidates[key] = {
            "line": line_number,
            "source_location": source_location,
            "dtype": result.group("dtype"),
            "layout": result.group("layout"),
            "logical_shape": dimensions,
            "bytes_per_word": 4,
            "static_occurrences": 1,
            "classification": "cross_lane_packed_byte_word",
            "confidence": "needs_word_alignment_and_performance_proof",
            "unproven_requirements": ["word_alignment", "consecutive_addresses", "lane_redistribution_cost"],
        }
    return list(candidates.values())


def _ttgir_store_width_candidates(ttgir: str) -> list[dict[str, Any]]:
    lines = ttgir.splitlines()
    layouts = _ttgir_layout_info(lines)
    locations = _ttgir_source_locations(lines)
    candidates: dict[tuple[Any, ...], dict[str, Any]] = {}
    for line_number, line in enumerate(lines, start=1):
        store = _TTGIR_BUFFER_STORE_RESULT.search(line)
        location = _TTGIR_OPERATION_LOCATION.search(line)
        if not store:
            continue
        element_bytes = _ttgir_element_bytes(store.group("dtype"))
        if element_bytes is None:
            continue
        layout = store.group("layout")
        layout_info = layouts.get(layout)
        contiguous_elements = layout_info["last_dimension_contiguous_elements_per_lane"] if layout_info else None
        source_location = locations.get(location.group("location")) if location else None
        key = (source_location, store.group("dtype"), layout, contiguous_elements)
        if key in candidates:
            candidates[key]["static_occurrences"] += 1
            continue
        candidate_bytes = min(16, contiguous_elements * element_bytes) if contiguous_elements else None
        candidates[key] = {
            "line":
            line_number,
            "source_location":
            source_location,
            "dtype":
            store.group("dtype"),
            "layout":
            layout,
            "contiguous_elements_per_lane":
            contiguous_elements,
            "candidate_bytes_per_lane":
            candidate_bytes,
            "static_occurrences":
            1,
            "classification":
            (f"{layout_info['kind']}_layout_store_width" if layout_info else "requires_output_layout_conversion"),
            "confidence":
            "needs_mask_bounds_and_performance_proof",
            "unproven_requirements": ["base_alignment", "bounds", "mask_equivalence", "layout_conversion_cost"],
        }
    return list(candidates.values())


def _ttgir_direct_to_lds_sites(ttgir: str) -> list[dict[str, Any]]:
    lines = ttgir.splitlines()
    layouts = _ttgir_layout_info(lines)
    locations = _ttgir_source_locations(lines)
    sites = []
    for line_number, line in enumerate(lines, start=1):
        if "amdg.buffer_load_to_local" not in line:
            continue
        pointer = _TTGIR_POINTER_LAYOUT.search(line)
        location = _TTGIR_OPERATION_LOCATION.search(line)
        if not pointer or pointer.group("layout") not in layouts:
            continue
        element_bytes = _ttgir_element_bytes(pointer.group("dtype"))
        if element_bytes is None:
            continue
        elements_per_thread = layouts[pointer.group("layout")]["contiguous_elements_per_lane"]
        useful_bytes = elements_per_thread * element_bytes
        contiguity = _TTGIR_CONTIGUITY.search(line)
        if contiguity:
            useful_bytes = min(useful_bytes, int(contiguity.group("count")) * element_bytes)
        sites.append({
            "line": line_number,
            "source_location": locations.get(location.group("location")) if location else None,
            "dtype": pointer.group("dtype"),
            "layout": pointer.group("layout"),
            "elements_per_thread": elements_per_thread,
            "useful_bytes_per_lane": useful_bytes,
        })
    return sites


def _add_ttgir_evidence(report: dict[str, Any], ttgir: str) -> None:
    contiguity_caps = _ttgir_explicit_contiguity_caps(ttgir)
    for candidate in contiguity_caps:
        candidate["corroborated_by_hot_loop_headroom"] = report["hot_loop_excess_issue_count"] > 0
        source_opcodes = report["source_opcode_counts"].get(candidate["source_location"], {})
        isa_corroboration_scope = "source_location"
        if not source_opcodes:
            source_opcodes = report["opcode_counts"]
            isa_corroboration_scope = "kernel"
        candidate["narrow_load_instruction_count"] = sum(
            count for opcode, count in source_opcodes.items()
            if "_load_" in opcode and (_memory_width_bytes(opcode) or 16) < candidate["candidate_bytes_per_lane"])
        candidate["isa_corroboration_scope"] = isa_corroboration_scope
        candidate["corroborated_by_narrow_isa"] = candidate["narrow_load_instruction_count"] > 0
    report["explicit_contiguity_caps"] = contiguity_caps
    report["explicit_contiguity_cap_count"] = len(contiguity_caps)
    report["explicit_contiguity_cap_static_occurrences"] = sum(candidate["static_occurrences"]
                                                               for candidate in contiguity_caps)
    report["explicit_contiguity_hot_loop_cap_count"] = sum(candidate["corroborated_by_hot_loop_headroom"]
                                                           for candidate in contiguity_caps)
    report["explicit_contiguity_unresolved_count"] = sum(candidate["corroborated_by_narrow_isa"]
                                                         for candidate in contiguity_caps)
    report["explicit_contiguity_already_wide_count"] = (report["explicit_contiguity_cap_count"] -
                                                        report["explicit_contiguity_unresolved_count"])
    packed_byte_candidates = _ttgir_packed_byte_load_candidates(ttgir)
    for candidate in packed_byte_candidates:
        source_opcodes = report["source_opcode_counts"].get(candidate["source_location"], {})
        isa_corroboration_scope = "source_location"
        if not source_opcodes:
            source_opcodes = report["opcode_counts"]
            isa_corroboration_scope = "kernel"
        candidate["scalar_byte_load_instruction_count"] = sum(
            count for opcode, count in source_opcodes.items()
            if "_load_" in opcode and _memory_width_bytes(opcode) == 1)
        candidate["isa_corroboration_scope"] = isa_corroboration_scope
        candidate["corroborated_by_scalar_byte_isa"] = candidate["scalar_byte_load_instruction_count"] > 0
    report["packed_byte_load_candidates"] = packed_byte_candidates
    report["packed_byte_load_candidate_count"] = len(packed_byte_candidates)
    report["packed_byte_load_unresolved_count"] = sum(candidate["corroborated_by_scalar_byte_isa"]
                                                      for candidate in packed_byte_candidates)
    store_candidates = _ttgir_store_width_candidates(ttgir)
    for candidate in store_candidates:
        candidate_bytes = candidate["candidate_bytes_per_lane"]
        source_opcodes = report["source_opcode_counts"].get(candidate["source_location"], {})
        candidate["narrow_store_instruction_count"] = sum(
            count for opcode, count in source_opcodes.items()
            if "_store_" in opcode and candidate_bytes is not None and (
                _memory_width_bytes(opcode) or candidate_bytes) < candidate_bytes)
        candidate["isa_corroboration_scope"] = "source_location"
        candidate["corroborated_by_narrow_isa"] = candidate["narrow_store_instruction_count"] > 0
    report["store_width_candidates"] = store_candidates
    report["store_width_candidate_count"] = len(store_candidates)
    report["store_width_unresolved_count"] = sum(candidate["corroborated_by_narrow_isa"]
                                                 for candidate in store_candidates)
    report["store_layout_conversion_count"] = sum(candidate["classification"] == "requires_output_layout_conversion"
                                                  for candidate in store_candidates)
    sites = _ttgir_direct_to_lds_sites(ttgir)
    by_source: dict[str, list[dict[str, Any]]] = {}
    for site in sites:
        if site["source_location"]:
            by_source.setdefault(site["source_location"], []).append(site)
    for candidate in report["direct_to_lds_opportunities"]:
        matches = by_source.get(candidate["source_location"], [])
        if not matches or any(site["useful_bytes_per_lane"] > candidate["bytes_per_lane"] for site in matches):
            continue
        candidate["ir_classification"] = "lane_saturated"
        candidate["confidence"] = "ir_proven_no_additional_useful_bytes"
        candidate["ir_evidence"] = matches
    saturated = sum(item.get("ir_classification") == "lane_saturated" for item in report["direct_to_lds_opportunities"])
    report["direct_to_lds_ir_lane_saturated_count"] = saturated
    report["direct_to_lds_unresolved_count"] = report["direct_to_lds_opportunity_count"] - saturated
    report["potential_widening_count"] = (report["merge_opportunity_count"] + report["direct_to_lds_unresolved_count"] +
                                          report["explicit_contiguity_unresolved_count"] +
                                          report["packed_byte_load_unresolved_count"])


def analyze_amdgcn_memory_width(assembly: str) -> dict[str, Any]:
    """Return static memory-width statistics and conservative merge candidates."""
    width_histogram: dict[str, int] = {}
    opcode_counts: dict[str, int] = {}
    unknown_opcodes: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    source_opcode_counts: dict[str, dict[str, int]] = {}
    known_width_count = 0
    total_bytes_per_lane = 0
    narrow_count = 0
    wide_count = 0

    lines = assembly.splitlines()
    current_source: str | None = None
    for line in lines:
        location = _source_location(line)
        if location is not None:
            current_source = location
        stripped = line.lstrip()
        if stripped.startswith((";", "//")):
            continue
        match = _MEMORY_INSTRUCTION.match(_strip_comment(line))
        if not match:
            continue
        opcode = match.group("opcode")
        family = f"{match.group('family')}_{match.group('direction')}"
        opcode_counts[opcode] = opcode_counts.get(opcode, 0) + 1
        if current_source is not None:
            source_counts = source_opcode_counts.setdefault(current_source, {})
            source_counts[opcode] = source_counts.get(opcode, 0) + 1
        family_counts[family] = family_counts.get(family, 0) + 1
        width = _memory_width_bytes(opcode)
        if width is None:
            unknown_opcodes[opcode] = unknown_opcodes.get(opcode, 0) + 1
            continue
        known_width_count += 1
        total_bytes_per_lane += width
        width_key = str(width)
        width_histogram[width_key] = width_histogram.get(width_key, 0) + 1
        if width < 16:
            narrow_count += 1
        else:
            wide_count += 1

    opportunities = _find_merge_opportunities(lines)
    direct_to_lds = _find_direct_to_lds_opportunities(lines)
    hot_loop = _hot_loop_load_width_report(assembly)
    return {
        "instruction_count": sum(opcode_counts.values()),
        "known_width_instruction_count": known_width_count,
        "unknown_width_instruction_count": sum(unknown_opcodes.values()),
        "total_bytes_per_lane": total_bytes_per_lane,
        "narrow_instruction_count": narrow_count,
        "wide_instruction_count": wide_count,
        "width_histogram_bytes": dict(sorted(width_histogram.items(), key=lambda item: int(item[0]))),
        "family_counts": dict(sorted(family_counts.items())),
        "opcode_counts": dict(sorted(opcode_counts.items())),
        "source_opcode_counts":
        {source: dict(sorted(counts.items()))
         for source, counts in sorted(source_opcode_counts.items())},
        "unknown_opcodes": dict(sorted(unknown_opcodes.items())),
        "merge_opportunity_count": len(opportunities),
        "estimated_static_issue_reduction": sum(item["estimated_issue_reduction"] for item in opportunities),
        "merge_opportunities": opportunities,
        "direct_to_lds_opportunity_count": len(direct_to_lds),
        "direct_to_lds_ir_lane_saturated_count": 0,
        "direct_to_lds_unresolved_count": len(direct_to_lds),
        "direct_to_lds_opportunities": direct_to_lds,
        "explicit_contiguity_cap_count": 0,
        "explicit_contiguity_cap_static_occurrences": 0,
        "explicit_contiguity_hot_loop_cap_count": 0,
        "explicit_contiguity_unresolved_count": 0,
        "explicit_contiguity_already_wide_count": 0,
        "explicit_contiguity_caps": [],
        "packed_byte_load_candidate_count": 0,
        "packed_byte_load_unresolved_count": 0,
        "packed_byte_load_candidates": [],
        "store_width_candidate_count": 0,
        "store_width_unresolved_count": 0,
        "store_layout_conversion_count": 0,
        "store_width_candidates": [],
        "potential_widening_count": len(opportunities) + len(direct_to_lds),
        "hot_loop_load_width": hot_loop,
        "hot_loop_found_count": int(hot_loop["found"]),
        "hot_loop_layout_candidate_count": int(hot_loop["requires_layout_redistribution"]),
        "hot_loop_excess_issue_count": hot_loop["excess_issue_count"],
        "scope": "static AMDGCN; merge candidates require IR proof and performance validation",
    }


def _merge_count_map(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + value


def _new_corpus_summary() -> dict[str, Any]:
    return {
        "kernel_count": 0,
        "fully_wide_kernel_count": 0,
        "kernel_with_4_byte_ops_count": 0,
        "kernel_with_8_byte_ops_count": 0,
        "instruction_count": 0,
        "known_width_instruction_count": 0,
        "unknown_width_instruction_count": 0,
        "narrow_instruction_count": 0,
        "wide_instruction_count": 0,
        "merge_opportunity_count": 0,
        "direct_to_lds_opportunity_count": 0,
        "direct_to_lds_ir_lane_saturated_count": 0,
        "direct_to_lds_unresolved_count": 0,
        "explicit_contiguity_cap_count": 0,
        "explicit_contiguity_cap_static_occurrences": 0,
        "explicit_contiguity_hot_loop_cap_count": 0,
        "explicit_contiguity_unresolved_count": 0,
        "explicit_contiguity_already_wide_count": 0,
        "packed_byte_load_candidate_count": 0,
        "packed_byte_load_unresolved_count": 0,
        "store_width_candidate_count": 0,
        "store_width_unresolved_count": 0,
        "store_layout_conversion_count": 0,
        "estimated_static_issue_reduction": 0,
        "hot_loop_found_count": 0,
        "hot_loop_layout_candidate_count": 0,
        "hot_loop_excess_issue_count": 0,
        "width_histogram_bytes": {},
        "opcode_counts": {},
        "unknown_opcodes": {},
    }


def _add_kernel_to_summary(summary: dict[str, Any], report: dict[str, Any]) -> None:
    summary["kernel_count"] += 1
    for key in (
            "instruction_count",
            "known_width_instruction_count",
            "unknown_width_instruction_count",
            "narrow_instruction_count",
            "wide_instruction_count",
            "merge_opportunity_count",
            "direct_to_lds_opportunity_count",
            "direct_to_lds_ir_lane_saturated_count",
            "direct_to_lds_unresolved_count",
            "explicit_contiguity_cap_count",
            "explicit_contiguity_cap_static_occurrences",
            "explicit_contiguity_hot_loop_cap_count",
            "explicit_contiguity_unresolved_count",
            "explicit_contiguity_already_wide_count",
            "packed_byte_load_candidate_count",
            "packed_byte_load_unresolved_count",
            "store_width_candidate_count",
            "store_width_unresolved_count",
            "store_layout_conversion_count",
            "estimated_static_issue_reduction",
            "hot_loop_found_count",
            "hot_loop_layout_candidate_count",
            "hot_loop_excess_issue_count",
    ):
        summary[key] += report[key]
    _merge_count_map(summary["width_histogram_bytes"], report["width_histogram_bytes"])
    _merge_count_map(summary["opcode_counts"], report["opcode_counts"])
    _merge_count_map(summary["unknown_opcodes"], report["unknown_opcodes"])
    widths = report["width_histogram_bytes"]
    summary["fully_wide_kernel_count"] += int(report["instruction_count"] > 0
                                              and report["unknown_width_instruction_count"] == 0
                                              and report["narrow_instruction_count"] == 0)
    summary["kernel_with_4_byte_ops_count"] += int(widths.get("4", 0) > 0)
    summary["kernel_with_8_byte_ops_count"] += int(widths.get("8", 0) > 0)


def analyze_dump_directory(directory: pathlib.Path, index_path: pathlib.Path | None = None) -> dict[str, Any]:
    """Analyze every AMDGCN file in a Triton dump directory."""
    resolved_index = index_path
    if resolved_index is None:
        candidate = directory / "index.json"
        if candidate.is_file():
            resolved_index = candidate
    metadata_by_dump: dict[str, dict[str, Any]] = {}
    indexed_case_count = 0
    if resolved_index is not None:
        index = json.loads(resolved_index.read_text())
        indexed_case_count = len(index.get("cases", []))
        for compile_entry in index.get("compiles", []):
            metadata = metadata_by_dump.setdefault(
                compile_entry["dump"],
                {"names": set(), "cases": set()},
            )
            metadata["names"].add(compile_entry.get("name"))
            metadata["cases"].add(compile_entry.get("case"))

    overall = _new_corpus_summary()
    by_family: dict[str, dict[str, Any]] = {}
    kernels = []
    search_root = directory / "dump" if (directory / "dump").is_dir() else directory
    for path in sorted(search_root.rglob("*.amdgcn")):
        report = analyze_amdgcn_memory_width(path.read_text())
        ttgir_path = path.with_suffix(".ttgir")
        if ttgir_path.is_file():
            _add_ttgir_evidence(report, ttgir_path.read_text())
        dump = path.parent.name
        metadata = metadata_by_dump.get(dump, {"names": set(), "cases": set()})
        cases = sorted(case for case in metadata["cases"] if case)
        names = sorted(name for name in metadata["names"] if name)
        families = sorted({case.split("/", 1)[0] for case in cases}) or ["unknown"]
        _add_kernel_to_summary(overall, report)
        for family in families:
            _add_kernel_to_summary(by_family.setdefault(family, _new_corpus_summary()), report)
        kernels.append({
            "dump": dump,
            "path": str(path),
            "names": names,
            "cases": cases,
            "families": families,
            "report": report,
        })

    for summary in [overall, *by_family.values()]:
        summary["width_histogram_bytes"] = dict(
            sorted(summary["width_histogram_bytes"].items(), key=lambda item: int(item[0])))
        summary["opcode_counts"] = dict(sorted(summary["opcode_counts"].items()))
        summary["unknown_opcodes"] = dict(sorted(summary["unknown_opcodes"].items()))
    ranked = sorted(
        (kernel for kernel in kernels if kernel["report"]["potential_widening_count"] > 0),
        key=lambda item: (
            item["report"]["potential_widening_count"],
            item["report"]["narrow_instruction_count"],
            item["report"]["instruction_count"],
        ),
        reverse=True,
    )
    hot_loop_ranked = sorted(
        (kernel for kernel in kernels if kernel["report"]["hot_loop_excess_issue_count"] > 0),
        key=lambda item: (
            item["report"]["hot_loop_excess_issue_count"],
            item["report"]["hot_loop_load_width"]["useful_bytes_per_lane"],
        ),
        reverse=True,
    )
    contiguity_cap_ranked = sorted(
        (kernel for kernel in kernels if kernel["report"]["explicit_contiguity_unresolved_count"] > 0),
        key=lambda item: (
            item["report"]["explicit_contiguity_hot_loop_cap_count"] > 0,
            item["report"]["hot_loop_excess_issue_count"],
            item["report"]["explicit_contiguity_cap_static_occurrences"],
        ),
        reverse=True,
    )
    packed_byte_ranked = sorted(
        (kernel for kernel in kernels if kernel["report"]["packed_byte_load_unresolved_count"] > 0),
        key=lambda item: (
            item["report"]["packed_byte_load_unresolved_count"],
            item["report"]["hot_loop_excess_issue_count"],
        ),
        reverse=True,
    )
    store_width_ranked = sorted(
        (kernel for kernel in kernels if kernel["report"]["store_width_unresolved_count"] > 0),
        key=lambda item: (
            item["report"]["store_width_unresolved_count"],
            item["report"]["narrow_instruction_count"],
        ),
        reverse=True,
    )
    return {
        "indexed_case_count": indexed_case_count,
        "overall": overall,
        "by_family": dict(sorted(by_family.items())),
        "kernels": kernels,
        "top_potential_widening": ranked,
        "top_hot_loop_layout_headroom": hot_loop_ranked,
        "top_explicit_contiguity_caps": contiguity_cap_ranked,
        "top_packed_byte_load_candidates": packed_byte_ranked,
        "top_store_width_candidates": store_width_ranked,
    }


def _find_llvm_objdump(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    environment = os.environ.get("TRITON_AMD_LLVM_OBJDUMP")
    if environment:
        return environment
    candidates = (
        "/opt/rocm/llvm/bin/llvm-objdump",
        shutil.which("llvm-objdump"),
    )
    return next((candidate for candidate in candidates if candidate and pathlib.Path(candidate).is_file()), None)


def _read_input(path: pathlib.Path, input_format: str, objdump: str | None) -> str:
    resolved_format = input_format
    if resolved_format == "auto":
        resolved_format = "hsaco" if path.suffix == ".hsaco" else "amdgcn"
    if resolved_format == "amdgcn":
        return path.read_text()
    tool = _find_llvm_objdump(objdump)
    if tool is None:
        raise RuntimeError("llvm-objdump is required to analyze an hsaco")
    return subprocess.check_output([tool, "-d", "-l", "--arch=amdgcn", str(path)], text=True)


def _format_report(report: dict[str, Any]) -> str:
    lines = [
        "AMD memory-width analysis",
        f"  memory instructions: {report['instruction_count']}",
        f"  known widths: {report['known_width_instruction_count']}",
        f"  narrow (<16 B/lane): {report['narrow_instruction_count']}",
        f"  wide (>=16 B/lane): {report['wide_instruction_count']}",
        f"  width histogram: {report['width_histogram_bytes']}",
        f"  merge opportunities: {report['merge_opportunity_count']}",
        f"  direct-to-LDS width opportunities: {report['direct_to_lds_opportunity_count']}",
        f"  direct-to-LDS IR lane-saturated: {report['direct_to_lds_ir_lane_saturated_count']}",
        f"  direct-to-LDS unresolved: {report['direct_to_lds_unresolved_count']}",
        f"  explicit contiguity caps: {report['explicit_contiguity_cap_count']}",
        f"  explicit contiguity caps with narrow ISA: {report['explicit_contiguity_unresolved_count']}",
        f"  explicit contiguity caps already widened by codegen: {report['explicit_contiguity_already_wide_count']}",
        f"  packed-byte load candidates: {report['packed_byte_load_candidate_count']}",
        f"  packed-byte candidates with scalar-byte ISA: {report['packed_byte_load_unresolved_count']}",
        f"  store-width candidates: {report['store_width_candidate_count']}",
        f"  store-width candidates with narrow ISA: {report['store_width_unresolved_count']}",
        f"  stores requiring output layout conversion: {report['store_layout_conversion_count']}",
        f"  explicit contiguity caps in headroom kernels: {report['explicit_contiguity_hot_loop_cap_count']}",
        f"  hot-loop load widths: {report['hot_loop_load_width']['load_width_histogram_bytes']}",
        f"  hot-loop excess issues vs 16 B/lane: {report['hot_loop_excess_issue_count']}",
        f"  estimated static issue reduction: {report['estimated_static_issue_reduction']}",
    ]
    for item in report["merge_opportunities"]:
        source = f" source={item['source_location']}" if item["source_location"] else ""
        lines.append(f"  lines {item['line_start']}-{item['line_end']}: {item['instruction_count']} x "
                     f"{item['from_opcode']} -> {item['target_opcode']} ({item['confidence']}){source}")
    if report["unknown_opcodes"]:
        lines.append(f"  unknown-width opcodes: {report['unknown_opcodes']}")
    for item in report["direct_to_lds_opportunities"]:
        source = f" source={item['source_location']}" if item["source_location"] else ""
        lines.append(f"  line {item['line']}: {item['opcode']} lds -> investigate {item['target_opcode']} "
                     f"({item['confidence']}){source}")
    for item in report["explicit_contiguity_caps"]:
        source = f" source={item['source_location']}" if item["source_location"] else ""
        lines.append(
            f"  TTGIR line {item['line']}: contiguity {item['explicit_contiguity']} -> investigate "
            f"{item['candidate_contiguity']} elements/lane ({item['confidence']}, "
            f"occurrences={item['static_occurrences']}, narrow_isa={item['narrow_load_instruction_count']}){source}")
    for item in report["store_width_candidates"]:
        source = f" source={item['source_location']}" if item["source_location"] else ""
        lines.append(f"  TTGIR line {item['line']}: store layout={item['layout']} "
                     f"candidate={item['candidate_bytes_per_lane']} B/lane ({item['classification']}, "
                     f"narrow_isa={item['narrow_store_instruction_count']}){source}")
    return "\n".join(lines)


def _format_corpus_report(corpus: dict[str, Any], top: int) -> str:
    overall = corpus["overall"]
    lines = [
        "AMD memory-width corpus analysis",
        f"  indexed cases: {corpus['indexed_case_count']}",
        f"  compiled kernels: {overall['kernel_count']}",
        f"  fully wide kernels: {overall['fully_wide_kernel_count']}",
        f"  kernels with 4 B/lane ops: {overall['kernel_with_4_byte_ops_count']}",
        f"  kernels with 8 B/lane ops: {overall['kernel_with_8_byte_ops_count']}",
        f"  width histogram: {overall['width_histogram_bytes']}",
        f"  adjacent merge opportunities: {overall['merge_opportunity_count']}",
        f"  direct-to-LDS investigations: {overall['direct_to_lds_opportunity_count']}",
        f"  direct-to-LDS IR lane-saturated: {overall['direct_to_lds_ir_lane_saturated_count']}",
        f"  direct-to-LDS unresolved: {overall['direct_to_lds_unresolved_count']}",
        f"  explicit contiguity caps: {overall['explicit_contiguity_cap_count']}",
        f"  explicit contiguity cap occurrences: {overall['explicit_contiguity_cap_static_occurrences']}",
        f"  explicit contiguity caps with narrow ISA: {overall['explicit_contiguity_unresolved_count']}",
        f"  explicit contiguity caps already widened by codegen: {overall['explicit_contiguity_already_wide_count']}",
        f"  packed-byte load candidates: {overall['packed_byte_load_candidate_count']}",
        f"  packed-byte candidates with scalar-byte ISA: {overall['packed_byte_load_unresolved_count']}",
        f"  store-width candidates: {overall['store_width_candidate_count']}",
        f"  store-width unresolved: {overall['store_width_unresolved_count']}",
        f"  store layout conversions: {overall['store_layout_conversion_count']}",
        f"  explicit contiguity caps in headroom kernels: {overall['explicit_contiguity_hot_loop_cap_count']}",
        f"  load-bearing hot loops: {overall['hot_loop_found_count']}",
        f"  hot-loop layout candidates: {overall['hot_loop_layout_candidate_count']}",
        f"  hot-loop excess issues: {overall['hot_loop_excess_issue_count']}",
    ]
    for family, summary in corpus["by_family"].items():
        lines.append(f"  {family}: kernels={summary['kernel_count']} fully_wide={summary['fully_wide_kernel_count']} "
                     f"widths={summary['width_histogram_bytes']} merge={summary['merge_opportunity_count']} "
                     f"direct_lds={summary['direct_to_lds_opportunity_count']} "
                     f"lane_saturated={summary['direct_to_lds_ir_lane_saturated_count']} "
                     f"unresolved={summary['direct_to_lds_unresolved_count']} "
                     f"contiguity_caps={summary['explicit_contiguity_cap_count']} "
                     f"contiguity_unresolved={summary['explicit_contiguity_unresolved_count']} "
                     f"packed_byte_unresolved={summary['packed_byte_load_unresolved_count']} "
                     f"store_unresolved={summary['store_width_unresolved_count']} "
                     f"hot_loop_candidates={summary['hot_loop_layout_candidate_count']}")
    lines.append("  top unresolved widening kernels:")
    for kernel in corpus["top_potential_widening"][:top]:
        report = kernel["report"]
        cases = ", ".join(kernel["cases"][:3]) or "unindexed"
        lines.append(
            f"    {kernel['names'] or [pathlib.Path(kernel['path']).stem]}: "
            f"potential={report['potential_widening_count']} widths={report['width_histogram_bytes']} cases={cases}")
    if not corpus["top_potential_widening"]:
        lines.append("    none")
    lines.append("  top explicit contiguity caps:")
    for kernel in corpus["top_explicit_contiguity_caps"][:top]:
        report = kernel["report"]
        cases = ", ".join(kernel["cases"][:3]) or "unindexed"
        lines.append(f"    {kernel['names'] or [pathlib.Path(kernel['path']).stem]}: "
                     f"caps={report['explicit_contiguity_cap_count']} "
                     f"occurrences={report['explicit_contiguity_cap_static_occurrences']} "
                     f"hot_loop_excess={report['hot_loop_excess_issue_count']} cases={cases}")
    if not corpus["top_explicit_contiguity_caps"]:
        lines.append("    none")
    lines.append("  top store-width candidates:")
    for kernel in corpus["top_store_width_candidates"][:top]:
        report = kernel["report"]
        cases = ", ".join(kernel["cases"][:3]) or "unindexed"
        lines.append(f"    {kernel['names'] or [pathlib.Path(kernel['path']).stem]}: "
                     f"unresolved={report['store_width_unresolved_count']} "
                     f"layout_conversions={report['store_layout_conversion_count']} cases={cases}")
    if not corpus["top_store_width_candidates"]:
        lines.append("    none")
    lines.append("  top hot-loop layout headroom:")
    for kernel in corpus["top_hot_loop_layout_headroom"][:top]:
        report = kernel["report"]
        hot_loop = report["hot_loop_load_width"]
        cases = ", ".join(kernel["cases"][:3]) or "unindexed"
        lines.append(f"    {kernel['names'] or [pathlib.Path(kernel['path']).stem]}: "
                     f"excess={hot_loop['excess_issue_count']} actual={hot_loop['load_issue_count']} "
                     f"minimum={hot_loop['minimum_16byte_issue_count']} "
                     f"widths={hot_loop['load_width_histogram_bytes']} cases={cases}")
    if not corpus["top_hot_loop_layout_headroom"]:
        lines.append("    none")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=pathlib.Path, help="AMDGCN text, hsaco code object, or Triton dump directory")
    parser.add_argument("--format", choices=("auto", "amdgcn", "hsaco"), default="auto", dest="input_format")
    parser.add_argument("--objdump", help="llvm-objdump path for hsaco input")
    parser.add_argument("--index", type=pathlib.Path, help="index.json mapping dump hashes to benchmark cases")
    parser.add_argument("--top", type=int, default=20, help="number of corpus candidates in the text report")
    parser.add_argument("--json", action="store_true", dest="as_json", help="emit machine-readable JSON")
    args = parser.parse_args(argv)
    if args.input.is_dir():
        report = analyze_dump_directory(args.input, args.index)
        output = json.dumps(report, indent=2, sort_keys=True) if args.as_json else _format_corpus_report(
            report, args.top)
    else:
        report = analyze_amdgcn_memory_width(_read_input(args.input, args.input_format, args.objdump))
        output = json.dumps(report, indent=2, sort_keys=True) if args.as_json else _format_report(report)
    sys.stdout.write(output + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
