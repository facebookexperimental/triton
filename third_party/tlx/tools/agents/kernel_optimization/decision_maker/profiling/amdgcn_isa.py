"""Static evidence from Triton's compiled AMDGCN for one AMD kernel.

``analyze_amdgcn`` turns a ``.amdgcn`` dump into a small, prompt-sized summary:
register/scratch/occupancy resources, the instruction mix of the hottest MFMA
loop, the instruction mix of the loop-boundary code (prologue/epilogue around
it), the source lines behind spills and byte-gathered operands, and factual
findings derived from those counts. ``collect_amdgcn_isa`` produces the dump by
rerunning a profile workload with Triton's kernel dump enabled.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_MAX_LOCATIONS = 6
_MAX_FINDINGS = 8

# Thresholds for findings. They only describe the evidence; Codex decides what
# to change.
_BYTE_GATHER_MIN_READS = 8
_NOP_CYCLES_PER_MFMA_HINT = 0.5

_LABEL = re.compile(r"^(\.LBB\d+_\d+):")
_DEPTH = re.compile(r"Loop Header: Depth=(\d+)")
_LOC = re.compile(r"^\s*\.loc\s+(\d+)\s+(\d+)\s+\d+")
_FILE = re.compile(r'^\s*\.file\s+(\d+)\s+"[^"]*"\s+"([^"]+)"')
_BRANCH = re.compile(r"^s_(?:cbranch_\w+|branch)\s+(\.LBB\d+_\d+)")
_NOP = re.compile(r"^s_nop\s+(\d+)")
_WAITCNT = re.compile(r"^s_waitcnt\b(.*)")
_COMMENT_RESOURCE = {
    "NumVgprs": "vgprs",
    "NumAgprs": "agprs",
    "TotalNumVgprs": "total_vgprs",
    "NumSgprs": "sgprs",
    "ScratchSize": "scratch_bytes",
    "Occupancy": "occupancy_waves_per_simd",
}
_METADATA_RESOURCE = {
    "vgpr_spill_count": "vgpr_spills",
    "sgpr_spill_count": "sgpr_spills",
    "group_segment_fixed_size": "lds_bytes",
    "private_segment_fixed_size": "scratch_bytes",
}
_BYTE_ASSEMBLY = ("v_perm_b32", "v_lshl_or_b32", "v_bfi_b32", "v_alignbyte_b32", "v_or3_b32")


@dataclass
class _Block:
    label: str
    depth: int = 0
    instructions: list[tuple[str, tuple[int, int] | None]] = field(default_factory=list)


def analyze_amdgcn(text: str, *, kernel_name: str | None = None) -> dict[str, Any]:
    """Summarize one kernel from a ``.amdgcn`` dump."""
    files = _file_table(text)
    body, metadata = _kernel_body(text, kernel_name)
    blocks = _blocks(body)
    resources = _resources(text, metadata)
    if not blocks:
        return {"resources": resources, "diagnostics": ["no instructions found"]}

    back_edges = _back_edge_targets(blocks)
    loop_blocks = [
        block
        for block in blocks
        if block.depth > 0 and block.label in back_edges and _count(block, "v_mfma") > 0
    ]
    hot = max(
        loop_blocks or [b for b in blocks if _count(b, "v_mfma") > 0] or blocks,
        key=lambda block: (_count(block, "v_mfma"), block.depth),
    )
    boundary = [
        block
        for block in blocks
        if block is not hot
        and block not in loop_blocks
        and any(_is_store(op) or _is_lds_dma(op) for op, _ in block.instructions)
    ]

    hot_mix = _mix(hot.instructions)
    boundary_instructions = [item for block in boundary for item in block.instructions]
    boundary_mix = _mix(boundary_instructions)
    locations = {
        "hot_loop_scratch_reloads": _locations(hot.instructions, files, _is_scratch_load),
        "hot_loop_byte_reads": _locations(hot.instructions, files, _is_byte_read),
        "boundary_scratch_reloads": _locations(boundary_instructions, files, _is_scratch_load),
    }
    reload_drains = _reload_drains_after_stores(boundary_instructions)
    summary: dict[str, Any] = {
        "resources": resources,
        "hot_loop": {"label": hot.label, "loop_depth": hot.depth, **hot_mix},
        "boundary": {"blocks": len(boundary), "reload_drains_after_stores": reload_drains, **boundary_mix},
        "source_locations": {key: value for key, value in locations.items() if value},
    }
    summary["findings"] = _findings(resources, hot_mix, boundary_mix, reload_drains, locations)
    return summary


def collect_amdgcn_isa(
    workload: Sequence[str],
    artifacts_dir: Path,
    *,
    kernel_filter: str,
    timeout_seconds: float = 600.0,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run ``workload`` once with Triton's kernel dump enabled and analyze ``kernel_filter``."""
    if not workload:
        raise ValueError("amdgcn_isa workload must not be empty")
    if not kernel_filter:
        return {"error": "amdgcn_isa requires a kernel name"}
    artifacts_dir = artifacts_dir.resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="amdgcn-isa-", dir=artifacts_dir))
    dump_dir = run_dir / "dump"
    env = dict(os.environ if environment is None else environment)
    env.update(
        {
            "TRITON_KERNEL_DUMP": "1",
            "TRITON_DUMP_DIR": str(dump_dir),
            "TRITON_ALWAYS_COMPILE": "1",
        }
    )
    try:
        completed = subprocess.run(
            [str(argument) for argument in workload],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
            cwd=run_dir,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"error": f"amdgcn_isa workload failed: {type(error).__name__}: {error}"}
    (run_dir / "stderr.txt").write_text(completed.stderr or "")
    artifacts = {"directory": str(run_dir), "stderr": str(run_dir / "stderr.txt")}
    if completed.returncode != 0:
        return {
            "error": f"amdgcn_isa workload failed with code {completed.returncode}",
            "artifacts": artifacts,
        }
    name = kernel_filter.removesuffix(".kd")
    dumps = sorted(dump_dir.rglob(f"{name}.amdgcn"))
    if not dumps:
        available = sorted({path.stem for path in dump_dir.rglob("*.amdgcn")})[:10]
        return {
            "error": f"no AMDGCN dump for kernel {name!r}",
            "available_kernels": available,
            "artifacts": artifacts,
        }
    chosen = max(dumps, key=lambda path: path.stat().st_mtime)
    kept = run_dir / chosen.name
    shutil.copyfile(chosen, kept)
    shutil.rmtree(dump_dir, ignore_errors=True)
    analysis = analyze_amdgcn(kept.read_text(errors="replace"), kernel_name=name)
    return {
        "tool": "amdgcn_isa",
        "schema_version": 1,
        "kernel": name,
        **analysis,
        "artifacts": {**artifacts, "amdgcn": str(kept)},
    }


def _file_table(text: str) -> dict[int, str]:
    table: dict[int, str] = {}
    for line in text.splitlines():
        match = _FILE.match(line)
        if match:
            table[int(match.group(1))] = match.group(2)
    return table


def _kernel_body(text: str, kernel_name: str | None) -> tuple[list[str], str]:
    lines = text.splitlines()
    start = 0
    if kernel_name:
        for index, line in enumerate(lines):
            if line.startswith(f"{kernel_name}:"):
                start = index + 1
                break
    end = next(
        (index for index in range(start, len(lines)) if lines[index].startswith(".Lfunc_end")),
        len(lines),
    )
    metadata_start = next(
        (index for index, line in enumerate(lines) if line.strip() == "amdhsa.kernels:"),
        len(lines),
    )
    return lines[start:end], "\n".join(lines[metadata_start:])


def _blocks(lines: Sequence[str]) -> list[_Block]:
    blocks = [_Block(label="entry")]
    pending_depth = 0
    location: tuple[int, int] | None = None
    for raw in lines:
        label = _LABEL.match(raw)
        if label:
            depth = _DEPTH.search(raw)
            blocks.append(_Block(label=label.group(1), depth=int(depth.group(1)) if depth else pending_depth))
            pending_depth = 0
            continue
        depth = _DEPTH.search(raw)
        if depth:
            # "; =>  This Loop Header" can trail the label on its own line.
            if blocks[-1].instructions:
                pending_depth = int(depth.group(1))
            else:
                blocks[-1].depth = int(depth.group(1))
            continue
        loc = _LOC.match(raw)
        if loc:
            location = (int(loc.group(1)), int(loc.group(2)))
            continue
        stripped = raw.strip()
        if not stripped or stripped.startswith((";", ".", "%")) or stripped.endswith(":"):
            continue
        blocks[-1].instructions.append((stripped.split(";", 1)[0].strip(), location))
    return [block for block in blocks if block.instructions]


def _back_edge_targets(blocks: Sequence[_Block]) -> set[str]:
    order = {block.label: index for index, block in enumerate(blocks)}
    targets: set[str] = set()
    for index, block in enumerate(blocks):
        for op, _ in block.instructions:
            match = _BRANCH.match(op)
            if match and order.get(match.group(1), len(blocks)) <= index:
                targets.add(match.group(1))
    return targets


def _resources(text: str, metadata: str) -> dict[str, int]:
    resources: dict[str, int] = {}
    for key, name in _COMMENT_RESOURCE.items():
        match = re.search(rf"^; {key}: (\d+)", text, re.MULTILINE)
        if match:
            resources[name] = int(match.group(1))
    for key, name in _METADATA_RESOURCE.items():
        match = re.search(rf"\.{key}:\s+(\d+)", metadata)
        if match and name not in resources:
            resources[name] = int(match.group(1))
    lds = re.search(r"^; LDSByteSize: (\d+)", text, re.MULTILINE)
    if lds and "lds_bytes" not in resources:
        resources["lds_bytes"] = int(lds.group(1))
    return resources


def _mix(instructions: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    ops = [op for op, _ in instructions]
    mnemonics = Counter(op.split()[0] for op in ops if op)
    ds_reads = {
        name.removeprefix("ds_read_"): count
        for name, count in sorted(mnemonics.items())
        if name.startswith("ds_read")
    }
    waits = [_WAITCNT.match(op).group(1) for op in ops if _WAITCNT.match(op)]
    nop_cycles = sum(int(_NOP.match(op).group(1)) + 1 for op in ops if _NOP.match(op))
    return {
        "instructions": len(ops),
        "mfma": sum(count for name, count in mnemonics.items() if name.startswith("v_mfma")),
        "scaled_mfma": sum(count for name, count in mnemonics.items() if name.startswith("v_mfma_scale")),
        "ds_read_by_width": ds_reads,
        "ds_write": sum(count for name, count in mnemonics.items() if name.startswith("ds_write")),
        "lds_dma_loads": sum(1 for op in ops if _is_lds_dma(op)),
        "global_loads": sum(1 for op in ops if _is_global_load(op) and not _is_lds_dma(op)),
        "global_stores": sum(1 for op in ops if _is_store(op)),
        "scratch_loads": sum(1 for op in ops if _is_scratch_load(op)),
        "scratch_stores": sum(1 for op in ops if op.startswith("scratch_store")),
        "waitcnt": len(waits),
        "waitcnt_vmcnt0": sum(1 for wait in waits if "vmcnt(0)" in wait),
        "waitcnt_lgkmcnt0": sum(1 for wait in waits if "lgkmcnt(0)" in wait),
        "barriers": mnemonics.get("s_barrier", 0),
        "nops": mnemonics.get("s_nop", 0),
        "nop_cycles": nop_cycles,
        "byte_assembly_valu": sum(mnemonics.get(name, 0) for name in _BYTE_ASSEMBLY),
        "m0_writes": sum(1 for op in ops if op.startswith("s_mov_b32 m0")),
    }


def _reload_drains_after_stores(instructions: Sequence[tuple[str, Any]]) -> int:
    """Scratch reloads followed by vmcnt(0) while earlier global stores may be in flight."""
    drains = 0
    seen_store = False
    pending_reload = False
    for op, _ in instructions:
        if _is_store(op):
            seen_store = True
        elif _is_scratch_load(op):
            pending_reload = True
        elif pending_reload and op.startswith("s_waitcnt") and "vmcnt(0)" in op:
            if seen_store:
                drains += 1
            pending_reload = False
    return drains


def _locations(
    instructions: Sequence[tuple[str, tuple[int, int] | None]],
    files: Mapping[int, str],
    predicate: Any,
) -> list[str]:
    counts = Counter(
        f"{files.get(location[0], location[0])}:{location[1]}"
        for op, location in instructions
        if location is not None and location[1] > 0 and predicate(op)
    )
    return [f"{place} x{count}" for place, count in counts.most_common(_MAX_LOCATIONS)]


def _findings(
    resources: Mapping[str, int],
    hot: Mapping[str, Any],
    boundary: Mapping[str, Any],
    reload_drains: int,
    locations: Mapping[str, list[str]],
) -> list[str]:
    findings: list[str] = []
    vgprs = resources.get("total_vgprs", resources.get("vgprs"))
    spills = resources.get("vgpr_spills", 0) + resources.get("sgpr_spills", 0)
    if spills:
        ceiling = " at the 256-VGPR ceiling" if vgprs == 256 else ""
        findings.append(
            f"{resources.get('vgpr_spills', 0)} VGPR + {resources.get('sgpr_spills', 0)} SGPR spills"
            f"{ceiling} ({resources.get('scratch_bytes', 0)} scratch bytes); "
            "new values kept live across the hot loop will spill further"
        )
    if hot.get("scratch_loads"):
        where = ", ".join(locations.get("hot_loop_scratch_reloads", [])[:3])
        findings.append(f"hot loop has {hot['scratch_loads']} scratch reloads ({where})")
    if reload_drains:
        where = ", ".join(locations.get("boundary_scratch_reloads", [])[:3])
        findings.append(
            f"loop boundary: {reload_drains} scratch reloads are each followed by s_waitcnt vmcnt(0) "
            f"after {boundary.get('global_stores', 0)} global/buffer stores, so the reload also waits "
            f"for those stores to complete ({where})"
        )
    byte_reads = sum(
        count for width, count in hot.get("ds_read_by_width", {}).items() if width in {"u8", "i8", "u16", "i16"}
    )
    if byte_reads >= _BYTE_GATHER_MIN_READS and hot.get("byte_assembly_valu"):
        where = ", ".join(locations.get("hot_loop_byte_reads", [])[:3])
        findings.append(
            f"hot loop: {byte_reads} sub-dword ds_reads + {hot['byte_assembly_valu']} byte-assembly VALU "
            f"per {hot.get('mfma', 0)} MFMA; operand bytes packed into one register are not contiguous "
            f"in LDS ({where})"
        )
    if hot.get("mfma") and hot.get("nop_cycles", 0) / hot["mfma"] >= _NOP_CYCLES_PER_MFMA_HINT:
        findings.append(
            f"hot loop: {hot['nop_cycles']} s_nop cycles per {hot['mfma']} MFMA (dependency stalls on "
            "values produced just before MFMA issue)"
        )
    occupancy = resources.get("occupancy_waves_per_simd")
    if occupancy is not None:
        findings.append(f"occupancy {occupancy} waves/SIMD with {vgprs} VGPRs")
    return findings[:_MAX_FINDINGS]


def _count(block: _Block, prefix: str) -> int:
    return sum(1 for op, _ in block.instructions if op.startswith(prefix))


def _is_store(op: str) -> bool:
    return op.startswith(("buffer_store", "global_store", "flat_store"))


def _is_global_load(op: str) -> bool:
    return op.startswith(("buffer_load", "global_load", "flat_load"))


def _is_lds_dma(op: str) -> bool:
    return _is_global_load(op) and (" lds" in op or op.startswith("global_load_lds"))


def _is_scratch_load(op: str) -> bool:
    return op.startswith("scratch_load")


def _is_byte_read(op: str) -> bool:
    return op.startswith(("ds_read_u8", "ds_read_i8", "ds_read_u16", "ds_read_i16"))


__all__ = ["analyze_amdgcn", "collect_amdgcn_isa"]
