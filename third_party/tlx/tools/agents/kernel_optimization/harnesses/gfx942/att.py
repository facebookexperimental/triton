"""rocprofv3 Advanced Thread Trace (ATT) driver for the gfx942 harnesses.

Invoked by ``harness.profile()``, which the optimizer calls once per candidate
per round -- automatically, not on request from the prompt.

ATT is the only profiler on this part that reports *per-instruction* hitcount,
latency, stall and idle. That is the raw data the optimization loop needs: a
kernel-level TFLOP/s number tells you that you are 0.83x aten but not which
instruction class is holding the pipe.

Two structural facts drive the design here.

**ATT cannot be collected in-process.** It needs the application launched under
``rocprofv3 -- <app>``. The harness ``profile()`` hook, by contrast, is called
inside an already-running worker that is holding a live candidate module. So
``collect()`` re-execs :mod:`att_child` as a fresh child under rocprofv3 rather
than tracing the caller.

**The ATT CLI is conditionally present.** Older rocprofv3 drivers hide the
entire ``--att*`` option group unless they find a legacy decoder on
``ROCPROF_ATT_LIBRARY_PATH``. Newer drivers bundle
``librocprof-trace-decoder`` and expose ATT directly. ``TLX_ROCPROFV3`` can
select a compatible driver when the system default is older than the runtime.
When ATT is absent, :func:`collect` degrades to counters rather than failing
the optimization round.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

# The exact names rocprofv3's search_path() regex accepts. Only "trace" is
# needed for the stats CSV we consume; the others enable extra parse modes.
LEGACY_DECODER_NAMES = (
    "libatt_decoder_trace.so",
    "libatt_decoder_summary.so",
    "libatt_decoder_debug.so",
    "libatt_decoder_testing.so",
)
DECODER_GLOBS = (*LEGACY_DECODER_NAMES, "librocprof-trace-decoder.so*")

# Counter fallback for when ATT is unavailable. Deliberately small: rocprofv3
# fails the whole job if the set cannot be collected in one pass, so this is
# scoped to wave occupancy, MFMA utilisation and memory stalls.
FALLBACK_PMC = (
    "SQ_WAVES",
    "SQ_BUSY_CYCLES",
    "SQ_WAIT_ANY",
    "SQ_INSTS_MFMA",
    "SQ_INSTS_VALU",
    "SQ_INSTS_LDS",
)

_HELP_CACHE: dict[str, Any] | None = None


def _rocprofv3() -> str | None:
    override = os.environ.get("TLX_ROCPROFV3")
    if override:
        return shutil.which(override)
    return shutil.which("rocprofv3")


def capability() -> dict[str, Any]:
    """Probe whether this rocprofv3 exposes ATT, and if not, say why.

    Cached per process: it shells ``rocprofv3 --help``, which is not free and
    cannot change underneath a single worker.
    """
    global _HELP_CACHE
    if _HELP_CACHE is not None:
        return _HELP_CACHE
    binary = _rocprofv3()
    if binary is None:
        _HELP_CACHE = {
            "att_available":
            False,
            "reason": (f"TLX_ROCPROFV3={os.environ['TLX_ROCPROFV3']!r} is not executable"
                       if os.environ.get("TLX_ROCPROFV3") else "rocprofv3 not on PATH"),
            "searched": [],
        }
        return _HELP_CACHE
    try:
        help_text = subprocess.run([binary, "--help"], capture_output=True, text=True, timeout=60, check=False).stdout
    except (OSError, subprocess.SubprocessError) as error:
        _HELP_CACHE = {
            "att_available": False,
            "reason": f"could not run rocprofv3 --help: {error}",
            "searched": [],
        }
        return _HELP_CACHE
    available = "--att" in help_text
    searched = _decoder_search_path()
    found = sorted(
        str(path)
        for directory in searched
        for pattern in DECODER_GLOBS
        for path in directory.glob(pattern)
        if path.is_file())
    _HELP_CACHE = {
        "att_available": available,
        "rocprofv3": binary,
        "decoders_found": found,
        "searched": [str(p) for p in searched],
    }
    if not available:
        _HELP_CACHE["reason"] = (f"{binary} does not expose ATT. Set TLX_ROCPROFV3 to a newer "
                                 "rocprofv3 with librocprof-trace-decoder, or install the legacy "
                                 "libatt_decoder_trace.so at the top level of a directory on "
                                 "ROCPROF_ATT_LIBRARY_PATH.")
    return _HELP_CACHE


def _decoder_search_path() -> list[Path]:
    """Mirror rocprofv3's own search order, so our diagnostics match its behaviour."""
    explicit = os.environ.get("ROCPROF_ATT_LIBRARY_PATH")
    if explicit:
        raw = explicit.split(":")
    else:
        raw = os.environ.get("LD_LIBRARY_PATH", "").split(":")
        binary = _rocprofv3()
        if binary is not None:
            rocm_root = Path(binary).resolve().parent.parent
            raw.append(str(rocm_root / "lib"))
    seen: list[Path] = []
    for entry in raw:
        if not entry:
            continue
        path = Path(entry)
        if path not in seen:
            seen.append(path)
    return seen


def collect(
    *,
    kernel_path: Path,
    case: dict[str, Any],
    output_dir: Path,
    entry_point: str = "matmul",
    kernel_regex: str = ".*matmul.*",
    warmup: int = 3,
    pin_config: dict[str, Any] | None = None,
    timeout_s: float = 900.0,
) -> dict[str, Any]:
    """Trace one dispatch of ``kernel_path`` and return a compact summary.

    Falls back to counter collection when ATT is unavailable, and reports which
    mode ran so a reader never mistakes counters for a thread trace.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=output_dir))
    cap = capability()
    if cap["att_available"]:
        result = _run(
            mode="att",
            kernel_path=kernel_path,
            case=case,
            output_dir=run_dir,
            entry_point=entry_point,
            kernel_regex=kernel_regex,
            warmup=warmup,
            pin_config=pin_config,
            timeout_s=timeout_s,
        )
    else:
        result = _run(
            mode="counters",
            kernel_path=kernel_path,
            case=case,
            output_dir=run_dir,
            entry_point=entry_point,
            kernel_regex=kernel_regex,
            warmup=warmup,
            pin_config=pin_config,
            timeout_s=timeout_s,
        )
        result["att_unavailable_reason"] = cap.get("reason", "")
    result["capability"] = cap
    return result


def _run(
    *,
    mode: str,
    kernel_path: Path,
    case: dict[str, Any],
    output_dir: Path,
    entry_point: str,
    kernel_regex: str,
    warmup: int,
    pin_config: dict[str, Any] | None,
    timeout_s: float,
) -> dict[str, Any]:
    binary = _rocprofv3()
    if binary is None:
        return {"mode": "unavailable", "error": "rocprofv3 not on PATH"}
    child = Path(__file__).with_name("att_child.py")

    profiler_args = [binary, "-d", str(output_dir), "-o", "run"]
    if mode == "att":
        profiler_args += [
            "--att",
            # Keep a dispatch record beside the trace. Among other things, its
            # global dispatch id lets us audit the iteration-range selection.
            "--kernel-trace",
            # gfx9-only shorthand that turns on the SQ perfcounter tokens; 8 is
            # AMD's own Instinct example value.
            "--att-activity",
            "8",
            # CU 1 produced an activity-only, header-only trace for the
            # one-workgroup-per-CU 256x256 tile on MI300X. CU 0 produced the
            # full instruction trace for both protected GEMM shapes.
            "--att-target-cu",
            os.environ.get("TLX_ATT_TARGET_CU", "0"),
            "--att-shader-engine-mask",
            "0x1",
            "--att-simd-select",
            "0xF",  # gfx9 default: all four SIMDs
            "--kernel-include-regex",
            kernel_regex,
            # One dispatch only. With the config pinned there is no autotune
            # search, so dispatch (warmup + 1) is the steady-state call.
            "--kernel-iteration-range",
            f"[{warmup + 1}]",
        ]
    else:
        profiler_args += [
            "--kernel-trace",
            "--stats",
            "--pmc",
            *FALLBACK_PMC,
            "--kernel-include-regex",
            kernel_regex,
            "--output-format",
            "csv",
        ]

    environment = os.environ.copy()
    environment["TLX_ATT_CASE"] = json.dumps(case)
    environment["TLX_ATT_KERNEL"] = str(kernel_path)
    environment["TLX_ATT_ENTRY"] = entry_point
    environment["TLX_ATT_WARMUP"] = str(warmup)
    if pin_config:
        environment["TLX_AGENT_PIN_CONFIG"] = json.dumps(pin_config)

    command = [*profiler_args, "--", sys.executable, str(child)]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=environment,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"mode": mode, "error": f"rocprofv3 timed out after {timeout_s:.0f}s"}

    summary: dict[str, Any] = {
        "mode": mode,
        "returncode": completed.returncode,
        "output_dir": str(output_dir),
        "command": " ".join(command),
    }
    if completed.returncode != 0:
        summary["error"] = (completed.stderr or completed.stdout)[-4000:]
        return summary

    if mode == "att":
        summary.update(_summarize_att(output_dir))
    else:
        summary.update(_summarize_counters(output_dir))
    summary["viewer_dirs"] = [
        str(p.relative_to(output_dir)) for p in sorted(output_dir.rglob("ui_output_agent_*_dispatch_*")) if p.is_dir()
    ]
    summary["traced_dispatches"] = len(summary["viewer_dirs"])
    if mode == "att" and summary["traced_dispatches"] != 1:
        summary["collection_error"] = ("expected exactly one ATT dispatch, found "
                                       f"{summary['traced_dispatches']}")
    return summary


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_TOP_N_SITES = 20
_TOP_N_OPCODES = 15
_ATT_REQUIRED_COLUMNS = {
    "vaddr",
    "instruction",
    "hitcount",
    "latency",
    "stall",
    "idle",
    "source",
}
_ATT_NUMERIC_COLUMNS = ("hitcount", "latency", "stall", "idle")


def _summarize_att(output_dir: Path) -> dict[str, Any]:
    """Roll the per-instruction stats CSV up into something promptable.

    The raw CSV is one row per instruction address; a 256x256 GEMM hot loop
    produces thousands. Two views are kept: an opcode-class rollup (where is
    the stall going?) and the worst individual sites (which instruction?).
    Everything else stays on disk for the Compute Viewer.
    """
    stats_files = sorted(output_dir.rglob("*stats*.csv"))
    instruction_files = [path for path in stats_files if _has_columns(path, _ATT_REQUIRED_COLUMNS)]
    if not instruction_files:
        return {
            "parse_error": "no stats CSV with the required ATT columns found",
            "csv_files": [str(p.relative_to(output_dir)) for p in stats_files],
            "csv_headers": {str(path.relative_to(output_dir)): _header(path)
                            for path in stats_files},
        }

    rows: list[dict[str, Any]] = []
    for path in instruction_files:
        with path.open(newline="") as stream:
            for raw in csv.DictReader(stream):
                normalized = {(key or "").strip().lower(): (value or "").strip() for key, value in raw.items()}
                rows.append(normalized)

    if not rows:
        return {
            "parse_error": "ATT stats CSV contains a header but no instruction rows",
            "csv_files": [str(p.relative_to(output_dir)) for p in instruction_files],
        }

    invalid_numeric: list[dict[str, str]] = []
    for row in rows:
        for field in _ATT_NUMERIC_COLUMNS:
            value = row.get(field, "")
            try:
                float(value or 0.0)
            except ValueError:
                invalid_numeric.append({"field": field, "value": value})
                if len(invalid_numeric) == 5:
                    break
        if len(invalid_numeric) == 5:
            break
    if invalid_numeric:
        return {
            "parse_error": "ATT stats CSV contains non-numeric metric values",
            "examples": invalid_numeric,
            "csv_files": [str(p.relative_to(output_dir)) for p in instruction_files],
        }

    def number(row: dict[str, Any], key: str) -> float:
        try:
            return float(row.get(key) or 0.0)
        except ValueError as error:  # validated above; keep the invariant local
            raise AssertionError(f"invalid ATT {key}: {row.get(key)!r}") from error

    totals = {field: sum(number(row, field) for row in rows) for field in ("hitcount", "latency", "stall", "idle")}

    by_opcode: dict[str,
                    dict[str,
                         float]] = defaultdict(lambda: {"hitcount": 0.0, "latency": 0.0, "stall": 0.0, "idle": 0.0})
    for row in rows:
        opcode = _opcode_class(row.get("instruction", ""))
        for field in ("hitcount", "latency", "stall", "idle"):
            by_opcode[opcode][field] += number(row, field)

    stall_total = totals["stall"] or 1.0
    opcode_rollup = sorted(
        ({
            "opcode": opcode,
            "stall": values["stall"],
            "stall_pct": 100.0 * values["stall"] / stall_total,
            "latency": values["latency"],
            "hitcount": values["hitcount"],
        } for opcode, values in by_opcode.items()),
        key=lambda entry: entry["stall"],
        reverse=True,
    )[:_TOP_N_OPCODES]

    top_sites = sorted(rows, key=lambda row: number(row, "stall"), reverse=True)[:_TOP_N_SITES]
    return {
        "totals":
        totals,
        "stall_by_opcode":
        opcode_rollup,
        "top_stall_sites": [{
            "vaddr": row.get("vaddr", ""),
            "instruction": row.get("instruction", ""),
            "hitcount": number(row, "hitcount"),
            "latency": number(row, "latency"),
            "stall": number(row, "stall"),
            "idle": number(row, "idle"),
            "source": row.get("source", ""),
        } for row in top_sites],
        "instruction_rows":
        len(rows),
        "csv_files": [str(p.relative_to(output_dir)) for p in instruction_files],
    }


def _has_columns(path: Path, required: set[str]) -> bool:
    return required.issubset(set(_header(path)))


def _header(path: Path) -> list[str]:
    try:
        with path.open(newline="") as stream:
            header = next(csv.reader(stream), [])
    except (OSError, StopIteration):
        return []
    return [column.strip().lower() for column in header]


_OPCODE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)")


def _opcode_class(instruction: str) -> str:
    """Bucket an instruction to its mnemonic, then to a coarse family.

    Families are what an optimization decision actually turns on -- MFMA versus
    LDS versus global versus wait -- so the rollup groups by family and keeps
    the mnemonic only when it does not fit one.
    """
    match = _OPCODE_RE.match(instruction)
    if not match:
        return "unknown"
    mnemonic = match.group(1).lower()
    for prefix, family in (
        ("v_mfma", "mfma"),
        ("ds_read", "lds_read"),
        ("ds_write", "lds_write"),
        ("buffer_load", "global_load"),
        ("global_load", "global_load"),
        ("flat_load", "global_load"),
        ("buffer_store", "global_store"),
        ("global_store", "global_store"),
        ("s_waitcnt", "waitcnt"),
        ("s_barrier", "barrier"),
        ("v_accvgpr", "accvgpr_move"),
        ("s_setprio", "setprio"),
    ):
        if mnemonic.startswith(prefix):
            return family
    if mnemonic.startswith("v_"):
        return "valu_other"
    if mnemonic.startswith("s_"):
        return "salu_other"
    return mnemonic


def _summarize_counters(output_dir: Path) -> dict[str, Any]:
    """Fallback summary: whatever --pmc produced, plus kernel durations."""
    result: dict[str, Any] = {}
    counter_files = sorted(output_dir.rglob("*counter_collection*.csv"))
    if counter_files:
        aggregate: dict[str, float] = defaultdict(float)
        for path in counter_files:
            with path.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    name = (row.get("Counter_Name") or "").strip()
                    if not name:
                        continue
                    try:
                        aggregate[name] += float(row.get("Counter_Value") or 0.0)
                    except ValueError:
                        continue
        result["counters"] = dict(sorted(aggregate.items()))
        result["counter_files"] = [str(p.relative_to(output_dir)) for p in counter_files]
    stats_files = sorted(output_dir.rglob("*kernel_stats*.csv"))
    if stats_files:
        with stats_files[0].open(newline="") as stream:
            result["kernel_stats"] = list(csv.DictReader(stream))[:10]
    if not result:
        result["parse_error"] = "no counter or kernel-stats CSV found"
        result["files"] = [str(p.relative_to(output_dir)) for p in sorted(output_dir.rglob("*.csv"))]
    return result
