"""gfx950 TLX latency/throughput microbenchmarks.

This is a pragmatic first-pass device-timed suite for instruction/model
calibration. Kernels use ``tlx.clock64`` around the measured region and return
raw target timer ticks from synchronized host launches. The baseline row is a
diagnostic only; it is not subtracted from primary metrics.

Coverage:
  - baseline: clock64 + loop overhead
  - VALU: runtime-seeded dependent recurrence and four explicit independent
    scalar recurrences
  - LDS: runtime-seeded pure ``tlx.local_gather`` pointer chase over a
    preinitialized 1D i32 LDS table, plus independent gather streams
  - GLOBAL: dependent ``tl.load`` latency, plus direct-to-LDS composite via
    ``tlx.buffer_load_to_local`` commit/wait + ``tlx.local_load``
  - MFMA: gfx9-style fp16 ``tl.dot`` shapes with BLOCK_K=32 and
    ``matrix_instr_nonkdim=16`` launch metadata

Example:
  python third_party/tlx/tools/microbench/amd_latency.py --bench all --niter 256 --reps 5
  python third_party/tlx/tools/microbench/amd_latency.py --bench valu,lds --json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


DEFAULT_NITER = 256
DEFAULT_WARMUP = 2
DEFAULT_REPS = 5

# Calibrated on gfx950 by comparing a long s_memtime-bracketed runtime loop
# against host monotonic time. rocminfo reports a 2.2 GHz maximum shader clock.
SMEMTIME_HZ = 2.183e9
GFX950_MAX_SCLK_HZ = 2.2e9
CYCLES_PER_TICK = GFX950_MAX_SCLK_HZ / SMEMTIME_HZ

ALL_BENCHES = ("baseline", "valu", "lds", "global", "mfma")


@triton.jit
def _k_baseline(seed_ptr, out_ptr, check_ptr, NITER):
    tid = tlx.thread_id(0)
    x = tl.load(seed_ptr).to(tl.float32)
    t0 = tlx.clock64()
    for _ in tl.range(0, NITER, loop_unroll_factor=1, num_stages=1):
        x += 1.0
    t1 = tlx.clock64()
    if tid == 0:
        tl.store(out_ptr, t1 - t0)
        tl.store(check_ptr, x)


@triton.jit
def _k_valu_dependent(seed_ptr, out_ptr, check_ptr, NITER):
    tid = tlx.thread_id(0)
    x = tl.load(seed_ptr).to(tl.float32)
    mul = tl.load(seed_ptr + 1).to(tl.float32)
    inc = tl.load(seed_ptr + 2).to(tl.float32)
    t0 = tlx.clock64()
    for _ in tl.range(0, NITER, loop_unroll_factor=1, num_stages=1):
        x = x * mul + inc
    t1 = tlx.clock64()
    if tid == 0:
        tl.store(out_ptr, t1 - t0)
        tl.store(check_ptr, x)


@triton.jit
def _k_valu_independent_x4(seed_ptr, out_ptr, check_ptr, NITER):
    tid = tlx.thread_id(0)
    x0 = tl.load(seed_ptr + 0).to(tl.float32)
    x1 = tl.load(seed_ptr + 1).to(tl.float32)
    x2 = tl.load(seed_ptr + 2).to(tl.float32)
    x3 = tl.load(seed_ptr + 3).to(tl.float32)
    mul0 = tl.load(seed_ptr + 4).to(tl.float32)
    mul1 = tl.load(seed_ptr + 5).to(tl.float32)
    mul2 = tl.load(seed_ptr + 6).to(tl.float32)
    mul3 = tl.load(seed_ptr + 7).to(tl.float32)
    inc0 = tl.load(seed_ptr + 8).to(tl.float32)
    inc1 = tl.load(seed_ptr + 9).to(tl.float32)
    inc2 = tl.load(seed_ptr + 10).to(tl.float32)
    inc3 = tl.load(seed_ptr + 11).to(tl.float32)
    t0 = tlx.clock64()
    for _ in tl.range(0, NITER, loop_unroll_factor=1, num_stages=1):
        x0 = x0 * mul0 + inc0
        x1 = x1 * mul1 + inc1
        x2 = x2 * mul2 + inc2
        x3 = x3 * mul3 + inc3
    t1 = tlx.clock64()
    if tid == 0:
        tl.store(out_ptr, t1 - t0)
        tl.store(check_ptr, x0 + x1 + x2 + x3)


@triton.jit
def _k_lds_dependent_gather_chase(table_ptr, out_ptr, check_ptr, NITER):
    tid = tlx.thread_id(0)
    offsets = tl.arange(0, 64)
    table = tlx.local_alloc((64,), tl.int32, 1)
    tlx.local_store(table[0], tl.load(table_ptr + offsets))
    tl.debug_barrier()

    idx = tl.load(table_ptr + 64).to(tl.int32)
    t0 = tlx.clock64()
    for _ in tl.range(0, NITER, loop_unroll_factor=1, num_stages=1):
        gathered = tlx.local_gather(table[0], tl.full((1,), idx, tl.int32), 0)
        idx = tl.sum(gathered, axis=0).to(tl.int32)
    t1 = tlx.clock64()
    if tid == 0:
        tl.store(out_ptr, t1 - t0)
        tl.store(check_ptr, idx.to(tl.float32))


@triton.jit
def _k_lds_independent_gather_x4(table_ptr, out_ptr, check_ptr, NITER):
    tid = tlx.thread_id(0)
    offsets = tl.arange(0, 64)
    table = tlx.local_alloc((64,), tl.int32, 1)
    tlx.local_store(table[0], tl.load(table_ptr + offsets))
    tl.debug_barrier()

    idx0 = tl.load(table_ptr + 64).to(tl.int32)
    idx1 = tl.load(table_ptr + 65).to(tl.int32)
    idx2 = tl.load(table_ptr + 66).to(tl.int32)
    idx3 = tl.load(table_ptr + 67).to(tl.int32)
    t0 = tlx.clock64()
    for _ in tl.range(0, NITER, loop_unroll_factor=1, num_stages=1):
        idx0 = tl.sum(tlx.local_gather(table[0], tl.full((1,), idx0, tl.int32), 0), axis=0).to(tl.int32)
        idx1 = tl.sum(tlx.local_gather(table[0], tl.full((1,), idx1, tl.int32), 0), axis=0).to(tl.int32)
        idx2 = tl.sum(tlx.local_gather(table[0], tl.full((1,), idx2, tl.int32), 0), axis=0).to(tl.int32)
        idx3 = tl.sum(tlx.local_gather(table[0], tl.full((1,), idx3, tl.int32), 0), axis=0).to(tl.int32)
    t1 = tlx.clock64()
    if tid == 0:
        tl.store(out_ptr, t1 - t0)
        tl.store(check_ptr, (idx0 + idx1 + idx2 + idx3).to(tl.float32))


@triton.jit
def _k_global_tl_load_dependent(src_ptr, out_ptr, check_ptr, NITER):
    tid = tlx.thread_id(0)
    idx = tl.load(src_ptr + 1).to(tl.int64)
    acc = tl.full((), 0, tl.int64)
    t0 = tlx.clock64()
    for _ in tl.range(0, NITER, loop_unroll_factor=1, num_stages=1):
        # src contains a self pointer at 0; this creates a serialized global load
        # dependency without changing the address range.
        idx = tl.load(src_ptr + idx).to(tl.int64)
        acc += idx
    t1 = tlx.clock64()
    if tid == 0:
        tl.store(out_ptr, t1 - t0)
        tl.store(check_ptr, acc + idx)


@triton.jit
def _k_global_direct_to_lds_composite(src_ptr, out_ptr, check_ptr, NITER):
    tid = tlx.thread_id(0)
    smem = tlx.local_alloc((32, 32), tl.float16, 1)
    offs = tl.arange(0, 32)[:, None] * 32 + tl.arange(0, 32)[None, :]
    acc = tl.zeros((32, 32), tl.float32)
    t0 = tlx.clock64()
    for _ in tl.range(0, NITER, loop_unroll_factor=1, num_stages=1):
        tok = tlx.buffer_load_to_local(smem[0], src_ptr, offs)
        tlx.async_load_commit_group([tok])
        tlx.async_load_wait_group(0)
        v = tlx.local_load(smem[0], relaxed=True)
        acc += v
    t1 = tlx.clock64()
    s = tl.sum(tl.sum(acc, axis=1), axis=0)
    if tid == 0:
        tl.store(out_ptr, t1 - t0)
        tl.store(check_ptr, s)


@triton.jit
def _k_mfma_dependent(a_ptr, b_ptr, out_ptr, check_ptr, NITER):
    tid = tlx.thread_id(0)
    offs_m = tl.arange(0, 32)
    offs_n = tl.arange(0, 32)
    offs_k = tl.arange(0, 32)
    a = tl.load(a_ptr + offs_m[:, None] * 32 + offs_k[None, :])
    b = tl.load(b_ptr + offs_k[:, None] * 32 + offs_n[None, :])
    acc = tl.zeros((32, 32), tl.float32)
    t0 = tlx.clock64()
    for _ in tl.range(0, NITER, loop_unroll_factor=1, num_stages=1):
        acc = tl.dot(a, b, acc, allow_tf32=False)
    t1 = tlx.clock64()
    s = tl.sum(tl.sum(acc, axis=1), axis=0)
    if tid == 0:
        tl.store(out_ptr, t1 - t0)
        tl.store(check_ptr, s)


@triton.jit
def _k_mfma_independent_x4(a_ptr, b_ptr, out_ptr, check_ptr, NITER):
    tid = tlx.thread_id(0)
    offs_m = tl.arange(0, 32)
    offs_n = tl.arange(0, 32)
    offs_k = tl.arange(0, 32)
    a0 = tl.load(a_ptr + offs_m[:, None] * 32 + offs_k[None, :])
    a1 = tl.load(a_ptr + 1024 + offs_m[:, None] * 32 + offs_k[None, :])
    a2 = tl.load(a_ptr + 2048 + offs_m[:, None] * 32 + offs_k[None, :])
    a3 = tl.load(a_ptr + 3072 + offs_m[:, None] * 32 + offs_k[None, :])
    b0 = tl.load(b_ptr + offs_k[:, None] * 32 + offs_n[None, :])
    b1 = tl.load(b_ptr + 1024 + offs_k[:, None] * 32 + offs_n[None, :])
    b2 = tl.load(b_ptr + 2048 + offs_k[:, None] * 32 + offs_n[None, :])
    b3 = tl.load(b_ptr + 3072 + offs_k[:, None] * 32 + offs_n[None, :])
    acc0 = tl.zeros((32, 32), tl.float32)
    acc1 = tl.zeros((32, 32), tl.float32)
    acc2 = tl.zeros((32, 32), tl.float32)
    acc3 = tl.zeros((32, 32), tl.float32)
    t0 = tlx.clock64()
    for _ in tl.range(0, NITER, loop_unroll_factor=1, num_stages=1):
        acc0 = tl.dot(a0, b0, acc0, allow_tf32=False)
        acc1 = tl.dot(a1, b1, acc1, allow_tf32=False)
        acc2 = tl.dot(a2, b2, acc2, allow_tf32=False)
        acc3 = tl.dot(a3, b3, acc3, allow_tf32=False)
    t1 = tlx.clock64()
    acc = acc0 + acc1 + acc2 + acc3
    s = tl.sum(tl.sum(acc, axis=1), axis=0)
    if tid == 0:
        tl.store(out_ptr, t1 - t0)
        tl.store(check_ptr, s)


@dataclass(frozen=True)
class Case:
    name: str
    family: str
    unit: str
    kernel: Any
    make_args: Callable[[], list[Any]]
    ops_per_iter: int = 1
    baseline_key: str = "baseline.loop"
    num_warps: int = 4
    launch_meta: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    expected_check: Callable[[int], float] | None = None


def _active_device() -> torch.device:
    return triton.runtime.driver.active.get_active_torch_device()


def _device_info() -> dict[str, Any]:
    dev = _active_device()
    props = torch.cuda.get_device_properties(dev)
    return {
        "name": torch.cuda.get_device_name(dev),
        "device": str(dev),
        "hip": torch.version.hip,
        "gcn_arch_name": getattr(props, "gcnArchName", ""),
        "multi_processor_count": props.multi_processor_count,
    }


def is_gfx950() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    try:
        arch = str(getattr(torch.cuda.get_device_properties(_active_device()), "gcnArchName", ""))
    except Exception:
        return False
    return "gfx950" in arch


def _percentile(sorted_samples: list[float], percentile: float) -> float:
    if len(sorted_samples) == 1:
        return sorted_samples[0]
    rank = (len(sorted_samples) - 1) * percentile
    lo = int(rank)
    hi = min(lo + 1, len(sorted_samples) - 1)
    frac = rank - lo
    return sorted_samples[lo] * (1.0 - frac) + sorted_samples[hi] * frac


def _summarize(samples: list[float]) -> dict[str, Any]:
    ordered = sorted(samples)
    mean = statistics.mean(samples)
    stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return {
        "median": statistics.median(samples),
        "p20": _percentile(ordered, 0.20),
        "p80": _percentile(ordered, 0.80),
        "min": min(samples),
        "max": max(samples),
        "mean": mean,
        "stdev": stdev,
        "cv": stdev / mean if mean else 0.0,
        "samples": samples,
    }


def _compiled_asm(compiled_or_none: Any) -> str | None:
    asm = getattr(compiled_or_none, "asm", None)
    if isinstance(asm, dict):
        return "\n".join(str(value) for value in asm.values()).lower()
    if isinstance(asm, str):
        return asm.lower()
    return None


def _compiled_asm_section(compiled_or_none: Any, key: str) -> str | None:
    asm = getattr(compiled_or_none, "asm", None)
    if isinstance(asm, dict):
        value = asm.get(key)
        return str(value).lower() if value is not None else None
    return None


def _count_tokens(asm_text: str | None, tokens: tuple[str, ...]) -> dict[str, int] | None:
    if not asm_text:
        return None
    return {token: len(re.findall(rf"\b{re.escape(token)}\b", asm_text)) for token in tokens}


def _count_mfma(asm_text: str | None) -> int | None:
    if not asm_text:
        return None
    return len(re.findall(r"\b(v_)?mfma", asm_text))


def _timed_region_text(asm_text: str | None) -> str | None:
    if not asm_text:
        return None
    first = re.search(r"\bs_memtime\b", asm_text)
    if not first:
        return None
    second = re.search(r"\bs_memtime\b", asm_text[first.end() :])
    if not second:
        return None
    start = first.end()
    end = first.end() + second.start()
    return asm_text[start:end]


def _isa_case_sanity(case: Case, asm_text: str | None, ttgir_text: str | None = None) -> dict[str, Any]:
    if not asm_text:
        return {
            "available": False,
            "ok": None,
            "checks": {},
            "valu_instruction_counts": None,
            "mfma_instruction_count": None,
            "estimated_mfma_per_loop_iter": None,
            "estimated_mfma_per_loop_iter_note": "assembly unavailable",
            "timed_region_instruction_counts": None,
        }
    checks: dict[str, bool] = {"clock64": any(token in asm_text for token in ("s_memtime", "s.memtime", "clock64"))}
    valu_counts = _count_tokens(asm_text, ("v_fma_f32", "v_fmac_f32", "v_mul_f32", "v_add_f32"))
    if case.family == "valu":
        checks["valu_arithmetic"] = any(valu_counts.values()) if valu_counts else False
    if case.family == "mfma":
        checks["mfma"] = "mfma" in asm_text
    timed_region = _timed_region_text(asm_text)
    timed_counts = _count_tokens(timed_region, ("ds_read_b32", "ds_write_b32", "ds_write"))
    if case.family == "lds":
        checks["ds_read"] = "ds_read" in asm_text
        if case.meta.get("timed_stores") is False:
            checks["ttgir_local_gather"] = ttgir_text is not None and "local_gather" in ttgir_text
            checks["timed_ds_read_b32"] = timed_counts is not None and timed_counts["ds_read_b32"] > 0
            checks["no_timed_ds_write"] = timed_region is not None and "ds_write" not in timed_region
        else:
            checks["ds_write"] = "ds_write" in asm_text
    if case.name == "global.tl_load_dependent":
        checks["global_or_flat_load"] = any(token in asm_text for token in ("global_load", "flat_load", "buffer_load"))
    if case.name == "global.direct_to_lds_composite_32x32":
        checks["direct_to_lds_hint"] = any(
            token in asm_text for token in ("global_load_lds", "buffer_load_to_local", "ds_write")
        )
    return {
        "available": True,
        "ok": all(checks.values()),
        "checks": checks,
        "valu_instruction_counts": valu_counts,
        "mfma_instruction_count": _count_mfma(asm_text),
        "estimated_mfma_per_loop_iter": None,
        "estimated_mfma_per_loop_iter_note": "not estimated: fixed prologue/epilogue MFMA count is not separated",
        "timed_region_instruction_counts": timed_counts,
    }


def _launch(case: Case, args: list[Any]) -> Any:
    return case.kernel[(1,)](*args, num_warps=case.num_warps, **case.launch_meta)


def _is_known_direct_to_lds_unsupported(case: Case, exc: Exception) -> bool:
    error = f"{type(exc).__name__}: {exc}"
    return case.name == "global.direct_to_lds_composite_32x32" and (
        "builtin.unrealized_conversion_cast" in error or "failed to translate module to LLVM IR" in error
    )


def _unsupported_result(case: Case, exc: Exception, baseline_raw_per_iter: float | None) -> dict[str, Any]:
    error = f"{type(exc).__name__}: {exc}"
    if case.name == "global.direct_to_lds_composite_32x32" and "failed to translate module to LLVM IR" in error:
        error += " (compiler stderr reports builtin.unrealized_conversion_cast)"
    return {
        "name": case.name,
        "family": case.family,
        "unit": case.unit,
        "ops_per_iter": case.ops_per_iter,
        "ok": False,
        "unsupported": True,
        "error": error,
        "stats": None,
        "raw_ticks_per_iter": None,
        "baseline_raw_ticks_per_iter": baseline_raw_per_iter,
        "diagnostic_baseline_delta_ticks_per_op": None,
        "net_ticks_per_op": None,
        "check_values": [],
        "expected_check": None,
        "correctness_ok": False,
        "isa_sanity": {"available": False, "ok": None, "checks": {}, "mfma_instruction_count": None},
        "meta": case.meta,
    }


def _run_case(case: Case, niter: int, warmup: int, reps: int, baseline_raw_per_iter: float | None) -> dict[str, Any]:
    out = torch.zeros(1, dtype=torch.int64, device=_active_device())
    check = torch.zeros(1, dtype=torch.float32, device=_active_device())
    args = case.make_args() + [out, check, niter]

    compiled = None
    for _ in range(warmup):
        maybe_compiled = _launch(case, args)
        compiled = maybe_compiled if maybe_compiled is not None else compiled
    torch.cuda.synchronize()

    raw_per_iter_samples = []
    raw_per_op_samples = []
    check_values = []
    for _ in range(reps):
        out.zero_()
        check.zero_()
        maybe_compiled = _launch(case, args)
        compiled = maybe_compiled if maybe_compiled is not None else compiled
        torch.cuda.synchronize()
        elapsed = int(out.item())
        raw_per_iter = elapsed / max(1, niter)
        raw_per_iter_samples.append(raw_per_iter)
        raw_per_op_samples.append(raw_per_iter / max(1, case.ops_per_iter))
        check_values.append(float(check.item()))

    expected_check = case.expected_check(niter) if case.expected_check is not None else None
    correctness_ok = expected_check is None or all(value == expected_check for value in check_values)
    ok = all(sample > 0 for sample in raw_per_iter_samples) and all(value == value for value in check_values) and correctness_ok
    asm_text = _compiled_asm(compiled)
    ttgir_text = _compiled_asm_section(compiled, "ttgir")
    result = {
        "name": case.name,
        "family": case.family,
        "unit": case.unit,
        "ops_per_iter": case.ops_per_iter,
        "ok": ok,
        "stats": _summarize(raw_per_op_samples),
        "normalized_cycles_per_op": _summarize([sample * CYCLES_PER_TICK for sample in raw_per_op_samples]),
        "raw_ticks_per_iter": _summarize(raw_per_iter_samples),
        "baseline_raw_ticks_per_iter": baseline_raw_per_iter,
        "diagnostic_baseline_delta_ticks_per_op": None,
        "net_ticks_per_op": None,
        "check_values": check_values,
        "expected_check": expected_check,
        "correctness_ok": correctness_ok,
        "isa_sanity": _isa_case_sanity(case, asm_text, ttgir_text),
    }
    if case.meta:
        result["meta"] = case.meta
    return result


def _make_seed(values: list[float]) -> list[Any]:
    return [torch.tensor(values, dtype=torch.float32, device=_active_device())]


def _lds_table_values() -> list[int]:
    return [((i * 17) + 5) & 63 for i in range(64)]


def _pointer_chase_expected(starts: list[int], niter: int) -> float:
    table = _lds_table_values()
    total = 0
    for start in starts:
        idx = start
        for _ in range(niter):
            idx = table[idx]
        total += idx
    return float(total)


def _make_lds_pointer_chase(starts: list[int]) -> list[Any]:
    return [torch.tensor(_lds_table_values() + starts, dtype=torch.int32, device=_active_device())]


def _make_global_pointer_chase() -> list[Any]:
    return [torch.zeros(2, dtype=torch.int64, device=_active_device())]


def _make_global_tile() -> list[Any]:
    return [torch.arange(1, 32 * 32 + 1, dtype=torch.float16, device=_active_device())]


def _make_mfma_inputs(streams: int = 1) -> list[Any]:
    elems = streams * 32 * 32
    a = torch.arange(1, elems + 1, dtype=torch.float16, device=_active_device()).reshape(streams, 32, 32)
    b = torch.ones((streams, 32, 32), dtype=torch.float16, device=_active_device())
    return [a, b]


def _cases() -> list[Case]:
    mfma_meta = {"matrix_instr_nonkdim": 16, "waves_per_eu": 0}
    return [
        Case(
            "baseline.loop",
            "baseline",
            "raw_ticks_per_loop_iter",
            _k_baseline,
            lambda: _make_seed([1.0]),
            baseline_key="",
            meta={
                "measures": "clock64 bracketing + scalar loop overhead",
                "timer": "target-specific tlx.clock64 ticks",
            },
        ),
        Case(
            "valu.dependent_fma_expr",
            "valu",
            "ticks_per_high_level_fma_expression",
            _k_valu_dependent,
            lambda: _make_seed([1.0, 1.0000001, 0.9999999]),
            ops_per_iter=1,
            meta={"streams": 1, "high_level_fma_expressions_per_iter": 1, "runtime_seeded": True},
        ),
        Case(
            "valu.independent_fma_expr_x4",
            "valu",
            "ticks_per_high_level_fma_expression",
            _k_valu_independent_x4,
            lambda: _make_seed([1.0, 2.0, 3.0, 4.0, 1.0000001, 1.0000002, 1.0000004, 1.0000008, 0.5, 0.75, 1.0, 1.25]),
            ops_per_iter=4,
            meta={"streams": 4, "high_level_fma_expressions_per_iter": 4, "runtime_seeded": True},
        ),
        Case(
            "lds.dependent_gather_chase_i32",
            "lds",
            "ticks_per_dependent_local_gather",
            _k_lds_dependent_gather_chase,
            lambda: _make_lds_pointer_chase([0]),
            ops_per_iter=1,
            meta={
                "table_shape": "64xi32",
                "streams": 1,
                "pointer_chase": "idx_{i+1}=table[idx_i]",
                "preinitialized_lds_table": True,
                "timed_stores": False,
                "hardware_ops_per_iter": {"lds_gather_read": 1, "lds_store": 0},
            },
            expected_check=lambda niter: _pointer_chase_expected([0], niter),
        ),
        Case(
            "lds.independent_gather_i32_x4",
            "lds",
            "ticks_per_independent_local_gather",
            _k_lds_independent_gather_x4,
            lambda: _make_lds_pointer_chase([0, 7, 19, 31]),
            ops_per_iter=4,
            meta={
                "table_shape": "64xi32",
                "streams": 4,
                "preinitialized_lds_table": True,
                "timed_stores": False,
                "hardware_ops_per_iter": {"lds_gather_read": 4, "lds_store": 0},
            },
            expected_check=lambda niter: _pointer_chase_expected([0, 7, 19, 31], niter),
        ),
        Case(
            "global.tl_load_dependent",
            "global",
            "ticks_per_tl_load",
            _k_global_tl_load_dependent,
            _make_global_pointer_chase,
            meta={"path": "tl.load scalar pointer chase"},
        ),
        Case(
            "global.direct_to_lds_composite_32x32",
            "global",
            "ticks_per_direct_to_lds_composite",
            _k_global_direct_to_lds_composite,
            _make_global_tile,
            ops_per_iter=1,
            meta={
                "shape": "32x32xf16",
                "path": "tlx.buffer_load_to_local + async_load_commit_group + async_load_wait_group(0) + tlx.local_load",
                "composite_per_iter": True,
            },
        ),
        Case(
            "mfma.dependent_acc_32x32x32",
            "mfma",
            "ticks_per_tl_dot",
            _k_mfma_dependent,
            lambda: _make_mfma_inputs(1),
            launch_meta=mfma_meta,
            meta={"shape": "32x32x32", "matrix_instr_nonkdim": 16},
        ),
        Case(
            "mfma.independent_acc_32x32x32_x4",
            "mfma",
            "ticks_per_tl_dot",
            _k_mfma_independent_x4,
            lambda: _make_mfma_inputs(4),
            ops_per_iter=4,
            launch_meta=mfma_meta,
            meta={"streams": 4, "shape": "32x32x32", "matrix_instr_nonkdim": 16},
        ),
    ]


def _select_cases(bench: str) -> list[Case]:
    requested = [part.strip().lower() for part in bench.split(",") if part.strip()]
    if not requested or requested == ["all"]:
        requested = list(ALL_BENCHES)
    valid = set(ALL_BENCHES)
    unknown = sorted(set(requested) - valid)
    if unknown:
        raise ValueError(f"unknown --bench value(s): {', '.join(unknown)}; valid: all,{','.join(ALL_BENCHES)}")
    return [case for case in _cases() if case.family in requested]


def _baseline_case() -> Case:
    for case in _cases():
        if case.name == "baseline.loop":
            return case
    raise AssertionError("baseline.loop case missing")


def run_all(
    bench: str = "all",
    niter: int = DEFAULT_NITER,
    warmup: int = DEFAULT_WARMUP,
    reps: int = DEFAULT_REPS,
    require_gfx950: bool = True,
) -> dict[str, Any]:
    if require_gfx950 and not is_gfx950():
        raise RuntimeError(
            f"gfx950 HIP device required, got {_device_info() if torch.cuda.is_available() else 'no cuda'}"
        )
    selected = _select_cases(bench)
    baseline_case = _baseline_case()
    baseline_result = _run_case(baseline_case, niter, warmup, reps, None)
    baseline_raw = baseline_result["raw_ticks_per_iter"]["median"]
    benchmarks = []
    for case in selected:
        if case.name == baseline_case.name:
            benchmarks.append(baseline_result)
        else:
            try:
                benchmarks.append(_run_case(case, niter, warmup, reps, baseline_raw))
            except Exception as exc:
                if _is_known_direct_to_lds_unsupported(case, exc):
                    benchmarks.append(_unsupported_result(case, exc, baseline_raw))
                else:
                    raise
    isa_cases = {item["name"]: item["isa_sanity"] for item in benchmarks}
    return {
        "device": _device_info(),
        "timer": {
            "api": "tlx.clock64",
            "lowering": "s_memtime",
            "measured_hz": SMEMTIME_HZ,
            "shader_clock_hz": GFX950_MAX_SCLK_HZ,
            "cycles_per_tick": CYCLES_PER_TICK,
        },
        "niter": niter,
        "warmup": warmup,
        "reps": reps,
        "bench": bench,
        "baseline": baseline_result,
        "benchmarks": benchmarks,
        "isa_sanity": {"available": any(v["available"] for v in isa_cases.values()), "cases": isa_cases},
    }


def _print_human(result: dict[str, Any], stream=sys.stdout) -> None:
    dev = result["device"]
    print(
        f"device: {dev['name']} ({dev['gcn_arch_name']}), niter={result['niter']}, "
        f"warmup={result['warmup']}, reps={result['reps']}",
        file=stream,
    )
    print(
        f"baseline.loop raw median: {result['baseline']['raw_ticks_per_iter']['median']:.2f} ticks/iter",
        file=stream,
    )
    print(
        f"{'bench':<42} {'unit':<40} {'raw_med':>10} {'p20':>10} {'p80':>10} {'cv':>8} ok",
        file=stream,
    )
    for item in result["benchmarks"]:
        stats = item["stats"]
        if stats is None:
            error = item.get("error", "unsupported")
            print(
                f"{item['name']:<42} {item['unit']:<40} {'unsupported':>10} "
                f"{'':>10} {'':>10} {'':>8} {item['ok']}  {error}",
                file=stream,
            )
            continue
        print(
            f"{item['name']:<42} {item['unit']:<40} {stats['median']:>10.2f} "
            f"{stats['p20']:>10.2f} {stats['p80']:>10.2f} {stats['cv']:>8.3f} {item['ok']}",
            file=stream,
        )
    print(f"isa_sanity: {result['isa_sanity']}", file=stream)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="gfx950 TLX clock64 target-timer latency/throughput microbenchmarks")
    parser.add_argument("--bench", default="all", help="comma list from baseline,valu,lds,global,mfma or all")
    parser.add_argument("--niter", type=int, default=DEFAULT_NITER)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--json", action="store_true", help="emit JSON to stdout")
    parser.add_argument("--no-gfx950-check", action="store_true", help="allow running on non-gfx950 HIP devices")
    args = parser.parse_args(argv)

    result = run_all(
        bench=args.bench,
        niter=args.niter,
        warmup=args.warmup,
        reps=args.reps,
        require_gfx950=not args.no_gfx950_check,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        _print_human(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
