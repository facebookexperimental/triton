"""Reusable measurement core for the TLX op perf suite.

Public surface only; everything else in this package is private. Depends on
``torch`` and ``triton`` alone -- no tritonbench wheel. Where behaviour is
ported from tritonbench, the porting module names its source.

Modules land per phase: ``denoise`` (2), ``compile`` (3), ``verdict`` and
``report`` (5).
"""

from .compile import COLD_COMPILE_CAP_S, CompileStat, cold_compile, fresh_triton_cache, prewarm
from . import report, verdict
from .verdict import MAX_CV, MIN_SPEEDUP, judge
from .contract import SCHEMA_VERSION, Case, Result, Stat, Status, artifact
from .denoise import (MAX_CLOCK_IDR, DEGRADING_REASONS, EVENT_REASONS, ClockTrace, GpuState, capture_env, check,
                      decode_event_reasons, foreign_processes, gpu_state, numa_bound, numa_node, nvml, parse_cpulist,
                      stable)
from .measure import (DEFAULT_REPLICATES, DEFAULT_WARMUP_ITERS, MAX_REPLICATE_DEVIATION, MIN_REP_MS, MIN_TOTAL_SAMPLES,
                      estimate_runtime_ms, host_overhead_us, measure, percentiles, relative_interdecile_range,
                      reject_outliers_iqr, resolve_warmup_and_rep, summarize, to_tflops, window_for)

__all__ = [
    "report",
    "verdict",
    "MAX_CV",
    "MIN_SPEEDUP",
    "judge",
    "COLD_COMPILE_CAP_S",
    "CompileStat",
    "cold_compile",
    "fresh_triton_cache",
    "prewarm",
    "SCHEMA_VERSION",
    "Case",
    "Result",
    "Stat",
    "Status",
    "artifact",
    "MAX_CLOCK_IDR",
    "DEGRADING_REASONS",
    "EVENT_REASONS",
    "ClockTrace",
    "GpuState",
    "capture_env",
    "check",
    "decode_event_reasons",
    "foreign_processes",
    "gpu_state",
    "numa_bound",
    "numa_node",
    "nvml",
    "parse_cpulist",
    "relative_interdecile_range",
    "stable",
    "DEFAULT_REPLICATES",
    "DEFAULT_WARMUP_ITERS",
    "MAX_REPLICATE_DEVIATION",
    "MIN_REP_MS",
    "MIN_TOTAL_SAMPLES",
    "window_for",
    "estimate_runtime_ms",
    "host_overhead_us",
    "measure",
    "percentiles",
    "reject_outliers_iqr",
    "resolve_warmup_and_rep",
    "summarize",
    "to_tflops",
]
