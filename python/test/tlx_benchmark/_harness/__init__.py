"""Reusable measurement core for the TLX op perf suite.

Public surface only; everything else in this package is private. Depends on
``torch`` and ``triton`` alone -- no tritonbench wheel. Where behaviour is
ported from tritonbench, the porting module names its source.

Modules land per phase: ``denoise`` (2), ``compile`` (3), ``baseline`` and
``report`` (5).
"""

from .compile import COLD_COMPILE_CAP_S, CompileStat, cold_compile, fresh_triton_cache, prewarm
from . import baseline, report
from .baseline import SPEEDUP_TOLERANCE, judge
from .contract import SCHEMA_VERSION, Case, Result, Stat, Status, artifact
from .denoise import (CLOCK_SPREAD_LIMIT, DEGRADING_REASONS, EVENT_REASONS, ClockTrace, GpuState, capture_env, check,
                      decode_event_reasons, foreign_processes, gpu_state, numa_bound, numa_node, nvml, parse_cpulist,
                      stable)
from .measure import (DEFAULT_REP_MS, DEFAULT_REPLICATES, DEFAULT_WARMUP_MS, HOST_BOUND_RATIO, NOISE_FLOOR,
                      estimate_runtime_ms, host_overhead_us, measure, quantile_spread, reject_outliers_iqr,
                      resolve_warmup_and_rep, summarize)

__all__ = [
    "baseline",
    "report",
    "SPEEDUP_TOLERANCE",
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
    "CLOCK_SPREAD_LIMIT",
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
    "quantile_spread",
    "stable",
    "DEFAULT_REP_MS",
    "DEFAULT_REPLICATES",
    "DEFAULT_WARMUP_MS",
    "HOST_BOUND_RATIO",
    "NOISE_FLOOR",
    "estimate_runtime_ms",
    "host_overhead_us",
    "measure",
    "reject_outliers_iqr",
    "resolve_warmup_and_rep",
    "summarize",
]
