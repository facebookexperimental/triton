"""Map tritonbench provider results into the existing nightly artifact."""

from __future__ import annotations

import statistics

from .contract import Result, Stat, Status
from . import verdict

PROVIDERS = {
    "mm": ("gemm", "aten_matmul", "tlx_ops_mm"),
    "addmm": ("addmm", "aten_addmm", "tlx_ops_addmm"),
    "flash_attn": ("flash_attention", "sdpa", "tlx_ops_flash_attn"),
    "mm_torchtlx": ("gemm", "aten_matmul", "torch_tlx_mm"),
    "addmm_torchtlx": ("addmm", "aten_addmm", "torch_tlx_addmm"),
}


def throughput_stat(latency, flops):
    values = sorted(flops / (ms * 1e9) for ms in latency.times)
    mean = statistics.fmean(values)

    def percentile(q):
        import math
        return values[max(0, math.ceil(q * len(values)) - 1)]

    return Stat(
        mean=mean,
        cv=max(latency.cv, latency.raw_cv),
        p50=flops / (latency.p50 * 1e9),
        p95=percentile(.95),
        p99=percentile(.99),
        min=values[0],
        max=values[-1],
        rel_max_deviation=None,
        rel_idr=(percentile(.9) - percentile(.1)) / percentile(.5),
        replicates=1,
        n_kept=len(values),
        n_raw=len(latency.raw_times),
    )


def result(case, metrics, reference, provider, flops, cap_s=120):
    from tritonbench.utils.tlx_l2 import MAX_CV, latency_stats
    from .compile import CompileStat

    tlx, ref = metrics.get(provider), metrics.get(reference)
    if tlx is None or ref is None or tlx.latency is None or ref.latency is None:
        errors = [getattr(m, "error_msg", None) for m in (tlx, ref)]
        return Result(case, status=Status.ERROR, notes=[str(error) for error in errors if error]
                      or ["tritonbench provider returned no latency"])
    cold = None if tlx.cold_compile_ms is None else CompileStat(tlx.cold_compile_ms / 1000, cap_s=cap_s)
    out = verdict.judge(
        case,
        throughput_stat(tlx.latency, flops),
        throughput_stat(ref.latency, flops),
        correct=tlx.accuracy,
        tlx_host_us=tlx.host_us,
        compile_stat=cold,
        speedup=ref.latency.p50 / tlx.latency.p50,
        max_cv=MAX_CV,
    )
    out.flop_count = flops
    out.ref_host_us = ref.host_us
    out.extra.update(statistic="p50_latency", harness="tritonbench", tlx_latency=latency_stats(tlx.latency),
                     ref_latency=latency_stats(ref.latency))
    return out


def run(bench, cases, shapes, cold_mode="none", cudagraph=False):
    from tritonbench.utils.tlx_l2 import run_operator

    op_name, reference, provider = PROVIDERS[bench.OP]
    shape_map = {(tuple(shape[:-1]), str(bench.DTYPES[shape[-1]]).removeprefix("torch.")): shape for shape in shapes}
    results = {}
    for direction in dict.fromkeys(case.direction for case in cases):
        selected = [case for case in cases if case.direction == direction]
        inputs = [shape_map[(case.shape, case.dtype)] for case in selected]
        try:
            rows = run_operator(op_name, provider, reference, inputs, direction, cold=cold_mode, cudagraph=cudagraph)
        except Exception as exc:
            for case in selected:
                results[case.key] = Result(case, status=Status.ERROR, notes=[f"{type(exc).__name__}: {exc}"])
            continue
        for i, case in enumerate(selected):
            if i >= len(rows):
                results[case.key] = Result(case, status=Status.ERROR, notes=["tritonbench stopped before this input"])
                continue
            flops = (2 * case.shape[0] * case.shape[1] *
                     case.shape[2] if op_name in ("gemm", "addmm") else bench.flops(*case.shape, direction))
            results[case.key] = result(case, rows[i][1], reference, provider, flops,
                                       getattr(bench, "COMPILE_CAP_S", 120))
    if op_name == "flash_attention":
        for value in results.values():
            value.extra["ref_backend"] = "flash"
    return [results[case.key] for case in cases]


def installation():
    import json
    from pathlib import Path
    import tritonbench

    metadata = Path(tritonbench.__file__).parent / "tlx_l2_install.json"
    if metadata.exists():
        return json.loads(metadata.read_text())
    return {"revision": "local", "source": str(Path(tritonbench.__file__).parent)}
