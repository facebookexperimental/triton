"""Perf and compile-time guard for ``tlx.ops.mm`` on Blackwell.

The flagship op file, and the template for every other one. Everything
op-specific is here -- the reference implementation, the FLOP formula, the
input builder -- and everything about *how* to measure lives in ``_harness``.
Adding an op should mean copying this file and changing those three things.

Shapes come from ``triton.tlx.ops.kernels.mm._shapes``, shared with the L1
correctness suite, so a shape disabled for a correctness bug cannot remain
benchmarked here.

Run it two ways. As a deterministic command, which is what CI and any review
agent should use::

    CUDA_VISIBLE_DEVICES=0 third_party/tlx/denoise.sh \\
        python python/test/tlx_benchmark/bench_mm.py \\
            --measure latency --guard enforce --json /tmp/mm.json

or under pytest, for the junitxml that the existing b200 reporting consumes::

    CUDA_VISIBLE_DEVICES=0 third_party/tlx/denoise.sh \\
        python -m pytest python/test/tlx_benchmark/bench_mm.py

``--measure`` matters. ``latency`` is minutes; ``compile`` is roughly four
minutes *per case* at ``--space full``, because a cold first call to
``tlx.ops.mm`` autotunes several hundred configs. They are separate modes so
that the fast one stays fast.
"""

from __future__ import annotations

import argparse
import sys

import torch

from triton.tlx.ops.kernels.mm._shapes import SHAPES, flops, operand

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from _harness import (Case, Status, capture_env, cold_compile, host_overhead_us, measure,  # noqa: E402
                      stable)
from _harness import baseline as baseline_mod  # noqa: E402
from _harness import report as report_mod  # noqa: E402

OP = "mm"
ARCH = "sm100"
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


def cases(dtypes=("fp16", "bf16")) -> list[Case]:
    return [
        Case(op=OP, arch=ARCH, dtype=str(DTYPES[d]).removeprefix("torch."), shape=tuple(shape))
        for shape in SHAPES
        for d in dtypes
    ]


def _operands(case: Case):
    M, N, K, a_row_major, b_row_major = case.shape
    dtype = getattr(torch, case.dtype)
    return operand(M, K, dtype, a_row_major), operand(K, N, dtype, b_row_major)


def run_case(case: Case, *, space: str, measure_compile: bool, baseline: dict):
    """Measure one case and judge it.

    Latency and cold-compile are two passes over two different cache states:
    ``t_cold`` needs an empty Triton cache and steady-state latency needs a
    warm one, so neither can be derived from the other's call.
    """
    from triton.tlx.ops import mm as tlx_mm

    a, b = _operands(case)
    tlx_fn = lambda: tlx_mm(a, b, arch=ARCH, space=space)  # noqa: E731
    ref_fn = lambda: torch.matmul(a, b)  # noqa: E731

    compile_stat = cold_compile(tlx_fn) if measure_compile else None

    tlx_fn()  # tune and compile outside the measured window
    ref_fn()
    torch.cuda.synchronize()

    tlx = measure(tlx_fn)
    ref = measure(ref_fn)
    host_us = host_overhead_us(tlx_fn)

    result = baseline_mod.judge(case, tlx, ref, tlx_host_us=host_us, compile_stat=compile_stat,
                                baseline=baseline.get(case.key))
    M, N, K = case.shape[0], case.shape[1], case.shape[2]
    result.tlx_tflops = flops(M, N, K) / (tlx.p50 * 1e-3) / 1e12
    if ref.p50:
        result.ref_tflops = flops(M, N, K) / (ref.p50 * 1e-3) / 1e12

    # The operands are freed when this frame exits; empty_cache then returns the
    # blocks to the driver so the next case (up to 2 GB at 1000000x512) can be
    # allocated. Do not `del` them -- the closures above still name them.
    torch.cuda.empty_cache()
    return result


def run(*, space="heuristic", measure_compile=False, dtypes=("fp16", "bf16"), strict=False):
    baseline = baseline_mod.load(OP, ARCH, space)
    env = capture_env()
    results = []
    with stable(strict=strict) as info:
        for case in cases(dtypes):
            try:
                results.append(run_case(case, space=space, measure_compile=measure_compile, baseline=baseline))
            except Exception as exc:  # a broken case must not hide the others
                results.append(_errored(case, exc))
    # The autotune space is part of what a number means: a heuristic-space
    # latency and a full-space latency for the same shape differ by 4x. Record
    # it so a baseline can never be silently compared across spaces.
    env["space"] = space
    env["measure_compile"] = measure_compile
    env["run"] = {k: info[k] for k in ("problems", "clock_trace", "elapsed_s") if k in info}
    return results, env


def _errored(case: Case, exc: Exception):
    from _harness import Result

    result = Result(case=case, status=Status.ERROR)
    result.notes.append(f"{type(exc).__name__}: {exc}")
    return result


def supported() -> bool:
    """Whether this op can run on the current device at all."""
    from triton._internal_testing import is_blackwell

    return is_blackwell()


# --------------------------------------------------------------------------
# CLI entry point -- the deterministic command
# --------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--measure", choices=("latency", "compile", "all"), default="latency",
                        help="latency is minutes; compile is ~4 min per case at --space full")
    parser.add_argument("--space", choices=("full", "heuristic", "smoke"), default="heuristic",
                        help="autotune search space; 'heuristic' is what tlx.ops.mm now uses by default")
    parser.add_argument("--dtype", choices=("fp16", "bf16", "both"), default="both")
    parser.add_argument("--guard", choices=("off", "report", "enforce"), default="report",
                        help="enforce exits non-zero on a regression or a compile-cap breach")
    parser.add_argument("--json", default=None, help="write the machine-readable artifact here")
    parser.add_argument("--update-baseline", action="store_true",
                        help="record this run as the baseline; refuses noisy and host-bound cases")
    parser.add_argument("--strict-env", action="store_true",
                        help="fail instead of warning when the environment is not denoised")
    args = parser.parse_args(argv)

    results, env = run(
        space=args.space,
        measure_compile=args.measure in ("compile", "all"),
        dtypes=("fp16", "bf16") if args.dtype == "both" else (args.dtype, ),
        strict=args.strict_env,
    )
    print(report_mod.render(results, env, args.json))

    if args.update_baseline:
        print(f"baseline written: {baseline_mod.save(OP, ARCH, results, env)}")
        return 0
    if args.guard == "enforce" and report_mod.failures(results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
