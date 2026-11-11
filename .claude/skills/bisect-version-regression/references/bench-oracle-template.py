# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Per-config bench harness and git-bisect oracle template.

Two uses:

1. Bench mode (default): run every config, print one JSON line per config
   with compile_s, n_regs, n_spills, shared, runtime_ms, and verify which
   Triton build produced the numbers.

2. Oracle mode (--oracle): measure one config repeatedly and exit with
   the git bisect contract: 0 = good (fast), 1 = bad (slow), 125 = skip
   (inside the guard band, failed to compile, or otherwise undecidable).

Adaptation required: replace `launch_kernel` with the standalone
mini-repro extracted from the reported kernel (custom-op wrappers
stripped; constexprs, strides, and grid identical to the report), and set
CONFIGS / GOOD_MAX_MS / BAD_MIN_MS from endpoint measurements.
"""

import argparse
import json
import sys
import time

import torch
import triton

# --- Adapt these -----------------------------------------------------------
CONFIGS = [
    # (constexpr_dict, num_warps, num_stages)
    ({"BLOCK_BQ": 1, "BLOCK_N": 32}, 4, 3),  # worst config in the report
    ({"BLOCK_BQ": 2, "BLOCK_N": 64}, 8, 2),  # autotune winner in the report
]
GOOD_MAX_MS = 0.68  # runtime at or below this is "good"
BAD_MIN_MS = 0.73  # runtime at or above this is "bad"; between = skip
KERNEL_NAME = "attention"


def launch_kernel(cfg, num_warps, num_stages):
    """Replace with the mini-repro launch. Must return after sync."""
    raise NotImplementedError("adapt launch_kernel to the reported kernel")


def bench_one(cfg, num_warps, num_stages, repeats=5):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    launch_kernel(cfg, num_warps, num_stages)  # first call compiles
    torch.cuda.synchronize()
    compile_s = time.perf_counter() - t0

    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        launch_kernel(cfg, num_warps, num_stages)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return compile_s, times[len(times) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", action="store_true", help="measure CONFIGS[0] and exit 0/1/125")
    ap.add_argument("--config-index", type=int, default=0, help="which CONFIGS entry the oracle measures")
    args = ap.parse_args()

    print(f"triton {triton.__version__} file={triton.__file__} "
          f"torch {torch.__version__} cuda {torch.version.cuda}", file=sys.stderr)

    if args.oracle:
        cfg, warps, stages = CONFIGS[args.config_index]
        try:
            _, ms = bench_one(cfg, warps, stages)
        except Exception as e:  # compile failure etc: undecidable
            print(f"oracle skip: {e}", file=sys.stderr)
            sys.exit(125)
        if ms <= GOOD_MAX_MS:
            sys.exit(0)
        if ms >= BAD_MIN_MS:
            sys.exit(1)
        sys.exit(125)  # guard band

    for cfg, warps, stages in CONFIGS:
        label = " ".join(f"{k}={v}" for k, v in cfg.items())
        label += f" warps={warps} stages={stages}"
        try:
            compile_s, ms = bench_one(cfg, warps, stages)
            print(
                json.dumps(
                    {"config": label, "kernel": KERNEL_NAME, "compile_s": compile_s, "status": "ok", "runtime_ms": ms}))
        except Exception as e:
            print(
                json.dumps({
                    "config": label, "kernel": KERNEL_NAME, "compile_s": None, "status": "failed", "error": str(e)[:200]
                }))


if __name__ == "__main__":
    main()
