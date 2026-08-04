#!/usr/bin/env python3
"""Run the official AITER-comparison harness across all shapes, best tile per shape.

Why this exists: there are four tile kernels and no dispatcher, so "what is our
status" means running the right kernel per shape and collating by hand. Doing
that by hand produced two wrong tables in one sitting -- once by running the
256x256 kernel on a shape the 128x128 tile owns, once by forgetting that the
256x256 kernel is the one kernel whose best config has the LLIR scheduler ON.
Both are encoded below instead.

Selection rule, confirmed on every shape measured so far: pick the tile whose
grid lands on 256 = the CU count of an MI350X. Tiles that overshoot (two
workgroups per CU with half the arithmetic intensity each) or undershoot (idle
CUs) both lose, typically by 10-20%.

    grid = ceil(M / BLOCK_M) * ceil(N / BLOCK_N)

This is a BENCHMARK driver, not a dispatcher. A real dispatcher additionally
needs a tile-independent input format: shuffle_a_scale is parameterised by
BLOCK_M, and only the 32x128/64x128 kernels accept pre-shuffled B weights, so
the required input form currently differs by tile -- and no single shuffle
callback sees both M and N to choose one.

Usage:
    python bench_best_tile.py [--harness PATH] [--shape MxNxK ...]
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The 256x256 kernel REQUIRES both the LLIR scheduler and force-AGPR, together.
# Without force-AGPR the scheduler spills 48 registers (512 vgpr / 256 agpr) and
# the kernel is 3-5x slower; with it, 484/256 and zero spills. 0.888 -> 1.088.
#
# NOTE the flag has NO leading dash: TRITON_LLVM_OPTS looks the name up in
# llvm::cl::getRegisteredOptions(). "-amdgpu-mfma-vgpr-form=false" is silently
# ignored (it warns on stderr, which sweep scripts routinely discard). Confirm
# with:  [triton] TRITON_LLVM_OPTS: amdgpu-mfma-vgpr-form = false
SCHED_AGPR = {
    "TRITON_ENABLE_LLIR_SCHED": "1",
    "TRITON_LLVM_OPTS": "amdgpu-mfma-vgpr-form=false",
}

# shape -> (kernel module file, extra env, note)
# Ratios in the comments are AITER/ours measured 2026-08-02; > 1.0 is a win.
PLAN = {
    (256, 4096, 4096): ("matmul_kernel_32.py", {}, "grid 8x32=256"),
    (256, 8192, 4096): ("matmul_kernel_64.py", {}, "grid 4x64=256"),
    (512, 4096, 4096): ("matmul_kernel_64.py", {}, "grid 8x32=256"),
    (512, 8192, 4096): ("matmul_kernel_128.py", {}, "grid 4x64=256"),
    # 128x256 wants the scheduler but NOT force-AGPR, unlike its 256x256 parent:
    # 0.984/0.982/0.983 vs 0.981/0.974/0.972 over three harness reps each. Forcing
    # AGPR here costs 164 agpr vs 105 and buys nothing -- there are no spills to
    # fix. AITER's own kernel reports .vgpr_count 512 with no .agpr_count at all.
    (2048, 4096, 8192): ("matmul_kernel_128x256.py", {"TRITON_ENABLE_LLIR_SCHED": "1"}, "grid 16x16=256"),
    (2048, 8192, 4096): ("matmul_kernel.py", SCHED_AGPR, "grid 8x32=256"),
    (2048, 8192, 8192): ("matmul_kernel.py", SCHED_AGPR, "grid 8x32=256"),
}

ROW = re.compile(r"^\s*(\d+)x(\d+)x(\d+)\s+([\d.]+)\s+\S+\s+\S+\s+([\d.]+)\s+([\d.]+)\s*$")


def run_one(harness, kernel, shape, env_extra):
    import os

    env = dict(os.environ)
    env.update(env_extra)
    cmd = [
        sys.executable,
        str(harness),
        "--tlx-intra-wave-source",
        str(HERE / kernel),
        "--allow-source-mismatch",
        "--shape",
        "x".join(str(v) for v in shape),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, env=env).stdout
    for line in out.splitlines():
        m = ROW.match(line)
        if m and tuple(int(m.group(i)) for i in (1, 2, 3)) == shape:
            return float(m.group(4)), float(m.group(5)), float(m.group(6))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", type=Path, default=Path("/tmp/aiter_baseline_repro.py"))
    ap.add_argument("--shape", action="append", dest="shapes")
    args = ap.parse_args()

    shapes = list(PLAN)
    if args.shapes:
        want = {tuple(int(v) for v in s.split("x")) for s in args.shapes}
        shapes = [s for s in shapes if s in want]
        for s in sorted(want - set(PLAN)):
            print(f"!! no tile planned for {s[0]}x{s[1]}x{s[2]}", file=sys.stderr)

    print(f"{'shape':<16}{'tile':<14}{'AITER us':>9}{'ours us':>9}{'ratio':>8}   note")
    ratios = []
    for shape in shapes:
        kernel, env_extra, note = PLAN[shape]
        got = run_one(args.harness, kernel, shape, env_extra)
        label = f"{shape[0]}x{shape[1]}x{shape[2]}"
        tile = kernel.replace("matmul_kernel", "").replace(".py", "").strip("_") or "256"
        if got is None:
            print(f"{label:<16}{tile:<14}{'FAILED':>9}")
            continue
        aiter_us, ours_us, ratio = got
        ratios.append(ratio)
        flag = " <-- win" if ratio >= 1.0 else ""
        print(f"{label:<16}{tile:<14}{aiter_us:9.2f}{ours_us:9.2f}{ratio:8.3f}   {note}{flag}")
    if ratios:
        print(
            f"\ngeomean ratio: {(lambda v: __import__('math').exp(sum(map(__import__('math').log, v)) / len(v)))(ratios):.3f}"
            f"   ({sum(r >= 1.0 for r in ratios)}/{len(ratios)} shapes at or above AITER)")


if __name__ == "__main__":
    main()
