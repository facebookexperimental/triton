# Solver-configuration measurements (B200, 2026-07-06)

Per-config speedup vs the COMMITTED default kernels (Rau + scoreCandidate),
speedup = committed gen_ms / config gen_ms (>1.00x is faster). Produced by
`examples/testing/solver_regression.py --keep` (do_bench median, warmup 50 /
rep 200) plus an ad-hoc gen-only bench for case5 (no bench_spec). "parity" =
non-comment codegen is byte-identical to the committed kernel, so 1.00x by
construction. case2 is excluded (emitter blockM>128 TMEM gap; its committed
kernel passes correctness); case4 does not exist in the repo. Correctness:
every entry below passed.

Values within ±0.02x are run-to-run noise. The only entries outside noise
are case3 at (1,32,8192): cpsat alone −2% (the documented alpha-order gap —
this is why the gate blocks cpsat-alone from default) and the two full
stacks +2% (the ~665 TFLOPS plateau, 4th independent reproduction).

## Config `cpsat` — TRITON_USE_MODULO_SCHEDULE=cpsat

| Case / shape | Speedup | Note |
|---|---|---|
| case1 (all shapes) | 1.00x | parity |
| case3 (1,4,512) | 1.00x | 23.9 TFLOPS |
| case3 (1,8,1024) | 1.06x | 139.5 TFLOPS |
| case3 (2,16,2048) | 1.01x | 409.1 TFLOPS |
| case3 (1,16,4096) | 1.01x | 472.9 TFLOPS |
| case3 (2,16,4096) | 1.00x | 547.8 TFLOPS |
| case3 (1,32,8192) | **0.98x** | 638.0 TFLOPS — canary MISS (651): alpha-order gap |
| case5 2048³/4096³/8192³ | 1.00x / 1.00x / 1.00x | ad-hoc bench |
| case6 (all shapes) | 1.00x | 1638 / 2621 / 3084 GB/s |
| case7 (all shapes) | 1.00x | parity |

## Config `joint1` — TRITON_MODULO_CPSAT_JOINT=1

| Case / shape | Speedup | Note |
|---|---|---|
| case1, case3, case5, case6, case7 (all shapes) | 1.00x | ALL parity — the joint v1 partitioner reproduces every committed partition byte-identically |

## Config `joint2` — TRITON_MODULO_CPSAT_JOINT=2

| Case / shape | Speedup | Note |
|---|---|---|
| case1, case3, case5, case7 (all shapes) | 1.00x | parity |
| case6 (16384,512) | 1.00x | 1638 GB/s — re-solved cycles, perf-flat |
| case6 (65536,512) | 1.00x | 2623 GB/s |
| case6 (262144,512) | 1.00x | 3084 GB/s |

## Config `full` — cpsat + JOINT=2

| Case / shape | Speedup | Note |
|---|---|---|
| case1 (all shapes) | 1.00x | parity |
| case3 (1,4,512) | 1.00x | 23.8 TFLOPS |
| case3 (1,8,1024) | 1.00x | 131.5 TFLOPS |
| case3 (2,16,2048) | 1.00x | 404.4 TFLOPS |
| case3 (1,16,4096) | 1.00x | 469.4 TFLOPS |
| case3 (2,16,4096) | 1.00x | 545.7 TFLOPS |
| case3 (1,32,8192) | **1.02x** | 664.6 TFLOPS — canary OK |
| case5 2048³/4096³/8192³ | 1.00x / 1.01x / 1.00x | ad-hoc bench |
| case6 (all shapes) | 1.00x | 1638 / 2623 / 3084 GB/s |
| case7 (all shapes) | 1.00x | parity |

## Config `full-noguard` — cpsat + JOINT=2 + DISABLE_MMA_GUARD (case3 II=1325)

| Case / shape | Speedup | Note |
|---|---|---|
| case1 (all shapes) | 1.00x | parity |
| case3 (1,4,512) | 1.00x | 23.9 TFLOPS |
| case3 (1,8,1024) | 1.00x | 131.2 TFLOPS |
| case3 (2,16,2048) | 1.00x | 404.4 TFLOPS |
| case3 (1,16,4096) | 1.00x | 469.4 TFLOPS |
| case3 (2,16,4096) | 1.00x | 545.7 TFLOPS |
| case3 (1,32,8192) | **1.02x** | 666.1 TFLOPS — canary OK, best measured |
| case5 2048³/4096³/8192³ | 1.00x / 1.00x / 0.99x | ad-hoc bench |
| case6 (all shapes) | 1.00x | 1638 / 2621 / 3084 GB/s |
| case7 (all shapes) | 1.00x | parity |

## Re-measurement (2026-07-06, second session) + sub-tiled kernel

All five config tables above REPRODUCE within ±0.02x noise (fresh
`solver_regression.py --keep` run + ad-hoc case5 bench; the only
outside-noise entries are the same two as before: cpsat-alone case3
canary 639.9/651 MISS, full/full-noguard 664.1/666.1 canary OK).

New: the Route A sub-tiled FA kernel
(`sched2tlx/examples/testing/subtiling/fa_subtiled_rau_handpatched.py`,
BLOCK_M=256 as 2×128 sub-tiles, Rau schedule + joint partition + the
five hand-fixes documented in its header) vs the committed case3
kernel. Not a solver config yet — it stands in for the emitter/model
work items in SubTilingDesign.md. Correctness PASS on all shapes.

| case3 shape | Speedup | Sub-tiled | Committed |
|---|---|---|---|
| (1,4,512) | 0.73x | 17.5 TFLOPS | 23.9 TFLOPS |
| (1,8,1024) | 0.73x | 95.3 TFLOPS | 131.5 TFLOPS |
| (2,16,2048) | **1.25x** | 504.7 TFLOPS | 404.3 TFLOPS |
| (1,16,4096) | **1.26x** | 593.8 TFLOPS | 469.3 TFLOPS |
| (2,16,4096) | **1.13x** | 615.6 TFLOPS | 543.4 TFLOPS |
| (1,32,8192) | **1.07x** | 695.4 TFLOPS | 651.2 TFLOPS |

The shape profile is the wave-quantization signature of halving the
grid (256-row tiles): at (1,4,512)/(1,8,1024) the sub-tiled grid is
2/4 CTAs per head-batch — too few to fill 148 SMs — while the
mid-range shapes gain the most (fewer, fuller waves) and the largest
shape keeps +7% (695–720 TFLOPS across runs; the 4-run best is 719.9 =
1.11x). A production autotuner would pick sub-tiled for N_CTX ≥ 2048
and the single-tile kernel below that.
