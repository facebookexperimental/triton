# B200 FMHA bar-benchmark results (paper replication)

Machine: 1x NVIDIA B200 (183 GB), driver-reported boost ~1965 MHz (no clock
locking — see notes).  Config: fp16, non-causal, BATCH=4, NUM_HEADS=32,
HEAD_DIM=128.  FLOPs: fwd = 4·B·H·S²·D, bwd = 10·B·H·S²·D.  Timing:
`triton.testing.do_bench(warmup=500, rep=500, quantiles=[0.5, 0.2, 0.8])`;
cells are median TFLOPS with [q20, q80] TFLOPS (derived from the q80/q20 ms
quantiles).  Every reported number passed its correctness gate (fwd rel <
1e-2 vs SDPA; bwd grad rel < 3e-2 vs autograd reference).

Raw data: `bench/results_fwd.json`, `bench/results_bwd.json` (per-bar
`env_probe` before/after nvidia-smi snapshots included).

## Forward (median TFLOPS, [q20, q80])

| bench bar | paper-figure bar (neutral name) | S=2048 | S=4096 | S=8192 | S=16384 |
|---|---|---|---|---|---|
| `triton_ws_off` | Triton (WS off) | 749.8 [741.3, 758.2] | 794.2 [780.4, 795.3] | 813.1 [810.8, 815.1] | 826.7 [824.7, 828.9] |
| `triton_ws_on` | Triton-WS | 481.1 [477.5, 484.5] | 521.5 [520.0, 523.5] | 559.5 [556.8, 560.9] | 605.1 [579.1, 622.5] |
| `triton_tiled` | (control: Triton at JOS sub-tile granularity; not a paper bar) | 337.6 [336.8, 338.1] | 375.3 [374.8, 376.2] | 405.8 [405.1, 406.2] | 425.6 [425.5, 425.6] |
| `tlx_default` | TLX-Default | 504.7 [503.7, 506.5] | 607.6 [606.3, 608.4] | 688.4 [683.9, 692.7] | 723.6 [721.2, 724.1] |
| `jos` | JOS | 184.0 [183.9, 184.1] | 199.8 [199.8, 200.0] | 209.5 [209.4, 209.5] | 215.3 [215.3, 215.3] |
| `cudnn` | cuDNN | 1188.1 [1172.4, 1193.7] | 1238.4 [1234.2, 1242.7] | 1256.2 [1253.3, 1284.3] | 1294.4 [1278.0, 1314.5] |
| `fa4` | FA4-official | 962.3 [932.7, 976.1] | 991.4 [983.1, 1008.2] | 1029.2 [1024.4, 1034.2] | 1047.1 [1036.4, 1051.3] |

The `jos` bar ran the solver-schedule kernel
`../sched2tlx/examples/case3_FA_fp16_subtiled/generated_jos.py`
(`fa_fwd_kernel_nows_subtiled`, grid `(cdiv(N,128), Z·H)`, num_warps=4,
num_stages=1), i.e. the FA4-exact-strategy schedule, B200-verified correct.

## Backward (median TFLOPS, [q20, q80])

| bench bar | paper-figure bar (neutral name) | S=2048 | S=4096 | S=8192 | S=16384 |
|---|---|---|---|---|---|
| `tlx_bwd_default` | TLX-Default | 244.1 [243.7, 244.5] | 257.5 [257.3, 257.7] | 265.3 [265.2, 265.4] | 269.2 [269.2, 269.3] |
| `jos_bwd` | JOS | SKIP | SKIP | SKIP | SKIP |
| `cudnn_bwd` | cuDNN | 878.4 [844.9, 900.9] | 978.6 [976.5, 987.7] | 1051.0 [1048.5, 1065.1] | 1084.9 [1078.3, 1088.3] |
| `fa4_bwd` | FA4-official | 896.0 [888.9, 899.7] | 1037.7 [1014.1, 1043.3] | 1107.2 [1096.7, 1123.6] | 1140.9 [1085.5, 1169.7] |

`tlx_bwd_default` ran `case4_FA_bwd/generated_hd128.py::fa_bwd_dkdv_5mma`
(the TMEM-aliasing re-verified emit) and times the fused dK/dV/dQ kernel only
(M/D preprocessing on host, excluded); `cudnn_bwd`/`fa4_bwd` report
(fwd+bwd − fwd-median), so the methodologies differ slightly (README
deviation 2/3).

## Key ratios

| ratio | value |
|---|---|
| JOS / FA4-official, fwd S=16384 | 215.3 / 1047.1 = **0.21×** (FA4 4.9× faster) |
| JOS / TLX-Default, fwd S=2048..16384 | 0.36×, 0.33×, 0.30×, 0.30× |
| cuDNN / FA4-official, fwd S=16384 | 1294.4 / 1047.1 = **1.24×** |
| cuDNN / FA4-official, bwd S=16384 | 1084.9 / 1140.9 = **0.95×** |
| TLX-Default / Triton-WS-off, fwd S=16384 | 723.6 / 826.7 = 0.88× |

## Skipped bars

- **`jos_bwd` (all seqlens): SKIPPED — emitter gap on the bwd topology.**
  Built per protocol: `paper_joint_solver.graph_writer.rewrite_schedule_graph`
  applied `bwd_joint_solution_v6.json` (ii=95, length=273, str keys coerced to
  int) to `case4_FA_bwd/sg_hd128_depth2.json` → `sg_hd128_jos.json`, then
  `python -m sched2tlx` emitted `generated_hd128_jos.py` (14235 bytes, with a
  benign emitter note: "register request over budget (69376 > 65536); scaling
  3 compute task(s) from num_regs=[152] to 136").
  **Failure signature:** the emitted kernel *uses* barriers for SMEM buffer
  `L0_smem_3` — `tlx.barrier_wait(L0_smem_3_full[buf], phase)` (lines 177,
  195) and `mBarriers=[L0_smem_3_empty[buf]]` on two `tlx.async_dot`s (lines
  179, 198) — but the barrier-allocation block never emits
  `L0_smem_3_full/_empty = tlx.alloc_barriers(...)` (its siblings `L0_smem_0`
  and `L0_smem_1` do get them).  Triton JIT tracing therefore fails with
  `CompilationError: ... NameError('L0_smem_3_full is not defined')` at
  kernel line 168+.  Per the replication mandate, no emitter surgery was
  attempted; the skip (with this reason) is recorded in
  `bench/results_bwd.json`.
- No other bar was skipped.

## Correctness-gate outcomes

- All fwd bars passed rel < 1e-2 vs fp16 SDPA (typical rel ~6e-4 for the
  Triton/TLX kernels).
- `tlx_bwd_default` passed grad rel < 3e-2 at every seqlen vs autograd of
  SDPA(scale=1.0) on pre-scaled Q (base-2 reference model).
- `cudnn_bwd` grads passed vs default-backend SDPA autograd; the FA4 worker
  gates vs fp32 SDPA (fwd and grads) and passed at every seqlen.

## Environment probe summary

- Each bar records `nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu`
  before/after (all 56 fwd + 32 bwd probes present).
- SM clock range across all probes: 120–1965 MHz; temperature 30–54 °C;
  power 195–967 W.  The 120 MHz readings are *idle* clocks captured by the
  "before" probe of the first bar of each fresh process (and after an idle
  gap between bars); every in-run "after" probe sat at boost (~1950–1965 MHz).
  `do_bench`'s 500 ms warmup absorbs the ramp, and per-bar q20/q80 spreads
  are tight (mostly < 2%), so no measurement was discarded for clock drift.
  Clocks could not be locked (no root) — README deviation 1.

## Measurement deviations added during the run (bench-side only)

1. **`triton_ws_on` runs under `TRITON_USE_META_WS=1` with a pruned autotune
   list** (`BLOCK_M=128, BLOCK_N∈{64,32}, num_stages∈{3,4}, num_warps=8`).
   On this beta build the stock (upstream) AutoWS pipeline fails for *every*
   tutorial autotune config with `PassManager::run failed: 'ttng.tmem_alloc'
   op operation destroyed but still has uses`
   (`TritonGPULoadMMASpecialization`); under Meta WS, `BLOCK_M=64` configs
   fail with "Only supported for scales as we pad the allocation",
   `num_stages=2` with "pipeliner doesn't know how to predicate this op", and
   `num_warps=4` exceeds the thread budget implied by the tutorial's
   `maxnreg=168`.  The three surviving configs were B200-probed correct.
   `triton_ws_off` keeps the stock pipeline and the full 36-config sweep.
2. **FA4 worker argv fix**: the CuTe DSL inside `flash_attn` parses
   `sys.argv` at import (a "Process diagnostic status" argparse) and died on
   the worker's flags; argv is now cleared before the import.
3. **TLX bwd reference runs SDPA in 4D** `(BH, 1, S, D)`: on 3D tensors SDPA
   silently falls back to the math backend (O(S²) memory — a 128 GiB
   allocation at S=16384).
4. **`do_bench` single-quantile return**: this Triton returns a bare float
   for `quantiles=[0.5]`; the harness and worker now handle both forms.

## Headline observations

- cuDNN is the fastest fwd bar at every seqlen (1188→1294 TFLOPS), 24%
  ahead of FA4-official at S=16384 on this stack.
- FA4-official leads bwd (1141 TFLOPS at S=16384), 5% ahead of cuDNN.
- The JOS fwd schedule is far below TLX-Default here (0.30× at S=16384),
  and TLX-Default itself trails the plain Triton tutorial with WS off
  (0.88×).  Triton-WS-on is *slower* than WS-off under the only compilable
  WS pipeline (Meta WS, pruned configs) on this build.
- `triton_tiled` (the sub-tile-granularity control) is the slowest working
  fwd Triton bar, ~2× below `triton_ws_off`.
