# Gap analysis: `generated_jos.py` (215 TFLOPS) vs TLX-Default `generated.py` (724 TFLOPS)

Kernels: `examples/case3_FA_fp16_subtiled/generated_jos.py` (`fa_fwd_kernel_nows_subtiled`,
5 warp groups) vs `examples/case3_FA_fp16/generated.py` (`fa_fwd_kernel_nows`, 6 warp
groups). All profiling at (Z,H,N,D) = (1,32,8192,128) fp16 fwd on B200, where the gap
reproduces at the same ratio as the paper shape: **205.1 vs 666.2 TFLOPS (5.36 ms vs
1.65 ms, 3.25x)**; correctness PASS for both (rel 4.8e-4).

## Headline decomposition

ncu (`--set basic` + `WarpStateStats`/`SchedulerStats`, 1 launch each):

| metric | jos | TLX-Default | ratio |
|---|---|---|---|
| duration / SM cycles | 5.34 ms / 9.57M | 1.73 ms / 2.86M | 3.35x |
| inst_executed (% peak x cycles, relative) | 0.2298 x 9.57M = 2.20 | 0.3033 x 2.86M = 0.87 | **2.54x more instructions** |
| issued warp-inst per scheduler cycle | 0.24 | 0.31 | **1.3x lower issue rate** |
| warp cycles per issued instruction | 25.5 | 12.8 | 2.0x |
| stalled: long_scoreboard (cyc/inst) | **15.74** | 8.00 | 62% of latency in both |
| stalled: barrier (cyc/inst) | **5.00** | 1.09 | 4.6x |
| stalled: short_scoreboard (cyc/inst) | 1.50 | 0.79 | 1.9x |
| Compute (SM) throughput | 24.5% | 41.2% | |
| Mem Busy / L1 hit rate | 39.1% / 95.1% | 15.6% / 0% | spill+channel traffic lives in L1 |
| achieved warps/SM (occupancy) | 24 (37.5%) | 16 (25%) | more warps, less issue |

2.54x instruction inflation x 1.3x issue-rate loss ≈ 3.3x ≈ the observed gap. The
inflation is spill code + SMEM channel round-trips + a dead-code warp group; the issue
loss is barrier serialization of the MMA-issuing group.

## Ranked causes

### 1. Partition puts TMEM unloads + both rescales on the MMA-issuing group (wg3), gated by 16 barrier waits/iter — biggest lever
In jos, wg3 (`role=TC`, 4 warps) per iteration: issues both QK MMAs, **tmem_loads both
64x64 fp32 QK results and stores them to SMEM channels** `L0_smem_12/13`, tmem_loads +
rescales + tmem_stores both 64x128 fp32 accumulators (`acc_tmem`, `acc_tmem_5`), then
issues both PV MMAs — all in strict series behind **16 `barrier_wait`s per iteration**
(lines 312-361). Whole-kernel waits/iter: **35 (jos) vs 12 (TLX-Default)**. In
TLX-Default the MMA issuers are 1-warp groups that do nothing else, and the softmax
group reads QK scores **directly from TMEM** (no SMEM bounce). Evidence: barrier stall
5.00 vs 1.09 cyc/inst; "No Eligible" scheduler cycles 76.5% vs 69.1%; the QK-score SMEM
detour also costs 2x16 KB fp32 stores + 3 loads per iter (`L0_smem_12` is read twice, by
wg1 and wg2).

### 2. 5-group plan oversubscribes the register file → spills (880B/996B) vs 0
Emitter stderr at re-emit: `register request over budget (88832 > 65536); scaling 4
compute task(s) from num_regs=[152] to 104`. ptxas (`TRITON_DUMP_PTXAS_LOG=1`):

- jos: **512 B stack frame, 880 B spill stores, 996 B spill loads**, 80 regs
- TLX-Default: **0 / 0 / 0**, 128 regs

21 warps/CTA (4 compute groups x 4 + TMA + default) vs 16 in TLX-Default, which keeps 2
fat compute groups at 152 regs spill-free. Spill traffic shows up as the dominant
long_scoreboard stall (15.74 vs 8.00 cyc/inst) and the 95% L1 hit rate / 39% Mem Busy.

### 3. Serial fallback: the intended wg3 software pipeline was dropped for TMEM budget, and the skew path deadlocks when re-enabled
Emitter stderr: `skew plan for loop0 wg3 dropped (TMEM budget exceeded with ring depths
{'op_269860064': 2}) — serial fallback`. So wg3 runs its 4-MMA + 2-rescale chain fully
serially each iteration. Variant V-B (graph-JSON only: shrink unused `L0_acc_tmem_3`
slot 2→1, freeing TMEM) makes the emitter keep the skew plan (`acc_tmem_7` becomes a
depth-2 intra-WG ring, QK1 MMA issued 1 iter ahead) — but the emitted kernel
**deadlocks on the first launch** (confirmed twice, 120 s timeout, even at 1x4x4096;
recovered with `killgpu.sh`). The serial fallback is currently load-bearing; the skewed
codegen path has a synchronization bug that blocks recovering this headroom.

### 4. Dead-code warp group wg4 + same-WG channel round-trips
wg4 (`role=CUDA+NONE`, 4 warps) loads 64x128 + 64x64 fp32 from TMEM per iteration and
computes `mulf_56/57` that are **never stored anywhere** — pure instruction/TMEM-bandwidth
waste that also inflates arrive counts (`sem15_b15_empty` arrive_count=3) and eats warp
slots/registers (feeds cause 2). Additionally wg1 and wg2 each do a pointless
**same-warp-group SMEM round trip** (store `L0_smem_9`/`L0_smem_7`, barrier_arrive,
barrier_wait, load back: lines 240-245, 284-289) — 2 extra barriers + 2 SMEM ops per
iteration each; short_scoreboard 1.50 vs 0.79 cyc/inst.

### 5. Channel ring depth is NOT the binding constraint (negative result)
Variant V-A (graph-JSON only: QK-score channels `L0_smem_12/13` depth 1→2, V ring 5→3 to
stay within SMEM): **197.1 TFLOPS — no improvement** (vs 205.1 base). TMA is also not a
factor: DRAM throughput is only 48.7 GB/s (jos) vs 149.5 GB/s (TLX-Default), and the K/V
rings are already 3/5-deep. The bottleneck is the serialized wg3 critical path and
spills, not producer-consumer buffering depth.

## Variant results (graph-JSON edits + re-emit only)

| variant | edit | result @ (1,32,8192) |
|---|---|---|
| base | — | 205.1 TFLOPS, rel 4.8e-4 |
| V-A | buf12,13 count 2; buf0/4 (V ring) 5→3 | 197.1 TFLOPS, rel 4.8e-4 (no gain) |
| V-B | buf3/5 count 2→1 (frees TMEM, skew plan kept) | **deadlock** on first launch |
| V-C | A + B | not benched (contains V-B's skew plan) |
| TLX-Default | — | 666.2 TFLOPS, rel 4.8e-4 |

## Single biggest lever

The solver's 5-group partition itself: colocate the accumulator rescale with the
softmax/correction consumers (as TLX-Default does) and shrink the compute-group count so
the QK scores flow TMEM→registers directly and 152 regs/thread fits again. That one
partition change simultaneously removes causes 1, 2, and 4 (the 2.54x instruction
inflation and the 4.6x barrier-stall blowup). No ring-depth edit can get there: the only
JSON-level route to pipelining wg3 (V-B) trips a deadlock bug in the skew-plan emitter,
which is worth fixing independently (cause 3).

*Method: ptxas logs via `TRITON_DUMP_PTXAS_LOG=1 TRITON_ALWAYS_COMPILE=1`; ncu profiles
in `--set basic` and `--section WarpStateStats/SchedulerStats/MemoryWorkloadAnalysis`,
`-k regex:fa_fwd --launch-count 1`; timing `triton.testing.do_bench(warmup=100, rep=100)`;
variants re-emitted with `python -m sched2tlx <edited graph>.json`.*
