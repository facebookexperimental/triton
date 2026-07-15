---
name: autows-match-tlx
description: >
  Enable Meta-Triton automatic warp specialization (autoWS) on a Triton kernel
  via annotations and tune it to match a hand-written TLX warp-specialized
  kernel. Use when asked to "enable autoWS", "warp-specialize a triton kernel",
  "make autoWS match/beat TLX", "close the autoWS-vs-TLX gap", or to add
  `warp_specialize=True` / meta-WS to an attention/GEMM kernel and benchmark it
  against a TLX baseline on Hopper/Blackwell.
---

# Enable autoWS (Meta-Triton) and match TLX via annotations

Meta Triton's automatic warp specialization (autoWS / meta-WS) partitions a
kernel's warps into producer/consumer groups from a plain `@triton.jit` kernel,
driven by an **annotation** on the main loop rather than hand-written `tlx.*`
primitives. This skill is the tested recipe to (1) turn autoWS on, (2) validate
it, (3) compare it to a TLX kernel, and (4) close the perf gap so autoWS matches
TLX.

Reference kernels in `third_party/tlx/tutorials/`:
- autoWS attention: `hstu_cross_attn/`, `hstu_self_attn/`, and the FA
  `fused_attention_ws_device_tma.py`.
- Hand-TLX equivalents: `tlx_bw_*.py`, `blackwell_fa_ws_pipelined_persistent.py`.
- Related complementary skills: `autows-authoring` (enable/structure autoWS),
  `autows-testing`, `ir-override-ablation`. For the autoWS-vs-TLX comparison
  recipe see `.../WarpSpecialization/docs/DebuggingAccuracyAndDeadlocks.md` -
  especially section 6 "Diff against the TLX base implementation", the
  `[ws-summary]` memory-planner dump, and "Lockstep instrumentation" - plus
  `ReuseGroups.md` and the known partition-scheduler bugs in
  `.llms/rules/partition-scheduler-bugs.md`.

## Step 0 - Prerequisites

- Meta Triton (`triton.knobs.nvidia.use_meta_ws` must exist). Editable install.
- Env (BOTH required at runtime):
  - `TRITON_USE_META_WS=1` - turns the meta-WS compiler pass on.
  - `TRITON_DISABLE_WSBARRIER_REORDER=1` - required by the WS lowering.
- Blackwell (sm_100) or Hopper (sm_90) GPU.

## Step 1 - Enable autoWS with the annotation

The primary trigger is `warp_specialize=True` on the kernel's main `tl.range`
loop (the loop the compiler will split into producer/consumer partitions):

```python
for start_n in tl.range(0, hi, BLOCK_N, warp_specialize=True):
    # loads (TMA), MMAs, and elementwise -> compiler partitions these
    ...
```

That plus the two env vars is the minimum. A plain `@triton.jit` kernel WITHOUT
this annotation gets NO warp specialization even with `TRITON_USE_META_WS=1`
(verified: 0 `ttg.warp_specialize`, `num-warps` unchanged) - the pass only acts
on annotated loops with an MMA/TMA structure it can partition.

Persistent variant: annotate the outer persistent tile loop the same way
(`for tile_id in tl.range(start, num_tiles, NUM_SMS, warp_specialize=True)`).

Two gotchas when gating the annotation by a flag (tested on HSTU self-attn):
- The flag MUST be a `tl.constexpr`, not a plain module global, or you get
  "Cannot access global variable ... from within @jit'ed function". Use
  `_AUTOWS = tl.constexpr(os.environ.get("X") == "1")` and
  `tl.range(..., warp_specialize=_AUTOWS)`.
- The autotune CONFIG must be big enough or meta-WS silently does nothing
  (0 `ttg.warp_specialize`, kernel stays SIMT). A tiny config (num_warps=2,
  BLOCK_M=16) will NOT partition; you need `num_warps>=4` (8 is better) and a
  real tile (BLOCK_M/BLOCK_N >= 64) with `num_stages>=1`. Pin/select such a
  config for the autoWS variant.

In the host wrapper, gate on the knobs so failures are loud:
```python
assert triton.knobs.nvidia.use_meta_ws, "requires TRITON_USE_META_WS=1"
assert triton.knobs.nvidia.disable_wsbarrier_reorder, "requires TRITON_DISABLE_WSBARRIER_REORDER=1"
```

## Step 2 - Validate correctness

Compile + run and confirm warp specialization actually happened, then check
accuracy vs a trusted reference (torch-float or the non-WS kernel):
```bash
TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/ws ... python driver.py
G=$(find /tmp/ws -name '*fwd*.ttgir'|head -1)
grep -c 'ttg.warp_specialize' "$G"     # expect >0
grep -oE 'ttg.partition.types = \[[^]]*\]' "$G" | head -1   # e.g. [computation, gemm, load, ...]
```
- If autoWS crashes in `NVGPUWarpSpecialization` (e.g. "SiLU product consumed by
  the PV MMA across partition boundaries", or a loop-carried-accumulator assert),
  that's a compiler-pass bug - see the partition-scheduler notes in
  `.llms/rules/partition-scheduler-bugs.md`.
- `numStages >= 1` assert = a `num_stages=0` config reached meta-WS. autoWS needs
  `num_stages >= 1`.
- **If autoWS is WRONG or DEADLOCKS: dump BOTH final ttgirs (TLX with meta-WS
  off, autoWS with it on, separate processes) and run the `barrier-visualization`
  skill on EACH, then diff them side by side.** TLX lowers to the same
  `ttg.warp_specialize`/`wait_barrier`/`arrive_barrier`/`init_barrier` ops, so the
  two are directly comparable at the mbarrier level. Diff cheapest-first:
  per-partition `num_warps` + total (`<=16`); the `init_barrier` count histogram
  (a barrier released by an N-warp partition must be `init N` — a TLX `init 4` vs
  autoWS `init 1` on the same channel is a silent phase desync; but TLX's
  CLC-persistent `clc_context` barrier `init num_consumers` has no autoWS analog,
  don't mistake it for the bug); then per-channel arrive/wait/phase for each
  reused/accumulator/reduction buffer (same-iteration-counter lockstep check).
  Full recipe: `.../WarpSpecialization/docs/DebuggingAccuracyAndDeadlocks.md` §6.

## Step 3 - Compare against TLX (measure them SEPARATELY)

CRITICAL gotcha: `TRITON_USE_META_WS=1` is a **global** env applied to every
kernel compiled in the process. It will run the meta-WS pass on a hand-TLX
kernel too and typically CRASH it (e.g. TLX fwd uses `num_stages=0` ->
`numStages >= 1` assert). So autoWS and TLX **cannot run in the same process**.

- Run autoWS: `TRITON_USE_META_WS=1 TRITON_DISABLE_WSBARRIER_REORDER=1 ... --only <autows_variant>`
- Run TLX separately: `TRITON_DISABLE_WSBARRIER_REORDER=1 ... --only <tlx_variant>` (no META_WS)
- Use the same shape/metric and compare the two numbers. A plain-Triton kernel
  (e.g. a non-WS baseline) is unaffected by META_WS, so it can co-run with autoWS
  as the accuracy baseline.

## Step 3.5 - Data partitioning (DP) = the autoWS analog of TLX `replicate=NUM_MMA_GROUPS`

TLX gets its multi-compute-group structure from `tlx.async_task(...,
replicate=NUM_MMA_GROUPS)`. The autoWS analog is a **`data_partition_factor=N`
kwarg on the annotated `tl.range`** (a `tl.constexpr`), e.g.:
```python
for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=True,
                        data_partition_factor=2, merge_epilogue=True,
                        separate_epilogue_store=True):
```
It becomes the `tt.data_partition_factor` loop attr that `WSDataPartition` /
ModuloSchedule consume to split the loop's MMAs into N groups (2 compute
partitions). Reference: `fused_attention_ws_device_tma.py` (`DP_FACTOR`).

GOTCHA (verified on HSTU fwd): the attr can wire through the IR yet the pass
**silently declines to partition** if the tile is too small. `WSDataPartition.cpp`
requires each **M-slice >= 128 rows** (TMEM allocations are 128 lanes; an M-slice
below blockM is "not representable"), so `data_partition_factor=2` needs
**BLOCK_M >= 256** to split along M (or BLOCK_N >= 256 for N-DP). At BLOCK_M=64/128
it is a no-op. Confirm it actually fired: the final ttgir should show more
partitions / a higher `tc_gen5_mma` count, not the same as DP=1.

DOUBLE THE SPLIT DIM: DP halves the split dim per partition, so to keep each
partition's tile at the reference size (and clear the >=128 slice rule) the
config must DOUBLE the split dim. TLX fwd uses BLOCK_M=256 with NUM_MMA_GROUPS=2
(2 groups of 128); the autoWS equivalent is `data_partition_factor=2` with
BLOCK_M=256 (each slice = 128). Build a dedicated doubled config for the DP path
(BLOCK_M = 128*dp).

CAPABILITY GAP vs TLX (verified end-to-end on HSTU self-attn fwd, GB200): even
with the doubled BLOCK_M=256 DP fires (attr applied, pass no longer skips,
tc_gen5_mma count rises) but then hits **OutOfResources: tensor memory** -
Required 1280 (BN=128) / 896 (BN=64) / 704 (BN=32) vs the 512-column TMEM limit.
The compiler's DP allocates the **full pre-split 256-row TMEM tile** and then
slices, so peak TMEM is the un-partitioned footprint; TLX's manual `replicate`
allocates **per-group 128-row TMEM** (each fits ~half the budget). So compiler DP
is NOT per-partition at TMEM-allocation time and cannot reproduce TLX's 2x128 fit
at BLOCK_M=256. Net: on a TMEM-heavy attention kernel, matching TLX's DP structure
needs a COMPILER fix (DP-aware per-partition TMEM allocation), not a config knob.
The two constraints squeeze from both sides: slice>=128 => BLOCK_M>=256; but
BLOCK_M=256 => TMEM OOM.

### Manual DP - the workaround for the compiler-DP TMEM gap (verified, bwd)

When compiler `data_partition_factor` can't fit (above), replicate TLX's
structure BY HAND: process N blocks per step explicitly in the kernel, giving
each block DISJOINT per-block `buffer.id`s (the `_2KV` / `_2KV_B1` dot-attrs) so
the compiler allocates each partition's tiles separately - the manual analog of
TLX `replicate`, bypassing the `WSDataPartition` pass that over-allocates the
pre-split TMEM tile. Fit the budget with dp-half TMEM reuse (dp0/dp1 share one
tile) so BLOCK_N=128 fits the 512-col TMEM limit. Worked example:
`hstu_cross_attn` `_hstu_attn_bwd_redq_2kv` (`BwdVariant.TRITON_AUTOWS_2KV`,
shared-KV + compute-fold), correct at ~1.9e-3.

This needs compiler support that had to be added for chained accumulators:
`handleOperandD` shared-opndD support + coalesce chained MMAv5 in
`OptimizeAccumulatorInit`, plus the reduce_dq store-token-wait co-location fix
(T279388065; otherwise the reduce is cloned into two partitions and dq is
reduced twice) - see `.llms/rules/partition-scheduler-bugs.md` bug #12. Net:
manual DP DOES reproduce TLX's per-partition TMEM fit today; the compiler
`data_partition_factor` path still needs the DP-aware per-partition TMEM
allocation fix to match it hands-free.

## Step 4 - Close the perf gap (make autoWS match TLX)

An untuned autoWS kernel is usually SLOWER than TLX (HSTU self-attn fwd example:
autoWS 76 vs TLX 106 vs GR-triton 123 TFLOPS). Dump BOTH final ttgirs and diff
the WS structure, then make autoWS match TLX's:

1. **Warp counts per partition** - `num_warps(...)` on each partition region and
   `ttg.num-warps`. TLX often puts the heavy compute partition on more warps.
2. **num_stages / pipelining** - match TLX's stage count on the annotated loop.
   Too many stages -> register spills; too few -> no overlap.
3. **Register budget (the big lever)** - compare `requestedRegisters = array<i32:...>`
   and `partition.types`. The winning pattern (from FA bwd) is ROLE-BASED, not
   positional: the register-heavy **computation partition = the default region**
   (gets the residual budget), and producers/reduction get a **balanced** modest
   budget (~88) so the default's residual stays under the setmaxnreg cap (256).
   Check spills with CUDA-13 `ptxas -arch=sm_100a -v` (expect 0 spill bytes).
4. **Per-dot / buffer annotations** - for FA-style kernels, the autotune config
   carries per-MMA scheduling (`_BWD_DOT_ATTRS_*`), buffer copy counts, and
   `EPILOGUE_SUBTILE`; match TLX's tile (BLOCK_M/N) and dsT-in-TMEM-vs-SMEM choice.
5. **2-CTA** - TLX bwd may use a 2-CTA collaborative MMA (`ctas_per_cga`) that
   autoWS configs lack; that is a separate ~13% that needs a 2-CTA autoWS config.
6. **Memory-plan search (autotune-native TMEM/SMEM allocation)** - the WS memory
   planner can SEARCH allocation plans instead of using the single cost-best one:
   `TRITON_WS_SMEM_PLAN_SEARCH=1`, `TRITON_WS_MEM_PLAN_TOPK=K`, and a
   `mem_plan_pick` `tl.range` kwarg (a `tl.constexpr` the autotuner sweeps; stamps
   `tt.mem_plan_pick`). This is the config-knob lever for the TMEM-fit gap in
   Step 3.5. Validated on FA fwd (search engages, results correct).
7. **Instruction / modulo-schedule search** - `TRITON_USE_LIST_SCHEDULE=1`,
   `TRITON_LIST_SCHEDULE_TOPK`, and a `list_schedule_pick` kwarg reorder the loop
   body (top-K/beam) for autotuning; `TRITON_MODULO_TOPK` does the same for the
   modulo scheduler. CAUTION: currently BUGGY on the bwd reduce_dq path - the
   picked schedule corrupts `dq` (rel-L2 ~0.7), tracked in T279643623. Usable for
   exploration, not yet trustworthy for bwd correctness.

Iterate: change annotation/config -> dump ttgir -> confirm the WS layout matches
TLX -> re-measure. The register ASSIGNMENT (role-based budget) is often the
decisive lever.

## Gotchas / methodology

- META_WS is global -> never benchmark autoWS and TLX in one process (Step 3).
- Perf on a host that cannot lock GPU clocks (`nvidia-smi -lgc` denied, no root)
  is clock-noise-contaminated ~+-6%; the `ir_override` measurement path in
  tritonbench is systematically inflated. Compare via the native path, or lock
  clocks.
- autoWS bwd autotune spaces can be huge -> compile gets SIGTERM'd. Pin the
  autotune to one config while iterating (env-gated `..._PIN`).
- "Correct but slower" is the normal starting point; matching TLX is the tuning
  work in Step 4, not a one-flag switch.

## Worked example (HSTU self-attention, GB200)

Two ways were tried, both correct (rel-L2 ~2e-3, tritonbench accuracy=1 vs GR;
the old NVGPUWarpSpecialization SiLU/PV-MMA crash no longer reproduces on current
Meta Triton):
- Standalone purpose-built autoWS kernel (`triton_autows_ragged_hstu`,
  `tl.range(warp_specialize=True)`): fwd 76.4 TFLOPS.
- SAME-kernel: added `warp_specialize=tl.constexpr(...)` to the hammer Triton
  kernel's main KV loop (env `HSTU_SELF_AUTOWS=1`). Untuned (num_warps=4,
  BLOCK_M=64) = 55.8 and produced `partition.types=[epilogue,gemm,load,
  computation]` with warps 1/1/4. Bumping to num_warps=8 -> 76.3 (matches the
  purpose-built kernel). Both still ~28% behind TLX (105.7) and below plain
  Triton (90.8) / GR (126).
Takeaway: enabling+correct is easy; the config (warps/tile) is the first lever
(55.8->76.3), and matching TLX needs the full Step 4 (register budget, per-
partition warp counts, num_stages, 2-CTA). Closing 76->106 is open work.
