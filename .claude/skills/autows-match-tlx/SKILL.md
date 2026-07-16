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
  - `TRITON_DISABLE_WSBARRIER_REORDER=1` - required by the WS lowering for attention.
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

**FIRST - pin TLX to its best config, then force autoWS to match it.** Before any
comparison, lock TLX to the winning config for your `(seq_len, HEAD_DIM)` and force
autoWS to the SAME tile / `num_warps` / `num_stages` (full procedure in Step 3.3);
comparing TLX-autotuned vs autoWS-autotuned pits two different tiles against each
other and HIDES the real gap. Everything below - and the Step 3.2 matching workflow
- assumes this ONE fully-specified config.

CRITICAL gotcha: `TRITON_USE_META_WS=1` is a **global** env applied to every
kernel compiled in the process. It will run the meta-WS pass on a hand-TLX
kernel too and typically CRASH it (e.g. TLX fwd uses `num_stages=0` ->
`numStages >= 1` assert). So autoWS and TLX **cannot run in the same process**.

- Run autoWS: `TRITON_USE_META_WS=1 TRITON_DISABLE_WSBARRIER_REORDER=1 ... --only <autows_variant>`
- Run TLX separately: `TRITON_DISABLE_WSBARRIER_REORDER=1 ... --only <tlx_variant>` (no META_WS)
- Use the same shape/metric and compare the two numbers. A plain-Triton kernel
  (e.g. a non-WS baseline) is unaffected by META_WS, so it can co-run with autoWS
  as the accuracy baseline.

## Step 3.1 - Data partitioning (DP) = the autoWS analog of TLX `replicate=NUM_MMA_GROUPS`

**What DP is.** Data partitioning splits ONE logical loop tile into N independent
compute groups that run concurrently on separate warp partitions, each handling
`tile / N` of the split dimension (usually M). It is how a kernel gets multiple
compute groups (a ping-pong / N-way overlap) instead of one, roughly N× the
compute-partition parallelism. TLX expresses it by hand as
`tlx.async_task(..., replicate=NUM_MMA_GROUPS)` (+ `BLOCK_M_SPLIT = BLOCK_M //
NUM_MMA_GROUPS`, indexed by a per-group `cid`); autoWS expresses it as the
`data_partition_factor=N` loop attr the `WSDataPartition` pass consumes. It is
distinct from *pipelining depth* (`num_stages` / buffer copies), which overlaps
iterations of a SINGLE group — a kernel can have DP, pipeline depth, both, or
neither.

**VERIFY whether TLX actually uses DP for the DIRECTION you are matching — do not
assume.** `NUM_MMA_GROUPS` / `replicate` / `cid` are frequently **forward-only**.
Concrete trap (HSTU, cost me three wrong claims): the HSTU *forward* persistent
kernel uses `NUM_MMA_GROUPS=2` (in `get_fwd_persistent_configs`,
`_softmax_inner_loop`/`_silu_inner_loop`), but the *backward*
(`_hstu_attn_bwd_ws_non_persistent`) uses **none** — `get_hstu_bwd_configs` never
sets `NUM_MMA_GROUPS` and the bwd body never references `cid`/`BLOCK_M_SPLIT`. A
`grep NUM_MMA_GROUPS` hit tells you nothing until you confirm its **enclosing
function** (`awk 'NR<=L && /^def /{f=$0} NR==L{print f}'`) and which config dict
sets it. If TLX-for-this-direction has no DP, then a DP=1 autoWS run is already
matched on that axis and the gap is elsewhere (pipeline depth / WS quality) — do
not chase DP.

TLX gets its multi-compute-group structure (when it has one) from
`tlx.async_task(..., replicate=NUM_MMA_GROUPS)`. The autoWS analog is a
**`data_partition_factor=N` kwarg on the annotated `tl.range`** (a `tl.constexpr`),
e.g.:
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

## Step 3.2 - THE MATCHING WORKFLOW (canonical ordered steps)

Match autoWS to TLX in THIS order, doing a **side-by-side final-ttgir diff after
EACH step** (compare **program order**, not `loop.cluster`; runtime correctness is a
SEPARATE parallel track - a broken kernel still emits a ttgir). Do NOT reorder -
each later step is computed against the earlier structure:

1. **Loop structure / persistency** - match persistent-vs-non-persistent; a single
   causal loop is fine (do NOT force TLX's masked+unmasked two-loop split).
2. **Tile** (BLOCK_M/N) - verify the ACTUAL compiled tile in the ttgir, not the
   knob. DP (Step 3.1: `data_partition_factor=N` = TLX's `replicate=NUM_MMA_GROUPS`)
   SPLITS one loop tile into N compute groups, each doing tile/N of the split dim
   (usually M) -> per-group GEMM = BLOCK_M/N. So tile-match on the GEMM shape in the
   IR **after DP**, NOT config BLOCK_M: to hit TLX's per-group gemm at DP=N set the
   annotated BLOCK_M = **N× TLX's per-group tile**. Ex (HSTU fwd): TLX BM=128 +
   NUM_MMA_GROUPS=2 (BLOCK_M=256) -> autoWS DP=2 needs BLOCK_M=256 (2x 128-row gemms
   = TLX); BLOCK_M=128+DP=2 gives 64-row gemms != TLX. VERIFY via `tc_gen5_mma`
   operand shapes AFTER the DP/partition pass, not the knob. See Step 3.1 for the
   fwd-vs-bwd caveat (NUM_MMA_GROUPS often fwd-only; a bwd with no DP needs no
   adjustment).
3. **Partition** - op CATEGORIES per partition (MMAs->gemm, loads->load,
   reduces->reduction, softmax->compute), NOT op counts.
4. **Schedule (SWP)** - derive `stage`/`order` from TLX's prologue/body/epilogue
   (prologue GEMM->stage0, epilogue->last stage, body GEMM order->`order`); needs
   `num_stages>=2`; SAME-stage dots keep PROGRAM order -> reorder by swapping call
   sites, not `order`. Also verify each gemm's **opndA location (SMEM vs TMEM)**
   matches TLX.
5. **Reuse EXCLUDING staging buffers** - match TMEM reuse GROUPS (shared alloc);
   verify with `[ws-summary]` (TLX has no `buffer.id` -> map by variable name).
6. **Staging buffers (depth)** - align dq staging depth; resolve OOM here
   (mem-plan search).
7. **Warps per partition**, then **2-CTA**.
8. **Register budget** - LAST.

The DP section above (Step 3.1) plus the sections below (Step 3.3 pin-TLX-config,
Step 3.4 reuse+`[ws-summary]`) and Step 4 (the ttgir-diff verification method +
detailed per-step notes) are the MECHANISM REFERENCE these numbered steps cite -
not a competing numbering.

## Step 3.3 - Fix TLX on its best config FIRST, then match autoWS to it

The mechanism behind the Step 3 baseline: this section only ESTABLISHES the shared
config; the IR alignment itself is the Step 3.2 workflow (do NOT re-walk tile /
partition / warps here). Do the comparison at ONE fully-specified config, not "TLX
autotuned vs autoWS autotuned" (that compares two different tiles and hides the real
gap).

1. **Pin TLX to its best config.** Read `get_<kernel>_bwd_configs()` (e.g.
   `get_hstu_bwd_configs`) and identify the winner for your `(seq_len, HEAD_DIM)`
   autotune key — or shrink the space to one entry (an env-gated `..._PIN`) so
   compile is fast and deterministic. Record EVERY field: `BLOCK_M1/N1/M2/N2`,
   `num_warps`, `num_stages`, and the staging/buffer knobs (`NUM_BUFFERS_Q/KV/DO/
   DS/TMEM`, `EPILOGUE_SUBTILE`, `DQ_REDUCE_FACTOR`, `DQ_REDUCE_STAGES`,
   `EARLY_RELEASE_SUBTILES`). These are the target autoWS must reproduce.
2. **Force autoWS to the SAME config.** Add an env knob (e.g. `HSTU_SELF_AUTOWS_BM`/
   `_BN`) to pin `BLOCK_M`/`BLOCK_N`/`num_warps`/`num_stages` to TLX's. This only
   fixes the config knobs; making the compiled IR actually MATCH (tile shape,
   partition, warps, schedule, reuse) is the Step 3.2 workflow.
3. Only now is a perf delta attributable to WS *quality* rather than config skew.
   Worked example (HSTU self-attn bwd, GB200): once autoWS dq-reduce was pinned to
   TLX's BM1=BN1=128 (and confirmed TLX bwd has no DP, Step 3.1), the residual
   ~8x bwd gap isolated cleanly to pipeline/staging depth + WS codegen quality.

## Step 3.4 - Align buffer reuse & staging with TLX (annotations + `[ws-summary]`)

TLX hand-places every TMEM/SMEM buffer (reuse groups, staging depth); autoWS's
memory planner decides heuristically and often differently. Two levers make
autoWS's memory plan match TLX's:

**ORDERING (important): settle the SWP schedule FIRST, then do these
memory-planner annotations.** Align the software-pipelining schedule — partition
structure and #GEMMs via `num_stages` / unroll / subtiling (Step 4 items 1–2) —
BEFORE touching reuse-group / staging-depth annotations. The memory planner
allocates against the *scheduled* op liveness (which buffers are simultaneously
live is set by the pipeline schedule), so annotations applied against an unsettled
schedule mismatch the eventual schedule or get silently invalidated when you later
change `num_stages`/unroll. EXCEPTION: a reuse annotation needed for CORRECTNESS or
TMEM-fit (e.g. the dq deadlock/512-col fix below) is a prerequisite and can go in
early; PERF-oriented staging/reuse tuning follows SWP alignment.

**(a) Reuse groups via per-dot channel annotations.** Pass
`attrs={"channels":["opndD,tmem,<copy>,<bufId>"]}` to `tl.dot`; MMAs sharing a
`bufId` form one TMEM reuse group (the compiler analog of TLX's `REUSE_DP_FOR_DQ`
/ `NUM_BUFFERS_TMEM=1`). Full spec:
`.../WarpSpecialization/docs/AnnotationBasedBufferPreAssignment.md` (partial
annotation is supported — un-annotated buffers stay heuristic). Worked example
(HSTU self-attn bwd, commit 603af6048): four groups pack to exactly 512 TMEM cols
at BM=BN=HEAD_DIM=128 — `id2` qk(+act reuses it as dv's opndA), `id5` dp(+dq =
REUSE_DP_FOR_DQ), `id7` dv, `id10` dk — which BOTH fits TMEM and fixes a
single-buffered-dq deadlock (dq inherits dp's cross-iteration WAR). Gotchas:
- Reuse validity needs the two accumulators **shape-compatible**; e.g. dp
  `[BLOCK_N,BLOCK_M]` and dq `[HEAD_DIM,BLOCK_M]` match only at `BLOCK_N==HEAD_DIM`.
  At a mismatched tile the reuse silently falls back.
- Ordering hazard (`BwdTmemReuseSlotHazard.md`): if two same-partition MMAs share a
  slot, emit the reader before the writer (e.g. dk before dq).
- The attrs value must be a **trace-time literal or constexpr kernel arg**, NOT a
  dict module-global (`NameError`) and NOT `constexpr(FrozenAttrs).value.get()`
  (tracer rejects the method call). Simplest: inline dict literal gated by a
  `tl.constexpr` bool — `attrs=({"channels":[...]} if _REUSE else None)`.

**(b) Staging depth via the `[ws-summary]` MemoryPlanner dump.** To compare
autoWS's actual per-buffer staging against TLX's config (`DQ_REDUCE_STAGES`,
`NUM_BUFFERS_Q`, `dq_store_buf` depth), dump the memory planner's summary:
```bash
MLIR_ENABLE_DUMP=1 TRITON_KERNEL_DUMP=1 ... python driver.py 2>&1 | grep -A80 '\[ws-summary\]'
# or run the pass alone with -debug-only on the pre-MP IR:
triton-opt --nvgpu-ws-memory-planner -debug-only=... pre_mp.mlir 2>&1 | grep -A80 'ws-summary'
```
`dumpPartitionAndBufferSummary` emits one row per buffer: `<tmem|smem> "name"
id=<buffer.id> cols=N rows=M colOffset=K ch#.. <prodTask> -> <consTask(s)>`.
Read off: which buffers share a `buffer.id` (reuse groups), each buffer's `copy`
(staging depth), and `colOffset` (packing). Compare against TLX: if TLX stages dq
through a depth-2 `dq_store_buf` (`DQ_REDUCE_STAGES=2`) but the ws-summary shows
autoWS's dq staging at `copy=1`, that single-slot staging is a concrete perf
(and sometimes deadlock) gap — raise it via the annotation copy count or a
memory-plan-search pick (Step 4.6). NOTE the emitter is behind `LLVM_DEBUG`; if
`[ws-summary]` is silent, confirm the `dumpPartitionAndBufferSummary` call site is
wired (it was dead code on some branches) and you built with asserts.

## Step 4 - Close the perf gap (make autoWS match TLX)

An untuned autoWS kernel is usually SLOWER than TLX (HSTU self-attn fwd example:
autoWS 76 vs TLX 106 vs GR-triton 123 TFLOPS).

### Verify by ttgir diff after EVERY fix - do NOT infer from config

The final ttgir is GROUND TRUTH; a config knob may silently not take effect (wrong
tile, DP declined, subtiling off, annotation fell back). NEVER attribute the gap
from config values or "should" reasoning - dump BOTH final ttgirs (autoWS meta-WS
on, TLX meta-WS off, separate processes; Step 3) and diff the checklist below after
EACH change. Every fix must move a metric TOWARD TLX; if a change leaves the ttgir
unchanged, the knob didn't apply (fix that before measuring). A perf number without
a ttgir diff behind it is not evidence.

**Runtime correctness and the ttgir comparison are SEPARATE, PARALLEL tracks - do
NOT block structural alignment on a runtime fix.** A kernel that hangs, OOMs, or
returns wrong results STILL emits its final ttgir at compile time (the dump is
written before launch). So at every step: (1) dump + diff the final ttgir toward
TLX regardless of runtime state, and (2) keep a running NOTE of any runtime issue
(deadlock / wrong grads / OOM) to fix on the parallel correctness track. Example
(HSTU bwd, num_stages=2): grads were wrong (dv rel-L2 0.5) yet the gemm-partition
ttgir comparison was fully informative - it showed BOTH the stage split
(qk/dv/dp=stage0, dk/dq=stage1) AND the order MATCHED TLX, so the failure was a
pure runtime/correctness bug (accumulator-carry at ns>=2), not a schedule mismatch.
**For the FINAL ttgir comparison, compare the ACTUAL PROGRAM ORDER of the ops in
the loop - do NOT use `loop.cluster`.** `loop.cluster`/`loop.stage` are
INTERMEDIATE scheduling attrs (post-ScheduleLoops, pre-pipeline); by the FINAL
ttgir the pipeliner has MATERIALIZED the schedule into the physical program order
(prologue peeled out + steady-state body in scheduled order), and `loop.cluster`
is stale/gone. So read the literal op sequence in the final loop body (and the
peeled prologue) and diff that against TLX's final-ttgir loop program order. Use
`loop.cluster` only when inspecting the pre-pipeline IR to see the intended
schedule; for the ground-truth final comparison, program order is authoritative.
   - **RULE - the `order` annotation only orders dots ACROSS clusters, NOT within a
     stage. Two dots in the SAME stage with the SAME `order` keep their SOURCE
     PROGRAM ORDER.** So to reorder two same-stage/same-order dots to match TLX, you
     must SWAP THEIR CALL SITES in the kernel source - changing `order` won't do it
     (and giving them different `order` would move one to a different cluster, which
     you may not want). Worked example (HSTU bwd): the only final-ttgir order diff
     vs TLX was dv/dp (both stage 0, order 2) - autoWS emitted dv-then-dp, TLX is
     dp-then-dv; fix = move the dp (`dact`) `tl.dot` above the dv `tl.dot` in the
     source (they are data-independent - dv reads act_qk, dp reads v/do^T - so the
     swap is safe), which makes the final program order match TLX.

### Compare PROLOGUE / LOOP-BODY / EPILOGUE vs TLX (do this once the SWP schedule annotation is aligned and autoWS emits a final ttgir)

Once Step 4's schedule (stage/order) is aligned and you HAVE a final autoWS ttgir,
split the gemm partition into its three physical regions and diff each against TLX
region-by-region - do NOT compare the flat MMA list. The 2-stage software pipeline
is MATERIALIZED by the **`TritonGPUPipeline` pass (the pipeline expander), which runs
AFTER the WS pass**; it peels a prologue + steady body + epilogue from the single
annotated loop. `TritonGPUScheduleLoops` DECIDES the schedule (`loop.stage` /
`loop.cluster`, driven by the `tt.autows` stage/order annotations); the expander only
EXECUTES it. So the prologue/epilogue shape is a consequence of the annotation, and
comparing the three regions is how you see whether autoWS's peel matches TLX's
hand-crafted 2-stage skew.

How to isolate which pass built the peel (when the peel looks wrong): recompile with
`MLIR_ENABLE_DUMP=1` (stderr), then for the gemm partition count `tc_gen5_mma`
inside-vs-outside the inner loop at each `IR Dump Before <pass>` / `WarpSpec internal
IR Dump After: <step>` boundary. The MMA count stays flat through AssignLatencies /
ScheduleLoops / all WS-internal steps (incl. `doLoopSchedule`) and JUMPS at
`TritonGPUPipeline` - that jump IS the peel. (Worked example, HSTU bwd ns=2: 5 MMAs
in one loop before `TritonGPUPipeline` -> 8-MMA prologue + 5-MMA body + `scf.if`
epilogue after it.)

Region-by-region table (autoWS | TLX), one row per region, list the MMA sequence
with use_acc (F=init/false, T=accumulate/true) per channel:
```
region     | autoWS                              | TLX (hand 2-stage)
prologue   | qk,dp,dv(F) | qk,dk,dq,dp,dv(T)  8  | qk,dp,dv(F)                3
body       | qk,dk(T),dq,dp,dv(T)                | qk,dq,dk(T),dp,dv(T)
epilogue   | (commits only, 0 MMAs)             | dk(T),dq(F)                2
```
What the diff tells you (HSTU bwd): TLX skews ONLY dk/dq BACKWARD (they lag one
block -> land in the EPILOGUE), keeping qk->act->dv depth-1 (prologue peels just the
dv INIT). An un-fixed autoWS expander instead does a FULL 2-stage fill: it hoists
iter-1's stage-0 `dv(T)` accumulate into the PROLOGUE (alongside iter-0's stage-1
dk/dq) -> the prologue is 8 MMAs / 2 pipeline time-slots instead of 3.

ROOT of the over-peel = a LATENCY, not the stage annotation. The per-op `loop.stage`
comes from the `tt.autows` annotation (ScheduleLoops), but the PEEL DEPTH (maxStage)
is inflated by the accumulator's `mmaSelfLatency`. In `AssignLatencies.cpp` the
loop-carried-accumulator branch (`useMetaWS && allUsersAreLoopCarried`) sets
`opLatency=0` then `continue`s, SKIPPING the line that would zero `mmaSelfLatency`
- so dv/dk keep self_latency=1, which raises maxStage -> the 2-part prologue.
**FIX (verified): add `mmaSelfLatency[mma]=0` before that `continue`** (upstream commit
`7f187af446`, FA bwd). Effect on HSTU bwd ns=2: gemm 13->10, prologue 8->3
(dv-init only, dv-acc back in the body), epilogue dk/dq restored = TLX structure;
dk NaN->finite. VERIFY the structure fix by: prologue dv-MMA count == 1.

CRITICAL LESSON: **matching the prologue/body/epilogue structure to TLX is NECESSARY
but NOT SUFFICIENT for correctness.** On HSTU bwd, after the mmaSelfLatency fix the
region split matched TLX exactly, yet dv was STILL wrong (rel-L2 0.5, growing with
seq_len) - a SEPARATE autoWS barrier/reuse-sync bug (the act/id2 cross-partition WAR),
independent of the peel. So run TWO checks: (1) structure (region diff vs TLX) AND
(2) correctness, ISOLATED - compile the same kernel with meta-WS OFF (plain triton):
if grads are correct there but wrong under autoWS, the residual bug is in the WS
barrier/reuse sync, not the schedule/peel - chase it with barrier-visualization
(per-channel FULL/EMPTY phase diff), NOT more schedule tuning.

**Produce a SIDE-BY-SIDE table (autoWS | TLX) of the metrics below AFTER EACH
alignment step - not just at the end.** Each step must show its metric moved to
TLX's column before you move on; a step is "matched" only when the side-by-side
row matches. Keep the whole table so regressions from later steps are visible.
Layout (one row per metric, two columns):

```
metric                     | autoWS                    | TLX
partition.types            | [reduction,gemm,load,comp]| (roles)
per-partition num_warps     | compute=8 gemm=1 load=1 …  | compute=8 gemm=1 …
op categories per partition | MMAs->gemm, reduce->reduc… | (same mapping)
gemm PROGRAM order (body)   | qk->dk->dq->dp->dv        | qk->dk->dq->dp->dv
prologue peel vs in-body    | in-body interleave        | 3-MMA prologue peel
TMEM alloc / reuse          | 4 groups id2/5/7/10 =512c | REUSE_DP_FOR_DQ, N allocs
SMEM staging depth          | multibuf N                | dq_store_buf depth 2
requestedRegisters          | (TUNE LAST)               | 192/80/80
```
CAVEATS when building the table:
- **Capture the DEFAULT region** - it is a partition too (often the reduction or
  computation role); an `awk` that only keys on `partitionN(` misses it.
- **TLX is hand-WS and has NO `buffer.id`** attrs - do NOT compare TMEM reuse by
  `buffer.id`; instead compare the count of DISTINCT `tmem_alloc`s and total TMEM
  columns (and which logical tensors share an alloc).
- Compare op CATEGORY->partition mapping, not op COUNTS (counts differ from
  pipelining/peel; see schedule step).

Metric checklist (grep both final ttgirs):
- `partition.types` AND which op CATEGORIES live in each partition: for each
  `partitionN(` region grep `tc_gen5_mma` / `async_tma_*` / `tt.reduce` /
  `tmem_load` to read its role. The same op CATEGORY must land in the same-role
  partition as TLX (compare category-to-partition mapping, NOT op counts).
- per-partition `num_warps(` (sum = true warp budget), NOT the config num_warps.
- **gemm PROGRAM order** in the final ttgir loop body (NOT `loop.cluster`).
- **# of GEMMs = `tc_gen5_mma` count** (align via the SWP schedule, below).
- `async_tma_reduce` count + SMEM multibuf depth (`memdesc<[2-9]x...`) = staging.
- TMEM: distinct `tmem_alloc` count + total cols + which tensors share an alloc.
- `wait_barrier` count.
- `requestedRegisters` - TUNE LAST.

### Alignment ORDER - loop structure first, registers LAST

Do these STRICTLY in order, and **dump both final ttgirs and compare after EACH
step** (a step isn't done until the ttgir metric moved toward TLX). Do not start a
later step until the earlier ones match - each later decision is computed against
the earlier structure, so aligning out of order is wasted/invalidated.

1. **Loop structure FIRST = PERSISTENCY alignment.** The loop-structure axis that
   must match is **persistency**: is there a persistent outer tile loop / CLC
   context, or is the kernel grid-launched one-tile-per-program? Compare the SAME
   mode - a persistent autoWS vs non-persistent TLX (or vice-versa) is not
   apples-to-apple. Check both: autoWS grep `tl.program_id`/persistent-outer-loop;
   TLX grep `clc_create_context` / a `..._non_persistent` variant + its
   `HSTU_TLX_PERSISTENT`-style selector (default may be PERSISTENT). Pick the TLX
   variant whose persistency matches autoWS (usually: autoWS bwd is non-persistent
   -> compare vs TLX `_non_persistent`).
   NOTE - do NOT force the inner causal masked/unmasked TWO-LOOP split. TLX often
   splits the causal iteration into a masked-boundary loop + an unmasked loop
   (2x static MMAs); autoWS uses ONE causal loop that masks uniformly (half the
   MMAs). A single causal loop is the RIGHT choice for autoWS (two loops are
   problematic for the WS pass), so the resulting `tc_gen5_mma` count difference
   (e.g. 5 vs 10) is EXPECTED and NOT a mismatch to fix - do not rewrite into two
   loops. Only persistency must align here.
2. **Tile match (BLOCK_M/BLOCK_N).** Force autoWS's tile to TLX's `BLOCK_M1/N1`.
   VERIFY THE ACTUAL COMPILED TILE IN THE TTGIR (mma operand `memdesc<AxB>` shapes
   and loop `step %cN`) - a config knob that "returns" a value can be the wrong
   knob or wrong function (real bug hit on HSTU: the bwd tile knob is
   `HSTU_SELF_AUTOWS_BWD_BM/_BN`, not `..._BM`; runs were silently at 64 not 128
   for a long time). OOM at the matched tile is OK for now - fix it in step 5/6.
3. **Partition match (op CATEGORIES per partition, NOT counts).** Check that the
   same *category* of op lands in the same-role partition as TLX - MMAs in the gemm
   partition, TMA loads in the load partition, `tt.reduce`/`async_tma_reduce` in the
   reduction partition, softmax/elementwise in the compute partition. Do NOT compare
   the NUMBER of ops per category here (5 vs 10 MMAs is the loop-structure/schedule
   concern from steps 1/4, not a partition mismatch). Only the category-to-partition
   mapping must match. Adjust via `merge_epilogue`, `separate_epilogue_store`, DP
   (Steps 3.1/3.3).
4. **Schedule match (SWP) via per-dot `stage`/`order` - works on the DEFAULT
   scheduler.** The default pipeliner `ScheduleLoops.cpp` (`add_schedule_loops`)
   DOES consume `tt.autows` `stage`/`order`: `scheduleKeyOpsAnnotation` (L703-742)
   parses `stage`/`order` off each MMA and builds the CoarseSchedule from them, and
   `scheduleKeyOps` (L762) tries it FIRST - "takes priority over all other
   scheduling." (This is how FA bwd's `_BWD_DOT_ATTRS` stage/order takes effect - no
   modulo/llm/list scheduler needed; those are separate, experimental, and buggy on
   bwd reduce_dq, T279643623 - avoid.)
   - **TWO gates can silently drop the annotation (verify with an MLIR pass-dump
     trace of the annotated `tc_gen5_mma`'s `loop.stage` before/after ScheduleLoops):**
     (a) `scheduleLoop` L886-908: if the loop ALREADY has `loop.stage`
     (`stageAssigned`) from a prior pass, annotation scheduling is DISABLED and the
     existing schedule is kept; (b) `getInitialSchedule` L788: `scheduleKeyOps` (the
     annotation consumer) is only called if `hasLatenciesAssigned` AND
     `isSafeToPipeline`. DOMINANT cause of (b): `AssignLatencies` BAILS for
     `num_stages <= 1` (AssignLatencies.cpp:289-291) -> a num_stages=1 loop gets NO
     latencies -> `hasLatenciesAssigned=false` -> annotation never consumed. VERIFIED
     end-to-end on HSTU bwd via pass-dump `loop.stage` trace: at num_stages=1 the
     annotated MMAs stay loop.stage-ABSENT through every pass (inert); switching to
     **num_stages>=2** -> AssignLatencies assigns latencies -> ScheduleLoops ->
     `scheduleKeyOpsAnnotation` sets `loop.stage` to EXACTLY the annotation (annotated
     stage 0/0/0/1/1 -> loop.stage 0/0/0/1/1). So **stage/order-driven scheduling
     REQUIRES num_stages>=2** (this is how FA's works). Use the correct per-direction
     num_stages knob (HSTU bwd = `HSTU_SELF_AUTOWS_BWD_STAGES`, not the fwd knob) and
     verify with the pass-dump `loop.stage` trace, not a single final diff.
   - So verify effect the right way: a **clean semantic ttgir diff** (strip
     `loc(...)` and the `tt.autows` string) - NOT op counts and NOT the dump hash
     (both change trivially with the source and mislead).
   - **DERIVE the stage/order values from the TLX kernel's SWP schedule - do NOT
     guess them.** A hand-TLX kernel's software-pipeline is structured as
     `prologue + steady-state loop body + epilogue`. Read that structure and map:
       * a GEMM peeled into the **prologue** -> `stage = 0`;
       * a GEMM in the **epilogue** -> `stage = 1` (i.e. last stage = num_stages-1);
       * the **left-to-right order of the GEMMs within the steady-state loop body**
         -> the `order` values (0,1,2,... in body order).
     Then set each autoWS dot's `stage`/`order` to those values so the default
     scheduler reproduces TLX's pipeline. CAUTION: running accumulators (`acc +=`,
     e.g. dv/dk) have a loop-carried dependency - their stage assignment must keep
     the accumulator read/write in a consistent stage or the pipelined value goes
     stale (observed failure: a guessed FA-style split gave dv rel-L2 0.5 / a
     deadlock at num_stages=2). Get the split from TLX's actual prologue/epilogue
     placement, not from another kernel's table.
5. **Reuse match EXCLUDING staging buffers.** Align the TMEM reuse GROUPS (which
   accumulators share a `buffer.id`) via the per-dot `channels` annotations (Step
   3.4a) - `REUSE_DP_FOR_DQ` etc. Do NOT try to match staging-buffer depth yet.
   **How to verify the reuse grouping matches (TLX has NO `buffer.id`, so map by
   variable name + `reuse=`):**
   - TLX side: grep `storage_kind.tmem` / `reuse=` (ttgir2tlx) or `tmem_alloc` +
     `memdesc_reinterpret` (raw) to build the variable->group map - which NAMED
     tensors share one TMEM alloc. Example (HSTU bwd): `qk_tiles`+`qk_tiles_11`
     (reuse) = {qk, act}; `dp_tiles`+dq (REUSE_DP_FOR_DQ) = {dp, dq}; `dv_tiles`;
     `dk_tiles` -> **4 TMEM allocs, 512 cols**.
   - autoWS side: group `ttng.tmem_alloc` by `buffer.id` (same id = one group);
     confirm they are TMEM not SMEM (`ttg.local_alloc` with a buffer.id is SMEM -
     those ids are a SEPARATE space, don't count them as TMEM allocs).
   - Then check: SAME set of tensors grouped, SAME count of distinct TMEM allocs +
     total cols. Worked example: autoWS id2={qk,act}, id5={dp,dq}, id7={dv},
     id10={dk} = 4 allocs/512c -> EXACTLY TLX's grouping. Match confirmed.
6. **Staging buffers (depth) - after 1-5 match.** Now align dq staging depth /
   `NUM_BUFFERS_*` and resolve any OOM, using `[ws-summary]` (Step 3.4b) and
   memory-plan search (`TRITON_WS_SMEM_PLAN_SEARCH=1` + `mem_plan_pick`).
7. **Warp counts per partition**, then **2-CTA** (`ctas_per_cga`) if TLX uses it.
8. **Register budget - LAST.** ONLY after 1-7 match in the ttgir.
   `requestedRegisters` role-based (FA bwd): the register-heavy computation
   partition = the default region (residual budget); producers/reduction get a
   balanced modest budget (~88) under the setmaxnreg cap (256). Check spills with
   CUDA-13 `ptxas -arch=sm_100a -v` (0 spill). Tuning registers on a
   structurally-mismatched kernel is wasted - that is why it is last.

Iterate: change ONE thing -> dump BOTH ttgirs -> confirm the metric moved toward
TLX -> next step. Loop structure -> tile -> partition -> schedule -> reuse
(excl staging) -> staging -> warps/2-CTA -> registers.

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
