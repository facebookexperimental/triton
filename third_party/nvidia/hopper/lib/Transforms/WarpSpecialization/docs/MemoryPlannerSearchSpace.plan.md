# Memory Planner Search Space — Design & Implementation Plan

**Status**: SMEM + TMEM search implemented (default off)

## Implementation status

Landed (all files under `WarpSpecialization/`, gated by `--smem-plan-search` /
`TRITON_WS_SMEM_PLAN_SEARCH`, default off — the one flag enables both pools):
- Interfaces `WSMemoryPlanSearch.h`; ordering (`LivenessStartOrder` +
  `TopologicalOrder`), cost, greedy copies, beam driver — pure modules.
- SMEM: `SmemBufferModel` + `SmemPacker`, wired via `allocateSmemBuffersViaSearch`
  with a floor safety net; reuse gated by encoding + reuseScope (same-block).
- TMEM: `TmemBufferModel` + `TmemPacker` (time-multiplexed, liveness-disjoint
  column reuse; copies pinned to 1), wired via `allocateTmemBuffersViaSearch`.
- Validated on sm100 with the full `run_all.sh` suite, **off and on**: all 7
  autoWS files pass identically (GEMM 216, addmm 73, quantized 3, FA-correctness
  72, FA tutorial 30, cross-attn-bwd 18 + 6 xfail; rest skipped) — the search
  introduces no correctness difference. WarpSpecialization LIT: 113 passed /
  11 XFAIL / 0 unexpected. (Full LIT tree has 3 failures in TLX-layout and
  AMD-atomic tests, unrelated to this change and pre-existing.)
- A full-suite hang in `fused_attention_ws_device_tma.py` was found and fixed:
  TMEM reuse now also requires a bidirectional data dependency (not just disjoint
  liveness), matching `hasPotentialReuse` — independent buffers can be concurrent
  across WS partitions despite disjoint op-id intervals.
- SMEM search does **per-buffer multi-buffering only — no circular reuse
  grouping**. Encoding + basic-block match is NOT a sufficient reuse condition
  (two concurrently-live operands, e.g. GEMM/addmm A and B, satisfy it but must
  not share a circular buffer — deterministic wrong results / races). The
  upstream heuristic defaults `smem-circular-reuse` OFF for the same reason.
- Falls back to the heuristic for pins, multi-store (subtiled) staging (SMEM),
  and subtiled regions / scaled MMA (TMEM) — see §8. Single-store staging (S=1)
  is searched (own block, floor copy).

Top-K search space (mirrors the list/modulo schedulers):
- `TRITON_WS_MEM_PLAN_TOPK=K` generate K ranked plans; `TRITON_WS_MEM_PLAN_PICK=i`
  apply rank i (0 = cost-best, clamped per pool); `TRITON_WS_MEM_PLAN_TOPK_DUMP`
  dumps each plan as JSON. A harness sweeps PICK over 0..K-1 and times each (the
  cost model only ranks). Validated: PICK 0..3 with TOPK=4 all legal + pass.
- **Autotune-native**: `tl.range(..., mem_plan_pick=<constexpr>)` stamps
  `tt.mem_plan_pick` on the loop (mirroring `tt.list_schedule_pick`); the planner
  reads it (attr > env > 0), and being a `tl.constexpr` it is part of the
  compilation key so `@triton.autotune` sweeps it without env vars. Validated:
  attr flows tl.range → TTIR → TTGIR (planner level).

Validation status: both pools are runtime-validated on the full `run_all.sh`
suite, **search on**: GEMM 216, addmm 73, quantized 3, FA-correctness 72, FA
tutorial 30, cross-attn-bwd 18 + 6 xfail — all pass, no regressions, no hangs.
The SMEM search engages on FA/GEMM/addmm (per-buffer multi-buffering); the TMEM
search engages on GEMM. Getting here surfaced and fixed two real bugs the suite
caught: a TMEM reuse hang (missing data-dependency) and an SMEM reuse-grouping
corruption (now disabled).

Staging copies (search axis): `TRITON_WS_STAGING_COPIES=K` caps the fused
TMA-staging pipeline depth (`increaseFusedEpilogueCopies`) to `min(numBuffers, K)`
— K|S divisibility, cross-stage floor, and budget all still enforced, so a
harness sweeping K explores only legal staging depths via the proven path.
Staging copies exist only for subtiled staging (S>1); S=1 is fixed at K=1.
Validated: sweeping K on a subtiled addmm case gives depths {floor, 2, 3}, all
pass. (This is a heuristic-path knob; the beam itself still defers multi-store
staging — reimplementing fusion + K|S in the beam is deliberately avoided.)

Remaining (perf / coverage, not correctness):
- **Safe SMEM reuse grouping** — the true rotating-entry condition (not just
  encoding + scope), to reclaim the reuse the heuristic gets under
  `smem-circular-reuse`. Currently disabled. (Studied: the safe condition is
  exactly-2 same-priority innermost buffers with `copy = 2·N−1`; reimplementing
  in the beam is low-value since the heuristic already does it.)
- **Staging reuse** (spatial aliasing, Phase 3.6 `findReuseCandidate` /
  `mergeStagingReuseIntoHost` — a non-innermost buffer aliases another's bytes).
  **Attempted and reverted**: a `TRITON_WS_STAGING_REUSE` knob to run Phase 3.6
  *proactively* (not only under budget pressure) regressed FA (4/4 fa_dp) and an
  addmm case. The reuse's liveness-disjointness is *approximate*
  (partition-unaware, last-consumer-order based); the `baseTotal > smemBudget`
  trigger is **load-bearing** for its safety envelope, so forcing it broadly
  aliases buffers that aren't actually disjoint → wrong results. Making staging
  reuse a *legal* search axis needs a partition-aware liveness/disjointness
  check (a real project), not just exposing the existing trigger.
- Accumulator per-outer-tile TMEM multi-copy (currently pinned to 1).
- Model scaled-MMA scale columns for TMEM.
- **N-way (≥3) TMEM reuse grouping, aligned with code partitioning** — see the
  dedicated section below. This is the gap that keeps FA-bwd `_BWD_DOT_ATTRS_TMEM`
  from being reproducible without hand-pinned `channels` (task T279873316).
**Covers (future)**: `WSMemoryPlanner.cpp`, new `WSMemoryPlanSearch.{h,cpp}`
**Related docs**: [SmemAllocationDesign.md](SmemAllocationDesign.md),
[TMEMAllocationHeuristics.md](TMEMAllocationHeuristics.md),
[ReuseGroups.md](ReuseGroups.md), [AccumulationCounters.md](AccumulationCounters.md),
[BwdTmemReuseSlotHazard.md](BwdTmemReuseSlotHazard.md)

---

## N-way (≥3) TMEM reuse grouping — align the planner with code partitioning

**Status: dependency-walk unification DONE (commit `ff0bff2e4`); sound N-way
predicate + verify harness DONE (`80dde009a`); enumeration-path N-way join DONE
(`b019a73be`, NFC); planner N-way GROUPING now SOLVED via a post-pass
(`repairUnsafeReuseGroups`, working change) — `test_bwd_bm128_memtype_only_xfail`
PASSES. Task T279873316. See "Implementation log & final design (2026-07-16)"
below for the shipped design and the dead ends.**

**Done (`ff0bff2e4`, refined by `8997960ee`):** one shared `dependsThroughMemory`
(CodePartitionUtility) used by both passes; a `getRootBuffer` subview-climb lets
the walk cross a multi-buffered alloc (store into `memdesc_index(base,i)` connects
to a read from `memdesc_index(base,j)` — the shape that hid `dsT→dq`). The climb
is gated behind `followBufferReuse`, used ONLY by code partitioning's
`hasDependencyChain` (to ORDER an already-decided reuse); the planner's
`isDataDependent` keeps the narrow walk (the wide walk over-forms reuse groups).

**REVERTED (`8997960ee`) — the same-partition restriction is UNSOUND (again).**
`ff0bff2e4` also restricted the program-order fallback to a single partition
(dropping cross-partition textual order). A full-suite run caught the regression
this warns about below: **HSTU 2-KV cross-attention bwd (`reduce_dq`) produced
wrong `dq` (rel-L2 ~0.79)** — a real kernel with a legitimate cross-partition
program-order reuse that `getRootBuffer` does not capture as a memory dep. Fix:
restore cross-partition program order. **Consequence: the hand-pinned
`{qkT,dpT,dq}` bad-group no longer fast-fails** (it is spuriously orderable via
program order → downstream blowup). Genuinely-unorderable groups
(`{dpT,dsT,dq}` in `ws_code_partition_tmem_3group_no_chain`) still fail fast.
So the badgroup fast-fail remains the OPEN N-way crux, exactly as below.

Validated after the fix: FA-bwd (40), HSTU cross-attn bwd (18), autoWS
GEMM+addmm+quant (140), autoWS FA (72), HSTU self-attn fwd/bwd (2/2), LIT
WarpSpecialization (125; decision checks in `verify_reuse_group_decisions.mlir`).

**Still TODO:** the planner's `hasPotentialReuse` is pairwise/greedy and does not
*assemble* the N≥3 group `{dpT,dsT,dq}` even though the pairwise deps are now all
visible (`test_bwd_bm128_memtype_only_xfail` still xfails). That is the remaining
group-formation work — see "What's left — 3 steps + gate" below — no longer blocked
on the dep predicate.

### The gap
Today the planner forms TMEM reuse groups **pairwise** via `hasPotentialReuse`
(disjoint op-id liveness + a *direct* data dependency). That cannot form a real
N≥3 chain like FA-bwd `{dpT, dsT, dq}` (config `_BWD_DOT_ATTRS_TMEM`), because the
`dq↔dsT` pair has **no pairwise dependency**: `dq` reads `dsT` from SMEM while
`dk` reads it from TMEM, so they are common-ancestor siblings, not a direct
producer→consumer. So the group only ever forms **by hand-pinned `channels`**.
Consequences observed (BM128 memtype-only, `test_bwd_bm128_memtype_only_xfail`):
the planner instead picks `{dpT,dq}`+`{dsT,ppT,qkT}` and **OOBs TMEM**, because the
tight 3-way `{dpT,dq,dsT}` (which fits) is unreachable.

### The alignment idea
Code partitioning already has the **group-level** safety predicate the planner
lacks: `verifyReuseGroupCrossPartition` + `orderReuseGroupChain`
(`CodePartitionUtility.cpp`). It accepts an N≥3 single-copy TMEM group iff its
channels admit a **unique total dependency-chain order** (Kahn with a unique head
each step; edges from `hasDependencyChain` = SSA use-def **or** same-block program
order). For `{dpT,dsT,dq}` that is `dpT→dsT` (SSA) then `dsT→dq`. Factor this into
**one shared predicate** used by both passes so the planner forms only groups
code partitioning can prove safe — making the existing "handled-or-bail /
`report_fatal_error`" contract unreachable instead of a late surprise.

### Two lessons from reverted attempts (do NOT repeat)

1. **Pairwise "same-partition disjoint reuse" in the planner is unsound.** It
   found a fitting packing (peak 448) but (a) blew up compile >5 min and (b) chose
   `{qkT,dpT,dq}` — `qkT`'s slot is not truly free (its value lives on through
   `pT`). Op-id-disjoint within a partition is *not* real disjointness.

2. **Tightening `hasDependencyChain`'s program-order fallback to same-partition,
   *without* a memory-complete data-dep walk, is unsound.** On its own it
   regressed 11 FA-bwd + 1 lit test: group `{dpT,dq}` (channels 9/12, `buffer.id
   5`) went `verifyReuseGroup2 chain=1 → chain=0`, dropping its cross-iteration WAR
   barrier → wrong `dq`. **Root cause (later found):** the `dpT→dsT→dq` dependency
   *is* real, but the walk couldn't trace it — `dsT` lives in a multi-buffered
   alloc and the store/read use different `memdesc_index` slot-views, so the walk
   dead-ended at the store's subview. The good binary was leaning on the
   cross-partition program-order fallback to paper over this. **Resolution
   (`ff0bff2e4`):** make the walk climb subview→base (`getRootBuffer`) so the real
   dep is found; *then* the same-partition-only rule is sound and
   non-regressing. Lesson: don't drop the program-order fallback until the
   data-dep walk is memory-complete.

### The two helper stacks — the divergence to unify
The two passes maintain **parallel, duplicated** reuse machinery that has drifted
apart:

| Concern | Memory planner (`WSMemoryPlanner.cpp`) | Code partitioning (`CodePartitionUtility.cpp`) |
|---|---|---|
| dependency walk | `isDataDependent` (l.3025) — **memory-aware**: follows SSA results **and** store→memdesc→load | `hasDependencyChain` (l.780) — **memory-blind** (SSA results only) **+ a program-order fallback** |
| pairwise reuse legality | `hasPotentialReuse` (l.3707): size-fits **AND** op-id **liveness disjoint** (`bufferRange` intervals don't intersect) **AND** a data dependency (either dir) | `verifyReuseGroup2` (l.905): column **overlap** **AND** a chain (data-dep **or** same-block program order) |
| N≥3 group legality | **— none —** (pairwise, greedy) | `orderReuseGroupChain` / `verifyReuseGroupCrossPartition` (unique Kahn order) |
| ordering evidence | **op-id liveness disjointness** (a scheduling fact) | re-derives order from chain direction / program order |

`verifyReuseGroup2` is called **only** by code partitioning (`WSCodePartition.cpp:3370`);
the planner never calls it. So the planner *decides* a reuse from liveness+data-dep,
and code partitioning independently *re-proves* it from chain direction to place the
barrier — two predicates that can disagree.

Note the dq path (corrects an earlier misreading): `dq`'s **producer** is the GEMM
`tc_gen5_mma %dq_trans(=memdesc_trans %dsT), %k, %dq`, which reads `dsT` **directly**
from the shared SMEM buffer — a clean `dpT→dsT→dq_producer` chain, no staging. The
early-TMA **staging is on dq's *consumer*** side (`tmem_load %dq` → reshape/trans/
split → `descriptor_reduce` for the subtiled dQ atomic add). Staging is irrelevant to
reuse ordering.

### Cases checked and the decision each pass should reach
| group | data dep? | op-id liveness disjoint? | desired decision | reason |
|---|---|---|---|---|
| `{dpT,dq}` (id5) | **yes** — `dpT→dsT→dq` via `memdesc_trans` of shared `%dsT` | yes | **accept**, order `dpT`→`dq`, emit cross-iter WAR barrier | real producer→consumer |
| `{dpT,dsT,dq}` (3-way, target) | `dpT→dsT` (SSA), `dsT→dq` (shared buf) | yes | **accept**, unique chain order | the config only hand-pinning produces today |
| `{qkT,ppT}` | `qkT→…→ppT` (p feeds ppT) | yes | **accept** | real chain |
| `{qkT,dpT,dq}` (badgroup) | **no** (`qkT⊥dpT`) | **no** — `qkT` lives on through `pT` | **reject** | independent + overlapping liveness → race, currently hangs code-partition >5 min |
| `{ppT,dsT}` (badgroup) | **no** (`ppT⊥dsT`) | — | **reject** | independent members, no safe order |

The planner's own liveness gate (`hasPotentialReuse` l.3713: `if intersects → return 0`)
**already rejects** `{qkT,dpT,dq}` — which is why the planner never proposes it and it
only arises via hand-pinned `channels`. The robustness gap is purely that a
pinned-illegal group is not *fast-failed*.

### The real discriminator (corrected)
`{qkT,dpT,dq}` (unsafe) vs `{dpT,dsT,dq}` (safe) differ by whether the members
admit a **genuine dependency-chain order**, **not** by op-id liveness. Op-id
liveness is NOT a sound signal — the `test_bwd_bm128_memtype_only_xfail` reason
confirms it **underestimates `qkT`'s lifetime** (its value lives on through `pT`),
so a liveness-disjoint test would wrongly accept `qkT` sharing. The sound gate is
the **group-level dependency-chain** predicate `orderReuseGroupChain`: a unique
total order over the members whose edges come from `dependsThroughMemory`
(memory-complete, `followBufferReuse=true`) or in-partition program order.
`{dpT,dsT,dq}` orders uniquely (`dpT→dsT` data dep, then `dsT→dq`); `{qkT,dpT,dq}`
has no order (`qkT⊥dpT`) → rejected. (An earlier revision of this doc proposed a
`livenessOrder` edge; that is superseded — see "Empirical note — RESOLVED".)

### What's left — 3 steps + gate (to make `test_bwd_bm128_memtype_only_xfail` pass)
**Prereq DONE:** the shared memory-complete `dependsThroughMemory` (+`getRootBuffer`)
and code partitioning's `orderReuseGroupChain` / `verifyReuseGroupCrossPartition`,
which already order the 3-way — proven by `ws_code_partition_tmem_3group_chain.mlir`
and the hand-pinned `_BWD_DOT_ATTRS_TMEM`. The only gap is that the **planner does
not FORM the group**.

1. **Share the group-level legality predicate.** Expose `orderReuseGroupChain` /
   `verifyReuseGroupCrossPartition` (backed by `dependsThroughMemory(followBufferReuse
   =true)`) so the planner can validate a candidate N≥3 TMEM group — even when a
   member pair (`dq↔dsT(tmem)`) has **no pairwise dep** (common-ancestor siblings:
   `dk` reads the TMEM copy, `dq` the SMEM copy).

2. **N-way group formation in the planner (the new logic).** Extend reuse formation
   from pairwise-greedy to: propose **extending** a reuse group with a further member
   and accept iff the shared predicate returns a **unique chain**. Keep the
   **pairwise packing gate narrow** (`hasPotentialReuse` / `followBufferReuse=false`)
   — only the N-way *extension* check uses the wide/group predicate. This forms
   `{dpT,dsT,dq}` **without** the wide walk over-forming pairwise groups (which
   regressed HSTU 2-KV). Wire it into the plan-space search's TMEM packer, which today
   does only pairwise liveness-disjoint column reuse (copies pinned to 1).

3. **Feasibility-aware ranking.** Prefer the grouping that **fits** — `{dpT,dsT,dq}`
   (512 cols) — over the greedy `{dpT,dq}`+`{dsT,ppT,qkT}` that **OOBs**. Reject
   infeasible (OOB) plans in the static feasibility gate and let the search explore
   the alternative grouping; rank by the shared predicate + occupancy, not occupancy
   alone.

**Gate:** `test_bwd_bm128_memtype_only_xfail` flips to pass (planner forms
`{dpT,dsT,dq}` from memtype-only annotations); and NO regression — `test_bwd_tmem_
dsT_reuse_3group` (hand-pinned 3-way), FA-bwd (40), HSTU cross-attn bwd (18), autoWS
GEMM+addmm+quant (140)/FA (72)/self-attn (2/2), WS LIT (125) all stay green. The key
invariant: N-way formation must NOT broaden the *pairwise* packing gate (keep
`followBufferReuse=false` there), or HSTU 2-KV over-forms and miscompiles again.

(Separately, the pinned-illegal `t_bwd_badgroup.py` fast-fail — reaching the
unorderable-group `report_fatal_error` before the >5-min downstream blowup — is a
robustness nicety, not required for the gate above.)

### Empirical note — RESOLVED
The earlier `chain=0` puzzle was the multi-buffer/subview blind spot above: the
`{dpT,dq}` chain flows `dpT.consumer(tmem_load) → dsT(store into memdesc_index base,i)
→ dq(reads memdesc_index base,j)`, and the walk dead-ended at the store's subview.
`getRootBuffer` (climb subview→base, fan to all slot-views) fixes it — confirmed by
instrumented `dataDep` traces going `0→1` and the FA-bwd suite going green. The
ordering source is therefore the **data dependency** (through the buffer), not a
separate `livenessOrder`; same-partition program order remains only for genuine
in-partition ordering. Regression-guarded by `reuse_group_2buffer_multibuf.mlir`.

---

## Implementation log & final design (2026-07-16)

This section closes the N-way TMEM reuse thread above. It documents the shipped
design (a **post-pass repair**, `repairUnsafeReuseGroups`), the sound predicate it
rests on, every alternative that was tried and rejected, and two supporting bug
fixes that had to land first. Task **T279873316**.

Commits on this branch (`git log --oneline`), oldest→newest:
- `f99b9c0ce` — Fix f16-in-f32 TMEM reuse materialization validation (`TMEMAlloc1D.cpp`)
- `45e976ce9` — Fix use-after-free on reuse-folded channel endpoints (`defunct` flag)
- `80dde009a` — Sound N-way reuse predicate (`crossPartitionProgOrder`) + memory-planner verify harness
- `b019a73be` — [NFC] N-way sibling reuse formation in the TMEM **enumeration** path (`canJoinReuseGroupChain`)
- **working change (uncommitted)** — `repairUnsafeReuseGroups` post-pass in `WSMemoryPlanner.cpp` (the fix that makes BM128 pass)

### 1. Problem recap — why BM128 needs `{dpT, dsT, dq}`

FA-bwd at `BLOCK_M1=128` runs five MMAs whose TMEM accumulators/operands must be
packed into ≤512 columns: `qkT`, `dpT`, `dv`, `dq`, `dk`. The hand-tuned config
`_BWD_DOT_ATTRS_TMEM`
(`fused_attention_ws_device_tma.py:607`) makes this fit by pinning **buffer
ids** (the trailing int of each `"opnd..,mem,copies,ID"` channel string):

```
qkT: opndD,tmem,1,2   dpT: opndD,tmem,1,5   dv: opndA,tmem,1,2 / opndD,tmem,1,7
dq:  opndD,tmem,1,5   dk:  opndA,tmem,1,5  / opndD,tmem,1,10
```

Decoding the shared ids: **id 2** = `{qkT`'s output (opndD), `ppT` = dv's opndA}`
(the "qk shares with ppT" 2-way); **id 5** = `{dpT`'s output, `dq`'s output,
`dsT` = dk's opndA}` — the **3-way** `{dpT, dsT, dq}`; id 7 = dv; id 10 = dk.

The 3-way is the crux. `dq` and `dsT` are **common-ancestor siblings**, not a
producer→consumer pair:
- `dpT → dsT` is a real SSA data dep (`dsT` is computed from `dpT`).
- `dk` reads `dsT` from **TMEM** (the `opndA,tmem,1,5` operand).
- `dq`'s producer GEMM reads `dsT` from **SMEM** via `memdesc_trans %dsT`.

So there is **no pairwise `dsT→dq` (or `dq↔dsT`) producer→consumer edge**: both
`dq` and `dk` consume the same ancestor `dsT` through different memory spaces. The
planner's pairwise gate `hasPotentialReuse`
(`WSMemoryPlanner.cpp:3742`) requires size-fit **AND** op-id liveness
disjointness **AND** a bidirectional data dependency (`isDataDependent`,
`:3757–3758`). With no `dq↔dsT` data dep, `hasPotentialReuse` returns 0 for that
pair, so the 3-way can never be assembled from pairwise steps. Without it the
planner packs `{dpT,dq}` + `{dsT,ppT,qkT}` and **OOBs TMEM** (the tight 512-col
3-way is unreachable) — the `xfail` at `fused_attention_ws_device_tma.py:1920`.

### 2. Architecture: search vs heuristics vs post-pass

TMEM allocation lives in `allocateTMemAllocs2` (`WSMemoryPlanner.cpp:4230`) and
has **two search modes**, both driven by the **same heuristics**:

| Mode | Entry | When |
|---|---|---|
| First-fit (default) | `tryAllocate` (`:3907`), a greedy DFS returning the **first** feasible packing | always, unless top-K opt-in |
| Top-K enumeration | `enumerateTMemAllocations` (`:4128`), collects distinct feasible packings, ranks by peak columns | opt-in: `TRITON_WS_MEM_PLAN_TOPK>1` or a `mem_plan_pick`>0 |

Both modes call the identical legality/placement heuristics:
- `hasPotentialReuse` (`:3742`) — the pairwise reuse gate (size + liveness + data dep).
- `computeColOffset` (`:3775`) — where a candidate packs inside an owner's columns.
- `findPlacements` (`:3696`) — where to open a brand-new column range (row-group gap search).
- `tmemStatePeakCols` (`:4084`) — the peak-column occupancy used to rank enumerated plans.

**Owner identity is emergent, not chosen.** A buffer becomes a reuse-group
**owner** only when it opens new space (`findPlacements` → `addOwnerToState`);
every other buffer becomes a **reuser** of some owner (`hasPotentialReuse` +
`computeColOffset`). Nothing declares "dsT is an owner"; whichever buffer first
needs fresh columns holds the slot. This is central to why the dead ends below
failed — you cannot make `dpT` an owner just by relaxing a predicate.

**Default-path guard (byte-identical).** The dispatch at
`WSMemoryPlanner.cpp:4297–4301` reads `topK`/`pick` and only enters enumeration
when `topK > 1 || pick > 0`; the comment states *"Default (topK=1, pick=0) keeps
the exact first-fit path so non-search compiles are byte-identical."* Rank 0 is
always pinned to the first-fit packing (`:4326–4338`), so `pick=0` under search
still equals the default. This guard is what lets the N-way experiments (and the
verify harness) exist without touching any shipping compile.

### 3. The sound predicate (`80dde009a`)

The group-level legality question — *"do these N channels admit a unique
dependency-chain order?"* — was already answered for **code partitioning** by
`orderReuseGroupChain` / `verifyReuseGroupCrossPartition`
(`CodePartitionUtility.cpp`), a Kahn topological sort over edges from
`hasDependencyChain` (`:869`). The edge policy had two sources: (1) a
memory-complete data dep (`dependsThroughMemory(..., followBufferReuse=true)`),
and (2) **same-block program order** — used *even across partitions*.

`80dde009a` adds a `crossPartitionProgOrder` flag to `hasDependencyChain` and
`orderReuseGroupChain` (default `true`):

```cpp
// hasDependencyChain(A, B, crossPartitionProgOrder=true)
// (2) program order within the same block.
if (aConsumer->getBlock() == bProducer->getBlock()) {
  if (!crossPartitionProgOrder) {
    // Sound-gate mode: textual order is a happens-before ONLY when a single
    // partition (shared async_task_id) runs both ops.
    ... if (!sharePartition) return false; ...
  }
  return appearsBefore(aConsumer, bProducer);
}
```

**Why cross-partition textual order is not a happens-before.** Distinct async
tasks (warp groups) run **concurrently**; their relative position in the IR text
says nothing about execution order. For *ordering an already-decided* reuse (what
code partitioning does — it just needs to place the WAR barrier), that
approximation happens to be load-bearing for some real kernels, so the default
stays `true`. But for **deciding whether to FORM** a group it is unsound: it would
"order" independent siblings that actually race.

**What dropping it buys (the group-formation gate, `crossPartitionProgOrder=false`):**
- `{qkT, ppT, dsT}` — `ppT` and `dsT` are cross-partition and data-independent →
  no edge → **REJECT** (this is the group first-fit wrongly forms at BM128).
- `{qkT, dpT, dq}` — `qkT ⊥ dpT` → **REJECT**.
- `{dpT, dsT, dq}` — `dpT→dsT` (SSA) and `dsT→dq` (through the shared buffer, via
  `getRootBuffer`) are genuine data deps → orders uniquely `dpT→dsT→dq` →
  **ACCEPT**.

So the sound gate accepts exactly the real chain and rejects the two spurious
cross-partition sibling groups — without changing code partitioning (which keeps
the default `true`).

**Verify harness + lit test (the fast-iteration net).** `80dde009a` also wires an
observation-only mode into the planner
(`WSMemoryPlanner.cpp:3403`, gated by `TRITON_WS_MEM_PLAN_VERIFY_GROUPS`): it
enumerates every candidate pair/triple of a loop's TMEM allocs, runs
`orderReuseGroupChain(..., /*crossPartitionProgOrder=*/false)`, and prints
`[ws-mem-plan-verify] group {..} => ACCEPT order=… / REJECT` per group. It
**changes no planning decision** (prints via `llvm::errs()`, no assertions build
needed), so it stays on a plain RUN line. `test/Hopper/WarpSpecialization/ws_mem_plan_verify_reuse_groups.mlir`
pins the three canonical FA-bwd verdicts (real chain ACCEPT, two badgroups
REJECT) — the safety net for iterating on the predicate without a GPU.

### 4. Things tried and WHY they failed (do not repeat)

**(a) Relaxing `hasPotentialReuse` / adding `canJoin` in the DEFAULT first-fit
`tryAllocate` — reverted.** The obvious move is to let the first-fit path form the
sibling group directly. It does not work because first-fit is **greedy
first-feasible** (`:3944–3969`: sort candidates, take the first that recurses to a
full packing). In buffer order, `dpT` reaches `dk` as a feasible reuse **owner**
before the 3-way can assemble, so `dpT` becomes a **reuser of `dk`**, not an owner
holding the `{dpT,dsT,dq}` slot. Under the emergent-owner structure (§2) the 3-way
then cannot form at all. It also risks perturbing the byte-identical default path
(regression surface across the whole green suite). Rejected.

**(b) `canJoinReuseGroupChain` in the ENUMERATION path (`b019a73be`, kept as NFC)
+ `mem_plan_pick` sweep — insufficient on its own.** `canJoinReuseGroupChain`
(`WSMemoryPlanner.cpp:3820`) lets a candidate join an owner's group as a
time-multiplexed (offset-0) member when the *whole* prospective ≥3 group orders
under the sound predicate — exactly the common-ancestor sibling case. It is wired
**only** into `enumerateTMemAllocations` (`:4155–4165`), so default compiles are
untouched (verified: default BM128 still xfails, LIT unchanged). But enumeration
explores **owner assignments incrementally in buffer order**, and assembling
`{dpT,dsT,dq}` needs the 2-member seed `{dpT,dsT}` to exist **before** `dq` can
join. The bridge member `dsT` sorts **last** in buffer order, while `dq` and `dpT`
have no pairwise dep to seed on — so the group is **never seeded**. Empirically
confirmed: a `mem_plan_pick` sweep 0..17 with `topK=32` produced **0 passes**.

**Lesson:** incremental, order-dependent formation (first-fit *or* enumeration)
**cannot** assemble a common-ancestor sibling group whose bridge member sorts
last, regardless of how clever the join predicate is. The group must be repaired
**after** a complete packing exists, not built up member-by-member. That is what
motivated the post-pass (§5).

**(c) Earlier: tightening `hasDependencyChain`'s program-order fallback to
same-partition *globally* — reverted (`8997960ee`).** Doing this as a global
change (not a separate flag) regressed **HSTU 2-KV cross-attention bwd
`reduce_dq`** (wrong `dq`, rel-L2 ~0.79) — a real kernel with a legitimate
cross-partition program-order reuse edge. This is precisely why `80dde009a`
introduces `crossPartitionProgOrder` as a **separate opt-in flag** (default
`true`) rather than changing the global edge policy: code partitioning keeps the
permissive policy it needs; only the planner's group-formation gate uses the
strict one.

### 5. The chosen design: post-pass repair (`repairUnsafeReuseGroups`)

The shipped fix (working change, `WSMemoryPlanner.cpp:3844–3906`) operates on the
**final** first-fit packing, so it is immune to the ordering problem in §4(b):

1. First-fit (`tryAllocate`) finalizes the packing as usual.
2. `repairUnsafeReuseGroups` runs to a fixpoint (`:4356–4358`, `while (repair…) {}`).
3. Each call: for every reuse group that is **not chain-orderable** (≥3 members,
   single copy, `orderReuseGroupChain(&g, /*crossPartitionProgOrder=*/false)`
   returns empty), it tries to **relocate a reuser member** (never the owner — the
   owner holds the slot) into another group where it forms a chain-orderable slot
   (`canJoinReuseGroupChain`), **provided removing it leaves the source group
   orderable** (`orderable(rest)`). One relocation per call; the loop repeats.

Termination: each move takes a member from an unsafe group to a safe one, so it
converges. Order-independence: it inspects the *complete* packing and its group
membership, not an incremental build order — so the "bridge sorts last" trap
(§4b) does not apply.

**Inert / NFC-until-triggered.** `orderable(mem)` returns `true` immediately for
any group with `< 3` members and for any already-orderable ≥3 group, so the pass
fires **only** when first-fit produced an unorderable ≥3 single-copy group. Every
packing first-fit already gets right is byte-identical — the whole green suite is
untouched by construction.

**What it does for BM128.** First-fit yields `{qkT,ppT,dsT}` (unorderable —
`ppT⊥dsT`) plus `{dpT,dq}`. The pass detects the unsafe group, finds that removing
`dsT` leaves `{qkT,ppT}` orderable, and that `dsT` can join `{dpT,dq}` as a
chain-orderable member (`dpT→dsT→dq`), so it relocates `dsT`:

```
{qkT,ppT,dsT}(unsafe) + {dpT,dq}   →   {qkT,ppT} + {dpT,dsT,dq}
```

This is **exactly** the hand-pinned `_BWD_DOT_ATTRS_TMEM` packing (id 2 =
`{qkT,ppT}`, id 5 = `{dpT,dsT,dq}`). Result:
`test_bwd_bm128_memtype_only_xfail` now **PASSES** from memtype-only annotations,
with no hand-pinned buffer ids.

Note the post-pass reuses `canJoinReuseGroupChain` (from `b019a73be`) as its
"can this member land here safely?" check, so `b019a73be` — NFC on its own — is a
load-bearing dependency of the fix.

### 6. Two supporting bug fixes (had to land first)

**f16-in-f32 materialization validation (`f99b9c0ce`, `TMEMAlloc1D.cpp`).**
`sliceAndReinterpretMDTMEM` validated a reuse subslice using the **reuser's
logical** column count, but the planner assigns `buffer.offset` in **physical**
TMEM columns. For an f16 reuser packed inside an f32 owner, two f16 elements share
one 32-bit column, so the reuser occupies `blockN/2` physical columns — the code
already halves the width in the subslice branch, but the bounds check used the full
`blockN`, spuriously rejecting a valid non-zero offset (e.g. `offset=64,
blockN=128 → 64+128 > 128` OOB) and aborting. Fix: compute `sliceCols = blockN/2`
when `oldElemTyWidth == elemTyWidth*2` and validate against that same physical
width. Without this, the `{dpT,dsT,dq}` packing (mixed element widths) fails to
materialize.

**Reuse-folded-endpoint use-after-free (`45e976ce9`, `defunct` flag).** When a
reuse group is folded into its representative, the non-representative channel's
`allocOp` is erased (`replaceBufferReuse`, `WSCodePartition.cpp`), leaving that
`Channel`'s endpoints dangling. A later reuse-group walk
(`needAccumCntForReuse → enclosing → isProperAncestor`) dereferenced the freed op.
Whether a given endpoint was already erased is **iteration-order dependent**, so
it was a **layout-sensitive heisenbug** (vanished under IR dumping). Fix: add a
`defunct` flag to `Channel` set at both erase sites; `getSrcOp()/getDstOp()`
return null when defunct, and `needAccumCntForReuse` skips defunct channels (the
representative covers their space) plus null-guards the endpoints. This is what
made the N-way folding reliable enough to test.

### 7. Validation status & remaining work

**Verified:**
- WS LIT suite: 116 pass + 11 xfail (includes the new
  `ws_mem_plan_verify_reuse_groups.mlir`).
- autoWS FA correctness: 72 passed.
- `test_bwd_bm128_memtype_only_xfail`: now **PASSES** with the post-pass build
  (the group forms as `{dpT,dsT,dq}` + `{qkT,ppT}`).

**Remaining work:**
- **Full regression re-run** with the post-pass compiled in (the full `run_all.sh`
  autoWS matrix + HSTU cross-attn bwd) to confirm the "inert unless triggered"
  claim end-to-end on a GPU.
- **Flip the xfail marker** on `test_bwd_bm128_memtype_only_xfail`
  (`fused_attention_ws_device_tma.py:1911–1919`) to a normal test once the
  post-pass lands.
- **Extend the post-pass** if needed: it currently does single-reuser relocation
  into another owner's group. Consider whether it should also try **2-way
  relocations** (move two members) or handle **multi-buffer** (copies>1) groups —
  neither is required for the BM128 gate but both are plausible for other configs.
- **Land the working change** (`repairUnsafeReuseGroups` is currently uncommitted).
- Meta task: **T279873316**.

---

## 1. Goal

Replace the memory planner's fixed heuristic (priority buckets P0/P1/P2 +
iterative copy increase + TMEM round-robin) with a **search over the space of
legal, feasible allocation plans**, scored by a latency-aware cost model, keeping
the **top-K** plans.

Key reframing: we search the planner's **output** (the allocation plan), not its
**inputs** (the knobs `smem-circular-reuse`, `num-buffers`, `smem-alloc-algo`,
`tmem-alloc-algo`, `partition-condition`). Every knob combination only ever
produced *some* plan in this space, and never all the good ones. Searching plans
directly subsumes the knobs; the heuristic planner becomes, at most, a fallback /
bound seed.

### Locked design decisions

| Decision | Choice |
|---|---|
| Objective | **Perf**, with a hard feasibility gate (never emit a deadlocking/OOR plan) |
| Feasibility | **Static** legality+feasibility check (GPU-free), reject before scoring |
| SMEM vs TMEM | **Independent** packing (separate pools, separate searches, combine scores) |
| Search strategy | **Beam** (width `W`) to start; branch-and-bound upgrade later |
| Buffer ordering | **Swappable policy**; liveness-start first, topological later |
| Priority signal | **Latency-driven** marginal benefit (producer latency from `ttng::NVLatencyModel`; `II` from `tt.modulo_ii`) |
| Scope | SMEM + TMEM, all decision dimensions (grouping, copies, placement) |

---

## 2. Problem formulation

### 2.1 A "plan"

Given the buffer set `B` (every SMEM `local_alloc` and TMEM alloc), a **plan** `P`
assigns to each buffer:

- **grouping** — a partition of `B` into physical **blocks**; buffers in one
  block share physical space (reuse). → `buffer.id`
- **copies** — a multi-buffer depth per block. → `buffer.copy`
- **placement** — for TMEM, `(rowOffset, colOffset)` within the block; SMEM
  blocks are sized by `max(size)·copies`. → `buffer.offset`

This is exactly the tuple the downstream lowering already consumes, so a plan is a
complete, replayable decision — serializable via the existing
`BufferDecisionList`.

### 2.2 Legal ∧ feasible = the pruning predicate

**Legal (correctness):**
1. reuse encoding compat (same elem type/encoding, or `neutral_reuse`)
2. cross-stage floor: `copies ≥ stageSpan` for every buffer in the block
3. slot-collision floor: `copies ≥ #entries` in the block after data-partition
   expansion (the `(accumCnt + theIdx) % numBuffers` math)
4. TMEM rotation semantics valid: accumulator (per-outer-tile) vs operand
   (per-inner-iter) blocks not mixed incompatibly
5. TMEM column-share only among liveness-disjoint (or subslice-fitting) buffers

**Feasible (resources):**
6. `Σ_block max(size)·copies ≤ smem_budget` (SMEM pool)
7. TMEM blocks fit `512 rows × cols` (minus reserved scale cols)

These are the same five hazard checks discussed as the "static hazard checker,"
here repurposed as the branch predicate that carves the legal∧feasible region out
of the raw partition space.

### 2.3 The copy factorization (why the search is tractable)

The raw space is Bell-number huge (partitions) × copy counts × placements. But
**copies factor out**: given a fixed grouping+placement, each block's
footprint-per-copy and its floor are fixed, and the latency benefit of the c-th
copy is

```
marginal(block, c) = freq · II   while c·II < L_block,   else 0     (concave)
```

Separable concave gains under a shared budget → **greedy-by-marginal-benefit-per-byte
is the exact optimum** (concave resource allocation). Therefore:

- **Branch only over grouping + placement** (the genuinely combinatorial part).
- **Solve copies in closed form** per grouping via the concave knapsack — exact,
  fast, no branching. This also yields the plan's score.

---

## 3. Architecture (modular)

### 3.1 The invariant that makes ordering swappable

> **Legality and feasibility are checked explicitly against the partial plan —
> never inferred from a buffer's position in the order.**

Every check is an order-agnostic predicate (interval disjointness, entry count,
type check). Ordering then affects only which partials the beam keeps and how hard
the bound prunes — never whether a plan is admitted. This is what lets
`TopologicalOrder` drop in later with zero changes to legality/feasibility/cost.

Honest consequence: under a *beam* (approximate), changing the order changes the
*result set*. Under full branch-and-bound it changes only speed. Ordering is a
quality knob, not a correctness knob.

### 3.2 Module seams

```
BufferModel     — normalized per-buffer facts (built once from IR + ttng::NVLatencyModel + tt.modulo_ii)
OrderingPolicy  — buffer → sequence                         [SWAPPABLE]
Packer          — what a block is; join/place/footprint     [per-kind: SMEM | TMEM]
CostModel       — leaf score + admissible/optimistic bound  [SWAPPABLE]
CopySolver      — copies given a grouping (concave knapsack) [SWAPPABLE]
BeamSearch      — generic driver over all of the above       (kind-agnostic)
```

Interfaces (land in `WSMemoryPlanSearch.h`):

```cpp
struct BufferModel {
  ArrayRef<BufferId> buffers();
  Footprint      size(BufferId);        // bytes (SMEM) or rows×cols (TMEM)
  Interval<size_t> liveness(BufferId);
  unsigned       stageSpan(BufferId);   // cross-stage floor
  unsigned       entries(BufferId);     // data-partition expansion → slot floor (SHIPPED: stubbed to 1, TODO step9)
  EncodingKey    encoding(BufferId);    // reuse-compat key
  BufferKind     kind(BufferId);        // TMA / operand / accumulator / staging
  unsigned       reuseScope(BufferId);  // SMEM same-block reuse-group gate
  double         latency(BufferId);     // L_b  (producer latency, from ttng::NVLatencyModel — see Step 0 revised)
  double         freq(BufferId);        // trip count (SHIPPED: stubbed to 1.0, TODO step9)
  bool           dependsOn(BufferId, BufferId);   // fwd slice via SSA + memory — topo order & TMEM reuse
};

struct OrderingPolicy {                 // the swap seam
  virtual SmallVector<BufferId> order(const BufferModel&) const = 0;
};

struct Packer {                         // SmemPacker | TmemPacker
  virtual bool      legalJoin(const PartialPlan&, BufferId, BlockId) const = 0;
  virtual bool      feasible (const PartialPlan&, Budget) const = 0;
  virtual Footprint footprint(const Block&) const = 0;
  virtual Placement place    (const Block&, BufferId) const = 0; // TMEM offsets; SMEM no-op
};

struct CostModel {
  virtual double score(const Plan&) const = 0;                    // Σ hidden_latency − λ·occ
  virtual double bound(const PartialPlan&, ArrayRef<BufferId> rest) const = 0;
};

struct CopySolver {
  virtual CopyMap solve(const Grouping&, Budget, const CostModel&) const = 0;
};

TopKPlans beamSearch(const BufferModel&, const OrderingPolicy&, const Packer&,
                     const CostModel&, const CopySolver&, unsigned W, unsigned K);
```

### 3.3 Engine loop (beam form)

```
seq  = ordering.order(model)            // the ONLY place ordering enters
beam = { emptyPartialPlan }
for b in seq:
    next = []
    for P in beam:
        for G in P.blocks if packer.legalJoin(P, b, G):
            P' = P.join(b, G)
            if packer.feasible(P', budget): next.push(P', costModel.bound(P', rest))
        P'' = P.openBlock(b)
        if packer.feasible(P'', budget): next.push(P'', costModel.bound(P'', rest))
    beam = top-W of next by score        // SHIPPED: ranks each partial by its copy-solved score (scoreWithCopies), not bound() — all partials share the same placed prefix, so score is apples-to-apples. log drops (no silent cap)
for P in beam:
    P.copies = copySolver.solve(P.grouping, budget, costModel)
    topK.offer(P, costModel.score(P))
return topK
```

### 3.4 SMEM/TMEM independence

Two instantiations of the same engine, per the locked decision:

```cpp
auto smem = beamSearch(smemModel, *ordering, SmemPacker{}, *cost, *copies, W, K);
auto tmem = beamSearch(tmemModel, *ordering, TmemPacker{}, *cost, *copies, W, K);
```

They differ only in the `Packer` and the budget (SMEM bytes / 512 rows). NOTE
(SHIPPED): the planned SMEM circular multi-buffer reuse grouping is NOT
implemented — `SmemPacker::legalJoin` returns false unconditionally (matching
encoding + basic block is not sufficient for safe circular reuse), so the SMEM
search gives each buffer its own block and only tunes copies; reuse grouping is
future work (§5.3, §8). TMEM = row-owner + column subslice + time-multiplexed
(liveness-disjoint + data-dependent) column reuse. `OrderingPolicy`, `CostModel`, `CopySolver` are shared verbatim. Final
plans are the cross-product of the two top-K lists, scored by combined occupancy
(or kept separate — they are physically independent pools).

---

## 4. Cost model

The same function orders candidates offline and (via the knapsack) drives copy
allocation.

```
Score(P) = Σ_block hidden_latency(block, copies) − λ · occupancy_penalty(P)
hidden_latency(block, c) = freq · min(c · II, L_block)
```

- `L_block` = producer latency, `freq` = trip count, `II` = `tt.modulo_ii`.
- **Latency source (revised)**: `L_block` is obtained *on demand* inside the
  `BufferModel` builder by instantiating `ttg::NVLatencyModel` and calling
  `getLatency(producer).latency` (issue-to-result — the quantity multi-buffering
  hides). `II` is read from the existing `tt.modulo_ii` loop attribute. **No new
  scheduler annotation is required** — see the Step 0 revision in §6. This avoids
  stamping attrs on every scheduled op (which would change IR output and risk
  breaking `lit` CHECK lines). The `tt.issue_cycle` annotation is only needed for
  cycle-accurate TMEM liveness and is deferred (§8).
- `bound(partial, rest)` = optimistic completion: best-case latency hidden for the
  unplaced suffix assuming free reuse + full copies. Admissible ⇒ branch-and-bound
  is exhaustive w.r.t. top-K; under beam it is just the ranking key.
- Start with `λ = 0` (pure latency hiding); add the occupancy term only if
  benchmarks show high-copy plans losing to occupancy. Keeping `λ=0` first also
  makes the proxy easy to validate against runtime.

**Schedule dependence**: the cost model needs `II` + latencies, so a plan search
is always *relative to a fixed schedule*. Both inputs come from the already-
scheduled IR (`tt.modulo_ii` + `NVLatencyModel`), so no per-schedule re-annotation
is needed. When later folded into the modulo-schedule top-K search, the plan
enumeration re-runs per schedule (memory pick is conditional on schedule pick).

---

## 5. Helper inventory (reuse / refactor / new)

All line numbers refer to `WSMemoryPlanner.cpp` unless noted. **reuse** = call
as-is; **refactor** = extract/generalize existing logic behind the new interface;
**new** = must be written.

### 5.1 `BufferModel` — normalized per-buffer facts

| Fact | Existing helper | Action |
|---|---|---|
| op-ID space for liveness | `MemoryPlannerBase::buildOperationIdMap` (:80) | reuse |
| SMEM size | `getSmemAllocSizeBytes` (:1306) | reuse |
| TMEM size | `ttng::getTmemAllocSizes` / `TMemAllocation` | reuse |
| liveness interval | `resolveLiveness` (:539), `resolveExplicitBufferLiveness` (:510), `livenessForTmemChannel` (:2422), `getAllAcutalUsersForChannel` (:266), `getAllTmemUsers` (:2391), `updateLiveOpsAcrossScopes` (:399), `getUserScopes` (:327), `getLiftedScope` (:304) | reuse |
| stage span (SMEM floor) | `getSmemCrossStageDepth` (:1248), `isSmemCrossStage` (:1233), `getLoopStage` (:1217), `getLoopCluster` (:1225) | reuse |
| stage span (**TMEM** floor) | — none — | **new** `getTmemCrossStageDepth` |
| entries → slot floor | inline in `enforceMinBufferCopy` (:689, counting per `buffer.id` at :664–704) | refactor → `slotFloor(block)` |
| encoding compat | `areReuseEncodingsCompatible` (:1702), `neutralReuseEnabled` (:1698) | reuse |
| kind (TMA/innermost/staging/accum) | `isSmemTMAChannel` (:1197), `isInnermostLoop` (:137), `usersInInnermostLoop` (:464), `isInnermostSmemChannel` (:1145), `WSBuffer.tmaStaging`, channel `isOperandD`, `hasLoopCarriedAccToken` (`CodePartitionUtility.cpp:64`) | reuse |
| **latency `L_b`** | `ttg::NVLatencyModel::getLatency(op).latency` (LatencyModel.h) | reuse (on-demand; no annotation) |
| **`II` / freq** | `tt.modulo_ii` loop attr; trip counts | reuse / **new** (freq reader) |
| `dependsOn(a,b)` | `isDataDependent` (:2487), `alongDependencyChain` | reuse |
| the struct itself | `WSBuffer` (:853) ≈ SMEM model; TMEM uses `BufferT` | refactor → one generic `BufferModel` |

### 5.2 `OrderingPolicy`

| Impl | Existing | Action |
|---|---|---|
| `LivenessStartOrder` | `sortChannelsByProgramOrder` (:3862), `getWSBufferUsageOrder` (:873), `getLogicalProducerOp` (:233), `getLastConsumerOrder` (:1678) / `Detailed` (:1627), `ConsumerOrder` (:1621) | refactor → wrap behind interface |
| `TopologicalOrder` | edge oracle `isDataDependent` (:2487) | **new** driver (topo sort over existing edges) |

### 5.3 `SmemPacker`

| Method | Existing | Action |
|---|---|---|
| `footprint` (Σ max·copies) | `computeTotalSmem` (:1324) | reuse |
| `feasible` (budget) | budget check in `allocateSmemBuffers` (:1845); `smemBudget` auto-detect | refactor → `SmemPacker::feasible` |
| `legalJoin` (reuse pairing) | pairing + `findReuseCandidate` (:2073) in `allocateSmemBuffers`; `areReuseEncodingsCompatible`; slot/cross-stage floors | **SHIPPED: returns false** — SMEM circular reuse grouping is future work (§8); encoding+scope is not a sufficient safety condition |
| epilogue disjoint-liveness fusion | `fuseEpilogueWSBuffers` (:1347), `fuseEpilogueBuffers` (:712), `allAllocsCompatible` (:191), `increaseFusedEpilogueCopies` (:1427) | not yet wired into the search (deferred with SMEM reuse grouping) |
| `place` | (SMEM sized by max — no offsets) | trivial no-op |

### 5.4 `TmemPacker`

| Method | Existing | Action |
|---|---|---|
| `legalJoin` | `hasPotentialReuse` (:3171), `findReuseChannel` (:3654), `samePartition` (:3518), `alongDependencyChain`, `checkOtherReuses`, `isDataDependent` (:2487) | reuse |
| `place` / `footprint` | `AllocationState` (:3064), `OwnerPlacement` (:3058), `addOwnerToState` (:3078), `computeColOffset` (:3204), `findReuseSpace`, `allocateNewSpace`, `applyAllocationState` (:3315), `tryAllocate` (:3235), `kNumRowGroups` | reuse / refactor (wrap behind `Packer`) |
| `feasible` (512) | 512-row/col limit checks; `getMinFutureScaleTmemCols` (:2587) | reuse |
| cross-stage floor | — none — | **new** `getTmemCrossStageDepth` |
| copy>1 legality guard | `hasLoopCarriedAccToken` (`CodePartitionUtility.cpp:64`) | **new** guard (reject non-accumulator `copies>1`; see §8) |

### 5.5 `CostModel` + `CopySolver`

| Piece | Existing | Action |
|---|---|---|
| `score` | crude precursor: `WSBufferPriority` buckets | **new** (latency-hiding; delete buckets) |
| `bound` (admissible) | — none — | **new** |
| `CopySolver` (concave knapsack) | Phase-4 iterative increase in `allocateSmemBuffers`; TMEM round-robin (:3010–3022); `enforceMinBufferCopy` (:689) floor | refactor → knapsack; keep floors as constraints |

### 5.6 `BeamSearch` engine

| Piece | Existing | Action |
|---|---|---|
| DFS/backtrack skeleton | `tryAllocate` (:3235) — first-fit DFS | refactor → generalize to beam + cost-bound + top-K |
| beam width / top-K queue | — none (returns first solution) | **new** (bounded priority queue, width `W`) |

### 5.7 Output / plumbing

SHIPPED: the search stamps `buffer.id`/`buffer.copy`(/`buffer.offset`) directly
on the allocs from the picked plan and dumps the ranked plans as JSON to
`TRITON_WS_MEM_PLAN_TOPK_DUMP` (rank selected by `mem_plan_pick` /
`TRITON_WS_MEM_PLAN_PICK` — see §top). The original plan of emitting each top-K
plan as a `BufferDecisionList` was NOT used; the `BufferDecision*` carrier below
remains available but the search does not route through it.

| Piece | Existing | Action |
|---|---|---|
| plan ↔ decision | `BufferDecision` (:3825), `BufferDecisionList` (:3844), `extractBufferDecision` (:3873), `applyBufferDecision` (:3897), `serializeBufferDecisions` (:3916), `serializeBufferDecisionsToString` (:3954), `writeDecisionsToFile` (:4044), `readDecisionsFromFile` (:4062) | reuse |
| debug viz | `dumpBuffers` (:772), `dumpSmem/TmemBufferLiveness`, `getLocName` (:801) | reuse |

### 5.8 Net-new code (the real work)

1. **Unified `BufferModel`** — generalize `WSBuffer` + TMEM `BufferT` into one interface.
2. **Latency/freq readers** — latency from `ttng::NVLatencyModel` on demand (Step 0 REVISED — no `tt.self_latency` annotation), `II` from `tt.modulo_ii`; `freq` (trip count) SHIPPED stubbed to 1.0 (TODO step9).
3. **`getTmemCrossStageDepth`** — TMEM cross-stage floor (no helper today).
4. **`CostModel::score` + `bound`** — latency-hiding objective and its optimistic bound.
5. **`CopySolver`** — concave-benefit knapsack (replaces two ad-hoc copy loops).
6. **Generic `BeamSearch`** — beam width + top-K queue + cost-bound pruning.
7. **`OrderingPolicy` interface + `TopologicalOrder`**; lift existing sorts behind `LivenessStartOrder`.
8. **`Packer` interface** — thin; wraps existing SMEM pairing and TMEM backtracking.
9. **TMEM copy>1 legality guard** — until the general per-inner-iter rotation exists (§8).

Everything in reuse/refactor is ~70% of the surface (liveness, sizes, TMEM
placement, reuse legality, and the entire serialization/output path). The
genuinely new code is the **cost model, the knapsack, the beam/top-K driver, and
the `BufferModel`/`Packer`/`OrderingPolicy` interfaces** that tie them together.

---

## 6. Implementation steps

Ordered by dependency. Each step is independently testable against a synthetic
`BufferModel` (no IR, no GPU) unless noted.

### Step 0 — Latency source (REVISED: no scheduler annotation)
Original plan stamped `tt.self_latency`/`tt.issue_cycle` on every scheduled op.
That changes IR output (breaks `lit` CHECK lines) for no benefit, because the
cost model's inputs are already reachable from the scheduled IR:
- **`L_b`**: instantiate `ttg::NVLatencyModel` (LatencyModel.h, same library) in
  the `BufferModel` builder and call `getLatency(producer).latency` on demand.
- **`II`**: read the existing `tt.modulo_ii` loop attribute.

So Step 0 folds into the `BufferModel` builder (Step 1/8) — **no separate
scheduler change, no behavior change**. The `tt.issue_cycle` annotation is only
needed for cycle-accurate TMEM liveness and is deferred (§8).

### Step 1 — `BufferModel` interface + SMEM builder
- New `WSMemoryPlanSearch.h` with the interfaces from §3.2.
- SMEM builder: generalize `WSBuffer` (:853). Wire each field to its existing
  helper: `getSmemAllocSizeBytes` (:1306), `resolveLiveness` (:539),
  `getSmemCrossStageDepth` (:1248), `areReuseEncodingsCompatible` (:1702),
  `isSmemTMAChannel`/`isInnermostSmemChannel` (:1197/:1145), `WSBuffer.tmaStaging`.
- `entries()`: **refactor** the per-`buffer.id` counting from
  `enforceMinBufferCopy` (:689, :664–704) into a standalone `slotFloor` helper.
- `latency()`: **new**, via on-demand `ttg::NVLatencyModel::getLatency`.
  `freq()`: **new** trip-count reader. `II` from `tt.modulo_ii` (Step 0 revised).
- `dependsOn()`: reuse `isDataDependent` (:2487).
- **Test**: synthetic + one real kernel; assert fields match the current planner's
  derived values.

### Step 2 — `OrderingPolicy` + `LivenessStartOrder`
- Lift the existing sorts (`sortChannelsByProgramOrder` :3862,
  `getWSBufferUsageOrder` :873, `getLogicalProducerOp` :233,
  `getLastConsumerOrder` :1678) behind `LivenessStartOrder`.
- Register via a factory keyed by `--mem-plan-order=liveness|topo`.
- **Test**: deterministic order on a synthetic model.

### Step 3 — `SmemPacker`
- `legalJoin`: **refactor** the reuse-pairing legality from `allocateSmemBuffers`
  (:1845) + `findReuseCandidate` (:2073), plus checks 1–3. Include the
  disjoint-liveness epilogue fusion (`fuseEpilogueWSBuffers` :1347,
  `allAllocsCompatible` :191) as a join capability.
- `footprint`: reuse `computeTotalSmem` (:1324).
- `feasible`: **refactor** the budget check from `allocateSmemBuffers`.
- `place`: no-op.
- **Guard**: keep every check an explicit predicate — no "processed in order so…"
  shortcuts (protects the ordering seam).
- **Test**: legalJoin accept/reject table; feasibility at budget boundary.

### Step 4 — `CopySolver` (concave knapsack)
- **Refactor** the Phase-4 iterative increase (`allocateSmemBuffers`) and the TMEM
  round-robin (:3010–3022) into one knapsack. Keep the floors (`minCopies`,
  cross-stage, slot-collision) as hard constraints seeded from `BufferModel`.
- **Test**: knapsack optimality vs brute force on small instances; floor
  enforcement.

### Step 5 — `CostModel`
- Implement `score` (§4) and an admissible `bound`. Delete the `WSBufferPriority`
  buckets once parity is shown.
- **Test**: monotonicity (more hidden latency ⇒ higher score); bound ≥ any
  completion's score on small instances.

### Step 6 — `BeamSearch` engine
- **Refactor** the DFS skeleton of `tryAllocate` (:3235) into the generic driver:
  add beam width `W`, a bounded top-K priority queue, and cost-bound pruning.
- **Test**: on a synthetic model, beam with `W=∞` reproduces brute-force top-K.

### Step 7 — `TmemPacker`
- Wrap the existing TMEM backtracking placer behind `Packer`: `hasPotentialReuse`
  (:3171), `findReuseChannel`, `computeColOffset` (:3204), `AllocationState`
  (:3064), `addOwnerToState` (:3078), `applyAllocationState` (:3315),
  `allocateNewSpace`, 512 limit, `getMinFutureScaleTmemCols` (:2587).
- **New** `getTmemCrossStageDepth` (TMEM cross-stage floor — no helper today).
- **Legality guard**: until the general per-inner-iteration TMEM copy path exists
  in `createBufferPost` (see [AccumulationCounters.md] / §8), `legalJoin` must
  reject `copies>1` for non-accumulator TMEM blocks. Accumulator multi-copy
  (per-outer-tile) is already supported via `hasLoopCarriedAccToken`.
- **Test**: reuse legality table; 512-row overflow rejection.

### Step 8 — TMEM `BufferModel` builder
- TMEM facts via `livenessForTmemChannel` (:2422), `getAllTmemUsers` (:2391),
  `ttng::getTmemAllocSizes`, channel `isOperandD`, `hasLoopCarriedAccToken`.
- **Test**: parity with current TMEM liveness/sizes.

### Step 9 — Wire into `doMemoryPlanner`
- Add `smem-plan-search` / `tmem-plan-search` pass options (default off).
- When on: build models → run `beamSearch` twice → emit the top-1 plan via
  `applyBufferDecision` (:3897). Emit top-K as ranked `BufferDecisionList`s
  (reuse `writeDecisionsToFile` :4044) for the downstream autotuner pick.
- Keep the heuristic path as the default fallback.
- **Test**: autoWS correctness suite (see `autows-testing` skill) with search on,
  top-1 plan, on the standard GEMM/FA kernels.

### Step 10 — `TopologicalOrder` (deferred, validates the seam)
- New `OrderingPolicy` impl using `isDataDependent` edges → topo sort.
- No changes to any other module. If this requires touching legality/feasibility,
  the §3.1 invariant was violated — fix that instead.
- **Test**: same top-K quality or better on kernels where reuse is
  dependency-dominated (TMEM).

---

## 7. Testing strategy

- **Unit** (per module, synthetic `BufferModel`, no GPU): ordering, legalJoin,
  feasible, knapsack, cost/bound, beam.
- **Parity**: search top-1 vs current heuristic on standard kernels — must be
  feasible and no worse in SMEM/TMEM footprint.
- **Correctness**: autoWS suite with search enabled (`autows-testing` skill).
- **Hazard**: replay the FA-bwd TMEM-reuse and cross-stage deadlock cases
  ([BwdTmemReuseSlotHazard.md], partition-scheduler-bugs #8) — the static gate
  must reject the deadlocking plans.

---

## 8. Open items / deferred

1. **Branch-and-bound upgrade**: replace beam with admissible-bound B&B for
   exhaustive top-K once the bound is tight enough to prune.
2. **General TMEM copy rotation**: the per-inner-iteration semantics in
   `createBufferPost` (blocks non-accumulator TMEM `copies>1`). Until then the
   Step-7 legality guard is in force.
3. **Joint schedule search**: fold `mem_decision_pick` into the modulo top-K
   config vector (conditional on schedule pick) — cross-turn follow-up.
4. **`λ` occupancy term**: enable only if benchmarks show occupancy losses.
5. **SMEM/TMEM coupling**: current plan keeps them independent; revisit if a
   kernel has buffers that can live in either pool.
6. **`tt.issue_cycle` annotation**: needed only for cycle-accurate TMEM liveness
   (tighter reuse packing than op-ID intervals). Requires a scheduler change
   (`ModuloSchedulePass`); deferred since the op-ID intervals already work and
   the cost-model latency comes from `NVLatencyModel` on demand (Step 0 revised).
