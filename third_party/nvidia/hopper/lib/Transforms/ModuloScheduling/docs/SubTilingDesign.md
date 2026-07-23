# Sub-tiling in front of the modulo solver — design note

Status: design, 2026-07-06 (SolverMigrationNotes step-4 item 4). Written
against branch hwu27/modulo-cpsat-joint-v2; all file:line references as of
that branch.

## Why

Ping-pong/FA4-class schedules — two M-half-tiles whose softmax and MMA
phases interleave in anti-phase — are ABSENT from the solver's solution
space today, not merely unreached: the DDG has one node per tile-level op,
so "sub-tile 0's softmax overlaps sub-tile 1's MMA" is not expressible.
Twill reports its joint formulation is UNSAT without sub-tiling; our
measured ~665 TFLOPS plateau on case3 (reproduced by three independent
solver configs) is the single-instance softmax-chain bound of the current
design point. Sub-tiling is the only lever that changes that bound rather
than approaching it.

## The target, concretely (from the hand-written kernels)

- **Hopper ping-pong** (`hopper_fa_ws_pipelined_pingpong.py`): BLOCK_M
  128 → 2×64, the consumer body replicated (`replicate=2`), and the ONLY
  addition over the non-pingpong variant is a named-barrier 9/10 handshake
  around the QK `async_dot` issue. In modulo-DDG terms that handshake is
  exactly one distance-0 edge (QK0@i → QK1@i) plus one distance-1 back
  edge (QK1@i → QK0@i+1): a recurrence pinning RecMII ≥ 2× TC occupancy
  and forcing the half-period offset. The schedule-graph IR already has
  the lowering vehicle (`CrossGroupBarrier.kind = "named"`).
- **Blackwell FA4-class** (`blackwell_fa_ws_pipelined_persistent.py`):
  same M-split, no named barriers — the interleave is program order
  inside the single 1-warp MMA task (`q0@k → p1@v_prev → q1@k → p0@v`),
  coupled to two replicated softmax tasks via per-cid mbarrier arrays and
  per-cid TMEM tiles.
- Shared vs duplicated: K/V TMA loads, their SMEM rings, and the scalar
  index math stay SHARED (empty barriers get `arrive_count=2` — the
  semaphore layer already derives this from `partition_count`,
  semaphore_ir.py:271-311). Everything from Q/QK down duplicates: QK MMA,
  the softmax chain, m_i/l_i recurrences, PV MMA, accumulator, epilogue
  stores at row offsets +0/+64.

## What it does to case3's numbers (M-split ×2, from the DDG)

- 31 nodes → ~55 (24 split, 7 shared); 40 edges → ~80; loop-carried
  iter_args 5 → 9 (m_i, l_i, qk token, acc token duplicate; the acc-init
  flag stays shared).
- The split axis MUST be M: `tt.reduce` is axis=1 (the kv axis), so an
  M-split needs no cross-sub-tile combiner; an N-split would need
  max/sum joins and serialize the PV MMAs (WSDataPartition.cpp:1978-1980
  asserts reduce axis ≠ partition dim).
- MMA latency does NOT halve (NVLatencyModel keys it on K only); MMA
  occupancy halves. Net: ResMII ~unchanged, RecMII 1325 → ~942 (the acc
  circuit's tmem_load/mulf/tmem_store halve, the 559 MMA back edge does
  not), MinII 1459 → ~1198 (CUDA-bound). **The payoff is schedulability,
  not MinII**: two independent ~1450-cycle half-chains can overlap each
  other's TC windows, which one indivisible ~2900-cycle chain cannot.
- The latency model needs ZERO changes for an IR-level split — it is
  fully shape-driven.

## Preconditions (status)

1. Guard 1 must be off for the sub-tiled graph — with 4 chained TC nodes
   its occupancy=latency inflation would price the split at TC
   ResMII=2918 (strictly worse than unsplit). DONE: the deletion is
   validated for the joint path (TRITON_MODULO_DISABLE_MMA_GUARD;
   665.7 TFLOPS at the unguarded MinII, SolverMigrationNotes step-4
   item 2).
2. A partitioner that discovers cohesion/splits from constraints rather
   than tuned scores. DONE (joint v1/v2).
3. Emission-order legality for re-solved cycles (iter-arg WAR edges).
   DONE.
4. Optional refinement: `getMinWarps` returns 4 for 64-row ops; a
   min_warps=2 refinement would let one 4-warp WG time-share both
   half-chains. Not blocking.

## Existing machinery inventory (none sufficient alone)

- **Pass A.5 DataPartitionPlan** (ModuloSchedulePass.cpp:873-967): M-split
  of the MMA accumulator, but the MMA stays ONE DDG "bundle" node with
  occupancy ×N — the sub-MMAs cannot be placed at different cycles, and
  nothing downstream (softmax) duplicates. Emit-time fan-out only.
- **Pass A.7 epilogue subtile** (TRITON_MODULO_EPILOGUE_SUBTILE): N-split
  of the outer-loop store chain only; node-level subtile fields exist in
  the JSON schema and Python parser but the C++ dumper never emits them
  (ModuloSchedulePass.cpp: "A.7 subtile fields remain a separate
  follow-up").
- **AutoWS SubtiledRegionOp** (GenerateSubtiledRegion.cpp): a pattern
  recognizer + region wrapper for split trees that ALREADY exist in the
  IR downstream of a tmem_load — it does not create splits, cannot split
  MMAs or reduces, and runs far too late (inside NVGPUWarpSpecialization).
  Not reusable as the transform; its structural-equivalence checker is a
  useful reference.
- **WSDataPartition.cpp** (~2600 lines): the real M-split slicing core —
  clones ops per partition with shape[dim]/N types, rewrites TMEM
  encodings (blockM divisibility gates at :154-171, encoding rebuild at
  :1294-1301), threads sliced loop-carried iter_args (:712-717). Its
  driver is welded to AutoWS async-task-IDs and produces per-task copies;
  reuse needs a new driver that inlines both slices in one block.

## Plan: Route A first, graduate to Route B

**Route A — env-gated IR pre-split before the modulo pass** (the cheap
experiment). A new small pass (or pre-step inside ModuloSchedulePass)
reusing WSDataPartition's slicing core with a new single-block driver:
M-split the FA-shaped inner loop ×2 (Q slice, QK MMA, softmax chain,
PV MMA, accumulators, iter_args; K/V loads shared). Downstream blast
radius is ~zero — DDG, latency model, CP-SAT, JSON, emitter are all
shape-driven and just see 55 smaller nodes. Gate:
`TRITON_MODULO_SUBTILE_M=2`. Risks: TTGIR/layout validity (blockM=64
TMEM encodings, warp-locked tcgen05.ld rows) surfaces at compile time;
split-before-solve pre-commitment is mitigated by solving both variants
and letting solver_regression compare (the A.5/A.7 env-flag precedent).

**Route B — DDG-level virtual split** (where this wants to end up: the
solver prices split vs unsplit inside ONE search, no IR risk until
emission). Requires breaking two load-bearing invariants: one node per
Operation* in the DDG builder (opToIdx; edge phases 2/3) and
one-variable-per-op_id memoization in the emitter's inner-loop renderer,
plus a shape-scaled latency query and multi-producer buffer lifetimes.
Materially larger diff concentrated in the two most delicate components.
Do it after Route A has proven the win.

## Acceptance

- Sub-tiled case3 passes correctness under solver_regression, and the
  full solver stack on the sub-tiled graph BEATS the 665-TFLOPS plateau
  at (1,32,8192) — that is the entire point; if it cannot, the design
  point hypothesis is wrong and this note gets a measured post-mortem.
- Unsplit cases (GEMM family, layernorm) are untouched (the transform
  only fires on the FA shape or behind the env flag).
- The solver reproduces the anti-phase structure (QK0/QK1 offset ≈ half
  period) without any ping-pong-specific code — the named-barrier
  recurrence should EMERGE from TC NoOverlap + the flight-window
  constraints, the same way softmax cohesion emerged in step 4 item 2.

## Experiment log (2026-07-06) — Route A executed, measured post-mortem

Vehicle: cheaper than the planned IR pre-split — a source-level split,
`sched2tlx/examples/testing/subtiling/fa_subtiled_experiment.py`
(BLOCK_M=256 written as two 128-row sub-tiles sharing each iteration's
K/V; no WS in source; the solver stack decides everything). Sanity: the
ping-pong tutorial measures 1197 TFLOPS at (1,32,8192) = 1.8× the 665
plateau, so the design-point hypothesis is real; the vehicle itself
passes correctness on the default pipeline (312 TFLOPS).

What worked end-to-end:

- DDG grows 31→55 nodes exactly as predicted; joint_solver solves II=2396, Rau
  II=2487 (recurrence-bound; MinII ~1198 was the resource floor).
- The partitioner produced a perfect ping-pong-SHAPED 6-WG partition
  from first principles — 2 TMA, 2 TC, 2 softmax WGs, each sub-tile's
  chain in its own WG. The structural half of the emergence criterion
  is met. [CORRECTED 2026-07-06: re-derivation shows this partition is
  the HEURISTIC scoreCandidate partitioner's output (byte-identical
  warp groups, same partition_cost 2509.994) — the winning kernel is
  Rau schedule + heuristic partition, no CP-SAT anywhere in its
  production. joint-v1 on the same graph chooses a DIFFERENT 5-WG
  partition ([1,1,1,4,4]: both MMA dots in ONE TC warp — the
  tutorial's in-order single-MMA-WG shape). This is the first case
  where heuristic and joint diverge; the joint variant is unemitted/
  unmeasured and worth a follow-up, since the dissection suggests
  single-MMA-WG issue order is exactly the anti-phase mechanism.]
- After hand-patching the emitted kernel (below), correctness PASSES.

Emitter: three defect classes, hand-patch spec committed as the diff
pair `subtiling/fa_subtiled_rau_generated.py` (raw, launch-fails) vs
`subtiling/fa_subtiled_rau_handpatched.py` (runs): (1) cross-region
channels (`epi_*`, the acc carve-out) are name-keyed singletons — the
second sub-tile's instances shadow the first's in Python; (2) empty
barriers count distinct consumer WGs, but tcgen05 arrives are per-MMA —
a K tile read by two QK dots in one WG needs arrive_count=2, not 1
(symptom: "unspecified launch failure" in the TC warp); (3) the M_lse
epilogue stores are dropped entirely.

MEASURED: **206.7 TFLOPS** at (1,32,8192) — below the plateau AND below
the unsplit default pipeline. A K-ring depth probe (1→2, hand-mirrored
from the V ring) changed nothing (206.4): ring depth is not the limiter.

Root cause: the two sub-tile chains run IN PHASE, not anti-phase. The
Rau schedule puts sub0's and sub1's softmax tmem_loads 128 cycles apart
(cyc 1458 vs 1586, ~5% of II) — the partition is ping-pong-shaped but
the schedule is lockstep. Each chain's per-iteration acc rescale is a
full TMEM round-trip inside the loop-carry cycle (PV MMA(i−1) → sem7 →
tmem_load 128×128 f32 → ×alpha → tmem_store → sem3 → PV MMA(i)), and
the TMEM ld/st port and SFU are per-SM resources shared by both softmax
WGs. Measured steady state ≈11.3K cycles/iteration vs modeled II 2487
(4.5×). The model prices tmem_load/store and exp2 as per-WG
fixed-latency/occupancy ops with NO shared-across-WG engine, so
overlapping the two chains is modeled as free when in hardware it
halves (or worse) both. The hand ping-pong's named barriers are exactly
a cross-WG mutual exclusion on this phase — the structure the model
cannot currently prefer because it cannot see the contention.

Verdict and order of work (the design point is NOT falsified — the
tutorial proves 1197 is reachable; this is a model + emitter fidelity
gap):

1. Latency model: add per-SM shared engines (TMEM ld/st port, SFU)
   as cross-WG reservations in the joint formulation. With NoOverlap on
   a shared engine, anti-phase staggering becomes the only feasible
   packing — emergent ping-pong, no special-case code. This is the
   research item with the measurement (11.3K vs 2487) attached.
2. Emitter multi-instance generalization: per-instance channel naming
   (defect 1), per-MMA arrive counts (defect 2), don't drop epilogue
   ops (defect 3). Prerequisite for ANY sub-tiled emission; the
   hand-patch diff is the spec.
3. Only then re-run this experiment; acceptance (beat 665) stands.

## The 1197-TFLOPS tutorial dissected (2026-07-06)

`blackwell_fa_ws_pipelined_persistent.py`, best config BLOCK_M=256
(2×128 split), BLOCK_N=128, NUM_MMA_SLICES=2, KV depth 3–6, CLC
persistent. Read before touching the model — it sharpens what "add
shared engines" must mean and adds two items the post-mortem missed.

Task structure (7 tasks): load(1 warp) + mma(1 warp) + 2× softmax
(4 warps, `replicate=2`) + the default task = correction + epilogue +
CLC producer. Compare our generated kernel: 2×TMA + 2×TC + 2×softmax.

Where the acc rescale lives — three stacked mitigations, strongest
first:

1. **Data-dependent skip (RESCALE_OPT)**: softmax clamps alpha to 1
   when `(m_i−m_ij)·scale ≥ −8` (max didn't grow by >2⁸ in exp2 space —
   the overwhelmingly common steady case); the correction WG does a
   warp `vote_ballot_sync` on `alpha<1` and skips the ENTIRE acc
   round-trip when no row needs it. Most iterations: correction =
   barrier + 128-float load + ballot, no 128×128 traffic at all. This
   is a numerically-tolerant ALGORITHMIC transform — a scheduler cannot
   invent it; it bounds what any solver output can reach on this graph.
2. **Dedicated correction WG, off the softmax chain**: alpha is
   produced EARLY in softmax (right after row-max, before the exp2/P
   bulk) and handed off through a tiny TMEM channel (128×1 f32). The
   rescale, when it fires, runs on the correction WG CONCURRENTLY with
   softmax's exp2/P tail — off both chains' critical paths. One
   correction WG serves BOTH sub-tiles (cid loop), which also
   serializes the two chains' TMEM port usage by construction.
3. **Subsliced acc access**: rescale and epilogue touch acc in
   2×(128×64) HEAD_DIM slices; P is stored in 2 slices with per-slice
   full barriers so each PV dot starts on half of softmax's output.

Anti-phase mechanism: NO named barriers anywhere. The single 1-warp MMA
task issues `QK0(i) → PV1(i−1) → QK1(i) → PV0(i)` and tcgen05 executes
one warp's MMAs IN ORDER — the half-iteration skew between the chains
is program order on the shared tensor core. The WAR hazard on the
depth-1 qk tiles is closed by the same in-order property plus the
p_fulls waits — zero extra synchronization. Softmax's m_i/l_i
loop-carry stays in registers (same WG), like ours.

TMEM budget: qk(2×128) + acc(2×128) = 512 columns — full. P, alpha, l,
m all live INSIDE the qk region via `storage_alias_spec` (P overlaps
the first half after qk is consumed; alpha/l/m the second). At
BLOCK_N=128, sub-tiled FA does not fit TMEM without aliasing.

What this changes about the order of work:

- Shared-engine reservations (item 1 above) must cover the **TC
  engine** too, not just the TMEM port and SFU: our partition put QK
  and PV in two different TC WGs, and the model treats each as its own
  engine — the tutorial shows one tensor core, one issue order, and
  gets anti-phase from exactly that. Modeling TC as per-SM shared makes
  the single-MMA-WG partition (or an equivalent stagger) the solver's
  own conclusion.
- The partitioner must be ABLE to pull the 3-node rescale group
  (tmem_load acc → mul → tmem_store) out of the softmax WG into a
  shared correction WG. The nodes exist separately in the DDG already;
  what glues them today is that the flight-window/merge terms see no
  cost for in-WG TMEM traffic. The alpha handoff it buys is tiny
  (128×1 f32). This is a pricing fix, not an expressiveness fix.
- The rescale-skip (mitigation 1) is out of scheduling scope. Even a
  perfect solve of our current graph pays the full rescale every
  iteration; expect a gap vs 1197 from this alone. Estimate its size
  from the microbench (TMEM round-trip cost per iteration vs II).
- Vehicle fix for the re-run: BLOCK_N=128 (ours was 64 — twice the
  per-iteration fixed costs relative to work), KV depth ≥3, and the
  emitter (or hand-patch) needs TMEM aliasing to fit.
- Emitter capability list grows by one: `storage_alias_spec` emission
  (defect class 4 — without it, BLOCK_N=128 sub-tiled FA cannot
  allocate).

## TMEM port microbench (2026-07-06) — contention measured, hypothesis revised

`third_party/tlx/tools/microbench/tmem_port_contention.py` (B200, one
CTA, two 4-warp WGs, serial ld→fma→st chain per WG on disjoint TMEM
buffers, named-barrier-aligned windows):

| tile (f32)  | bytes | solo cyc/round-trip | dual (both WGs) | dual/solo |
|-------------|-------|--------------------:|----------------:|----------:|
| 128×32      | 16K   | 109.3               | 145.0           | 1.33      |
| 128×64      | 32K   | 174.5               | 243.0           | 1.39      |
| 128×128     | 64K   | 310.0               | 460.0           | 1.48      |

Fit: round-trip ≈ 41 cyc fixed + 4.2 cyc/KB. Both WGs' windows agree to
the cycle, so the ratio is engine interference, not skew.

Two conclusions, one revision:

- The TMEM ld/st path IS a shared per-SM resource (interference grows
  with size → the bandwidth component is shared), and the model should
  carry it as a cross-WG engine with these constants. But contention
  costs ×1.5, not ×2 — treating it as a fully serializing NoOverlap
  engine would over-penalize.
- REVISED: contention (×1.48) cannot explain the 4.5× gap (11.3K
  measured vs 2487 modeled) on its own, and the raw round-trip (310
  cyc) fits comfortably inside II=2487. The next diagnostic (before
  touching the model) was a clock64-instrumented replay — below.

## Where the 11.3K cycles actually went (2026-07-06) — PLATEAU BEATEN

clock64 stamps around every segment of wg3's loop body (9 slots × 16
iterations, CTA(0,0), full (1,32,8192) grid), steady-state medians:

| segment                   | as emitted | after fix 4 (+5) |
|---------------------------|-----------:|-----------------:|
| wait sem7 (PV done)       |        215 |              491 |
| tmem ld acc 128×128       |        320 |               20 |
| wait sem1 (QK done)       |        126 |               53 |
| tmem ld qk 128×64         |   **4371** |              317 |
| softmax math + exp2       |   **5328** |             1472 |
| tmem st acc + arrive      |        163 |              157 |
| wait P-bridge empty       |        552 |              791 |
| tmem st P + arrive        |        103 |              132 |
| **iteration total**       |  **11516** |         **3497** |

The two monster segments were not TMEM or waits at all: **register
spills**. The schedule places each softmax WG's acc tmem_load (128 f32
registers/thread of live range) at the loop top — cycle 2 of the
modulo schedule, ~5000 cycles before its only consumer (the alpha
rescale near the loop end). Under the 152-register cap ptxas spills the
acc tile to local memory, and the spill/fill traffic lands in whatever
segments touch the register file (the "qk load" and "math" rows).
Sinking the two loads to their use sites — a 2-line move per WG —
collapses the iteration 3.1× and the kernel goes 206.7 → 557 TFLOPS.
P-bridge depth 2 (fix 5) adds a little (recurrence-bound: the wait
mostly relocates, 3682 → 3497).

**Clean result (no instrumentation): 703–720 TFLOPS at (1,32,8192)
over 4 runs — ABOVE the 665–666 plateau (best committed single-tile
config: 666.1). The Route A acceptance criterion is met**: sub-tiling
plus the solver's own schedule/partition beats every single-tile
config, once the emitter defects are patched and the spill-inducing
load placement is corrected. (Instrumentation note: the 9 masked debug
stores cost ~13% — 561 vs 717 TFLOPS — never leave stamps in a bench.)

Model/emitter work items, re-ranked by measured impact:

1. **Register live-range (×3.1, dominant)**: the model prices
   tmem_load by latency/occupancy but not by the live range its result
   occupies. Two remedies, either sufficient here: (a) emitter rule —
   emit a WG-local tmem_load at its first consumer, not at its
   scheduled cycle (legal: the load's wait stays where the schedule
   put it; ~20 lines); (b) model-side — per-WG register capacity as a
   resource, load-to-use distance × tile-registers as pressure (Twill
   gap 5 "liveness in-formulation", now with a measured 3.1×
   instance). Do (a) first, it is schedule-independent insurance.
2. **Buffer depths for shared/bridge channels** (fix 5 class): minor
   for recurrence-bound loops; the joint solver already owns depths.
3. **Shared-engine contention (×1.33–1.48)**: real, second-order.
   Remaining gap after fixes: 3497 measured vs 2487 modeled ≈ the
   in-phase SFU/TMEM sharing between the two softmax WGs (math segment
   1472 vs ~900 modeled) plus handoff waits. Worth modeling for the
   next increment, not the first.
4. The tutorial-dissection items (correction-WG hoisting, rescale
   skip, BLOCK_N=128, TMEM aliasing) remain the path from ~717 toward
   the tutorial's 1197.
