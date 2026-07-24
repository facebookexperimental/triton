# Buffer-Depth Model for the Modulo Scheduler

How the modulo scheduler sizes multi-buffered SMEM/TMEM rings, why the
`lifetime/II + 1` formula under-provisioned them, the fix, and the open
questions.

## 1. Problem

The scheduler chose SMEM prefetch-ring depths that were too shallow. For the
`case1` fp16 GEMM (128x64 / 64x128 tiles, II=256) it allocated **3** buffers,
while an autotuned hand-written kernel peaks at **5–6**. Measured on B200, going
from depth 3 → 5/6 is worth **+14–18%** — the generated kernel was leaving that
on the table purely from ring depth (structure was otherwise ~equal at equal
depth).

The depth is set by `computeBufferCount` (`ModuloSchedulePass.cpp`):

```
num_buffers = lifetime / II + 1
lifetime    = lastConsumerEnd - producerStart
```

The bug: `producerStart` was the `local_alloc` cycle (data-**ready**), which
already folds in the TMA latency. That drops the whole load-transfer window from
the lifetime. The fix (`bufferOccupancyStart`) walks back through the producer's
incoming TMA-load edge and starts the lifetime at the load **issue** cycle.

## 2. The core model: two clocks

A ring buffer's depth is governed by two independent quantities. Conflating them
is the source of most of the confusion.

### 2.1 `R` — the resident span (a serial dependency chain)

`R` is how long **one** buffer slot is occupied, from the moment its load is
issued to the moment it is freed:

```
load_issue ──transfer──▶ data_ready ──▶ MMA_issue ──compute──▶ MMA_done ──▶ slot freed
|<----------------------------- R ----------------------------->|
```

`R` uses **latencies** (result-available delays), because it is a true
dependency chain that **cannot overlap within itself**:

- MMA depends on the load's result (RAW): `MMA_issue ≥ data_ready`.
- The next load into this slot depends on the MMA finishing with it (WAR):
  `next_load_issue ≥ MMA_done`.

For `case1`: `R = load_latency(556) + gap(30) + MMA_latency(559) = 1145`.

Critically, `R` is **not** "just the TMA time" nor "just the MMA time" — it is
the whole `load→MMA` span. The pre-fix code used only the MMA half
(`R = 589 = gap + MMA_latency`), which is exactly why it undercounted.

### 2.2 `II` — the initiation interval (a pipelined start rate)

`II` is how often a **new** iteration (a new chain) may start. Unlike `R`, this
*is* pipelined: both the TMA engine and the tensor core are multi-outstanding,
so successive loads/MMAs overlap. `II` therefore uses **occupancies**
(issue-to-issue throughput), not latencies. (Details in §3.)

For `case1`: `II = 256`.

### 2.3 depth = how many chains overlap

New chains start every `II`; each chain occupies its slot for `R`. During one
slot's life, `R/II` other chains launch, each needing its own slot:

```
cycle:   0        256       512       768      1024      1280
chain0:  |============ R = 1145 ============|                      slot0
chain1:            |============ R ============|                   slot1
chain2:                      |============ R ============|         slot2
chain3:                                |=========== R ...          slot3
chain4:                                          |======= R ...    slot4
                             ^ up to ~5 chains simultaneously live
```

```
depth = ceil(R / II)            (Little's law / interval-overlap)
      = ceil(1145 / 256) = 5
```

(The code currently writes `floor(R/II) + 1`, which equals `ceil` except when
`R` divides `II` evenly, where it yields one extra. See §6.4.)

**This is the whole model.** Everything below is about computing `R` and `II`
correctly and knowing when `ceil(R/II)` is not the final answer.

## 3. Computing `II` precisely

`II = max(ResMII, RecMII, SuperNodeII)`.

### 3.1 ResMII — resource pressure

For each hardware pipeline `P ∈ {TMA, TC, CUDA, SFU}`:

```
ResMII(P) = Σ_{op on P} occupancy(op)          (per iteration)
ResMII    = max_P ResMII(P)
```

`occupancy(op)` is the number of cycles the op *holds its pipeline* before the
next op of that class can issue — **not** its latency. This is where
multi-outstanding is modeled:

| op                | occupancy                              | latency (for `R`, §4)        |
|-------------------|----------------------------------------|------------------------------|
| TMA load          | `max(30, 6·KB)`  (HBM bandwidth share) | `460 + 6·KB + 240·min(insts-1,2)` |
| TMA store         | `max(30, 52·KB)`                       | `130 + 52·KB`                |
| MMA (tcgen05)     | `MACs / MACs_per_cycle` (fp16 4096/cyc), clamped `[30, 4096]` | `900` (K≥128) / `559` (K=64) |
| CUDA elementwise  | `selfLatency` (NCU pipe-active)        | RAW chain cost               |
| SFU (exp2/log2)   | `selfLatency` (insts/4 subpartitions)  | transcendental wait          |

Worked example, `case1`:
- TMA pipeline: two 16 KB loads → `2 · (6·16) = 192`.
- TC pipeline: one 128x128x64 fp16 MMA → `128·128·64 / 4096 = 256`.
- `ResMII = max(192, 256) = 256`.

If loads were (incorrectly) charged their **latency** for ResMII, TMA would read
`2·556 = 1112` and II would be ~1112 — the classic "serial load" mistake. The
model already avoids it; that is why II is 256 and not ~1100.

### 3.2 RecMII — dependency recurrences

For each recurrence circuit `C` (a dependency cycle across loop iterations):

```
RecMII(C) = Σ_{e in C} latency(e) / Σ_{e in C} distance(e)
RecMII    = max_C RecMII(C)
```

`case1` has one recurrence: the MMA accumulating into the same TMEM tile across
K-iterations — a self-edge `MMA → MMA`, `distance=1`, `latency=256`. So
`RecMII = 256/1 = 256`.

### 3.3 `case1` result

`II = max(ResMII=256, RecMII=256) = 256`. Confirmed by the DDG debug log:
`[DDG] Pipeline TMA load: 192  Pipeline TC load: 256  RecMII=256  ResMII=256  MinII=256`.

## 4. Computing `R` precisely

```
R = liveEnd - liveStart
liveStart = bufferOccupancyStart(producer)
          = min over the producer's incoming TMA-load edges of (load.cycle)
          = load ISSUE cycle   (data_ready - load_latency)
liveEnd   = walkLastConsumerEnd(producer)
          = max over consumers of (consumer.cycle + consumer.latency + dist·II)
```

`liveStart` uses the fix; `liveEnd` was already correct (it uses the consumer's
**latency**, e.g. MMA 559 — the completion/free point, not the MMA occupancy).

Note the asymmetry, and it is deliberate:
- `II` uses **occupancy** (throughput; overlappable).
- `R` uses **latency** (completion; the non-overlappable chain).

## 5. Compute-bound vs memory-bound

`depth = ceil(R/II)` holds in **both** regimes; only the *composition* changes.
This directly answers "what if it's compute-bound, load slow vs MMA slow."

| regime          | what sets `II`            | dominant term in `R`     | typical depth behavior |
|-----------------|---------------------------|--------------------------|------------------------|
| **compute-bound** (case1) | TC occupancy / MMA recurrence (256) | MMA latency (559)        | II is small relative to `R` → deeper rings needed to feed the fast-draining consumer |
| **memory-bound** | TMA occupancy (`6·KB` × #loads) | load latency             | II is large (loads slow) → `R/II` smaller → shallower rings, but II itself is the thing to attack |

Worked hypothetical (memory-bound, low arithmetic intensity):
`II_TMA = 400`, `II_TC = 200` → `II = 400`. `R = load_lat(900) + MMA(300) = 1200`
→ `depth = ceil(1200/400) = 3`. Fewer buffers, because the consumer is starved
by bandwidth, not by ring depth — adding buffers wouldn't help; **lowering II
would** (see §6.2).

The key insight: in the compute-bound regime the ring depth is the lever
(feed the hungry tensor core); in the memory-bound regime the ring depth
saturates and the lever moves to `II` (bandwidth / split-K / bigger loads).
`ceil(R/II)` self-adjusts between the two, but it does **not** by itself do the
memory-bound "buy lower II with more buffers" trade — that is §6.2.

## 6. Where `ceil(R/II)` is not the final answer (open questions)

These are the iteration points. `ceil(R/II)` is the steady-state, mean-latency
**minimum** to not stall; it is a floor, not always the optimum.

### 6.1 Latency variance → a slack margin

`R` uses *mean* modeled latency. Real TMA latency has variance (L2/DRAM
contention across 148 SMs, TLB, bank conflicts). The model predicts `case1`
depth **5**; empirically **5–6** is best — the extra slot absorbs jitter so a
slow load doesn't stall the MMA. Options: a small additive/multiplicative slack
(`ceil(R/II) + k`, or `ceil(R·(1+ε)/II)`), capped by the SMEM budget. We should
calibrate `ε`/`k` on measured data, not guess.

### 6.2 Latency-bound regime: buy lower II with more buffers

`depth = ceil(R/II)` treats `II` as fixed. But in the memory-bound regime, `II`
*is a function of ring depth*: with `N` buffers you can sustain `II = max(II_floor, R/N)`.
Today the code fixes `II` from ResMII/RecMII and then derives `N`, and the budget
reducer only ever *cuts* `N`. A fuller design would, when load-bound, **fill `N`
toward the SMEM budget** and recompute `II = R/N`:

```
if R / N_fit > II_res:        # load-bound: buffers are the bottleneck
    N  = N_fit                # N_fit = floor(SMEM_budget / buffer_bytes)
    II = ceil(R / N)          # deeper pipe → lower II
else:                         # resource-bound (case1)
    N  = ceil(R / II_res)
    II = II_res
```

This is the II(N) curve — throughput is concave/saturating in N ("1 buffer ≠ A,
2 buffers ≠ 2A"): `II(N) = max(II_floor, R/N)`, flat once `N ≥ R/II_floor`.

### 6.3 SMEM ring depth vs TMEM accumulator depth are separate

`R/II` sizes the **SMEM operand ring** (load→MMA). The **TMEM accumulator**
(MMA→epilogue) is a different buffer with its own lifetime (produced by the MMA,
consumed by the epilogue store) and should be sized by the same model applied to
*that* chain. They interact through the shared SMEM budget but are not the same
number. Current code sizes both via `computeBufferCount`; verify the accumulator
path also gets the right `R` (its producer is the MMA, not a TMA load, so the
§4 walk-back does not apply — good, but confirm its `R` is complete).

### 6.4 `floor(R/II) + 1` vs `ceil(R/II)`

When `R` is an exact multiple of `II`, `floor+1` gives one extra buffer vs
`ceil`. Decide whether that extra is desired (a free half-slot of slack) or a
rounding artifact to remove. Minor, but it should be intentional.

### 6.5 Per-operand vs shared ring depth

A/B operands feed the same MMA and today are equalized to a common depth
(co-consumed group). If A and B have different `R` (different tile bytes →
different load latency), the shared depth is `max`. That is correct for
correctness; confirm it is not over-allocating the cheaper operand.

### 6.6 Interaction with the budget reducer — FIXED

`reduceBuffersForBudget` cuts the cheapest ring when SMEM overflows and then
recomputes `II = R/depth`. Deeper default rings from this model make the reducer
fire for the first time on large tiles (e.g. BK=128), which exposed a **latent
bug**: `computeTotalSmem` charges each merge-group's *physical* footprint, but
the reducer decremented only the *logical* `buf.count` and never refreshed the
physical buffers — so the total never dropped, the reducer over-reduced every
ring to depth 1, and recomputed a serial `II` (≈ MMA latency). Net: any
over-budget tile collapsed to a non-pipelined depth-1 schedule (worse than
pre-fix).

Fix: call `buildPhysicalBuffers(loop)` after each reduction step (the forward
decl already anticipated this), and make `computeBufferLifetime` use
`bufferOccupancyStart` so the post-reduction `II` recompute uses the same `R` as
`computeBufferCount`. Result on BK=128: reduces 4→3, sees 192 KB fits the 227 KB
budget, stops, raises II 512→528 — depth 3, pipelined. This is the graceful cap:
**over-budget → largest depth that fits, keep pipelining; never collapse to
serial.**

## 7. Validation plan

1. **Mechanism** (done): `case1` depth 3→5, II unchanged at 256, clean build.
2. **Perf** (done, case1/case2): case1 gen/hw 0.80→0.95 @2048³, 0.83→0.88
   @4096³; case2 generated +15% (hw also improved so ratio flat — case2's
   residual is not depth).
3. **Falsification** (DONE — PASSED): on the unseen BLOCK_K=128 tile the model
   predicts depth **3** (ceil(1582/512)=4, capped to 3 by the SMEM budget), and
   the hand-written autotune optimum is **3** (depth ≥4 OOMs). BK=64 stays 5.
   Prediction matches measurement on a tile the model was not tuned to →
   principle, not pattern-match. (A first attempt on a hand-edited BK=128 ttgir
   gave a false negative — the edited input broke the TC→TC recurrence and the
   budget reducer collapsed to depth 1; both were artifacts, not the model.)
4. **Regression** (DONE): regenerated + revalidated on master (post-tensordesc-
   migration, with the fix) for case1/2/3/4/6/7/9 — deeper rings confirmed
   (case1 3→5, case2 3→5, case3 buf1 1→2, case4 3→4, case7 3→5, case9 1→2; case6
   unchanged), all correctness PASS on B200, II unchanged, no budget cap fired,
   no OOM. case5 excluded (epilogue-subtile emitter bug, tracked separately).
   The emitter needed no change (its descriptor regex already matches the
   migrated tensordesc syntax).
5. **Memory-bound** (todo): a load-bound shape to test §6.2 (does II have
   headroom that filling buffers would recover?).

## 8. Current status

- Implemented: `bufferOccupancyStart` + `computeBufferCount` using it; the
  budget-reducer graceful cap (§6.6: `buildPhysicalBuffers` refresh +
  `computeBufferLifetime` consistency) — all in `ModuloSchedulePass.cpp`.
- Validated: case1 (mechanism + perf, 3→5, +15%); falsification passed on BK=128
  (predicts 3, measured 3); BK=64 no regression; II independently confirmed
  correct (occupancy-based, MMA-bound).
- Fixtures regenerated on master for case1/2/3/4/6/7/9 (§7.4); case5 pending its
  epilogue-subtile emitter bug.
- Not yet: variance margin (§6.1), latency-bound fill-to-budget (§6.2), case5.
