# AutoWS DP=2 WAR race: dropped `gemm→compute` "P-consumed" backward edge

Root-cause analysis of the runtime failure in the HSTU self-attention **forward**
kernel (`@_hstu_attn_fwd`) under Meta autoWS with `data_partition_factor=2`
(`num_warps=4`) on Blackwell (`cuda:100`). The kernel compiles and
warp-specializes into the TLX structure but **fails at launch**. The failure is a
**WAR (write-after-read) race**, not a clean barrier deadlock, and it manifests
non-deterministically depending on the memory plan:

- `num_stages=1` and `num_stages=2` → **hang** (GPU pegged at 100% util, kernel
  never returns; `compute-sanitizer synccheck` is clean).
- `num_stages=2` + `TRITON_WS_SMEM_PLAN_SEARCH=1` → **CUDA illegal memory access**
  (no hang, util ~1%).

Non-deterministic manifestation across schedules = **undefined behavior / race**,
not a deterministic phase-cycle deadlock — consistent with the phase analysis
finding no unsatisfiable phase-0 wait and no first-execution barrier cycle (see
"Why it is topological" below).

## One-line root cause

Code partitioning emits the WAR-ordering barrier that must keep a **computation**
partition's **P (softmax) write** from clobbering the **gemm** partition's **PV
read** as a **degenerate self-edge** (`wait_barrier … {dstTask = own-partition}`)
instead of a cross-partition `gemm → compute` **backward "P-consumed"** edge. On
the DP group whose PV MMA is scheduled one pipeline stage later, this severed
edge lets the writes and reads on a single-buffered TMEM slot race with nothing
serializing them → the unserialized async `tcgen5`/`tmem_store` accesses on that
slot corrupt state and desync the tensor-core completion accounting (hang) or
touch invalid state (illegal access).

This is **not** an mbarrier phase-cycle deadlock; it is a **topological** defect
(an *absent* cross-partition edge collapsed to a `dstTask=self` no-op) that
produces a WAR race, invisible to arrive/wait-count balancing.

## The buffer structure that sets it up

`data_partition_factor=2` produces two per-group QK accumulators, each a
**single-buffered (`buffer.copy=1`) TMEM slot reused in place** for two different
logical values within one KV iteration:

```
id8  qk_0  TMEM 128x128  copy=1  (allocation.shareGroup=1)
   phase A: QK-dp0 result   gemm writes (QK MMA)  ->  task4 reads (softmax)
   phase B: P-dp0 (bf16 reinterpret of the same slot)   task4 writes  ->  gemm reads (PV-dp0)
id9  qk_1  TMEM 128x128  copy=1  (allocation.shareGroup=0)
   phase A: QK-dp1 result   gemm writes  ->  task3 reads
   phase B: P-dp1           task3 writes  ->  gemm reads (PV-dp1)
```

Because `copy=1`, the QK result (phase A) and the softmax output P (phase B) live
in the **same physical slot**, so there must be a WAR barrier stopping the
compute partition from overwriting the slot with `P(i+1)` before gemm has read
`P(i)`.

Partition map (`ttg.partition.types = ["epilogue","gemm","load","computation","computation"]`):
`task0`=epilogue, `task1`=gemm, `task2`=load, `task3`=compute-dp1 (P for id9),
`task4`=compute-dp0 (P for id8).

The KV loop carries `tt.data_partition_factor=2`,
`ttg.partition.stages = [0,1,0,0,0]`. The four MMAs:

| MMA | A × B → C | id | loop.stage |
|-----|-----------|----|------------|
| QK-dp0 | q_0 × Kᵀ → **id8** | qk_0 | 0 |
| QK-dp1 | q_1 × Kᵀ → **id9** | qk_1 | 0 |
| PV-dp0 | P(id8) × V → acc_0(id6) | | **0** |
| PV-dp1 | P(id9) × V → acc_1(id7) | | **1** |

## The broken IR (the WAR edge is a self-edge)

Line numbers are for the code-partition output of the deadlocking kernel
(`_hstu_attn_fwd`).

The compute partition's acquire, immediately before it stores P into the slot:

```mlir
// task3 (compute-dp1), before overwriting id9 with P:
L607:  ttng.wait_barrier %191, %192 {WSBarrier = {dstTask = 3}}   // %191 = %arg78[0] = %55
L608:  ttng.tmem_store   P -> id9
// task4 (compute-dp0): same shape at L716 (barrier %51, dstTask = 4), then P -> id8
```

`%55` is **the very barrier task3 arrived on 16 lines earlier** (L591 — its
"QK-consumed" signal to gemm). So the acquire waits on its *own* partition's
arrive. Tell-tale signature of a dropped cross-partition channel collapsed onto
the partition's own QK-consumed barrier:

- `dstTask = 3` (its own id) — not the producer partition,
- no `direction = "backward"`, no `channelGraph`.

Correspondingly, gemm's PV MMAs carry **no backward operand toward the compute
partitions**:

```mlir
L356 gemm PV-dp0: tc_gen5_mma P(id8),V,acc_0, %arg84,%true, %205[%true]            // %205=%2 (self QK-WAR only)
L371 gemm PV-dp1: tc_gen5_mma P(id9),V,acc_1, %arg84,%true, %218[%true],%219[%true] // %218=%4(V-empty), %219=%0(self)
```

Neither arrives a "P-consumed" barrier back to task3/task4.

## The FA contrast (what a correct kernel emits)

FA fwd DP=2 has the **same single-buffered acc/QK layout** but the real edge:

```mlir
// FA compute P-write:
L777:  ttng.wait_barrier %acc_0_192, %acc_193 {WSBarrier = {direction = "backward", dstTask = 1}}
L778:  ttng.tmem_store   P
// FA gemm PV MMA arrives that backward barrier:
L539/L582:  tc_gen5_mma ... %acc_0_211 / %acc_0_239, %acc_0_242
```

So in FA:

- **Yes** — the gemm PV MMA carries a backward barrier operand toward the compute
  partition (the `acc_0`/`acc_1` trailing barriers).
- **Yes** — the compute P-write waits on a genuine `task1 → compute`,
  `direction="backward"`, `dstTask=1` barrier, **not** a self-edge.

HSTU is missing exactly this.

## Why it bites dp1 (id9) specifically

Both slots are `copy=1` and reused in place; the difference is the SWP schedule:

- **PV-dp0 is `loop.stage=0`** — producer (task4 P-write) and consumer
  (gemm PV-dp0) run in the *same* pipeline stage, so they stay lockstep and the
  broken edge is *accidentally* masked.
- **PV-dp1 is `loop.stage=1`** — P must stay live in the `copy=1` id9 slot
  **across a pipeline-stage boundary** while task3 already wants to produce the
  next P. This is precisely the case that *requires* the backward edge. Collapsed
  to a self-edge, task3's P-write into id9 is unordered against gemm's stage-1
  PV-dp1 read; the two async `tcgen5` ops on the same TMEM with no token
  dependency are the WAR race.

One observed manifestation (num_stages=1/2, no plan-search): gemm spins at the
PV-dp1 P-ready wait (`wait_barrier %220 = %62`) together with task3's severed
acquire at L607 — the tensor-core completion the wait expects never arrives in the
required order. Under plan-search the same race instead faults (illegal access).
This is a *symptom* of the UB, not a proof of a specific never-satisfied barrier;
pinning the exact stuck warp/barrier would need `cuda-gdb` on the hung kernel.

## Why it is topological, not a phase cycle (and why the O-accumulator is NOT it)

Applying the corrected mbarrier phase model (a freshly `init_barrier … ,1`
barrier with no pre-arm: `wait phase=1` passes on first execution — empty/reuse
barriers are pre-inverted so the producer's first acquire is free — while
`wait phase=0` blocks until one arrive):

- All wait/arrive/commit sites resolve to a producer → **no "no-producer" hang**.
- The two acquires on `%55`/`%51` poll **opposite parities** with **one
  arrive/iter** → the benign redundant-acquire signature, **not** cadence
  starvation.
- The self-edge waits (L607/L716) are satisfied by the **same partition's own
  arrive** two lines earlier → they never gate another partition, so **no
  cross-partition phase cycle** either.

So the defect is a **severed synchronization edge**, invisible to arrive/wait
count balancing (the self-edge is individually "balanced").

The **O-accumulator** channel (id6/id7, epilogue↔gemm) was independently
**confirmed acyclic and correct**: the epilogue's pre-loop backward acc-init wait
is `phase=1` (pre-inverted) and passes on first execution, the init arrive
satisfies gemm's pre-loop wait, and gemm's post-loop commit satisfies the
epilogue's forward drain wait. An earlier analysis mis-flagged this as "cycle 1"
by treating the `phase=1` backward wait as blocking — a phase-model misread. That
is also why removing the O-acc zero-init and bumping `num_stages=2` did **not**
fix the hang: neither touches the real (P-consumed) defect.

## Compiler location

This is the reuse-owner-consumed-cross-partition channel that `WSCodePartition` /
`CodePartitionUtility` (the `isFullOverwriteReuseOwner` / `handleOperandD` path)
is supposed to emit. It is being **dropped / self-routed specifically for the
HSTU-fwd DP=2 pattern where P is reinterpreted into the QK TMEM slot** (the
in-place QK→P reuse), rather than P getting its own buffer as FA effectively
does.

## Fix (either; both mirror FA)

1. **Restore the backward `gemm → compute` "P-consumed" edge** (preferred,
   smaller, FA-consistent): make gemm's PV-dp0/PV-dp1 MMAs arrive a dedicated
   barrier toward task4/task3, and replace the self-edge acquires (L716/L607)
   with `wait_barrier {direction="backward", dstTask=1}` before the P
   `tmem_store`.
2. **Give P its own buffer** — allocate P separately or bump the P slot to
   `buffer.copy=2`, so `P(i)` and `QK(i+1)`/`P(i+1)` occupy different slots and
   the stage-1 PV-dp1 no longer needs a same-slot WAR back-edge.

## References

- Deadlock IR (code-partition output): `_hstu_attn_fwd`, DP=2, `num_warps=4`.
- Working reference: FA fwd DP=2 (`@_attn_fwd`, `fused_attention_ws_device_tma.py`).
- Compiler: `WSCodePartition.cpp`, `CodePartitionUtility.cpp`
  (`isFullOverwriteReuseOwner`, `handleOperandD`, reuse-group channel emission).
- Regression test: `test/Hopper/WarpSpecialization/ws_code_partition_dp_idle_acc_init_deadlock.mlir`
  (currently xfail; its `CHECK-NOT` targets the *benign* O-acc backward wait and
  should be **retargeted** to the real signature — the `dstTask=self` P-write
  self-edge / missing `direction="backward"` gemm→compute edge).
- Detection methodology: `.claude/skills/barrier-visualization/SKILL.md`
  (mbarrier phase model; the corrected `phase=1`-free-first rule is essential to
  avoid the O-acc false positive).
