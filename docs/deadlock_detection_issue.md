# Barrier deadlock detection for TLX warp-specialized kernels

TLX warp-specialized kernels coordinate producer-consumer tasks through hardware mbarriers. Incorrect barrier usage (missing arrives, mismatched counts, wrong phases) causes silent GPU deadlocks that are extremely hard to debug.

This issue tracks the work to add a **constraint-based barrier deadlock detector** as an MLIR pass operating on TTGIR. The approach encodes barrier semantics as Z3 constraints following the design in `triton_lint`'s `barrier_deadlock_design.tex`, adapted to work on MLIR IR instead of Python AST.

## Advantages over the Python AST approach (triton_lint)

- No `# triton-lint: assume` annotations needed — constexprs already specialized in IR
- Barrier identity is clear from SSA def-use chains (no name resolution)
- Replicas already expanded into separate `warp_specialize` partition regions
- Can run on any TTGIR, not just Python-authored kernels

## Design

The detector:
1. Walks `ttg.warp_specialize` regions to identify concurrent tasks
2. Unrolls `scf.for` loops with concrete evaluation of barrier indices and phases
3. Produces a concrete operation trace per task
4. Generates Z3 constraints (Phi_ord, Phi_async, Phi_stall, Phi_B, Phi_R) encoding barrier semantics
5. SAT = deadlock witness; UNSAT = safe within unroll bound

Development phase dumps a standalone Python Z3 script; production phase will migrate to MLIR SMT dialect + SMT-LIB export.

## PR Plan

- [x] **PR 1: Program model extraction** — `BarrierDeadlockAnalysis` class with concrete barrier trace extraction from TTGIR.
- [x] **PR 2: Z3 constraint encoding + pass** — `dumpPythonZ3Script()` method + `BarrierDeadlockDetection` pass.
- [ ] **PR 3: Lit tests** — More lit tests covering additional bug patterns.
- [ ] **PR 4: Integration** — Wire into TLX compilation pipeline as an optional diagnostic pass.
- [ ] **(Future) PR 5: MLIR SMT dialect migration** — Replace Python Z3 script with `smt.*` dialect ops.

## Key files

| File | Purpose |
|------|---------|
| `include/triton/Analysis/BarrierAnalysis.h` | Data structures and `BarrierDeadlockAnalysis` class |
| `lib/Analysis/BarrierAnalysis.cpp` | Implementation: trace extraction (PR 1) + Z3 encoding (PR 2) |
| `lib/Dialect/TritonGPU/Transforms/BarrierDeadlockDetection.cpp` | Pass entry point (PR 2) |
| `docs/barrier_deadlock_detection_design.md` | Design doc (MLIR adaptation of `barrier_deadlock_design.tex`) |

---

## Current Status (2026-04-06)

### PR 1 — Merged

Branch `shuofei/barrier-deadlock-analysis-pr1`, GitHub PR #1157 (draft).
- Trace extraction working: walks `warp_specialize`, unrolls `scf.for`, tracks barrier ops with slot/phase.

### PR 2 — Committed locally, NOT yet pushed

Branch `shuofei/barrier-deadlock-analysis-pr2` (stacked on PR 1).
- Z3 constraint encoding (`dumpPythonZ3Script`) fully implemented.
- `BarrierDeadlockDetection` pass registered as `tritongpu-barrier-deadlock-detection`.
- `collectInitialArrives()` handles pre-task barrier arrives.
- Lit tests: 6 total (1 UNSAT correct pipeline + 5 SAT bug patterns). All verified with Z3 solver.

---

## Validation on Real TLX Hopper Kernels

### hopper_gemm_ws — **64/64 UNSAT, 0 false positives**

All 64 TTIR variants (different tile sizes, stages, etc.) pass deadlock detection with UNSAT result. This is a strong validation that the detector produces no false positives on correct warp-specialized kernels.

### hopper_gemm_pipelined — NO-WS (not applicable)

This kernel does not use `warp_specialize`, so no analysis is performed.

### hopper_fa_ws_pipelined_pingpong_persistent — **Z3 timeout**

The FA kernel generates 470 barrier ops across all tasks, producing a 3060-line Z3 script. The Z3 solver returns `unknown` (timeout). See scalability section below.

---

## Technical Notes

### 1. Handling `perThread` Arrives

**Problem**: `ttng.arrive_barrier` supports a `perThread` attribute. When `perThread` is set, each thread in the warp group arrives independently, so the effective arrive count is `count × numWarps × threadsPerWarp`.

**Which ops support perThread**: Only `ArriveBarrierOp` (`ttng.arrive_barrier`) has the `perThread` attribute. Other barrier ops (`barrier_expect`, `wait_barrier`, `async_tma_copy_global_to_local`) do not have it.

**Solution**:
- In `tryAppendBarrierOp`, when processing an `arrive_barrier` with `getPerThread() == true`, multiply the arrive count by `numThreads`:
  ```cpp
  if (arriveOp.getPerThread())
    count *= numThreads;
  ```
- `numThreads` is computed per-task from:
  - `partitionNumWarps`: per-partition warp count from `WarpSpecializeOp.getPartitionNumWarps()` array attribute.
  - `threadsPerWarp`: from the module attribute `ttg.threads-per-warp` (typically 32).
  - `numThreads = partitionNumWarps[taskIdx] × threadsPerWarp`
- For the default region (task 0), `numWarps` is derived from the total module `ttg.num-warps` minus the sum of all partition warp counts.

**Validation**: Without this handling, hopper_gemm_ws produced 32 SAT false positives (barriers initialized with `arrive_count=128` but analysis only saw 1 arrive per `perThread` arrive op). After the fix, all 64 variants are UNSAT.

### 2. Handling Cluster Barriers (cta_bars)

**Problem**: Some Hopper kernels use cluster barriers for inter-CTA synchronization. These barriers use `ttng.map_to_remote_buffer` to arrive on a remote CTA's barrier. The analysis only models a single CTA, so it sees only the local arrive (count=1) but the barrier expects arrives from multiple CTAs (e.g., `arrive_count=2`), causing false SAT results.

**Current workaround**: In `collectBarrierAllocs()`, detect cluster barriers by checking if any user of the barrier allocation flows into a `MapToRemoteBufferOp`, and skip the entire alloc:
```cpp
bool isClusterBarrier = false;
for (auto *user : result.getUsers()) {
  if (auto indexOp = dyn_cast<gpu::MemDescIndexOp>(user)) {
    for (auto *indexUser : indexOp.getResult().getUsers()) {
      if (isa<nvidia_gpu::MapToRemoteBufferOp>(indexUser)) {
        isClusterBarrier = true;
        break;
      }
    }
  }
  if (isClusterBarrier) break;
}
if (isClusterBarrier) return;  // skip this alloc entirely
```
When a barrier alloc is skipped, all operations referencing it (arrives, waits, expects, TMA loads) are also skipped — they fail barrier resolution and are silently dropped.

**Why this doesn't cause false positives**: The skip is at alloc granularity. A cluster barrier alloc and a local barrier alloc are always separate `local_alloc` ops — they are never mixed. Skipping the entire alloc means both the waits and arrives on that barrier are dropped together, so there is no arrive/wait imbalance. The remaining local barriers are unaffected and independently analyzable.

**⚠️ Known limitation — false negatives**: This workaround **cannot detect deadlocks in cluster barrier usage**. If a cluster barrier has a bug (e.g., missing remote arrive, wrong phase), the detector will not catch it. This is a soundness gap: no false positives, but possible false negatives for multi-CTA synchronization patterns. The correct fix is to implement full multi-CTA barrier modeling (see Future Work), not to keep skipping.

**Validation**: Without this handling, hopper_gemm_ws produced 32 SAT false positives from cluster barriers. After the fix, combined with perThread handling, all 64 variants are UNSAT.

### 3. `tryEvalInt` Refactoring for `scf.while` iter_args

**Problem**: The original `tryEvalInt` only tracked the `scf.for` induction variable as a known value. Real kernels pass barrier slot indices and phase values through `scf.for` iter_args (e.g., `accum_cnt` for round-robin slot indexing), which could not be resolved, producing `slotIndex=-1`.

**Solution**: Changed `tryEvalInt` to accept a `DenseMap<Value, int64_t> &knownValues` map instead of a single `(loopVar, loopInductionVar)` pair. The map is populated with:
- The induction variable → `lowerBound + k * step`
- All `scf.for` iter_arg block arguments → their initial values (from `getInitArgs()`) for iteration 0, then yield values from the previous iteration

This unified approach handles all value resolution through a single mechanism and naturally supports the phase XOR pattern (`phase ^= (buf == NUM_STAGES - 1)`).

### 4. `scf.while` Loop Traversal

**Problem**: Persistent kernels wrap the `scf.for` K-loop in an `scf.while` tile loop. The analysis needs to find and unroll the inner `scf.for` loops.

**Solution**: `findAndUnrollLoops()` recursively traverses `scf.while` bodies to find `scf.for` loops:
- Walk each region of each op in the task
- If the op is `scf.for`, unroll it
- If the op is `scf.while`, recurse into its `before` and `after` regions
- This finds the K-loop regardless of how many `scf.while` wrappers exist

### 5. Z3 slotKey Sanitization

Barrier slot keys like `alloc_0_neg1` (for slot index -1 from unresolved values) need to be valid Python identifiers. Negative indices are converted using a `neg` prefix (e.g., `-1` → `neg1`).

---

## Z3 Solver Scalability

### The Problem

The FA kernel (`hopper_fa_ws_pipelined_pingpong_persistent`) generates significantly more barrier operations than GEMM:

| Kernel | Barrier Ops | Z3 Script Lines | Solver Result |
|--------|------------|-----------------|---------------|
| hopper_gemm_ws (per variant) | ~40-60 | ~500-800 | UNSAT (<1s) |
| hopper_fa_ws_pipelined_pingpong_persistent | ~470 | ~3060 | unknown (timeout) |

### Root Cause

The constraint system grows quadratically with the number of operations:
- **Φ_ord** constraints: O(n²) timestamp ordering within each task
- **Φ_R** (release) constraints: each WAIT needs an `arrive_count_before(b, tau_w)` predicate that sums over all arrive ops with `If(effect_time < tau_w, count, 0)` — this creates O(n × m) terms where n = waits and m = arrives
- **Φ_B** (blocking) constraints: similarly require aggregating over all arrive ops

With 470 ops, the solver gets ~220,000 constraint terms, overwhelming Z3's integer arithmetic solver.

### Potential Solutions (Future Work)

1. **Barrier-local decomposition**: Solve each barrier independently instead of one global formula. Most barriers are independent (don't share arrive ops), so this would reduce to many small problems.
2. **Symmetry reduction**: Exploit the pipeline stage symmetry — iterations 0..N-1 of the unrolled loop have identical structure shifted by one stage.
3. **Bounded model checking with abstraction**: Instead of full Z3, use a simpler fixed-point check on the barrier state machine per barrier slot.
4. **Reduce unroll bound**: The FA kernel may need a smarter unroll bound selection. Currently it unrolls based on `NUM_STAGES + 1`, but FA has more complex phase patterns.
5. **In-process solving (MLIR SMT dialect)**: Eliminating the Python overhead and using C++ Z3 API directly would help, though the fundamental complexity remains.

---

## Future Work

### Short-term (PR 3-4)

1. **More lit tests** (PR 3): phase mismatch, circular dependency, arrive count mismatch.
2. **Pipeline integration** (PR 4): Wire into `CUDABackend.run_static_analysis()` behind `TRITON_ENABLE_STATIC_ANALYSIS` env var. Currently `run_static_analysis` is a no-op.

### Medium-term

3. **Multi-CTA / cluster barrier modeling** ⚠️: Currently cluster barriers are **silently skipped**, creating a false-negative gap (see Technical Note §2). The correct fix is to model multi-CTA semantics: track `map_to_remote_buffer` to identify which remote CTA arrives on which barrier, and include remote arrives in the constraint encoding. This is needed for Hopper cluster kernels and is critical for Blackwell 2-CTA kernels (`blackwell_gemm_2cta`).
4. **FA scalability**: Investigate barrier-local decomposition or abstraction to handle FA kernel's 470 ops.
5. **`warp_group_dot` / `warp_group_dot_wait` support**: These are async tensor core ops that affect barrier semantics (ARRIVE_ASYNC kind). Real FA kernels use these extensively.
6. **`scf.while` full support**: Currently we traverse `scf.while` bodies to find `scf.for` loops, but don't unroll the `scf.while` itself. For persistent kernels, the outer while loop's iteration count may matter.

### Long-term

7. **MLIR SMT dialect migration** (PR 5): Replace Python Z3 script generation with `smt.*` dialect ops + SMT-LIB export for in-process solving.
8. **Named barrier support**: Thread-count-based barriers (`bar.arrive` / `bar.sync`) have different completion semantics.
9. **Blackwell kernel support**: `tcgen05_commit`, TMEM barriers, CLC patterns.
