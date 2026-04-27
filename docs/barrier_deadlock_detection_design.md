# Barrier Deadlock Detection for TTGIR — Design Document

This document describes the design of a **constraint-based barrier deadlock
detector** operating on Triton GPU IR (TTGIR). The detector is an MLIR
analysis pass that extracts a concrete barrier operation trace from
`ttg.warp_specialize` regions, encodes barrier semantics as Z3 constraints,
and checks satisfiability: SAT = deadlock witness, UNSAT = safe within the
unroll bound.

The constraint encoding follows the authoritative design in
`triton_lint/docs/deadlock_detection/barrier_deadlock_design.tex` (Sections
2-5). This document focuses on the **MLIR-specific adaptation** — how the
program model is extracted from TTGIR rather than Python AST, and what each
PR delivers.

## 1. Background

### 1.1 Problem

TLX warp-specialized kernels coordinate producer-consumer tasks through
hardware mbarriers. Incorrect barrier usage — missing arrives, mismatched
counts, wrong phases — causes silent GPU deadlocks that are extremely hard to
debug.

### 1.2 Why TTGIR instead of Python AST

The `triton_lint` implementation operates on the Python AST of TLX kernels.
An MLIR-based detector has several advantages:

| Aspect | Python AST (`triton_lint`) | TTGIR (this work) |
|--------|---------------------------|-------------------|
| Constexpr resolution | Needs `# triton-lint: assume` annotations | Already specialized in IR |
| Barrier identity | Name resolution across scopes | SSA def-use chains (`memdesc_index` → `local_alloc`) |
| Replicas | Must expand `replicate=N` manually | Already separate `warp_specialize` partition regions |
| Helper functions | Must inline Python calls | Already inlined by compiler |
| Language coupling | Python-only | Works on any frontend that emits TTGIR |

### 1.3 Constraint encoding summary

The detector encodes the mbarrier state machine as Z3 constraints (see tex
doc Sections 2, 5 for full details):

- **Variables**: cut-point `c_i` per task (where stuck), timestamp `τ_j` per
  op (global order), async completion timestamp `τ'_l` for TMA loads and
  async arrives.
- **Φ_ord**: intra-task program ordering via timestamps.
- **Φ_async**: causal constraint `τ < τ'` for async ops; `async_dot_wait`
  ordering.
- **Φ_stall**: tasks can only stall at WAIT positions.
- **Φ_B** (blocking): stalled WAIT's barrier is permanently blocked — either
  arrive/byte deficit or phase cycling.
- **Φ_R** (reachability): every passed-through WAIT was satisfiable *at the
  time it was reached*, using timestamp-filtered arrive counts.
- **Deadlock query**: `∨(c_i < n_i)` — at least one task is stuck.

## 2. Program Model

After loop unrolling with bound U, the program model is a tuple
`(T, B, O)`:

- **T**: set of concurrent tasks (from `warp_specialize` regions).
- **B**: set of barrier slots. Each slot `b` has arrive count `ac(b)`.
- **O**: set of concrete barrier operation instances (after unrolling).

### 2.1 Operation kinds

Each operation has a `kind` from the following classification (tex doc
Section 2.2):

| Kind | TTGIR op | Barrier effect | Effect time |
|------|----------|----------------|-------------|
| **ARRIVE_SYNC** | `ttng.arrive_barrier` | `pending_arrives -= cnt` | τ (synchronous) |
| **EXPECT_BYTES** | `ttng.barrier_expect` | `pending_arrives -= 1` AND `pending_bytes += sz` | τ (synchronous) |
| **TMA_LOAD** | `ttng.async_tma_copy_global_to_local` | `pending_bytes -= xfer` on DMA completion | τ' (async) |
| **ARRIVE_ASYNC** | from `async_dot` / `tcgen05_commit` lowering | `pending_arrives -= 1` on engine completion | τ' (async) |
| **WAIT** | `ttng.wait_barrier` | blocks while `phase == φ` | τ (timestamp of return) |
| **ASYNC_DOT_WAIT** | `ttng.async_dot_wait` | forces completion of prior async dots | τ |

Key semantic points (from tex doc Remarks):
- **`barrier_expect`** (PTX `mbarrier.arrive.expect_tx`) does both an arrive
  (cnt=1) AND sets expected bytes. It is a full arrive operation.
- **TMA load** only does byte completion, NOT an arrive.
- **`async_dot`** arrives are asynchronous — the barrier update happens when
  the tensor core completes, not when the instruction retires.

### 2.2 Operation attributes

Each concrete operation carries (all concrete integers, no symbolic
variables):

- `task(o)`: enclosing task.
- `pos(o)`: position within task (program order after unrolling).
- `kind(o)`: from the table above.
- `slot(o) = (alloc_name, slot_index)`: target barrier slot.
- `cnt(o)`: arrive count (for arrive kinds).
- `sz(o)`: byte count (for EXPECT_BYTES).
- `xfer(o)`: transfer size in bytes (for TMA_LOAD).
- `phase(o)` / `parity(o)`: target phase for WAIT.
- `iteration`: which unrolled loop iteration produced this op.

## 3. TTGIR to Program Model (PR 1)

This is the MLIR-specific adaptation of tex doc Section 4 ("From AST to
Program Model"). The translation extracts concrete barrier traces from TTGIR.

### 3.1 `warp_specialize` structure

A TLX kernel in TTGIR has the form:

```mlir
ttg.warp_specialize(%cap1, %cap2, ...)
    default {
        // Producer task (task 0, "default")
        scf.for %k = %c0 to %K step %c1 {
            ttng.wait_barrier %bar_empty, %phase_e
            ttng.barrier_expect %bar_full, %size
            ttng.async_tma_copy_global_to_local %desc, %buf, %bar_full
            // phase update...
        }
    }
    partition0(%arg0 = %cap1, %arg1 = %cap2, ...) {
        // Consumer task (task 1, "partition0")
        scf.for %k = %c0 to %K step %c1 {
            ttng.wait_barrier %bar_full, %phase_f
            // compute...
            ttng.arrive_barrier %bar_empty
            // phase update...
        }
    }
```

Key observations:
- The **default region** and each **partition region** are separate concurrent
  tasks.
- Shared values (barriers, descriptors) are passed via **explicit captures** —
  operands of `warp_specialize` that appear as block arguments in partition
  regions.
- Replicas are already expanded: `replicate=2` produces `partition0` and
  `partition1` with separate block arguments.

### 3.2 Barrier allocation collection

Walk the function for `ttng.local_alloc` ops that produce barrier memdesc
types. For each allocation:

```
BarrierAllocInfo {
    name:         unique identifier (e.g., "alloc_0" from op ordering)
    numSlots:     number of barrier slots (from memdesc shape)
    arriveCount:  from init_barrier's arrive_count attribute
}
```

Build a map `allocOp → name` for later barrier resolution.

### 3.3 Barrier resolution via def-use chains

Given a barrier `Value` used by a barrier op, resolve it to
`(allocName, slotIndex)` by tracing the SSA def-use chain:

```
barrier Value
  ↓ (defining op)
memdesc_subview or memdesc_index
  ↓ (source operand)
block argument of partition region
  ↓ (corresponding capture operand of warp_specialize)
memdesc_subview or local_alloc in the parent scope
  ↓
local_alloc (→ allocName from map)
```

The slot index comes from the index operand of `memdesc_subview`, which must
evaluate to a concrete integer (it depends on the loop induction variable
after unrolling).

### 3.4 Concrete evaluation (`tryEvalInt`)

During loop unrolling, we evaluate integer expressions by walking the SSA
def-use chain with the loop induction variable bound to a concrete value.
Supported patterns:

- `arith.constant` → literal value
- `arith.addi`, `arith.subi`, `arith.muli`, `arith.remsi`, `arith.divsi` →
  binary arithmetic
- `arith.xori`, `arith.andi`, `arith.ori` → bitwise ops
- `arith.cmpi` → comparison (returns 0 or 1)
- `arith.extui`, `arith.extsi`, `arith.trunci` → unary casts (must be
  handled before the binary-op check since they have only 1 operand)
- `arith.select` → conditional selection
- Loop induction variable → current iteration value
- Block arguments → trace through captures to parent scope

If any expression cannot be resolved to a concrete integer, the operation is
skipped with a warning (consistent with the tex doc's no-symbolic-fallback
policy).

### 3.5 Loop unrolling

For each `scf.for` in a task region:

1. Determine unroll bound U:
   - If `unrollBound` pass option > 0, use that.
   - Else if loop bounds are static, use `min(tripCount, NUM_STAGES + 1)`.
   - Else default to U = 2.
2. For each iteration k = 0, 1, ..., U-1:
   - Bind the induction variable to `lowerBound + k * step`.
   - Walk the loop body for barrier ops.
   - For each barrier op, evaluate all parameters (slot index, phase, size,
     count) using `tryEvalInt`.
   - Append a `ConcreteBarrierOp` to the task's trace.

Phase tracking is handled by **faithful evaluation** of the programmer's
actual expressions (tex doc Section 4.2). For example, the pattern
`phase ^= (buf == NUM_STAGES - 1)` naturally evaluates correctly: `buf` and
`NUM_STAGES` are concrete, so `cmpi eq` returns 0 or 1, `extui` extends it,
and `xori` flips the phase when appropriate.

### 3.6 Output: TaskTrace

The result is a `std::vector<TaskTrace>`, where each `TaskTrace` contains the
ordered sequence of `ConcreteBarrierOp`s for one task.

### 3.7 Data structures (C++)

```cpp
enum class BarrierOpKind {
    ArriveSync,      // ttng.arrive_barrier
    ExpectBytes,     // ttng.barrier_expect (arrive + set bytes)
    TmaLoad,         // ttng.async_tma_copy_global_to_local (bytes only)
    ArriveAsync,     // from async_dot / tcgen05_commit
    Wait,            // ttng.wait_barrier
    AsyncDotWait,    // ttng.async_dot_wait
};

struct ConcreteBarrierOp {
    BarrierOpKind kind;
    std::string allocName;
    int64_t slotIndex;
    int64_t phase;          // for Wait
    int64_t arriveCount;    // for arrive kinds (0 for TmaLoad)
    int64_t expectedBytes;  // for ExpectBytes
    int64_t xferBytes;      // for TmaLoad
    int64_t iteration;
    size_t position;        // within task trace
};

struct TaskTrace {
    int64_t taskId;
    std::string taskName;
    std::vector<ConcreteBarrierOp> ops;
};

struct BarrierAllocInfo {
    std::string name;
    int64_t numSlots;
    int64_t arriveCount;
};
```

## 4. Z3 Constraint Encoding (PR 2)

PR 2 generates a standalone Python Z3 script from the program model. This
follows tex doc Section 5 and the reference implementation in
`z3_encoding.py`.

### 4.1 Variables

```python
# Cut-point per task: where execution is stuck
c_<task> = Int('c_<task>')    # 0 <= c_i <= n_i

# Timestamp per operation: global execution order
tau_<idx> = Int('tau_<idx>')  # >= 0

# Async completion timestamp (TMA_LOAD, ARRIVE_ASYNC only)
tau_prime_<idx> = Int('tau_prime_<idx>')  # >= 0
```

### 4.2 Helper predicates

```python
# exec(o) = pos(o) < c_task(o)
def exec_op(o): return o.pos < c[o.task]

# effect_time(o) = tau'_o for async ops, tau_o for sync ops
def effect_time(o):
    if o.kind in {TMA_LOAD, ARRIVE_ASYNC}: return tau_prime[o]
    else: return tau[o]

# A(b) = Σ If(exec(o), cnt(o), 0) for arrive-kind ops on slot b
# A^{<t}(b) = Σ If(exec(o) ∧ effect_time(o) < t, cnt(o), 0)
```

### 4.3 Completion and blocking (arrive-only simplification)

For the initial implementation, we use the **arrive-only** completion model
(tex doc Eq. 6, simplified without per-cycle byte tracking):

```python
# ct(b, i) = A(b) >= (i+1) * ac(b) ∧ total_loaded >= total_expected
# blocked(b, i) = ¬ct(b,i) ∨ ct(b,i+1)
```

The byte condition uses **cumulative** (not per-cycle) tracking: sum all
executed EXPECT_BYTES sizes vs. sum all executed TMA_LOAD transfer sizes.
This is the approach used in the reference implementation (`completion()`
method), which avoids the complexity of timestamp-based per-cycle byte
attribution while remaining sound.

### 4.4 Parity-based blocking (Φ_B)

When the wait's phase/parity is concretely known (which it always is after
our concrete evaluation), we use the parity-based blocked predicate:

```python
# blocked_parity(b, p) = (A(b) / ac(b)) % 2 == p
# i.e., the barrier's current phase equals the wait's parity → still blocked
```

This is equivalent to the cycle-based `blocked(b, i)` but uses only the
global arrive count, which is simpler. Falls back to cycle-based when
parity is unknown.

### 4.5 Release constraint (Φ_R)

For each WAIT that was passed through (pos < c_i):

```python
# Parity-based: phase at τ_w must differ from wait's parity
a_before = arrive_count_before(b, tau_w)
phase_before = (a_before / ac) % 2
release_cond = (phase_before != parity)

# Cycle-based fallback:
# arrive_ready = A^{<τ_w}(b) >= (i_w+1) * ac(b)
# anti_cycle = ¬ct^{<τ_w}(b, i_w+1)
# release_cond = arrive_ready ∧ anti_cycle
```

### 4.6 Constraint assembly

```python
solver = Solver()

# Structural
solver.add(0 <= c_i, c_i <= n_i)
solver.add(tau >= 0, tau_prime >= 0)

# Φ_ord: consecutive ops in same task
solver.add(Implies(exec(a) ∧ exec(b), tau_a < tau_b))

# Φ_async: causal ordering
solver.add(Implies(exec(l), tau_l < tau_prime_l))

# Φ_stall: stall only at WAIT positions
solver.add(Implies(c_i < n_i, Or(c_i == w.pos for w in waits)))

# Φ_B: blocked barrier at stall point
solver.add(Implies(c_i == w.pos ∧ c_i < n_i, blocked(w)))

# Φ_R: passed-through waits were satisfiable
solver.add(Implies(w.pos < c_i, release_cond(w)))

# Deadlock query
solver.add(Or(c_i < n_i for each task))
```

### 4.7 Development vs. production

- **Development (PR 2)**: Generate a standalone Python script that imports
  `z3` and prints SAT/UNSAT with diagnostic extraction. Run via
  `python3 /tmp/triton_deadlock_check.py`.
- **Future (PR 5)**: Migrate to MLIR SMT dialect + SMT-LIB export for
  in-process solving without Python dependency.

## 5. Pass Registration (PR 2)

The pass is registered as `--triton-barrier-deadlock-detection` in
`triton-opt`:

```tablegen
def TritonBarrierDeadlockDetection
    : Pass<"triton-barrier-deadlock-detection", "ModuleOp"> {
  let summary = "Detect potential barrier deadlocks in warp-specialized kernels";
  let options = [
    Option<"outputPath", "output-path", "std::string", /*default=*/"\"\"",
           "Path to write Z3 script (empty = stderr)">,
    Option<"runSolver", "run-solver", "bool", /*default=*/"false",
           "Run python3 on the generated Z3 script">,
    Option<"unrollBound", "unroll-bound", "int", /*default=*/"0",
           "Loop unrolling bound (0 = auto)">,
  ];
}
```

Usage:
```bash
triton-opt --triton-barrier-deadlock-detection \
           --triton-barrier-deadlock-detection-output-path=/tmp/check.py \
           input.ttgir
python3 /tmp/check.py
```

## 6. Lit Tests (PR 3)

Test cases as TTGIR `.mlir` files:

| Test | Scenario | Expected |
|------|----------|----------|
| `correct_pipeline.mlir` | Standard producer-consumer with correct phases | UNSAT |
| `missing_arrive.mlir` | Consumer waits but no arrive from producer | SAT |
| `missing_tma_load.mlir` | `barrier_expect` without matching TMA load | SAT |
| `phase_mismatch.mlir` | Consumer waits with wrong phase | SAT |
| `circular_dependency.mlir` | Two tasks each waiting on the other's arrive | SAT |
| `arrive_count_mismatch.mlir` | `arrive_count=2` but only 1 arrive | SAT |

Each test uses `FileCheck` to verify the generated Z3 script structure (or
uses `--run-solver` to verify SAT/UNSAT directly if Z3 is available in CI).

## 7. PR Plan

### PR 1: Program model extraction

**Scope**: Extract concrete barrier operation traces from TTGIR.

**Files**:
- `include/triton/Analysis/BarrierAnalysis.h` — Data structures
  (`ConcreteBarrierOp`, `TaskTrace`, `BarrierAllocInfo`) and
  `BarrierDeadlockAnalysis` class declaration (trace-building methods only,
  no Z3).
- `lib/Analysis/BarrierAnalysis.cpp` — Implementation of:
  - `collectBarrierAllocs()`: walk for `local_alloc` + `init_barrier`.
  - `buildTaskTraces()`: walk `warp_specialize`, dispatch to
    `processWarpRegion()`.
  - `processWarpRegion()`: walk region for barrier ops and `scf.for` loops.
  - `unrollLoop()`: iterate k=0..U-1, evaluate parameters, build
    `ConcreteBarrierOp`s.
  - `resolveBarrier()`: trace def-use chain to `(allocName, slotIndex)`.
  - `tryEvalInt()`: concrete integer expression evaluator.
- `lib/Analysis/CMakeLists.txt` — Add `BarrierAnalysis.cpp`.

**Does NOT include**: Z3 encoding, pass registration, `dumpPythonZ3Script()`.

**Testing**: Unit-style verification by dumping traces (print method) and
checking against expected output for a hand-written TTGIR test case.

### PR 2: Z3 constraint encoding + pass

**Scope**: Generate Z3 constraints from the program model and register the
`triton-opt` pass.

**Files**:
- `include/triton/Analysis/BarrierAnalysis.h` — Add `dumpPythonZ3Script()`
  method.
- `lib/Analysis/BarrierAnalysis.cpp` — Implement `dumpPythonZ3Script()`:
  emit Python Z3 script with all constraint components (Φ_ord, Φ_async,
  Φ_stall, Φ_B, Φ_R, deadlock query).
- `include/triton/Dialect/TritonGPU/Transforms/Passes.td` — Register
  `TritonBarrierDeadlockDetection` pass with options.
- `lib/Dialect/TritonGPU/Transforms/BarrierDeadlockDetection.cpp` — Pass
  entry point: instantiate `BarrierDeadlockAnalysis`, run, dump script.
- `lib/Dialect/TritonGPU/Transforms/CMakeLists.txt` — Add source file.

### PR 3: Lit tests

Hand-crafted TTGIR test cases covering correct and buggy scenarios.

### PR 4: Pipeline integration

Wire into TLX compilation pipeline as an optional diagnostic pass.

### PR 5 (future): MLIR SMT dialect migration

Replace Python Z3 script generation with `smt.*` dialect ops and SMT-LIB
export.

## 8. Relationship to `BarrierExecutionOrderAnalysis`

The existing `BarrierExecutionOrderAnalysis` on the `mren/barrier-ana` branch
is a separate analysis with a different purpose: it collects barrier ops,
groups by warp group, and identifies producer-consumer dependencies using a
graph-based approach. It does **not** do loop unrolling, concrete evaluation,
or constraint encoding.

`BarrierDeadlockAnalysis` is a new, independent analysis. It may reuse
some utility code (e.g., `barrierOpKindToString`, value tracing through
block args) but is not built on top of `BarrierExecutionOrderAnalysis`.
The two analyses can coexist.

## 9. Scope Limitations

The initial implementation focuses on the core producer-consumer mbarrier
pattern. The following are out of scope for the initial PRs:

- **Named barriers** (`bar.arrive` / `bar.sync`): different completion
  semantics (thread-count based). Tex doc Section 2.2.6.
- **Multi-CTA barriers**: remote CTA rank, predicated operations.
- **Path sensitivity**: branches in unrolled TTGIR are rare (constexprs
  already specialized). If encountered, operations under unresolvable
  conditions are included unconditionally (over-approximation).
- **`async_dot_wait` ordering** (Φ_async^adw): requires tracking wgmma
  groups. Added in a follow-up.
- **Per-cycle byte tracking**: uses cumulative byte balance instead.
  Sound but may miss some byte-timing-related deadlocks.

## References

- **Authoritative design**: `triton_lint/docs/deadlock_detection/barrier_deadlock_design.tex`
- **Reference implementation**: `triton_lint/analysis/barrier/z3_encoding.py`,
  `triton_lint/analysis/barrier/program_model.py`
- GCatch: Liu et al., "Automatically Detecting and Fixing Concurrency Bugs
  in Go Software Systems", ASPLOS 2021.
