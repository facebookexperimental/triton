# Automatic wave specialization

Status: initial implementation. The reusable MLIR uniformity analysis is in
`include/triton/Analysis/Uniformity.h` and `lib/Analysis/Uniformity.cpp`. The
AMD pass is in
`third_party/amd/lib/TritonAMDGPUTransforms/AutomaticWaveSpecialization.cpp`,
and `HIPOptions.enable_wave_specialization` gates the pipeline. The implementation
selects one invariant predicate (including equivalent repeated definitions),
extracts its slice to entry, and creates two versions.

## Objective

Automatically specialize most of a Triton/TLX kernel body on runtime predicates
that are uniform within each hardware wave. Different waves may choose different
versions. Within a version, the selected predicates become constants, exposing
constant folding, mask elimination, dead computation elimination, and simpler
address calculations before LLVM optimization and machine-code generation.

The transformation uses a small dispatch tree outside the original loops and
clones the remaining kernel body into its leaves. It extracts an ordered common
prefix to make dispatch possible. It does not create small specialized regions
throughout the kernel and does not introduce specialization dispatch inside loop
bodies. Amortizing control-flow overhead over the body is a fundamental design
constraint.

Run the transformation on lowered, per-thread MLIR after TTGIR conversion,
software pipelining, and shared-memory allocation. The input is predominantly
LLVM/ROCDL operations and explicit CFGs, not distributed tensor operations.

Cross-wave communication remains supported inside the versions. On AMDGPU,
different waves may rendezvous through different static instances of a barrier
instruction. Cloning the complete remaining CFG and substituting a predicate
with its actual value for the executing wave preserves the original paths and
their synchronization. No separate synchronization proof, event correspondence
analysis, or barrier-sequence validator is required. Constant folding may remove
unreachable paths, including their barriers.

This transformation is distinct from Triton's existing automatic warp
specialization, which partitions producer/consumer work into warp groups. The
proposed pass performs runtime partial evaluation of the kernel body.

## Kernel option

The AMD kernel option is disabled by default:

```python
enable_wave_specialization: bool = False
```

Example launch:

```python
kernel[grid](..., enable_wave_specialization=True)
```

The option is a compiler option, not a kernel argument or a language-level
`constexpr`. It applies to Triton/TLX kernels compiled through the AMD backend.

- With the option disabled, do not run the analysis or transformation pipeline.
- With it enabled, attempt specialization of eligible predicates. A kernel with
  no eligible or useful predicate retains its original body.
- Include the option in compilation cache identity and allow it in autotuning
  configurations through the normal backend-option mechanism.
- Do not use an environment variable as the enablement mechanism.
- Initially expose only the boolean option. Version and code-growth budgets are
  internal pass parameters, with test/debug overrides if needed.

The field is in `HIPOptions` in `third_party/amd/backend/compiler.py`. The
existing option parser collects dataclass fields, and `HIPOptions.hash()` hashes
the option values. Tests must verify both parsing and cache separation rather
than relying only on that implementation detail.

Integration uses the common AMD `make_llir` pipeline for Triton/TLX and Gluon.
Compile tests cover both frontends. The option has no integration in other
backends. Autotuning can pass it through
`triton.Config({"enable_wave_specialization": True})`.

## Transformation shape

Given a scalar wave-uniform predicate `p`:

```text
original body:
    operations needed to compute p
    other operations
    original loops
    epilogue
    return
```

Produce:

```text
ordered common prefix
p = ...
if p:
    remaining body with p = true
else:
    remaining body with p = false
return
```

The common prefix need not have been a contiguous prefix of the original block.
It is an order-preserving extraction of operations whose movement is legal. All
remaining operations, including the original loops, are cloned in their original
relative order into both versions.

The `if` above describes execution structure. Implement it as an LLVM-dialect
conditional branch to cloned CFGs, using `IRMapping` to remap values, blocks,
successor operands, and loop backedges. Create all destination blocks before
cloning operations so forward edges and cycles can be remapped. Keep a common
return block and pass any live-outs through block arguments. Preserve the original
exit behavior when redirecting returns.

Values defined in the extracted prefix dominate both versions. Substitute
selected predicates only in the cloned bodies; the dispatch uses their original
runtime values. Before dispatch, freeze a zero-extended predicate and broadcast
the chosen bit with `readfirstlane`. This keeps dispatch defined and uniform even
when a speculated computation produces poison on a path that originally did not
use it. Defined uniform predicates retain their original value.
Every retained CFG edge targets the corresponding block in its
own version, except edges to the common exit. The pass does not require retaining
SCF through TTGIR lowering or reconstructing `scf.if` around existing CFGs.

Multiple predicates form a bounded decision tree. For the initial implementation,
select one predicate and create at most two versions. A subsequent extension may
select several predicates available from the same prefix, with at most four
leaves as an initial experimental budget. Both leaf count and estimated code
growth need limits because leaf count alone does not bound duplication cost.

The pass currently targets defined external LLVM functions returning void,
matching the AMD lowering convention for kernel entries. Internal device helpers
are skipped; their arguments do not have the uniform kernel ABI. Supported bodies
are region-free CFGs with ordinary LLVM branches, switches, returns, and
unreachable terminators. Every cloned return branches to one common void return.
The function attribute `tritonamdgpu.wave_specialized` prevents a repeated pass
invocation from multiplying versions again.

## Uniformity analysis

Analyze per-thread integer, boolean, and other modeled SSA values in common
middle-level MLIR dialects, including `arith`, `math`, `index`, `vector`, `gpu`,
`scf`, and `cf`. The same analysis also accepts LLVM/ROCDL MLIR operations; it does
not require translation to LLVM IR. Select only scalar `i1` values as
specialization predicates initially. Restricting the analysis itself to booleans
would miss predicates derived from wave ID or uniform address/index arithmetic.

The implemented domain distinguishes subgroup-uniform values from values that
may vary:

```text
Uninitialized -> Uniform (= ThreadId<1>) -> ThreadId<2> -> ThreadId<4> -> ... -> Unknown
```

MLIR's dataflow solver computes the fixed point. Uniform seeds initialize value
facts, and transfer functions monotonically join incoming facts and propagate
unknown values and control decisions. Queries publish results only after
convergence. Uninitialized, unknown, unsupported, and unreachable values never
count as proofs of uniformity. CTA uniformity is stronger than the property
needed here and is not separately represented. Constant values remain distinct
from uniformity.

Initial sources and transfer rules:

| Source or operation | Rule |
| --- | --- |
| Numeric constants and modeled workgroup/program ID | Uniform |
| Function arguments | Unknown unless the caller explicitly classifies them as uniform, e.g. kernel parameters |
| AMD wave-ID intrinsics | Wave-uniform |
| Linear thread ID and lane ID | `ThreadId<subgroupSize>` when the required target assumptions are supplied |
| Constant unsigned power-of-two division, logical right shift, AND mask | Reduce a `ThreadId` low-bit bound; `ThreadId<1>` becomes uniform |
| Other deterministic scalar arithmetic and comparisons | Uniform for uniform operands; any `ThreadId` operand becomes unknown |
| Select | Account for the condition and alternatives; equal alternatives may improve precision |
| Unknown operations or calls | Unknown unless an operation summary supplies a proof |
| Loads and atomics | Conservative initially; a uniform pointer alone does not prove uniform results for arbitrary memory semantics |
| Shuffles and other communication intrinsics | Use explicit intrinsic semantics; otherwise unknown |
| Inline assembly | Unknown unless a modeled form supplies a proof |

`ThreadId<N>` means only the low `log2(N)` bits may vary within a subgroup;
all higher bits agree. `N` is a power of two, not the number of distinct lane
values. Joining two bounds takes their maximum, treating uniform as `N = 1`.
For a wave32, `tid / 4` yields `ThreadId<8>` and a subsequent `>> 3` becomes
uniform. AND retains the bound implied by the highest remaining varying bit:
`tid & 7` yields `ThreadId<8>`, while `tid & 16` remains `ThreadId<32>`.
Clearing all varying bits produces uniformity immediately. Both operand orders
of a constant AND are supported. Unsigned division by non-powers of two and
other computations on `ThreadId` values become unknown. Arithmetic on values
that have already become uniform follows the ordinary uniform rules.

These transfers inspect their own operands' dataflow states, without looking
through defining operations. They apply to scalar `arith`, `index`, and LLVM
operations and compose through CFG and SCF argument/result forwarding. Divergent
control loses the bound unless the incoming value is identical on all edges;
loop induction arithmetic also falls back to unknown for `ThreadId` operands.

The enclosing module's `ttg.threads-per-warp` attribute supplies the seed bound;
missing, zero, or non-power-of-two sizes disable these seeds. There is no
separate subgroup-size analysis or pass option. `gpu.lane_id` only needs that
size. `gpu.thread_id x` and
`rocdl.workitem.id.x` additionally require `Options.linearThreadIdX`, which
asserts subgroup-aligned linear X coordinates, as in Triton's one-dimensional
AMD workgroups. Y/Z IDs are not seeded. The specialization pass supplies this
linear-X assumption and contains no thread-ID expression matcher. The compiler
already records the target wave size in the IR before lowering.

`UniformityAnalysis` takes a function-like operation and optional argument and
operation inference callbacks. `isUniform(value)` queries agreement at the
original definition; `isUniform(value, user)` and `isUniform(operand)` additionally
conservatively account for temporal divergence at a use. Callbacks receive the
original operations and arguments. Recompute after modifying the IR.

The basic implementation uses MLIR's `SparseForwardDataFlowAnalysis` directly on
the original IR, alongside `DeadCodeAnalysis` and `SparseConstantPropagation`.
The solver owns reachability, dependency scheduling, and fixed-point iteration.
There is no private IR copy, SCF normalization, observation-marker scheme,
postdominance calculation, or cycle analysis.

- Use a monotone value lattice: uninitialized, uniform, bounded `ThreadId`,
  unknown. Only uniform is a proof of agreement; a `ThreadId` control decision
  is potentially divergent. Explicit seeds and operation models establish facts.
- Use `RegionBranchOpInterface` and the region-successor transfer hook to join
  incoming operands for structured control flow. Include the selector for
  `scf.if` and `scf.index_switch`. Loop region arguments model recurrences;
  `scf.for` induction variables depend on the lower bound and step. Loop results
  additionally depend on the exit decision.
- Use `BranchOpInterface` and MLIR's executable-edge states for CFG block
  arguments, including duplicate successor edges. Different uniform incoming
  values can only establish a uniform join when the controlling decisions are
  uniform. Identical incoming values can retain uniformity.
- Initially, conservatively consider all live branch decisions in the CFG
  region when checking a join. This catches ordinary diamonds whose immediate
  predecessors end in unconditional branches, without custom control-dependence
  machinery. It also loses precision at unrelated joins after reconvergence.
- Conservatively check cross-block uses of non-entry definitions against the
  region's control decisions. This handles potentially different loop iterations
  without discovering cycles, but can reject uniform uses in acyclic CFGs.
  Literal constants and values defined at region entry are exempt.

The implementation is intra-procedural. Unknown calls, memory reads, concurrent
regions, and unmodeled operations stay conservative; operation callbacks can add
semantic rules. In particular, `Pure` and `ConstantLike` alone do not establish
uniformity: thread IDs, undef/poison, and thread-local addresses need distinct
handling. Distributed tensor values require a separate layout-aware analysis.
No LLVM IR uniformity pass, LLVM target machine, or AxisInfo is required.

Structured control flow retains more precision than an equivalent lowered CFG.
In particular, a varying-trip `scf.for` can have a uniform induction variable
within an iteration and an unknown result; a CFG recurrence in a region with
varying control is conservatively unknown even within the loop. More precise
reconvergence and temporal reasoning can be added if specialization needs it.
Inference for additional arithmetic identities, memory operations, and
target-specific idioms can also improve precision. The test pass
`test-uniformity` checks result/use expectations and verifies that the analysis
does not change the input function.

### Control flow and temporal invariance

Uniform incoming values do not imply a uniform merge. For example, selecting
between zero and one under a lane-varying condition produces a varying result.
Control-flow transfer rules for CFG block arguments must include which
participants select each incoming edge. Generic sparse SSA joins alone are
insufficient.

Loop-carried values require a fixed-point analysis accounting for loop control.
Wave uniformity describes agreement across lanes at an execution point; it does
not imply that a value is invariant across iterations.

Candidate predicates must be available at entry dispatch after legal extraction.
An iteration-dependent predicate is ineligible. A predicate invariant within a
loop is eligible only if its computation can legally move outside all enclosing
original loops to the entry dispatch. The pass never falls back to dispatching
inside the loop. It also does not initially synthesize entry guards that imply
properties of every iteration.

### Lowered tensor predicates

Original tensor predicates may already have become several scalar per-thread
`i1` values. Each such value is independently eligible if it is wave-uniform and
available at entry. Selecting one does not imply that other register components
have the same value. Values still packed in vectors or aggregates need explicit
component handling or remain outside the initial candidate set.

This avoids requiring a proof that one boolean describes an entire distributed
tensor. Grouping related register predicates into a compact dispatch is a later
profitability improvement subject to the same version budget.

### Integer ranges and predicate canonicalization

Before choosing a predicate, run MLIR's `dataflow::IntegerRangeAnalysis` with
`DeadCodeAnalysis` and sparse constant propagation. Range inference is independent
of uniformity: a value can have a small range and still vary between lanes.

The common analysis library registers external `InferIntRangeInterface` models
for scalar LLVM integer arithmetic, comparisons, casts, and selects, using
MLIR's existing inference helpers. ROCDL ID models consume their `range`
attributes, including wrapped ranges. `readfirstlane` preserves the global
operand range; `freeze` remains conservative because it can turn poison into
an arbitrary value. Existing GPU, arith, and index interfaces remain unchanged.

The AMD specialization pass supplies missing ID bounds from the kernel's
`ttg.num-warps` and `ttg.threads-per-warp` metadata. When explicit warp
specialization allocated additional waves, `ttg.total-num-warps` supplies the
physical bound. Existing range attributes are preserved. Missing launch
metadata does not justify inventing a bound.

Use these ranges to fold constant comparisons and normalize bit tests before
CSE and candidate grouping. For example, when `w` is in `[0,3]`:

```text
((3 >> w) & 1) != 0  -> (w & 2) == 0
((w ^ 2) & 2) == 0  -> !((w & 2) == 0)
```

The lookup rewrite requires both an in-width shift and the tighter bound that
makes the predicates equivalent. In an eight-wave kernel the first rewrite
does not apply. These are algebraic rewrites backed by standard interval
analysis, without per-wave value enumeration or custom control-flow propagation.

Group equivalent predicates and their boolean complements into the same plan.
Each clone substitutes the appropriate constant for every grouped definition,
including body uses of definitions in the extracted prefix. Existing cleanup removes the
resulting dead descriptor computations without additional body versions.

This does not add branch-sensitive ranges for dominating SSA values. For
example, eliminating `w == 0` specifically inside the false branch of
`(w & 2) == 0` remains a separate implication-folding improvement.

## Ordered common-prefix extraction

For a candidate set, compute the backward dependency closure needed to evaluate
its predicates. Include required common setup and ordering predecessors.
Classify operations as extracted or retained, then emit each class in original
relative order. Do not independently topologically sort extracted operations when
the source order already provides a valid schedule.

Example:

```text
original:    common A; specialized B; common C; specialized D; loop
prefix:     common A; common C
each body:  specialized B'; specialized D'; loop'
```

This partition is valid only if moving `C` across `B` is legal.

Extraction rules:

1. Extract dependencies required to evaluate the selected predicates and any
   required common entry setup.
2. Preserve SSA dominance and account for memory, synchronization, control, and
   other ordering dependencies across the extraction boundary.
3. Extract additional ordering predecessors only when they are legal to extract.
4. Retain operations whose values or execution depend on selected specialization
   assumptions. The predicate-producing slice itself necessarily executes before
   those assumptions are established and is the dispatch prerequisite.
5. If an ordering or data dependency requires extracting a specialized operation,
   reject that candidate set. Do not move it to the prefix and erase the intended
   specialization opportunity.
6. Keep the prefix minimal subject to these requirements. Independence from a
   selected predicate alone is not a reason to extract an operation.

Use dominance, memory-effect/alias information, speculation legality, and explicit
synchronization semantics to justify movement. Loads cannot move across possibly
aliasing writes just because their addresses are uniform. Speculation must not
introduce executions that were absent on the original path.

Discover predicates throughout the CFG and clone the complete remaining CFG,
including all loop blocks. For predicates outside entry, require every non-entry
dependency to be movable, with all dependency leaves being entry arguments or
operations in the slice. Reject non-entry block arguments, including induction
variables and loop-carried state. This admits loop-invariant fused-TDM selectors
without extracting an original loop. Order the slice along its dominance chain,
preserving operation order within each source block even when blocks are printed
in a different order.

Plan extraction and cloning before mutating the kernel. Failure to construct a
legal plan leaves the original body intact.

The initial planner uses an intentionally conservative movement rule. A
dependency that is already in the contiguous initial slice stays in place.
Moving a dependency across retained work requires a pure, speculatable,
deterministic per-thread computation or a modeled invariant hardware ID read.
Calls, inline assembly, and convergent broadcasts are not moved across retained
work. No alias analysis or extraction of additional ordering predecessors is
implemented yet; a candidate needing those capabilities is rejected.

## Communication and synchronization

Cross-wave communication is part of the supported transformation, including LDS
exchange and barriers inside the cloned loops. Do not reject a body merely
because it contains communication, or split it into small regions around each
barrier.

### Full-body cloning preserves existing synchronization

Assume the original lowered kernel is valid. For each wave, dispatch chooses the
version whose substituted predicate equals its original runtime value. Cloning
the complete remaining CFG preserves the paths that wave would have executed,
including memory operations, barriers, waits, and asynchronous operations. Preserve
operation attributes and operands as part of ordinary cloning.

The pass does not compare barrier counts between versions, attach correspondence
IDs to synchronization events, or require a separate proof of synchronization
compatibility. It does not pin barriers that become unreachable under a valid
branch assumption. Different static barrier instruction instances are allowed
under the AMDGPU execution model targeted by this pass.

For example, consider an originally valid kernel containing:

```text
if p:
    work_A
    barrier_A
else:
    work_B
    barrier_B
```

The `p=true` version retains the first path, and the `p=false` version retains
the second. Folding away the other path also removes its barrier. This is valid:
each wave retains the synchronization it would actually have executed. The same
reasoning applies to conditional paths inside cloned loops.

An unconditional barrier after this conditional remains reachable in both
versions and is retained by ordinary side-effect-aware simplification. No special
rule requiring the same textual barrier sequence in every version is needed.

Prefix extraction still obeys ordinary motion legality. Full-body cloning does
not authorize moving a memory operation across a barrier when extracting the
prefix; that is an extraction issue, not an additional cloning proof.

### Shared resources are already lowered

Run after shared-memory allocation and synchronization lowering. Preserve the
existing shared-memory globals/base addresses, offsets, barrier identities, and
lowered asynchronous state. Clone executable blocks, not module-level resource
definitions. A cloned address calculation refers to the same physical storage
as the original calculation; do not rerun TTGIR allocation separately on versions.

This removes the need for allocation correspondence metadata or a special
allocator for simultaneously executing versions. Existing resource initialization
is cloned with its original control flow or remains in the legal common prefix.

### Ordinary per-thread constant semantics

Late lowering makes data exchange explicit. For example:

```text
value = select(p, a, b)
store shared[address], value
barrier
result = load shared[other_address]
```

Substituting `p=true` simplifies the store value to `a`. It does not make the
subsequent load return `a`: the load can observe data from a wave executing the
other version, according to the existing memory semantics.

There are no distributed tensor constants or implicit cross-wave layout
conversions at this point. Use ordinary per-thread constants and existing
side-effect/convergence-aware simplification. No new specialization-scope tensor
representation or specialized tensor folding rules are required.

TTGIR layout selection, software pipelining, synchronization insertion, and shared
allocation have already run once on the original kernel. Versions inherit their
lowered results. Subsequent LLVM optimization and machine scheduling operate on
the specialized code under the existing operation semantics.

## Candidate selection and simplification

Prefer entry-available predicates with substantial uses in masks, selects,
conditionals, addressing, or repeated work inside the original loops. Ignore
predicates already known constant. A large extracted prefix or little remaining
specializable work lowers a candidate's priority.

Initial selection can use deterministic use-based ranking and hard budgets.
More expensive estimates can later simplify temporary clones and count removed
operations. No fine-grained region-placement heuristic is required.

The current ranking assigns one point per entry-block use and four per use in
other blocks, doubling the weight of select and conditional-branch conditions.
Vector selects are additionally weighted by their element count. Structurally
equivalent invariant predicates are grouped, ignoring locations, and their use
scores are combined. Extract only a representative slice and substitute all
equivalent results in each body. This handles the repeated A/B descriptor
selectors emitted by fused TDM loads without extracting descriptor computations.

Candidates need at least two points. Ties prefer a smaller prefix, then source
order. The `max-extra-ops` pass parameter defaults to 65536 and bounds the estimated
extra body plus dispatch operations before mutation; it is not a machine-code
size or register-pressure model. The original 4096-operation limit rejected
gfx1250 grouped GEMM, whose 256x256 lowered body contains roughly 29,400 operations.

After a split, use constant propagation, CFG simplification, and dead-code
elimination under the branch assumptions. These may remove unreachable blocks
containing synchronization as well as ordinary computation. Reevaluate related
predicates so implication and constant propagation can avoid redundant leaves.
Do not repeatedly specialize newly cloned predicates without a shared kernel-wide
budget.

Measure both final code growth and runtime effects. Separate body versions do not
receive independent register-allocation budgets by assumption. Code size,
instruction-cache pressure, increased liveness, and prefix cost can offset the
removed work.

## Pipeline integration

Implemented components:

| Component | Responsibility |
| --- | --- |
| `UniformityAnalysis` | Prove subgroup uniformity on common MLIR values, including control and temporal dependencies |
| Entry specialization planner | Choose predicates and plan ordered prefix extraction |
| Body versioning transform | Clone complete CFGs and introduce entry dispatch and branch-local constants |
| Cleanup | Fold constants, simplify CFGs, and eliminate dead code using existing operation semantics |

Keep the generic dataflow and CFG machinery reusable. Put AMD intrinsic models and
initial backend integration in the AMD implementation. Pass name:
`tritonamdgpu-automatic-wave-specialization`.

Run near the end of the MLIR portion of `HIPBackend.make_llir`, after TTGIR and
warp-specialization lowering, residual CF/arith conversion, and cleanup, before
debug-scope generation and LLVM translation. The current pipeline lowers SCF to
CFG before TTGIR conversion; it does not provide a substantial common-dialect
`scf`/`arith`/`vector` stage with all distributed operations already lowered.

The insertion point must expose LLVM/ROCDL operations and supported residual
operations with per-thread semantics. Reconcile materialization casts or finish
any relevant helper lowering before analysis if they obscure candidate values.
Do not move the transformation back to TTGIR merely to retain structured loops.

```text
TTGIR layout optimization and software pipelining
SCF to CFG
shared-memory allocation
TTGIR and warp-specialization/predicate lowering
remaining common-dialect conversion and cleanup
if enable_wave_specialization:
    builtin helper lowering and cast reconciliation
    common-dialect MLIR uniformity analysis
    ordered prefix extraction + whole-body CFG versioning
    LLVM select/boolean/branch cleanup
    constant propagation / CFG simplification / dead-code elimination
debug information and remaining LLVM translation preparation
MLIR to LLVM IR
LLVM optimization and AMDGPU code generation
```

Recompute or invalidate analysis when cloning and simplification change values or
control flow. Run specialization once, with one shared budget for all decisions.
The kernel option remains the sole enablement gate. The LLVM-specific cleanup
extends the existing `canonicalize-llvm-ir` pass with constant boolean operations
and conditional branches; its existing select folding handles the other main
specialization use. Greedy region simplification removes unreachable blocks,
including barriers on paths excluded by specialization.

Existing `ttg.warp_predicate` and `ttg.warp_specialize` operations have already
lowered. Their high-level region restrictions are not the legality policy for
this pass. Clone their resulting control flow as part of the complete body and
analyze their lowered values using the same entry-availability rules.

### Placement tradeoffs

Late placement makes communication explicit and inherits allocated resources and
an already constructed software pipeline. It avoids specialization-aware tensor
folding and coordination of separate TTGIR pipelines for different versions.

The cost is larger IR and more target-specific uniformity rules, including models
for intrinsics and selected inline assembly. Some high-level facts may be harder
to recover. Versions also inherit the same layout and software-pipeline choices;
this pass does not optimize those choices independently per version. LLVM can
still simplify and schedule each specialized path. Preserve selected earlier
analysis facts only if measurements justify the additional plumbing.

## Diagnostics and failure behavior

Provide pass remarks or debug output for selected predicates, extraction size,
version count, and estimated growth. Rejection reasons should distinguish lack
of uniformity, entry availability, illegal extraction, unsupported lowered
operations/CFG forms, and code-growth budget. The presence of synchronization or
different textual barrier sequences after folding is not a rejection reason.

Enabling the option requests an optimization, not a promise that every kernel
specializes. Rejecting a candidate is a normal outcome. A later violation of an
accepted plan is a compiler error unless the implementation retains enough state
to recover the original unspecialized body; never silently emit invalid code.

## Validation

Add focused cases to existing suitable lit and pytest files. New test files
require separate agreement under the repository's testing conventions.

Analysis and transformation coverage:

- CTA-uniform and wave-uniform arithmetic/comparison chains, including wave-ID
  intrinsics, lowered idioms, and configured wave size.
- Negative cases for lane-varying conditions, divergent merges, unknown memory
  results, and iteration-dependent predicates.
- A body containing loops cloned behind one entry dispatch, with no new dispatch
  inside any original loop; verify remapped block arguments, backedges, and exits.
- Ordered noncontiguous prefix extraction, preserved retained-operation order,
  and rejection when ordering would require extracting specialized work.
- Branch-local constant replacement and useful predicate/select/dead-work
  elimination in lowered IR.
- Version budgets, correlated predicates, and a no-op when no candidate qualifies.

Communication coverage must be part of the first useful implementation:

- Waves taking opposite versions exchange data through the same LDS allocation
  and rendezvous through corresponding barrier instruction instances.
- Multiple barriers inside a repeated loop preserve their dynamic sequence.
- Unreachable conditional paths and their barriers disappear after specialization,
  while reachable unconditional barriers remain. Use an originally valid kernel
  with waves taking opposite paths through different static barrier instances.
- Cross-wave redistribution of data derived from the selected predicate produces
  the original values, including values from the opposite version.
- Already allocated shared addresses, lowered pipeline loops, waits, and
  asynchronous state retain their original behavior after complete CFG cloning.

Inspect generated LLVM IR and assembly to check entry dispatch, barrier sequences,
and eliminated computation. Run GPU comparisons with different waves choosing
different versions; tests where every wave chooses the same leaf are insufficient.
Use bounded execution for synchronization regressions so failures do not hang the
test suite indefinitely.

Verify option-off behavior, option-on behavior, and cache separation. Benchmark
representative kernels with the option off/on, recording runtime, code size,
register usage, LDS usage, and compilation time.

For native/compiler implementation changes, rebuild with `make` before tests.
Run lit from the build directory and pytest with `-s --tb=short`, as required by
the repository instructions.

Current regression coverage is in:

- `test/Conversion/amd/warp_id_to_llvm.mlir`: raw CFG cloning, loop backedges,
  ordered extraction, repeated-pass and growth limits, wave32/wave64 inference,
  negative cases, boolean simplification, and removal of unreachable barriers.
- `python/test/unit/language/test_tlx_codegen.py`: option parsing and hash
  separation, autotuning kwargs, and disabled/enabled pipeline checks for TLX
  and Gluon with allocated shared memory.
- `python/test/unit/language/test_tlx_amd_hip.py`: bounded native execution
  comparing enabled/disabled kernels with alternating waves exchanging values
  through LDS over 1, 2, and 7 iterations. Both versions retain the original LDS
  allocation size. This test has passed on an MI350X (gfx950).
- `python/test/unit/language/test_tlx_amd_gfx1250.py`: packed/list grouped GEMM
  compile regressions verifying that specialization removes fused-TDM vector
  descriptor selects and scalar descriptor membership tests from the bodies,
  and preserves shared-memory allocation.

For gfx1250 grouped GEMM, the selector `((3 >> wave_id) & 1) != 0` divides the
waves between the A and B descriptors. Packed clustered and ragged list kernels
have passed off/on correctness comparisons on the gfx1250 functional simulator
with bitwise-identical outputs. This is simulator validation, not a hardware
performance result; the larger body versions increase code size.

These checks establish functional behavior for the covered cases. Representative
performance, register-pressure, and code-size characterization remains future
work; the option stays disabled by default.

## Implementation stages

1. Implemented: common-dialect MLIR uniformity analysis, initial AMD models,
   regression coverage, and the disabled-by-default kernel option.
2. Implemented: one-predicate, two-version whole-body specialization with ordered
   prefix extraction, complete CFG cloning, and no dispatch inside original loops.
   Cross-wave communication is supported without a separate synchronization
   analysis.
3. Implemented: late MLIR integration and constant/CFG cleanup, with compile
   and native communication tests. Broader performance evaluation remains.
4. Future: extend to multiple predicates with bounded whole-body versions and improve
   candidate discovery/ranking for predicates originating from tensor components.

Keep the kernel option disabled by default throughout initial development.
