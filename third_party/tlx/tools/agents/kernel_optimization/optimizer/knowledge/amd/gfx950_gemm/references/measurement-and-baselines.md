# gfx950 GEMM Measurement And Baselines

The production workload is the performance contract. Diagnostic benchmarks are
useful only when their relationship to that contract is explicit.

## Keep Four Reference Roles Separate

1. The **correctness oracle** supplies expected values and may be slow.
2. The **incumbent** is the best implementation currently accepted for the
   exact contract and is what every candidate must preserve or improve.
3. An **external comparator** is another measured implementation or design
   point; it may have different preprocessing, layouts, or dispatch behavior.
4. A **performance bound** estimates remaining headroom from useful work,
   traffic, launch cost, occupancy, or instruction issue.

Only the incumbent is required. A new fusion or layout can be optimized against
its simplest correct implementation and analytical bounds when no vendor
library implements the same operation.

Reserve **SOTA** for an implementation selected from the relevant available
alternatives and remeasured under the same contract. Otherwise name the
comparator precisely.

## Freeze The Complete Workload Identity

Record more than M/N/K:

```text
operation, fusion, epilogue, and numerical mode
batch dimensions and input/accumulator/output dtypes
logical shapes and transposes
physical strides, storage offsets, alignment, and broadcast state
shared operands and preprocessing outside the timed region
device/CU count, runtime, library, compiler, and resolved backend options
public provider, dispatcher, selected plan, and workspace behavior
output-allocation ownership
cache, stream, concurrency, warmup, sample, and synchronization regime
correctness oracle, tolerance, and deterministic-order requirements
```

Two runs with different identities are diagnostics, not a regression
comparison. A stride-zero shared operand, a row-major B, or preprocessing
outside the kernel can change the best comparator and kernel family by much
more than the expected optimization.

## Resolve The Real Incumbent Ladder

Measure in this order:

1. current public/internal dispatch for the exact workload;
2. the named framework or library path; and
3. a pinned exhaustive external winner, when available.

A public fallback may already select a stronger family than the kernel named in
the task. Preserve a callable for the incumbent before editing dispatch, then
verify after every edit that incumbent and candidate still resolve to distinct
implementations. A label such as `public` can silently become a self-comparison.

For an external solution pool, use the scanner only to nominate a winner. Pin
the solution index and full implementation name, then remeasure it beside the
incumbent in the promotion harness. Never divide absolute times from different
harnesses, cache states, or process states.

For hipBLASLt, do an unpruned final search before calling the winner SOTA. Log
the selected solution name in an untimed launch because logging perturbs
latency. Normalize its column-major problem representation back to the public
API before interpreting macro-tile M/N, MI wave tile, wave group, mapping, or
XCD fields. In transposed representations, library A/B and M/N roles can be
reversed.

A pinned solution is versioned evidence. Rescan after runtime/library changes,
or when the current framework heuristic unexpectedly beats it beyond noise.

## Choose The Timing Regime Deliberately

- **Production timing** is the promotion gate, whether hot, cold, concurrent,
  or end-to-end.
- **Cold-cache timing** measures compulsory traffic. Flush with a
  read-modify-write over storage larger than the relevant caches; a write-only
  fill can bypass cache and leave the previous working set warm.
- **Hot repeated-launch timing** measures reuse and launch amortization and can
  reverse locality-sensitive rankings.
- **Interleaved same-process timing** rotates contestants to reduce clock,
  thermal, and temporal bias.
- **Kernel timestamps** isolate device work only when framework/dispatch/
  allocation overhead is not part of the contract.
- **End-to-end timing** is required when allocation, preprocessing, fusion,
  graph execution, or dispatch belongs to the API contract.

Preallocate equivalent outputs and persistent workspaces unless allocation is
explicitly measured. Audit the actual callable passed to the timer: creating an
output variable does not help if one lambda still calls with `out=None`.

Warm compilation and one-time initialization. Retain raw samples rather than
only medians. Rotate order and use an odd sample count large enough to expose
variance. Repeat independent processes for close results, multiple clock
plateaus, or address-sensitive behavior. Keep the incumbent when equivalent
runs change sign.

Long corpus runs need absolute-latency sentinels before and after the sequence.
If a sentinel leaves its established band, use the sequence only to nominate
cases for isolated remeasurement. A stable ratio cannot validate a thermally or
cache-contaminated run.

The set and order of contestants can change cache residency. Do not mix an
absolute time from a two-way run with one from a three-way run.

## Validate Correctness Before Timing

Exercise multiple seeds, partial M/N tiles, every advertised K remainder,
nonzero offsets and alignment boundaries, short trip counts, deterministic
repeated launches for race-sensitive pipelines, and dynamic-range/cancellation
cases when reduction order changes.

Use bit equality when the numerical order is part of the contract. Otherwise
justify tolerances against a higher-precision oracle and record observed error.
Do not relabel missing K blocks, unsafe vectorization, uninitialized storage, or
a race as harmless rounding drift.

## Declare The Acceptance Rule Up Front

State whether success means every-case improvement, zero regressions plus a
geometric-mean improvement, a weighted production score, or parity after a
refactor. Define the noise margin and minimum per-case ratio before measuring.
Always report aggregate ratio together with win/parity/regression counts and
the minimum ratio; a geometric-mean win is not an every-shape win.

For every run retain the complete case identity, source/compiler revision,
selected dispatch and plan, correctness, raw baseline and candidate samples,
profile/artifact summaries, and ratio direction. Exact campaign numbers belong
in run artifacts, not in reusable dispatch instructions.
