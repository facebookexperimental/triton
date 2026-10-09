# gfx950 GEMM Shape Families And Promotion

Use this reference when one implementation or dispatcher must cover multiple
GEMM workloads. Do not turn an isolated winning shape into a default rule.

## Separate Three Kinds Of State

- The **geometry ranker** proposes promising tiles and wave grids from useful
  work, padding, CU rounds, resources, and measured soft priors.
- The **capability catalog** states which parameter combinations each shared
  kernel family can compile and execute correctly.
- The **promotion policy** contains measured plans and bounded predicates safe
  for default dispatch.

A high-ranked unsupported geometry exposes a capability gap; filtering it out
does not make the geometry poor. Successful compilation establishes neither
correctness nor suitability for default dispatch.

Represent a plan with fields for MFMA geometry, wave grid, block K, staging
family, buffers, operand residency, persistence or K partitioning, mapping,
cache policy, compiler options, and launch grid. Shape rules select plans; they
must not clone device implementations.

## Build Mechanism-Based Corpora

Keep targets, one-variable neighbors, and blind holdouts disjoint. Cover short
and deep K, balanced and rectangular outputs, useful/padded-work boundaries,
CU-round transitions, partial tiles, supported K tails, and production layouts.
Name corpora by workload purpose rather than a shape count.

For a claimed irregular-tile capability, record the problem shape, logical
macro tile/MFMA grid, and physical movement backing. Verify the resolved launch
actually executes the irregular MFMA grid. A strong reusable boundary is a
smaller tile that reduces padded work without increasing the tile count, for
example equal ceiling quotients across two tile sizes; test the neighboring
values where the equality changes.

Do not inspect holdout winners while deriving a rule. When evaluating the
optimization workflow itself, include a blind positive control whose incumbent
has independently established headroom above the noise floor.

## Measure The Real Incumbent Ladder

Measure the current public dispatch first, then a named framework or library
path, then an exhaustively selected external winner when available. Pin the
winner and remeasure it beside the incumbent in the promotion harness; do not
divide standalone scanner latency by another harness's latency.

Use the production cache and allocation regime as the promotion gate. Warm all
contestants, preallocate equivalent outputs/workspaces unless allocation is in
the API contract, rotate order in one process, retain raw samples, and repeat
independent processes for close results. Preserve absolute-latency sentinels
around long corpus runs so thermal, clock, allocation, or cache drift cannot be
hidden by a ratio.

Normalize an external library's matrix convention before interpreting its tile
or mapping. A row-major public GEMM may be represented as a transposed
column-major problem, reversing reported M/N roles.

## Derive Bounded Dispatch Rules

Explain a family win with observable features such as useful/padded work, tile
and CU rounds, aspect ratio, K-loop depth, occupancy, or reuse distance. Turn
those features into a bounded predicate and test both endpoints, an interior
point, and negative neighbors across every admitted boundary.

When Stream-K is useful only for a sparse final device wave, compare a hybrid
decomposition: complete waves use a lean data-parallel kernel and only the
remainder uses Stream-K. Admit it only when full-wave work amortizes the extra
launch and the tail is sparse enough; test the adjacent denser-tail case.

For range generalization, compare every admitted point against the previous
dispatch and verify that old and new labels resolve to distinct implementations.
A public label can silently become a self-comparison after a dispatch edit.
Preserve the incumbent as fallback when the minimum ratio is unresolved.

## Preserve Dispatch Equivalence Across Refactors

Before and after a rebase, consolidation, or API migration, capture a dispatch
manifest for every protected case. Compare selected family, complete plan,
launch grid, tile, wave count, block K, pipeline flags, mapping, cache policy,
and compiler options. Identical kernel source does not prove equivalent public
behavior when a broader heuristic can shadow a tuned plan.

Run focused correctness and artifact checks first, then target and neighbor
timing, the full existing regression corpus, and blind holdouts. Report minimum
ratio, geometric mean, win/parity/regression counts, correctness coverage,
compile-time effect, dispatch overhead, and workspace cost. Keep exact campaign
measurements in run artifacts; only promote a decision rule after evidence spans
the boundary claimed by that rule.
