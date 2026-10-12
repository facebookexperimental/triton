# AMD gfx950 GEMM Optimization

Apply this guidance only when `optimize-amd-gfx950-gemm` is selected for a
GEMM-like Triton or TLX kernel on gfx950/CDNA4. It covers GEMM, batched GEMM,
fused matmul, and shared shape-family dispatch. The Decision Maker remains the
authority for correctness, measurement, promotion, and stopping.

The objective is to improve the best verified implementation for the exact
workload contract. An external library is useful evidence when it implements
the same contract, but it is not a design specification and is not required.

## Freeze The Workload Contract

Before proposing a candidate, identify:

- operation, epilogue, batch dimensions, M/N/K, and every dtype;
- transpose, shape, stride, storage offset, alignment, and broadcast state;
- public provider and resolved dispatch path;
- output and workspace ownership;
- device, runtime, compiler revision, and resolved backend options; and
- the production cache, stream, concurrency, correctness, and timing rules.

Do not substitute a convenient dense layout or raw library call for the
production path. Treat compiler options that change MFMA selection, wave policy,
packing, vectorization, or scheduling as part of the candidate and compilation
cache identity.

Keep four roles distinct: the correctness oracle, the accepted incumbent, an
optional external comparator, and an analytical performance bound. Call a
comparator SOTA only after searching the relevant alternatives and remeasuring
the selected winner under the same contract.

## Establish Ground Truth

Verify correctness and benchmark the current public dispatch before editing.
Resolve the selected launcher, complete compile-time plan, launch grid, and
workspace behavior. Confirm in final IR or AMDGCN that the capability under
study is present; an irregular problem dimension handled by a masked regular
tile does not exercise irregular macro-tile decomposition.

Write a headroom verdict containing the incumbent latency distribution, noise
floor, strongest same-contract comparator or analytical bound, gap, and a
mechanism capable of closing that gap. A case with no actionable gap is a
guardrail, not a positive optimization result.

## Diagnose Before Rewriting

Classify the first-order limitation using measured and compiled evidence:

```text
device work distribution and launch cost
useful versus padded tile work
MFMA geometry, register pressure, and occupancy
global-memory traffic and cache locality
LDS traffic, bank behavior, and synchronization
pipeline depth, dependency exposure, and machine scheduling
epilogue, reduction, or framework overhead
```

Calculate useful work, tile count, CU rounds, K-loop trips, operand and output
traffic, reuse scope, waves per workgroup, LDS footprint, VGPR/AGPR use, scratch,
and launch or reduction cost. Do not infer the bottleneck from arithmetic
intensity, one counter, or resemblance to another kernel.

For pipeline or scheduling claims, compare instruction traces for the incumbent
and candidate when available, plus the strongest launchable comparator when it
answers the hypothesis. Inspect resident waves, VMEM, LDS, MFMA, waits,
barriers, and the instructions behind the largest stalls. Normalize traces per
CTA or unit of useful work. Label a diagnosis provisional when only static
evidence is available.

## Run Attributable Experiments

For each candidate state one mechanism, predicted artifact change, predicted
timing effect, correctness risk, and falsifier. Prefer a bounded ladder in this
dependency order:

1. device work distribution and CU-round utilization;
2. orientation, macro tile, and boundary work;
3. MFMA ownership and wave grid;
4. register/LDS/direct-to-LDS staging, block K, and buffering;
5. persistence, Stream-K, LocalSplitU, mapping, and cache policy;
6. source-approved memory/compute scheduling windows; and
7. post-lowering machine scheduling.

Reject before timing when the compiled artifact spills, loses the intended
occupancy, changes to an unintended load or MFMA form, violates a pipeline
invariant, or does not exercise the hypothesis. Measure survivors against the
incumbent in the same process and regime, rotate execution order, retain raw
samples, and repeat independent processes for close or multi-plateau results.

Use the accompanying design-space reference to choose bounded kernel
mechanisms. Use the shape-family reference when changing a shared dispatcher or
promoting a rule beyond one case.

## Knowledge Map

The injected references separate decisions that should not be collapsed into
one recipe:

- **Measurement and baselines** defines the workload identity, incumbent
  ladder, timing regimes, hipBLASLt comparison rules, and stability checks.
- **Experiment workflow** defines the required artifacts, bounded candidate
  ladder, fail-fast gates, staged expansion, experiment record, and valid stop
  conditions.
- **Production kernel architecture** maps the public gfx950 catalog from API
  dispatch through compile-time plan, reusable family, and final artifact.
- **Design space** covers MFMA grids, register/LDS/direct-to-LDS movement,
  PGR2, persistence, and artifact gates.
- **Work distribution** distinguishes ordinary tiling, persistent execution,
  Stream-K, hybrid tail launches, global Split-K, and LocalSplitU.
- **Pipeline and machine scheduling** covers inter-wave versus intra-wave
  overlap, current scheduling-region forms, lowering-aware cover, and final
  wait/issue validation.
- **Irregular tiles and K tails** covers non-power-of-two logical tiles,
  physical backing, boundary lanes, operand-specific alignment, and tail
  peeling.
- **Locality, resources, and cache** covers occupancy, cache scope, shared
  operands, XCD/PID mapping, cache modifiers, and LDS/register tradeoffs.
- **Profiling and artifact analysis** explains which wall-clock, counter, ATT,
  and AMDGCN evidence can support each claim and the common failure modes.
- **Compiler integration contract** requires all lowering-affecting facts and
  options to survive JIT/framework boundaries and compilation caching.
- **Shape families and promotion** separates geometry ranking, executable
  capability, measured dispatch, corpus construction, and regression gates.

Use the sections relevant to the measured mechanism. Do not replay every lever
on every kernel.

## Promote Conservatively

Promote only after focused correctness, boundary and tail coverage, target
timing, one-variable neighbors, the prior regression corpus, and blind holdouts
pass. Report target latency, speedup over the incumbent, sample stability,
minimum per-case ratio, weighted or unweighted geometric mean as declared,
win/parity/regression counts, and compile/dispatch/workspace cost.

An aggregate win never licenses a known production regression unless the
acceptance policy explicitly says so. When evidence is unresolved, keep the
incumbent or fall back outside the candidate's proven contract. Record rejected
mechanisms as evidence, but promote a rule into this knowledge only after its
claimed boundary has been tested.
