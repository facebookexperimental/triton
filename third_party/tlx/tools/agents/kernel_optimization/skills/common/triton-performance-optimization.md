# General Triton Performance Optimization

Apply this guidance to any Triton or TLX candidate. The promotion target is
lower correctness-gated end-to-end latency across every protected case. A
reference kernel is optional: use it as design evidence when supplied, and use
the benchmark plus profiler evidence when it is absent.

## Establish The Bottleneck

Start from measured evidence. Separate wrapper latency, individual kernel
latency, compilation time, and profiler instrumentation overhead. Use Proton or
the platform-native profiler to locate expensive launches and then distinguish:

- launch or host scheduling overhead;
- redundant global-memory traffic and temporary tensors;
- tensor-core, vector, or scalar compute throughput;
- occupancy limits from registers, shared memory, or threads;
- dependency, scoreboard, barrier, or async-copy stalls;
- layout conversions, spills, and cache behavior.

Do not optimize a metric that is outside the measured critical path. If a
profiler is unavailable, make the narrowest hypothesis supported by source
structure and end-to-end timings, and keep the claimed evidence explicit.

For a persistent kernel, calculate the logical tile count before changing the
compiler or synchronization: `ceil(M / BLOCK_M) * ceil(N / BLOCK_N)` (times any
batch or split dimension). Compare it with the launched persistent grid and the
GPU SM count. A tile that exposes substantially fewer CTAs than SMs can leave a
large fraction of the device idle even when its per-CTA compute is efficient.
Sweep tile shapes that produce roughly one or more full waves before attributing
the gap to warp specialization. Record both the logical CTA tile and the
per-compute-group MMA tile when a reference splits one CTA across multiple MMA
groups. Derive the persistent stride and grid from the selected runtime device;
do not reuse a generated `multi_processor_count` captured on another GPU. Pay
special attention to a tiny final wave such as `148 + 148 + 4` blocks, which can
cost an entire additional wave even though only a few blocks remain.

CUDA-event timing around a Python callable is not automatically kernel-only:
when the stream becomes idle after the start event, Python registry lookup,
descriptor construction, allocation, and launch submission can delay the stop
event and appear in the elapsed interval. Compare equivalent call stacks. For a
kernel-throughput claim, pre-bind descriptors and dispatch and time the direct
kernel launch, while retaining the complete wrapper measurement as a separate
end-to-end metric. Do not subtract host overhead heuristically.

## Prefer Generalizable Transformations

Search for generated patterns that a compiler could safely recognize:

- fold pointwise producers, casts, padding, or views into consumer loads;
- fold pointwise or reduction epilogues into a producer when the tile owns all
  required data, or introduce an explicit multi-tile reduction when it does not;
- remove dead launches, allocations, conversions, stores, and reloads after
  fusion;
- preserve tensor-core-friendly input types and layouts when the numerical
  contract permits them;
- select block shapes, warps, stages, persistent grids, and launch ordering from
  shape, dtype, hardware resources, and measured occupancy;
- keep reductions and epilogues in registers, TMEM, or shared memory when their
  complete lifetime and synchronization can be proven;
- specialize invariant shape and stride decisions while retaining guarded
  fallbacks for unsupported inputs.

Express each candidate as one coherent hypothesis. In its evidence metadata,
state both the local source change and the general compiler recognition rule:
the source pattern, legality conditions, expected lowering, and required
fallback. This makes successful hill-climbing results actionable as compiler
optimizations rather than permanent edits to generated code.

## Use Optional Reference Implementations Carefully

When `reference_kernel.py` exists, read the full file. Compare algorithm,
precision, tiling, memory traffic, scheduling, and synchronization with the
candidate. Translate an idea only when input/output semantics and hardware
assumptions match. Never import, call, or paste the reference implementation as
a shortcut; the candidate must remain independently executable.

Validate the reference leg independently before using its latency as a target.
If its default heuristic exceeds target resources or triggers a large cold
autotune, tune the reference once under the same cache policy as the candidate,
then pin one correctness-checked, architecture-legal reference configuration
and reuse it for every candidate. Validate every configuration considered by a
custom sweep; a native autotuner may time candidates without checking their
outputs. Serialize both kernel kwargs and launch metadata such as `num_warps`,
`num_stages`, and cluster dimensions. Record that complete configuration with
the run; do not change it in response to candidate performance.

For an NVIDIA TLX reference that uses TMA and persistent scheduling, first port
and measure those independently useful structures in a plain Triton candidate.
Enable AutoWS only after that candidate is correct and stable. This keeps TMA,
persistence, and warp-specialization effects separately attributable.

Treat TMA placement as multiple independent choices rather than an all-or-none
conversion. Measure operand loads and epilogue loads/stores separately. If
descriptor stores introduce extra staging copies, token waits, or partitions,
retain TMA for the recurring GEMM operands and test a direct pointer epilogue.
Accept the hybrid only when final IR, correctness, and direct timing all confirm
the change; a hand-written reference's preferred TMA strategy need not be the
best lowering produced by the current compiler.

Also keep full-TMA epilogues in the search space when the residual or output is
large. Descriptor residual loads and descriptor stores can outperform direct
pointer I/O even when the GEMM operand strategy is unchanged. Compare the full
TMA and hybrid variants with identical tiles and scheduling instead of assuming
that output staging is always overhead.

Materialized padding is another independent data-movement target. A descriptor
load may replace zero-padding only when its out-of-bounds fill is exactly the
graph's padding value and the descriptor coordinates preserve the original
layout. Likewise, preserve explicit numerical barriers when fusing casts: if
the graph rounds a GEMM to BF16 and promotes it before an FP32 epilogue, perform
that BF16 round in the fused kernel before the epilogue. Passing a loose final
tolerance does not justify reassociating or deleting the intermediate round.

For shallow-K or otherwise small GEMMs, benchmark a plain descriptor/TMA kernel
before requiring persistence or AutoWS. A small tile can expose more independent
CTAs, and a single K iteration may have too little recurring work to amortize a
specialized producer/consumer pipeline. Sweep plain and AutoWS forms over the
same tile space, and keep the plain form when AutoWS is slower; matching the
reference's scheduling structure is evidence to test, not a promotion rule.

Without a reference, use source inspection and profiles to generate hypotheses.
Prefer changes supported by counters or launch attribution over broad rewrites.

## Preserve Correctness

Keep public entry points and the full input/output ABI. Preserve boundary masks,
aliasing, mutation, numerical tolerance, deterministic initialization, and
repeated-launch behavior. For asynchronous or warp-specialized code, prove
producer/consumer ownership, buffer lifetime, barrier phase, arrival count, and
store completion before changing scheduling or storage.

When a TLX reference exists, final validation must include a direct numerical
comparison of every returned Triton and TLX tensor using identical deterministic
inputs. Record shape, dtype, exact-match fraction, maximum and mean absolute
error, and maximum relative error per output. Keep this untimed; transferring a
reference result between isolated TLX and AutoWS processes must not enter either
performance measurement.

Also write an operation-level precision trace for both implementations:

```text
step                 Triton                         TLX
operand load         dtype / conversion             dtype / conversion
dot                  input precision / accumulator  input precision / accumulator
intermediate         cast and rounding boundary     cast and rounding boundary
epilogue arithmetic  operation dtype                operation dtype
store                output dtype                   output dtype
```

Do not infer hidden intermediate equality from final allclose. If the traces
differ or the final comparison is not exact, use diagnostic-only kernel copies
that publish the relevant intermediate values and compare them at each semantic
boundary. Never time or promote an instrumented kernel. For reductions, note
that equal input and accumulator dtypes can still differ because of reduction
order; identify the first divergent boundary before deciding whether the change
is acceptable.

Reject changes that fail any protected case, increase measurement variance
beyond the configured threshold, or improve only an instrumented profile while
regressing the uninstrumented end-to-end benchmark.
