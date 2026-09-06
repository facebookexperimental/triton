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

Without a reference, use source inspection and profiles to generate hypotheses.
Prefer changes supported by counters or launch attribution over broad rewrites.

## Preserve Correctness

Keep public entry points and the full input/output ABI. Preserve boundary masks,
aliasing, mutation, numerical tolerance, deterministic initialization, and
repeated-launch behavior. For asynchronous or warp-specialized code, prove
producer/consumer ownership, buffer lifetime, barrier phase, arrival count, and
store completion before changing scheduling or storage.

Reject changes that fail any protected case, increase measurement variance
beyond the configured threshold, or improve only an instrumented profile while
regressing the uninstrumented end-to-end benchmark.
