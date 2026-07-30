# Correctness and Performance Evidence

Correctness precedes performance. A fast wrong kernel, a kernel that succeeds
only for one phase cycle, or a kernel that no longer implements the reviewed
mapping is not a result.

## Build and static evidence

Record:

- exact source revision, compiler/toolchain, CUDA driver/runtime, target SM,
  build command, and launch configuration;
- compiler resource report: registers, spills, SMEM, TMEM, barriers, occupancy,
  and any warnings;
- disassembly anchors for expected TMA, TCGEN05, mbarrier, and fence families;
- mapping coverage and source-anchor checks;
- manifest and pipelined-IR hashes embedded in or shipped with the result.

If disassembly changes instruction shape, order, scope, or synchronization,
update the mapping/sync review before testing. Do not tune from FA4/FA3
disassembly or profiler timelines.

## Correctness matrix

Compare against an independent semantic reference across:

- supported dtypes and accumulation modes;
- minimum, typical, and maximum head/tile dimensions;
- one-tile, multi-tile, and non-multiple boundary sizes;
- zero/empty work where the public contract allows it;
- causal/noncausal, masking, bias, scaling, dropout, or other enabled modes;
- batch/head combinations and irregular strides/alignment;
- values stressing overflow, underflow, NaN/Inf policy, and softmax range;
- enough loop iterations to wrap every ring slot and mbarrier phase repeatedly;
- enough CTAs to expose scale-dependent synchronization failures.

Derive tolerances from dtype, accumulation order, and the public numerical
contract. Report maximum/mean error and failing coordinates, not only pass/fail.
Use deterministic seeds and retain failing cases.

FA4/FA3 may be invoked through their black-box public interfaces for output
comparison, but their outputs are not the only oracle and their internals must
remain uninspected. A matching output does not prove matching roles, MMA order,
or barriers.

## Runtime safety and hang testing

Run appropriate NVIDIA tools for:

- out-of-bounds and misaligned memory accesses;
- uninitialized values;
- shared-memory races and synchronization misuse;
- invalid barrier/fence behavior supported by the toolchain.

Use the repository's `compute-sanitizer` skill for commands and limitations.
Use bounded timeouts for hang tests, retain the exact reproducer, and reset a
poisoned GPU according to the repository GPU-recovery workflow. Test multiple
slot wraps and large grids; many phase bugs are invisible in a single CTA.

Tool silence is supporting evidence, not a substitute for the phase audit.

## Schedule-fidelity evidence

After correctness, verify:

- all IR nodes and edges remain covered by the manifests;
- per-group dynamic issue order agrees with the reviewed mapping;
- source and disassembly contain intended B200 primitive shapes;
- compiler resource allocation agrees with the memory plan;
- no debug synchronization, fallback path, or unreviewed baseline helper is
  active;
- prologue, steady state, and epilogue all execute in targeted tests.

Correctness cannot waive a schedule deviation. Label deviations and rerun all
reviews.

## Performance gate

Run performance tests only when the user explicitly requests them. Load the
repository `kernel-perf-testing` and `running-with-buck` skills before choosing
commands; do not guess a Buck target or GPU configuration.

For an approved B200 run, record:

- exact B200 SKU, clocks/power mode, driver, CUDA version, and competing load;
- input corpus, warmups, repetitions, timing mechanism, cache policy, and
  timeout;
- raw samples plus median and dispersion/tail statistics;
- achieved throughput/bandwidth and clearly stated denominator;
- resource/occupancy data for the tested binary;
- all compiler flags and whether denoising was used;
- manifest/source/binary hashes.

Use identical inputs and methodology for black-box FA4/FA3 baselines. Measure
them only through the allowed public harness. Do not use their source, SASS,
internal counters, or timeline to guide the manual kernel.

## Interpreting results

- A speedup or slowdown is a result for this manual implementation, not proof
  or disproof of the Twill scheduling algorithm.
- Do not attribute performance to Twill unless the mapping is faithful and all
  other manual choices are disclosed.
- Report tuning changes as memory, layout, synchronization, or instruction
  selection decisions, with manifest revisions.
- Keep correctness and performance results for historical non-paper emitters
  separate from this implementation.
- Always name the authoring mode:
  `authoring_mode=agent-assisted-manual-cuda`.

The acceptable summary is "agent-assisted manual B200 CUDA C++ lowering of
the hashed Twill IR, compared with FA4/FA3 as black boxes." It is not
"Twill-generated CUDA matched FA4."
