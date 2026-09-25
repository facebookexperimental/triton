---
name: tlx-kernel-tuning
description: Tune or port a Triton TLX kernel against a user-provided correctness and performance benchmark, including GEO-informed fusion design, controlled parameter search, profiling, and evidence-based reporting.
---

# TLX Kernel Tuning

Use this skill when the user provides a kernel benchmark or asks to tune, port,
or optimize a TLX kernel. Treat the supplied benchmark as the authority for the
operation boundary and timing comparison; do not silently replace or optimize
the baseline.

## Establish the contract first

Before changing the kernel, extract and record:

- shapes, strides, layouts, dtypes, output ABI, hardware, and compared paths;
- required intermediate rounding boundaries and accumulator precision;
- whether reduction reassociation is allowed;
- whether approximate transcendental or other algorithmic math is allowed;
- correctness metrics, thresholds, seeds, and any saved outputs or side effects.

Distinguish requirements from observations. An existing result being bit exact
does not make bit exactness a requirement. Floating-point reductions may change
low bits when their order changes unless the contract explicitly forbids it.
Keep exact and approximate math as separately named variants. Never change
accumulator precision silently: report the old and new precision prominently
before treating such a candidate as promotable.

If the benchmark does not state a necessary numerical choice, keep the current
precision boundaries and ask only when the choice would materially change the
accepted result. A tolerance selected during exploration is provisional until
it is justified by the benchmark or confirmed by the user.

## Choose references

When editing TLX, load `tlx-api-reference`. When measuring performance, load
`kernel-perf-testing`. Load `ir-debugging` for IR, register-spill, or lowering
investigations; `compute-sanitizer` for suspected memory or synchronization
errors; `debug-failing-gpu` after GPU-busy or wedged-device failures;
`tma-illegal-instruction` for TMA illegal-instruction failures; and
`autows-authoring` when the comparison includes an AutoWS implementation.

Use GEO as a read-only design library when fbsource is available:

```text
fbcode/gem/next_gen/geo/kernel_library/kernels/
```

Search for the complete semantic pattern, not only an operation name. Compare:

- prologue, epilogue, or combined fusion boundary;
- operand orientation and which operand is reconstructed;
- registered shape and tile divisibility;
- BF16/FP16 rounding points and FP32 accumulation;
- exact versus approximate math;
- TMA/direct loads, SMEM/TMEM buffering, task topology, persistence, and CTA
  clustering.

Do not transplant a GEO configuration solely because the fused graph looks
similar. Record which ideas transfer and which are invalidated by shape,
layout, precision, or ABI differences. Do not modify the GEO checkout unless
the user explicitly requests it.

For RLLayer work, consult `RLLAYER_FUSED_KERNEL_TUNING.md` before selecting a
donor or repeating an experiment. A useful SwiGLU-prologue example is GEO's
`compute/bf16/fused/bwd/mul_mul_sigmoid_grad_cat_matmul` family.

## Tune with measured evidence

1. Reproduce correctness and timing before edits. Record the GPU, software
   environment, warmups, samples, repetitions, and statistic used.
2. Measure the unfused baseline, original fused seed, and a plain GEMM floor
   such as cuBLAS and `tlx.ops.mm` when applicable. This separates GEMM cost
   from fusion overhead.
3. Classify the fusion. Estimate repeated pointwise work across output tiles;
   prologue fusion often loses when an expensive reconstructed operand is
   recomputed for many independent output tiles.
4. Preserve a known-correct variant. Add algorithmic approximations, topology
   changes, or precision alternatives as explicit variants with contract
   metadata.
5. Hill-climb one coupled group at a time: tile shape, reuse factor, pipeline
   depth, task warps/register caps, epilogue subtiling, persistence, then CTA
   topology. Recheck correctness after every candidate.
6. Profile when measurements plateau or a change behaves unexpectedly. At
   minimum inspect registers/thread, dynamic SMEM, resident blocks, occupancy,
   tensor/compute throughput, DRAM throughput, and grid-tail effects. A
   requested register cap is not the same as measured register use.
7. Re-run the winner with the benchmark's full correctness matrix and stable
   performance protocol on an idle device. On NVIDIA, use `denoise.sh` as
   directed by `kernel-perf-testing`.

Do not run performance tests unless the user has asked for tuning or
measurement. If a GPU run hangs for several minutes, use
`third_party/tlx/killgpu.sh` and investigate before retrying.

## Provided benchmark and optimization agent

For a one-off benchmark with a clear CLI, tune it directly. For repeatable or
multi-case searches, adapt the benchmark to the existing harness contract in
`third_party/tlx/tools/agents/kernel_optimization/README.md`:

```python
build(kernel_source, target)
verify(build_artifact, case)
benchmark(build_artifact, case, repetitions)
profile(build_artifact, case)  # optional
```

The Decision Maker owns authoritative correctness, timing, promotion, and run
state; the Optimizer proposes candidates. Put shapes and protected status in
`cases.json`, platform metadata in `target.json`, and keep numerical semantics
in `verify` diagnostics/metrics. Use `--no-commit-winner` unless the user has
explicitly authorized commits. Consult the NVIDIA knowledge under
`third_party/tlx/tools/agents/kernel_optimization/optimizer/knowledge/nvidia/`
for applicable scheduling patterns.

## Report and preserve evidence

Report the winning config and all relevant numerical semantics, including
whether math is approximate and the accumulator dtype. Give latency against
both the acceptance baseline and original fused seed, accuracy ranges across
seeds, profiler constraints, and important rejected experiments. Do not call a
candidate a win when it only improves the fused seed but still loses to the
acceptance baseline.

Update the task or tuning ledger when requested. Commit only when explicitly
authorized, and leave unrelated workspace changes untouched.
