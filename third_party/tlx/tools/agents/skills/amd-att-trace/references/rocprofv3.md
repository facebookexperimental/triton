# rocprofv3 ATT collection

## Inputs that must be recorded

- Complete application argv and working directory.
- GPU visibility selection.
- Kernel include regex.
- Steady-state matching-dispatch number.
- Target CU, shader-engine mask, and SIMD selection.
- rocprofv3 executable and decoder directory.
- Relevant workload controls such as warmup and iteration counts.

`--kernel-iteration-range` applies to dispatch selection. Confirm its numbering
with the adjacent kernel trace rather than assuming that application call
numbers and matching-dispatch numbers are identical.

## Capability checks

A usable collector needs both:

1. `rocprofv3 --help` containing `--att` or `--advanced-thread-trace`.
2. A decoder directory containing `librocprof-trace-decoder.so*` or the legacy
   `libatt_decoder_trace.so`.

Do not combine an arbitrary profiler with an incompatible runtime merely
because both exist on the machine. Prefer the profiler shipped with the ROCm
bundle used by the target. Record `ldd` evidence for standalone applications
when library selection matters.

## gfx9 defaults

The collection helper defaults to the settings that produce useful Instinct
traces on gfx9:

```text
--att-activity 8
--att-target-cu 0
--att-shader-engine-mask 0x1
--att-simd-select 0xF
```

These are defaults, not universal constants. Change them when the selected
workgroup does not execute on the target CU or when another architecture
requires different settings. Prefer one shader engine and one steady-state
dispatch unless the user requests broader coverage.

## Collection failures

- Header-only or activity-only output: try a CU known to receive the selected
  workgroup and verify the dispatch filter.
- Raw `.att` plus a flat CSV but no results database or UI directory: the trace
  was not decoded into a viewer-ready bundle; recollect with a working decoder.
- Empty trace: verify kernel regex, matching-dispatch number, GPU visibility,
  and that the workload launches enough matching kernels.
- Profiler/runtime crash: check ROCm compatibility before changing ATT knobs.
