# Interpreting AMD IR live-range reports

## Contents

- Artifact levels
- Interval definitions
- LDS analysis
- Tensor-register analysis
- Loop and control-flow limits
- Comparing compiler passes
- Validation checklist

## Artifact levels

| Artifact | What can be stated | What cannot be stated reliably |
|---|---|---|
| Triton/TLX IR | SSA dependencies and source-level regions | AMD layouts, physical storage, issue order |
| TTGIR | Ordered logical definitions/uses, layouts, LDS views, async waits, scheduled MFMA sites | Exact physical VGPR assignment or cycle lifetime |
| AMDGPU/LLVM IR | Lowered memory operations and instruction-level dependencies | Final scheduled issue order unless preserved by the backend dump |
| Machine IR with live intervals | Virtual/physical register liveness and allocator decisions | Dynamic latency without execution data |
| ISA + metadata | Final instructions, resource totals, LDS total, spills, occupancy inputs | Complete mapping back to every tensor SSA value |

Do not combine intervals from different artifact levels into one unlabeled timeline.

## Interval definitions

The bundled analyzer reports a conservative textual interval:

```text
[definition IR line, last textual SSA use IR line]
```

The interval is exact for straight-line textual SSA use order in the selected artifact. It is not a cycle range. It may overstate or understate dynamic liveness across branches, loops, phi-like block arguments, or backend rematerialization.

`span` is a source-line distance, not latency. `uses` counts textual SSA references after excluding the defining result. The bar chart is normalized to the selected function's line range.

## LDS analysis

An LDS allocation root is normally produced by `local_alloc`. Views such as `local_view`, `local_slice`, `memdesc_subslice`, reinterpretations, and explicit aliases do not allocate new data. Group those values with their source allocation. The root remains live through the last use of any grouped alias.

The analyzer derives `logical bytes` from the first shaped field of `!ttg.memdesc<...>`. Treat it as unknown for dynamic shapes or unrecognized element types. A logical size is not proof of a unique physical range: the compiler may overlay allocations with non-overlapping lifetimes, add alignment, or reserve scratch space.

To claim physical LDS reuse, obtain a post-allocation artifact or allocator dump containing offsets and sizes. Verify that barriers and outstanding async operations complete before reusing an overlapping range.

LDS liveness does not diagnose bank conflicts by itself. Combine access layout, lane-to-address mapping, operation width, and profiler counters.

## Tensor-register analysis

A `tensor<...>` SSA result in TTGIR is treated as a distributed register-resident logical tensor unless the original operation proves otherwise. The report's logical bytes equal full tensor elements times element width. They are not per-thread or per-wave register bytes because the encoding controls distribution and some values may be rematerialized.

Use the report to find candidates, not to compute occupancy directly:

- values defined much earlier than first/last use;
- prefetched fragments overlapping many accumulators;
- large accumulator chains kept across a runtime loop;
- duplicated layout conversions;
- values live across waits, barriers, or phase boundaries;
- results whose final use can move earlier without violating dependencies.

Confirm actual VGPR count, AGPR use, spills, and occupancy from compiler metadata or ISA. Exact per-value physical VGPR ranges require machine-level live-interval/register-allocation output.

## Loop and control-flow limits

An SSA value used inside a loop can be dynamically live for multiple iterations even when its textual interval is short. Loop results, iter arguments, and yielded accumulators are especially important. Inspect `scf.for`, `scf.while`, `scf.yield`, block arguments, and lowered `cf.br`/`cf.cond_br` manually when the report marks a cross-region value.

For branches, a linear textual overlap is conservative: mutually exclusive values can appear simultaneously alive in the report. For loops, textual order does not express repeated dynamic execution. Report both the static interval and the dynamic interpretation.

## Comparing compiler passes

Generate one report per saved pass artifact, then compare:

- definition and last-use lines relative to stable operations, not raw absolute line numbers alone;
- alias-root membership and logical LDS overlap;
- added or removed tensor values;
- hoisted loads and sunk stores/conversions;
- loop-carried accumulator changes;
- peak-overlap membership.

Canonicalization, CSE, DCE, LICM, layout-conversion hoisting/sinking, loop scheduling, pipelining, and buffer lowering can change intervals. A pass named for one of these transformations may still be a no-op for a particular specialization; verify the actual diff.

## Validation checklist

- Pin the exact function and specialization.
- Check every parser warning.
- Confirm allocation roots and aliases against the IR.
- Inspect long-lived tensors at their definition and last-use lines.
- Check loop iter arguments and yields manually.
- Label logical bytes separately from physical resources.
- Use machine-level data for exact physical VGPR claims.
- Use post-allocation offsets for exact LDS overlap claims.
- Preserve IR program order in any accompanying pseudocode.
