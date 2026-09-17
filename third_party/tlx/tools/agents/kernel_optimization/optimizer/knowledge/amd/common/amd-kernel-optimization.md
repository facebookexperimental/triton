# AMD Kernel Optimization

Apply this guidance to Triton and TLX kernels on AMD GPUs. Base changes on the supplied source, protected cases, measurements, and target architecture; do not assume an attention, GEMM, or reduction workload that the source does not implement.

## Optimize The Measured Bottleneck

Use steady-state end-to-end latency as the promotion criterion. Treat profiler duration, memory traffic, occupancy, register counts, spills, instruction mix, and stalls as diagnostic evidence. Keep one candidate attributable to one hypothesis, and do not choose configurations from resource metadata alone.

## Preserve Layout And Memory Contracts

Trace data through global memory, LDS, registers, and MFMA operands before changing layouts or staging. Distinguish metadata-only views from physical movement in final IR. Preserve coalescing, LDS bank behavior, vector width, descriptor legality, and the ownership expected by each consumer.

Treat `tlx.amd_register_resident` and `tlx.amd_scheduled_mfma` as parts of an explicit native-layout design. Use them only when the source and target support the corresponding accumulator, operand-layout, fragment, scheduling, and commit contract; do not copy residency annotations onto an unrelated implicit-layout dot.

## Preserve Pipeline Correctness

For manually managed asynchronous or warp pipelines, derive buffer slots, commit groups, waits, barriers, and reuse phases from the exact producer/consumer order. Use compiler staging only when it does not overlap a separately managed pipeline. Handle short and boundary loop ranges without speculative out-of-bounds loads, and prove that every asynchronous operation has completed before its storage is reused.

## Tune For The Selected Architecture

Sweep tile geometry, warps, stages or buffers, `waves_per_eu`, persistence, and work distribution only where the protected cases exercise them. Treat CU/XCD placement and persistent scheduling as conditional strategies whose value depends on reuse, workload size, and load balance. Preserve a valid fallback for unsupported shapes, dtypes, alignments, or topology.

Logical tensor bytes in TTGIR are not per-wave VGPR allocation, and textual SSA lifetime is not machine live range. Confirm physical VGPRs, spills, LDS allocation, and occupancy with compiler or machine-level evidence before using them to justify a change.
