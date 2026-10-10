# gfx942/CDNA3 Kernel Optimization

Treat MI300X as a wave64 CDNA3 target with eight XCDs. Diagnose the selected
specialization before editing: use physical VGPR/SGPR counts, LDS allocation,
occupancy, MFMA utilization, cache traffic, and instruction mix from the
profiler and compiler artifacts. Do not infer occupancy from source tensor
sizes.

For work distribution, distinguish insufficient parallelism from excessive
per-CTA overhead. XCD striping and persistent scheduling are useful only when
they preserve load balance and improve measured locality. For narrow GEMMs,
measure inactive lanes, edge masking, address-generation cost, and repeated B
traffic before expanding the ordinary two-dimensional tile space.

Prefer MFMA-compatible shapes and coalesced vector loads. Audit generated
layout conversions and spills whenever a larger tile, K pack, pipeline stage,
or wave count changes resource use. Preserve a general fallback when adding a
shape family specialization.
