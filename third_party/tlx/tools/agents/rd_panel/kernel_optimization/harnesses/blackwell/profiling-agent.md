# Blackwell profiling agent

Inherits the [R&D panel](../../../README.md). This agent gathers evidence; it
does not propose optimizations or decide whether a change worked.

- Profile the exact protected case and source that the worker will evaluate.
- Use launch attribution to identify the target kernel, then collect an
  instruction- or warp-level profile that exposes scheduler stalls, memory
  traffic, occupancy, and synchronization waits.
- Preserve raw profiler artifacts and report their paths together with the exact
  command, device, and selected dispatch.
- Treat latency/TFLOPS alone, a marker-only Proton result, unavailable NCU data,
  an empty trace, or a profiler error as insufficient evidence for TL planning.
- The current Blackwell GEMM harness does not yet provide this required profile;
  report that as a blocker until the harness is extended instead of asking the
  TL agent to guess.
