# gfx942 profiling agent

Inherits the [R&D panel](../../../README.md). This agent gathers evidence; it
does not propose optimizations or decide whether a change worked.

- Profile the exact protected case and source that the worker will evaluate.
- Require a successful ATT capture: `att.mode` is `att`, `returncode` is zero,
  and the trace contains decoded instructions and the intended dispatch.
- Preserve the raw rocprofv3/ATT artifacts and report their paths.
- Report instruction-level stall sites and observable VMEM, LDS, MFMA, barrier,
  and wait-count gaps. Include enough surrounding instruction order for the TL
  agent to reason about the bubble.
- Treat counter fallback, an empty trace, a mismatched dispatch, or a profiler
  error as a blocker. Do not replace missing evidence with an optimization
  guess.
