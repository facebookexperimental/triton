# TLX barrier hotspot workflow

Start with the [Triton Agentic Reference](https://www.internalfb.com/wiki/Triton_Core/Agentic_Reference/)
and its TLX synchronization and diagnostic-tool routes. Consult
`third-party/triton/beta/triton/third_party/tlx/doc/tlx_barriers.md` for the
phase, implicit-arrive, and named-barrier semantics.

## Gather a static lead

For an existing NVIDIA TLX compilation, opt into the read-only compiler
diagnostic with `TRITON_REPORT_BARRIER_HOTSPOTS=1`. If its result is cached,
also set `TRITON_ALWAYS_COMPILE=1`. Capture compiler remarks from stderr;
each reports a looped `ttng.wait_barrier` or `ttng.wait_barrier_named` with its
source location and nesting depth. An mbarrier remark calls out explicit
`deps` where present. No named-barrier ID is allocated, no IR is rewritten,
and `TRITON_PROMOTE_MBAR_TO_NAMED_BARRIER` need not be enabled.

For already dumped late TTGIR, use:

```bash
triton-opt kernel.ttgir \
  --triton-nvidia-gpu-promote-mbarrier-to-named-barrier='report-likely-barrier-perf-locations=true'
```

The reporter requires the TLX fixup module attribute
`triton.skip_generic_pipeline`. Run it on TTGIR before LLVM conversion, where
the waits still exist. No remarks does **not** establish that synchronization
is free: waits outside loops, barriers lowered to other operations, or modules
without the source marker are outside this static report.

## Interpret, then investigate

- Treat loop depth as a *ranking hint*, never as measured latency. Determine
  whether the wait actually stalls using Triton-MPP for general performance,
  Proton for a source-scoped timeline, or CUTracer for hangs/races and executed
  warp progress. Follow the relevant tool's workflow; do not run a performance
  benchmark unless the user has requested one.
- Trace the waited-on buffer and both phases. An mbarrier wait may pass
  immediately when its phase is already complete, and multiple waits may
  observe the **same** completion. `barrier_expect_bytes`, TMA implicit
  arrivals, MMA commits, explicit `deps`, remote CTA barriers, and shared-memory
  reuse impose memory-completion and visibility requirements. A named barrier
  instead counts waiting and arriving threads in a fresh rendezvous. A remark
  is never evidence that these mechanisms can be exchanged or removed.
- For a correctness failure or deadlock, pair the runtime trace with the
  existing `.claude/skills/barrier-visualization/SKILL.md` workflow; check
  arrival count, parity, per-slot reuse, first/last loop iterations, and
  producer/consumer task guards before suggesting a fix.

Report the source location, loop depth, barrier kind, known `deps`/async
producer, observed profiler or trace evidence (or that it is missing), and a
specific next check. Do not claim a performance win from compiler remarks.
