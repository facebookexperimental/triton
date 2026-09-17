# Architecture Discovery

Use current-file-only tuning only while measurements support a concrete local hypothesis. One
rejected candidate without repository evidence can be enough to question the current source's
assumptions, regardless of how the candidate labels its hypothesis; do not require a long sequence
of nearby failures before considering repository research or a higher-level hypothesis.

Before comparing implementations, establish the semantic contract: input and output dtypes,
quantization and scaling modes, activation approximations, masking, scheduling assumptions,
and the exact timed scope. Do not attribute a performance gap to architecture when numerical
or benchmark modes differ.

Do not require a reference to use the same dtype, quantization mode, framework, or operator name.
A BF16, FP16, or otherwise semantically adjacent kernel can provide the strongest evidence for
work decomposition, operand reuse, memory lifetime, or synchronization. Search by structural
relationship first, then record numerical and semantic differences as transfer limitations.

When the current source and profiling evidence cannot identify a useful edit, formulate one
specific repository research question. Search for transferable patterns rather than a named
answer. Useful comparison dimensions include:

- Work decomposition across programs, CTAs, warpgroups, warps, and persistent workers.
- Reuse of operands or intermediates across logical output tiles.
- Producer/consumer pipeline depth and overlap of loads, compute, reductions, and stores.
- Register, shared-memory, and tensor-memory layouts and value lifetimes.
- Barrier ownership, arrival counts, phase transitions, and overwrite safety.
- Publication and reduction topology.
- Compiler layout or lowering constraints that shape the implementation.

A source pattern is evidence, not permission to transplant it. Verify architecture, shapes,
dtypes, operation semantics, memory capacity, and synchronization invariants before adapting
it. For an asynchronous structural candidate, state the producer, consumer, ownership,
visibility point, reuse point, and proof that storage is not overwritten while still live.

Choose one coherent hypothesis. Classify it as parameter, memory_layout, pipeline, topology,
synchronization, compiler_layout, algorithmic, or unknown. When moving beyond local tuning, record why
the previous level was insufficient and cite the source-research evidence that supports the
new level. Repository research and profiler evidence guide candidate generation; only the
external correctness and uninstrumented benchmark harness may promote a candidate.
