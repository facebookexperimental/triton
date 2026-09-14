# AMD TLX Attention Optimization

Apply this guidance only when `optimize-amd-tlx-attention` was explicitly selected for an AMD Triton or TLX attention kernel. Preserve the kernel's numerical and synchronization contracts; optimize data movement and scheduling without assuming a particular attention variant. The accompanying attention-variant reference is injected immediately after this guide and contains the semantic taxonomy, grid formulas, and scheduling choices needed to classify the kernel.

## Classify Semantics Before Scheduling

Write the attention equation, tensor ownership, offset/length sources, mask, and output domain before modifying the kernel. Distinguish dense self-attention from cross-attention, normalized softmax from HSTU SiLU scores, independent Q/KV sequence domains from grouped-query head mapping, dense from packed jagged, mapped IKBO, and paged representations, and forward from dQ/dK/dV, fused, or multi-launch backward phases.

## Derive The Work Model

For each active query tile, derive the logical and physical/persistent grid, Q-tile count, exact KV lower and upper bounds, bulk mask-free block count, masked boundary/diagonal block count, and reuse across heads, sequences, CTAs, and XCDs. Use these formulas to choose diagnostic shapes and interpret profiles. Do not compare kernels with a FLOP count that ignores materially different causal or windowed work without labeling it.

## Establish A Correct Streaming Baseline

For standard softmax attention, preserve online max/sum correction for every streamed KV block and normalize only after the final block. Apply the implementation's log2 scaling consistently when using `exp2`. For HSTU self-attention, do not introduce online-softmax state: its score is SiLU-scaled and divided by a fixed maximum sequence length. Validate masks against an explicit reference before optimizing loads.

## Optimize In Layers

Keep semantic, memory, pipeline, and grid changes attributable, one coherent change per candidate:

- **Bounds and masks:** separate bulk blocks from causal, jagged, or tail boundaries; peel masks out of the hot loop when valid.
- **Memory staging:** compare direct loads, single-slot async LDS staging, double-buffer prefetch, and deeper rotated cluster pipelines.
- **Pipeline schedule:** build explicit prologue/steady-state/epilogue regions; derive commit/wait counts and avoid out-of-range speculative loads.
- **Persistence:** use a fixed-CU grid when tiles per batch/head justify it; pin related work to XCDs when it improves K/V cache reuse.
- **Load balance:** flatten jagged grids, skip padding slots safely, and permute sequences when length variance causes imbalance.
- **Tile/config tuning:** sweep query/KV block sizes, warps, MFMA dimensions, stages/buffers, `waves_per_eu`, and persistent geometry over the production distribution.
- **Backward-specific reuse:** treat dQ and dK/dV ownership, reductions, atomics, and multi-launch preprocessing as distinct scheduling problems.

For packed jagged backward inputs, preserve direct global-to-LDS lowering by peeling masked prefix and tail blocks away from a contiguous mask-free interior when the mask geometry permits it. Keep one async ring live across all regions.

## Treat Native MFMA Design As A Matched Set

Treat `tlx.amd_register_resident` and `tlx.amd_scheduled_mfma` as a matched native-layout design, not independent hints. Establish explicit MFMA accumulator and dot-operand layouts, slice tensors into native fragments, and schedule independent accumulator chains in source order before adding register-class residency. Never copy a residency annotation onto an ordinary implicit-layout `tl.dot` result; port the complete layout/fragment/scheduling/commit contract or leave the ordinary dot lowering intact.

## Preserve Synchronization And Numerical Correctness

Use `num_stages=1` for manually managed async or warp pipelines. Derive waits from the exact K/V commit order and handle short loop ranges with a separate path when they cannot fill the steady-state pipeline. Rescale both the softmax denominator and the output accumulator when the running maximum changes. Apply causal alignment explicitly for unequal Q/KV lengths; do not assume ordinary `k <= q` for rectangular cross-attention. Include sequence length, masking mode, and mapping-sensitive dimensions in autotune keys when they change the best schedule. Re-test large negative logits, fully masked boundaries, short sequences, and non-divisible tails.

## Benchmark The Production Distribution

Use the protected cases and measurements as the production distribution for this optimization run. Report latency and speed ratio when dense-equivalent TFLOP/s misrepresents causal, windowed, or jagged work. Retain per-shape winning configurations; do not force one universal winner when crossover behavior is measured.
