# Blackwell Persistent CLC Scheduling

Use this guidance only for NVIDIA Blackwell kernels that use persistent scheduling or Cluster Launch Control (CLC). Consult `.claude/skills/tlx-api-reference/SKILL.md`, `docs/design/triton-clc-tile-scheduler.md`, and `third_party/tlx/language/tlx/dynamic_launch.py` for API and protocol details.

## Find Runtime Scheduling Overhead

Inspect persistent loops for repeated integer division, modulo, `cdiv`, section lookup, head lookup, and flattened tile-ID decoding. These operations are especially costly when repeated for every tile by every async task.

When batch size, head count, group size, tile count, or launch geometry are compile-time or launch-time metadata, prefer encoding their stable structure in `tl.constexpr` values and the launch grid. Do not specialize runtime numerical parameters such as `sm_scale`.

## Preserve Structured CLC Coordinates

Prefer a 2D or 3D launch grid when its axes can directly represent head lanes, head chunks, sections, batches, or tile rows. When appropriate, consume CLC responses with `return_3d=True` and preserve the structured coordinates instead of flattening them and immediately decoding them with runtime div/mod.

If a logical tile ID is still required, reconstruct it with compile-time strides derived from the launch grid. All warp-specialized tasks must derive the same logical work item and advance the CLC phase exactly once per consumed response.

## Handle Groups And Tails Explicitly

Prove the mapping for all of these cases:

- `H <= GROUP_SIZE_N`;
- full head groups where `H % GROUP_SIZE_N == 0`;
- a tail group where `H % GROUP_SIZE_N != 0`;
- single-CTA and multi-CTA launch modes;
- invalid or exhausted CLC responses.

Compute full-section counts, rows per section, and chunks per section at compile time where possible. Ensure the tail path cannot divide by zero or produce an out-of-range head. Preserve the kernel's original causal tile order.

## Gate And Measure The Optimization

Introduce a separate `tl.constexpr` or autotune flag for compact scheduling and retain the original scheduler as a fallback. Enable it only for validated combinations of stage, group size, grid geometry, CTA count, and synchronization mode.

Validate the target causal path and protect dense, backward, alternate head counts, and tail-group cases from regressions. Treat fewer scheduling instructions as a hypothesis; promotion still requires stable end-to-end latency improvement with passing correctness.
