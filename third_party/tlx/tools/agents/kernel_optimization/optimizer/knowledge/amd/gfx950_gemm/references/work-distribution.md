# gfx950 GEMM Work Distribution

Choose how output and reduction work reach the device before fine scheduling.
Count useful CTAs, complete CU rounds, final-wave occupancy, K work per owner,
launch cost, workspace traffic, and numerical-order changes.

## Ordinary Output Tiling

Use ordinary data-parallel tiling when the M×N grid already provides full,
balanced device waves. Optimize tile geometry and data movement before adding
coordination. Split-K or persistence adds overhead without solving a utilization
problem when the natural grid already oversubscribes the device.

A sparse final device wave is discontinuous: a small tile-count change can add
an entire CU round. Consider tiles and orientations that improve complete-wave
occupancy, but include their padded arithmetic and per-CTA overhead.

## Persistent Execution

Persistent kernels let each CTA process multiple output tiles. They can
amortize launch/setup cost, preserve locality, and redistribute work, but they
also create cross-tile pipeline lifetimes. Prove that stores and asynchronous
operations have completed before operand or staging storage is reused by the
next tile.

Choose worker count from logical work and the measured device, not from a fixed
preference. Preserve a nonpersistent fallback for workloads too small to
amortize the persistent loop or too irregular to balance it.

## Stream-K And Global Split-K

Use K partitioning when the natural output grid underfills the device or leaves
large imbalance and K is deep enough to amortize coordination.

Account for:

- K work and MFMA chain length per partition;
- partial-output workspace or atomic traffic;
- reduction launch and synchronization;
- accumulation order and tolerance;
- persistent scheduling overhead; and
- the output-grid cases that already have enough parallelism.

The split count need not be a power of two. Choose the smallest legal count
that fills useful device waves without making each partition too short.

If only the final wave is sparse, compare a hybrid decomposition: send complete
waves through the lean data-parallel kernel and only the remainder through
Stream-K. The extra launch is worthwhile only when the bulk work amortizes it
and the tail is sufficiently sparse. Derive and test the adjacent denser-tail
boundary.

## Region-Split Multi-Launch

Another sparse-wave repair is to split a long output axis into disjoint regions
served by different already-efficient tile widths. Derive region sizes from
launch algebra rather than memorized dimensions:

```text
cta_i = ceil(M / tile_m_i) * ceil(N_i / tile_n_i)
cta_i % device_cu_count == 0
sum(N_i) == N
```

Compare total physical output work with the incumbent and time the complete
multi-launch operation. Reject when filling device waves costs more padded MFMA
work or launch/cache overhead than it saves.

## LocalSplitU

Use LocalSplitU when very small M gives too little natural output parallelism,
K is deep enough to partition among waves inside one workgroup, and an
in-workgroup FP32 reduction is cheaper than leaving waves idle.

LocalSplitU is not global Split-K: partials remain inside the workgroup, reduce
through LDS, and store once without a global workspace or atomic output.

Generate a bounded legal set from output tile, local split count, K per wave,
and MFMA K width. Reject plans unless the split exactly covers K or a separate
tail path is proven, each wave owns a disjoint K range, the output grid is large
enough, and the reduction fits resource/synchronization budgets.

More local splits shorten MFMA chains and may increase latency hiding, but add
waves, FP32 partial state, LDS traffic, and reduction depth. Larger K per wave
reduces partials and barriers but lengthens operand live ranges. Verify final
VGPRs, spills, LDS behavior, and reduction ordering.

Keep one parameterized implementation. A known-plan table is a tuning cache,
not the supported-shape list. Validate unseen M, N, and K values, advertised
tails, fallback behavior, first-call tuning cost, and host dispatch overhead.

## Workgroup And XCD Mapping

Mapping changes locality and load balance without changing the device kernel.
Model operand reuse per CU/XCD, the number of tiles in each band, and whether a
narrow grid starves a stripe. Re-evaluate mapping whenever tile geometry or
work distribution changes; a mapping measured on a large balanced grid can
hurt a narrow or one-wave grid.
