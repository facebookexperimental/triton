# Blackwell Persistent Pipeline Efficiency

Apply this guidance to NVIDIA Blackwell kernels with persistent workers, preprocessing launches, asynchronous task pipelines, or staged output publication. Preserve the kernel's numerical and synchronization contracts; optimize data movement and scheduling without assuming a particular algorithm.

## Amortize Launch And Scheduling Overhead

When the logical work grid contains many small work groups, consider persistent workers that process multiple groups. Prefer native multidimensional program coordinates for direct launches and a compact task counter for persistent launches. All warp-specialized tasks must decode the same logical work item and use one shared outer-task/ring phase convention.

Select the persistent grid from measured occupancy and available CTAs rather than launching one worker for every logical task. Protect small workloads where persistent scheduling or multi-CTA coordination costs more than the launch overhead it removes.

## Fuse Compatible Preprocessing

If a preprocessing pass already traverses an input/output domain, consider fusing other bandwidth-compatible elementwise work into that pass:

- applying a normalization or scale needed by the main kernel;
- materializing a transformed input that removes repeated work from the main loop;
- initializing an accumulation destination required by reduce-add publication.

Only fuse operations with compatible traversal and lifetime. Prove the transformed representation preserves the required rounding and numerical contract. Repeated launches and autotune warmups must initialize accumulation destinations deterministically and exactly once.

## Remove Redundant Main-Kernel State

When preprocessing makes a main-kernel input or transformation redundant, remove its load, shared-memory allocation, barriers, and repeated arithmetic together. Keep the original path behind a compile-time fallback until the transformed contract is validated across all protected cases.

Packed native arithmetic can reduce conversions and instruction count when operand layout and dtype permit it. Use it only when its rounding behavior satisfies the protected tolerance; preserve a wider-precision fallback for unsupported dtypes or stricter numerical requirements.

## Schedule By True Data Readiness

Order pipeline stages by producer/consumer readiness rather than source order. Defer a load or barrier wait until immediately before first use when doing so creates useful overlap, but never move it past a dependency.

TMEM and SMEM reuse require explicit lifetime proofs. Before aliasing storage:

- prove every compute operation and local load completed its final read;
- prove every asynchronous store drained before its staging slot is overwritten;
- include reuse across persistent outer-loop iterations, not only one logical item;
- add an explicit barrier when different warps can overwrite storage still being read.

## Parameterize Publication Rings Consistently

For sliced output publication, give each outstanding store a distinct staging or ring slot. Use pending-store waits only when same-slot reuse distance matches the ring depth, then drain stores before releasing aliased source storage.

Represent ring depth with one compile-time contract shared by:

- storage and barrier allocation;
- slot indexing and phase calculation;
- producer publication and consumer acquisition;
- asynchronous store wait depth;
- host configuration and pruning.

Deeper rings may improve overlap on long workloads but regress short workloads through resource pressure. Select ring depth and persistent worker count by measured workload class, independently from numerical changes.

## Keep One Authoritative Topology Policy

Use one host-side selection path for descriptor geometry, CTA topology, persistence, ring depth, and fallbacks. Prune unsupported or predictably unprofitable configurations before autotuning.

Gate specialized paths by the actual requirements they depend on, such as dtype, shape, alignment, storage capacity, and CTA topology. Preserve fallback routes for boundary tiles, alternate dtypes, small workloads, and configurations that cannot satisfy the optimized lifetime proof.

## Validate The Pipeline

Test short and long workloads, boundary shapes, supported dtypes, one-CTA and multi-CTA modes, persistent and direct launches, repeated executions, and every selected ring depth. Require finite, repeatable outputs and reference correctness before performance promotion.

Use profiling to distinguish launch overhead, dependency stalls, barrier waits, local-memory traffic, spill pressure, and publication serialization. Treat lower instruction count or deeper buffering as hypotheses; promotion still requires stable end-to-end latency improvement.
