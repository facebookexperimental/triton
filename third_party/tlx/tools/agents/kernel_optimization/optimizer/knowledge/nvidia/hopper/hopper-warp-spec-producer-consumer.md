# Hopper Warp-Spec Producer-Consumer

Use this guidance only for NVIDIA Hopper (H100, sm_90) kernels with an inner loop that loads tiles into SMEM and consumes them in compute. It builds the 2-task producer/consumer scaffold that all other Hopper warp-spec patterns extend.

## Recognize When Warp Specialization Wins

Run NCU on the baseline first (Tier-1, a single launch). Apply only when BOTH hold:

- The tensor pipe sits idle a majority of cycles, AND
- the long-scoreboard stall is the top stall component (warps spend a meaningful fraction of issue cycles waiting on DRAM).

Confirming signal: the NCU bottleneck verdict names memory or latency as dominant. Either signal alone is insufficient.

Reject when ANY holds (the rewrite regresses at bit-exact correctness):

- Tensor pipe at or near peak: compute-saturated; the split removes per-CTA registers and adds a starving handshake.
- NCU verdict "well-balanced": no slack to fill.
- Small long-scoreboard share with the tensor pipe on the critical path: grey zone, no headroom.
- Small wall time with multiple resident blocks per SM at the SMEM-binding ceiling and mid-saturation: barrier-amortization trap; consumer register demand exceeds budget and occupancy drops.
- Small wall time with fewer than 2 blocks per SM on a mostly idle SM and HMMA far below peak: every barrier wait lands on the critical path.

Low occupancy alone decides nothing: tensor pipe near peak plus small long-scoreboard means high-register WGMMA (reject; relieve registers instead), while tensor pipe idle plus long-scoreboard on top means memory-bound underutilization (allow).

Without NCU, use the safer-side fallback: `Block_Limit_Regs` binding with regs/thread near the ceiling rejects. Otherwise prefer the simpler chain (software pipelining, PID swizzle, `num_warps` widening) until NCU evidence exists.

## Require Persistent Style First

Warp specialization requires a persistent kernel: a single 1D grid with `for tile_id in range(pid, total_tiles, NUM_SMS)`, grid `(min(NUM_SMS, total_tiles),)`, and `NUM_SMS` from `torch.cuda.get_device_properties(device).multi_processor_count`. Producer and consumer share SMEM and barriers that persist across tiles. If the kernel uses a 2D grid (one CTA per tile), convert it to persistent style first; do not bundle the conversion silently into the warp-spec rewrite.

## Size The Consumer Register Budget

Pick the consumer `registers=` value from this table (Hopper warp-spec FA / GEMM):

| Family | Single consumer (`replicate=1`) | Pingpong consumer (`replicate=2`) |
|---|---|---|
| Attention (FA-WS) | 224 | 232 |
| GEMM (GEMM-WS) | 232 | 232 |

```python
with tlx.async_task(num_warps=4, replicate=2, registers=232):
```

Sanity-check the table against a per-thread lower bound — normalize warpgroup-wide accumulator storage by the consumer thread count before comparing with the per-thread `registers=` budget:

```
per_thread_regs ~= (BLOCK_M_SPLIT * HEAD_DIM * acc_regs) / (consumer_num_warps * 32) + per_thread_softmax_overhead
```

where `BLOCK_M_SPLIT = BLOCK_M / replicate`, `acc_regs` is 1 for fp32 (0.5 for fp16/bf16), and per-thread softmax overhead is ~16/24/40 regs for HEAD_DIM 64/128/256. If this lower bound exceeds the table default, the default under-budgets the consumer — raise it and audit for exotic accumulator dtype or softmax overhead. If the bound sits below the default, the default stands (it is a cap, not a prediction).

Skip silently when the consumer `tlx.async_task` already carries an explicit `registers=` value; re-emitting a default over it is a no-op at best and an override at worst. The keyword is `registers=N`, not `num_regs=N`.

## Partition Warps Around The Default Task

Declare exactly one `tlx.async_task("default")`: it owns warps `[0, N)` with `N` = the `num_warps` from the kernel's autotune `triton.Config`. Explicit partitions add warps on top of the default, so always place the task that scales with warp count in the default — in this scaffold, the consumer — and size every explicit partition from its ops' warp minimums:

- WGMMA: at least 4 warps (one warpgroup, 128 threads) per dot group.
- TMA descriptor issue: 1 warp (single-thread issue; elect one thread).

```python
with tlx.async_tasks():  # num_warps=4 in the autotune config
    with tlx.async_task("default"):  # consumer: 4 warps (one WGMMA group)
        ...
    with tlx.async_task(num_warps=1):  # producer: TMA issue only
        ...
```

The default partition is always at least 4 warps: keep the autotune `num_warps` at or above the default task's minimum (a WGMMA default needs 4).

Replication on the default is manual: `replicate=` does not split the default partition, so for N warp groups on the default role (e.g., 2 WGMMA warp groups), copy the task body into an additional explicit `async_task(num_warps=4)` block per extra group instead of a single `replicate=N` default.

## Check The SMEM Precondition Before Rewriting

Compute this lower bound (bytes) for every autotune config BEFORE rewriting, sizing the manual rings the rewrite will actually allocate — NOT `num_stages` (which is 1 in this scaffold):

```
operand_per_buffer = (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * dtype_bytes
producer_smem      = NUM_BUFFERS * operand_per_buffer
barrier_bytes      ~= 2 * NUM_BUFFERS * 16 + 4 * 16
output_smem        ~= BLOCK_M * BLOCK_N * out_dtype_bytes
extension_smem     += BLOCK_M * BLOCK_N * 4 * ACC_STAGES   # when adding a sink-writer acc ring
smem_bytes         ~= producer_smem + output_smem + barrier_bytes + extension_smem + margin
```

Sum every planned `tlx.local_alloc` (shape × dtype × depth) plus barriers, padding, and a 2–4 KB margin, against the 232,448-byte H100 cap (see `hopper-machine-limits.md`): if the smallest config exceeds the cap, stop and shrink first — `NUM_BUFFERS` (3→2) first, then `BLOCK_N` (256→128), then `BLOCK_M` last. If some configs fit, use the smallest fitting config, not the largest. Default to `NUM_BUFFERS=2` for warp-spec attention; `NUM_BUFFERS=3` adds SMEM pressure without a stable win and belongs only as an extra autotune config.

## Build The Scaffold

Set `num_stages=1` in the `triton.Config` (staging is manual), allocate per-iteration operands as multi-stage rings via `tlx.local_alloc(..., NUM_BUFFERS)` with full/empty barrier pairs, and give per-tile operands single-slot barriers. Pre-compute every `tlx.barrier_expect_bytes` size as `tl.constexpr` BEFORE `tlx.async_tasks()` — arithmetic inside task scope loses constexpr status and fails.

Scoping rules (violations are compile errors or silent corruption):

- `RankedTensorType` producers (`tl.arange`, `tl.zeros`, `tl.load`, tensor arithmetic) go INSIDE the consuming `tlx.async_task`; each task recomputes its own scalars and offsets.
- `tlx.local_alloc`, `tlx.alloc_barriers`, and constexprs go at kernel scope BEFORE `tlx.async_tasks()`.
- Producer and consumer iterate tiles in identical order.

The producer waits empty `phase ^ 1`, expects bytes, and issues `tlx.async_descriptor_load` per operand; the consumer waits full `phase`, computes, and arrives empty. Do not introduce TMA into a kernel that uses pointer loads — keep `tl.load` + `tlx.local_store` + `tlx.fence("async_shared")` + `barrier_arrive`.

## Track Phase With Monotonic Counters

Derive `(buf_id, phase)` from a counter initialized OUTSIDE the tile loop and never reset inside it:

```python
buf_id = accum_cnt % NUM_BUFFERS
phase = (accum_cnt // NUM_BUFFERS) & 1
```

Producer and consumer advance identical counters. Resetting phase per tile deadlocks as soon as per-tile trip counts vary, because the barrier's internal phase no longer matches. Per-tile single-slot barriers use a tile-index counter with depth 1. In multi-phase loops, give different-shaped tensors phase-suffixed names (`_p1`, `_p2`); Triton requires shape consistency per variable.

## Avoid The Known Failure Modes

- Deadlock on tile 2+: phase or counter reset inside the tile loop.
- `WarpSpecializeOp captures RankedTensorType`: tensor op outside `async_task` scope.
- `'tensor' has no attribute 'value'` at `barrier_expect_bytes`: byte size not constexpr.
- SMEM overflow: shape stored into a buffer allocated for another shape.
- Wrong results: producer/consumer tile-order mismatch.
- Sharp regression at bit-exact output on a compute-saturated baseline: the gate misjudged — revert.
- Undersized default: autotune `num_warps` below 4 starves a WGMMA default — raise `num_warps` to at least one warpgroup.
- `replicate=N` on the default task: the default is not split — copy the body into explicit warp-group tasks instead.

## Validate The Rewrite

Verify correctness at the kernel's standard dtype tolerance across shapes, boundary tiles, and repeated launches. Re-run NCU: the long-scoreboard share must fall, the tensor pipe must rise, and `Stall_Barrier` must not balloon. Promotion requires a stable end-to-end win; fitting the budget is necessary but not sufficient.
