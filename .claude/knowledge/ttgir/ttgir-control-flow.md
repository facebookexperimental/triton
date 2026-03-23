# TTGIR Control Flow Ops

Warp specialization structure, pipeline control, and cluster launch control.

## Warp Specialization

### `ttg.warp_specialize`
Top-level op for executing different code on different warp groups
simultaneously. Contains a "default" region (implicit capture, runs on the
original warp group) and N "partition" regions (isolated from above, each
with its own warp count).

All regions start simultaneously and are joined at the end of the op.
Partition regions receive explicit captures as block arguments.

```mlir
%out = ttg.warp_specialize(%a, %b)
default {
    %result = some_op(%a)   // implicit capture
    ttg.warp_yield %result : f32
}
partition0(%arg0: ..., %arg1: ...) num_warps(4) {
    async_work(%arg0, %arg1)
    ttg.warp_return
} : (...) -> f32
```

Attributes:
- `partitionNumWarps`: number of warps per partition
- `warpGroupStartIds`: starting warp IDs (optional)
- `requestedRegisters` / `actualRegisters`: register budget hints

### `ttg.warp_specialize.partitions`
Container op that holds the partition regions of a `warp_specialize` op.
Required because MLIR needs entire operations to be `IsolatedFromAbove`.
This op is the `IsolatedFromAbove` boundary; it's a terminator of the
parent `warp_specialize`.

### `ttg.warp_yield`
Terminator for the default region of `warp_specialize`. Operands become
the SSA results of the `warp_specialize` op.

```mlir
ttg.warp_yield %a, %b : i32, tensor<32xbf16, #blocked>
```

### `ttg.warp_return`
Terminator for partition regions. Has no operands — partitions cannot
return values (they communicate via shared memory / barriers).

```mlir
ttg.warp_return
```

## Pipeline Control

### `ttg.predicate_stage`
Generates a predicate for a pipeline stage. Given the loop induction
variable, upper bound, step, max number of stages, and current stage,
returns whether this stage should execute on this iteration.

Used by software pipelining to guard operations in different pipeline stages.

```mlir
%pred = ttg.predicate_stage %iv, %ub, %step maxStage 3 stage 1 : i32 -> i1
```

### `ttg.mask`
Mask region for pipelining. Contains operations that should only execute
when the predicate is true. The region returns values via `mask.return`.

```mlir
%result = ttg.mask %pred {
    %val = some_op(...)
    ttg.mask.return %val : f32
} : f32
```

### `ttg.mask.return`
Terminator for `mask` regions. Returns values to the parent `mask` op.

```mlir
ttg.mask.return %result : f32
```

## Cluster Launch Control (CLC, Blackwell only)

CLC enables dynamic persistent kernels with work stealing on Blackwell.
A CTA can try to cancel a pending cluster launch and steal its work.

### `ttng.async_clc_try_cancel`
Requests atomic cancellation of a cluster not yet launched. Writes an
opaque 16-byte CLC response to SMEM. Completion tracked via mbarrier.
Uses PTX `clusterlaunchcontrol.try_cancel.async.shared::cta`.

```mlir
ttng.async_clc_try_cancel %mbar_alloc, %clc_res_alloc : ...
```

### `ttng.clc_query_cancel`
Extracts the CTA ID from a CLC cancel response in SMEM. Returns -1 if
the cancellation was not successful (cluster already launched).

```mlir
%cta_id = ttng.clc_query_cancel %clc_res_alloc : ... -> i32
```
