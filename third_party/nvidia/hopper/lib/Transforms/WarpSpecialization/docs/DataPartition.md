# Data Partitioning

Data partitioning physically splits tensor dimensions across multiple consumer
warp groups. After task assignment (which determines *which* ops run on
producers vs consumers), data partitioning determines *how* each consumer warp
group gets its slice of the data. For example, an M=256 accumulator is split
into two M=128 pieces for two consumer groups.

**File**: `WSDataPartition.cpp`
**Function**: `doDataPartition(funcOp, numConsumerGroups)`

## Pipeline Context

```
doTaskPartition          ← assigns ops to partitions
  → doTaskIdPropagate   ← propagates partition IDs to all ops
  → doDataPartition     ← THIS STEP: splits tensor dimensions
  → doPingPongPrep
```

Data partitioning is exposed as `nvgpu-ws-data-partition` and is not Hopper
only. Hopper-style AutoWS typically reaches it after partition ID propagation, while
Blackwell flows may run it as a separate pass when an explicit data partition
factor is attached or multiple warp groups require per-consumer slices.

## `DataPartitionScheme`

The central data structure tracking what to partition and how:

```cpp
struct DataPartitionScheme {
    unsigned numPartitions;                          // number of consumer groups
    SetVector<Operation *> ops;                      // ops to partition
    DenseMap<Operation *, unsigned> opPartitionDims;  // op → which dim to split
    DenseMap<Operation *, unsigned> dotPartitionOperand; // dot → which operand
    DenseMap<Operation *, SetVector<unsigned>> rematerializedOps; // ops to clone
    DenseSet<Operation *> opsToSkip;                 // ops exempt from partitioning
    DenseMap<unsigned, unsigned> funcArgPartitionDims; // func arg → partition dim
};
```

- `noOpPartitionDim`: Special sentinel value — ops with this dim are
  duplicated (cloned for each partition) rather than sliced.

## Algorithm

### Step 1: Task ID Fixup (`fixTaskId`)

Before partitioning, ensures all ops in def-use chains carry correct
`ttg.partition` attributes via bidirectional propagation:

- **Backward**: If an op uses a value defined by an `arith` op that lacks the
  consumer's partition ID, propagate backward.
- **Forward**: If a `YieldOp`, `ConditionOp`, `IfOp`, or `WhileOp` has a
  single-use operand whose defining op has extra partition IDs, propagate forward.

Runs to a fixed point.

### Step 2: Compute Partition Scheme (`computePartitionScheme`)

Drives partitioning from dot/MMA ops:

1. Collect all `WarpGroupDotOp` operations and all operations implementing
   `MMAv5OpInterface`.
2. For each dot with multiple `ttg.partition` values, determine the partition
   dimension from the accumulator shape:
   - **M dimension** (dim 0): if `shapePerCTA[0] / numPartitions >= 64`
   - **N dimension** (dim 1): if `shapePerCTA[1] / numPartitions >= 128`
   - M is preferred; N is fallback.
3. Call `getSliceToPartition` to trace the partition dimension through the
   dataflow graph.
4. Reject trial schemes that would require changing an existing TMEM encoding
   when partitioning along M. For example, splitting a `128x128` TMEM
   allocation with `blockM=128` into two `64x128` slices is rejected before any
   data partitioning rewrite; splitting `256x128` into `128x128` slices remains
   valid. This applies only to regular `TensorMemoryEncodingAttr` TMEM buffers;
   `TensorMemoryScalesEncodingAttr` scale buffers keep their scale encoding
   when sliced. When this guard fires, data partitioning is skipped and the
   function is otherwise left unchanged.

### Step 3: Slice Propagation (`getSliceToPartition`)

Traces the partition dimension backward and forward from the accumulator:

- **`getBackwardSliceToPartition`**: From the accumulator, walks backward
  through operand definitions. Tracks how the partition dimension transforms
  through transposes (`TransOp`), expands (`ExpandDimsOp`), reshapes, and
  other shape-changing ops. Stops at loads, block arguments, and ops that
  produce scalar types.

- **`getForwardSliceToPartition`**: From the accumulator, walks forward
  through result users. Handles `YieldOp` (follow to `scf.for` / `scf.if`
  results, or to the `scf.while` before-region backedge), `ConditionOp`
  (follow to `scf.while` results and after-region arguments), and tracks
  dimension remapping through layout-changing ops.

For `ReshapeOp` and `MemDescReshapeOp`, the pass remaps partition dimensions
by comparing flattened element intervals for each candidate source/destination
dimension. This permits reshapes such as:

```mlir
tensor<1x1x1x4x256xi8> -> tensor<256x4xi8>
```

to carry an M-dimension partition from logical scale dim 0 back to the packed
scale dimension. If no interval-preserving dimension exists, ordinary tensor
reshapes reject the partition trial; memdesc reshapes are allowed to become a
partition boundary, and the rewrite clones the full reshape before inserting a
`ttg.memdesc_subslice` on the reshaped result.

### Step 4: Rematerialization (`rewriteRematerializedOps`)

When an op is reached with **conflicting partition dimensions** (e.g., used by
two dots partitioning along different dims), it is marked for rematerialization.
Only `LocalAllocOp` and `arith::ConstantOp` are eligible. The op is cloned —
one copy per partition dimension — and users are updated to reference the
appropriate clone.

### Step 5: Rewrite (`sliceOp`)

For each partition offset (0 to `numPartitions - 1`):

1. Clone each partitioned op with types adjusted — divide
   `shape[partitionDim]` by `numPartitions`.
2. An op with `ttg.partition = [1, 2]` gets split into two copies: one with
   `[1]` and one with `[2]`.
3. Function arguments with `TensorDescType` have their block type sliced to
   match the partition factor.

`ttg.local_load` is shape-preserving and can be sliced with the same simple
clone/retype path as `ttg.local_alloc`. This matters for scale paths that load
packed scale tiles from SMEM, reshape them into logical `(BLOCK_MN,
BLOCK_K / scale_vec_size)`, and then allocate scale TMEM for
`ttng.tc_gen5_mma_scaled`.

For `scf.while`, slicing may append extra loop-carried values. The new
before-region argument is forwarded through `scf.condition` to create the
matching while result and after-region argument, and the after-region
`scf.yield` appends the sliced next-iteration value.

### Step 6: Cleanup (`doDeepCleanup`)

After rewriting, runs dead code elimination and removes orphaned operations
that are no longer referenced after partitioning.

## Key Design Points

### Partition Dimension Tracking

The partition dimension is tracked through shape-changing operations:
- `TransOp`: remaps dimension via permutation order
- `ExpandDimsOp`: shifts dimension index if expansion is before the partition
  dim
- `SplatOp`, `BroadcastOp`: partition dim propagates unchanged
- `MakeRangeOp`, `LoadOp`: stop — these produce fresh data

### Function Argument Slicing

When a `TensorDescType` function argument feeds a partitioned op, its block
type is sliced. The `funcArgPartitionDims` map tracks which arguments need
slicing and along which dimension.

If a `TensorDescType` function argument is also used as a loop init value, the
pass currently bails out. Updating both the function signature and the loop
carried argument/result types consistently requires additional handling.

### MMAv5 Scaled MMA

Scaled MMAv5 ops are partitioned through `MMAv5OpInterface` for the ordinary
MMA operands and accumulator. Scale operands are handled only for
`TCGen5MMAScaledOp`, because scale operands are not part of the generic MMAv5
interface.

The scale TMEM memdesc is treated logically as `(BLOCK_MN,
BLOCK_K / scale_vec_size)`. Therefore:

- M/output dim 0 partitioning slices `A` and `A_scale`; `A_scale` is sliced
  along logical scale dim 0.
- N/output dim 1 partitioning slices `B` and `B_scale`; `B_scale` is sliced
  along logical scale dim 0.
- Logical scale dim 1 is the K/scale-vector dimension and is not sliced by the
  normal M/N data partitioning path.

Scale buffers populated by `TMEMCopyOp` also need their copy source sliced to
match the destination scale rows. The copy source uses packed 32x128b chunks,
so the source partition dimension is the packed `repRows` dimension rather than
the logical scale dimension directly.

Scale buffers populated by `ttng.tmem_alloc %src` follow the same logical
partitioning rule. When the allocation result uses
`TensorMemoryScalesEncodingAttr`, slicing preserves that encoding instead of
trying to reinterpret it as a regular accumulator TMEM encoding. The compatible
distributed source layout is recomputed from the sliced scale memdesc type, so
the source tensor and destination scale TMEM agree after partitioning.

Example: a `BLOCK_M=256` scaled MMA with data partition factor 2 is rewritten
into two M partitions:

```mlir
%acc0, %tok0 = ttng.tmem_alloc
  : () -> (!ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
%acc1, %tok1 = ttng.tmem_alloc
  : () -> (!ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

%a0 = tt.descriptor_load ... : !tt.tensordesc<tensor<128x128xf8E4M3FN, ...>>
%a1 = tt.descriptor_load ... : !tt.tensordesc<tensor<128x128xf8E4M3FN, ...>>

%a_scale0 = ttng.tmem_alloc %scale0
  : (tensor<128x4xi8, ...>) -> !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>
%a_scale1 = ttng.tmem_alloc %scale1
  : (tensor<128x4xi8, ...>) -> !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>

ttng.tc_gen5_mma_scaled %a0, %b, %acc0[%tok0], %a_scale0, %b_scale, ...
  : !ttg.memdesc<128x128xf8E4M3FN, ...>,
    !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>,
    !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>, ...
ttng.tc_gen5_mma_scaled %a1, %b, %acc1[%tok1], %a_scale1, %b_scale, ...
  : !ttg.memdesc<128x128xf8E4M3FN, ...>,
    !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>,
    !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>, ...
```

### Interaction with Task IDs

Data partitioning operates **after** partition ID assignment. The offset parameter
selects which partition ID from the original array. This is how N consumer warp
groups each get their slice of the data.

## Regression Tests

- `test/Hopper/WarpSpecialization/ws_data_partition_scaled_memdesc_reshape_reproducer.mlir`
  covers the positive `BLOCK_M=256` scaled-MMA case. It checks that M
  partitioning produces two `128x256` accumulator TMEM allocations, two
  `128x4` A-scale TMEM allocations, and two `tc_gen5_mma_scaled` ops with
  `128x128` A operands.
- `test/Hopper/WarpSpecialization/ws_data_partition_tmem_encoding_bail.mlir`
  covers the bailout path where M partitioning would slice regular accumulator
  TMEM below its `blockM=128` encoding, including a scaled-MMA
  `memdesc_reshape` reproducer.
