# Packing merged-epilogue slices

[← Back to README](README.md)

## Contents

- [Target shape](#target-shape)
- [Exact eligibility contract](#exact-eligibility-contract)
- [Slice construction](#slice-construction)
- [Movement legality](#movement-legality)
- [Bridge placement](#bridge-placement)
- [Preserved metadata and output](#preserved-metadata-and-output)
- [Conservative failure behavior](#conservative-failure-behavior)
- [Test example](#test-example)
- [Code and test map](#code-and-test-map)

`NVWSPackEpilogueSlices` is a narrow scheduling optimization in the
[MetaAWS–NVWS bridge](meta-aws-nvws-bridge.md). It packs the producer slice for
the first of two data-partitioned epilogue outputs next to its shared-memory
store. In the generated pattern targeted by the pass, this ends the first large
register live range before the remaining second-slice tail instead of keeping
both outputs live until a pair of adjacent stores. The matcher does not prove
that effect for arbitrary accepted IR or search for later uses of the value.

The pass is registered as `nvws-pack-epilogue-slices` on `ModuleOp`. It has no
options. The production MetaAWS–NVWS bridge invokes it; the default NVWS route
does not.

## Target shape

The optimization models the two-way merged epilogue produced for the current
data-partitioned kernels. In schematic form, allocation materialization can
leave this order after the last inner loop:

```text
first output, step A
second output, step A
first output, step B
second output, step B
local_store first output -> first staging buffer
local_store second output -> second staging buffer
TMA store first staging buffer
TMA store second staging buffer
wait first TMA token
wait second TMA token
```

The pass changes only the first producer/store slice:

```text
first output, step A
first output, step B
local_store first output -> first staging buffer
second output, step A
second output, step B
local_store second output -> second staging buffer
TMA store first staging buffer
TMA store second staging buffer
wait first TMA token
wait second TMA token
```

In the intended generated order, once the first value has been stored, the
second slice is already the natural tail. The current implementation therefore
packs only the first slice; it does not independently reschedule the second one
or verify that all second-slice producers follow the first slice's anchor.

[↑ Back to contents](#contents)

## Exact eligibility contract

The pass walks every `scf.for` in the module. A loop is considered only when
all of the following structural conditions hold:

1. The loop has `tt.warp_specialize`.
2. At least one of `tt.merge_epilogue` or
   `tt.merge_epilogue_to_computation` is present. The implementation tests
   attribute presence, not a Boolean value stored in the attribute.
3. `tt.data_partition_factor` is an integer attribute whose value is exactly
   2.
4. The loop body has at least one directly nested `scf.for`. If it has more
   than one, the last direct child in block order is selected. The candidate
   epilogue segment consists only of the direct operations after that loop and
   before the outer loop's terminator.
5. The segment contains exactly two
   `ttng.async_tma_copy_local_to_global` operations.

Each of those two TMA stores must then satisfy the same buffer and completion
shape:

- its source value is defined directly by a `ttg.local_alloc`; views and other
  aliases are not followed;
- the allocation is source-free and is a shared-memory allocation;
- exactly one `ttg.local_store` in the candidate segment writes that exact
  allocation value, and that local store precedes the corresponding TMA store;
- the TMA token has exactly one use;
- that user is a `ttng.async_tma_store_token_wait` in the outer loop body and
  after the TMA store; and
- the wait has no ordinary barriers, barrier predicates, NVWS tokens, or NVWS
  token indices.

The two matching local stores are paired in TMA issue order. The first matching
local store must precede the second matching local store, so producer/store
order agrees with externally visible TMA-store order. A different number of
TMA stores, reuse of a staging buffer, multiple stores to either buffer, an
aliased buffer operand, or a richer TMA-wait form makes the loop ineligible.

[↑ Back to contents](#contents)

## Slice construction

For an eligible pair, the first `ttg.local_store` is the slice endpoint. The
pass obtains its backward SSA slice with these limits:

- only direct operations in the post-inner-loop segment may enter the slice;
- block arguments terminate traversal; and
- values captured from above the segment do not pull their producers into the
  slice.

The endpoint store is added explicitly. Every other slice operation must have no
regions, must not be a terminator, and must either be memory-effect-free or
report only `MemoryEffects::Read` effects. Thus the endpoint's write to the
selected shared-memory staging allocation is the only write moved as part of
the slice. The operations are retained in their original block order, and the
endpoint must be the last operation in that order.

The earliest operation in the slice is kept fixed as an **anchor**. All later
slice operations, ending with the local store, are moved consecutively after
that anchor. Keeping the anchor fixed is important for tokenless memory
dependencies: a tokenless `ttng.tmem_load`, typically reading a source-free
TMEM allocation, can depend on a preceding TMEM store that materialized a loop
result even though no SSA token connects the two. The pass never moves the
anchor across that preceding store.

[↑ Back to contents](#contents)

## Movement legality

Before moving anything, the pass examines every operation strictly between the
anchor and the endpoint store. An intervening operation may be crossed when it
is:

- another member of the selected slice;
- memory-effect-free, including a region-bearing operation when MLIR's
  recursive effect query classifies it as free;
- an operation with no regions whose reported memory effects are all reads;
  or
- `tt.store`, the Triton pointer-store operation. This is an explicit
  exception based on the intended generated shape, where the slice reads TMEM
  and writes an independent shared-memory staging buffer while the relative
  order among global pointer stores is unchanged. The matcher does not enforce
  TMEM-only reads and does not perform a resource or alias query for this
  exception.

Packing is abandoned if the interval contains a region-bearing operation that
is not memory-effect-free, an unknown or non-read memory effect other than the
`tt.store` exception above, or any of these scheduling boundaries:

- `ttng.async_tma_copy_local_to_global`;
- `ttng.async_tma_reduce`;
- `ttng.async_tma_store_token_wait`;
- `ttng.wait_barrier` or `ttng.arrive_barrier`;
- `ttng.arrive_barrier_named` or `ttng.wait_barrier_named`;
- `ttng.async_copy_mbarrier_arrive`; or
- an operation named `gpu.barrier`.

This prevents the first producer slice from crossing synchronization, an
already issued TMA operation, or non-effect-free nested control flow. Moving
the selected operations earlier preserves their original relative order.
Operations not in the slice also retain their relative order, and the move
keeps the SSA producers at least as early as before. These are the
implementation's structural/effect checks; because the `tt.store` exception
has no alias query, they are not a general memory-safety proof for arbitrary
IR.

[↑ Back to contents](#contents)

## Bridge placement

The bridge places the pass in this sequence:

```text
allocation materialization
-> canonical MetaAWS TMEM-store hoisting
-> NVWSPackEpilogueSlices
-> canonical MemoryPlanner
```

It runs after allocation materialization because its contract is expressed in
terms of already explicit, source-free SMEM staging allocations, local stores,
lowered TMA stores, and their token waits. Both bridge allocation variants
produce that form before reaching the pass; the packing pass does not create
or infer communication buffers itself.

It runs after canonical MetaAWS TMEM-store hoisting so the redundant
loop-invariant accumulator initializer has already been moved and the
remaining in-loop TMEM memory order is settled. For remaining tokenless TMEM
load/store dependencies, the fixed-anchor rule described above preserves that
order.

It runs before `MemoryPlanner` because packing changes operation order and the
points at which register results are consumed by their staging stores. The
planner builds operation-order-based buffer liveness from the IR it receives;
placing this optimization first makes the packed epilogue, rather than the
old interleaved order, its planning input. The pass does not repair or
recompute an already authored physical buffer plan.

[↑ Back to contents](#contents)

## Preserved metadata and output

The transformation uses `moveAfter` on existing operations. It does not clone,
create, or erase operations and does not change operands, results, types,
locations, or attributes. In particular, it preserves:

- MetaAWS task assignments such as `async_task_id`;
- NVWS ownership such as `ttg.partition` and `ttg.partition.outputs`;
- warp-specialization tags and loop attributes; and
- allocation, buffer, staging, and reuse metadata already attached to the
  operations.

The containing block and loop are unchanged. TMA stores, token waits,
allocations, and the second local-store slice are not directly moved. The only
observable output is a new order for the movable portion of the first
producer/store slice.

This pass does not choose partitions, allocate or reuse SMEM/TMEM, change
buffer depth, add synchronization, reorder TMA issues, or handle arbitrary
epilogue factors. It is not a general epilogue scheduler: region-bearing
producers, tails containing non-memory-effect-free nested operations, and
patterns other than the exact two-store shape are outside its contract.

[↑ Back to contents](#contents)

## Conservative failure behavior

The optimization is best-effort. Every failed structural check, an unsupported
effect, a scheduling boundary, or failure to compute the backward slice leaves
that loop unchanged and returns success without a diagnostic. All checks for a
loop complete before its first move, so a rejected loop is not partially
packed. Other eligible loops in the same module can still be transformed.

The pass driver has generic failure plumbing, but `packLoopEpilogue` currently
turns all match and slice-analysis failures into successful no-ops. Malformed
IR can still be rejected by the normal MLIR verifier outside this
optimization.

[↑ Back to contents](#contents)

## Test example

[`test/NVWS/pack_epilogue_slices.mlir`](../test/NVWS/pack_epilogue_slices.mlir)
runs the standalone pass with:

```text
triton-opt test/NVWS/pack_epilogue_slices.mlir \
  --nvws-pack-epilogue-slices
```

Its `pack_two_slices` loop contains an inner loop followed by a TMEM store,
interleaved `%a0`/`%a1` and `%b0`/`%b1` computations, two local stores, two TMA
stores, and two plain token waits. The first local store's backward slice is
`%a0 -> %b0 -> local_store %b0`. `%a0` remains anchored after the TMEM store;
`%b0` and its local store move next to it. The checked result is:

```text
ttng.tmem_store ...
%a0 = arith.addf ...
%b0 = arith.mulf %a0, ...
ttg.local_store %b0, %buf0
%a1 = arith.addf ...
%b1 = arith.mulf %a1, ...
ttg.local_store %b1, %buf1
```

The test also checks that the prerequisite TMEM store remains before the
anchored producer. The TMA stores and waits remain after both local stores.

[↑ Back to contents](#contents)

## Code and test map

- [`PackEpilogueSlices.cpp`](../third_party/nvidia/lib/Dialect/NVWS/Transforms/PackEpilogueSlices.cpp):
  eligibility, slice construction, movement legality, and reordering.
- [`Passes.td`](../third_party/nvidia/include/Dialect/NVWS/Transforms/Passes.td):
  `nvws-pack-epilogue-slices` registration and dependent dialects.
- [`AutomaticWarpSpecialization.cpp`](../lib/Dialect/TritonGPU/Transforms/WarpSpecialization/AutomaticWarpSpecialization.cpp):
  production placement between MetaAWS TMEM-store hoisting and memory
  planning.
- [`WSMemoryPlanner.cpp`](../third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/WSMemoryPlanner.cpp):
  operation-order and liveness analysis that consumes the packed IR.
- [`pack_epilogue_slices.mlir`](../test/NVWS/pack_epilogue_slices.mlir):
  focused two-slice scheduling test.

[↑ Back to contents](#contents)
