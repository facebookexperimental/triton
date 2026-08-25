# InsertAllocas

[← Back to README](README.md)

## Contents

- [Purpose](#purpose)
- [Input contract](#input-contract)
- [Algorithm](#algorithm)
- [Output contract](#output-contract)
- [Separation from synchronization](#separation-from-synchronization)
- [Code map](#code-map)

## Purpose

`NVWSInsertAllocas` converts cross-partition SSA communication into explicit
mutable SMEM or TMEM accesses. It inserts no synchronization. The pass runs
both on the default NVWS route and in the optional NVWS allocation variant of
the [MetaAWS–NVWS bridge](meta-aws-nvws-bridge.md). Both routes are summarized
in [NVWS-MetaAWS passes](nvws-aws-overview.md).

On the bridge route, an early `MetaToNVWSConvert` first promotes Meta's inner
scheduling root to the outermost enclosing task-bearing loop, then translates
completed MetaAWS task ownership into `ttg.partition` and
`ttg.partition.outputs`. Inner Triton software-pipeliner metadata remains on the
scheduled loop. `NVWSInsertAllocas` can then reuse the same ownership-driven
communication-buffer materialization as the default route. In particular, it
replaces cross-partition canonical `tt.descriptor_load` and
`tt.descriptor_gather` producers with destination-form
`nvws.descriptor_load` and `nvws.descriptor_gather` operations that write the
new communication allocations. The canonical MetaAWS MemoryPlanner recognizes
those NVWS operations as channel producers, so planning can continue after
this early conversion. The bridge document describes the complete pass
handoff.

[↑ Back to contents](#contents)

## Input contract

The pass expects warp-specialized loops with finalized partition ownership:

- producers and consumers carry `ttg.partition`;
- region results carry `ttg.partition.outputs`;
- scheduled operations retain `loop.stage` and `loop.cluster`;
- WS tags distinguish nested or post-loop warp-specialized scopes.

It handles loop iter-args, ranked-tensor WS-loop results, sourceful
`ttg.local_alloc` operations, descriptor loads/gathers, ordinary tensor/scalar
results (with scalar support limited to integer and floating-point types),
values produced by regular `tt.load`, and sourceful `ttng.tmem_alloc`
operations.

[↑ Back to contents](#contents)

## Algorithm

Before materializing SSA communication, a function containing a partitioned WS
loop normalizes each sourceful `ttng.tmem_alloc` whose allocation and uses span
more than one owner. Unpartitioned root code counts as an owner, so root plus
one partition also triggers normalization. Allocations confined to one owner
remain sourceful:

- a sourceless mutable backing is placed before the owning WS loop while
  remaining in the allocation's top-level CFG block;
- an explicit `ttng.tmem_store` remains at the original scheduled point;
- memory-plan attributes stay on the backing, while partition and loop
  schedule attributes stay on the store; and
- an initializer token seed is replaced with poison without breaking a
  loop-carried MMA token recurrence.

For each produced value:

1. Group uses by consumer partition after removing producer partitions.
2. Choose the communication memory. A produced `ttg.local_alloc` memdesc keeps
   its memory space. With `NVWS_USE_SSA_TMEM` set, a CUDA capability of 100+,
   and a rank-1 floating-point tensor of extent 64/128 and element width 16/32
   whose layout can be expanded to the rank-1 TMEM form
   (`getExpandedRank1TensorType`), the value may use TMEM; remaining tensors
   and supported integer or floating-point scalars use SMEM. Other scalar
   types are unsupported.
3. Allocate one mutable communication buffer before the owning WS loop.
4. Materialize the producer write:
   - descriptor operations write directly into the buffer;
   - regular loads and sourceful `ttg.local_alloc` operations store their value
     into it;
   - other tensors use an SMEM or TMEM store;
   - floating-point and integer scalars are splatted and stored in SMEM.
5. Materialize consumer-side accesses as needed for each consumer partition
   and rewrite the uses. Tensor and scalar values receive partition-local
   loads. A memdesc value (sourceful `ttg.local_alloc`) is instead rewired in
   one shot, replaying supported aliases as needed
   (`replaceUsesAndPropagateType`); this assumes a single consumer partition,
   checked only by a debug assertion.
6. Remove stale sourceful allocation/descriptor operations after rewiring.

Generated producer and consumer accesses carry the selected owner partition
and available `loop.stage`/`loop.cluster` annotation, plus a WS tag when the
rewritten path supplies one. The allocation operation itself deliberately
carries none of these annotations.

[↑ Back to contents](#contents)

## Output contract

The output contains explicit producer writes and consumer reads over mutable
allocations. Cross-partition TMEM communication has no sourceful allocations,
while single-owner TMEM remains unchanged. There are no `nvws.semaphore.*`
operations. The buffers keep their direct shape: this pass does not add a
leading copy dimension.

The later depth decision depends on the route. On the default route,
`InsertSemas` chooses and materializes the initial backing depth;
`LowerSemaphore` may subsequently widen eligible TMA-load-fed SMEM backings.
On the MetaAWS–NVWS bridge route, the canonical MemoryPlanner assigns
`buffer.copy` and related memory-plan metadata. The final
`MetaToNVWSConvert` preserves the established ownership and allocations. For
SMEM it adds circular annotations when selected and translates valid
`allocation.reuseTarget` into a shared `buffer.id` and `buffer.start` while
preserving `buffer.copy`. The resulting memory plan flows to InsertSemas,
which uses it when materializing backings and views for synchronized groups
and eligible tokenless local reuse groups.

[↑ Back to contents](#contents)

## Separation from synchronization

Upstream partitioning supplies the ownership sets for the original
computation. `NVWSInsertAllocas` elects the first producer partition as the
writer of each generated communication buffer and creates accesses for the
remaining consumer partitions; it does not repartition the original
computation. `InsertSemas` then derives the required handoffs and their
placement from those accesses without changing the computation partitions.

[↑ Back to contents](#contents)

## Code map

[`InsertAllocas.cpp`](../third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertAllocas.cpp):

- `createCommunicationBuffer`: memory-space and allocation choice.
- `normalizeSourcefulTmemAlloc`: hoisted TMEM backing and explicit initializer
  store.
- `createSemaphoreProducer` / `createSemaphoreConsumer`: retained historical
  names for producer/consumer access materialization.
- `insertSemaphoresForUses`: per-produced-value transformation.
- `NVWSInsertAllocas::runOnOperation`: selects allocation-only mode.

[↑ Back to contents](#contents)
