# MetaAWS–NVWS bridge

[← Back to README](README.md)

## Contents

- [Purpose](#purpose)
- [Bridge sequence](#bridge-sequence)
- [Required input](#required-input)
- [Ownership mapping](#ownership-mapping)
- [Operation mapping](#operation-mapping)
- [Memory-plan mapping](#memory-plan-mapping)
- [Managed-allocation CFG locality](#managed-allocation-cfg-locality)
- [Errors](#errors)
- [Implementation and tests](#implementation-and-tests)

## Purpose

The bridge converts a completed MetaAWS ownership and storage plan into the
representation consumed by NVWS. `MetaToNVWSConvert` is a representation
boundary. It does not choose ownership, storage, schedules, synchronization, or
codegen.

On Blackwell, `TRITON_NVWS_USE_META=1` selects the bridge and takes precedence
over `TRITON_USE_META_WS=1`. This NVWS route is currently single-CTA.

[↑ Back to contents](#contents)

## Bridge sequence

```
1. PartitionSchedulingMeta
2. TaskIdPropagation
3. redundant-TMEM-zero normalization
4. Allocation materialization:
   - Default: WSBufferAllocation
   - With TRITON_NVWS_USE_META_NVWS_ALLOCAS=1:
     1. MetaToNVWSConvert promotes the NVWS physical WS root and converts
        ownership.
     2. NVWSInsertAllocas materializes allocations.

5. WSHoistTMEMStore
6. NVWSPackEpilogueSlices
7. WSMemoryPlanner
    - Extended to treat nvws.descriptor_load and nvws.descriptor_gather as
      buffer producers.
8. AnnotateTMAStoreWaits
9. TMA-store-wait validation
10. MetaToNVWSConvert:
    - Default: converts ownership and the completed MetaAWS memory plan.
    - With TRITON_NVWS_USE_META_NVWS_ALLOCAS=1: preserves ownership from
      step 4 and converts the completed MetaAWS memory plan.
    - Localizes eligible source-free managed groups to their unique top-level
      function CFG use block.
    - Validates the block-local representation required by NVWS planning.
    - Preserves partition ownership on external memdesc-alias chains that feed
      a WS scope, so aliases replayed inside that scope remain verifier-valid.
    - Promotes Meta's inner scheduling root to the outermost enclosing
      `scf.for` with the same `async_task_id` domain. The WS marker, tag,
      partition stages, and partition types move together; inner
      `tt.scheduled_max_stage`, `loop.stage`, and `loop.cluster` remain on the
      scheduled loop.

11. NVWSOrderBufferGroups
12. Handoff to NVWSInsertSemas
```

`NVWSInsertSemas` and the following NVWS codegen passes are outside the bridge.

[↑ Back to contents](#contents)

## Required input

For each function containing a `tt.warp_specialize` loop:

- every WS loop has `ttg.warp_specialize.tag`;
- each converted WS-scoped operation has a nonempty, nonnegative
  `async_task_id` or an already matching `ttg.partition`;
- planned storage may carry `buffer.id`, `buffer.copy`, `buffer.offset`,
  `buffer.tmaStaging`, `allocation.shareGroup`, and
  `allocation.reuseTarget`; and
- in the NVWS allocation variant, the second conversion keeps the ownership
  attributes produced by the first conversion and converts the completed
  MetaAWS memory-plan attributes into NVWS buffer representation.

A zero-operand `scf.yield` without its own assignment inherits its parent
region's owner. If the parent has no resolvable owner, conversion fails.
`ub.poison` requires no owner. Source-free allocations use the completion
rules below.

[↑ Back to contents](#contents)

## Ownership mapping

| Input | NVWS output | Rule |
|---|---|---|
| `async_task_id=[...]` | `ttg.partition=[...]` | Sort and remove duplicates. Reject an empty or negative assignment. A pre-existing partition must match. Remove `async_task_id` from WS loops and nested operations, plus external assignments consumed for result ownership. |
| Owners of values yielded by result `N` of `scf.for`, `scf.if`, or `tt.reduce` | `ttg.partition.outputs[N]` | Every result owner must also belong to the region operation. Existing outputs are preserved by a second conversion. |
| Async-token result ownership | Token result entry in `ttg.partition.outputs` | Infer it from token users or yielded token producers. |
| Meta scheduling root: `tt.warp_specialize`, tag, partition stages, and partition types | One NVWS physical WS root | Move the complete root bundle to the outermost enclosing `scf.for` with the same `async_task_id` domain. Preserve the inner Triton software-pipeliner schedule and result-ownership metadata. |
| Partition attributes outside the promoted `tt.warp_specialize` loop | No NVWS ownership, except associated external managed aliases | Remove partition/output/stage/tag attributes before rebuilding converted ownership. Preserve the partition and unique WS tag for an external managed alias that InsertSemas will replay inside the WS scope. |

Meta uses `tt.warp_specialize` to select the inner loop whose schedule it
analyzes, while its final specialization covers the enclosing task-bearing
loop nest. NVWS uses the same marker as its physical loop-level specialization
root. Before ownership conversion, the bridge therefore moves the WS-root
attribute bundle to the outermost enclosing `scf.for` with the same authored
task domain. This normalization is idempotent, so it is applied by both bridge
invocations when the NVWS allocation route is enabled.

The inner loop remains the schedule owner for the Triton software pipeliner. Its
`tt.num_stages`, `tt.scheduled_max_stage`, `loop.stage`, `loop.cluster`,
`ttg.partition`, and `ttg.partition.outputs` are not moved. A mismatched
enclosing task domain is rejected rather than treated as a specialization
root.

[↑ Back to contents](#contents)

## Operation mapping

| Input shape | NVWS output | Rule |
|---|---|---|
| `tt.descriptor_load` + same-owner `ttg.local_store` | `nvws.descriptor_load` writing the destination buffer | Compute `txCount` from the per-CTA byte count. Copy descriptor-producer attributes, not store attributes, and remove the store. |
| `tt.descriptor_gather` + same-owner `ttg.local_store` | `nvws.descriptor_gather` writing the destination buffer | Use the same ownership, destination, and attribute rules as descriptor load conversion. |
| `%t = ttng.async_tma_reduce` owned by `P`; `ttng.async_tma_store_token_wait %t` owned by `Q` | Same operations and token, with the wait owned by `P` | Change only the wait's ownership. Do not convert or lower either operation. Note: add `nvws.descriptor_reduce`, the counterpart of `nvws.descriptor_load` and `nvws.descriptor_gather`, to handle this directly. |
| Unassigned source-free `ttg.local_alloc` with exactly one assigned producer store | Allocation owned by that producer | Reject more than one assigned producer store because this repair recognizes only Meta's split of one sourceful allocation into one source-free allocation and one store. This is not a general restriction on stores. |
| Unassigned source-free `ttng.tmem_alloc` with an assigned direct TMEM store | Allocation owned by the first assigned direct store | Deterministic bookkeeping for the storage handle. Note: `PartitionLoops` expects every operation inside a `tt.warp_specialize` loop to have a partition assignment. |

[↑ Back to contents](#contents)

## Memory-plan mapping

| Planned input | NVWS output | Rule |
|---|---|---|
| Ordinary `buffer.id`, `buffer.copy`, `buffer.offset`, `buffer.tmaStaging`, and `allocation.shareGroup` | Same allocations and attributes | Preserve the completed memory plan. The bridge does not collapse allocations or create views. |
| Repeated ordinary SMEM `buffer.id` under the default policy | Same allocations and IDs | Preserve every allocation. InsertSemas receives one ordinary same-ID group. |
| Repeated `buffer.id` on allocations without `allocation.reuseTarget`, under allocation algorithm 0 or explicit circular reuse | Same allocations plus `buffer.circular` and ordinal `buffer.start` | Preserve each logical ring entry. Apply only to groups with at least two entries. |
| `buffer.id=M`, positive `buffer.copy=C`, and valid `allocation.reuseTarget=N` | Same allocation with `buffer.id=N`, `buffer.copy=C`, and `buffer.start=k`; no `reuseTarget` | Assign `k` in allocation order for original ID `M`, modulo `C`. Require an earlier non-staging allocation with `buffer.id=N` in the same block, matching encoding, memory space, and element type, and enough storage for all `C` copies of the allocation. |
| Missing or incompatible nonnegative reuse target | Original allocation and memory plan without `reuseTarget` | Preserve the original `buffer.id` and other attributes. A negative target is an error. |

`tt.smem_alloc_algo` and `tt.smem_circular_reuse` select whether repeated SMEM
IDs on allocations without `allocation.reuseTarget` receive circular
annotations. The converter consults these attributes but does not remove them.
Reuse-target translation is independent of that policy.

The bridge never materializes a reuse view. Valid reuse can intentionally
produce one same-ID SMEM group in which the allocation supplying storage has
`buffer.copy=1` and the allocations translated from `allocation.reuseTarget`
retain their larger `buffer.copy` and assigned `buffer.start` values.
InsertSemas consumes those annotations when it materializes required backings
and views.

[↑ Back to contents](#contents)

## Managed-allocation CFG locality

Meta buffer allocation may hoist a source-free managed allocation to the
function entry block even when every access is in a different top-level
function CFG block, such as the continuation of an early-return branch.
`NVWSInsertSemas` models structured control flow recursively, but it does not
model dataflow between top-level function CFG blocks.

After translating the completed buffer plan, `MetaToNVWSConvert` localizes each
eligible managed allocation group. It follows every allocation result,
including TMEM tokens, and supported memdesc alias chains to determine the
unique top-level function-body block containing the group's uses. If all uses
resolve to one block, it moves the complete source-free group to the beginning
of that block while preserving member order and all memory-plan attributes.
Nested `scf` regions belong to their enclosing top-level function block;
allocations are not sunk into a loop or conditional region.

The converter then runs the strict locality validator exposed by
`MetaToNVWSConvert.h`. `NVWSOrderBufferGroups` and `NVWSInsertSemas` call the
same validator at their boundaries. After it succeeds, both discover groups
and construct access DAGs independently for each top-level function block;
OrderBufferGroups also reorders within each block independently. InsertSemas
accumulates the validated block-local DAGs for its later synchronization
schedule and one function-atomic IR emission. Neither pass moves managed
allocations between blocks or infers cross-block storage flow.

Every allocation in a managed group and its complete memdesc/token use closure
must resolve to one top-level function block. A group spanning blocks, a
cross-block alias or access, or storage forwarded through a function CFG block
argument is rejected. Cases that cannot be localized are not guessed or
cloned; conversion fails before semaphore planning.

[↑ Back to contents](#contents)

## Errors

Missing ownership, conflicting ownership, an owner outside its region, a
missing WS loop tag, a missing or ambiguous external WS-tag association,
mismatched descriptor/store owners, and invalid negative reuse targets are
errors. Mismatched enclosing task domains and multiple Meta scheduling roots
that collide on one promoted NVWS root are rejected. An external managed alias
that feeds a WS scope may not also retain a root-scope use. A managed
allocation group spanning function CFG blocks, cross-block memdesc/token flow,
or storage carried by a function CFG block argument is also an error.

[↑ Back to contents](#contents)

## Implementation and tests

| Resource | Coverage |
|---|---|
| [`MetaToNVWSConvert.cpp`](../third_party/nvidia/lib/Dialect/NVWS/Transforms/MetaToNVWSConvert.cpp) | WS-root promotion, ownership, external managed aliases, descriptor and allocation completion, memory-plan conversion, managed-group localization, and locality validation |
| [`MetaToNVWSConvert.h`](../third_party/nvidia/lib/Dialect/NVWS/Transforms/MetaToNVWSConvert.h) | Locality-validator contract called by NVWS planning boundaries |
| [`AutomaticWarpSpecialization.cpp`](../lib/Dialect/TritonGPU/Transforms/WarpSpecialization/AutomaticWarpSpecialization.cpp) | Bridge selection and conversion points |
| [`InsertAllocas.cpp`](../third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertAllocas.cpp) | Optional allocation route |
| [`meta_to_nvws_convert_partitions.mlir`](../test/NVWS/meta_to_nvws_convert_partitions.mlir) | Ownership, result-owner mapping, external-alias ownership, nested WS-root promotion, and conversion idempotence |
| [`meta_to_nvws_convert_errors.mlir`](../test/NVWS/meta_to_nvws_convert_errors.mlir) | Rejected ownership, mismatched promoted task domains, and cross-CFG managed-storage input |
| [`meta_to_nvws_convert_descriptor.mlir`](../test/NVWS/meta_to_nvws_convert_descriptor.mlir) | Descriptor/store mapping |
| [`meta_to_nvws_convert_buffer_plan.mlir`](../test/NVWS/meta_to_nvws_convert_buffer_plan.mlir) | Allocation preservation, circular mapping, reuse-target translation, CFG localization, and conversion idempotence |
| [`meta_to_nvws_convert_tma_reduce_wait.mlir`](../test/NVWS/meta_to_nvws_convert_tma_reduce_wait.mlir) | Direct TMA-reduce wait ownership |
| [`insert_semas_local_mixed_copy_reuse.mlir`](../test/NVWS/insert_semas_local_mixed_copy_reuse.mlir) | Same-ID SMEM backing and view materialization |
| [`meta_nvws_automatic_warp_specialization.mlir`](../test/NVWS/meta_nvws_automatic_warp_specialization.mlir) | Integrated bridge route |

[↑ Back to contents](#contents)
