# Memory and Layout Plan

Create `memory_plan.json` after instruction mapping and before allocating CUDA
storage. Derive it from source-operation semantics, mapped dynamic instances,
and dependency lifetimes. Do not copy FA4/FA3 layouts or ring depths.

## Required manifest structure

```json
{
  "schema_version": "twill-manual-cuda-memory-v1",
  "authoring_mode": "agent-assisted-manual-cuda",
  "target": {"gpu": "B200", "cuda_arch": "sm_100a"},
  "pipelined_ir": {
    "schema_version": "twill-pipelined-warp-ir-v1",
    "sha256": "<sha256>"
  },
  "status": "manual_completion_required",
  "capacities": {
    "register_words": 65536,
    "shared_bytes": 232448,
    "tmem_columns": 512,
    "physical_warps": 32
  },
  "allocations": [],
  "alias_sets": [],
  "descriptors": [],
  "resource_budget": {"planned": {}, "compiler_reported": {}},
  "lifetime_audit": [],
  "review": {}
}
```

Every manifest allocation must have a stable ID referenced by the mapping and
synchronization manifests. Capacity keys and values are fixed by the B200
machine model; they are not author-selected knobs. A completed audit rejects
empty allocations, compiler spills, incomplete lifetime audits, or a plan
whose status/review is not `approved`.

## Per-allocation contract

For every register fragment, SMEM buffer, TMEM allocation, barrier object, and
descriptor, record:

- semantic value and producing/consuming IR node IDs;
- storage kind and address space;
- element type, logical shape, physical shape, byte size, alignment, and base
  offset;
- layout, swizzle, interleave, transpose, and descriptor interpretation;
- owner CTA/group/warp and allowed remote access;
- number of slots, slot-index expression, and phase-index expression;
- per-slot size and total allocation size, where total size includes every
  slot rather than relying on an implicit multiplier;
- first write, publication event, last read/completion event, and release;
- alias-set ID and proof that overlapping dynamic lifetimes never share bytes;
- initialization and epilogue-drain requirements;
- source anchor and B200 hardware constraint used.

Treat barrier storage and tensor descriptors as real allocations. Include
padding, alignment gaps, and fixed runtime overhead in aggregate budgets.
The final bundle derives a reuse obligation for each mutable SMEM/TMEM/barrier
allocation and requires an allocation-specific release-to-producer edge in
`sync_manifest.json`. Its loop distance equals the slot count so the edge
protects reuse of the same physical slot.

## Lifetime derivation

For each dynamic value:

1. Start its lifetime before the first instruction that can write or launch an
   asynchronous write.
2. Keep it live until the documented completion witness for the last consumer,
   not merely until that consumer issues an instruction.
3. Apply dependency distance when relating producer and consumer iterations.
4. Expand prologue, at least two steady-state periods, and epilogue.
5. Derive the minimum safe ring depth from maximum overlapping live instances.
6. Add a release synchronization entry before any slot is reused by a producer.

`copies` is pipeline information, not automatically the correct slot count for
every value. A smaller ring requires a proof; a larger ring changes resource
use and must remain within the reviewed schedule/resource contract.

## Layout contract

For every TMA or TCGEN05 operand, record a bidirectional explanation:

- how logical tensor indices map to global memory;
- how a TMA box maps into SMEM bytes and swizzle banks;
- how SMEM/TMEM coordinates are interpreted by the exact MMA shape;
- how lane ownership maps register fragments to logical output elements;
- how epilogue stores reconstruct the public output layout.

Add executable or independently checkable layout tests where possible. A TMA
descriptor that moves the right byte count into the wrong swizzle is not
validated by barrier completion.

## Aliasing and pooling

For every alias set:

- list all members and exact byte/column ranges;
- show the last asynchronous read/write completion before ownership changes;
- identify the synchronization edge that protects the transition;
- include predicated and zero-iteration paths;
- prove that prologue fill and epilogue drain do not overlap the next use.

Do not pool memory solely because two operations are far apart in source. Use
dynamic pipeline lifetimes. TMEM reuse must account for pending TCGEN05 work;
SMEM reuse must account for pending TMA and MMA operand reads.

## Resource budget

Record planned and compiler-reported values for:

- registers per thread and per physical group;
- static and dynamic SMEM, including barriers and descriptors;
- TMEM columns and any CTA-group constraints;
- active warps, CTAs per cluster, and expected occupancy limits;
- descriptor count, barrier count, and named-barrier IDs.

If compiler allocation differs from the plan, update and re-review the plan.
Do not silently reduce stages, change group width, spill a scheduled value, or
alias buffers merely to make the kernel launch.

## Memory review gate

The reviewer must be able to answer:

- Can every mapping operand be located by allocation ID and byte/column range?
- Are all layouts compatible with the selected B200 primitive?
- Is every live interval closed by an actual completion witness?
- Are every slot and alias transition protected on all predicates?
- Do aggregate resources fit the exact B200 launch configuration?
- Are out-of-range tiles rejected or mapped safely before a TMA issue?
- Does the epilogue drain all pending stores and releases before exit?

Do not begin performance tuning until these answers are recorded.
