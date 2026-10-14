# TMA Descriptors: Host-Side versus Device-Side

AutoWS is built around TMA. Loads become a load partition that issues
asynchronous TMA copies, and stores become TMA store staging (see
[Memory Lowering](MemoryLowering.md)). A kernel can supply the TMA descriptor
in two ways. Both reach AutoWS as `tt.descriptor_load` / `tt.descriptor_store`
on a `!tt.tensordesc` value, so partitioning, channels, and barriers are the
same. What differs is where the descriptor value comes from. That affects how
AutoWS passes that change a tile's block shape rewrite it, what the runtime
does at launch, and which memory the descriptor itself uses.

This is a kernel-authoring pattern, not a compile option. Pointer loads are a
third input form. They do not form a TMA load partition unless `auto_tma`
rewrites them (see [Auto-TMA Promotion](AutoTMAPromotion.md)).

| | Host-side | Device-side |
|---|---|---|
| Source | `triton.tools.tensor_descriptor.TensorDescriptor` passed as a kernel argument | `tl.make_tensor_descriptor(...)` in the kernel |
| IR at AutoWS time | Function argument of type `!tt.tensordesc<...>` | `tt.make_tensor_descriptor` result |
| Descriptor built | By the launcher, on every launch, from `tensordesc_meta` | In the kernel, by `TMACreateDescLowering` (`TMALowering.cpp`) after AutoWS |
| Memory for the descriptor | Kernel parameter | 128 B of global scratch per `tt.make_tensor_descriptor` (`GlobalScratchAllocOp`, `TMA_SIZE_BYTES`) unless a `desc_ptr` is supplied, plus a `ttng.tensormap_create` whose op scratch is 128 B of SMEM (`lib/Analysis/Allocation.cpp`) |

## Host-side descriptors

`getTensorDescMetadata` (exposed as `get_tensordesc_metadata`) reads the block
shape and encoding from the **final** IR type of each descriptor argument.
The compiler stores the result as `metadata["tensordesc_meta"]` after the final IR stages,
and the launcher encodes the `CUtensorMap` from it. The box the hardware uses
is therefore whatever block type the compiler left on the argument. That can
differ from the block shape written on the Python `TensorDescriptor`.

Two AutoWS-adjacent passes rely on this. They narrow a host descriptor by
**retyping the function argument in place** and updating the function
signature:

- **`Transform2CTALoads`** (`transformBLoad`) halves B's N block for 2-CTA
  MMAs. In the split-V path (`transformSplitBLoad`) it retypes the argument
  only if the descriptor load is its sole user. Otherwise it leaves the load
  untransformed.
- **`WSDataPartition`** records each sliced descriptor argument in
  `funcArgPartitionDims` and slices its block type. Two cases reject the
  partition scheme: two uses of one argument that need different partition
  dimensions, and a descriptor argument used as a loop init argument
  (`LoopLikeOpInterface` inits). Unpartitioned `tt.descriptor_store` users of
  a retyped argument are rewritten to match.

Because the argument is shared by every use in the kernel, retyping changes
the box for all of its loads and stores.

## Device-side descriptors

Passes that change a block shape **clone** the `tt.make_tensor_descriptor`
with the new result type instead of mutating a shared value:

- `Transform2CTALoads` clones it for the half-width B descriptor.
- `WSDataPartition` clones `tt.make_tensor_descriptor` and
  `ttng.reinterpret_tensor_desc` with sliced result types.

Other uses keep the original descriptor.

`tt.make_tensor_descriptor` is lowered by `TMACreateDescLowering`. It
allocates global scratch, emits `ttng.tensormap_create` and
`ttng.tensormap_fenceproxy_acquire`, and replaces the descriptor with
`ttng.reinterpret_tensor_desc`. That pass runs after AutoWS in `make_ttgir`
(`add_tma_lowering`). Two consequences follow for the memory planner:

- **Tensormap scratch is not reserved.** The auxiliary SMEM reservation
  (`estimateAuxiliarySmemBytes` in `WSMemoryPlanner.cpp`) counts 128 B per
  `ttng.tensormap_create`. When the planner runs, there are none yet: device
  descriptors are still `tt.make_tensor_descriptor`, so the reservation is
  zero. `estimateOpScratchBytes` explicitly skips tensormap creation. As a
  result, the 128-byte op scratch of each later `ttng.tensormap_create` is
  not part of the planned budget.
- **Captures.** Function arguments, including host descriptor arguments, are
  counted in the capture estimate. A device descriptor is not a function
  argument, so it is not counted there.

`auto_tma` with device mode, or with any op that needs a per-program base
offset, produces device descriptors. It hoists them above enclosing loops
while their operands stay loop-invariant.

## Resources

- **SMEM.** The operand rings and staging buffers are the same for both
  forms. Device descriptors add 128 B of SMEM op scratch per
  `ttng.tensormap_create` that the planner does not reserve.
- **Global memory.** Device descriptors use 128 B of global scratch each.
- **Launch.** Host descriptors are encoded on the host at every launch.
  Device descriptors are built and fenced in the kernel.
- **Registers, TMEM, warps.** No difference.
