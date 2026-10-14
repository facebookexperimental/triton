# Auto-TMA Promotion (`auto_tma`)

**Pass**: `triton-nvidia-promote-load-to-tma`
(`TritonNvidiaGPUPromoteLoadToTMAPass` in
`lib/Dialect/TritonNvidiaGPU/Transforms/PromoteLoadToTMA.cpp`)

**Frontend**: `third_party/nvidia/backend/compiler.py` (`CUDAOptions.auto_tma`,
`CUDAOptions.auto_tma_device`, `CUDABackend.make_ttir`), `python/triton/knobs.py`
(`knobs.nvidia.auto_tma`, `knobs.nvidia.auto_tma_device`)

Auto-TMA rewrites pointer-based `tt.load` operations whose addresses form a
tiled, affine block into the explicit TMA form:
`tt.make_tensor_descriptor` (or a host-built descriptor argument) plus
`tt.descriptor_load`. Downstream the descriptor load is handled exactly like
one written by hand with `TensorDescriptor`. Under AutoWS that means
`doConvertDescriptorLoadsToNVWS` turns it into a buffered NVWS load that the
load partition issues as an asynchronous TMA copy (see
[Memory Lowering](MemoryLowering.md)). In warp-specialized kernels the pass
also promotes eligible `tt.store` and relaxed `tt.atomic_rmw` add operations
to `tt.descriptor_store` / `tt.descriptor_reduce`.

The pass is not part of AutoWS itself. It runs at TTIR time, before AutoWS
sees the kernel, and decides whether AutoWS gets descriptor loads to work
with.

## Enabling it

| Control | How it is set | Default | Effect |
|---|---|---|---|
| `auto_tma` | Compile option: `triton.Config(..., auto_tma=True)` or a launch kwarg | `False` | Runs the pass for this compile. Part of `CUDAOptions`, so it is in the compile cache key. |
| `TRITON_AUTO_TMA` | Environment variable (`knobs.nvidia.auto_tma`) | off | Runs the pass for every compile in the process. `make_ttir` ORs it with `auto_tma`. |
| `auto_tma_device` | Compile option (launch kwarg). `triton.Config` has no field for it and does not forward it. | `False` | Forces device-built descriptors for every promoted op (see below). Only matters when the pass runs. |
| `TRITON_AUTO_TMA_DEVICE` | Environment variable (`knobs.nvidia.auto_tma_device`) | off | ORed with `auto_tma_device`. The pass also reads it directly with `std::getenv` and accepts only the exact value `1` on that path. |

`make_ttir` adds the pass only when `capability // 10 >= 9` (Hopper and
Blackwell). On older targets every control above is silently ignored.

`TRITON_AUTO_TMA` and `TRITON_AUTO_TMA_DEVICE` are **not** listed in
`CACHE_INVALIDATING_ENV_VARS` (`include/triton/Tools/Sys/GetEnv.h`), and the
knob values are not copied into `CUDAOptions`. Toggling either environment
variable therefore does not change the compile cache key. A kernel already in
the cache is reused as compiled, with or without promotion. The per-config
`auto_tma` / `auto_tma_device` options do change the key.

## Load eligibility

Phase 1 of `runOnOperation` walks every `tt.load` in the (single) kernel.
`decomposePointer` must recover, for every dimension, a block size, a tile
offset, and a stride from the address expression. The recognized forms are:

- **Contiguous**: `make_range(0, B) + splat(off)`.
- **Strided**: `(make_range(0, B) + splat(off)) * splat(stride)`, including
  the `offs[:, None] * stride` form applied after `expand_dims`.
- **Constant offset**: a dense constant offset, as produced by a subtiled
  epilogue slice.
- **Loop-advanced pointer**: a pointer carried by an `scf.for` and advanced by
  a constant each iteration, i.e. `ptrs += BLOCK_K * stride`.
- **Per-program scalar base offsets** such as `off_z * stride_z`. These force
  device mode.

`isTMAEligible` and `isTMAEligibleCommon` then require all of the following:

- The element type is TMA-encodable: 8/16/32/64-bit integers, f16, bf16, f32,
  f64, or fp8 e4m3/e5m2.
- Rank is 1 to 5.
- Exactly one dimension is contiguous, and its block is a positive multiple
  of 16 bytes.
- Every strided dimension's stride is a direct kernel argument.
- The load is not volatile.
- `other` is absent or a zero constant.

For rank 3 and above, only a single `expand_dims` level is peeled, so most
rank-3+ tiles stay pointer loads.

Each dimension also needs a global extent. Sources are tried in this order:

1. A `% arg` or `select(offs < arg, offs, 0)` clamp on the raw index
   (`match1DOffset`).
2. A comparison `offs < arg` tied to this dimension's tile offset,
   looked up first in the load's own mask and then anywhere in the kernel
   (`findShapeArgForDim`).
3. A per-tile remaining-extent bound `arange < E - tileOffset`
   (`findShapeArgFromTileBound`), or its loop-advanced form
   (`findShapeArgForLoopDim`).
4. For the innermost contiguous dimension only: the next-outer dimension's
   stride argument. This forces device mode.
5. For a whole, untiled dimension whose mask has no unattributed bound: the
   block size as a constant extent. This also forces device mode.

If any dimension has no extent, the load is left alone.

Finally, every strided dimension's stride argument and the base pointer must
carry a `tt.divisibility` making them provably 16-byte aligned. If the
alignment is not provable, for example because the argument was not
specialized as divisible by 16, the load is not promoted.

## Host recipes versus device descriptors

Each promoted op independently uses one of two descriptor sources.

- **Host recipe (default).** The pass appends a descriptor kernel argument
  tagged `tt.auto_tma` after the user arguments, and records a recipe in the
  module attribute `ttg.auto_tma_recipes`. The recipe holds the base pointer,
  shape and stride argument indices, block shape, and element type. The compile
  metadata carries it as `auto_tma_recipes`, and the launcher builds the
  `CUtensorMap` from the scalar arguments on every launch. The generated C
  launcher source (`make_launcher_src`) refuses to emit a launcher, with a
  warning, if a recipe's argument mapping is not one-to-one or exceeds the TMA
  descriptor/dimension caps.
- **Device descriptor.** A `tt.make_tensor_descriptor` is built in the kernel.
  It is hoisted above enclosing `scf.for` loops while its scalar base offsets
  stay loop-invariant. This mode is used when `auto_tma_device` is set, or
  automatically for any op that needs a scalar base offset, an outer-stride
  inner extent, or a constant extent. If a scalar base offset is defined inside
  the enclosing loop, the op is not promoted.

## Store and atomic-add promotion

Stores and atomics are promoted only when `kernelIsWarpSpecialized` finds an
`scf.for` carrying `tt.warp_specialize`. A warp-specialized `scf.while` written
with `tl.condition` does not count. On top of the load rules, a promoted store
or atomic needs:

- The contiguous dimension already innermost.
- No loop-advanced dimension.
- A mask that is exactly a rectangle, `AND_d(offs_d < E_d)`
  (`storeMaskIsRectangular`).
- For `tt.atomic_rmw`: an `add`/`fadd` with relaxed semantics and an unused
  result.

Store descriptors in device mode are deduplicated so that subtiled stores to
one output share a descriptor, which lets the memory planner fuse their
staging buffers ([TMA Store Wait Pipeline](TMAStoreWaitPipeline.md)).

## Resources

- **SMEM.** A promoted load lands in an NVMMA-swizzled SMEM buffer. Under
  AutoWS this becomes a channel buffer that the memory planner multi-buffers
  against `smemBudget`, the same as a hand-written descriptor load. A promoted
  store adds a TMA store staging buffer, whose copy count is set by Phase 3.7
  of the planner (and capped by `TRITON_WS_STAGING_COPIES`).
- **Registers.** Address tensors for promoted ops are no longer materialized
  per thread.
- **Warps/partitions.** Promotion enables the partition structure that
  depends on TMA: a load partition for descriptor loads and an
  `epilogue_store` partition for TMA stores (see
  [PartitionSchedulingMeta](PartitionSchedulingMeta.md)).
- **Kernel ABI.** Host mode adds one hidden descriptor argument per promoted
  op. Device mode adds in-kernel descriptor construction.

## Semantics the rewrite does not preserve

A promoted load returns the TMA box clipped to the descriptor's global
extents, with out-of-bounds elements zero-filled. The rewrite replaces the
`tt.load` and drops its mask. The mask is never re-applied, and unlike the
store path the load mask is not checked for rectangularity. The promoted load
therefore matches the original only when the original mask is exactly the
per-dimension boundary that the chosen extents encode. These cases are
promoted but read different values:

- **Extra mask predicates.** A load with `other=0` whose mask has a term
  beyond the per-dimension bounds. Examples: a causal term such as
  `offs_m[:, None] >= offs_n[None, :]`, or a lower bound `offs >= lo`.
  In-bounds elements that the original masked to `0` are now read from
  memory.
- **Wrap-around indexing.** `(arange + splat(off)) % arg` is peeled as an
  out-of-bounds idiom, with `arg` as the extent. Indices at or past `arg` read
  `0` instead of the wrapped element. This is equivalent only when those
  elements are later discarded, as with the `% M` / `% N` row and column
  clamps in matmul operand loads.
- **A bound taken from elsewhere.** If the load's own mask bound is not a bare
  kernel argument, the extent can come from the first matching `offs < arg`
  comparison anywhere in the kernel, or from the outer stride for the
  innermost dimension. Either can be wider than the load's real bound.

Nothing diagnoses these cases.

## Silent fallbacks

Every eligibility failure leaves the op as an ordinary pointer load or store
with no message. To see what was promoted, look for `tt.descriptor_load`,
`tt.descriptor_store`, and `tt.descriptor_reduce` in the TTIR, or check the
`tt.auto_tma` arguments and `auto_tma_recipes` metadata. The pass emits an
error only for a module containing more than one kernel function.
