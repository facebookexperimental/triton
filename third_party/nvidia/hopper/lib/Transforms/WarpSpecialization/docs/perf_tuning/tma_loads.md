# TMA operand loads

Scope: how GEMM operands (A, B) reach SMEM under Meta WS (`TRITON_USE_META_WS=1`) on Blackwell. Covered here: descriptor loads versus pointer loads, operands consumed by more than one partition, and `auto_tma` promotion. Epilogue and prologue side inputs loaded with TMA are covered in `epilogue_prologue_tma_loads.md`. Operand ring depth is covered in [tiles_and_stages.md](tiles_and_stages.md).

## Knobs

| Knob | How set | One line | Existing doc |
|---|---|---|---|
| TMA descriptor load | `TensorDescriptor` (host) or `tl.make_tensor_descriptor` (device), then `desc.load(...)` | The only load form that the partition scheduler routes to the producer (load) partition. | [MemoryLowering.md](../MemoryLowering.md), section "Entry Points"; [BufferAllocation.md](../BufferAllocation.md), section "Step 4: separateLocalAllocWithSrc" |
| `auto_tma` | compile option `auto_tma=True` or env `TRITON_AUTO_TMA=1` (sm_90+) | Promotes eligible pointer `tl.load`s to host TMA descriptors before warp specialization. In a WS kernel it also promotes eligible stores. | [AutoTMAPromotion.md](../AutoTMAPromotion.md) |
| `auto_tma_device` | compile option or env `TRITON_AUTO_TMA_DEVICE=1` | Makes `auto_tma` build device-side descriptors instead of host recipes. Loads that a host recipe cannot express switch to device mode on their own. | [AutoTMAPromotion.md](../AutoTMAPromotion.md#host-recipes-versus-device-descriptors) |
| descriptor `padding` | `TensorDescriptor(..., padding="zero"\|"nan")` | Out-of-bounds fill value. The default is `"zero"`. | none |
| `desc.load(multicast=True)` / `multicast` option | kernel source / compile option | Plans multicast axes only. See "Never try (any problem)" below. | `docs/design/triton_tma_multicast.md` |

## Never try (any problem)

| Condition | Status | Why | Code ref |
|---|---|---|---|
| Pointer `tl.load` feeding `tl.dot` in a WS loop, without `auto_tma` | requires | A TMA descriptor: only descriptor loads and gathers are categorized as loads and placed in the load partition. A pointer load stays in a compute partition, and its value reaches the MMA through a register-to-SMEM copy. It has also been reported to assert or hang ("Pointer-loaded MMA operand" row in [TRACKER.md](TRACKER.md#reported-wrong-answers--hangs-needing-repro--code-root-cause)). | `PartitionSchedulingMeta.cpp`, `categorizeLoads` and the Phase 3 load scheduling in `getInitialSchedule` |
| `multicast=True` on a descriptor load, or the `multicast` compile option | silently ignored | `TMAMulticast.cpp` only attaches `tt.multicast_axes`. No TMA lowering reads that attribute, so no `.multicast::cluster` copy is emitted. | `TMAMulticast.cpp` (sets `kMulticastAxesAttrName`); `WSLowerMem.cpp`, `optimizeTMALoads` (no multicast operand) |
| `auto_tma` on a load with a non-rectangular mask (`other=0` plus an extra predicate), a real `% arg` wraparound, or a mask bound that is not a bare kernel argument | wrong answer | The promotion drops the load mask. See [hazards.md](hazards.md#auto_tma-promotion-drops-the-load-mask-from-code-inspection-no-reproducer-yet). | `PromoteLoadToTMA.cpp`, `isTMAEligible`, `match1DOffset`, `findShapeArgForDim` |
| `auto_tma` on a load whose strides or base pointer are not provably 16-byte aligned, or whose `other` is non-zero | silently ignored | Promotion is skipped and the pointer load stays. | `PromoteLoadToTMA.cpp`, `isTMAEligible`, `dimStrideIs16BAligned`, `argIs16BAligned` |

## Mechanism

**Single-use operand: TMA writes straight into the MMA buffer.**
- **What changes:** when a descriptor load's only user is the MMA operand's `local_alloc` and its encoding is TMA-writable, `doConvertDescriptorLoadsToNVWS` makes the TMA write directly into that buffer.
- **What it relieves:** there is no register traffic, no extra SMEM, and no consumer-side release barrier.
- **What it costs:** one SMEM buffer per copy, sized from the descriptor block (BLOCK_M x BLOCK_K for A, BLOCK_N x BLOCK_K for B), not from K.
- Code: `WSLowerMem.cpp`, `doConvertDescriptorLoadsToNVWS` and `getTMAWritableEncoding`.

**Operand also read in registers (multi-consumer) pays for two buffers.**
- **What changes:** when a descriptor load has any second user, for example a row reduction over A alongside the MMA, the load gets its own NVMMA landing buffer. Each register consumer reads it with a `local_load`.
  - The MMA's `local_alloc` is then fed from registers, so it becomes a separate SMEM buffer written with generic stores in a compute partition.
- **What it costs:**
  - two SMEM buffers per stage for that operand;
  - a full register round trip;
  - a partition-wide barrier on every release (next item).
  - The register-written buffer is also the trigger for two hazards: [hazards.md](hazards.md#tma-tile-read-into-registers-released-without-a-proxy-fence) and, under 2-CTA, [hazards.md](hazards.md#register-written-mma-operand-read-by-a-2-cta-mma).
- Code: `WSLowerMem.cpp`, `doConvertDescriptorLoadsToNVWS` (register-consumed path); `WSCodePartition.cpp`, `isChannelAnchorOp` (a `local_alloc` with a register source is a channel producer).
- The same path applies to an operand that is cast in-kernel after a TMA load, e.g. fp32 loaded and truncated to bf16 for the MMA.

**Each register-side release is a partition barrier.**
- **What changes:** a consumer release and a producer commit both lower to a non-per-thread mbarrier arrive. That arrive is preceded by a `bar.sync` across the issuing partition, and then a leader thread arrives.
- **What it costs:** every register-consumed TMA buffer adds one partition-wide barrier per release, i.e. per K step for an operand ring.
- Code: `WSLowerToken.cpp`, `processConsumerReleaseOp` and `processProducerCommitOp`; `BarrierOpToLLVM.cpp`, `ArriveBarrierOpConversion` (`perThread=false` path).

**Pointer-loaded operands get no producer pipelining.**
- **What changes:** the partition scheduler puts only `DescriptorLoadOp`/`DescriptorGatherOp` in the load partition. A pointer load runs in whichever compute partition uses it, and its SMEM copy becomes a register-written channel.
- **What it relieves:** nothing. **What it costs:** that load issues and waits inside the consumer partition.
- Code: `PartitionSchedulingMeta.cpp`, `categorizeLoads`.

**`auto_tma` turns eligible pointer loads into descriptor loads before warp specialization.**
- **What changes:** `PromoteLoadToTMA` runs before partition scheduling. It rewrites a pointer load whose address decomposes into an affine block offset and whose extents can be found into a descriptor load. That load then goes through the paths above.
  - It does this only when strides and the base pointer are 16-byte aligned and `other` is zero.
  - It also promotes stores, but only when the kernel has a `tt.warp_specialize` loop.
  - It switches to device-side descriptors when a per-program base offset or a constant extent is needed.
- Code: `PromoteLoadToTMA.cpp`; `third_party/nvidia/backend/compiler.py`, `make_ttgir` (pass ordering).

**Out-of-bounds TMA reads return the descriptor's padding value.**
- With the default `padding="zero"`, an operand tile that runs past M, N or K reads zeros. Masking M/N/K tails for TMA operands is therefore unnecessary for correctness. The output tail is discarded by the TMA store's bounds.
- Code: `python/triton/tools/tensor_descriptor.py`, `TensorDescriptor`.

## When to try it (depends on the problem)

- IF an MMA operand is pointer-loaded under Meta WS, THEN only try configs that load it with a TMA descriptor (or `auto_tma`), BECAUSE pointer loads are not routed to the load partition (`categorizeLoads`).
- IF an operand is read by both the MMA and register code, THEN count two SMEM buffers per stage for it when judging whether a tile and depth fit, BECAUSE the MMA copy is a separate register-written buffer.
- IF an operand is read by both the MMA and register code, THEN prefer moving the register consumer onto the accumulator or the epilogue when the math allows it, BECAUSE that removes the second buffer, the register round trip and the per-release partition barrier.
- IF tuning on Blackwell under Meta WS, THEN skip the `multicast` axis, BECAUSE it changes no generated code.
- IF `auto_tma` is enabled on a kernel whose loads use masks other than per-dimension `offs < extent` bounds, or that use `%` for real broadcasting, THEN skip `auto_tma` for that kernel, BECAUSE the promoted load drops the mask.

## Interactions

See [interactions.md](interactions.md). Pairs this file expects to hold:
- **TMA loads x 2-CTA:** B is split across the CTA pair only when it is descriptor-loaded.
- **TMA loads x data partitioning:** `auto_tma`-promoted loads compose with `data_partition_factor=2`.
- **TMA loads x CLC / dynamic persistence:** the tile-id owner is the partition that holds the TMA loads.
- **TMA loads x multicast:** multicast is silently ignored.
- **Register-consumed operand x 2-CTA:** see the hazard above.

## Hazards

- [TMA tile read into registers, released without a proxy fence](hazards.md#tma-tile-read-into-registers-released-without-a-proxy-fence)
- [Register-written MMA operand read by a 2-CTA MMA](hazards.md#register-written-mma-operand-read-by-a-2-cta-mma)
- [auto_tma promotion drops the load mask](hazards.md#auto_tma-promotion-drops-the-load-mask-from-code-inspection-no-reproducer-yet)

Reported but not yet located in code (rows in [TRACKER.md](TRACKER.md#reported-wrong-answers--hangs-needing-repro--code-root-cause)): pointer-loaded MMA operand; single-copy register copy of a multi-consumer operand with few K tiles.

## Not yet established

Tracked in [TRACKER.md](TRACKER.md).


- Whether a second load warp ever helps TMA issue. The observation so far is that one load warp saturates issue, and stalls come from slots freeing late.
- The cost of a duplicate TMA ring for a register consumer, compared with the shared-buffer path above.
- Whether an operand that hits in L2 still needs 2 or more stages.
- Narrower TMA boxes for one consumer versus fewer, larger boxes.
- Host-side versus device-side descriptor cost on GEMM.
- Whether WS is what makes TMA operand loads pay off at small K.
- The per-K-step cost of the register-side release barrier, and the gain from grouping releases to one per stage.
- Shared-operand performance once the register copy is folded away. That fold is not in the compiler.
