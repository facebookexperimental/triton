# TMA operand loads

Scope: how GEMM operands (A, B) reach SMEM under Meta WS (`TRITON_USE_META_WS=1`) on Blackwell. Covered here: descriptor loads versus pointer loads, operands consumed by more than one partition, and `auto_tma` promotion. Epilogue and prologue side inputs loaded with TMA are covered in [epilogue_prologue_tma_loads.md](epilogue_prologue_tma_loads.md). Operand ring depth is covered in [tiles_and_stages.md](tiles_and_stages.md).

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
| Pointer `tl.load` feeding `tl.dot` in a WS loop | requires | A TMA descriptor (or `auto_tma`). Only descriptor loads and gathers go to the load partition, so a pointer load gets no producer pipelining. A pointer-loaded B under 2-CTA is a compile error. | `PartitionSchedulingMeta.cpp`, `categorizeLoads` |
| `multicast=True` on a descriptor load, or the `multicast` compile option | silently ignored | `TMAMulticast.cpp` only attaches `tt.multicast_axes`. No TMA lowering reads that attribute, so no `.multicast::cluster` copy is emitted. | `TMAMulticast.cpp` (sets `kMulticastAxesAttrName`); `WSLowerMem.cpp`, `optimizeTMALoads` (no multicast operand) |
| `auto_tma` on a load with a non-rectangular mask (`other=0` plus an extra predicate), a real `% arg` wraparound, or a mask bound that is not a bare kernel argument | wrong answer (reproduced) | The promotion drops the load mask. See [hazards.md](hazards.md#auto_tma-promotion-drops-the-load-mask). | `PromoteLoadToTMA.cpp`, `isTMAEligible`, `match1DOffset`, `findShapeArgForDim` |
| `auto_tma` on a load whose strides or base pointer are not provably 16-byte aligned, or whose `other` is non-zero | silently ignored | Promotion is skipped and the pointer load stays. | `PromoteLoadToTMA.cpp`, `isTMAEligible`, `dimStrideIs16BAligned`, `argIs16BAligned` |

## Mechanism

**Single-use operand: TMA writes straight into the MMA buffer.**
- **What changes:** if a descriptor load's only user is the MMA operand's `local_alloc` and its encoding is TMA-writable, TMA writes directly into that buffer.
- **What it relieves:** no register traffic, no extra SMEM, no consumer-side release barrier.
- **What it costs:** one SMEM buffer per copy, sized from the descriptor block (BLOCK_M x BLOCK_K for A, BLOCK_N x BLOCK_K for B), not from K.
- Code: `WSLowerMem.cpp`, `doConvertDescriptorLoadsToNVWS` and `getTMAWritableEncoding`.

**Operand also read in registers (multi-consumer) pays for an extra buffer.**
- **What changes:** any second user (e.g. a row reduction over A alongside the MMA) gives the load its own NVMMA landing buffer, read by each register consumer with a `local_load`. The MMA's copy is then fed from registers: for B (and for A when it cannot be promoted) a separate SMEM buffer written with generic stores in a compute partition; on sm100 a register-sourced A is normally promoted to TMEM instead (`PromoteLHSToTMem.cpp`, `LHSToTMem`), so A needs no second SMEM buffer.
- Same path for an operand cast in-kernel after a TMA load, e.g. fp32 loaded and truncated to bf16 for the MMA.
- **What it costs:** for B, two SMEM buffers per stage; a full register round trip; a partition-wide barrier on every release (next item).
- Hazards: the register read of the TMA tile triggers an unreproduced code-inspection hazard ([hazards.md](hazards.md#tma-tile-read-into-registers-released-without-a-proxy-fence)); under 2-CTA a register-modified B is not split and gives a wrong answer ([hazards.md](hazards.md#2-cta-with-an-unsplit-b)).
- Code: `WSLowerMem.cpp`, `doConvertDescriptorLoadsToNVWS` (register-consumed path); `WSCodePartition.cpp`, `isChannelAnchorOp` (a `local_alloc` with a register source is a channel producer).

**Each register-side release is a partition barrier.**
- **What changes:** consumer release and producer commit both lower to a non-per-thread mbarrier arrive: a `bar.sync` across the issuing partition, then a leader-thread arrive.
- **What it costs:** one partition-wide barrier per release (per K step for an operand ring) for every register-consumed TMA buffer.
- Code: `WSLowerToken.cpp`, `processConsumerReleaseOp` and `processProducerCommitOp`; `BarrierOpToLLVM.cpp`, `ArriveBarrierOpConversion` (`perThread=false` path).

**Pointer-loaded operands get no producer pipelining.** Only `DescriptorLoadOp`/`DescriptorGatherOp` go to the load partition; a pointer load issues and waits inside the compute partition that uses it, and its SMEM copy becomes a register-written channel. Code: `PartitionSchedulingMeta.cpp`, `categorizeLoads`.

**`auto_tma` turns eligible pointer loads into descriptor loads before warp specialization**, which then take the paths above. Eligibility, host versus device descriptors and silent fallbacks: [AutoTMAPromotion.md](../AutoTMAPromotion.md). The global `knobs.nvidia.auto_tma` is not in the compile cache key, so set `auto_tma=` per launch when comparing configs.

**Out-of-bounds TMA reads return the descriptor's padding value.** With the default `padding="zero"`, operand tiles past M, N or K read zeros, so M/N/K tail masks on TMA operands are unnecessary for correctness; the output tail is discarded by the TMA store's bounds. Code: `python/triton/tools/tensor_descriptor.py`, `TensorDescriptor`.

## Interactions

See [interactions.md](interactions.md). Pairs this file expects to hold:
- **TMA loads x 2-CTA:** B is split across the CTA pair only when it is descriptor-loaded.
- **TMA loads x data partitioning:** `auto_tma`-promoted loads compose with `data_partition_factor=2`.
- **TMA loads x CLC / dynamic persistence:** the tile-id owner is the partition that holds the TMA loads.
- **TMA loads x multicast:** multicast is silently ignored.
- **Register-modified operand x 2-CTA:** A is promoted to TMEM and is exact; B is left unsplit and gives a wrong answer (see the hazard below).

## Hazards

- [TMA tile read into registers, released without a proxy fence](hazards.md#tma-tile-read-into-registers-released-without-a-proxy-fence)
- [2-CTA with an unsplit B](hazards.md#2-cta-with-an-unsplit-b)
- [auto_tma promotion drops the load mask](hazards.md#auto_tma-promotion-drops-the-load-mask)

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
