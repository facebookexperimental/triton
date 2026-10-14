# TMA operand loads

Scope: how GEMM operands (A, B) reach SMEM under Meta WS (`TRITON_USE_META_WS=1`) on Blackwell. Covered here: descriptor loads versus pointer loads, operands consumed by more than one partition, and `auto_tma` promotion. Epilogue and prologue side inputs loaded with TMA are covered in `epilogue_prologue_tma_loads.md`. Operand ring depth is covered in [tiles_and_stages.md](tiles_and_stages.md).

## Knobs

| Knob | Set via | Legal values | Default | Priority | Applies when | Prereqs |
|---|---|---|---|---|---|---|
| Descriptor loads/stores | kernel pattern (`tl.make_tensor_descriptor` or host `TensorDescriptor`; `desc.load` / `desc.store`) | — | — | P0 | Every autoWS GEMM. Only descriptor loads go to the load partition; a pointer-loaded B under `two_ctas` is a compile error. | 16-byte-aligned base and strides; inner dim >= 16 B. |
| `auto_tma` / `auto_tma_device` | compile option, or env `TRITON_AUTO_TMA`‡ / `TRITON_AUTO_TMA_DEVICE`‡ | bool | False | P2 | 16-byte-aligned pointer loads with `other=0`; otherwise promotion is silently skipped. A non-rectangular mask, a real `% arg` wraparound or a non-argument bound gives a wrong answer. | Paused; mask-drop hazard open. |

## Pruning ruleset

| Condition | Status | Why |
|---|---|---|
| Pointer `tl.load` feeding `tl.dot` in a WS loop | requires | A TMA descriptor (or `auto_tma`). |
| `multicast=True` on a descriptor load, or the `multicast` compile option | silently ignored | `TMAMulticast.cpp` only attaches `tt.multicast_axes`. |
| `auto_tma` on a load with a non-rectangular mask (`other=0` plus an extra predicate), a real `% arg` wraparound, or a mask bound that is not a bare kernel argument | wrong answer | The promotion drops the load mask. |
| `auto_tma` on a load whose strides or base pointer are not provably 16-byte aligned, or whose `other` is non-zero | silently ignored | Promotion is skipped and the pointer load stays. |

## Mechanism

- **Single-use operand:** changes the load so that, if a descriptor load's only user is the MMA operand's `local_alloc` and its encoding is TMA-writable, TMA writes directly into that buffer (`WSLowerMem.cpp`, `doConvertDescriptorLoadsToNVWS` and `getTMAWritableEncoding`) -> relieves register traffic, extra SMEM and a consumer-side release barrier -> costs one SMEM buffer per copy, sized from the descriptor block (BLOCK_M x BLOCK_K for A, BLOCK_N x BLOCK_K for B), not from K.
- **Operand also read in registers (multi-consumer, or cast in-kernel after a TMA load, e.g. fp32 truncated to bf16 for the MMA):** changes the load so any second user (e.g. a row reduction over A alongside the MMA) gets its own NVMMA landing buffer read with a `local_load`, and the MMA's copy is fed from registers: for B (and for A when it cannot be promoted) a separate SMEM buffer written with generic stores in a compute partition, while on sm100 a register-sourced A is normally promoted to TMEM instead (`PromoteLHSToTMem.cpp`, `LHSToTMem`) so A needs no second SMEM buffer (`WSLowerMem.cpp`, `doConvertDescriptorLoadsToNVWS` register-consumed path; `WSCodePartition.cpp`, `isChannelAnchorOp`: a `local_alloc` with a register source is a channel producer) -> costs, for B, two SMEM buffers per stage, a full register round trip and a partition-wide barrier on every release (next bullet); hazards: the register read of the TMA tile triggers an unreproduced code-inspection hazard ([hazards.md](hazards.md#tma-tile-read-into-registers-released-without-a-proxy-fence)), and under 2-CTA a register-modified B is not split and gives a wrong answer ([hazards.md](hazards.md#2-cta-with-an-unsplit-b)).
- **Register-side release:** changes consumer release and producer commit to lower to a non-per-thread mbarrier arrive, a `bar.sync` across the issuing partition then a leader-thread arrive (`WSLowerToken.cpp`, `processConsumerReleaseOp` and `processProducerCommitOp`; `BarrierOpToLLVM.cpp`, `ArriveBarrierOpConversion` `perThread=false` path) -> costs one partition-wide barrier per release (per K step for an operand ring) for every register-consumed TMA buffer.
- **Pointer-loaded operands:** get no producer pipelining, since only `DescriptorLoadOp`/`DescriptorGatherOp` go to the load partition; a pointer load issues and waits inside the compute partition that uses it, and its SMEM copy becomes a register-written channel (`PartitionSchedulingMeta.cpp`, `categorizeLoads`).
- **`auto_tma`:** turns eligible pointer loads into descriptor loads before warp specialization, which then take the paths above (eligibility, host versus device descriptors and silent fallbacks: [AutoTMAPromotion.md](../AutoTMAPromotion.md)); the global `knobs.nvidia.auto_tma` is not in the compile cache key, so set `auto_tma=` per launch when comparing configs.
- **Out-of-bounds TMA reads:** return the descriptor's padding value; with the default `padding="zero"`, operand tiles past M, N or K read zeros, so M/N/K tail masks on TMA operands are unnecessary for correctness, and the output tail is discarded by the TMA store's bounds (`python/triton/tools/tensor_descriptor.py`, `TensorDescriptor`).

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
