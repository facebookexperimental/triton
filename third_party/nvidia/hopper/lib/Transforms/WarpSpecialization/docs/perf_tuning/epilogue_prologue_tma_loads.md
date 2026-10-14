# Epilogue and prologue TMA loads

Scope: side inputs of a Meta WS (`TRITON_USE_META_WS=1`) GEMM on Blackwell that are loaded with TMA descriptors and then consumed in registers. Typical side inputs are:
- a bias, a residual or a scale read in the epilogue;
- an operand loaded wide and cast in-kernel before the MMA (cast prologue);
- an M/K pad folded into the descriptor bounds (pad prologue).

The MMA operands themselves are covered in [tma_loads.md](tma_loads.md). The output store is covered in [tma_stores.md](tma_stores.md).

## Knobs

| Knob | Set via | Legal values | Default | Priority | Applies when | Prereqs |
|---|---|---|---|---|---|---|
| Epilogue/prologue TMA loads | kernel pattern (`desc.load` of a side input read in registers) | — | — | P1 | Kernels with a side input read in registers (bias, residual, scale, cast prologue). | #3938 (no-K-loop TMA addend), #3939 (sub-128 B landing buffers), #3940 (proxy fence), all OPEN. A B-side prologue under `two_ctas` gives a wrong answer. |

## Pruning ruleset

| Condition | Status | Why |
|---|---|---|
| No K loop in the IR (K a compile-time constant <= BLOCK_K) with a TMA-loaded bias, or any `[M, N]` addend of the GEMM result | wrong answer | The addend is folded into the accumulator init, and its commit lands in the MMA partition. |
| A side-input landing buffer per subtile that is not a multiple of 128 bytes, at `EPILOGUE_SUBTILE >= 2` | requires | A single copy, or a box of at least 128 bytes: subtile buffers form a group that the planner may multi-buffer, and the planner never checks TMA alignment. |
| A prologue on B under `two_ctas`: B TMA-loaded, then scaled or cast in registers before the dot | wrong answer | B is no longer a direct descriptor load, so the 2-CTA B split is skipped with only a debug log. |

## Mechanism

- **TMA side input staged through SMEM:** each side-input `desc.load` becomes an `nvws.descriptor_load` into its own descriptor-layout SMEM buffer, which is also the cross-partition channel, and register consumers read it with a `local_load` tagged with the consumer partitions (`WSLowerMem.cpp`, `doConvertDescriptorLoadsToNVWS`, register-consumed path) -> the producer partition issues the load asynchronously, off the epilogue's critical path -> the planner counts each buffer against the SMEM budget and may give it several copies, so N side inputs cost N landing buffers x copies x the box size, and every release is a partition-wide barrier ([tma_loads.md](tma_loads.md#mechanism)).
- **Pointer load of a side input:** a pointer `tl.load` of the same input, issued in the partition that uses it, takes no planner SMEM.
- **Release without proxy fence:** an unreproduced code-inspection hazard: the release of a side-input buffer back to the TMA producer emits no proxy fence ([hazards.md](hazards.md#tma-tile-read-into-registers-released-without-a-proxy-fence)).
- **Cast prologue keeps WS:** an operand loaded wide (e.g. fp32) and cast for the MMA is register-consumed, so it gets a wide landing buffer; the narrow MMA copy goes to TMEM for A on sm100 (`PromoteLHSToTMem.cpp`, `LHSToTMem`) or to a second, register-written SMEM buffer for B (or for A when promotion is blocked), and an MMA that consumes the wide type directly (e.g. TF32) needs only the landing buffer (`WSLowerMem.cpp`, `doConvertDescriptorLoadsToNVWS`; `WSCodePartition.cpp`, `isChannelAnchorOp`) -> for B, SMEM per stage is `BLOCK_N x BLOCK_K x (wide + narrow element size)`, shrinking the legal stage or tile set; under 2-CTA a B prologue is a wrong answer (see "Pruning ruleset").
- **Pad prologue is exact:** with `padding="zero"`, TMA returns zeros past the descriptor's global extent, so zero-padding M or K is expressed by setting the descriptor shape to the unpadded extent, with no pad kernel and no extra buffer (`python/triton/tools/tensor_descriptor.py`, `TensorDescriptor`) -> costs nothing beyond the normal operand load; it only replaces a pad, and a prologue that computes anything else still needs a register path.

## Interactions

See [interactions.md](interactions.md). Pairs this file expects to hold:
- **Epilogue TMA load x epilogue subtiling:** one landing buffer per subtile, and the 128-byte limit applies per subtile.
- **Epilogue TMA load x single K tile:** wrong answer, see above.
- **Prologue x 2-CTA:** on A it is exact (A goes to TMEM); on B it is a wrong answer (unsplit B).

## Hazards

- [TMA tile read into registers, released without a proxy fence](hazards.md#tma-tile-read-into-registers-released-without-a-proxy-fence)
- [Single K tile with an additive epilogue loaded by TMA](hazards.md#single-k-tile-with-an-additive-epilogue-loaded-by-tma)
- [TMA landing buffer smaller than 128 bytes, multi-buffered](hazards.md#tma-landing-buffer-smaller-than-128-bytes-multi-buffered)
- [2-CTA with an unsplit B](hazards.md#2-cta-with-an-unsplit-b)

Reported but not yet located in code (row in [TRACKER.md](TRACKER.md#reported-wrong-answers--hangs-needing-repro--code-root-cause)): rank-1 TMA bias with a subtiled epilogue.

## Not yet established

Tracked in [TRACKER.md](TRACKER.md).

- Whether TMA-loading a 1 x N broadcast bias ever beats a vector pointer load.
- How fused-epilogue performance degrades with the number of TMA side inputs, and at what count it stops paying.
- The problem-size threshold, in bytes of the wide operand and in grid shape, above which a cast prologue beats a separate cast kernel.
- Whether fusing both operands' casts (wide A and B, narrow MMA) pays in general.
- Whether giving the prologue its own partition ever helps, given that it takes registers from the computation partition.
