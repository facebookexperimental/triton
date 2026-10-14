# Epilogue and prologue TMA loads

Scope: side inputs of a Meta WS (`TRITON_USE_META_WS=1`) GEMM on Blackwell that are loaded with TMA descriptors and then consumed in registers. Typical side inputs are:
- a bias, a residual or a scale read in the epilogue;
- an operand loaded wide and cast in-kernel before the MMA (cast prologue);
- an M/K pad folded into the descriptor bounds (pad prologue).

The MMA operands themselves are covered in [tma_loads.md](tma_loads.md). The output store is covered in [tma_stores.md](tma_stores.md).

## Knobs

| Knob | How set | One line | Existing doc |
|---|---|---|---|
| TMA side-input load | `desc.load(...)` for the side input in place of a pointer `tl.load` | Turns the side input into a planner-owned SMEM buffer that feeds registers. | [BufferAllocation.md](../BufferAllocation.md), section "Step 4: separateLocalAllocWithSrc"; [MemoryLowering.md](../MemoryLowering.md), section "Entry Points" |
| descriptor rank and box | descriptor shape, e.g. rank-1 `[BLOCK_N]` versus 2-D `[1, BLOCK_N]` for a bias | Sets the landing-buffer size per subtile. | none |
| descriptor `padding` | `TensorDescriptor(..., padding="zero"\|"nan")` | Out-of-bounds fill, `"zero"` by default. This is what makes a pad prologue exact. | none |
| `EPILOGUE_SUBTILE` | kernel constexpr | Splits the side input into per-subtile loads, each with its own landing buffer. | [epilogue.md](epilogue.md); [SubtileOperator.md](../SubtileOperator.md), section "Motivation" |

## Never try (any problem)

| Condition | Status | Why | Code ref |
|---|---|---|---|
| No K loop in the IR (K a compile-time constant <= BLOCK_K) with a TMA-loaded bias, or any `[M, N]` addend of the GEMM result | wrong answer (reproduced) | The addend is folded into the accumulator init, and its commit lands in the MMA partition. A runtime K launched with K <= BLOCK_K keeps its K loop and is not affected. See [hazards.md](hazards.md#single-k-tile-with-an-additive-epilogue-loaded-by-tma). | `Combine.cpp`, `CombineDotAddFPattern`, `CombineDotAddIPattern`; `WSCodePartition.cpp`, `insertAsyncComm` |
| A side-input landing buffer per subtile that is not a multiple of 128 bytes, at `EPILOGUE_SUBTILE >= 2` | requires | A single copy, or a box of at least 128 bytes: subtile buffers form a group that the planner may multi-buffer, and the planner never checks TMA alignment. See [hazards.md](hazards.md#tma-landing-buffer-smaller-than-128-bytes-multi-buffered). | `WSMemoryPlanner.cpp`, `increaseFusedEpilogueCopies` |
| A prologue on B under `two_ctas`: B TMA-loaded, then scaled or cast in registers before the dot | wrong answer (reproduced) | B is no longer a direct descriptor load, so the 2-CTA B split is skipped with only a debug log. A prologue on A is fine (A is promoted to TMEM). See [hazards.md](hazards.md#2-cta-with-an-unsplit-b) | `third_party/nvidia/hopper/lib/Transforms/Transform2CTALoads.cpp`, `runOnOperation` ("B not from descriptor load") |

## Mechanism

**Every TMA side input is staged through SMEM.**
- **What changes:** each `desc.load` of a side input becomes an `nvws.descriptor_load` into its own descriptor-layout SMEM buffer. Its register consumers read that buffer with a `local_load` tagged with the consumer partitions, and the buffer itself serves as the cross-partition channel.
  - The planner counts every such buffer against the SMEM budget and may give it several copies.
  - A pointer `tl.load` of the same input, issued in the partition that uses it, takes no planner SMEM.
- **What it relieves:** the load is issued asynchronously by the producer partition, off the epilogue's critical path.
- **What it costs:**
  - N side inputs cost N landing buffers x copies x the box size;
  - every release is a partition-wide barrier ([tma_loads.md](tma_loads.md#mechanism));
  - an unreproduced code-inspection hazard: the release back to the TMA producer emits no proxy fence ([hazards.md](hazards.md#tma-tile-read-into-registers-released-without-a-proxy-fence)).
- Code: `WSLowerMem.cpp`, `doConvertDescriptorLoadsToNVWS` (register-consumed path).

**A cast prologue keeps WS; on B it doubles the operand's SMEM.**
- **What changes:** an operand loaded wide (e.g. fp32) and cast for the MMA is register-consumed, so it gets a wide landing buffer. The narrow copy for the MMA goes to TMEM for A on sm100 (`PromoteLHSToTMem.cpp`, `LHSToTMem`), and to a second, register-written SMEM buffer for B (or for A when promotion is blocked).
- **What it costs:** for B, SMEM per stage is `BLOCK_N x BLOCK_K x (wide + narrow element size)`, which shrinks the legal stage or tile set. Under 2-CTA a B prologue is a wrong answer (see "Never try (any problem)").
- An MMA that consumes the wide type directly (e.g. TF32) needs only the landing buffer.
- Code: `WSLowerMem.cpp`, `doConvertDescriptorLoadsToNVWS`; `WSCodePartition.cpp`, `isChannelAnchorOp`.

**A pad prologue is exact.**
- **What changes:** with `padding="zero"`, TMA returns zeros past the descriptor's global extent. Zero-padding M or K can therefore be expressed by setting the descriptor shape to the unpadded extent, with no pad kernel and no extra buffer.
- **What it costs:** nothing beyond the normal operand load.
- **Limit:** this only replaces a pad. A prologue that computes anything else still needs a register path.
- Code: `python/triton/tools/tensor_descriptor.py`, `TensorDescriptor`.

## When to try it (depends on the problem)

- IF the kernel reads a TMA-loaded side input in registers, THEN prefer (ranks first; removes nothing) a pointer `tl.load` for that input, BECAUSE the TMA form's release has no proxy fence (code inspection; not reproduced in testing).
- IF the kernel has no K loop (K a compile-time constant <= BLOCK_K) and the epilogue adds a tensor to the GEMM result, THEN skip TMA for that tensor, BECAUSE the folded accumulator init lands in the wrong partition.
- IF a side input's per-subtile box is under 128 bytes (e.g. a bf16 bias slice of fewer than 64 elements), THEN skip TMA for it, BECAUSE its landing buffer can be multi-buffered at a misaligned offset.
- IF judging SMEM fit with TMA side inputs, THEN add one landing buffer per input per copy to the operand ring and staging, BECAUSE the planner allocates each one.
- IF a prologue is a pure zero pad of M or K, THEN prefer (ranks first; removes nothing) folding it into the descriptor extent, BECAUSE TMA zero-fill makes it exact and free.

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
