# Blackwell Overlapping TMEM Accumulators

Use this guidance only for NVIDIA Blackwell persistent GEMM-like kernels whose accumulator occupies most of TMEM, so a second full accumulator buffer does not fit. The canonical case is a 128x256 fp32 accumulator (256 columns) plus block-scaled MMA scales. Consult `.claude/skills/tlx-api-reference/SKILL.md` for TLX APIs and CUTLASS `cutlass/detail/sm100_tmem_helper.hpp` (`make_sm100_accumulator` with `IsOverlappingAccum`) for the reference layout.

## Recognize The Bottleneck

Suspect a single-accumulator stall when a reference kernel with the same tile, cluster, and stage count reaches clearly higher tensor-pipe utilization in NCU (for example 82% versus 63%), and the kernel uses `NUM_TMEM_BUFFERS == 1`. With one accumulator, the MMA task waits for the whole epilogue: every subtile TMEM load, conversion, SMEM store, TMA store, and, when the epilogue does `store_wait(0)` before arriving, every global writeback. Check the TTGIR for `async_tma_store_wait {pendings = 0}` before each `arrive_barrier` on the TMEM-empty barrier.

## The Overlapped Layout

Place two accumulator slots in TMEM that overlap by exactly one epilogue subtile:

- slot 0 owns columns `[0, N)`;
- slot 1 owns columns `[N - N/S, 2N - N/S)`, where `S` is `EPILOGUE_SUBTILE`;
- for `N = 256, S = 4`, the slots are `[0, 256)` and `[192, 448)`, and `[448, 512)` remains for scales.

Alternate slots by tile parity (the TMEM barrier phase with one buffer). Drain slot 0 in reverse subtile order and slot 1 in forward order, so the first subtile read is always the shared columns `[N - N/S, N)`. Release the accumulator immediately after that first subtile is loaded into registers and set the TMEM-empty barrier arrive count to one release per CTA (`NUM_CTAS`) instead of `S * NUM_CTAS`. The next tile's MMA then overlaps the remaining `S - 1` epilogue subtiles. Prove that the next MMA writes only columns already drained by the current epilogue.

## Coordinate The Two Slots

Keep a single `tmem_full`/`tmem_empty` barrier pair (`NUM_TMEM_BUFFERS == 1`) for both slots. The MMA task and the epilogue warps must walk the same tile sequence and advance identical tile counters, so the barrier phase both orders the handshake and selects the slot without extra communication:

```
MMA task                                  epilogue warps
wait tmem_empty(phase ^ 1)      <---+
MMAs over K into slot[phase]        |
tcgen05_commit -> tmem_full  -------+-->  wait tmem_full(phase)
                                    |     load shared subtile first
                                    +---  arrive tmem_empty (once per tile)
wait tmem_empty -> next tile              load/store remaining S - 1 subtiles
MMAs into the other slot                  (overlaps the next tile's MMA)
```

Prove these invariants before promoting the change:

- The MMA task runs at most one tile ahead. Tile `t + 2` reuses tile `t`'s slot only after the release for tile `t + 1`, and an in-order epilogue issues that release only after draining every subtile of tile `t`. This program-order argument is what makes one barrier pair sufficient; do not move the epilogue's release later or reorder tiles across epilogue warps without re-deriving it.
- Because the producer is never more than one phase ahead, `tmem_full` cannot complete two phases before the epilogue waits, so phase parity cannot alias.
- The release happens exactly once per tile per CTA. A leftover per-subtile arrive breaks the arrive count and lets the MMA overwrite undrained columns.
- In 2CTA mode only the leader CTA issues the MMA; `tcgen05_commit(..., two_ctas=True)` signals `tmem_full` in both CTAs, and both epilogues arrive on the leader's `tmem_empty` with `remote_cta_rank=0`, so its arrive count is `NUM_CTAS`.
- A single scale TMEM buffer needs no barrier: `tcgen05.cp` and `tcgen05.mma` are issued in order by the same MMA thread, and the epilogue never reads scales.

## Express It In TLX

These constraints shape the implementation:

- `local_alloc` shapes must be powers of two, and the TMEM allocator derives column counts from linear layouts, so a 448-column allocation still costs 512 columns. Do not try to allocate `2N - N/S` columns directly.
- If the MMA keeps SMEM scales, the compiler inserts its own scale TMEM allocations after the accumulator, for every `async_dot_scaled` call site. With a 512-column accumulator view this exceeds TMEM (576 > 512), and a runtime slot branch duplicates the scale allocations.
- Use one `tlx.storage_alias_spec(storage=tmem)`: a shared group of a `(128, 2N)` fp32 accumulator view and a distinct group of placeholder buffers (`N`, `N/2`, `N/4` columns for `N = 256`) followed by the A and B scale TMEM buffers. Distinct children are aligned to their own column width, so slot 1 cannot be its own allocation at column 192; take both slots as `tlx.subslice` views of the 2N view.
- Stage scales explicitly with `tlx.tmem_copy(smem_scale, tmem_scale)` before each `async_dot_scaled`, as CUTLASS does per k-block; `tcgen05.cp` and `tcgen05.mma` issued by the same thread execute in order, so one scale TMEM buffer suffices. Match the compiler's scale TMEM shapes: `(BLOCK_M, bytes / BLOCK_M)` for A and `(BLOCK_N_PER_CTA, bytes / BLOCK_N_PER_CTA)` for B. In 2CTA mode the B source is `[1, 2, REP_K, 2, 256]` with a `(128, 16)`-style destination.
- TMEM subslice offsets must be compile-time constants. Write the offset expression inline at the call site; assigning it to a local variable (or a constexpr inside `static_range`) materializes a runtime value. Select the slot with a runtime `if` on the phase around separate constant-offset subslices.
- Inline the slot-specific MMA and epilogue code in the kernel body. Passing TMEM memdescs into helper `@triton.jit` functions can fail with a `tt.call` operand type mismatch after TMEM layout assignment.
- A non-per-thread `tlx.barrier_arrive` synchronizes the issuing warp group first, so arriving right after `tlx.local_load` of the shared subtile is safe for all epilogue warps.

## Validate And Measure

Run the full correctness suite on both 1CTA and 2CTA paths, multiple persistent waves, uneven and zero-size groups, and CUDA-graph replay. Measure with same-GPU A/B runs, at least three per variant, comparing ratios against the reference kernel from the same run; single runs of this class of kernel vary by several percent. In a block-scaled grouped GEMM whose 2CTA and 1CTA paths both ran a single accumulator, this change raised throughput about 3% geomean across twelve shapes, almost all of it on the 2CTA path.
