# gfx9 Codegen Patterns

Each profiled case on CDNA3/CDNA4 carries an `amdgcn_isa` profile entry: static compiler output for the dominant kernel. It reports register, scratch, and occupancy resources; the instruction mix of the hottest MFMA loop (`hot_loop`); the code around that loop that issues stores or direct-to-LDS loads (`boundary`); the source lines behind scratch reloads and sub-dword LDS reads (`source_locations`); and factual `findings`. Use it to name the mechanism behind a timing result and to check that a candidate changed that mechanism. Line numbers refer to the current `candidate.py`. Promotion still depends only on measured latency and correctness.

## Values Hoisted Out Of A Persistent Loop Spill

Symptom: `findings` reports scratch reloads at the loop boundary, each followed by `s_waitcnt vmcnt(0)` after global/buffer stores. `boundary_scratch_reloads` points at `tl.arange`-derived tensors used per tile, such as epilogue store offsets/masks or prologue load offsets. These depend only on lane IDs and constants, so they never change between tiles.

Cause: LLVM hoists loop-invariant tensors, such as per-lane load offsets computed from `tl.arange`, above the persistent tile loop. They are then live across the hot loop at the VGPR ceiling and spill. `vmcnt` counts both loads and stores, so the wait for each reload also waits for the previous tile's output stores. Store latency is then exposed at every tile boundary instead of overlapping the next tile's prologue.

Change: recompute those values inside the tile. Make them depend on a runtime value the compiler cannot fold, for example `opaque_zero = tile_idx // 0x7FFFFFFF` passed into the tile function. Multiply it by the alignment the offsets need, such as `+ opaque_zero * 256`, so `multiple_of`/`max_contiguous` facts and direct-to-LDS vector widths survive. Start with the shared lane-offset tensors that feed the most loads or stores; scalar bases are cheap to recompute. Removing one hoisted tensor frees registers, so the remaining reloads can move, and the tensor the findings name may not be the one whose removal helps most. Re-read `amdgcn_isa` after each attempt.

Confirm: `vgpr_spills` and `reload_drains_after_stores` drop, and outputs stay bitwise identical.

## Sub-Dword Operand Gathers Before MFMA

Symptom: `hot_loop.ds_read_by_width` contains `u8` or `u16` reads, `byte_assembly_valu` (`v_perm_b32`, `v_lshl_or_b32`) is nonzero, and `nop_cycles` is high, all feeding `v_mfma_scale` or packed MFMA operands. Block-scale factors are the common case: each scaled MFMA packs four E8M0 bytes from different row groups into one VGPR.

Cause: the bytes the instruction packs into one register are not adjacent in LDS, so the compiler gathers them one byte at a time and reassembles them.

Change: store those bytes adjacently, then read them back through a `tlx.local_reinterpret` view whose shared linear layout maps the logical `[rows, k]` tile onto the new byte order. That read becomes one `ds_read_b32` or `b64`. Direct-to-LDS copies cannot permute bytes, so reorder where the data is produced or already repacked. Host-side scale packing that copies anyway is a natural place. Copy the atom as a flat byte tile with a layout that gives each lane a 4- or 16-byte contiguous chunk. Specialize with a `constexpr` so callers that hand over an already-packed layout keep their zero-copy path. Adding a per-call host permute to a zero-copy path usually costs more than it saves. Keep every alternate layout bitwise equivalent, and keep the cold tail paths reading the same byte order.

Confirm: sub-dword reads and byte-assembly VALU in `hot_loop` drop to zero, `nop_cycles` falls, and outputs stay bitwise identical.

## Post-RA Scheduler Settings Are Tunables

`TRITON_DISABLE_POST_MISCHED=1` or the `("amdgpu-post-sched-strategy", "nop")` LLVM attribute copied from another kernel is a tuning choice, not an invariant. Kernels with explicit `warp_pipeline_stage` or `amd_sched_barrier` ordering may still run faster with the default post-RA scheduler. Toggle the environment guard and the attribute together as one config experiment, and keep `sched_barrier` fences in place.

## At The VGPR Ceiling

At 256 VGPRs with spills (2 waves/SIMD on CDNA3/CDNA4), any change that keeps more values live across the hot loop or the epilogue usually adds spills. This includes cross-tile prefetch, next-tile address precomputation, wider accumulator tiles, and extra pipeline buffers held in registers. Check the spill delta in `amdgcn_isa` before attributing a timing change to the intended overlap. Narrowing accumulators before the epilogue helps only if spills do not grow.

## Layout Conversions Inside Warp-Pipelined Loops

Do not `convert_layout` MFMA operands or scale tensors inside `tlx.warp_pipeline_stage` loops. The conversion routes through LDS scratch, which the phase-shifted wave groups share, and they race on it. Load from LDS directly into the consumer layout with `tlx.local_load(..., layout=...)`, or pin the layout with `tlx.require_layout`.
