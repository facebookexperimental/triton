"""TLX MXFP4 GEMM for gfx950 -- 32x128 tile, the narrowest point of the curve.

Exists for exactly one shape: 256x4096x4096, where it is the only tile that fills
the machine. AITER selects its own 32x128 kernel there for the same reason.

    tile      grid on 256x4096x4096          ratio vs AITER
    32x128    (256/32) x (4096/128) = 256      **1.010**
    64x128    (256/64) x (4096/128) = 128       0.889
    128x128   (256/128) x (4096/128) = 64       ~0.6

Do NOT use it anywhere else -- it is correctly worse when the grid overshoots:

    shape            32x128    better tile
    256x4096x4096     1.010    (this one)
    256x8192x4096     0.762    64x128  (~0.95)
    512x4096x4096     0.780    64x128  (~0.98)
    512x8192x4096       --     128x128 (0.996)

Both off-target shapes put the grid at 512, i.e. two workgroups per CU with half
the arithmetic intensity each. The rule across all four tiles is simply: pick the
one whose grid lands on 256.

Derived from matmul_kernel_64.py; only the M-dependent layouts change
(shared_layout_a loses a row base, and the A-scale / accumulator / store layouts
each lose one M repeat). Everything on the B side -- b_operand_layout, the B
offsets, shuffle_b_weight, scale_mfma_layout_b -- is independent of BLOCK_M and
carries over verbatim. All four re-derived layouts were right first try and
compiled exact, using the conventions established in the 64x128 kernel; see its
"HOW THE LAYOUTS HERE WERE DERIVED" section before changing any of them.

HOW THE LAYOUTS HERE WERE DERIVED
  Every pinned layout in this file was read off the TTGIR, not guessed. Guessing
  stride tuples cost several cycles: the first attempt at the scale layouts
  assumed warpsPerCTA=[2,2] because that is what the accumulator's *blocked*
  layout shows, but the MFMA is [1,4], which inverts which operand is
  warp-broadcast (A, 8 e8m0 per lane) and which is warp-split (B, 4). The symptom
  was an opaque "size mismatch when packing elements for LLVM struct expected 4
  but got 8" out of make_llir.

  The reliable procedure, for any future tile shape:
    TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=... python <driver>
    grep '^#mma\|^#blocked\|^#linear' <dump>/*.ttgir
  then translate the #ttg.linear basis vectors to Shape:Stride. tlx.dump_layout
  on a loaded value is NOT enough -- it reports the blocked layout the compiler
  chose for the load, not the operand layout the MFMA requires.

EPILOGUE
  Pinning the accumulator before the narrowing (see store_layout_c /
  accumulator_layout) took the C-store round trip from 8 ds_write_b128 to 4 and
  s_barrier from 15 to 11, worth +0.1/+3.1/+1.2% on the three shapes above.

  The remaining 4 cannot be removed by pre-shuffling. Pre-shuffle works for the
  scales because they are *inputs*: the host permutes them outside the timed
  region and AITER's contract does the same. C is an output whose layout the
  caller dictates, so there is no free permutation to exploit. The round trip
  buys coalescing and is worth it: the MFMA hands each lane 4 contiguous columns
  of one row (lane l -> row l&15), which stores as 16 scattered 32-byte
  transactions per wave, versus 4 x 256 B after redistributing to store_layout_c.
  Reaching zero here needs an in-register cross-lane transpose (v_permlane),
  which is what AITER does -- not a host-side shuffle.

Original scaffold notes follow.

Why it exists: the 128x128 kernel is grid-limited on these shapes, launching
128 workgroups on 256 CUs (50% of the machine) for 256x8192x4096 and
512x4096x4096. AITER selects 64x128 for both, which puts the grid at exactly
256. Its 64x128 kernel is the same design point as its 128x128 one -- 4 waves,
all 160 KB of LDS, 512 VGPR, 16x16x128 MFMA -- just half the MFMAs per wave
(128 vs 256), i.e. only BLOCK_M changes.

What transfers from matmul_kernel_128.py and what does not:
  * smem_b is still (128, 128), so the padded shared layout carries over.
  * smem_a becomes (64, 128) = 2**13 elements and needs a 13-base padded
    layout, i.e. the same list minus [64, 0] from the m-bases.
  * A scales become a (64, 8) tile while B scales stay (128, 8), so they can
    no longer share one layout/offsets tensor, and the (64, 8) MFMA scale
    layout has no counterpart in the 256x256 kernel to copy. That derivation
    is the remaining work; until it is done the scale path here will
    scalarise the way the 128 kernel did before its layouts were pinned
    (which cost ~20%).

Original 128x128 header follows.

TLX MXFP4 GEMM for gfx950 -- 128x128 tile, for occupancy-starved shapes.

Companion to the 256x256 kernel in matmul_kernel.py. Same ABI, same intra-wave
structure; the only reason it exists is grid fill.

A 256x256 tile is one workgroup per CU, so on a 256-CU MI350X it needs
M*N >= 256*65536 to fill the machine. Below that it strands CUs: 256x4096x4096
launches 16 workgroups on 256 CUs. Measured against AITER on the four small
shapes the 256x256 kernel lands at 0.385-0.562 -- the loop body is the same one
that reaches parity on large shapes, it is simply running on 6-25% of the GPU.

AITER's tuned configs make the rule explicit: it selects 32x128 / 64x128 /
128x128 / 128x128 on those four shapes, i.e. whichever tile makes the grid come
out at exactly 256. This kernel supplies the 128x128 point of that curve:

    512x8192x4096   ->  4 x 64 = 256 workgroups   (exact fill)
    256x8192x4096   ->  2 x 64 = 128
    512x4096x4096   ->  4 x 32 = 128
    256x4096x4096   ->  2 x 32 =  64

The narrower tiles (and/or split-K) are what the two smallest shapes still need.

Differences from the 256x256 kernel, all consequences of BLOCK_N:
  * No left/right N split. BLOCK_N=128 is one MFMA group, so there is a single
    accumulator and one tl.dot_scaled per k-step instead of two.
  * No hand-pinned tlx.layout constants. Those in matmul_kernel.py are derived
    for the 256x256x256 tile specifically and do not carry over; this kernel
    lets the compiler choose, which is correct but not yet tuned. Use
    tlx.dump_layout() to read back what it picked before pinning anything.
  * Scales still take the LDS round-trip. The pre-shuffle + direct-to-VGPR path
    (shuffle_a_scale in matmul_kernel.py) needs its own derivation per tile.

Inputs use the same ABI as the Gluon a4w4 tutorial:
  * A: packed e2m1, shape (M, K // 2), K-contiguous
  * B: packed e2m1, shape (N, K // 2), K-contiguous; computes A @ B.T
  * scales: e8m0 uint8, shapes (M, K // 32) and (N, K // 32),
    contiguous along M/N so scale tiles are coalesced
  * C: bfloat16, shape (M, N)
"""

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

BLOCK_M = 32
BLOCK_N = 128
# 512, not 256: the loop steps 2 stages per body, so this is K=1024 per loop body
# against the same 4 s_barrier -- matching AITER's 64x128 kernel, which issues 64
# MFMAs per body where BLOCK_K=256 gives us 32. Barrier and loop overhead are
# amortised over twice the math. Costs 2x the LDS (~99 KB of 160 KB, still one
# workgroup per CU) and more VGPR, both of which were sitting unused: at these
# shapes the grid is exactly 256 = the CU count, so occupancy above 1 buys nothing.
#
# This is NOT the 4-stage K=1024 variant that regressed the 128x128 kernel 33%.
# That one added buffers (4 stages x BLOCK_K=256), which adds barriers with the
# K it adds; this doubles the K per buffer and leaves the barrier count alone.
BLOCK_K = 512
NUM_CU = 256
# Launch knobs, module-level so a sweep can set them without editing the source.
NUM_WARPS = 4
GROUP_SIZE_M = 4
NUM_XCDS = 8


@triton.jit
def _a4w4_kernel_32(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scales_ptr,
    b_scales_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bsn,
    stride_bsk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    GRID_MN: tl.constexpr,
):
    SCALE_GROUP_SIZE: tl.constexpr = 32
    # Without these the compiler picks sizePerThread=[1,1] for the scale tiles and
    # the whole scale path scalarizes: 20 ds_write_b8 + 16 ds_read_u8 +
    # 8 buffer_load_ubyte + 14 v_perm per iteration, ~96 overhead instructions
    # against 64 MFMAs of real work. The A/B path does not need pinning -- it
    # already comes out as ds_read_b128 / buffer_load_dwordx4.
    # A scales are a (BLOCK_M, BLOCK_K_SCALE) = (64, 8) tile and B scales a
    # (BLOCK_N, BLOCK_K_SCALE) = (128, 8) one, so unlike the 128x128 kernel they
    # need separate layouts and separate offsets. Both put
    #   row = l0 + 16*l2 + 32*r1,  k = l1 + 4*r0
    # with l3 a broadcast; only the size of r1 differs (2 vs 4) because it is what
    # covers the row dimension. 4 e8m0 per lane for A, 8 for B.
    # Read off the TTGIR after tritonamdgpu-accelerate-matmul, not guessed:
    #   #mma = amd_mfma<{warpsPerCTA = [1, 4], instrShape = [16, 16, 128]}>
    #   A (64x8) : register=[[0,4],[16,0],[32,0]]  warp=[[0,0],[0,0]]
    #   B (128x8): register=[[0,4],[64,0]]         warp=[[16,0],[32,0]]
    # warpsPerCTA is [1, 4], not [2, 2] -- all four waves lie along N. So A is
    # broadcast across every warp and carries 8 e8m0 per lane while B is split
    # across them and carries 4, inverted from the 128x128 tile. That inversion is
    # what the "expected 4 but got 8" struct-packing error was reporting.
    # BLOCK_K=512 makes these (64, 16) and (128, 16) tiles, so relative to the
    # BLOCK_K=256 versions every stride scales with the new ncols=16 and the k
    # register dim doubles 2 -> 4 (BLOCK_K spans 4 MFMAs of K=128, not 2). Per
    # lane: 16 e8m0 for A, 8 for B.
    scale_mfma_layout_a: tl.constexpr = tlx.layout(
        shape=((16, 4, 4), (4, 2)),
        stride=((16, 1, 0), (4, 256)),
    )
    scale_mfma_layout_b: tl.constexpr = tlx.layout(
        shape=((16, 4, 2, 2), (4, 2)),
        stride=((16, 1, 256, 512), (4, 1024)),
    )
    # Epilogue layouts, both read off the TTGIR rather than guessed.
    #
    # store_layout_c is #blocked<sizePerThread=[1,8], threadsPerWarp=[4,16],
    # warpsPerCTA=[4,1], order=[1,0]> in Shape:Stride form: 16 lanes of 8
    # contiguous bf16 cover a 128-wide row (one global_store_dwordx4 each), 64
    # lanes cover 4 rows, the 4 warps cover 16, and the register dim repeats that
    # 4 times down to 64.
    store_layout_c: tl.constexpr = tlx.layout(
        shape=((64, 4), (8, 2)),
        stride=((8, 512), (1, 2048)),
    )
    # accumulator_layout is #mma<warpsPerCTA=[1,4], instrShape=[16,16,128],
    # isTransposed=true>. One MFMA tile is lane -> row (16, stride 128) x
    # col (4, stride 4) with 4 contiguous cols per lane; the 4 warps tile N
    # contiguously (stride 16), so the CTA covers 16x64 per round and the
    # registers repeat it 2x across N and 4x down M.
    # The MFMA B dot-operand layout: dot_op<opIdx=1, parent=#mma, kWidth=16> on
    # the (K_packed=256, N=128) operand. lane(a,b,c) x reg(e,f,g) ->
    #   n = a + 16c + 64g,   k = 16b + e + 64f
    # i.e. lane l holds n = l%16 and 16 contiguous K bytes at k = 16*(l/16), c is
    # the warp (4 warps tiling N by 16), and the registers repeat 16x and 4x
    # along K and 2x along N.
    #
    # Confirmed, not guessed: pinning the LDS-loaded B to this produced a
    # byte-identical kernel (same op histogram, same 234 VGPR), so require_layout
    # was a no-op and this IS the layout the MFMA already asks for. That is what
    # makes it safe to bypass LDS and load B straight into these registers.
    b_operand_layout: tl.constexpr = tlx.layout(
        shape=((16, 4, 4), (16, 4, 2)),
        stride=((1, 2048, 16), (128, 8192, 64)),
    )
    accumulator_layout: tl.constexpr = tlx.layout(
        shape=((16, 4, 4), (4, 2, 2)),
        stride=((128, 4, 16), (1, 64, 2048)),
    )
    # A is now the only tile in LDS -- B goes global -> VGPR (see b_operand_layout
    # and shuffle_b_weight), so the (BLOCK_N, BLOCK_K/2) companion layout that used
    # to back smem_b is gone with it. That also drops LDS from ~99 KB to ~33 KB.
    #
    # smem_a is (BLOCK_M, BLOCK_K/2) = (32, 256) = 2**13.
    #
    # The pad stays 32 and this tile RUNS WITH 2.9% LDS bank conflicts. That is
    # deliberate and measured. Halving BLOCK_M re-introduced a 2-way conflict
    # (0.00 -> 2.90) that the 64x128 tile does not have, and pad=64 does remove
    # it completely -- LDSBankConflict 0.000 -- but the kernel gets ~2% SLOWER:
    #
    #   pad  LDSBankConflict   256x4096x4096 ratio (2 reps, alternating)
    #    32       2.90%              1.021 / 1.000   <- kept
    #    64       0.000              0.992 / 0.983
    #    16       5.795              (not benchmarked)
    #
    # Do not "fix" the counter here. The conflict is real but cheap, and the
    # wider pad costs more elsewhere than the conflicts cost. Note also that the
    # stride/gcd argument that predicts which pads work is wrong for this tile
    # (it predicts 16, which is the worst of the three) -- sweep and measure.
    shared_layout_a: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [[1024, 32]],
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [8, 0],
            [0, 16],
            [0, 32],
            [0, 64],
            [0, 128],
            [16, 0],
            [1, 0],
            [2, 0],
            [4, 0],
        ],
        [BLOCK_M, BLOCK_K // 2],
    )
    BLOCK_K_PACKED: tl.constexpr = BLOCK_K // 2
    BLOCK_K_SCALE: tl.constexpr = BLOCK_K // SCALE_GROUP_SIZE

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # Spread consecutive pids across XCDs so each XCD gets a contiguous stripe
    # of the (m, n) grid rather than a strided one.
    if NUM_XCDS != 1:
        pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
        tall_xcds = GRID_MN % NUM_XCDS
        tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
        if xcd < tall_xcds:
            pid = xcd * pids_per_xcd + local_pid
        else:
            pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid

    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        tl.assume(group_size_m > 0)
        pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
        pid_n = (pid % num_pid_in_group) // group_size_m

    # 2 x 16 KB for A/B plus 2 x 1 KB of scale scratch = 66 KB, so unlike the
    # 256x256 kernel (138 KB) this fits two workgroups per CU on the 160 KB
    # CDNA4 LDS -- which is the point, since these shapes cannot fill the grid.
    smem_a = tlx.local_alloc((BLOCK_M, BLOCK_K_PACKED), tlx.dtype_of(a_ptr), 2, layout=shared_layout_a)
    # 2-deep, not 1: a 1-deep scratchpad is store->load->(next iter)store on the
    # same buffer, a cross-iteration WAR across the 4 warps that showed up as a
    # nondeterministic result (same inputs, different output run to run).

    offs_am = tl.arange(0, BLOCK_M)
    offs_ak = tl.arange(0, BLOCK_K_PACKED)
    a_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    a_offsets_next = a_offsets + BLOCK_K_PACKED * stride_ak
    a_base = a_ptr + pid_m * BLOCK_M * stride_am

    # B is pre-shuffled on the host (shuffle_b_weight) so a plain coalesced global
    # load lands directly in b_operand_layout -- B never enters LDS. This is
    # AITER's BpreShuffle contract, which its kernel already gets from the
    # harness; matching it removes the last structural difference in the loop.
    # Byte for (k, n) within a (BLOCK_K_PACKED, BLOCK_N) tile:
    #   4096*((k//64) + 4*(n//64))                      <- which dwordx4
    #     + 16*((n%16) + 16*((k//16)%4) + 64*((n//16)%4))  <- 16 B per lane
    #     + (k%16)
    # i.e. b_operand_layout linearised with the LANE index innermost inside each
    # 16-byte chunk, NOT register-innermost.
    #
    # The register-innermost form (each lane's whole 128 bytes contiguous) is the
    # obvious generalisation of the scale shuffle and it is 32% SLOWER. A lane
    # needs 8 dwordx4, so with lane base 128*lane the j-th instruction addresses
    # 128*lane + 16j -- stride 128 across lanes, touching 64 separate 128-byte
    # lines and using 16 bytes of each. The scales get away with register-
    # innermost only because 16 B/lane is exactly one instruction wide.
    # What has to be contiguous is one INSTRUCTION across lanes, so each of the 8
    # loads is one 4096-byte burst.
    TILE_B: tl.constexpr = BLOCK_N * BLOCK_K_PACKED
    offs_bn = tl.arange(0, BLOCK_N)
    offs_bk = tl.arange(0, BLOCK_K_PACKED)
    b_koff = (offs_bk % 16) + ((offs_bk // 16) % 4) * 256 + (offs_bk // 64) * 4096
    b_noff = (offs_bn % 16) * 16 + ((offs_bn // 16) % 4) * 1024 + (offs_bn // 64) * 16384
    b_offsets = tl.add(b_koff[:, None], b_noff[None, :], sanitize_overflow=False)
    b_offsets = tlx.require_layout(b_offsets, b_operand_layout)
    b_offsets_next = tl.add(b_offsets, TILE_B, sanitize_overflow=False)
    b_base = b_ptr + pid_n * (K // 2 // BLOCK_K_PACKED) * TILE_B

    # Pre-shuffled scale addressing, one tile per (row_tile, k_tile). A is a
    # (BLOCK_M, BLOCK_K_SCALE) = (64, 16) tile and B a (128, 16) one, so they need
    # separate offsets. Within a tile the byte for (row, k) is at
    #   A: (row%16)*16 + ((row//16)%2)*4 + (row//32)*8
    #                  + (k%4)*256 + (k//4)
    #   B: (row%16)*8  + ((row//16)%2)*512 + ((row//32)%2)*1024 + (row//64)*4
    #                  + (k%4)*128 + (k//4)
    # i.e. each layout's (lane, register) -> element map linearised with the
    # register index innermost, so a lane's bytes are contiguous and the load is
    # one buffer_load_dwordx4 (A, 16 B) / dwordx2 (B, 8 B) straight into MFMA
    # operand layout, with no LDS round trip. _shuffle_scale below is the host
    # side of exactly these two formulas; both are verified bijections.
    TILE_AS: tl.constexpr = BLOCK_M * BLOCK_K_SCALE
    TILE_BS: tl.constexpr = BLOCK_N * BLOCK_K_SCALE
    offs_sk = tl.arange(0, BLOCK_K_SCALE)
    offs_asm = tl.arange(0, BLOCK_M)
    offs_bsn = tl.arange(0, BLOCK_N)
    a_row = (offs_asm % 16) * 8 + (offs_asm // 16) * 4
    a_col = (offs_sk % 4) * 128 + (offs_sk // 4)
    a_scale_offsets = tl.add(a_row[:, None], a_col[None, :], sanitize_overflow=False)
    a_scale_offsets = tlx.require_layout(a_scale_offsets, scale_mfma_layout_a)
    a_scale_offsets_next = tl.add(a_scale_offsets, TILE_AS, sanitize_overflow=False)
    b_row = ((offs_bsn % 16) * 8 + ((offs_bsn // 16) % 2) * 512 + ((offs_bsn // 32) % 2) * 1024 + (offs_bsn // 64) * 4)
    b_col = (offs_sk % 4) * 128 + (offs_sk // 4)
    b_scale_offsets = tl.add(b_row[:, None], b_col[None, :], sanitize_overflow=False)
    b_scale_offsets = tlx.require_layout(b_scale_offsets, scale_mfma_layout_b)
    b_scale_offsets_next = tl.add(b_scale_offsets, TILE_BS, sanitize_overflow=False)
    a_scales_base = a_scales_ptr + pid_m * (K // 32 // BLOCK_K_SCALE) * TILE_AS
    b_scales_base = b_scales_ptr + pid_n * (K // 32 // BLOCK_K_SCALE) * TILE_BS

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Runtime trip count so every supported K shares one compiled kernel.
    iter_max = K // BLOCK_K
    tl.assume(iter_max > 1)

    # Prime both stages, one async group each.
    tlx.buffer_load_to_local(smem_a[0], a_base, a_offsets)
    b_buf0 = tlx.require_layout(tlx.buffer_load(b_base, b_offsets, contiguity=16), b_operand_layout)
    a_sc_buf0 = tlx.require_layout(tlx.buffer_load(a_scales_base, a_scale_offsets, contiguity=8), scale_mfma_layout_a)
    b_sc_buf0 = tlx.require_layout(tlx.buffer_load(b_scales_base, b_scale_offsets, contiguity=8), scale_mfma_layout_b)
    tlx.async_load_commit_group()

    tlx.buffer_load_to_local(smem_a[1], a_base, a_offsets_next)
    b_buf1 = tlx.require_layout(tlx.buffer_load(b_base, b_offsets_next, contiguity=16), b_operand_layout)
    a_sc_buf1 = tlx.require_layout(tlx.buffer_load(a_scales_base, a_scale_offsets_next, contiguity=8),
                                   scale_mfma_layout_a)
    b_sc_buf1 = tlx.require_layout(tlx.buffer_load(b_scales_base, b_scale_offsets_next, contiguity=8),
                                   scale_mfma_layout_b)
    tlx.async_load_commit_group()

    a_base += BLOCK_K_PACKED * stride_ak * 2
    b_base += TILE_B * 2
    a_scales_base += TILE_AS * 2
    b_scales_base += TILE_BS * 2

    tlx.async_load_wait_group(1)
    a = tlx.local_load(smem_a[0])
    b = b_buf0
    a_sc = a_sc_buf0
    b_sc = b_sc_buf0

    # Unrolled by 2 so the two LDS stages alternate without an index variable.
    # Issue the refill for the stage we just consumed BEFORE waiting on the other
    # stage. With only two stages there is exactly one group per stage, so if the
    # refill came after the wait there would be a single group outstanding and
    # async_load_wait_group(1) would be satisfied by 1 <= 1 without waiting for
    # anything -- the next local_load would then read a buffer whose DMA was still
    # in flight. That raced: ~20% of the elements of a tile (one warp's share)
    # differed run to run. Refill-then-wait keeps two groups in flight, so the
    # wait actually retires the one being read and still overlaps the other.
    for _ in tl.range(0, iter_max - 2, 2, num_stages=1):
        acc = tl.dot_scaled(a, a_sc, "e2m1", b, b_sc, "e2m1", acc)
        tlx.buffer_load_to_local(smem_a[0], a_base, a_offsets)
        b_buf0 = tlx.require_layout(tlx.buffer_load(b_base, b_offsets, contiguity=16), b_operand_layout)
        a_sc_buf0 = tlx.require_layout(tlx.buffer_load(a_scales_base, a_scale_offsets, contiguity=8),
                                       scale_mfma_layout_a)
        b_sc_buf0 = tlx.require_layout(tlx.buffer_load(b_scales_base, b_scale_offsets, contiguity=8),
                                       scale_mfma_layout_b)
        tlx.async_load_commit_group()
        tlx.async_load_wait_group(1)
        a_n = tlx.local_load(smem_a[1])
        b_n = b_buf1
        a_sc_n = a_sc_buf1
        b_sc_n = b_sc_buf1

        acc = tl.dot_scaled(a_n, a_sc_n, "e2m1", b_n, b_sc_n, "e2m1", acc)
        tlx.buffer_load_to_local(smem_a[1], a_base, a_offsets_next)
        b_buf1 = tlx.require_layout(tlx.buffer_load(b_base, b_offsets_next, contiguity=16), b_operand_layout)
        a_sc_buf1 = tlx.require_layout(tlx.buffer_load(a_scales_base, a_scale_offsets_next, contiguity=8),
                                       scale_mfma_layout_a)
        b_sc_buf1 = tlx.require_layout(tlx.buffer_load(b_scales_base, b_scale_offsets_next, contiguity=8),
                                       scale_mfma_layout_b)
        tlx.async_load_commit_group()
        tlx.async_load_wait_group(1)
        a = tlx.local_load(smem_a[0])
        b = b_buf0
        a_sc = a_sc_buf0
        b_sc = b_sc_buf0

        a_base += BLOCK_K_PACKED * stride_ak * 2
        b_base += TILE_B * 2
        a_scales_base += TILE_AS * 2
        b_scales_base += TILE_BS * 2

    # Drain the two in-flight stages.
    acc = tl.dot_scaled(a, a_sc, "e2m1", b, b_sc, "e2m1", acc)
    tlx.async_load_wait_group(0)
    a_n = tlx.local_load(smem_a[1])
    b_n = b_buf1
    a_sc_n = a_sc_buf1
    b_sc_n = b_sc_buf1
    acc = tl.dot_scaled(a_n, a_sc_n, "e2m1", b_n, b_sc_n, "e2m1", acc)

    offs_cm = tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_offsets = tl.add(
        tl.mul(stride_cm, offs_cm, sanitize_overflow=False)[:, None],
        tl.mul(stride_cn, offs_cn, sanitize_overflow=False)[None, :],
        sanitize_overflow=False,
    )
    c_offsets = tlx.require_layout(c_offsets, store_layout_c)
    c_tile_base = c_ptr + pid_m * BLOCK_M * stride_cm
    # Pin the accumulator layout before narrowing so that the store-layout
    # requirement redistributes bf16 rather than propagating back to f32. Left
    # unpinned the compiler hoists the conversion above the truncf -- the TTGIR
    # reads convert_layout(f32, #mma -> #blocked) then truncf -- and the epilogue
    # round-trips 4 bytes per element through LDS instead of 2.
    acc = tlx.require_layout(acc, accumulator_layout)
    c = acc.to(c_ptr.dtype.element_ty)
    c = tlx.require_layout(c, store_layout_c)
    tlx.buffer_store(c, c_tile_base, c_offsets)


def _shuffle_scale(scales, tile_rows):
    """Permute raw (rows, K//32) e8m0 scales into the layout the kernel loads.

    Host-side and done once, mirroring aiter.ops.shuffle.shuffle_scale. A and B
    differ because warpsPerCTA is [1, 4]: A is warp-broadcast with 8 e8m0 per lane,
    B warp-split with 4. Both make a lane's bytes contiguous so the load is one
    buffer_load straight into MFMA operand layout, with no LDS round-trip.
    Pure permutation; shape and dtype preserved.
    """
    rows, nk = scales.shape
    KS = BLOCK_K // 32
    assert rows % tile_rows == 0 and nk % KS == 0, "scales must tile by tile_rows x KS"
    src = scales.contiguous()
    dev = src.device
    r = torch.arange(tile_rows, device=dev)
    k = torch.arange(KS, device=dev)
    if tile_rows == BLOCK_M:
        pos = ((r % 16)[:, None] * 8 + (r // 16)[:, None] * 4 + (k % 4)[None, :] * 128 + (k // 4)[None, :])
    else:
        pos = (((r % 16) * 8 + ((r // 16) % 2) * 512 + ((r // 32) % 2) * 1024 + (r // 64) * 4)[:, None] +
               (k % 4)[None, :] * 128 + (k // 4)[None, :])
    nt, nkt, tile = rows // tile_rows, nk // KS, tile_rows * KS
    blocks = src.reshape(nt, tile_rows, nkt, KS).permute(0, 2, 1, 3).reshape(nt, nkt, tile)
    out = torch.empty(nt, nkt, tile, dtype=src.dtype, device=dev)
    out.scatter_(2, pos.reshape(1, 1, -1).expand(nt, nkt, tile).contiguous(), blocks)
    return out.reshape(rows, nk)


def shuffle_a_scale(a_scales):
    return _shuffle_scale(a_scales, BLOCK_M)


def shuffle_b_scale(b_scales):
    return _shuffle_scale(b_scales, BLOCK_N)


def shuffle_b_weight(b):
    """Permute raw packed-e2m1 B, shape (N, K // 2), into b_operand_layout order.

    Host-side and done once, the same deal AITER's kernel gets from
    aiter.ops.shuffle.shuffle_weight (its selected kernel is literally named
    ..._BpreShuffle_64x128). This is what lets B skip LDS entirely: the shuffled
    buffer is a sequence of (BLOCK_K_PACKED, BLOCK_N) tiles in which each lane's
    128 bytes are contiguous, so a plain coalesced global load delivers the MFMA
    B operand with no staging, no ds_read and no barrier.

    Pure permutation; shape and dtype preserved.
    """
    n, kp = b.shape
    assert n % BLOCK_N == 0 and kp % (BLOCK_K // 2) == 0, "B must tile by BLOCK_N x BLOCK_K_PACKED"
    KP = BLOCK_K // 2
    src = b.contiguous()
    dev = src.device
    kk = torch.arange(KP, device=dev)[:, None]
    nn = torch.arange(BLOCK_N, device=dev)[None, :]
    pos = ((nn % 16) * 16 + ((nn // 16) % 4) * 1024 + (nn // 64) * 16384 + (kk % 16) + ((kk // 16) % 4) * 256 +
           (kk // 64) * 4096)
    nt, nkt, tile = n // BLOCK_N, kp // KP, BLOCK_N * KP
    # (n, kp) -> per-tile (k, n) blocks, matching the kernel's operand orientation
    blocks = (src.reshape(nt, BLOCK_N, nkt, KP).permute(0, 2, 3, 1).reshape(nt, nkt, tile))
    out = torch.empty(nt, nkt, tile, dtype=src.dtype, device=dev)
    out.scatter_(2, pos.reshape(1, 1, -1).expand(nt, nkt, tile).contiguous(), blocks)
    return out.reshape(n, kp)


def matmul(a, b, a_scales, b_scales):
    assert a.dtype is torch.uint8
    assert b.dtype is torch.uint8
    assert a_scales.dtype is torch.uint8
    assert b_scales.dtype is torch.uint8
    assert a.is_cuda and b.is_cuda and a_scales.is_cuda and b_scales.is_cuda

    M = a.shape[0]
    K_packed = a.shape[1]
    K = K_packed * 2
    N = b.shape[0]

    assert b.shape[1] == K_packed, "B must have shape (N, K // 2)"
    assert a_scales.shape == (M, K // 32), "A scales must have shape (M, K // 32)"
    assert b_scales.shape == (N, K // 32), "B scales must have shape (N, K // 32)"
    assert a_scales.is_contiguous(), "A scales must be pre-shuffled (see shuffle_a_scale)"
    assert b_scales.is_contiguous(), "B scales must be pre-shuffled (see shuffle_b_scale)"
    assert b.is_contiguous(), "B must be pre-shuffled (see shuffle_b_weight)"

    assert M % BLOCK_M == 0, "M must be a multiple of 128"
    assert N % BLOCK_N == 0, "N must be a multiple of 128"
    assert K >= 2 * BLOCK_K and K % (2 * BLOCK_K) == 0, "K must be at least 512 and a multiple of 512"

    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    grid_mn = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    _a4w4_kernel_32[(grid_mn, )](
        a,
        b,
        c,
        a_scales,
        b_scales,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        a_scales.stride(0),
        a_scales.stride(1),
        b_scales.stride(0),
        b_scales.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        NUM_XCDS=NUM_XCDS,
        GRID_MN=grid_mn,
        num_warps=NUM_WARPS,
        num_stages=1,
        matrix_instr_nonkdim=16,
    )
    return c
