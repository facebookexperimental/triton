"""Occupancy-oriented MXFP4 skinny GEMM kernels for gfx950.

This module owns the 32/64/128x128 skinny tile family. The inter-wave A4W4
module imports this implementation and retains the public shape dispatcher.
"""

import os

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

__all__ = ["is_skinny", "skinny_matmul"]

# Preserve the scheduling contract when this module is imported directly,
# rather than through the inter-wave dispatcher.
os.environ.setdefault("TRITON_DISABLE_POST_MISCHED", "1")

BLOCK_K = 256
NUM_WARPS = 8
GROUP_SIZE_M = 4
NUM_XCDS = 8
NUM_CU = 256
SKINNY_TARGET_WGS = NUM_CU
DISPATCH_BLOCK_M = 256
DISPATCH_BLOCK_N = 256


@triton.jit
def _reduce_k_kernel(workspace_ptr, c_ptr, M, N, SPLIT_K: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
                     BLOCK_SIZE_N: tl.constexpr, OUTPUT_DTYPE: tl.constexpr):
    # Sum contiguous fp32 partial slabs into C with fp32 accumulation.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    base_offs = offs_m[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for s in range(SPLIT_K):
        partial = tl.load(workspace_ptr + base_offs + s * M * N, mask=mask, other=0.0)
        acc += partial.to(tl.float32)
    tl.store(c_ptr + base_offs, acc.to(OUTPUT_DTYPE), mask=mask)


# 32/64x128 variants use four waves; 128x128 uses eight. These kernels use
# direct-to-LDS loads, pinned layouts, a buffered operand ring, one accumulator,
# and no inter-wave software pipeline. Split-K is used only when the natural
# tile grid cannot fill the machine.
#
# The smaller tiles are reserved for occupancy-starved shapes because they
# repeat operand work when the natural grid is already large.
SKINNY_TINY_BLOCK_M = 32
SKINNY_SMALL_BLOCK_M = 64
SKINNY_BLOCK_M = 128
SKINNY_BLOCK_N = 128


@triton.jit
def _a4w4_skinny_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    workspace_ptr,
    a_scales_ptr,
    b_scales_ptr,
    M,
    N,
    K: tl.constexpr,
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
    SPLIT_K: tl.constexpr,
    BUFFER_COUNT: tl.constexpr,
):
    SCALE_GROUP_SIZE: tl.constexpr = 32
    NG: tl.constexpr = BLOCK_K // SCALE_GROUP_SIZE
    BKP: tl.constexpr = BLOCK_K // 2
    KS: tl.constexpr = K // SPLIT_K
    iter_max: tl.constexpr = KS // BLOCK_K

    # The 32x128 variant assigns one 32x32 MFMA output tile per wave in N.
    # Its 32x8 byte A-scale copy is too small/non-injective for the AMD async
    # copy lowering, so four adjacent M bytes are copied as one aligned dword
    # and the LDS image is viewed as bytes only at local-load time.
    if BLOCK_M == 32:
        g_load_layout_a: tl.constexpr = tlx.layout(
            shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2)),
            stride=((16, 32, 64, 128, 256, 512, 1024, 2048), (1, 2, 4, 8)),
        )
        packed_scales_a: tl.constexpr = tlx.layout(
            shape=((2, 2, 2, 2, 2, 2, 2, 2), ()),
            stride=((8, 16, 32, 1, 2, 4, 0, 0), ()),
        )
        shared_tile_a: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
            [[1024, 16]],
            [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64],
             [1, 0], [2, 0], [4, 0], [8, 0], [16, 0]],
            [BLOCK_M, BKP],
        )
        scale_a_layout: tl.constexpr = tlx.layout(
            shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2)),
            stride=((8, 16, 32, 64, 128, 1, 0, 0), (2, 4)),
        )
        accumulator_layout: tl.constexpr = tlx.layout(
            shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2)),
            stride=((128, 256, 512, 1024, 2048, 4, 32, 64), (1, 2, 8, 16)),
        )
        store_layout_c: tl.constexpr = tlx.layout(
            shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2)),
            stride=((8, 16, 32, 64, 128, 256, 512, 1024), (1, 2, 4, 2048)),
        )
    # The 64x128 variant assigns two 32x32 MFMA output tiles to each of four
    # waves (one M tile and two N tiles). It drops the second per-wave M tile
    # used by the 128x128 variant and keeps all
    # global A loads as adjacent 16-byte groups.
    elif BLOCK_M == 64:
        g_load_layout_a: tl.constexpr = tlx.layout(
            shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2)),
            stride=((16, 32, 64, 128, 256, 512, 1024, 2048), (1, 2, 4, 8, 4096)),
        )
        blocked_scales_a: tl.constexpr = tlx.layout(
            shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2)),
            stride=((32, 64, 128, 256, 1, 2, 0, 4), (8, 16)),
        )
        shared_tile_a: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
            [[1024, 16]],
            [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64],
             [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]],
            [BLOCK_M, BKP],
        )
        scale_a_layout: tl.constexpr = tlx.layout(
            shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2)),
            stride=((8, 16, 32, 64, 128, 1, 0, 0), (2, 4, 256)),
        )
        accumulator_layout: tl.constexpr = tlx.layout(
            shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2)),
            stride=((128, 256, 512, 1024, 2048, 4, 32, 64), (1, 2, 8, 16, 4096)),
        )
        store_layout_c: tl.constexpr = tlx.layout(
            shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2)),
            stride=((8, 16, 32, 64, 128, 256, 512, 1024), (1, 2, 4, 2048, 4096)),
        )
    else:
        g_load_layout_a: tl.constexpr = tlx.layout(
            shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2)),
            stride=((16, 32, 64, 128, 4096, 8192, 256, 512, 1024), (1, 2, 4, 8, 2048)),
        )
        blocked_scales_a: tl.constexpr = tlx.layout(
            shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2)),
            stride=((32, 64, 128, 256, 512, 1, 0, 2, 4), (8, 16)),
        )
        shared_tile_a: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
            [[1024, 16]],
            [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64],
             [1, 0], [32, 0], [64, 0], [2, 0], [4, 0], [8, 0], [16, 0]],
            [BLOCK_M, BKP],
        )
        scale_a_layout: tl.constexpr = tlx.layout(
            shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2)),
            stride=((8, 16, 32, 64, 128, 1, 0, 0, 256), (2, 4, 512)),
        )
        accumulator_layout: tl.constexpr = tlx.layout(
            shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2)),
            stride=((128, 256, 512, 1024, 2048, 4, 32, 64, 4096), (1, 2, 8, 16, 8192)),
        )
        store_layout_c: tl.constexpr = tlx.layout(
            shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2, 2)),
            stride=((8, 16, 32, 64, 512, 1024, 0, 0, 2048), (1, 2, 4, 128, 256, 4096, 8192)),
        )
    if BLOCK_M <= 64:
        g_load_layout_b: tl.constexpr = tlx.layout(
            shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2)),
            stride=((16, 32, 64, 128, 4096, 8192, 256, 512), (1, 2, 4, 8, 1024, 2048)),
        )
        blocked_scales_b: tl.constexpr = tlx.layout(
            shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2)),
            stride=((32, 64, 128, 256, 512, 1, 2, 4), (8, 16)),
        )
        scale_b_layout: tl.constexpr = tlx.layout(
            shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2)),
            stride=((8, 16, 32, 64, 128, 1, 256, 512), (2, 4)),
        )
    else:
        g_load_layout_b: tl.constexpr = tlx.layout(
            shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2)),
            stride=((16, 32, 64, 128, 4096, 8192, 256, 512, 1024), (1, 2, 4, 8, 2048)),
        )
        blocked_scales_b: tl.constexpr = tlx.layout(
            shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2)),
            stride=((32, 64, 128, 256, 512, 1, 0, 2, 4), (8, 16)),
        )
        scale_b_layout: tl.constexpr = tlx.layout(
            shape=((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2)),
            stride=((8, 16, 32, 64, 128, 1, 256, 512, 0), (2, 4)),
        )
    shared_tile_b: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [[1024, 16]], [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [32, 0], [64, 0], [2, 0],
                       [4, 0], [8, 0], [16, 0]], [BLOCK_N, BKP])
    shared_scales: tl.constexpr = tlx.swizzled_layout(0, 0, 0, order=[0, 1])
    # Byte-level physical view of row-major [M/4, K-group] dwords. The byte
    # lane is contiguous, followed by packed M, then K-group.
    shared_scale_bytes: tl.constexpr = tlx.shared_linear_layout_encoding(
        offset_bases=[
            [0, 0, 1],
            [0, 0, 2],
            [1, 0, 0],
            [2, 0, 0],
            [4, 0, 0],
            [0, 1, 0],
            [0, 2, 0],
            [0, 4, 0],
        ],
        block_bases=[],
        alignment=4,
    )

    split_id = tl.program_id(0) // GRID_MN
    pid = tl.program_id(0) % GRID_MN
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
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

    smem_a = tlx.local_alloc((BLOCK_M, BKP), tlx.dtype_of(a_ptr), BUFFER_COUNT, layout=shared_tile_a)
    smem_b = tlx.local_alloc((BLOCK_N, BKP), tlx.dtype_of(b_ptr), BUFFER_COUNT, layout=shared_tile_b)
    if BLOCK_M == 32:
        smem_asc = tlx.local_alloc((BLOCK_M // 4, NG), tl.uint32, BUFFER_COUNT, layout=shared_scales)
    else:
        smem_asc = tlx.local_alloc((BLOCK_M, NG), tlx.dtype_of(a_scales_ptr), BUFFER_COUNT, layout=shared_scales)
    smem_bsc = tlx.local_alloc((BLOCK_N, NG), tlx.dtype_of(b_scales_ptr), BUFFER_COUNT, layout=shared_scales)

    offs_am = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BKP)
    a_off = tlx.require_layout(offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak, g_load_layout_a)
    a_base = a_ptr + pid_m * BLOCK_M * stride_am + split_id * (KS // 2) * stride_ak
    offs_bn = tl.arange(0, BLOCK_N)
    b_off = tlx.require_layout(offs_bn[:, None] * stride_bn + offs_k[None, :] * stride_bk, g_load_layout_b)
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn + split_id * (KS // 2) * stride_bk
    offs_sg = tl.arange(0, NG)
    if BLOCK_M == 32:
        # Scales are contiguous in M. Reinterpret each four-M byte group as a
        # dword; strides and offsets are consequently expressed in dwords.
        offs_asm_packed = tl.arange(0, BLOCK_M // 4)
        asc_off = tlx.require_layout(
            offs_asm_packed[:, None] +
            tl.mul(offs_sg[None, :], stride_ask // 4, sanitize_overflow=False),
            packed_scales_a,
        )
        a_scales_load_ptr = (a_scales_ptr + pid_m * BLOCK_M).to(tl.pointer_type(tl.uint32))
    else:
        offs_asm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        asc_off = tlx.require_layout(
            tl.mul(offs_asm[:, None], stride_asm, sanitize_overflow=False) +
            tl.mul(offs_sg[None, :], stride_ask, sanitize_overflow=False), blocked_scales_a)
        a_scales_load_ptr = a_scales_ptr
    a_scales_load_ptr += split_id * (KS // SCALE_GROUP_SIZE) * stride_ask // (4 if BLOCK_M == 32 else 1)
    offs_bsn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    bsc_off = tlx.require_layout(
        tl.mul(offs_bsn[:, None], stride_bsn, sanitize_overflow=False) +
        tl.mul(offs_sg[None, :], stride_bsk, sanitize_overflow=False), blocked_scales_b)
    b_scales_ptr += split_id * (KS // SCALE_GROUP_SIZE) * stride_bsk

    ak = BKP * stride_ak
    bk = BKP * stride_bk
    sck_a = NG * stride_ask // (4 if BLOCK_M == 32 else 1)
    sck_b = NG * stride_bsk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    tl.assume(iter_max > 1)

    # Keep a true ring of independent K-tiles live.  A static inner loop makes
    # every stage visible to LLVM while the outer loop remains rolled.
    for stage in tl.static_range(0, BUFFER_COUNT):
        tlx.buffer_load_to_local(smem_a[stage], a_base + stage * ak, a_off)
        tlx.buffer_load_to_local(
            smem_asc[stage], a_scales_load_ptr + stage * sck_a, asc_off
        )
        tlx.buffer_load_to_local(smem_b[stage], b_base + stage * bk, b_off)
        tlx.buffer_load_to_local(
            smem_bsc[stage], b_scales_ptr + stage * sck_b, bsc_off
        )
        tlx.async_load_commit_group()
    a_base += ak * BUFFER_COUNT
    b_base += bk * BUFFER_COUNT
    a_scales_load_ptr += sck_a * BUFFER_COUNT
    b_scales_ptr += sck_b * BUFFER_COUNT

    for k in tl.range(
        0, iter_max - BUFFER_COUNT, BUFFER_COUNT, num_stages=1
    ):
        for stage in tl.static_range(0, BUFFER_COUNT):
            tlx.async_load_wait_group(BUFFER_COUNT - 1)
            # The wait is wave-local; all waves must finish their direct-to-LDS
            # writes before the cooperative tile is consumed.
            tl.debug_barrier()
            a = tlx.local_load(smem_a[stage], relaxed=True)
            b = tlx.local_load(tlx.local_trans(smem_b[stage]), relaxed=True)
            if BLOCK_M == 32:
                asc_view = tlx.local_reinterpret(
                    smem_asc[stage],
                    tl.uint8,
                    [BLOCK_M // 4, NG, 4],
                    shared_scale_bytes,
                )
                asc_view = tlx.local_trans(asc_view, (0, 2, 1))
                asc_view = tlx.local_reshape(asc_view, [BLOCK_M, NG])
                asc = tlx.local_load(asc_view, layout=scale_a_layout)
            else:
                asc = tlx.local_load(
                    smem_asc[stage], layout=scale_a_layout
                )
            bsc = tlx.local_load(
                smem_bsc[stage], layout=scale_b_layout
            )
            acc = tl.dot_scaled(
                a, asc, "e2m1", b, bsc, "e2m1", acc
            )
            # Do not let a faster wave overwrite this ring slot while another
            # wave is still reading its share of the cooperative tile.
            tl.debug_barrier()
            tlx.buffer_load_to_local(
                smem_a[stage], a_base + stage * ak, a_off
            )
            tlx.buffer_load_to_local(
                smem_asc[stage],
                a_scales_load_ptr + stage * sck_a,
                asc_off,
            )
            tlx.buffer_load_to_local(
                smem_b[stage], b_base + stage * bk, b_off
            )
            tlx.buffer_load_to_local(
                smem_bsc[stage], b_scales_ptr + stage * sck_b, bsc_off
            )
            tlx.async_load_commit_group()
        a_base += ak * BUFFER_COUNT
        b_base += bk * BUFFER_COUNT
        a_scales_load_ptr += sck_a * BUFFER_COUNT
        b_scales_ptr += sck_b * BUFFER_COUNT

    # Drain the ring without refilling it.
    for stage in tl.static_range(0, BUFFER_COUNT):
        tlx.async_load_wait_group(BUFFER_COUNT - 1 - stage)
        tl.debug_barrier()
        a = tlx.local_load(smem_a[stage], relaxed=True)
        b = tlx.local_load(tlx.local_trans(smem_b[stage]), relaxed=True)
        if BLOCK_M == 32:
            asc_view = tlx.local_reinterpret(
                smem_asc[stage],
                tl.uint8,
                [BLOCK_M // 4, NG, 4],
                shared_scale_bytes,
            )
            asc_view = tlx.local_trans(asc_view, (0, 2, 1))
            asc_view = tlx.local_reshape(asc_view, [BLOCK_M, NG])
            asc = tlx.local_load(asc_view, layout=scale_a_layout)
        else:
            asc = tlx.local_load(smem_asc[stage], layout=scale_a_layout)
        bsc = tlx.local_load(smem_bsc[stage], layout=scale_b_layout)
        acc = tl.dot_scaled(a, asc, "e2m1", b, bsc, "e2m1", acc)

    offs_cm = tl.arange(0, BLOCK_M)
    offs_cn = tl.arange(0, BLOCK_N)
    if SPLIT_K == 1:
        c_off = tl.mul(stride_cm, offs_cm[:, None], sanitize_overflow=False) + tl.mul(
            stride_cn, offs_cn[None, :], sanitize_overflow=False)
        c_off = tlx.require_layout(c_off, store_layout_c)
        c_base = c_ptr + pid_m * BLOCK_M * stride_cm + pid_n * BLOCK_N * stride_cn
        et = c_ptr.dtype.element_ty
        acc = tlx.require_layout(acc, accumulator_layout)
        c = tlx.require_layout(acc.to(et), store_layout_c)
        tlx.buffer_store(c, c_base, c_off)
    else:
        rb = split_id * M
        rows = rb + pid_m * BLOCK_M + offs_cm
        cols = pid_n * BLOCK_N + offs_cn
        tl.store(workspace_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn, acc)


def choose_skinny_block_m(M, N, K=None):
    """Select the measured skinny-M tile.

    The native 32x128 tile wins only for the 256x4096x4096 production shape:
    it fills all 256 CUs without split-K. Larger M/N/K values do too much
    repeated B work, so retain the 64/128 policy everywhere else.
    """
    if M == 256 and N == 4096 and K == 4096:
        return SKINNY_TINY_BLOCK_M
    grid_64 = triton.cdiv(M, SKINNY_SMALL_BLOCK_M) * triton.cdiv(
        N, SKINNY_BLOCK_N
    )
    return SKINNY_SMALL_BLOCK_M if grid_64 <= NUM_CU else SKINNY_BLOCK_M


def choose_split_k_skinny(M, N, K, block_m=None):
    """Smallest-cost SPLIT_K for the selected skinny tile.

    Use split-K only until the compute grid reaches SKINNY_TARGET_WGS. Each
    split must retain a whole BLOCK_K-aligned K chunk. Cold-L2 sweeps on gfx950
    show that filling all 256 CUs is worthwhile for the M=256 production
    shapes; naturally full 128x128 grids retain SPLIT_K=1.
    """
    if block_m is None:
        block_m = choose_skinny_block_m(M, N, K)
    grid_mn = triton.cdiv(M, block_m) * triton.cdiv(N, SKINNY_BLOCK_N)
    best = 1
    for sk in range(2, SKINNY_TARGET_WGS // grid_mn + 1):
        ks = K // sk
        if K % sk == 0 and ks % BLOCK_K == 0 and ks >= 2 * BLOCK_K:
            best = sk
    return best


def choose_skinny_buffer_count(M, N, K):
    """Use a deeper operand ring for measured full-grid skinny tiles."""
    return 4 if (M, N, K) in {
        (256, 4096, 4096),
        (256, 4096, 8192),
        (256, 8192, 4096),
        (256, 8192, 8192),
        (512, 4096, 4096),
        (512, 4096, 8192),
    } else 2


def skinny_matmul(a, b, a_scales, b_scales, SPLIT_K=None, BLOCK_M=None):
    """32/64/128x128 TLX path for occupancy-starved shapes."""
    M = a.shape[0]
    K = a.shape[1] * 2
    N = b.shape[0]
    BM = choose_skinny_block_m(M, N, K) if BLOCK_M is None else BLOCK_M
    if BM == SKINNY_TINY_BLOCK_M:
        # The dword view requires four contiguous, aligned M-scale bytes and a
        # dword-expressible K-group stride. Fall back for exotic strided views.
        if (
            a_scales.stride(0) != 1
            or a_scales.stride(1) % 4 != 0
            or a_scales.data_ptr() % 4 != 0
        ):
            BM = SKINNY_SMALL_BLOCK_M
    BN = SKINNY_BLOCK_N
    if SPLIT_K is None:
        SPLIT_K = choose_split_k_skinny(M, N, K, BM)
    KS = K // SPLIT_K
    assert K % SPLIT_K == 0 and KS % BLOCK_K == 0
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    grid_mn = triton.cdiv(M, BM) * triton.cdiv(N, BN)
    workspace = torch.empty((SPLIT_K * M, N), device=a.device, dtype=torch.float32) if SPLIT_K > 1 else c
    buffer_count = choose_skinny_buffer_count(M, N, K)
    group_size_m = GROUP_SIZE_M
    num_xcds = NUM_XCDS
    # The tiny tile is memory-clause limited; larger skinny tiles prefer ILP.
    sched_strategy = "max-memory-clause" if BM == SKINNY_TINY_BLOCK_M else "max-ilp"
    _a4w4_skinny_kernel[(grid_mn * SPLIT_K, )](
        a,
        b,
        c,
        workspace,
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
        BLOCK_M=BM,
        BLOCK_N=BN,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=group_size_m,
        NUM_XCDS=num_xcds,
        GRID_MN=grid_mn,
        SPLIT_K=SPLIT_K,
        BUFFER_COUNT=buffer_count,
        num_warps=4 if BM <= SKINNY_SMALL_BLOCK_M else NUM_WARPS,
        num_stages=1,
        matrix_instr_nonkdim=32,
        llvm_fn_attrs=(
            ("amdgpu-agpr-alloc", "0,0"),
            (
                "amdgpu-sched-strategy",
                sched_strategy,
            ),
        ),
    )
    if SPLIT_K > 1:
        rbm, rbn, rw = (32, 32, 4)
        _reduce_k_kernel[(triton.cdiv(M, rbm), triton.cdiv(N, rbn))](workspace, c, M, N, SPLIT_K=SPLIT_K,
                                                                     BLOCK_SIZE_M=rbm, BLOCK_SIZE_N=rbn,
                                                                     OUTPUT_DTYPE=tl.bfloat16, num_warps=rw)
    return c


def is_skinny(M, N, K):
    """Whether the occupancy-oriented skinny path should handle this shape."""
    grid_mn = triton.cdiv(M, DISPATCH_BLOCK_M) * triton.cdiv(
        N, DISPATCH_BLOCK_N
    )
    return grid_mn <= NUM_CU // 4
