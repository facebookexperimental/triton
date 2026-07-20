"""A separate TLX rendering of Wave's gfx950 8-wave f16 GEMM structure.

This is intentionally not a v10 tutorial step: v9 remains the tuned TLX/LLVM
kernel.  The variant stages full 256x64 and 64x256 tiles in double-buffered
LDS.  Its ordinary tensor operands are factored into the 8x4 instruction-tile
grid owned by each wave, so each TLX dot and local load has the same structural
granularity as the corresponding high-level Wave kernel.  Fragment types stay
entirely inside dot lowering.
"""

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


@triton.jit
def _load_a_fragment(smem_a, buffer, phase: tl.constexpr,
                     tile: tl.constexpr):
    # Keep Wave's physical LDS slot order: within each K phase, the eight
    # instruction-M tiles are contiguous inside each of the two M-wave
    # groups.  Select the tile structurally while retaining the M-wave group
    # as part of the distributed tensor.
    region = smem_a[buffer]
    fragment_view = tlx.local_slice(
        region,
        [0, phase, 0, tile, 0, 0],
        [1, 1, 2, 1, 16, 32],
    )
    fragment_layout: tl.constexpr = tlx.distributed_linear_layout_encoding.make(
        reg_bases=[
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 4],
        ],
        lane_bases=[
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 2, 0],
            [0, 0, 0, 0, 4, 0],
            [0, 0, 0, 0, 8, 0],
            [0, 0, 0, 0, 0, 8],
            [0, 0, 0, 0, 0, 16],
        ],
        warp_bases=[
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ],
        block_bases=[],
        shape=[1, 1, 2, 1, 16, 32],
    )
    fragment = tlx.require_layout(
        tlx.local_load(fragment_view), fragment_layout)
    fragment = tl.reshape(fragment, (32, 32))
    return tlx.release_layout(fragment)


@triton.jit
def _load_b_fragment(smem_b, buffer, phase: tl.constexpr,
                     tile: tl.constexpr):
    # Likewise group the four instruction-N tiles within each N-wave group,
    # matching the producer slot order used by Wave's direct-to-LDS DMA.
    region = smem_b[buffer]
    fragment_view = tlx.local_slice(
        region,
        [1, phase, 0, tile, 0, 0],
        [1, 1, 4, 1, 32, 16],
    )
    # The shared-memory view is physically [N-wave, K, N-inner]; permuting it
    # to [K, N-wave, N-inner] before the reshape is structural and preserves
    # the upstream producer-wave slot ownership.
    fragment_layout: tl.constexpr = tlx.distributed_linear_layout_encoding.make(
        reg_bases=[
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 2, 0],
            [0, 0, 0, 0, 4, 0],
        ],
        lane_bases=[
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 4],
            [0, 0, 0, 0, 0, 8],
            [0, 0, 0, 0, 8, 0],
            [0, 0, 0, 0, 16, 0],
        ],
        warp_bases=[
            [0, 0, 1, 0, 0, 0],
            [0, 0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        block_bases=[],
        shape=[1, 1, 4, 1, 32, 16],
    )
    fragment = tlx.require_layout(
        tlx.local_load(fragment_view), fragment_layout)
    fragment = tl.permute(fragment, (0, 1, 4, 3, 2, 5))
    fragment = tl.reshape(fragment, (32, 64))
    return tlx.release_layout(fragment)


@triton.jit
def _load_phase(smem_a, smem_b, buffer, phase: tl.constexpr):
    a = (
        _load_a_fragment(smem_a, buffer, phase, 0),
        _load_a_fragment(smem_a, buffer, phase, 1),
        _load_a_fragment(smem_a, buffer, phase, 2),
        _load_a_fragment(smem_a, buffer, phase, 3),
        _load_a_fragment(smem_a, buffer, phase, 4),
        _load_a_fragment(smem_a, buffer, phase, 5),
        _load_a_fragment(smem_a, buffer, phase, 6),
        _load_a_fragment(smem_a, buffer, phase, 7),
    )
    b = (
        _load_b_fragment(smem_b, buffer, phase, 0),
        _load_b_fragment(smem_b, buffer, phase, 1),
        _load_b_fragment(smem_b, buffer, phase, 2),
        _load_b_fragment(smem_b, buffer, phase, 3),
    )
    return a, b


@triton.jit
def _mma_phase(a, b, acc):
    # One 16x16x32 MFMA per wave for every dot below.  The 32 ordinary
    # 32x64 accumulator tensors collectively cover the 256x256 CTA tile.
    mma_layout: tl.constexpr = tlx.amd_mfma_layout_encoding.make(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[2, 4],
    )
    a_fragment_layout: tl.constexpr = tlx.dot_operand_layout_encoding.make(
        0, mma_layout, 8)
    b_fragment_layout: tl.constexpr = tlx.dot_operand_layout_encoding.make(
        1, mma_layout, 8)
    a = (
        tlx.require_layout(a[0], a_fragment_layout),
        tlx.require_layout(a[1], a_fragment_layout),
        tlx.require_layout(a[2], a_fragment_layout),
        tlx.require_layout(a[3], a_fragment_layout),
        tlx.require_layout(a[4], a_fragment_layout),
        tlx.require_layout(a[5], a_fragment_layout),
        tlx.require_layout(a[6], a_fragment_layout),
        tlx.require_layout(a[7], a_fragment_layout),
    )
    b = (
        tlx.require_layout(b[0], b_fragment_layout),
        tlx.require_layout(b[1], b_fragment_layout),
        tlx.require_layout(b[2], b_fragment_layout),
        tlx.require_layout(b[3], b_fragment_layout),
    )
    acc = (
        tlx.require_layout(acc[0], mma_layout),
        tlx.require_layout(acc[1], mma_layout),
        tlx.require_layout(acc[2], mma_layout),
        tlx.require_layout(acc[3], mma_layout),
        tlx.require_layout(acc[4], mma_layout),
        tlx.require_layout(acc[5], mma_layout),
        tlx.require_layout(acc[6], mma_layout),
        tlx.require_layout(acc[7], mma_layout),
        tlx.require_layout(acc[8], mma_layout),
        tlx.require_layout(acc[9], mma_layout),
        tlx.require_layout(acc[10], mma_layout),
        tlx.require_layout(acc[11], mma_layout),
        tlx.require_layout(acc[12], mma_layout),
        tlx.require_layout(acc[13], mma_layout),
        tlx.require_layout(acc[14], mma_layout),
        tlx.require_layout(acc[15], mma_layout),
        tlx.require_layout(acc[16], mma_layout),
        tlx.require_layout(acc[17], mma_layout),
        tlx.require_layout(acc[18], mma_layout),
        tlx.require_layout(acc[19], mma_layout),
        tlx.require_layout(acc[20], mma_layout),
        tlx.require_layout(acc[21], mma_layout),
        tlx.require_layout(acc[22], mma_layout),
        tlx.require_layout(acc[23], mma_layout),
        tlx.require_layout(acc[24], mma_layout),
        tlx.require_layout(acc[25], mma_layout),
        tlx.require_layout(acc[26], mma_layout),
        tlx.require_layout(acc[27], mma_layout),
        tlx.require_layout(acc[28], mma_layout),
        tlx.require_layout(acc[29], mma_layout),
        tlx.require_layout(acc[30], mma_layout),
        tlx.require_layout(acc[31], mma_layout),
    )
    c00 = tlx.dot(a[0], b[0], acc[0], tiles_per_warp=[1, 1])
    c01 = tlx.dot(a[0], b[1], acc[1], tiles_per_warp=[1, 1])
    c02 = tlx.dot(a[0], b[2], acc[2], tiles_per_warp=[1, 1])
    c03 = tlx.dot(a[0], b[3], acc[3], tiles_per_warp=[1, 1])
    c10 = tlx.dot(a[1], b[0], acc[4], tiles_per_warp=[1, 1])
    c11 = tlx.dot(a[1], b[1], acc[5], tiles_per_warp=[1, 1])
    c12 = tlx.dot(a[1], b[2], acc[6], tiles_per_warp=[1, 1])
    c13 = tlx.dot(a[1], b[3], acc[7], tiles_per_warp=[1, 1])
    c20 = tlx.dot(a[2], b[0], acc[8], tiles_per_warp=[1, 1])
    c21 = tlx.dot(a[2], b[1], acc[9], tiles_per_warp=[1, 1])
    c22 = tlx.dot(a[2], b[2], acc[10], tiles_per_warp=[1, 1])
    c23 = tlx.dot(a[2], b[3], acc[11], tiles_per_warp=[1, 1])
    c30 = tlx.dot(a[3], b[0], acc[12], tiles_per_warp=[1, 1])
    c31 = tlx.dot(a[3], b[1], acc[13], tiles_per_warp=[1, 1])
    c32 = tlx.dot(a[3], b[2], acc[14], tiles_per_warp=[1, 1])
    c33 = tlx.dot(a[3], b[3], acc[15], tiles_per_warp=[1, 1])
    c40 = tlx.dot(a[4], b[0], acc[16], tiles_per_warp=[1, 1])
    c41 = tlx.dot(a[4], b[1], acc[17], tiles_per_warp=[1, 1])
    c42 = tlx.dot(a[4], b[2], acc[18], tiles_per_warp=[1, 1])
    c43 = tlx.dot(a[4], b[3], acc[19], tiles_per_warp=[1, 1])
    c50 = tlx.dot(a[5], b[0], acc[20], tiles_per_warp=[1, 1])
    c51 = tlx.dot(a[5], b[1], acc[21], tiles_per_warp=[1, 1])
    c52 = tlx.dot(a[5], b[2], acc[22], tiles_per_warp=[1, 1])
    c53 = tlx.dot(a[5], b[3], acc[23], tiles_per_warp=[1, 1])
    c60 = tlx.dot(a[6], b[0], acc[24], tiles_per_warp=[1, 1])
    c61 = tlx.dot(a[6], b[1], acc[25], tiles_per_warp=[1, 1])
    c62 = tlx.dot(a[6], b[2], acc[26], tiles_per_warp=[1, 1])
    c63 = tlx.dot(a[6], b[3], acc[27], tiles_per_warp=[1, 1])
    c70 = tlx.dot(a[7], b[0], acc[28], tiles_per_warp=[1, 1])
    c71 = tlx.dot(a[7], b[1], acc[29], tiles_per_warp=[1, 1])
    c72 = tlx.dot(a[7], b[2], acc[30], tiles_per_warp=[1, 1])
    c73 = tlx.dot(a[7], b[3], acc[31], tiles_per_warp=[1, 1])
    return (
        tlx.release_layout(c00), tlx.release_layout(c01),
        tlx.release_layout(c02), tlx.release_layout(c03),
        tlx.release_layout(c10), tlx.release_layout(c11),
        tlx.release_layout(c12), tlx.release_layout(c13),
        tlx.release_layout(c20), tlx.release_layout(c21),
        tlx.release_layout(c22), tlx.release_layout(c23),
        tlx.release_layout(c30), tlx.release_layout(c31),
        tlx.release_layout(c32), tlx.release_layout(c33),
        tlx.release_layout(c40), tlx.release_layout(c41),
        tlx.release_layout(c42), tlx.release_layout(c43),
        tlx.release_layout(c50), tlx.release_layout(c51),
        tlx.release_layout(c52), tlx.release_layout(c53),
        tlx.release_layout(c60), tlx.release_layout(c61),
        tlx.release_layout(c62), tlx.release_layout(c63),
        tlx.release_layout(c70), tlx.release_layout(c71),
        tlx.release_layout(c72), tlx.release_layout(c73),
    )


@triton.jit
def _mma_phase_with_next_reads(a, b, acc, smem_a, smem_b, next_buffer):
    # This is the cadence in Wave's _emit_mma_phase_with_dma_reads.  Each next
    # LDS read follows the last use of the physical operand register it will
    # replace on the next iteration.
    mma_layout: tl.constexpr = tlx.amd_mfma_layout_encoding.make(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[2, 4],
    )
    a_fragment_layout: tl.constexpr = tlx.dot_operand_layout_encoding.make(
        0, mma_layout, 8)
    b_fragment_layout: tl.constexpr = tlx.dot_operand_layout_encoding.make(
        1, mma_layout, 8)
    a = (
        tlx.require_layout(a[0], a_fragment_layout),
        tlx.require_layout(a[1], a_fragment_layout),
        tlx.require_layout(a[2], a_fragment_layout),
        tlx.require_layout(a[3], a_fragment_layout),
        tlx.require_layout(a[4], a_fragment_layout),
        tlx.require_layout(a[5], a_fragment_layout),
        tlx.require_layout(a[6], a_fragment_layout),
        tlx.require_layout(a[7], a_fragment_layout),
    )
    b = (
        tlx.require_layout(b[0], b_fragment_layout),
        tlx.require_layout(b[1], b_fragment_layout),
        tlx.require_layout(b[2], b_fragment_layout),
        tlx.require_layout(b[3], b_fragment_layout),
    )
    acc = (
        tlx.require_layout(acc[0], mma_layout),
        tlx.require_layout(acc[1], mma_layout),
        tlx.require_layout(acc[2], mma_layout),
        tlx.require_layout(acc[3], mma_layout),
        tlx.require_layout(acc[4], mma_layout),
        tlx.require_layout(acc[5], mma_layout),
        tlx.require_layout(acc[6], mma_layout),
        tlx.require_layout(acc[7], mma_layout),
        tlx.require_layout(acc[8], mma_layout),
        tlx.require_layout(acc[9], mma_layout),
        tlx.require_layout(acc[10], mma_layout),
        tlx.require_layout(acc[11], mma_layout),
        tlx.require_layout(acc[12], mma_layout),
        tlx.require_layout(acc[13], mma_layout),
        tlx.require_layout(acc[14], mma_layout),
        tlx.require_layout(acc[15], mma_layout),
        tlx.require_layout(acc[16], mma_layout),
        tlx.require_layout(acc[17], mma_layout),
        tlx.require_layout(acc[18], mma_layout),
        tlx.require_layout(acc[19], mma_layout),
        tlx.require_layout(acc[20], mma_layout),
        tlx.require_layout(acc[21], mma_layout),
        tlx.require_layout(acc[22], mma_layout),
        tlx.require_layout(acc[23], mma_layout),
        tlx.require_layout(acc[24], mma_layout),
        tlx.require_layout(acc[25], mma_layout),
        tlx.require_layout(acc[26], mma_layout),
        tlx.require_layout(acc[27], mma_layout),
        tlx.require_layout(acc[28], mma_layout),
        tlx.require_layout(acc[29], mma_layout),
        tlx.require_layout(acc[30], mma_layout),
        tlx.require_layout(acc[31], mma_layout),
    )
    c00 = tlx.dot(a[0], b[0], acc[0], tiles_per_warp=[1, 1])
    c01 = tlx.dot(a[0], b[1], acc[1], tiles_per_warp=[1, 1])
    c02 = tlx.dot(a[0], b[2], acc[2], tiles_per_warp=[1, 1])
    na0 = _load_a_fragment(smem_a, next_buffer, 0, 0)
    c03 = tlx.dot(a[0], b[3], acc[3], tiles_per_warp=[1, 1])
    na8 = _load_a_fragment(smem_a, next_buffer, 1, 0)

    c10 = tlx.dot(a[1], b[0], acc[4], tiles_per_warp=[1, 1])
    nb0 = _load_b_fragment(smem_b, next_buffer, 0, 0)
    c11 = tlx.dot(a[1], b[1], acc[5], tiles_per_warp=[1, 1])
    c12 = tlx.dot(a[1], b[2], acc[6], tiles_per_warp=[1, 1])
    na1 = _load_a_fragment(smem_a, next_buffer, 0, 1)
    c13 = tlx.dot(a[1], b[3], acc[7], tiles_per_warp=[1, 1])
    na9 = _load_a_fragment(smem_a, next_buffer, 1, 1)

    c20 = tlx.dot(a[2], b[0], acc[8], tiles_per_warp=[1, 1])
    c21 = tlx.dot(a[2], b[1], acc[9], tiles_per_warp=[1, 1])
    nb1 = _load_b_fragment(smem_b, next_buffer, 0, 1)
    c22 = tlx.dot(a[2], b[2], acc[10], tiles_per_warp=[1, 1])
    c23 = tlx.dot(a[2], b[3], acc[11], tiles_per_warp=[1, 1])
    na2 = _load_a_fragment(smem_a, next_buffer, 0, 2)
    na10 = _load_a_fragment(smem_a, next_buffer, 1, 2)

    c30 = tlx.dot(a[3], b[0], acc[12], tiles_per_warp=[1, 1])
    c31 = tlx.dot(a[3], b[1], acc[13], tiles_per_warp=[1, 1])
    nb2 = _load_b_fragment(smem_b, next_buffer, 0, 2)
    c32 = tlx.dot(a[3], b[2], acc[14], tiles_per_warp=[1, 1])
    c33 = tlx.dot(a[3], b[3], acc[15], tiles_per_warp=[1, 1])
    na11 = _load_a_fragment(smem_a, next_buffer, 1, 3)

    c40 = tlx.dot(a[4], b[0], acc[16], tiles_per_warp=[1, 1])
    na3 = _load_a_fragment(smem_a, next_buffer, 0, 3)
    c41 = tlx.dot(a[4], b[1], acc[17], tiles_per_warp=[1, 1])
    c42 = tlx.dot(a[4], b[2], acc[18], tiles_per_warp=[1, 1])
    nb3 = _load_b_fragment(smem_b, next_buffer, 0, 3)
    c43 = tlx.dot(a[4], b[3], acc[19], tiles_per_warp=[1, 1])
    na12 = _load_a_fragment(smem_a, next_buffer, 1, 4)

    c50 = tlx.dot(a[5], b[0], acc[20], tiles_per_warp=[1, 1])
    na4 = _load_a_fragment(smem_a, next_buffer, 0, 4)
    c51 = tlx.dot(a[5], b[1], acc[21], tiles_per_warp=[1, 1])
    c52 = tlx.dot(a[5], b[2], acc[22], tiles_per_warp=[1, 1])
    c53 = tlx.dot(a[5], b[3], acc[23], tiles_per_warp=[1, 1])
    na5 = _load_a_fragment(smem_a, next_buffer, 0, 5)
    na13 = _load_a_fragment(smem_a, next_buffer, 1, 5)

    c60 = tlx.dot(a[6], b[0], acc[24], tiles_per_warp=[1, 1])
    c61 = tlx.dot(a[6], b[1], acc[25], tiles_per_warp=[1, 1])
    na6 = _load_a_fragment(smem_a, next_buffer, 0, 6)
    c62 = tlx.dot(a[6], b[2], acc[26], tiles_per_warp=[1, 1])
    c63 = tlx.dot(a[6], b[3], acc[27], tiles_per_warp=[1, 1])
    na7 = _load_a_fragment(smem_a, next_buffer, 0, 7)
    na14 = _load_a_fragment(smem_a, next_buffer, 1, 6)

    c70 = tlx.dot(a[7], b[0], acc[28], tiles_per_warp=[1, 1])
    nb4 = _load_b_fragment(smem_b, next_buffer, 1, 0)
    c71 = tlx.dot(a[7], b[1], acc[29], tiles_per_warp=[1, 1])
    nb5 = _load_b_fragment(smem_b, next_buffer, 1, 1)
    c72 = tlx.dot(a[7], b[2], acc[30], tiles_per_warp=[1, 1])
    nb6 = _load_b_fragment(smem_b, next_buffer, 1, 2)
    c73 = tlx.dot(a[7], b[3], acc[31], tiles_per_warp=[1, 1])
    nb7 = _load_b_fragment(smem_b, next_buffer, 1, 3)
    na15 = _load_a_fragment(smem_a, next_buffer, 1, 7)

    next_a0 = (na0, na1, na2, na3, na4, na5, na6, na7)
    next_b0 = (nb0, nb1, nb2, nb3)
    next_a1 = (na8, na9, na10, na11, na12, na13, na14, na15)
    next_b1 = (nb4, nb5, nb6, nb7)
    next_acc = (
        tlx.release_layout(c00), tlx.release_layout(c01),
        tlx.release_layout(c02), tlx.release_layout(c03),
        tlx.release_layout(c10), tlx.release_layout(c11),
        tlx.release_layout(c12), tlx.release_layout(c13),
        tlx.release_layout(c20), tlx.release_layout(c21),
        tlx.release_layout(c22), tlx.release_layout(c23),
        tlx.release_layout(c30), tlx.release_layout(c31),
        tlx.release_layout(c32), tlx.release_layout(c33),
        tlx.release_layout(c40), tlx.release_layout(c41),
        tlx.release_layout(c42), tlx.release_layout(c43),
        tlx.release_layout(c50), tlx.release_layout(c51),
        tlx.release_layout(c52), tlx.release_layout(c53),
        tlx.release_layout(c60), tlx.release_layout(c61),
        tlx.release_layout(c62), tlx.release_layout(c63),
        tlx.release_layout(c70), tlx.release_layout(c71),
        tlx.release_layout(c72), tlx.release_layout(c73),
    )
    return next_acc, next_a0, next_b0, next_a1, next_b1


@triton.jit
def _store_acc_tile(c_ptr, acc, pid_m, pid_n, stride_cm, stride_cn,
                    local_m: tl.constexpr, local_n: tl.constexpr):
    m = tl.arange(0, 32)
    n = tl.arange(0, 64)
    offs_m = pid_m * 256 + (m // 16) * 128 + local_m * 16 + m % 16
    offs_n = pid_n * 256 + (n // 16) * 64 + local_n * 16 + n % 16
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc.to(tl.float16))


@triton.jit
def _store_accumulators(c_ptr, acc, pid_m, pid_n, stride_cm, stride_cn):
    _store_acc_tile(c_ptr, acc[0], pid_m, pid_n, stride_cm, stride_cn, 0, 0)
    _store_acc_tile(c_ptr, acc[1], pid_m, pid_n, stride_cm, stride_cn, 0, 1)
    _store_acc_tile(c_ptr, acc[2], pid_m, pid_n, stride_cm, stride_cn, 0, 2)
    _store_acc_tile(c_ptr, acc[3], pid_m, pid_n, stride_cm, stride_cn, 0, 3)
    _store_acc_tile(c_ptr, acc[4], pid_m, pid_n, stride_cm, stride_cn, 1, 0)
    _store_acc_tile(c_ptr, acc[5], pid_m, pid_n, stride_cm, stride_cn, 1, 1)
    _store_acc_tile(c_ptr, acc[6], pid_m, pid_n, stride_cm, stride_cn, 1, 2)
    _store_acc_tile(c_ptr, acc[7], pid_m, pid_n, stride_cm, stride_cn, 1, 3)
    _store_acc_tile(c_ptr, acc[8], pid_m, pid_n, stride_cm, stride_cn, 2, 0)
    _store_acc_tile(c_ptr, acc[9], pid_m, pid_n, stride_cm, stride_cn, 2, 1)
    _store_acc_tile(c_ptr, acc[10], pid_m, pid_n, stride_cm, stride_cn, 2, 2)
    _store_acc_tile(c_ptr, acc[11], pid_m, pid_n, stride_cm, stride_cn, 2, 3)
    _store_acc_tile(c_ptr, acc[12], pid_m, pid_n, stride_cm, stride_cn, 3, 0)
    _store_acc_tile(c_ptr, acc[13], pid_m, pid_n, stride_cm, stride_cn, 3, 1)
    _store_acc_tile(c_ptr, acc[14], pid_m, pid_n, stride_cm, stride_cn, 3, 2)
    _store_acc_tile(c_ptr, acc[15], pid_m, pid_n, stride_cm, stride_cn, 3, 3)
    _store_acc_tile(c_ptr, acc[16], pid_m, pid_n, stride_cm, stride_cn, 4, 0)
    _store_acc_tile(c_ptr, acc[17], pid_m, pid_n, stride_cm, stride_cn, 4, 1)
    _store_acc_tile(c_ptr, acc[18], pid_m, pid_n, stride_cm, stride_cn, 4, 2)
    _store_acc_tile(c_ptr, acc[19], pid_m, pid_n, stride_cm, stride_cn, 4, 3)
    _store_acc_tile(c_ptr, acc[20], pid_m, pid_n, stride_cm, stride_cn, 5, 0)
    _store_acc_tile(c_ptr, acc[21], pid_m, pid_n, stride_cm, stride_cn, 5, 1)
    _store_acc_tile(c_ptr, acc[22], pid_m, pid_n, stride_cm, stride_cn, 5, 2)
    _store_acc_tile(c_ptr, acc[23], pid_m, pid_n, stride_cm, stride_cn, 5, 3)
    _store_acc_tile(c_ptr, acc[24], pid_m, pid_n, stride_cm, stride_cn, 6, 0)
    _store_acc_tile(c_ptr, acc[25], pid_m, pid_n, stride_cm, stride_cn, 6, 1)
    _store_acc_tile(c_ptr, acc[26], pid_m, pid_n, stride_cm, stride_cn, 6, 2)
    _store_acc_tile(c_ptr, acc[27], pid_m, pid_n, stride_cm, stride_cn, 6, 3)
    _store_acc_tile(c_ptr, acc[28], pid_m, pid_n, stride_cm, stride_cn, 7, 0)
    _store_acc_tile(c_ptr, acc[29], pid_m, pid_n, stride_cm, stride_cn, 7, 1)
    _store_acc_tile(c_ptr, acc[30], pid_m, pid_n, stride_cm, stride_cn, 7, 2)
    _store_acc_tile(c_ptr, acc[31], pid_m, pid_n, stride_cm, stride_cn, 7, 3)


@triton.jit
def wave_8wave(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    GRID_MN: tl.constexpr,
):
    raw_pid_m = tl.program_id(0)
    raw_pid_n = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    tl.assume(M > 0)
    tl.assume(N > 0)
    tl.assume(raw_pid_m >= 0)
    tl.assume(raw_pid_m < num_pid_m)
    tl.assume(raw_pid_n >= 0)
    tl.assume(raw_pid_n < num_pid_n)

    pid = raw_pid_n * num_pid_m + raw_pid_m
    if NUM_XCDS != 1:
        pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
        tall_xcds = GRID_MN % NUM_XCDS
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
        # Keep the common balanced mapping structural.  Besides matching
        # Wave's CTA mapping contract, this prevents an unreachable ragged
        # branch from turning constant divisors into runtime signed division.
        if tall_xcds == 0:
            pid = xcd * pids_per_xcd + local_pid
        else:
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
        if M % (BLOCK_M * GROUP_SIZE_M) == 0:
            group_size_m: tl.constexpr = GROUP_SIZE_M
        else:
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            tl.assume(group_size_m > 0)
        pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
        pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    K_PHASE: tl.constexpr = BLOCK_K // 2
    MMA_M: tl.constexpr = 16
    MMA_N: tl.constexpr = 16
    A_M_TILES: tl.constexpr = BLOCK_M // MMA_M
    B_N_TILES: tl.constexpr = BLOCK_N // MMA_N
    K_PHASES: tl.constexpr = BLOCK_K // K_PHASE
    ring_layout: tl.constexpr = tlx.swizzled_shared_layout_encoding.with_order(
        [2, 1, 0], vectorSize=8, perPhase=2, maxPhase=8)
    a_load_layout: tl.constexpr = tlx.swizzled_shared_layout_encoding.with_order(
        [5, 4, 3, 2, 1, 0], vectorSize=8, perPhase=2, maxPhase=8)
    b_load_layout: tl.constexpr = tlx.swizzled_shared_layout_encoding.with_order(
        [4, 5, 3, 2, 1, 0], vectorSize=8, perPhase=2, maxPhase=8)

    # Match Wave's single 128 KiB double-buffered ring.  Every descriptor below
    # spans the complete allocation, so lowering can express each as a pure
    # same-size reinterpret.  Each ring is [A 32 KiB, B 32 KiB].
    smem_a = tlx.local_alloc(
        (2, 2, 2, 8, 16, 32), tl.float16, 2, layout=a_load_layout)
    smem_b = tlx.local_alloc(
        (2, 2, 4, 4, 32, 16), tl.float16, 2, reuse=smem_a,
        layout=b_load_layout)
    smem_a_dma = tlx.local_alloc(
        (16, 32, 32), tl.float16, 4, reuse=smem_a,
        layout=ring_layout)
    smem_b_dma = tlx.local_alloc(
        (8, 64, 32), tl.float16, 4, reuse=smem_a,
        layout=ring_layout)

    offs_a_slot = tl.arange(0, K_PHASES * 8)
    offs_b_slot = tl.arange(0, K_PHASES * 4)
    offs_a_m = tl.arange(0, 32)
    offs_b_n = tl.arange(0, 64)
    offs_k_inner = tl.arange(0, K_PHASE)
    wave_id = tlx.thread_id(0) // 64
    # The DMA destination is component-major: register component ``r`` from
    # wave ``w`` is written to slot ``r * 8 + w``.  Recover that pair from
    # the logical coordinates assigned by each blocked source layout before
    # selecting the corresponding global tile.  Indexing the logical slot
    # and row dimensions directly would instead transpose register and warp
    # bits (for example A component zero would fetch tiles
    # [0, 8, 1, 9, ...] across waves while LDS slots expect [0, 1, 2, 3, ...]).
    a_component = offs_a_slot[:, None] // 4
    a_m_tile = wave_id + (a_component % 2) * 8
    offs_am = (
        pid_m * BLOCK_M
        + a_m_tile * MMA_M
        + offs_a_m[None, :] % MMA_M
    )

    b_component = offs_b_slot[:, None] // 2
    b_n_tile = wave_id + (b_component % 2) * 8
    offs_bn = (
        pid_n * BLOCK_N
        + b_n_tile * MMA_N
        + offs_b_n[None, :] % MMA_N
    )
    a_off = (
        offs_am[:, :, None] * stride_am
        + ((a_component[:, :, None] // 2) * K_PHASE
           + offs_k_inner[None, None, :]) * stride_ak
    )
    b_off = (
        ((b_component[:, :, None] // 2) * K_PHASE
         + offs_k_inner[None, None, :]) * stride_bk
        + offs_bn[:, :, None] * stride_bn
    )
    zero = tl.zeros((32, 64), dtype=tl.float32)
    acc = (
        zero, zero, zero, zero, zero, zero, zero, zero,
        zero, zero, zero, zero, zero, zero, zero, zero,
        zero, zero, zero, zero, zero, zero, zero, zero,
        zero, zero, zero, zero, zero, zero, zero, zero,
    )
    iter_max: tl.constexpr = K // BLOCK_K

    initial_a_dma0 = smem_a_dma[0]
    initial_b_dma0 = smem_b_dma[1]
    tlx.buffer_load_to_local(initial_a_dma0, a_ptr, a_off)
    tlx.buffer_load_to_local(initial_b_dma0, b_ptr, b_off)
    tlx.async_load_commit_group(issue_group_size=7, issue_delay_cycles=68)

    initial_a_dma1 = smem_a_dma[2]
    initial_b_dma1 = smem_b_dma[3]
    tlx.buffer_load_to_local(
        initial_a_dma1, a_ptr, a_off + BLOCK_K * stride_ak)
    tlx.buffer_load_to_local(
        initial_b_dma1, b_ptr, b_off + BLOCK_K * stride_bk)
    tlx.async_load_commit_group(issue_group_size=7, issue_delay_cycles=68)

    tlx.async_load_wait_group(1)
    a0, b0 = _load_phase(smem_a, smem_b, 0, 0)
    a1, b1 = _load_phase(smem_a, smem_b, 0, 1)

    for k in tl.range(0, iter_max - 2, num_stages=1):
        consumed = k % 2
        next_buffer = 1 - consumed
        refill_k = (k + 2) * BLOCK_K

        # Refill the buffer whose payload was consumed by the CTA.  The dot
        # operands read LDS written by all eight waves.
        refill_a_dma = smem_a_dma[consumed * 2]
        refill_b_dma = smem_b_dma[consumed * 2 + 1]
        tlx.buffer_load_to_local(
            refill_a_dma, a_ptr, a_off + refill_k * stride_ak)
        tlx.buffer_load_to_local(
            refill_b_dma, b_ptr, b_off + refill_k * stride_bk)
        tlx.async_load_commit_group(
            issue_group_size=7,
            issue_delay_cycles=46,
            issue_delay_overlap_cycles=33,
            issue_delay_skip_thread_threshold=256,
        )

        acc = _mma_phase(a0, b0, acc)

        # Complete only the older DMA group before consuming next_buffer.
        # The refill issued above remains live and overlaps the second MMA
        # phase, exactly as required by wait_group(1).
        tlx.async_load_wait_group(1)

        acc, a0, b0, a1, b1 = _mma_phase_with_next_reads(
            a1, b1, acc, smem_a, smem_b, next_buffer)

    acc = _mma_phase(a0, b0, acc)
    acc = _mma_phase(a1, b1, acc)
    tlx.async_load_wait_group(0)

    last_buffer: tl.constexpr = (iter_max - 1) % 2
    a0, b0 = _load_phase(smem_a, smem_b, last_buffer, 0)
    a1, b1 = _load_phase(smem_a, smem_b, last_buffer, 1)
    acc = _mma_phase(a0, b0, acc)
    acc = _mma_phase(a1, b1, acc)

    _store_accumulators(c_ptr, acc, pid_m, pid_n, stride_cm, stride_cn)


def matmul(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert b.stride(0) == 1, "wave_8wave expects a K-contiguous transposed-B view"
    M, K = a.shape
    K, N = b.shape
    BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 64
    assert M % BLOCK_M == 0 and N % BLOCK_N == 0
    NUM_XCDS = 8
    GROUP_SIZE_M = 4
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid_m = triton.cdiv(M, BLOCK_M)
    grid_n = triton.cdiv(N, BLOCK_N)
    grid_mn = grid_m * grid_n
    wave_8wave[(grid_m, grid_n)](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        NUM_XCDS=NUM_XCDS,
        GRID_MN=grid_mn,
        num_warps=8,
        num_stages=1,
        matrix_instr_nonkdim=16,
    )
    return c
