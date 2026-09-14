"""TLX AMD tests -- CDNA4 (gfx950)."""
import pytest
import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
import traceback
from triton._internal_testing import is_hip_cdna4
from triton.language.extra.tlx.tutorials.amd_fa_cluster import (
    _cluster_causal_query_tile as _amd_fa_cluster_causal_query_tile,
    _cluster_direct_workgroup_window as _amd_fa_cluster_direct_workgroup_window,
    persistent_attention as _amd_fa_cluster_persistent_attention,
)
from triton.language.extra.tlx.tutorials.gfx9_gemm.intra_wave.a4w4.bench import (
    generate_mxfp4_inputs as _generate_a4w4_inputs,
    launch_matmul as _launch_a4w4,
    torch_reference as _a4w4_reference,
)
from triton.language.extra.tlx.tutorials.gfx9_gemm.inter_wave.a4w4.matmul_kernel import (
    BLOCK_M as _A4W4_INTER_WAVE_BLOCK_M,
    BLOCK_N as _A4W4_INTER_WAVE_BLOCK_N,
    MIN_K as _A4W4_INTER_WAVE_MIN_K,
    matmul as _a4w4_inter_wave_matmul,
    matmul_merged_scales as _a4w4_inter_wave_matmul_merged_scales,
    matmul_preshuffled as _a4w4_inter_wave_matmul_preshuffled,
    preshuffle_mxfp4_a_scales as _preshuffle_a4w4_a_scales,
    preshuffle_mxfp4_b_scales as _preshuffle_a4w4_b_scales,
    preshuffle_mxfp4_scales as _preshuffle_a4w4_scales,
    select_matmul_path as _select_a4w4_inter_wave_path,
)


@triton.jit
def _amd_fa_cluster_causal_query_order_kernel(output, N_CTX: tl.constexpr, BLOCK_M: tl.constexpr):
    raw_pid_m = tl.program_id(0)
    pid_m = _amd_fa_cluster_causal_query_tile(raw_pid_m, N_CTX, BLOCK_M)
    tl.store(output + raw_pid_m, pid_m)


@triton.jit
def _amd_fa_cluster_workgroup_order_kernel(
    output,
    H: tl.constexpr,
    N_CTX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    USE_24_HEAD_WINDOW: tl.constexpr,
    USE_FACTORED_24: tl.constexpr,
):
    raw_off_h = tl.program_id(0)
    raw_pid_m = tl.program_id(1)
    off_h, pid_m = _amd_fa_cluster_direct_workgroup_window(
        raw_off_h,
        raw_pid_m,
        H,
        N_CTX,
        BLOCK_M,
        IS_CAUSAL,
        USE_24_HEAD_WINDOW,
        USE_FACTORED_24,
    )
    if IS_CAUSAL:
        pid_m = _amd_fa_cluster_causal_query_tile(pid_m, N_CTX, BLOCK_M)
    num_m_blocks: tl.constexpr = (N_CTX + BLOCK_M - 1) // BLOCK_M
    raw_linear = raw_off_h + raw_pid_m * H
    mapped_linear = off_h + pid_m * H
    tl.store(output + raw_linear, mapped_linear, mask=raw_pid_m < num_m_blocks)


@triton.jit
def _warp_predicate_update(lhs, rhs, increment, side_ptr, offsets):
    tl.store(side_ptr + offsets, lhs)
    return lhs + increment, rhs - increment


@triton.jit
def _warp_predicate_kernel(x_ptr, lhs_ptr, rhs_ptr, side_ptr, size: tl.constexpr):
    offsets = tl.arange(0, size)
    lhs = tl.load(x_ptr + offsets)
    rhs = lhs * 2.0
    predicate = (offsets >= 64) & (offsets < 128) & (offsets % 5 < 2)
    lhs, rhs = tlx.warp_predicate(
        predicate,
        (lhs, rhs),
        _warp_predicate_update,
        args=(3.0, side_ptr, offsets),
    )
    tl.store(lhs_ptr + offsets, lhs)
    tl.store(rhs_ptr + offsets, rhs)


@triton.jit
def _warp_predicate_warp_local_reduce(value):
    row_sum = tl.sum(value, axis=1)
    return value + row_sum[:, None]


@triton.jit
def _warp_predicate_warp_local_reduce_kernel(x_ptr, output_ptr):
    offsets = tl.arange(0, 256)
    value = tl.reshape(tl.load(x_ptr + offsets), (4, 64))
    wave = tlx.thread_id(0) // 64
    predicate = wave >= 2
    value = tlx.warp_predicate(predicate, value, _warp_predicate_warp_local_reduce, wave_uniform=True)
    tl.store(output_ptr + offsets, tl.reshape(value, (256, )))


@triton.jit
def _warp_vote_kernel(x_ptr, all_ptr, any_ptr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    predicate = tl.load(x_ptr + offsets) != 0
    all_value = tlx.warp_all(predicate).to(tl.int32)
    any_value = tlx.warp_any(predicate).to(tl.int32)
    tl.store(all_ptr + offsets, all_value)
    tl.store(any_ptr + offsets, any_value)


@triton.jit
def _concrete_predicate_scale(value, scale):
    return value * scale


@triton.jit
def _concrete_dot_control_flow_helper(a, b, condition, predicate, MMA: tl.constexpr, DOT0: tl.constexpr,
                                      DOT1: tl.constexpr):
    a = tlx.require_layout(a, DOT0, pin=False)
    b = tlx.require_layout(b, DOT1, pin=False)
    acc = tlx.require_layout(tl.zeros((16, 64), tl.float32), MMA, pin=False)
    result = tl.dot(a, b, acc)
    if condition:
        result = result * 2.0
    else:
        result = result + 1.0
    return tlx.warp_predicate(predicate, result, _concrete_predicate_scale, args=(0.5, ))


@triton.jit
def _concrete_helper_release_kernel(a_ptr, b_ptr, output_ptr, condition):
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    dot0: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=8)
    dot1: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=8)
    rows = tl.arange(0, 16)
    reduction = tl.arange(0, 32)
    cols = tl.arange(0, 64)
    a = tl.load(a_ptr + rows[:, None] * 32 + reduction[None, :])
    b = tl.load(b_ptr + reduction[:, None] * 64 + cols[None, :])
    predicate = (rows[:, None] < 8) & (cols[None, :] >= 0)
    concrete = _concrete_dot_control_flow_helper(a, b, condition, predicate, mma, dot0, dot1)

    offsets = rows[:, None] * 64 + cols[None, :]
    # Fixup must specialize this pointer use when the helper result acquires
    # its concrete MFMA layout.
    tl.store(output_ptr + offsets, concrete)
    # The call result is still encoding-free while the Python frontend builds
    # this operation. The release remains as a deliberate layout-domain edge
    # after helper-ABI specialization and lets this store choose a fresh layout.
    generic = tlx.release_layout(concrete)
    tl.store(output_ptr + 1024 + offsets, generic)


@triton.jit
def _async_load_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    buffers = tlx.local_alloc((BLOCK_SIZE, ), tl.float32, 2)

    buf0 = tlx.local_view(buffers, 0)
    buf1 = tlx.local_view(buffers, 1)
    tok_x = tlx.async_load(x_ptr + offs, buf0, mask=mask)
    tok_y = tlx.async_load(y_ptr + offs, buf1, mask=mask)
    tlx.async_load_commit_group([tok_x, tok_y])
    tlx.async_load_wait_group(0)

    x = tlx.local_load(buf0)
    y = tlx.local_load(buf1)
    tl.store(output_ptr + offs, x + y, mask=mask)


@triton.jit
def _extract_slice_dot1_kernel(
    x_ptr,
    y_ptr,
    ROW_OFFSET: tl.constexpr,
    COL_OFFSET: tl.constexpr,
):
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    dot1: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=8)
    rows = tl.arange(0, 256)
    cols = tl.arange(0, 128)
    values = tl.load(x_ptr + rows[:, None] * 128 + cols[None, :])
    values = tlx.require_layout(values, dot1, pin=False)
    band = tlx.extract_slice(values, [32, 64], [ROW_OFFSET, COL_OFFSET])
    band_rows = tl.arange(0, 32)
    band_cols = tl.arange(0, 64)
    out_ptrs = y_ptr + band_rows[:, None] * 64 + band_cols[None, :]
    out_ptrs = tlx.require_layout(out_ptrs, dot1, pin=False)
    tl.store(out_ptrs, band)


@triton.jit
def _extract_slice_mfma_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    BAND: tl.constexpr,
):
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    dot0: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=8)
    dot1: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=8)
    rows = tl.arange(0, 16)
    reduction = tl.arange(0, 256)
    cols = tl.arange(0, 64)
    a = tl.load(a_ptr + rows[:, None] * 256 + reduction[None, :])
    b = tl.load(b_ptr + reduction[:, None] * 64 + cols[None, :])
    a = tlx.require_layout(a, dot0, pin=False)
    b = tlx.require_layout(b, dot1, pin=False)
    a_band = tlx.extract_slice(a, [16, 32], [0, BAND * 32])
    b_band = tlx.extract_slice(b, [32, 64], [BAND * 32, 0])
    acc = tlx.zeros((16, 64), tl.float32, layout=mma)
    result = tl.dot(a_band, b_band, acc)
    output_offsets = output_ptr + rows[:, None] * 64 + cols[None, :]
    output_offsets = tlx.require_layout(output_offsets, mma, pin=False)
    tl.store(output_offsets, result)


@triton.jit
def _amd_register_class_anchor_kernel(x_ptr, y_ptr, REGISTER_CLASS: tl.constexpr):
    offsets = tl.arange(0, 2048)
    values = tl.load(x_ptr + offsets)
    values = tlx.amd_register_class_anchor(values, register_class=REGISTER_CLASS)
    tl.store(y_ptr + offsets, values)


@triton.jit
def _amd_scheduled_mfma_kernel(a_ptr, b_ptr, output_ptr, K_WIDTH: tl.constexpr):
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    dot0: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=K_WIDTH)
    dot1: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=K_WIDTH)
    rows = tl.arange(0, 16)
    reduction = tl.arange(0, 32)
    cols = tl.arange(0, 64)
    a = tl.load(a_ptr + rows[:, None] * 32 + reduction[None, :])
    b = tl.load(b_ptr + reduction[:, None] * 64 + cols[None, :])
    a = tlx.require_layout(a, dot0, pin=False)
    b = tlx.require_layout(b, dot1, pin=False)
    b = tlx.amd_register_resident(b, register_class="agpr", registers_per_group=4)
    acc = tl.full((16, 64), 7.0, tl.float32)
    acc = tlx.require_layout(acc, mma, pin=False)
    result = tlx.amd_scheduled_mfma(
        a,
        b,
        acc,
        resident_operand=1,
        accumulator_role="transient",
        initialize=True,
    )
    result, _ = tlx.amd_mfma_commit(result, b)
    output_offsets = output_ptr + rows[:, None] * 64 + cols[None, :]
    output_offsets = tlx.require_layout(output_offsets, mma, pin=False)
    tl.store(output_offsets, result)


@triton.jit
def _amd_scheduled_mfma_chain_kernel(a_ptr, b_ptr, output_ptr, BANDS: tl.constexpr):
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    dot0: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=8)
    dot1: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=8)
    rows = tl.arange(0, 16)
    reduction = tl.arange(0, 256)
    cols = tl.arange(0, 64)
    a = tl.load(a_ptr + rows[:, None] * 256 + reduction[None, :])
    b = tl.load(b_ptr + reduction[:, None] * 64 + cols[None, :])
    a = tlx.require_layout(a, dot0, pin=False)
    b = tlx.require_layout(b, dot1, pin=False)
    b = tlx.amd_register_resident(b, register_class="agpr", registers_per_group=32)
    acc = tlx.zeros((16, 64), tl.float32, layout=mma)

    for band in tl.static_range(BANDS):
        a_band = tlx.extract_slice(a, [16, 32], [0, band * 32])
        b_band = tlx.extract_slice(b, [32, 64], [band * 32, 0])
        acc = tlx.amd_scheduled_mfma(
            a_band,
            b_band,
            acc,
            resident_operand=1,
            accumulator_role="transient",
            initialize=band == 0,
        )
    acc, _ = tlx.amd_mfma_commit(acc, b_band)
    output_offsets = output_ptr + rows[:, None] * 64 + cols[None, :]
    output_offsets = tlx.require_layout(output_offsets, mma, pin=False)
    tl.store(output_offsets, acc)


@triton.jit
def _amd_scheduled_mfma_persistent_acc_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    USE_VGPR: tl.constexpr,
):
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    dot0: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=8)
    dot1: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=8)
    rows = tl.arange(0, 16)
    reduction = tl.arange(0, 64)
    cols = tl.arange(0, 64)
    a = tl.load(a_ptr + rows[:, None] * 64 + reduction[None, :])
    b = tl.load(b_ptr + reduction[:, None] * 64 + cols[None, :])
    a = tlx.require_layout(a, dot0, pin=False)
    b = tlx.require_layout(b, dot1, pin=False)
    a0 = tlx.extract_slice(a, [16, 32], [0, 0])
    b0 = tlx.extract_slice(b, [32, 64], [0, 0])
    acc = tlx.zeros((16, 64), tl.float32, layout=mma)
    acc = tlx.amd_scheduled_mfma(
        a0,
        b0,
        acc,
        accumulator_role="persistent",
        accumulator_register_class="vgpr" if USE_VGPR else None,
        initialize=True,
    )
    a1 = tlx.extract_slice(a, [16, 32], [0, 32])
    b1 = tlx.extract_slice(b, [32, 64], [32, 0])
    acc = tlx.amd_scheduled_mfma(
        a1,
        b1,
        acc,
        accumulator_role="persistent",
        accumulator_register_class="vgpr" if USE_VGPR else None,
    )
    output_offsets = output_ptr + rows[:, None] * 64 + cols[None, :]
    output_offsets = tlx.require_layout(output_offsets, mma, pin=False)
    tl.store(output_offsets, acc)


@triton.jit
def _amd_scheduled_mfma_32x32_kernel(a_ptr, b_ptr, output_ptr):
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[32, 32, 16],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    dot0: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=8)
    dot1: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=8)
    rows = tl.arange(0, 128)
    reduction = tl.arange(0, 16)
    cols = tl.arange(0, 32)
    a = tlx.require_layout(
        tl.load(a_ptr + rows[:, None] * 16 + reduction[None, :]),
        dot0,
        pin=False,
    )
    b = tlx.require_layout(
        tl.load(b_ptr + reduction[:, None] * 32 + cols[None, :]),
        dot1,
        pin=False,
    )
    acc = tlx.zeros((128, 32), tl.float32, layout=mma)
    result = tlx.amd_scheduled_mfma(
        a,
        b,
        acc,
        accumulator_role="transient",
        initialize=True,
    )
    result, _ = tlx.amd_mfma_commit(result, b)
    offsets = tlx.require_layout(
        output_ptr + rows[:, None] * 32 + cols[None, :],
        mma,
        pin=False,
    )
    tl.store(offsets, result)


@triton.jit
def _amd_scheduled_mfma_fragmented_nd_kernel(a_ptr, b_ptr, output_ptr):
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[32, 32, 16],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    dot0: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=8)
    dot1: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=8)
    rows = tl.arange(0, 256)
    reduction = tl.arange(0, 16)
    cols = tl.arange(0, 128)
    a = tl.load(a_ptr + rows[:, None] * 16 + reduction[None, :])
    b = tl.load(b_ptr + reduction[:, None] * 128 + cols[None, :])
    a = tlx.require_layout(a, dot0, pin=False)
    b = tlx.require_layout(b, dot1, pin=False)

    a_lo = tlx.extract_slice(a, [128, 16], [0, 0])
    a_hi = tlx.extract_slice(a, [128, 16], [128, 0])
    b0 = tlx.extract_slice(b, [16, 32], [0, 0])
    b1 = tlx.extract_slice(b, [16, 32], [0, 32])
    b2 = tlx.extract_slice(b, [16, 32], [0, 64])
    b3 = tlx.extract_slice(b, [16, 32], [0, 96])
    acc = tlx.zeros((256, 128), tl.float32, layout=mma)
    c00 = tlx.extract_slice(acc, [128, 32], [0, 0])
    c10 = tlx.extract_slice(acc, [128, 32], [128, 0])
    c01 = tlx.extract_slice(acc, [128, 32], [0, 32])
    c11 = tlx.extract_slice(acc, [128, 32], [128, 32])
    c02 = tlx.extract_slice(acc, [128, 32], [0, 64])
    c12 = tlx.extract_slice(acc, [128, 32], [128, 64])
    c03 = tlx.extract_slice(acc, [128, 32], [0, 96])
    c13 = tlx.extract_slice(acc, [128, 32], [128, 96])

    tl.debug_barrier()
    c00 = tlx.amd_scheduled_mfma(a_lo, b0, c00, accumulator_role="transient", initialize=True)
    c10 = tlx.amd_scheduled_mfma(a_hi, b0, c10, accumulator_role="transient", initialize=True)
    c01 = tlx.amd_scheduled_mfma(a_lo, b1, c01, accumulator_role="transient", initialize=True)
    c11 = tlx.amd_scheduled_mfma(a_hi, b1, c11, accumulator_role="transient", initialize=True)
    c02 = tlx.amd_scheduled_mfma(a_lo, b2, c02, accumulator_role="transient", initialize=True)
    c12 = tlx.amd_scheduled_mfma(a_hi, b2, c12, accumulator_role="transient", initialize=True)
    c03 = tlx.amd_scheduled_mfma(a_lo, b3, c03, accumulator_role="transient", initialize=True)
    c13 = tlx.amd_scheduled_mfma(a_hi, b3, c13, accumulator_role="transient", initialize=True)
    c00, _ = tlx.amd_mfma_commit(c00, b3)
    c10, _ = tlx.amd_mfma_commit(c10, b3)
    c01, _ = tlx.amd_mfma_commit(c01, b3)
    c11, _ = tlx.amd_mfma_commit(c11, b3)
    c02, _ = tlx.amd_mfma_commit(c02, b3)
    c12, _ = tlx.amd_mfma_commit(c12, b3)
    c03, _ = tlx.amd_mfma_commit(c03, b3)
    c13, _ = tlx.amd_mfma_commit(c13, b3)

    fragment_rows = tl.arange(0, 128)
    fragment_cols = tl.arange(0, 32)
    out00 = output_ptr + fragment_rows[:, None] * 128 + fragment_cols[None, :]
    out10 = out00 + 128 * 128
    out01 = out00 + 32
    out11 = out10 + 32
    out02 = out00 + 64
    out12 = out10 + 64
    out03 = out00 + 96
    out13 = out10 + 96
    tl.store(tlx.require_layout(out00, mma, pin=False), c00)
    tl.store(tlx.require_layout(out10, mma, pin=False), c10)
    tl.store(tlx.require_layout(out01, mma, pin=False), c01)
    tl.store(tlx.require_layout(out11, mma, pin=False), c11)
    tl.store(tlx.require_layout(out02, mma, pin=False), c02)
    tl.store(tlx.require_layout(out12, mma, pin=False), c12)
    tl.store(tlx.require_layout(out03, mma, pin=False), c03)
    tl.store(tlx.require_layout(out13, mma, pin=False), c13)


@triton.jit
def _amd_scheduled_mfma_fragmented_nd_update_kernel(
    a0_ptr,
    b0_ptr,
    a1_ptr,
    b1_ptr,
    output_ptr,
    DIRECT_STORE: tl.constexpr,
):
    """Match the GQA dK path: a full update, then eight persistent updates."""
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[32, 32, 16],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    dot0: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=8)
    dot1: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=8)
    rows = tl.arange(0, 256)
    reduction = tl.arange(0, 16)
    cols = tl.arange(0, 128)
    a0 = tlx.require_layout(
        tl.load(a0_ptr + rows[:, None] * 16 + reduction[None, :]),
        dot0,
        pin=False,
    )
    b0 = tlx.require_layout(
        tl.load(b0_ptr + reduction[:, None] * 128 + cols[None, :]),
        dot1,
        pin=False,
    )
    a1 = tlx.require_layout(
        tl.load(a1_ptr + rows[:, None] * 16 + reduction[None, :]),
        dot0,
        pin=False,
    )
    b1 = tlx.require_layout(
        tl.load(b1_ptr + reduction[:, None] * 128 + cols[None, :]),
        dot1,
        pin=False,
    )

    acc = tlx.zeros((256, 128), tl.float32, layout=mma)
    acc = tl.dot(a0, b0, acc)
    tl.debug_barrier()

    lhs0 = tlx.extract_slice(a1, [128, 16], [0, 0])
    lhs1 = tlx.extract_slice(a1, [128, 16], [128, 0])
    rhs0 = tlx.extract_slice(b1, [16, 32], [0, 0])
    rhs1 = tlx.extract_slice(b1, [16, 32], [0, 32])
    rhs2 = tlx.extract_slice(b1, [16, 32], [0, 64])
    rhs3 = tlx.extract_slice(b1, [16, 32], [0, 96])
    c00 = tlx.extract_slice(acc, [128, 32], [0, 0])
    c10 = tlx.extract_slice(acc, [128, 32], [128, 0])
    c01 = tlx.extract_slice(acc, [128, 32], [0, 32])
    c11 = tlx.extract_slice(acc, [128, 32], [128, 32])
    c02 = tlx.extract_slice(acc, [128, 32], [0, 64])
    c12 = tlx.extract_slice(acc, [128, 32], [128, 64])
    c03 = tlx.extract_slice(acc, [128, 32], [0, 96])
    c13 = tlx.extract_slice(acc, [128, 32], [128, 96])

    c00 = tlx.amd_scheduled_mfma(lhs0, rhs0, c00, accumulator_role="persistent")
    c10 = tlx.amd_scheduled_mfma(lhs1, rhs0, c10, accumulator_role="persistent")
    c01 = tlx.amd_scheduled_mfma(lhs0, rhs1, c01, accumulator_role="persistent")
    c11 = tlx.amd_scheduled_mfma(lhs1, rhs1, c11, accumulator_role="persistent")
    c02 = tlx.amd_scheduled_mfma(lhs0, rhs2, c02, accumulator_role="persistent")
    c12 = tlx.amd_scheduled_mfma(lhs1, rhs2, c12, accumulator_role="persistent")
    c03 = tlx.amd_scheduled_mfma(lhs0, rhs3, c03, accumulator_role="persistent")
    c13 = tlx.amd_scheduled_mfma(lhs1, rhs3, c13, accumulator_role="persistent")

    if DIRECT_STORE:
        fragment_rows = tl.arange(0, 128)
        fragment_cols = tl.arange(0, 32)
        out00 = output_ptr + fragment_rows[:, None] * 128 + fragment_cols[None, :]
        out10 = out00 + 128 * 128
        out01 = out00 + 32
        out11 = out10 + 32
        out02 = out00 + 64
        out12 = out10 + 64
        out03 = out00 + 96
        out13 = out10 + 96
        tl.store(tlx.require_layout(out00, mma, pin=False), c00)
        tl.store(tlx.require_layout(out10, mma, pin=False), c10)
        tl.store(tlx.require_layout(out01, mma, pin=False), c01)
        tl.store(tlx.require_layout(out11, mma, pin=False), c11)
        tl.store(tlx.require_layout(out02, mma, pin=False), c02)
        tl.store(tlx.require_layout(out12, mma, pin=False), c12)
        tl.store(tlx.require_layout(out03, mma, pin=False), c03)
        tl.store(tlx.require_layout(out13, mma, pin=False), c13)
    else:
        row0 = tl.cat(
            tl.cat(c00, c01, dim=1),
            tl.cat(c02, c03, dim=1),
            dim=1,
        )
        row1 = tl.cat(
            tl.cat(c10, c11, dim=1),
            tl.cat(c12, c13, dim=1),
            dim=1,
        )
        result = tlx.require_layout(tl.cat(row0, row1, dim=0), mma, pin=False)
        offsets = tlx.require_layout(
            output_ptr + rows[:, None] * 128 + cols[None, :],
            mma,
            pin=False,
        )
        tl.store(offsets, result)


@triton.jit
def _amd_scheduled_mfma_interleaved_chains_kernel(a_ptr, b_ptr, output_ptr, BANDS: tl.constexpr):
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    dot0: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=8)
    dot1: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=8)
    rows = tl.arange(0, 16)
    reduction = tl.arange(0, 256)
    cols = tl.arange(0, 128)
    a = tl.load(a_ptr + rows[:, None] * 256 + reduction[None, :])
    b = tl.load(b_ptr + reduction[:, None] * 128 + cols[None, :])
    a = tlx.require_layout(a, dot0, pin=False)
    b = tlx.require_layout(b, dot1, pin=False)
    b = tlx.amd_register_resident(b, register_class="agpr", registers_per_group=32)
    acc0 = tlx.zeros((16, 64), tl.float32, layout=mma)
    acc1 = tlx.zeros((16, 64), tl.float32, layout=mma)

    for band in tl.static_range(BANDS):
        a_band = tlx.extract_slice(a, [16, 32], [0, band * 32])
        b0 = tlx.extract_slice(b, [32, 64], [band * 32, 0])
        b1 = tlx.extract_slice(b, [32, 64], [band * 32, 64])
        acc0 = tlx.amd_scheduled_mfma(
            a_band,
            b0,
            acc0,
            resident_operand=1,
            accumulator_role="transient",
            initialize=band == 0,
        )
        acc1 = tlx.amd_scheduled_mfma(
            a_band,
            b1,
            acc1,
            resident_operand=1,
            accumulator_role="transient",
            initialize=band == 0,
        )
    acc0, acc1, b1 = tlx.amd_mfma_commit((acc0, acc1), b1)
    half_cols = tl.arange(0, 64)
    output_offsets0 = output_ptr + rows[:, None] * 128 + half_cols[None, :]
    output_offsets1 = output_offsets0 + 64
    output_offsets0 = tlx.require_layout(output_offsets0, mma, pin=False)
    output_offsets1 = tlx.require_layout(output_offsets1, mma, pin=False)
    tl.store(output_offsets0, acc0)
    tl.store(output_offsets1, acc1)


@triton.jit
def _amd_scheduled_mfma_split_resident_chains_kernel(
    a_ptr,
    b_ptr,
    v_ptr,
    output_ptr,
    v_output_ptr,
    USE_LOCAL: tl.constexpr,
    EXACT_LOCAL_LAYOUT: tl.constexpr,
    FULL_COMMIT: tl.constexpr,
):
    """Match dQ's 128+64+32+32 resident-K decomposition."""
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    v_mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    dot0: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=8)
    dot1: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=8)
    v_layout: tl.constexpr = tlx.dot_operand_layout(0, v_mma, k_width=8)
    rows = tl.arange(0, 16)
    reduction = tl.arange(0, 256)
    cols = tl.arange(0, 128)
    a = tlx.require_layout(
        tl.load(a_ptr + rows[:, None] * 256 + reduction[None, :]),
        dot0,
        pin=False,
    )
    b = tl.load(b_ptr + reduction[:, None] * 128 + cols[None, :])
    if USE_LOCAL:
        if EXACT_LOCAL_LAYOUT:
            b_smem_layout: tl.constexpr = (tlx.shared_linear_layout_encoding(
                offset_bases=[
                    [0, 1],
                    [0, 2],
                    [0, 4],
                    [0, 8],
                    [0, 64],
                    [1, 0],
                    [2, 0],
                    [4, 0],
                    [8, 64],
                    [0, 16],
                    [0, 32],
                    [16, 0],
                    [32, 0],
                    [64, 0],
                    [128, 0],
                ],
                block_bases=[],
                alignment=16,
            ))
            b_buffers = tlx.local_alloc(
                (256, 128),
                tl.bfloat16,
                1,
                layout=b_smem_layout,
            )
        else:
            b_buffers = tlx.local_alloc((256, 128), tl.bfloat16, 1)
        b_buffer = tlx.local_view(b_buffers, 0)
        tlx.local_store(b_buffer, b)
        tl.debug_barrier()
        b_lo = tlx.local_load(
            tlx.local_slice(b_buffer, [0, 0], [128, 128]),
            layout=dot1,
            relaxed=True,
        )
        b_mid = tlx.local_load(
            tlx.local_slice(b_buffer, [128, 0], [64, 128]),
            layout=dot1,
            relaxed=True,
        )
        b6 = tlx.local_load(
            tlx.local_slice(b_buffer, [192, 0], [32, 128]),
            layout=dot1,
            relaxed=True,
        )
        b7 = tlx.local_load(
            tlx.local_slice(b_buffer, [224, 0], [32, 128]),
            layout=dot1,
            relaxed=True,
        )
    else:
        b = tlx.require_layout(b, dot1, pin=False)
        b_lo = tlx.extract_slice(b, [128, 128], [0, 0])
        b_mid = tlx.extract_slice(b, [64, 128], [128, 0])
        b6 = tlx.extract_slice(b, [32, 128], [192, 0])
        b7 = tlx.extract_slice(b, [32, 128], [224, 0])

    acc0 = tlx.zeros((16, 64), tl.float32, layout=mma)
    acc1 = tlx.zeros((16, 64), tl.float32, layout=mma)
    for band in tl.static_range(4):
        a_band = tlx.extract_slice(a, [16, 32], [0, band * 32])
        b0 = tlx.extract_slice(b_lo, [32, 64], [band * 32, 0])
        b1 = tlx.extract_slice(b_lo, [32, 64], [band * 32, 64])
        acc0 = tlx.amd_scheduled_mfma(
            a_band,
            b0,
            acc0,
            resident_operand=1,
            accumulator_role="transient",
            initialize=band == 0,
        )
        acc1 = tlx.amd_scheduled_mfma(
            a_band,
            b1,
            acc1,
            resident_operand=1,
            accumulator_role="transient",
            initialize=band == 0,
        )
    for band in tl.static_range(2):
        a_band = tlx.extract_slice(a, [16, 32], [0, (band + 4) * 32])
        b0 = tlx.extract_slice(b_mid, [32, 64], [band * 32, 0])
        b1 = tlx.extract_slice(b_mid, [32, 64], [band * 32, 64])
        acc0 = tlx.amd_scheduled_mfma(
            a_band,
            b0,
            acc0,
            resident_operand=1,
            accumulator_role="transient",
        )
        acc1 = tlx.amd_scheduled_mfma(
            a_band,
            b1,
            acc1,
            resident_operand=1,
            accumulator_role="transient",
        )
    a_band6 = tlx.extract_slice(a, [16, 32], [0, 192])
    b60 = tlx.extract_slice(b6, [32, 64], [0, 0])
    b61 = tlx.extract_slice(b6, [32, 64], [0, 64])
    acc0 = tlx.amd_scheduled_mfma(
        a_band6,
        b60,
        acc0,
        resident_operand=1,
        accumulator_role="transient",
    )
    acc1 = tlx.amd_scheduled_mfma(
        a_band6,
        b61,
        acc1,
        resident_operand=1,
        accumulator_role="transient",
    )
    a_band7 = tlx.extract_slice(a, [16, 32], [0, 224])
    b70 = tlx.extract_slice(b7, [32, 64], [0, 0])
    b71 = tlx.extract_slice(b7, [32, 64], [0, 64])
    acc0 = tlx.amd_scheduled_mfma(
        a_band7,
        b70,
        acc0,
        resident_operand=1,
        accumulator_role="transient",
    )
    acc1 = tlx.amd_scheduled_mfma(
        a_band7,
        b71,
        acc1,
        resident_operand=1,
        accumulator_role="transient",
    )
    if FULL_COMMIT:
        v_resident = tlx.require_layout(
            tl.load(v_ptr + reduction[:, None] * 128 + cols[None, :]),
            v_layout,
            pin=False,
        )
        acc0, acc1, v_resident = tlx.amd_mfma_commit((acc0, acc1), v_resident)
        v_offsets = tlx.require_layout(
            v_output_ptr + reduction[:, None] * 128 + cols[None, :],
            v_layout,
            pin=False,
        )
        tl.store(v_offsets, v_resident)
    else:
        acc0, acc1, b71 = tlx.amd_mfma_commit((acc0, acc1), b71)
    half_cols = tl.arange(0, 64)
    output_offsets0 = tlx.require_layout(
        output_ptr + rows[:, None] * 128 + half_cols[None, :],
        mma,
        pin=False,
    )
    output_offsets1 = tlx.require_layout(
        output_ptr + rows[:, None] * 128 + half_cols[None, :] + 64,
        mma,
        pin=False,
    )
    tl.store(output_offsets0, acc0)
    tl.store(output_offsets1, acc1)


@triton.jit
def _warp_pipe_bmm_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    sab,
    sam,
    sak,
    sbb,
    sbn,
    sbk,
    scb,
    scm,
    scn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
):
    """C[b] = A[b] @ B[b]; B fed [b, N, K] (K-contiguous) + local_trans; 64-bit batch base."""
    bid = tl.program_id(1)
    pid = tl.program_id(0)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    # 64-bit base: batch offset can exceed 2**31 for the production shape.
    a_base = bid.to(tl.int64) * sab + offs_m[:, None].to(tl.int64) * sam
    b_base = bid.to(tl.int64) * sbb + offs_n[:, None].to(tl.int64) * sbn
    K_ITERS = tl.cdiv(K, BLOCK_K)

    smemA = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(A), NUM_BUFFERS)
    smemB = tlx.local_alloc((BLOCK_N, BLOCK_K), tlx.dtype_of(B), NUM_BUFFERS)

    for i in tl.range(0, NUM_BUFFERS, loop_unroll_factor=NUM_BUFFERS):
        ks = i * BLOCK_K
        m = offs_k[None, :] < K - ks  # partial-K mask (folds away when K % BLOCK_K == 0)
        ta = tlx.async_load(A + a_base + (ks + offs_k[None, :]) * sak, tlx.local_view(smemA, i), mask=m, other=0.0)
        tb = tlx.async_load(B + b_base + (ks + offs_k[None, :]) * sbk, tlx.local_view(smemB, i), mask=m, other=0.0)
        tlx.async_load_commit_group([ta, tb])

    tlx.async_load_wait_group(NUM_BUFFERS - 2)
    a_tile = tlx.local_load(tlx.local_view(smemA, 0))
    b_tile = tlx.local_load(tlx.local_trans(tlx.local_view(smemB, 0)))
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for tile_id in tl.range(0, K_ITERS - NUM_BUFFERS):
        pf = (tile_id % NUM_BUFFERS).to(tl.int32)
        nb = ((tile_id + 1) % NUM_BUFFERS).to(tl.int32)
        kpf = (tile_id + NUM_BUFFERS) * BLOCK_K
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)
        with tlx.warp_pipeline_stage("mem", priority=1):
            m = offs_k[None, :] < K - kpf
            ta = tlx.async_load(A + a_base + (kpf + offs_k[None, :]) * sak, tlx.local_view(smemA, pf), mask=m,
                                other=0.0)
            tb = tlx.async_load(B + b_base + (kpf + offs_k[None, :]) * sbk, tlx.local_view(smemB, pf), mask=m,
                                other=0.0)
            tlx.async_load_commit_group([ta, tb])
            a_tile = tlx.local_load(tlx.local_view(smemA, nb))
            b_tile = tlx.local_load(tlx.local_trans(tlx.local_view(smemB, nb)))
        tlx.async_load_wait_group(NUM_BUFFERS - 2)

    acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)
    tlx.async_load_wait_group(0)
    for i in tl.range(0, NUM_BUFFERS - 1, loop_unroll_factor=NUM_BUFFERS - 1):
        buf = ((K_ITERS - (NUM_BUFFERS - 1) + i) % NUM_BUFFERS).to(tl.int32)
        a_tile = tlx.local_load(tlx.local_view(smemA, buf))
        b_tile = tlx.local_load(tlx.local_trans(tlx.local_view(smemB, buf)))
        acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)

    ocm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    ocn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptr = C + bid.to(tl.int64) * scb + scm * ocm[:, None].to(tl.int64) + scn * ocn[None, :]
    tl.store(c_ptr, acc.to(tlx.dtype_of(C)), mask=(ocm[:, None] < M) & (ocn[None, :] < N))


def _run_warp_pipe_bmm(device, bt, M, N, K):
    """Build fp16 operands and launch the warp-pipe bmm (B fed [bt, N, K] for local_trans)."""
    BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS = 128, 64, 64, 2
    a = torch.randn((bt, M, K), device=device, dtype=torch.float16) * 0.1
    b = torch.randn((bt, K, N), device=device, dtype=torch.float16) * 0.1
    bT = b.transpose(1, 2).contiguous()  # [bt, N, K], K-contiguous
    c = torch.empty((bt, M, N), device=device, dtype=torch.float16)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), bt)
    _warp_pipe_bmm_kernel[grid](
        a,
        bT,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        bT.stride(0),
        bT.stride(1),
        bT.stride(2),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        NUM_BUFFERS=NUM_BUFFERS,
        num_warps=8,
        num_stages=1,
        matrix_instr_nonkdim=16,
    )
    return a, b, c


@triton.jit
def _row_stride_async_load_kernel(a_ptr, out_ptr, stride_am, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs = offs_m[:, None] * stride_am + offs_k[None, :]
    smem = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_ptr), 1)
    tok = tlx.async_load(a_ptr + offs, tlx.local_view(smem, 0))  # unmasked -- full tile
    tlx.async_load_commit_group([tok])
    tlx.async_load_wait_group(0)
    t = tlx.local_load(tlx.local_view(smem, 0))
    tl.store(out_ptr + offs_m[:, None] * BLOCK_K + offs_k[None, :], t)


@triton.jit
def _noncontiguous_gather_async_load_kernel(V, out_ptr, stride_b, stride_po, stride_d, stride_x, N: tl.constexpr,
                                            HEAD_DIM: tl.constexpr, PAGE: tl.constexpr):
    n = tl.arange(0, N)
    d = tl.arange(0, HEAD_DIM)
    page = n // PAGE
    token = n % PAGE
    # V is laid out [block, page // 8, head_dim, 8]. Reconstructing the logical
    # [token, head_dim] tile makes the async-load pointer tensor non-contiguous
    # (a gather: sizePerThread=[1,1]).
    ptrs = (V + page[:, None] * stride_b + (token[:, None] // 8) * stride_po + d[None, :] * stride_d +
            (token[:, None] % 8) * stride_x)
    smem = tlx.local_alloc((N, HEAD_DIM), tlx.dtype_of(V), 2)
    tok = tlx.async_load(ptrs, tlx.local_view(smem, 0))
    tlx.async_load_commit_group([tok])
    tlx.async_load_wait_group(0)
    value = tlx.local_load(tlx.local_view(smem, 0))
    tl.store(out_ptr + n[:, None] * HEAD_DIM + d[None, :], value)


@triton.jit
def _local_load_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    buf = tlx.local_alloc((BLOCK_SIZE, ), tl.float32, 1)
    buf0 = tlx.local_view(buf, 0)
    tok = tlx.async_load(x_ptr + offs, buf0, mask=mask)
    tlx.async_load_commit_group([tok])
    tlx.async_load_wait_group(0)

    x = tlx.local_load(buf0)
    tl.store(output_ptr + offs, x, mask=mask)


@triton.jit
def _local_slice_runtime_offset_kernel(x_ptr, output_ptr, row):
    value_layout: tl.constexpr = tlx.layout(
        shape=((8, 32), (2, )),
        stride=((64, 2), (1, )),
    )
    smem_layout: tl.constexpr = tlx.shared_linear_layout_encoding(
        offset_bases=[
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [1, 0],
            [2, 8],
            [4, 16],
        ],
        block_bases=[],
        alignment=8,
    )
    rows = tl.arange(0, 8)
    cols = tl.arange(0, 64)
    offsets = rows[:, None] * 64 + cols[None, :]
    offsets = tlx.require_layout(offsets, value_layout, pin=False)
    values = tl.load(x_ptr + offsets)
    buffers = tlx.local_alloc((8, 64), tl.float32, 1, layout=smem_layout)
    buffer = tlx.local_view(buffers, 0)
    tlx.local_store(buffer, values)
    tl.debug_barrier()
    view = tlx.local_slice(buffer, [row, 0], [1, 64])
    selected = tl.reshape(tlx.local_load(view, relaxed=True), (64, ))
    tl.store(output_ptr + cols, selected)


@triton.jit
def _assume_uniform_ptr_kernel(ptr_array, out_ptr, BLOCK: tl.constexpr):
    # A pointer loaded from memory is not provably uniform, so the backend would
    # otherwise waterfall every buffer access built on it.
    base = tl.load(ptr_array).to(tl.pointer_type(tl.float32))
    base = tlx.assume_uniform(base)
    offs = tl.arange(0, BLOCK).to(tl.int32)
    tlx.buffer_store(tlx.buffer_load(base, offs), out_ptr, offs)


@triton.jit
def _async_load_1d_kernel(
    src_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    OTHER_VAL: tl.constexpr,
    HAS_WRITE_MASK: tl.constexpr,
    INITIALIZE_LOCAL: tl.constexpr,
):
    """Load via async_load (pointer-tensor path) and write result to output."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    buf = tlx.local_alloc((BLOCK_SIZE, ), tl.float32, 1)
    if INITIALIZE_LOCAL:
        val = tl.zeros((BLOCK_SIZE, ), tl.float32)
        tlx.local_store(tlx.local_view(buf, 0), val)
    tok = tlx.async_load(src_ptr + offs, tlx.local_view(buf, 0), mask=mask, other=OTHER_VAL)
    tlx.async_load_commit_group([tok])
    tlx.async_load_wait_group(0)
    val = tlx.local_load(tlx.local_view(buf, 0))
    write_mask = offs < n_elements if HAS_WRITE_MASK else None
    tl.store(out_ptr + offs, val, mask=write_mask)


@triton.jit
def _buffer_load_to_local_1d_kernel(
    src_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    OTHER_VAL: tl.constexpr,
    HAS_WRITE_MASK: tl.constexpr,
    INITIALIZE_LOCAL: tl.constexpr,
):
    """Load via buffer_load_to_local (scalar-ptr + offsets path) and write result to output."""
    pid = tl.program_id(0)
    offs = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int32)
    mask = offs < n_elements
    buf = tlx.local_alloc((BLOCK_SIZE, ), tl.float32, 1)
    if INITIALIZE_LOCAL:
        val = tl.zeros((BLOCK_SIZE, ), tl.float32)
        tlx.local_store(tlx.local_view(buf, 0), val)
    tlx.buffer_load_to_local(tlx.local_view(buf, 0), src_ptr, offs, mask=mask, other=OTHER_VAL)
    tlx.async_load_commit_group()
    tlx.async_load_wait_group(0)
    val = tlx.local_load(tlx.local_view(buf, 0))
    write_mask = offs < n_elements if HAS_WRITE_MASK else None
    tl.store(out_ptr + offs, val, mask=write_mask)


def _run_load_to_local_1d(device, kernel_fn, size, n_valid, other_val, block_size=256, has_write_mask=True,
                          init_local=False):
    """Helper: run a 1D load-to-local kernel and return the output tensor.

    Uses float32 with block_size=256 and num_warps=4 so each thread handles
    exactly one 32-bit element.  This gives per-element mask granularity that
    is compatible with the 32-bit minimum direct-to-LDS width on CDNA4.
    """
    x = torch.randn(size, dtype=torch.float32, device=device)
    out = torch.full((size, ), float("nan"), dtype=torch.float32, device=device)
    grid = (triton.cdiv(size, block_size), )
    kernel_fn[grid](
        x,
        out,
        n_valid,
        BLOCK_SIZE=block_size,
        OTHER_VAL=other_val,
        HAS_WRITE_MASK=has_write_mask,
        INITIALIZE_LOCAL=init_local,
        num_warps=4,
        num_stages=1,
    )
    return x, out


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize(
    "config,match",
    [
        ({"NUM_XCDS": 0}, "NUM_XCDS must be positive"),
        ({"NUM_XCDS": 8, "NUM_SMS": 4}, "NUM_SMS .* must be >= NUM_XCDS"),
        ({"NUM_XCDS": 4, "NUM_SMS": 10}, "NUM_SMS .* must be divisible by NUM_XCDS"),
    ],
    ids=["zero-xcds", "too-few-sms", "nondivisible-sms"],
)
def test_amd_fa_cluster_rejects_invalid_persistent_scheduler(config, match):
    q = torch.empty((1, 1, 8, 64), device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match=match):
        _amd_fa_cluster_persistent_attention(q, q, q, 1.0, False, config=config)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize("N_CTX,BLOCK_M", [(1024, 256), (1025, 256)])
def test_amd_fa_cluster_causal_query_order_gfx950(N_CTX, BLOCK_M):
    num_m_blocks = triton.cdiv(N_CTX, BLOCK_M)
    actual = torch.empty(num_m_blocks, device="cuda", dtype=torch.int32)
    _amd_fa_cluster_causal_query_order_kernel[(num_m_blocks, )](
        actual,
        N_CTX=N_CTX,
        BLOCK_M=BLOCK_M,
        num_warps=1,
    )
    expected = torch.arange(num_m_blocks - 1, -1, -1, device="cuda", dtype=torch.int32)
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize(
    "H,N_CTX,BLOCK_M,IS_CAUSAL,USE_24_HEAD_WINDOW,USE_FACTORED_24,CHECK_INDEX,EXPECTED_MAPPED",
    [
        (64, 4096, 256, False, False, False, 64, 512),
        (64, 4096, 256, True, False, False, 0, 960),
        (64, 16384, 256, True, True, False, 3072, 4080),
        (64, 16384, 256, True, True, True, 3072, 4080),
        (4, 1025, 256, True, False, False, 0, 16),
    ],
)
def test_amd_fa_cluster_workgroup_window_is_bijective_gfx950(
    H,
    N_CTX,
    BLOCK_M,
    IS_CAUSAL,
    USE_24_HEAD_WINDOW,
    USE_FACTORED_24,
    CHECK_INDEX,
    EXPECTED_MAPPED,
):
    num_m_blocks = triton.cdiv(N_CTX, BLOCK_M)
    num_workgroups = H * num_m_blocks
    actual = torch.empty(num_workgroups, device="cuda", dtype=torch.int32)
    _amd_fa_cluster_workgroup_order_kernel[(H, num_m_blocks)](
        actual,
        H=H,
        N_CTX=N_CTX,
        BLOCK_M=BLOCK_M,
        IS_CAUSAL=IS_CAUSAL,
        USE_24_HEAD_WINDOW=USE_24_HEAD_WINDOW,
        USE_FACTORED_24=USE_FACTORED_24,
        num_warps=1,
    )
    expected = torch.arange(num_workgroups, device="cuda", dtype=torch.int32)
    torch.testing.assert_close(actual.sort().values, expected)
    assert actual[CHECK_INDEX].item() == EXPECTED_MAPPED


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_warp_predicate_scalar_warp_local_reduce_gfx950():
    source = torch.arange(256, device="cuda", dtype=torch.float32).reshape(4, 64)
    output = torch.empty_like(source)
    _warp_predicate_warp_local_reduce_kernel[(1, )](source, output, num_warps=4)
    row_sum = source.sum(axis=1)
    active_wave = torch.arange(4, device="cuda") >= 2
    expected = torch.where(active_wave[:, None], source + row_sum[:, None], source)
    torch.testing.assert_close(output, expected)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_warp_predicate_false_lanes_keep_inits_gfx950():
    size = 256
    source = torch.arange(size, device="cuda", dtype=torch.float32)
    lhs = torch.empty_like(source)
    rhs = torch.empty_like(source)
    side = torch.full_like(source, -1.0)
    _warp_predicate_kernel[(1, )](
        source,
        lhs,
        rhs,
        side,
        size=size,
        num_warps=4,
    )

    offsets = torch.arange(size, device="cuda")
    predicate = (offsets >= 64) & (offsets < 128) & (offsets % 5 < 2)
    torch.testing.assert_close(lhs, torch.where(predicate, source + 3.0, source))
    torch.testing.assert_close(rhs, torch.where(predicate, source * 2.0 - 3.0, source * 2.0))
    torch.testing.assert_close(side, torch.where(predicate, source, torch.full_like(source, -1.0)))


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_amd_warp_votes_correct_gfx950():
    values = torch.ones(64, device="cuda", dtype=torch.int32)
    all_output = torch.empty_like(values)
    any_output = torch.empty_like(values)

    _warp_vote_kernel[(1, )](values, all_output, any_output, BLOCK=64, num_warps=1)
    torch.testing.assert_close(all_output, torch.ones_like(values))
    torch.testing.assert_close(any_output, torch.ones_like(values))

    values[17] = 0
    _warp_vote_kernel[(1, )](values, all_output, any_output, BLOCK=64, num_warps=1)
    torch.testing.assert_close(all_output, torch.zeros_like(values))
    torch.testing.assert_close(any_output, torch.ones_like(values))

    values.zero_()
    _warp_vote_kernel[(1, )](values, all_output, any_output, BLOCK=64, num_warps=1)
    torch.testing.assert_close(all_output, torch.zeros_like(values))
    torch.testing.assert_close(any_output, torch.zeros_like(values))


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize("condition", [False, True])
def test_concrete_helper_release_preserves_values_gfx950(condition):
    torch.manual_seed(0)
    a = torch.randn((16, 32), device="cuda", dtype=torch.float16)
    b = torch.randn((32, 64), device="cuda", dtype=torch.float16)
    output = torch.empty(2048, device="cuda", dtype=torch.float32)
    _concrete_helper_release_kernel[(1, )](
        a,
        b,
        output,
        condition,
        num_warps=4,
        matrix_instr_nonkdim=16,
    )

    concrete = output[:1024].reshape(16, 64)
    released = output[1024:].reshape(16, 64)
    assert torch.isfinite(concrete).all()
    expected = a.float() @ b.float()
    expected = expected * 2.0 if condition else expected + 1.0
    expected[:8] *= 0.5
    torch.testing.assert_close(concrete, expected, atol=5e-2, rtol=1e-2)
    # release_layout changes only the register distribution, so its generic
    # store must preserve the concrete store's logical tensor exactly.
    torch.testing.assert_close(released, concrete, atol=0, rtol=0)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_pinned_buffer_load_layout_correctness_gfx950(device):
    from triton.language.extra.tlx.tutorials.amd_fa_bwd import (
        _attn_bwd_dq_native_convert_kernel, )

    m = torch.arange(128, device=device, dtype=torch.int64)[:, None]
    d = torch.arange(128, device=device, dtype=torch.int64)[None, :]
    expected = ((m * 131 + d * 7) % 2048).to(torch.bfloat16)
    local_m = m & 15
    tile_m = m - local_m
    d_swizzled = ((d & 1) | ((d & 2) << 6) | ((d & 12) << 3) | ((d & 48) << 5) | ((d & 64) << 2))
    physical = tile_m * 128 + (local_m << 1) + d_swizzled
    native = torch.empty(128 * 128, device=device, dtype=torch.bfloat16)
    native[physical.flatten()] = expected.flatten()
    actual = torch.empty_like(expected)

    _attn_bwd_dq_native_convert_kernel[(1, 1)](
        native,
        actual,
        N=128,
        D=128,
        BLOCK_M=128,
        num_warps=4,
        matrix_instr_nonkdim=16,
    )

    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_async_load_correctness(device):
    """async_load produces correct results on gfx950 hardware."""
    size = 256
    x = torch.rand(size, dtype=torch.float32, device=device)
    y = torch.rand(size, dtype=torch.float32, device=device)
    output = torch.empty_like(x)
    grid = (triton.cdiv(size, 64), )
    _async_load_kernel[grid](x, y, output, size, BLOCK_SIZE=64)
    torch.testing.assert_close(x + y, output)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_extract_slice_dot1_correct_gfx950():
    x = torch.arange(256 * 128, device="cuda", dtype=torch.float32).reshape(256, 128).to(torch.bfloat16)
    actual = torch.empty((32, 64), device="cuda", dtype=torch.bfloat16)
    for row in range(0, 256, 32):
        for col in (0, 64):
            _extract_slice_dot1_kernel[(1, )](
                x,
                actual,
                ROW_OFFSET=row,
                COL_OFFSET=col,
                num_warps=4,
                matrix_instr_nonkdim=16,
            )
            torch.testing.assert_close(actual, x[row:row + 32, col:col + 64])


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_extract_slice_mfma_correct_gfx950():
    torch.manual_seed(0)
    a = torch.randn((16, 256), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((256, 64), device="cuda", dtype=torch.bfloat16)
    actual = torch.empty((16, 64), device="cuda", dtype=torch.float32)
    for band in range(8):
        _extract_slice_mfma_kernel[(1, )](
            a,
            b,
            actual,
            BAND=band,
            num_warps=4,
            matrix_instr_nonkdim=16,
        )
        expected = (a[:, band * 32:(band + 1) * 32].float() @ b[band * 32:(band + 1) * 32].float())
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize("register_class", ["vgpr", "agpr"])
def test_amd_register_class_anchor_correct_gfx950(register_class):
    x = torch.arange(2048, device="cuda", dtype=torch.float32)
    actual = torch.empty_like(x)
    _amd_register_class_anchor_kernel[(1, )](x, actual, register_class, num_warps=4)
    torch.testing.assert_close(actual, x)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize("k_width", [8, 4])
def test_amd_scheduled_mfma_correct_gfx950(k_width):
    torch.manual_seed(0)
    a = torch.randn((16, 32), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((32, 64), device="cuda", dtype=torch.bfloat16)
    actual = torch.empty((16, 64), device="cuda", dtype=torch.float32)
    _amd_scheduled_mfma_kernel[(1, )](
        a,
        b,
        actual,
        K_WIDTH=k_width,
        num_warps=4,
        matrix_instr_nonkdim=16,
    )
    torch.testing.assert_close(actual, a.float() @ b.float(), atol=2e-4, rtol=2e-4)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_amd_scheduled_mfma_chain_correct_gfx950():
    torch.manual_seed(0)
    a = torch.randn((16, 256), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((256, 64), device="cuda", dtype=torch.bfloat16)
    actual = torch.empty((16, 64), device="cuda", dtype=torch.float32)
    _amd_scheduled_mfma_chain_kernel[(1, )](
        a,
        b,
        actual,
        BANDS=8,
        num_warps=4,
        matrix_instr_nonkdim=16,
    )
    expected = a.float() @ b.float()
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_amd_scheduled_mfma_persistent_acc_correct_gfx950():
    torch.manual_seed(0)
    a = torch.randn((16, 64), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((64, 64), device="cuda", dtype=torch.bfloat16)
    actual = torch.empty((16, 64), device="cuda", dtype=torch.float32)
    expected = a.float() @ b.float()
    for use_vgpr in (False, True):
        _amd_scheduled_mfma_persistent_acc_kernel[(1, )](
            a,
            b,
            actual,
            USE_VGPR=use_vgpr,
            num_warps=4,
            matrix_instr_nonkdim=16,
        )
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_amd_scheduled_mfma_32x32_correct_gfx950():
    torch.manual_seed(0)
    a = torch.randn((128, 16), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((16, 32), device="cuda", dtype=torch.bfloat16)
    actual = torch.empty((128, 32), device="cuda", dtype=torch.float32)
    _amd_scheduled_mfma_32x32_kernel[(1, )](
        a,
        b,
        actual,
        num_warps=4,
        matrix_instr_nonkdim=16,
    )
    torch.testing.assert_close(actual, a.float() @ b.float(), atol=2e-4, rtol=2e-4)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_amd_scheduled_mfma_fragmented_nd_correct_gfx950():
    torch.manual_seed(0)
    a = torch.randn((256, 16), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((16, 128), device="cuda", dtype=torch.bfloat16)
    actual = torch.empty((256, 128), device="cuda", dtype=torch.float32)
    _amd_scheduled_mfma_fragmented_nd_kernel[(1, )](
        a,
        b,
        actual,
        num_warps=4,
        matrix_instr_nonkdim=16,
    )
    expected = a.float() @ b.float()
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize("direct_store", [True, False])
def test_amd_scheduled_mfma_fragmented_nd_update_correct_gfx950(direct_store, ):
    torch.manual_seed(0)
    a0 = torch.randn((256, 16), device="cuda", dtype=torch.bfloat16)
    b0 = torch.randn((16, 128), device="cuda", dtype=torch.bfloat16)
    a1 = torch.randn((256, 16), device="cuda", dtype=torch.bfloat16)
    b1 = torch.randn((16, 128), device="cuda", dtype=torch.bfloat16)
    actual = torch.empty((256, 128), device="cuda", dtype=torch.float32)
    _amd_scheduled_mfma_fragmented_nd_update_kernel[(1, )](
        a0,
        b0,
        a1,
        b1,
        actual,
        DIRECT_STORE=direct_store,
        num_warps=4,
        matrix_instr_nonkdim=16,
    )
    expected = a0.float() @ b0.float() + a1.float() @ b1.float()
    torch.testing.assert_close(actual, expected, atol=4e-4, rtol=4e-4)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_amd_scheduled_mfma_interleaved_chains_correct_gfx950():
    torch.manual_seed(0)
    a = torch.randn((16, 256), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((256, 128), device="cuda", dtype=torch.bfloat16)
    actual = torch.empty((16, 128), device="cuda", dtype=torch.float32)
    _amd_scheduled_mfma_interleaved_chains_kernel[(1, )](
        a,
        b,
        actual,
        BANDS=8,
        num_warps=4,
        matrix_instr_nonkdim=16,
    )
    expected = a.float() @ b.float()
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize(
    "exact_local_layout,full_commit",
    [(False, False), (True, False), (True, True)],
)
def test_amd_scheduled_mfma_split_resident_chains_correct_gfx950(
    exact_local_layout,
    full_commit,
):
    torch.manual_seed(0)
    a = torch.randn((16, 256), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((256, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn((256, 128), device="cuda", dtype=torch.bfloat16)
    actual = torch.empty((16, 128), device="cuda", dtype=torch.float32)
    v_actual = torch.empty_like(v)
    _amd_scheduled_mfma_split_resident_chains_kernel[(1, )](
        a,
        b,
        v,
        actual,
        v_actual,
        USE_LOCAL=True,
        EXACT_LOCAL_LAYOUT=exact_local_layout,
        FULL_COMMIT=full_commit,
        num_warps=4,
        matrix_instr_nonkdim=16,
    )
    expected = a.float() @ b.float()
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
    if full_commit:
        torch.testing.assert_close(v_actual, v)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_warp_pipe_bmm_aligned_k_gfx950(device):
    """Warp-pipe bmm with K a multiple of BLOCK_K compiles + runs correctly (positive control)."""
    a, b, c = _run_warp_pipe_bmm(device, bt=8, M=256, N=256, K=2560)  # 2560 % 64 == 0
    torch.testing.assert_close(c.float(), torch.bmm(a.float(), b.float()), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_warp_pipe_bmm_partial_k_gfx950(device):
    """Warp-pipe bmm with a partial-K tail (K not a multiple of BLOCK_K).

    Same kernel and config as the aligned-K positive control; only K differs (prime 2309, the
    production compression-bmm K). The partial-K mask makes the async_load un-lowerable as a
    direct-to-LDS copy on CDNA4 (vec=1 -> 16-bit); CoalesceAsyncCopy now falls back to a
    synchronous tt.load + ttg.local_store so it compiles and runs correctly.
    Previously this aborted make_llir with an unrealized_conversion_cast.
    """
    a, b, c = _run_warp_pipe_bmm(device, bt=8, M=256, N=256, K=2309)  # 2309 % 64 == 5
    torch.testing.assert_close(c.float(), torch.bmm(a.float(), b.float()), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize("K", [2320, 2309, 2312, 1956])
def test_async_load_row_stride_gfx950(device, K):
    """Unmasked full-tile async_load with a non-16-aligned global row stride (T280910119).

    A row stride not a multiple of 16 elements collapses the direct-to-LDS vector width
    below a supported bitwidth (fp16 -> 16-bit) on CDNA4, so the copy cannot be lowered as
    a direct-to-LDS load (its swizzled dst hits loadContig == 0). CoalesceAsyncCopy now
    falls back to a synchronous tt.load + ttg.local_store for both swizzled and padded
    dsts, so it compiles and runs correctly. Previously K % 16 != 0 aborted make_llir with
    an unrealized_conversion_cast. K=2320 (% 16 == 0) is the positive control and keeps the
    fast direct-to-LDS path.
    """
    BLOCK_M, BLOCK_K = 128, 64
    a = torch.randn((BLOCK_M, K), device=device, dtype=torch.float16)
    out = torch.empty((BLOCK_M, BLOCK_K), device=device, dtype=torch.float16)
    _row_stride_async_load_kernel[(1, )](a, out, a.stride(0), BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K)
    torch.testing.assert_close(out, a[:, :BLOCK_K])


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_async_load_noncontiguous_gather_gfx950(device):
    """Non-contiguous gather-pointer async_load in bf16 (P2440272260).

    A third way (besides a partial-K mask or a non-16-aligned row stride) to collapse the
    direct-to-LDS vector width to 16-bit on CDNA4: a genuinely non-contiguous pointer tensor.
    The gather offsets force the async src blocked layout to sizePerThread=[1,1] (vec=1), so
    bf16 -> 16-bit, canLoadDirectToLDS() rejects it (loadContig == 0), and CoalesceAsyncCopy
    falls back to a synchronous tt.load + ttg.local_store. Previously this aborted make_llir
    with an unrealized_conversion_cast. Uses bfloat16 -- the other 16-bit dtype; the mask and
    row-stride tests cover fp16.
    """
    N, HEAD_DIM, PAGE = 128, 64, 64
    v = torch.randn((2, 8, HEAD_DIM, 8), device=device, dtype=torch.bfloat16)
    out = torch.empty((N, HEAD_DIM), device=device, dtype=torch.bfloat16)
    _noncontiguous_gather_async_load_kernel[(1, )](v, out, *v.stride(), N=N, HEAD_DIM=HEAD_DIM, PAGE=PAGE, num_warps=4)
    n = torch.arange(N, device=device)
    page = n // PAGE
    token = n % PAGE
    ref = v[page, token // 8, :, token % 8]
    torch.testing.assert_close(out, ref)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize("row", [0, 3, 7])
def test_local_slice_runtime_offset_correct_gfx950(row):
    x = torch.arange(8 * 64, device="cuda", dtype=torch.float32).reshape(8, 64)
    actual = torch.empty(64, device="cuda", dtype=torch.float32)
    _local_slice_runtime_offset_kernel[(1, )](x, actual, row, num_warps=4)
    torch.testing.assert_close(actual, x[row], atol=0.0, rtol=0.0)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_local_load_correctness(device):
    """local_load after async_wait produces correct results on gfx950 hardware."""
    size = 256
    x = torch.rand(size, dtype=torch.float32, device=device)
    output = torch.empty_like(x)
    grid = (triton.cdiv(size, 64), )
    _local_load_kernel[grid](x, output, size, BLOCK_SIZE=64)
    torch.testing.assert_close(x, output)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_a4w4_shape_stride_layouts_correctness_gfx950(device):
    m = n = 256
    for k in (1024, 1536):
        a, b, a_scales, b_scales = _generate_a4w4_inputs(m, n, k)
        actual = _launch_a4w4(a, b, a_scales, b_scales)
        expected = _a4w4_reference(a, b, a_scales, b_scales)
        torch.testing.assert_close(actual, expected, atol=0.1, rtol=0.0)


@pytest.mark.parametrize("k, split_k", [(1024, 1), (4096, 2)])
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_a4w4_inter_wave_preshuffled_scale_correctness_gfx950(device, k, split_k):
    m = n = 256
    a, b, a_scales, b_scales = _generate_a4w4_inputs(m, n, k)
    a_scales_preshuffled = _preshuffle_a4w4_a_scales(a_scales)
    b_scales_preshuffled = _preshuffle_a4w4_b_scales(b_scales)
    actual = _a4w4_inter_wave_matmul_preshuffled(a, b, a_scales_preshuffled, b_scales_preshuffled, SPLIT_K=split_k)
    expected = _a4w4_reference(a, b, a_scales, b_scales)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("k, split_k", [(1024, 1), (4096, 2)])
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_a4w4_inter_wave_merged_scale_correctness_gfx950(device, k, split_k):
    m = n = 256
    a, b, a_scales, b_scales = _generate_a4w4_inputs(m, n, k)
    scales = _preshuffle_a4w4_scales(a_scales, b_scales)
    actual = _a4w4_inter_wave_matmul_merged_scales(a, b, scales, SPLIT_K=split_k)
    expected = _a4w4_reference(a, b, a_scales, b_scales)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "k, expected_path",
    [(1536, "intra_wave_256x256"), (2048, "inter_wave_256x256")],
)
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_a4w4_inter_wave_large_grid_dispatch_correctness_gfx950(device, k, expected_path):
    # A 2x33 grid exceeds the skinny threshold. K=1536 selects the measured
    # lower-overhead 4-wave path; K=2048 selects the 8-wave pipeline.
    m, n = 512, 8448
    assert _select_a4w4_inter_wave_path(m, n, k) == expected_path
    a, b, a_scales, b_scales = _generate_a4w4_inputs(m, n, k)
    actual = _a4w4_inter_wave_matmul(a, b, a_scales, b_scales)
    expected = _a4w4_reference(a, b, a_scales, b_scales)
    torch.testing.assert_close(actual, expected, atol=0.1, rtol=0.0)


@pytest.mark.parametrize("m, n", [(256, 16640), (512, 8448)])
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_a4w4_short_k_dispatch_stress_gfx950(device, m, n):
    # Both public shapes exceed the skinny threshold by the smallest possible
    # grid margins. K=1024 must dispatch to the measured 4-wave path.
    k = _A4W4_INTER_WAVE_MIN_K
    grid_mn = triton.cdiv(m, _A4W4_INTER_WAVE_BLOCK_M) * triton.cdiv(n, _A4W4_INTER_WAVE_BLOCK_N)
    assert grid_mn in (65, 66)
    assert _select_a4w4_inter_wave_path(m, n, k) == "intra_wave_256x256"
    assert k == 1024

    a, b, a_scales, b_scales = _generate_a4w4_inputs(m, n, k)
    expected = _a4w4_reference(a, b, a_scales, b_scales)
    for launch in range(500):
        actual = _a4w4_inter_wave_matmul(a, b, a_scales, b_scales)
        torch.testing.assert_close(actual, expected, atol=0.1, rtol=0.0, msg=f"failed on launch {launch}")


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_a4w4_inter_wave_skinny_correctness_gfx950(device):
    # 512x256x1536 -> 256-tile grid = 2*1 = 2 <= NUM_CU/32, so the dispatcher takes
    # the occupancy-starved 128x128 + split-K TLX path (and its fp32 reduce).
    m = 512
    n = 256
    k = 1536
    a, b, a_scales, b_scales = _generate_a4w4_inputs(m, n, k)
    actual = _a4w4_inter_wave_matmul(a, b, a_scales, b_scales)
    expected = _a4w4_reference(a, b, a_scales, b_scales)
    torch.testing.assert_close(actual, expected, atol=0.1, rtol=0.0)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_assume_uniform_correctness_gfx950(device):
    """assume_uniform returns its argument unchanged."""
    size = 64
    x = torch.rand(size, dtype=torch.float32, device=device)
    out = torch.empty_like(x)
    ptr_array = torch.tensor([x.data_ptr()], dtype=torch.int64, device=device)
    _assume_uniform_ptr_kernel[(1, )](ptr_array, out, BLOCK=size)
    torch.testing.assert_close(out, x)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_buffer_load_to_local_no_mask(device, monkeypatch):
    # set env var AMDGCN_USE_BUFFER_OPS=0 to ensure async_load to be lowered to global_load
    # so can compare behaviors of global_load and buffer_load
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", 0)
    """buffer_load_to_local without mask matches async_load without mask."""
    size = 256
    torch.manual_seed(42)
    x_a, out_a = _run_load_to_local_1d(device, _async_load_1d_kernel, size, size, None)
    torch.manual_seed(42)
    x_b, out_b = _run_load_to_local_1d(device, _buffer_load_to_local_1d_kernel, size, size, None)
    torch.testing.assert_close(out_a, out_b)
    torch.testing.assert_close(out_a, x_a)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_buffer_load_to_local_masked_other_none(device, monkeypatch):
    # set env var AMDGCN_USE_BUFFER_OPS=0 to ensure async_load to be lowered to global_load
    # so can compare behaviors of global_load and buffer_load
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", 0)
    """buffer_load_to_local with mask and other=None zero-fills masked elements, matching async_load."""
    size = 256
    n_valid = 128
    torch.manual_seed(42)
    x_a, out_a = _run_load_to_local_1d(device, _async_load_1d_kernel, size, n_valid, None, has_write_mask=False,
                                       init_local=True)
    torch.manual_seed(42)
    x_b, out_b = _run_load_to_local_1d(device, _buffer_load_to_local_1d_kernel, size, n_valid, None,
                                       has_write_mask=False, init_local=True)
    # Valid region must match the source data.
    torch.testing.assert_close(out_a[:n_valid], x_a[:n_valid])
    torch.testing.assert_close(out_b[:n_valid], x_b[:n_valid])
    # Masked region must be nan in both paths.
    assert torch.all(out_a[n_valid:] == 0)
    assert torch.all(out_b[n_valid:] == 0)
    # Overall outputs must be identical.
    torch.testing.assert_close(out_a, out_b, equal_nan=True)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_buffer_load_to_local_masked_other_zero(device, monkeypatch):
    # set env var AMDGCN_USE_BUFFER_OPS=0 to ensure async_load to be lowered to global_load
    # so can compare behaviors of global_load and buffer_load
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", 0)
    """buffer_load_to_local with mask and other=0.0 zero-fills masked elements, matching async_load."""
    size = 256
    n_valid = 128
    torch.manual_seed(42)
    x_a, out_a = _run_load_to_local_1d(device, _async_load_1d_kernel, size, n_valid, 0.0, has_write_mask=False)
    torch.manual_seed(42)
    x_b, out_b = _run_load_to_local_1d(device, _buffer_load_to_local_1d_kernel, size, n_valid, 0.0,
                                       has_write_mask=False)
    torch.testing.assert_close(out_a[:n_valid], x_a[:n_valid])
    torch.testing.assert_close(out_b[:n_valid], x_b[:n_valid])
    torch.testing.assert_close(out_b[n_valid:], torch.zeros(size - n_valid, dtype=torch.float32, device=device))
    torch.testing.assert_close(out_a[n_valid:], torch.zeros(size - n_valid, dtype=torch.float32, device=device))
    torch.testing.assert_close(out_a, out_b)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize("n_valid", [128, 200], ids=["half", "near_full"])
def test_buffer_load_to_local_masked_other_none_boundary(device, n_valid, monkeypatch):
    # set env var AMDGCN_USE_BUFFER_OPS=0 to ensure async_load to be lowered to global_load
    # so can compare behaviors of global_load and buffer_load
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", 0)
    """buffer_load_to_local with other=None at various mask boundaries."""
    size = 256
    torch.manual_seed(42)
    x_a, out_a = _run_load_to_local_1d(device, _async_load_1d_kernel, size, n_valid, None)
    torch.manual_seed(42)
    x_b, out_b = _run_load_to_local_1d(device, _buffer_load_to_local_1d_kernel, size, n_valid, None)
    torch.testing.assert_close(out_a[:n_valid], x_a[:n_valid])
    torch.testing.assert_close(out_b[:n_valid], x_b[:n_valid])
    torch.testing.assert_close(out_a, out_b, equal_nan=True)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_buffer_load_to_local_multi_cta_masked_other_none(device, monkeypatch):
    # set env var AMDGCN_USE_BUFFER_OPS=0 to ensure async_load to be lowered to global_load
    # so can compare behaviors of global_load and buffer_load
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", 0)
    """buffer_load_to_local with mask, other=None, and multiple CTAs (partial last tile)."""
    # Two CTAs: first full (256 elements), second partial (128 valid out of 256).
    size = 512
    n_valid = 384  # First CTA fully valid, second CTA half masked.
    block_size = 256
    torch.manual_seed(42)
    x_a, out_a = _run_load_to_local_1d(device, _async_load_1d_kernel, size, n_valid, None, block_size=block_size)
    torch.manual_seed(42)
    x_b, out_b = _run_load_to_local_1d(device, _buffer_load_to_local_1d_kernel, size, n_valid, None,
                                       block_size=block_size)
    torch.testing.assert_close(out_a[:n_valid], x_a[:n_valid])
    torch.testing.assert_close(out_b[:n_valid], x_b[:n_valid])
    torch.testing.assert_close(out_a, out_b, equal_nan=True)


@triton.jit
def _workgroup_barrier_sum_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    # One workgroup, BLOCK lanes across several warps. Each lane publishes its input
    # into its own LDS slot, then every lane sums the WHOLE buffer -- so each lane
    # must observe every other (cross-warp) lane's store. workgroup_barrier provides
    # the LDS fence + rendezvous; without it the sum races and drops cross-warp
    # stores.
    off = tl.arange(0, BLOCK)
    smem = tlx.local_alloc((BLOCK, ), tl.int32, 1)
    buf = tlx.local_view(smem, 0)
    tlx.local_store(buf, tl.load(x_ptr + off))
    tlx.workgroup_barrier()
    total = tl.sum(tlx.local_load(buf))
    tl.store(out_ptr + off, total + tl.zeros((BLOCK, ), tl.int32))


@triton.jit
def _cond_barrier_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    # Exercise the cond_barrier phase-shift bracket (cond_barrier(wg != 0) ...
    # cond_barrier(wg == 0), the split-M ping-pong pattern) around a
    # workgroup_barrier. The payload is a per-lane copy -- independent of the
    # (deliberately out-of-phase) barriers -- so the assertion checks what a unit
    # test can: the paired bracket compiles and RECONVERGES without deadlock (a
    # broken barrier-count contract would hang or drop the stores). cond_barrier's
    # cross-warp phase-shift correctness is covered by the grouped-gemm integration.
    off = tl.arange(0, BLOCK)
    wg = tlx.thread_id(0) // 256
    tlx.cond_barrier(wg != 0)
    tlx.workgroup_barrier()
    tlx.cond_barrier(wg == 0)
    tl.store(out_ptr + off, tl.load(x_ptr + off))


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_amd_workgroup_barrier(device):
    BLOCK = 256  # 4 warps x 64 lanes -> exercises cross-warp LDS visibility
    x = torch.randint(-(2**20), 2**20, (BLOCK, ), device=device, dtype=torch.int32)
    out = torch.empty_like(x)
    compiled = _workgroup_barrier_sum_kernel[(1, )](x, out, BLOCK=BLOCK, num_warps=4)
    torch.cuda.synchronize()
    expected = torch.full((BLOCK, ), int(x.sum().item()), device=device, dtype=torch.int32)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    # Lowers to a fenced ttg.barrier bracketed by rocdl.sched.barrier fences.
    ttgir = compiled.asm["ttgir"]
    assert ttgir.count("ttg.barrier") >= 1, f"TTGIR {ttgir}"
    assert ttgir.count("rocdl.sched.barrier") >= 2, f"TTGIR {ttgir}"


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_amd_cond_barrier(device):
    BLOCK = 512  # 8 warps -> two 256-lane warp-groups for the cond_barrier phase shift
    x = torch.randint(-(2**20), 2**20, (BLOCK, ), device=device, dtype=torch.int32)
    out = torch.empty_like(x)
    compiled = _cond_barrier_kernel[(1, )](x, out, BLOCK=BLOCK, num_warps=8)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, x, atol=0, rtol=0)
    # The paired bracket lowers to two amdg.cond_barrier ops.
    ttgir = compiled.asm["ttgir"]
    assert ttgir.count("amdg.cond_barrier") == 2, f"TTGIR {ttgir}"


@triton.jit
def _test_get_fp8_format_name_kernel(
    output_ptr,
    DTYPE: tl.constexpr,
    EXPECTED: tl.constexpr,
):
    result: tl.constexpr = tlx.get_fp8_format_name(DTYPE)
    if result == EXPECTED:
        tl.store(output_ptr, 1)
    else:
        tl.store(output_ptr, 0)


@triton.jit
def _test_get_fp8_format_name_unsupported_kernel(
    output_ptr,
    DTYPE: tl.constexpr,
):
    result: tl.constexpr = tlx.get_fp8_format_name(DTYPE)
    tl.store(output_ptr, result == "e5m2")


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_thread_id_gfx950(device):

    @triton.jit
    def store_from_thread_0_kernel(
        output_ptr,
        value,
        n_elements,
        axis: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        tid = tlx.thread_id(axis)
        if tid == 0:
            tl.store(output_ptr + offsets, value, mask=mask)

    output = torch.zeros(32, dtype=torch.int32, device=device)
    n_elements = output.numel()
    value = 42
    store_from_thread_0_kernel[(1, )](output, value, n_elements, 0, 32, num_warps=1)
    torch.cuda.synchronize()
    expected_output = torch.zeros(32, dtype=torch.int32, device=device)
    expected_output[0] = value
    torch.testing.assert_close(output, expected_output)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_loop_carry_var_check_gfx950(device):

    @triton.jit
    def loop_carry_shadow():
        x = tlx.local_alloc((16, 16), tl.int16, tl.constexpr(2))
        y = x
        for _ in range(0, 128):
            zeros = tl.zeros((16, 16), dtype=tl.int16)
            # shadow x with different type
            x = tlx.local_view(y, 0)
            tlx.local_store(x, zeros)

    grid = lambda meta: (1, 1)

    with pytest.raises(triton.CompilationError) as e:
        loop_carry_shadow[grid]()
    list_msg = traceback.format_exception(e.type, e.value, e.tb, chain=True)
    assert "Please make sure that the type stays consistent" in "\n".join(list_msg)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_size_of_gfx950(device):

    @triton.jit
    def size_of_kernel(output_ptr):
        # Test size_of for various dtypes
        size_fp32 = tlx.size_of(tl.float32)
        size_fp16 = tlx.size_of(tl.float16)
        size_int32 = tlx.size_of(tl.int32)
        size_int8 = tlx.size_of(tl.int8)
        size_int64 = tlx.size_of(tl.int64)

        # Store results
        tl.store(output_ptr + 0, size_fp32)
        tl.store(output_ptr + 1, size_fp16)
        tl.store(output_ptr + 2, size_int32)
        tl.store(output_ptr + 3, size_int8)
        tl.store(output_ptr + 4, size_int64)

    # Expected sizes in bytes
    expected_sizes = torch.tensor([4, 2, 4, 1, 8], dtype=torch.int32, device=device)
    output = torch.zeros(5, dtype=torch.int32, device=device)

    grid = lambda meta: (1, )
    size_of_kernel[grid](output)

    torch.testing.assert_close(output, expected_sizes)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_size_of_constexpr_gfx950(device):

    @triton.jit
    def size_of_constexpr_kernel(output_ptr, DTYPE: tl.constexpr):
        # Test size_of with constexpr dtype argument
        size = tlx.size_of(DTYPE)
        tl.store(output_ptr, size)

    output = torch.zeros(1, dtype=torch.int32, device=device)

    # Test with float32 (4 bytes)
    grid = lambda meta: (1, )
    size_of_constexpr_kernel[grid](output, tl.float32)
    assert output.item() == 4, f"Expected 4 for float32, got {output.item()}"

    # Test with float16 (2 bytes)
    size_of_constexpr_kernel[grid](output, tl.float16)
    assert output.item() == 2, f"Expected 2 for float16, got {output.item()}"

    # Test with int8 (1 byte)
    size_of_constexpr_kernel[grid](output, tl.int8)
    assert output.item() == 1, f"Expected 1 for int8, got {output.item()}"

    # Test with int64 (8 bytes)
    size_of_constexpr_kernel[grid](output, tl.int64)
    assert output.item() == 8, f"Expected 8 for int64, got {output.item()}"


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize(
    "dtype,expected",
    [
        (tl.float8e5, "e5m2"),
        (tl.float8e4nv, "e4m3"),
    ],
)
def test_get_fp8_format_name_gfx950(dtype, expected, device):
    """Test that FP8 dtypes return correct format strings."""
    output = torch.zeros(1, dtype=torch.int32, device=device)
    _test_get_fp8_format_name_kernel[(1, )](output, DTYPE=dtype, EXPECTED=expected)
    assert output.item() == 1


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize(
    "dtype",
    [
        tl.float32,
        tl.float16,
        tl.int32,
    ],
)
def test_get_fp8_format_name_unsupported_dtype_raises_error_gfx950(dtype, device):
    """Test that non-FP8 dtypes raise a CompilationError during compilation."""
    output = torch.zeros(1, dtype=torch.int32, device=device)
    with pytest.raises(triton.CompilationError) as exc_info:
        _test_get_fp8_format_name_unsupported_kernel[(1, )](output, DTYPE=dtype)
    # Check that the underlying cause mentions the supported types
    assert "only supports tl.float8e5" in str(exc_info.value.__cause__)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_clock64_gfx950(device):

    @triton.jit
    def clock64_from_thread_0_kernel(
        output_ptr,
        elapsed_ptr,
        value,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        tid = tlx.thread_id(0)
        if pid == 0 and tid == 0:
            start = tlx.clock64()
            tl.store(output_ptr + offsets, value, mask=mask)
            end = tlx.clock64()
            tl.store(elapsed_ptr, end - start)

    output = torch.zeros(32, dtype=torch.int32, device=device)
    elapsed = torch.zeros(1, dtype=torch.int64, device=device)
    n_elements = output.numel()
    value = 42
    kernel = clock64_from_thread_0_kernel[(1, )](output, elapsed, value, n_elements, 32, num_warps=1)
    assert kernel.asm["ttgir"].count("ttg.clock64") == 2
    assert kernel.asm["amdgcn"].count("s_memtime") == 2
    assert elapsed.item() > 0


_PARTIAL_MASK_BLOCK_M = 128
_PARTIAL_MASK_BLOCK_K = 64


@triton.jit
def _partial_mask_async_load_kernel(
    a_ptr,
    out_ptr,
    VALID_K: tl.constexpr,
    USE_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs = offs_m[:, None] * BLOCK_K + offs_k[None, :]
    smem = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_ptr), 1)
    if USE_MASK:
        # Ensure masked lanes must be overwritten rather than relying on
        # freshly allocated shared memory happening to contain zero.
        poison = tl.full((BLOCK_M, BLOCK_K), 7.0, tlx.dtype_of(a_ptr))
        tlx.local_store(tlx.local_view(smem, 0), poison)
        tok = tlx.async_load(a_ptr + offs, tlx.local_view(smem, 0), mask=offs_k[None, :] < VALID_K)
    else:
        tok = tlx.async_load(a_ptr + offs, tlx.local_view(smem, 0))
    tlx.async_load_commit_group([tok])
    tlx.async_load_wait_group(0)
    t = tlx.local_load(tlx.local_view(smem, 0))
    tl.store(out_ptr + offs, t)


@triton.jit
def _partial_mask_sync_load_kernel(
    a_ptr,
    out_ptr,
    VALID_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # The "sync-load the tail" fix: a masked *synchronous* tl.load lowers fine, so a
    # partial K-tile is handled by tl.load instead of tlx.async_load.
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs = offs_m[:, None] * BLOCK_K + offs_k[None, :]
    t = tl.load(a_ptr + offs, mask=offs_k[None, :] < VALID_K, other=0.0)
    tl.store(out_ptr + offs, t)


def _partial_mask_a_out(device):
    a = torch.randn(_PARTIAL_MASK_BLOCK_M, _PARTIAL_MASK_BLOCK_K, device=device, dtype=torch.float16)
    return a, torch.zeros_like(a)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_async_load_no_mask_ok_gfx950(device):
    a, out = _partial_mask_a_out(device)
    _partial_mask_async_load_kernel[(1, )](
        a,
        out,
        VALID_K=_PARTIAL_MASK_BLOCK_K,
        USE_MASK=False,
        BLOCK_M=_PARTIAL_MASK_BLOCK_M,
        BLOCK_K=_PARTIAL_MASK_BLOCK_K,
        num_warps=4,
    )
    torch.testing.assert_close(out, a)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_async_load_all_true_mask_ok_gfx950(device):
    # mask present but all-true (like an ALIGNED K, where every K-tile is full).
    a, out = _partial_mask_a_out(device)
    _partial_mask_async_load_kernel[(1, )](
        a,
        out,
        VALID_K=_PARTIAL_MASK_BLOCK_K,
        USE_MASK=True,
        BLOCK_M=_PARTIAL_MASK_BLOCK_M,
        BLOCK_K=_PARTIAL_MASK_BLOCK_K,
        num_warps=4,
    )
    torch.testing.assert_close(out, a)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_async_load_partial_mask_ok_gfx950(device):
    a, out = _partial_mask_a_out(device)
    _partial_mask_async_load_kernel[(1, )](
        a,
        out,
        VALID_K=5,
        USE_MASK=True,
        BLOCK_M=_PARTIAL_MASK_BLOCK_M,
        BLOCK_K=_PARTIAL_MASK_BLOCK_K,
        num_warps=4,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out[:, :5], a[:, :5])
    torch.testing.assert_close(out[:, 5:], torch.zeros_like(out[:, 5:]))


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_sync_load_partial_mask_fix_ok_gfx950(device):
    # THE FIX: the same partial mask via a synchronous tl.load compiles and is correct.
    a, out = _partial_mask_a_out(device)
    _partial_mask_sync_load_kernel[(1, )](
        a,
        out,
        VALID_K=5,
        BLOCK_M=_PARTIAL_MASK_BLOCK_M,
        BLOCK_K=_PARTIAL_MASK_BLOCK_K,
        num_warps=4,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out[:, :5], a[:, :5])


@triton.jit
def _plain_unaligned_vector_load_kernel(
    x_ptr,
    y_ptr,
    OFFSET: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    values = tl.load(x_ptr + OFFSET + offsets)
    tl.store(y_ptr + offsets, values)


@triton.jit
def _masked_unaligned_vector_load_kernel(
    x_ptr,
    y_ptr,
    OFFSET: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    VALID_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < VALID_SIZE
    values = tl.load(x_ptr + OFFSET + offsets, mask=mask, other=0)
    tl.store(y_ptr + offsets, values)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize(
    "dtype,block_size",
    [
        (torch.float16, 1024),
        (torch.float16, 2048),
        (torch.uint8, 4096),
    ],
)
@pytest.mark.parametrize("offset", [1, 2, 3, 7])
def test_plain_unaligned_vector_load_correctness_gfx950(device, offset, dtype, block_size):
    """Exercise several naturally aligned but non-vector-aligned addresses."""
    x = torch.arange(block_size + 8, device=device, dtype=dtype)
    actual = torch.empty(block_size, device=device, dtype=dtype)
    _plain_unaligned_vector_load_kernel[(1, )](
        x,
        actual,
        OFFSET=offset,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    torch.testing.assert_close(actual, x[offset:offset + block_size], atol=0.0, rtol=0.0)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_masked_unaligned_vector_load_correctness_gfx950(device):
    block_size = 2048
    valid_size = block_size - 8
    x = torch.arange(block_size + 1, device=device, dtype=torch.float16)
    actual = torch.empty(block_size, device=device, dtype=torch.float16)
    compiled = _masked_unaligned_vector_load_kernel[(1, )](
        x,
        actual,
        OFFSET=1,
        BLOCK_SIZE=block_size,
        VALID_SIZE=valid_size,
        num_warps=4,
    )
    expected = torch.zeros_like(actual)
    expected[:valid_size] = x[1:1 + valid_size]
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    assert "buffer_load_dwordx4" in compiled.asm["amdgcn"]
