"""
Correctness tests for Flash Attention kernels using the autoWS (automatic warp
specialization) flow.

The kernel is ported from tritonbench's blackwell_triton_fused_attention_dp
to remove the external dependency.
"""

import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_blackwell
from triton.tools.tensor_descriptor import TensorDescriptor

# =============================================================================
# Ported Flash Attention DP kernel
# =============================================================================


@triton.jit
def _mask_scalar(qk, col_limit_right, s, i):
    col_lim_right_s = col_limit_right - s
    col_lim_right_cur = max(col_lim_right_s, 0)
    mask = -1 << col_lim_right_cur
    mask_i_bit = (mask & (1 << i)) == 0
    return tl.where(mask_i_bit, qk, -float("inf"))


@triton.jit
def _apply_causal_mask(qk, col_limit_right, BLOCK_N: tl.constexpr):
    offs_n = tl.arange(0, BLOCK_N)[None, :]
    s = offs_n & ~0xF
    i = offs_n & 0xF
    return tl.map_elementwise(_mask_scalar, qk, col_limit_right, s, i)


@triton.jit
def _mul_f32x2(a, b):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b64 ra, rb, rc;
            mov.b64 ra, { $2, $3 };
            mov.b64 rb, { $4, $5 };
            mul.f32x2 rc, ra, rb;
            mov.b64 { $0, $1 }, rc;
        }
        """,
        "=r,=r,r,r,r,r",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=2,
    )


@triton.jit
def _fma_f32x2(a, b, c):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b64 ra, rb, rc, rd;
            mov.b64 ra, { $2, $3 };
            mov.b64 rb, { $4, $5 };
            mov.b64 rc, { $6, $7 };
            fma.rn.f32x2 rd, ra, rb, rc;
            mov.b64 { $0, $1 }, rd;
        }
        """,
        "=r,=r,r,r,r,r,r,r",
        [a, b, c],
        dtype=tl.float32,
        is_pure=True,
        pack=2,
    )


@triton.jit
def _reduce_fadd2(p0a, p1a, p0b, p1b):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b64 rc, ra, rb;
            mov.b64 ra, { $2, $4 };
            mov.b64 rb, { $3, $5 };
            add.f32x2 rc, ra, rb;
            mov.b64 { $0, $1 }, rc;
        }
        """,
        "=r,=r,r,r,r,r",
        [p0a, p0b, p1a, p1b],
        dtype=[tl.float32, tl.float32],
        is_pure=True,
        pack=1,
    )


@triton.jit
def _attn_fwd_subtile(
    q,
    k,
    offs_m,
    start_n,
    BLOCK_N,
    offs_n,
    qk_scale,
    l_i0,
    l_i1,
    m_i,
    acc,
    v,
    dtype: tl.constexpr,
    STAGE: tl.constexpr,
    SUBTILING: tl.constexpr,
    SUBTILING_P: tl.constexpr,
    VECT_MUL: tl.constexpr,
    FADD2_REDUCE: tl.constexpr,
):
    qk = tl.dot(q, k)
    if STAGE == 3:
        col_limit_right = (offs_m - start_n + 1)[:, None]
        qk = _apply_causal_mask(qk, col_limit_right, BLOCK_N)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        if VECT_MUL & 2:
            qk = _fma_f32x2(qk, qk_scale, -m_ij[:, None])
        else:
            qk = qk * qk_scale - m_ij[:, None]
    else:
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        if VECT_MUL & 2:
            qk = _fma_f32x2(qk, qk_scale, -m_ij[:, None])
        else:
            qk = qk * qk_scale - m_ij[:, None]

    PM: tl.constexpr = qk.shape[0]
    PN: tl.constexpr = qk.shape[1]

    if SUBTILING_P:
        qk0, qk1 = qk.reshape([PM, 2, PN // 2]).permute(0, 2, 1).split()
        p0 = tl.math.exp2(qk0)
        p0_bf16 = p0.to(dtype)
        p1 = tl.math.exp2(qk1)
        p1_bf16 = p1.to(dtype)
        p = tl.join(p0, p1).permute(0, 2, 1).reshape([PM, PN])
    else:
        p = tl.math.exp2(qk)

    alpha = tl.math.exp2(m_i - m_ij)
    if not FADD2_REDUCE:
        l_ij = tl.sum(p, 1)

    BM: tl.constexpr = acc.shape[0]
    BN: tl.constexpr = acc.shape[1]

    if SUBTILING:
        acc0, acc1 = acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
        if VECT_MUL & 1:
            acc0 = _mul_f32x2(acc0, alpha[:, None])
            acc1 = _mul_f32x2(acc1, alpha[:, None])
        else:
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
        acc = tl.join(acc0, acc1).permute(0, 2, 1).reshape([BM, BN])
    else:
        acc = acc * alpha[:, None]

    if FADD2_REDUCE:
        p0, p1 = p.reshape([PM, 2, PN // 2]).permute(0, 2, 1).split()
        l_ij0, l_ij1 = tl.reduce((p0, p1), axis=1, combine_fn=_reduce_fadd2)
        l_i0 = l_i0 * alpha + l_ij0
        l_i1 = l_i1 * alpha + l_ij1

    if not SUBTILING_P:
        p_bf16 = p.to(dtype)
    else:
        p_bf16 = tl.join(p0_bf16, p1_bf16).permute(0, 2, 1).reshape([PM, PN])
    acc = tl.dot(p_bf16, v, acc)
    if not FADD2_REDUCE:
        l_i0 = l_i0 * alpha + l_ij
    m_i = m_ij

    return l_i0, l_i1, m_i, acc


@triton.jit
def _attn_fwd_inner_oss_dp(
    acc0,
    acc1,
    l_i0,
    l_i0_1,
    l_i1,
    l_i1_1,
    m_i0,
    m_i1,
    q0,
    q1,
    desc_k,
    desc_v,
    offset_y,
    dtype: tl.constexpr,
    start_m,
    qk_scale,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m0: tl.constexpr,
    offs_m1: tl.constexpr,
    offs_n: tl.constexpr,
    N_CTX: tl.constexpr,
    warp_specialize: tl.constexpr,
    SUBTILING: tl.constexpr,
    SUBTILING_P: tl.constexpr,
    VECT_MUL: tl.constexpr,
    FADD2_REDUCE: tl.constexpr,
):
    if STAGE == 1:
        lo, hi = 0, N_CTX
    else:
        lo, hi = 0, (start_m + 1) * BLOCK_M

    offsetkv_y = offset_y + lo

    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize, disallow_acc_multi_buffer=True):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        k = desc_k.load([offsetkv_y, 0]).T
        v = desc_v.load([offsetkv_y, 0])

        l_i0, l_i0_1, m_i0, acc0 = _attn_fwd_subtile(
            q0,
            k,
            offs_m0,
            start_n,
            BLOCK_N,
            offs_n,
            qk_scale,
            l_i0,
            l_i0_1,
            m_i0,
            acc0,
            v,
            dtype,
            STAGE,
            SUBTILING,
            SUBTILING_P,
            VECT_MUL,
            FADD2_REDUCE,
        )
        l_i1, l_i1_1, m_i1, acc1 = _attn_fwd_subtile(
            q1,
            k,
            offs_m1,
            start_n,
            BLOCK_N,
            offs_n,
            qk_scale,
            l_i1,
            l_i1_1,
            m_i1,
            acc1,
            v,
            dtype,
            STAGE,
            SUBTILING,
            SUBTILING_P,
            VECT_MUL,
            FADD2_REDUCE,
        )

        offsetkv_y += BLOCK_N

    return acc0, acc1, l_i0, l_i0_1, l_i1, l_i1_1, m_i0, m_i1


@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, triton.language.core.tensor_descriptor_base):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


@triton.jit
def _attn_fwd_tma_dp(
    sm_scale,
    M,
    Z,
    H,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    start_m,
    off_hz,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    warp_specialize: tl.constexpr,
    dtype: tl.constexpr,
    SUBTILING: tl.constexpr,
    SUBTILING_P: tl.constexpr,
    VECT_MUL: tl.constexpr,
    FADD2_REDUCE: tl.constexpr,
):
    off_z = off_hz // H
    off_h = off_hz % H

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    offs_m0 = start_m * BLOCK_M + tl.arange(0, BLOCK_M // 2)
    offs_m1 = start_m * BLOCK_M + tl.arange(BLOCK_M // 2, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    m_i0 = tl.zeros([BLOCK_M // 2], dtype=tl.float32) - float("inf")
    l_i0_0 = tl.zeros([BLOCK_M // 2], dtype=tl.float32) + 1.0
    acc0 = tl.zeros([BLOCK_M // 2, HEAD_DIM], dtype=tl.float32)

    m_i1 = tl.zeros([BLOCK_M // 2], dtype=tl.float32) - float("inf")
    l_i1_0 = tl.zeros([BLOCK_M // 2], dtype=tl.float32) + 1.0
    acc1 = tl.zeros([BLOCK_M // 2, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)

    q0 = desc_q.load([qo_offset_y, 0])
    q1 = desc_q.load([qo_offset_y + BLOCK_M // 2, 0])

    if FADD2_REDUCE:
        l_i0_1 = tl.zeros([BLOCK_M // 2], dtype=tl.float32)
        l_i1_1 = tl.zeros([BLOCK_M // 2], dtype=tl.float32)
    else:
        l_i0_1 = 0
        l_i1_1 = 0

    acc0, acc1, l_i0_0, l_i0_1, l_i1_0, l_i1_1, m_i0, m_i1 = _attn_fwd_inner_oss_dp(
        acc0,
        acc1,
        l_i0_0,
        l_i0_1,
        l_i1_0,
        l_i1_1,
        m_i0,
        m_i1,
        q0,
        q1,
        desc_k,
        desc_v,
        offset_y,
        dtype,
        start_m,
        qk_scale,
        BLOCK_M,
        HEAD_DIM,
        BLOCK_N,
        STAGE,
        offs_m0,
        offs_m1,
        offs_n,
        N_CTX,
        warp_specialize,
        SUBTILING,
        SUBTILING_P,
        VECT_MUL,
        FADD2_REDUCE,
    )

    if FADD2_REDUCE:
        l_i0 = l_i0_0 + l_i0_1
        l_i1 = l_i1_0 + l_i1_1
    else:
        l_i0 = l_i0_0
        l_i1 = l_i1_0

    m_i0 += tl.math.log2(l_i0)
    acc0 = acc0 / l_i0[:, None]
    m_ptrs0 = M + off_hz * N_CTX + offs_m0
    tl.store(m_ptrs0, m_i0)
    desc_o.store([qo_offset_y, 0], acc0.to(dtype))

    m_i1 += tl.math.log2(l_i1)
    acc1 = acc1 / l_i1[:, None]
    m_ptrs1 = M + off_hz * N_CTX + offs_m1
    tl.store(m_ptrs1, m_i1)
    desc_o.store([qo_offset_y + BLOCK_M // 2, 0], acc1.to(dtype))


@triton.jit
def _attn_fwd_persist(
    sm_scale,
    M,
    Z,
    H,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    warp_specialize: tl.constexpr,
    OUTER_LOOP: tl.constexpr,
    dtype: tl.constexpr,
    SUBTILING: tl.constexpr,
    SUBTILING_P: tl.constexpr,
    VECT_MUL: tl.constexpr,
    FADD2_REDUCE: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr,
):
    prog_id = tl.program_id(0)
    num_progs = tl.num_programs(0)
    num_pid_m = tl.cdiv(N_CTX, BLOCK_M)
    num_pid_n = Z * H
    num_pid_in_group = num_pid_m * GROUP_SIZE_N
    total_tiles = num_pid_m * Z * H

    tiles_per_sm = total_tiles // num_progs
    if prog_id < total_tiles % num_progs:
        tiles_per_sm += 1

    tile_idx = prog_id
    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(
        desc_q,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M // 2, HEAD_DIM],
    )
    desc_k = _maybe_make_tensor_desc(
        desc_k,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_v = _maybe_make_tensor_desc(
        desc_v,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_o = _maybe_make_tensor_desc(
        desc_o,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M // 2, HEAD_DIM],
    )

    for _ in tl.range(0, tiles_per_sm, warp_specialize=warp_specialize and OUTER_LOOP):
        group_id = tile_idx // num_pid_in_group
        first_pid_n = group_id * GROUP_SIZE_N
        group_size_n = min(num_pid_n - first_pid_n, GROUP_SIZE_N)
        off_hz = first_pid_n + ((tile_idx % num_pid_in_group) % group_size_n)
        start_m = (tile_idx % num_pid_in_group) // group_size_n

        _attn_fwd_tma_dp(
            sm_scale,
            M,
            Z,
            H,
            desc_q,
            desc_k,
            desc_v,
            desc_o,
            start_m,
            off_hz,
            N_CTX,
            HEAD_DIM,
            BLOCK_M,
            BLOCK_N,
            STAGE,
            warp_specialize and not OUTER_LOOP,
            dtype,
            SUBTILING,
            SUBTILING_P,
            VECT_MUL,
            FADD2_REDUCE,
        )
        tile_idx += num_progs


# =============================================================================
# Flash Attention: Launcher & test utilities
# =============================================================================


def attention_forward(q, k, v, causal, sm_scale):
    """Launch the persistent WS flash attention DP kernel."""
    HEAD_DIM = q.shape[-1]
    Z, H, N_CTX = q.shape[0], q.shape[1], q.shape[2]
    o = torch.empty_like(q)
    stage = 3 if causal else 1

    lse = torch.empty((Z, H, N_CTX), device=q.device, dtype=torch.float32)

    BLOCK_M = 256
    BLOCK_N = 128
    y_dim = Z * H * N_CTX

    desc_q = TensorDescriptor(
        q,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M // 2, HEAD_DIM],
    )
    desc_k = TensorDescriptor(
        k,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_v = TensorDescriptor(
        v,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_o = TensorDescriptor(
        o,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M // 2, HEAD_DIM],
    )

    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    grid = lambda META: (
        min(NUM_SMS,
            triton.cdiv(N_CTX, META["BLOCK_M"]) * Z * H),
        1,
        1,
    )

    _attn_fwd_persist[grid](
        sm_scale,
        lse,
        Z,
        H,
        desc_q,
        desc_k,
        desc_v,
        desc_o,
        N_CTX=N_CTX,
        HEAD_DIM=HEAD_DIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        STAGE=stage,
        warp_specialize=True,
        OUTER_LOOP=True,
        dtype=tl.bfloat16,
        SUBTILING=True,
        SUBTILING_P=False,
        VECT_MUL=0,
        FADD2_REDUCE=False,
        GROUP_SIZE_N=1,
        num_stages=3,
        num_warps=4,
        maxnreg=128,
    )
    return o


class FlashAttention:
    """Common utilities for Flash Attention autoWS correctness tests."""

    # (Z, H, N_CTX, HEAD_DIM)
    SHAPES = [(4, 32, 8192, 128)]

    @staticmethod
    def create_inputs(Z, H, N_CTX, HEAD_DIM, dtype=torch.bfloat16):
        torch.manual_seed(20)
        q = torch.empty((Z, H, N_CTX, HEAD_DIM), device="cuda", dtype=dtype).normal_(mean=0.0, std=0.5)
        k = torch.empty((Z, H, N_CTX, HEAD_DIM), device="cuda", dtype=dtype).normal_(mean=0.0, std=0.5)
        v = torch.empty((Z, H, N_CTX, HEAD_DIM), device="cuda", dtype=dtype).normal_(mean=0.0, std=0.5)
        return q, k, v

    @staticmethod
    def get_reference(q, k, v, sm_scale, causal):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=sm_scale, is_causal=causal)


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.parametrize("causal", [False, True], ids=["non_causal", "causal"])
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_blackwell_fa_autows_dp(causal, dtype):
    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True
        triton.knobs.nvidia.use_meta_partition = True

        for Z, H, N_CTX, HEAD_DIM in FlashAttention.SHAPES:
            sm_scale = 1.0 / (HEAD_DIM**0.5)
            q, k, v = FlashAttention.create_inputs(Z, H, N_CTX, HEAD_DIM, dtype)
            ref_out = FlashAttention.get_reference(q, k, v, sm_scale, causal)
            tri_out = attention_forward(q, k, v, causal, sm_scale)
            torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=1e-2)
