import pytest
import torch

import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_host_descriptor():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def is_blackwell():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


def is_hopper():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 9


@triton.jit
def _attn_fwd_subtile(
    q,
    k,
    offs_m,
    start_n,
    offs_n,
    qk_scale,
    l_i0,
    l_i1,  # used when FADD2_REDUCE is true
    m_i,
    acc,
    v,
    dtype: tl.constexpr,
    STAGE: tl.constexpr,
    SUBTILING: tl.constexpr,
    VECT_MUL: tl.constexpr,
    FADD2_REDUCE: tl.constexpr,
):
    qk = tl.dot(q, k)
    if STAGE == 2:
        mask = offs_m[:, None] >= (start_n + offs_n[None, :])
        qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
    else:
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        if VECT_MUL == 2 or VECT_MUL == 3:
            qk = _fma_f32x2(qk, qk_scale, -m_ij[:, None])
        else:
            qk = qk * qk_scale - m_ij[:, None]
    p = tl.math.exp2(qk)
    # -- compute correction factor
    alpha = tl.math.exp2(m_i - m_ij)
    if not FADD2_REDUCE:
        l_ij = tl.sum(p, 1)

    # -- update output accumulator --
    BM: tl.constexpr = acc.shape[0]
    BN: tl.constexpr = acc.shape[1]

    if SUBTILING:
        acc0, acc1 = acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
        if VECT_MUL == 1 or VECT_MUL == 3:
            acc0 = _mul_f32x2(acc0, alpha[:, None])
            acc1 = _mul_f32x2(acc1, alpha[:, None])
        else:
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
        acc = tl.join(acc0, acc1).permute(0, 2, 1).reshape([BM, BN])
    else:
        acc = acc * alpha[:, None]

    # update m_i and l_i
    # place this at the end of the loop to reduce register pressure
    PM: tl.constexpr = p.shape[0]
    PN: tl.constexpr = p.shape[1]
    if FADD2_REDUCE:
        p0, p1 = p.reshape([PM, 2, PN // 2]).permute(0, 2, 1).split()
        l_ij0, l_ij1 = tl.reduce((p0, p1), axis=1, combine_fn=_reduce_fadd2)
        l_i0 = l_i0 * alpha + l_ij0
        l_i1 = l_i1 * alpha + l_ij1

    # prepare p and v for the dot
    p = p.to(dtype)
    # note that this non transposed v for FP8 is only supported on Blackwell
    acc = tl.dot(p, v, acc)
    # update m_i and l_i
    # place this at the end of the loop to reduce register pressure
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
    q1,  #
    desc_k,
    desc_v,  #
    offset_y,
    dtype: tl.constexpr,
    start_m,
    qk_scale,  #
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,
    offs_m0: tl.constexpr,
    offs_m1: tl.constexpr,  #
    offs_n: tl.constexpr,  #
    N_CTX: tl.constexpr,
    warp_specialize: tl.constexpr,
    SUBTILING: tl.constexpr,
    VECT_MUL: tl.constexpr,
    FADD2_REDUCE: tl.constexpr,
):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    offsetkv_y = offset_y + lo

    # loop over k, v and update accumulator
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize, disallow_acc_multi_buffer=True):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        k = desc_k.load([offsetkv_y, 0]).T
        v = desc_v.load([offsetkv_y, 0])

        l_i0, l_i0_1, m_i0, acc0 = _attn_fwd_subtile(
            q0,
            k,
            offs_m0,
            start_n,
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
            VECT_MUL,
            FADD2_REDUCE,
        )
        l_i1, l_i1_1, m_i1, acc1 = _attn_fwd_subtile(
            q1,
            k,
            offs_m1,
            start_n,
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
            VECT_MUL,
            FADD2_REDUCE,
        )

        offsetkv_y += BLOCK_N

    return acc0, acc1, l_i0, l_i0_1, l_i1, l_i1_1, m_i0, m_i1


def _host_descriptor_pre_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    nargs["desc_q"].block_shape = [BLOCK_M // 2, HEAD_DIM]  # due to data partitioning
    if nargs["FP8_OUTPUT"]:
        nargs["desc_v"].block_shape = [HEAD_DIM, BLOCK_N]
    else:
        nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_o"].block_shape = [BLOCK_M // 2, HEAD_DIM]


if is_hip():
    NUM_STAGES_OPTIONS = [1]
elif supports_host_descriptor():
    NUM_STAGES_OPTIONS = [3]
else:
    NUM_STAGES_OPTIONS = [3]

configs = [
    triton.Config(
        {
            "BLOCK_M": BM,
            "BLOCK_N": BN,
        },
        num_stages=s,
        num_warps=w,
        pre_hook=_host_descriptor_pre_hook,
        # ir_override=f"/home/mren/OpenSource/tritonbench/override/_attn_fwd_persist.ttgir"
    ) for BM in [256] for BN in [128] for s in NUM_STAGES_OPTIONS for w in [4]
]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    return not (is_cuda() and torch.cuda.get_device_capability()[0] == 9 and BLOCK_M * BLOCK_N < 128 * 128
                and conf.num_warps == 8)


def prune_invalid_configs(configs, named_args, **kwargs):
    N_CTX = kwargs["N_CTX"]

    # Filter out configs where BLOCK_M > N_CTX
    return [conf for conf in configs if conf.kwargs.get("BLOCK_M", 0) <= N_CTX]


@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


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
def _attn_fwd_tma_dp(
    sm_scale,
    M,  #
    Z,
    H,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    pid,
    off_hz,
    N_CTX: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    FP8_OUTPUT: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    warp_specialize: tl.constexpr,  #
    dtype: tl.constexpr,
    SUBTILING: tl.constexpr,
    VECT_MUL: tl.constexpr,
    FADD2_REDUCE: tl.constexpr,
):
    start_m = pid  # tl.program_id(0)
    # off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    # initialize offsets
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

    if STAGE & 1:
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
            q1,  #
            desc_k,
            desc_v,  #
            offset_y,
            dtype,
            start_m,
            qk_scale,  #
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,  #
            4 - STAGE,
            offs_m0,
            offs_m1,
            offs_n,
            N_CTX,  #
            warp_specialize,
            SUBTILING,
            VECT_MUL,
            FADD2_REDUCE,
        )
    if STAGE & 2:
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
            q1,  #
            desc_k,
            desc_v,  #
            offset_y,
            dtype,
            start_m,
            qk_scale,  #
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,  #
            2,
            offs_m0,
            offs_m1,
            offs_n,
            N_CTX,  #
            warp_specialize,
            SUBTILING,
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


@triton.autotune(
    configs=list(filter(keep, configs)),
    key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "warp_specialize"],
    prune_configs_by={"early_config_prune": prune_invalid_configs},
)
@triton.jit
def _attn_fwd(
    sm_scale,
    M,  #
    Z,
    H,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    N_CTX: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    FP8_OUTPUT: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    warp_specialize: tl.constexpr,  #
    dtype: tl.constexpr,
    SUBTILING: tl.constexpr,
    VECT_MUL: tl.constexpr,
    FADD2_REDUCE: tl.constexpr,
):
    pid = tl.program_id(0)
    off_hz = tl.program_id(1)
    _attn_fwd_tma_dp(
        sm_scale,
        M,
        Z,
        H,
        desc_q,
        desc_k,
        desc_v,
        desc_o,
        pid,
        off_hz,
        N_CTX,
        HEAD_DIM,
        BLOCK_M,
        BLOCK_N,
        FP8_OUTPUT,
        STAGE,
        warp_specialize,
        dtype,
        SUBTILING,
        VECT_MUL,
        FADD2_REDUCE,
    )


@triton.autotune(
    configs=list(filter(keep, configs)),
    key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "warp_specialize"],
    prune_configs_by={"early_config_prune": prune_invalid_configs},
)
@triton.jit
def _attn_fwd_persist(
    sm_scale,
    M,  #
    Z,
    H,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    N_CTX: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    FP8_OUTPUT: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    warp_specialize: tl.constexpr,  #
    OUTER_LOOP: tl.constexpr,
    dtype: tl.constexpr,
    SUBTILING: tl.constexpr,
    VECT_MUL: tl.constexpr,
    FADD2_REDUCE: tl.constexpr,
):
    n_tile_num = tl.cdiv(N_CTX, BLOCK_M)
    prog_id = tl.program_id(0)
    num_progs = tl.num_programs(0)
    total_tiles = n_tile_num * Z * H

    tiles_per_sm = total_tiles // num_progs
    if prog_id < total_tiles % num_progs:
        tiles_per_sm += 1

    tile_idx = prog_id
    # inner loop warpspec vs. outer loop warpspec
    for _ in tl.range(0, tiles_per_sm, warp_specialize=warp_specialize and OUTER_LOOP):
        pid = tile_idx % n_tile_num
        off_hz = tile_idx // n_tile_num
        _attn_fwd_tma_dp(
            sm_scale,
            M,
            Z,
            H,
            desc_q,
            desc_k,
            desc_v,
            desc_o,
            pid,
            off_hz,
            N_CTX,
            HEAD_DIM,
            BLOCK_M,
            BLOCK_N,
            FP8_OUTPUT,
            STAGE,
            warp_specialize and not OUTER_LOOP,
            dtype,
            SUBTILING,
            VECT_MUL,
            FADD2_REDUCE,
        )
        tile_idx += num_progs


@triton.jit
def _attn_bwd_preprocess(O, DO,  #
                         Delta,  #
                         Z, H, N_CTX,  #
                         BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr,  #
                         ):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    # load
    o = tl.load(O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :])
    do = tl.load(DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


def _bwd_pre_hook(nargs):
    """Zero out DQ before each autotune benchmark run.
    DQ is accumulated via atomic_add, so stale values from prior runs corrupt results."""
    nargs["DQ"].zero_()


configs_bwd = [
    triton.Config(
        {
            "BLOCK_M1": BM,
            "BLOCK_N1": BN,
        },
        num_warps=4,
        num_stages=1,
        pre_hook=_bwd_pre_hook,
    ) for BM in [64] for BN in [64, 128]
]


@triton.autotune(configs=configs_bwd, key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _attn_bwd(
    Q,
    K,
    V,
    sm_scale,
    DO,
    DQ,
    DK,
    DV,
    M,
    D,
    stride_z,
    stride_h,
    stride_tok,
    stride_d,
    H,
    N_CTX,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Monolithic backward kernel: one thread block per K/V block.
    Copied from the proven _bwd_simple pattern in test_bwd_debug.py."""
    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz

    offs_k = tl.arange(0, HEAD_DIM)
    start_n = pid * BLOCK_N1
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    # Load K and V for this block — they stay in SRAM for the entire inner loop.
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)

    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    # Iterate over all Q blocks (the entire inner loop is inlined here,
    # NOT delegated to a helper function — this is critical for correctness).
    RCP_LN2: tl.constexpr = 1.4426950408889634
    curr_m = 0
    for _ in range(N_CTX // BLOCK_M1):
        offs_m = curr_m + tl.arange(0, BLOCK_M1)

        q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
        do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
        m = tl.load(M + offs_m)
        Di = tl.load(D + offs_m)

        # Recompute P = softmax(QK^T * sm_scale) in log2 space
        qk = tl.dot(q, tl.trans(k))  # [M, N]
        qk = qk * (sm_scale * RCP_LN2)
        p = tl.math.exp2(qk - m[:, None])  # [M, N]

        # dV += P^T @ dO
        pp = p.to(tl.float16)
        dv += tl.dot(tl.trans(pp), do)

        # dP = dO @ V^T, dS = P * (dP - Delta)
        dp = tl.dot(do, tl.trans(v)).to(tl.float32)  # [M, N]
        ds = p * (dp - Di[:, None])  # [M, N]
        ds = ds.to(tl.float16)

        # dK += dS^T @ Q
        dk += tl.dot(tl.trans(ds), q)

        # dQ += dS @ K * sm_scale (accumulated via atomic add)
        dq = tl.dot(ds, k)  # [M, D]
        dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
        tl.atomic_add(dq_ptrs, dq.to(tl.float32) * sm_scale)

        curr_m += BLOCK_M1

    # Store dK (scaled) and dV
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    dk = dk * sm_scale
    tl.store(dk_ptrs, dk)
    tl.store(dv_ptrs, dv)


def torch_dtype_to_triton(dtype):
    if dtype == torch.float8_e5m2:
        return tl.float8e5
    return getattr(tl, str(dtype).split(".")[1])


class _attention_opt(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, baseVariant, SUBTILING, VECT_MUL, FADD2_REDUCE):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        stage = 3 if causal else 1
        extra_kern_args = {}

        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        warp_specialize = baseVariant == "ws" or baseVariant == "ws_persistent"
        # Use device_descriptor for Hopper + warpspec.
        if supports_host_descriptor() and not (is_hopper() and warp_specialize):
            # Note that on Hopper we cannot perform a FP8 dot with a non-transposed second tensor
            y_dim = q.shape[0] * q.shape[1] * q.shape[2]

            dummy_block = [1, 1]
            desc_q = TensorDescriptor(
                q,
                shape=[y_dim, HEAD_DIM_K],
                strides=[HEAD_DIM_K, 1],
                block_shape=dummy_block,
            )
            if q.dtype == torch.float8_e5m2:
                desc_v = TensorDescriptor(
                    v,
                    shape=[HEAD_DIM_K, y_dim],
                    strides=[q.shape[2], 1],
                    block_shape=dummy_block,
                )
            else:
                desc_v = TensorDescriptor(
                    v,
                    shape=[y_dim, HEAD_DIM_K],
                    strides=[HEAD_DIM_K, 1],
                    block_shape=dummy_block,
                )
            desc_k = TensorDescriptor(
                k,
                shape=[y_dim, HEAD_DIM_K],
                strides=[HEAD_DIM_K, 1],
                block_shape=dummy_block,
            )
            desc_o = TensorDescriptor(
                o,
                shape=[y_dim, HEAD_DIM_K],
                strides=[HEAD_DIM_K, 1],
                block_shape=dummy_block,
            )
        else:
            desc_q = q
            desc_v = v
            desc_k = k
            desc_o = o

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

        def grid(META):
            return (
                triton.cdiv(q.shape[2], META["BLOCK_M"]),
                q.shape[0] * q.shape[1],
                1,
            )

        def grid_persist(META):
            return (
                min(
                    NUM_SMS,
                    triton.cdiv(q.shape[2], META["BLOCK_M"]) * q.shape[0] * q.shape[1],
                ),
                1,
                1,
            )

        def grid_debug(META):
            return (
                1,
                1,
                1,
            )

        ctx.grid = grid
        persistent = baseVariant == "persistent" or baseVariant == "ws_persistent"
        if is_blackwell() and warp_specialize:
            if HEAD_DIM_K == 128 and (q.dtype == torch.float16 or q.dtype == torch.bfloat16):
                extra_kern_args["maxnreg"] = 128
            else:
                extra_kern_args["maxnreg"] = 128
        if persistent:
            _attn_fwd_persist[grid_persist](
                sm_scale,
                M,  #
                q.shape[0],
                q.shape[1],  #
                desc_q,
                desc_k,
                desc_v,
                desc_o,  #
                N_CTX=q.shape[2],  #
                HEAD_DIM=HEAD_DIM_K,  #
                FP8_OUTPUT=q.dtype == torch.float8_e5m2,  #
                STAGE=stage,  #
                warp_specialize=warp_specialize,
                OUTER_LOOP=True,
                dtype=torch_dtype_to_triton(q.dtype),
                SUBTILING=SUBTILING,
                VECT_MUL=VECT_MUL,
                FADD2_REDUCE=FADD2_REDUCE,
                **extra_kern_args,
            )
        else:
            _attn_fwd[grid](
                sm_scale,
                M,  #
                q.shape[0],
                q.shape[1],  #
                desc_q,
                desc_k,
                desc_v,
                desc_o,  #
                N_CTX=q.shape[2],  #
                HEAD_DIM=HEAD_DIM_K,  #
                FP8_OUTPUT=q.dtype == torch.float8_e5m2,  #
                STAGE=stage,  #
                warp_specialize=warp_specialize,
                dtype=torch_dtype_to_triton(q.dtype),
                SUBTILING=SUBTILING,
                VECT_MUL=VECT_MUL,
                FADD2_REDUCE=FADD2_REDUCE,
                **extra_kern_args,
            )

        ctx.save_for_backward(q, k, v, o, M)

        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        assert do.is_contiguous()
        assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        dq = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
        dk = torch.empty_like(k, dtype=torch.float32)
        dv = torch.empty_like(v, dtype=torch.float32)
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        PRE_BLOCK = 128
        assert N_CTX % PRE_BLOCK == 0
        pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        delta = torch.empty_like(M)
        _attn_bwd_preprocess[pre_grid](
            o, do,  #
            delta,  #
            BATCH, N_HEAD, N_CTX,  #
            BLOCK_M=PRE_BLOCK, HEAD_DIM=ctx.HEAD_DIM,  #
        )

        def grid(meta):
            return (
                triton.cdiv(N_CTX, meta["BLOCK_N1"]),
                1,
                BATCH * N_HEAD,
            )

        _attn_bwd[grid](
            q, k, v, ctx.sm_scale, do, dq, dk, dv,  #
            M, delta,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            N_HEAD, N_CTX,  #
            HEAD_DIM=ctx.HEAD_DIM,  #
        )

        return dq, dk, dv, None, None, None, None, None, None


attention = _attention_opt.apply


@pytest.mark.skipif(
    not is_blackwell(),
    reason="Requires Blackwell GPU",
)
@pytest.mark.parametrize("Z", [8])
@pytest.mark.parametrize("H", [16])
@pytest.mark.parametrize("N_CTX", [1024, 2048])
@pytest.mark.parametrize("HEAD_DIM", [64, 128])
@pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("mode", ["fwd", "bwd"])
@pytest.mark.parametrize("provider", ["triton-fp16"])
@pytest.mark.parametrize("SUBTILING", [False, True])
@pytest.mark.parametrize("VECT_MUL", [0, 1, 2, 3])
@pytest.mark.parametrize("FADD2_REDUCE", [False, True])
def test_op(
    Z,
    H,
    N_CTX,
    HEAD_DIM,
    causal,
    mode,
    provider,
    SUBTILING,
    VECT_MUL,
    FADD2_REDUCE,
    dtype=torch.float16,
):
    if mode == "bwd" and "fp8" in provider:
        pytest.skip("Backward pass with FP8 is not supported.")
    torch.manual_seed(20)
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    sm_scale = 0.5
    # reference implementation
    ref_dtype = dtype
    if mode == "fwd" and "fp8" in provider:
        ref_dtype = torch.float32
    q = q.to(ref_dtype)
    k = k.to(ref_dtype)
    v = v.to(ref_dtype)
    M = torch.tril(torch.ones((N_CTX, N_CTX), device=DEVICE))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1)
    p = p.to(ref_dtype)
    # p = torch.exp(p)
    ref_out = torch.matmul(p, v).half()
    if mode == "bwd":
        dout = torch.randn_like(q)
        ref_out.backward(dout)
        ref_dv, v.grad = v.grad.clone(), None
        ref_dk, k.grad = k.grad.clone(), None
        ref_dq, q.grad = q.grad.clone(), None
    # triton implementation
    if mode == "fwd" and "fp8" in provider:
        q = q.to(torch.float8_e5m2)
        k = k.to(torch.float8_e5m2)
        v = v.permute(0, 1, 3, 2).contiguous()
        v = v.permute(0, 1, 3, 2)
        v = v.to(torch.float8_e5m2)
    tri_out = attention(q, k, v, causal, sm_scale, "ws_persistent", SUBTILING, VECT_MUL, FADD2_REDUCE).half()
    if mode == "fwd":
        atol = 3 if "fp8" in provider else 1e-2
        torch.testing.assert_close(tri_out, ref_out, atol=atol, rtol=0)
        return
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    # compare
    torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=0)
    rtol = 0.0
    # Relative tolerance workaround for known hardware limitation of CDNA2 GPU.
    # For details see https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
    if (torch.version.hip is not None and triton.runtime.driver.active.get_current_target().arch == "gfx90a"):
        rtol = 1e-2
    torch.testing.assert_close(tri_dv, ref_dv, atol=1e-2, rtol=rtol)
    torch.testing.assert_close(tri_dk, ref_dk, atol=1e-2, rtol=rtol)
    torch.testing.assert_close(tri_dq, ref_dq, atol=1e-2, rtol=rtol)


try:
    from flash_attn.flash_attn_interface import (
        flash_attn_qkvpacked_func as flash_attn_func, )

    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

TORCH_HAS_FP8 = False
BATCH, N_HEADS = 4, 32
# vary seq length for fixed head and batch=4
configs = []
for HEAD_DIM in [64, 128]:
    configs.append(
        triton.testing.Benchmark(
            x_names=["N_CTX"],
            x_vals=[2**i for i in range(10, 15)],
            line_arg="provider",
            line_vals=["triton-fp16"] + (["flash"] if HAS_FLASH else []),
            line_names=["Triton [FP16]"] + (["Flash-2"] if HAS_FLASH else []),
            styles=[("red", "-"), ("blue", "-"), ("green", "-")],
            ylabel="TFLOPS",
            plot_name=f"fused-attention-ws-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}",
            args={
                "H": N_HEADS,
                "BATCH": BATCH,
                "HEAD_DIM": HEAD_DIM,
                "mode": "fwd",
            },
        ))


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, HEAD_DIM, mode, provider, device=DEVICE):
    assert mode in ["fwd", "bwd"]
    dtype = torch.float16
    if "triton" in provider:
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        if mode == "fwd" and "fp8" in provider:
            q = q.to(torch.float8_e5m2)
            k = k.to(torch.float8_e5m2)
            v = v.permute(0, 1, 3, 2).contiguous()
            v = v.permute(0, 1, 3, 2)
            v = v.to(torch.float8_e5m2)
        sm_scale = 1.3
        SUBTILING = True
        VECT_MUL = False
        FADD2_REDUCE = False
        fn = lambda: attention(q, k, v, False, sm_scale, "ws_persistent", SUBTILING, VECT_MUL, FADD2_REDUCE)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)

    if provider == "flash":
        qkv = torch.randn(
            (BATCH, N_CTX, 3, H, HEAD_DIM),
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        fn = lambda: flash_attn_func(qkv)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops * 1e-12 / (ms * 1e-3)


if __name__ == "__main__":
    if is_blackwell():
        print("Running benchmarks...")
        bench_flash_attention.run(print_data=True)
    else:
        print("Skipping benchmarks, no Blackwell GPU found.")
