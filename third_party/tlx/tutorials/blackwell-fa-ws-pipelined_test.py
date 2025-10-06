import pytest
import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.tools.tensor_descriptor import TensorDescriptor
from triton._internal_testing import is_blackwell

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def _host_descriptor_pre_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    HEAD_DIM = nargs["HEAD_DIM"]
    NUM_MMA_GROUPS = nargs["NUM_MMA_GROUPS"]
    BLOCK_M_SPLIT = BLOCK_M // NUM_MMA_GROUPS
    nargs["desc_q"].block_shape = [BLOCK_M_SPLIT, HEAD_DIM]
    if nargs["FP8_OUTPUT"]:
        nargs["desc_v"].block_shape = [HEAD_DIM, BLOCK_N]
    else:
        nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_o"].block_shape = [BLOCK_M_SPLIT, HEAD_DIM]


configs = [
    # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'NUM_BUFFERS_KV': 3, 'NUM_BUFFERS_QK': 1, 'NUM_MMA_GROUPS': 1},
    #               num_stages=0, num_warps=4, pre_hook=_host_descriptor_pre_hook),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'NUM_BUFFERS_KV': 3, 'NUM_BUFFERS_QK': 1, 'NUM_MMA_GROUPS': 2},
                  num_stages=0, num_warps=4, pre_hook=_host_descriptor_pre_hook),
]


@triton.jit
def _get_bufidx_phase(accum_cnt, NUM_BUFFERS_KV):
    bufIdx = accum_cnt % NUM_BUFFERS_KV
    phase = (accum_cnt // NUM_BUFFERS_KV) & 1
    return bufIdx, phase


@triton.jit
def _compute_offsets(H, N_CTX, BLOCK_M):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    lo, hi = 0, N_CTX
    kv_offset_y = offset_y + lo
    return start_m, off_hz, lo, hi, qo_offset_y, kv_offset_y


# Original add_round_down(x, y) did: add.rm.ftz.f32  (round-down)
# We only need it to produce floor(x) when paired with the big-constant trick.
# In Triton, just compute floor(x) explicitly and add y.
@triton.jit
def add_round_down(x, y):
    # For the original usage (x rounded down; y is the big constant),
    # this is equivalent to add.rm on (x + y) then subtracting y later.
    return tl.math.floor(x) + y


# ============================================================================
# Custom exp2 Polynomial Approximation (the "emulation" path)
# ============================================================================


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
def evaluate_polynomial(x, coeffs: tl.constexpr):
    """
    Returns P(x) with scalar coeffs using Horner's rule and inline PTX FMA.
    P(t) = sum_{i=0..deg} coeffs[i] * t^i
    """
    deg = len(coeffs) - 1
    out = coeffs[deg]
    for i in range(deg - 1, -1, -1):
        out = _fma_f32x2(out, x, coeffs[i])   # out = out*x + coeffs[i]
    return out

@triton.jit
def combine_int_frac_ex2(x_rounded, frac_ex2):
    """
    Compose: out_bits = (x_rounded_bits << 23) + frac_ex2_bits
    where:
      - x_rounded carries the integer part (floor) in its low bits
      - frac_ex2 carries the mantissa approximation of 2**fraction(x)
    Returns fp32 with those bits.
    """
    out = tl.inline_asm_elementwise(
        asm="""
        {
          .reg .s32 x_rounded_i, frac_ex_i, x_rounded_e, out_i;
          mov.b32 x_rounded_i, $1;
          mov.b32 frac_ex_i, $2;
          shl.b32 x_rounded_e, x_rounded_i, 23;
          add.s32 out_i, x_rounded_e, frac_ex_i;
          mov.b32 $0, out_i;
        }
        """,
        constraints="=f, f, f",        # $0: out(float), $1: x_rounded(float), $2: frac_ex2(float)
        args=[x_rounded, frac_ex2],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    return out




@triton.jit
def ex2_emulation(x):
    # We assume x <= 127.0
    x = tl.maximum(x, -127.0)

    FP32_ROUND_INT = (2.0 ** 23) + (2.0 ** 22)  # same constant you used

    # Emulate your “round down fractional” split explicitly
    x_rounded = add_round_down(x, FP32_ROUND_INT)
    x_rounded_back = x_rounded - FP32_ROUND_INT
    x_frac = x - x_rounded_back                       # r in [0,1)

    # Your degree-3 approximation coefficients for 2**r, r in [0,1)

                    # ---- inline polynomial for 2**r (degree-3, Horner) ----
    # Coeffs: (C0, C1, C2, C3)
    C0 = 1.0
    C1 = 0.695146143436431884765625
    C2 = 0.227564394474029541015625
    C3 = 0.077119089663028717041015625

    x_frac_ex2 = C3
    x_frac_ex2 = _fma_f32x2(x_frac_ex2, x_frac, C2)   # t = C3*r + C2
    x_frac_ex2 = _fma_f32x2(x_frac_ex2, x_frac, C1)   # t = (..)*r + C1
    x_frac_ex2 = _fma_f32x2(x_frac_ex2, x_frac, C0)   # t = (..)*r + C0   ~ 2**r


    return combine_int_frac_ex2(x_rounded, x_frac_ex2)       # 2**n * 2**r


@triton.autotune(configs=configs, key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT"])
@triton.jit
def _attn_fwd_ws(sm_scale, M,  #
                 Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,  #
                 HEAD_DIM: tl.constexpr,  #
                 BLOCK_M: tl.constexpr,  #
                 BLOCK_N: tl.constexpr,  #
                 FP8_OUTPUT: tl.constexpr,  #
                 NUM_BUFFERS_KV: tl.constexpr,  #
                 NUM_BUFFERS_QK: tl.constexpr,  #
                 NUM_MMA_GROUPS: tl.constexpr,  #
                 ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    tl.static_assert(NUM_MMA_GROUPS == 2)
    tl.static_assert(NUM_BUFFERS_QK == 1)

    BLOCK_M_SPLIT: tl.constexpr = BLOCK_M // NUM_MMA_GROUPS

    # allocate SMEM buffers and barriers
    q_tiles = tlx.local_alloc((BLOCK_M_SPLIT, HEAD_DIM), tlx.dtype_of(desc_q), NUM_MMA_GROUPS)
    kv_tiles = tlx.local_alloc((BLOCK_N, HEAD_DIM), tlx.dtype_of(desc_k), NUM_BUFFERS_KV)

    q_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
    kv_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    kv_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)

    # allocate TMEM buffers and barriers
    qk_tiles = tlx.local_alloc((BLOCK_M_SPLIT, HEAD_DIM), tl.float32, NUM_MMA_GROUPS * NUM_BUFFERS_QK,
                               tlx.storage_kind.tmem)
    # Shared buffer for QK, P and Alpha, l, and m.
    # Alpha/l/m lives in the lower half of qk_buf, and P lives in the upper half.
    p_tiles = tlx.local_alloc((BLOCK_M_SPLIT, HEAD_DIM), tlx.dtype_of(desc_v), NUM_MMA_GROUPS * NUM_BUFFERS_QK * 2,
                              tlx.storage_kind.tmem, reuse=qk_tiles)
    alpha_tiles = tlx.local_alloc((BLOCK_M_SPLIT, 1), tl.float32, HEAD_DIM * NUM_MMA_GROUPS * NUM_BUFFERS_QK,
                                  tlx.storage_kind.tmem, reuse=qk_tiles)
    l_tiles = tlx.local_alloc((BLOCK_M_SPLIT, 1), tl.float32, HEAD_DIM * NUM_MMA_GROUPS * NUM_BUFFERS_QK,
                              tlx.storage_kind.tmem, reuse=qk_tiles)
    m_tiles = tlx.local_alloc((BLOCK_M_SPLIT, 1), tl.float32, HEAD_DIM * NUM_MMA_GROUPS * NUM_BUFFERS_QK,
                              tlx.storage_kind.tmem, reuse=qk_tiles)

    acc_tiles = tlx.local_alloc((BLOCK_M_SPLIT, HEAD_DIM), tl.float32, NUM_MMA_GROUPS * NUM_BUFFERS_QK,
                                tlx.storage_kind.tmem)

    qk_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS * NUM_BUFFERS_QK)
    p_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS * NUM_BUFFERS_QK)
    acc_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS * NUM_BUFFERS_QK)
    acc_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS * NUM_BUFFERS_QK)

    alpha_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS * NUM_BUFFERS_QK)
    alpha_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS * NUM_BUFFERS_QK)
    l_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)

    with tlx.async_tasks():
        # correction group
        with tlx.async_task("default"):
            # initialize offsets
            start_m, off_hz, lo, hi, qo_offset_y, kv_offset_y = _compute_offsets(H, N_CTX, BLOCK_M)
            accum_cnt = 0
            buf_idx = 0
            phase = 0

            for _ in tl.range(lo, hi, BLOCK_N):
                buf_idx, phase = _get_bufidx_phase(accum_cnt, NUM_BUFFERS_QK)
                for cid in tl.range(0, NUM_MMA_GROUPS, loop_unroll_factor=NUM_MMA_GROUPS):
                    buf_idx_2 = buf_idx + cid * NUM_BUFFERS_QK

                    # -- update output accumulator --
                    tlx.barrier_wait(alpha_fulls[buf_idx_2], phase)
                    # Use alpha[0] for cid=0, and alpha[HEAD_DIM * NUM_BUFFERS_QK] for cid=1
                    alpha_1 = tlx.local_load(alpha_tiles[cid * HEAD_DIM * NUM_BUFFERS_QK])
                    tlx.barrier_arrive(alpha_empties[buf_idx_2])

                    acc = tlx.local_load(acc_tiles[buf_idx_2])
                    acc = acc * alpha_1
                    tlx.local_store(acc_tiles[buf_idx_2], acc)
                    tlx.barrier_arrive(acc_fulls[buf_idx_2])
                accum_cnt += 1

            for cid in tl.range(0, NUM_MMA_GROUPS, loop_unroll_factor=NUM_MMA_GROUPS):
                # epilogue
                tlx.barrier_wait(l_fulls[cid], 0)
                # Use l[1]/l[1+HEAD_DIM * NUM_BUFFERS_QK] and m[2][2 + HEAD_DIM * NUM_BUFFERS_QK]
                # to disambigulate from alpha[0]/alpha[HEAD_DIM * NUM_BUFFERS_QK]
                l = tlx.local_load(l_tiles[cid * HEAD_DIM * NUM_BUFFERS_QK + 1])
                m = tlx.local_load(m_tiles[cid * HEAD_DIM * NUM_BUFFERS_QK + 2])
                m += tl.math.log2(l)
                offs_m = start_m * BLOCK_M + cid * BLOCK_M_SPLIT + tl.arange(0, BLOCK_M_SPLIT)
                m_ptrs = M + off_hz * N_CTX + offs_m
                tl.store(m_ptrs, tl.reshape(m, [BLOCK_M_SPLIT]))

                tlx.barrier_wait(acc_empties[cid], 0)
                acc = tlx.local_load(acc_tiles[cid])
                acc = acc / l
                qo_offset_y_split = qo_offset_y + cid * BLOCK_M_SPLIT
                desc_o.store([qo_offset_y_split, 0], acc.to(tlx.dtype_of(desc_o)))

        # softmax groups
        with tlx.async_task(num_warps=4, registers=152, replicate=NUM_MMA_GROUPS):
            # initialize offsets
            start_m, off_hz, lo, hi, qo_offset_y, kv_offset_y = _compute_offsets(H, N_CTX, BLOCK_M)
            # initialize pointer to m and l
            m_i = tl.zeros([BLOCK_M_SPLIT], dtype=tl.float32) - float("inf")
            l_i = tl.zeros([BLOCK_M_SPLIT], dtype=tl.float32) + 1.0
            acc = tl.zeros([BLOCK_M_SPLIT, HEAD_DIM], dtype=tl.float32)
            qk_scale = sm_scale
            qk_scale *= 1.44269504  # 1/log(2)

            accum_cnt_qk = 0
            cid = tlx.async_task_replica_id()
            for _ in tl.range(lo, hi, BLOCK_N):
                qk_bufIdx, qk_phase = _get_bufidx_phase(accum_cnt_qk, NUM_BUFFERS_QK)
                qk_bufIdx += cid * NUM_BUFFERS_QK

                tlx.barrier_wait(qk_fulls[qk_bufIdx], qk_phase)
                qk = tlx.local_load(qk_tiles[qk_bufIdx])

                # compute m_i, p in registers
                m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)

                # -- compute correction factor
                alpha = tl.math.exp2(m_i - m_ij)
                tlx.barrier_wait(alpha_empties[qk_bufIdx], qk_phase ^ 1)
                # Use alpha[0] for cid=0, and alpha[HEAD_DIM * NUM_BUFFERS_QK] for cid=1
                tlx.local_store(alpha_tiles[cid * HEAD_DIM * NUM_BUFFERS_QK], alpha[:, None])
                tlx.barrier_arrive(alpha_fulls[qk_bufIdx])

                qk = qk * qk_scale - m_ij[:, None]

                BM: tl.constexpr = qk.shape[0]
                BN: tl.constexpr = qk.shape[1]
                qk_0, qk_1 = qk.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()

                p_0 = ex2_emulation(qk_0)
                p_1 = tl.math.exp2(qk_1)
                p = tl.join(p_0, p_1).permute(0, 2, 1).reshape([BM, BN])


                l_ij = tl.sum(p, 1)
                p = p.to(tlx.dtype_of(desc_v))

                # prepare p for the v dot
                # Use p[1] for cid=0, and p[3] for cid=1
                p_bufIdx = 1 + cid * NUM_MMA_GROUPS * NUM_BUFFERS_QK
                tlx.local_store(p_tiles[p_bufIdx], p)
                tlx.barrier_arrive(p_fulls[qk_bufIdx])

                l_i = l_i * alpha + l_ij
                m_i = m_ij
                accum_cnt_qk += 1

            # prepare l_i for the epilog
            # Use l[1]/l[1+HEAD_DIM * NUM_BUFFERS_QK] and m[2][2 + HEAD_DIM * NUM_BUFFERS_QK]
            # to disambigulate from alpha[0]/alpha[HEAD_DIM * NUM_BUFFERS_QK]
            tlx.local_store(l_tiles[cid * HEAD_DIM * NUM_BUFFERS_QK + 1], l_i[:, None])
            tlx.local_store(m_tiles[cid * HEAD_DIM * NUM_BUFFERS_QK + 2], m_i[:, None])
            tlx.barrier_arrive(l_fulls[cid])

        # mma group
        with tlx.async_task(num_warps=1, registers=24):
            _, _, lo, hi, _, _ = _compute_offsets(H, N_CTX, BLOCK_M)

            # loop over k, v and update accumulator
            accum_cnt_kv = 0
            accum_cnt_qk = 0
            k_bufIdx, k_phase = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
            v_bufIdx, v_phase = _get_bufidx_phase(accum_cnt_kv + 1, NUM_BUFFERS_KV)

            # -- compute q @ k ----
            # wait for the K buffer to be populated by the producer
            tlx.barrier_wait(kv_fulls[k_bufIdx], k_phase)
            tlx.barrier_wait(q_fulls[0], 0)

            k_tile = tlx.local_trans(kv_tiles[k_bufIdx])
            _, qk_phase = _get_bufidx_phase(accum_cnt_qk, NUM_BUFFERS_QK)
            tlx.async_dot(
                q_tiles[0],
                k_tile,
                qk_tiles[0],
                use_acc=False,
                mBarriers=[qk_fulls[0]],
            )

            tlx.barrier_wait(q_fulls[1], 0)
            tlx.async_dot(
                q_tiles[1],
                k_tile,
                qk_tiles[1],
                use_acc=False,
                mBarriers=[qk_fulls[1], kv_empties[k_bufIdx]],
            )

            # -- compute p0 @ v ----
            # wait for the V buffer to be populated by the producer
            tlx.barrier_wait(kv_fulls[v_bufIdx], v_phase)
            tlx.barrier_wait(p_fulls[0], qk_phase)
            tlx.barrier_wait(acc_fulls[0], qk_phase)
            # As p shares the second half of the qk buffer, use p[2]/p[3] instead of p[0]/p[1]
            tlx.async_dot(
                p_tiles[1],
                kv_tiles[v_bufIdx],
                acc_tiles[0],
                use_acc=False,
            )

            acc1_init = False

            for i in tl.range(lo + BLOCK_N, hi, BLOCK_N):
                v_bufIdx_prev = v_bufIdx
                qk_phase_prev = qk_phase

                accum_cnt_qk += 1
                accum_cnt_kv += 2
                k_bufIdx, k_phase = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                v_bufIdx, v_phase = _get_bufidx_phase(accum_cnt_kv + 1, NUM_BUFFERS_KV)

                # -- compute q0 @ k ----
                _, qk_phase = _get_bufidx_phase(accum_cnt_qk, NUM_BUFFERS_QK)
                tlx.barrier_wait(kv_fulls[k_bufIdx], k_phase)
                k_tile = tlx.local_trans(kv_tiles[k_bufIdx])
                tlx.async_dot(
                    q_tiles[0],
                    k_tile,
                    qk_tiles[0],
                    use_acc=False,
                    mBarriers=[qk_fulls[0]],
                )

                # -- compute p1 @ v from the previous iteration----
                tlx.barrier_wait(p_fulls[1], qk_phase_prev)
                tlx.barrier_wait(acc_fulls[1], qk_phase_prev)
                tlx.async_dot(
                    p_tiles[3],
                    kv_tiles[v_bufIdx_prev],
                    acc_tiles[1],
                    use_acc=acc1_init,
                    mBarriers=[kv_empties[v_bufIdx_prev]],
                )

                acc1_init = True

                # -- compute q1 @ k ----
                tlx.async_dot(
                    q_tiles[1],
                    k_tile,
                    qk_tiles[1],
                    use_acc=False,
                    mBarriers=[qk_fulls[1], kv_empties[k_bufIdx]],
                )

                # -- compute p0 @ v ----
                # wait for the V buffer to be populated by the producer
                tlx.barrier_wait(kv_fulls[v_bufIdx], v_phase)
                tlx.barrier_wait(p_fulls[0], qk_phase)
                tlx.barrier_wait(acc_fulls[0], qk_phase)
                tlx.async_dot(
                    p_tiles[1],
                    kv_tiles[v_bufIdx],
                    acc_tiles[0],
                    use_acc=True,
                )

            tlx.tcgen05_commit(acc_empties[0])

            # -- compute p1 @ v ----
            tlx.barrier_wait(p_fulls[1], qk_phase)
            tlx.barrier_wait(acc_fulls[1], qk_phase)
            tlx.async_dot(
                p_tiles[3],
                kv_tiles[v_bufIdx],
                acc_tiles[1],
                use_acc=acc1_init,
                mBarriers=[acc_empties[1], kv_empties[v_bufIdx]],
            )

        # load
        with tlx.async_task(num_warps=1, registers=24):
            # initialize offsets
            start_m, off_hz, lo, hi, qo_offset_y, kv_offset_y = _compute_offsets(H, N_CTX, BLOCK_M)

            # load q0
            tlx.barrier_expect_bytes(q_fulls[0], 2 * BLOCK_M_SPLIT * HEAD_DIM)  # float16
            qo_offset_y_split = qo_offset_y
            tlx.async_descriptor_load(desc_q, q_tiles[0], [qo_offset_y_split, 0], q_fulls[0])

            # loop over loading k, v
            accum_cnt_kv = 0
            k_bufIdx, k_phase = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
            # wait for the K buffer to be released by the consumer
            k_empty = tlx.local_view(kv_empties, k_bufIdx)
            tlx.barrier_wait(k_empty, k_phase ^ 1)

            # load K
            k_full = tlx.local_view(kv_fulls, k_bufIdx)
            k_tile = tlx.local_view(kv_tiles, k_bufIdx)
            tlx.barrier_expect_bytes(k_full, 2 * BLOCK_N * HEAD_DIM)  # float16
            tlx.async_descriptor_load(desc_k, k_tile, [kv_offset_y, 0], k_full)

            # load q1
            tlx.barrier_expect_bytes(q_fulls[1], 2 * BLOCK_M_SPLIT * HEAD_DIM)  # float16
            qo_offset_y_split = qo_offset_y + BLOCK_M_SPLIT
            tlx.async_descriptor_load(desc_q, q_tiles[1], [qo_offset_y_split, 0], q_fulls[1])

            v_bufIdx, v_phase = _get_bufidx_phase(accum_cnt_kv + 1, NUM_BUFFERS_KV)
            # wait for the V buffer to be released by the consumer
            v_empty = tlx.local_view(kv_empties, v_bufIdx)
            tlx.barrier_wait(v_empty, v_phase ^ 1)
            # load V
            v_full = tlx.local_view(kv_fulls, v_bufIdx)
            v_tile = tlx.local_view(kv_tiles, v_bufIdx)
            tlx.barrier_expect_bytes(v_full, 2 * BLOCK_N * HEAD_DIM)  # float16
            tlx.async_descriptor_load(desc_v, v_tile, [kv_offset_y, 0], v_full)

            kv_offset_y += BLOCK_N
            accum_cnt_kv += 2

            for _ in tl.range(lo + BLOCK_N, hi, BLOCK_N):
                k_bufIdx, k_phase = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                # wait for the K buffer to be released by the consumer
                k_empty = tlx.local_view(kv_empties, k_bufIdx)
                tlx.barrier_wait(k_empty, k_phase ^ 1)
                # load K
                k_full = tlx.local_view(kv_fulls, k_bufIdx)
                k_tile = tlx.local_view(kv_tiles, k_bufIdx)
                tlx.barrier_expect_bytes(k_full, 2 * BLOCK_N * HEAD_DIM)  # float16
                tlx.async_descriptor_load(desc_k, k_tile, [kv_offset_y, 0], k_full)

                v_bufIdx, v_phase = _get_bufidx_phase(accum_cnt_kv + 1, NUM_BUFFERS_KV)
                # wait for the V buffer to be released by the consumer
                v_empty = tlx.local_view(kv_empties, v_bufIdx)
                tlx.barrier_wait(v_empty, v_phase ^ 1)
                # load V
                v_full = tlx.local_view(kv_fulls, v_bufIdx)
                v_tile = tlx.local_view(kv_tiles, v_bufIdx)
                tlx.barrier_expect_bytes(v_full, 2 * BLOCK_N * HEAD_DIM)  # float16
                tlx.async_descriptor_load(desc_v, v_tile, [kv_offset_y, 0], v_full)

                kv_offset_y += BLOCK_N
                accum_cnt_kv += 2


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, sm_scale):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        extra_kern_args = {}

        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        # Note that on Hopper we cannot perform a FP8 dot with a non-transposed second tensor
        y_dim = q.shape[0] * q.shape[1] * q.shape[2]

        dummy_block = [1, 1]
        desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
        if q.dtype == torch.float8_e5m2:
            desc_v = TensorDescriptor(v, shape=[HEAD_DIM_K, y_dim], strides=[q.shape[2], 1], block_shape=dummy_block)
        else:
            desc_v = TensorDescriptor(v, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
        desc_k = TensorDescriptor(k, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
        desc_o = TensorDescriptor(o, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        def grid(META):
            return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)

        ctx.grid = grid
        _attn_fwd_ws[grid](
            sm_scale, M,  #
            q.shape[0], q.shape[1],  #
            desc_q, desc_k, desc_v, desc_o,  #
            N_CTX=q.shape[2],  #
            HEAD_DIM=HEAD_DIM_K,  #
            FP8_OUTPUT=q.dtype == torch.float8_e5m2,  #
            **extra_kern_args)

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        return o


attention = _attention.apply


@pytest.mark.skipif(
    not is_blackwell(),
    reason="Requires Hopper GPU",
)
@pytest.mark.parametrize("Z", [8])
@pytest.mark.parametrize("H", [16])
@pytest.mark.parametrize("N_CTX", [1024])
@pytest.mark.parametrize("HEAD_DIM", [128])
@pytest.mark.parametrize("mode", ["fwd"])
@pytest.mark.parametrize("provider", ["triton-fp16"])
def test_op(Z, H, N_CTX, HEAD_DIM, mode, provider, dtype=torch.float16):
    if mode == "bwd":
        pytest.skip("Backward pass not supported.")
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
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    p = torch.softmax(p.float(), dim=-1)
    p = p.to(ref_dtype)
    # p = torch.exp(p)
    ref_out = torch.matmul(p, v).half()
    # triton implementation
    if mode == "fwd" and "fp8" in provider:
        q = q.to(torch.float8_e5m2)
        k = k.to(torch.float8_e5m2)
        v = v.permute(0, 1, 3, 2).contiguous()
        v = v.permute(0, 1, 3, 2)
        v = v.to(torch.float8_e5m2)
    tri_out = attention(q, k, v, sm_scale).half()
    if mode == "fwd":
        atol = 3 if "fp8" in provider else 1e-2
        torch.testing.assert_close(tri_out, ref_out, atol=atol, rtol=0)
        return


try:
    from flash_attn.flash_attn_interface import \
        flash_attn_qkvpacked_func as flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

TORCH_HAS_FP8 = False
BATCH, N_HEADS, HEAD_DIM = 4, 32, 128
# vary seq length for fixed head and batch=4
configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["N_CTX"],
        x_vals=[2**i for i in range(10, 15)],
        line_arg="provider",
        line_vals=["triton-fp16"] + (["flash"] if HAS_FLASH else []),
        line_names=["Triton [FP16]"] + (["Flash-2"] if HAS_FLASH else []),
        styles=[("red", "-"), ("blue", "-"), ("green", "-")],
        ylabel="TFLOPS",
        plot_name=f"fused-attention-ws-pipelined-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}",
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
        fn = lambda: attention(q, k, v, sm_scale)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)

    if provider == "flash":
        qkv = torch.randn((BATCH, N_CTX, 3, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
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
