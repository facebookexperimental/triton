"""Sub-tiled FA backward source for the paper joint-solver fixture.

Each outer-loop iteration shares one K/V tile across two independent Q/dO
sub-tiles.  Each sub-tile retains the full exp2 plus five-MMA backward chain,
so the lowered loop has two exp2 operations and ten tc_gen5_mma operations.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def fa_bwd_dkdv_subtiled(
    Q,
    K,
    V,
    dO,
    dQ,
    dK,
    dV,
    M,
    D,
    stride_m,
    stride_n,
    N_CTX,
    SUB_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    WARP_SPEC: tl.constexpr = True,
):
    pid_n = tl.program_id(0)
    off_hz = tl.program_id(1)

    base = off_hz * N_CTX * HEAD_DIM
    start_n = pid_n * BLOCK_N

    k_desc = tl.make_tensor_descriptor(
        K + base, [N_CTX, HEAD_DIM], [stride_n, 1], [BLOCK_N, HEAD_DIM]
    )
    v_desc = tl.make_tensor_descriptor(
        V + base, [N_CTX, HEAD_DIM], [stride_n, 1], [BLOCK_N, HEAD_DIM]
    )
    q_desc = tl.make_tensor_descriptor(
        Q + base, [N_CTX, HEAD_DIM], [stride_m, 1], [SUB_M, HEAD_DIM]
    )
    do_desc = tl.make_tensor_descriptor(
        dO + base, [N_CTX, HEAD_DIM], [stride_m, 1], [SUB_M, HEAD_DIM]
    )
    dq_desc = tl.make_tensor_descriptor(
        dQ + base, [N_CTX, HEAD_DIM], [stride_m, 1], [SUB_M, HEAD_DIM]
    )

    k = k_desc.load([start_n, 0])
    v = v_desc.load([start_n, 0])
    dk_acc = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    m_base = off_hz * N_CTX

    for start_m in tl.range(
        0, N_CTX, 2 * SUB_M, warp_specialize=WARP_SPEC
    ):
        q0 = q_desc.load([start_m, 0])
        do0 = do_desc.load([start_m, 0])
        offs_m0 = start_m + tl.arange(0, SUB_M)
        m0 = tl.load(M + m_base + offs_m0)
        Di0 = tl.load(D + m_base + offs_m0)

        qkT0 = tl.dot(k, tl.trans(q0))
        pT0 = tl.math.exp2(qkT0 * 1.4426950408889634 - m0[None, :])
        ppT0 = pT0.to(tl.float16)
        dpT0 = tl.dot(v, tl.trans(do0)).to(tl.float32)
        dv_acc += tl.dot(ppT0, do0)
        dsT0 = (pT0 * (dpT0 - Di0[None, :])).to(tl.float16)
        dq_partial0 = tl.dot(tl.trans(dsT0), k)
        dq_desc.atomic_add([start_m, 0], dq_partial0.to(tl.float16))
        dk_acc += tl.dot(dsT0, q0)

        start_m1 = start_m + SUB_M
        q1 = q_desc.load([start_m1, 0])
        do1 = do_desc.load([start_m1, 0])
        offs_m1 = start_m1 + tl.arange(0, SUB_M)
        m1 = tl.load(M + m_base + offs_m1)
        Di1 = tl.load(D + m_base + offs_m1)

        qkT1 = tl.dot(k, tl.trans(q1))
        pT1 = tl.math.exp2(qkT1 * 1.4426950408889634 - m1[None, :])
        ppT1 = pT1.to(tl.float16)
        dpT1 = tl.dot(v, tl.trans(do1)).to(tl.float32)
        dv_acc += tl.dot(ppT1, do1)
        dsT1 = (pT1 * (dpT1 - Di1[None, :])).to(tl.float16)
        dq_partial1 = tl.dot(tl.trans(dsT1), k)
        dq_desc.atomic_add([start_m1, 0], dq_partial1.to(tl.float16))
        dk_acc += tl.dot(dsT1, q1)

    dk_desc = tl.make_tensor_descriptor(
        dK + base, [N_CTX, HEAD_DIM], [stride_n, 1], [BLOCK_N, HEAD_DIM]
    )
    dv_desc = tl.make_tensor_descriptor(
        dV + base, [N_CTX, HEAD_DIM], [stride_n, 1], [BLOCK_N, HEAD_DIM]
    )
    dk_desc.store([start_n, 0], dk_acc.to(tl.float16))
    dv_desc.store([start_n, 0], dv_acc.to(tl.float16))
