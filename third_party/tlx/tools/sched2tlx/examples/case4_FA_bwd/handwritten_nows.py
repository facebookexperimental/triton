"""case4 FA backward — no-WS 5-MMA reference kernel (the modulo-schedule source).

Single fused inner loop computing the full FA backward for one K/V tile,
mirroring `_attn_bwd_dkdv_inner` from tritonbench. Per Q-tile iteration:

    qkT = K @ Q^T                 # MMA1
    pT  = exp2(qkT - m)           # CUDA + SFU   (base-2 softmax, no sm_scale)
    dpT = V @ dO^T                # MMA2
    dV += pT @ dO                 # MMA3
    dsT = pT * (dpT - D)          # CUDA
    dQ += dsT^T @ K               # MMA4  (atomic_add: many K-tiles -> same Q row)
    dK += dsT @ Q                 # MMA5

Convention (matches the kernel's exp2, NO sm_scale): the caller folds sm_scale
into Q, builds the reference forward with the SAME base-2 softmax, and passes
M = log2(sum exp2(S))  and  D = rowsum(dO * O). Then kernel dQ/dK/dV must match
torch.autograd of that forward.

Used two ways:
  - `python3 run_handwritten_nows.py`   -> GPU correctness vs torch.autograd
  - AOT dump (see C4-M2) -> fa_bwd_nows_pre_modulo.ttgir
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def fa_bwd_dkdv_5mma(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    WARP_SPEC: tl.constexpr = True,  # True for the modulo pre-dump; False for GPU run
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
        Q + base, [N_CTX, HEAD_DIM], [stride_m, 1], [BLOCK_M, HEAD_DIM]
    )
    do_desc = tl.make_tensor_descriptor(
        dO + base, [N_CTX, HEAD_DIM], [stride_m, 1], [BLOCK_M, HEAD_DIM]
    )
    dq_desc = tl.make_tensor_descriptor(
        dQ + base, [N_CTX, HEAD_DIM], [stride_m, 1], [BLOCK_M, HEAD_DIM]
    )

    k = k_desc.load([start_n, 0])
    v = v_desc.load([start_n, 0])

    dk_acc = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    m_base = off_hz * N_CTX

    for start_m in tl.range(0, N_CTX, BLOCK_M, warp_specialize=WARP_SPEC):
        q = q_desc.load([start_m, 0])
        do = do_desc.load([start_m, 0])

        offs_m = start_m + tl.arange(0, BLOCK_M)
        m = tl.load(M + m_base + offs_m)
        Di = tl.load(D + m_base + offs_m)

        qkT = tl.dot(k, tl.trans(q))  # MMA1 [N, M]
        # exp2(qk * log2e) == exp(qk): folds the base-2/base-e factor into the
        # exponent so the (no-ln2) `dsT = pT*(dpT-D)` backward formula is exact.
        pT = tl.math.exp2(qkT * 1.4426950408889634 - m[None, :])
        ppT = pT.to(tl.float16)
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)  # MMA2 [N, M]
        dv_acc += tl.dot(ppT, do)  # MMA3 [N, D]
        dsT = (pT * (dpT - Di[None, :])).to(tl.float16)
        dq_partial = tl.dot(tl.trans(dsT), k)  # MMA4 [M, D]
        dq_desc.atomic_add([start_m, 0], dq_partial.to(tl.float16))
        dk_acc += tl.dot(dsT, q)  # MMA5 [N, D]

    dk_desc = tl.make_tensor_descriptor(
        dK + base, [N_CTX, HEAD_DIM], [stride_n, 1], [BLOCK_N, HEAD_DIM]
    )
    dv_desc = tl.make_tensor_descriptor(
        dV + base, [N_CTX, HEAD_DIM], [stride_n, 1], [BLOCK_N, HEAD_DIM]
    )
    dk_desc.store([start_n, 0], dk_acc.to(tl.float16))
    dv_desc.store([start_n, 0], dv_acc.to(tl.float16))
