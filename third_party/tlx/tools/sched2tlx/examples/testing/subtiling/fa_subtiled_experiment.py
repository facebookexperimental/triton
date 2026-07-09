"""Route-A sub-tiling experiment (SubTilingDesign.md): source-level
two-sub-tile FA fwd, BLOCK_M=256 split as 2x128 (TMEM-friendly, the same
design point as the 1197-TFLOPS ping-pong tutorial), no WS in the source —
the modulo solver + joint partitioner decide everything.

Phase 1 of this script: compile the kernel with the solver env to dump the
pre_modulo schedule graph. Phase 2 (separate invocations): emit via
sched2tlx, correctness, bench.
"""

import os
import sys

import torch
import triton
import triton.language as tl


@triton.jit
def fa_fwd_kernel_nows_subtiled(Q, K, V, Out, M_lse, sm_scale, H, N_CTX,
                                BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
                                SUB_M: tl.constexpr):
    # One program handles BLOCK_M = 2*SUB_M query rows as TWO independent
    # sub-tiles sharing each iteration's K/V tiles. Written exactly like
    # case3's nows kernel, duplicated per sub-tile.
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    qo_offset = pid_bh * N_CTX
    qo_base = pid_m * (2 * SUB_M)

    q_desc = tl.make_tensor_descriptor(Q, [H * N_CTX, HEAD_DIM], [HEAD_DIM, 1],
                                       [SUB_M, HEAD_DIM])
    k_desc = tl.make_tensor_descriptor(K, [H * N_CTX, HEAD_DIM], [HEAD_DIM, 1],
                                       [BLOCK_N, HEAD_DIM])
    v_desc = tl.make_tensor_descriptor(V, [H * N_CTX, HEAD_DIM], [HEAD_DIM, 1],
                                       [BLOCK_N, HEAD_DIM])
    o_desc = tl.make_tensor_descriptor(Out, [H * N_CTX, HEAD_DIM], [HEAD_DIM, 1],
                                       [SUB_M, HEAD_DIM])

    q0 = q_desc.load([qo_offset + qo_base, 0])
    q1 = q_desc.load([qo_offset + qo_base + SUB_M, 0])

    qk_scale = sm_scale * 1.44269504

    m_i0 = tl.full([SUB_M], float("-inf"), tl.float32)
    m_i1 = tl.full([SUB_M], float("-inf"), tl.float32)
    l_i0 = tl.full([SUB_M], 1.0, tl.float32)
    l_i1 = tl.full([SUB_M], 1.0, tl.float32)
    acc0 = tl.zeros([SUB_M, HEAD_DIM], tl.float32)
    acc1 = tl.zeros([SUB_M, HEAD_DIM], tl.float32)

    for kv in range(0, N_CTX // BLOCK_N):
        koff = qo_offset + kv * BLOCK_N
        k = k_desc.load([koff, 0])
        v = v_desc.load([koff, 0])
        kt = tl.trans(k)

        # sub-tile 0
        qk0 = tl.dot(q0, kt)
        m_new0 = tl.maximum(m_i0, tl.max(qk0, 1) * qk_scale)
        alpha0 = tl.math.exp2(m_i0 - m_new0)
        p0 = tl.math.exp2(qk0 * qk_scale - m_new0[:, None])
        l_i0 = l_i0 * alpha0 + tl.sum(p0, 1)
        acc0 = acc0 * alpha0[:, None] + tl.dot(p0.to(tl.bfloat16), v)
        m_i0 = m_new0

        # sub-tile 1
        qk1 = tl.dot(q1, kt)
        m_new1 = tl.maximum(m_i1, tl.max(qk1, 1) * qk_scale)
        alpha1 = tl.math.exp2(m_i1 - m_new1)
        p1 = tl.math.exp2(qk1 * qk_scale - m_new1[:, None])
        l_i1 = l_i1 * alpha1 + tl.sum(p1, 1)
        acc1 = acc1 * alpha1[:, None] + tl.dot(p1.to(tl.bfloat16), v)
        m_i1 = m_new1

    acc0 = acc0 / l_i0[:, None]
    acc1 = acc1 / l_i1[:, None]
    o_desc.store([qo_offset + qo_base, 0], acc0.to(tl.bfloat16))
    o_desc.store([qo_offset + qo_base + SUB_M, 0], acc1.to(tl.bfloat16))

    offs = tl.arange(0, SUB_M)
    tl.store(M_lse + qo_offset + qo_base + offs, m_i0 + tl.math.log2(l_i0))
    tl.store(M_lse + qo_offset + qo_base + SUB_M + offs, m_i1 + tl.math.log2(l_i1))


def run(shape=(1, 32, 8192), bench=False):
    Z, H, N_CTX = shape
    D, SUB_M, BLOCK_N = 128, 128, 64
    torch.manual_seed(0)
    q = torch.randn(Z, H, N_CTX, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(Z, H, N_CTX, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(Z, H, N_CTX, D, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(q)
    m_lse = torch.empty(Z * H * N_CTX, device="cuda", dtype=torch.float32)
    sm = 1.0 / (D ** 0.5)
    triton.set_allocator(lambda size, align, _: torch.empty(size, dtype=torch.int8, device="cuda"))

    grid = (triton.cdiv(N_CTX, 2 * SUB_M), Z * H)
    call = lambda: fa_fwd_kernel_nows_subtiled[grid](
        q.view(-1, D), k.view(-1, D), v.view(-1, D), out.view(-1, D), m_lse,
        sm, Z * H, N_CTX, BLOCK_N=BLOCK_N, HEAD_DIM=D, SUB_M=SUB_M,
        num_warps=4)
    call()
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=sm)
    rel = (out.float() - ref.float()).abs().max().item() / ref.float().abs().max().item()
    nan = torch.isnan(out).sum().item()
    print(f"correctness rel={rel:.2e} nan={nan} {'PASS' if rel < 1e-2 and nan == 0 else 'FAIL'}")
    if bench:
        ms = triton.testing.do_bench(call, warmup=50, rep=200, quantiles=[0.5])
        tf = 4 * Z * H * N_CTX * N_CTX * D / (ms * 1e-3) / 1e12
        print(f"{shape}: {ms:.3f} ms = {tf:.1f} TFLOPS")


if __name__ == "__main__":
    run(bench="--bench" in sys.argv)
