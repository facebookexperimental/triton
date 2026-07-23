"""Sub-tiled fp16 non-causal FA forward, no warp specialization — dump source
for the case3_FA_fp16_subtiled fixture.

The M tile (2*SUB_M rows) is split into two sub-tiles with fully duplicated
online-softmax chains (two Q descriptor loads at +0/+SUB_M, separate
m/l/acc accumulators) sharing the K/V loads.  After TTGIR lowering the DDG
then contains 4 tc_gen5_mma nodes and two disjoint softmax chains joined only
at the shared K/V nodes — the sub-tile granularity the paper's system
schedules at (its Fig-9 graph shape), and the control for the
"no sub-tiling => UNSAT" ablation.

Per-op structure of each chain matches case3_FA_fp16's fa_fwd_kernel_nows.
SUB_M=64 is the paper-faithful split of BLOCK_M=128; SUB_M=128 (BLOCK_M=256)
is the fallback size.  Dump route: same as case3_FA_fp16.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def fa_fwd_kernel_nows_subtiled(Q, K, V, Out, M_lse, sm_scale, H, N_CTX,
                                SUB_M: tl.constexpr, BLOCK_N: tl.constexpr,
                                HEAD_DIM: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    kv_base = pid_bh * N_CTX
    qo_offset = kv_base + pid_m * (2 * SUB_M)
    rows = H * N_CTX  # H is really Z*H; layout flattened to [(Z*H)*N_CTX, HEAD_DIM]
    qo_desc = tl.make_tensor_descriptor(Q, [rows, HEAD_DIM], [HEAD_DIM, 1],
                                        [SUB_M, HEAD_DIM])
    k_desc = tl.make_tensor_descriptor(K, [rows, HEAD_DIM], [HEAD_DIM, 1],
                                       [BLOCK_N, HEAD_DIM])
    v_desc = tl.make_tensor_descriptor(V, [rows, HEAD_DIM], [HEAD_DIM, 1],
                                       [BLOCK_N, HEAD_DIM])
    out_desc = tl.make_tensor_descriptor(Out, [rows, HEAD_DIM], [HEAD_DIM, 1],
                                         [SUB_M, HEAD_DIM])
    q0 = qo_desc.load([qo_offset, 0])
    q1 = qo_desc.load([qo_offset + SUB_M, 0])
    m_i0 = tl.full([SUB_M], float("-inf"), tl.float32)
    m_i1 = tl.full([SUB_M], float("-inf"), tl.float32)
    l_i0 = tl.full([SUB_M], 1.0, tl.float32)
    l_i1 = tl.full([SUB_M], 1.0, tl.float32)
    acc0 = tl.zeros([SUB_M, HEAD_DIM], tl.float32)
    acc1 = tl.zeros([SUB_M, HEAD_DIM], tl.float32)
    qk_scale = sm_scale * 1.44269504
    for kv in range(N_CTX // BLOCK_N):
        koff = kv_base + kv * BLOCK_N
        k = k_desc.load([koff, 0])
        v = v_desc.load([koff, 0])
        kt = tl.trans(k)
        qk0 = tl.dot(q0, kt)
        m_ij0 = tl.maximum(m_i0, tl.max(qk0, 1) * qk_scale)
        alpha0 = tl.math.exp2(m_i0 - m_ij0)
        p0 = tl.math.exp2(qk0 * qk_scale - m_ij0[:, None])
        l_ij0 = tl.sum(p0, 1)
        acc0 = tl.dot(p0.to(tl.float16), v, acc0 * alpha0[:, None])
        l_i0 = l_i0 * alpha0 + l_ij0
        m_i0 = m_ij0
        qk1 = tl.dot(q1, kt)
        m_ij1 = tl.maximum(m_i1, tl.max(qk1, 1) * qk_scale)
        alpha1 = tl.math.exp2(m_i1 - m_ij1)
        p1 = tl.math.exp2(qk1 * qk_scale - m_ij1[:, None])
        l_ij1 = tl.sum(p1, 1)
        acc1 = tl.dot(p1.to(tl.float16), v, acc1 * alpha1[:, None])
        l_i1 = l_i1 * alpha1 + l_ij1
        m_i1 = m_ij1
    acc0 = acc0 / l_i0[:, None]
    acc1 = acc1 / l_i1[:, None]
    out_desc.store([qo_offset, 0], acc0.to(tl.float16))
    out_desc.store([qo_offset + SUB_M, 0], acc1.to(tl.float16))
    offs = tl.arange(0, SUB_M)
    tl.store(M_lse + qo_offset + offs, m_i0 + tl.math.log2(l_i0))
    tl.store(M_lse + qo_offset + SUB_M + offs, m_i1 + tl.math.log2(l_i1))


def run(Z, H, N_CTX, HEAD_DIM=128, SUB_M=64, BLOCK_N=64, check=True):
    torch.manual_seed(0)
    triton.set_allocator(
        lambda size, align, stream: torch.empty(size, device="cuda",
                                                dtype=torch.int8))
    q, k, v = (torch.randn(Z, H, N_CTX, HEAD_DIM, device="cuda",
                           dtype=torch.float16) for _ in range(3))
    qf, kf, vf = (t.contiguous().view(-1, HEAD_DIM) for t in (q, k, v))
    of = torch.empty_like(qf)
    m_lse = torch.empty(Z * H, N_CTX, device="cuda", dtype=torch.float32)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    grid = (triton.cdiv(N_CTX, 2 * SUB_M), Z * H)
    fa_fwd_kernel_nows_subtiled[grid](qf, kf, vf, of, m_lse, sm_scale, Z * H,
                                      N_CTX, SUB_M=SUB_M, BLOCK_N=BLOCK_N,
                                      HEAD_DIM=HEAD_DIM, num_warps=4,
                                      num_ctas=1, num_stages=2,
                                      maxRegAutoWS=152)
    if not check:
        return None
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v,
                                                           scale=sm_scale)
    out = of.view(Z, H, N_CTX, HEAD_DIM)
    return (out - ref).abs().max().item() / ref.abs().max().item()


if __name__ == "__main__":
    import sys
    sub_m = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    rel = run(Z=1, H=4, N_CTX=2048, SUB_M=sub_m)
    status = "PASS" if rel < 1e-2 else "FAIL"
    print(f"fa_fwd_kernel_nows_subtiled fp16 SUB_M={sub_m} (1,4,2048,128): "
          f"rel={rel:.2e} {status}")
