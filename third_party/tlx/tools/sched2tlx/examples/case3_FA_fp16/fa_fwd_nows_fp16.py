"""fp16 non-causal FA forward, no warp specialization — dump source for the
case3_FA_fp16 fixture (paper config: fp16, BATCH=4, HEADS=32, HEAD_DIM=128).

Same op-for-op structure as case3_FA's bf16 fa_fwd_kernel_nows (reconstructed
from fa_fwd_nows_pre_modulo.ttgir loc metadata); only the two element-type
casts and the launch tensor dtype differ.

Dump the pre-modulo TTGIR with (single GPU compile+launch, small shape):

    TRITON_ALWAYS_COMPILE=1 TRITON_USE_MODULO_SCHEDULE=1 \
    MLIR_ENABLE_DUMP=fa_fwd_kernel_nows MLIR_DUMP_PATH=$PWD/dump_fp16.mlir \
    env -u LD_LIBRARY_PATH <venv-python> fa_fwd_nows_fp16.py

then extract the module printed immediately before the modulo-schedule pass.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def fa_fwd_kernel_nows(Q, K, V, Out, M_lse, sm_scale, H, N_CTX,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                       HEAD_DIM: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    kv_base = pid_bh * N_CTX
    qo_offset = kv_base + pid_m * BLOCK_M
    rows = H * N_CTX  # H is really Z*H; layout flattened to [(Z*H)*N_CTX, HEAD_DIM]
    qo_desc = tl.make_tensor_descriptor(Q, [rows, HEAD_DIM], [HEAD_DIM, 1],
                                        [BLOCK_M, HEAD_DIM])
    k_desc = tl.make_tensor_descriptor(K, [rows, HEAD_DIM], [HEAD_DIM, 1],
                                       [BLOCK_N, HEAD_DIM])
    v_desc = tl.make_tensor_descriptor(V, [rows, HEAD_DIM], [HEAD_DIM, 1],
                                       [BLOCK_N, HEAD_DIM])
    out_desc = tl.make_tensor_descriptor(Out, [rows, HEAD_DIM], [HEAD_DIM, 1],
                                         [BLOCK_M, HEAD_DIM])
    q = qo_desc.load([qo_offset, 0])
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    qk_scale = sm_scale * 1.44269504  # log2(e), folded once, applied post-dot
    for kv in range(N_CTX // BLOCK_N):
        koff = kv_base + kv * BLOCK_N
        k = k_desc.load([koff, 0])
        v = v_desc.load([koff, 0])
        qk = tl.dot(q, tl.trans(k))
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(qk * qk_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        acc = tl.dot(p.to(tl.float16), v, acc * alpha[:, None])
        l_i = l_i * alpha + l_ij
        m_i = m_ij
    acc = acc / l_i[:, None]
    out_desc.store([qo_offset, 0], acc.to(tl.float16))
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(M_lse + kv_base + offs_m, m_i + tl.math.log2(l_i))


def run(Z, H, N_CTX, HEAD_DIM=128, BLOCK_M=128, BLOCK_N=64, check=True):
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
    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H)
    fa_fwd_kernel_nows[grid](qf, kf, vf, of, m_lse, sm_scale, Z * H, N_CTX,
                             BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                             HEAD_DIM=HEAD_DIM, num_warps=4, num_ctas=1,
                             num_stages=2, maxRegAutoWS=152)
    if not check:
        return None
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v,
                                                           scale=sm_scale)
    out = of.view(Z, H, N_CTX, HEAD_DIM)
    rel = (out - ref).abs().max().item() / ref.abs().max().item()
    return rel


if __name__ == "__main__":
    # Small dump shape: Z*H=4 keeps the H argument non-divisible-by-16,
    # matching the committed bf16 fixture's argument specialization.
    rel = run(Z=1, H=4, N_CTX=2048)
    status = "PASS" if rel < 1e-2 else "FAIL"
    print(f"fa_fwd_kernel_nows fp16 (1,4,2048,128): rel={rel:.2e} {status}")
