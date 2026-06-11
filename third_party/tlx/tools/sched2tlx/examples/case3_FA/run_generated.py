"""Run auto-emitted FA forward TLX target on B200 vs torch SDPA.

Generated kernel signature (matches handwritten_nows.py): Q, K, V, Out, M_lse,
sm_scale, H, N_CTX. Block sizes baked in: BLOCK_M=128, BLOCK_N=64, HEAD_DIM=128.
"""

from __future__ import annotations

import sys

import generated
import torch
import torch.nn.functional as F
import triton


def alloc_fn(size, alignment, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


def main() -> int:
    triton.set_allocator(alloc_fn)
    torch.manual_seed(0)
    BLOCK_M, BLOCK_N, HEAD_DIM = 128, 64, 128  # noqa: F841

    failed = 0
    for Z, H, N_CTX in [
        (1, 4, 512),
        (1, 8, 1024),
        (2, 16, 2048),
        (1, 16, 4096),
        (2, 16, 4096),
        (1, 32, 8192),
    ]:
        q = torch.randn(Z, H, N_CTX, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(Z, H, N_CTX, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(Z, H, N_CTX, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        out = torch.empty_like(q)
        m_lse = torch.empty(Z * H, N_CTX, device="cuda", dtype=torch.float32)
        sm_scale = 1.0 / (HEAD_DIM**0.5)
        grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H)
        try:
            generated.fa_fwd_kernel_nows[grid](
                q.contiguous().view(-1, HEAD_DIM),
                k.contiguous().view(-1, HEAD_DIM),
                v.contiguous().view(-1, HEAD_DIM),
                out.view(-1, HEAD_DIM),
                m_lse,
                sm_scale,
                Z * H,
                N_CTX,
                num_warps=4,
                num_ctas=1,
                num_stages=2,
            )
        except Exception as e:
            print(f"[FAIL-COMPILE] Z={Z} H={H} N_CTX={N_CTX}: {type(e).__name__}: {str(e)[:200]}")
            failed += 1
            continue
        ref = F.scaled_dot_product_attention(q, k, v, scale=sm_scale)
        err = (out.float() - ref.float()).abs().max().item()
        rel = err / max(ref.float().abs().max().item(), 1e-9)
        ok = rel < 1e-2
        marker = "PASS" if ok else "FAIL"
        print(f"[{marker}] Z={Z} H={H} N_CTX={N_CTX}  max abs={err:.3e}  rel={rel:.3e}")
        if not ok:
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
