"""Run hand-written FA forward TLX target on B200 vs torch SDPA."""

from __future__ import annotations

import sys

import handwritten
import torch
import torch.nn.functional as F
import triton


def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


def main() -> int:
    triton.set_allocator(alloc_fn)
    torch.manual_seed(0)
    BLOCK_M = 128
    BLOCK_N = 64
    HEAD_DIM = 128
    NUM_BUFFERS_KV = 2

    shapes = [
        # (Z, H, N_CTX)
        (1, 4, 512),
        (1, 8, 1024),
        (2, 16, 2048),
    ]
    failed = 0
    for Z, H, N_CTX in shapes:
        q = torch.randn(Z, H, N_CTX, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(Z, H, N_CTX, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(Z, H, N_CTX, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(Z, H, N_CTX, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        m_lse = torch.empty(Z * H, N_CTX, device="cuda", dtype=torch.float32)
        sm_scale = 1.0 / (HEAD_DIM**0.5)

        # Triton expects flat tensors with [batch*heads*N_CTX, HEAD_DIM] layout.
        q_flat = q.contiguous().view(-1, HEAD_DIM)
        k_flat = k.contiguous().view(-1, HEAD_DIM)
        v_flat = v.contiguous().view(-1, HEAD_DIM)
        out_flat = out.view(-1, HEAD_DIM)

        grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H)
        handwritten.fa_fwd_kernel[grid](
            q_flat,
            k_flat,
            v_flat,
            out_flat,
            m_lse,
            sm_scale,
            Z,
            H,
            N_CTX,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            HEAD_DIM=HEAD_DIM,
            NUM_BUFFERS_KV=NUM_BUFFERS_KV,
            num_warps=4,
            num_ctas=1,
            num_stages=2,
        )

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
