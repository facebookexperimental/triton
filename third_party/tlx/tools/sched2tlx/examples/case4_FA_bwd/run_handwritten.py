"""case4 FA bwd — correctness of the hand-written WS kernel vs torch.autograd.

Same base-e/log2e convention as run_handwritten_nows.py. HEAD_DIM=128 fits
because the WS kernel uses TMEM storage-aliasing.
"""

from __future__ import annotations

import math
import sys

import torch
import triton
from handwritten import fa_bwd_dkdv_ws

LOG2E = 1.4426950408889634


def alloc_fn(size, alignment, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


def ref_fwd(q, k, v):
    s = torch.matmul(q, k.transpose(-1, -2))
    p = torch.softmax(s, dim=-1)
    o = torch.matmul(p, v)
    m = torch.logsumexp(s, dim=-1) * LOG2E
    return o, m


def run(BH, N_CTX, HEAD_DIM=128, BLOCK_M=128, BLOCK_N=128, NUM_BUFFERS_Q=2):
    triton.set_allocator(alloc_fn)
    torch.manual_seed(0)
    sm = 1.0 / math.sqrt(HEAD_DIM)
    q = (
        (torch.randn(BH, N_CTX, HEAD_DIM, device="cuda", dtype=torch.float16) * sm)
        .detach()
        .requires_grad_(True)
    )
    k = (
        torch.randn(BH, N_CTX, HEAD_DIM, device="cuda", dtype=torch.float16)
        .detach()
        .requires_grad_(True)
    )
    v = (
        torch.randn(BH, N_CTX, HEAD_DIM, device="cuda", dtype=torch.float16)
        .detach()
        .requires_grad_(True)
    )
    do = torch.randn(BH, N_CTX, HEAD_DIM, device="cuda", dtype=torch.float16)

    o, m = ref_fwd(q, k, v)
    o.backward(do)
    ref_dq, ref_dk, ref_dv = q.grad, k.grad, v.grad

    M = m.detach().to(torch.float32).contiguous()
    D = (do.float() * o.detach().float()).sum(-1).contiguous()
    qf, kf, vf, dof = (
        q.detach().contiguous(),
        k.detach().contiguous(),
        v.detach().contiguous(),
        do.contiguous(),
    )
    dq = torch.zeros_like(qf)
    dk = torch.empty_like(kf)
    dv = torch.empty_like(vf)

    grid = (N_CTX // BLOCK_N, BH)
    fa_bwd_dkdv_ws[grid](
        qf,
        kf,
        vf,
        dof,
        dq,
        dk,
        dv,
        M,
        D,
        HEAD_DIM,
        HEAD_DIM,
        N_CTX,
        BLOCK_M,
        BLOCK_N,
        HEAD_DIM,
        # NUM_BUFFERS_Q + num_warps/num_stages/num_ctas injected by @triton.autotune.
    )
    torch.cuda.synchronize()

    def rel(a, b):
        return (a.float() - b.float()).abs().max().item() / max(
            b.float().abs().max().item(), 1e-9
        )

    rq, rk, rv = rel(dq, ref_dq), rel(dk, ref_dk), rel(dv, ref_dv)
    ok = max(rq, rk, rv) < 3e-2
    print(
        f"[{'PASS' if ok else 'FAIL'}] BH={BH} N={N_CTX} D={HEAD_DIM}  dQ={rq:.2e} dK={rk:.2e} dV={rv:.2e}"
    )
    return ok


def main() -> int:
    import os

    shapes = [(1, 256)] if os.environ.get("C4_TINY") else [(2, 512), (4, 1024)]
    failed = 0
    for BH, N in shapes:
        failed += 0 if run(BH, N) else 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
