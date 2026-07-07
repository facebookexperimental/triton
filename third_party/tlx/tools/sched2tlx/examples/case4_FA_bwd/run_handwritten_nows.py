"""case4 FA bwd — correctness of the no-WS 5-MMA kernel vs torch.autograd.

Reference forward uses the SAME base-2 softmax the kernel assumes (sm_scale
folded into Q, no extra scale, P = exp2(S)/sum exp2(S)). M = log2(sum exp2(S)),
D = rowsum(dO*O). Kernel dQ/dK/dV must match autograd of that forward.
"""

from __future__ import annotations

import math
import sys

import torch
import triton
from handwritten_nows import fa_bwd_dkdv_5mma

LN2 = math.log(2.0)


def alloc_fn(size, alignment, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


LOG2E = 1.4426950408889634


def ref_fwd(q, k, v):
    # q already scaled. Forward is base-e softmax; the kernel folds log2e into
    # its exp2 exponent so exp2(qk*log2e - m) == exp(qk)/sum == P. M is the
    # base-e logsumexp scaled by log2e so the kernel's subtraction matches.
    s = torch.matmul(q, k.transpose(-1, -2))  # [BH, M, N]
    p = torch.softmax(s, dim=-1)
    o = torch.matmul(p, v)
    m = torch.logsumexp(s, dim=-1) * LOG2E  # exp2(S*log2e - M) == P
    return o, m


def run(BH, N_CTX, HEAD_DIM, BLOCK_M=128, BLOCK_N=128):
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

    # bwd preprocessing constants (detached)
    M = m.detach().to(torch.float32).contiguous()  # [BH, N_CTX]
    D = (do.float() * o.detach().float()).sum(-1).contiguous()  # [BH, N_CTX]

    qf = q.detach().contiguous()
    kf = k.detach().contiguous()
    vf = v.detach().contiguous()
    dof = do.contiguous()
    dq = torch.zeros_like(qf)  # atomic_add accumulates
    dk = torch.empty_like(kf)
    dv = torch.empty_like(vf)

    grid = (N_CTX // BLOCK_N, BH)
    fa_bwd_dkdv_5mma[grid](
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
        WARP_SPEC=False,  # GPU correctness run; WS form is for the modulo dump
        num_warps=4,
        num_ctas=1,
        num_stages=1,  # correctness only; avoid SMEM overflow from pipelining
    )
    torch.cuda.synchronize()

    def rel(a, b):
        return (a.float() - b.float()).abs().max().item() / max(
            b.float().abs().max().item(), 1e-9
        )

    rq, rk, rv = rel(dq, ref_dq), rel(dk, ref_dk), rel(dv, ref_dv)
    ok = max(rq, rk, rv) < 3e-2
    print(
        f"[{'PASS' if ok else 'FAIL'}] BH={BH} N={N_CTX} D={HEAD_DIM}  "
        f"dQ={rq:.2e} dK={rk:.2e} dV={rv:.2e}"
    )
    return ok


def main() -> int:
    # HEAD_DIM=64: the un-reused 5-MMA accumulators fit in TMEM (qkT+dpT 128c
    # each + dk/dv/dq 64c each = 448 < 512); HD=128 overflows without TMEM reuse.
    failed = 0
    for BH, N in [(2, 512), (4, 1024), (1, 2048)]:
        failed += 0 if run(BH, N, 64) else 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
