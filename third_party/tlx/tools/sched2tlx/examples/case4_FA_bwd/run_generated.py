"""case4 FA bwd — correctness of the sched2tlx-generated WS kernel vs torch.

Same base-e/log2e convention as run_handwritten_nows.py. The generated kernel
bakes HEAD_DIM=128 into the schedule (descriptors/tiles are [128,128]); strides
are the per-(b,h) row stride = HEAD_DIM. dQ accumulates via TMA reduce so dq
must start zeroed.
"""

from __future__ import annotations

import importlib
import math
import sys

import torch
import triton

try:
    from generated import fa_bwd_dkdv_5mma
except ModuleNotFoundError:  # buck par: module lives under the dotted package
    _gen = importlib.import_module(
        (__package__ + ".generated") if __package__ else "generated"
    )
    fa_bwd_dkdv_5mma = _gen.fa_bwd_dkdv_5mma

LOG2E = 1.4426950408889634


def alloc_fn(size, alignment, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


def ref_fwd(q, k, v):
    s = torch.matmul(q, k.transpose(-1, -2))
    p = torch.softmax(s, dim=-1)
    o = torch.matmul(p, v)
    m = torch.logsumexp(s, dim=-1) * LOG2E
    return o, m


def run(BH, N_CTX, HEAD_DIM=64, BLOCK_M=64, BLOCK_N=128):
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
    dq = torch.zeros_like(qf)  # TMA reduce accumulates
    dk = torch.empty_like(kf)
    dv = torch.empty_like(vf)

    stride = HEAD_DIM  # per-(b,h) [N_CTX, HEAD_DIM] row stride
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
        stride,
        stride,
        N_CTX,
        num_warps=4,
        num_ctas=1,
        num_stages=1,
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
    import os

    shapes = [(1, 256)] if os.environ.get("C4_TINY") else [(2, 512), (4, 1024)]
    failed = 0
    for BH, N in shapes:
        failed += 0 if run(BH, N) else 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
