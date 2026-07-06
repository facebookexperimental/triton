"""Perf: sched2tlx-generated FA-bwd WS kernel vs no-WS baseline vs hand-written
WS ground truth. All at the generated config HEAD_DIM=64, BLOCK_M=64, BLOCK_N=128.

FA-bwd 5-MMA flops = 5 matmuls x 2 (MAC) x BH x N_CTX^2 x HEAD_DIM.
"""

from __future__ import annotations

import math
import sys

import torch
import triton

from generated import fa_bwd_dkdv_5mma as gen_kernel
from handwritten_nows import fa_bwd_dkdv_5mma as nows_kernel
from handwritten import fa_bwd_dkdv_ws as ws_kernel

LOG2E = 1.4426950408889634
HEAD_DIM, BLOCK_M, BLOCK_N = 64, 64, 128


def alloc_fn(size, alignment, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


def _inputs(BH, N_CTX):
    torch.manual_seed(0)
    sm = 1.0 / math.sqrt(HEAD_DIM)
    q = torch.randn(BH, N_CTX, HEAD_DIM, device="cuda", dtype=torch.float16) * sm
    k = torch.randn(BH, N_CTX, HEAD_DIM, device="cuda", dtype=torch.float16)
    v = torch.randn(BH, N_CTX, HEAD_DIM, device="cuda", dtype=torch.float16)
    do = torch.randn(BH, N_CTX, HEAD_DIM, device="cuda", dtype=torch.float16)
    s = torch.matmul(q, k.transpose(-1, -2))
    o = torch.matmul(torch.softmax(s, -1), v)
    m = (torch.logsumexp(s, -1) * LOG2E).to(torch.float32).contiguous()
    D = (do.float() * o.float()).sum(-1).contiguous()
    return (
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        do.contiguous(),
        m,
        D,
    )


def _bench(fn):
    return triton.testing.do_bench(fn, warmup=25, rep=100)


def run(BH, N_CTX):
    triton.set_allocator(alloc_fn)
    q, k, v, do, M, D = _inputs(BH, N_CTX)
    dq = torch.zeros_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    stride = HEAD_DIM
    grid = (N_CTX // BLOCK_N, BH)
    flops = 5 * 2 * BH * N_CTX * N_CTX * HEAD_DIM

    def gen():
        dq.zero_()
        gen_kernel[grid](
            q,
            k,
            v,
            do,
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

    def nows():
        dq.zero_()
        nows_kernel[grid](
            q,
            k,
            v,
            do,
            dq,
            dk,
            dv,
            M,
            D,
            stride,
            stride,
            N_CTX,
            BLOCK_M,
            BLOCK_N,
            HEAD_DIM,
            False,
            num_warps=4,
            num_ctas=1,
            num_stages=1,
        )

    def ws():
        dq.zero_()
        ws_kernel[grid](
            q,
            k,
            v,
            do,
            dq,
            dk,
            dv,
            M,
            D,
            stride,
            stride,
            N_CTX,
            BLOCK_M,
            BLOCK_N,
            HEAD_DIM,
            2,
            num_warps=4,
            num_ctas=1,
            num_stages=1,
        )

    res = {}
    for name, fn in (("nows", nows), ("ws", ws), ("gen", gen)):
        try:
            ms = _bench(fn)
            res[name] = (ms, flops / (ms * 1e-3) / 1e12)
        except Exception as e:
            res[name] = (None, f"ERR:{type(e).__name__}")

    def tf(x):
        return f"{x[1]:6.1f}" if isinstance(x[1], float) else f"{x[1]:>8}"

    def ms(x):
        return f"{x[0]:7.3f}" if x[0] is not None else "    n/a"

    g = res["gen"]
    base = res["nows"]
    ratio = (
        (g[1] / base[1])
        if isinstance(g[1], float) and isinstance(base[1], float)
        else 0
    )
    print(
        f"BH={BH:2d} N={N_CTX:5d}  "
        f"nows {ms(base)}ms {tf(base)}TF | "
        f"ws {ms(res['ws'])}ms {tf(res['ws'])}TF | "
        f"gen {ms(g)}ms {tf(g)}TF | gen/nows={ratio:.2f}x"
    )


def main() -> int:
    # 6 shapes, small -> large (N sweep 1k..32k)
    for BH, N in [
        (2, 1024),
        (4, 2048),
        (8, 4096),
        (8, 8192),
        (8, 16384),
        (2, 32768),
    ]:
        run(BH, N)
    return 0


if __name__ == "__main__":
    sys.exit(main())
