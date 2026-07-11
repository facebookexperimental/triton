"""Benchmark spec for case4 (FA bwd dK/dV 5-MMA) consumed by
examples/testing/perf_engine.py.

Launch logic mirrors perf_generated.py (generated) and handwritten.fa_bwd_dkdv_ws
(reference). The kernel writes dK/dV directly and accumulates dQ via atomics,
so calls zero dQ first (counted in the timed region, same as perf_generated.py).
Outputs are compared as one concatenated (dK, dV, dQ) vector.
"""

from __future__ import annotations

import math

import torch

LOG2E = 1.4426950408889634
HEAD_DIM, BLOCK_M, BLOCK_N = 64, 64, 128
TOL = 3e-2

SHAPES = [(4, 2048), (8, 4096), (8, 8192)]


def make_inputs(shape):
    BH, N_CTX = shape
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
    with torch.no_grad():
        s = torch.matmul(q, k.transpose(-1, -2))
        p = torch.softmax(s, dim=-1)
        o = torch.matmul(p, v)
        m = (torch.logsumexp(s, dim=-1) * LOG2E).to(torch.float32).contiguous()
        D = (do.float() * o.float()).sum(-1).contiguous()
        del s, p, o
    return {
        "q": q,
        "k": k,
        "v": v,
        "do": do,
        "m": m,
        "D": D,
        "dq": torch.zeros(BH, N_CTX, HEAD_DIM, device="cuda", dtype=torch.float16),
        "dk": torch.empty(BH, N_CTX, HEAD_DIM, device="cuda", dtype=torch.float16),
        "dv": torch.empty(BH, N_CTX, HEAD_DIM, device="cuda", dtype=torch.float16),
        "BH": BH,
        "N_CTX": N_CTX,
    }


def _cat(dk, dv, dq):
    return torch.cat([dk.float().flatten(), dv.float().flatten(), dq.float().flatten()])


def gen_call(generated, inputs):
    q, k, v, do = inputs["q"], inputs["k"], inputs["v"], inputs["do"]
    dq, dk, dv = inputs["dq"], inputs["dk"], inputs["dv"]
    grid = (inputs["N_CTX"] // BLOCK_N, inputs["BH"])
    dq.zero_()
    generated.fa_bwd_dkdv_5mma[grid](
        q, k, v, do, dq, dk, dv,
        inputs["m"], inputs["D"],
        HEAD_DIM, HEAD_DIM, inputs["N_CTX"],
        num_warps=4, num_ctas=1, num_stages=1,
    )
    return _cat(dk, dv, dq)


def hw_call(handwritten, inputs):
    q, k, v, do = inputs["q"], inputs["k"], inputs["v"], inputs["do"]
    dq, dk, dv = inputs["dq"], inputs["dk"], inputs["dv"]
    grid = (inputs["N_CTX"] // BLOCK_N, inputs["BH"])
    dq.zero_()
    handwritten.fa_bwd_dkdv_ws[grid](
        q, k, v, do, dq, dk, dv,
        inputs["m"], inputs["D"],
        HEAD_DIM, HEAD_DIM, inputs["N_CTX"],
        BLOCK_M, BLOCK_N, HEAD_DIM, 2,
        num_warps=4, num_ctas=1, num_stages=1,
    )
    return _cat(dk, dv, dq)


def metric(shape):
    BH, N_CTX = shape
    return (5 * 2 * BH * N_CTX * N_CTX * HEAD_DIM, 1e12, "TFLOPS")


def reference(inputs):
    q, k, v, do = inputs["q"], inputs["k"], inputs["v"], inputs["do"]
    for t in (q, k, v):
        t.grad = None
    s = torch.matmul(q, k.transpose(-1, -2))
    o = torch.matmul(torch.softmax(s, dim=-1), v)
    o.backward(do)
    return _cat(k.grad, v.grad, q.grad)
