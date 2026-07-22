"""Benchmark spec for case4 (Flash-Attention backward) consumed by
examples/testing/perf_regression/perf_harness.py.

Launch logic mirrors run_generated.py (the sched2tlx-emitted 5-MMA WS kernel)
and run_handwritten.py (the hand-written TLX-WS reference), side by side.

Shape note: case4 uses a combined batch*heads leading dim, so SHAPES are
``(BH, N_CTX, HEAD_DIM)`` (Q/K/V/dO are ``[BH, N_CTX, HEAD_DIM]``). HEAD_DIM is
pinned to 64 because the generated kernel ``fa_bwd_dkdv_5mma`` bakes HEAD_DIM=64
into its descriptors/tiles (``[N_CTX, 64]``, per-(b,h) stride 64); it cannot run
any other head dim, so gen/hw/reference all share HEAD_DIM=64.

Three-output note: FA-bwd produces dQ, dK, dV, but the harness compares ONE
tensor. gen_call/hw_call/reference each return ``cat([dQ, dK, dV])`` (identically
flattened) so a wrong dQ, dK, OR dV surfaces in the harness's ``_rel`` check.

dQ accumulates in-place (generated: TMA ``store_reduce="add"``; handwritten:
``tl.atomic_add``), so dQ is re-zeroed before every gen/hw launch.
"""

from __future__ import annotations

import math

import torch

LOG2E = 1.4426950408889634

# Baked into the generated kernel's grid (K/V tile is BLOCK_N=128 wide).
GEN_BLOCK_N = 128
# Hand-written kernel block config (matches run_handwritten.py). NUM_BUFFERS_Q,
# num_warps, num_ctas, num_stages are all supplied by @triton.autotune.
HW_BLOCK_M = 128
HW_BLOCK_N = 128

TOL = 3e-2  # matches the max(rq, rk, rv) PASS bound in case4's runners

# (BH, N_CTX, HEAD_DIM) — shapes used by case4's runners; HEAD_DIM pinned to 64.
SHAPES = [
    (1, 256, 64),
    (2, 512, 64),
    (4, 1024, 64),
    (1, 2048, 64),
]


def _ref_fwd(q, k, v):
    # case4 base-e/log2e convention (see run_generated.py): q is pre-scaled by
    # 1/sqrt(HEAD_DIM); forward is base-e softmax; M = logsumexp_e * log2e so the
    # kernel's exp2(qkT*log2e - M) reproduces P.
    s = torch.matmul(q, k.transpose(-1, -2))
    p = torch.softmax(s, dim=-1)
    o = torch.matmul(p, v)
    m = torch.logsumexp(s, dim=-1) * LOG2E
    return o, m


def make_inputs(shape):
    BH, N_CTX, HEAD_DIM = shape
    sm = 1.0 / math.sqrt(HEAD_DIM)
    q = (
        (torch.randn(BH, N_CTX, HEAD_DIM, device="cuda", dtype=torch.float16) * sm)
        .detach()
        .requires_grad_(True)
    )
    k = torch.randn(BH, N_CTX, HEAD_DIM, device="cuda", dtype=torch.float16).detach().requires_grad_(True)
    v = torch.randn(BH, N_CTX, HEAD_DIM, device="cuda", dtype=torch.float16).detach().requires_grad_(True)
    do = torch.randn(BH, N_CTX, HEAD_DIM, device="cuda", dtype=torch.float16)

    o, m = _ref_fwd(q, k, v)
    o.backward(do)
    ref_dq, ref_dk, ref_dv = q.grad, k.grad, v.grad

    M = m.detach().to(torch.float32).contiguous()
    D = (do.float() * o.detach().float()).sum(-1).contiguous()
    qf = q.detach().contiguous()
    kf = k.detach().contiguous()
    vf = v.detach().contiguous()
    dof = do.contiguous()
    dq = torch.zeros_like(qf)  # accumulated in place → zero before each launch
    dk = torch.empty_like(kf)
    dv = torch.empty_like(vf)
    return {
        "qf": qf, "kf": kf, "vf": vf, "dof": dof,
        "dq": dq, "dk": dk, "dv": dv, "M": M, "D": D,
        "BH": BH, "N_CTX": N_CTX, "HEAD_DIM": HEAD_DIM,
        "ref_dq": ref_dq, "ref_dk": ref_dk, "ref_dv": ref_dv,
    }


def _cat3(dq, dk, dv):
    return torch.cat([dq.flatten(), dk.flatten(), dv.flatten()])


def gen_call(generated, inputs):
    inputs["dq"].zero_()
    stride = inputs["HEAD_DIM"]  # per-(b,h) [N_CTX, HEAD_DIM] row stride
    grid = (inputs["N_CTX"] // GEN_BLOCK_N, inputs["BH"])
    generated.fa_bwd_dkdv_5mma[grid](
        inputs["qf"], inputs["kf"], inputs["vf"], inputs["dof"],
        inputs["dq"], inputs["dk"], inputs["dv"], inputs["M"], inputs["D"],
        stride, stride, inputs["N_CTX"],
        num_warps=4, num_ctas=1, num_stages=1,
    )
    return _cat3(inputs["dq"], inputs["dk"], inputs["dv"])


def hw_call(handwritten, inputs):
    inputs["dq"].zero_()
    stride = inputs["HEAD_DIM"]
    grid = (inputs["N_CTX"] // HW_BLOCK_N, inputs["BH"])
    # NUM_BUFFERS_Q / num_warps / num_ctas / num_stages come from @triton.autotune.
    handwritten.fa_bwd_dkdv_ws[grid](
        inputs["qf"], inputs["kf"], inputs["vf"], inputs["dof"],
        inputs["dq"], inputs["dk"], inputs["dv"], inputs["M"], inputs["D"],
        stride, stride, inputs["N_CTX"],
        HW_BLOCK_M, HW_BLOCK_N, inputs["HEAD_DIM"],
    )
    return _cat3(inputs["dq"], inputs["dk"], inputs["dv"])


def metric(shape):
    BH, N_CTX, HEAD_DIM = shape
    # FA-bwd is a 5-MMA kernel (dkdv 5mma): 5 matmuls, each 2 FLOP per MAC over an
    # N_CTX x N_CTX x HEAD_DIM tile per (b,h) → 10 * BH * N_CTX^2 * HEAD_DIM, i.e.
    # ~2.5x the forward matmul FLOPs.
    return (10 * BH * N_CTX * N_CTX * HEAD_DIM, 1e12, "TFLOPS")


def reference(inputs):
    return _cat3(inputs["ref_dq"], inputs["ref_dk"], inputs["ref_dv"])
