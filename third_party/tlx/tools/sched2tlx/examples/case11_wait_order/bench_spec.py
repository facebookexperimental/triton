"""Benchmark spec for case11 (wait_order) consumed by
examples/testing/perf_regression/perf_harness.py.

Launch logic mirrors run_generated.py. No handwritten reference exists for
this case (it is a synthetic schedule-order fixture), so the harness's hw
column stays empty.
"""

from __future__ import annotations

import torch

BLOCK_M, BLOCK_N, KD, ON = 128, 64, 64, 128
TOL = 2e-2

SHAPES = [(148, 32), (592, 64)]  # (G, T): CTAs x inner iterations


def make_inputs(shape):
    G, T = shape
    x = torch.randn(G * T * BLOCK_M, KD, device="cuda", dtype=torch.float16) * 0.25
    y = torch.randn(G * T * BLOCK_M, KD, device="cuda", dtype=torch.float16) * 0.25
    w1 = torch.randn(BLOCK_N, KD, device="cuda", dtype=torch.float16) * 0.25
    w2 = torch.randn(BLOCK_N, KD, device="cuda", dtype=torch.float16) * 0.25
    v = torch.randn(BLOCK_N, ON, device="cuda", dtype=torch.float16) * 0.25
    out = torch.full((G * BLOCK_M, ON), float("nan"), device="cuda",
                     dtype=torch.float16)
    return {"x": x, "y": y, "w1": w1, "w2": w2, "v": v, "out": out,
            "G": G, "T": T}


def gen_call(generated, inputs):
    G, T = inputs["G"], inputs["T"]
    generated.wait_order_kernel_nows[(G,)](
        inputs["x"], inputs["y"], inputs["w1"], inputs["w2"], inputs["v"],
        inputs["out"], 1.0, T, num_warps=4, num_ctas=1, num_stages=2)
    return inputs["out"]


def metric(shape):
    G, T = shape
    flops = G * T * 2 * (BLOCK_M * KD * BLOCK_N * 2 + BLOCK_M * BLOCK_N * ON)
    return (flops, 1e12, "TFLOPS")


def reference(inputs):
    G, T = inputs["G"], inputs["T"]
    xf = inputs["x"].float().view(G, T, BLOCK_M, KD)
    yf = inputs["y"].float().view(G, T, BLOCK_M, KD)
    ref = torch.zeros(G, BLOCK_M, ON, device="cuda")
    for i in range(T):
        s1 = xf[:, i] @ inputs["w1"].float().T
        s2 = yf[:, i] @ inputs["w2"].float().T
        ref += (torch.exp2(s1) - s2).half().float() @ inputs["v"].float()
    return ref.view(G * BLOCK_M, ON).half()
