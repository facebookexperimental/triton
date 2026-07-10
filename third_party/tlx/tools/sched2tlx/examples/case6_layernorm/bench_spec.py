"""Benchmark spec for case6 (LayerNorm fwd) consumed by examples/testing/perf_engine.py.

Launch logic mirrors perf_generated_vs_handwritten.py. Memory-bound → GB/s metric.
The generated kernel bakes N=512 and BLOCK_M=8 in as literals.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

N = 512
EPS = 1e-5
BLOCK_M = 8
TOL = 1e-2

SHAPES = [(16384, N), (65536, N), (262144, N)]


def _num_sms():
    return torch.cuda.get_device_properties(0).multi_processor_count


def make_inputs(shape):
    M, n = shape
    x = torch.randn(M, n, device="cuda", dtype=torch.float16)
    w = torch.randn(n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, device="cuda", dtype=torch.float16)
    y = torch.full((M, n), float("nan"), device="cuda", dtype=torch.float16)
    yh = torch.empty((M, n), device="cuda", dtype=torch.float16)
    return {"x": x, "w": w, "b": b, "y": y, "yh": yh, "M": M, "N": n}


def gen_call(generated, inputs):
    # Launch hints emitted by the modulo pass for memory-bound kernels
    # (absent → legacy launch: 1x-SMS grid, register file auto-fill).
    grid = _num_sms() * getattr(generated, "RECOMMENDED_GRID_MULTIPLIER", 1)
    kw = {"num_warps": 4}
    if (mreg := getattr(generated, "RECOMMENDED_MAXNREG", None)) is not None:
        kw["maxnreg"] = mreg
    generated.layernorm_fwd_nows[(grid,)](
        inputs["x"], inputs["w"], inputs["b"], inputs["y"], inputs["M"], EPS, **kw
    )
    return inputs["y"]


def hw_call(handwritten, inputs):
    num_persist = _num_sms() * 4
    handwritten.layernorm_fwd_tma[(num_persist,)](
        inputs["x"], inputs["w"], inputs["b"], inputs["yh"], inputs["M"], EPS,
        row_stride=inputs["x"].stride(0), N=N, BLOCK_M=BLOCK_M, BLOCK_N=512,
        NUM_PERSIST=num_persist, num_warps=4,
    )
    return inputs["yh"]


def metric(shape):
    M, n = shape
    return (M * n * 2 * 2, 1e9, "GB/s")  # read x + write y, fp16


def reference(inputs):
    return F.layer_norm(inputs["x"].float(), (inputs["N"],), inputs["w"].float(), inputs["b"].float(), EPS)
