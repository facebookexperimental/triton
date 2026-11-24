"""Perf and compile-time reporting for ``tlx.ops.rmsnorm_gate``.

The reference is ``torch.compile`` of the eager expression (FP32 per-head RMSNorm times the sigmoid of the gate). The
op is memory bound, so the case's ``flop_count`` is its compulsory traffic in bytes (BF16 input, gate and output) and
the ``TF/s`` columns read as TB/s of achieved bandwidth. Both sides are timed as the replay of a graph holding one
call: a decode call takes about 2 us of GPU time, less than its Python launch.
"""

from __future__ import annotations

import pathlib
import sys

import torch
from triton.tlx.ops.kernels.fused.rmsnorm_gate._shapes import FOCUS as SHAPE_SUITES
from triton.tlx.ops.kernels.fused.rmsnorm_gate._shapes import (SYNTHETIC, RmsnormGateShape, bytes_moved, inputs, label)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _harness import Case, Prepared, close_enough, driver  # noqa: E402

OP = "rmsnorm_gate"
REF_NAME = "torch.compile of the eager expression"
DEFAULT_SPACE = "heuristic"
EXTRA_COLUMNS = (("MB", "mb"), )
COLD_COMPILE = "first"


def shapes(synthetic: bool = False, suites=None) -> list:
    return list(SYNTHETIC if synthetic else SHAPE_SUITES.shapes(driver.arch(), suites))


def cases(synthetic: bool = False, suites=None) -> list[Case]:
    return [
        Case(op=OP, arch=driver.arch(), dtype="bfloat16", shape=tuple(shape), label=label(shape))
        for shape in shapes(synthetic, suites)
    ]


def _eager(x, gate, eps):
    T, N = x.shape
    y = torch.nn.functional.rms_norm(x.float().view(T, N // 64, 64), (64, ), None, eps)
    return (y * torch.sigmoid(gate.float().view(T, N // 64, 64))).view(T, N).to(torch.bfloat16)


def _graphed(fn):
    """The first call runs ``fn`` eagerly, so the cold-compile pass times the real first call, then captures it in a
    graph; later calls replay the graph."""
    replay = []

    def call():
        if replay:
            return replay[0]()
        fn()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fn()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        replay.append(graph.replay)

    return call


def prepare(case: Case, space: str) -> Prepared:
    from triton.tlx.ops import rmsnorm_gate

    shape = RmsnormGateShape(*case.shape)
    kw = inputs(shape)
    out = torch.empty(shape.tokens, shape.heads * 64, device="cuda", dtype=torch.bfloat16)
    ref_compiled = torch.compile(_eager, dynamic=False)

    tlx_fwd = lambda: rmsnorm_gate(**kw, out=out, space=space)  # noqa: E731
    ref_fwd = lambda: ref_compiled(**kw)  # noqa: E731

    def check():
        return close_enough(tlx_fwd(), ref_fwd(), 8e-3)

    traffic = bytes_moved(shape)
    ref_fn = _graphed(ref_fwd)
    ref_fn()
    return Prepared(tlx_fn=_graphed(tlx_fwd), ref_fn=ref_fn, flop_count=traffic, check=check,
                    extra={"mb": round(traffic / 1e6, 1)})


supported, default_json, run, main = driver.bind(sys.modules[__name__])

if __name__ == "__main__":
    raise SystemExit(main())
