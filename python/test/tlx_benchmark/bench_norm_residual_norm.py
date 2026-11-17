"""Perf and compile-time reporting for ``tlx.ops.norm_residual_norm``.

The reference is the unfused module sequence (BF16 post norm, FP32 gated residual add, pre norm) compiled by
``torch.compile``. The op is memory bound, so the case's ``flop_count`` is its compulsory traffic in bytes (BF16
branch and output, FP32 stream and new stream) and the ``TF/s`` columns read as TB/s of achieved bandwidth. Both sides
are timed as the replay of a graph holding one call: a decode call takes about 4 us of GPU time, less than its
Python launch.
"""

from __future__ import annotations

import pathlib
import sys

import torch
from triton.tlx.ops.kernels.fused.norm_residual_norm._shapes import FOCUS as SHAPE_SUITES
from triton.tlx.ops.kernels.fused.norm_residual_norm._shapes import (SYNTHETIC, NormResidualNormShape, bytes_moved,
                                                                     inputs, label, reference)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _harness import Case, Prepared, close_enough, driver  # noqa: E402

OP = "norm_residual_norm"
REF_NAME = "torch.compile of the unfused module sequence"
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
    from triton.tlx.ops import norm_residual_norm

    shape = NormResidualNormShape(*case.shape)
    kw = inputs(shape)
    out = torch.empty(shape.tokens, shape.H, device="cuda", dtype=torch.bfloat16)
    new_stream = torch.empty(shape.tokens, shape.H, device="cuda")
    ref_compiled = torch.compile(reference, dynamic=False)

    tlx_fwd = lambda: norm_residual_norm(**kw, out=out, new_stream=new_stream, space=space)  # noqa: E731
    ref_fwd = lambda: ref_compiled(**kw)  # noqa: E731

    def check():
        got, ref = tlx_fwd(), ref_fwd()
        return close_enough(got[0], ref[0], 8e-3)

    traffic = bytes_moved(shape)
    ref_fn = _graphed(ref_fwd)
    ref_fn()
    return Prepared(tlx_fn=_graphed(tlx_fwd), ref_fn=ref_fn, flop_count=traffic, check=check,
                    extra={"mb": round(traffic / 1e6, 1)})


supported, default_json, run, main = driver.bind(sys.modules[__name__])

if __name__ == "__main__":
    raise SystemExit(main())
