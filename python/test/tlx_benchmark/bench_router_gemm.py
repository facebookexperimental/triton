"""Perf and compile-time reporting for ``tlx.ops.router_gemm``.

The reference is ``F.linear`` on the fp32 upcast of the bf16 hidden states with the fp32 router weight: what a model
runs for an fp32 router without this op (hipBLASLt fp32 on gfx950). ``flop_count`` is ``2 * T * E * K``. Both sides
are timed as the replay of a HIP graph holding one call, as in serving, where a decode call costs less GPU time than
its Python launch; the reference's upcast of ``x`` is outside its graph.
"""

from __future__ import annotations

import pathlib
import sys

import torch
import torch.nn.functional as F
from triton.tlx.ops.kernels.router_gemm._shapes import FOCUS as SHAPE_SUITES
from triton.tlx.ops.kernels.router_gemm._shapes import SYNTHETIC, RouterGemmShape, inputs, label, reference

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _harness import Case, Prepared, driver  # noqa: E402

OP = "router_gemm"
REF_NAME = "F.linear on the fp32 upcast (hipBLASLt fp32)"
DEFAULT_SPACE = "heuristic"
EXTRA_COLUMNS = ()
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
    from triton.tlx.ops import router_gemm

    shape = RouterGemmShape(*case.shape)
    x, weight = inputs(shape)
    x_fp32 = x.float()
    out = torch.empty(shape.tokens, shape.experts, device="cuda")

    tlx_fwd = lambda: router_gemm(x, weight, out=out, space=space)  # noqa: E731
    ref_fwd = lambda: F.linear(x_fp32, weight)  # noqa: E731

    def check():
        exact = reference(x, weight)
        err = (tlx_fwd().double() - exact).abs().max().item()
        ref_err = (ref_fwd().double() - exact).abs().max().item()
        # Same accuracy class as the fp32 GEMM: within 4x its worst error against the fp64 result.
        if err > 4 * ref_err + 1e-30:
            return False, f"max error {err:.3g} vs fp32 GEMM {ref_err:.3g}"
        return True, ""

    return Prepared(tlx_fn=_graphed(tlx_fwd), ref_fn=_graphed(ref_fwd),
                    flop_count=2 * shape.tokens * shape.experts * shape.hidden, check=check)


supported, default_json, run, main = driver.bind(sys.modules[__name__])

if __name__ == "__main__":
    raise SystemExit(main())
