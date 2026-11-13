"""Perf and compile-time reporting for ``tlx.ops.moe_routing``.

The reference is the same routing (sigmoid, biased stable sort, gather, left-to-right sum, scale) compiled by
``torch.compile``: what a model gets without a routing kernel. The op is memory bound, so the case's ``flop_count``
is its compulsory traffic in bytes (logits and bias read once, weights and ids written once) and the ``TF/s`` columns
read as TB/s of achieved bandwidth. Both sides are timed as the replay of a HIP graph holding one call, as in
serving, where a decode call costs less GPU time than its Python launch.
"""

from __future__ import annotations

import pathlib
import sys

import torch
from triton.tlx.ops.kernels.moe._routing_shapes import FOCUS as SHAPE_SUITES
from triton.tlx.ops.kernels.moe._routing_shapes import (ROUTE_EPS, ROUTER_MULTIPLIER, SYNTHETIC, RoutingShape, inputs,
                                                        label, reference)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _harness import Case, Prepared, driver  # noqa: E402

OP = "moe_routing"
REF_NAME = "torch.compile of the eager routing"
DEFAULT_SPACE = "heuristic"
EXTRA_COLUMNS = (("MB", "mb"), )
COLD_COMPILE = "first"

ROUTER = dict(router_multiplier=ROUTER_MULTIPLIER, route_scale=1.0, eps=ROUTE_EPS)


def shapes(synthetic: bool = False, suites=None) -> list:
    return list(SYNTHETIC if synthetic else SHAPE_SUITES.shapes(driver.arch(), suites))


def cases(synthetic: bool = False, suites=None) -> list[Case]:
    return [
        Case(op=OP, arch=driver.arch(), dtype="float32", shape=tuple(shape), label=label(shape))
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
    from triton.tlx.ops import moe_routing

    shape = RoutingShape(*case.shape)
    logits, bias = inputs(shape)
    out = (torch.empty(shape.tokens, shape.topk,
                       device="cuda"), torch.empty(shape.tokens, shape.topk, device="cuda", dtype=torch.int32))
    ref_compiled = torch.compile(reference, dynamic=False)

    tlx_fwd = lambda: moe_routing(logits, bias, shape.topk, out=out, space=space, **ROUTER)  # noqa: E731
    ref_fwd = lambda: ref_compiled(logits, bias, shape.topk, **ROUTER)  # noqa: E731

    def check():
        (w, ids), (rw, rids) = tlx_fwd(), ref_fwd()
        # Random logits can tie to within an ulp of sigmoid; most rows must agree exactly.
        same = (ids == rids).all(dim=1)
        if same.float().mean() <= 0.99:
            return False, f"ids differ on {int((~same).sum())} of {same.numel()} rows"
        if not torch.allclose(w[same], rw[same], rtol=2e-6, atol=1e-7):
            return False, "weights do not match the reference"
        return True, ""

    traffic = shape.tokens * shape.experts * 4 + shape.experts * 4 + shape.tokens * shape.topk * 8
    ref_fn = _graphed(ref_fwd)
    ref_fn()  # compiled and captured here, ahead of the harness's cold-compile pass, which times only the op
    return Prepared(tlx_fn=_graphed(tlx_fwd), ref_fn=ref_fn, flop_count=traffic, check=check,
                    extra={"mb": round(traffic / 1e6, 2)})


supported, default_json, run, main = driver.bind(sys.modules[__name__])

if __name__ == "__main__":
    raise SystemExit(main())
