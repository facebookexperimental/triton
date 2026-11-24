"""Perf and compile-time reporting for ``tlx.ops.moe_router_topk_align``.

The reference is the chain of launches the op replaces: ``tlx.ops.router_gemm``, ``tlx.ops.moe_routing`` and the slot
sort of ``tlx.ops.moe``. A serving stack may run the same steps with its own kernels (for example ``router_gemm``,
``moe_routing``, a local-id map and sgl-kernel's align, five launches per layer). ``flop_count`` is the router GEMM's
``2 * T * E * K``. Both sides are timed as the replay of a HIP graph holding one call, as in serving,
where a decode call costs less GPU time than its Python launches.
"""

from __future__ import annotations

import pathlib
import sys

import torch
from triton.tlx.ops.kernels.fused.moe_router_topk_align._shapes import FOCUS as SHAPE_SUITES
from triton.tlx.ops.kernels.fused.moe_router_topk_align._shapes import (SYNTHETIC, MoeRouterTopkAlignShape, inputs,
                                                                        label, reference)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _harness import Case, Prepared, driver  # noqa: E402

OP = "moe_router_topk_align"
REF_NAME = "router_gemm + moe_routing + tlx.ops.moe slot sort"
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
    from triton.tlx.ops import moe_router_topk_align

    shape = MoeRouterTopkAlignShape(*case.shape)
    args = inputs(shape)
    tlx_fwd = lambda: moe_router_topk_align(**args, space=space)  # noqa: E731
    ref_fwd = lambda: reference(**args)  # noqa: E731

    def check():
        for name, got, ref in zip(("weights", "ids", "sorted_ids", "expert_ids", "num_post"), tlx_fwd(), ref_fwd()):
            if not torch.equal(got, ref):
                return False, f"{name} differs from the separate ops"
        return True, ""

    return Prepared(tlx_fn=_graphed(tlx_fwd), ref_fn=_graphed(ref_fwd),
                    flop_count=2 * shape.tokens * shape.experts * shape.hidden, check=check)


supported, default_json, run, main = driver.bind(sys.modules[__name__])

if __name__ == "__main__":
    raise SystemExit(main())
