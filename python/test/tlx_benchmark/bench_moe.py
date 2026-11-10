"""Perf and compile-time reporting for ``tlx.ops.moe``.

The op's natural reference, FlyDSL's Muse MoE, ships with aiter rather than torch, so the suite reports TLX
throughput without a speedup gate; correctness against a torch reference lives in the unit tests. A call is timed as
the replay of a HIP graph holding it, as in serving: at decode its four to five launches cost more host time than
device time.
"""

from __future__ import annotations

import pathlib
import sys

import torch
from triton.tlx.ops.kernels.moe._shapes import FOCUS as SHAPE_SUITES
from triton.tlx.ops.kernels.moe._shapes import SYNTHETIC, MoeShape, flops, inputs, label

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _harness import Case, Prepared, driver  # noqa: E402

OP = "moe"
REF_NAME = ""
DEFAULT_SPACE = "heuristic"
EXTRA_COLUMNS = (("local rows", "local_rows"), )
COLD_COMPILE = "first"

DTYPES = {"bf16": torch.bfloat16}


def shapes(synthetic: bool = False, suites=None) -> list:
    return list(SYNTHETIC if synthetic else SHAPE_SUITES.shapes(driver.arch(), suites))


def cases(synthetic: bool = False, suites=None) -> list[Case]:
    return [
        Case(op=OP, arch=driver.arch(), dtype=str(DTYPES[shape.dtype]).removeprefix("torch."), shape=tuple(shape),
             label=label(shape)) for shape in shapes(synthetic, suites)
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


_LOCAL_ROWS: dict[tuple, int] = {}


def prepare(case: Case, space: str) -> Prepared:
    from triton.tlx.ops import moe

    shape = MoeShape(*case.shape)
    x, weights, ids, w13, w2, norm = inputs(shape)
    local = (ids >= shape.expert_start) & (ids < shape.expert_start + shape.local_experts)
    local_rows = int(local.sum())
    _LOCAL_ROWS[tuple(case.shape)] = local_rows
    out = torch.empty(shape.tokens, shape.H, device="cuda", dtype=x.dtype)

    def tlx_call():
        return moe(x, weights, ids, w13, w2, expert_start=shape.expert_start, post_norm_weight=norm, out=out,
                   space=space)

    return Prepared(
        tlx_fn=_graphed(tlx_call),
        ref_fn=None,
        flop_count=flops(local_rows, shape.K, shape.I, shape.H),
        check=None,
    )


def annotate(result) -> None:
    result.extra["local_rows"] = _LOCAL_ROWS.get(tuple(result.case.shape))


supported, default_json, run, main = driver.bind(sys.modules[__name__])

if __name__ == "__main__":
    raise SystemExit(main())
