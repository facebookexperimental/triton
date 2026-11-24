"""Perf and compile-time reporting for ``tlx.ops.rmsnorm_cat``.

The reference is the same expression (part norms, concatenation, final norm per output weight) compiled by
``torch.compile``: the Inductor-fused kernels a model gets without this op, which is what the production model runs
in production. It is the baseline to replace. The op is memory bound, so the case's ``flop_count`` is its
compulsory traffic in bytes (each part read once, each output written once) and the ``TF/s`` columns read as TB/s
of achieved bandwidth (H100 HBM3 peaks at about 3.35). The suite derives speedup from that rate, so it stays
"higher is faster".

Both sides are timed as a replay of a CUDA graph holding one call. A call takes 10-60 us of GPU time, less than the
Python launch of either side (about 0.15 ms for the op's 70 kernel arguments), so timing the eager calls would
measure the host. Production runs both through AOTInductor, whose launches cost no Python.
"""

from __future__ import annotations

import pathlib
import sys

import torch
from triton.tlx.ops.kernels.norm._cat_shapes import FOCUS as SHAPE_SUITES
from triton.tlx.ops.kernels.norm._cat_shapes import SYNTHETIC, bytes_moved, inputs, label, reference

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _harness import Case, Prepared, close_enough, driver  # noqa: E402

OP = "rmsnorm_cat"
REF_NAME = "torch.compile of the eager expression (Inductor-fused norms)"
DEFAULT_SPACE = "heuristic"
EXTRA_COLUMNS = (("MB", "mb"), )
COLD_COMPILE = "first"

REL_PRECISION = {"float16": 1e-3, "bfloat16": 8e-3}


def shapes(synthetic: bool = False, suites=None) -> list:
    return list(SYNTHETIC if synthetic else SHAPE_SUITES.shapes(driver.arch(), suites))


def cases(synthetic: bool = False, suites=None) -> list[Case]:
    dtypes = {"fp16": "float16", "bf16": "bfloat16"}
    return [
        Case(op=OP, arch=driver.arch(), dtype=dtypes[shape.dtype], shape=tuple(shape), label=label(*shape))
        for shape in shapes(synthetic, suites)
    ]


def _graphed(fn):
    """A zero-argument callable whose first call runs ``fn`` eagerly (so the harness's cold-compile pass times the
    real first call, compilation included) and captures it in a CUDA graph; later calls replay the graph."""
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
    from triton.tlx.ops import rmsnorm_cat

    del space  # One fixed launch config.
    parts, part_weights, weights = inputs(case.shape)
    ref_compiled = torch.compile(reference, dynamic=False)

    tlx_fwd = lambda: rmsnorm_cat(parts, weights, part_weights=part_weights)  # noqa: E731
    ref_fwd = lambda: ref_compiled(parts, part_weights, weights)  # noqa: E731

    def check():
        return close_enough(torch.stack(tlx_fwd()), torch.stack(ref_fwd()), REL_PRECISION[case.dtype])

    traffic = bytes_moved(parts, len(weights))
    ref_fn = _graphed(ref_fwd)
    ref_fn()  # compiled and captured here, ahead of the harness's cold-compile pass, which times only the op
    return Prepared(tlx_fn=_graphed(tlx_fwd), ref_fn=ref_fn, flop_count=traffic, check=check,
                    extra={"mb": round(traffic / 1e6, 1)})


supported, default_json, run, main = driver.bind(sys.modules[__name__])

if __name__ == "__main__":
    raise SystemExit(main())
