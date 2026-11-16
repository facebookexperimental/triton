"""Perf and compile-time reporting for ``tlx.ops.swiglu_ffn``.

The reference is the same expression (RMSNorm, the SwiGLU feed-forward, the residual) compiled by ``torch.compile``:
the Inductor kernels and cuBLAS GEMMs a model gets without this op. Both sides are timed as a replay of a CUDA graph
holding one call, because a call takes about as long on the GPU as either side's Python launch; production runs both
through AOTInductor, whose launches cost no Python.
"""

from __future__ import annotations

import pathlib
import sys

import torch
from triton.tlx.ops.kernels.swiglu_ffn._shapes import FOCUS as SHAPE_SUITES
from triton.tlx.ops.kernels.swiglu_ffn._shapes import SYNTHETIC, flops, inputs, label, reference

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _harness import Case, Prepared, close_enough, driver  # noqa: E402

OP = "swiglu_ffn"
REF_NAME = "torch.compile of the eager expression"
DEFAULT_SPACE = "heuristic"
EXTRA_COLUMNS = ()

DTYPES = {"fp16": torch.float16}
REL_PRECISION = 1e-2


def shapes(synthetic: bool = False, suites=None) -> list:
    return list(SYNTHETIC if synthetic else SHAPE_SUITES.shapes(driver.arch(), suites))


def cases(synthetic: bool = False, suites=None) -> list[Case]:
    return [
        Case(op=OP, arch=driver.arch(), dtype=str(DTYPES[shape.dtype]).removeprefix("torch."), shape=tuple(shape[:2]),
             label=label(*shape)) for shape in shapes(synthetic, suites)
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
    from triton.tlx.ops import swiglu_ffn

    tokens, d_model = case.shape
    args = inputs(tokens, d_model, getattr(torch, case.dtype))
    compiled = torch.compile(reference, dynamic=False)

    def tlx_call():
        return swiglu_ffn(*args, space=space)

    ref_fn = _graphed(lambda: compiled(*args))
    ref_fn()  # compiled and captured here, ahead of the harness's cold-compile pass, which times only the op
    return Prepared(
        tlx_fn=_graphed(tlx_call),
        ref_fn=ref_fn,
        flop_count=flops(tokens, d_model),
        check=lambda: close_enough(tlx_call(), reference(*args), REL_PRECISION),
    )


supported, default_json, run, main = driver.bind(sys.modules[__name__])

if __name__ == "__main__":
    raise SystemExit(main())
