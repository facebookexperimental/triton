"""Perf and compile-time reporting for ``tlx.ops.mla_window_attn``.

The reference is the same module (RMSNorm, the low-rank q/kv/gate projections, the gate and the residual) with the
windowed attention as FlexAttention under a sequence-and-window block mask, compiled by ``torch.compile``. Both sides
are timed as a replay of a CUDA graph holding one call, because a call takes about as long on the GPU as either side's
Python launch; production runs both through AOTInductor, whose launches cost no Python.
"""

from __future__ import annotations

import pathlib
import sys

import torch
from triton.tlx.ops.kernels.mla_window_attn._shapes import FOCUS as SHAPE_SUITES
from triton.tlx.ops.kernels.mla_window_attn._shapes import (SYNTHETIC, flops, inputs, label, reference,
                                                            window_block_mask)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _harness import Case, Prepared, close_enough, driver  # noqa: E402

OP = "mla_window_attn"
REF_NAME = "torch.compile of the module, attention as FlexAttention"
DEFAULT_SPACE = "heuristic"
EXTRA_COLUMNS = ()

DTYPES = {"fp16": torch.float16}
REL_PRECISION = 1e-2


def shapes(synthetic: bool = False, suites=None) -> list:
    return list(SYNTHETIC if synthetic else SHAPE_SUITES.shapes(driver.arch(), suites))


def cases(synthetic: bool = False, suites=None) -> list[Case]:
    return [
        Case(op=OP, arch=driver.arch(), dtype=str(DTYPES[shape.dtype]).removeprefix("torch."), shape=tuple(shape[:-1]),
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
    from triton.tlx.ops import mla_window_attn

    tokens, seqs, d_model, heads, ranks, window = case.shape
    x, cu_seqlens, w = inputs(tokens, seqs, d_model, heads, ranks, window, getattr(torch, case.dtype))
    block_mask = window_block_mask(cu_seqlens, tokens, window)
    compiled = torch.compile(reference, dynamic=False)

    def tlx_call():
        return mla_window_attn(x, cu_seqlens, w["norm"], w["q"], w["kv"], w["gate"], num_heads=heads,
                               window_size=window, space=space)

    ref_fn = _graphed(lambda: compiled(x, w, heads, block_mask))
    ref_fn()  # compiled and captured here, ahead of the harness's cold-compile pass, which times only the op
    return Prepared(
        tlx_fn=_graphed(tlx_call),
        ref_fn=ref_fn,
        flop_count=flops(tokens, seqs, d_model, heads, ranks, window),
        # Eager FlexAttention materializes the full score matrix; the check uses the compiled reference too.
        check=lambda: close_enough(tlx_call(), compiled(x, w, heads, block_mask), REL_PRECISION),
    )


supported, default_json, run, main = driver.bind(sys.modules[__name__])

if __name__ == "__main__":
    raise SystemExit(main())
