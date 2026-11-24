"""Perf and compile-time reporting for ``tlx.ops.moe_mxfp4_sort_quant``.

The reference is ``torch.compile`` of a vectorized PyTorch implementation of the same contract (expert-masked slot
sort, per-row MXFP4 quantization, scale tile sort; the same bytes as the op); a serving stack runs the same steps as
five or six launches of its own kernels (for example AITER's ``moe_sorting`` and ``moe_mxfp4_sort`` plus a row
quantization). ``flop_count`` is the compulsory traffic in bytes (bf16 input, packed output and scales), so the
``TF/s`` columns read as TB/s. Both sides are timed as the replay of a HIP graph holding one call, as in serving.
"""

from __future__ import annotations

import pathlib
import sys

import torch
from triton.tlx.ops.kernels.fused.moe_mxfp4_sort_quant._shapes import FOCUS as SHAPE_SUITES
from triton.tlx.ops.kernels.fused.moe_mxfp4_sort_quant._shapes import (SYNTHETIC, MoeMxfp4SortQuantShape, inputs, label,
                                                                       reference_torch)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _harness import Case, Prepared, driver  # noqa: E402

OP = "moe_mxfp4_sort_quant"
REF_NAME = "torch.compile of the slot sort, MXFP4 row quantization and scale sort (same bytes)"
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
    from triton.tlx.ops import moe_mxfp4_sort_quant

    shape = MoeMxfp4SortQuantShape(*case.shape)
    kw = inputs(shape)
    ref_compiled = torch.compile(reference_torch, dynamic=False)
    tlx_fwd = lambda: moe_mxfp4_sort_quant(**kw, space=space)  # noqa: E731
    ref_fwd = lambda: ref_compiled(**kw)  # noqa: E731

    def check():
        got, ref = tlx_fwd(), ref_fwd()
        n = int(ref[3][0])
        ok = (torch.equal(got[0], ref[0]) and torch.equal(got[3], ref[3]) and torch.equal(got[1][:n], ref[1][:n])
              and torch.equal(got[2][:n // 32], ref[2][:n // 32]))
        written = ref[4].view(-1) != 0  # scale bytes of real slots (padding rows are left unwritten by the op)
        ok = ok and torch.equal(got[4].view(-1)[written], ref[4].view(-1)[written])
        return ok, "" if ok else "differs from the PyTorch implementation"

    traffic = shape.tokens * shape.hidden * 2 + shape.tokens * shape.hidden // 2 + shape.tokens * shape.topk * (
        shape.hidden // 32)
    ref_fn = _graphed(ref_fwd)
    ref_fn()
    return Prepared(tlx_fn=_graphed(tlx_fwd), ref_fn=ref_fn, flop_count=traffic, check=check)


supported, default_json, run, main = driver.bind(sys.modules[__name__])

if __name__ == "__main__":
    raise SystemExit(main())
