"""Perf and compile-time reporting for ``tlx.ops.moe_finalize``.

The op serves training after the combine on sm100, so cases take the ``(T, K, H)`` layout (with the expert mask when
the rank holds a subset of the experts). The forward reference is the finalize of that layout as one unfused Triton
kernel (a ``(K, H)`` tile per token), which gives the same bits. Each shape also has a backward case: the gradients of
all three inputs against autograd of ``torch.compile`` of the eager finalize.

The op is memory bound, so the case's ``flop_count`` is its compulsory traffic in bytes (live expert rows, norm weight,
routing weights and expert ids read once, output written once; the backward reads the expert rows and the output
gradient and writes the expert row gradients) and the ``TF/s`` columns read as TB/s of achieved bandwidth. Both sides
are timed as the replay of a graph holding one call.
"""

from __future__ import annotations

import pathlib
import sys

import torch
from triton.tlx.ops.kernels.moe_finalize._shapes import FOCUS as SHAPE_SUITES
from triton.tlx.ops.kernels.moe_finalize._shapes import (EPS, SYNTHETIC, MoeFinalizeShape, inputs, label, reference,
                                                         unfused_finalize, unpermute)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _harness import Case, Prepared, close_enough, driver  # noqa: E402

OP = "moe_finalize"
REF_NAME = "unfused Triton finalize; bwd: torch.compile of the eager finalize"
DEFAULT_SPACE = "heuristic"
EXTRA_COLUMNS = (("MB", "mb"), )
COLD_COMPILE = "first"


def shapes(synthetic: bool = False, suites=None) -> list:
    return list(SYNTHETIC if synthetic else SHAPE_SUITES.shapes(driver.arch(), suites))


def cases(synthetic: bool = False, suites=None) -> list[Case]:
    return [
        Case(op=OP, arch=driver.arch(), dtype="bfloat16", shape=tuple(shape), label=label(shape), direction=direction)
        for shape in shapes(synthetic, suites)
        for direction in ("fwd", "bwd")
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
    from triton.tlx.ops import moe_finalize

    shape = MoeFinalizeShape(*case.shape)
    data = inputs(shape)
    y, rows, w, gamma = data["expert_out"], data["row_index"], data["topk_weights"], data["norm_weight"]
    y = unpermute(y, rows)
    masked = shape.local < shape.experts
    layout = dict(topk_ids=data["topk_ids"], expert_mask=data["expert_mask"]) if masked else {}
    live = (rows >= 0).float() if masked else None
    if case.direction == "bwd":
        return _prepare_bwd(shape, y, w, gamma, layout, live, data["expert_out"].shape[0], space)
    out = torch.empty(shape.tokens, shape.H, device="cuda", dtype=torch.bfloat16)
    tlx_fwd = lambda: moe_finalize(y, w, norm_weight=gamma, eps=EPS, out=out, space=space, **layout)  # noqa: E731
    ref_fwd = lambda: unfused_finalize(y, w, gamma)  # noqa: E731

    def check():
        got = tlx_fwd()
        if not torch.equal(got, ref_fwd()):
            return False, "outputs differ from the unfused finalize"
        want = reference(y, w, gamma)
        if (got == want).float().mean() <= 0.99:
            return False, "fewer than 99% of the outputs equal the reference"
        if not torch.allclose(got.float(), want.float(), rtol=8e-3, atol=8e-3):
            return False, "outputs do not match the reference"
        return True, ""

    read = data["expert_out"].numel() * 2 + shape.tokens * shape.topk * 4
    traffic = read + shape.H * 2 + shape.tokens * shape.topk * 4 + shape.tokens * shape.H * 2
    ref_fn = _graphed(ref_fwd)
    ref_fn()  # compiled and captured here, ahead of the harness's cold-compile pass, which times only the op
    return Prepared(tlx_fn=_graphed(tlx_fwd), ref_fn=ref_fn, flop_count=traffic, check=check,
                    extra={"mb": round(traffic / 1e6, 2)})


def _prepare_bwd(shape, y, w, gamma, layout, live, live_rows, space):
    """Both sides run the forward once here and replay only the gradient of all three inputs."""
    from triton.tlx.ops import moe_finalize

    gout = torch.randn(shape.tokens, shape.H, device="cuda",
                       generator=torch.Generator(device="cuda").manual_seed(7)).bfloat16()
    leaves = [t.detach().clone().requires_grad_() for t in (y, w, gamma)]
    ref_leaves = [t.detach().clone().requires_grad_() for t in (y, w, gamma)]
    compiled = torch.compile(lambda a, b, c: reference(a, b, c, live=live), fullgraph=True)

    def side(fn, args):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            result = fn(*args)
        torch.cuda.current_stream().wait_stream(stream)
        return _graphed(lambda: torch.autograd.grad(result, args, gout, retain_graph=True))

    tlx_fn = side(lambda a, b, c: moe_finalize(a, b, norm_weight=c, eps=EPS, space=space, **layout), leaves)
    ref_fn = side(compiled, ref_leaves)
    ref_fn()

    def check():
        got = torch.autograd.grad(moe_finalize(*leaves[:2], norm_weight=leaves[2], eps=EPS, space=space, **layout),
                                  leaves, gout)
        want = torch.autograd.grad(compiled(*ref_leaves), ref_leaves, gout)
        for name, g, r in zip(("expert_out", "topk_weights", "norm_weight"), got, want):
            ok, note = close_enough(g.float(), r.float(), 0.02)
            if not ok:
                return False, f"{name}: {note}"
        return True, ""

    # The backward reads only the live expert rows and writes the gradient of every (token, slot) row.
    traffic = (live_rows * shape.H * 2 + y.numel() * 2 + shape.tokens * shape.H * 2 + shape.tokens * shape.topk * 8 +
               shape.H * 2)
    return Prepared(tlx_fn=tlx_fn, ref_fn=ref_fn, flop_count=traffic, check=check,
                    extra={"mb": round(traffic / 1e6, 2)})


supported, default_json, run, main = driver.bind(sys.modules[__name__])

if __name__ == "__main__":
    raise SystemExit(main())
