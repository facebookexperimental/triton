"""Perf and compile-time reporting for ``tlx.ops.qk_norm_rope_kv_cache``.

The reference is the same computation (head RMSNorm, RoPE, FP8 cast, paged cache scatter) in eager torch compiled by
``torch.compile``: what a model gets without a fused kernel. The op is memory bound, so the case's ``flop_count`` is
its compulsory traffic in bytes (q, k, v, positions, table rows and slots read once; q and the cache rows written once)
and the ``TF/s`` columns read as TB/s of achieved bandwidth. Both sides are timed as the replay of a HIP graph holding
one call, as in serving, where a decode call costs less GPU time than its Python launch.
"""

from __future__ import annotations

import pathlib
import sys

import torch
from triton.tlx.ops.kernels.fused.qk_norm_rope_kv_cache._shapes import FOCUS as SHAPE_SUITES
from triton.tlx.ops.kernels.fused.qk_norm_rope_kv_cache._shapes import (SYNTHETIC, QKNormRopeShape, inputs, label,
                                                                        reference)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _harness import Case, Prepared, driver  # noqa: E402

OP = "qk_norm_rope_kv_cache"
REF_NAME = "torch.compile of the eager norm, RoPE and cache scatter"
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


def traffic_bytes(shape: QKNormRopeShape) -> int:
    T, HQ, HK, D = shape.tokens, shape.q_heads, shape.kv_heads, shape.head_dim
    total = T * (HQ + 2 * HK) * D * 2 + T * HQ * D * (1 if shape.q_fp8 else 2) + T * 2 * HK * D + T * 8
    if shape.rope:
        total += T * (8 + D * 4)
    if shape.remap:
        total += T * 8
    return total


def prepare(case: Case, space: str) -> Prepared:
    from triton.tlx.ops import qk_norm_rope_kv_cache

    shape = QKNormRopeShape(*case.shape)
    args = inputs(shape)
    ref_compiled = torch.compile(reference, dynamic=False)
    q_out = torch.empty(shape.tokens, shape.q_heads * shape.head_dim, device="cuda",
                        dtype=torch.float8_e4m3fn if shape.q_fp8 else torch.bfloat16)

    tlx_fwd = lambda: qk_norm_rope_kv_cache(**args, q_out=q_out, space=space)  # noqa: E731
    ref_fwd = lambda: ref_compiled(**args, inplace=True)  # noqa: E731

    def check():
        k_cache, v_cache = args["k_cache"].clone(), args["v_cache"].clone()
        got = qk_norm_rope_kv_cache(**dict(args, k_cache=k_cache, v_cache=v_cache))
        ref = reference(**args)
        for g, r, name in ((got, ref[0], "q"), (k_cache, ref[1], "k_cache"), (v_cache, ref[2], "v_cache")):
            gb, rb = g.view(torch.uint8), r.view(torch.uint8)
            # The reference sums each head in torch's order, so an element may round one step apart.
            if (gb != rb).float().mean() > 0.01:
                return False, f"{name} differs from the reference in {int((gb != rb).sum())} bytes"
        return True, ""

    traffic = traffic_bytes(shape)
    ref_fn = _graphed(ref_fwd)
    ref_fn()  # compiled and captured here, ahead of the harness's cold-compile pass, which times only the op
    return Prepared(tlx_fn=_graphed(tlx_fwd), ref_fn=ref_fn, flop_count=traffic, check=check,
                    extra={"mb": round(traffic / 1e6, 2)})


supported, default_json, run, main = driver.bind(sys.modules[__name__])

if __name__ == "__main__":
    raise SystemExit(main())
