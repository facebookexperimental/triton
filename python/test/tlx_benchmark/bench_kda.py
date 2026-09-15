"""Perf guardrail for `tlx.ops.kimi_delta_attention`. No reference.

The only KDA reference in the tree is the per-chunk python loop in
`test_kimi_delta_attention.py`, which does a `solve_triangular` per chunk and
cannot be timed. So there is no `speedup` column, and the perf gate is the
absolute `FLOOR_TFLOPS` -- currently None, i.e. this op reports rather than
gates until a clean run exists to seed the floor from.

TODO: `fla` (flash-linear-attention) ships `chunk_kda`, which would be a real
reference. It is not a dependency of this suite and the README promises torch
and triton only, so adding it means an optional import that skips the reference
when absent -- deliberately deferred rather than done quietly.
"""

from __future__ import annotations

import importlib
import pathlib
import sys

import torch

from triton.tlx.ops.kernels.kda._shapes import CHUNK, FLOOR_TFLOPS, flops, inputs, label

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _harness import Case, Prepared, driver  # noqa: E402

OP = "kimi_delta_attention"
REF_NAME = ""  # no runnable reference; see the module docstring
DEFAULT_SPACE = "full"
#: TFLOP/s is a poor headline for a chunked recurrence -- see `_shapes.flops` --
#: so the honest rate gets the column.
EXTRA_COLUMNS = (("Mtok/s", "mtokens_per_s"), )
#: One fresh-cache first call per direction, not per case: the cold pass
#: compiles this op's whole autotune space, and "does a first call take too
#: long" is not a per-shape question.
COLD_COMPILE = "first"

COMPILE_CAP_S = 900.0

#: bf16 only: the catalog entry admits fp16 but the L1 suite only covers bf16,
#: and this suite does not measure what nothing checks.
DTYPES = {"bf16": torch.bfloat16}
DIRECTIONS = ("fwd", "bwd")


def shapes(synthetic: bool = False) -> list[list]:
    if synthetic:
        from triton.tlx.ops.kernels.kda._shapes import SYNTHETIC

        return list(SYNTHETIC)
    return list(importlib.import_module(f"triton.tlx.ops.kernels.kda.{driver.arch()}").PERF_SHAPES)


def cases(synthetic: bool = False) -> list[Case]:
    return [
        Case(op=OP, arch=driver.arch(), dtype=str(DTYPES[entry[4]]).removeprefix("torch."), shape=tuple(entry[:4]),
             direction=direction, label=label(*entry, direction))
        for entry in shapes(synthetic)
        for direction in DIRECTIONS
    ]


def prepare(case: Case, space: str) -> Prepared:
    from triton.tlx.ops import kimi_delta_attention as tlx_kda

    B, T, H, head_dim = case.shape
    backward = case.direction == "bwd"
    dtype = getattr(torch, case.dtype)
    q, k, v, g, beta, cu_seqlens = inputs(B, T, H, head_dim, dtype, requires_grad=backward)

    # The op returns a TritonBench-compatible `(output, None)` pair; the index
    # is free and keeps the closure returning something differentiable.
    tlx_fwd = lambda: tlx_kda(  # noqa: E731
        q, k, v, g, beta, scale=1.0, cu_seqlens=cu_seqlens, arch=driver.arch(), space=space)[0]

    flop_count = flops(*case.shape, case.direction)
    extra = {"chunks": (B * T) // CHUNK, "chunk": CHUNK}

    if not backward:
        return Prepared(tlx_fn=tlx_fwd, ref_fn=None, flop_count=flop_count, floor_tflops=FLOOR_TFLOPS,
                        check=None,  # no runnable reference to check against
                        extra=extra, cap_s=COMPILE_CAP_S)

    out = tlx_fwd()
    do = torch.randn_like(out)
    return Prepared(
        tlx_fn=lambda: out.backward(do, retain_graph=True),
        ref_fn=None,
        flop_count=flop_count,
        grad_to_none=[q, k, v, g, beta],
        floor_tflops=FLOOR_TFLOPS,
        check=None,
        extra=extra,
        cap_s=COMPILE_CAP_S,
    )


def annotate(result) -> None:
    """The driver's post-measurement hook: this rate needs the measured mean.

    Recovered from the throughput rather than timed separately, because
    `flop_count / mean` is exactly the latency `measure` converted away.
    """
    if result.tlx and result.flop_count:
        latency_s = result.flop_count / (result.tlx.mean * 1e12)
        tokens = result.case.shape[0] * result.case.shape[1]
        result.extra["mtokens_per_s"] = tokens / latency_s / 1e6


supported, default_json, run, main = driver.bind(sys.modules[__name__])

if __name__ == "__main__":
    raise SystemExit(main())
