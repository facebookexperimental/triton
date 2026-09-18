"""torchTLX mm: `a @ b` through torch.compile with the TLX Inductor template.
"""

from __future__ import annotations

import functools

import torch
from torch._inductor import config

from ._shapes import SM100_FOCUS

PERF_SHAPES = SM100_FOCUS


@functools.lru_cache(maxsize=None)
def _compiled(mode):
    # One code object per mode, so dynamo caches the two modes separately
    # instead of reusing whichever artifact was compiled first.
    def f(x, y):
        return x @ y

    return torch.compile(f, dynamic=False)


def mm(a, b, *, mode="allow"):
    """`a @ b` through the TLX Inductor template.

    "allow" is what a torchTLX user gets: TLX competes with aten and the stock
    Triton templates and may lose. "force" leaves TLX as the only candidate,
    which is what the perf suite measures and what the tests assert on.
    """
    with config.patch({"triton.tlx_mode": mode}):
        return _compiled(mode)(a, b)


def ref(a, b):
    """`a @ b` through stock Inductor -- aten or its own Triton template."""
    with config.patch({"triton.tlx_mode": None}):
        return _compiled(None)(a, b)
