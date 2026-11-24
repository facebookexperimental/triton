"""TorchTLX ``bmm`` through the gfx950 Inductor templates."""

from __future__ import annotations

import functools

import torch
from torch._inductor import config


@functools.lru_cache(maxsize=None)
def _compiled(mode):

    def f(a, b):
        return torch.bmm(a, b)

    return torch.compile(f, dynamic=False)


def bmm(a, b, *, mode="allow"):
    """Run ``torch.bmm`` with the gfx950 TLX templates enabled.

    ``allow`` makes TLX compete with stock Inductor choices; ``force`` retains
    only TLX candidates.
    """
    # torchTLX is independent of max_autotune: TLX configs come from the
    # template heuristics.
    with config.patch({"triton.tlx_mode": mode, "max_autotune": False, "max_autotune_gemm": False}):
        return _compiled(mode)(a, b)


def ref(a, b):
    """Run the same graph through stock Inductor, with TLX disabled."""
    with config.patch({"triton.tlx_mode": None}):
        return _compiled(None)(a, b)
