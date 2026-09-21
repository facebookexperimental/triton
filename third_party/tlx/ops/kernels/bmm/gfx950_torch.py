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
    only TLX candidates. Bmm needs max-autotune enabled to enter its template
    selection path, unlike the Blackwell ``mm`` lowering.
    """
    settings = {"triton.tlx_mode": mode, "max_autotune": True}
    if mode == "force":
        settings["max_autotune_gemm_backends"] = "TRITON"
    with config.patch(settings):
        return _compiled(mode)(a, b)


def ref(a, b):
    """Run the same graph through stock Inductor, with TLX disabled."""
    with config.patch({"triton.tlx_mode": None}):
        return _compiled(None)(a, b)
