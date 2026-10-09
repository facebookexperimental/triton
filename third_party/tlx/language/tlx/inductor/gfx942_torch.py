"""TorchTLX gfx942 MM callable for tests and benchmarks."""

from __future__ import annotations

import functools

import torch
from torch._inductor import config


@functools.lru_cache(maxsize=None)
def _compiled(mode):
    def f(x, y):
        return x @ y

    return torch.compile(f, dynamic=False)


def mm(a, b, *, mode="allow"):
    """Run ``torch.mm`` with the gfx942 TorchTLX template enabled."""
    settings = {"triton.tlx_mode": mode, "max_autotune": True}
    if mode == "force":
        settings["max_autotune_gemm_backends"] = "TRITON"
    with config.patch(settings):
        return _compiled(mode)(a, b)


def ref(a, b):
    """Run the same graph through stock Inductor with TLX disabled."""
    with config.patch({"triton.tlx_mode": None}):
        return _compiled(None)(a, b)
