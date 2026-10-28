"""TorchTLX gfx942 AddMM callable for tests and benchmarks."""

from __future__ import annotations

import functools

import torch
from torch._inductor import config


@functools.lru_cache(maxsize=None)
def _compiled(mode):

    def f(bias, a, b):
        return torch.addmm(bias, a, b)

    return torch.compile(f, dynamic=False)


def addmm(bias, a, b, *, mode="allow"):
    """Run ``torch.addmm`` with the gfx942 TorchTLX template enabled."""
    settings = {"triton.tlx_mode": mode, "max_autotune": True}
    if mode == "force":
        settings["max_autotune_gemm_backends"] = "TRITON"
    with config.patch(settings):
        return _compiled(mode)(bias, a, b)


def ref(bias, a, b):
    """Run the same graph through stock Inductor with TLX disabled."""
    with config.patch({"triton.tlx_mode": None}):
        return _compiled(None)(bias, a, b)
