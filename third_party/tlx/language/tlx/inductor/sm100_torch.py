"""TorchTLX SM100 MM callable for tests and benchmarks."""

from __future__ import annotations

import functools

import torch
from torch._inductor import config


@functools.lru_cache(maxsize=None)
def _compiled(mode):
    # INVARIANT: Each mode needs a distinct code object and Dynamo cache entry.

    def f(x, y):
        return x @ y

    return torch.compile(f, dynamic=False)


def mm(a, b, *, mode="allow"):
    """Compile ``a @ b`` with TLX allowed or forced as an Inductor candidate."""
    with config.patch({"triton.tlx_mode": mode}):
        return _compiled(mode)(a, b)


def ref(a, b):
    """Compile ``a @ b`` with TLX disabled."""
    with config.patch({"triton.tlx_mode": None}):
        return _compiled(None)(a, b)
