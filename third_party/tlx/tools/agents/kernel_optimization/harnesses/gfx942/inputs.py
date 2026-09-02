"""Operand construction for the gfx942 harness and its two child processes.

Shared so the in-process gate, the default-environment correctness check and
the rocprofv3 traced dispatch all run the same kernel on the same data. They
previously built inputs three times, and the ATT child's copy had drifted to a
different distribution.

The distribution matches ``testing/test_correctness.py``'s GEMM case, which is
the tolerance the repository's real correctness suite applies. Large unscaled
random inputs plus the default fp16 ``assert_close`` band let schedule changes
through the agent gate that that suite rejects.
"""

from __future__ import annotations

from typing import Any


def make_inputs(case: dict[str, Any]):
    """``(a, b)`` for ``case``, on the case's device and dtype."""
    import torch

    parameters = case["parameters"]
    device = parameters.get("device", "cuda")
    dtype = getattr(torch, parameters.get("dtype", "float16"))
    m, n, k = int(parameters["m"]), int(parameters["n"]), int(parameters["k"])
    torch.manual_seed(int(parameters.get("seed", 0)))
    a = (torch.randn((m, k), device=device, dtype=dtype) + 1) / k
    b = (torch.randn((k, n), device=device, dtype=dtype) + 1) / k
    return a, b
