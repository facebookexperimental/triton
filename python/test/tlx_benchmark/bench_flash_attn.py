"""Perf and compile-time guardrail for `tlx.ops.flash_attn`, against SDPA.

Forward and backward are separate cases: the backward is 2.5x the FLOPs, is a
different kernel, and is where most of the recent work went, so folding it into
the forward's number would hide it.
"""

from __future__ import annotations

import pathlib
import sys

import torch

from triton.tlx.ops.kernels.flash_attn._shapes import FOCUS as SHAPE_SUITES
from triton.tlx.ops.kernels.flash_attn._shapes import FORWARD_ONLY, SYNTHETIC, label, qkv
from triton.tlx.ops.kernels.flash_attn._shapes import flops as flops

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _harness import Case, driver  # noqa: E402

OP = "flash_attn"
REF_NAME = "torch.nn.functional.scaled_dot_product_attention"
DEFAULT_SPACE = "heuristic"
#: Which SDPA kernel torch would dispatch. Earns a column because the headline
#: speedup is unreadable without it: 1.4x over `math` is a broken benchmark,
#: 1.4x over cuDNN is a result.
EXTRA_COLUMNS = (("ref bknd", "ref_backend"), )

#: A full autotune on a cold cache is minutes, not seconds. The default 120s cap
#: is calibrated for mm's heuristic path and would fail every case here on
#: compile alone, which says nothing about the kernel.
COMPILE_CAP_S = 900.0

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
DIRECTIONS = ("fwd", "bwd")

#: atol tracks the magnitude of the result, not of the element compared. Same
#: values as `test_flash_attn.py`, so a case cannot pass there and fail here.
REL_PRECISION = {"float16": 1e-3, "bfloat16": 8e-3}


def shapes(synthetic: bool = False, suites=None) -> list:
    arch = driver.arch()
    entries = SYNTHETIC if synthetic else SHAPE_SUITES.shapes(arch, suites)
    # The shared synthetic list mirrors the original NVIDIA correctness cases,
    # which are FP16. gfx950 FlashAttention is deliberately BF16-only, so keep
    # the same small shapes while selecting the dtype accepted by its catalog
    # entry instead of turning every synthetic case into an InvalidInput row.
    if synthetic and arch == "gfx950":
        entries = (entry._replace(dtype="bf16") for entry in entries)
    return list(entries)


def cases(synthetic: bool = False, suites=None) -> list[Case]:
    return [
        Case(op=OP, arch=driver.arch(), dtype=str(DTYPES[entry[5]]).removeprefix("torch."), shape=tuple(entry[:5]),
             direction=direction, label=label(*entry, direction))
        for entry in shapes(synthetic, suites)
        for direction in (("fwd", ) if entry in FORWARD_ONLY else DIRECTIONS)
    ]


def ref_backend(case: Case) -> str:
    """Which kernel `scaled_dot_product_attention` would pick for this case.

    A capability probe in the dispatcher's own preference order, not a trace of
    the call -- torch exposes no "which one did you just run".
    """
    from torch.backends.cuda import (SDPAParams, can_use_cudnn_attention, can_use_efficient_attention,
                                     can_use_flash_attention)

    Z, H, N_CTX, HEAD_DIM, causal = case.shape
    q, k, v = qkv(Z, H, N_CTX, HEAD_DIM, getattr(torch, case.dtype), requires_grad=case.direction == "bwd")
    params = SDPAParams(q, k, v, None, 0.0, causal, False)
    for name, usable, enabled in (
        ("cudnn", can_use_cudnn_attention, torch.backends.cuda.cudnn_sdp_enabled),
        ("flash", can_use_flash_attention, torch.backends.cuda.flash_sdp_enabled),
        ("efficient", can_use_efficient_attention, torch.backends.cuda.mem_efficient_sdp_enabled),
    ):
        try:
            if enabled() and usable(params, False):
                return name
        except RuntimeError:  # a probe may reject the params outright
            continue
    return "math"


supported, default_json, run, main = driver.bind(sys.modules[__name__])

if __name__ == "__main__":
    raise SystemExit(main())
