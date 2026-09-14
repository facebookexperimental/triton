"""Perf and compile-time guardrail for `tlx.ops.flash_attn`, against SDPA.

Forward and backward are separate cases: the backward is 2.5x the FLOPs, is a
different kernel, and is where most of the recent work went, so folding it into
the forward's number would hide it.
"""

from __future__ import annotations

import importlib
import pathlib
import sys

import torch
import torch.nn.functional as F

from triton.tlx.ops.kernels.flash_attn._shapes import flops, label, qkv

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _harness import Case, Prepared, close_enough, driver  # noqa: E402

OP = "flash_attn"
REF_NAME = "torch.nn.functional.scaled_dot_product_attention"
#: No `heuristic_config` yet, so the op's own default is a full autotune -- see
#: the `tlx.ops` module docstring. Measuring "smoke" instead would measure a
#: path no user takes, so the suite pays the compile time and raises the cap.
DEFAULT_SPACE = "full"
#: Which SDPA kernel torch would dispatch. Earns a column because the headline
#: speedup is unreadable without it: 1.4x over `math` is a broken benchmark,
#: 1.4x over cuDNN is a result.
EXTRA_COLUMNS = (("ref bknd", "ref_backend"), )

#: One fresh-cache first call per direction, not per case: the cold pass
#: compiles this op's whole autotune space, and "does a first call take too
#: long" is not a per-shape question.
COLD_COMPILE = "first"

#: A full autotune on a cold cache is minutes, not seconds. The default 120s cap
#: is calibrated for mm's heuristic path and would fail every case here on
#: compile alone, which says nothing about the kernel.
COMPILE_CAP_S = 900.0

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
DIRECTIONS = ("fwd", "bwd")

#: atol tracks the magnitude of the result, not of the element compared. Same
#: values as `test_flash_attn.py`, so a case cannot pass there and fail here.
REL_PRECISION = {"float16": 1e-3, "bfloat16": 8e-3}


def shapes(synthetic: bool = False) -> list[list]:
    if synthetic:
        from triton.tlx.ops.kernels.flash_attn._shapes import SYNTHETIC

        return list(SYNTHETIC)
    return list(importlib.import_module(f"triton.tlx.ops.kernels.flash_attn.{driver.arch()}").PERF_SHAPES)


def cases(synthetic: bool = False) -> list[Case]:
    return [
        Case(op=OP, arch=driver.arch(), dtype=str(DTYPES[entry[5]]).removeprefix("torch."), shape=tuple(entry[:5]),
             direction=direction, label=label(*entry, direction))
        for entry in shapes(synthetic)
        for direction in DIRECTIONS
    ]


def _ref_backend(q, k, v, causal) -> str:
    """Which kernel `scaled_dot_product_attention` would pick for these inputs.

    A capability probe in the dispatcher's own preference order, not a trace of
    the call -- torch exposes no "which one did you just run". Good enough for
    the question it answers, which is whether the ratio is against a real
    attention kernel or against the math fallback.
    """
    from torch.backends.cuda import (SDPAParams, can_use_cudnn_attention, can_use_efficient_attention,
                                     can_use_flash_attention)

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


def prepare(case: Case, space: str) -> Prepared:
    from triton.tlx.ops import flash_attn as tlx_flash_attn

    Z, H, N_CTX, HEAD_DIM, causal = case.shape
    backward = case.direction == "bwd"
    dtype = getattr(torch, case.dtype)
    q, k, v = qkv(Z, H, N_CTX, HEAD_DIM, dtype, requires_grad=backward)

    tlx_fwd = lambda: tlx_flash_attn(q, k, v, causal=causal, arch=driver.arch(), space=space)  # noqa: E731
    ref_fwd = lambda: F.scaled_dot_product_attention(  # noqa: E731
        q, k, v, is_causal=causal, scale=HEAD_DIM**-0.5)
    extra = {"ref_backend": _ref_backend(q, k, v, causal)}

    if not backward:
        return Prepared(tlx_fn=tlx_fwd, ref_fn=ref_fwd, flop_count=flops(*case.shape, "fwd"),
                        check=lambda: close_enough(tlx_fwd(), ref_fwd(), REL_PRECISION[case.dtype]), extra=extra,
                        cap_s=COMPILE_CAP_S)

    # Build both graphs once, here, so the measured window is the backward and
    # nothing else. `retain_graph` because the same graph is walked several
    # hundred times; `grad_to_none` because otherwise each iteration accumulates
    # into the last one's gradients.
    tlx_out, ref_out = tlx_fwd(), ref_fwd()
    do = torch.randn_like(tlx_out)
    return Prepared(
        tlx_fn=lambda: tlx_out.backward(do, retain_graph=True),
        ref_fn=lambda: ref_out.backward(do, retain_graph=True),
        flop_count=flops(*case.shape, "bwd"),
        grad_to_none=[q, k, v],
        # No accuracy check: the two `.backward()` calls write to the same
        # `.grad` tensors, so comparing them here would need a third set of
        # operands and a second graph. `test_flash_attn.py::test_flash_attn_bwd`
        # already covers the numerics.
        check=None,
        extra=extra,
        cap_s=COMPILE_CAP_S,
    )


supported, default_json, run, main = driver.bind(sys.modules[__name__])

if __name__ == "__main__":
    raise SystemExit(main())
