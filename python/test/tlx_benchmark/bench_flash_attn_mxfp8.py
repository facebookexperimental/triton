"""Perf and compile-time reporting for ``tlx.ops.flash_attn_mxfp8``."""

from __future__ import annotations

import pathlib
import sys

import torch
import torch.nn.functional as F

from triton.tlx.ops.kernels.flash_attn_mxfp8._shapes import FOCUS as SHAPE_SUITES
from triton.tlx.ops.kernels.flash_attn_mxfp8._shapes import SYNTHETIC, flops, label, qkv

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _harness import Case, Prepared, driver  # noqa: E402

OP = "flash_attn_mxfp8"
REF_NAME = "torch.nn.functional.scaled_dot_product_attention"
DEFAULT_SPACE = "full"
EXTRA_COLUMNS = ()
COLD_COMPILE = "first"
COMPILE_CAP_S = 900.0

DTYPES = {"bf16": torch.bfloat16}
DIRECTIONS = ("fwd", "bwd")


def _close_enough(out, ref) -> tuple[bool, str]:
    try:
        torch.testing.assert_close(out, ref, atol=0.2, rtol=0)
    except AssertionError as mismatch:
        return False, f"output does not match the reference: {str(mismatch).splitlines()[0]}"
    return True, ""


def shapes(synthetic: bool = False, suites=None) -> list:
    return list(SYNTHETIC if synthetic else SHAPE_SUITES.shapes(driver.arch(), suites))


def cases(synthetic: bool = False, suites=None) -> list[Case]:
    return [
        Case(
            op=OP,
            arch=driver.arch(),
            dtype=str(DTYPES[entry[5]]).removeprefix("torch."),
            shape=tuple(entry[:5]),
            direction=direction,
            label=label(*entry, direction),
        ) for entry in shapes(synthetic, suites) for direction in DIRECTIONS
    ]


def prepare(case: Case, space: str) -> Prepared:
    from triton.tlx.ops import flash_attn_mxfp8

    batch, heads, context, head_dim, causal = case.shape
    backward = case.direction == "bwd"
    dtype = getattr(torch, case.dtype)
    q, k, v = qkv(batch, heads, context, head_dim, dtype, requires_grad=backward)

    tlx_fwd = lambda: flash_attn_mxfp8(q, k, v, causal=causal, sm_scale=0.5, space=space)  # noqa: E731
    ref_fwd = lambda: F.scaled_dot_product_attention(  # noqa: E731
        q, k, v, is_causal=causal, scale=0.5)

    if not backward:
        return Prepared(
            tlx_fn=tlx_fwd,
            ref_fn=ref_fwd,
            flop_count=flops(*case.shape, "fwd"),
            check=lambda: _close_enough(tlx_fwd(), ref_fwd()),
            cap_s=COMPILE_CAP_S,
        )

    tlx_out, ref_out = tlx_fwd(), ref_fwd()
    do = torch.randn_like(tlx_out)
    return Prepared(
        tlx_fn=lambda: tlx_out.backward(do, retain_graph=True),
        ref_fn=lambda: ref_out.backward(do, retain_graph=True),
        flop_count=flops(*case.shape, "bwd"),
        grad_to_none=[q, k, v],
        check=None,
        cap_s=COMPILE_CAP_S,
    )


supported, default_json, run, main = driver.bind(sys.modules[__name__])

if __name__ == "__main__":
    raise SystemExit(main())
