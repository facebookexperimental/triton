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


def _backward_close_enough(tlx_out, ref_out, inputs, do) -> tuple[bool, str]:
    actual = torch.autograd.grad(tlx_out, inputs, do, retain_graph=True)
    expected = torch.autograd.grad(ref_out, inputs, do, retain_graph=True)
    for name, got, ref in zip(("dQ", "dK", "dV"), actual, expected):
        got, ref = got.float(), ref.float()
        if not torch.isfinite(got).all() or not torch.isfinite(ref).all():
            return False, f"{name} or its reference contains nonfinite values"
        cosine = F.cosine_similarity(got.flatten(), ref.flatten(), dim=0).item()
        ref_ms = ref.square().mean()
        relative_rms = ((got - ref).square().mean() / ref_ms.clamp_min(1e-30)).sqrt().item()
        if (ref_ms > 0 and cosine < 0.98) or relative_rms >= 0.15:
            return False, f"{name} cosine={cosine:.6f}, relative RMS error={relative_rms:.6f}"
    return True, ""


def shapes(synthetic: bool = False, suites=None) -> list:
    return list(SYNTHETIC if synthetic else SHAPE_SUITES.shapes(driver.arch(), suites))


def _directions(arch) -> tuple[str, ...]:
    """Backward cases exist only on an arch whose catalog entry implements them.

    Keep this catalog-driven so forward-only implementations on future
    architectures do not become failing backward benchmark cases.
    """
    if arch is None:
        return DIRECTIONS
    from triton.tlx.ops._catalog import CATALOG

    backward = any(spec.op == OP and spec.arch == arch and spec.supports_backward for spec in CATALOG)
    return DIRECTIONS if backward else ("fwd", )


def cases(synthetic: bool = False, suites=None) -> list[Case]:
    arch = driver.arch()
    return [
        Case(
            op=OP,
            arch=arch,
            dtype=str(DTYPES[entry[5]]).removeprefix("torch."),
            shape=tuple(entry[:5]),
            direction=direction,
            label=label(*entry, direction),
        ) for entry in shapes(synthetic, suites) for direction in _directions(arch)
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
        check=(lambda: _backward_close_enough(tlx_out, ref_out, (q, k, v), do)) if case.arch == "gfx950" else None,
        cap_s=COMPILE_CAP_S,
    )


supported, default_json, run, main = driver.bind(sys.modules[__name__])

if __name__ == "__main__":
    raise SystemExit(main())
