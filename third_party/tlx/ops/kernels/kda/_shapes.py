from __future__ import annotations

from typing import NamedTuple

from .._shape_suites import FocusRegistry, FocusSuite


class KDAShape(NamedTuple):
    batch: int
    tokens: int
    heads: int
    head_dim: int
    dtype: str


class KDAPrefillShape(NamedTuple):
    total_tokens: int
    sequences: int
    heads: int
    key_dim: int
    value_dim: int
    dtype: str


class KDADecodeShape(NamedTuple):
    batch: int
    heads: int
    key_dim: int
    value_dim: int
    dtype: str


SYNTHETIC: tuple[KDAShape, ...] = (KDAShape(2, 64, 2, 128, "bf16"), )

# TODO: Replace placeholders with captured shapes.
SM100_FOCUS_SUITE = FocusSuite(
    name="sm100_baseline",
    op="kimi_delta_attention",
    shapes=(
        KDAShape(4, 4096, 8, 128, "bf16"),
        KDAShape(4, 4096, 16, 128, "bf16"),
        KDAShape(2, 8192, 8, 128, "bf16"),
        KDAShape(8, 2048, 8, 128, "bf16"),
    ),
)
FOCUS_SUITES = (SM100_FOCUS_SUITE, )

GFX950_PREFILL_SYNTHETIC: tuple[KDAPrefillShape, ...] = (KDAPrefillShape(64, 1, 4, 128, 128, "bf16"), )

GFX950_PREFILL_FOCUS_SUITE = FocusSuite(
    name="gfx950_prefill_baseline",
    op="kda_paged_prefill",
    shapes=(
        KDAPrefillShape(4096, 1, 4, 128, 128, "bf16"),
        KDAPrefillShape(4096, 4, 4, 128, 128, "bf16"),
        KDAPrefillShape(131072, 1, 4, 128, 128, "bf16"),
        KDAPrefillShape(131072, 8, 4, 128, 128, "bf16"),
        KDAPrefillShape(4096, 1, 12, 128, 128, "bf16"),
        KDAPrefillShape(4096, 4, 12, 128, 128, "bf16"),
        KDAPrefillShape(131072, 1, 12, 128, 128, "bf16"),
        KDAPrefillShape(131072, 8, 12, 128, 128, "bf16"),
    ),
)
PREFILL_FOCUS_SUITES = (GFX950_PREFILL_FOCUS_SUITE, )

GFX950_DECODE_SYNTHETIC: tuple[KDADecodeShape, ...] = (KDADecodeShape(1, 4, 128, 128, "bf16"), )

GFX950_DECODE_FOCUS_SUITE = FocusSuite(
    name="gfx950_decode_baseline",
    op="kda_recurrent_decode",
    shapes=tuple(KDADecodeShape(batch, heads, 128, 128, "bf16") for heads in (4, 12) for batch in (1, 2, 4, 8, 16, 32)),
)
DECODE_FOCUS_SUITES = (GFX950_DECODE_FOCUS_SUITE, )

DEFAULT_SUITES = {"sm100": ("sm100_baseline", )}
PREFILL_DEFAULT_SUITES = {"gfx950": ("gfx950_prefill_baseline", )}
DECODE_DEFAULT_SUITES = {"gfx950": ("gfx950_decode_baseline", )}

FOCUS = FocusRegistry("kimi_delta_attention", FOCUS_SUITES, DEFAULT_SUITES)
PREFILL_FOCUS = FocusRegistry("kda_paged_prefill", PREFILL_FOCUS_SUITES, PREFILL_DEFAULT_SUITES)
DECODE_FOCUS = FocusRegistry("kda_recurrent_decode", DECODE_FOCUS_SUITES, DECODE_DEFAULT_SUITES)

SM100_FOCUS = FOCUS.shapes("sm100")
GFX950_PREFILL_FOCUS = PREFILL_FOCUS.shapes("gfx950")
GFX950_DECODE_FOCUS = DECODE_FOCUS.shapes("gfx950")

CHUNK = 64

#: Absolute TFLOP/s gate -- the only perf gate available, since KDA has no
#: runnable reference. None = report only, until a clean run exists to seed it.
FLOOR_TFLOPS = None


def inputs(B, T, H, head_dim, dtype, requires_grad=False, device="cuda"):
    """Packed `[1, B*T, H, D]` inputs plus `cu_seqlens`, as the op wants them.

    Mirrors `test_kimi_delta_attention.py::_inputs`. The normalization and the
    negative softplus are load-bearing: the delta rule diverges on plain random
    inputs, which would time arithmetic no real caller produces.
    """
    import torch
    import torch.nn.functional as F

    gen = torch.Generator(device=device).manual_seed(0)

    def rn(*shape):
        return torch.randn(*shape, generator=gen, device=device, dtype=torch.float32)

    total = B * T
    q = F.normalize(rn(1, total, H, head_dim), dim=-1).to(dtype).requires_grad_(requires_grad)
    k = F.normalize(rn(1, total, H, head_dim), dim=-1).to(dtype).requires_grad_(requires_grad)
    v = rn(1, total, H, head_dim).to(dtype).requires_grad_(requires_grad)
    g = (-F.softplus(rn(1, total, H, head_dim))).requires_grad_(requires_grad)
    beta = torch.sigmoid(rn(1, total, H)).requires_grad_(requires_grad)
    cu_seqlens = torch.arange(0, (B + 1) * T, T, device=device, dtype=torch.int64)
    return q, k, v, g, beta, cu_seqlens


def flops(B, T, H, HEAD_DIM, direction="fwd", chunk=CHUNK):
    """Approximate: NOT comparable to mm or attention, only to other KDA runs.

    Per chunk per (sequence, head), the matmuls in `sm100.py`'s docstring are
    four of `2*C*C*D` and three of `2*C*D*D`; the triangular inverse is not
    counted and the backward is taken as 2x for its two passes. The uncounted
    work is a real fraction of the runtime, so `mtokens_per_s` in
    `Result.extra` is the honest rate.
    """
    per_seq_head = 8 * T * chunk * HEAD_DIM + 6 * T * HEAD_DIM * HEAD_DIM
    total = B * H * per_seq_head
    if direction == "bwd":
        total *= 2.0
    return int(total)


def label(B, T, H, HEAD_DIM, dtype, direction="fwd") -> str:
    return (f"((), {{'dtype': '{dtype}', 'dir': '{direction}', "
            f"'B': '{B}', 'T': '{T}', 'H': '{H}', 'HEAD_DIM': '{HEAD_DIM}'}})")
