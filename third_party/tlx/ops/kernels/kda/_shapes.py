from __future__ import annotations

from typing import NamedTuple

from .._shape_suites import FocusRegistry, FocusSuite


class KDAShape(NamedTuple):
    batch: int
    tokens: int
    heads: int
    head_dim: int
    dtype: str


SYNTHETIC: tuple[KDAShape, ...] = (KDAShape(2, 64, 2, 128, "bf16"), )

# TODO: Replace placeholders with captured shapes.
SM100_1 = FocusSuite(
    name="sm100_1",
    op="kimi_delta_attention",
    shapes=(
        KDAShape(4, 4096, 8, 128, "bf16"),
        KDAShape(4, 4096, 16, 128, "bf16"),
        KDAShape(2, 8192, 8, 128, "bf16"),
        KDAShape(8, 2048, 8, 128, "bf16"),
    ),
)
FOCUS_SUITES = (SM100_1, )
DEFAULT_SUITES = {"sm100": ("sm100_1", )}
FOCUS = FocusRegistry("kimi_delta_attention", FOCUS_SUITES, DEFAULT_SUITES)
CORRECTNESS_SHAPES = tuple(dict.fromkeys((*SYNTHETIC, *FOCUS.all_shapes())))

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
