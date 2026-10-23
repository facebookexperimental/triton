from __future__ import annotations

from typing import NamedTuple

from .._shape_suites import FocusRegistry, FocusSuite


class FlashAttentionShape(NamedTuple):
    batch: int
    heads: int
    context: int
    head_dim: int
    causal: bool
    dtype: str


#: Identical to `test_flash_attn.py::SHAPES`.
SYNTHETIC: tuple[FlashAttentionShape, ...] = (
    FlashAttentionShape(1, 1, 256, 64, False, "fp16"),
    FlashAttentionShape(1, 2, 512, 64, True, "fp16"),
    FlashAttentionShape(2, 4, 1024, 64, False, "fp16"),
    FlashAttentionShape(2, 4, 1024, 128, False, "fp16"),
    FlashAttentionShape(2, 4, 1024, 128, True, "fp16"),
    FlashAttentionShape(4, 8, 2048, 128, True, "fp16"),
    FlashAttentionShape(1, 16, 4096, 128, False, "fp16"),
    FlashAttentionShape(2, 32, 2048, 64, False, "fp16"),
    FlashAttentionShape(4, 8, 512, 64, True, "fp16"),
    FlashAttentionShape(1, 1, 8192, 128, True, "fp16"),
)

FOCUS_SUITES = (
    FocusSuite(
        name="sm90_1",
        op="flash_attn",
        shapes=(
            FlashAttentionShape(4, 48, 1024, 128, False, "bf16"),
            FlashAttentionShape(4, 48, 2048, 128, True, "bf16"),
            FlashAttentionShape(4, 48, 4096, 128, False, "bf16"),
            FlashAttentionShape(4, 48, 4096, 128, True, "bf16"),
            FlashAttentionShape(4, 48, 8192, 128, True, "bf16"),
            FlashAttentionShape(4, 48, 4096, 128, False, "fp16"),
        ),
    ),
    # TODO: Replace placeholders with captured shapes.
    FocusSuite(
        name="sm100_1",
        op="flash_attn",
        shapes=(
            FlashAttentionShape(4, 32, 4096, 128, False, "bf16"),
            FlashAttentionShape(4, 32, 4096, 128, True, "bf16"),
            FlashAttentionShape(2, 32, 8192, 128, True, "bf16"),
            FlashAttentionShape(1, 16, 16384, 128, True, "bf16"),
            FlashAttentionShape(4, 32, 4096, 64, False, "bf16"),
            FlashAttentionShape(4, 32, 4096, 128, False, "fp16"),
        ),
    ),
    FocusSuite(
        name="gfx950_1",
        op="flash_attn",
        shapes=(
            FlashAttentionShape(1, 16, 4096, 64, False, "bf16"),
            FlashAttentionShape(16, 16, 1024, 128, False, "bf16"),
            FlashAttentionShape(16, 16, 1024, 128, True, "bf16"),
        ),
    ),
    # Self-attention captured from production ranking models (some sites run
    # fp16 there; gfx950 takes bf16 only).
    FocusSuite(
        name="gfx950_2",
        op="flash_attn",
        shapes=(
            FlashAttentionShape(16, 1, 200, 128, False, "bf16"),
            FlashAttentionShape(16, 1, 400, 128, False, "bf16"),
            FlashAttentionShape(16, 1, 2000, 128, False, "bf16"),
            FlashAttentionShape(22, 1, 100, 128, False, "bf16"),
            FlashAttentionShape(22, 1, 150, 128, False, "bf16"),
        ),
    ),
    # The same capture's sites whose head dims `tlx.ops.flash_attn` does not
    # accept yet, so no default selects them.
    FocusSuite(
        name="gfx950_3",
        op="flash_attn",
        shapes=(
            FlashAttentionShape(44, 1, 200, 192, False, "bf16"),
            FlashAttentionShape(60, 1, 200, 192, False, "bf16"),
            FlashAttentionShape(22, 1, 200, 256, False, "bf16"),
            FlashAttentionShape(22, 1, 300, 256, False, "bf16"),
            FlashAttentionShape(22, 1, 1500, 256, False, "bf16"),
            FlashAttentionShape(22, 1, 2000, 256, False, "bf16"),
            FlashAttentionShape(22, 1, 3200, 256, False, "bf16"),
            FlashAttentionShape(78, 1, 200, 256, False, "bf16"),
        ),
    ),
    FocusSuite(
        name="gfx950_all",
        op="flash_attn",
        includes=("gfx950_1", "gfx950_2"),
    ),
)

DEFAULT_SUITES = {
    "sm90": ("sm90_1", ),
    "sm100": ("sm100_1", ),
    "gfx950": ("gfx950_all", ),
}
FOCUS = FocusRegistry("flash_attn", FOCUS_SUITES, DEFAULT_SUITES)
CORRECTNESS_SHAPES = tuple(dict.fromkeys((*SYNTHETIC, *FOCUS.all_shapes())))
#: Inference captures: production never runs their backward, and no backend's
#: backward accepts their sequence lengths, so the perf suite times the forward.
FORWARD_ONLY = frozenset((*FOCUS.resolved_shapes("gfx950_2"), *FOCUS.resolved_shapes("gfx950_3")))


def qkv(Z, H, N_CTX, HEAD_DIM, dtype, requires_grad=False, device="cuda"):
    import torch

    return [
        torch.randn((Z, H, N_CTX, HEAD_DIM), device=device, dtype=dtype).requires_grad_(requires_grad) for _ in range(3)
    ]


def flops(Z, H, N_CTX, HEAD_DIM, causal, direction="fwd"):
    """The tutorials' and tritonbench's count, so the numbers are comparable.

    `tutorials/fused_attention_ws_auto_tma.py`: 2.5x on the backward is 2.0 plus
    0.5 to recompute the scores.
    """
    total = 2 * (2.0 * Z * H * N_CTX * N_CTX * HEAD_DIM)
    if causal:
        total *= 0.5
    if direction == "bwd":
        total *= 2.5
    return int(total)


def label(Z, H, N_CTX, HEAD_DIM, causal, dtype, direction="fwd") -> str:
    return (f"((), {{'dtype': '{dtype}', 'causal': '{causal}', 'dir': '{direction}', "
            f"'Z': '{Z}', 'H': '{H}', 'N_CTX': '{N_CTX}', 'HEAD_DIM': '{HEAD_DIM}'}})")
