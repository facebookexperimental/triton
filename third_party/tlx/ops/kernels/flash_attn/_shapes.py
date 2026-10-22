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
        name="sm90_baseline",
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
        name="sm100_baseline",
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
)

DEFAULT_SUITES = {
    "sm90": ("sm90_baseline", ),
    "sm100": ("sm100_baseline", ),
}
FOCUS = FocusRegistry("flash_attn", FOCUS_SUITES, DEFAULT_SUITES)

SM90_FOCUS = FOCUS.shapes("sm90")
SM100_FOCUS = FOCUS.shapes("sm100")


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
