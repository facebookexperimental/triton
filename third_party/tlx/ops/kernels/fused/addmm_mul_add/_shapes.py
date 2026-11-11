from __future__ import annotations

from typing import NamedTuple

from ..._shape_suites import FocusRegistry, FocusSuite


class AddMMMulAddShape(NamedTuple):
    m: int
    n: int
    k: int
    has_res: bool
    dtype: str


SYNTHETIC: tuple[AddMMMulAddShape, ...] = (
    AddMMMulAddShape(256, 512, 256, False, "fp16"),
    AddMMMulAddShape(256, 512, 256, True, "bf16"),
    AddMMMulAddShape(200, 776, 200, True, "fp16"),
)

# Production DCNv2 cross layers on H100 inference: up-projections to dim 20992 with
# inner 256 or 512 and a dynamic batch M of 706 rows on average (641-768).
FOCUS_SUITES = (
    FocusSuite(
        name="sm90_1",
        op="addmm_mul_add",
        shapes=(
            AddMMMulAddShape(706, 20992, 512, True, "fp16"),
            AddMMMulAddShape(706, 20992, 512, False, "fp16"),
            AddMMMulAddShape(706, 20992, 256, True, "fp16"),
            AddMMMulAddShape(706, 20992, 256, False, "fp16"),
        ),
    ),
    FocusSuite(
        name="sm90_2",
        op="addmm_mul_add",
        shapes=(
            AddMMMulAddShape(768, 20992, 512, True, "fp16"),
            AddMMMulAddShape(768, 20992, 256, True, "fp16"),
            AddMMMulAddShape(641, 20992, 512, True, "fp16"),
            AddMMMulAddShape(641, 20992, 256, True, "fp16"),
        ),
    ),
    # Production DCNv2 gates with dim 22016, inner 512 and a dynamic batch M = 320-767 (mostly 685-764).
    # Without res is the first cross block (x is x0), with res the second.
    FocusSuite(
        name="sm90_3",
        op="addmm_mul_add",
        shapes=(
            AddMMMulAddShape(741, 22016, 512, False, "fp16"),
            AddMMMulAddShape(741, 22016, 512, True, "fp16"),
            AddMMMulAddShape(685, 22016, 512, False, "fp16"),
            AddMMMulAddShape(685, 22016, 512, True, "fp16"),
            AddMMMulAddShape(525, 22016, 512, False, "fp16"),
            AddMMMulAddShape(525, 22016, 512, True, "fp16"),
        ),
    ),
    # DCNv2 cross layers with dim 22016, inner 512, dynamic batch M = 600-768 (mean 671): five calls with
    # x is x0 (no res) and five with a distinct x per forward.
    FocusSuite(
        name="sm90_4",
        op="addmm_mul_add",
        shapes=(
            AddMMMulAddShape(671, 22016, 512, False, "fp16"),
            AddMMMulAddShape(671, 22016, 512, True, "fp16"),
            AddMMMulAddShape(600, 22016, 512, False, "fp16"),
            AddMMMulAddShape(600, 22016, 512, True, "fp16"),
            AddMMMulAddShape(768, 22016, 512, False, "fp16"),
            AddMMMulAddShape(768, 22016, 512, True, "fp16"),
        ),
    ),
    FocusSuite(
        name="sm90_all",
        op="addmm_mul_add",
        includes=("sm90_1", "sm90_2", "sm90_3", "sm90_4"),
    ),
)

DEFAULT_SUITES = {"sm90": ("sm90_1", )}
FOCUS = FocusRegistry("addmm_mul_add", FOCUS_SUITES, DEFAULT_SUITES)
CORRECTNESS_SHAPES = tuple(dict.fromkeys((*SYNTHETIC, *FOCUS.all_shapes())))


def inputs(entry, dtype, device="cuda"):
    """``bias, a, b, x0, res`` with ``b`` column-major, as an ``nn.Linear`` passes it."""
    m, n, k, has_res, _ = entry
    import torch

    a = torch.randn(m, k, device=device, dtype=dtype)
    b = (torch.randn(n, k, device=device, dtype=dtype) / k**0.5).t()
    bias = torch.randn(n, device=device, dtype=dtype) * 0.1
    x0 = torch.randn(m, n, device=device, dtype=dtype)
    res = torch.randn(m, n, device=device, dtype=dtype) if has_res else None
    return bias, a, b, x0, res


def flops(m, n, k):
    return 2 * m * n * k + 2 * m * n


def label(m, n, k, has_res, dtype) -> str:
    return f"((), {{'dtype': '{dtype}', 'M': '{m}', 'N': '{n}', 'K': '{k}', 'res': {has_res}}})"
