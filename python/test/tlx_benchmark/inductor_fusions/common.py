from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import Any


@dataclasses.dataclass(frozen=True)
class CodeContract:
    """Generated-code evidence required before a timing is reportable."""

    required_after: tuple[str, ...]
    forbidden_after: tuple[str, ...] = ()
    forbidden_before: tuple[str, ...] = ()
    expected_after_launches: int | None = None

    def validate_before(self, before: str) -> None:
        forbidden_before = self.forbidden_before or self.required_after
        for marker in forbidden_before:
            if marker in before:
                raise AssertionError(f"baseline unexpectedly contained {marker!r}")

    def validate_after(self, after: str) -> None:
        for marker in self.required_after:
            if marker not in after:
                raise AssertionError(f"optimized code did not contain {marker!r}")
        for marker in self.forbidden_after:
            if marker in after:
                raise AssertionError(f"optimized code unexpectedly contained {marker!r}")
        if self.expected_after_launches is not None:
            launches = after.count(".run(")
            if launches != self.expected_after_launches:
                raise AssertionError("optimized code produced "
                                     f"{launches} launches, expected {self.expected_after_launches}")

    def validate(self, before: str, after: str) -> None:
        self.validate_before(before)
        self.validate_after(after)

    def classify_autotuned(self, code: str) -> str:
        """Describe whether candidate code survived production selection."""
        present = [marker in code for marker in self.required_after]
        if present and all(present):
            return "candidate_present"
        if any(present):
            raise AssertionError("autotuned code contained only part of the candidate marker set")
        return "fallback_selected"


@dataclasses.dataclass(frozen=True)
class FusionCase:
    """One self-contained before/after TorchInductor fusion experiment."""

    name: str
    problem: str
    model: Callable[..., Any]
    make_inputs: Callable[[], tuple[Any, ...]]
    code: CodeContract
    after_mode: str = "force"
    autotuned_mode: str = "allow"
    config_overrides: Mapping[str, object] = dataclasses.field(default_factory=dict)
    forced_before_config_overrides: Mapping[str, object] = dataclasses.field(default_factory=dict)
    forced_after_config_overrides: Mapping[str, object] = dataclasses.field(default_factory=dict)
    autotuned_before_config_overrides: Mapping[str, object] = dataclasses.field(default_factory=dict)
    autotuned_after_config_overrides: Mapping[str, object] = dataclasses.field(default_factory=dict)
    atol: float = 3.0e-2
    rtol: float = 3.0e-2
    requires_custom_op_autotuning: bool = False

    def __post_init__(self) -> None:
        if not self.name or any(char.isspace() for char in self.name):
            raise ValueError("fusion case names must be non-empty and contain no whitespace")

    def variant(self, comparison: str, side: str) -> tuple[str | None, Mapping[str, object]]:
        if comparison not in ("forced", "autotuned") or side not in ("before", "after"):
            raise ValueError(f"invalid fusion variant: {comparison}/{side}")
        if side == "before":
            mode = None
        else:
            mode = self.after_mode if comparison == "forced" else self.autotuned_mode
        overrides = getattr(self, f"{comparison}_{side}_config_overrides")
        return mode, overrides


def make_gemm_norm_inputs(
    shape: tuple[int, int, int],
    *,
    with_norm_bias: bool,
) -> tuple[Any, ...]:
    import torch

    m, k, n = shape
    torch.manual_seed(0)
    tensors = (
        torch.randn((m, k), device="cuda", dtype=torch.bfloat16),
        # Production weights are stored [N, K]. The graph consumes weight.t(),
        # giving logical B [K, N] with stride [1, K].
        torch.randn((n, k), device="cuda", dtype=torch.bfloat16),
        torch.randn((n, ), device="cuda", dtype=torch.bfloat16),
        torch.randn((n, ), device="cuda", dtype=torch.bfloat16),
    )
    if not with_norm_bias:
        return tensors
    return *tensors, torch.randn((n, ), device="cuda", dtype=torch.bfloat16)
