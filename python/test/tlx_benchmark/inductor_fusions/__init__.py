from __future__ import annotations

import importlib
from collections.abc import Sequence

from .common import FusionCase

ARCH_PACKAGES = {
    "sm90": "h100",
    "sm100": "b200",
    "gfx942": "mi300",
    "gfx950": "mi350",
}


def cases_for_arch(arch: str) -> Sequence[FusionCase]:
    package = ARCH_PACKAGES.get(arch)
    if package is None:
        return ()
    module = importlib.import_module(f"{__name__}.{package}")
    cases = tuple(module.CASES)
    names = [case.name for case in cases]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate fusion case name in {arch}: {names}")
    return cases


def catalog() -> dict[str, Sequence[FusionCase]]:
    return {arch: cases_for_arch(arch) for arch in ARCH_PACKAGES}
