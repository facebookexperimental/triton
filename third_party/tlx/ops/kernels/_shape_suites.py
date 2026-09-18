from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Hashable, Iterable, Mapping, TypeVar

ShapeT = TypeVar("ShapeT", bound=Hashable)


@dataclass(frozen=True)
class FocusSuite(Generic[ShapeT]):
    name: str
    op: str
    shapes: tuple[ShapeT, ...]


@dataclass(frozen=True)
class FocusRegistry(Generic[ShapeT]):
    op: str
    suites: tuple[FocusSuite[ShapeT], ...]
    defaults: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        names = [suite.name for suite in self.suites]
        duplicates = sorted(name for name in set(names) if names.count(name) > 1)
        if duplicates:
            raise ValueError(f"duplicate {self.op} focus suite name(s): {duplicates}")
        mismatched = sorted(suite.name for suite in self.suites if suite.op != self.op)
        if mismatched:
            raise ValueError(f"focus suite(s) {mismatched} do not map to op {self.op!r}")

        by_name = {suite.name: suite for suite in self.suites}
        for arch, default_names in self.defaults.items():
            if len(default_names) != len(set(default_names)):
                raise ValueError(f"duplicate default {self.op} focus suites for architecture {arch!r}")
            unknown = [name for name in default_names if name not in by_name]
            if unknown:
                raise ValueError(f"unknown default {self.op} focus suite(s) {unknown} for architecture {arch!r}")

    def suite(self, name: str) -> FocusSuite[ShapeT]:
        for suite in self.suites:
            if suite.name == name:
                return suite
        available = ", ".join(suite.name for suite in self.suites)
        raise ValueError(f"unknown focus suite {name!r}; available suites: {available}")

    def _selected(self, arch: str, names: Iterable[str] | None) -> tuple[FocusSuite[ShapeT], ...]:
        if names is None:
            if arch not in self.defaults:
                raise ValueError(f"no default {self.op} focus suites for architecture {arch!r}")
            requested = self.defaults[arch]
        else:
            requested = tuple(names)
        return tuple(self.suite(name) for name in dict.fromkeys(requested))

    def shapes(self, arch: str, suites: Iterable[str] | None = None) -> tuple[ShapeT, ...]:
        selected = self._selected(arch, suites)
        seen: set[ShapeT] = set()
        result: list[ShapeT] = []
        for suite in selected:
            for shape in suite.shapes:
                if shape not in seen:
                    seen.add(shape)
                    result.append(shape)
        return tuple(result)

    def selected_suite_names(self, arch: str, suites: Iterable[str] | None = None) -> tuple[str, ...]:
        return tuple(suite.name for suite in self._selected(arch, suites))
