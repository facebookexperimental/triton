from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


_TARGETS_ROOT = Path(__file__).resolve().parent


def _normalize(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


@dataclass(frozen=True)
class TargetBundle:
    operation: str
    vendor: str
    architecture: str
    aliases: tuple[str, ...]
    cuda_major: int | None
    directory: Path

    @property
    def harness_path(self) -> Path:
        return self.directory / "harness.py"

    @property
    def cases_path(self) -> Path:
        return self.directory / "cases.json"

    @property
    def target_path(self) -> Path:
        return self.directory / "target.json"

    def matches_architecture(self, architecture: str) -> bool:
        needle = _normalize(architecture)
        return needle in {_normalize(self.architecture), *map(_normalize, self.aliases)}


@lru_cache(maxsize=1)
def target_bundles() -> tuple[TargetBundle, ...]:
    bundles: list[TargetBundle] = []
    for manifest_path in sorted(_TARGETS_ROOT.glob("**/bundle.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        directory = manifest_path.parent
        relative = directory.relative_to(_TARGETS_ROOT)
        if len(relative.parts) < 2:
            raise RuntimeError(f"invalid target bundle location: {manifest_path}")
        vendor = relative.parts[0]
        architecture = relative.parts[-2]
        bundles.append(
            TargetBundle(
                operation=str(payload.get("operation", directory.name)),
                vendor=str(payload.get("vendor", vendor)),
                architecture=str(payload.get("architecture", architecture)),
                aliases=tuple(str(alias) for alias in payload.get("aliases", ())),
                cuda_major=(int(payload["cuda_major"]) if payload.get("cuda_major") is not None else None),
                directory=directory,
            )
        )
    return tuple(bundles)


def resolve_target_paths(
    kernel: Path,
    harness: Path | None,
    cases: Path | None,
    target: Path | None,
    architecture: str | None,
) -> tuple[Path, Path, Path]:
    if harness is not None and cases is not None and target is not None:
        return harness, cases, target

    matches = [bundle for bundle in target_bundles() if bundle.operation == kernel.stem]
    if architecture is not None:
        matches = [bundle for bundle in matches if bundle.matches_architecture(architecture)]
    if matches:
        bundle = matches[0]
        harness = harness or bundle.harness_path
        cases = cases or bundle.cases_path
        target = target or bundle.target_path

    if harness is None or cases is None or target is None:
        missing = "/".join(
            name for name, value in (("harness", harness), ("cases", cases), ("target", target)) if value is None
        )
        raise SystemExit(
            f"missing required {missing}; pass paths explicitly or register a target bundle for {kernel.stem!r}"
        )
    return harness, cases, target


def expected_cuda_major(architecture: str) -> int | None:
    for bundle in target_bundles():
        if bundle.matches_architecture(architecture):
            return bundle.cuda_major
    return None
