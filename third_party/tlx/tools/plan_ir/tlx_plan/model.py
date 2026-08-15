"""Versioned, deterministic Plan IR data model."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "0.1"


class PlanError(ValueError):
    """Raised when a plan is malformed or cannot be replayed."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, separators=(",", ": ")) + "\n"


def digest(value: Any) -> str:
    payload = canonical_json(value).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class KernelCase:
    name: str
    family: str
    shape: dict[str, int]
    dtype: str
    causal: bool
    grouped_query: bool


@dataclass(frozen=True)
class ScheduleSpec:
    name: str
    kernel: str
    implementation: str
    config: dict[str, int]
    accumulation: str
    algorithm: str


@dataclass
class BaselineManifest:
    case: KernelCase
    schedule: ScheduleSpec
    source_revision: str
    compiler_revision: str
    device: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    measurements: dict[str, float] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BaselineManifest":
        return cls(
            case=KernelCase(**value["case"]),
            schedule=ScheduleSpec(**value["schedule"]),
            source_revision=value["source_revision"],
            compiler_revision=value["compiler_revision"],
            device=value.get("device", {}),
            artifacts=value.get("artifacts", {}),
            measurements=value.get("measurements", {}),
            schema_version=value.get("schema_version", SCHEMA_VERSION),
        )


@dataclass
class PlanBundle:
    kernel: str
    case: dict[str, Any]
    provenance: dict[str, Any]
    operations: list[dict[str, Any]]
    dot_fragments: list[dict[str, Any]]
    storage: list[dict[str, Any]]
    synchronization: list[dict[str, Any]]
    schedule: list[str]
    layouts: dict[str, str]
    normalized_ir_hash: str
    diagnostics: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION
    layer_hashes: dict[str, str] = field(default_factory=dict)

    _LAYERS = (
        "operations",
        "dot_fragments",
        "storage",
        "synchronization",
        "schedule",
        "layouts",
    )

    def refresh_hashes(self) -> None:
        self.layer_hashes = {name: digest(getattr(self, name)) for name in self._LAYERS}

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise PlanError(
                f"unsupported PlanBundle schema {self.schema_version!r}; expected {SCHEMA_VERSION!r}"
            )
        op_ids = [op.get("id") for op in self.operations]
        if any(not value for value in op_ids):
            raise PlanError("every operation must have a non-empty id")
        if len(op_ids) != len(set(op_ids)):
            raise PlanError("operation ids are not unique")
        op_id_set = set(op_ids)
        missing = [value for value in self.schedule if value not in op_id_set]
        if missing:
            raise PlanError(f"schedule refers to unknown operations: {missing[:4]}")
        fragment_ids = [fragment.get("id") for fragment in self.dot_fragments]
        if len(fragment_ids) != len(set(fragment_ids)):
            raise PlanError("dot fragment ids are not unique")
        expected = {name: digest(getattr(self, name)) for name in self._LAYERS}
        if self.layer_hashes and self.layer_hashes != expected:
            raise PlanError("one or more layer hashes do not match the plan payload")

    def to_dict(self) -> dict[str, Any]:
        self.refresh_hashes()
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PlanBundle":
        bundle = cls(**value)
        bundle.validate()
        return bundle

    def write(self, path: Path) -> None:
        path.write_text(canonical_json(self.to_dict()))

    @classmethod
    def read(cls, path: Path) -> "PlanBundle":
        return cls.from_dict(json.loads(path.read_text()))
