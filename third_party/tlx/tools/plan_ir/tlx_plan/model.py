"""Versioned, deterministic Plan IR data model."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "0.3"
BASELINE_SCHEMA_VERSION = "0.1"


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
    schema_version: str = BASELINE_SCHEMA_VERSION

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
            schema_version=value.get("schema_version", BASELINE_SCHEMA_VERSION),
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
    values: list[dict[str, Any]] = field(default_factory=list)
    lineage_edges: list[dict[str, Any]] = field(default_factory=list)
    blocks: list[dict[str, Any]] = field(default_factory=list)
    live_segments: list[dict[str, Any]] = field(default_factory=list)
    lds_aliases: list[dict[str, Any]] = field(default_factory=list)
    memory_accesses: list[dict[str, Any]] = field(default_factory=list)
    lds_allocations: list[dict[str, Any]] = field(default_factory=list)
    value_graph_fingerprint: str = ""
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
        "values",
        "lineage_edges",
        "blocks",
        "live_segments",
        "lds_aliases",
        "memory_accesses",
        "lds_allocations",
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
        value_ids = [value.get("id") for value in self.values]
        if any(not value for value in value_ids):
            raise PlanError("every value must have a non-empty id")
        if len(value_ids) != len(set(value_ids)):
            raise PlanError("value ids are not unique")
        value_id_set = set(value_ids)
        for edge in self.lineage_edges:
            if edge.get("source") not in value_id_set or edge.get("destination") not in value_id_set:
                raise PlanError("lineage edge refers to an unknown value")
            distance = edge.get("iteration_distance", 0)
            if edge.get("kind") == "loop_backedge":
                if not isinstance(distance, int) or distance < 1:
                    raise PlanError("loop backedge must have a positive iteration distance")
            elif distance != 0:
                raise PlanError("only loop backedges may have nonzero iteration distance")
        block_ids = [block.get("id") for block in self.blocks]
        if any(not value for value in block_ids) or len(block_ids) != len(set(block_ids)):
            raise PlanError("block ids must be non-empty and unique")
        block_sizes: dict[str, int] = {}
        for block in self.blocks:
            block_operations = block.get("operations", [])
            if any(operation not in op_id_set for operation in block_operations):
                raise PlanError("block refers to an unknown operation")
            block_sizes[block["id"]] = len(block_operations)

        def validate_segment(segment: dict[str, Any], expected_value: str | None = None) -> None:
            value_id = segment.get("value")
            block_id = segment.get("block")
            start = segment.get("start_position")
            end = segment.get("end_position")
            if value_id not in value_id_set or (expected_value is not None and value_id != expected_value):
                raise PlanError("live segment refers to an unknown or inconsistent value")
            if block_id not in block_sizes:
                raise PlanError("live segment refers to an unknown block")
            if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start <= end <= block_sizes[block_id]:
                raise PlanError("live segment positions are outside the block")
            distance = segment.get("iteration_distance", 0)
            if not isinstance(distance, int) or distance < 0:
                raise PlanError("live segment iteration distance must be nonnegative")
            if bool(segment.get("crosses_backedge")) != (distance > 0):
                raise PlanError("live segment backedge flag and distance disagree")

        for segment in self.live_segments:
            validate_segment(segment)

        def validate_paths(paths: list[dict[str, Any]]) -> None:
            for path in paths:
                if path.get("root_value") not in value_id_set:
                    raise PlanError("LDS slot path refers to an unknown root")
                for index in path.get("indices", []):
                    base = index.get("base_value", "")
                    if base and base not in value_id_set:
                        raise PlanError("LDS slot expression refers to an unknown value")

        for alias in self.lds_aliases:
            if alias.get("value") not in value_id_set:
                raise PlanError("LDS alias refers to an unknown value")
            source = alias.get("source_value")
            if source is not None and source not in value_id_set:
                raise PlanError("LDS alias refers to an unknown source")
            if any(root not in value_id_set for root in alias.get("root_values", [])):
                raise PlanError("LDS alias refers to an unknown root")
            validate_paths(alias.get("slot_paths", []))
        for access in self.memory_accesses:
            if access.get("operation") not in op_id_set or access.get("value") not in value_id_set:
                raise PlanError("memory access refers to an unknown operation or value")
            if any(root not in value_id_set for root in access.get("root_values", [])):
                raise PlanError("memory access refers to an unknown root")
            validate_paths(access.get("slot_paths", []))
        for allocation in self.lds_allocations:
            root = allocation.get("root_value")
            if root not in value_id_set or allocation.get("allocation_operation") not in op_id_set:
                raise PlanError("LDS allocation refers to an unknown operation or root")
            if any(alias not in value_id_set for alias in allocation.get("aliases", [])):
                raise PlanError("LDS allocation refers to an unknown alias")
            for segment in allocation.get("live_segments", []):
                validate_segment(segment, root)
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
        value = dict(value)
        schema_version = value.get("schema_version", "0.1")
        if schema_version in {"0.1", "0.2"}:
            additions = {
                "blocks": [],
                "live_segments": [],
                "lds_aliases": [],
                "memory_accesses": [],
                "lds_allocations": [],
            }
            if schema_version == "0.1":
                additions.update(
                    values=[],
                    lineage_edges=[],
                    value_graph_fingerprint="",
                )
            value.update(
                schema_version=SCHEMA_VERSION,
                layer_hashes={},
                **additions,
            )
        bundle = cls(**value)
        if not bundle.layer_hashes:
            bundle.refresh_hashes()
        bundle.validate()
        return bundle

    def write(self, path: Path) -> None:
        path.write_text(canonical_json(self.to_dict()))

    @classmethod
    def read(cls, path: Path) -> "PlanBundle":
        return cls.from_dict(json.loads(path.read_text()))
