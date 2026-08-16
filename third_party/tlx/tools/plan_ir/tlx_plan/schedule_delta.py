"""M1.5a block-local schedule-delta contract."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .model import PlanBundle, PlanError, canonical_json


SCHEDULE_DELTA_SCHEMA_VERSION = "plan-schedule-delta/0.1"
FINAL_STRUCTURED_TTGIR_POSITION = "after_warp_pipeline_conversion_before_scf_to_cf"


@dataclass
class BlockScheduleDelta:
    block: str
    baseline_order: list[str]
    desired_order: list[str]
    reason: str = ""

    def validate(self) -> None:
        if not self.block:
            raise PlanError("schedule delta block id must be non-empty")
        if not self.baseline_order:
            raise PlanError(f"schedule delta block {self.block!r} has an empty baseline order")
        if len(self.baseline_order) != len(set(self.baseline_order)):
            raise PlanError(f"schedule delta block {self.block!r} repeats a baseline operation")
        if len(self.desired_order) != len(set(self.desired_order)):
            raise PlanError(f"schedule delta block {self.block!r} repeats a desired operation")
        if set(self.baseline_order) != set(self.desired_order):
            raise PlanError(
                f"schedule delta block {self.block!r} desired order is not a complete permutation"
            )


@dataclass
class PlanScheduleDelta:
    kernel: str
    input_value_graph_fingerprint: str
    blocks: list[BlockScheduleDelta]
    provenance: dict[str, Any] = field(default_factory=dict)
    pass_position: str = FINAL_STRUCTURED_TTGIR_POSITION
    schema_version: str = SCHEDULE_DELTA_SCHEMA_VERSION

    def validate(self, plan: PlanBundle | None = None) -> None:
        if self.schema_version != SCHEDULE_DELTA_SCHEMA_VERSION:
            raise PlanError(
                f"unsupported schedule-delta schema {self.schema_version!r}; "
                f"expected {SCHEDULE_DELTA_SCHEMA_VERSION!r}"
            )
        if not self.kernel:
            raise PlanError("schedule delta kernel must be non-empty")
        if not self.input_value_graph_fingerprint:
            raise PlanError("schedule delta value-graph fingerprint must be non-empty")
        if self.pass_position != FINAL_STRUCTURED_TTGIR_POSITION:
            raise PlanError("schedule delta targets an unsupported compiler pass position")
        block_ids = [block.block for block in self.blocks]
        if not block_ids:
            raise PlanError("schedule delta must contain at least one block")
        if len(block_ids) != len(set(block_ids)):
            raise PlanError("schedule delta repeats a block")
        for block in self.blocks:
            block.validate()

        if plan is None:
            return
        plan.validate()
        if self.kernel != plan.kernel:
            raise PlanError(
                f"schedule delta kernel {self.kernel!r} does not match plan kernel {plan.kernel!r}"
            )
        if self.input_value_graph_fingerprint != plan.value_graph_fingerprint:
            raise PlanError("schedule delta value-graph fingerprint does not match the plan")
        plan_blocks = {block["id"]: block.get("operations", []) for block in plan.blocks}
        for block in self.blocks:
            if block.block not in plan_blocks:
                raise PlanError(f"schedule delta refers to unknown block {block.block!r}")
            if block.baseline_order != plan_blocks[block.block]:
                raise PlanError(f"schedule delta baseline order does not match block {block.block!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: Path) -> None:
        self.validate()
        path.write_text(canonical_json(self.to_dict()))

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PlanScheduleDelta":
        blocks = [BlockScheduleDelta(**block) for block in value.get("blocks", [])]
        delta = cls(
            kernel=value.get("kernel", ""),
            input_value_graph_fingerprint=value.get("input_value_graph_fingerprint", ""),
            blocks=blocks,
            provenance=value.get("provenance", {}),
            pass_position=value.get("pass_position", FINAL_STRUCTURED_TTGIR_POSITION),
            schema_version=value.get("schema_version", ""),
        )
        delta.validate()
        return delta

    @classmethod
    def read(cls, path: Path) -> "PlanScheduleDelta":
        return cls.from_dict(json.loads(path.read_text()))


def make_identity_schedule_delta(
    plan: PlanBundle,
    block_ids: Iterable[str] | None = None,
    *,
    reason: str = "identity schedule",
) -> PlanScheduleDelta:
    """Create a delta that exactly preserves selected PlanBundle block orders."""

    plan.validate()
    selected = set(block_ids) if block_ids is not None else None
    known = {block["id"] for block in plan.blocks}
    if selected is not None:
        missing = sorted(selected - known)
        if missing:
            raise PlanError(f"identity schedule requested unknown blocks: {missing}")
    blocks = []
    for block in plan.blocks:
        block_id = block["id"]
        if selected is not None and block_id not in selected:
            continue
        order = list(block.get("operations", []))
        blocks.append(
            BlockScheduleDelta(
                block=block_id,
                baseline_order=order,
                desired_order=list(order),
                reason=reason,
            )
        )
    delta = PlanScheduleDelta(
        kernel=plan.kernel,
        input_value_graph_fingerprint=plan.value_graph_fingerprint,
        blocks=blocks,
        provenance={"source_plan_schema": plan.schema_version},
    )
    delta.validate(plan)
    return delta


def make_identity_schedule_delta_from_value_graph(
    value_graph: dict[str, Any],
    kernel: str,
    *,
    block_ids: Iterable[str] | None = None,
    reason: str = "identity schedule",
) -> PlanScheduleDelta:
    """Create an identity delta directly from a native value-graph sidecar."""

    function = next(
        (entry for entry in value_graph.get("functions", []) if entry.get("function") == kernel),
        None,
    )
    if function is None:
        raise PlanError(f"native value graph does not contain kernel {kernel!r}")
    selected = set(block_ids) if block_ids is not None else None
    known = {block.get("id") for block in function.get("blocks", [])}
    if selected is not None:
        missing = sorted(selected - known)
        if missing:
            raise PlanError(f"identity schedule requested unknown blocks: {missing}")
    blocks = []
    for block in function.get("blocks", []):
        block_id = block.get("id", "")
        if selected is not None and block_id not in selected:
            continue
        order = list(block.get("operations", []))
        blocks.append(
            BlockScheduleDelta(
                block=block_id,
                baseline_order=order,
                desired_order=list(order),
                reason=reason,
            )
        )
    delta = PlanScheduleDelta(
        kernel=kernel,
        input_value_graph_fingerprint=function.get("semantic_fingerprint", ""),
        blocks=blocks,
        provenance={"source_value_graph_schema": value_graph.get("schema_version", "")},
    )
    delta.validate()
    return delta
