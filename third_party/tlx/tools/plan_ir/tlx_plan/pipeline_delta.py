"""M1.5b cross-iteration pipeline-delta contract and dry-run validation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .model import PlanBundle, PlanError, canonical_json


PIPELINE_DELTA_SCHEMA_VERSION = "plan-pipeline-delta/0.1"
PIPELINE_DELTA_VALIDATION_SCHEMA_VERSION = "plan-pipeline-delta-validation/0.1"
PIPELINE_APPLY_POSITION = "before_update_async_wait_count"
TRANSACTION_ACTIONS = frozenset({"set_prefetch_distance"})
STAGING_ACTIONS = frozenset({"global_to_lds", "register_to_lds"})
LOOP_KINDS = frozenset({"scf.for", "scf.while"})


def _require_positive_int(value: object, description: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise PlanError(f"{description} must be a positive integer")
    return value


@dataclass
class TransactionPipelineIntent:
    group: str
    action: str
    distance: int
    buffer_depth: int

    def validate(self) -> None:
        if not self.group:
            raise PlanError("pipeline transaction group id must be non-empty")
        if self.action not in TRANSACTION_ACTIONS:
            raise PlanError(f"unsupported pipeline transaction action {self.action!r}")
        distance = _require_positive_int(self.distance, "pipeline prefetch distance")
        depth = _require_positive_int(self.buffer_depth, "pipeline transaction buffer depth")
        if depth < distance:
            raise PlanError("pipeline transaction buffer depth cannot be less than prefetch distance")


@dataclass
class StagingPipelineIntent:
    value: str
    action: str
    consumers: list[str]
    buffer_depth: int
    alignment: int

    def validate(self) -> None:
        if not self.value:
            raise PlanError("pipeline staging value id must be non-empty")
        if self.action not in STAGING_ACTIONS:
            raise PlanError(f"unsupported pipeline staging action {self.action!r}")
        if not self.consumers:
            raise PlanError(f"pipeline staging for {self.value!r} must name at least one consumer")
        if len(self.consumers) != len(set(self.consumers)):
            raise PlanError(f"pipeline staging for {self.value!r} repeats a consumer")
        _require_positive_int(self.buffer_depth, "pipeline staging buffer depth")
        alignment = _require_positive_int(self.alignment, "pipeline staging alignment")
        if alignment & (alignment - 1):
            raise PlanError("pipeline staging alignment must be a power of two")


@dataclass
class LoopPipelineDelta:
    loop: str
    transactions: list[TransactionPipelineIntent] = field(default_factory=list)
    staging: list[StagingPipelineIntent] = field(default_factory=list)

    def validate(self) -> None:
        if not self.loop:
            raise PlanError("pipeline delta loop id must be non-empty")
        if not self.transactions and not self.staging:
            raise PlanError(f"pipeline delta loop {self.loop!r} contains no mutations")
        transaction_groups = [transaction.group for transaction in self.transactions]
        if len(transaction_groups) != len(set(transaction_groups)):
            raise PlanError(f"pipeline delta loop {self.loop!r} repeats a transaction group")
        staging_values = [staging.value for staging in self.staging]
        if len(staging_values) != len(set(staging_values)):
            raise PlanError(f"pipeline delta loop {self.loop!r} repeats a staging value")
        for transaction in self.transactions:
            transaction.validate()
        for staging in self.staging:
            staging.validate()


@dataclass
class PlanPipelineDelta:
    kernel: str
    input_value_graph_fingerprint: str
    loops: list[LoopPipelineDelta]
    provenance: dict[str, Any] = field(default_factory=dict)
    pass_position: str = PIPELINE_APPLY_POSITION
    schema_version: str = PIPELINE_DELTA_SCHEMA_VERSION

    def validate(self, plan: PlanBundle | None = None) -> None:
        if self.schema_version != PIPELINE_DELTA_SCHEMA_VERSION:
            raise PlanError(
                f"unsupported pipeline-delta schema {self.schema_version!r}; "
                f"expected {PIPELINE_DELTA_SCHEMA_VERSION!r}"
            )
        if not self.kernel:
            raise PlanError("pipeline delta kernel must be non-empty")
        if not self.input_value_graph_fingerprint:
            raise PlanError("pipeline delta value-graph fingerprint must be non-empty")
        if self.pass_position != PIPELINE_APPLY_POSITION:
            raise PlanError("pipeline delta targets an unsupported compiler pass position")
        loop_ids = [loop.loop for loop in self.loops]
        if len(loop_ids) != len(set(loop_ids)):
            raise PlanError("pipeline delta repeats a loop")
        for loop in self.loops:
            loop.validate()

        if plan is not None:
            _validate_against_plan(self, plan)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: Path) -> None:
        self.validate()
        path.write_text(canonical_json(self.to_dict()))

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PlanPipelineDelta":
        loops = []
        for loop_value in value.get("loops", []):
            loops.append(
                LoopPipelineDelta(
                    loop=loop_value.get("loop", ""),
                    transactions=[
                        TransactionPipelineIntent(**transaction)
                        for transaction in loop_value.get("transactions", [])
                    ],
                    staging=[
                        StagingPipelineIntent(**staging)
                        for staging in loop_value.get("staging", [])
                    ],
                )
            )
        delta = cls(
            kernel=value.get("kernel", ""),
            input_value_graph_fingerprint=value.get("input_value_graph_fingerprint", ""),
            loops=loops,
            provenance=value.get("provenance", {}),
            pass_position=value.get("pass_position", PIPELINE_APPLY_POSITION),
            schema_version=value.get("schema_version", ""),
        )
        delta.validate()
        return delta

    @classmethod
    def read(cls, path: Path) -> "PlanPipelineDelta":
        return cls.from_dict(json.loads(path.read_text()))


def _validate_against_plan(delta: PlanPipelineDelta, plan: PlanBundle) -> None:
    plan.validate()
    if delta.kernel != plan.kernel:
        raise PlanError(
            f"pipeline delta kernel {delta.kernel!r} does not match plan kernel {plan.kernel!r}"
        )
    if delta.input_value_graph_fingerprint != plan.value_graph_fingerprint:
        raise PlanError("pipeline delta value-graph fingerprint does not match the plan")

    operations = {operation["id"]: operation for operation in plan.operations}
    values = {value["id"]: value for value in plan.values}
    blocks_by_parent: dict[str, list[dict[str, Any]]] = {}
    for block in plan.blocks:
        parent = block.get("parent_operation")
        if parent:
            blocks_by_parent.setdefault(parent, []).append(block)
    groups = {group["id"]: group for group in plan.async_groups}
    transactions = {transaction["id"]: transaction for transaction in plan.async_transactions}

    for loop_delta in delta.loops:
        loop_operation = operations.get(loop_delta.loop)
        if loop_operation is None:
            raise PlanError(f"pipeline delta refers to unknown loop {loop_delta.loop!r}")
        if loop_operation.get("kind") not in LOOP_KINDS:
            raise PlanError(f"pipeline delta target {loop_delta.loop!r} is not a structured loop")
        loop_blocks = blocks_by_parent.get(loop_delta.loop, [])
        if not loop_blocks:
            raise PlanError(f"pipeline delta loop {loop_delta.loop!r} has no structured body blocks")
        loop_operations = {
            operation for block in loop_blocks for operation in block.get("operations", [])
        }
        requested_groups = {intent.group for intent in loop_delta.transactions}

        for intent in loop_delta.transactions:
            group = groups.get(intent.group)
            if group is None:
                raise PlanError(f"pipeline delta refers to unknown async group {intent.group!r}")
            if group.get("commit_operation") not in loop_operations:
                raise PlanError(
                    f"async group {intent.group!r} is not committed in loop {loop_delta.loop!r}"
                )
            transaction_ids = group.get("transactions", [])
            if not transaction_ids:
                raise PlanError(f"async group {intent.group!r} contains no transactions")
            complete_transaction_ids = {
                transaction_id
                for transaction_id, transaction in transactions.items()
                if transaction.get("commit_group") == intent.group
            }
            if set(transaction_ids) != complete_transaction_ids:
                raise PlanError(f"async group {intent.group!r} is not a complete transaction group")
            for transaction_id in transaction_ids:
                transaction = transactions.get(transaction_id)
                if transaction is None:
                    raise PlanError(
                        f"async group {intent.group!r} refers to unknown transaction {transaction_id!r}"
                    )
                if transaction.get("producer_operation") not in loop_operations:
                    raise PlanError(
                        f"async transaction {transaction_id!r} is not produced in loop "
                        f"{loop_delta.loop!r}"
                    )
                if not transaction.get("slot_paths"):
                    raise PlanError(
                        f"async transaction {transaction_id!r} has no resolved LDS slot path"
                    )
                if any(
                    index.get("kind") == "unknown"
                    for path in transaction.get("slot_paths", [])
                    for index in path.get("indices", [])
                ):
                    raise PlanError(
                        f"async transaction {transaction_id!r} has an unresolved LDS slot index"
                    )
                consumer_distances = {
                    frontier.get("iteration_distance", 0)
                    for frontier in transaction.get("consumer_frontiers", [])
                    if frontier.get("iteration_distance", 0) > 0
                }
                if consumer_distances != {intent.distance}:
                    raise PlanError(
                        f"async transaction {transaction_id!r} does not have exact prefetch "
                        f"distance {intent.distance}"
                    )
                slot_depths = {
                    index.get("modulus", 0)
                    for path in transaction.get("slot_paths", [])
                    for index in path.get("indices", [])
                    if index.get("modulus", 0) > 0
                }
                if slot_depths != {intent.buffer_depth}:
                    raise PlanError(
                        f"async transaction {transaction_id!r} does not have exact buffer "
                        f"depth {intent.buffer_depth}"
                    )

            wait_families = []
            for wait in plan.async_waits:
                selected_completions = [
                    completion
                    for completion in wait.get("completed_groups", [])
                    if completion.get("group") == intent.group
                    and completion.get("iteration_distance", 0) > 0
                ]
                for selected in selected_completions:
                    distance = selected["iteration_distance"]
                    family = {
                        completion.get("group")
                        for completion in wait.get("completed_groups", [])
                        if completion.get("iteration_distance", 0) == distance
                        and groups.get(completion.get("group"), {}).get("commit_operation")
                        in loop_operations
                    }
                    wait_families.append((wait.get("operation"), family))
            if not wait_families:
                raise PlanError(
                    f"async group {intent.group!r} has no positive-distance completion wait"
                )
            for wait_operation, family in wait_families:
                missing = family - requested_groups
                if missing:
                    raise PlanError(
                        f"pipeline delta omits groups {sorted(missing)} sharing wait "
                        f"{wait_operation!r}"
                    )

        for intent in loop_delta.staging:
            value = values.get(intent.value)
            if value is None:
                raise PlanError(f"pipeline delta refers to unknown staging value {intent.value!r}")
            category = value.get("type", {}).get("category")
            if category != "tensor_register_logical":
                raise PlanError(
                    f"pipeline staging value {intent.value!r} must be a tensor, got {category!r}"
                )
            uses = {use.get("operation") for use in value.get("uses", [])}
            for consumer in intent.consumers:
                if consumer not in operations:
                    raise PlanError(f"pipeline staging refers to unknown consumer {consumer!r}")
                if consumer not in loop_operations:
                    raise PlanError(
                        f"pipeline staging consumer {consumer!r} is not in loop {loop_delta.loop!r}"
                    )
                if consumer not in uses:
                    raise PlanError(
                        f"operation {consumer!r} does not consume staging value {intent.value!r}"
                    )


def make_identity_pipeline_delta(plan: PlanBundle) -> PlanPipelineDelta:
    """Create a valid no-op pipeline delta pinned to a PlanBundle."""

    plan.validate()
    delta = PlanPipelineDelta(
        kernel=plan.kernel,
        input_value_graph_fingerprint=plan.value_graph_fingerprint,
        loops=[],
        provenance={"source_plan_schema": plan.schema_version},
    )
    delta.validate(plan)
    return delta


def validate_pipeline_delta(
    delta: PlanPipelineDelta, plan: PlanBundle
) -> dict[str, Any]:
    """Dry-run the contract against Plan IR and return materialization requirements."""

    delta.validate(plan)
    transaction_count = sum(len(loop.transactions) for loop in delta.loops)
    staging_count = sum(len(loop.staging) for loop in delta.loops)
    requires_modulo_scheduling = any(
        transaction.distance > 0 or transaction.buffer_depth > 1
        for loop in delta.loops
        for transaction in loop.transactions
    ) or any(
        staging.buffer_depth > 1 for loop in delta.loops for staging in loop.staging
    )
    return {
        "schema_version": PIPELINE_DELTA_VALIDATION_SCHEMA_VERSION,
        "passed": True,
        "kernel": delta.kernel,
        "identity": not delta.loops,
        "loops": len(delta.loops),
        "transaction_groups": transaction_count,
        "staging_values": staging_count,
        "requires_modulo_scheduling": requires_modulo_scheduling,
        "changes_iteration_placement": bool(transaction_count or staging_count),
        "changes_iteration_storage": bool(staging_count),
        "changes_synchronization": bool(transaction_count or staging_count),
        "changes_dot_decomposition": False,
        "materialization_status": "not_applied",
    }
