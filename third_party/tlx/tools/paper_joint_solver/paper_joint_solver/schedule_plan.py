"""Strict, versioned handoff from the joint solver to pipelined IR.

The paper's schedule is more than ``(II, L)``.  A compilable artifact owns
every operation's cycle and its physical warp assignment, and it is valid
only for the DDG, machine model, and solver sources that produced it.  This
module validates that contract before the software-pipelined, warp-annotated
IR is emitted.

There are two entries and therefore two artifact shapes.  The paper entry
assigns each operation a *set* of physical warps (``warp_sets``, overlap
allowed) and its revalidation here is a literal mirror of the Fig 4/5/6
system: no fixed overheads, no SM-wide register bound, no spill footprint, no
same-lane placement rule.  The emitter entry keeps the code generator's
group/lane encoding and every realizability constraint that goes with it;
that branch is frozen and is not paper content.  Which one applies is decided
by ``provenance.model`` alone -- there are no model options.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .machine import (
    MachineModel,
    physical_group_width,
    register_allocation_words,
    required_group_width,
)


PAPER_SOLUTION_SCHEMA = "paper-joint-solution-v2"
EMITTER_SOLUTION_SCHEMA = "emitter-joint-solution-v1"
PAPER_MODEL_PROFILE = "paper"
EMITTER_MODEL_PROFILE = "emitter"
_SOLUTION_SCHEMAS = {
    PAPER_MODEL_PROFILE: PAPER_SOLUTION_SCHEMA,
    EMITTER_MODEL_PROFILE: EMITTER_SOLUTION_SCHEMA,
}
_SOLVER_SOURCES = (
    "__main__.py",
    "ddg.py",
    "joint_smt.py",
    "machine.py",
    "modulo_ilp.py",
    "normalize.py",
    "pipelined_ir.py",
    "resource_model.py",
    "schedule_plan.py",
    "search.py",
    "../skc/__main__.py",
    "../skc/_schema.py",
    "../skc/audit.py",
    "../skc/compiler.py",
    "../skc/scaffold.py",
)
# The paper solve path carries no fixed overhead at all; a non-zero knob can
# only come from a hand-edited artifact.  This is a provenance-integrity
# check, not a model gate.
_PAPER_ZERO_MACHINE_FIELDS = (
    "warp_fixed_overhead",
    "regs_fixed_overhead",
    "smem_fixed_overhead",
    "tmem_fixed_cols",
)


class ScheduleArtifactError(ValueError):
    pass


class LegacySolutionError(ScheduleArtifactError):
    pass


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def solver_sources_sha256() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in _SOLVER_SOURCES:
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def machine_manifest(machine: MachineModel) -> dict:
    manifest = asdict(machine)
    manifest["group_widths"] = list(machine.group_widths)
    return manifest


def experiment_inputs(
    allow_cross_warp: bool,
    num_warps_override: int | None,
    reg_budget: int | None,
) -> dict:
    """The paper's own experiment inputs, recorded as observations."""
    return {
        "allow_cross_warp": allow_cross_warp,
        "num_warps_override": num_warps_override,
        "reg_budget": reg_budget,
    }


def solution_provenance(
    ddg_path: str | Path,
    machine: MachineModel,
    normalization_u: int,
    *,
    model: str = PAPER_MODEL_PROFILE,
    inputs: dict | None = None,
    baseline_graph_path: str | Path | None = None,
) -> dict:
    """Stamp an artifact with the chain that produced it.

    ``ddg_sha256`` is the solved file itself: the curated artifact on the
    paper path, the raw dump on the emitter path.  The raw hashes of the
    paper path live one level down, inside ``curation_source``.
    """
    provenance = {
        "model": model,
        "ddg_sha256": sha256_file(ddg_path),
        "solver_sources_sha256": solver_sources_sha256(),
        "normalization_u": normalization_u,
        "machine": machine_manifest(machine),
    }
    if model == PAPER_MODEL_PROFILE:
        # Transcribed verbatim so the raw dump -> curated -> solution hash
        # chain closes (schema section 1).
        provenance["curation_source"] = curation_source_of(ddg_path)
        provenance["experiment_inputs"] = dict(
            inputs if inputs is not None else experiment_inputs(True, None, None)
        )
    else:
        provenance["baseline_graph_sha256"] = (
            sha256_file(baseline_graph_path)
            if baseline_graph_path is not None
            else None
        )
    return provenance


def curation_source_of(curated_ddg_path: str | Path) -> dict:
    data = json.loads(Path(curated_ddg_path).read_text())
    source = data.get("curation_source")
    if not isinstance(source, dict):
        raise ScheduleArtifactError(
            "curated ddg is missing its curation_source object"
        )
    return dict(source)


def _int_dict(raw: dict, name: str) -> dict[int, int]:
    try:
        return {int(key): int(value) for key, value in raw.items()}
    except (AttributeError, TypeError, ValueError) as error:
        raise ScheduleArtifactError(f"{name} must be an integer map") from error


def _lane_dict(raw: dict) -> dict[int, tuple[int, ...]]:
    try:
        return {
            int(key): tuple(sorted(int(lane) for lane in lanes))
            for key, lanes in raw.items()
        }
    except (AttributeError, TypeError, ValueError) as error:
        raise ScheduleArtifactError(
            "lane_masks must map node ids to integer lane lists"
        ) from error


def _warp_sets_dict(raw: dict) -> dict[int, tuple[int, ...]]:
    try:
        parsed = {
            int(key): [int(warp) for warp in warps]
            for key, warps in raw.items()
        }
    except (AttributeError, TypeError, ValueError) as error:
        raise ScheduleArtifactError(
            "warp_sets must map node ids to integer warp lists"
        ) from error
    for node_id, warps in sorted(parsed.items()):
        if len(set(warps)) != len(warps):
            raise ScheduleArtifactError(
                f"node {node_id} repeats a physical warp in {warps}"
            )
    return {node_id: tuple(sorted(warps)) for node_id, warps in parsed.items()}


def _same_warp_set(edge, warp_sets) -> bool:
    """Paper reading: only an identical warp set avoids the spill (I-1)."""
    return warp_sets[edge.src] == warp_sets[edge.dst]


def _paper_edge_spill_cost(problem, edge, warp_sets) -> int:
    if _same_warp_set(edge, warp_sets):
        return 0
    return problem.spill[edge.src]


def _same_physical_warp(edge, warp, lane_masks) -> bool:
    return (
        warp[edge.src] == warp[edge.dst]
        and lane_masks[edge.src] == lane_masks[edge.dst]
    )


def _edge_spill_cost(problem, edge, warp, lane_masks) -> int:
    if _same_physical_warp(edge, warp, lane_masks):
        return 0
    return problem.spill[edge.src]


def _figure5_liveness(problem, cycles, ii, copies, horizon):
    liveness_edges = problem.edges
    consumers: dict[int, list] = {}
    for edge in liveness_edges:
        consumers.setdefault(edge.src, []).append(edge)
    carried = {edge.src for edge in liveness_edges if edge.distance > 0}
    starts = {
        (node_id, copy): cycles[node_id] + copy * ii
        for node_id in problem.nodes
        for copy in range(copies)
    }
    live: dict[tuple[int, int, int], bool] = {}
    for node_id in problem.nodes:
        for copy in range(copies):
            live[(node_id, copy, horizon)] = (
                copy == copies - 1 and node_id in carried
            )
            for time in range(horizon, 0, -1):
                is_live = live[(node_id, copy, time)]
                defines_value = (
                    time < horizon and starts[(node_id, copy)] == time
                )
                used = any(
                    copy + edge.distance < copies
                    and time < horizon
                    and starts[(edge.dst, copy + edge.distance)] == time
                    for edge in consumers.get(node_id, ())
                )
                live[(node_id, copy, time - 1)] = (
                    not defines_value if is_live else used
                )
    return live, starts


def _validate_figure5_paper(
    problem, cycles, warp_sets, ii, copies, horizon
) -> None:
    """Fig 5/6 verbatim: REGISTERLIMIT per physical warp, MEMORYCAPACITY per
    value.  The paper has no fixed overhead, no SM-wide register bound, no
    spill footprint and no same-lane rule, so this mirror has none either."""
    machine = problem.machine
    live, _ = _figure5_liveness(problem, cycles, ii, copies, horizon)
    # REGISTERLIMIT charges every member warp ceil(regs(v)/warps(v)) (R5).
    charge = {
        node_id: -(-problem.regs[node_id] // node.min_warps)
        for node_id, node in problem.nodes.items()
    }
    register_nodes = [
        node_id for node_id in problem.nodes if problem.regs[node_id] > 0
    ]
    warps = sorted({warp for warps in warp_sets.values() for warp in warps})
    for time in range(horizon + 1):
        for physical_warp in warps:
            per_warp_registers = sum(
                charge[node_id]
                for node_id in register_nodes
                if physical_warp in warp_sets[node_id]
                for copy in range(copies)
                if live[(node_id, copy, time)]
            )
            if per_warp_registers > machine.regs_per_warp:
                raise ScheduleArtifactError(
                    "REGISTERLIMIT violation at "
                    f"time {time}, warp {physical_warp}: "
                    f"{per_warp_registers} > {machine.regs_per_warp}"
                )

    for kind, capacity in (
        ("smem", machine.smem_bytes),
        ("tmem", machine.tmem_cols),
    ):
        owners = [v for v in problem.nodes if problem.footprint(v, kind) > 0]
        for time in range(horizon + 1):
            use = sum(
                problem.footprint(v, kind)
                for v in owners
                for copy in range(copies)
                if live[(v, copy, time)]
            )
            if use > capacity:
                raise ScheduleArtifactError(
                    f"{kind.upper()} capacity exceeded at time {time}: "
                    f"{use} > {capacity}"
                )


def _validate_figure5_constraints(
    problem, cycles, warp, lane_masks, ii, copies, horizon
) -> None:
    """Emitter branch: the frozen extended model, hardcoded."""
    machine = problem.machine
    if not 0 <= machine.regs_fixed_overhead < machine.regs_per_sm:
        raise ScheduleArtifactError(
            "fixed register overhead must leave register capacity"
        )
    if not 0 <= machine.smem_fixed_overhead <= machine.smem_bytes:
        raise ScheduleArtifactError("fixed SMEM overhead exceeds machine capacity")
    if not 0 <= machine.tmem_fixed_cols <= machine.tmem_cols:
        raise ScheduleArtifactError("fixed TMEM overhead exceeds machine capacity")

    live, starts = _figure5_liveness(problem, cycles, ii, copies, horizon)
    required_widths = {
        node_id: required_group_width(node.min_warps)
        for node_id, node in problem.nodes.items()
    }
    ws_data_edges = problem.edges

    for edge in ws_data_edges:
        if (
            edge.distance >= 1
            and edge.src != edge.dst
            and problem.regs.get(edge.src, 0) > 0
            and problem.nodes[edge.src].pipeline in ("CUDA", "SFU", "NONE")
            and not _same_physical_warp(edge, warp, lane_masks)
        ):
            raise ScheduleArtifactError(
                "reg-carried same-lane violation on edge "
                f"{edge.src}->{edge.dst} distance {edge.distance}"
            )

    register_nodes = [
        node_id for node_id in problem.nodes if problem.regs[node_id] > 0
    ]
    for time in range(horizon + 1):
        global_registers = sum(
            register_allocation_words(
                problem.regs[node_id], required_widths[node_id]
            )
            for node_id in register_nodes
            for copy in range(copies)
            if live[(node_id, copy, time)]
        )
        if global_registers + machine.regs_fixed_overhead > machine.regs_per_sm:
            raise ScheduleArtifactError(
                f"register-file capacity exceeded at time {time}: "
                f"{global_registers} + fixed {machine.regs_fixed_overhead} "
                f"> {machine.regs_per_sm}"
            )

        for group in sorted(set(warp.values())):
            group_lanes = {
                lane
                for node_id, owner in warp.items()
                if owner == group
                for lane in lane_masks[node_id]
            }
            for lane in sorted(group_lanes):
                per_warp_registers = sum(
                    -(-problem.regs[node_id] // required_widths[node_id])
                    for node_id in register_nodes
                    for copy in range(copies)
                    if warp[node_id] == group
                    and lane in lane_masks[node_id]
                    and live[(node_id, copy, time)]
                )
                if per_warp_registers > machine.regs_per_warp:
                    raise ScheduleArtifactError(
                        "REGISTERLIMIT violation at "
                        f"time {time}, group {group}, lane {lane}: "
                        f"{per_warp_registers} > {machine.regs_per_warp}"
                    )

    spill_edges: dict[int, list] = {}
    for edge in ws_data_edges:
        if problem.regs.get(edge.src, 0) > 0:
            spill_edges.setdefault(edge.src, []).append(edge)

    for kind, capacity, fixed in (
        ("smem", machine.smem_bytes, machine.smem_fixed_overhead),
        ("tmem", machine.tmem_cols, machine.tmem_fixed_cols),
    ):
        owners = [v for v in problem.nodes if problem.footprint(v, kind) > 0]
        for time in range(horizon + 1):
            use = sum(
                problem.footprint(v, kind)
                for v in owners
                for copy in range(copies)
                if live[(v, copy, time)]
            )
            if kind == "smem":
                use += sum(
                    problem.regs[producer] * 4 + machine.barrier_pair_bytes
                    for producer, edges in spill_edges.items()
                    for copy in range(copies)
                    if starts[(producer, copy)] <= time
                    and any(
                        copy + edge.distance < copies
                        and not _same_physical_warp(
                            edge, warp, lane_masks
                        )
                        and starts[(edge.dst, copy + edge.distance)] >= time
                        for edge in edges
                    )
                )
            if use + fixed > capacity:
                raise ScheduleArtifactError(
                    f"{kind.upper()} capacity exceeded at time {time}: "
                    f"{use} + fixed {fixed} > {capacity}"
                )


def _register_predecessors(problem) -> dict[int, list]:
    result: dict[int, list] = {}
    for edge in problem.edges:
        if edge.src == edge.dst or problem.regs.get(edge.src, 0) == 0:
            continue
        result.setdefault(edge.dst, []).append(edge)
    return result


def _concurrency_window_clash(
    problem, cycles, ii, copies, consumer, other
) -> None:
    window = problem.lat[other]
    if window == 0:
        return
    for consumer_copy in range(copies):
        consumer_cycle = cycles[consumer] + consumer_copy * ii
        for other_copy in range(copies):
            other_cycle = cycles[other] + other_copy * ii
            if consumer_cycle - (window - 1) <= other_cycle <= consumer_cycle:
                raise ScheduleArtifactError(
                    "CONCURRENCY violation: "
                    f"node {other} copy {other_copy} at {other_cycle} "
                    f"overlaps blocking node {consumer} copy "
                    f"{consumer_copy} at {consumer_cycle}"
                )


def _validate_concurrency_paper(problem, cycles, warp_sets, ii, copies) -> None:
    blocking_consumers = {consumer for _, consumer in problem.blocking}
    register_predecessors = _register_predecessors(problem)
    for consumer in problem.nodes:
        has_cross_warp_predecessor = any(
            not _same_warp_set(edge, warp_sets)
            for edge in register_predecessors.get(consumer, ())
        )
        if consumer not in blocking_consumers and not has_cross_warp_predecessor:
            continue
        for other in problem.nodes:
            # Co-resident == sharing at least one physical warp, so an
            # overlapping (but unequal) set still participates.
            if other == consumer or not set(warp_sets[other]).intersection(
                warp_sets[consumer]
            ):
                continue
            _concurrency_window_clash(
                problem, cycles, ii, copies, consumer, other
            )


def _validate_concurrency(problem, cycles, warp, lane_masks, ii, copies) -> None:
    blocking_consumers = {consumer for _, consumer in problem.blocking}
    register_predecessors = _register_predecessors(problem)
    for consumer in problem.nodes:
        has_cross_warp_predecessor = any(
            not _same_physical_warp(edge, warp, lane_masks)
            for edge in register_predecessors.get(consumer, ())
        )
        if consumer not in blocking_consumers and not has_cross_warp_predecessor:
            continue
        for other in problem.nodes:
            if (
                other == consumer
                or warp[other] != warp[consumer]
                or not set(lane_masks[other]).intersection(lane_masks[consumer])
            ):
                continue
            _concurrency_window_clash(
                problem, cycles, ii, copies, consumer, other
            )


@dataclass(frozen=True)
class ScheduledNode:
    node_id: int
    op_ref: str
    op_kind: str
    pipeline: str
    cycle: int
    stage: int
    offset: int
    group: int | None = None
    group_width: int | None = None
    lanes: tuple[int, ...] = ()
    warps: tuple[int, ...] = ()


@dataclass(frozen=True)
class ScheduledEdge:
    index: int
    src: int
    dst: int
    distance: int
    latency: int
    ws_semantics: bool
    spill_cost: int
    producer_group: int | None = None
    consumer_group: int | None = None
    producer_lanes: tuple[int, ...] = ()
    consumer_lanes: tuple[int, ...] = ()
    producer_warps: tuple[int, ...] = ()
    consumer_warps: tuple[int, ...] = ()


@dataclass(frozen=True)
class SchedulePlan:
    ii: int
    length: int
    copies: int
    horizon: int
    nodes: tuple[ScheduledNode, ...]
    edges: tuple[ScheduledEdge, ...]
    solution_sha256: str
    model: str
    provenance: dict
    machine: dict
    normalization_u: int
    group_widths: dict[int, int] = field(default_factory=dict)

    @property
    def cycles(self) -> dict[int, int]:
        return {node.node_id: node.cycle for node in self.nodes}

    @property
    def warp_sets(self) -> dict[int, tuple[int, ...]]:
        return {node.node_id: node.warps for node in self.nodes}

    @property
    def warp(self) -> dict[int, int]:
        return {
            node.node_id: node.group for node in self.nodes if node.group is not None
        }

    @property
    def lane_masks(self) -> dict[int, tuple[int, ...]]:
        return {
            node.node_id: node.lanes for node in self.nodes if node.group is not None
        }

    @property
    def experiment_inputs(self) -> dict:
        return dict(self.provenance.get("experiment_inputs") or {})

    @property
    def ddg_sha256(self) -> str:
        return self.provenance["ddg_sha256"]

    @property
    def solver_sources_sha256(self) -> str:
        return self.provenance["solver_sources_sha256"]

    @property
    def baseline_graph_sha256(self) -> str | None:
        if self.model == PAPER_MODEL_PROFILE:
            return self.provenance["curation_source"]["baseline_graph_sha256"]
        return self.provenance["baseline_graph_sha256"]

    def _warp_traces(self) -> list[dict]:
        """Per-warp issue order.  A node with |W(v)| > 1 appears in every
        member warp's trace; the sort key is warp-independent, so co-resident
        nodes keep the same relative order in every warp they share (I-5)."""
        warps = sorted({warp for node in self.nodes for warp in node.warps})
        return [
            {
                "id": physical_warp,
                "issue_trace": [
                    {
                        "node": node.node_id,
                        "cycle": node.cycle,
                        "stage": node.stage,
                        "offset": node.offset,
                    }
                    # Intra-cycle order is model-undetermined; per-warp
                    # expansion with a node_id-ascending tiebreak is a
                    # documented serialization choice.
                    for node in sorted(
                        (n for n in self.nodes if physical_warp in n.warps),
                        key=lambda n: (n.stage, n.offset, n.node_id),
                    )
                ],
            }
            for physical_warp in warps
        ]

    def _paper_manifest(self) -> dict:
        return {
            "warps": self._warp_traces(),
            "nodes": [
                {
                    "id": node.node_id,
                    "op_ref": node.op_ref,
                    "op_kind": node.op_kind,
                    "pipeline": node.pipeline,
                    "cycle": node.cycle,
                    "stage": node.stage,
                    "offset": node.offset,
                    "warps": list(node.warps),
                }
                for node in self.nodes
            ],
            "edges": [
                {
                    "index": edge.index,
                    "src": edge.src,
                    "dst": edge.dst,
                    "distance": edge.distance,
                    "latency": edge.latency,
                    "producer_warps": list(edge.producer_warps),
                    "consumer_warps": list(edge.consumer_warps),
                    "ws_semantics": edge.ws_semantics,
                    "spill_cost": edge.spill_cost,
                }
                for edge in self.edges
            ],
        }

    def _emitter_manifest(self) -> dict:
        return {
            "groups": [
                {
                    "id": group,
                    "width": width,
                    "issue_trace": [
                        {
                            "node": node.node_id,
                            "cycle": node.cycle,
                            "stage": node.stage,
                            "offset": node.offset,
                            "lanes": list(node.lanes),
                        }
                        for node in sorted(
                            (n for n in self.nodes if n.group == group),
                            key=lambda n: (n.stage, n.offset, n.node_id),
                        )
                    ],
                }
                for group, width in sorted(self.group_widths.items())
            ],
            "nodes": [
                {
                    "id": node.node_id,
                    "op_ref": node.op_ref,
                    "op_kind": node.op_kind,
                    "pipeline": node.pipeline,
                    "cycle": node.cycle,
                    "stage": node.stage,
                    "offset": node.offset,
                    "group": node.group,
                    "group_width": node.group_width,
                    "lanes": list(node.lanes),
                }
                for node in self.nodes
            ],
            "edges": [
                {
                    "index": edge.index,
                    "src": edge.src,
                    "dst": edge.dst,
                    "distance": edge.distance,
                    "latency": edge.latency,
                    "producer_group": edge.producer_group,
                    "consumer_group": edge.consumer_group,
                    "producer_lanes": list(edge.producer_lanes),
                    "consumer_lanes": list(edge.consumer_lanes),
                    "ws_semantics": edge.ws_semantics,
                    "spill_cost": edge.spill_cost,
                }
                for edge in self.edges
            ],
        }

    def manifest(self) -> dict:
        manifest = {
            "schema_version": _SOLUTION_SCHEMAS[self.model],
            "ii": self.ii,
            "length": self.length,
            "copies": self.copies,
            "horizon": self.horizon,
            "solution_sha256": self.solution_sha256,
            "provenance": dict(self.provenance),
        }
        if self.model == PAPER_MODEL_PROFILE:
            manifest.update(self._paper_manifest())
        else:
            manifest.update(self._emitter_manifest())
        return manifest


def _load_machine(raw: dict) -> MachineModel:
    if not isinstance(raw, dict):
        raise ScheduleArtifactError("provenance.machine must be an object")
    values = dict(raw)
    if "group_widths" in values:
        values["group_widths"] = tuple(values["group_widths"])
    try:
        return MachineModel(**values)
    except TypeError as error:
        raise ScheduleArtifactError(f"invalid machine manifest: {error}") from error


_COMMON_SOLUTION_FIELDS = (
    "ii",
    "length",
    "copies",
    "horizon",
    "cycles",
    "stats",
    "provenance",
    "status",
    "satisfiable",
)
_PAPER_SOLUTION_FIELDS = ("warp_sets",)
_EMITTER_SOLUTION_FIELDS = ("warp", "group_widths", "lane_masks")


def _require_modern_solution(payload: dict, model_profile: str) -> None:
    schema = _SOLUTION_SCHEMAS[model_profile]
    if model_profile == PAPER_MODEL_PROFILE:
        required = set(_COMMON_SOLUTION_FIELDS) | set(_PAPER_SOLUTION_FIELDS)
        forbidden = sorted(set(_EMITTER_SOLUTION_FIELDS) & payload.keys())
    else:
        required = set(_COMMON_SOLUTION_FIELDS) | set(_EMITTER_SOLUTION_FIELDS)
        forbidden = []
    if payload.get("schema_version") != schema:
        missing = sorted(required - payload.keys())
        suffix = f"; missing {', '.join(missing)}" if missing else ""
        raise LegacySolutionError(
            f"solution is not {schema}{suffix}; refusing to infer "
            "physical scheduling data; rerun the joint solver"
        )
    missing = sorted(required - payload.keys())
    if missing:
        raise ScheduleArtifactError(
            "versioned solution is missing required fields: " + ", ".join(missing)
        )
    if forbidden:
        raise ScheduleArtifactError(
            "paper solutions carry warp_sets only; found group/lane fields: "
            + ", ".join(forbidden)
        )
    if payload["status"] != "sat" or payload["satisfiable"] is not True:
        raise ScheduleArtifactError(
            "only a satisfiable, optimal-search result can be compiled"
        )


def _require_provenance_model(provenance: dict, model_profile: str) -> None:
    model = provenance.get("model")
    if model != model_profile:
        raise ScheduleArtifactError(
            f"solution model {model!r} is not the requested production model "
            f"{model_profile!r}"
        )


def _check_hash_chain(
    provenance: dict,
    model_profile: str,
    graph_path: Path,
    baseline_graph_path: Path | None,
) -> None:
    expected = {
        "ddg_sha256": sha256_file(graph_path),
        "solver_sources_sha256": solver_sources_sha256(),
    }
    if model_profile == EMITTER_MODEL_PROFILE:
        expected["baseline_graph_sha256"] = sha256_file(baseline_graph_path)
    for key, actual in expected.items():
        if provenance.get(key) != actual:
            raise ScheduleArtifactError(
                f"{key} mismatch: solution={provenance.get(key)!r}, "
                f"input={actual!r}"
            )
    if model_profile != PAPER_MODEL_PROFILE:
        return
    recorded = provenance.get("curation_source")
    if not isinstance(recorded, dict):
        raise ScheduleArtifactError(
            "provenance.curation_source must be an object"
        )
    curated = curation_source_of(graph_path)
    if recorded != curated:
        raise ScheduleArtifactError(
            f"curation_source mismatch: solution={recorded!r}, "
            f"input={curated!r}"
        )
    if baseline_graph_path is not None:
        baseline = sha256_file(baseline_graph_path)
        if recorded.get("baseline_graph_sha256") != baseline:
            raise ScheduleArtifactError(
                "baseline graph is not the one the input was curated from: "
                f"solution={recorded.get('baseline_graph_sha256')!r}, "
                f"input={baseline!r}"
            )


def _require_normalization_u(provenance: dict) -> int:
    normalization_u = provenance.get("normalization_u")
    if (
        isinstance(normalization_u, bool)
        or not isinstance(normalization_u, int)
        or normalization_u <= 0
    ):
        raise ScheduleArtifactError(
            "provenance.normalization_u must be a positive integer"
        )
    return normalization_u


def _require_zero_fixed_overheads(raw_machine: dict) -> None:
    for name in _PAPER_ZERO_MACHINE_FIELDS:
        if raw_machine.get(name) != 0:
            raise ScheduleArtifactError(
                f"paper artifacts carry no fixed overhead: "
                f"provenance.machine.{name}={raw_machine.get(name)!r}"
            )


def _require_pipeline_shape(ii: int, length: int, copies: int, horizon: int) -> None:
    if ii < 1 or length < 0:
        raise ScheduleArtifactError(f"invalid II/length: {ii}/{length}")
    expected_copies = -(-length // ii)
    expected_horizon = (copies - 1) * ii + length
    if copies != expected_copies:
        raise ScheduleArtifactError(
            f"copies={copies} != ceil(length/ii)={expected_copies}"
        )
    if horizon != expected_horizon:
        raise ScheduleArtifactError(f"horizon={horizon} != {expected_horizon}")


def _validate_timing(problem, cycles, edge_spill_costs, ii, length) -> None:
    for node_id, cycle in cycles.items():
        if cycle < 0 or cycle + max(1, problem.lat[node_id]) > length:
            raise ScheduleArtifactError(
                f"node {node_id} violates COMPLETION: cycle={cycle}, "
                f"latency={problem.lat[node_id]}, length={length}"
            )
    for edge in problem.edges:
        available = cycles[edge.dst] + edge.distance * ii
        edge_delay = problem.edge_latency(edge) + edge_spill_costs[edge.index]
        required = cycles[edge.src] + edge_delay
        if available < required:
            raise ScheduleArtifactError(
                "dependence violation "
                f"{edge.src}->{edge.dst} distance {edge.distance}: "
                f"{available} < {required}"
            )
    machine = problem.machine
    for pipeline, capacity in machine.capacities.items():
        for offset in range(ii):
            use = sum(
                amount
                for node_id in problem.nodes
                for resource, occupied, amount in problem.reservations(node_id)
                if resource == pipeline and (cycles[node_id] + occupied) % ii == offset
            )
            if use > capacity:
                raise ScheduleArtifactError(
                    f"{pipeline} capacity exceeded at modulo offset "
                    f"{offset}: {use} > {capacity}"
                )


def _require_node_cover(assignment: dict, all_nodes: set[int], name: str) -> None:
    if set(assignment) != all_nodes:
        raise ScheduleArtifactError(
            f"{name} must cover exactly the DDG nodes: "
            f"missing={sorted(all_nodes - set(assignment))}, "
            f"extra={sorted(set(assignment) - all_nodes)}"
        )


def _validate_warp_sets(problem, warp_sets, machine: MachineModel) -> None:
    _require_node_cover(warp_sets, set(problem.nodes), "warp_sets")
    for node_id in sorted(warp_sets):
        warps = warp_sets[node_id]
        if not warps:
            raise ScheduleArtifactError(f"node {node_id} has an empty warp set")
        if any(warp < 0 or warp >= machine.num_warps for warp in warps):
            raise ScheduleArtifactError(
                f"node {node_id} warp set {list(warps)} leaves the physical "
                f"budget [0, {machine.num_warps})"
            )
        # WARPUNIQUENESS, multi-warp reading (I-3): exactly warps(v) distinct
        # physical warps, with no adjacency, alignment or power-of-two rule.
        required = problem.nodes[node_id].min_warps
        if len(warps) != required:
            raise ScheduleArtifactError(
                f"node {node_id} occupies {len(warps)} physical warps, "
                f"requires {required}"
            )
    used = {warp for warps in warp_sets.values() for warp in warps}
    if len(used) > machine.num_warps:
        raise ScheduleArtifactError(
            f"schedule uses {len(used)} physical warps, budget is "
            f"{machine.num_warps}"
        )


def _validate_variable_latency_paper(problem, warp_sets) -> None:
    """v is variable-latency iff warp 0 is one of its physical warps; the
    other members of a multi-warp W_vl may be shared (spec-4 I-5)."""
    misplaced = [
        node_id
        for node_id in problem.nodes
        if (node_id in problem.variable_latency) != (0 in warp_sets[node_id])
    ]
    if misplaced:
        raise ScheduleArtifactError(
            "VARIABLELATENCY iff placement violated by nodes "
            f"{sorted(misplaced)}"
        )


def _validate_variable_latency_groups(problem, warp) -> None:
    vl_groups = {warp[node_id] for node_id in problem.variable_latency}
    if len(vl_groups) > 1:
        raise ScheduleArtifactError(
            f"variable-latency nodes span groups {sorted(vl_groups)}"
        )
    if not vl_groups:
        return
    vl_group = next(iter(vl_groups))
    misplaced = [
        node_id
        for node_id, group in warp.items()
        if (node_id in problem.variable_latency) != (group == vl_group)
    ]
    if misplaced:
        raise ScheduleArtifactError(
            "VARIABLELATENCY iff placement violated by nodes "
            f"{sorted(misplaced)}"
        )


def _validate_emitter_groups(problem, warp, group_widths, lane_masks) -> None:
    machine = problem.machine
    for group, width in group_widths.items():
        if width not in machine.group_widths:
            raise ScheduleArtifactError(f"group {group} has unsupported width {width}")
        members = [node_id for node_id, owner in warp.items() if owner == group]
        active_warps = len(
            {lane for node_id in members for lane in lane_masks[node_id]}
        )
        required = physical_group_width(active_warps, machine.group_widths)
        if width != required:
            raise ScheduleArtifactError(
                f"group {group} width {width} != allocation width {required} "
                f"for {active_warps} active warps"
            )
        for node_id in members:
            lanes = lane_masks[node_id]
            node_width = required_group_width(problem.nodes[node_id].min_warps)
            if len(lanes) != node_width or len(set(lanes)) != len(lanes):
                raise ScheduleArtifactError(
                    f"node {node_id} has invalid lane mask {lanes}; "
                    f"requires {node_width} lanes"
                )
            if any(lane < 0 or lane >= width for lane in lanes):
                raise ScheduleArtifactError(
                    f"node {node_id} lane mask {lanes} exceeds group " f"width {width}"
                )
            # The emitter's lane model is prefix + full group, hardcoded.
            if lanes != tuple(range(node_width)):
                raise ScheduleArtifactError(
                    f"node {node_id} lane mask {lanes} is not the required "
                    f"emitter prefix {tuple(range(node_width))}"
                )
            if width != node_width:
                raise ScheduleArtifactError(
                    f"node {node_id} width {node_width} does not occupy its "
                    f"full emitter group width {width}"
                )


def _require_stats(payload: dict) -> None:
    # Solver statistics are observations; nothing here compares their values.
    if not isinstance(payload["stats"], dict):
        raise ScheduleArtifactError("stats must be an object")


def _load_context(
    solution_path: Path,
    graph_path: Path,
    baseline_graph_path: Path | None,
    model_profile: str,
):
    payload = json.loads(solution_path.read_text())
    if not isinstance(payload, dict):
        raise ScheduleArtifactError("solution root must be an object")
    _require_modern_solution(payload, model_profile)
    provenance = payload["provenance"]
    if not isinstance(provenance, dict):
        raise ScheduleArtifactError("provenance must be an object")
    _require_provenance_model(provenance, model_profile)
    _check_hash_chain(provenance, model_profile, graph_path, baseline_graph_path)
    raw_machine = provenance.get("machine")
    if not isinstance(raw_machine, dict):
        raise ScheduleArtifactError("provenance.machine must be an object")
    if model_profile == PAPER_MODEL_PROFILE:
        _require_zero_fixed_overheads(raw_machine)
    machine = _load_machine(raw_machine)
    normalization_u = _require_normalization_u(provenance)

    from .ddg import load_problem

    problem = load_problem(graph_path, machine=machine, u=normalization_u)
    cycles = _int_dict(payload["cycles"], "cycles")
    _require_node_cover(cycles, set(problem.nodes), "cycles")
    _require_pipeline_shape(
        int(payload["ii"]),
        int(payload["length"]),
        int(payload["copies"]),
        int(payload["horizon"]),
    )
    _require_stats(payload)
    return payload, provenance, machine, problem, cycles


def _paper_plan(
    payload, provenance, machine, problem, cycles, solution_sha256
) -> SchedulePlan:
    ii = int(payload["ii"])
    length = int(payload["length"])
    copies = int(payload["copies"])
    horizon = int(payload["horizon"])
    warp_sets = _warp_sets_dict(payload["warp_sets"])
    _validate_warp_sets(problem, warp_sets, machine)

    edge_spill_costs = {
        edge.index: _paper_edge_spill_cost(problem, edge, warp_sets)
        for edge in problem.edges
    }
    _validate_timing(problem, cycles, edge_spill_costs, ii, length)
    _validate_figure5_paper(problem, cycles, warp_sets, ii, copies, horizon)
    _validate_concurrency_paper(problem, cycles, warp_sets, ii, copies)
    _validate_variable_latency_paper(problem, warp_sets)

    nodes = tuple(
        ScheduledNode(
            node_id=node_id,
            op_ref=problem.nodes[node_id].op_ref,
            op_kind=problem.nodes[node_id].op_kind,
            pipeline=problem.nodes[node_id].pipeline,
            cycle=cycles[node_id],
            stage=cycles[node_id] // ii,
            offset=cycles[node_id] % ii,
            warps=warp_sets[node_id],
        )
        for node_id in sorted(problem.nodes)
    )
    edges = tuple(
        ScheduledEdge(
            index=edge.index,
            src=edge.src,
            dst=edge.dst,
            distance=edge.distance,
            latency=problem.edge_latency(edge),
            producer_warps=warp_sets[edge.src],
            consumer_warps=warp_sets[edge.dst],
            # Curated E has no infra edges left (G1).
            ws_semantics=True,
            spill_cost=edge_spill_costs[edge.index],
        )
        for edge in problem.edges
    )
    return SchedulePlan(
        ii=ii,
        length=length,
        copies=copies,
        horizon=horizon,
        nodes=nodes,
        edges=edges,
        solution_sha256=solution_sha256,
        model=PAPER_MODEL_PROFILE,
        provenance=dict(provenance),
        machine=machine_manifest(machine),
        normalization_u=provenance["normalization_u"],
    )


def _emitter_plan(
    payload, provenance, machine, problem, cycles, solution_sha256
) -> SchedulePlan:
    ii = int(payload["ii"])
    length = int(payload["length"])
    copies = int(payload["copies"])
    horizon = int(payload["horizon"])
    warp = _int_dict(payload["warp"], "warp")
    group_widths = _int_dict(payload["group_widths"], "group_widths")
    lane_masks = _lane_dict(payload["lane_masks"])
    all_nodes = set(problem.nodes)
    _require_node_cover(warp, all_nodes, "warp")
    _require_node_cover(lane_masks, all_nodes, "lane_masks")
    used_groups = set(warp.values())
    if set(group_widths) != used_groups:
        raise ScheduleArtifactError(
            "group_widths must cover exactly used groups: "
            f"missing={sorted(used_groups - set(group_widths))}, "
            f"extra={sorted(set(group_widths) - used_groups)}"
        )

    edge_spill_costs = {
        edge.index: _edge_spill_cost(problem, edge, warp, lane_masks)
        for edge in problem.edges
    }
    _validate_timing(problem, cycles, edge_spill_costs, ii, length)
    _validate_emitter_groups(problem, warp, group_widths, lane_masks)
    _validate_figure5_constraints(
        problem, cycles, warp, lane_masks, ii, copies, horizon
    )
    _validate_concurrency(problem, cycles, warp, lane_masks, ii, copies)
    scheduled_warps = sum(group_widths.values())
    if scheduled_warps > machine.scheduled_warp_budget():
        raise ScheduleArtifactError(
            f"schedule uses {scheduled_warps} physical warps, budget is "
            f"{machine.scheduled_warp_budget()}"
        )
    _validate_variable_latency_groups(problem, warp)

    nodes = tuple(
        ScheduledNode(
            node_id=node_id,
            op_ref=problem.nodes[node_id].op_ref,
            op_kind=problem.nodes[node_id].op_kind,
            pipeline=problem.nodes[node_id].pipeline,
            cycle=cycles[node_id],
            stage=cycles[node_id] // ii,
            offset=cycles[node_id] % ii,
            group=warp[node_id],
            group_width=group_widths[warp[node_id]],
            lanes=lane_masks[node_id],
        )
        for node_id in sorted(problem.nodes)
    )
    edges = tuple(
        ScheduledEdge(
            index=edge.index,
            src=edge.src,
            dst=edge.dst,
            distance=edge.distance,
            latency=problem.edge_latency(edge),
            producer_group=warp[edge.src],
            consumer_group=warp[edge.dst],
            producer_lanes=lane_masks[edge.src],
            consumer_lanes=lane_masks[edge.dst],
            ws_semantics=True,
            spill_cost=edge_spill_costs[edge.index],
        )
        for edge in problem.edges
    )
    return SchedulePlan(
        ii=ii,
        length=length,
        copies=copies,
        horizon=horizon,
        nodes=nodes,
        edges=edges,
        group_widths=group_widths,
        solution_sha256=solution_sha256,
        model=EMITTER_MODEL_PROFILE,
        provenance=dict(provenance),
        machine=machine_manifest(machine),
        normalization_u=provenance["normalization_u"],
    )


def load_schedule_plan(
    solution_path: str | Path,
    graph_path: str | Path,
    baseline_graph_path: str | Path | None = None,
    *,
    model_profile: str = PAPER_MODEL_PROFILE,
) -> SchedulePlan:
    """Revalidate a solution artifact against the graph that produced it.

    ``graph_path`` is the curated ddg on the paper path and the raw ddg on the
    emitter path.  ``baseline_graph_path`` is required by the emitter branch
    and is an IR-emission input on the paper branch, where it is checked
    against the baseline the input was curated from.
    """
    if model_profile not in _SOLUTION_SCHEMAS:
        raise ScheduleArtifactError(
            f"unknown production model profile: {model_profile!r}"
        )
    solution_path = Path(solution_path)
    graph_path = Path(graph_path)
    if baseline_graph_path is not None:
        baseline_graph_path = Path(baseline_graph_path)
    elif model_profile == EMITTER_MODEL_PROFILE:
        raise ScheduleArtifactError(
            "the emitter branch requires its baseline schedule graph"
        )
    payload, provenance, machine, problem, cycles = _load_context(
        solution_path, graph_path, baseline_graph_path, model_profile
    )
    build = (
        _paper_plan if model_profile == PAPER_MODEL_PROFILE else _emitter_plan
    )
    return build(
        payload,
        provenance,
        machine,
        problem,
        cycles,
        sha256_file(solution_path),
    )


def load_schedule_context(
    solution_path: str | Path,
    graph_path: str | Path,
    model: str = PAPER_MODEL_PROFILE,
    baseline_graph_path: str | Path | None = None,
):
    """Load a validated plan and reconstruct its exact normalized problem."""
    plan = load_schedule_plan(
        solution_path,
        graph_path,
        baseline_graph_path,
        model_profile=model,
    )
    machine = _load_machine(plan.machine)
    from .ddg import load_problem

    problem = load_problem(
        graph_path,
        machine=machine,
        u=plan.normalization_u,
    )
    return problem, plan
