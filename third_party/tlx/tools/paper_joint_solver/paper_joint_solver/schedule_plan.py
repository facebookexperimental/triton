"""Strict, versioned handoff from the joint solver to pipelined IR.

The paper's schedule is more than ``(II, L)``.  A compilable artifact owns
every operation's cycle, physical group, and physical-lane mask, and it is
valid only for the DDG, schedule graph, machine model, and solver sources that
produced it.  This module validates that contract before Twill emits its
software-pipelined, warp-annotated IR.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .machine import MachineModel, physical_group_width, required_group_width


SOLUTION_SCHEMA = "twill-joint-solution-v1"
_SOLVER_SOURCES = (
    "ddg.py",
    "joint_smt.py",
    "machine.py",
    "normalize.py",
    "resource_model.py",
    "search.py",
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


def solution_provenance(
    ddg_path: str | Path,
    baseline_graph_path: str | Path | None,
    machine: MachineModel,
    normalization_u: int,
) -> dict:
    return {
        "ddg_sha256": sha256_file(ddg_path),
        "baseline_graph_sha256": (
            sha256_file(baseline_graph_path)
            if baseline_graph_path is not None
            else None
        ),
        "solver_sources_sha256": solver_sources_sha256(),
        "normalization_u": normalization_u,
        "machine": machine_manifest(machine),
    }


def _int_dict(raw: dict, name: str) -> dict[int, int]:
    try:
        return {int(key): int(value) for key, value in raw.items()}
    except (AttributeError, TypeError, ValueError) as error:
        raise ScheduleArtifactError(f"{name} must be an integer map") from error


def _lane_dict(raw: dict) -> dict[int, tuple[int, ...]]:
    try:
        return {
            int(key): tuple(int(lane) for lane in lanes) for key, lanes in raw.items()
        }
    except (AttributeError, TypeError, ValueError) as error:
        raise ScheduleArtifactError(
            "lane_masks must map node ids to integer lane lists"
        ) from error


@dataclass(frozen=True)
class ScheduledNode:
    node_id: int
    op_ref: str
    op_kind: str
    pipeline: str
    cycle: int
    stage: int
    offset: int
    group: int | None
    group_width: int | None
    lanes: tuple[int, ...]


@dataclass(frozen=True)
class ScheduledEdge:
    index: int
    src: int
    dst: int
    distance: int
    latency: int
    producer_group: int | None
    consumer_group: int | None


@dataclass(frozen=True)
class SchedulePlan:
    ii: int
    length: int
    copies: int
    horizon: int
    nodes: tuple[ScheduledNode, ...]
    edges: tuple[ScheduledEdge, ...]
    group_widths: dict[int, int]
    solution_sha256: str
    ddg_sha256: str
    baseline_graph_sha256: str
    solver_sources_sha256: str
    machine: dict
    normalization_u: int

    @property
    def cycles(self) -> dict[int, int]:
        return {node.node_id: node.cycle for node in self.nodes}

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

    def manifest(self) -> dict:
        return {
            "schema_version": SOLUTION_SCHEMA,
            "ii": self.ii,
            "length": self.length,
            "copies": self.copies,
            "horizon": self.horizon,
            "solution_sha256": self.solution_sha256,
            "provenance": {
                "ddg_sha256": self.ddg_sha256,
                "baseline_graph_sha256": self.baseline_graph_sha256,
                "solver_sources_sha256": self.solver_sources_sha256,
                "normalization_u": self.normalization_u,
                "machine": self.machine,
            },
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
                }
                for edge in self.edges
            ],
        }


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


def _require_modern_solution(payload: dict) -> None:
    required = {
        "ii",
        "length",
        "copies",
        "horizon",
        "cycles",
        "warp",
        "group_widths",
        "lane_masks",
        "stats",
        "provenance",
        "status",
        "satisfiable",
    }
    missing = sorted(required - payload.keys())
    if payload.get("schema_version") != SOLUTION_SCHEMA:
        physical = sorted({"group_widths", "lane_masks"} - payload.keys())
        suffix = f"; missing {', '.join(physical)}" if physical else ""
        raise LegacySolutionError(
            f"solution is not {SOLUTION_SCHEMA}{suffix}; refusing to infer "
            "physical scheduling data; rerun the joint solver"
        )
    if missing:
        raise ScheduleArtifactError(
            "versioned solution is missing required fields: " + ", ".join(missing)
        )
    if payload["status"] != "sat" or payload["satisfiable"] is not True:
        raise ScheduleArtifactError(
            "only a satisfiable, optimal-search result can be compiled"
        )


def _maximum_useful_group_labels(problem, machine: MachineModel) -> int:
    required_width = {
        node_id: required_group_width(node.min_warps)
        for node_id, node in problem.nodes.items()
    }
    variable_width = max(
        (required_width[node_id] for node_id in problem.variable_latency),
        default=0,
    )
    remaining_budget = machine.scheduled_warp_budget() - variable_width
    regular_widths = [
        width
        for node_id, width in required_width.items()
        if node_id not in problem.variable_latency
    ]
    if not regular_widths:
        return 1
    widest = max(regular_widths)
    regular_widths.remove(widest)
    used = widest
    regular_groups = 1
    for width in sorted(regular_widths):
        if used + width > remaining_budget:
            break
        used += width
        regular_groups += 1
    return 1 + regular_groups


def load_schedule_plan(
    solution_path: str | Path,
    ddg_path: str | Path,
    baseline_graph_path: str | Path,
) -> SchedulePlan:
    solution_path = Path(solution_path)
    ddg_path = Path(ddg_path)
    baseline_graph_path = Path(baseline_graph_path)
    payload = json.loads(solution_path.read_text())
    if not isinstance(payload, dict):
        raise ScheduleArtifactError("solution root must be an object")
    _require_modern_solution(payload)

    provenance = payload["provenance"]
    if not isinstance(provenance, dict):
        raise ScheduleArtifactError("provenance must be an object")
    expected = {
        "ddg_sha256": sha256_file(ddg_path),
        "baseline_graph_sha256": sha256_file(baseline_graph_path),
        "solver_sources_sha256": solver_sources_sha256(),
    }
    for key, actual in expected.items():
        if provenance.get(key) != actual:
            raise ScheduleArtifactError(
                f"{key} mismatch: solution={provenance.get(key)!r}, "
                f"input={actual!r}"
            )

    machine = _load_machine(provenance.get("machine"))
    normalization_u = provenance.get("normalization_u")
    if not isinstance(normalization_u, int) or normalization_u <= 0:
        raise ScheduleArtifactError(
            "provenance.normalization_u must be a positive integer"
        )

    from .ddg import load_problem

    problem = load_problem(
        ddg_path,
        machine=machine,
        u=normalization_u,
        baseline_graph=baseline_graph_path,
    )
    cycles = _int_dict(payload["cycles"], "cycles")
    warp = _int_dict(payload["warp"], "warp")
    group_widths = _int_dict(payload["group_widths"], "group_widths")
    lane_masks = _lane_dict(payload["lane_masks"])

    all_nodes = set(problem.nodes)
    assignable = all_nodes
    if set(cycles) != all_nodes:
        raise ScheduleArtifactError(
            "cycles must cover exactly the DDG nodes: "
            f"missing={sorted(all_nodes - set(cycles))}, "
            f"extra={sorted(set(cycles) - all_nodes)}"
        )
    if set(warp) != assignable:
        raise ScheduleArtifactError(
            "warp must cover exactly the DDG nodes: "
            f"missing={sorted(assignable - set(warp))}, "
            f"extra={sorted(set(warp) - assignable)}"
        )
    if set(lane_masks) != assignable:
        raise ScheduleArtifactError(
            "lane_masks must cover exactly the DDG nodes: "
            f"missing={sorted(assignable - set(lane_masks))}, "
            f"extra={sorted(set(lane_masks) - assignable)}"
        )
    used_groups = set(warp.values())
    if set(group_widths) != used_groups:
        raise ScheduleArtifactError(
            "group_widths must cover exactly used groups: "
            f"missing={sorted(used_groups - set(group_widths))}, "
            f"extra={sorted(set(group_widths) - used_groups)}"
        )

    ii = int(payload["ii"])
    length = int(payload["length"])
    copies = int(payload["copies"])
    horizon = int(payload["horizon"])
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

    for node_id, cycle in cycles.items():
        if cycle < 0 or cycle + max(1, problem.lat[node_id]) > length:
            raise ScheduleArtifactError(
                f"node {node_id} violates COMPLETION: cycle={cycle}, "
                f"latency={problem.lat[node_id]}, length={length}"
            )
    for edge in problem.edges:
        available = cycles[edge.dst] + edge.distance * ii
        edge_delay = problem.edge_latency(edge)
        if edge.src in assignable and edge.dst in assignable:
            same_lanes = (
                warp[edge.src] == warp[edge.dst]
                and lane_masks[edge.src] == lane_masks[edge.dst]
            )
            if not same_lanes:
                edge_delay += problem.spill.get(edge.src, 0)
        required = cycles[edge.src] + edge_delay
        if available < required:
            raise ScheduleArtifactError(
                "dependence violation "
                f"{edge.src}->{edge.dst} distance {edge.distance}: "
                f"{available} < {required}"
            )

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

    scheduled_warps = sum(group_widths.values())
    if scheduled_warps > machine.scheduled_warp_budget():
        raise ScheduleArtifactError(
            f"schedule uses {scheduled_warps} physical warps, budget is "
            f"{machine.scheduled_warp_budget()}"
        )
    stats = payload["stats"]
    if not isinstance(stats, dict):
        raise ScheduleArtifactError("stats must be an object")
    expected_stats = {
        "T": horizon,
        "num_groups": len(group_widths),
        "scheduled_warps": scheduled_warps,
        "fixed_warps": machine.warp_fixed_overhead,
        "num_warps": scheduled_warps + machine.warp_fixed_overhead,
        "physical_warp_budget": machine.num_warps,
        "max_groups": _maximum_useful_group_labels(problem, machine),
        "regs_per_warp": machine.regs_per_warp,
        "regs_per_sm": machine.regs_per_sm,
        "regs_fixed_overhead": machine.regs_fixed_overhead,
        "smem_capacity": machine.smem_bytes,
        "smem_fixed_overhead": machine.smem_fixed_overhead,
        "barrier_pair_bytes": machine.barrier_pair_bytes,
        "tmem_capacity": machine.tmem_cols,
        "tmem_fixed_cols": machine.tmem_fixed_cols,
        "emitter_infra_nodes": len(problem.emitter_infra),
        "lane_mask_model": "paper_physical_masks",
    }
    for key, value in expected_stats.items():
        if stats.get(key) != value:
            raise ScheduleArtifactError(
                f"stats.{key}={stats.get(key)!r} != derived {value}"
            )

    vl_groups = {warp[node_id] for node_id in problem.variable_latency}
    if len(vl_groups) > 1:
        raise ScheduleArtifactError(
            f"variable-latency nodes span groups {sorted(vl_groups)}"
        )
    if vl_groups:
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
        solution_sha256=sha256_file(solution_path),
        ddg_sha256=expected["ddg_sha256"],
        baseline_graph_sha256=expected["baseline_graph_sha256"],
        solver_sources_sha256=expected["solver_sources_sha256"],
        machine=machine_manifest(machine),
        normalization_u=normalization_u,
    )
