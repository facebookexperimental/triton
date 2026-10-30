"""Read a curated G and apply the paper's sec 5.2 / 5.3 cost-model logic.

The input is a ``curated-ddg-0.2`` file produced by ``curate_ddg``; its field
authority is the frozen schema document ``curated-ddg-0.2-schema.md``, which
this module does not restate. Everything the paper's formulation needs is an
explicit field there -- pipeline, rrt, regs, spill_cost, per-node smem/tmem
footprints, variable_latency, streaming, and per-edge blocking -- so this
loader performs no classification, filtering or defaulting of its own. Unknown
fields are a hard error rather than something to ignore.

Two pieces of paper logic remain here because they are cost-model rules rather
than input curation:

  * sec 5.3 zeroes the outgoing latency of streaming operations;
  * sec 5.2 normalizes the cost multiset C. Every enumerated entry joins C --
    each RRT segment duration, each edge latency, and each node's spill cost,
    zeros included -- and the ZLP runs once at the requested U, its result
    consumed entry by entry.

The adapter splits every RRT into maximal constant-vector runs so that segment
durations can be normalized independently; the rebuilt RRT satisfies
``cycles(v) == len(RRT[v][f])`` for every row.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from .machine import MachineModel
from .normalize import DEFAULT_U, normalize_costs

CURATED_SCHEMA = "curated-ddg-0.2"

_NODE_FIELDS = frozenset(
    (
        "id",
        "op_ref",
        "op_kind",
        "pipeline",
        "latency",
        "occupancy",
        "rrt",
        "min_warps",
        "regs",
        "spill_cost",
        "smem_footprint",
        "tmem_footprint",
        "variable_latency",
        "streaming",
    )
)
_EDGE_FIELDS = frozenset(("src", "dst", "distance", "latency", "blocking"))
_LOOP_FIELDS = frozenset(("loop_id", "trip_count", "ddg"))
_TOP_FIELDS = frozenset(("schema_version", "curation_source", "loops"))


@dataclass
class Node:
    id: int
    op_ref: str
    op_kind: str
    pipeline: str
    latency: int
    occupancy: int
    min_warps: int
    spill_cost: int = 0


@dataclass
class Edge:
    src: int
    dst: int
    distance: int  # iteration delay (paper's delta)
    latency: int  # clock-cycle delay (paper's d)
    index: int = -1  # stable identity for parallel edges
    blocking: bool = False


@dataclass
class Problem:
    nodes: dict[int, Node]
    edges: list[Edge]
    machine: MachineModel
    trip_count: int | None
    # Curated annotations (paper secs 4.3 / 5.3), read from the file.
    variable_latency: set[int] = field(default_factory=set)
    streaming: set[int] = field(default_factory=set)
    blocking: set[tuple[int, int]] = field(default_factory=set)
    regs: dict[int, int] = field(default_factory=dict)
    smem_footprint: dict[int, int] = field(default_factory=dict)
    tmem_footprint: dict[int, int] = field(default_factory=dict)
    # Normalized costs (paper sec 5.2).
    lat: dict[int, int] = field(default_factory=dict)  # cycles(v), normalized
    # `occ` is compatibility metadata: normalized cycles with any resource
    # demand. All resource constraints consume `rrt` directly.
    occ: dict[int, int] = field(default_factory=dict)
    rrt: dict[int, dict[str, tuple[int, ...]]] = field(default_factory=dict)
    edge_lat: dict[int, int] = field(default_factory=dict)
    normalization_u: int = DEFAULT_U
    normalization_f: int = 0
    normalization_cost_pairs: tuple[tuple[int, int], ...] = ()
    spill: dict[int, int] = field(default_factory=dict)

    def normalization_stats(self) -> dict[str, int]:
        return {
            "normalization_u": self.normalization_u,
            "normalization_f": self.normalization_f,
        }

    def edge_latency(self, edge: Edge) -> int:
        return self.edge_lat[edge.index]

    def footprint(self, node_id: int, kind: str) -> int:
        if kind == "smem":
            return self.smem_footprint.get(node_id, 0)
        if kind == "tmem":
            return self.tmem_footprint.get(node_id, 0)
        raise ValueError(f"unknown storage kind {kind!r}")

    def reservations(self, node_id: int):
        for functional_unit, row in self.rrt[node_id].items():
            for cycle, amount in enumerate(row):
                if amount:
                    yield functional_unit, cycle, amount


def _rrt_amount(value, node_id: int, functional_unit: str, cycle: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"node {node_id} RRT[{functional_unit},{cycle}] must be a "
            "nonnegative integer"
        )
    return value


def _rrt_row(value, node_id: int, functional_unit: str) -> list[int]:
    if isinstance(value, list):
        return [
            _rrt_amount(amount, node_id, functional_unit, cycle)
            for cycle, amount in enumerate(value)
        ]
    if isinstance(value, dict):
        entries = []
        for raw_cycle, amount in value.items():
            try:
                cycle = int(raw_cycle)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"node {node_id} RRT cycle {raw_cycle!r} is not an integer"
                ) from error
            if cycle < 0:
                raise ValueError(
                    f"node {node_id} RRT cycle {cycle} must be nonnegative"
                )
            entries.append(
                (cycle, _rrt_amount(amount, node_id, functional_unit, cycle))
            )
        row = [0] * (max((cycle for cycle, _ in entries), default=-1) + 1)
        for cycle, amount in entries:
            row[cycle] = amount
        return row
    raise ValueError(
        f"node {node_id} RRT row {functional_unit!r} must be a list or map"
    )


def _explicit_rrt(
    raw, node_id: int, latency: int, machine: MachineModel
):
    if not isinstance(raw, dict):
        raise ValueError(f"node {node_id} rrt must be an object")
    rows = {}
    for functional_unit, value in raw.items():
        if functional_unit not in machine.capacities:
            raise ValueError(
                f"node {node_id} uses unknown functional unit "
                f"{functional_unit!r}"
            )
        row = _rrt_row(value, node_id, functional_unit)
        if isinstance(value, dict) and len(row) <= latency:
            row.extend([0] * (latency - len(row)))
        elif len(row) != latency:
            raise ValueError(
                f"node {node_id} RRT row {functional_unit!r} has span "
                f"{len(row)}, expected latency {latency}"
            )
        rows[functional_unit] = row
    return rows


def _rrt_segments(rows: dict[str, list[int]], span: int):
    """Split an RRT into maximal segments with one resource-demand vector.

    Scaling run durations is a compiler-adapter extension: the paper defines
    the final RRT but no projection from compiler cycle measurements. It keeps
    integer demand vectors exact and establishes one final cycles(v) shared by
    COMPLETION and CAPACITY.
    """
    if span == 0:
        return []
    functional_units = tuple(rows)
    segments = []
    vector = tuple(rows[unit][0] for unit in functional_units)
    duration = 1
    for cycle in range(1, span):
        current = tuple(rows[unit][cycle] for unit in functional_units)
        if current == vector:
            duration += 1
        else:
            segments.append((functional_units, vector, duration))
            vector = current
            duration = 1
    segments.append((functional_units, vector, duration))
    return segments


def _append_rrt_segment(rows, functional_units, vector, duration):
    for functional_unit, amount in zip(functional_units, vector):
        rows[functional_unit].extend([amount] * duration)


def _require_fields(obj: dict, allowed: frozenset, label: str) -> None:
    """Curated input is read fail-closed: an unknown field means the file was
    produced by a different curation contract, not something to ignore."""
    unknown = sorted(set(obj) - allowed)
    if unknown:
        raise ValueError(f"{label} has unknown curated field(s) {unknown}")


def _checked_int(value, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _checked_bool(value, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def load_problem(path: str | Path, machine: MachineModel | None = None,
                 loop_index: int = 0, u: int = DEFAULT_U) -> Problem:
    """Read a curated-ddg-0.2 file and apply the paper's sec 5.2/5.3 logic.

    The curated file is the whole semantic input: this loader classifies,
    filters and defaults nothing. Produce one with `curate_ddg` first.
    """
    data = json.loads(Path(path).read_text())
    machine = machine or MachineModel()
    if not isinstance(data, dict):
        raise ValueError("curated ddg must be an object")
    if data.get("schema_version") != CURATED_SCHEMA:
        raise ValueError(
            f"expected schema {CURATED_SCHEMA}, got "
            f"{data.get('schema_version')!r}; run "
            "`python -m paper_joint_solver.curate_ddg` on the raw dump first"
        )
    _require_fields(data, _TOP_FIELDS, "curated ddg")
    loop = data["loops"][loop_index]
    _require_fields(loop, _LOOP_FIELDS, f"loop {loop_index}")
    ddg = loop["ddg"]

    nodes = {}
    raw_rrt = {}
    for n in ddg["nodes"]:
        node_id = n["id"]
        _require_fields(n, _NODE_FIELDS, f"node {node_id}")
        pipeline = n["pipeline"]
        if pipeline not in machine.capacities and pipeline != "NONE":
            raise ValueError(
                f"node {node_id} uses unknown functional unit {pipeline!r}"
            )
        latency = _checked_int(n["latency"], f"node {node_id} latency")
        occupancy = _checked_int(n["occupancy"], f"node {node_id} occupancy")
        if occupancy > latency:
            raise ValueError(
                f"node {node_id} occupancy {occupancy} exceeds latency {latency}"
            )
        nodes[node_id] = Node(
            node_id,
            n["op_ref"],
            n["op_kind"],
            pipeline,
            latency,
            occupancy,
            _checked_int(n["min_warps"], f"node {node_id} min_warps", minimum=1),
            _checked_int(n["spill_cost"], f"node {node_id} spill_cost"),
        )
        raw_rrt[node_id] = _explicit_rrt(n["rrt"], node_id, latency, machine)

    edges = []
    for index, e in enumerate(ddg["edges"]):
        _require_fields(e, _EDGE_FIELDS, f"edge {index}")
        edges.append(
            Edge(
                e["src"],
                e["dst"],
                _checked_int(e["distance"], f"edge {index} distance"),
                _checked_int(e["latency"], f"edge {index} latency"),
                index,
                _checked_bool(e["blocking"], f"edge {index} blocking"),
            )
        )

    prob = Problem(nodes=nodes, edges=edges, machine=machine,
                   trip_count=loop.get("trip_count"))
    for n in ddg["nodes"]:
        node_id = n["id"]
        prob.regs[node_id] = _checked_int(n["regs"], f"node {node_id} regs")
        prob.smem_footprint[node_id] = _checked_int(
            n["smem_footprint"], f"node {node_id} smem_footprint"
        )
        prob.tmem_footprint[node_id] = _checked_int(
            n["tmem_footprint"], f"node {node_id} tmem_footprint"
        )
        if _checked_bool(n["variable_latency"], f"node {node_id} variable_latency"):
            prob.variable_latency.add(node_id)
        if _checked_bool(n["streaming"], f"node {node_id} streaming"):
            prob.streaming.add(node_id)
    prob.blocking = {(e.src, e.dst) for e in edges if e.blocking}

    raw_rrt_segments = {
        node_id: _rrt_segments(raw_rrt[node_id], nodes[node_id].latency)
        for node_id in nodes
    }

    # Paper sec 5.3: streaming ops run ahead on their own warp; zero their
    # outgoing latency so consumers schedule precisely.
    raw_edge_lat = {}
    for e in edges:
        raw_edge_lat[e.index] = 0 if e.src in prob.streaming else e.latency

    raw_spill = {node_id: nodes[node_id].spill_cost for node_id in nodes}

    # Paper sec 5.2 applies the ZLP directly to the full cost list C,
    # assembled as a positional multiset; every enumerated entry joins C,
    # zeros included.
    pool = (
        [
            duration
            for node_id in nodes
            for _, _, duration in raw_rrt_segments[node_id]
        ]
        + list(raw_edge_lat.values())
        + list(raw_spill.values())
    )
    result = normalize_costs(pool, u=u)
    scaled = iter(result.scaled)
    prob.lat = {}
    prob.rrt = {}
    prob.occ = {}
    for node_id in nodes:
        rows = {functional_unit: [] for functional_unit in raw_rrt[node_id]}
        normalized_cycles = 0
        for functional_units, vector, _ in raw_rrt_segments[node_id]:
            duration = next(scaled)
            _append_rrt_segment(rows, functional_units, vector, duration)
            normalized_cycles += duration
        prob.rrt[node_id] = {
            functional_unit: tuple(row)
            for functional_unit, row in rows.items()
        }
        prob.lat[node_id] = normalized_cycles
        prob.occ[node_id] = sum(
            any(row[cycle] for row in rows.values())
            for cycle in range(normalized_cycles)
        )
    prob.edge_lat = {key: next(scaled) for key in raw_edge_lat}
    prob.normalization_u = u
    prob.normalization_f = result.objective
    prob.normalization_cost_pairs = tuple(zip(pool, result.scaled))
    prob.spill = {key: next(scaled) for key in raw_spill}
    return prob
