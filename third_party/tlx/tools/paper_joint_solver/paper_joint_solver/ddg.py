"""Load a ddg.json dump and derive the paper's solver inputs.

The dump (schema ddg-0.1, produced by -nvgpu-modulo-schedule with
TRITON_MODULO_DUMP_DDG) carries the dependence graph G=(V,E) with per-node
pipeline / latency / occupancy and per-edge distance / latency. On top of
that this module derives inputs the paper's formulation needs that the dump
does not label explicitly:

Nodes may override the legacy pipeline/occupancy pair with
``rrt: {functional_unit: [usage_at_cycle_0, ...]}``; a row may instead be a
sparse ``{cycle: usage}`` object. Every explicit row must span exactly the
node's raw latency. ``spill_cost`` (or ``spillcost``) is an optional per-node
raw-cycle annotation; unannotated register producers use the machine default.

  * variable_latency(v) — ops with high dynamic latency range (TMA loads),
    which VARIABLELATENCY pins to the dedicated warp W_vl;
  * streaming(v)        — variable-latency ops with no incoming dependence;
    per paper sec 5.3 their outgoing latency is zeroed (they run ahead) and
    their pipeline depth becomes an external tunable;
  * blocking(u, v)      — edges whose consumer needs blocking synchronization
    (results of the asynchronous TC / TMA units), driving CONCURRENCY;
  * regs(v) / memory_objects — register values plus alias-aware SMEM/TMEM
    storage objects, derived from result types and allocation/view edges.

The compiler adapter splits every raw RRT into maximal constant-vector runs
and places their durations, edge latencies, and spill latencies in the paper's
normalization cost multiset. The formal solver receives the rebuilt RRT with
``cycles(v) == len(RRT[v][f])`` for every row.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from .machine import MachineModel
from .normalize import DEFAULT_U, normalize_costs
from .resource_model import StorageObject, derive_resources


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
    explicit_spill_cost: bool = False


@dataclass
class Edge:
    src: int
    dst: int
    distance: int  # iteration delay (paper's delta)
    latency: int  # clock-cycle delay (paper's d)
    index: int = -1  # stable identity for parallel edges
    src_result_idx: int = 0
    signal_only: bool = False


@dataclass
class Problem:
    nodes: dict[int, Node]
    edges: list[Edge]
    machine: MachineModel
    trip_count: int | None
    raw_min_ii: int
    # Derived inputs (paper secs 4.3 / 5.3).
    variable_latency: set[int] = field(default_factory=set)
    streaming: set[int] = field(default_factory=set)
    blocking: set[tuple[int, int]] = field(default_factory=set)
    # Baseline-graph annotation retained for provenance only. Every DDG node
    # participates in the paper model regardless of emitter ownership.
    emitter_infra: set[int] = field(default_factory=set)
    regs: dict[int, int] = field(default_factory=dict)
    memory_objects: dict[str, StorageObject] = field(default_factory=dict)
    # Normalized costs (paper sec 5.2).
    lat: dict[int, int] = field(default_factory=dict)  # cycles(v), normalized
    # `occ` is compatibility metadata: normalized cycles with any resource
    # demand. All resource constraints consume `rrt` directly.
    occ: dict[int, int] = field(default_factory=dict)
    rrt: dict[int, dict[str, tuple[int, ...]]] = field(default_factory=dict)
    edge_lat: dict[int, int] = field(default_factory=dict)
    normalization_f: int = 0
    spill: dict[int, int] = field(default_factory=dict)

    def edge_latency(self, edge: Edge) -> int:
        return self.edge_lat[edge.index]

    def reservations(self, node_id: int):
        for functional_unit, row in self.rrt[node_id].items():
            for cycle, amount in enumerate(row):
                if amount:
                    yield functional_unit, cycle, amount

    def res_mii(self) -> int:
        best = 1
        for p in self.machine.capacities:
            use = sum(
                amount
                for v in self.nodes
                for functional_unit, _, amount in self.reservations(v)
                if functional_unit == p
            )
            cap = self.machine.cap(p)
            if cap:
                best = max(best, -(-use // cap))
        return best

    def _has_positive_cycle(self, ii: int) -> bool:
        ids = list(self.nodes)
        idx = {v: i for i, v in enumerate(ids)}
        n = len(ids)
        NEG = float("-inf")
        dist = [[NEG] * n for _ in range(n)]
        for e in self.edges:
            w = self.edge_latency(e) - e.distance * ii
            i, j = idx[e.src], idx[e.dst]
            dist[i][j] = max(dist[i][j], w)
        for k in range(n):
            dk = dist[k]
            for i in range(n):
                dik = dist[i][k]
                if dik == NEG:
                    continue
                row = dist[i]
                for j in range(n):
                    if dk[j] != NEG and dik + dk[j] > row[j]:
                        row[j] = dik + dk[j]
        return any(dist[i][i] > 0 for i in range(n))

    def rec_mii(self) -> int:
        # Smallest II with no positive-weight cycle; monotone in II, so
        # binary-search through the sum of all distinct edge delays. This is
        # also a valid upper bound when edge latency exceeds node latency.
        lo, hi = 1, max(1, sum(self.edge_lat.values()))
        while lo < hi:
            mid = (lo + hi) // 2
            if self._has_positive_cycle(mid):
                lo = mid + 1
            else:
                hi = mid
        return lo

    def min_ii(self) -> int:
        return max(self.res_mii(), self.rec_mii())


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


def load_problem(path: str | Path, machine: MachineModel | None = None,
                 loop_index: int = 0, u: int = DEFAULT_U,
                 streaming_zero_latency: bool = True,
                 baseline_graph: str | Path | None = None) -> Problem:
    data = json.loads(Path(path).read_text())
    machine = machine or MachineModel()
    loop = data["loops"][loop_index]
    ddg = loop["ddg"]
    def pipeline_of(n: dict) -> str:
        # Blackwell TMEM ports are their own functional unit (machine.py).
        if "tmem_load" in n["op_kind"] or "tmem_store" in n["op_kind"]:
            return "TMEM"
        return n.get("pipeline", "NONE")

    nodes = {}
    raw_rrt = {}
    for n in ddg["nodes"]:
        node_id = n["id"]
        pipeline = pipeline_of(n)
        latency = n.get("latency", 0)
        if (
            isinstance(latency, bool)
            or not isinstance(latency, int)
            or latency < 0
        ):
            raise ValueError(
                f"node {node_id} latency must be a nonnegative integer"
            )
        if "rrt" in n:
            rows = _explicit_rrt(n["rrt"], node_id, latency, machine)
            occupancy = latency
        else:
            occupancy = n.get("occupancy", latency)
            if (
                isinstance(occupancy, bool)
                or not isinstance(occupancy, int)
                or occupancy < 0
            ):
                raise ValueError(
                    f"node {node_id} occupancy must be a nonnegative integer"
                )
            if occupancy > latency:
                raise ValueError(
                    f"node {node_id} occupancy {occupancy} exceeds "
                    f"latency {latency}"
                )
            rows = (
                {pipeline: [1] * occupancy + [0] * (latency - occupancy)}
                if pipeline in machine.capacities and latency > 0
                else {}
            )
        explicit_spill_cost = "spill_cost" in n or "spillcost" in n
        spill_cost = n.get("spill_cost", n.get("spillcost", machine.spill_cost))
        if (
            isinstance(spill_cost, bool)
            or not isinstance(spill_cost, int)
            or spill_cost < 0
        ):
            raise ValueError(
                f"node {node_id} spill cost must be a nonnegative integer"
            )
        nodes[node_id] = Node(
            node_id,
            n.get("op_ref", ""),
            n["op_kind"],
            pipeline,
            latency,
            occupancy,
            n.get("min_warps", 1),
            spill_cost,
            explicit_spill_cost,
        )
        raw_rrt[node_id] = rows
    ops_table = data.get("ops", {})
    edges = []
    for index, e in enumerate(ddg["edges"]):
        src_result_idx = e.get("src_result_idx", 0)
        result_types = ops_table.get(
            nodes[e["src"]].op_ref, {}
        ).get("result_types", [])
        signal_only = (
            0 <= src_result_idx < len(result_types)
            and result_types[src_result_idx] == "!ttg.async.token"
        )
        edges.append(
            Edge(
                e["src"],
                e["dst"],
                e.get("distance", 0),
                e.get("latency", 0),
                index,
                src_result_idx,
                signal_only,
            )
        )
    prob = Problem(nodes=nodes, edges=edges, machine=machine,
                   trip_count=loop.get("trip_count"),
                   raw_min_ii=loop.get("min_ii", 0))
    if baseline_graph is not None:
        graph_data = json.loads(Path(baseline_graph).read_text())
        graph_nodes = graph_data["loops"][loop_index]["schedule_loop"]["graph"][
            "nodes"
        ]
        prob.emitter_infra = {
            node["id"]
            for node in graph_nodes
            if node.get("warp_group", -1) < 0 and node["id"] in nodes
        }

    data_edges = [edge for edge in edges if not edge.signal_only]
    prob.regs, prob.memory_objects = derive_resources(
        nodes,
        data_edges,
        ops_table,
        tmem_column_bytes=machine.tmem_column_bytes,
    )
    for node_id, node in nodes.items():
        if not node.explicit_spill_cost:
            node.spill_cost = machine.spill_cost if prob.regs[node_id] > 0 else 0

    # Paper sec 5.3 defines streaming operations as variable-latency
    # operations with no incoming data dependence in G.
    has_incoming = {edge.dst for edge in edges}
    for v in nodes.values():
        if v.pipeline == "TMA" and "load" in v.op_kind:
            prob.variable_latency.add(v.id)
            if v.id not in has_incoming:
                prob.streaming.add(v.id)
    async_producers = {
        v.id
        for v in nodes.values()
        if v.pipeline in ("TC", "TMA")
    }
    prob.blocking = {
        (e.src, e.dst)
        for e in edges
        if e.src in async_producers or e.signal_only
    }

    raw_rrt_segments = {
        node_id: _rrt_segments(raw_rrt[node_id], nodes[node_id].latency)
        for node_id in nodes
    }

    # Paper sec 5.3: streaming ops run ahead on their own warp; zero their
    # outgoing latency so consumers schedule precisely.
    raw_edge_lat = {}
    for e in edges:
        if streaming_zero_latency and e.src in prob.streaming:
            lat = 0
        else:
            lat = e.latency
        raw_edge_lat[e.index] = lat

    spill_producers = {edge.src for edge in data_edges}
    cross_spill_producers = {
        edge.src for edge in data_edges if edge.src != edge.dst
    }
    raw_spill = {
        node_id: nodes[node_id].spill_cost
        for node_id in nodes
        if node_id in spill_producers
    }

    # Paper sec 5.2 applies the ZLP directly to the full cost list C. Keep
    # duplicates because every occurrence contributes to sum(C').
    pool = (
        [
            duration
            for node_id in nodes
            for _, _, duration in raw_rrt_segments[node_id]
        ]
        + [c for c in raw_edge_lat.values() if c > 0]
        + [
            cost
            for node_id, cost in raw_spill.items()
            if node_id in cross_spill_producers and cost > 0
        ]
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
    prob.edge_lat = {
        key: next(scaled) if cost > 0 else 0
        for key, cost in raw_edge_lat.items()
    }
    prob.normalization_f = result.objective
    prob.spill = {
        key: (
            next(scaled)
            if key in cross_spill_producers and cost > 0
            else 0
        )
        for key, cost in raw_spill.items()
    }
    return prob
