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
  * streaming(v)        — variable-latency ops with no incoming non-infra data
    dependence; per paper sec 5.3 their outgoing latency is zeroed (they run
    ahead) and their pipeline depth becomes an external tunable;
  * blocking(u, v)      — edges whose consumer needs blocking synchronization
    (results of the asynchronous TC / TMA units), driving CONCURRENCY;
  * regs(v) / memory_objects — register values plus alias-aware SMEM/TMEM
    storage objects, derived from result types and allocation/view edges.

The compiler adapter splits every raw RRT into maximal constant-vector runs
and places their durations, edge latencies, and spill latencies in the paper's
normalization cost multiset. The formal solver receives the rebuilt RRT with
``cycles(v) == len(RRT[v][f])`` for every row.

If the ZLP maps any positive multiset entry to zero, the adapter doubles U up
to a bounded 256x ceiling. The committed sub-tiled backward fixture already
consumes the 128x retry level, so one more doubling covers modest measurement
or fixture drift without tying the limit to that fixture. A collapse at the
ceiling remains a hard error; the paper's ZLP and duplicate-entry semantics
remain unchanged.
"""

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

from .machine import MachineModel
from .normalize import (
    DEFAULT_U,
    MAX_U_MULTIPLIER,
    NormalizationResolutionError,
    NormalizationResult,
    normalize_costs,
)
from .resource_model import StorageObject, derive_resources


_SCALAR_RESULT_TYPES = frozenset(
    ("bf16", "f16", "f32", "f64", "i1", "i8", "i16", "i32", "i64", "index")
)


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
    # Baseline-graph ownership annotation. Every DDG node remains scheduled,
    # but scalar infrastructure edges retain dependence semantics only.
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
    normalization_u: int = DEFAULT_U
    normalization_u_effective: int = DEFAULT_U
    normalization_f: int = 0
    normalization_zero_collapses: int = 0
    normalization_retries: int = 0
    normalization_cost_pairs: tuple[tuple[int, int], ...] = ()
    spill: dict[int, int] = field(default_factory=dict)

    def normalization_stats(self) -> dict[str, int]:
        return {
            "normalization_u": self.normalization_u,
            "normalization_u_effective": self.normalization_u_effective,
            "normalization_f": self.normalization_f,
            "normalization_zero_collapses": self.normalization_zero_collapses,
            "normalization_retries": self.normalization_retries,
        }

    def edge_latency(self, edge: Edge) -> int:
        return self.edge_lat[edge.index]

    def edge_has_ws_semantics(self, edge: Edge) -> bool:
        return edge.src not in self.emitter_infra

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


def _normalize_cost_pool(costs: list[int], u: int) -> NormalizationResult:
    """Double U until positive costs survive, within the bounded headroom.

    The current sub-tiled backward multiset consumes the 128x retry level, so
    MAX_U_MULTIPLIER deliberately leaves one additional doubling for modest
    measurement or fixture drift instead of making that fixture the cap.
    """
    if isinstance(u, bool) or not isinstance(u, int) or u < 1:
        raise ValueError("normalization budget U must be a positive integer")

    max_u = u * MAX_U_MULTIPLIER
    u_eff = u
    retries = 0
    while True:
        result = normalize_costs(costs, u=u_eff)
        if len(result.scaled) != len(costs):
            raise RuntimeError(
                "cost normalization returned "
                f"{len(result.scaled)} values for {len(costs)} costs"
            )
        zero_collapses = sum(
            raw > 0 and normalized == 0
            for raw, normalized in zip(costs, result.scaled)
        )
        result = replace(
            result,
            u_eff=u_eff,
            zero_collapses=zero_collapses,
            retries=retries,
        )
        if zero_collapses == 0:
            return result
        if u_eff == max_u:
            raise NormalizationResolutionError(
                initial_u=u,
                max_u=max_u,
                zero_collapses=zero_collapses,
                positive_costs=sum(cost > 0 for cost in costs),
                collapsed_cost_values=tuple(
                    sorted(
                        {
                            raw
                            for raw, normalized in zip(costs, result.scaled)
                            if raw > 0 and normalized == 0
                        }
                    )
                ),
            )
        u_eff = min(u_eff * 2, max_u)
        retries += 1


def _is_scalar_result_type(result_type: str) -> bool:
    return result_type in _SCALAR_RESULT_TYPES or result_type.startswith("!tt.ptr<")


def _validate_infra_data_edges(
    prob: Problem, data_edges: list[Edge], ops_table: dict
) -> None:
    for edge in data_edges:
        if prob.edge_has_ws_semantics(edge):
            continue
        result_types = ops_table.get(prob.nodes[edge.src].op_ref, {}).get(
            "result_types", []
        )
        result_type = (
            result_types[edge.src_result_idx]
            if 0 <= edge.src_result_idx < len(result_types)
            else None
        )
        if result_type is None or not _is_scalar_result_type(result_type):
            raise ValueError(
                f"infra data edge {edge.src}->{edge.dst} result "
                f"{edge.src_result_idx} must be scalar, got {result_type!r}"
            )


def _baseline_infra_nodes(
    graph_nodes: object,
    nodes: dict[int, Node],
    *,
    require_full_coverage: bool,
) -> set[int]:
    if not isinstance(graph_nodes, list):
        raise ValueError("baseline graph nodes must be a list")
    by_id: dict[int, dict] = {}
    for position, raw_node in enumerate(graph_nodes):
        if not isinstance(raw_node, dict):
            raise ValueError(f"baseline graph node {position} must be an object")
        node_id = raw_node.get("id")
        if isinstance(node_id, bool) or not isinstance(node_id, int):
            raise ValueError(f"baseline graph node {position} has an invalid id")
        if node_id in by_id:
            raise ValueError(f"baseline graph has duplicate node id {node_id}")
        if node_id not in nodes:
            raise ValueError(f"baseline graph has unknown DDG node id {node_id}")
        warp_group = raw_node.get("warp_group")
        if "warp_group" in raw_node and (
            isinstance(warp_group, bool) or not isinstance(warp_group, int)
        ):
            raise ValueError(
                f"baseline graph node {node_id} warp_group must be an integer"
            )
        by_id[node_id] = raw_node
    if require_full_coverage and set(by_id) != set(nodes):
        missing = sorted(set(nodes) - set(by_id))
        raise ValueError(f"baseline graph is missing DDG nodes {missing}")
    for node_id, raw_node in by_id.items():
        ddg_node = nodes[node_id]
        for field, expected in (
            ("op_ref", ddg_node.op_ref),
            ("op_kind", ddg_node.op_kind),
        ):
            if require_full_coverage and field not in raw_node:
                raise ValueError(f"baseline graph node {node_id} is missing {field}")
            if field in raw_node and raw_node[field] != expected:
                raise ValueError(
                    f"baseline graph node {node_id} {field} does not match the DDG"
                )
    return {
        node_id
        for node_id, node in by_id.items()
        if node.get("warp_group", 0) < 0
    }


def load_problem(path: str | Path, machine: MachineModel | None = None,
                 loop_index: int = 0, u: int = DEFAULT_U,
                 streaming_zero_latency: bool = True,
                 baseline_graph: str | Path | None = None,
                 strict_baseline: bool = True,
                 allow_zero_normalized_costs: bool = False) -> Problem:
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
        prob.emitter_infra = _baseline_infra_nodes(
            graph_nodes,
            nodes,
            require_full_coverage=strict_baseline,
        )

    data_edges = [edge for edge in edges if not edge.signal_only]
    _validate_infra_data_edges(prob, data_edges, ops_table)
    ws_data_edges = [
        edge for edge in data_edges if prob.edge_has_ws_semantics(edge)
    ]
    prob.regs, prob.memory_objects = derive_resources(
        nodes,
        data_edges,
        ops_table,
        tmem_column_bytes=machine.tmem_column_bytes,
    )
    for node_id, node in nodes.items():
        if not node.explicit_spill_cost:
            node.spill_cost = machine.spill_cost if prob.regs[node_id] > 0 else 0

    # The compiler DDG includes scalar address arithmetic that paper sec 5.3's
    # G omits. Signal-only and emitter-infrastructure edges therefore do not
    # disqualify a variable-latency operation from streaming.
    prob.variable_latency = {
        v.id
        for v in nodes.values()
        if v.pipeline == "TMA" and "load" in v.op_kind
    }
    has_incoming = {edge.dst for edge in ws_data_edges}
    prob.streaming = prob.variable_latency - has_incoming
    async_producers = {
        v.id
        for v in nodes.values()
        if v.pipeline in ("TC", "TMA")
    }
    prob.blocking = {
        (e.src, e.dst)
        for e in edges
        if prob.edge_has_ws_semantics(e)
        and (e.src in async_producers or e.signal_only)
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

    spill_producers = {edge.src for edge in ws_data_edges}
    cross_spill_producers = {
        edge.src for edge in ws_data_edges if edge.src != edge.dst
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
    # This escape hatch exists only for inspecting superseded artifacts and
    # tests whose subject is downstream of normalization. New solves must use
    # the fail-closed default.
    result = (
        normalize_costs(pool, u=u)
        if allow_zero_normalized_costs
        else _normalize_cost_pool(pool, u=u)
    )
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
    prob.normalization_u = u
    prob.normalization_u_effective = result.u_eff
    prob.normalization_f = result.objective
    prob.normalization_zero_collapses = result.zero_collapses
    prob.normalization_retries = result.retries
    prob.normalization_cost_pairs = tuple(zip(pool, result.scaled))
    prob.spill = {
        key: (
            next(scaled)
            if key in cross_spill_producers and cost > 0
            else 0
        )
        for key, cost in raw_spill.items()
    }
    return prob
