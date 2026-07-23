"""Load a ddg.json dump and derive the paper's solver inputs.

The dump (schema ddg-0.1, produced by -nvgpu-modulo-schedule with
TRITON_MODULO_DUMP_DDG) carries the dependence graph G=(V,E) with per-node
pipeline / latency / occupancy and per-edge distance / latency.  On top of
that this module derives the three inputs the paper's formulation needs that
the dump does not label explicitly:

  * variable_latency(v) — ops with high dynamic latency range (TMA loads),
    which VARIABLELATENCY pins to the dedicated warp W_vl;
  * streaming(v)        — variable-latency ops with no in-loop producer;
    per paper sec 5.3 their outgoing latency is zeroed (they run ahead) and
    their pipeline depth becomes an external tunable;
  * blocking(u, v)      — edges whose consumer needs blocking synchronization
    (results of the asynchronous TC / TMA units), driving CONCURRENCY;
  * regs(v)             — register footprint of v's live result, from the
    result tensor type in the ops table.

Costs are then normalized (paper sec 5.2) over the union of edge latencies
and node occupancies so the time-indexed problems stay tractable.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .machine import MachineModel
from .normalize import (DEFAULT_CLUSTER_TOL, DEFAULT_U, cluster_costs,
                        normalize_costs)

_TENSOR_RE = re.compile(
    r"(?:tensor|!ttg\.memdesc)<([0-9x]+)x(bf16|f16|f32|f64|i1|i8|i16|i32|i64)")
_SCALAR_KIND_PREFIXES = ("arith.", "math.", "index.", "tt.addptr")
_DTYPE_BYTES = {"bf16": 2, "f16": 2, "f32": 4, "f64": 8, "i1": 1, "i8": 1,
                "i16": 2, "i32": 4, "i64": 8}


@dataclass
class Node:
    id: int
    op_ref: str
    op_kind: str
    pipeline: str
    latency: int
    occupancy: int
    min_warps: int


@dataclass
class Edge:
    src: int
    dst: int
    distance: int  # iteration delay (paper's delta)
    latency: int  # clock-cycle delay (paper's d)


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
    regs: dict[int, int] = field(default_factory=dict)
    # Normalized costs (paper sec 5.2).
    lat: dict[int, int] = field(default_factory=dict)  # cycles(v), normalized
    occ: dict[int, int] = field(default_factory=dict)  # RRT rows, normalized
    edge_lat: dict[tuple[int, int, int], int] = field(default_factory=dict)
    normalization_f: int = 0
    spill: int = 1  # cross-warp spill delay, normalized with the cost pool

    def res_mii(self) -> int:
        best = 1
        for p in self.machine.capacities:
            use = sum(self.occ[v.id] for v in self.nodes.values()
                      if v.pipeline == p)
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
            w = self.edge_lat[(e.src, e.dst, e.distance)] - e.distance * ii
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
        # binary-search over [1, sum(latencies)].
        lo, hi = 1, max(1, sum(self.lat.values()))
        while lo < hi:
            mid = (lo + hi) // 2
            if self._has_positive_cycle(mid):
                lo = mid + 1
            else:
                hi = mid
        return lo

    def min_ii(self) -> int:
        return max(self.res_mii(), self.rec_mii())


def _result_bytes(op_entry: dict) -> int:
    total = 0
    for t in op_entry.get("result_types", []):
        m = _TENSOR_RE.search(t)
        if m:
            elems = 1
            for d in m.group(1).split("x"):
                elems *= int(d)
            total += elems * _DTYPE_BYTES[m.group(2)]
    return total


def load_problem(path: str | Path, machine: MachineModel | None = None,
                 loop_index: int = 0, u: int = DEFAULT_U,
                 streaming_zero_latency: bool = True) -> Problem:
    data = json.loads(Path(path).read_text())
    machine = machine or MachineModel()
    loop = data["loops"][loop_index]
    ddg = loop["ddg"]
    def pipeline_of(n: dict) -> str:
        # Blackwell TMEM ports are their own functional unit (machine.py).
        if "tmem_load" in n["op_kind"] or "tmem_store" in n["op_kind"]:
            return "TMEM"
        return n.get("pipeline", "NONE")

    nodes = {
        n["id"]: Node(n["id"], n.get("op_ref", ""), n["op_kind"],
                      pipeline_of(n), n.get("latency", 0),
                      n.get("occupancy", n.get("latency", 0)),
                      n.get("min_warps", 1))
        for n in ddg["nodes"]
    }
    edges = [Edge(e["src"], e["dst"], e.get("distance", 0),
                  e.get("latency", 0)) for e in ddg["edges"]]
    prob = Problem(nodes=nodes, edges=edges, machine=machine,
                   trip_count=loop.get("trip_count"),
                   raw_min_ii=loop.get("min_ii", 0))

    ops_table = data.get("ops", {})
    for v in nodes.values():
        entry = ops_table.get(v.op_ref, {})
        prob.regs[v.id] = _result_bytes(entry) // 4  # 32-bit registers

    # Streaming (paper sec 5.3): variable-latency ops with no incoming data
    # dependence.  Scalar address arithmetic on the induction variable is not
    # a data dependence — those values are known ahead, so the load can still
    # run ahead of the pipeline.  A load stops being streaming only when some
    # transitive intra-iteration producer computes a tensor (regs > 0).
    preds: dict[int, set[int]] = {}
    for e in edges:
        if e.distance == 0:
            preds.setdefault(e.dst, set()).add(e.src)

    def has_tensor_ancestor(v: int) -> bool:
        seen, stack = set(), list(preds.get(v, ()))
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            if prob.regs.get(x, 0) > 0:
                return True
            stack.extend(preds.get(x, ()))
        return False

    for v in nodes.values():
        if v.pipeline == "TMA" and "load" in v.op_kind:
            prob.variable_latency.add(v.id)
            if not has_tensor_ancestor(v.id):
                prob.streaming.add(v.id)
    async_producers = {v.id for v in nodes.values()
                       if v.pipeline in ("TC", "TMA")}
    prob.blocking = {(e.src, e.dst) for e in edges
                     if e.src in async_producers}

    # The paper schedules tile-level operations (sec 3.1) — its cost pool has
    # a bounded dynamic range.  The dump instead prices warp-parallel
    # elementwise CUDA ops at 1 cycle (hidden-by-parallelism convention) and
    # includes scalar address arithmetic; both would wreck the normalization
    # ratio spread.  Restore the paper's granularity: scalar ops (no tensor
    # result) cost zero (kept only for dependence structure), and elementwise
    # tile ops cost at least ceil(elements / 128 lanes) cycles.
    def op_tile_units(v: Node) -> int:
        """32-bit units the op touches: max of result and operand tiles."""
        entry = ops_table.get(v.op_ref, {})
        units = prob.regs[v.id]
        for operand in entry.get("operands", []):
            ref = operand.get("op") if isinstance(operand, dict) else operand
            src = ops_table.get(ref, {})
            units = max(units, _result_bytes(src) // 4)
        return units

    def tile_cost(v: Node, dump_cost: int) -> int:
        # Scalar address/index arithmetic is free (kept for dependences only).
        if (prob.regs[v.id] == 0
                and v.op_kind.startswith(_SCALAR_KIND_PREFIXES)):
            return 0
        # Elementwise/reduce tile ops on the general-purpose cores: at least
        # ceil(tile elements / 128 lanes) cycles (the dump prices them at 1,
        # a hidden-by-warp-parallelism convention the RRT view cannot use).
        if v.pipeline == "CUDA":
            return max(dump_cost, -(-op_tile_units(v) // 128))
        return dump_cost

    raw_node_lat, raw_node_occ = {}, {}
    for v in nodes.values():
        lat = 0 if (streaming_zero_latency and v.id in prob.streaming) \
            else tile_cost(v, v.latency)
        raw_node_lat[v.id] = lat
        raw_node_occ[v.id] = tile_cost(v, v.occupancy)

    # Paper sec 5.3: streaming ops run ahead on their own warp; zero their
    # outgoing latency so consumers schedule precisely.  Other edges carry
    # the producer's (re-priced) latency.
    raw_edge_lat = {}
    for e in edges:
        if streaming_zero_latency and e.src in prob.streaming:
            lat = 0
        elif e.latency == prob.nodes[e.src].latency:
            lat = raw_node_lat[e.src]  # producer-latency edge, re-priced
        else:
            lat = e.latency
        raw_edge_lat[(e.src, e.dst, e.distance)] = lat

    # Cost floor: ops cheaper than 1/32 of the pool maximum are scheduling
    # noise at tile granularity, but their extreme ratios to the MMA-scale
    # costs make the normalization ILP's F useless (a 1:900 pair cannot be
    # represented within any practical U).  Zero them; they remain dependence
    # nodes.
    all_costs = [c for c in list(raw_node_lat.values())
                 + list(raw_node_occ.values()) if c > 0]
    floor = max(all_costs, default=0) // 32
    raw_node_lat = {i: (0 if c <= floor else c)
                    for i, c in raw_node_lat.items()}
    raw_node_occ = {i: (0 if c <= floor else c)
                    for i, c in raw_node_occ.items()}
    raw_edge_lat = {k: (0 if c <= floor else c)
                    for k, c in raw_edge_lat.items()}

    raw_values = [c for c in list(raw_node_lat.values())
                  + list(raw_node_occ.values()) if c > 0]
    if raw_values:
        # Costs are measurement estimates: merge values within tolerance
        # before the normalization ILP so near-equal integers don't pin F.
        # The cross-warp spill delay is a cost like any other and must live
        # in the same normalized time unit.
        spill_raw = max(machine.spill_cost, 1)
        clusters = cluster_costs(
            raw_values + [c for c in raw_edge_lat.values() if c > 0]
            + [spill_raw], tol=DEFAULT_CLUSTER_TOL)
        pool = sorted(set(clusters[c] for c in raw_values + [spill_raw]))
        result = normalize_costs(pool, u=u)
        rep_scaled = dict(zip(pool, result.scaled))
        mapping = {c: rep_scaled[r] for c, r in clusters.items()}
        mapping[0] = 0
        prob.normalization_f = result.objective
        prob.spill = max(1, mapping[spill_raw])
    else:
        mapping = {0: 0}
    # A tile op whose cost rounds to zero still occupies its unit one slot;
    # zero-cost scalar ops stay free.
    prob.lat = {i: mapping[c] for i, c in raw_node_lat.items()}
    prob.occ = {i: (max(mapping[c], 1) if c > 0 else 0)
                for i, c in raw_node_occ.items()}
    prob.edge_lat = {k: mapping[c] for k, c in raw_edge_lat.items()}
    return prob
