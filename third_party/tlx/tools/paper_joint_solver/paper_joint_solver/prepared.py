"""The prepared regime: a disclosed parallel input-preparation path.

The literal path is a strict transcription of the paper: its cost pool is the
positional multiset of every RRT maximal-segment duration, every edge latency
and every node spill cost, and one ZLP pass at U=300 decides the scale
(``ddg.py``).  On compiler-extracted inputs that pass is degenerate --
``fwd_subtiled`` collapses 104/128 positive entries and the search settles at
II=2 -- and four controlled experiments localized the gap in the *inputs*, not
in the solver: normalization post-processing cannot help because the collapse
is the strict ZLP optimum, and the U cliff is binary.

This module is the other half of that finding, implemented as a regime that
runs beside the literal one and never inside it.  Where the literal path
answers "what does the paper say", this one answers "what must additionally be
assumed to reach the published numbers" -- and every such assumption is listed
here and in ``DEVIATIONS.md`` under "The prepared regime".  Nothing here is
paper content, and nothing here is reachable from ``python -m
paper_joint_solver``.

Two layers, eight assumptions:

  A: cost-pool assembly.  A1 one pool entry per node instead of one per RRT
     maximal segment, plus only the *non-zero* edge latencies and spills;
     A2 CUDA-pipeline elementwise operators are re-priced at
     ``ceil(tile elements / 128)``, and scalar/address arithmetic is charged
     zero; A3 asynchronous MMA results are tokens, not scalars, and keep their
     measured cost; A4 TMEM ports are their own functional unit.

  B: normalization stack.  B1 10% relative-tolerance clustering; B2 a 1/32
     floor; B3 the paper's own ZLP, imported unmodified; B4 a lexicographic
     second stage that maximizes sum(C') over the F-optimal face.

B4 is what makes the small pool survive U=300: on ``fwd_subtiled`` the
F-optimal face contains both a degenerate point (sum(C')=16, 87.5% of positive
entries at zero) and a rich one (sum(C')=272, 16.1%), and the paper's
minimize-F program is indifferent between them.

Every source file listed in ``schedule_plan._SOLVER_SOURCES`` is untouched by
this module; it imports them and adds nothing to them.  That includes the
curated-input validators reused below: they are private to ``ddg`` only in the
naming sense, and copying them would fork the fail-closed reading of the
curated schema, which is the one thing this regime must not do.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from pyscipopt import Model, quicksum

from .ddg import (
    _checked_bool,
    _checked_int,
    _explicit_rrt,
    _LOOP_FIELDS,
    _NODE_FIELDS,
    _EDGE_FIELDS,
    _require_fields,
    _TOP_FIELDS,
    CURATED_SCHEMA,
    Edge,
    Node,
    Problem,
)
from .machine import MachineModel
from .normalize import normalize_costs

PREPARED_SOLUTION_SCHEMA = "prepared-joint-solution-v1"
PREPARED_PROBE_SCHEMA = "prepared-joint-probe-v1"

# A2: 4 warps x 32 lanes is the issue width one CUDA-pipeline elementwise
# instruction covers per issue slot.
ISSUE_WIDTH = 128
# A2: op_kind dialects whose tensor operations are elementwise. `tt.reduce` is
# deliberately absent -- a reduction is not elementwise and keeps its measured
# occupancy.
ELEMENTWISE_DIALECTS = ("arith", "math")
# A4: op_kind markers that must sit on the TMEM functional unit.
TMEM_OP_MARKERS = ("tmem_load", "tmem_store")
# B1: v joins the open cluster while v <= anchor * 11/10.  Held as a rational
# so the scan never depends on binary floating point.
CLUSTER_TOLERANCE = (11, 10)
# B2: after clustering, an entry below max/32 is floored to zero.
FLOOR_DIVISOR = 32
# B6: frozen at the 2026-08-11 ruling.  The climb takes the FIRST rung that
# clears P1, never the loosest, and the ladder is never extended: a pool that
# is still degenerate at 8xF* is reported as such.
F_RELAXATION_LADDER = (1, 2, 3, 4, 6, 8)
# P1, also frozen at that ruling: these are the thresholds B6 climbs toward,
# and they may not be revised in response to a result.
P1_MAX_COLLAPSE_RATIO = 0.20
P1_MIN_SURVIVOR_SPREAD = 8
# B5: the per-case normalization budget.  `fwd` is the one case whose formula
# size explodes at 300; the others are the paper's own U.
DEFAULT_U = 300
CASE_U = {
    "fwd_subtiled": 300,
    "fwd": 150,
    "bwd": 300,
    "bwd_lr4096": 300,
}

_TENSOR_SHAPE = re.compile(r"^tensor<([0-9]+(?:x[0-9]+)*)x")


def prepared_sources_sha256() -> str:
    """This module's own digest.

    The preparation layer decides what the solver is even asked, so a prepared
    artifact is meaningless without it.  Stamped beside -- never instead of --
    the core's ``solver_sources_sha256``, the same split ``probe.py`` uses.
    """
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class PreparationError(ValueError):
    """A curated input the preparation layer refuses to guess about."""


# --------------------------------------------------------------------------
# A layer: cost-pool assembly
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PoolEntry:
    """One member of the prepared cost pool C.

    ``kind``/``key`` identify what the entry prices ("node" and a node id,
    "edge" and an edge index, "spill" and a node id); ``basis`` records which
    A-layer rule produced ``cost``, so a reader can see per entry whether the
    number is measured or assumed.
    """

    kind: str
    key: int
    cost: int
    basis: str


def tensor_elements(result_type: str) -> int | None:
    """Element count of an MLIR tensor type, or None if it is not one.

    ``!ttg.async.token`` and ``!ttg.memdesc<...>`` results deliberately return
    None: A3 requires that an asynchronous MMA result is never mistaken for a
    scalar and re-priced to zero.
    """
    match = _TENSOR_SHAPE.match(result_type)
    if match is None:
        return None
    elements = 1
    for extent in match.group(1).split("x"):
        elements *= int(extent)
    return elements


def _is_elementwise(node: dict) -> bool:
    dialect = node["op_kind"].split(".")[0]
    return node["pipeline"] == "CUDA" and dialect in ELEMENTWISE_DIALECTS


def node_cost(node: dict, result_types: list[str]) -> tuple[int, str]:
    """A2/A3: the single cost that prices one node.

    Non-elementwise operations keep their measured RRT occupancy -- the span
    on which they actually hold a functional unit.  Their *latency* is not
    dropped: curation writes every non-streaming edge's latency equal to its
    producer's latency, so the long result delays reach the solver through the
    edge entries below, and charging them twice would inflate the pool.

    Elementwise CUDA operations are re-priced instead.  The dump records
    latency 1 (often 0) for them, which is the warp-parallel hiding convention
    of the extractor rather than an issue cost, and is unusable from the RRT's
    point of view: it makes a 64-wide multiply and a 16384-wide multiply cost
    the same.
    """
    if not _is_elementwise(node):
        return node["occupancy"], "occupancy"
    elements = tensor_elements(result_types[0] if result_types else "")
    if elements is None:
        # Scalar and address arithmetic keeps only its dependence structure.
        return 0, "scalar-zero"
    return -(-elements // ISSUE_WIDTH), "elementwise"


def _require_tmem_unit(node: dict) -> None:
    """A4: TMEM ports are distinct hardware from the general-purpose ALUs.

    The stock ``MachineModel`` already carries TMEM as its own functional unit
    of capacity 1, and curation already routes these operations onto it, so
    the assumption needs no machine of its own here -- only this check, so a
    curated input that folds TMEM back into CUDA fails loudly instead of
    silently saturating the ALU column.
    """
    if any(marker in node["op_kind"] for marker in TMEM_OP_MARKERS):
        if node["pipeline"] != "TMEM":
            raise PreparationError(
                f"node {node['id']} ({node['op_kind']}) sits on pipeline "
                f"{node['pipeline']!r}; the prepared regime requires TMEM "
                "operations on the TMEM functional unit (A4)"
            )


def assemble_pool(ddg: dict, ops_table: dict) -> list[PoolEntry]:
    """A1: the prepared cost pool.

    One entry per node, then every non-zero edge latency, then every non-zero
    spill cost.  The literal pool instead enumerates each RRT maximal segment,
    every edge and every node's spill including the zeros; those zero entries
    consume ZLP variables and pairwise constraints without carrying any ratio,
    and each node's trailing idle segment duplicates information already held
    by its out-edges.
    """
    nodes = ddg["nodes"]
    streaming = {node["id"] for node in nodes if node["streaming"]}
    entries: list[PoolEntry] = []
    for node in nodes:
        _require_tmem_unit(node)
        op_ref = node["op_ref"]
        if op_ref not in ops_table:
            raise PreparationError(
                f"node {node['id']} references {op_ref!r}, absent from the raw "
                "dump's op table; the shapes A2 needs are unavailable"
            )
        cost, basis = node_cost(node, ops_table[op_ref]["result_types"])
        entries.append(PoolEntry("node", node["id"], cost, basis))
    for index, edge in enumerate(ddg["edges"]):
        # Paper sec 5.3, unchanged: a streaming producer's out-edges are zero.
        latency = 0 if edge["src"] in streaming else edge["latency"]
        if latency:
            entries.append(PoolEntry("edge", index, latency, "edge-latency"))
    for node in nodes:
        if node["spill_cost"]:
            entries.append(
                PoolEntry("spill", node["id"], node["spill_cost"], "spill")
            )
    return entries


# --------------------------------------------------------------------------
# B layer: normalization stack
# --------------------------------------------------------------------------


def cluster_representatives(
    values: list[int],
) -> tuple[dict[int, int], list[tuple[int, tuple[int, ...]]]]:
    """B1: 10% relative-tolerance clustering of the positive values.

    Ascending scan; the first value of a cluster is its anchor and every later
    value within ``anchor * 11/10`` joins it; the representative is the
    members' mean, rounded half up.  Near-but-coprime measured costs otherwise
    pin F at a large value for no modelling reason -- the ZLP has to reproduce
    an accidental ratio like 133:128 exactly, and pays for it everywhere else
    in the pool.

    Returns the value -> representative map and the clusters themselves, so a
    report can show what was merged.
    """
    numerator, denominator = CLUSTER_TOLERANCE
    clusters: list[tuple[int, list[int]]] = []
    for value in sorted(v for v in values if v > 0):
        if clusters and value * denominator <= clusters[-1][0] * numerator:
            clusters[-1][1].append(value)
        else:
            clusters.append((value, [value]))
    representatives: dict[int, int] = {}
    for _, members in clusters:
        total, count = sum(members), len(members)
        mean = (2 * total + count) // (2 * count)  # round half up, integer
        for member in members:
            representatives[member] = mean
    return representatives, [
        (anchor, tuple(members)) for anchor, members in clusters
    ]


def apply_floor(values: list[int]) -> list[int]:
    """B2: zero every entry below max/32.

    A pool spanning three orders of magnitude cannot be represented inside
    ``sum(C') <= 300`` at any useful resolution.  The floor compresses the
    spectrum to 32:1 before the ZLP sees it, which is a decision taken here
    and disclosed, rather than one the ZLP takes implicitly by collapsing
    whatever it likes.
    """
    top = max(values, default=0)
    return [0 if FLOOR_DIVISOR * value < top else value for value in values]


def _lexicographic_second_stage(
    costs: list[int], u: int, f_star: int
) -> tuple[list[int], float]:
    """B4: maximize sum(C') subject to the paper's constraints at F = F*.

    The paper's program minimizes F and says nothing about which point of the
    F-optimal face is returned.  That face is not a singleton: on the prepared
    ``fwd_subtiled`` pool it holds both sum(C')=16 and sum(C')=272 at the same
    optimal F, and the degenerate end is the one SCIP happens to report.
    Taking the richest point is a second lexicographic objective the paper
    does not state.

    A fresh model rather than a re-optimized one: changing the objective sense
    on a solved PySCIPOpt model does not re-solve the problem being asked.
    """
    n = len(costs)
    model = Model("prepared_lexicographic_second_stage")
    model.hideOutput()
    cp = [model.addVar(f"cp_{i}", vtype="I", lb=0, ub=u) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            expr = costs[i] * cp[j] - costs[j] * cp[i]
            model.addCons(expr <= f_star)
            model.addCons(-f_star <= expr)
    cp_sum = quicksum(cp)
    model.addCons(cp_sum >= 1)
    model.addCons(cp_sum <= u)
    model.setObjective(cp_sum, "maximize")

    model.optimize()
    status = model.getStatus()
    if status != "optimal":
        raise PreparationError(
            f"lexicographic second stage ended {status} at F*={f_star}"
        )
    scaled = [round(model.getVal(value)) for value in cp]
    return scaled, model.getSolvingTime()


@dataclass(frozen=True)
class PreparedNormalization:
    """Everything the B layer did, entry by entry."""

    entries: tuple[PoolEntry, ...]
    clustered: tuple[int, ...]
    processed: tuple[int, ...]  # after B2, the list the ZLP actually saw
    scaled: tuple[int, ...]  # after B4, at the selected rung
    first_stage_scaled: tuple[int, ...]  # after B3
    objective: int  # F*
    clusters: tuple[tuple[int, tuple[int, ...]], ...]
    u: int
    first_stage_solve_s: float
    second_stage_solve_s: float
    # B6: the selected rung of the frozen ladder, and the whole climb.
    relaxation_k: int = 1
    ladder: tuple[dict, ...] = ()
    p1_cleared: bool = True

    @property
    def positive_entries(self) -> tuple[int, ...]:
        """Indices whose A-layer cost started positive."""
        return tuple(
            index
            for index, entry in enumerate(self.entries)
            if entry.cost > 0
        )

    @property
    def collapsed_entries(self) -> tuple[int, ...]:
        return tuple(
            index for index in self.positive_entries if self.scaled[index] == 0
        )

    @property
    def collapse_ratio(self) -> float:
        positives = self.positive_entries
        if not positives:
            return 0.0
        return len(self.collapsed_entries) / len(positives)

    @property
    def survivor_spread(self) -> float:
        """max/min over the surviving scaled values -- the resolution left."""
        survivors = [value for value in self.scaled if value > 0]
        if not survivors:
            return 0.0
        return max(survivors) / min(survivors)

    def health(self) -> dict[str, object]:
        return {
            "pool_size": len(self.entries),
            "positive_entries": len(self.positive_entries),
            "collapsed_entries": len(self.collapsed_entries),
            "collapse_ratio": round(self.collapse_ratio, 4),
            "survivor_spread": round(self.survivor_spread, 4),
            "distinct_survivors": sorted(
                {value for value in self.scaled if value > 0}
            ),
            "objective_f": self.objective,
            "first_stage_sum": sum(self.first_stage_scaled),
            "second_stage_sum": sum(self.scaled),
            "u": self.u,
            "clusters": len(self.clusters),
            "relaxation_k": self.relaxation_k,
            "p1_cleared": self.p1_cleared,
        }


def _p1_health(
    entries: list[PoolEntry], scaled: list[int]
) -> tuple[float, float, bool]:
    """P1 measured on one candidate scaling: (collapse ratio, spread, clears)."""
    positives = [index for index, e in enumerate(entries) if e.cost > 0]
    collapsed = [index for index in positives if scaled[index] == 0]
    ratio = len(collapsed) / len(positives) if positives else 0.0
    survivors = [value for value in scaled if value > 0]
    spread = max(survivors) / min(survivors) if survivors else 0.0
    clears = (
        ratio <= P1_MAX_COLLAPSE_RATIO and spread >= P1_MIN_SURVIVOR_SPREAD
    )
    return ratio, spread, clears


def normalize_prepared(
    entries: list[PoolEntry], u: int = DEFAULT_U
) -> PreparedNormalization:
    """Run B1-B4, then B6's climb, over an assembled pool.

    B6 exists because one pool -- `fwd` -- has an F-optimal face that is poor
    on its own terms: every point inside F* zeroes almost everything, so B4's
    choice among them cannot help.  The climb tries progressively looser F
    bounds and stops at the *first* rung that clears P1.

    First, not loosest: the ladder is not monotone.  Past a point a looser F
    lets the maximizer spend the whole budget on a few large entries and
    starve the rest, so collapse rises again (measured on `fwd` at U=150:
    24.2% at 2xF*, 0.0% at 3xF* and 4xF*, 30.3% at 8xF*).

    A pool that clears at k=1 -- which is every pool but `fwd` -- selects k=1
    and therefore sits exactly on the paper's own optimum, with no relaxation
    applied at all.  If no rung clears, the result falls back to k=1 rather
    than to the loosest rung: an unhelpful relaxation is not taken, and
    ``p1_cleared`` records that the pool was not rescued.
    """
    costs = [entry.cost for entry in entries]
    representatives, clusters = cluster_representatives(costs)
    clustered = [representatives.get(cost, 0) for cost in costs]
    processed = apply_floor(clustered)
    first = normalize_costs(processed, u=u)

    ladder: list[dict] = []
    rungs: dict[int, tuple[list[int], float]] = {}
    selected = None
    for k in F_RELAXATION_LADDER:
        scaled, solve_s = _lexicographic_second_stage(
            processed, u, first.objective * k
        )
        rungs[k] = (scaled, solve_s)
        ratio, spread, clears = _p1_health(entries, scaled)
        ladder.append(
            {
                "k": k,
                "f_bound": first.objective * k,
                "sum": sum(scaled),
                "collapse_ratio": round(ratio, 4),
                "survivor_spread": round(spread, 4),
                "clears_p1": clears,
            }
        )
        if clears:
            selected = k
            break

    p1_cleared = selected is not None
    chosen = selected if p1_cleared else F_RELAXATION_LADDER[0]
    scaled, second_solve_s = rungs[chosen]
    return PreparedNormalization(
        entries=tuple(entries),
        clustered=tuple(clustered),
        processed=tuple(processed),
        scaled=tuple(scaled),
        first_stage_scaled=tuple(first.scaled),
        objective=first.objective,
        clusters=tuple(clusters),
        u=u,
        first_stage_solve_s=first.solve_time_s,
        second_stage_solve_s=second_solve_s,
        relaxation_k=chosen,
        ladder=tuple(ladder),
        p1_cleared=p1_cleared,
    )


# --------------------------------------------------------------------------
# Curated input -> prepared Problem
# --------------------------------------------------------------------------


@dataclass
class CuratedGraph:
    """A validated curated-ddg-0.2 loop, before any cost model is applied."""

    ddg: dict
    trip_count: int | None
    nodes: dict[int, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)


def read_curated(
    path: str | Path, machine: MachineModel, loop_index: int = 0
) -> CuratedGraph:
    """Validate a curated file with ``ddg``'s own fail-closed readers.

    This is ``ddg.load_problem`` minus its section 5.2 step: identical field
    authority, identical unknown-field rejection, identical RRT shape check.
    Only the cost model differs, and that is the entire point of the regime.
    """
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise PreparationError("curated ddg must be an object")
    if data.get("schema_version") != CURATED_SCHEMA:
        raise PreparationError(
            f"expected schema {CURATED_SCHEMA}, got "
            f"{data.get('schema_version')!r}"
        )
    _require_fields(data, _TOP_FIELDS, "curated ddg")
    loop = data["loops"][loop_index]
    _require_fields(loop, _LOOP_FIELDS, f"loop {loop_index}")
    ddg = loop["ddg"]

    graph = CuratedGraph(ddg=ddg, trip_count=loop.get("trip_count"))
    for raw in ddg["nodes"]:
        node_id = raw["id"]
        _require_fields(raw, _NODE_FIELDS, f"node {node_id}")
        pipeline = raw["pipeline"]
        if pipeline not in machine.capacities and pipeline != "NONE":
            raise PreparationError(
                f"node {node_id} uses unknown functional unit {pipeline!r}"
            )
        latency = _checked_int(raw["latency"], f"node {node_id} latency")
        occupancy = _checked_int(raw["occupancy"], f"node {node_id} occupancy")
        if occupancy > latency:
            raise PreparationError(
                f"node {node_id} occupancy {occupancy} exceeds latency "
                f"{latency}"
            )
        _explicit_rrt(raw["rrt"], node_id, latency, machine)
        graph.nodes[node_id] = Node(
            node_id,
            raw["op_ref"],
            raw["op_kind"],
            pipeline,
            latency,
            occupancy,
            _checked_int(raw["min_warps"], f"node {node_id} min_warps", minimum=1),
            _checked_int(raw["spill_cost"], f"node {node_id} spill_cost"),
        )
    for index, raw in enumerate(ddg["edges"]):
        _require_fields(raw, _EDGE_FIELDS, f"edge {index}")
        graph.edges.append(
            Edge(
                raw["src"],
                raw["dst"],
                _checked_int(raw["distance"], f"edge {index} distance"),
                _checked_int(raw["latency"], f"edge {index} latency"),
                index,
                _checked_bool(raw["blocking"], f"edge {index} blocking"),
            )
        )
    return graph


def _annotations(graph: CuratedGraph, prob: Problem) -> None:
    for raw in graph.ddg["nodes"]:
        node_id = raw["id"]
        prob.regs[node_id] = _checked_int(raw["regs"], f"node {node_id} regs")
        prob.smem_footprint[node_id] = _checked_int(
            raw["smem_footprint"], f"node {node_id} smem_footprint"
        )
        prob.tmem_footprint[node_id] = _checked_int(
            raw["tmem_footprint"], f"node {node_id} tmem_footprint"
        )
        if _checked_bool(
            raw["variable_latency"], f"node {node_id} variable_latency"
        ):
            prob.variable_latency.add(node_id)
        if _checked_bool(raw["streaming"], f"node {node_id} streaming"):
            prob.streaming.add(node_id)
    prob.blocking = {(e.src, e.dst) for e in graph.edges if e.blocking}


def build_problem(
    graph: CuratedGraph,
    normalization: PreparedNormalization,
    machine: MachineModel,
) -> Problem:
    """Assemble the ``Problem`` the unmodified solver core will consume.

    Each node reserves its functional unit for the whole of its normalized
    cost, so cycles(v) and the RRT span coincide.  The literal path instead
    carries a second, idle segment per node; here that span is represented by
    the out-edge latencies, which the curation stage sets to the producer's
    latency.
    """
    prob = Problem(
        nodes=graph.nodes,
        edges=graph.edges,
        machine=machine,
        trip_count=graph.trip_count,
    )
    _annotations(graph, prob)

    prob.lat = {node_id: 0 for node_id in graph.nodes}
    prob.occ = {node_id: 0 for node_id in graph.nodes}
    prob.rrt = {node_id: {} for node_id in graph.nodes}
    prob.edge_lat = {edge.index: 0 for edge in graph.edges}
    prob.spill = {node_id: 0 for node_id in graph.nodes}

    for entry, scaled in zip(normalization.entries, normalization.scaled):
        if entry.kind == "node":
            prob.lat[entry.key] = scaled
            prob.occ[entry.key] = scaled
            pipeline = graph.nodes[entry.key].pipeline
            if scaled and pipeline != "NONE":
                prob.rrt[entry.key] = {pipeline: (1,) * scaled}
        elif entry.kind == "edge":
            prob.edge_lat[entry.key] = scaled
        else:
            prob.spill[entry.key] = scaled

    prob.normalization_u = normalization.u
    prob.normalization_f = normalization.objective
    prob.normalization_cost_pairs = tuple(
        zip(normalization.processed, normalization.scaled)
    )
    return prob


def load_ops_table(raw_ddg_path: str | Path, curated_path: str | Path) -> dict:
    """Read the raw dump's op table, refusing one the curated file disowns.

    A2 needs result shapes, which the curated schema does not carry: curation
    keeps what the paper's formulation reads, and a tile extent is not one of
    those fields.  The raw dump is therefore a second input to this regime,
    and it is bound to the curated file by the hash curation already recorded
    rather than by the caller's word.
    """
    curated = json.loads(Path(curated_path).read_text())
    recorded = (curated.get("curation_source") or {}).get("ddg_sha256")
    actual = hashlib.sha256(Path(raw_ddg_path).read_bytes()).hexdigest()
    if recorded != actual:
        raise PreparationError(
            f"raw dump {raw_ddg_path} has sha256 {actual}, but "
            f"{curated_path} was curated from {recorded}"
        )
    raw = json.loads(Path(raw_ddg_path).read_text())
    ops_table = raw.get("ops")
    if not isinstance(ops_table, dict):
        raise PreparationError(f"{raw_ddg_path} carries no op table")
    return ops_table


@dataclass(frozen=True)
class PreparedProblem:
    problem: Problem
    normalization: PreparedNormalization
    curated_sha256: str
    raw_ddg_sha256: str


def load_prepared_problem(
    curated_path: str | Path,
    raw_ddg_path: str | Path,
    machine: MachineModel | None = None,
    u: int = DEFAULT_U,
    loop_index: int = 0,
) -> PreparedProblem:
    """Curated file + raw dump -> a Problem carrying prepared costs."""
    machine = machine or MachineModel()
    graph = read_curated(curated_path, machine, loop_index=loop_index)
    ops_table = load_ops_table(raw_ddg_path, curated_path)
    entries = assemble_pool(graph.ddg, ops_table)
    normalization = normalize_prepared(entries, u=u)
    return PreparedProblem(
        problem=build_problem(graph, normalization, machine),
        normalization=normalization,
        curated_sha256=hashlib.sha256(
            Path(curated_path).read_bytes()
        ).hexdigest(),
        raw_ddg_sha256=hashlib.sha256(
            Path(raw_ddg_path).read_bytes()
        ).hexdigest(),
    )


def preparation_manifest(prepared: PreparedProblem) -> dict:
    """The ``provenance.preparation`` block of a prepared artifact.

    ``provenance.model`` stays ``"paper"`` on a prepared solution: the solve is
    the unmodified Fig 4/5/6 system.  What changed is upstream of it, and this
    block is where that shows up -- every A/B parameter, the per-entry cost
    chain, and this module's own digest.
    """
    normalization = prepared.normalization
    return {
        "regime": "prepared",
        "prepared_sources_sha256": prepared_sources_sha256(),
        "curated_ddg_sha256": prepared.curated_sha256,
        "raw_ddg_sha256": prepared.raw_ddg_sha256,
        "assumptions": {
            "a1_pool_members": "one entry per node, non-zero edge latencies, "
                               "non-zero spills",
            "a2_issue_width": ISSUE_WIDTH,
            "a2_elementwise_dialects": list(ELEMENTWISE_DIALECTS),
            "a3_async_tokens_keep_measured_cost": True,
            "a4_tmem_functional_unit": "TMEM",
            "b1_cluster_tolerance": list(CLUSTER_TOLERANCE),
            "b2_floor_divisor": FLOOR_DIVISOR,
            "b3_zlp": "paper_joint_solver.normalize.normalize_costs",
            "b4_second_objective": "maximize sum(C') at F = F*",
            "b5_u": normalization.u,
        },
        "health": normalization.health(),
        "pool": [
            {
                "kind": entry.kind,
                "key": entry.key,
                "basis": entry.basis,
                "raw": entry.cost,
                "clustered": clustered,
                "processed": processed,
                "scaled": scaled,
            }
            for entry, clustered, processed, scaled in zip(
                normalization.entries,
                normalization.clustered,
                normalization.processed,
                normalization.scaled,
            )
        ],
    }


# --------------------------------------------------------------------------
# Solve, classify, refit
# --------------------------------------------------------------------------


def preparation_provenance(prepared: PreparedProblem, u: int) -> dict:
    """`provenance.preparation`: the parameters, not the per-entry table.

    The per-entry chain is bulky and lives in its own artifact (`--pool-out`);
    what has to travel with every solution is the digest of this module plus
    the values of the eight assumptions it applied.
    """
    manifest = preparation_manifest(prepared)
    del manifest["pool"]
    manifest["b5_u_applied"] = u
    return manifest


def solve_prepared(
    prepared: PreparedProblem,
    num_warps_override: int | None = None,
    allow_cross_warp: bool = True,
):
    """Algorithm 1 over a prepared Problem.

    `run_search` is the literal search, imported unmodified: the regime
    changes what the solver is asked, never how it searches.  Imported here
    rather than at module scope so the preparation layer stays usable -- and
    testable -- without the SMT shared library present.
    """
    from .search import run_search

    return run_search(
        prepared.problem,
        num_warps_override=num_warps_override,
        allow_cross_warp=allow_cross_warp,
    )


def refit_window(
    prepared: PreparedProblem,
    template_path: str | Path,
    ii: int,
    length: int,
    inputs: dict,
    curation_manifest: dict | None = None,
    window: int = 6,
) -> dict:
    """P2's refit half: is the exact FA4 partition reachable near (II*, L*)?

    The free search returns the first satisfiable point, and the exact
    template may need a little more room at the same II, so the audit probes
    ``L*`` through ``L* + window`` and stops at the first SAT.  A failure to
    find one anywhere in the window is the answer, not an error.
    """
    from .strategy_report import refit_fa4_template

    probed = []
    for candidate in range(length, length + window + 1):
        metadata, satisfiable = refit_fa4_template(
            prepared.problem,
            template_path,
            ii,
            candidate,
            inputs,
            curation_manifest=curation_manifest,
        )
        probed.append({"length": candidate, "verdict": metadata["verdict"]})
        if satisfiable:
            return {
                "satisfiable": True,
                "at_length": candidate,
                "window": probed,
                "metadata": metadata,
            }
    return {"satisfiable": False, "at_length": None, "window": probed}


def probe_order(window: int) -> list[int]:
    """Offsets within [0, window]: ascend first, jump to the middle if that
    start misses twice, backfill the remainder.

    Two earlier orders were tried and are recorded here because each was wrong
    in an instructive way.

    Plain ascending, `L*` first, is right on *loose* constraints and only
    there.  Small L is both where satisfiable points cluster when the
    constraint is easy and where the formula is smallest -- L fixes T and so
    the model size -- so it wins twice.

    Middle-first, `L*+4`, is right on *tight* constraints and was adopted
    after `fwd_subtiled`'s G=4 turned out satisfiable only at L*+4, having
    missed at L*, +1, +2 and +3.  It cut that probe from five points to one,
    and then cost 36x on the loose end: the same graph's G=9 went from 50s at
    L* to 1808s at L*+4.

    The rule below needs neither guess.  Ascend from L*; if the first two
    points both miss -- the signature of a tight instance whose satisfiable
    point is hidden deep -- jump to the middle, which reproduces the G=4
    saving; then backfill everything, because a refutation still requires the
    whole window.  A satisfiable point stops the scan wherever it is found, so
    the ordering only ever changes cost, never a verdict.
    """
    ascending_start = [offset for offset in (0, 1) if offset <= window]
    middle = [
        offset
        for offset in (4, 2, 6)
        if offset <= window and offset not in ascending_start
    ]
    backfill = [
        offset
        for offset in range(window + 1)
        if offset not in ascending_start and offset not in middle
    ]
    return ascending_start + middle + backfill


def _audit_window(
    prepared: PreparedProblem,
    ii: int,
    length: int,
    groups: int,
    inputs: dict,
    window: int,
    offsets: list[int] | None = None,
) -> tuple[object | None, int | None, list[dict]]:
    """Probe `exact_warp_sets == groups` over L* .. L*+window at fixed II.

    There is one probing protocol and one evidence grade: a G is refuted only
    by the whole window.  A narrower window was tried and abolished -- it
    would have declared `fwd_subtiled`'s G=4 unsatisfiable and returned
    G_min=5, which is wrong in the direction that happens to match the
    published number.
    """
    from .joint_smt import solve_joint_audit

    probes: list[dict] = []
    if offsets is None:
        offsets = probe_order(window)
    for candidate in (length + offset for offset in offsets):
        started = time.time()
        solution, verdict = solve_joint_audit(
            prepared.problem,
            ii,
            candidate,
            exact_warp_sets=groups,
            allow_cross_warp=inputs["allow_cross_warp"],
            num_warps_override=inputs["num_warps_override"],
        )
        probes.append(
            {
                "groups": groups,
                "length": candidate,
                "verdict": verdict,
                "seconds": round(time.time() - started, 1),
            }
        )
        print("[min-g]", probes[-1], flush=True)
        if verdict == "sat":
            return solution, candidate, probes
    return None, None, probes


def min_warp_sets_descent(
    prepared: PreparedProblem,
    ii: int,
    length: int,
    upper: int,
    inputs: dict,
    window: int = 6,
    decisive_window: int = 8,
    budget_s: float | None = None,
) -> dict:
    """C1: the fewest distinct warp sets the paper's own system admits.

    Figure 6 quantifies `opw` over the machine's physical warps and contains
    no term that rewards reusing one, so a free solve scatters -- 18 distinct
    sets on `fwd_subtiled` where the reported shape has five.  Eighteen is a
    correct answer to the question the model asks; it is the wrong answer to
    the question the paper's *prose* describes, which is stated throughout in
    groups.  This is the missing selection criterion, and it is structurally
    the same move B4 makes one layer down: keep the paper's satisfiability
    question, then choose among its answers by something the paper never
    states.

    No new solver entry.  `solve_joint_audit`'s `exact_warp_sets` is already a
    pure "exactly N distinct warp sets" counting constraint -- the FA4 refit
    passes it alongside `colocate`/`separate` to pin a specific partition, and
    passing it alone leaves the count and nothing else.  `joint_smt.py` is
    untouched, so `_SOLVER_SOURCES` and every existing artifact's
    `solver_sources_sha256` are untouched with it.

    Binary search assumes SAT is upward-closed in G, which is why `G_min - 1`
    is verified UNSAT across the whole window rather than inferred.

    The result is an instrument, not a solution: the regime's primary artifact
    stays the free solve, whose `provenance.model` is honestly `"paper"`,
    exactly as the FA4 refit is reported beside a solution rather than as one.
    """
    started = time.time()

    def exhausted() -> bool:
        return budget_s is not None and time.time() - started >= budget_s

    probes: list[dict] = []
    best: dict = {}
    # Widest interval still consistent with what has been proven: every G at
    # or above `proven_sat` is achievable, every G at or below `proven_unsat`
    # is not.  An interrupted descent still reports these honestly.
    proven_sat, proven_unsat = upper, 0
    low, high = 1, upper
    interrupted = False
    while low < high:
        if exhausted():
            interrupted = True
            break
        middle = (low + high) // 2
        solution, at_length, window_probes = _audit_window(
            prepared, ii, length, middle, inputs, window
        )
        probes.extend(window_probes)
        if solution is not None:
            best = {"groups": middle, "at_length": at_length,
                    "solution": solution}
            proven_sat = middle
            high = middle
        else:
            proven_unsat = max(proven_unsat, middle)
            low = middle + 1

    settled = not interrupted and low == high
    if settled and best.get("groups") != low:
        solution, at_length, window_probes = _audit_window(
            prepared, ii, length, low, inputs, window
        )
        probes.extend(window_probes)
        if solution is not None:
            best = {"groups": low, "at_length": at_length,
                    "solution": solution}
            proven_sat = low
        else:
            proven_unsat = max(proven_unsat, low)
            settled = False

    complete = settled and best.get("solution") is not None
    report: dict = {
        "instrument": True,
        "ii": ii,
        "free_length": length,
        "window": window,
        "upper_bound": upper,
        "experiment_inputs": dict(inputs),
        "complete": complete,
        "interrupted": interrupted,
        "budget_s": budget_s,
        # Proven regardless of completion; `g_min` is set only once the two
        # bounds meet.
        "proven_sat_at": proven_sat,
        "proven_unsat_at_or_below": proven_unsat or None,
        "g_min": proven_sat if complete else None,
        "at_length": best.get("at_length"),
        "wall_s": None,
        "probes": probes,
    }
    if complete and proven_sat > 1 and proven_unsat != proven_sat - 1:
        _, _, below = _audit_window(
            prepared, ii, length, proven_sat - 1, inputs, window
        )
        probes.extend(below)
        if all(probe["verdict"] == "unsat" for probe in below):
            proven_unsat = max(proven_unsat, proven_sat - 1)
        else:
            # Upward-closedness failed: SAT below a G that probed UNSAT.
            report["monotonicity_violated"] = True
            report["g_min"] = None
        report["proven_unsat_at_or_below"] = proven_unsat or None
    # Harden the one refutation the declaration rests on past the standard
    # window; the bisection needs it for the same reason the anchor descent
    # does.
    hardened = None
    if report["g_min"] is not None and report["g_min"] > 1:
        hardened, extra = harden_refutation(
            prepared, ii, length, report["g_min"] - 1, inputs,
            window, decisive_window,
        )
        probes.extend(extra)
        if not hardened:
            report["g_min"] = None
            report["complete"] = False
    report["decisive_window"] = decisive_window
    report["decisive_refutation_hardened"] = hardened
    report["probe_order"] = probe_order(window)
    report["g_min_minus_one_unsat"] = (
        None if report["g_min"] is None or report["g_min"] == 1
        else proven_unsat == report["g_min"] - 1 and bool(hardened)
    )
    report["wall_s"] = round(time.time() - started, 1)
    return report, best.get("solution")


def harden_refutation(
    prepared: PreparedProblem,
    ii: int,
    length: int,
    groups: int,
    inputs: dict,
    window: int,
    decisive_window: int,
) -> tuple[bool, list[dict]]:
    """Extend one refutation past the standard window.

    `fwd_subtiled` first admits G=4 at L*+4, which leaves a +6 window only two
    points of margin.  The single G value a `G_min` declaration rests on is
    therefore pushed to `decisive_window`; intermediate steps, which only
    steer the search, keep the standard one.  Costs a few extra points once
    per case.
    """
    extra = list(range(window + 1, decisive_window + 1))
    if not extra:
        return True, []
    solution, _, probes = _audit_window(
        prepared, ii, length, groups, inputs, decisive_window, offsets=extra
    )
    for entry in probes:
        entry["decisive"] = True
    return solution is None, probes


def anchor_guided_descent(
    prepared: PreparedProblem,
    ii: int,
    length: int,
    anchor: int,
    inputs: dict,
    window: int = 6,
    decisive_window: int = 8,
    budget_s: float | None = None,
) -> dict:
    """C1 seeded at a v6-era anchor.

    The full bisection costs hours per case because a *counting* constraint is
    the hardest shape for the SMT solver -- the exact-partition refit is two
    orders of magnitude cheaper at the same II because `colocate` and
    `separate` hand it the answer.  Where an anchor exists, stepping from it
    is both cheaper and better targeted than bisecting from the free
    solution's scattered count, and the path is short: two or three steps.

    One window, one evidence grade.  Every G is refuted only by the whole
    window; the saving comes entirely from `probe_order` and from stopping on
    the first satisfiable point.  The G that the declaration rests on is
    hardened further by `harden_refutation`.
    """
    started = time.time()

    def exhausted() -> bool:
        return budget_s is not None and time.time() - started >= budget_s

    probes: list[dict] = []

    def probe(groups: int) -> tuple[object | None, int | None]:
        solution, at_length, window_probes = _audit_window(
            prepared, ii, length, groups, inputs, window
        )
        probes.extend(window_probes)
        return solution, at_length

    best: dict = {}
    solution, at_length = probe(anchor)
    if solution is None:
        # The anchor is not achievable here.  Climb for one that is, by
        # bisection over (anchor, num_warps]; the upward branch accepts the
        # same whole-window refutations as the downward one.
        low, high = anchor + 1, prepared.problem.machine.num_warps
        while low <= high and not exhausted():
            middle = (low + high) // 2
            solution, at_length = probe(middle)
            if solution is not None:
                best = {"groups": middle, "at_length": at_length,
                        "solution": solution}
                high = middle - 1
            else:
                low = middle + 1
    else:
        best = {"groups": anchor, "at_length": at_length, "solution": solution}

    refuted_at = None
    while best.get("solution") is not None and best["groups"] > 1:
        if exhausted():
            break
        candidate = best["groups"] - 1
        solution, at_length = probe(candidate)
        if solution is None:
            refuted_at = candidate
            break
        best = {"groups": candidate, "at_length": at_length,
                "solution": solution}

    g_min = best["groups"] if best.get("solution") is not None else None
    complete = g_min is not None and refuted_at == g_min - 1
    hardened = None
    if complete and not exhausted():
        hardened, extra = harden_refutation(
            prepared, ii, length, refuted_at, inputs, window, decisive_window
        )
        probes.extend(extra)
        if not hardened:
            # The decisive G turned out satisfiable past the standard window,
            # exactly the failure the hardening exists to catch.
            best = {"groups": refuted_at}
            g_min = None
            complete = False

    report = {
        "instrument": True,
        "method": "anchor-guided",
        "anchor": anchor,
        "ii": ii,
        "free_length": length,
        "window": window,
        "decisive_window": decisive_window,
        "probe_order": probe_order(window),
        "experiment_inputs": dict(inputs),
        "budget_s": budget_s,
        "interrupted": exhausted(),
        "g_min": g_min,
        "at_length": best.get("at_length"),
        # A descent stopped by the budget rather than by a refutation has
        # found an upper bound, not a minimum.
        "complete": complete,
        "g_min_minus_one_unsat": (
            None if g_min is None or g_min == 1 else complete
        ),
        "decisive_refutation_hardened": hardened,
        "refutation_points": decisive_window + 1,
        "probes": probes,
        "wall_s": round(time.time() - started, 1),
    }
    return report, best.get("solution")


def shape_cost_probe(
    prepared: PreparedProblem,
    target_groups: int,
    ii_start: int,
    length: int,
    inputs: dict,
    window: int = 6,
    max_ii: int | None = None,
) -> dict:
    """How much II does a target group count cost?

    Only meaningful when `G_min` exceeds the reported shape's group count: it
    answers "the published shape is not free here -- what does it cost", which
    is a number rather than an absence.
    """
    ceiling = max_ii if max_ii is not None else ii_start * 3
    probes: list[dict] = []
    for candidate_ii in range(ii_start, ceiling + 1):
        solution, at_length, window_probes = _audit_window(
            prepared, candidate_ii, length, target_groups, inputs, window
        )
        probes.extend(window_probes)
        if solution is not None:
            return {
                "target_groups": target_groups,
                "satisfiable": True,
                "ii": candidate_ii,
                "at_length": at_length,
                "ii_cost": candidate_ii - ii_start,
                "probes": probes,
            }
    return {
        "target_groups": target_groups,
        "satisfiable": False,
        "searched_to_ii": ceiling,
        "probes": probes,
    }


def classify_prepared(
    prepared: PreparedProblem,
    solution,
    curation_manifest: dict | None = None,
    curation_manifest_sha256: str | None = None,
) -> dict:
    """The unmodified classifier, fed directly rather than through an artifact.

    `strategy_report.main` reaches its Problem through
    `schedule_plan.load_schedule_context`, which validates a
    `paper-joint-solution-v2` artifact.  A prepared solution deliberately
    carries a different schema, so it must not pass that gate; `classify` and
    `refit_fa4_template` take the Problem itself and are called here directly.
    """
    from .strategy_report import classify, warp_set_labels

    return classify(
        prepared.problem,
        warp_set_labels(solution.warp_sets),
        solution.cycles,
        curation_manifest,
        curation_manifest_sha256,
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _write(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=1) + "\n")


def build_parser():
    """A separate entry, deliberately.

    The literal CLI gains no flag from this regime: `python -m
    paper_joint_solver` must keep meaning exactly what the paper describes, so
    the prepared path is reached only by naming this module.
    """
    import argparse

    from .__main__ import add_experiment_inputs

    parser = argparse.ArgumentParser(prog="paper_joint_solver.prepared")
    parser.add_argument("curated_ddg", help="curated-ddg-0.2 artifact")
    parser.add_argument(
        "--raw-ddg",
        required=True,
        help="the raw dump the curated file records; supplies the tile "
        "extents A2 needs and is checked against curation_source.ddg_sha256",
    )
    parser.add_argument("-o", "--out", required=True)
    parser.add_argument(
        "--pool-out", help="per-entry cost chain: raw, clustered, floored, C'"
    )
    parser.add_argument(
        "--strategy-out", help="classifier report for the free solution"
    )
    parser.add_argument(
        "--fa4-template",
        help="exact-partition template for the P2 refit audit",
    )
    parser.add_argument("--curation-manifest")
    parser.add_argument(
        "--refit-window",
        type=int,
        default=6,
        help="probe L* .. L*+window for the exact-partition refit",
    )
    parser.add_argument(
        "--min-warp-sets",
        action="store_true",
        help="C1: descend on the number of distinct warp sets and report the "
        "smallest the paper's own system admits, as an instrument beside the "
        "free solution",
    )
    parser.add_argument(
        "--min-warp-sets-anchor",
        type=int,
        default=None,
        help="seed the descent at this group count instead of bisecting from "
        "the free solution's; enables the asymmetric-window protocol",
    )
    parser.add_argument(
        "--min-warp-sets-window",
        type=int,
        default=6,
        help="L window for satisfiability-seeking probes",
    )
    parser.add_argument(
        "--min-warp-sets-decisive-window",
        type=int,
        default=8,
        help="wider L window for the one refutation a G_min declaration rests "
        "on (anchor-guided only)",
    )
    parser.add_argument(
        "--min-warp-sets-budget-s",
        type=float,
        default=None,
        help="wall-clock bound on the descent, checked between L windows; an "
        "interrupted descent reports the bounds it proved, never a guess",
    )
    parser.add_argument(
        "--shape-probe-groups",
        type=int,
        default=None,
        help="if the descent lands above this count, measure how far II must "
        "rise before it is reachable",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="run the published first-window protocol instead of Algorithm 1; "
        "the bounded evidence an UNSAT ablation needs, since Algorithm 1 does "
        "not terminate on a structurally unsatisfiable model",
    )
    add_experiment_inputs(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    from .__main__ import configure_machine, recorded_experiment_inputs
    from .schedule_plan import PAPER_MODEL_PROFILE, solution_provenance

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.refit_window < 0:
        parser.error("--refit-window must be nonnegative")

    machine = MachineModel()
    configure_machine(parser, args, machine)
    prepared = load_prepared_problem(
        args.curated_ddg,
        args.raw_ddg,
        machine=machine,
        u=args.normalization_u,
    )
    health = prepared.normalization.health()
    print(f"[preparation] {json.dumps(health, sort_keys=True)}", flush=True)

    if args.pool_out:
        _write(args.pool_out, preparation_manifest(prepared))
        print(f"[preparation] wrote {args.pool_out}")

    base_provenance = solution_provenance(
        args.curated_ddg,
        machine,
        args.normalization_u,
        model=PAPER_MODEL_PROFILE,
        inputs=recorded_experiment_inputs(args),
    )
    # The solve is the unmodified paper model, so `model` stays "paper"; what
    # this regime changed sits one level up, in its own block.
    base_provenance["preparation"] = preparation_provenance(
        prepared, args.normalization_u
    )

    if args.probe:
        from .probe import EXECUTION_MODE, probe_first_window, probe_sources_sha256

        base_provenance["probe_sources_sha256"] = probe_sources_sha256()
        window = probe_first_window(
            prepared.problem,
            num_warps_override=args.num_warps_override,
            allow_cross_warp=not args.no_cross_warp,
        )
        _write(
            args.out,
            {
                "schema_version": PREPARED_PROBE_SCHEMA,
                "execution_mode": EXECUTION_MODE,
                "provenance": base_provenance,
                **window,
            },
        )
        print(
            f"[prepared probe] I={window['ii']} "
            f"unsat {window['unsat_count']}/{window['window_size']} "
            f"-> {args.out}",
            flush=True,
        )
        return 0

    result = solve_prepared(
        prepared,
        num_warps_override=args.num_warps_override,
        allow_cross_warp=not args.no_cross_warp,
    )

    provenance = base_provenance
    payload = {
        "schema_version": PREPARED_SOLUTION_SCHEMA,
        "provenance": provenance,
        "attempts": result.attempts,
        "wall_s": round(result.wall_s, 1),
        "status": result.status,
        "satisfiable": result.solution is not None,
    }
    solution = result.solution
    if solution is not None:
        payload.update(
            {
                "ii": solution.ii,
                "length": solution.length,
                "copies": solution.copies,
                "horizon": solution.horizon,
                "cycles": solution.cycles,
                "warp_sets": solution.warp_sets,
                "stats": solution.stats,
            }
        )
    _write(args.out, payload)
    print(
        f"[prepared] {result.status} after {result.wall_s:.1f}s -> {args.out}",
        flush=True,
    )
    if solution is None:
        return 0

    curation_manifest = curation_manifest_sha256 = None
    if args.curation_manifest:
        manifest_bytes = Path(args.curation_manifest).read_bytes()
        curation_manifest = json.loads(manifest_bytes)
        curation_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

    min_g_report = min_g_solution = None
    if args.min_warp_sets:
        inputs = provenance["experiment_inputs"]
        if args.min_warp_sets_anchor is not None:
            min_g_report, min_g_solution = anchor_guided_descent(
                prepared,
                solution.ii,
                solution.length,
                args.min_warp_sets_anchor,
                inputs,
                window=args.min_warp_sets_window,
                decisive_window=args.min_warp_sets_decisive_window,
                budget_s=args.min_warp_sets_budget_s,
            )
        else:
            min_g_report, min_g_solution = min_warp_sets_descent(
                prepared,
                solution.ii,
                solution.length,
                solution.stats["num_warp_sets"],
                inputs,
                window=args.min_warp_sets_window,
                decisive_window=args.min_warp_sets_decisive_window,
                budget_s=args.min_warp_sets_budget_s,
            )
        if (
            args.shape_probe_groups is not None
            and min_g_report.get("g_min") is not None
            and min_g_report["g_min"] > args.shape_probe_groups
        ):
            min_g_report["shape_cost"] = shape_cost_probe(
                prepared,
                args.shape_probe_groups,
                solution.ii,
                solution.length,
                inputs,
                window=args.min_warp_sets_window,
            )
        # A summary travels with the solution; the probe trace stays in the
        # strategy report, which is where instruments belong.
        provenance["preparation"]["min_warp_sets"] = {
            key: value
            for key, value in min_g_report.items()
            if key not in ("probes", "shape_cost")
        }
        payload["provenance"] = provenance
        _write(args.out, payload)
        print(
            f"[min-g] G_min={min_g_report['g_min']} "
            f"complete={min_g_report['complete']} "
            f"after {min_g_report['wall_s']}s",
            flush=True,
        )

    if args.strategy_out:
        report = classify_prepared(
            prepared, solution, curation_manifest, curation_manifest_sha256
        )
        report.update(
            {
                "regime": "prepared",
                "ii": solution.ii,
                "length": solution.length,
                "wall_s": payload["wall_s"],
                "preparation": preparation_provenance(
                    prepared, args.normalization_u
                ),
            }
        )
        if min_g_report is not None:
            # P2/P3 shape verdicts are read off the instrument solution; the
            # free classification above stays in the same file so both are
            # archived and the difference is visible.
            if min_g_solution is not None:
                min_g_report["classification"] = classify_prepared(
                    prepared,
                    min_g_solution,
                    curation_manifest,
                    curation_manifest_sha256,
                )
                min_g_report["warp_sets"] = min_g_solution.warp_sets
                min_g_report["cycles"] = min_g_solution.cycles
                min_g_report["stats"] = min_g_solution.stats
            report["min_warp_sets"] = min_g_report
        if args.fa4_template:
            report["fa4_template_refit"] = refit_window(
                prepared,
                args.fa4_template,
                solution.ii,
                solution.length,
                provenance["experiment_inputs"],
                curation_manifest=curation_manifest,
                window=args.refit_window,
            )
        _write(args.strategy_out, report)
        print(f"[prepared] wrote {args.strategy_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
