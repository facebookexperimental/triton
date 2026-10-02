"""Input curation: raw compiler dump -> the paper's tile-level graph G.

The paper's G is a hand-curated tile-level dependence graph: sec 5.3's
streaming rule ("no incoming data dependencies") only holds once the scalar
address arithmetic the compiler emits is gone. This module performs that
curation explicitly, as a pre-solver stage, so the solver itself contains no
filtering, classification or defaulting.

    raw ddg.json + baseline schedule_graph.json
        -> [curate_ddg] -> curated_ddg.json + curation_manifest.json

Every curation judgment is transcribed into the manifest, which is the only
channel that remembers what was dropped and why. The field-level authority for
both artifacts is the frozen schema document `curated-ddg-0.2-schema.md`; this
module does not restate it.

Rules C1-C9 run in the order they are numbered below.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .machine import MachineModel
from .resource_model import derive_resources

CURATED_SCHEMA = "curated-ddg-0.2"
MANIFEST_SCHEMA = "curation-manifest-0.2"

_CURATOR_SOURCES = ("curate_ddg.py", "resource_model.py")

_SCALAR_RESULT_TYPES = frozenset(
    ("bf16", "f16", "f32", "f64", "i1", "i8", "i16", "i32", "i64", "index")
)

_DROPPED_FIELD_NAMES = [
    "node.self_latency",
    "node.is_super_node",
    "edge.kind",
    "edge.src_result_idx",
]

_DROPPED_LOOP_FIELDS = (
    "min_ii",
    "res_mii",
    "rec_mii",
    "is_outer",
    "induction_var",
    "lower_bound",
    "upper_bound",
    "step",
    "trip_count_estimated",
)


class CurationError(ValueError):
    """A raw input the curation stage refuses (fail-closed input gate)."""


@dataclass
class _RawNode:
    id: int
    op_ref: str
    op_kind: str
    pipeline: str  # after C3
    raw_pipeline: str
    latency: int
    occupancy: int
    min_warps: int
    rrt: dict[str, list[int]] = field(default_factory=dict)
    rrt_rule: str = "explicit"
    explicit_spill_cost: bool = False
    spill_cost: int = 0
    regs: int = 0
    smem_footprint: int = 0
    tmem_footprint: int = 0
    variable_latency: bool = False
    streaming: bool = False


@dataclass
class _RawEdge:
    src: int
    dst: int
    distance: int
    latency: int
    order: int
    src_result_idx: int = 0
    token: bool = False
    blocking: bool = False
    blocking_rule: str | None = None
    synthesized_via: list[int] | None = None
    synthesized_reason: str | None = None


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def curator_sources_sha256() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in _CURATOR_SOURCES:
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def _nonnegative_int(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CurationError(f"{label} must be a nonnegative integer")
    return value


def _is_scalar_result_type(result_type: str) -> bool:
    return (
        result_type in _SCALAR_RESULT_TYPES
        or result_type.startswith("!tt.ptr<")
    )


def _rrt_row(value, node_id: int, functional_unit: str) -> list[int]:
    if isinstance(value, list):
        return [
            _nonnegative_int(
                amount, f"node {node_id} RRT[{functional_unit},{cycle}]"
            )
            for cycle, amount in enumerate(value)
        ]
    if isinstance(value, dict):
        entries = []
        for raw_cycle, amount in value.items():
            try:
                cycle = int(raw_cycle)
            except (TypeError, ValueError) as error:
                raise CurationError(
                    f"node {node_id} RRT cycle {raw_cycle!r} is not an integer"
                ) from error
            if cycle < 0:
                raise CurationError(
                    f"node {node_id} RRT cycle {cycle} must be nonnegative"
                )
            entries.append(
                (
                    cycle,
                    _nonnegative_int(
                        amount, f"node {node_id} RRT[{functional_unit},{cycle}]"
                    ),
                )
            )
        row = [0] * (max((c for c, _ in entries), default=-1) + 1)
        for cycle, amount in entries:
            row[cycle] = amount
        return row
    raise CurationError(
        f"node {node_id} RRT row {functional_unit!r} must be a list or map"
    )


def _explicit_rrt(raw, node_id: int, latency: int, machine: MachineModel):
    if not isinstance(raw, dict):
        raise CurationError(f"node {node_id} rrt must be an object")
    rows = {}
    for functional_unit, value in raw.items():
        if functional_unit not in machine.capacities:
            raise CurationError(
                f"node {node_id} uses unknown functional unit "
                f"{functional_unit!r}"
            )
        row = _rrt_row(value, node_id, functional_unit)
        if isinstance(value, dict) and len(row) <= latency:
            row.extend([0] * (latency - len(row)))
        elif len(row) != latency:
            raise CurationError(
                f"node {node_id} RRT row {functional_unit!r} has span "
                f"{len(row)}, expected latency {latency}"
            )
        rows[functional_unit] = row
    return rows


def _read_nodes(raw_nodes, machine: MachineModel) -> dict[int, _RawNode]:
    """C3 (functional-unit reassignment) + C4 (RRT made explicit) + C9."""
    nodes: dict[int, _RawNode] = {}
    for raw in raw_nodes:
        node_id = raw["id"]
        op_kind = raw["op_kind"]
        raw_pipeline = raw.get("pipeline", "NONE")
        # C3: Blackwell TMEM ports are their own functional unit.
        if "tmem_load" in op_kind or "tmem_store" in op_kind:
            pipeline = "TMEM"
        else:
            pipeline = raw_pipeline
        if pipeline not in machine.capacities and pipeline != "NONE":
            raise CurationError(
                f"node {node_id} uses unknown functional unit {pipeline!r}"
            )
        latency = _nonnegative_int(
            raw.get("latency", 0), f"node {node_id} latency"
        )
        # C4: the curated file always carries an explicit RRT.
        if "rrt" in raw:
            rows = _explicit_rrt(raw["rrt"], node_id, latency, machine)
            occupancy = latency
            rrt_rule = "explicit"
        else:
            occupancy = _nonnegative_int(
                raw.get("occupancy", latency), f"node {node_id} occupancy"
            )
            if occupancy > latency:
                raise CurationError(
                    f"node {node_id} occupancy {occupancy} exceeds "
                    f"latency {latency}"
                )
            rows = (
                {pipeline: [1] * occupancy + [0] * (latency - occupancy)}
                if pipeline in machine.capacities and latency > 0
                else {}
            )
            rrt_rule = "synthesized_from_occupancy"
        explicit_spill = "spill_cost" in raw or "spillcost" in raw
        spill_cost = raw.get("spill_cost", raw.get("spillcost", 0))
        if explicit_spill:
            spill_cost = _nonnegative_int(
                spill_cost, f"node {node_id} spill cost"
            )
        nodes[node_id] = _RawNode(
            id=node_id,
            op_ref=raw.get("op_ref", ""),
            op_kind=op_kind,
            pipeline=pipeline,
            raw_pipeline=raw_pipeline,
            latency=latency,
            occupancy=occupancy,
            min_warps=raw.get("min_warps", 1),
            rrt=rows,
            rrt_rule=rrt_rule,
            explicit_spill_cost=explicit_spill,
            spill_cost=spill_cost if explicit_spill else 0,
        )
    return nodes


def _read_edges(raw_edges, nodes, ops_table) -> list[_RawEdge]:
    """C2: token-edge detection, the only semantic use of the raw ops table."""
    edges = []
    for order, raw in enumerate(raw_edges):
        src_result_idx = raw.get("src_result_idx", 0)
        result_types = ops_table.get(nodes[raw["src"]].op_ref, {}).get(
            "result_types", []
        )
        token = (
            0 <= src_result_idx < len(result_types)
            and result_types[src_result_idx] == "!ttg.async.token"
        )
        edges.append(
            _RawEdge(
                src=raw["src"],
                dst=raw["dst"],
                distance=raw.get("distance", 0),
                latency=raw.get("latency", 0),
                order=order,
                src_result_idx=src_result_idx,
                token=token,
            )
        )
    return edges


def _baseline_infra_nodes(
    graph_nodes, nodes, *, require_full_coverage: bool
) -> set[int]:
    """C1.1: the paper gives no curation criterion, so we proxy it with the
    baseline emitter's own infrastructure ownership (warp_group < 0)."""
    if not isinstance(graph_nodes, list):
        raise CurationError("baseline graph nodes must be a list")
    by_id: dict[int, dict] = {}
    for position, raw_node in enumerate(graph_nodes):
        if not isinstance(raw_node, dict):
            raise CurationError(f"baseline graph node {position} must be an object")
        node_id = raw_node.get("id")
        if isinstance(node_id, bool) or not isinstance(node_id, int):
            raise CurationError(f"baseline graph node {position} has an invalid id")
        if node_id in by_id:
            raise CurationError(f"baseline graph has duplicate node id {node_id}")
        if node_id not in nodes:
            raise CurationError(f"baseline graph has unknown DDG node id {node_id}")
        warp_group = raw_node.get("warp_group")
        if "warp_group" in raw_node and (
            isinstance(warp_group, bool) or not isinstance(warp_group, int)
        ):
            raise CurationError(
                f"baseline graph node {node_id} warp_group must be an integer"
            )
        by_id[node_id] = raw_node
    if require_full_coverage and set(by_id) != set(nodes):
        missing = sorted(set(nodes) - set(by_id))
        raise CurationError(f"baseline graph is missing DDG nodes {missing}")
    for node_id, raw_node in by_id.items():
        ddg_node = nodes[node_id]
        for name, expected in (
            ("op_ref", ddg_node.op_ref),
            ("op_kind", ddg_node.op_kind),
        ):
            if require_full_coverage and name not in raw_node:
                raise CurationError(
                    f"baseline graph node {node_id} is missing {name}"
                )
            if name in raw_node and raw_node[name] != expected:
                raise CurationError(
                    f"baseline graph node {node_id} {name} does not match the DDG"
                )
    return {
        node_id
        for node_id, node in by_id.items()
        if node.get("warp_group", 0) < 0
    }


def _validate_infra_data_edges(edges, nodes, infra, ops_table) -> None:
    """C1.2: infrastructure may only carry scalar/pointer results."""
    for edge in edges:
        if edge.src not in infra or edge.token:
            continue
        result_types = ops_table.get(nodes[edge.src].op_ref, {}).get(
            "result_types", []
        )
        result_type = (
            result_types[edge.src_result_idx]
            if 0 <= edge.src_result_idx < len(result_types)
            else None
        )
        if result_type is None or not _is_scalar_result_type(result_type):
            raise CurationError(
                f"infra data edge {edge.src}->{edge.dst} result "
                f"{edge.src_result_idx} must be scalar, got {result_type!r}"
            )


def _drop_infra(edges, infra, node_records):
    """C1.3: drop infra nodes and rewire kept -> infra* -> kept chains."""
    kept_edges = [
        edge for edge in edges if edge.src not in infra and edge.dst not in infra
    ]
    removed = [
        edge for edge in edges if edge.src in infra or edge.dst in infra
    ]
    out_edges: dict[int, list[_RawEdge]] = {}
    for edge in edges:
        out_edges.setdefault(edge.src, []).append(edge)

    synthesized: list[_RawEdge] = []
    order = len(edges)
    for edge in edges:
        if edge.src in infra or edge.dst not in infra:
            continue
        # Walk every pure-infra path out of edge.dst until it lands on a kept
        # node, summing d and delta along the chain (I9).
        stack = [(edge.dst, edge.distance, edge.latency, [edge.dst])]
        while stack:
            current, distance, latency, via = stack.pop()
            for nxt in out_edges.get(current, ()):
                total_d = distance + nxt.distance
                total_lat = latency + nxt.latency
                if nxt.dst in infra:
                    if nxt.dst in via:
                        continue
                    stack.append(
                        (nxt.dst, total_d, total_lat, [*via, nxt.dst])
                    )
                else:
                    synthesized.append(
                        _RawEdge(
                            src=edge.src,
                            dst=nxt.dst,
                            distance=total_d,
                            latency=total_lat,
                            order=order,
                            synthesized_via=list(via),
                            synthesized_reason="rewired_through_dropped_chain",
                        )
                    )
                    order += 1
    for node_id in infra:
        node_records.pop(node_id, None)
    return kept_edges + synthesized, removed, synthesized


def _resolve_footprints(nodes, edges, ops_table, machine):
    """C5: regs plus per-node footprints with a unique owner per storage."""
    registers, objects = derive_resources(
        nodes, edges, ops_table, tmem_column_bytes=machine.tmem_column_bytes
    )
    for node_id, words in registers.items():
        nodes[node_id].regs = words

    out_edges: dict[int, list[_RawEdge]] = {}
    for edge in edges:
        out_edges.setdefault(edge.src, []).append(edge)

    records = []
    synthesized: list[_RawEdge] = []
    order = max((edge.order for edge in edges), default=-1) + 1
    for storage_id in sorted(objects):
        storage = objects[storage_id]
        members = sorted(storage.members)
        owner = _select_owner(storage, members, nodes)
        if storage.kind == "smem":
            nodes[owner].smem_footprint += storage.amount
            basis = {"bytes": storage.amount, "rule": "max_union"}
        else:
            nodes[owner].tmem_footprint += storage.amount
            basis = {
                "bytes": storage.amount * machine.tmem_column_bytes,
                "tmem_column_bytes": machine.tmem_column_bytes,
                "rule": "max_union",
            }
        # B5: the owner's Fig-5 liveness interval must cover every member's
        # use. Liveness is quantified over the whole curated E, so a use is
        # covered exactly when it is reachable from the owner -- which is how
        # the desugared MMA completion token extends accumulator liveness.
        reachable = _reachable_from(owner, out_edges)
        cover = "desugared_use_edges"
        for member in members:
            if member == owner:
                continue
            for edge in out_edges.get(member, ()):
                if edge.dst in members or edge.dst in reachable:
                    continue
                synthesized.append(
                    _RawEdge(
                        src=owner,
                        dst=edge.dst,
                        distance=edge.distance,
                        latency=edge.latency,
                        order=order,
                        synthesized_reason="footprint_liveness_use",
                    )
                )
                reachable.add(edge.dst)
                order += 1
                cover = "synthesized"
        records.append(
            {
                "object": storage_id,
                "kind": storage.kind,
                "amount": storage.amount,
                "amount_basis": basis,
                "owner": owner,
                "members": members,
                "rules": _footprint_rules(storage, nodes),
                "liveness_cover": cover,
            }
        )
    return records, synthesized


def _reachable_from(start: int, out_edges) -> set[int]:
    seen = {start}
    work = [start]
    while work:
        current = work.pop()
        for edge in out_edges.get(current, ()):
            if edge.dst not in seen:
                seen.add(edge.dst)
                work.append(edge.dst)
    return seen


def _select_owner(storage, members, nodes) -> int:
    """Deterministic owner choice (schema §5); ties break on the lowest id."""
    if storage.kind == "smem":
        loads = [
            m
            for m in members
            if nodes[m].op_kind == "tt.descriptor_load"
            and nodes[m].pipeline == "TMA"
        ]
        if loads:
            return min(loads)
        allocs = [m for m in members if nodes[m].op_kind == "ttg.local_alloc"]
        if allocs:
            return min(allocs)
        return min(members)
    allocs = [m for m in members if nodes[m].op_kind == "ttng.tmem_alloc"]
    if allocs:
        return min(allocs)
    return min(members)


def _footprint_rules(storage, nodes) -> list[str]:
    rules = []
    for member in sorted(storage.members):
        kind = nodes[member].op_kind
        if kind == "ttg.local_alloc":
            rules.append("local_alloc")
        elif kind == "tt.descriptor_load":
            rules.append("descriptor_load")
        elif kind == "ttng.tmem_alloc":
            rules.append("tmem_alloc")
        elif kind == "ttng.tmem_store":
            rules.append("tmem_store_operand0")
        elif "mma" in kind:
            rules.append("mma_operand2_accumulator")
    return sorted(set(rules))


def _annotate(nodes, edges, machine) -> None:
    """C6 (blocking), C7 (spill cost), C8 (variable latency / streaming)."""
    for edge in edges:
        src = nodes[edge.src]
        if src.pipeline in ("TC", "TMA"):
            edge.blocking = True
            edge.blocking_rule = f"async_producer_unit_{src.pipeline}"
        elif edge.token:
            edge.blocking = True
            edge.blocking_rule = "async_token_completion"
        else:
            edge.blocking = False
            edge.blocking_rule = None

    for node in nodes.values():
        if node.explicit_spill_cost:
            continue
        node.spill_cost = machine.spill_cost if node.regs > 0 else 0

    has_incoming = {edge.dst for edge in edges}
    for node in nodes.values():
        node.variable_latency = (
            node.pipeline == "TMA" and "load" in node.op_kind
        )
        # Paper sec 5.3, literally, on curated E: no exemptions remain.
        node.streaming = node.variable_latency and node.id not in has_incoming


def _spill_rule(node: _RawNode) -> str:
    if node.explicit_spill_cost:
        return "explicit"
    return "default_regs_producer" if node.regs > 0 else "memory_backed_zero"


def _sorted_edges(edges):
    return sorted(
        edges, key=lambda e: (e.src, e.dst, e.distance, e.latency, e.order)
    )


def curate(
    raw_ddg_path: str | Path,
    baseline_graph_path: str | Path,
    *,
    loop_index: int = 0,
    strict_baseline: bool = True,
    machine: MachineModel | None = None,
) -> tuple[dict, dict]:
    """Run C1-C9 and return the (curated, manifest) artifact pair."""
    machine = machine or MachineModel()
    data = json.loads(Path(raw_ddg_path).read_text())
    graph_data = json.loads(Path(baseline_graph_path).read_text())
    loop = data["loops"][loop_index]
    ops_table = data.get("ops", {})

    nodes = _read_nodes(loop["ddg"]["nodes"], machine)
    edges = _read_edges(loop["ddg"]["edges"], nodes, ops_table)

    graph_nodes = graph_data["loops"][loop_index]["schedule_loop"]["graph"][
        "nodes"
    ]
    infra = _baseline_infra_nodes(
        graph_nodes, nodes, require_full_coverage=strict_baseline
    )
    _validate_infra_data_edges(edges, nodes, infra, ops_table)

    dropped_nodes = {
        node_id: nodes[node_id] for node_id in sorted(infra)
    }
    token_edges = sum(1 for edge in edges if edge.token)
    edges, removed_edges, rewired = _drop_infra(edges, infra, nodes)

    footprints, liveness_edges = _resolve_footprints(
        nodes, edges, ops_table, machine
    )
    edges.extend(liveness_edges)
    _annotate(nodes, edges, machine)

    curated = {
        "schema_version": CURATED_SCHEMA,
        "curation_source": {
            "ddg_sha256": sha256_file(raw_ddg_path),
            "baseline_graph_sha256": sha256_file(baseline_graph_path),
            "curator_sources_sha256": curator_sources_sha256(),
        },
        "loops": [
            {
                "loop_id": loop.get("loop_id", loop_index),
                "trip_count": loop.get("trip_count"),
                "ddg": {
                    "nodes": [
                        {
                            "id": node.id,
                            "op_ref": node.op_ref,
                            "op_kind": node.op_kind,
                            "pipeline": node.pipeline,
                            "latency": node.latency,
                            "occupancy": node.occupancy,
                            "rrt": {
                                unit: list(row)
                                for unit, row in sorted(node.rrt.items())
                            },
                            "min_warps": node.min_warps,
                            "regs": node.regs,
                            "spill_cost": node.spill_cost,
                            "smem_footprint": node.smem_footprint,
                            "tmem_footprint": node.tmem_footprint,
                            "variable_latency": node.variable_latency,
                            "streaming": node.streaming,
                        }
                        for node in sorted(nodes.values(), key=lambda n: n.id)
                    ],
                    "edges": [
                        {
                            "src": edge.src,
                            "dst": edge.dst,
                            "distance": edge.distance,
                            "latency": edge.latency,
                            "blocking": edge.blocking,
                        }
                        for edge in _sorted_edges(edges)
                    ],
                },
            }
        ],
    }

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "curation_source": dict(curated["curation_source"]),
        "loops": [
            {
                "loop_id": loop.get("loop_id", loop_index),
                "nodes": _manifest_nodes(nodes, dropped_nodes),
                "edges": _manifest_edges(edges, removed_edges),
                "footprints": footprints,
                "observations": {
                    "dropped_node_kinds": _kind_counts(dropped_nodes),
                    "token_edges_resolved": token_edges,
                    "edges_synthesized": len(rewired) + len(liveness_edges),
                    "dropped_loop_fields": {
                        name: loop[name]
                        for name in _DROPPED_LOOP_FIELDS
                        if name in loop
                    },
                    "dropped_field_names": list(_DROPPED_FIELD_NAMES),
                },
            }
        ],
    }
    return curated, manifest


def _manifest_nodes(nodes, dropped_nodes) -> list[dict]:
    records = []
    for node_id, node in sorted(dropped_nodes.items()):
        records.append(
            {
                "id": node_id,
                "op_kind": node.op_kind,
                "action": "dropped",
                "reason": "emitter-infra (baseline warp_group<0)",
            }
        )
    for node in sorted(nodes.values(), key=lambda n: n.id):
        record = {
            "id": node.id,
            "action": "kept",
            "pipeline": {"from": node.raw_pipeline, "to": node.pipeline},
            "rrt": node.rrt_rule,
            "spill_cost": {
                "value": node.spill_cost,
                "rule": _spill_rule(node),
            },
            "variable_latency": node.variable_latency,
            "streaming": node.streaming,
        }
        if node.raw_pipeline != node.pipeline:
            record["pipeline_rule"] = "tmem_port_unit"
        records.append(record)
    return sorted(records, key=lambda r: (r["id"], r["action"]))


def _manifest_edges(edges, removed_edges) -> list[dict]:
    records = []
    for edge in _sorted_edges(removed_edges):
        records.append(
            {
                "src": edge.src,
                "dst": edge.dst,
                "action": "removed",
                "reason": "src dropped (emitter-infra)",
            }
        )
    for edge in _sorted_edges(edges):
        if edge.synthesized_reason is not None:
            record = {
                "src": edge.src,
                "dst": edge.dst,
                "action": "synthesized",
                "reason": edge.synthesized_reason,
                "distance": edge.distance,
                "latency": edge.latency,
            }
            if edge.synthesized_via is not None:
                record["via"] = edge.synthesized_via
        else:
            record = {"src": edge.src, "dst": edge.dst, "action": "kept"}
            if edge.token:
                record["resolution"] = "token_to_plain_dependence"
        record["blocking"] = edge.blocking
        if edge.blocking:
            record["blocking_rule"] = edge.blocking_rule
        records.append(record)
    return records


def _kind_counts(dropped_nodes) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in dropped_nodes.values():
        counts[node.op_kind] = counts.get(node.op_kind, 0) + 1
    return dict(sorted(counts.items()))


def dumps(artifact: dict) -> str:
    return json.dumps(artifact, sort_keys=True, indent=1) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="paper_joint_solver.curate_ddg")
    ap.add_argument("ddg", help="raw ddg.json dump")
    ap.add_argument(
        "--baseline-graph",
        required=True,
        help="baseline schedule graph; the curation criterion for G membership",
    )
    ap.add_argument("-o", "--out", required=True, help="curated_ddg.json")
    ap.add_argument(
        "--manifest-out", required=True, help="curation_manifest.json"
    )
    ap.add_argument("--loop-index", type=int, default=0)
    ap.add_argument("--no-strict-baseline", action="store_true")
    args = ap.parse_args(argv)

    curated, manifest = curate(
        args.ddg,
        args.baseline_graph,
        loop_index=args.loop_index,
        strict_baseline=not args.no_strict_baseline,
    )
    Path(args.out).write_text(dumps(curated))
    Path(args.manifest_out).write_text(dumps(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
