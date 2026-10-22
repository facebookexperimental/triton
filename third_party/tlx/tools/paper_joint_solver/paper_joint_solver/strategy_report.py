"""Classify paper-visible warp-specialization strategies.

The forward classifier checks the FA4 shape reported for Blackwell forward
FMHA (paper Fig. 9).  The backward classifiers distinguish the two
sub-tiled strategies reported in section 6.2.3: two compute groups that
ping-pong exponential and reduction-staging work, or two exponential groups
plus a third reduction-staging group under the reduced-register budget.

Every field here is an observation.  Each structural criterion is one
operationalization of the paper's prose, and the report states its own
criteria so a reader can see what was measured; nothing in this file is a
gate, and a False verdict is a recorded result, not a failure.

Usage: python -m paper_joint_solver.strategy_report <curated_ddg.json>
       <solution.json> [--curation-manifest <curation_manifest.json>]
       [--output <strategy.json>] [--fa4-template <legacy-solution.json>]
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .ddg import Node, Problem
from .joint_smt import solve_joint_audit
from .schedule_plan import load_schedule_context


STRATEGY_SCHEMA = "paper-joint-strategy-v2"
TMEM_SOURCE_SEARCH_MAX_DEPTH = 4


def warp_set_labels(warp_sets: dict[int, tuple[int, ...]]) -> dict[int, int]:
    """Label the distinct physical warp sets of a paper solution.

    The paper has no group concept -- an operation owns a set of physical
    warps -- so the classifier's "group" is the warp set itself: two nodes
    share a group exactly when they occupy the same warps.
    """
    order = sorted({tuple(warps) for warps in warp_sets.values()})
    index = {warps: label for label, warps in enumerate(order)}
    return {
        node_id: index[tuple(warps)] for node_id, warps in warp_sets.items()
    }


@dataclass(frozen=True)
class Fa4TemplatePartition:
    groups: tuple[tuple[int, ...], ...]
    colocate: list[list[int]]
    separate: list[tuple[int, int]]
    sha256: str
    projected_nodes: tuple[int, ...]


def curation_dropped_nodes(manifest: dict | None) -> frozenset[int]:
    """Node ids the curation stage recorded as dropped.

    The manifest node dispositions are curation's own record of what it
    removed; they are the only surviving link between a raw node set and a
    curated one.  Without a manifest the answer is "nothing was dropped",
    which leaves the caller's coverage rule exactly as strict as before.
    """
    if manifest is None:
        return frozenset()
    return frozenset(
        record["id"]
        for loop in manifest.get("loops", [])
        for record in loop.get("nodes", [])
        if record.get("action") == "dropped"
    )


def load_fa4_template_partition(
    template_path: str | Path,
    expected_nodes: set[int],
    dropped_nodes: frozenset[int] = frozenset(),
) -> Fa4TemplatePartition:
    """Load a complete, label-independent partition from a legacy solution.

    The template is a frozen artifact keyed on the *raw* node ids, so it still
    carries the emitter-infra nodes curation deletes.  It is projected onto the
    curated node set through the curation manifest's own disposition records --
    never by intersecting with whatever the graph happens to contain -- so a
    template that disagrees with the curated graph for any reason curation did
    not record still fails the coverage check below.
    """
    template_path = Path(template_path)
    payload = json.loads(template_path.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("warp"), dict):
        raise ValueError(f"{template_path} must contain a warp object")

    warp: dict[int, int] = {}
    try:
        for raw_node, raw_group in payload["warp"].items():
            if isinstance(raw_node, bool) or isinstance(raw_group, bool):
                raise TypeError
            node, group = int(raw_node), int(raw_group)
            if node in warp:
                raise ValueError(f"duplicate normalized node id {node}")
            warp[node] = group
    except (TypeError, ValueError) as error:
        raise ValueError(f"{template_path} warp must be an integer map") from error

    projected = tuple(sorted(node for node in warp if node in dropped_nodes))
    for node in projected:
        del warp[node]

    actual_nodes = set(warp)
    if actual_nodes != expected_nodes:
        raise ValueError(
            f"{template_path} partition does not cover the DDG exactly: "
            f"missing={sorted(expected_nodes - actual_nodes)}, "
            f"extra={sorted(actual_nodes - expected_nodes)}"
        )
    by_label: dict[int, list[int]] = defaultdict(list)
    for node, group in warp.items():
        by_label[group].append(node)
    groups = tuple(tuple(sorted(by_label[label])) for label in sorted(by_label))
    if not groups:
        raise ValueError(f"{template_path} partition has no groups")
    representatives = [group[0] for group in groups]
    return Fa4TemplatePartition(
        groups=groups,
        colocate=[list(group) for group in groups if len(group) > 1],
        separate=list(itertools.combinations(representatives, 2)),
        sha256=hashlib.sha256(template_path.read_bytes()).hexdigest(),
        projected_nodes=projected,
    )


def refit_fa4_template(
    prob: Problem,
    template_path: str | Path,
    ii: int,
    length: int,
    inputs: dict[str, object],
    solver: Callable[..., tuple[object | None, str]] | None = None,
    *,
    curation_manifest: dict | None = None,
) -> tuple[dict[str, object], bool]:
    """Refit an exact FA4 partition at the free solution's schedule point.

    An AUDIT instrument: it runs one extra partition-pinned solve that the
    paper never performs, through the audit entry, under the single paper
    model and the same experiment inputs as the solution being audited.
    """
    partition = load_fa4_template_partition(
        template_path,
        set(prob.nodes),
        curation_dropped_nodes(curation_manifest),
    )
    solve = solve_joint_audit if solver is None else solver
    solution, verdict = solve(
        prob,
        ii,
        length,
        exact_warp_sets=len(partition.groups),
        allow_cross_warp=inputs["allow_cross_warp"],
        num_warps_override=inputs["num_warps_override"],
        colocate=partition.colocate,
        separate=partition.separate,
    )
    if verdict not in {"sat", "unsat"}:
        raise RuntimeError(f"unexpected FA4 template refit verdict {verdict!r}")
    if verdict == "sat" and solution is None:
        raise RuntimeError("FA4 template refit returned SAT without a solution")
    metadata: dict[str, object] = {
        "verdict": verdict,
        "ii": ii,
        "length": length,
        "audit_instrument": True,
        "experiment_inputs": dict(inputs),
        "template_sha256": partition.sha256,
        "template_groups": len(partition.groups),
        "template_projected_nodes": list(partition.projected_nodes),
    }
    return metadata, verdict == "sat"


def _group_ids(
    groups: dict[int, list[int]],
    prob: Problem,
    predicate: Callable[[Node], bool],
) -> set[int]:
    return {
        group
        for group, node_ids in groups.items()
        if any(predicate(prob.nodes[node_id]) for node_id in node_ids)
    }


def _group_counts(
    node_ids: set[int], warp: dict[int, int]
) -> dict[int, int]:
    return dict(sorted(Counter(warp[node_id] for node_id in node_ids).items()))


def _reduction_staging_nodes(prob: Problem, manifest: dict | None) -> set[int] | None:
    """Producers feeding a reduction store.

    Curation drops the `descriptor_reduce` nodes (they are emitter infra), so
    this observation is reconstructed from the curation manifest's removed
    edges. Without a manifest the observation is simply unavailable -- it is
    an observation, not a gate.
    """
    if manifest is None:
        return None
    dropped_reduces = set()
    for loop in manifest.get("loops", []):
        for record in loop.get("nodes", []):
            if record.get("action") == "dropped" and "descriptor_reduce" in (
                record.get("op_kind") or ""
            ):
                dropped_reduces.add(record["id"])
    staging = set()
    for loop in manifest.get("loops", []):
        for record in loop.get("edges", []):
            if (
                record.get("action") == "removed"
                and record.get("dst") in dropped_reduces
                and prob.regs.get(record.get("src"), 0) > 1
            ):
                staging.add(record["src"])
    return staging


def _exp_tmem_sources(
    prob: Problem, exp_nodes: set[int]
) -> dict[int, set[int]]:
    predecessors: dict[int, set[int]] = defaultdict(set)
    for edge in prob.edges:
        predecessors[edge.dst].add(edge.src)

    sources_by_exp: dict[int, set[int]] = {}
    for exp_node in exp_nodes:
        sources: set[int] = set()
        visited = {exp_node}
        frontier = {exp_node}
        for _ in range(TMEM_SOURCE_SEARCH_MAX_DEPTH):
            next_frontier: set[int] = set()
            for node_id in frontier:
                for predecessor in predecessors[node_id]:
                    if predecessor in visited:
                        continue
                    visited.add(predecessor)
                    if "tmem_load" in prob.nodes[predecessor].op_kind:
                        sources.add(predecessor)
                    else:
                        next_frontier.add(predecessor)
            frontier = next_frontier
            if not frontier:
                break
        sources_by_exp[exp_node] = sources
    return sources_by_exp


def _classify_backward(
    prob: Problem,
    groups: dict[int, list[int]],
    warp: dict[int, int],
    mma_groups: set[int],
    curation_manifest: dict | None = None,
) -> dict[str, object]:
    vl_groups = {
        warp[node_id]
        for node_id in prob.variable_latency
        if node_id in warp
    }
    # Curated G has no infra nodes, so every group is active.
    active_groups = set(groups)
    compute_groups = active_groups - vl_groups - mma_groups
    exp_nodes = {
        node_id
        for node_id, node in prob.nodes.items()
        if "exp2" in node.op_kind
    }
    staging_nodes = _reduction_staging_nodes(prob, curation_manifest)
    exp_counts = _group_counts(exp_nodes, warp)
    staging_counts = (
        None if staging_nodes is None
        else _group_counts(staging_nodes, warp)
    )
    exp_groups = set(exp_counts)
    staging_groups = set(staging_counts or {})
    exp_tmem_sources = _exp_tmem_sources(prob, exp_nodes)
    exp_tmem_sources_in_group = {
        exp_node: sorted(
            source
            for source in exp_tmem_sources[exp_node]
            if warp[source] == warp[exp_node]
        )
        for exp_node in sorted(exp_nodes)
    }
    exp_reads_tmem_in_group = bool(exp_nodes) and all(
        bool(sources)
        and len(exp_tmem_sources_in_group[exp_node]) == len(sources)
        for exp_node, sources in exp_tmem_sources.items()
    )
    two_exp_groups = (
        len(exp_nodes) == 2
        and len(exp_groups) == 2
        and all(count == 1 for count in exp_counts.values())
    )
    return {
        "vl_groups": sorted(vl_groups),
        "compute_groups": sorted(compute_groups),
        "exp_group_counts": exp_counts,
        "exp_tmem_source_search_max_depth": TMEM_SOURCE_SEARCH_MAX_DEPTH,
        "exp_tmem_sources": {
            exp_node: sorted(sources)
            for exp_node, sources in sorted(exp_tmem_sources.items())
        },
        "exp_tmem_sources_in_group": exp_tmem_sources_in_group,
        "exp_reads_tmem_in_group": exp_reads_tmem_in_group,
        "reduction_staging_nodes": (
            None if staging_nodes is None else sorted(staging_nodes)
        ),
        "reduction_staging_group_counts": staging_counts,
        # Both verdicts rest on the staging observation, which only the
        # curation manifest can supply.  Without it the question is
        # undecidable, not answered in the negative.
        "bwd_2wg_pingpong": None if staging_nodes is None else bool(
            len(compute_groups) == 2
            and two_exp_groups
            and exp_groups == compute_groups
            and staging_groups == compute_groups
        ),
        "bwd_3wg_fa4": None if staging_nodes is None else bool(
            len(compute_groups) == 3
            and two_exp_groups
            and exp_groups < compute_groups
            and staging_groups == compute_groups - exp_groups
            and exp_reads_tmem_in_group
        ),
    }


def _criteria(curation_manifest_sha256: str | None) -> dict[str, object]:
    """State the report's own reading of the paper's prose."""
    return {
        "softmax_marker": "op_kind contains 'exp2' (proxy for the paper's "
                          "'softmax calculations', Fig 9 / p.10)",
        "tmem_source_search_max_depth": TMEM_SOURCE_SEARCH_MAX_DEPTH,
        "tmem_source_rule": "backward BFS over ALL curated edges (E is "
                            "homogeneous; token edges desugared at curation, "
                            "curated-ddg-0.2 section 3), stop at first "
                            "tmem_load; ALL sources must share the exp node's "
                            "group (operationalizes p.12 'two groups read "
                            "accumulators from Tensor Memory')",
        "staging_rule": "curated src (regs>1) of a curation-removed edge into "
                        "a curation-dropped descriptor_reduce node, read from "
                        "curation_manifest (curated-ddg-0.2 section 4-D1); "
                        "operationalizes p.12 'stages accumulators in shared "
                        "memory for the outgoing atomic reduction'; "
                        "unavailable without a curation manifest, in which "
                        "case bwd_2wg_pingpong and bwd_3wg_fa4 are null "
                        "rather than false",
        "compute_group_rule": "active_groups - vl_groups - mma_groups "
                              "(reading of p.12 'two/three groups of warps'); "
                              "curated G has no infra nodes, so every group "
                              "is active",
        "rescale_rule": "residual tmem_load groups minus VL/MMA/exp groups "
                        "(operationalizes p.10 'accumulator rescaling ... "
                        "third group')",
        "tma_isolated_rule": "TMA groups admit occupancy==0 nodes (allocs "
                             "adjacent to TMA loads in Fig 9; node.occupancy "
                             "is a curated observational field, "
                             "curated-ddg-0.2 section 2)",
        "two_exp_rule": "exactly 2 exp2 nodes, one per group "
                        "(sub-tiling realization assumption)",
        "group_rule": "a group is one distinct physical warp set; the paper "
                      "assigns every operation a set of warps and has no "
                      "group label",
        "curation_manifest": curation_manifest_sha256 or "absent",
    }


def classify(
    prob: Problem,
    warp: dict[int, int],
    cycles: dict[int, int] | None = None,
    curation_manifest: dict | None = None,
    curation_manifest_sha256: str | None = None,
) -> dict[str, object]:
    groups: dict[int, list[int]] = defaultdict(list)
    for node_id, group in warp.items():
        groups[group].append(node_id)

    active_groups = set(groups)
    vl_groups = {
        warp[node_id]
        for node_id in prob.variable_latency
        if node_id in warp
    }
    tma_groups = _group_ids(
        groups, prob, lambda node: node.pipeline == "TMA" and "load" in node.op_kind
    )
    mma_groups = _group_ids(groups, prob, lambda node: "mma" in node.op_kind)
    exp_nodes = {
        node_id
        for node_id, node in prob.nodes.items()
        if "exp2" in node.op_kind
    }
    exp_groups = {warp[node_id] for node_id in exp_nodes}
    tmem_groups = _group_ids(
        groups, prob, lambda node: "tmem_load" in node.op_kind
    )
    rescale_groups = tmem_groups - vl_groups - mma_groups - exp_groups
    fa4_role_groups = {
        "variable_latency": vl_groups,
        "tensor_core": mma_groups,
        "softmax": exp_groups,
        "rescale": rescale_groups,
    }
    all_role_groups = set().union(*fa4_role_groups.values())
    roles_disjoint = sum(map(len, fa4_role_groups.values())) == len(all_role_groups)
    roles_cover_active = all_role_groups == active_groups

    report: dict[str, object] = {
        "num_groups_used": len(groups),
        "num_active_groups": len(active_groups),
        "active_groups": sorted(active_groups),
        "group_sizes": {
            group: len(node_ids) for group, node_ids in sorted(groups.items())
        },
        "group_ops": {
            group: dict(
                sorted(
                    Counter(
                        prob.nodes[node_id].op_kind.split(".")[-1]
                        for node_id in node_ids
                    ).items()
                )
            )
            for group, node_ids in sorted(groups.items())
        },
        "tma_isolated": bool(tma_groups)
        and all(
            prob.nodes[node_id].pipeline == "TMA"
            or prob.nodes[node_id].occupancy == 0
            for group in tma_groups
            for node_id in groups[group]
        ),
        "tma_groups": sorted(tma_groups),
        "tma_single_group": len(tma_groups) == 1,
        "mma_groups": sorted(mma_groups),
        "mma_single_group": len(mma_groups) == 1,
        "exp_groups": sorted(exp_groups),
        "softmax_two_groups": len(exp_groups) == 2,
        "tmem_groups": sorted(tmem_groups),
        "rescale_groups": sorted(rescale_groups),
        "rescale_separate": len(rescale_groups) == 1,
        "fa4_role_groups": {
            role: sorted(role_groups)
            for role, role_groups in fa4_role_groups.items()
        },
        "fa4_roles_disjoint": roles_disjoint,
        "fa4_roles_cover_active": roles_cover_active,
    }
    if cycles and len(exp_nodes) >= 2:
        phase_offsets: dict[int, list[int]] = defaultdict(list)
        for node_id in exp_nodes:
            phase_offsets[warp[node_id]].append(cycles[node_id])
        report["softmax_phase_offsets"] = {
            str(group): sorted(offsets)
            for group, offsets in sorted(phase_offsets.items())
        }
    report["fa4_like"] = bool(
        report["tma_isolated"]
        and tma_groups == vl_groups
        and len(vl_groups) == 1
        and len(mma_groups) == 1
        and len(exp_groups) == 2
        and len(rescale_groups) == 1
        and len(active_groups) == 5
        and roles_disjoint
        and roles_cover_active
    )
    report.update(
        _classify_backward(prob, groups, warp, mma_groups, curation_manifest)
    )
    report["criteria"] = _criteria(curation_manifest_sha256)
    report["curation_manifest_sha256"] = curation_manifest_sha256
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Classify a joint solution's warp-specialization strategy"
    )
    parser.add_argument("curated_ddg")
    parser.add_argument("solution")
    parser.add_argument(
        "--curation-manifest",
        type=Path,
        help="curation_manifest.json; source of the dropped-node observations",
    )
    parser.add_argument(
        "--fa4-template",
        type=Path,
        help="AUDIT instrument for the paper's 'exactly FA4' claim; runs one "
        "extra partition-pinned SMT solve absent from the paper",
    )
    parser.add_argument("-o", "--output", help="write the JSON report to this path")
    args = parser.parse_args(argv)

    prob, plan = load_schedule_context(args.solution, args.curated_ddg)
    solution_path = Path(args.solution)
    solution = json.loads(solution_path.read_text())
    curation_manifest = curation_manifest_sha256 = None
    if args.curation_manifest:
        manifest_bytes = args.curation_manifest.read_bytes()
        curation_manifest = json.loads(manifest_bytes)
        curation_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    report = classify(
        prob,
        warp_set_labels(plan.warp_sets),
        plan.cycles,
        curation_manifest,
        curation_manifest_sha256,
    )
    report.update(
        {
            "schema_version": STRATEGY_SCHEMA,
            "solution_sha256": hashlib.sha256(solution_path.read_bytes()).hexdigest(),
            "ii": plan.ii,
            "length": plan.length,
            "wall_s": solution.get("wall_s"),
        }
    )
    if args.fa4_template is not None:
        refit, optimal_set_member = refit_fa4_template(
            prob,
            args.fa4_template,
            plan.ii,
            plan.length,
            plan.experiment_inputs,
            curation_manifest=curation_manifest,
        )
        report["fa4_template_refit"] = refit
        report["fa4_optimal_set_member"] = optimal_set_member
    rendered = json.dumps(report, indent=1) + "\n"
    if args.output:
        Path(args.output).write_text(rendered)
        print(f"[strategy_report] wrote {args.output}")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
