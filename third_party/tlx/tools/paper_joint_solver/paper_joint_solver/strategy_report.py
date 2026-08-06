"""Classify paper-visible warp-specialization strategies.

The forward classifier checks the FA4 shape reported for Blackwell forward
FMHA (paper Fig. 9).  The backward classifiers distinguish the two
sub-tiled strategies reported in section 6.2.3: two compute groups that
ping-pong exponential and reduction-staging work, or two exponential groups
plus a third reduction-staging group under the reduced-register budget.

Usage: python -m paper_joint_solver.strategy_report <ddg.json> <solution.json>
       --baseline-graph <schedule_graph.json> [--output <strategy.json>]
       [--fa4-template <legacy-solution.json>]
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
from .joint_smt import solve_joint
from .schedule_plan import load_schedule_context


STRATEGY_SCHEMA = "paper-joint-strategy-v1"
TMEM_SOURCE_SEARCH_MAX_DEPTH = 4


@dataclass(frozen=True)
class Fa4TemplatePartition:
    groups: tuple[tuple[int, ...], ...]
    colocate: list[list[int]]
    separate: list[tuple[int, int]]
    sha256: str


def load_fa4_template_partition(
    template_path: str | Path,
    expected_nodes: set[int],
) -> Fa4TemplatePartition:
    """Load a complete, label-independent partition from a legacy solution."""
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
    )


def refit_fa4_template(
    prob: Problem,
    template_path: str | Path,
    ii: int,
    length: int,
    timeout_s: float,
    solver: Callable[..., tuple[object | None, str]] | None = None,
) -> tuple[dict[str, object], bool]:
    """Refit an exact FA4 partition at the free solution's schedule point."""
    if timeout_s <= 0:
        raise ValueError("FA4 template refit timeout must be positive")
    partition = load_fa4_template_partition(template_path, set(prob.nodes))
    solve = solve_joint if solver is None else solver
    solution, verdict = solve(
        prob,
        ii,
        length,
        max_groups=len(partition.groups),
        exact_groups=len(partition.groups),
        allow_cross_warp=True,
        timeout_s=timeout_s,
        colocate=partition.colocate,
        separate=partition.separate,
    )
    if verdict not in {"sat", "unsat", "unknown"}:
        raise RuntimeError(f"unexpected FA4 template refit verdict {verdict!r}")
    if verdict == "sat" and solution is None:
        raise RuntimeError("FA4 template refit returned SAT without a solution")
    metadata: dict[str, object] = {
        "verdict": verdict,
        "ii": ii,
        "length": length,
        "timeout_s": timeout_s,
        "template_sha256": partition.sha256,
        "template_groups": len(partition.groups),
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


def _reduction_staging_nodes(prob: Problem) -> set[int]:
    descriptor_reduces = {
        node_id
        for node_id, node in prob.nodes.items()
        if "descriptor_reduce" in node.op_kind
    }
    return {
        edge.src
        for edge in prob.edges
        if edge.dst in descriptor_reduces
        and not edge.signal_only
        and prob.edge_has_ws_semantics(edge)
        and prob.regs.get(edge.src, 0) > 1
    }


def _exp_tmem_sources(
    prob: Problem, exp_nodes: set[int]
) -> dict[int, set[int]]:
    predecessors: dict[int, set[int]] = defaultdict(set)
    for edge in prob.edges:
        if edge.signal_only or not prob.edge_has_ws_semantics(edge):
            continue
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
) -> dict[str, object]:
    vl_groups = {
        warp[node_id]
        for node_id in prob.variable_latency
        if node_id in warp
    }
    active_groups = {
        group
        for group, node_ids in groups.items()
        if any(node_id not in prob.emitter_infra for node_id in node_ids)
    }
    compute_groups = active_groups - vl_groups - mma_groups
    exp_nodes = {
        node_id
        for node_id, node in prob.nodes.items()
        if "exp2" in node.op_kind
    }
    staging_nodes = _reduction_staging_nodes(prob)
    exp_counts = _group_counts(exp_nodes, warp)
    staging_counts = _group_counts(staging_nodes, warp)
    exp_groups = set(exp_counts)
    staging_groups = set(staging_counts)
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
        "reduction_staging_nodes": sorted(staging_nodes),
        "reduction_staging_group_counts": staging_counts,
        "bwd_2wg_pingpong": bool(
            len(compute_groups) == 2
            and two_exp_groups
            and exp_groups == compute_groups
            and staging_groups == compute_groups
        ),
        "bwd_3wg_fa4": bool(
            len(compute_groups) == 3
            and two_exp_groups
            and exp_groups < compute_groups
            and staging_groups == compute_groups - exp_groups
            and exp_reads_tmem_in_group
        ),
    }


def classify(
    prob: Problem,
    warp: dict[int, int],
    cycles: dict[int, int] | None = None,
) -> dict[str, object]:
    groups: dict[int, list[int]] = defaultdict(list)
    for node_id, group in warp.items():
        groups[group].append(node_id)

    active_groups = {
        group
        for group, node_ids in groups.items()
        if any(node_id not in prob.emitter_infra for node_id in node_ids)
    }
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
    report.update(_classify_backward(prob, groups, warp, mma_groups))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Classify a joint solution's warp-specialization strategy"
    )
    parser.add_argument("ddg")
    parser.add_argument("solution")
    parser.add_argument(
        "--baseline-graph",
        required=True,
        help="schedule_graph.json carrying emitter-infrastructure ownership",
    )
    parser.add_argument(
        "--fa4-template",
        type=Path,
        help="legacy solution whose complete warp map defines the FA4 partition",
    )
    parser.add_argument(
        "--fa4-refit-timeout-s",
        type=float,
        default=1800.0,
        help="SMT timeout for --fa4-template (default: %(default)s)",
    )
    parser.add_argument("-o", "--output", help="write the JSON report to this path")
    args = parser.parse_args(argv)
    if args.fa4_refit_timeout_s <= 0:
        parser.error("--fa4-refit-timeout-s must be positive")

    prob, plan = load_schedule_context(
        args.solution,
        args.ddg,
        args.baseline_graph,
    )
    solution_path = Path(args.solution)
    solution = json.loads(solution_path.read_text())
    report = classify(prob, plan.warp, plan.cycles)
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
            args.fa4_refit_timeout_s,
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
