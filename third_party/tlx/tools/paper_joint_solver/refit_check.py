"""Verify the paper-visible claims carried by the lit1 solution artifacts.

This is an artifact verifier, not another search driver.  It validates each
solution against its recorded curated DDG, machine, and normalization
provenance, records the backward strategy observations, and performs one
independent refit: the recorded FA4 partition is constrained at the free
optimum's ``(II, L)``.  The free solution's FA4 shape is recorded only as an
observation because the paper defines no tie-break within the optimal set.
The legacy template contributes only its partition; its historical cycles and
``(II, L)`` are deliberately ignored.

The refit is an AUDIT instrument: it goes through ``solve_joint_audit``, under
the single paper model and the audited solution's own experiment inputs.  It
is uncapped -- bounding a long solve is the harness watchdog's job.

Usage::

    python refit_check.py --solution-dir solutions
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from paper_joint_solver.joint_smt import solve_joint_audit
from paper_joint_solver.schedule_plan import load_schedule_context
from paper_joint_solver.strategy_report import (
    classify,
    curation_dropped_nodes,
    warp_set_labels,
)


ROOT = Path(__file__).resolve().parent
FA4_REFIT_STEM = "fwd_subtiled_lit1"


@dataclass(frozen=True)
class Case:
    stem: str
    required_strategy: str | None
    observed_strategy: str | None = None

    def curated_ddg(self, solution_dir: Path) -> Path:
        """The curated artifact the run wrote beside the solution."""
        return solution_dir / f"{self.stem}.curated_ddg.json"

    def curation_manifest(self, solution_dir: Path) -> Path:
        """The manifest recording what curation dropped from the raw graph."""
        return solution_dir / f"{self.stem}.curation_manifest.json"


CASES = (
    Case(FA4_REFIT_STEM, None, "fa4_like"),
    Case("fwd_lit1", None),
    Case("bwd_lit1", "bwd_2wg_pingpong"),
    Case("bwd_lr4096_lit1", "bwd_3wg_fa4"),
)


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str

    def as_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class PartitionConstraints:
    groups: tuple[tuple[int, ...], ...]
    colocate: list[list[int]]
    separate: list[tuple[int, int]]


def load_partition_constraints(
    template_path: str | Path,
    expected_nodes: set[int],
    dropped_nodes: frozenset[int] = frozenset(),
) -> PartitionConstraints:
    """Translate a complete warp map into label-independent constraints.

    The template still carries the raw node ids, so it is projected onto the
    curated node set through the curation manifest's own disposition records
    before the coverage check -- see ``strategy_report``.
    """
    template_path = Path(template_path)
    payload = json.loads(template_path.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("warp"), dict):
        raise ValueError(f"{template_path} must contain a warp object")
    try:
        warp = {int(node): int(group) for node, group in payload["warp"].items()}
    except (TypeError, ValueError) as error:
        raise ValueError(f"{template_path} warp must be an integer map") from error
    for node in [node for node in warp if node in dropped_nodes]:
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
    groups = tuple(
        tuple(sorted(by_label[label])) for label in sorted(by_label)
    )
    if not groups:
        raise ValueError(f"{template_path} partition has no groups")
    representatives = [group[0] for group in groups]
    return PartitionConstraints(
        groups=groups,
        colocate=[list(group) for group in groups if len(group) > 1],
        separate=list(itertools.combinations(representatives, 2)),
    )


def verify_fa4_refit(
    prob,
    plan,
    template_path: str | Path,
    solver: Callable = solve_joint_audit,
    *,
    curation_manifest: dict | None = None,
) -> Check:
    """Refit the FA4 partition at the free solution's optimal point."""
    constraints = load_partition_constraints(
        template_path,
        set(prob.nodes),
        curation_dropped_nodes(curation_manifest),
    )
    inputs = plan.experiment_inputs
    solution, verdict = solver(
        prob,
        plan.ii,
        plan.length,
        exact_warp_sets=len(constraints.groups),
        allow_cross_warp=inputs["allow_cross_warp"],
        num_warps_override=inputs["num_warps_override"],
        colocate=constraints.colocate,
        separate=constraints.separate,
    )
    passed = verdict == "sat" and solution is not None
    return Check(
        f"{FA4_REFIT_STEM}.fa4_exact_refit",
        passed,
        f"free optimum (II={plan.ii}, L={plan.length}), "
        f"template_groups={len(constraints.groups)}, verdict={verdict}",
    )


def run_checks(
    solution_dir: str | Path,
    template_path: str | Path,
    *,
    context_loader: Callable = load_schedule_context,
    classifier: Callable = classify,
    solver: Callable = solve_joint_audit,
) -> list[Check]:
    solution_dir = Path(solution_dir)
    checks: list[Check] = []
    loaded: dict[str, tuple[object, object]] = {}

    for case in CASES:
        solution_path = solution_dir / f"{case.stem}.json"
        try:
            prob, plan = context_loader(
                solution_path,
                case.curated_ddg(solution_dir),
            )
        except Exception as error:
            checks.append(
                Check(
                    f"{case.stem}.artifact",
                    False,
                    f"{type(error).__name__}: {error}",
                )
            )
            continue

        loaded[case.stem] = (prob, plan)
        checks.append(
            Check(
                f"{case.stem}.artifact",
                True,
                f"validated II={plan.ii}, L={plan.length}, "
                f"warp_sets={len(set(plan.warp_sets.values()))}",
            )
        )
        if case.required_strategy is None and case.observed_strategy is None:
            continue
        try:
            report = classifier(
                prob, warp_set_labels(plan.warp_sets), plan.cycles
            )
            if case.required_strategy is not None:
                actual = report.get(case.required_strategy)
                checks.append(
                    Check(
                        f"{case.stem}.{case.required_strategy}",
                        actual is True,
                        f"expected true, got {actual!r}; "
                        f"II={plan.ii}, L={plan.length}",
                    )
                )
            if case.observed_strategy is not None:
                actual = report.get(case.observed_strategy)
                checks.append(
                    Check(
                        f"{case.stem}.{case.observed_strategy}_observation",
                        True,
                        f"observed {actual!r}; II={plan.ii}, L={plan.length}",
                    )
                )
        except Exception as error:
            strategy = case.required_strategy or case.observed_strategy
            checks.append(
                Check(
                    f"{case.stem}.{strategy}",
                    False,
                    f"{type(error).__name__}: {error}",
                )
            )

    fwd = loaded.get(FA4_REFIT_STEM)
    if fwd is not None:
        try:
            fa4_case = next(
                case for case in CASES if case.stem == FA4_REFIT_STEM
            )
            # No manifest means no recorded curation, so no projection -- the
            # coverage rule below stays exactly as strict as it was.
            manifest_path = fa4_case.curation_manifest(solution_dir)
            manifest = (
                json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.is_file()
                else None
            )
            checks.append(
                verify_fa4_refit(
                    *fwd,
                    template_path,
                    solver=solver,
                    curation_manifest=manifest,
                )
            )
        except Exception as error:
            checks.append(
                Check(
                    f"{FA4_REFIT_STEM}.fa4_exact_refit",
                    False,
                    f"{type(error).__name__}: {error}",
                )
            )
    else:
        checks.append(
            Check(
                f"{FA4_REFIT_STEM}.fa4_exact_refit",
                False,
                "skipped because the free artifact could not be loaded",
            )
        )
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify lit1 strategies and the FA4 exact-template refit"
    )
    parser.add_argument(
        "--solution-dir",
        type=Path,
        default=ROOT / "solutions",
        help="directory containing *_lit1.json artifacts (default: %(default)s)",
    )
    parser.add_argument(
        "--fa4-template",
        type=Path,
        default=ROOT / "subtiled_fa4exact_solution.json",
        help="JSON whose warp map defines the FA4 partition (default: %(default)s)",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="optional path for a machine-readable check report",
    )
    args = parser.parse_args(argv)

    checks = run_checks(args.solution_dir, args.fa4_template)
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {check.name}: {check.detail}", flush=True)
    passed = all(check.passed for check in checks)
    print(f"REFIT-CHECK {'PASS' if passed else 'FAIL'}", flush=True)

    if args.json_output:
        payload = {
            "passed": passed,
            "solution_dir": str(args.solution_dir),
            "fa4_template": str(args.fa4_template),
            "checks": [check.as_json() for check in checks],
        }
        args.json_output.write_text(json.dumps(payload, indent=1) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
