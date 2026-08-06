import json
import sys
from pathlib import Path
from types import SimpleNamespace


PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))

import desym_check


def _write_solution(path, *, minimality="not_proven"):
    path.write_text(json.dumps({"stats": {"group_minimality": minimality}}))


def test_main_returns_nonzero_on_mismatched_verdict(tmp_path, monkeypatch):
    solution = tmp_path / "solution.json"
    _write_solution(solution)
    plan = SimpleNamespace(ii=7, length=19, group_widths={0: 1, 1: 1})
    monkeypatch.setattr(
        desym_check,
        "load_schedule_context",
        lambda *args: (object(), plan),
    )
    monkeypatch.setattr(
        desym_check,
        "solve_joint",
        lambda *args, **kwargs: (None, "unknown"),
    )

    assert (
        desym_check.main(
            [
                "ddg.json",
                str(solution),
                "1",
                "--baseline-graph",
                "schedule_graph.json",
            ]
        )
        == 1
    )


def test_main_checks_proven_group_boundary(tmp_path, monkeypatch):
    solution = tmp_path / "solution.json"
    _write_solution(solution, minimality="proven")
    plan = SimpleNamespace(ii=7, length=19, group_widths={0: 1, 1: 1})
    verdicts = iter(("sat", "unsat"))
    seen_groups = []
    monkeypatch.setattr(
        desym_check,
        "load_schedule_context",
        lambda *args: (object(), plan),
    )

    def fake_solve(*args, **kwargs):
        seen_groups.append(kwargs["max_groups"])
        return object(), next(verdicts)

    monkeypatch.setattr(desym_check, "solve_joint", fake_solve)

    assert (
        desym_check.main(
            [
                "ddg.json",
                str(solution),
                "1",
                "--baseline-graph",
                "schedule_graph.json",
            ]
        )
        == 0
    )
    assert seen_groups == [2, 1]
