import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))

import refit_check


def _write_template(path, warp, *, ii=1, length=2):
    path.write_text(
        json.dumps(
            {
                "ii": ii,
                "length": length,
                "cycles": {str(node): 0 for node in warp},
                "warp": {str(node): group for node, group in warp.items()},
            }
        )
    )


def test_partition_constraints_ignore_historical_schedule_point(tmp_path):
    template = tmp_path / "template.json"
    _write_template(template, {0: 9, 1: 9, 2: 20}, ii=3, length=7)

    constraints = refit_check.load_partition_constraints(template, {0, 1, 2})

    assert constraints.groups == ((0, 1), (2,))
    assert constraints.colocate == [[0, 1]]
    assert constraints.separate == [(0, 2)]


def test_partition_constraints_require_exact_node_coverage(tmp_path):
    template = tmp_path / "template.json"
    _write_template(template, {0: 0, 2: 1})

    with pytest.raises(ValueError) as error:
        refit_check.load_partition_constraints(template, {0, 1})
    message = str(error.value)
    assert "missing=[1]" in message
    assert "extra=[2]" in message


def test_fa4_refit_uses_free_v8_optimum_and_exact_groups(tmp_path):
    template = tmp_path / "template.json"
    _write_template(template, {0: 4, 1: 4, 2: 7}, ii=3, length=7)
    prob = SimpleNamespace(nodes={0: object(), 1: object(), 2: object()})
    plan = SimpleNamespace(ii=71, length=153)
    call = {}

    def fake_solver(problem, ii, length, **kwargs):
        call.update(problem=problem, ii=ii, length=length, **kwargs)
        return object(), "sat"

    check = refit_check.verify_fa4_refit(
        prob,
        plan,
        template,
        timeout_s=42,
        solver=fake_solver,
    )

    assert check.passed
    assert call["problem"] is prob
    assert (call["ii"], call["length"]) == (71, 153)
    assert call["max_groups"] == call["exact_groups"] == 2
    assert call["colocate"] == [[0, 1]]
    assert call["separate"] == [(0, 2)]
    assert call["timeout_s"] == 42


def test_fa4_refit_reports_non_sat_verdict_as_failure(tmp_path):
    template = tmp_path / "template.json"
    _write_template(template, {0: 0})
    prob = SimpleNamespace(nodes={0: object()})
    plan = SimpleNamespace(ii=11, length=29)

    check = refit_check.verify_fa4_refit(
        prob,
        plan,
        template,
        timeout_s=1,
        solver=lambda *args, **kwargs: (None, "timeout"),
    )

    assert not check.passed
    assert "verdict=timeout" in check.detail
    assert "II=11, L=29" in check.detail


def test_run_checks_uses_configurable_solution_directory(tmp_path):
    solution_dir = tmp_path / "custom-solutions"
    solution_dir.mkdir()
    template = tmp_path / "template.json"
    _write_template(template, {0: 0})
    seen = []

    def fake_context_loader(solution_path, ddg, baseline_graph):
        seen.append((solution_path, ddg, baseline_graph))
        plan = SimpleNamespace(
            ii=13,
            length=31,
            warp={0: 0},
            cycles={0: 0},
            group_widths={0: 1},
        )
        return SimpleNamespace(nodes={0: object()}), plan

    def fake_classifier(prob, warp, cycles):
        return {
            "fa4_like": False,
            "bwd_2wg_pingpong": True,
            "bwd_3wg_fa4": True,
        }

    checks = refit_check.run_checks(
        solution_dir,
        template,
        timeout_s=5,
        context_loader=fake_context_loader,
        classifier=fake_classifier,
        solver=lambda *args, **kwargs: (object(), "sat"),
    )

    assert all(check.passed for check in checks)
    observation = next(
        check for check in checks if check.name.endswith("fa4_like_observation")
    )
    assert "observed False" in observation.detail
    assert [item[0] for item in seen] == [
        solution_dir / "fwd_subtiled_v8.json",
        solution_dir / "fwd_v8.json",
        solution_dir / "bwd_v8.json",
        solution_dir / "bwd_lr4096_v8.json",
    ]


def test_main_returns_nonzero_and_writes_explicit_failure(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        refit_check,
        "run_checks",
        lambda *args, **kwargs: [
            refit_check.Check("probe", False, "verdict=unsat")
        ],
    )
    report = tmp_path / "report.json"

    rc = refit_check.main(["--json-output", str(report)])

    assert rc == 1
    assert "[FAIL] probe: verdict=unsat" in capsys.readouterr().out
    assert json.loads(report.read_text())["passed"] is False
