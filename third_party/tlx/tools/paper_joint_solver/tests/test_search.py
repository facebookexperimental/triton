import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_joint_solver.search import run_search


class Problem:
    def min_ii(self):
        return 1


def modulo_sat(prob, ii, max_seconds):
    return SimpleNamespace(ii=ii, length=1)


def test_unknown_joint_result_stops_the_optimal_search():
    calls = []

    def joint_unknown(prob, ii, length, **kwargs):
        calls.append((ii, length))
        return None, "unknown"

    result = run_search(
        Problem(),
        modulo_solver=modulo_sat,
        joint_solver=joint_unknown,
    )

    assert result.status == "unknown"
    assert result.solution is None
    assert calls == [(1, 1)]


def test_default_search_does_not_minimize_emitter_groups():
    calls = []
    solution = SimpleNamespace(warp={0: 0, 1: 1}, stats={})

    def joint_sat(prob, ii, length, **kwargs):
        calls.append(kwargs.get("max_groups"))
        return solution, "sat"

    result = run_search(
        Problem(),
        modulo_solver=modulo_sat,
        joint_solver=joint_sat,
    )

    assert result.status == "sat"
    assert result.solution is solution
    assert calls == [None]


def test_explicit_probe_cap_is_resource_limited():
    calls = []

    def joint_unsat(prob, ii, length, **kwargs):
        calls.append((ii, length))
        return None, "unsat"

    result = run_search(
        Problem(),
        max_ii_span=1,
        max_probes_per_ii=1,
        modulo_solver=modulo_sat,
        joint_solver=joint_unsat,
    )

    assert result.status == "resource_limited"
    assert result.solution is None
    assert calls == [(1, 1)]


def test_optional_group_minimization_preserves_unknown_status():
    calls = []
    solution = SimpleNamespace(
        ii=1,
        length=1,
        warp={0: 0, 1: 1, 2: 2, 3: 3},
        stats={},
    )

    def joint(prob, ii, length, **kwargs):
        groups = kwargs.get("max_groups")
        calls.append(groups)
        if groups is None:
            return solution, "sat"
        return None, "unknown"

    result = run_search(
        Problem(),
        minimize_groups=True,
        modulo_solver=modulo_sat,
        joint_solver=joint,
    )

    assert result.status == "sat"
    assert result.solution is solution
    assert solution.stats["group_minimality"] == "unknown"
    assert calls == [None, 3]


def test_full_group_lane_capability_is_forwarded():
    seen = []
    solution = SimpleNamespace(warp={0: 0}, stats={})

    def joint(prob, ii, length, **kwargs):
        seen.append((kwargs["prefix_lane_masks"],
                     kwargs["full_group_lane_masks"]))
        return solution, "sat"

    result = run_search(
        Problem(),
        prefix_lane_masks=True,
        full_group_lane_masks=True,
        modulo_solver=modulo_sat,
        joint_solver=joint,
    )

    assert result.status == "sat"
    assert seen == [(True, True)]


def test_cli_ir_handoff_does_not_add_emitter_constraints(tmp_path, monkeypatch):
    from paper_joint_solver import __main__ as cli
    from paper_joint_solver import pipelined_ir
    from paper_joint_solver.search import SearchResult

    problem = SimpleNamespace()
    solution = SimpleNamespace(
        ii=1,
        length=1,
        copies=1,
        horizon=1,
        cycles={},
        warp={},
        group_widths={},
        lane_masks={},
        stats={},
    )
    monkeypatch.setattr(cli, "load_problem", lambda *args, **kwargs: problem)
    search_kwargs = {}

    def fake_search(*args, **kwargs):
        search_kwargs.update(kwargs)
        return SearchResult(solution, [], 0.0, "sat")

    monkeypatch.setattr(cli, "run_search", fake_search)
    handoff_kwargs = {}

    def fake_handoff(**kwargs):
        handoff_kwargs.update(kwargs)
        return SimpleNamespace(
            ir_path=kwargs["ir_out"], manifest_path=kwargs["manifest_out"]
        )

    monkeypatch.setattr(pipelined_ir, "prepare_manual_cuda_handoff", fake_handoff)
    ddg = tmp_path / "ddg.json"
    baseline = tmp_path / "schedule_graph.json"
    out = tmp_path / "solution.json"
    ir_out = tmp_path / "pipelined_ir.json"
    manifest_out = tmp_path / "manual_cuda_handoff.json"
    ddg.write_text("{}")
    baseline.write_text("{}")

    assert cli.main(
        [
            str(ddg),
            "-o",
            str(out),
            "--baseline-graph",
            str(baseline),
            "--ir-out",
            str(ir_out),
            "--handoff-manifest-out",
            str(manifest_out),
        ]
    ) == 0
    assert "prefix_lane_masks" not in search_kwargs
    assert "full_group_lane_masks" not in search_kwargs
    assert handoff_kwargs["solution_path"] == str(out)
    assert handoff_kwargs["ir_out"] == str(ir_out)
    assert handoff_kwargs["manifest_out"] == str(manifest_out)
