import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_joint_solver.search import run_search


class Problem:
    def min_ii(self):
        return 1


def test_cli_requires_baseline_graph(tmp_path):
    from paper_joint_solver import __main__ as cli

    with pytest.raises(SystemExit) as error:
        cli.main([str(tmp_path / "ddg.json"), "-o", str(tmp_path / "out.json")])

    assert error.value.code == 2


def test_cli_rejects_diagnostic_model_handoff(tmp_path):
    from paper_joint_solver import __main__ as cli

    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                str(tmp_path / "ddg.json"),
                "-o",
                str(tmp_path / "solution.json"),
                "--baseline-graph",
                str(tmp_path / "schedule_graph.json"),
                "--paper-pure",
                "--ir-out",
                str(tmp_path / "ir.json"),
                "--handoff-manifest-out",
                str(tmp_path / "handoff.json"),
            ]
        )

    assert error.value.code == 2


@pytest.mark.parametrize(
    "diagnostic_args",
    (
        ("--ignore-fixed-overheads",),
        ("--minimize-groups",),
        ("--max-groups", "2"),
        ("--num-warps", "16"),
    ),
)
def test_cli_rejects_nonproduction_handoff(tmp_path, diagnostic_args):
    from paper_joint_solver import __main__ as cli

    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                str(tmp_path / "ddg.json"),
                "-o",
                str(tmp_path / "solution.json"),
                "--baseline-graph",
                str(tmp_path / "schedule_graph.json"),
                *diagnostic_args,
                "--ir-out",
                str(tmp_path / "ir.json"),
                "--handoff-manifest-out",
                str(tmp_path / "handoff.json"),
            ]
        )

    assert error.value.code == 2


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


def test_no_cross_warp_structural_conflict_is_globally_unsat():
    edge = SimpleNamespace(src=0, dst=1, index=7, signal_only=False)
    problem = SimpleNamespace(
        edges=[edge],
        variable_latency={0},
        regs={0: 1},
        edge_has_ws_semantics=lambda candidate: candidate is edge,
    )

    def should_not_run(*args, **kwargs):
        raise AssertionError("structural contradiction should bypass solvers")

    result = run_search(
        problem,
        allow_cross_warp=False,
        modulo_solver=should_not_run,
        joint_solver=should_not_run,
    )

    assert result.status == "unsat"
    assert result.solution is None
    assert result.attempts == [
        {
            "stage": "structural",
            "result": "unsat",
            "reason": "VARIABLELATENCY conflicts with no-cross-warp",
            "edge": 7,
        }
    ]


def test_no_cross_warp_reg_data_precheck_ignores_buffer_edge():
    edge = SimpleNamespace(src=0, dst=1, index=7, signal_only=False)
    solution = SimpleNamespace(warp={0: 0, 1: 1}, stats={})
    problem = SimpleNamespace(
        edges=[edge],
        variable_latency={0},
        regs={0: 0},
        edge_has_ws_semantics=lambda candidate: candidate is edge,
        min_ii=lambda: 1,
    )
    calls = []

    def joint_sat(*args, **kwargs):
        calls.append(kwargs["no_cross_warp_domain"])
        return solution, "sat"

    result = run_search(
        problem,
        allow_cross_warp=False,
        modulo_solver=modulo_sat,
        joint_solver=joint_sat,
    )

    assert result.status == "sat"
    assert result.solution is solution
    assert calls == ["reg-data"]


def test_no_cross_warp_all_ws_precheck_includes_buffer_edge():
    edge = SimpleNamespace(src=0, dst=1, index=7, signal_only=False)
    problem = SimpleNamespace(
        edges=[edge],
        variable_latency={0},
        regs={0: 0},
        edge_has_ws_semantics=lambda candidate: candidate is edge,
    )

    def should_not_run(*args, **kwargs):
        raise AssertionError("structural contradiction should bypass solvers")

    result = run_search(
        problem,
        allow_cross_warp=False,
        no_cross_warp_domain="all-ws",
        modulo_solver=should_not_run,
        joint_solver=should_not_run,
    )

    assert result.status == "unsat"
    assert result.solution is None
    assert result.attempts[0]["stage"] == "structural"


def test_disabling_structural_precheck_records_joint_attempt():
    edge = SimpleNamespace(src=0, dst=1, index=7, signal_only=False)
    solution = SimpleNamespace(warp={0: 0, 1: 1}, stats={})
    problem = SimpleNamespace(
        edges=[edge],
        variable_latency={0},
        regs={0: 1},
        edge_has_ws_semantics=lambda candidate: candidate is edge,
        min_ii=lambda: 1,
    )

    result = run_search(
        problem,
        allow_cross_warp=False,
        structural_precheck=False,
        modulo_solver=modulo_sat,
        joint_solver=lambda *args, **kwargs: (solution, "sat"),
    )

    assert result.status == "sat"
    assert result.solution is solution
    assert result.attempts[0]["stage"] == "joint"


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


def test_joint_semantics_options_are_forwarded():
    seen = []
    solution = SimpleNamespace(warp={0: 0}, stats={})

    def joint(prob, ii, length, **kwargs):
        seen.append(
            (
                kwargs["prefix_lane_masks"],
                kwargs["full_group_lane_masks"],
                kwargs["no_cross_warp_domain"],
                kwargs["spill_smem_footprint"],
                kwargs["reg_carried_same_lane"],
            )
        )
        return solution, "sat"

    result = run_search(
        Problem(),
        prefix_lane_masks=True,
        full_group_lane_masks=True,
        no_cross_warp_domain="all-ws",
        spill_smem_footprint=False,
        reg_carried_same_lane=False,
        modulo_solver=modulo_sat,
        joint_solver=joint,
    )

    assert result.status == "sat"
    assert seen == [(True, True, "all-ws", False, False)]


def test_cli_ir_handoff_does_not_add_emitter_constraints(tmp_path, monkeypatch):
    from paper_joint_solver import __main__ as cli
    from paper_joint_solver import pipelined_ir
    from paper_joint_solver.search import SearchResult

    problem = SimpleNamespace(
        normalization_stats=lambda: {
            "normalization_u": 300,
            "normalization_u_effective": 300,
            "normalization_f": 0,
            "normalization_zero_collapses": 0,
            "normalization_retries": 0,
        },
        normalization_cost_pairs=(),
        normalization_u_effective=300,
    )
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
    assert search_kwargs["prefix_lane_masks"] is False
    assert search_kwargs["full_group_lane_masks"] is False
    assert handoff_kwargs["solution_path"] == str(out)
    assert handoff_kwargs["ir_out"] == str(ir_out)
    assert handoff_kwargs["manifest_out"] == str(manifest_out)


def test_cli_forwards_paper_diagnostic_switches(tmp_path, monkeypatch):
    from paper_joint_solver import __main__ as cli
    from paper_joint_solver.search import SearchResult

    captured = {}
    problem = SimpleNamespace(
        normalization_stats=lambda: {
            "normalization_u": 300,
            "normalization_u_effective": 300,
            "normalization_f": 0,
            "normalization_zero_collapses": 0,
            "normalization_retries": 0,
        },
        normalization_cost_pairs=(),
        normalization_u_effective=300,
    )

    def fake_load(*args, **kwargs):
        captured["machine"] = kwargs["machine"]
        return problem

    def fake_search(*args, **kwargs):
        captured["search"] = kwargs
        return SearchResult(None, [], 0.0, "resource_limited")

    monkeypatch.setattr(cli, "load_problem", fake_load)
    monkeypatch.setattr(cli, "run_search", fake_search)
    ddg = tmp_path / "ddg.json"
    baseline = tmp_path / "schedule_graph.json"
    out = tmp_path / "solution.json"
    ddg.write_text("{}")
    baseline.write_text("{}")

    assert cli.main(
        [
            str(ddg),
            "-o",
            str(out),
            "--baseline-graph",
            str(baseline),
            "--no-cross-warp",
            "--no-cross-warp-domain",
            "all-ws",
            "--no-structural-precheck",
            "--paper-pure",
            "--warp-fixed-overhead",
            "4",
            "--reg-fixed-overhead",
            "1",
            "--smem-fixed-overhead",
            "1",
            "--tmem-fixed-cols",
            "1",
        ]
    ) == 2
    machine = captured["machine"]
    assert machine.warp_fixed_overhead == 0
    assert machine.regs_fixed_overhead == 0
    assert machine.smem_fixed_overhead == 0
    assert machine.tmem_fixed_cols == 0
    assert captured["search"]["allow_cross_warp"] is False
    assert captured["search"]["no_cross_warp_domain"] == "all-ws"
    assert captured["search"]["structural_precheck"] is False
    assert captured["search"]["prefix_lane_masks"] is False
    assert captured["search"]["full_group_lane_masks"] is False
    assert captured["search"]["spill_smem_footprint"] is False
    assert captured["search"]["reg_carried_same_lane"] is False


def test_cli_emitter_lane_constraints_are_explicit_and_provenanced(
    tmp_path, monkeypatch
):
    from paper_joint_solver import __main__ as cli
    from paper_joint_solver.search import SearchResult

    problem = SimpleNamespace(
        normalization_stats=lambda: {
            "normalization_u": 300,
            "normalization_u_effective": 300,
            "normalization_f": 0,
            "normalization_zero_collapses": 0,
            "normalization_retries": 0,
        },
        normalization_cost_pairs=(),
        normalization_u_effective=300,
    )
    captured = {}
    monkeypatch.setattr(cli, "load_problem", lambda *args, **kwargs: problem)

    def fake_search(*args, **kwargs):
        captured.update(kwargs)
        return SearchResult(None, [], 0.0, "resource_limited")

    monkeypatch.setattr(cli, "run_search", fake_search)
    ddg = tmp_path / "ddg.json"
    baseline = tmp_path / "schedule_graph.json"
    out = tmp_path / "solution.json"
    ddg.write_text("{}")
    baseline.write_text("{}")

    assert (
        cli.main(
            [
                str(ddg),
                "-o",
                str(out),
                "--baseline-graph",
                str(baseline),
                "--emitter-compatible-lane-masks",
            ]
        )
        == 2
    )
    payload = json.loads(out.read_text())
    assert captured["prefix_lane_masks"] is True
    assert captured["full_group_lane_masks"] is True
    assert payload["provenance"]["model_options"]["prefix_lane_masks"] is True
    assert payload["provenance"]["model_options"]["full_group_lane_masks"] is True


def test_cli_writes_normalization_resolution_failure(tmp_path, monkeypatch):
    from paper_joint_solver import __main__ as cli
    from paper_joint_solver.normalize import NormalizationResolutionError

    error = NormalizationResolutionError(
        initial_u=300,
        max_u=1200,
        zero_collapses=2,
        positive_costs=10,
        collapsed_cost_values=(1, 2),
    )
    def fail_load(*args, **kwargs):
        raise error

    monkeypatch.setattr(cli, "load_problem", fail_load)
    ddg = tmp_path / "ddg.json"
    baseline = tmp_path / "schedule_graph.json"
    out = tmp_path / "solution.json"
    ddg.write_text("{}")
    baseline.write_text("{}")

    assert cli.main(
        [
            str(ddg),
            "-o",
            str(out),
            "--baseline-graph",
            str(baseline),
        ]
    ) == 2
    payload = json.loads(out.read_text())
    assert payload["status"] == "normalization_resolution_error"
    assert payload["satisfiable"] is False
    assert payload["provenance"]["normalization_u"] == 300
    assert payload["provenance"]["normalization_u_effective"] == 1200
    assert payload["normalization_error"]["zero_collapses"] == 2
