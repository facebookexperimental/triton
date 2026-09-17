import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_joint_solver.search import run_search


class Problem:
    pass


_CURATED_STUB = {
    "schema_version": "curated-ddg-0.2",
    "curation_source": {
        "ddg_sha256": "0" * 64,
        "baseline_graph_sha256": "0" * 64,
        "curator_sources_sha256": "0" * 64,
    },
    "loops": [],
}


def _solution_stub():
    return SimpleNamespace(
        ii=1,
        length=1,
        copies=1,
        horizon=1,
        cycles={},
        warp_sets={},
        stats={},
    )


def test_cli_rejects_baseline_graph_without_ir_emission(tmp_path):
    """R6: --baseline-graph is an IR-emission input, not a solver input."""
    from paper_joint_solver import __main__ as cli

    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                str(tmp_path / "curated_ddg.json"),
                "-o",
                str(tmp_path / "out.json"),
                "--baseline-graph",
                str(tmp_path / "schedule_graph.json"),
            ]
        )

    assert error.value.code == 2


@pytest.mark.parametrize(
    "removed_flag",
    (
        ("--paper-pure",),
        ("--minimize-groups",),
        ("--max-groups", "2"),
        ("--ignore-fixed-overheads",),
        ("--no-spill-smem-footprint",),
        ("--no-reg-carried-same-lane",),
        ("--no-cross-warp-domain", "reg-data"),
        ("--no-structural-precheck",),
        ("--emitter-compatible-lane-masks",),
        ("--ilp-seconds", "10"),
        ("--smt-seconds", "10"),
        ("--max-wall-s", "10"),
        ("--max-ii-span", "4"),
        ("--warp-fixed-overhead", "4"),
        ("--reg-fixed-overhead", "4"),
        ("--smem-fixed-overhead", "4"),
        ("--tmem-fixed-cols", "4"),
        ("--profile", "paper"),
    ),
)
def test_cli_has_no_mode_flags(tmp_path, removed_flag):
    """The canonical surface has no model switches left; an old caller gets
    an argparse usage error, which is an infrastructure failure (never a
    verdict)."""
    from paper_joint_solver import __main__ as cli

    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                str(tmp_path / "ddg.json"),
                "-o",
                str(tmp_path / "solution.json"),
                *removed_flag,
            ]
        )

    assert error.value.code == 2


def modulo_sat(prob, ii):
    return SimpleNamespace(ii=ii, length=1)


def test_search_returns_first_sat_unmodified():
    calls = []
    solution = SimpleNamespace(warp_sets={0: (1,), 1: (2,)}, stats={})

    def joint_sat(prob, ii, length, **kwargs):
        calls.append((ii, length))
        return solution, "sat"

    result = run_search(
        Problem(),
        modulo_solver=modulo_sat,
        joint_solver=joint_sat,
    )

    assert result.status == "sat"
    assert result.solution is solution
    assert result.solution.stats == {}
    assert calls == [(1, 1)]


def test_search_ascends_from_ii_one():
    def modulo(prob, ii):
        return None if ii < 3 else SimpleNamespace(ii=ii, length=3)

    def joint_sat(prob, ii, length, **kwargs):
        return SimpleNamespace(warp_sets={}, stats={}), "sat"

    result = run_search(
        Problem(), modulo_solver=modulo, joint_solver=joint_sat
    )

    assert result.status == "sat"
    assert result.attempts[:2] == [
        {"ii": 1, "stage": "modulo", "result": "unsat"},
        {"ii": 2, "stage": "modulo", "result": "unsat"},
    ]
    assert result.attempts[-1]["ii"] == 3
    assert result.attempts[-1]["stage"] == "joint"


def test_l_window_closes_on_ceiling_growth():
    """Algorithm 1 line 9: L grows while ceil(L/I) is unchanged, then I does."""
    calls = []

    def modulo(prob, ii):
        return None if ii < 2 else SimpleNamespace(ii=ii, length=3)

    def joint(prob, ii, length, **kwargs):
        calls.append((ii, length))
        if ii == 3:
            return SimpleNamespace(warp_sets={}, stats={}), "sat"
        return None, "unsat"

    result = run_search(Problem(), modulo_solver=modulo, joint_solver=joint)

    assert result.status == "sat"
    # I=2 probes L in {3, 4} (ceil = 2); L=5 would make ceil 3, closing the
    # window.  I=3 probes only L=3.
    assert calls == [(2, 3), (2, 4), (3, 3)]


def test_run_search_passes_no_model_switches():
    seen = []

    def joint(prob, ii, length, **kwargs):
        seen.append(kwargs)
        return SimpleNamespace(warp_sets={}, stats={}), "sat"

    run_search(Problem(), modulo_solver=modulo_sat, joint_solver=joint)

    assert [set(kwargs) for kwargs in seen] == [
        {"num_warps_override", "allow_cross_warp"}
    ]


def test_joint_semantics_options_are_forwarded():
    seen = []

    def joint(prob, ii, length, **kwargs):
        seen.append((kwargs["num_warps_override"], kwargs["allow_cross_warp"]))
        return SimpleNamespace(warp_sets={}, stats={}), "sat"

    result = run_search(
        Problem(),
        num_warps_override=16,
        allow_cross_warp=False,
        modulo_solver=modulo_sat,
        joint_solver=joint,
    )

    assert result.status == "sat"
    assert seen == [(16, False)]


def test_non_verdict_from_the_joint_solver_is_an_error():
    def joint(prob, ii, length, **kwargs):
        return None, "unknown"

    with pytest.raises(RuntimeError, match="non-verdict"):
        run_search(
            Problem(), modulo_solver=modulo_sat, joint_solver=joint
        )


def test_cli_ir_emission_requires_baseline_graph_tri_pair(tmp_path, monkeypatch):
    from paper_joint_solver import __main__ as cli
    from paper_joint_solver import pipelined_ir
    from paper_joint_solver.search import SearchResult

    problem = SimpleNamespace(
        normalization_stats=lambda: {
            "normalization_u": 300,
            "normalization_f": 0,
        },
        normalization_cost_pairs=(),
    )
    solution = _solution_stub()
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
    ddg.write_text(json.dumps(_CURATED_STUB))
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
    assert set(search_kwargs) == {"num_warps_override", "allow_cross_warp"}
    payload = json.loads(out.read_text())
    assert payload["warp_sets"] == {}
    assert "lane_masks" not in payload
    assert handoff_kwargs["solution_path"] == str(out)
    assert handoff_kwargs["ir_out"] == str(ir_out)
    assert handoff_kwargs["manifest_out"] == str(manifest_out)

    # Two thirds of the triple is a usage error, not a partial emission.
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                str(ddg),
                "-o",
                str(out),
                "--ir-out",
                str(ir_out),
                "--handoff-manifest-out",
                str(manifest_out),
            ]
        )
    assert error.value.code == 2


def test_cli_default_call_is_paper_literal(tmp_path, monkeypatch):
    """No flags at all: the search receives the paper's inputs and nothing else."""
    from paper_joint_solver import __main__ as cli
    from paper_joint_solver.search import SearchResult

    captured = {}
    problem = SimpleNamespace(
        normalization_stats=lambda: {
            "normalization_u": 300,
            "normalization_f": 0,
        },
        normalization_cost_pairs=(),
    )
    monkeypatch.setattr(cli, "load_problem", lambda *a, **k: problem)

    def fake_search(*args, **kwargs):
        captured.update(kwargs)
        return SearchResult(_solution_stub(), [], 0.0, "sat")

    monkeypatch.setattr(cli, "run_search", fake_search)
    ddg = tmp_path / "ddg.json"
    out = tmp_path / "solution.json"
    ddg.write_text(json.dumps(_CURATED_STUB))

    assert cli.main([str(ddg), "-o", str(out)]) == 0
    assert captured == {"num_warps_override": None, "allow_cross_warp": True}


def test_cli_returns_zero_on_sat(tmp_path, monkeypatch):
    """rc=0 is the only normal termination; the verdict lives in the JSON."""
    from paper_joint_solver import __main__ as cli
    from paper_joint_solver.schedule_plan import PAPER_SOLUTION_SCHEMA
    from paper_joint_solver.search import SearchResult

    problem = SimpleNamespace(
        normalization_stats=lambda: {
            "normalization_u": 300,
            "normalization_f": 0,
        },
        normalization_cost_pairs=(),
    )
    monkeypatch.setattr(cli, "load_problem", lambda *a, **k: problem)
    monkeypatch.setattr(
        cli,
        "run_search",
        lambda *a, **k: SearchResult(_solution_stub(), [], 0.0, "sat"),
    )
    ddg = tmp_path / "ddg.json"
    out = tmp_path / "solution.json"
    ddg.write_text(json.dumps(_CURATED_STUB))

    assert cli.main([str(ddg), "-o", str(out)]) == 0
    payload = json.loads(out.read_text())
    assert payload["schema_version"] == PAPER_SOLUTION_SCHEMA
    assert payload["status"] == "sat" and payload["satisfiable"] is True
    assert "warp_sets" in payload
    assert not {"warp", "group_widths", "lane_masks"} & payload.keys()


def test_cli_records_experiment_inputs_as_observations(tmp_path, monkeypatch):
    """Ablation inputs are transcribed; the model transcription is untouched."""
    from paper_joint_solver import __main__ as cli
    from paper_joint_solver.search import SearchResult

    problem = SimpleNamespace(
        normalization_stats=lambda: {
            "normalization_u": 300,
            "normalization_f": 0,
        },
        normalization_cost_pairs=(),
    )
    monkeypatch.setattr(cli, "load_problem", lambda *a, **k: problem)
    monkeypatch.setattr(
        cli,
        "run_search",
        lambda *a, **k: SearchResult(_solution_stub(), [], 0.0, "sat"),
    )
    ddg = tmp_path / "ddg.json"
    ddg.write_text(json.dumps(_CURATED_STUB))
    free = tmp_path / "free.json"
    ablated = tmp_path / "ablated.json"

    assert cli.main([str(ddg), "-o", str(free)]) == 0
    assert cli.main(
        [
            str(ddg),
            "-o",
            str(ablated),
            "--num-warps-override",
            "16",
            "--no-cross-warp",
            "--reg-budget",
            "4096",
        ]
    ) == 0
    free_provenance = json.loads(free.read_text())["provenance"]
    ablated_provenance = json.loads(ablated.read_text())["provenance"]
    assert free_provenance["experiment_inputs"] == {
        "allow_cross_warp": True,
        "num_warps_override": None,
        "reg_budget": None,
    }
    assert ablated_provenance["experiment_inputs"] == {
        "allow_cross_warp": False,
        "num_warps_override": 16,
        "reg_budget": 4096,
    }
    # There is no model transcription to change: one model, no options.
    assert free_provenance["model"] == ablated_provenance["model"] == "paper"
    assert "model_options" not in ablated_provenance


def test_cli_forwards_experiment_inputs(tmp_path, monkeypatch):
    from paper_joint_solver import __main__ as cli
    from paper_joint_solver.search import SearchResult

    captured = {}
    problem = SimpleNamespace(
        normalization_stats=lambda: {
            "normalization_u": 300,
            "normalization_f": 0,
        },
        normalization_cost_pairs=(),
    )

    def fake_load(*args, **kwargs):
        captured["machine"] = kwargs["machine"]
        return problem

    def fake_search(*args, **kwargs):
        captured["search"] = kwargs
        return SearchResult(_solution_stub(), [], 0.0, "sat")

    monkeypatch.setattr(cli, "load_problem", fake_load)
    monkeypatch.setattr(cli, "run_search", fake_search)
    ddg = tmp_path / "ddg.json"
    out = tmp_path / "solution.json"
    ddg.write_text(json.dumps(_CURATED_STUB))

    assert cli.main(
        [
            str(ddg),
            "-o",
            str(out),
            "--num-warps-override",
            "16",
            "--no-cross-warp",
            "--reg-budget",
            "128",
        ]
    ) == 0
    assert captured["search"] == {
        "num_warps_override": 16,
        "allow_cross_warp": False,
    }
    machine = captured["machine"]
    assert machine.num_warps == 16
    assert machine.regs_per_warp == 128
    # The paper model has no fixed overhead to configure at all.
    assert machine.warp_fixed_overhead == 0
    provenance = json.loads(out.read_text())["provenance"]
    assert provenance["experiment_inputs"] == {
        "allow_cross_warp": False,
        "num_warps_override": 16,
        "reg_budget": 128,
    }
