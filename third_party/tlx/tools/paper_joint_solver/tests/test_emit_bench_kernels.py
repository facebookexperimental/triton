import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SOLVER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOLVER_ROOT))

import emit_bench_kernels

RUN_EMITTER_CASES = SOLVER_ROOT / "run_emitter_cases.sh"


def test_case_outputs_match_bench_registry(tmp_path):
    assert emit_bench_kernels.WORKFLOW_SCHEMA == "jos-bench-emission-v1"
    cases = emit_bench_kernels._cases(tmp_path)

    assert [case.solution.name for case in cases] == [
        "fwd_subtiled_emitter_v8.json",
        "bwd_emitter_v8.json",
        "bwd_lr4096_emitter_v8.json",
    ]
    assert [case.graph_output.name for case in cases] == [
        "schedule_graph_jos_v8.json",
        "schedule_graph_jos_v8.json",
        "schedule_graph_jos_lr4096_v8.json",
    ]
    assert [case.kernel_output.name for case in cases] == [
        "generated_jos_v8.py",
        "generated_jos_v8.py",
        "generated_jos_lr4096_v8.py",
    ]
    assert cases[0].kernel_output.parent.name == "case3_FA_fp16_subtiled"
    assert cases[1].kernel_output.parent.name == "case4_FA_bwd_subtiled"


def test_emit_graph_rejects_non_sat_solution_without_output(tmp_path):
    from paper_joint_solver.schedule_plan import EMITTER_SOLUTION_SCHEMA

    solution = tmp_path / "solution.json"
    ddg = tmp_path / "ddg.json"
    baseline = tmp_path / "schedule_graph.json"
    output = tmp_path / "output.json"
    solution.write_text(
        json.dumps(
            {
                "schema_version": EMITTER_SOLUTION_SCHEMA,
                "ii": 1,
                "length": 1,
                "copies": 1,
                "horizon": 1,
                "cycles": {},
                "warp": {},
                "group_widths": {},
                "lane_masks": {},
                "stats": {},
                "provenance": {},
                "status": "unsat",
                "satisfiable": False,
            }
        )
    )
    ddg.write_text("{}\n")
    baseline.write_text("{}\n")

    assert (
        emit_bench_kernels.main(
            [
                "emit-graph",
                "--case-name",
                "test",
                "--solution",
                str(solution),
                "--ddg",
                str(ddg),
                "--baseline-graph",
                str(baseline),
                "--output",
                str(output),
                "--expected-regs-per-warp",
                "8160",
                "--expected-warp-fixed-overhead",
                "0",
            ]
        )
        == 2
    )
    assert not output.exists()


def test_bench_loader_requires_explicit_emitter_profile(tmp_path, monkeypatch):
    solution = tmp_path / "fwd_subtiled_emitter_v8.json"
    solution.write_text(
        json.dumps(
            {
                "attempts": [
                    {"stage": "joint", "result": "sat", "ii": 1, "L": 1}
                ]
            }
        )
    )
    expected_machine = emit_bench_kernels._expected_machine(8160, 4)
    plan = SimpleNamespace(
        normalization_u=300,
        machine=expected_machine,
        ii=1,
        length=1,
    )
    problem = object()
    captured = {}

    def fake_load(solution_path, ddg, model_profile, baseline):
        captured["model_profile"] = model_profile
        return problem, plan

    monkeypatch.setattr(emit_bench_kernels, "load_schedule_context", fake_load)

    assert emit_bench_kernels._load_emitter_plan(
        name="fwd_subtiled_emitter_v8",
        solution=solution,
        ddg=tmp_path / "ddg.json",
        baseline=tmp_path / "schedule_graph.json",
        regs_per_warp=8160,
        warp_fixed_overhead=4,
        require_tmem_evidence=False,
    ) == (problem, plan)
    assert captured["model_profile"] == emit_bench_kernels.EMITTER_MODEL_PROFILE


def test_emit_graph_never_overwrites_existing_output(tmp_path):
    output = tmp_path / "output.json"
    output.write_text("sentinel\n")

    assert (
        emit_bench_kernels.main(
            [
                "emit-graph",
                "--case-name",
                "test",
                "--solution",
                str(tmp_path / "missing-solution.json"),
                "--ddg",
                str(tmp_path / "missing-ddg.json"),
                "--baseline-graph",
                str(tmp_path / "missing-graph.json"),
                "--output",
                str(output),
                "--expected-regs-per-warp",
                "8160",
                "--expected-warp-fixed-overhead",
                "0",
            ]
        )
        == 2
    )
    assert output.read_text() == "sentinel\n"


def test_publish_is_exclusive_and_keeps_record_copy_independent(tmp_path):
    staged = tmp_path / "staged.py"
    output = tmp_path / "generated.py"
    staged.write_text("first\n")

    emit_bench_kernels._publish_exclusive(staged, output)
    staged.write_text("changed\n")

    assert output.read_text() == "first\n"
    with pytest.raises(emit_bench_kernels.WorkflowError, match="overwrite"):
        emit_bench_kernels._publish_exclusive(staged, output)
    assert output.read_text() == "first\n"


def test_run_rejects_missing_solutions_before_creating_record(tmp_path):
    solution_dir = tmp_path / "solutions"
    solution_dir.mkdir()
    record_dir = solution_dir / "record"

    assert (
        emit_bench_kernels.main(
            [
                "run",
                "--solution-dir",
                str(solution_dir),
                "--record-dir",
                str(record_dir),
            ]
        )
        == 2
    )
    assert not record_dir.exists()


def test_emitter_solve_workflow_is_retired():
    """The automatic emitter arm is retired: the batch must refuse to run
    rather than quietly produce artifacts that no longer have a solver."""
    source = RUN_EMITTER_CASES.read_text()

    assert "RETIRED" in source
    assert "exit 1" in source
    # None of the guarded-era machinery may survive in an executable form.
    assert "--emitter-compatible-lane-masks" not in source
    assert 'taskset -c "$SOLVER_CPU"' not in source
    assert "fwd_subtiled_emitter_v8" not in source
    assert "bwd_lr4096_emitter_v8" not in source


def test_retired_emitter_batch_refuses_to_run(tmp_path):
    completed = subprocess.run(
        ["/bin/bash", str(RUN_EMITTER_CASES)],
        text=True,
        capture_output=True,
        cwd=tmp_path,
        check=False,
    )

    assert completed.returncode == 1
    assert "RETIRED" in completed.stderr
