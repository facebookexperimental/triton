import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

_SOLVER_PATH = Path(__file__).parents[3] / "triton/tools/modulo_joint_solver.py"
_SPEC = importlib.util.spec_from_file_location("modulo_joint_solver", _SOLVER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
modulo_joint_solver = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(modulo_joint_solver)

requires_ortools = pytest.mark.skipif(
    modulo_joint_solver.cp_model is None, reason="OR-Tools is not in the Buck runtime"
)


def _edge(src, dst, latency):
    edge = {
        "src": src,
        "dst": dst,
        "latency": latency,
        "distance": 0,
        "freq": 1,
        "rt": 0,
        "xissue": 0,
        "chan_bytes": 0,
    }
    cluster = {0: 0, 1: 1, 2: 1, 3: 2, 4: 1, 5: 1}
    if cluster[src] != cluster[dst]:
        edge["src_cluster"] = cluster[src]
        edge["dst_cluster"] = cluster[dst]
    return edge


def _async_reader_template(template_id, src, dst, src_cluster):
    common = {
        "frequency": 1,
        "buffer_id": None,
        "bytes": 0,
        "depth": 1,
        "semaphore": "full",
        "fusion_group": template_id,
        "dedup_group": template_id,
    }
    return {
        "id": template_id,
        "relation": "always",
        "src_node": src,
        "dst_node": dst,
        "src_cluster": src_cluster,
        "dst_cluster": 1,
        "events": [
            {
                "id": 0,
                "kind": "tc_commit",
                "owner": "src",
                "anchor_node": src,
                "placement": "after",
                "pipeline": "NONE",
                "issue_duration": 1,
                "completion_latency": 10,
                "blocking": False,
                "async": True,
                "distance": 0,
                **common,
            },
            {
                "id": 1,
                "kind": "wait",
                "owner": "dst",
                "anchor_node": dst,
                "placement": "before",
                "pipeline": "NONE",
                "issue_duration": 1,
                "completion_latency": 0,
                "blocking": True,
                "async": False,
                "distance": 0,
                **common,
            },
        ],
    }


def _wait_order_problem():
    nodes = [
        {
            "id": 0,
            "cycle": 0,
            "duration": 1,
            "latency": 10,
            "pipeline": "TC",
            "freq": 1,
        },
        {
            "id": 1,
            "cycle": 10,
            "duration": 2,
            "latency": 2,
            "pipeline": "CUDA",
            "freq": 1,
        },
        {
            "id": 2,
            "cycle": 12,
            "duration": 3,
            "latency": 3,
            "pipeline": "SFU",
            "freq": 1,
        },
        {
            "id": 3,
            "cycle": 1,
            "duration": 1,
            "latency": 10,
            "pipeline": "TC",
            "freq": 1,
        },
        {
            "id": 4,
            "cycle": 11,
            "duration": 2,
            "latency": 2,
            "pipeline": "CUDA",
            "freq": 1,
        },
        {
            "id": 5,
            "cycle": 20,
            "duration": 1,
            "latency": 1,
            "pipeline": "CUDA",
            "freq": 1,
        },
    ]
    return {
        "version": "joint-solver-0.2",
        "mode": "joint",
        "ii": 50,
        "clusters": [
            {"id": 0, "min_warps": 1, "nodes": [0]},
            {"id": 1, "min_warps": 4, "nodes": [1, 2, 4, 5]},
            {"id": 2, "min_warps": 1, "nodes": [3]},
        ],
        "nodes": nodes,
        "edges": [
            _edge(0, 1, 10),
            _edge(1, 2, 2),
            _edge(3, 4, 10),
            _edge(2, 5, 3),
            _edge(4, 5, 2),
        ],
        "buffers": [],
        "lowering_templates": [
            _async_reader_template(0, 0, 1, 0),
            _async_reader_template(1, 3, 4, 2),
        ],
        "max_wgs": 3,
        "committed_smem": 0,
        "fixed_smem": 0,
        "smem_budget": 100000,
        "warp_footprint": [0, 1, 2, 0, 4, 0, 0, 0, 8],
        "default_wg_footprint": 0,
        "sm_regs": 1000,
        "default_slack": 0,
        "time_limit_s": 5.0,
    }


@requires_ortools
def test_joint_solver_defers_blocking_wait_below_independent_work():
    problem = _wait_order_problem()
    modulo_joint_solver._validate_problem_schema(problem)

    result = modulo_joint_solver.solve_joint(problem)
    assert result["status"] == "ok"
    assert result["cycles"]["2"] < result["cycles"]["4"]
    assert result["lowering_objective"] == 2

    plan = modulo_joint_solver._instantiate_lowering_plan(problem, result)
    wait = plan["templates"][1]["events"][1]
    assert wait["cycle"] == result["cycles"]["4"] - 1
    assert wait["wg"] == result["wg"]["1"]


def test_partition_schema_requires_lowering_aware_version():
    problem = _wait_order_problem()
    problem["version"] = "joint-solver-0.1"
    with pytest.raises(ValueError, match="joint-solver-0.2"):
        modulo_joint_solver._validate_problem_schema(problem)


def test_fingerprint_does_not_require_problem_files(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [str(_SOLVER_PATH), "--fingerprint"])
    assert modulo_joint_solver.main() == 0
    fingerprint = json.loads(capsys.readouterr().out)
    assert fingerprint["ortools_version"]
    assert fingerprint["python_version"]
    assert len(fingerprint["source_sha256"]) == 64


@requires_ortools
def test_solver_uses_deterministic_search_budget():
    solver = modulo_joint_solver._new_cp_solver(
        1.25, num_search_workers=1, random_seed=0
    )
    assert solver.parameters.max_deterministic_time == 1.25
    assert solver.parameters.max_time_in_seconds == float("inf")


def test_missing_ortools_reports_structured_error(monkeypatch, capsys):
    monkeypatch.setattr(modulo_joint_solver, "cp_model", None)
    monkeypatch.setattr(sys, "argv", [str(_SOLVER_PATH), "input.json", "out.json"])

    assert modulo_joint_solver.main() == 2
    error = json.loads(capsys.readouterr().out)
    assert error == {
        "status": "error",
        "message": "ortools not installed (pip install ortools)",
    }


def test_ii_sweep_stops_at_unknown_lower_ii(monkeypatch, tmp_path):
    problem_path = tmp_path / "problem.json"
    solution_path = tmp_path / "solution.json"
    problem_path.write_text(
        json.dumps(
            {
                "version": "joint-solver-0.1",
                "min_ii": 7,
                "max_ii": 8,
                "normalize_u": 0,
                "nodes": [],
                "edges": [],
                "buffers": [],
            }
        )
    )
    tried = []

    def solve_unknown(_prob, ii, _time_limit_s, _hint=None):
        tried.append(ii)
        return "unknown", None, None

    monkeypatch.setattr(modulo_joint_solver, "cp_model", object())
    monkeypatch.setattr(modulo_joint_solver, "solve_at_ii", solve_unknown)
    monkeypatch.setattr(
        sys,
        "argv",
        [str(_SOLVER_PATH), str(problem_path), str(solution_path)],
    )

    assert modulo_joint_solver.main() == 1
    assert tried == [7]
    solution = json.loads(solution_path.read_text())
    assert solution["status"] == "error"
    assert solution["stats"]["unknown_iis"] == [7]


@requires_ortools
def test_multiple_waits_at_one_anchor_are_serialized():
    problem = _wait_order_problem()
    problem["edges"].append(_edge(0, 4, 10))
    shared_commit = _async_reader_template(2, 0, 4, 0)
    shared_commit["events"][0]["issue_duration"] = 0
    problem["lowering_templates"].append(shared_commit)

    result = modulo_joint_solver.solve_joint(problem)
    assert result["status"] == "ok"
    plan = modulo_joint_solver._instantiate_lowering_plan(problem, result)
    waits = [plan["templates"][index]["events"][1] for index in (1, 2)]
    assert [event["cycle"] for event in waits] == [
        result["cycles"]["4"] - 2,
        result["cycles"]["4"] - 1,
    ]


@requires_ortools
def test_lowering_frequency_reserves_every_issue(monkeypatch):
    problem = _wait_order_problem()
    problem["ii"] = 3
    problem["nodes"] = deepcopy(problem["nodes"][:2])
    problem["nodes"][0].update(cycle=0, duration=1, latency=2)
    problem["nodes"][1].update(cycle=2, duration=1, latency=1)
    problem["clusters"] = [
        {"id": 0, "min_warps": 1, "nodes": [0]},
        {"id": 1, "min_warps": 1, "nodes": [1]},
    ]
    problem["edges"] = [dict(_edge(0, 1, 2), src_cluster=0, dst_cluster=1)]
    problem["lowering_templates"] = [_async_reader_template(0, 0, 1, 0)]
    problem["lowering_templates"][0]["events"][1]["frequency"] = 2
    problem["max_wgs"] = 2

    result = modulo_joint_solver.solve_joint(problem)
    assert result["status"] == "ok"
    plan = modulo_joint_solver._instantiate_lowering_plan(problem, result)
    assert plan["templates"][0]["events"][1]["cycle"] == 0

    problem["lowering_templates"][0]["events"][0]["frequency"] = 3
    assert modulo_joint_solver.solve_joint(problem)["status"] == "error"

    monkeypatch.setenv("TRITON_MODULO_INTRAWG_ASYNC_LEGALITY", "0")
    problem["lowering_templates"][0]["relation"] = "different_wg"
    conditional_result = modulo_joint_solver.solve_joint(problem)
    assert conditional_result["status"] == "ok"
    assert conditional_result["wg"]["0"] == conditional_result["wg"]["1"]
    conditional_plan = modulo_joint_solver._instantiate_lowering_plan(
        problem, conditional_result
    )
    assert not conditional_plan["templates"][0]["active"]


@requires_ortools
def test_channel_budget_distinguishes_producer_results():
    problem = _wait_order_problem()
    problem["smem_budget"] = 100
    first = problem["edges"][0]
    first.update(src_result_idx=0, chan_bytes=60)
    second = dict(first, src_result_idx=1)
    problem["edges"].append(second)

    result = modulo_joint_solver.solve_partition(
        problem, fixed_assign={0: 0, 1: 1, 2: 2}
    )
    assert result["status"] == "infeasible"


def test_arbitration_keeps_v2_when_probe_is_unknown(monkeypatch):
    problem = _wait_order_problem()
    problem["buffers"] = [
        {
            "producer": 0,
            "size_bytes": 1,
            "kind": "smem",
            "consumers": [{"node": 1, "latency": 0, "distance": 0}],
        }
    ]
    responses = iter(
        [
            {
                "status": "ok",
                "wg": {"0": 0, "1": 1, "2": 2},
                "used_wgs": 3,
                "objective": 1,
                "combined": 1,
            },
            {"status": "unknown"},
        ]
    )
    monkeypatch.setattr(
        modulo_joint_solver, "solve_partition", lambda *_args, **_kwargs: next(responses)
    )

    assert modulo_joint_solver._arbitrate_v2(problem, {0: 0, 1: 1, 2: 2}) is None


@requires_ortools
def test_joint_solver_preserves_committed_stages():
    problem = _wait_order_problem()
    problem["nodes"][4]["cycle"] = 61
    problem["nodes"][5]["cycle"] = 70

    result = modulo_joint_solver.solve_joint(problem)

    assert result["status"] == "ok"
    ii = problem["ii"]
    committed_stages = {
        str(node["id"]): node["cycle"] // ii for node in problem["nodes"]
    }
    solved_stages = {
        node_id: cycle // ii for node_id, cycle in result["cycles"].items()
    }
    assert solved_stages == committed_stages


@requires_ortools
def test_joint_solver_is_deterministic():
    problem = _wait_order_problem()
    problem["buffers"] = [
        {
            "producer": 0,
            "size_bytes": 1,
            "kind": "smem",
            "consumers": [{"node": 1, "latency": 0, "distance": 0}],
        }
    ]
    assert modulo_joint_solver._has_full_graph(problem)
    results = [
        modulo_joint_solver.solve_joint(deepcopy(problem)) for _ in range(4)
    ]

    signatures = {
        json.dumps(
            {
                "wg": result["wg"],
                "cycles": result["cycles"],
                "objective": result["objective"],
                "lowering_objective": result["lowering_objective"],
            },
            sort_keys=True,
        )
        for result in results
    }
    assert len(signatures) == 1


@requires_ortools
def test_lowering_events_occupy_only_the_owner_wg_stream():
    problem = _wait_order_problem()
    problem["ii"] = 3
    problem["nodes"] = deepcopy(problem["nodes"][:4])
    problem["nodes"][1].update(cycle=1, duration=1, latency=1)
    problem["nodes"][2].update(cycle=1, duration=1, latency=1, pipeline="TC")
    problem["nodes"][3].update(cycle=2, duration=1, latency=1, pipeline="CUDA")
    problem["clusters"] = [
        {"id": index, "min_warps": 1, "nodes": [index]} for index in range(4)
    ]
    problem["edges"] = [
        dict(_edge(0, 1, 1), src_cluster=0, dst_cluster=1),
        dict(_edge(2, 3, 1), src_cluster=2, dst_cluster=3),
    ]
    problem["lowering_templates"] = [
        _async_reader_template(0, 0, 1, 0),
        _async_reader_template(1, 2, 3, 2),
    ]
    for template in problem["lowering_templates"]:
        template["dst_cluster"] = template["dst_node"]
        template["events"][1]["issue_duration"] = 0
    problem["max_wgs"] = 4

    separate_result = modulo_joint_solver.solve_joint(problem)
    assert separate_result["status"] == "ok", separate_result

    problem["clusters"] = [
        {"id": 0, "min_warps": 1, "nodes": [0, 2]},
        {"id": 1, "min_warps": 1, "nodes": [1]},
        {"id": 2, "min_warps": 1, "nodes": [3]},
    ]
    problem["edges"][1].update(src_cluster=0, dst_cluster=2)
    problem["lowering_templates"][1].update(src_cluster=0, dst_cluster=2)
    problem["max_wgs"] = 3
    assert modulo_joint_solver.solve_joint(problem)["status"] == "error"
