import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from paper_joint_solver.ddg import Edge, Node, load_problem
from paper_joint_solver import strategy_report
from paper_joint_solver.strategy_report import (
    classify,
    curation_dropped_nodes,
    load_fa4_template_partition,
    refit_fa4_template,
)

EXAMPLES = Path(__file__).resolve().parents[2] / "sched2tlx" / "examples"
BWD_SUBTILED = EXAMPLES / "case4_FA_bwd_subtiled"
_FREE_INPUTS = {
    "allow_cross_warp": True,
    "num_warps_override": None,
    "reg_budget": None,
}


def _node(node_id: int, op_kind: str, pipeline: str = "CUDA") -> Node:
    return Node(node_id, f"op_{node_id}", op_kind, pipeline, 1, 1, 1)


def _problem(*, three_compute_groups: bool) -> tuple[SimpleNamespace, dict[int, int]]:
    nodes = {
        0: _node(0, "tt.descriptor_load", "TMA"),
        1: _node(1, "ttng.tc_gen5_mma", "TC"),
        2: _node(2, "math.exp2", "SFU"),
        3: _node(3, "math.exp2", "SFU"),
        4: _node(4, "ttg.convert_layout"),
        5: _node(5, "ttg.convert_layout"),
        8: _node(8, "ttng.tmem_load", "TMEM"),
        9: _node(9, "ttng.tmem_load", "TMEM"),
        10: _node(10, "arith.mulf"),
        11: _node(11, "arith.subf"),
    }
    edges = [
        Edge(8, 10, 0, 1),
        Edge(10, 2, 0, 1),
        Edge(9, 11, 0, 1),
        Edge(11, 3, 0, 1),
    ]
    problem = SimpleNamespace(
        nodes=nodes,
        edges=edges,
        variable_latency={0},
        regs={node_id: 16 if node_id in {4, 5} else 1 for node_id in nodes},
    )
    warp = {0: 0, 1: 1, 2: 2, 3: 3, 8: 2, 9: 3, 10: 2, 11: 3}
    if three_compute_groups:
        warp.update({4: 4, 5: 4})
    else:
        warp.update({4: 2, 5: 3})
    return problem, warp


# Curation drops the two `tt.descriptor_reduce` nodes as emitter infra, so the
# reduction-staging observation is reconstructed from the manifest (WI-8.2).
STAGING_MANIFEST = {
    "loops": [
        {
            "nodes": [
                {"id": 6, "op_kind": "tt.descriptor_reduce", "action": "dropped"},
                {"id": 7, "op_kind": "tt.descriptor_reduce", "action": "dropped"},
            ],
            "edges": [
                {"src": 4, "dst": 6, "action": "removed"},
                {"src": 5, "dst": 7, "action": "removed"},
            ],
        }
    ]
}


@pytest.fixture(scope="module")
def bwd_subtiled_fixture():
    import tempfile

    from paper_joint_solver.curate_ddg import curate, dumps

    curated_data, manifest = curate(
        BWD_SUBTILED / "ddg.json", BWD_SUBTILED / "schedule_graph.json"
    )
    curated = Path(tempfile.mkdtemp()) / "curated_ddg.json"
    curated.write_text(dumps(curated_data))
    problem = load_problem(curated)
    problem.curation_manifest = manifest
    graph = json.loads((BWD_SUBTILED / "schedule_graph.json").read_text())
    graph_nodes = graph["loops"][0]["schedule_loop"]["graph"]["nodes"]
    # Curated G no longer contains the dropped emitter-infra nodes.
    warp = {
        node["id"]: node["warp_group"]
        for node in graph_nodes
        if node["id"] in problem.nodes
    }
    return problem, warp


def _fa4_problem() -> tuple[SimpleNamespace, dict[int, int]]:
    nodes = {
        0: _node(0, "tt.descriptor_load", "TMA"),
        1: _node(1, "ttng.tc_gen5_mma", "TC"),
        2: _node(2, "math.exp2", "SFU"),
        3: _node(3, "math.exp2", "SFU"),
        4: _node(4, "ttng.tmem_load"),
        5: _node(5, "ttng.tmem_load"),
    }
    problem = SimpleNamespace(
        nodes=nodes,
        edges=[],
        variable_latency={0},
        regs={node_id: 16 for node_id in nodes},
    )
    return problem, {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 1}


def _write_template(path, warp: dict[int, int]) -> None:
    path.write_text(json.dumps({"warp": {str(k): v for k, v in warp.items()}}))


def test_classify_forward_requires_exact_five_disjoint_roles() -> None:
    problem, warp = _fa4_problem()

    report = classify(problem, warp, None, STAGING_MANIFEST)

    assert report["fa4_like"] is True
    assert report["num_active_groups"] == 5
    assert report["fa4_roles_disjoint"] is True
    assert report["fa4_roles_cover_active"] is True
    assert report["fa4_role_groups"] == {
        "variable_latency": [0],
        "tensor_core": [1],
        "softmax": [2, 3],
        "rescale": [4],
    }


def test_classify_forward_rejects_extra_active_group() -> None:
    problem, warp = _fa4_problem()
    problem.nodes[6] = _node(6, "arith.addf")
    warp[6] = 5

    report = classify(problem, warp, None, STAGING_MANIFEST)

    assert report["fa4_like"] is False
    assert report["num_active_groups"] == 6
    assert report["fa4_roles_cover_active"] is False


def test_classify_forward_rejects_overlapping_roles() -> None:
    problem, warp = _fa4_problem()
    warp[1] = 2

    report = classify(problem, warp, None, STAGING_MANIFEST)

    assert report["fa4_like"] is False
    assert report["fa4_roles_disjoint"] is False


def test_fa4_template_requires_exact_ddg_coverage(tmp_path) -> None:
    template = tmp_path / "template.json"
    _write_template(template, {0: 0, 2: 1})

    with pytest.raises(ValueError, match=r"missing=\[1\], extra=\[2\]"):
        load_fa4_template_partition(template, {0, 1})


def test_curation_dropped_nodes_reads_manifest_dispositions() -> None:
    assert curation_dropped_nodes(None) == frozenset()
    assert curation_dropped_nodes(STAGING_MANIFEST) == frozenset({6, 7})


def test_fa4_template_projects_curation_dropped_nodes(tmp_path) -> None:
    """The frozen template keeps raw ids; curation's record removes them."""
    template = tmp_path / "template.json"
    _write_template(template, {0: 4, 1: 4, 2: 0, 3: 1})
    manifest = {
        "loops": [
            {
                "nodes": [
                    {"id": 0, "op_kind": "arith.muli", "action": "dropped"},
                    {"id": 1, "op_kind": "arith.addi", "action": "dropped"},
                    {"id": 2, "op_kind": "tt.descriptor_load", "action": "kept"},
                ]
            }
        ]
    }

    partition = load_fa4_template_partition(
        template, {2, 3}, curation_dropped_nodes(manifest)
    )

    assert partition.projected_nodes == (0, 1)
    assert partition.groups == ((2,), (3,))


def test_fa4_template_rejects_extra_nodes_curation_did_not_drop(tmp_path) -> None:
    """A projection, not an intersection: unrecorded extras still fail."""
    template = tmp_path / "template.json"
    _write_template(template, {0: 4, 1: 4, 2: 0})
    manifest = {"loops": [{"nodes": [{"id": 0, "action": "dropped"}]}]}

    with pytest.raises(ValueError, match=r"missing=\[\], extra=\[1\]"):
        load_fa4_template_partition(template, {2}, curation_dropped_nodes(manifest))


def test_refit_fa4_template_discloses_the_projection(tmp_path) -> None:
    template = tmp_path / "template.json"
    _write_template(template, {0: 4, 1: 4, 2: 0, 3: 1})
    problem = SimpleNamespace(nodes={2: object(), 3: object()})
    manifest = {
        "loops": [
            {
                "nodes": [
                    {"id": 0, "action": "dropped"},
                    {"id": 1, "action": "dropped"},
                ]
            }
        ]
    }

    metadata, _ = refit_fa4_template(
        problem,
        template,
        ii=2,
        length=5,
        inputs=_FREE_INPUTS,
        solver=lambda *args, **kwargs: (object(), "sat"),
        curation_manifest=manifest,
    )

    assert metadata["template_projected_nodes"] == [0, 1]
    assert metadata["template_groups"] == 2


def test_refit_fa4_template_uses_free_optimum_and_exact_partition(tmp_path) -> None:
    template = tmp_path / "template.json"
    _write_template(template, {0: 9, 1: 9, 2: 20, 3: 30, 4: 40, 5: 50})
    problem = SimpleNamespace(nodes={node: object() for node in range(6)})
    call = {}

    def fake_solver(prob, ii, length, **kwargs):
        call.update(prob=prob, ii=ii, length=length, **kwargs)
        return object(), "sat"

    inputs = {
        "allow_cross_warp": False,
        "num_warps_override": 16,
        "reg_budget": 4096,
    }
    metadata, member = refit_fa4_template(
        problem,
        template,
        ii=71,
        length=153,
        inputs=inputs,
        solver=fake_solver,
    )

    assert member is True
    assert metadata["verdict"] == "sat"
    assert metadata["template_groups"] == 5
    assert metadata["audit_instrument"] is True
    assert metadata["experiment_inputs"] == inputs
    assert (call["ii"], call["length"]) == (71, 153)
    assert call["exact_warp_sets"] == 5
    # The audit entry is uncapped; bounding is the harness's job.
    assert "timeout_s" not in call
    # The refit inherits the audited solution's inputs verbatim.
    assert call["allow_cross_warp"] is False
    assert call["num_warps_override"] == 16
    assert call["colocate"] == [[0, 1]]
    assert len(call["separate"]) == 10


def test_refit_fa4_template_rejects_an_unexpected_verdict(tmp_path) -> None:
    """The verdict vocabulary is {sat, unsat}; 'unknown' no longer exists."""
    template = tmp_path / "template.json"
    _write_template(template, {0: 0})
    problem = SimpleNamespace(nodes={0: object()})

    with pytest.raises(RuntimeError, match="unexpected FA4 template refit"):
        refit_fa4_template(
            problem,
            template,
            ii=11,
            length=29,
            inputs=_FREE_INPUTS,
            solver=lambda *args, **kwargs: (None, "unknown"),
        )


@pytest.mark.parametrize("verdict", ["unsat"])
def test_refit_fa4_template_preserves_non_sat_verdict(
    tmp_path, verdict: str
) -> None:
    template = tmp_path / "template.json"
    _write_template(template, {0: 0})
    problem = SimpleNamespace(nodes={0: object()})

    metadata, member = refit_fa4_template(
        problem,
        template,
        ii=11,
        length=29,
        inputs=_FREE_INPUTS,
        solver=lambda *args, **kwargs: (None, verdict),
    )

    assert member is False
    assert metadata["verdict"] == verdict
    assert (metadata["ii"], metadata["length"]) == (11, 29)


def test_main_emits_fa4_template_refit_metadata(
    tmp_path, monkeypatch, capsys
) -> None:
    problem, warp = _fa4_problem()
    plan = SimpleNamespace(
        ii=67,
        length=149,
        warp_sets={node: (group,) for node, group in warp.items()},
        cycles={node: 0 for node in warp},
        experiment_inputs=_FREE_INPUTS,
    )
    solution = tmp_path / "solution.json"
    solution.write_text(json.dumps({"wall_s": 12.5}))
    template = tmp_path / "template.json"
    _write_template(template, warp)
    monkeypatch.setattr(
        strategy_report,
        "load_schedule_context",
        lambda *args: (problem, plan),
    )
    monkeypatch.setattr(
        strategy_report,
        "solve_joint_audit",
        lambda *args, **kwargs: (object(), "sat"),
    )

    strategy_report.main(
        [
            "curated_ddg.json",
            str(solution),
            "--fa4-template",
            str(template),
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert report["fa4_template_refit"]["verdict"] == "sat"
    assert report["fa4_template_refit"]["ii"] == 67
    assert report["fa4_template_refit"]["length"] == 149
    assert report["fa4_template_refit"]["audit_instrument"] is True
    assert report["fa4_template_refit"]["experiment_inputs"] == _FREE_INPUTS
    assert "timeout_s" not in report["fa4_template_refit"]
    assert report["fa4_optimal_set_member"] is True


def test_classify_backward_two_group_pingpong() -> None:
    problem, warp = _problem(three_compute_groups=False)

    report = classify(problem, warp, None, STAGING_MANIFEST)

    assert report["bwd_2wg_pingpong"] is True
    assert report["bwd_3wg_fa4"] is False
    assert report["compute_groups"] == [2, 3]


def test_classify_backward_three_group_fa4() -> None:
    problem, warp = _problem(three_compute_groups=True)

    report = classify(problem, warp, None, STAGING_MANIFEST)

    assert report["bwd_2wg_pingpong"] is False
    assert report["bwd_3wg_fa4"] is True
    assert report["compute_groups"] == [2, 3, 4]
    assert report["exp_tmem_sources"] == {2: [8], 3: [9]}
    assert report["exp_reads_tmem_in_group"] is True
    assert report["exp_tmem_sources_in_group"] == {2: [8], 3: [9]}


def test_classify_backward_three_group_requires_each_exp_to_read_tmem_in_group(
) -> None:
    problem, warp = _problem(three_compute_groups=True)
    warp[8] = 4

    report = classify(problem, warp, None, STAGING_MANIFEST)

    assert report["bwd_3wg_fa4"] is False
    assert report["exp_tmem_sources"] == {2: [8], 3: [9]}
    assert report["exp_reads_tmem_in_group"] is False
    assert report["exp_tmem_sources_in_group"] == {2: [], 3: [9]}


def test_classify_backward_requires_all_tmem_sources_in_exp_group() -> None:
    problem, warp = _problem(three_compute_groups=True)
    problem.nodes[12] = _node(12, "ttng.tmem_load", "TMEM")
    problem.regs[12] = 1
    problem.edges.append(Edge(12, 10, 0, 1))
    warp[12] = 4

    report = classify(problem, warp, None, STAGING_MANIFEST)

    assert report["bwd_3wg_fa4"] is False
    assert report["exp_tmem_sources"] == {2: [8, 12], 3: [9]}
    assert report["exp_tmem_sources_in_group"] == {2: [8], 3: [9]}
    assert report["exp_reads_tmem_in_group"] is False


def test_scalar_reduce_offset_is_not_reduction_staging() -> None:
    problem, warp = _problem(three_compute_groups=False)
    problem.nodes[8] = _node(8, "arith.addi")
    problem.regs[8] = 1
    problem.edges = []
    warp[8] = 3
    manifest = {
        "loops": [
            {
                "nodes": STAGING_MANIFEST["loops"][0]["nodes"],
                "edges": [
                    {"src": 4, "dst": 6, "action": "removed"},
                    {"src": 8, "dst": 7, "action": "removed"},
                ],
            }
        ]
    }

    report = classify(problem, warp, None, manifest)

    assert report["bwd_2wg_pingpong"] is False
    assert report["reduction_staging_nodes"] == [4]


def test_real_bwd_subtiled_fixture_binds_each_exp_to_its_tmem_source(
    bwd_subtiled_fixture,
) -> None:
    problem, baseline_warp = bwd_subtiled_fixture

    report = classify(problem, baseline_warp, None, problem.curation_manifest)

    assert report["exp_tmem_sources"] == {17: [12], 55: [50]}
    assert report["exp_reads_tmem_in_group"] is True
    assert report["reduction_staging_nodes"] == [34, 72]


def test_real_bwd_subtiled_fixture_classifies_two_group_partition(
    bwd_subtiled_fixture,
) -> None:
    problem, _ = bwd_subtiled_fixture
    warp = {node_id: 2 for node_id in problem.nodes}
    for node_id in problem.variable_latency:
        warp[node_id] = 0
    for node_id, node in problem.nodes.items():
        if "mma" in node.op_kind:
            warp[node_id] = 1
    for node_id in (50, 55, 72):
        warp[node_id] = 3

    report = classify(problem, warp, None, problem.curation_manifest)

    assert report["compute_groups"] == [2, 3]
    assert report["exp_reads_tmem_in_group"] is True
    assert report["bwd_2wg_pingpong"] is True
    assert report["bwd_3wg_fa4"] is False


def test_real_bwd_subtiled_fixture_classifies_three_group_partition(
    bwd_subtiled_fixture,
) -> None:
    problem, _ = bwd_subtiled_fixture
    warp = {node_id: 4 for node_id in problem.nodes}
    for node_id in problem.variable_latency:
        warp[node_id] = 0
    for node_id, node in problem.nodes.items():
        if "mma" in node.op_kind:
            warp[node_id] = 1
    for group, (exp_node, source) in enumerate(((17, 12), (55, 50)), start=2):
        warp[exp_node] = group
        warp[source] = group

    report = classify(problem, warp, None, problem.curation_manifest)

    assert report["compute_groups"] == [2, 3, 4]
    assert report["exp_reads_tmem_in_group"] is True
    assert report["bwd_2wg_pingpong"] is False
    assert report["bwd_3wg_fa4"] is True


def test_classify_report_carries_criteria_block() -> None:
    """The report states the reading of the paper's prose it measured."""
    problem, warp = _problem(three_compute_groups=False)

    report = classify(problem, warp, None, STAGING_MANIFEST, "a" * 64)

    criteria = report["criteria"]
    assert criteria["tmem_source_search_max_depth"] == 4
    assert "exp2" in criteria["softmax_marker"]
    assert criteria["curation_manifest"] == "a" * 64
    assert report["curation_manifest_sha256"] == "a" * 64


def test_classify_without_manifest_degrades_staging_to_null() -> None:
    """Staging is an observation read from the manifest, never a gate."""
    problem, warp = _problem(three_compute_groups=False)

    report = classify(problem, warp)

    assert report["reduction_staging_nodes"] is None
    assert report["reduction_staging_group_counts"] is None
    # Undecidable without the staging observation, not decided negative.
    assert report["bwd_2wg_pingpong"] is None
    assert report["bwd_3wg_fa4"] is None
    assert "unavailable without a curation manifest" in (
        report["criteria"]["staging_rule"]
    )
    assert report["criteria"]["curation_manifest"] == "absent"
    assert report["curation_manifest_sha256"] is None


def test_warp_set_labels_group_nodes_by_identical_warp_sets() -> None:
    """The classifier's 'group' is the warp set; overlap is not sameness."""
    labels = strategy_report.warp_set_labels(
        {0: (2,), 1: (0, 1), 2: (2,), 3: (1, 2)}
    )

    assert labels[0] == labels[2]
    assert len({labels[0], labels[1], labels[3]}) == 3
