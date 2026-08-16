"""M1 coverage for the prepared regime's input-preparation layer.

The gates these tests encode are P1 (normalization health on the prepared
``fwd_subtiled`` pool) and P5's control half: the literal path must still
collapse on the very same fixture, or the comparison proves nothing.

Nothing here imports the SMT layer, so the whole file runs on a host that has
SCIP but no yices shared library.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_joint_solver.ddg import load_problem
from paper_joint_solver.machine import MachineModel
from paper_joint_solver.normalize import normalize_costs
from paper_joint_solver.prepared import (
    apply_floor,
    build_parser,
    assemble_pool,
    build_problem,
    CASE_U,
    cluster_representatives,
    ISSUE_WIDTH,
    load_ops_table,
    load_prepared_problem,
    node_cost,
    normalize_prepared,
    PREPARED_PROBE_SCHEMA,
    PREPARED_SOLUTION_SCHEMA,
    PreparationError,
    preparation_manifest,
    preparation_provenance,
    read_curated,
    tensor_elements,
)

GOLDEN = Path(__file__).resolve().parent / "golden_curated"
EXAMPLES = Path(__file__).resolve().parents[2] / "sched2tlx" / "examples"

# The raw dumps the committed golden fixtures were curated from.  `bwd` comes
# from the head-dim-128 backward dump, not the sub-tiled one beside it; the
# hash check in `load_ops_table` is what establishes that, so these pairings
# are asserted rather than assumed.
RAW_DUMPS = {
    "fwd": EXAMPLES / "case3_FA_fp16" / "ddg.json",
    "fwd_subtiled": EXAMPLES / "case3_FA_fp16_subtiled" / "ddg.json",
    "bwd": EXAMPLES / "case4_FA_bwd" / "ddg_hd128.json",
}

# P1's two thresholds.
MAX_COLLAPSE_RATIO = 0.20
MIN_SURVIVOR_SPREAD = 8


def curated_path(case: str) -> Path:
    return GOLDEN / case / f"{case}.curated_ddg.json"


@pytest.fixture(scope="module")
def fwd_subtiled():
    return load_prepared_problem(
        curated_path("fwd_subtiled"),
        RAW_DUMPS["fwd_subtiled"],
        u=CASE_U["fwd_subtiled"],
    )


def make_node(**overrides):
    node = {
        "id": 1,
        "op_ref": "op_1",
        "op_kind": "arith.mulf",
        "pipeline": "CUDA",
        "latency": 0,
        "occupancy": 0,
        "rrt": {},
        "min_warps": 1,
        "regs": 0,
        "spill_cost": 0,
        "smem_footprint": 0,
        "tmem_footprint": 0,
        "variable_latency": False,
        "streaming": False,
    }
    node.update(overrides)
    return node


# --------------------------------------------------------------------------
# A layer
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("result_type", "expected"),
    [
        ("tensor<64x128xf32, #ttg.linear<{}>>", 64 * 128),
        ("tensor<64xf32, #ttg.slice<{}>>", 64),
        ("tensor<2x4x8xf16>", 64),
        ("!ttg.async.token", None),
        ("!ttg.memdesc<64x128xf16, #ttg.nvmma_shared<{}>>", None),
        ("f32", None),
        ("", None),
    ],
)
def test_tensor_elements_reads_only_real_tensors(result_type, expected):
    assert tensor_elements(result_type) == expected


def test_elementwise_cuda_ops_are_repriced_by_issue_width():
    node = make_node(op_kind="arith.mulf", latency=34, occupancy=32)
    cost, basis = node_cost(node, ["tensor<64x64xf32>"])

    assert basis == "elementwise"
    assert cost == 4096 // ISSUE_WIDTH == 32


def test_reprice_rescues_narrow_tiles_the_dump_charges_nothing_for():
    """A 64-wide multiply costs 0 in the dump and one issue slot here."""
    node = make_node(op_kind="arith.addf", latency=0, occupancy=0)

    assert node_cost(node, ["tensor<64xf32>"]) == (1, "elementwise")


def test_non_elementwise_ops_keep_their_measured_occupancy():
    reduce_node = make_node(
        op_kind="tt.reduce", pipeline="CUDA", latency=115, occupancy=76
    )
    exp_node = make_node(
        op_kind="math.exp2", pipeline="SFU", latency=285, occupancy=32
    )

    assert node_cost(reduce_node, ["tensor<64xf32>"]) == (76, "occupancy")
    assert node_cost(exp_node, ["tensor<64x64xf32>"]) == (32, "occupancy")


def test_scalar_and_address_arithmetic_is_charged_zero():
    node = make_node(op_kind="arith.addi", latency=1, occupancy=1)

    assert node_cost(node, ["i32"]) == (0, "scalar-zero")


def test_a3_async_mma_token_is_not_mistaken_for_a_scalar():
    """The MMA is not elementwise, so the scalar rule must never reach it."""
    mma = make_node(
        op_kind="ttng.tc_gen5_mma", pipeline="TC", latency=900, occupancy=128
    )

    assert node_cost(mma, ["!ttg.async.token"]) == (128, "occupancy")


def test_a4_rejects_a_tmem_op_folded_onto_another_unit():
    ddg = {
        "nodes": [
            make_node(id=1, op_kind="ttng.tmem_load", pipeline="CUDA")
        ],
        "edges": [],
    }

    with pytest.raises(PreparationError, match="TMEM functional unit"):
        assemble_pool(ddg, {"op_1": {"result_types": ["tensor<64xf32>"]}})


def test_a1_pool_is_one_entry_per_node_plus_non_zero_edges_and_spills():
    ddg = {
        "nodes": [
            make_node(id=1, op_kind="arith.mulf"),
            make_node(id=2, op_kind="tt.reduce", latency=8, occupancy=8,
                      spill_cost=30),
            make_node(id=3, op_kind="tt.descriptor_load", pipeline="TMA",
                      latency=9, occupancy=5, streaming=True),
        ],
        "edges": [
            {"src": 1, "dst": 2, "distance": 0, "latency": 0,
             "blocking": False},
            {"src": 2, "dst": 3, "distance": 0, "latency": 7,
             "blocking": False},
            {"src": 3, "dst": 1, "distance": 1, "latency": 9,
             "blocking": False},
        ],
    }
    ops = {
        "op_1": {"result_types": ["tensor<128xf32>"]},
        "op_2": {"result_types": ["tensor<64xf32>"]},
        "op_3": {"result_types": ["tensor<64x128xf16>"]},
    }

    entries = assemble_pool(ddg, ops)

    # Three nodes, one non-zero edge (the zero one and the streaming-zeroed
    # one are both absent), one non-zero spill.
    assert [(e.kind, e.key, e.cost) for e in entries] == [
        ("node", 1, 1),
        ("node", 2, 8),
        ("node", 3, 5),
        ("edge", 1, 7),
        ("spill", 2, 30),
    ]


def test_pool_rejects_a_node_the_raw_dump_does_not_describe():
    ddg = {"nodes": [make_node(id=1, op_ref="op_missing")], "edges": []}

    with pytest.raises(PreparationError, match="absent from the raw dump"):
        assemble_pool(ddg, {})


def test_prepared_pool_is_smaller_than_the_literal_one(fwd_subtiled):
    literal = load_problem(curated_path("fwd_subtiled"))

    assert len(literal.normalization_cost_pairs) == 176
    assert len(fwd_subtiled.normalization.entries) == 129


# --------------------------------------------------------------------------
# B layer
# --------------------------------------------------------------------------


def test_b1_clusters_within_ten_percent_of_the_anchor():
    # 100 anchors; 110 is exactly at the boundary and joins, 111 does not.
    representatives, clusters = cluster_representatives([100, 110, 111])

    assert [members for _, members in clusters] == [(100, 110), (111,)]
    assert representatives == {100: 105, 110: 105, 111: 111}


def test_b1_anchors_on_the_cluster_head_not_the_running_mean():
    """A chain of 10% steps must not merge into one unbounded cluster."""
    _, clusters = cluster_representatives([100, 109, 118, 127])

    assert [members for _, members in clusters] == [(100, 109), (118, 127)]


def test_b1_representative_is_the_integer_mean_rounded_half_up():
    representatives, _ = cluster_representatives([10, 11])

    assert representatives == {10: 11, 11: 11}


def test_b1_ignores_zero_and_never_invents_a_representative_for_it():
    representatives, clusters = cluster_representatives([0, 0, 5])

    assert clusters == [(5, (5,))]
    assert representatives == {5: 5}


def test_b2_floors_everything_below_a_thirty_secondth_of_the_maximum():
    assert apply_floor([320, 10, 9, 0]) == [320, 10, 0, 0]


def test_b2_is_a_no_op_on_an_empty_or_flat_pool():
    assert apply_floor([]) == []
    assert apply_floor([7, 7, 7]) == [7, 7, 7]


def test_p1_prepared_fwd_subtiled_normalization_is_healthy(fwd_subtiled):
    health = fwd_subtiled.normalization.health()

    assert health["collapse_ratio"] <= MAX_COLLAPSE_RATIO
    assert health["survivor_spread"] >= MIN_SURVIVOR_SPREAD


def test_p5_control_the_literal_pool_still_collapses_on_this_fixture():
    """Without this the P1 result would say nothing about the change."""
    literal = load_problem(curated_path("fwd_subtiled"))
    pairs = literal.normalization_cost_pairs
    positive = [(raw, scaled) for raw, scaled in pairs if raw > 0]
    collapsed = [scaled for _, scaled in positive if scaled == 0]

    assert literal.normalization_f == 394
    assert len(collapsed) / len(positive) > MAX_COLLAPSE_RATIO


def test_b4_never_leaves_the_f_optimal_face(fwd_subtiled):
    normalization = fwd_subtiled.normalization
    costs = normalization.processed
    scaled = normalization.scaled

    assert 1 <= sum(scaled) <= normalization.u
    for i, cost_i in enumerate(costs):
        for j, cost_j in enumerate(costs):
            cross = cost_i * scaled[j] - cost_j * scaled[i]
            assert abs(cross) <= normalization.objective


def test_b4_second_stage_never_worsens_the_first(fwd_subtiled):
    normalization = fwd_subtiled.normalization

    assert sum(normalization.scaled) >= sum(normalization.first_stage_scaled)


def test_b4_is_what_makes_the_prepared_pool_survive_u300(fwd_subtiled):
    """The min-F face holds both a degenerate and a rich point here."""
    normalization = fwd_subtiled.normalization
    first = normalization.first_stage_scaled
    positive = normalization.positive_entries
    first_collapsed = [index for index in positive if first[index] == 0]

    assert sum(first) < sum(normalization.scaled)
    assert len(first_collapsed) / len(positive) > MAX_COLLAPSE_RATIO


def test_b3_is_the_unmodified_paper_zlp(fwd_subtiled):
    normalization = fwd_subtiled.normalization
    replay = normalize_costs(list(normalization.processed), u=normalization.u)

    assert replay.objective == normalization.objective


def test_b5_records_the_per_case_budget():
    assert CASE_U == {
        "fwd_subtiled": 300,
        "fwd": 150,
        "bwd": 300,
        "bwd_lr4096": 300,
    }


# --------------------------------------------------------------------------
# Curated input binding and the assembled Problem
# --------------------------------------------------------------------------


def test_raw_dump_must_be_the_one_the_curated_file_records():
    with pytest.raises(PreparationError, match="was curated from"):
        load_ops_table(RAW_DUMPS["fwd"], curated_path("fwd_subtiled"))


@pytest.mark.parametrize("case", sorted(RAW_DUMPS))
def test_every_golden_fixture_binds_to_its_recorded_raw_dump(case):
    assert load_ops_table(RAW_DUMPS[case], curated_path(case))


def test_read_curated_rejects_an_unknown_field(tmp_path):
    payload = json.loads(curated_path("fwd_subtiled").read_text())
    payload["loops"][0]["ddg"]["nodes"][0]["colour"] = "green"
    path = tmp_path / "curated_ddg.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="unknown curated field"):
        read_curated(path, MachineModel())


def test_problem_reserves_each_unit_for_the_whole_normalized_cost(
    fwd_subtiled,
):
    problem = fwd_subtiled.problem

    for node_id, rows in problem.rrt.items():
        span = problem.lat[node_id]
        assert problem.occ[node_id] == span
        if not span or problem.nodes[node_id].pipeline == "NONE":
            assert rows == {}
            continue
        assert rows == {problem.nodes[node_id].pipeline: (1,) * span}


def test_problem_keeps_the_tmem_ports_off_the_alu_column(fwd_subtiled):
    problem = fwd_subtiled.problem
    tmem_nodes = [
        node_id
        for node_id, node in problem.nodes.items()
        if "tmem_load" in node.op_kind or "tmem_store" in node.op_kind
    ]

    assert tmem_nodes
    for node_id in tmem_nodes:
        assert set(problem.rrt[node_id]) <= {"TMEM"}


def test_problem_zeroes_the_out_edges_of_streaming_producers(fwd_subtiled):
    problem = fwd_subtiled.problem

    assert problem.streaming
    for edge in problem.edges:
        if edge.src in problem.streaming:
            assert problem.edge_latency(edge) == 0


def test_problem_reports_the_prepared_normalization_stats(fwd_subtiled):
    problem = fwd_subtiled.problem
    normalization = fwd_subtiled.normalization

    assert problem.normalization_u == normalization.u
    assert problem.normalization_f == normalization.objective
    assert problem.normalization_cost_pairs == tuple(
        zip(normalization.processed, normalization.scaled)
    )


def test_build_problem_covers_every_node_and_edge(fwd_subtiled):
    problem = fwd_subtiled.problem

    assert set(problem.lat) == set(problem.nodes) == set(problem.spill)
    assert set(problem.edge_lat) == {edge.index for edge in problem.edges}


def test_build_problem_is_deterministic():
    machine = MachineModel()
    graph = read_curated(curated_path("fwd_subtiled"), machine)
    ops = load_ops_table(
        RAW_DUMPS["fwd_subtiled"], curated_path("fwd_subtiled")
    )
    normalization = normalize_prepared(assemble_pool(graph.ddg, ops), u=300)
    first = build_problem(graph, normalization, machine)

    graph_again = read_curated(curated_path("fwd_subtiled"), machine)
    second = build_problem(graph_again, normalization, machine)

    assert first.lat == second.lat
    assert first.rrt == second.rrt
    assert first.edge_lat == second.edge_lat
    assert first.spill == second.spill


# --------------------------------------------------------------------------
# Disclosure surface
# --------------------------------------------------------------------------


def test_manifest_discloses_every_assumption_and_every_pool_entry(
    fwd_subtiled,
):
    manifest = preparation_manifest(fwd_subtiled)

    assert manifest["regime"] == "prepared"
    assert len(manifest["prepared_sources_sha256"]) == 64
    assert manifest["raw_ddg_sha256"] == fwd_subtiled.raw_ddg_sha256
    assert set(manifest["assumptions"]) == {
        "a1_pool_members",
        "a2_issue_width",
        "a2_elementwise_dialects",
        "a3_async_tokens_keep_measured_cost",
        "a4_tmem_functional_unit",
        "b1_cluster_tolerance",
        "b2_floor_divisor",
        "b3_zlp",
        "b4_second_objective",
        "b5_u",
    }
    assert len(manifest["pool"]) == len(fwd_subtiled.normalization.entries)
    assert json.dumps(manifest)  # the block has to be serializable as-is


def test_manifest_records_the_basis_of_every_priced_node(fwd_subtiled):
    bases = {
        row["basis"] for row in preparation_manifest(fwd_subtiled)["pool"]
    }

    assert bases == {"occupancy", "elementwise", "edge-latency", "spill"}


def test_prepared_artifacts_declare_their_own_schema():
    assert PREPARED_SOLUTION_SCHEMA == "prepared-joint-solution-v1"


# --------------------------------------------------------------------------
# M2/M3 surface: CLI contract and the provenance block
# --------------------------------------------------------------------------


def test_provenance_block_carries_parameters_not_the_pool(fwd_subtiled):
    """The per-entry chain is bulky and lives in its own artifact."""
    provenance = preparation_provenance(fwd_subtiled, 300)

    assert "pool" not in provenance
    assert provenance["b5_u_applied"] == 300
    assert provenance["assumptions"]["b5_u"] == 300
    assert provenance["health"]["pool_size"] == 129


def test_probe_artifacts_declare_their_own_schema():
    assert PREPARED_PROBE_SCHEMA == "prepared-joint-probe-v1"


def test_cli_requires_the_raw_dump_the_shapes_come_from():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["curated.json", "-o", "out.json"])


def test_cli_defaults_match_the_regime_contract():
    args = build_parser().parse_args(
        ["curated.json", "--raw-ddg", "raw.json", "-o", "out.json"]
    )

    assert args.refit_window == 6
    assert args.probe is False
    assert args.num_warps_override is None
    assert args.no_cross_warp is False


def test_literal_cli_gains_no_prepared_flag():
    """`python -m paper_joint_solver` must keep meaning what the paper says.

    Read from disk rather than imported: importing the literal CLI pulls in
    the SMT layer, and this file must stay runnable without it.
    """
    package = Path(__file__).resolve().parents[1] / "paper_joint_solver"
    source = (package / "__main__.py").read_text()

    assert "--raw-ddg" not in source
    assert "prepared" not in source
