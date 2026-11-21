import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_joint_solver.curate_ddg import CurationError, curate, dumps

EXAMPLES = Path(__file__).resolve().parents[2] / "sched2tlx" / "examples"
GOLDEN = Path(__file__).resolve().parent / "golden_curated"

CANONICAL = {
    "fwd": (EXAMPLES / "case3_FA_fp16" / "ddg.json",
            EXAMPLES / "case3_FA_fp16" / "schedule_graph.json"),
    "fwd_subtiled": (EXAMPLES / "case3_FA_fp16_subtiled" / "ddg.json",
                     EXAMPLES / "case3_FA_fp16_subtiled" / "schedule_graph.json"),
    "bwd": (EXAMPLES / "case4_FA_bwd" / "ddg_hd128.json",
            EXAMPLES / "case4_FA_bwd" / "schedule_graph_hd128.json"),
}


def _write_raw(tmp_path, nodes, edges, ops=None, loop_extra=None):
    prepared = []
    ops_table = dict(ops or {})
    for values in nodes:
        values = values.copy()
        result_types = values.pop("result_types", ["i32"])
        node = {
            "op_ref": f"op_{values['id']}",
            "op_kind": "arith.addi",
            "pipeline": "NONE",
            "latency": 0,
            "occupancy": 0,
            "min_warps": 1,
            **values,
        }
        prepared.append(node)
        ops_table.setdefault(node["op_ref"], {"result_types": result_types})
    loop = {"trip_count": 4, "ddg": {"nodes": prepared, "edges": edges}}
    loop.update(loop_extra or {})
    path = tmp_path / "ddg.json"
    path.write_text(
        json.dumps(
            {"schema_version": "ddg-0.1", "ops": ops_table, "loops": [loop]}
        )
    )
    return path


def _write_graph(tmp_path, entries, name="schedule_graph.json"):
    path = tmp_path / name
    path.write_text(
        json.dumps(
            {"loops": [{"schedule_loop": {"graph": {"nodes": entries}}}]}
        )
    )
    return path


def _entry(node_id, warp_group=0, op_kind="arith.addi", op_ref=None):
    return {
        "id": node_id,
        "op_ref": op_ref if op_ref is not None else f"op_{node_id}",
        "op_kind": op_kind,
        "warp_group": warp_group,
    }


def _loop(curated):
    return curated["loops"][0]["ddg"]


def _nodes_by_id(curated):
    return {node["id"]: node for node in _loop(curated)["nodes"]}


# --- C1: emitter-infra membership ------------------------------------------


def test_curate_drops_baseline_infra_nodes_and_edges(tmp_path):
    ddg = _write_raw(
        tmp_path,
        [{"id": 0}, {"id": 1}, {"id": 2}],
        [{"src": 0, "dst": 1, "latency": 1}, {"src": 1, "dst": 2, "latency": 2}],
    )
    graph = _write_graph(
        tmp_path, [_entry(0, -1), _entry(1), _entry(2)]
    )

    curated, manifest = curate(ddg, graph)

    assert set(_nodes_by_id(curated)) == {1, 2}
    assert [(e["src"], e["dst"]) for e in _loop(curated)["edges"]] == [(1, 2)]
    records = manifest["loops"][0]["nodes"]
    dropped = [r for r in records if r["action"] == "dropped"]
    assert [r["id"] for r in dropped] == [0]
    assert dropped[0]["reason"] == "emitter-infra (baseline warp_group<0)"
    removed = [
        r for r in manifest["loops"][0]["edges"] if r["action"] == "removed"
    ]
    assert [(r["src"], r["dst"]) for r in removed] == [(0, 1)]


def test_curate_rewires_kept_chain_through_dropped_nodes(tmp_path):
    ddg = _write_raw(
        tmp_path,
        [{"id": 0}, {"id": 1}, {"id": 2}],
        [
            {"src": 0, "dst": 1, "distance": 1, "latency": 3},
            {"src": 1, "dst": 2, "distance": 2, "latency": 4},
        ],
    )
    graph = _write_graph(tmp_path, [_entry(0), _entry(1, -1), _entry(2)])

    curated, manifest = curate(ddg, graph)

    edges = _loop(curated)["edges"]
    assert len(edges) == 1
    # I9: d and delta add up along the dropped chain.
    assert (edges[0]["src"], edges[0]["dst"]) == (0, 2)
    assert edges[0]["distance"] == 3
    assert edges[0]["latency"] == 7
    synthesized = [
        r for r in manifest["loops"][0]["edges"] if r["action"] == "synthesized"
    ]
    assert synthesized[0]["reason"] == "rewired_through_dropped_chain"
    assert synthesized[0]["via"] == [1]


def test_curate_rejects_non_scalar_infra_result(tmp_path):
    ddg = _write_raw(
        tmp_path,
        [{"id": 0, "result_types": ["tensor<4xf32>"]}, {"id": 1}],
        [{"src": 0, "dst": 1}],
    )
    graph = _write_graph(tmp_path, [_entry(0, -1), _entry(1)])

    with pytest.raises(CurationError, match="must be scalar"):
        curate(ddg, graph)


def test_strict_baseline_requires_complete_ddg_coverage(tmp_path):
    ddg = _write_raw(tmp_path, [{"id": 0}, {"id": 1}], [])
    graph = _write_graph(tmp_path, [_entry(0)])

    with pytest.raises(CurationError, match="missing DDG nodes"):
        curate(ddg, graph)

    curated, _ = curate(ddg, graph, strict_baseline=False)
    assert set(_nodes_by_id(curated)) == {0, 1}


def test_strict_baseline_rejects_mismatched_node_identity(tmp_path):
    ddg = _write_raw(tmp_path, [{"id": 0}], [])
    graph = _write_graph(tmp_path, [_entry(0, op_ref="op_other")])

    with pytest.raises(CurationError, match="op_ref does not match"):
        curate(ddg, graph)


def test_strict_baseline_requires_node_identity(tmp_path):
    ddg = _write_raw(tmp_path, [{"id": 0}], [])
    graph = _write_graph(tmp_path, [{"id": 0, "warp_group": 0}])

    with pytest.raises(CurationError, match="missing op_ref"):
        curate(ddg, graph)


# --- C2 / C6: token desugaring and blocking --------------------------------


def test_curate_resolves_token_edges_to_plain_dependences(tmp_path):
    ddg = _write_raw(
        tmp_path,
        [
            {"id": 0, "op_kind": "ttng.tmem_store",
             "result_types": ["!ttg.async.token"]},
            {"id": 1, "op_kind": "ttng.tc_gen5_mma", "pipeline": "TC"},
        ],
        [{"src": 0, "dst": 1, "latency": 4}],
    )
    graph = _write_graph(
        tmp_path,
        [_entry(0, op_kind="ttng.tmem_store"),
         _entry(1, op_kind="ttng.tc_gen5_mma")],
    )

    curated, manifest = curate(ddg, graph)

    edge = _loop(curated)["edges"][0]
    # The token class is gone; what survives is an ordinary blocking edge.
    assert set(edge) == {"src", "dst", "distance", "latency", "blocking"}
    assert edge["blocking"] is True
    record = manifest["loops"][0]["edges"][0]
    assert record["resolution"] == "token_to_plain_dependence"
    assert record["blocking_rule"] == "async_token_completion"
    assert manifest["loops"][0]["observations"]["token_edges_resolved"] == 1


# --- C3 / C4: functional unit and RRT --------------------------------------


def test_curate_reassigns_tmem_ports_to_tmem_unit(tmp_path):
    ddg = _write_raw(
        tmp_path,
        [{"id": 0, "op_kind": "ttng.tmem_load", "pipeline": "CUDA",
          "latency": 2, "occupancy": 2}],
        [],
    )
    graph = _write_graph(
        tmp_path, [_entry(0, op_kind="ttng.tmem_load")]
    )

    curated, manifest = curate(ddg, graph)

    assert _nodes_by_id(curated)[0]["pipeline"] == "TMEM"
    record = manifest["loops"][0]["nodes"][0]
    assert record["pipeline"] == {"from": "CUDA", "to": "TMEM"}
    assert record["pipeline_rule"] == "tmem_port_unit"


def test_curate_synthesizes_rrt_from_occupancy(tmp_path):
    ddg = _write_raw(
        tmp_path,
        [{"id": 0, "pipeline": "SFU", "latency": 4, "occupancy": 2}],
        [],
    )
    graph = _write_graph(tmp_path, [_entry(0)])

    curated, manifest = curate(ddg, graph)

    node = _nodes_by_id(curated)[0]
    assert node["rrt"] == {"SFU": [1, 1, 0, 0]}
    # R4: occupancy stays as an observation field.
    assert node["occupancy"] == 2
    assert manifest["loops"][0]["nodes"][0]["rrt"] == "synthesized_from_occupancy"


# --- C7 / C8: spill cost, variable latency, streaming ----------------------


def test_curate_spillcost_rules(tmp_path):
    ddg = _write_raw(
        tmp_path,
        [
            {"id": 0, "spill_cost": 7},
            {"id": 1, "result_types": ["tensor<8xf32>"]},
            {"id": 2, "op_kind": "ttg.local_alloc",
             "result_types": ["!ttg.memdesc<8xf32, #ttg.shared_memory>"]},
        ],
        [],
    )
    graph = _write_graph(
        tmp_path,
        [_entry(0), _entry(1), _entry(2, op_kind="ttg.local_alloc")],
    )

    curated, manifest = curate(ddg, graph)

    nodes = _nodes_by_id(curated)
    assert nodes[0]["spill_cost"] == 7
    assert nodes[1]["spill_cost"] == 30
    assert nodes[2]["spill_cost"] == 0
    rules = {
        r["id"]: r["spill_cost"]["rule"]
        for r in manifest["loops"][0]["nodes"]
        if r["action"] == "kept"
    }
    assert rules == {
        0: "explicit",
        1: "default_regs_producer",
        2: "memory_backed_zero",
    }


def test_curate_variable_latency_and_streaming_flags(tmp_path):
    ddg = _write_raw(
        tmp_path,
        [
            {"id": 0, "op_kind": "tt.descriptor_load", "pipeline": "TMA"},
            {"id": 1, "op_kind": "tt.descriptor_load", "pipeline": "TMA"},
            {"id": 2, "op_kind": "ttng.tmem_store", "pipeline": "TMEM",
             "result_types": ["!ttg.async.token"]},
        ],
        [{"src": 2, "dst": 1}],
    )
    graph = _write_graph(
        tmp_path,
        [
            _entry(0, op_kind="tt.descriptor_load"),
            _entry(1, op_kind="tt.descriptor_load"),
            _entry(2, op_kind="ttng.tmem_store"),
        ],
    )

    curated, _ = curate(ddg, graph)

    nodes = _nodes_by_id(curated)
    assert nodes[0]["variable_latency"] is True
    assert nodes[1]["variable_latency"] is True
    assert nodes[0]["streaming"] is True
    # Any incoming curated edge -- including a desugared token edge --
    # disqualifies streaming, with no exemption (sec 5.3, literally).
    assert nodes[1]["streaming"] is False


# --- C5 / B5: footprint ownership and liveness cover -----------------------


def test_curate_resolves_footprint_ownership(tmp_path):
    ddg = _write_raw(
        tmp_path,
        [
            {"id": 0, "op_kind": "tt.descriptor_load", "pipeline": "TMA",
             "result_types": ["tensor<64x64xf16>"]},
            {"id": 1, "op_kind": "ttg.local_alloc",
             "result_types": ["!ttg.memdesc<64x64xf16, #ttg.shared_memory>"]},
        ],
        [{"src": 0, "dst": 1}],
    )
    graph = _write_graph(
        tmp_path,
        [_entry(0, op_kind="tt.descriptor_load"),
         _entry(1, op_kind="ttg.local_alloc")],
    )

    curated, manifest = curate(ddg, graph)

    nodes = _nodes_by_id(curated)
    footprints = manifest["loops"][0]["footprints"]
    assert len(footprints) == 1
    record = footprints[0]
    assert record["kind"] == "smem"
    assert record["members"] == [0, 1]
    # Exactly one owner carries the amount; the other member is zero, so the
    # solver's literal per-value sum stays physically correct (G6).
    owner = record["owner"]
    assert nodes[owner]["smem_footprint"] == record["amount"]
    other = next(m for m in record["members"] if m != owner)
    assert nodes[other]["smem_footprint"] == 0
    assert record["amount_basis"]["bytes"] == record["amount"]


def test_curate_verifies_b5_liveness_cover(tmp_path):
    """The accumulator's liveness reaches its last consumer through the
    desugared completion edge, so no extra edge is needed."""
    ddg = _write_raw(
        tmp_path,
        [
            {"id": 0, "op_kind": "ttng.tmem_alloc",
             "result_types": ["!ttg.memdesc<128x128xf32, #ttng.tensor_memory>"]},
            {"id": 1, "op_kind": "ttng.tmem_store",
             "result_types": ["!ttg.async.token"]},
        ],
        [{"src": 0, "dst": 1}],
        ops={
            "op_1": {
                "result_types": ["!ttg.async.token"],
                "operands": [{"op": "op_0"}],
            }
        },
    )
    graph = _write_graph(
        tmp_path,
        [_entry(0, op_kind="ttng.tmem_alloc"),
         _entry(1, op_kind="ttng.tmem_store")],
    )

    _curated, manifest = curate(ddg, graph)

    tmem = [f for f in manifest["loops"][0]["footprints"] if f["kind"] == "tmem"]
    assert tmem, "expected a TMEM storage object"
    assert all(f["liveness_cover"] == "desugared_use_edges" for f in tmem)
    assert manifest["loops"][0]["observations"]["edges_synthesized"] == 0


# --- C9: dropped metadata ---------------------------------------------------


def test_curate_drops_loop_metadata_and_dump_fields(tmp_path):
    ddg = _write_raw(
        tmp_path,
        [{"id": 0, "self_latency": 3, "is_super_node": True}],
        [{"src": 0, "dst": 0, "kind": "loop_carried", "distance": 1}],
        loop_extra={"min_ii": 1204, "res_mii": 1204, "rec_mii": 1033},
    )
    graph = _write_graph(tmp_path, [_entry(0)])

    curated, manifest = curate(ddg, graph)

    loop = curated["loops"][0]
    assert "min_ii" not in loop
    node = _nodes_by_id(curated)[0]
    assert "self_latency" not in node and "is_super_node" not in node
    assert "kind" not in _loop(curated)["edges"][0]
    observations = manifest["loops"][0]["observations"]
    assert observations["dropped_loop_fields"] == {
        "min_ii": 1204,
        "res_mii": 1204,
        "rec_mii": 1033,
    }
    assert observations["dropped_field_names"] == [
        "node.self_latency",
        "node.is_super_node",
        "edge.kind",
        "edge.src_result_idx",
    ]


# --- determinism, hash chain, golden regression ----------------------------


def test_curate_is_deterministic_and_hash_chained(tmp_path):
    ddg, graph = CANONICAL["fwd"]

    first_curated, first_manifest = curate(ddg, graph)
    second_curated, second_manifest = curate(ddg, graph)

    assert dumps(first_curated) == dumps(second_curated)
    assert dumps(first_manifest) == dumps(second_manifest)
    source = first_curated["curation_source"]
    assert set(source) == {
        "ddg_sha256",
        "baseline_graph_sha256",
        "curator_sources_sha256",
    }
    assert all(len(value) == 64 for value in source.values())
    # The manifest carries the identical triple so the chain closes.
    assert first_manifest["curation_source"] == source


@pytest.mark.parametrize("case", sorted(CANONICAL))
def test_curated_fixture_matches_golden(case):
    """Byte-for-byte regression anchor: any curation-rule drift shows up as a
    diff against the committed golden pair."""
    ddg, graph = CANONICAL[case]
    curated, manifest = curate(ddg, graph)

    golden_dir = GOLDEN / case
    assert dumps(curated) == (golden_dir / f"{case}.curated_ddg.json").read_text()
    assert dumps(manifest) == (
        golden_dir / f"{case}.curation_manifest.json"
    ).read_text()
