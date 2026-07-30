import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_joint_solver.graph_writer import (
    _allocated_buffer_bytes,
    _implicit_channel_barrier_bytes,
    _parse_register_type,
    _parse_tensor_type,
    _ring_depth,
    _snap_num_warps,
    _synthesize_register_channels,
    rewrite_schedule_graph,
    validate,
)

TOOLS = Path(__file__).resolve().parents[2]
SCHED2TLX = TOOLS / "sched2tlx"
BASELINE = SCHED2TLX / "examples" / "case3_FA_fp16" / "schedule_graph.json"
BWD_BASELINE = SCHED2TLX / "examples" / "case4_FA_bwd" / "schedule_graph_hd128.json"
VENV_PY = Path(__file__).resolve().parents[5] / ".venv" / "bin" / "python"
PYTHON = str(VENV_PY) if VENV_PY.exists() else sys.executable


def _emit(graph_path, out_py):
    env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    subprocess.run(
        [PYTHON, "-m", "sched2tlx", str(graph_path), "-o", str(out_py)],
        cwd=str(SCHED2TLX),
        env=env,
        check=True,
        capture_output=True,
    )
    return out_py.read_text()


def _non_comment(src):
    return "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))


def _identity_solution(baseline=BASELINE):
    data = json.loads(baseline.read_text())
    sl = data["loops"][0]["schedule_loop"]
    nodes = sl["graph"]["nodes"]
    cycles = {n["id"]: n["schedule"]["cycle"] for n in nodes}
    warp = {n["id"]: n["warp_group"] for n in nodes if n["warp_group"] >= 0}
    return sl["II"], cycles, warp, max(cycles.values()) + 1


def test_pointer_tensor_channel_shape_is_64_bit():
    assert _parse_tensor_type("tensor<64x!tt.ptr<f32>, #layout>") == ([64], 64)
    assert _parse_register_type("f32") == ([1], 32)


def test_ring_depth_is_never_truncated_to_a_byte_threshold():
    assert _ring_depth(0, 129, 128) == 2
    buffers = [
        {
            "kind": "smem",
            "total_bytes": 24576,
            "merge_group_id": 7,
        },
        {
            "kind": "smem",
            "total_bytes": 16384,
            "merge_group_id": 7,
        },
    ]
    assert _allocated_buffer_bytes(buffers, "smem") == 24576
    schedule_loop = {
        "buffers": [{"id": 1, "count": 2}],
        "graph": {
            "cross_wg_barriers": [
                {"paired_buffer_id": 1},
                {"paired_buffer_id": 1},
            ]
        },
    }
    assert _implicit_channel_barrier_bytes(schedule_loop) == 32


def test_identity_roundtrip_emits_identically(tmp_path):
    ii, cycles, warp, length = _identity_solution()
    out_json = tmp_path / "rewritten.json"
    rewrite_schedule_graph(
        BASELINE,
        out_json,
        ii=ii,
        cycles=cycles,
        warp=warp,
        length=length,
        materialize_all_spills=False,
    )
    base_py = _emit(BASELINE, tmp_path / "base.py")
    rew_py = _emit(out_json, tmp_path / "rewritten.py")
    assert _non_comment(rew_py) == _non_comment(base_py)


def test_identity_roundtrip_validates(tmp_path):
    ii, cycles, warp, length = _identity_solution()
    out_json = tmp_path / "rewritten.json"
    rewrite_schedule_graph(
        BASELINE, out_json, ii=ii, cycles=cycles, warp=warp, length=length
    )
    assert validate(out_json) == []


def test_bwd_tma_register_load_spills_are_materialized():
    data = json.loads(BWD_BASELINE.read_text())
    loop = data["loops"][0]
    nodes = loop["schedule_loop"]["graph"]["nodes"]
    by_id = {node["id"]: node for node in nodes}
    by_id[14]["warp_group"] = 3
    by_id[23]["warp_group"] = 3

    _synthesize_register_channels(data, loop, True)

    barriers = loop["schedule_loop"]["graph"]["cross_wg_barriers"]
    assert {7, 9} <= {barrier["producer_node"] for barrier in barriers}


def test_register_channel_keeps_each_fanout_edge():
    def node(node_id, op_ref, group, cycle):
        return {
            "id": node_id,
            "op_ref": op_ref,
            "op_kind": "arith.addf",
            "pipeline": "CUDA",
            "warp_group": group,
            "latency": 1,
            "schedule": {"cycle": cycle, "stage": 0, "cluster": cycle},
            "produces_buffer": None,
            "consumes_buffers": [],
        }

    data = {
        "ops": {
            "producer": {
                "kind": "arith.addf",
                "operands": [],
                "result_types": ["tensor<4xf32>"],
            },
            "consumer1": {
                "kind": "arith.addf",
                "operands": [{"op": "producer"}],
                "result_types": ["tensor<4xf32>"],
            },
            "consumer2": {
                "kind": "arith.addf",
                "operands": [{"op": "producer"}],
                "result_types": ["tensor<4xf32>"],
            },
        }
    }
    loop = {
        "schedule_loop": {
            "id": 0,
            "II": 4,
            "lower_bound": {"const": 0, "type": "i32"},
            "upper_bound": {"const": 4, "type": "i32"},
            "step": {"const": 1, "type": "i32"},
            "buffers": [],
            "graph": {
                "nodes": [
                    node(0, "producer", 0, 0),
                    node(1, "consumer1", 1, 1),
                    node(2, "consumer2", 1, 2),
                ],
                "edges": [
                    {"src": 0, "dst": 1, "kind": "data", "distance": 0},
                    {"src": 0, "dst": 2, "kind": "data", "distance": 0},
                ],
                "cross_wg_barriers": [],
            },
        }
    }

    _synthesize_register_channels(data, loop, True)

    barriers = loop["schedule_loop"]["graph"]["cross_wg_barriers"]
    assert {
        (barrier["producer_node"], barrier["consumer_node"]) for barrier in barriers
    } == {(0, 1), (0, 2)}
    assert len({barrier["paired_buffer_id"] for barrier in barriers}) == 1


def test_cross_wg_barriers_are_rebuilt_for_buffer_edges(tmp_path):
    ii, cycles, warp, length = _identity_solution(
        SCHED2TLX / "examples" / "case1_simple_gemm" / "schedule_graph.json"
    )
    baseline = SCHED2TLX / "examples" / "case1_simple_gemm" / "schedule_graph.json"
    data = json.loads(baseline.read_text())
    loop = data["loops"][0]["schedule_loop"]
    warp[1] = warp[4]
    out_json = tmp_path / "buffer-edges.json"

    rewrite_schedule_graph(
        baseline,
        out_json,
        ii=ii,
        cycles=cycles,
        warp=warp,
        length=length,
        dependence_edges=loop["graph"]["edges"],
    )

    rewritten = json.loads(out_json.read_text())["loops"][0]["schedule_loop"]
    barriers = rewritten["graph"]["cross_wg_barriers"]
    keys = {
        (barrier["producer_node"], barrier["consumer_node"], barrier["distance"])
        for barrier in barriers
    }
    assert keys == {(0, 1, 0), (3, 4, 0)}
    tma_alloc = next(barrier for barrier in barriers if barrier["producer_node"] == 0)
    assert tma_alloc["paired_buffer_id"] == 0
    assert tma_alloc["expect_bytes"] == 16384


def test_rebuilt_barriers_preserve_distance_despite_cycle_order(tmp_path):
    baseline = SCHED2TLX / "examples" / "case1_simple_gemm" / "schedule_graph.json"
    data = json.loads(baseline.read_text())
    loop = data["loops"][0]["schedule_loop"]
    nodes = loop["graph"]["nodes"]
    cycles = {node["id"]: node["schedule"]["cycle"] for node in nodes}
    warp = {node["id"]: node["warp_group"] for node in nodes}
    warp[1] = warp[4]
    cycles[3] = cycles[4] + 10
    edges = [dict(edge) for edge in loop["graph"]["edges"]]
    next(edge for edge in edges if edge["src"] == 0 and edge["dst"] == 1)[
        "distance"
    ] = 1
    out_json = tmp_path / "distance.json"

    rewrite_schedule_graph(
        baseline,
        out_json,
        ii=loop["II"],
        cycles=cycles,
        warp=warp,
        length=max(cycles.values()) + 1,
        dependence_edges=edges,
    )

    graph = json.loads(out_json.read_text())
    barriers = graph["loops"][0]["schedule_loop"]["graph"]["cross_wg_barriers"]
    distance = {
        (barrier["producer_node"], barrier["consumer_node"]): barrier["distance"]
        for barrier in barriers
    }
    assert distance[0, 1] == 1
    assert distance[3, 4] == 0


def test_parallel_edges_keep_order_distance_and_latency(tmp_path):
    baseline = SCHED2TLX / "examples" / "case1_simple_gemm" / "schedule_graph.json"
    data = json.loads(baseline.read_text())
    loop = data["loops"][0]["schedule_loop"]
    nodes = loop["graph"]["nodes"]
    cycles = {node["id"]: node["schedule"]["cycle"] for node in nodes}
    warp = {node["id"]: node["warp_group"] for node in nodes}
    original = loop["graph"]["edges"]
    parallel = [
        {"src": 0, "dst": 1, "distance": 0, "latency": 101},
        {"src": 0, "dst": 1, "distance": 0, "latency": 202},
        *original[1:],
    ]
    out_json = tmp_path / "parallel.json"

    rewrite_schedule_graph(
        baseline,
        out_json,
        ii=loop["II"],
        cycles=cycles,
        warp=warp,
        length=max(cycles.values()) + 1,
        dependence_edges=parallel,
    )

    rewritten = json.loads(out_json.read_text())["loops"][0]["schedule_loop"]
    assert [
        (edge["src"], edge["dst"], edge["distance"], edge["latency"])
        for edge in rewritten["graph"]["edges"][:2]
    ] == [(0, 1, 0, 101), (0, 1, 0, 202)]


def test_permuted_warp_assignment_validates(tmp_path):
    ii, cycles, warp, length = _identity_solution()
    ids = sorted(set(warp.values()))
    rot = {w: ids[(i + 1) % len(ids)] for i, w in enumerate(ids)}
    permuted = {nid: rot[w] for nid, w in warp.items()}
    out_json = tmp_path / "permuted.json"
    rewrite_schedule_graph(
        BASELINE, out_json, ii=ii, cycles=cycles, warp=permuted, length=length
    )
    assert validate(out_json) == []
    data = json.loads(out_json.read_text())
    loop = data["loops"][0]
    used = {
        n["warp_group"]
        for n in loop["schedule_loop"]["graph"]["nodes"]
        if n["warp_group"] >= 0
    }
    assert used == {w["id"] for w in loop["warp_groups"]}
    assert sorted(used) == list(range(len(used)))


def test_solver_assignment_overrides_negative_baseline_group(tmp_path):
    data = json.loads(BWD_BASELINE.read_text())
    loop = data["loops"][0]["schedule_loop"]
    nodes = loop["graph"]["nodes"]
    cycles = {node["id"]: node["schedule"]["cycle"] for node in nodes}
    warp = {
        node["id"]: (2 if node["warp_group"] < 0 else node["warp_group"])
        for node in nodes
    }
    out_json = tmp_path / "owned-infra.json"

    rewrite_schedule_graph(
        BWD_BASELINE,
        out_json,
        ii=loop["II"],
        cycles=cycles,
        warp=warp,
        length=max(cycles.values()) + 1,
        materialize_all_spills=False,
    )

    rewritten = json.loads(out_json.read_text())["loops"][0]["schedule_loop"]
    by_id = {node["id"]: node for node in rewritten["graph"]["nodes"]}
    assert by_id[35]["warp_group"] == 2


def test_explicit_group_widths_are_validated(tmp_path):
    ii, cycles, warp, length = _identity_solution()
    data = json.loads(BASELINE.read_text())
    widths = {
        group["id"]: group["num_warps"] for group in data["loops"][0]["warp_groups"]
    }
    nodes = data["loops"][0]["schedule_loop"]["graph"]["nodes"]
    masks = {
        node["id"]: tuple(range(_snap_num_warps(node["min_warps"])))
        for node in nodes
        if node["warp_group"] >= 0
    }
    out_json = tmp_path / "widths.json"

    rewrite_schedule_graph(
        BASELINE,
        out_json,
        ii=ii,
        cycles=cycles,
        warp=warp,
        length=length,
        group_widths=widths,
        lane_masks=masks,
    )
    assert validate(out_json) == []

    bad = dict(widths)
    first = next(iter(bad))
    bad[first] = 1 if bad[first] != 1 else 8
    with pytest.raises(ValueError, match="solver width"):
        rewrite_schedule_graph(
            BASELINE,
            tmp_path / "bad-widths.json",
            ii=ii,
            cycles=cycles,
            warp=warp,
            length=length,
            group_widths=bad,
        )

    bad_masks = dict(masks)
    first_node = next(iter(bad_masks))
    bad_masks[first_node] = (1,)
    with pytest.raises(ValueError, match="physical lane mask"):
        rewrite_schedule_graph(
            BASELINE,
            tmp_path / "bad-masks.json",
            ii=ii,
            cycles=cycles,
            warp=warp,
            length=length,
            group_widths=widths,
            lane_masks=bad_masks,
        )


def test_writer_rejects_physical_warp_overflow(tmp_path):
    ii, cycles, warp, length = _identity_solution()
    with pytest.raises(ValueError, match="physical warps"):
        rewrite_schedule_graph(
            BASELINE,
            tmp_path / "too-many-warps.json",
            ii=ii,
            cycles=cycles,
            warp=warp,
            length=length,
            warp_fixed_overhead=21,
        )


def test_writer_rejects_zero_latency_node_at_length(tmp_path):
    baseline = SCHED2TLX / "examples" / "case1_simple_gemm" / "schedule_graph.json"
    ii, cycles, warp, length = _identity_solution(baseline)
    cycles[1] = length

    with pytest.raises(ValueError, match="cycles outside"):
        rewrite_schedule_graph(
            baseline,
            tmp_path / "at-length.json",
            ii=ii,
            cycles=cycles,
            warp=warp,
            length=length,
        )


def test_writer_rejects_post_rewrite_smem_overflow(tmp_path):
    ii, cycles, warp, length = _identity_solution()
    with pytest.raises(ValueError, match="SMEM bytes"):
        rewrite_schedule_graph(
            BASELINE,
            tmp_path / "too-much-smem.json",
            ii=ii,
            cycles=cycles,
            warp=warp,
            length=length,
            smem_fixed_overhead=232448,
            materialize_all_spills=False,
        )
