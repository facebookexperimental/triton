import json
from copy import deepcopy
from pathlib import Path

import pytest

from sched2tlx import emitter, schedule_graph

FIXTURE = Path(__file__).parent / "examples/case11_wait_order/schedule_graph.json"
_WAIT_S1 = "tlx.barrier_wait(sem2_full[0], (_it & 1))"
_WAIT_S2 = "tlx.barrier_wait(sem3_full[0], (_it & 1))"
_RECYCLE_S1 = "tlx.barrier_arrive(sem4_full[0], 1)"
_RECYCLE_S2 = "tlx.barrier_arrive(sem5_full[0], 1)"
_TMEM_BRIDGE_TAIL = "\n".join(
    f"                {line}"
    for line in (
        "trunc_21 = subf_20.to(tl.float16)",
        "tlx.barrier_wait(L0_acc_tmem_2_empty[0], (_it & 1) ^ 1)  # TMEM bridge",
        "tlx.local_store(L0_acc_tmem_2[0], trunc_21)",
        "tlx.barrier_arrive(L0_acc_tmem_2_full[0], 1)",
    )
)


def _merge_softmax_warp_groups(
    graph: schedule_graph.ScheduleGraph, first_stage_order: tuple[int, ...]
) -> None:
    loop = graph.loops[0]
    nodes = {node.id: node for node in loop.schedule.nodes}
    merged_wg = nodes[8].warp_group
    absorbed_wg = nodes[11].warp_group
    for node_id in (11, 12, 13):
        nodes[node_id].warp_group = merged_wg
    loop.warp_groups = [wg for wg in loop.warp_groups if wg.id != absorbed_wg]

    loop.schedule.cross_wg_barriers = [
        barrier
        for barrier in loop.schedule.cross_wg_barriers
        if (barrier.producer_node, barrier.consumer_node) != (10, 12)
    ]
    for barrier in loop.schedule.cross_wg_barriers:
        barrier.producer_wg = nodes[barrier.producer_node].warp_group
        barrier.consumer_wg = nodes[barrier.consumer_node].warp_group

    for cluster, node_id in enumerate(first_stage_order):
        nodes[node_id].schedule_cluster = cluster


def _assert_substrings_in_order(source: str, substrings: tuple[str, ...]) -> None:
    cursor = 0
    for substring in substrings:
        cursor = source.index(substring, cursor) + len(substring)


def _assert_shared_resident_load_protocol(sources: tuple[str, str]) -> None:
    for source in sources:
        _assert_substrings_in_order(
            source,
            (_WAIT_S1, "tlx.local_load(acc_tmem_4[0])", _RECYCLE_S1),
        )
        assert source.count(_WAIT_S1) == 1
        assert source.count(_WAIT_S2) == 1
        for recycle in (_RECYCLE_S1, _RECYCLE_S2):
            assert source.splitlines().count(f"                {recycle}") == 1


def _add_lowering_contract(data: dict) -> None:
    schedule = data["loops"][0]["schedule_loop"]
    nodes = [node for node in schedule["graph"]["nodes"] if node["warp_group"] >= 0]
    src, dst = nodes[:2]
    src_wg, dst_wg = src["warp_group"], dst["warp_group"]
    dst_order = 1 if src_wg == dst_wg else 0
    common = {
        "frequency": 1,
        "buffer_id": None,
        "bytes": 0,
        "depth": 1,
        "semaphore": "full",
        "fusion_group": None,
        "dedup_group": None,
    }
    schedule["lowering_templates"] = [
        {
            "id": 0,
            "relation": "always",
            "src_node": src["id"],
            "dst_node": dst["id"],
            "src_cluster": 0,
            "dst_cluster": 1,
            "events": [
                {
                    "id": 0,
                    "kind": "arrive",
                    "owner": "src",
                    "anchor_node": src["id"],
                    "placement": "after",
                    "pipeline": src["pipeline"],
                    "issue_duration": 1,
                    "completion_latency": 0,
                    "blocking": False,
                    "async": False,
                    "distance": 0,
                    **common,
                },
                {
                    "id": 1,
                    "kind": "wait",
                    "owner": "dst",
                    "anchor_node": dst["id"],
                    "placement": "before",
                    "pipeline": dst["pipeline"],
                    "issue_duration": 1,
                    "completion_latency": 0,
                    "blocking": True,
                    "async": False,
                    "distance": 0,
                    **common,
                },
            ],
        }
    ]
    schedule["lowering_plan"] = {
        "version": "lowering-plan-0.1",
        "status": "shadow_verified",
        "templates": [
            {
                "id": 0,
                "active": True,
                "events": [
                    {
                        "id": 0,
                        "cycle": src["schedule"]["cycle"] + 1,
                        "wg": src_wg,
                        "stream_order": 0,
                    },
                    {
                        "id": 1,
                        "cycle": dst["schedule"]["cycle"] - 1,
                        "wg": dst_wg,
                        "stream_order": dst_order,
                    },
                ],
            }
        ],
    }


def test_load_graph_parses_shadow_lowering_contract(tmp_path):
    data = json.loads(FIXTURE.read_text())
    _add_lowering_contract(data)
    graph_path = tmp_path / "schedule_graph.json"
    graph_path.write_text(json.dumps(data))

    graph = schedule_graph.load_graph(graph_path)

    lowering_template = graph.loops[0].schedule.lowering_templates[0]
    lowering_plan = graph.loops[0].schedule.lowering_plan
    assert lowering_template.events[1].kind == "wait"
    assert lowering_plan.status == "shadow_verified"
    assert lowering_plan.templates[0].events[1].warp_group >= 0


def test_load_graph_rejects_verified_plan_with_wrong_owner(tmp_path):
    data = json.loads(FIXTURE.read_text())
    _add_lowering_contract(data)
    data["loops"][0]["schedule_loop"]["lowering_plan"]["templates"][0]["events"][0][
        "wg"
    ] = 999
    graph_path = tmp_path / "schedule_graph.json"
    graph_path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="lowering owner mismatch"):
        schedule_graph.load_graph(graph_path)


def test_transposed_resident_mma_operands_wait_for_tma():
    src = emitter.emit(schedule_graph.load_graph(FIXTURE))

    consumers = {
        "q_smem_6": "tlx.async_dot(L0_acc_tmem_2[0], q_smem_6[0]",
        "q_smem_7": "tlx.local_trans(q_smem_7[0])",
        "q_smem_8": "tlx.local_trans(q_smem_8[0])",
    }
    for alloc_var, consumer in consumers.items():
        wait = f"tlx.barrier_wait({alloc_var}_full[0], 0)"
        assert src.count(wait) == 1
        assert src.index(wait) < src.index(consumer)


def test_merged_softmax_resident_load_can_be_early_or_deferred():
    base = schedule_graph.load_graph(FIXTURE)
    early, deferred = deepcopy(base), deepcopy(base)
    _merge_softmax_warp_groups(early, (8, 11, 9, 10))
    _merge_softmax_warp_groups(deferred, (8, 9, 10, 11))

    early_src, deferred_src = emitter.emit(early), emitter.emit(deferred)
    _assert_shared_resident_load_protocol((early_src, deferred_src))

    _assert_substrings_in_order(
        early_src,
        (
            _WAIT_S2,
            "tlx.local_load(acc_tmem_5[0])",
            _RECYCLE_S2,
            "* scale)",
            "tl.math.exp2(",
        ),
    )
    _assert_substrings_in_order(
        deferred_src,
        (
            "* scale)",
            "tl.math.exp2(",
            _WAIT_S2,
            "tlx.local_load(acc_tmem_5[0])",
            _RECYCLE_S2,
        ),
    )
    assert _TMEM_BRIDGE_TAIL in early_src
    assert _TMEM_BRIDGE_TAIL in deferred_src
