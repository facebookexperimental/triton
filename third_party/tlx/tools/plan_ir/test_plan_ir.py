from __future__ import annotations

import copy
import json

import pytest

from tlx_plan.baseline import FA_BWD_D128_CASES, FA_BWD_D128_SCHEDULES, make_manifest
from tlx_plan.model import PlanBundle, PlanError
from tlx_plan.replay import compare_plans, verify_replay
from tlx_plan.ttgir import extract_plan, normalize_ttgir


TTGIR = r"""
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 4], instrShape = [16, 16, 32], isTransposed = true}>
#mma_p = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32, 16], isTransposed = true}>
#slice = #ttg.slice<{dim = 1, parent = #mma_p}>
module {
  tt.func public @_attn_bwd_dkdv_dq_d128_gqa_kernel() {
    %dv = arith.constant dense<0.0> : tensor<128x64xf32, #mma_p> loc(#loc1)
    %c0 = amdg.extract_slice %dv [0, 0] sizes [128, 32] strides [1, 1] : tensor<128x64xf32, #mma_p> to tensor<128x32xf32, #mma_p> loc(#loc2)
    %lhs = arith.constant dense<0.0> : tensor<128x16xbf16, #mma_p> loc(#loc3)
    %rhs = arith.constant dense<0.0> : tensor<16x32xbf16, #mma_p> loc(#loc4)
    %c1 = amdg.scheduled_mfma %lhs, %rhs, %c0 resident "none" accumulator "persistent" register_class "vgpr" initialize false : tensor<128x16xbf16, #mma_p>, tensor<16x32xbf16, #mma_p>, tensor<128x32xf32, #mma_p> -> tensor<128x32xf32, #mma_p> loc(#loc5)
    %dq0 = arith.constant dense<0.0> : tensor<16x64xf32, #mma> loc(#loc6)
    %ds0 = arith.constant dense<0.0> : tensor<16x32xbf16, #mma> loc(#loc7)
    %k0 = arith.constant dense<0.0> : tensor<32x64xbf16, #mma> loc(#loc8)
    %dq1 = amdg.scheduled_mfma %ds0, %k0, %dq0 resident "rhs" accumulator "transient" register_class "auto" initialize true : tensor<16x32xbf16, #mma>, tensor<32x64xbf16, #mma>, tensor<16x64xf32, #mma> -> tensor<16x64xf32, #mma> loc(#loc9)
    %buf = ttg.local_alloc : () -> !ttg.memdesc<2x256x128xbf16, #slice, #ttg.shared_memory, mutable> loc(#loc10)
    ttg.async_commit_group loc(#loc11)
    ttg.async_wait 0 : i32 loc(#loc12)
    ttg.barrier loc(#loc13)
    tt.return loc(#loc14)
  }
}
#loc1 = loc("kernel.py":1:1)
#loc2 = loc("kernel.py":2:1)
"""


def test_fixed_fa_backward_catalog() -> None:
    assert FA_BWD_D128_CASES["mha_n2048_d128"].shape["head_dim"] == 128
    assert FA_BWD_D128_CASES["gqa_b16_hq64_hkv8_n16384_d128"].grouped_query
    assert FA_BWD_D128_SCHEDULES["tlx_fused_bridge"].config == {
        "block_m": 16,
        "block_n": 256,
        "num_warps": 4,
        "num_stages": 1,
    }
    manifest = make_manifest("mha_n2048_d128", "tlx_fused_bridge", "source", "compiler")
    assert manifest.schedule.algorithm == "long_fused_bridge"


def test_extract_scheduled_mfma_contracts_and_storage() -> None:
    plan = extract_plan(TTGIR, source_name="fixture.ttgir")
    assert plan.kernel == "_attn_bwd_dkdv_dq_d128_gqa_kernel"
    assert [fragment["role"] for fragment in plan.dot_fragments] == ["dv", "dq"]
    assert plan.dot_fragments[0]["accumulator_slice_offsets"] == [0, 0]
    assert plan.dot_fragments[0]["accumulator"] == "persistent"
    assert plan.dot_fragments[0]["register_class"] == "vgpr"
    assert plan.dot_fragments[1]["resident"] == "rhs"
    assert plan.dot_fragments[1]["initialize"] is True
    assert "warpsPerCTA = [1, 4]" in plan.dot_fragments[1]["mma_layout"]
    assert len(plan.storage) == 1
    assert [entry["kind"] for entry in plan.synchronization] == [
        "ttg.async_commit_group",
        "ttg.async_wait",
        "ttg.barrier",
    ]
    assert not plan.diagnostics


def test_normalization_and_ids_ignore_debug_locations_and_ssa_names() -> None:
    changed = TTGIR.replace("%dq0", "%renamed0").replace("%dq1", "%renamed1")
    changed = changed.replace("#loc9", "#loc999")
    left = extract_plan(TTGIR)
    right = extract_plan(changed)
    assert normalize_ttgir(TTGIR) == normalize_ttgir(changed)
    assert left.normalized_ir_hash == right.normalized_ir_hash
    assert left.layer_hashes == right.layer_hashes


def test_round_trip_and_replay_report(tmp_path) -> None:
    path = tmp_path / "plan.json"
    expected = extract_plan(TTGIR)
    expected.write(path)
    loaded = PlanBundle.read(path)
    report = verify_replay(TTGIR, loaded)
    assert report["exact"]
    assert report["semantic_match"]
    assert json.loads(path.read_text())["schema_version"] == "0.4"


def test_native_value_graph_supplies_stable_ids_and_lineage() -> None:
    textual = extract_plan(TTGIR)
    operations = [
        {
            "id": f"op:native:{index}",
            "kind": operation["kind"],
            "ordinal": index,
            "locator": f"func/kernel/b0/o{index}",
            "identity_quality": "semantic",
        }
        for index, operation in enumerate(textual.operations)
    ]
    native = {
        "schema_version": "plan-value-graph/0.3",
        "functions": [
            {
                "function": textual.kernel,
                "semantic_fingerprint": "native-fingerprint",
                "operations": operations,
                "values": [
                    {"id": "value:a"},
                    {"id": "value:b"},
                ],
                "lineage_edges": [
                    {
                        "source": "value:a",
                        "destination": "value:b",
                        "kind": "loop_backedge",
                        "iteration_distance": 1,
                    }
                ],
                "blocks": [
                    {
                        "id": "block:native:0",
                        "operations": [operation["id"] for operation in operations],
                    }
                ],
                "live_segments": [
                    {
                        "value": "value:a",
                        "block": "block:native:0",
                        "start_position": 0,
                        "end_position": 1,
                        "crosses_backedge": False,
                        "iteration_distance": 0,
                    }
                ],
                "lds_aliases": [
                    {
                        "value": "value:a",
                        "source_value": None,
                        "root_values": ["value:a"],
                        "slot_paths": [{"root_value": "value:a", "indices": []}],
                    }
                ],
                "memory_accesses": [
                    {
                        "operation": operations[0]["id"],
                        "value": "value:a",
                        "root_values": ["value:a"],
                        "slot_paths": [{"root_value": "value:a", "indices": []}],
                    }
                ],
                "lds_allocations": [
                    {
                        "root_value": "value:a",
                        "allocation_operation": operations[0]["id"],
                        "aliases": ["value:a"],
                        "live_segments": [
                            {
                                "value": "value:a",
                                "block": "block:native:0",
                                "start_position": 0,
                                "end_position": 1,
                                "crosses_backedge": False,
                                "iteration_distance": 0,
                            }
                        ],
                    }
                ],
                "async_transactions": [
                    {
                        "id": "async-tx:native:0",
                        "producer_operation": operations[0]["id"],
                        "destination_value": "value:a",
                        "commit_group": "async-group:native:0",
                        "root_values": ["value:a"],
                        "slot_paths": [{"root_value": "value:a", "indices": []}],
                        "completion_frontiers": [
                            {
                                "operation": operations[1]["id"],
                                "block": "block:native:0",
                                "position": 1,
                                "iteration_distance": 1,
                            }
                        ],
                        "visibility_frontiers": [],
                        "consumer_frontiers": [],
                        "release_frontiers": [],
                        "overwrite_frontiers": [],
                    }
                ],
                "async_groups": [
                    {
                        "id": "async-group:native:0",
                        "commit_operation": operations[0]["id"],
                        "transactions": ["async-tx:native:0"],
                    }
                ],
                "async_waits": [
                    {
                        "operation": operations[1]["id"],
                        "retained_group_count": 0,
                        "completed_groups": [
                            {
                                "group": "async-group:native:0",
                                "iteration_distance": 1,
                            }
                        ],
                        "possibly_outstanding_groups": [],
                    }
                ],
                "lds_reuse_hazards": [],
                "diagnostics": [],
            }
        ],
    }
    plan = extract_plan(TTGIR, native_value_graph=native)
    assert [operation["id"] for operation in plan.operations] == [
        operation["id"] for operation in operations
    ]
    assert plan.value_graph_fingerprint == "native-fingerprint"
    assert plan.lineage_edges[0]["iteration_distance"] == 1
    assert plan.live_segments[0]["block"] == "block:native:0"
    assert plan.lds_allocations[0]["root_value"] == "value:a"
    assert plan.async_transactions[0]["completion_frontiers"][0]["iteration_distance"] == 1
    assert verify_replay(TTGIR, plan, native_value_graph=native)["semantic_match"]


def test_read_upgrades_plan_bundle_0_1(tmp_path) -> None:
    plan = extract_plan(TTGIR).to_dict()
    plan["schema_version"] = "0.1"
    plan.pop("values")
    plan.pop("lineage_edges")
    plan.pop("value_graph_fingerprint")
    path = tmp_path / "old-plan.json"
    path.write_text(json.dumps(plan))
    upgraded = PlanBundle.read(path)
    assert upgraded.schema_version == "0.4"
    assert upgraded.values == []


def test_read_upgrades_plan_bundle_0_2(tmp_path) -> None:
    plan = extract_plan(TTGIR).to_dict()
    plan["schema_version"] = "0.2"
    for field in (
        "blocks",
        "live_segments",
        "lds_aliases",
        "memory_accesses",
        "lds_allocations",
    ):
        plan.pop(field)
    path = tmp_path / "old-plan.json"
    path.write_text(json.dumps(plan))
    upgraded = PlanBundle.read(path)
    assert upgraded.schema_version == "0.4"
    assert upgraded.blocks == []


def test_read_upgrades_plan_bundle_0_3_without_losing_liveness(tmp_path) -> None:
    plan = extract_plan(TTGIR).to_dict()
    plan["schema_version"] = "0.3"
    plan["blocks"] = [{"id": "preserved", "operations": []}]
    for field in (
        "async_transactions",
        "async_groups",
        "async_waits",
        "lds_reuse_hazards",
    ):
        plan.pop(field)
    path = tmp_path / "old-plan.json"
    path.write_text(json.dumps(plan))
    upgraded = PlanBundle.read(path)
    assert upgraded.schema_version == "0.4"
    assert upgraded.blocks[0]["id"] == "preserved"
    assert upgraded.async_transactions == []


def test_layered_diff_localizes_schedule_change() -> None:
    expected = extract_plan(TTGIR)
    actual = copy.deepcopy(expected)
    actual.schedule[0], actual.schedule[1] = actual.schedule[1], actual.schedule[0]
    actual.refresh_hashes()
    report = compare_plans(expected, actual)
    assert not report["semantic_match"]
    assert not report["layers"]["schedule"]["match"]
    assert report["layers"]["dot_fragments"]["match"]


def test_validation_rejects_unknown_scheduled_operation() -> None:
    plan = extract_plan(TTGIR)
    plan.schedule.append("op/missing/0")
    plan.refresh_hashes()
    with pytest.raises(PlanError, match="unknown operations"):
        plan.validate()
