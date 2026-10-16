import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_joint_solver.machine import (
    MachineModel,
    register_allocation_words,
    required_group_width,
)
from paper_joint_solver.resource_model import (
    derive_resources,
    register_words,
    storage_amount,
)

EXAMPLES = Path(__file__).resolve().parents[2] / "sched2tlx" / "examples"
SUBTILED = EXAMPLES / "case3_FA_fp16_subtiled" / "ddg.json"
BWD = EXAMPLES / "case4_FA_bwd" / "ddg_hd128.json"
BWD_SUBTILED = EXAMPLES / "case4_FA_bwd_subtiled" / "ddg.json"


@pytest.mark.parametrize(
    "result_types, expected",
    [
        (["tensor<64x64xf32>"], 4096),
        (["tensor<3xf16>", "!ttg.async.token"], 2),
        (["f32"], 1),
        (["!tt.ptr<f32>"], 2),
        (["tensor<64x!tt.ptr<f32>>"], 128),
        (["!ttg.async.token"], 0),
        (
            ["!ttg.memdesc<64x128xf16, #ttg.shared_memory>"],
            0,
        ),
    ],
)
def test_register_words_only_counts_register_values(result_types, expected):
    assert register_words(result_types) == expected


def test_storage_amount_uses_memory_kind_units():
    shared = ["!ttg.memdesc<64x128xf16, #ttg.shared_memory>"]
    tensor = ["!ttg.memdesc<64x64xf16, #ttng.tensor_memory>"]

    assert storage_amount(shared, "smem", 512) == 16384
    assert storage_amount(tensor, "tmem", 512) == 16


def test_unknown_register_type_fails_closed():
    with pytest.raises(ValueError, match="unsupported result type"):
        register_words(["!unknown.value"])


def test_subtiled_fixture_has_alias_aware_storage_footprints():
    data = json.loads(SUBTILED.read_text())
    ddg = data["loops"][0]["ddg"]
    nodes = {n["id"]: SimpleNamespace(**n) for n in ddg["nodes"]}
    edges = [SimpleNamespace(**e) for e in ddg["edges"]]

    registers, objects = derive_resources(
        nodes, edges, data["ops"], tmem_column_bytes=512
    )

    assert objects["smem:5"].amount == 16384
    assert objects["smem:5"].members == frozenset({2, 5, 6})
    assert objects["smem:4"].members == frozenset({3, 4})
    assert objects[f"tmem:{nodes[21].op_ref}"].amount == 16
    assert objects[f"tmem:{nodes[45].op_ref}"].amount == 16
    first_acc = data["ops"][nodes[7].op_ref]["operands"][2]["op"]
    second_acc = data["ops"][nodes[28].op_ref]["operands"][2]["op"]
    assert objects[f"tmem:{first_acc}"].amount == 32
    assert objects[f"tmem:{second_acc}"].amount == 64

    for node_id in (2, 3, 4, 5, 6, 7, 21, 28, 31, 45, 52):
        assert registers[node_id] == 0
    assert registers[8] == 4096


@pytest.mark.parametrize("fixture", (SUBTILED, BWD_SUBTILED))
def test_real_mma_accumulators_are_token_live_storage_members(fixture):
    data = json.loads(fixture.read_text())
    ddg = data["loops"][0]["ddg"]
    nodes = {node["id"]: SimpleNamespace(**node) for node in ddg["nodes"]}
    edges = [SimpleNamespace(**edge) for edge in ddg["edges"]]
    _, objects = derive_resources(
        nodes, edges, data["ops"], tmem_column_bytes=512
    )

    mma_nodes = [
        node for node in nodes.values() if node.op_kind == "ttng.tc_gen5_mma"
    ]
    assert mma_nodes
    for mma in mma_nodes:
        operands = data["ops"][mma.op_ref]["operands"]
        accumulator = operands[2]["op"]
        storage = objects[f"tmem:{accumulator}"]
        assert storage.kind == "tmem"
        assert storage.amount > 0
        assert mma.id in storage.members
        assert data["ops"][mma.op_ref]["result_types"] == [
            "!ttg.async.token"
        ]
        assert any(
            edge.src == mma.id
            and nodes[edge.dst].op_kind
            in {"ttng.tc_gen5_mma", "ttng.tmem_load"}
            for edge in edges
        )


def test_bwd_pointer_tensors_are_register_values():
    data = json.loads(BWD.read_text())
    ddg = data["loops"][0]["ddg"]
    nodes = {n["id"]: SimpleNamespace(**n) for n in ddg["nodes"]}
    edges = [SimpleNamespace(**e) for e in ddg["edges"]]

    registers, _ = derive_resources(
        nodes, edges, data["ops"], tmem_column_bytes=512
    )

    assert registers[6] == 128
    assert registers[8] == 128


def test_one_tma_with_two_allocations_charges_both():
    nodes = {
        0: SimpleNamespace(
            id=0,
            op_ref="tma",
            op_kind="tt.descriptor_load",
            pipeline="TMA",
        ),
        1: SimpleNamespace(
            id=1,
            op_ref="alloc_a",
            op_kind="ttg.local_alloc",
            pipeline="NONE",
        ),
        2: SimpleNamespace(
            id=2,
            op_ref="alloc_b",
            op_kind="ttg.local_alloc",
            pipeline="NONE",
        ),
    }
    edges = [SimpleNamespace(src=0, dst=1), SimpleNamespace(src=0, dst=2)]
    shared = "!ttg.memdesc<64x128xf16, #ttg.shared_memory>"
    ops = {
        "tma": {"result_types": ["!ttg.async.token"]},
        "alloc_a": {"result_types": [shared]},
        "alloc_b": {"result_types": [shared]},
    }

    registers, objects = derive_resources(
        nodes, edges, ops, tmem_column_bytes=512
    )

    assert objects["smem:1"].members == frozenset({0, 1})
    assert objects["smem:2"].members == frozenset({0, 2})
    assert sum(obj.amount for obj in objects.values()) == 32768
    assert registers == {0: 0, 1: 0, 2: 0}


def test_unknown_memdesc_view_fails_closed():
    shared = "!ttg.memdesc<64x128xf16, #ttg.shared_memory>"
    nodes = {
        0: SimpleNamespace(
            id=0,
            op_ref="alloc",
            op_kind="ttg.local_alloc",
            pipeline="NONE",
        ),
        1: SimpleNamespace(
            id=1,
            op_ref="view",
            op_kind="ttg.unknown_memdesc_view",
            pipeline="NONE",
        ),
    }
    edges = [SimpleNamespace(src=0, dst=1)]
    ops = {
        "alloc": {"result_types": [shared]},
        "view": {"result_types": [shared]},
    }

    with pytest.raises(ValueError, match="memory-handle transformation"):
        derive_resources(nodes, edges, ops, tmem_column_bytes=512)


def test_emitter_group_width_and_global_register_boundaries():
    """Emitter-only accounting: width quantization, the fixed-warp deduction
    and the SM-wide register file. The paper entry reads none of these -- see
    test_paper_model_has_no_sm_level_register_cap and
    test_paper_model_ignores_fixed_overheads."""
    assert required_group_width(1) == 1
    assert required_group_width(3) == 4
    assert required_group_width(8) == 8
    with pytest.raises(ValueError):
        required_group_width(9)

    with_overhead = MachineModel(warp_fixed_overhead=4)
    assert with_overhead.scheduled_warp_budget(17) == 13
    with pytest.raises(ValueError):
        with_overhead.scheduled_warp_budget(4)
    with pytest.raises(ValueError):
        with_overhead.scheduled_warp_budget(33)

    exact = MachineModel(regs_per_sm=8000)
    over = MachineModel(regs_per_sm=7999)
    values = {
        0: [(4000, frozenset({0}))],
        1: [(4000, frozenset({0}))],
    }
    widths = {0: 1, 1: 1}

    assert exact.registers_fit(values, widths)
    assert not over.registers_fit(values, widths)

    per_warp_over = MachineModel(regs_per_sm=1000, regs_per_warp=100)
    two_lanes = {0: [(201, frozenset({0, 1}))]}
    assert not per_warp_over.registers_fit(two_lanes, {0: 2})
    assert register_allocation_words(201, 2) == 202
    assert not MachineModel(regs_per_sm=201).registers_fit(two_lanes, {0: 2})
    assert not MachineModel(
        regs_per_sm=202, regs_fixed_overhead=1
    ).registers_fit(two_lanes, {0: 2})

    mixed = {
        0: [
            (90, frozenset({0})),
            (40, frozenset({0, 1, 2, 3})),
        ]
    }
    assert not MachineModel(regs_per_warp=99).registers_fit(mixed, {0: 4})
    assert MachineModel(regs_per_warp=100).registers_fit(mixed, {0: 4})

    balanced = {
        0: [
            (90, frozenset({0})),
            (90, frozenset({1})),
            (40, frozenset({0, 1, 2, 3})),
        ]
    }
    assert MachineModel(regs_per_warp=100).registers_fit(balanced, {0: 4})
