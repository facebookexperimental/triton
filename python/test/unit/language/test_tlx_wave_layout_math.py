from types import SimpleNamespace

import pytest

from triton._C.libtriton.linear_layout import LinearLayout
from triton.backends import backends

pytestmark = pytest.mark.skipif(
    "tlx_wave" not in backends,
    reason="TLX Wave backend is not registered",
)

if "tlx_wave" in backends:
    from triton.backends.tlx_wave.converter import layouts

_LANE_WIDTH = 64
_PACKET_WIDTH = 8


def _distributed_layout(register, lane, warp, shape):
    return SimpleNamespace(
        kind="dot_operand",
        linear_layout=LinearLayout.from_bases(
            (
                ("register", register),
                ("lane", lane),
                ("warp", warp),
                ("block", ()),
            ),
            tuple(f"dim{dim}" for dim in range(len(shape))),
            shape,
            False,
        ),
    )


def _glu_b_layout():
    return _distributed_layout(
        (
            (1, 0),
            (2, 0),
            (4, 0),
            (32, 0),
            (0, 32),
            (0, 64),
        ),
        (
            (0, 1),
            (0, 2),
            (0, 4),
            (0, 8),
            (8, 0),
            (16, 0),
        ),
        (
            (0, 16),
            (0, 0),
        ),
        (64, 128),
    )


def _padded_b_layout():
    return _distributed_layout(
        (
            (1, 0),
            (2, 0),
            (8, 0),
            (16, 0),
            (32, 0),
            (0, 64),
        ),
        (
            (0, 1),
            (0, 2),
            (0, 4),
            (0, 8),
            (0, 16),
            (4, 0),
        ),
        (
            (0, 32),
            (0, 0),
            (0, 0),
        ),
        (64, 128),
    )


def _fa_v_layout():
    return _distributed_layout(
        (
            (1, 0),
            (2, 0),
            (8, 0),
            (16, 0),
            (32, 0),
            (64, 0),
            (0, 32),
        ),
        (
            (0, 1),
            (0, 2),
            (0, 4),
            (0, 8),
            (0, 16),
            (4, 0),
        ),
        (
            (0, 0),
            (0, 0),
        ),
        (128, 64),
    )


def _fa_v_shared_layout():
    inverse = LinearLayout.from_bases(
        (
            (
                "dim0",
                (
                    (512, 0),
                    (64, 0),
                    (128, 0),
                    (256, 0),
                    (1024, 0),
                    (2048, 0),
                    (4096, 0),
                ),
            ),
            (
                "dim1",
                tuple((1 << bit, 0) for bit in range(6)),
            ),
        ),
        ("offset", "block"),
        (128 * 64, 1),
        False,
    )
    return _shared_layout(
        "padded_shared",
        inverse,
        intervals=(512, ),
        paddings=(32, ),
        order=(1, 0),
    )


def _glu_b_global_layout():
    # Exact LinearLayout for the B offsets in the optimized GLU kernel:
    # sizePerThread=[1,8], threadsPerWarp=[4,16], warpsPerCTA=[8,1].
    linear = LinearLayout.from_bases(
        (
            ("register", ((0, 1), (0, 2), (0, 4), (4, 0))),
            ("lane", ((0, 8), (0, 16), (0, 32), (0, 64), (16, 0), (32, 0))),
            ("warp", ((1, 0), (2, 0), (8, 0))),
            ("block", ()),
        ),
        ("dim0", "dim1"),
        (64, 128),
        False,
    )
    return SimpleNamespace(
        kind="linear",
        linear_layout=linear,
        lane_width=_LANE_WIDTH,
    )


def _fa_k_global_layout():
    # Exact #linear1 layout of the async FA K offsets.
    linear = LinearLayout.from_bases(
        (
            ("register", ((0, 1), (0, 2), (0, 4), (4, 0))),
            ("lane", ((0, 8), (0, 16), (0, 32), (8, 0), (16, 0), (32, 0))),
            ("warp", ((1, 0), (2, 0))),
            ("block", ()),
        ),
        ("dim0", "dim1"),
        (64, 64),
        False,
    )
    return SimpleNamespace(
        kind="linear",
        linear_layout=linear,
        lane_width=_LANE_WIDTH,
    )


def _shared_layout(kind, inverse, **properties):
    return SimpleNamespace(
        kind=kind,
        linear_layout=inverse.invert(),
        properties=properties,
    )


def _glu_shared_layout():
    inverse = LinearLayout.from_bases(
        (
            (
                "dim0",
                (
                    ((1 << 3) | (1 << 7), 0),
                    ((1 << 4) | (1 << 8), 0),
                    ((1 << 5) | (1 << 9), 0),
                    ((1 << 6) | (1 << 10), 0),
                    (1 << 11, 0),
                    (1 << 12, 0),
                ),
            ),
            (
                "dim1",
                tuple((1 << bit, 0) for bit in range(7)),
            ),
        ),
        ("offset", "block"),
        (64 * 128, 1),
        False,
    )
    return _shared_layout(
        "swizzled_shared",
        inverse,
        vec=8,
        per_phase=1,
        max_phase=16,
        order=(1, 0),
    )


def _padded_shared_layout():
    inverse = LinearLayout.from_bases(
        (
            (
                "dim0",
                tuple((1 << bit, 0) for bit in range(7, 13)),
            ),
            (
                "dim1",
                tuple((1 << bit, 0) for bit in range(7)),
            ),
        ),
        ("offset", "block"),
        (64 * 128, 1),
        False,
    )
    return _shared_layout(
        "padded_shared",
        inverse,
        intervals=(4, ),
        paddings=(16, ),
        order=(1, 0),
    )


def _bit_offset_relation(
        distributed,
        shared,
        warp_count,
        allocation_bytes,
        shape=(64, 128),
):
    dsl = layouts.load_wave_dsl()
    blob = layouts.local_memory_bit_offset_relation(
        distributed,
        shared,
        shape,
        shape,
        (0, ) * len(shape),
        lane_width=_LANE_WIDTH,
        warp_count=warp_count,
        element_byte_width=2,
        allocation_bytes=allocation_bytes,
        stage="test",
        diagnostic="TLXW_TEST",
    )
    return dsl, dsl.ixs_deserialize(blob)


def _lane_major_relation():
    distributed = _distributed_layout(
        ((64, ), (128, ), (256, )),
        ((1, ), (2, ), (4, ), (8, ), (16, ), (32, )),
        (),
        (512, ),
    )
    inverse = LinearLayout.from_bases(
        (("dim0", tuple((1 << bit, 0) for bit in range(9))), ),
        ("offset", "block"),
        (512, 1),
        False,
    )
    return _bit_offset_relation(
        distributed,
        _shared_layout(
            "swizzled_shared",
            inverse,
            vec=1,
            per_phase=1,
            max_phase=1,
            order=(0, ),
        ),
        warp_count=1,
        allocation_bytes=512 * 2,
        shape=(512, ),
    )


def _transpose_frame(dsl, item):
    group = dsl.sym("group")
    within = dsl.sym("within")
    lane = dsl.mod(item, 64)
    wave_base = 64 * dsl.floor(item / 64)
    row_base = 16 * dsl.floor(lane / 16)
    lane_in_row = dsl.mod(lane, 16)
    source_item = wave_base + row_base + dsl.floor(lane_in_row / 4) + 4 * within
    source_lane = dsl.mod(source_item, 64)
    source_lane_in_row = dsl.mod(source_lane, 16)
    origin_item = (64 * dsl.floor(source_item / 64) + 16 * dsl.floor(source_lane / 16) +
                   4 * dsl.mod(source_lane_in_row, 4))
    origin_slot = 4 * group + dsl.floor(source_lane_in_row / 4)
    return group, within, origin_item, origin_slot


def _unpack_packet_relation(dsl, relation, slots, items):
    packed = dsl.ixs_deserialize(relation)
    source_slot = dsl.mod(packed, slots)
    source_item = dsl.mod(dsl.floor(packed / slots), items)
    source_block = dsl.floor(packed / (slots * items))
    facts = (
        *layouts._symbolic_range_predicates(dsl, dsl.sym("block"), 0, 0),
        *layouts._symbolic_range_predicates(dsl, dsl.sym("item"), 0, items - 1),
        *layouts._symbolic_range_predicates(dsl, dsl.sym("slot"), 0, slots - 1),
    )
    return dsl.ixs_check((source_block, source_item, source_slot), facts)[1]


def test_packet_relation_keeps_direct_physical_coordinates_literal():
    bases = {
        "register": ((1, 0), (2, 0), (4, 0)),
        "lane": ((0, 1), (0, 2), (0, 4), (0, 8), (8, 0), (16, 0)),
        "warp": ((0, 16), (0, 32)),
        "block": (),
    }

    def make_linear(order):
        return LinearLayout.from_bases(
            tuple((name, bases[name]) for name in order),
            ("dim0", "dim1"),
            (32, 64),
            False,
        )

    source = make_linear(("register", "lane", "warp", "block"))
    result = make_linear(("warp", "register", "lane", "block"))
    relation = layouts._packet_relation_blob(
        source,
        result,
        (32, 64),
        (32, 64),
        lane_width=_LANE_WIDTH,
        source_components=_PACKET_WIDTH,
        destination_components=_PACKET_WIDTH,
    )
    dsl = layouts.load_wave_dsl()
    source_block, source_item, source_slot = _unpack_packet_relation(dsl, relation, _PACKET_WIDTH, 256)
    assert str(source_block) == "0"
    assert str(source_item) == "item"
    assert str(source_slot) == "slot"


def test_binary_reshape_keeps_direct_physical_coordinates_literal():
    source_shape = (1, 1, 2, 1, 16, 32)
    result_shape = (32, 32)
    source_bases = (
        ("register", ((0, 0, 0, 0, 0, 1), (0, 0, 0, 0, 0, 2), (0, 0, 0, 0, 0, 4))),
        ("lane", ((0, 0, 0, 0, 1, 0), (0, 0, 0, 0, 2, 0), (0, 0, 0, 0, 4, 0), (0, 0, 0, 0, 8, 0), (0, 0, 0, 0, 0, 8),
                  (0, 0, 0, 0, 0, 16))),
        ("warp", ((0, 0, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0))),
        ("block", ()),
    )

    def reshape_coordinate(coordinate):
        linear = 0
        for value, extent in zip(coordinate, source_shape, strict=True):
            linear = linear * extent + value
        return (linear // result_shape[1], linear % result_shape[1])

    source = LinearLayout.from_bases(
        source_bases,
        tuple(f"dim{dim}" for dim in range(len(source_shape))),
        source_shape,
        False,
    )
    result = LinearLayout.from_bases(
        tuple((name, tuple(reshape_coordinate(basis) for basis in bases)) for name, bases in source_bases),
        ("dim0", "dim1"),
        result_shape,
        False,
    )
    relation = layouts._packet_relation_blob(
        source,
        result,
        source_shape,
        result_shape,
        lane_width=_LANE_WIDTH,
        source_components=_PACKET_WIDTH,
        destination_components=_PACKET_WIDTH,
        transform="reshape",
    )
    dsl = layouts.load_wave_dsl()
    source_block, source_item, source_slot = _unpack_packet_relation(dsl, relation, _PACKET_WIDTH, 256)
    assert str(source_block) == "0"
    assert str(source_item) == "item"
    assert str(source_slot) == "slot"


def _prove_b16_transactions(dsl, relation, item_count, packet_count=8):
    item = dsl.sym("item")
    slot = dsl.sym("slot")
    local_slot = dsl.sym("local_slot")
    group, within, origin_item, origin_slot = _transpose_frame(dsl, item)
    facts = (
        *layouts._symbolic_range_predicates(dsl, item, 0, item_count - 1),
        *layouts._symbolic_range_predicates(dsl, group, 0, 1),
        *layouts._symbolic_range_predicates(dsl, within, 0, 3),
    )
    for packet in range(packet_count):
        packet_relation = relation.subs({
            slot: local_slot + packet * _PACKET_WIDTH,
        })
        point = packet_relation.subs({local_slot: 4 * group + within})
        origin = packet_relation.subs({
            item: origin_item,
            local_slot: origin_slot,
        })
        checked, _normalized = dsl.ixs_check(
            (dsl.ixs_eq(point, origin + 16 * dsl.mod(item, 4)), ),
            facts,
        )
        assert checked == (True, )


def test_glu_b_layout_to_symbolic_relation_matches_linear_layout():
    distributed = _glu_b_layout()
    dsl, relation = _bit_offset_relation(
        distributed,
        _glu_shared_layout(),
        warp_count=4,
        allocation_bytes=64 * 128 * 2,
    )
    for item in range(256):
        for slot in range(64):
            logical = distributed.linear_layout.apply({
                "register": slot,
                "lane": item % 64,
                "warp": item // 64,
                "block": 0,
            })
            row = int(logical["dim0"])
            column = int(logical["dim1"])
            swizzled_column = ((column // 8) ^ (row % 16)) * 8 + column % 8
            expected = 16 * (swizzled_column + 128 * row)
            assert int(relation.eval({"item": item, "slot": slot})) == expected


def test_lane_major_symbolic_relation_proves_b16_contiguity():
    dsl, relation = _lane_major_relation()
    item = dsl.sym("item")
    slot = dsl.sym("slot")
    for physical_item in range(64):
        for physical_slot in range(8):
            assert int(relation.eval({
                "item": physical_item,
                "slot": physical_slot,
            })) == 16 * (physical_item + 64 * physical_slot)
    within = dsl.sym("within")
    lane = dsl.mod(item, 64)
    wave_base = item - lane
    linear = 8 * lane + within
    point = relation.subs({
        item: wave_base + dsl.mod(linear, 64),
        slot: dsl.floor(linear / 64),
    })
    origin_linear = 8 * lane
    origin = relation.subs({
        item: wave_base + dsl.mod(origin_linear, 64),
        slot: dsl.floor(origin_linear / 64),
    })
    facts = (
        *layouts._symbolic_range_predicates(dsl, item, 0, 63),
        *layouts._symbolic_range_predicates(dsl, within, 0, 7),
    )
    checked, _normalized = dsl.ixs_check(
        (dsl.ixs_eq(point, origin + 16 * within), ),
        facts,
    )
    assert checked == (True, )


def test_lane_major_global_relation_proves_direct_to_lds_contiguity():
    distributed = _distributed_layout(
        ((64, ), (128, ), (256, )),
        ((1, ), (2, ), (4, ), (8, ), (16, ), (32, )),
        (),
        (512, ),
    )
    distributed.lane_width = _LANE_WIDTH
    dsl = layouts.load_wave_dsl()
    relation = dsl.ixs_deserialize(
        layouts.global_memory_bit_offset_relation(
            distributed,
            dsl.sym("dim0"),
            element_byte_width=2,
        ))
    item = dsl.sym("item")
    slot = dsl.sym("slot")
    within = dsl.sym("within")
    linear = 8 * item + within
    point = relation.subs({
        item: dsl.mod(linear, 64),
        slot: dsl.floor(linear / 64),
    })
    origin_linear = 8 * item
    origin = relation.subs({
        item: dsl.mod(origin_linear, 64),
        slot: dsl.floor(origin_linear / 64),
    })
    facts = (
        *layouts._symbolic_range_predicates(dsl, item, 0, 63),
        *layouts._symbolic_range_predicates(dsl, within, 0, 7),
    )
    checked, _normalized = dsl.ixs_check(
        (dsl.ixs_eq(point, origin + 16 * within), ),
        facts,
    )
    assert checked == (True, )


def test_glu_b_global_layout_symbolic_relation_proves_width8_contiguity():
    dsl = layouts.load_wave_dsl()
    layout = _glu_b_global_layout()
    base = dsl.sym("base")
    iteration = dsl.sym("iteration")
    advance = dsl.sym("advance")
    stride = dsl.sym("stride")
    logical_offset = (base + iteration * advance + dsl.sym("dim0") * stride + dsl.sym("dim1"))
    relation = dsl.ixs_deserialize(
        layouts.global_memory_bit_offset_relation(
            layout,
            logical_offset,
            element_byte_width=2,
            wrap_i32=False,
        ))

    item = dsl.sym("item")
    slot = dsl.sym("slot")
    group = dsl.sym("group")
    within = dsl.sym("within")
    point = relation.subs({slot: 8 * group + within})
    origin = relation.subs({slot: 8 * group})
    facts = (
        *layouts._symbolic_range_predicates(dsl, item, 0, 511),
        *layouts._symbolic_range_predicates(dsl, group, 0, 1),
        *layouts._symbolic_range_predicates(dsl, within, 0, 7),
    )
    checked, _normalized = dsl.ixs_check(
        (dsl.ixs_eq(point, origin + 16 * within), ),
        facts,
    )
    assert checked == (True, )


def test_fa_k_layout_symbolic_formula_uses_maximal_integer_fields():
    dsl = layouts.load_wave_dsl()
    item = dsl.sym("item")
    slot = dsl.sym("slot")
    linear = layouts._packet_item_linear_layout(
        layouts._complete_packet_physical_dims(_fa_k_global_layout().linear_layout, ),
        _LANE_WIDTH,
        4,
        preserve_block=True,
    )
    formula = layouts._symbolic_layout_field_formula(
        dsl,
        linear,
        {"block": dsl.sym("block"), "item": item, "slot": slot},
    )
    assert formula == {
        "dim0": (dsl.xor(
            dsl.xor(
                dsl.mod(dsl.floor(item / 64), 4),
                8 * dsl.mod(dsl.floor(item / 8), 8),
            ),
            4 * dsl.mod(dsl.floor(slot / 8), 2),
        )),
        "dim1":
        dsl.xor(dsl.mod(slot, 8), 8 * dsl.mod(item, 8)),
    }
    for physical_item in range(256):
        for physical_slot in range(16):
            expected = _fa_k_global_layout().linear_layout.apply({
                "register": physical_slot,
                "lane": physical_item % 64,
                "warp": physical_item // 64,
                "block": 0,
            })
            assert {
                name: int(expr.eval({
                    "block": 0,
                    "item": physical_item,
                    "slot": physical_slot,
                }))
                for name, expr in formula.items()
            } == {name: int(value)
                  for name, value in expected.items()}


def test_glu_b_global_layout_contiguity_normalizes_nonlinear_index():
    dsl = layouts.load_wave_dsl()
    layout = _glu_b_global_layout()
    extent = dsl.sym("extent")
    base = dsl.sym("base")
    stride = dsl.sym("stride")
    logical_offset = (dsl.sym("dim0") * stride + dsl.mod(base + dsl.sym("dim1"), extent))
    relation = dsl.ixs_deserialize(
        layouts.global_memory_bit_offset_relation(
            layout,
            logical_offset,
            element_byte_width=2,
            element_contiguity=8,
            wrap_i32=False,
        ))

    item = dsl.sym("item")
    slot = dsl.sym("slot")
    group = dsl.sym("group")
    within = dsl.sym("within")
    point = relation.subs({slot: 8 * group + within})
    origin = relation.subs({slot: 8 * group})
    facts = (
        *layouts._symbolic_range_predicates(dsl, item, 0, 511),
        *layouts._symbolic_range_predicates(dsl, group, 0, 1),
        *layouts._symbolic_range_predicates(dsl, within, 0, 7),
    )
    checked, _normalized = dsl.ixs_check(
        (dsl.ixs_eq(point, origin + 16 * within), ),
        facts,
    )
    assert checked == (True, )


def test_glu_b_symbolic_relation_proves_b16_contiguity():
    dsl, relation = _bit_offset_relation(
        _glu_b_layout(),
        _glu_shared_layout(),
        warp_count=4,
        allocation_bytes=64 * 128 * 2,
    )
    _prove_b16_transactions(dsl, relation, 256)


def test_fa_v_layout_to_symbolic_formula_proves_b16_contiguity():
    dsl, relation = _bit_offset_relation(
        _fa_v_layout(),
        _fa_v_shared_layout(),
        warp_count=4,
        allocation_bytes=17408,
        shape=(128, 64),
    )
    item = dsl.sym("item")
    slot = dsl.sym("slot")
    unpadded = (dsl.mod(item, 32) + 128 * dsl.mod(dsl.floor(item / 32), 2) + 512 * dsl.mod(slot, 2) +
                64 * dsl.mod(dsl.floor(slot / 2), 2) + 256 * dsl.mod(dsl.floor(slot / 4), 2) +
                1024 * dsl.mod(dsl.floor(slot / 8), 2) + 2048 * dsl.mod(dsl.floor(slot / 16), 2) +
                4096 * dsl.mod(dsl.floor(slot / 32), 2) + 32 * dsl.mod(dsl.floor(slot / 64), 2))
    unpadded_formula = dsl.xor(
        dsl.mod(item, 2),
        2 * dsl.mod(dsl.floor(dsl.mod(item, 64) / 2), 2),
        4 * dsl.mod(dsl.floor(dsl.mod(item, 64) / 4), 2),
        8 * dsl.mod(dsl.floor(dsl.mod(item, 64) / 8), 2),
        16 * dsl.mod(dsl.floor(dsl.mod(item, 64) / 16), 2),
        32 * dsl.mod(dsl.floor(slot / 64), 2),
        64 * dsl.mod(dsl.floor(slot / 2), 2),
        128 * dsl.mod(dsl.floor(dsl.mod(item, 64) / 32), 2),
        256 * dsl.mod(dsl.floor(slot / 4), 2),
        512 * dsl.mod(slot, 2),
        1024 * dsl.mod(dsl.floor(slot / 8), 2),
        2048 * dsl.mod(dsl.floor(slot / 16), 2),
        4096 * dsl.mod(dsl.floor(slot / 32), 2),
    )
    padding_formula = 32 * dsl.xor(
        dsl.mod(slot, 2),
        2 * dsl.mod(dsl.floor(slot / 8), 2),
        4 * dsl.mod(dsl.floor(slot / 16), 2),
        8 * dsl.mod(dsl.floor(slot / 32), 2),
    )
    assert relation == 16 * (unpadded_formula + padding_formula)
    for physical_item in range(256):
        for physical_slot in range(128):
            unpadded_value = int(unpadded.eval({
                "item": physical_item,
                "slot": physical_slot,
            }))
            expected = 16 * (unpadded_value + 32 * (unpadded_value // 512))
            assert int(relation.eval({
                "item": physical_item,
                "slot": physical_slot,
            })) == expected
    _prove_b16_transactions(dsl, relation, 256, packet_count=16)


def test_padded_b_symbolic_relation_proves_b16_contiguity():
    dsl, relation = _bit_offset_relation(
        _padded_b_layout(),
        _padded_shared_layout(),
        warp_count=8,
        allocation_bytes=(64 * 128 + (64 * 128 // 4) * 16) * 2,
    )
    _prove_b16_transactions(dsl, relation, 512)
