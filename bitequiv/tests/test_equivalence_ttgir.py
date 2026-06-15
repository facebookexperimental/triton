"""Unit tests for the autotuner-facing TTGIR equivalence API
(`bitequiv/equivalence_ttgir.py`): the level registry, the `ir_config_prune`
predicate plumbing (incl. reference selection), and the adapters that wrap the
MLIR-native reduction checker.

The reduction checker's own behavior (warps/inner_tree/shape/combine/multi-operand/
MMA) is covered in `test_reduction_tree.py`. Tests that exercise the real checker
parse committed TTGIR fixtures (`tests/ttgir/*.ttgir`, from `gen_ttgir_fixtures.py`);
the prune-plumbing tests use opaque key functions and need no parsing. Requires the
built triton on PYTHONPATH for the TTGIR-parsing tests (calls into `libtriton.bitequiv`).
The PTX-level checker lives on the separate `bitequiv-m1-ptx` branch."""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from bitequiv.equivalence_ttgir import (  # noqa: E402
    reduction_signature, classify, reduction_equivalence_key, CHECKERS, ir_based_prune_configs,
    reduction_equivalence_prune)

_TTGIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ttgir")


def _fx(name):
    with open(os.path.join(_TTGIR, name + ".ttgir")) as f:
        return f.read()


# --- adapters that wrap the MLIR-native checker (fixture-backed) ---
def test_classify_groups_by_order():
    classes = classify(
        {"nw2": _fx("sum_uno_nw2"), "nw4a": _fx("sum_uno_nw4"), "nw4b": _fx("sum_uno_nw4b"), "nw8": _fx("sum_uno_nw8")})
    # nw2, nw8 each alone; nw4a + nw4b together.
    assert sorted(len(v) for v in classes.values()) == [1, 1, 2]
    assert len(classes) == 3


def test_equivalence_key_adapter_reads_ttgir():
    key_a = reduction_equivalence_key(None, {"ttgir": _fx("sum_uno_nw4")}, None)
    key_b = reduction_equivalence_key(None, {"ttgir": _fx("sum_uno_nw8")}, None)
    assert key_a == reduction_signature(_fx("sum_uno_nw4"))
    assert key_a != key_b


def test_no_reduction_yields_empty_signature():
    assert reduction_signature("") == ()


# --- level registry ---
def test_registry_has_ttgir_only():
    assert set(CHECKERS) == {"ttgir"}


# --- ir_config_prune predicate plumbing (opaque key functions; no parsing) ---
def test_ir_based_prune_keeps_reference_class():
    prune = ir_based_prune_configs(lambda config, asm, md: md)
    assert prune("c0", {}, "A")  # reference
    assert prune("c1", {}, "A")  # matches -> kept
    assert not prune("c2", {}, "B")  # differs -> pruned
    assert prune.classes["A"] == ["c0", "c1"]
    assert prune.pruned == {"c2": "not-equivalent-to-reference"}


def test_ir_based_prune_all_same_key_keeps_all():
    prune = ir_based_prune_configs(lambda c, a, m: 0)
    assert all(prune(f"c{i}", {}, None) for i in range(3))
    assert prune.pruned == {}
    assert len(prune.classes) == 1


def test_ir_based_prune_handles_falsy_reference_key():
    prune = ir_based_prune_configs(lambda c, a, m: None)
    assert prune("c0", {}, None)
    assert prune("c1", {}, None)
    assert prune.pruned == {}


def test_ir_based_prune_tuple_key_multilevel_semantics():
    prune = ir_based_prune_configs(lambda c, a, m: m)
    assert prune("ref", {}, (4, 256))
    assert prune("both_match", {}, (4, 256))
    assert not prune("l2_differs", {}, (4, 512))
    assert not prune("l1_differs", {}, (2, 256))
    assert set(prune.pruned) == {"l2_differs", "l1_differs"}


def test_reduction_equivalence_prune_ttgir_uses_signature():
    prune = reduction_equivalence_prune("ttgir")
    # reference = first config's TTGIR reduction order (warps=4); warps 2/8 reduce differently.
    assert prune("nw4", {"ttgir": _fx("sum_uno_nw4")}, None)
    assert not prune("nw2", {"ttgir": _fx("sum_uno_nw2")}, None)
    assert not prune("nw8", {"ttgir": _fx("sum_uno_nw8")}, None)
    assert set(prune.pruned) == {"nw2", "nw8"}
    assert len(prune.classes) == 3


def test_reduction_equivalence_prune_unknown_level_raises():
    with pytest.raises(ValueError):
        reduction_equivalence_prune("nope")


# --- the optional `reference` parameter (4th arg the autotuner passes) ---
def test_reference_arg_overrides_call_order():
    prune = ir_based_prune_configs(lambda config, asm, md: md)
    items = [("nw4", {}, "K4"), ("nw8", {}, "K8")]
    reference = items[1]  # reference order is K8
    kept = [c for (c, asm, md) in items if prune(c, asm, md, reference)]
    assert kept == ["nw8"]
    assert prune.pruned == {"nw4": "not-equivalent-to-reference"}


def test_reduction_equivalence_prune_external_anchor_ir_text():
    prune = reduction_equivalence_prune("ttgir", reference=_fx("sum_uno_nw8"))
    items = [("nw4", {"ttgir": _fx("sum_uno_nw4")}, None), ("nw8", {"ttgir": _fx("sum_uno_nw8")}, None)]
    kept = [c for (c, asm, md) in items if prune(c, asm, md, items[0])]
    assert kept == ["nw8"]  # only the warps=8 candidate matches the anchor


def test_reduction_equivalence_prune_external_anchor_compiled_kernel():

    class _Anchor:
        asm = {"ttgir": _fx("sum_uno_nw2")}
        metadata = None

    prune = reduction_equivalence_prune("ttgir", reference=_Anchor())
    items = [("nw4", {"ttgir": _fx("sum_uno_nw4")}, None), ("nw2", {"ttgir": _fx("sum_uno_nw2")}, None)]
    kept = [c for (c, asm, md) in items if prune(c, asm, md, items[0])]
    assert kept == ["nw2"]


def test_reference_resets_introspection_per_run():
    prune = ir_based_prune_configs(lambda config, asm, md: md)
    a, b, c = object(), object(), object()
    run1 = [(a, {}, "K1"), (b, {}, "K2")]
    for cfg, asm, md in run1:
        prune(cfg, asm, md, run1[0])
    assert set(prune.pruned) == {b}
    run2 = [(a, {}, "K1"), (c, {}, "K1")]  # fresh reference object -> resets state
    for cfg, asm, md in run2:
        prune(cfg, asm, md, run2[0])
    assert prune.pruned == {}
    assert len(prune.classes) == 1
