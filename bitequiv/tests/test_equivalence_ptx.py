"""Unit tests for the autotuner-facing PTX equivalence API
(`bitequiv/equivalence_ptx.py`): the level registry, the `ir_config_prune` predicate
plumbing (incl. reference selection), and the adapter that wraps the PTX reduction
checker.

The PTX checker's own parsing behavior (offsets / fusion / opcode multiset) is covered
in `test_ptx_reduction.py`. These tests use small inline PTX snippets (no GPU, no
compilation) for the checker adapter, and opaque key functions for the prune plumbing.
The TTGIR-level checker lives on the separate `bitequiv-m1` branch."""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from bitequiv.equivalence_ptx import (  # noqa: E402
    ptx_reduction_signature, CHECKERS, ir_based_prune_configs, reduction_equivalence_prune)


def _ptx(off=16, combine="add.f32"):
    """Minimal one-entry PTX reduction: load a leaf, butterfly-combine it, store the
    result. The new tree-based checker reconstructs ``store <- combine(leaf, shfl(leaf,
    OFF))``, so a different shuffle offset is a different tree (different signature)."""
    return f""".visible .entry k(.param .u64 .ptr .global k_param_0, .param .u64 .ptr .global k_param_1)
.reqntid 32
{{
  .reg .b32 %r<8>;
  .reg .b64 %rd<5>;
  ld.param.b64 %rd1, [k_param_0];
  ld.param.b64 %rd2, [k_param_1];
  mov.u32 %r1, %tid.x;
  mul.wide.u32 %rd3, %r1, 4;
  add.s64 %rd1, %rd1, %rd3;
  ld.global.b32 %r4, [%rd1];
  shfl.sync.bfly.b32 %r5, %r4, {off}, 31, -1;
  {combine} %r6, %r4, %r5;
  st.global.b32 [%rd2], %r6;
  ret;
}}
"""


# --- level registry ---
def test_registry_has_ptx_only():
    assert set(CHECKERS) == {"ptx"}


# --- the adapter that wraps the PTX checker ---
def test_ptx_checker_returns_signature():
    sig = ptx_reduction_signature(_ptx())
    assert sig != ()
    assert CHECKERS["ptx"](None, {"ptx": _ptx()}, None) == sig


def test_ptx_checker_empty_ptx_is_empty_signature():
    assert ptx_reduction_signature("") == ()


def test_ptx_equivalence_key_adapter_reads_ptx():
    key_a = CHECKERS["ptx"](None, {"ptx": _ptx(off=16)}, None)
    key_b = CHECKERS["ptx"](None, {"ptx": _ptx(off=8)}, None)
    assert key_a == ptx_reduction_signature(_ptx(off=16))
    assert key_a != key_b


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


def test_reduction_equivalence_prune_ptx_uses_signature():
    prune = reduction_equivalence_prune("ptx")
    # reference = first config's PTX reduction order; a different butterfly offset is a
    # different tree -> pruned.
    assert prune("ref", {"ptx": _ptx(off=16)}, None)
    assert prune("same", {"ptx": _ptx(off=16)}, None)
    assert not prune("diff", {"ptx": _ptx(off=8)}, None)
    assert set(prune.pruned) == {"diff"}


def test_reduction_equivalence_prune_default_level_is_ptx():
    prune = reduction_equivalence_prune()
    assert prune("ref", {"ptx": _ptx(off=16)}, None)
    assert not prune("diff", {"ptx": _ptx(off=8)}, None)


def test_reduction_equivalence_prune_unknown_level_raises():
    with pytest.raises(ValueError):
        reduction_equivalence_prune("nope")


# --- the optional `reference` parameter (4th arg the autotuner passes) ---
def test_reference_arg_overrides_call_order():
    prune = ir_based_prune_configs(lambda config, asm, md: md)
    items = [("a16", {}, "K16"), ("a8", {}, "K8")]
    reference = items[1]  # reference order is K8
    kept = [c for (c, asm, md) in items if prune(c, asm, md, reference)]
    assert kept == ["a8"]
    assert prune.pruned == {"a16": "not-equivalent-to-reference"}


def test_reduction_equivalence_prune_external_anchor_ir_text():
    prune = reduction_equivalence_prune("ptx", reference=_ptx(off=8))
    items = [("a16", {"ptx": _ptx(off=16)}, None), ("a8", {"ptx": _ptx(off=8)}, None)]
    kept = [c for (c, asm, md) in items if prune(c, asm, md, items[0])]
    assert kept == ["a8"]  # only the offset-8 candidate matches the anchor


def test_reduction_equivalence_prune_external_anchor_compiled_kernel():

    class _Anchor:
        asm = {"ptx": _ptx(off=8)}
        metadata = None

    prune = reduction_equivalence_prune("ptx", reference=_Anchor())
    items = [("a16", {"ptx": _ptx(off=16)}, None), ("a8", {"ptx": _ptx(off=8)}, None)]
    kept = [c for (c, asm, md) in items if prune(c, asm, md, items[0])]
    assert kept == ["a8"]


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
