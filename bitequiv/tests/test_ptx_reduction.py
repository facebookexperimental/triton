"""Module-level tests for the standalone PTX reduction-tree descriptor
(``bitequiv/ptx_reduction.py``).

The descriptor reconstructs the floating-point reduction *tree* from PTX and labels each
leaf with its recovered layout coordinate, so two configs match only when they reduce in
the same bitwise order. The internals (def-use, affine address recovery, tree builder,
canonicalization) are covered by ``test_ptx_linker.py`` / ``test_ptx_affine.py`` /
``test_ptx_builder.py``; here we test the public descriptor + the fixture cross-checks,
including the headline FMA-contraction case the TTGIR checker is blind to.
"""
import hashlib
import os

import pytest

from bitequiv.ptx_reduction import (_Unparseable, ptx_reduction_descriptor, ptx_reductions_equivalent)

_FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "ptx")


def _fixture(name, stage="ptx"):
    path = os.path.join(_FIXTURES, f"{name}.{stage}")
    if not os.path.exists(path):
        pytest.skip(f"missing fixture {name}.{stage} (regenerate offline)")
    with open(path) as f:
        return f.read()


# ---------------------------------------------------------------------------
# Module-level behavior (no GPU)
# ---------------------------------------------------------------------------
def test_empty_or_commentonly_ptx_is_empty():
    assert ptx_reduction_descriptor("") == ()
    assert ptx_reduction_descriptor("// just a comment") == ()


def test_descriptor_is_a_hashable_tuple_of_strings():
    d = ptx_reduction_descriptor(_fixture("sum_nw4"))
    assert isinstance(d, tuple) and d and all(isinstance(s, str) for s in d)
    assert hash(d) is not None  # usable as a dict/set key in the prune registry


def test_unparseable_ptx_is_a_never_equal_singleton():
    g = ptx_reduction_descriptor("not ptx <<<>>> .version garbage")
    assert isinstance(g, _Unparseable)
    assert g != g  # never merges, even with itself


def test_mma_entry_gets_unanalyzed_guard_not_empty():
    # A tensor-core accumulation has no reducible tree here; it must not collapse to an
    # empty (always-equal) signature -> a conservative guard keeps it distinct.
    mma = (".visible .entry k()\n.reqntid 128\n{\n.reg .b32 %r<4>;\n"
           "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {%r1}, %r2, %r3;\nret;\n}\n")
    (sig, ) = ptx_reduction_descriptor(mma)
    assert "unanalyzed-mma" in sig


# ---------------------------------------------------------------------------
# Fixture cross-checks (real captured PTX; no GPU at test time)
# ---------------------------------------------------------------------------
_SUM = ["sum_nw1", "sum_nw2", "sum_nw4", "sum_nw8"]


def test_each_num_warps_is_a_distinct_ptx_class():
    sigs = {name: ptx_reduction_descriptor(_fixture(name)) for name in _SUM}
    assert len(set(sigs.values())) == len(_SUM), sigs


def _has_fma(name):
    from bitequiv.ptx.builder import build_trees
    from bitequiv.ptx.treeir import FpOp
    from pyptx.parser import parse
    func = [d for d in parse(_fixture(name)).directives if getattr(d, "is_entry", False)][0]
    seen, stack, found = set(), list(build_trees(func)), False
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        found = found or (isinstance(n, FpOp) and n.fused)
        stack.extend(n.children)
    return found


def test_dot_fusion_is_visible_at_ptx():
    # The headline: enable_fp_fusion on vs off compiles to byte-identical TTGIR (the
    # sibling bitequiv-m1 checker is blind to it) but a different PTX reduction tree --
    # fma vs mul+add -- which this checker catches.
    on, off = _fixture("dot_fuse_on"), _fixture("dot_fuse_off")
    assert not ptx_reductions_equivalent(on, off)
    assert _has_fma("dot_fuse_on") and not _has_fma("dot_fuse_off")


def test_same_ptx_is_equivalent_to_itself():
    assert ptx_reductions_equivalent(_fixture("sum_nw4"), _fixture("sum_nw4"))


def test_ptx_level_prunes_fma_contraction_end_to_end():
    # Via the public prune predicate at level="ptx": with the fused dot as reference, the
    # unfused dot is pruned -- the pair that matches at TTGIR is correctly split at PTX.
    from bitequiv.equivalence_ptx import reduction_equivalence_prune
    prune = reduction_equivalence_prune("ptx")
    ref = {"ptx": _fixture("dot_fuse_on")}
    cand = {"ptx": _fixture("dot_fuse_off")}
    assert prune("fused", ref, None)
    assert not prune("unfused", cand, None)
    assert set(prune.pruned) == {"unfused"}


# Golden signature hashes captured from the real fixtures. Pins the full reconstructed
# tree (layout-labeled leaves + fold order + butterfly shuffles + fusion) so any parser /
# builder / canonicalization regression is caught on real Triton PTX. Regenerate with
# hashlib.sha1("␟".join(ptx_reduction_descriptor(ptx)).encode()).hexdigest()[:12].
_GOLDEN = {
    "sum_nw1": "95a83aa1dc04",
    "sum_nw2": "be27f98c38b2",
    "sum_nw4": "c6fda1d478b8",
    "sum_nw8": "dca122a2b82e",
    "dot_fuse_on": "80acbd23018a",
    "dot_fuse_off": "32aa7e471b9a",
}


@pytest.mark.parametrize("name", sorted(_GOLDEN))
def test_fixture_golden_signature(name):
    d = ptx_reduction_descriptor(_fixture(name))
    h = hashlib.sha1("␟".join(d).encode()).hexdigest()[:12]
    assert h == _GOLDEN[name]
