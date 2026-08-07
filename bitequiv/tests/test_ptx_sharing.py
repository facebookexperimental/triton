"""Signature-stage sharing: one memo per value-DAG, not one per output root, plus the node budget.

The interpreter builds ONE value-DAG whose output roots share most of their nodes; the stage after
it used to re-materialize that DAG once per root with a fresh memo, so a subtree k roots share was
rebuilt k times (O(n^2) nodes alive at once). These tests pin the two things that fix has to keep
true: the collapsed / hashed trees are byte-identical to the per-root ones (including where the
per-root collapse BOUNDARY legitimately differs between roots), and the node budget fails CLOSED to
the conservative fingerprint instead of merging. CPU-only."""
import bitequiv.ptx.forward.interp as interp_mod
from bitequiv.ptx.builder import _forest_postorder, collapse_balanced, collapse_balanced_forest, tree_hash, tree_hashes
from bitequiv.ptx.forward.interp import forward_module_descriptor
from bitequiv.ptx.treeir import FpOp, ITreeReduce, Leaf, OpaqueOp
from bitequiv.ptx_reduction import ptx_header

_HDR = ptx_header()


def _leaf(i):
    return Leaf(f"c{i}", frozenset({i}))


def _unique_nodes(trees):
    """How many DISTINCT node objects the trees are made of (shared nodes counted once)."""
    seen, stack = set(), list(trees)
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        stack.extend(n.children)
    return len(seen)


def _opaque_chain(n):
    """``n`` outputs over a chain each output extends: ``out[i] = f(out[i-1], leaf_i)``. The scan
    shape — ``out[i]`` CONTAINS ``out[i-1]``, so the outputs share almost everything. An OpaqueOp is
    never a reduce node, so nothing collapses and every node goes through the rebuild path."""
    node, roots = _leaf(0), []
    for i in range(1, n + 1):
        node = OpaqueOp("cvt.rn.f32.f32", (node, _leaf(i)))
        roots.append(node)
    return roots


def _add_chain(n):
    """The same scan shape built from plain ``add`` combines, so the collapse pass has real
    decisions to make: the first two links are AVL-balanced (hence collapsible) and the rest are
    not, which is what makes a node's region membership depend on WHICH root you look from."""
    node, roots = _leaf(0), []
    for i in range(1, n + 1):
        node = FpOp("add", (".f32", ), (node, _leaf(i)))
        roots.append(node)
    return roots


# -- one memo per DAG, not one per root ---------------------------------------


def test_forest_postorder_visits_each_shared_node_once():
    roots = _opaque_chain(40)
    order = _forest_postorder(roots)
    assert len(order) == _unique_nodes(roots)  # every unique node exactly once
    assert len({id(n) for n in order}) == len(order)  # and no node twice
    assert [id(n) for n in order[:2]] == [id(_forest_postorder([roots[0]])[i]) for i in range(2)]


def test_collapse_forest_rebuilds_a_shared_subtree_once():
    n = 40
    roots = _opaque_chain(n)
    per_root = [collapse_balanced(r) for r in roots]
    shared = collapse_balanced_forest(roots)
    assert [t.sig() for t in shared] == [t.sig() for t in per_root]  # byte-identical output
    # A fresh memo per root copies the shared chain once per root -> quadratic (n(n+1)/2 combines);
    # one memo shared by all the roots rebuilds each node once -> linear (n combines).
    assert _unique_nodes(shared) == _unique_nodes(roots) == 2 * n + 1
    assert _unique_nodes(per_root) == n * (n + 1) // 2 + (n + 1)


def test_tree_hashes_match_per_root_hashes():
    roots = _opaque_chain(40)
    assert tree_hashes(roots) == [tree_hash(r) for r in roots]


# -- the parent-context risk: a shared node's collapse BOUNDARY is per-root ----


def test_shared_collapse_keeps_the_per_root_region_boundary():
    # `add(leaf, leaf)` and `add(that, leaf)` are both AVL-balanced, so both collapse; the third link
    # is not. So `roots[0]` is a region ROOT on its own (it collapses to an ITreeReduce) but is
    # SWALLOWED by the larger region under `roots[1]`. One shared memo must not let whichever answer
    # came first win for everyone -- that would move the collapse boundary and change a descriptor.
    roots = _add_chain(6)
    per_root = [collapse_balanced(r) for r in roots]
    assert isinstance(per_root[0], ITreeReduce)  # alone: its own region
    assert isinstance(collapse_balanced(roots[2]), FpOp)  # third link: not collapsible
    assert [t.sig() for t in collapse_balanced_forest(roots)] == [t.sig() for t in per_root]


def test_multi_output_descriptor_is_unchanged_by_sharing(monkeypatch):
    # End to end: the multi-output descriptor must be the same whether the outputs are collapsed and
    # hashed together or one at a time.
    ptx = _two_row_sums(4)
    shared = forward_module_descriptor(ptx)
    monkeypatch.setattr("bitequiv.ptx.forward.interp.collapse_balanced_forest",
                        lambda roots: [collapse_balanced(r) for r in roots])
    monkeypatch.setattr("bitequiv.ptx.forward.interp.tree_hashes", lambda roots: [tree_hash(r) for r in roots])
    assert forward_module_descriptor(ptx) == shared


# -- the node budget fails CLOSED ---------------------------------------------


def _balanced_sum4(ntid):
    """A 4-element balanced within-thread sum at a given ``.reqntid`` (= num_warps). The collapse
    recovers num_warps, so two of these with different ntid share ONE descriptor."""
    return (_HDR + f".visible .entry k(.param .u64 pa, .param .u64 po) .reqntid {ntid}, 1, 1\n{{\n"
            ".reg .b64 %rd<6>;\n.reg .f32 %f<10>;\n"
            "ld.param.u64 %rd1, [pa];\ncvta.to.global.u64 %rd2, %rd1;\n"
            "ld.param.u64 %rd3, [po];\ncvta.to.global.u64 %rd4, %rd3;\n"
            "ld.global.f32 %f1, [%rd2];\nld.global.f32 %f2, [%rd2+4];\n"
            "ld.global.f32 %f3, [%rd2+8];\nld.global.f32 %f4, [%rd2+12];\n"
            "add.f32 %f5, %f1, %f2;\nadd.f32 %f6, %f3, %f4;\nadd.f32 %f7, %f5, %f6;\n"
            "st.global.f32 [%rd4], %f7;\nret;\n}\n")


def _two_row_sums(ntid):
    """Two independent balanced 4-element sums stored as two outputs (a multi-output entry)."""
    return (_HDR + f".visible .entry k(.param .u64 pa, .param .u64 po) .reqntid {ntid}, 1, 1\n{{\n"
            ".reg .b64 %rd<6>;\n.reg .f32 %f<20>;\n"
            "ld.param.u64 %rd1, [pa];\ncvta.to.global.u64 %rd2, %rd1;\n"
            "ld.param.u64 %rd3, [po];\ncvta.to.global.u64 %rd4, %rd3;\n"
            "ld.global.f32 %f1, [%rd2];\nld.global.f32 %f2, [%rd2+4];\n"
            "ld.global.f32 %f3, [%rd2+8];\nld.global.f32 %f4, [%rd2+12];\n"
            "add.f32 %f5, %f1, %f2;\nadd.f32 %f6, %f3, %f4;\nadd.f32 %f7, %f5, %f6;\n"
            "add.f32 %f8, %f1, %f3;\nadd.f32 %f9, %f2, %f4;\nadd.f32 %f10, %f8, %f9;\n"
            "st.global.f32 [%rd4], %f7;\nst.global.f32 [%rd4+4], %f10;\nret;\n}\n")


def test_budget_floor_falls_back_to_the_fingerprint(monkeypatch):
    small, big = _balanced_sum4(32), _balanced_sum4(256)
    assert forward_module_descriptor(small) == forward_module_descriptor(big)  # merged by the collapse
    assert "fwd-incomplete" not in forward_module_descriptor(small)[0]
    monkeypatch.setattr(interp_mod, "_MAX_DAG_NODES", 1)
    (floored, ) = forward_module_descriptor(small)
    assert floored.startswith("fwd-incomplete|ntid=x32")
    # Fail CLOSED: the floor must SPLIT what the collapsed hash merged, never merge more.
    assert forward_module_descriptor(small) != forward_module_descriptor(big)
