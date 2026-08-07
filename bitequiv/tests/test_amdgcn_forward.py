"""CPU unit tests for the AMDGCN forward equivalence checker.

Two layers, no GPU:
  * the ISA-neutral collapse (:mod:`bitequiv.amdgcn.core.canonicalize`) — the num_warps-recovery
    mechanisms (Horner equal-height fold, reduction extent, additive-neutral strip);
  * the public descriptor (:func:`bitequiv.amdgcn.forward.interp.forward_module_descriptor`) — a
    matrix-core entry keys on the sound MFMA fence, a plain entry on the reconstructed tree, and the
    descriptor is deterministic + hashable.
"""
from bitequiv.amdgcn import parser
from bitequiv.amdgcn.core.canonicalize import _fold_equal_height, _reduction_extent, collapse_balanced
from bitequiv.amdgcn.core.treeir import FpOp, ITreeReduce, Leaf, OpaqueLeaf
from bitequiv.amdgcn.forward.interp import forward_module_descriptor
from bitequiv.amdgcn.mma import mma_fence


def _kernel(body):
    return ("\t.type\tk,@function\nk:\n" + body + "\n\ts_endpgm\n"
            "\t.amdgpu_metadata\namdhsa.kernels:\n  - .name: k\n"
            "    .max_flat_workgroup_size: 256\n\t.end_amdgpu_metadata\n")


# ---- collapse: num_warps recovery mechanisms ----------------------------


def test_horner_fold_merges_equal_height_countup():
    # add(ITree_h, ITree_h) of the same count-up computation IS the next balanced level: this is how a
    # low-num_warps reconstructed add-chain of unequal-height partials folds back, bottom-up, to the
    # high-num_warps single balanced node. Tested through _fold_equal_height (the collapse's fold pass).
    a = ITreeReduce("add.f32", "L[]", 9, ("up", ))
    b = ITreeReduce("add.f32", "L[]", 9, ("up", ))
    [res] = _fold_equal_height([FpOp("add", (".f32", ), (a, b))])
    assert isinstance(res, ITreeReduce) and res.height == 10 and res.op == "add.f32"


def test_horner_fold_refuses_unordered():
    # count-DOWN partials (a raw offset, not the canonical ("up",) marker) must NOT fold — unordered
    # is num_warps-dependent, so keeping the physical shape is the sound (over-split) choice.
    a = ITreeReduce("add.f32", "L[]", 9, ("16", ))
    b = ITreeReduce("add.f32", "L[]", 9, ("16", ))
    [res] = _fold_equal_height([FpOp("add", (".f32", ), (a, b))])
    assert not (isinstance(res, ITreeReduce) and res.height == 10)


def test_reduction_extent_counts_fan_in():
    assert _reduction_extent(FpOp("add", (".f32", ), (Leaf("c0"), Leaf("c1")))) == 2
    assert _reduction_extent(ITreeReduce("add.f32", "L[]", 3, ("up", ))) == 8  # 2**height
    # add over two ITree(h3) = 8 + 8 = 16 elements
    two = FpOp("add", (".f32", ), (ITreeReduce("add.f32", "L[]", 3, ("up", )),
                                   ITreeReduce("add.f32", "L[]", 3, ("up", ))))
    assert _reduction_extent(two) == 16


def test_neg_zero_is_additive_neutral():
    # add(x, -0.0) == x for every x; codegen seeds a sum accumulator with -0.0 (bit-reverse of 1).
    col = collapse_balanced(FpOp("add", (".f32", ), (Leaf("c0"), OpaqueLeaf("fpconst:-0.0"))))
    assert isinstance(col, Leaf) and col.coord == "c0"


# ---- matrix-core fence ---------------------------------------------------


def test_mma_fence_present_for_mfma():
    f = parser.parse(_kernel("\tv_mfma_f32_16x16x16_f16 v[0:3], v4, v5, v[0:3]"))[0]
    fence = mma_fence(f)
    assert fence is not None and "mma{" in fence and "f16" in fence


def test_mma_fence_tiling_invariant_non_fp8():
    # f16 MFMA is tile-invariant on gfx942: one vs two MFMA of the same dtype family -> SAME fence.
    one = mma_fence(parser.parse(_kernel("\tv_mfma_f32_16x16x16_f16 v[0:3], v4, v5, v[0:3]"))[0])
    two = mma_fence(parser.parse(_kernel(
        "\tv_mfma_f32_16x16x16_f16 v[0:3], v4, v5, v[0:3]\n"
        "\tv_mfma_f32_16x16x16_f16 v[0:3], v6, v7, v[0:3]"))[0])
    assert one == two


def test_no_mma_fence_for_plain_kernel():
    f = parser.parse(_kernel("\tv_add_f32_e32 v0, v0, v1"))[0]
    assert mma_fence(f) is None


# ---- public descriptor ---------------------------------------------------


def test_descriptor_is_hashable_and_deterministic():
    asm = _kernel("\tv_mfma_f32_16x16x16_f16 v[0:3], v4, v5, v[0:3]")
    d1, d2 = forward_module_descriptor(asm), forward_module_descriptor(asm)
    assert d1 == d2
    assert hash(d1) == hash(d2)  # a tuple of strings -> hashable, usable as a partition key


def test_gemm_and_plain_descriptors_differ():
    gemm = forward_module_descriptor(_kernel("\tv_mfma_f32_16x16x16_f16 v[0:3], v4, v5, v[0:3]"))
    plain = forward_module_descriptor(_kernel("\tv_add_f32_e32 v0, v0, v1"))
    assert gemm != plain
