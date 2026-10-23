"""GPU-free API tests for `BitwiseInvarianceSpec`/`Space` and wrapper validation.

Validation runs before dispatch, so these tests use bare tensor stand-ins and
need no device. Enforcement coverage lives in the per-arch suites.
"""

import dataclasses
from types import SimpleNamespace

import pytest
from triton.tlx.ops import BitwiseInvarianceSpec, InvalidInput, Space, UnsupportedOp
from triton.tlx.ops import flash_attn as tlx_flash_attn
from triton.tlx.ops import grouped_gemm as tlx_grouped_gemm
from triton.tlx.ops import mm as tlx_mm
from triton.tlx.ops._catalog import any_enforces_invariance, invariant_vocab_for
from triton.tlx.ops._invariance import gemm_fixed_split_k, normalize_space

GEMM_VOCAB = frozenset({"M", "N", "K"})


def test_dims_canonicalizes_to_uppercase():
    assert BitwiseInvarianceSpec.dims("m", "K").invariant_dims == frozenset({"M", "K"})


def test_empty_spec_needs_nothing():
    spec = BitwiseInvarianceSpec()
    assert not spec.needs_bitwise()
    assert not spec.invariant_in_dim("M")
    assert spec.resolve(GEMM_VOCAB) is spec


def test_all_resolves_to_vocab():
    spec = BitwiseInvarianceSpec.all()
    assert spec.needs_bitwise()
    assert spec.resolve(GEMM_VOCAB).invariant_dims == GEMM_VOCAB


def test_all_rejects_explicit_dims():
    with pytest.raises(InvalidInput):
        BitwiseInvarianceSpec(invariant_dims=frozenset({"M"}), all_dims=True)


def test_unresolved_all_rejects_membership():
    with pytest.raises(InvalidInput):
        BitwiseInvarianceSpec.all().invariant_in_dim("M")


def test_resolve_rejects_unknown_dims():
    with pytest.raises(InvalidInput, match="ZZZ"):
        BitwiseInvarianceSpec.dims("M", "ZZZ").resolve(GEMM_VOCAB)


def test_spec_is_frozen_and_hashable():
    spec = BitwiseInvarianceSpec.dims("M")
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.all_dims = True
    assert len({spec, BitwiseInvarianceSpec.dims("M")}) == 1


def test_space_accepts_impl_specific_name():
    assert Space("heuristic_2cta").name == "heuristic_2cta"
    assert Space("bogus").name == "bogus"


def test_space_rejects_empty_name():
    with pytest.raises(InvalidInput):
        Space("")
    with pytest.raises(InvalidInput):
        Space(123)  # type: ignore[arg-type]


def test_space_rejects_non_spec_invariance():
    with pytest.raises(InvalidInput):
        Space("heuristic", invariance="M")


def test_space_defaults_to_no_invariance():
    assert Space("heuristic").invariance is None


def test_normalize_space():
    empty = BitwiseInvarianceSpec()
    assert normalize_space(None) == (None, empty)
    assert normalize_space(None, "heuristic") == ("heuristic", empty)
    assert normalize_space("full") == ("full", empty)
    spec = BitwiseInvarianceSpec.dims("M")
    assert normalize_space(Space("heuristic", invariance=spec)) == ("heuristic", spec)
    assert normalize_space(Space("full")) == ("full", empty)
    with pytest.raises(InvalidInput):
        normalize_space(42)


def test_all_with_empty_vocab_fails_closed():
    with pytest.raises(InvalidInput, match="vocabulary"):
        BitwiseInvarianceSpec.all().resolve(frozenset())
    assert not BitwiseInvarianceSpec().resolve(frozenset()).needs_bitwise()


def test_gemm_fixed_split_k():
    assert gemm_fixed_split_k(BitwiseInvarianceSpec()) is None
    assert gemm_fixed_split_k(BitwiseInvarianceSpec.dims("K")) == 1
    assert gemm_fixed_split_k(BitwiseInvarianceSpec.dims("M")) == 1
    assert gemm_fixed_split_k(BitwiseInvarianceSpec.all().resolve(GEMM_VOCAB)) == 1


def test_vocabularies():
    assert invariant_vocab_for("mm") == GEMM_VOCAB
    assert invariant_vocab_for("addmm") == GEMM_VOCAB
    assert invariant_vocab_for("mm_mxfp8") == GEMM_VOCAB
    assert invariant_vocab_for("flash_attn") == frozenset({"B", "H", "SQ", "SKV", "D"})
    assert invariant_vocab_for("flash_attn_mxfp8") == frozenset({"B", "H", "SQ", "SKV", "D"})
    assert invariant_vocab_for("flash_attn_varlen") == frozenset({"TQ", "TKV", "H", "D"})
    assert invariant_vocab_for("prepare_flash_attn_varlen_backward") == frozenset({"TQ", "TKV", "H", "D"})
    assert invariant_vocab_for("flash_attn_varlen_backward") == frozenset({"TQ", "TKV", "H", "D"})
    assert invariant_vocab_for("grouped_gemm") == frozenset({"G", "M", "N", "K"})
    assert invariant_vocab_for("grouped_gemm_mxfp8") == frozenset({"G", "M", "N", "K"})
    assert invariant_vocab_for("hstu_attn_dev") == frozenset({"B", "LQ", "LKV", "H", "D"})
    assert invariant_vocab_for("hstu_cross_attn") == frozenset({"B", "LQ", "LKV", "H", "D"})
    assert invariant_vocab_for("kimi_delta_attention") == frozenset({"T", "H", "D"})
    assert invariant_vocab_for("kda_paged_prefill") == frozenset({"T", "H", "D"})
    assert invariant_vocab_for("kda_recurrent_decode") == frozenset({"T", "H", "D"})
    assert invariant_vocab_for("no_such_op") == frozenset()


def _mm_tensors():
    a = SimpleNamespace(ndim=2, shape=(8, 16), dtype="fp16", device="cuda:0")
    b = SimpleNamespace(ndim=2, shape=(16, 32), dtype="fp16", device="cuda:0")
    return a, b


def test_mm_rejects_unknown_dim_before_dispatch():
    a, b = _mm_tensors()
    with pytest.raises(InvalidInput, match="ZZZ"):
        tlx_mm(a, b, space=Space("heuristic", invariance=BitwiseInvarianceSpec.dims("M", "ZZZ")))


def test_mm_enforced_per_arch():
    import triton.tlx.ops as tlx_ops
    from triton.tlx.ops._catalog import CATALOG

    spec = BitwiseInvarianceSpec.dims("M")
    assert any_enforces_invariance("mm")
    enforced = {entry.arch for entry in CATALOG if entry.op == "mm" and entry.enforces_invariance}
    assert enforced == {"sm90", "sm100"}
    for entry in [entry for entry in CATALOG if entry.op == "mm"]:
        if entry.arch in enforced:
            tlx_ops._check_enforced("mm", entry, spec)
        else:
            with pytest.raises(UnsupportedOp, match="does not enforce"):
                tlx_ops._check_enforced("mm", entry, spec)


def test_flash_attn_refuses_unenforced_spec():
    q = SimpleNamespace(ndim=4)
    with pytest.raises(UnsupportedOp, match="does not enforce"):
        tlx_flash_attn(q, q, q, space=Space("full", invariance=BitwiseInvarianceSpec.dims("B")))


def test_grouped_gemm_refuses_unenforced_spec():
    import torch

    a = torch.empty(4, 8)
    b = torch.empty(16, 8).t()
    with pytest.raises(UnsupportedOp, match="does not enforce"):
        tlx_grouped_gemm([a], [b], space=Space("heuristic", invariance=BitwiseInvarianceSpec.dims("G")))


def test_grouped_gemm_rejects_invalid_space_name():
    import torch

    a = torch.empty(4, 8)
    b = torch.empty(16, 8).t()
    with pytest.raises(InvalidInput, match="does not provide"):
        tlx_grouped_gemm([a], [b], space="typo")
    with pytest.raises(InvalidInput, match="does not provide"):
        tlx_grouped_gemm([a], [b], space=Space("full"))
