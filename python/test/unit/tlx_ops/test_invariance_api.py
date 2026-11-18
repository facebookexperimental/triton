"""GPU-free API tests for `BitwiseInvarianceSpec`/`Space` and wrapper validation.

Validation runs before dispatch, so these tests use bare tensor stand-ins and
need no device. Enforcement coverage lives in the per-arch suites.
"""

import dataclasses
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import create_autospec

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
    assert enforced == {"sm90", "sm100", "gfx942", "gfx950"}
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


def _paged_decode_tensors(kv_layout="HND", packed=False, q_fp8=False, kv_fp8=False):
    import torch

    query = torch.empty((14, 16, 64), dtype=torch.float8_e4m3fn if q_fp8 else torch.bfloat16, device="cpu")
    shape = (4, 2, 8, 64, 8) if packed else ((4, 2, 64, 64) if kv_layout == "HND" else (4, 64, 2, 64))
    key = torch.empty(shape, dtype=torch.float8_e4m3fn if kv_fp8 else torch.bfloat16, device="cpu")
    return (query, key, torch.empty_like(key), torch.tensor([64, 65],
                                                            dtype=torch.int32), torch.zeros((2, 2), dtype=torch.int32))


@pytest.fixture
def paged_decode_dispatch(monkeypatch):
    import torch
    import triton.tlx.ops as ops
    from triton.tlx.ops._catalog import impl_for

    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())

    def bind(arch, op="paged_attention_decode"):
        implementation, spec = impl_for(op, arch=arch)
        mock = create_autospec(implementation, return_value=object())
        monkeypatch.setattr(ops, "impl_for", lambda name, **kwargs: (mock, spec))
        return mock

    return bind


@pytest.mark.parametrize("arch,kv_layout,packed,q_fp8,kv_fp8", [
    ("gfx1250", "HND", False, False, False),
    ("gfx1250", "HND", True, False, False),
    ("gfx950", "HND", False, False, False),
    ("gfx950", "NHD", False, False, False),
    ("gfx950", "HND", True, False, False),
    ("gfx950", "NHD", False, False, True),
    ("gfx950", "HND", False, True, True),
])
def test_paged_decode_forwards_supported_options(paged_decode_dispatch, arch, kv_layout, packed, q_fp8, kv_fp8):
    import torch
    import triton.tlx.ops as ops

    tensors = _paged_decode_tensors(kv_layout, packed, q_fp8, kv_fp8)
    scales = {name: torch.ones(1, dtype=torch.float32) for name in ("k_descale", "v_descale")} if kv_fp8 else {}
    if q_fp8:
        scales["q_descale"] = torch.ones(1, dtype=torch.float32)
    mock = paged_decode_dispatch(arch)
    result = ops.paged_attention_decode(*tensors, query_length=7, sliding_window=2048, max_context_len=8192,
                                        kv_layout=kv_layout, **scales)
    assert result is mock.return_value
    mock.assert_called_once()
    args, kwargs = mock.call_args
    assert all(actual is expected for actual, expected in zip(args[:5], tensors))
    assert args[5:] == (0.125, 7, 2048, 8192, None, None)
    if arch == "gfx1250":
        assert not kwargs
    else:
        assert kwargs["kv_layout"] == kv_layout
        assert all(kwargs[name] is scales.get(name) for name in ("k_descale", "v_descale", "q_descale"))


@pytest.mark.parametrize("arch,kv_layout", [("gfx1250", "HND"), ("gfx950", "HND"), ("gfx950", "NHD")])
def test_paged_decode_workspace_forwards_layout(paged_decode_dispatch, arch, kv_layout):
    import triton.tlx.ops as ops

    query, key, value, _, table = _paged_decode_tensors(kv_layout)
    mock = paged_decode_dispatch(arch, "paged_attention_decode_workspace")
    result = ops.paged_attention_decode_workspace(query, key, value, table, query_length=7, sliding_window=2048,
                                                  max_context_len=8192, kv_layout=kv_layout)
    assert result is mock.return_value
    mock.assert_called_once()
    args, kwargs = mock.call_args
    assert all(actual is expected for actual, expected in zip(args[:4], (query, key, value, table)))
    assert args[4:] == (7, 2048, 8192)
    assert kwargs == ({} if arch == "gfx1250" else {"kv_layout": kv_layout})


@pytest.mark.parametrize("workspace", [False, True])
@pytest.mark.parametrize("kv_layout,q_fp8,kv_fp8", [("NHD", False, False), ("HND", False, True), ("HND", True, True)])
def test_paged_decode_gfx1250_rejects_unsupported_options(paged_decode_dispatch, workspace, kv_layout, q_fp8, kv_fp8):
    import torch
    import triton.tlx.ops as ops

    query, key, value, lengths, table = _paged_decode_tensors(kv_layout, q_fp8=q_fp8, kv_fp8=kv_fp8)
    op = "paged_attention_decode_workspace" if workspace else "paged_attention_decode"
    mock = paged_decode_dispatch("gfx1250", op)
    with pytest.raises(InvalidInput, match="gfx1250 expects BF16 Q/K/V and kv_layout='HND'"):
        if workspace:
            ops.paged_attention_decode_workspace(query, key, value, table, query_length=7, kv_layout=kv_layout)
        else:
            scales = {name: torch.ones(1) for name in ("k_descale", "v_descale")} if kv_fp8 else {}
            if q_fp8:
                scales["q_descale"] = torch.ones(1)
            ops.paged_attention_decode(query, key, value, lengths, table, query_length=7, kv_layout=kv_layout, **scales)
    mock.assert_not_called()


@pytest.mark.parametrize("name", ["k_descale", "v_descale", "q_descale"])
def test_paged_decode_gfx1250_rejects_bf16_descales(paged_decode_dispatch, name):
    import torch
    import triton.tlx.ops as ops

    mock = paged_decode_dispatch("gfx1250")
    with pytest.raises(InvalidInput, match="descale"):
        ops.paged_attention_decode(*_paged_decode_tensors(), query_length=7, **{name: torch.ones(1)})
    mock.assert_not_called()


@pytest.mark.parametrize("shape", [(), (2, 1)])
def test_paged_decode_rejects_nonvector_lengths(shape):
    import torch
    import triton.tlx.ops as ops

    query, key, value, _, table = _paged_decode_tensors()
    with pytest.raises(InvalidInput, match="1-D int32 context_lens"):
        ops.paged_attention_decode(query, key, value, torch.zeros(shape, dtype=torch.int32), table, query_length=7)
