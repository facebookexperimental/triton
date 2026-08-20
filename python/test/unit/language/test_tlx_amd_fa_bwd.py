"""Focused D64 tests for the AMD Flash-Attention backward tutorial."""

import ast
import dataclasses
import math
import re

import pytest
import torch
import triton

from triton.language.extra.tlx.tutorials import amd_fa_bwd
from triton.language.extra.tlx.tutorials.amd_fa_bwd import (
    ReferenceCase,
    _D64DQLaunch,
    _D64Dispatch,
    _D64_CAUSAL_GQA8_KV_SPLITS,
    _D64_CAUSAL_GQA8_VALIDATION_CASES,
    _D64_DELTA_NEGATED,
    _D64_DELTA_POSITIVE,
    _D64_DQ_KV_STAGES,
    _D64_GQA_DIRECT_D64,
    _D64_GQA_INDEPENDENT_D32,
    _D64_GQA_INTERLEAVED_D32,
    _D64_GQA_SIGNED,
    _D64_GQA_SPLIT_FAST,
    _D64_GQA_XCD,
    _D64_GQA_XCD_N_FAST,
    _D64_LSE_NATURAL_LOG,
    _D64_LSE_NEG_LOG2E,
    _D64_MHA_POSITIVE,
    _D64_VALIDATION_SHAPES,
    _allocate_bwd_d64_causal_gqa8_workspaces,
    _allocate_bwd_d64_fused_workspaces,
    _attn_bwd_d64_fused_dq_convert_kernel,
    _attn_bwd_d64_fused_n256_kernel,
    _attn_bwd_d64_fused_n256_update,
    _attn_bwd_dkdv_d64_causal_gqa8_kernel,
    _attn_bwd_dkdv_d64_causal_gqa8_reduce_kernel,
    _attn_bwd_dkdv_d64_direct_kernel,
    _attn_bwd_dkdv_d64_reduce_kernel,
    _attn_bwd_dq_d64_causal_finish32,
    _attn_bwd_dq_d64_causal_gqa8_kernel,
    _attn_bwd_dq_d64_causal_impl,
    _attn_bwd_dq_d64_causal_load_q64,
    _attn_bwd_dq_d64_causal_m256_unmasked_n32,
    _attn_bwd_dq_d64_causal_mha_kernel,
    _attn_bwd_dq_d64_causal_nslice,
    _attn_bwd_dq_d64_causal_score32,
    _attn_bwd_dq_d64_causal_step,
    _attn_bwd_dq_d64_causal_store_q64,
    _attn_bwd_dq_d64_direct_kernel,
    _attn_bwd_preprocess_kernel,
    _d64_causal_dkdv_first_query_block,
    _d64_causal_dq_key_blocks,
    _d64_causal_gqa8_batch_stats4,
    _d64_causal_owner_interval,
    _d64_causal_physical_frontier,
    _d64_causal_stat_values,
    _d64_causal_triangular_tail_schedule,
    _d64_decode_dq_pid,
    _d64_decode_gqa_pid,
    _d64_dq_launch_plan,
    _d64_encode_dq_pid,
    _d64_gqa8_async_stats4_direct_d64_impl,
    _d64_gqa8_d32_consume,
    _d64_gqa8_d32_impl,
    _d64_gqa8_d32_step,
    _d64_gqa8_direct_consume,
    _d64_gqa8_direct_d64_impl,
    _d64_gqa8_direct_d64_step,
    _d64_gqa8_issue_stage,
    _d64_gqa8_signed_front,
    _d64_gqa_grid_policy,
    _d64_gqa_lifetime,
    _d64_gqa_split_ownership,
    _d64_selected_causal_logical_n,
    _d64_selected_causal_owner_rows,
    _d64_use_dq_xcd,
    _is_d64_fused_n256_eligible,
    _is_d64_scheduled_causal_eligible,
    _launch_bwd_d64_causal_dq,
    _launch_bwd_d64_causal_gqa8_dkdv,
    _launch_bwd_d64_causal_gqa8_reduce,
    _launch_bwd_d64_causal_mha_dkdv,
    _launch_bwd_d64_dkdv,
    _launch_bwd_d64_fused_dq_convert,
    _launch_bwd_d64_fused_n256,
    _launch_bwd_d64_kv_reduce,
    _matrix_instr_nonkdim,
    _run_bwd_d64,
    _run_bwd_d64_direct,
    _run_bwd_preprocess,
    _select_d64_dispatch,
    _select_d64_dispatch_for_device,
    _validate_d64_sm_scale,
    fa_backward,
    is_hip_cdna4,
)


def _make_d64_gqa_smoke_case(shape=(1, 1, 1, 256, 256, 64), causal=False, seed=0, sm_scale=None):
    """Build a small D64 MHA/GQA reference with bottom-right causality."""
    batch, hq, hkv, sq, skv, head_dim = shape
    assert batch >= 1 and hq >= 1 and hkv >= 1 and hq % hkv == 0
    assert head_dim == 64
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    q = torch.randn((batch, hq, sq, head_dim), generator=generator, device="cuda", dtype=torch.bfloat16)
    k = torch.randn((batch, hkv, skv, head_dim), generator=generator, device="cuda", dtype=torch.bfloat16)
    v = torch.randn((batch, hkv, skv, head_dim), generator=generator, device="cuda", dtype=torch.bfloat16)
    do = torch.randn(q.shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    o = torch.empty_like(q)
    lse = torch.empty(q.shape[:-1], device="cuda", dtype=torch.float32)
    dq = torch.empty_like(q, dtype=torch.float32)
    dk = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32)
    if sm_scale is None:
        sm_scale = head_dim**-0.5
    group_size = hq // hkv
    causal_mask = None
    if causal:
        query_positions = torch.arange(sq, device="cuda")[:, None]
        key_positions = torch.arange(skv, device="cuda")[None, :]
        causal_mask = key_positions > query_positions + (skv - sq)

    for batch_idx in range(batch):
        for query_head in range(hq):
            kv_head = query_head // group_size
            q_ref = q[batch_idx, query_head].float().requires_grad_(True)
            k_ref = k[batch_idx, kv_head].float().requires_grad_(True)
            v_ref = v[batch_idx, kv_head].float().requires_grad_(True)
            scores = torch.matmul(q_ref, k_ref.transpose(0, 1)) * sm_scale
            if causal_mask is not None:
                scores = scores.masked_fill(causal_mask, float("-inf"))
            lse_ref = torch.logsumexp(scores, dim=-1)
            o_ref = torch.matmul(torch.softmax(scores, dim=-1), v_ref)
            grads = torch.autograd.grad(o_ref, (q_ref, k_ref, v_ref), do[batch_idx, query_head].float())
            with torch.no_grad():
                o[batch_idx, query_head].copy_(o_ref)
                lse[batch_idx, query_head].copy_(lse_ref)
                dq[batch_idx, query_head].copy_(grads[0])
                dk[batch_idx, kv_head].add_(grads[1])
                dv[batch_idx, kv_head].add_(grads[2])

    return ReferenceCase(q, k, v, o, do, lse, sm_scale, causal, (dq, dk, dv))


def _make_d64_aten_case(shape, seed, causal=False, sm_scale=None):
    """Build a full-size D64 case without materializing a dense score matrix."""
    batch, hq, hkv, sq, skv, head_dim = shape
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)

    def random(tensor_shape):
        return torch.randn(
            tensor_shape,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        ).contiguous()

    q = random((batch, hq, sq, head_dim))
    k = random((batch, hkv, skv, head_dim))
    v = random((batch, hkv, skv, head_dim))
    do = random(q.shape)
    if sm_scale is None:
        sm_scale = head_dim**-0.5
    state = torch.ops.aten._scaled_dot_product_flash_attention.default(q, k, v, 0.0, causal, False, scale=sm_scale)
    out, lse, cum_q, cum_k, max_q, max_k, rng, unused, _debug = state
    reference = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(
        do,
        q,
        k,
        v,
        out,
        lse,
        cum_q,
        cum_k,
        max_q,
        max_k,
        0.0,
        causal,
        rng,
        unused,
        scale=sm_scale,
    )
    return ReferenceCase(
        q,
        k,
        v,
        out.contiguous(),
        do,
        lse.contiguous(),
        sm_scale,
        causal,
        tuple(reference),
    )


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 1, 256, 256, 64), id="group1"),
        pytest.param((1, 4, 2, 256, 256, 64), id="group2-hkv2"),
        pytest.param((1, 3, 1, 256, 256, 64), id="group3"),
        pytest.param((1, 4, 1, 256, 256, 64), id="group4"),
        pytest.param((1, 16, 2, 256, 256, 64), id="group8-hkv2"),
    ],
)
def test_d64_public_contract_group_matrix_gfx950(shape, causal):
    group_size = shape[1] // shape[2]
    seed = 23 + group_size + int(causal)
    case = _make_d64_gqa_smoke_case(shape, causal=causal, seed=seed)
    actual_grads = fa_backward(*case.kernel_args)
    for name, actual, expected in zip(("dq", "dk", "dv"), actual_grads, case.grads):
        assert torch.isfinite(actual).all(), name
        relative_l2 = torch.linalg.vector_norm(actual.float() - expected) / torch.linalg.vector_norm(expected)
        assert relative_l2.item() < 5e-3, (name, relative_l2.item())


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_d64_causal_negative_scale_gfx950():
    case = _make_d64_gqa_smoke_case(
        (1, 1, 1, 256, 256, 64),
        causal=True,
        seed=47,
        sm_scale=-0.125,
    )
    actual_grads = fa_backward(*case.kernel_args)
    for name, actual, expected in zip(("dq", "dk", "dv"), actual_grads, case.grads):
        assert torch.isfinite(actual).all(), name
        expected_norm = torch.linalg.vector_norm(expected.float())
        assert expected_norm.item() > 0.0, name
        relative_l2 = torch.linalg.vector_norm(actual.float() - expected.float()) / expected_norm
        assert relative_l2.item() < 5e-3, (name, relative_l2.item())


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_d64_bottom_right_rectangular_causal_gqa_public_contract_gfx950():
    case = _make_d64_gqa_smoke_case((1, 8, 1, 256, 512, 64), causal=True, seed=41)
    actual_grads = fa_backward(*case.kernel_args)
    for name, actual, expected in zip(("dq", "dk", "dv"), actual_grads, case.grads):
        assert torch.isfinite(actual).all(), name
        relative_l2 = torch.linalg.vector_norm(actual.float() - expected) / torch.linalg.vector_norm(expected)
        assert relative_l2.item() < 5e-3, (name, relative_l2.item())


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize(
    ("causal", "expected_family", "seed"),
    (
        pytest.param(False, "noncausal_direct_n256", 127, id="noncausal"),
        pytest.param(True, "causal_m192", 131, id="bottom-right-causal"),
    ),
)
def test_d64_64_aligned_public_fallback_gfx950(monkeypatch, causal, expected_family, seed):
    case = _make_d64_gqa_smoke_case((1, 1, 1, 320, 384, 64), causal=causal, seed=seed)
    assert torch.count_nonzero(case.q).item() > 0
    dispatches = []
    original_run = _run_bwd_d64

    def record_dispatch(*args):
        dispatches.append(args[-1])
        return original_run(*args)

    monkeypatch.setitem(vars(amd_fa_bwd), "_run_bwd_d64", record_dispatch)
    actual_grads = fa_backward(*case.kernel_args)

    assert [dispatch.family for dispatch in dispatches] == [expected_family]
    assert not dispatches[0].selected_causal
    for name, actual, expected in zip(("dq", "dk", "dv"), actual_grads, case.grads):
        assert torch.isfinite(actual).all(), name
        assert torch.linalg.vector_norm(expected).item() > 0.0, name
        relative_l2 = torch.linalg.vector_norm(actual.float() - expected.float()) / torch.linalg.vector_norm(
            expected.float())
        assert relative_l2.item() < 5e-3, (name, relative_l2.item())


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize("case_name", ["mha_square_16k_causal", "gqa8_square_16k_causal"])
def test_d64_causal_m256_accuracy_gfx950(case_name):
    batch, sq, skv, hq, hkv, head_dim, causal = _D64_VALIDATION_SHAPES[case_name]
    assert causal and sq == skv == 16384
    dispatch = _select_d64_dispatch((batch, hq, sq, head_dim), (batch, hkv, skv, head_dim), causal)
    assert dispatch.family == "causal_m256"
    assert dispatch.kv_splits == (1 if hq == hkv else 4)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260807 + tuple(_D64_VALIDATION_SHAPES).index(case_name))

    def random(shape):
        return torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16).contiguous()

    q = random((batch, hq, sq, head_dim))
    k = random((batch, hkv, skv, head_dim))
    v = random((batch, hkv, skv, head_dim))
    do = random(q.shape)
    sm_scale = head_dim**-0.5
    state = torch.ops.aten._scaled_dot_product_flash_attention.default(q, k, v, 0.0, True, False, scale=sm_scale)
    o, lse, cum_q, cum_k, max_q, max_k, rng, unused, _debug = state
    reference = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(do, q, k, v, o, lse, cum_q, cum_k,
                                                                                    max_q, max_k, 0.0, True, rng,
                                                                                    unused, scale=sm_scale)

    actual = fa_backward(q, k, v, o.contiguous(), do, lse.contiguous(), sm_scale, True)

    for name, result, expected in zip(("dq", "dk", "dv"), actual, reference):
        assert torch.isfinite(result).all(), name
        relative_l2 = torch.linalg.vector_norm(result.float() - expected.float()) / torch.linalg.vector_norm(
            expected.float())
        assert relative_l2.item() < 5e-3, (name, relative_l2.item())


def test_d64_causal_stat_contract_formulas():
    generator = torch.Generator(device="cpu")
    generator.manual_seed(101)
    q = torch.randn((3, 4), generator=generator, dtype=torch.float32)
    k = torch.randn((5, 4), generator=generator, dtype=torch.float32)
    v = torch.randn((5, 4), generator=generator, dtype=torch.float32)
    o = torch.randn((3, 4), generator=generator, dtype=torch.float32)
    do = torch.randn((3, 4), generator=generator, dtype=torch.float32)
    lse = torch.randn((3, ), generator=generator, dtype=torch.float32)
    sm_scale = 0.5

    delta_mha, lse_term_mha = _d64_causal_stat_values(o, do, lse, sm_scale, _D64_MHA_POSITIVE)
    delta_gqa, lse_term_gqa = _d64_causal_stat_values(o, do, lse, sm_scale, _D64_GQA_SIGNED)
    with pytest.raises(ValueError, match=r"^unknown D64 stat mode 2$"):
        _d64_causal_stat_values(o, do, lse, sm_scale, 2)

    delta_positive = torch.sum(o.float() * do.float(), dim=-1)
    delta_signed = -delta_positive
    lse_term = -lse.float() * math.log2(math.e)

    torch.testing.assert_close(delta_mha, delta_positive)
    torch.testing.assert_close(delta_gqa, delta_signed)
    assert lse_term_mha is None
    torch.testing.assert_close(lse_term_gqa, lse_term)
    assert delta_gqa.dtype is lse_term_gqa.dtype is torch.float32
    assert delta_gqa.is_contiguous() and lse_term_gqa.is_contiguous()

    scores_mha = q.float() @ k.float().mT
    p_mha = torch.exp2((scores_mha * sm_scale - lse.float()[..., None]) * math.log2(math.e))
    ds_mha = p_mha * (do.float() @ v.float().mT - delta_positive[..., None])

    scores_gqa = q.float() @ k.float().mT
    p_gqa = torch.exp2(scores_gqa * (sm_scale * math.log2(math.e)) + lse_term[..., None])
    ds_gqa = p_gqa * (do.float() @ v.float().mT + delta_signed[..., None])

    torch.testing.assert_close(p_mha, p_gqa)
    torch.testing.assert_close(ds_mha, ds_gqa)


def test_d64_causal_owner_interval_exhaustive():
    for sq, owner_rows in (
        (4096, 192),
        (8192, 192),
        (12288, 192),
        (16384, 256),
        (16448, 192),
    ):
        owners = triton.cdiv(sq, owner_rows)
        for invalid_owner in (-1, owners):
            with pytest.raises(
                    ValueError,
                    match=r"^physical owner is outside the dQ grid$",
            ):
                _d64_causal_owner_interval(invalid_owner, sq, owner_rows)
        covered = []
        for physical_owner in range(owners):
            actual = _d64_causal_owner_interval(physical_owner, sq, owner_rows)
            expected = (
                max(
                    (owners - 1 - physical_owner) * owner_rows - (owners * owner_rows - sq),
                    0,
                ),
                min(
                    (owners - 1 - physical_owner) * owner_rows - (owners * owner_rows - sq) + owner_rows,
                    sq,
                ),
            )
            assert actual == expected
            covered.extend(range(*actual))
        assert len(covered) == len(set(covered))
        assert sorted(covered) == list(range(sq))


def test_d64_causal_dq_grid_bijection_exhaustive():
    launch_tiles = 5
    owner_pid_base = 3
    for batch, hq, hkv in ((2, 16, 16), (2, 64, 8)):
        for use_xcd in (
                False,
                _d64_use_dq_xcd(batch, hkv, 8192, 8192, 192),
        ):
            assert use_xcd is (batch * hkv % 8 == 0) if use_xcd else True
            decoded = []
            for pid in range(batch * hq * launch_tiles):
                coords = _d64_decode_dq_pid(
                    pid,
                    batch,
                    hq,
                    hkv,
                    launch_tiles,
                    use_xcd,
                    owner_pid_base,
                )
                decoded.append(coords)
                assert _d64_encode_dq_pid(
                    *coords,
                    batch,
                    hq,
                    hkv,
                    launch_tiles,
                    use_xcd,
                    owner_pid_base,
                ) == pid
            assert len(decoded) == len(set(decoded))
            assert set(decoded) == {(batch_id, hq_id, owner_pid_base + local_owner)
                                    for batch_id in range(batch)
                                    for hq_id in range(hq)
                                    for local_owner in range(launch_tiles)}


def test_d64_causal_dq_xcd_predicate_boundaries():
    assert not _d64_use_dq_xcd(1, 7, 8192, 8192, 192)
    assert _d64_use_dq_xcd(1, 8, 8192, 8192, 192)

    assert not _d64_use_dq_xcd(1, 8, 4096, 4096, 192)
    assert _d64_use_dq_xcd(1, 8, 4096, 4096, 256)
    assert _d64_use_dq_xcd(1, 8, 4096, 4160, 192)
    assert _d64_use_dq_xcd(1, 8, 4160, 4160, 192)
    assert _d64_use_dq_xcd(1, 8, 3904, 3904, 192)
    assert _d64_use_dq_xcd(1, 8, 5248, 5248, 192)
    assert _d64_use_dq_xcd(8, 1, 4096, 4096, 192)


def test_d64_causal_m192_launch_plan(monkeypatch):
    batch, hq, hkv = 4, 64, 8
    sq = skv = 8192
    owner_rows = 192
    cu_count = 256
    owners = triton.cdiv(sq, owner_rows)
    peeled = _d64_dq_launch_plan(
        batch,
        hq,
        hkv,
        sq,
        skv,
        owner_rows,
        cu_count,
        True,
    )
    assert peeled == (
        _D64DQLaunch(owners - 1, False, 0, owners - 1, 3, 0),
        _D64DQLaunch(1, False, owners - 1, 1, 2, 192),
    )

    false_boundaries = (
        (batch, hq, hkv, 8000, 8000, owner_rows, cu_count, True),
        (1, 7, 7, sq, skv, owner_rows, 1, True),
        (1, hq, hkv, sq, skv, owner_rows, cu_count, True),
        (batch, hq, hkv, sq, skv, owner_rows, cu_count, False),
        (batch, hq, hkv, 8256, 8256, owner_rows, cu_count, True),
    )
    for args in false_boundaries:
        local_owners = triton.cdiv(args[3], args[5])
        assert _d64_dq_launch_plan(*args) == (_D64DQLaunch(local_owners, args[-1], 0, 0, 3, 0), )

    with monkeypatch.context() as patch:
        patch.setattr(triton, "cdiv", lambda _sq, _owner_rows: 1)
        assert _d64_dq_launch_plan(
            batch,
            hq,
            hkv,
            sq,
            skv,
            owner_rows,
            cu_count,
            True,
        ) == (_D64DQLaunch(1, True, 0, 0, 3, 0), )


def test_d64_causal_m192_launch_coverage_and_order():
    batch, hq, hkv = 4, 64, 8
    sq = skv = 8192
    owner_rows = 192
    owners = triton.cdiv(sq, owner_rows)
    launches = _d64_dq_launch_plan(batch, hq, hkv, sq, skv, owner_rows, 256, True)
    assert launches == (
        _D64DQLaunch(owners - 1, False, 0, owners - 1, 3, 0),
        _D64DQLaunch(1, False, owners - 1, 1, 2, 192),
    )

    decoded_by_launch = []
    for launch in launches:
        decoded_by_launch.append([
            _d64_decode_dq_pid(
                pid,
                batch,
                hq,
                hkv,
                launch.launch_tiles,
                True,
                launch.owner_pid_base,
            ) for pid in range(batch * hq * launch.launch_tiles)
        ])

    assert all(coords[2] < owners - 1 for coords in decoded_by_launch[0])
    assert all(coords[2] == owners - 1 for coords in decoded_by_launch[1])
    combined = decoded_by_launch[0] + decoded_by_launch[1]
    assert len(combined) == len(set(combined))
    assert set(combined) == {(batch_id, hq_id, physical_owner)
                             for batch_id in range(batch)
                             for hq_id in range(hq)
                             for physical_owner in range(owners)}
    assert _d64_causal_owner_interval(owners - 1, sq, owner_rows) == (
        0,
        128,
    )


def test_d64_causal_owner_triangular_tail_schedule_exhaustive():
    for owner_fragments in (2, 3, 4):
        for tail_step in range(owner_fragments):
            modes = _d64_causal_triangular_tail_schedule(owner_fragments, owner_fragments, tail_step)
            assert modes.count("masked") == 1
            assert modes[tail_step] == "masked"
            assert modes[:tail_step] == ("skip", ) * tail_step
            assert modes[tail_step + 1:] == ("unmasked", ) * (owner_fragments - tail_step - 1)

        for valid_fragments in range(1, owner_fragments + 1):
            for tail_step in range(valid_fragments):
                modes = _d64_causal_triangular_tail_schedule(owner_fragments, valid_fragments, tail_step)
                assert modes.count("masked") == 1
                assert modes[tail_step] == "masked"
                assert all(mode == "skip" for mode in modes[valid_fragments:])


def test_d64_structural_dispatch_locked_host_interface():
    assert tuple(field.name for field in dataclasses.fields(_D64Dispatch)) == (
        "family",
        "owner_rows",
        "key_rows",
        "kv_splits",
        "selected_causal",
        "stat_mode",
        "dq_logical_n",
        "dq_use_xcd",
        "dq_launches",
        "gqa_grid_mode",
        "cyclic_query_split",
        "dkdv_lifetime",
    )
    for value in (
            _D64_MHA_POSITIVE,
            _D64_GQA_SIGNED,
            _D64_LSE_NATURAL_LOG,
            _D64_LSE_NEG_LOG2E,
            _D64_DELTA_POSITIVE,
            _D64_DELTA_NEGATED,
    ):
        assert type(value) is int
    assert dataclasses.asdict(_D64Dispatch("retained", 192, 64, 1)) == {
        "family": "retained",
        "owner_rows": 192,
        "key_rows": 64,
        "kv_splits": 1,
        "selected_causal": False,
        "stat_mode": _D64_MHA_POSITIVE,
        "dq_logical_n": 64,
        "dq_use_xcd": False,
        "dq_launches": (),
        "gqa_grid_mode": None,
        "cyclic_query_split": False,
        "dkdv_lifetime": None,
    }


def test_d64_structural_dispatch_dq_launcher_stat_mode_contract():
    q = torch.empty((1, 1, 64, 64), device="meta", dtype=torch.bfloat16)
    stats = torch.empty((1, 1, 64), device="meta", dtype=torch.float32)

    def launch(dispatch, lse_term):
        _launch_bwd_d64_causal_dq(
            q,
            q,
            q,
            q,
            q,
            stats,
            stats,
            lse_term,
            q,
            0.125,
            dispatch,
        )

    mha = _D64Dispatch("causal_scheduled_mha", 192, 64, 1, stat_mode=_D64_MHA_POSITIVE)
    with pytest.raises(ValueError) as exc_info:
        launch(mha, stats)
    assert exc_info.value.args == ("MHA positive dQ must not receive lse_term", )

    gqa = dataclasses.replace(mha, stat_mode=_D64_GQA_SIGNED)
    with pytest.raises(ValueError) as exc_info:
        launch(gqa, None)
    assert exc_info.value.args == ("GQA signed dQ requires lse_term", )

    unknown = dataclasses.replace(mha, stat_mode=7)
    with pytest.raises(ValueError) as exc_info:
        launch(unknown, None)
    assert exc_info.value.args == ("unknown dQ stat mode 7", )


def test_d64_structural_dispatch_direct_route_uses_selected_record(monkeypatch):
    q = torch.empty((1, 8, 256, 64), device="meta", dtype=torch.bfloat16)
    k = torch.empty((1, 1, 256, 64), device="meta", dtype=torch.bfloat16)
    stats = torch.empty((1, 8, 256), device="meta", dtype=torch.float32)
    dispatch = _D64Dispatch("causal_m192", owner_rows=192, key_rows=32, kv_splits=4)
    calls = []

    def forbid_reselection(*args, **kwargs):
        raise AssertionError("retained D64 route reselected dispatch")

    def preprocess(*args, **kwargs):
        calls.append(("preprocess", None))

    def launch_dq(*args, **kwargs):
        calls.append(("dq", args[-1]))

    def allocate(_k, kv_splits):
        calls.append(("allocate", kv_splits))
        return None, None

    def launch_dkdv(*args, **kwargs):
        calls.append(("dkdv", args[-1]))

    monkeypatch.setitem(vars(amd_fa_bwd), "_select_d64_dispatch", forbid_reselection)
    monkeypatch.setitem(vars(amd_fa_bwd), "_run_bwd_preprocess", preprocess)
    monkeypatch.setitem(vars(amd_fa_bwd), "_launch_bwd_d64_dq", launch_dq)
    monkeypatch.setitem(vars(amd_fa_bwd), "_allocate_bwd_d64_kv_partials", allocate)
    monkeypatch.setitem(vars(amd_fa_bwd), "_launch_bwd_d64_dkdv", launch_dkdv)

    _run_bwd_d64(
        q,
        k,
        k,
        q,
        q,
        stats,
        stats,
        q,
        k,
        k,
        0.125,
        True,
        dispatch,
    )

    assert calls == [
        ("preprocess", None),
        ("dq", dispatch),
        ("allocate", dispatch.kv_splits),
        ("dkdv", dispatch),
    ]


def test_d64_causal_gqa_grid_policy_validation_shapes():
    cu_count = 256
    expected = {
        "gqa8_square_16k_causal": (_D64_GQA_XCD, False),
        "gqa8_square_4k_causal": (_D64_GQA_XCD, False),
        "gqa8_rect_4k_16k_causal": (_D64_GQA_XCD, False),
        "gqa8_rect_4k_8k_causal": (_D64_GQA_XCD, False),
        "gqa8_rect_4k_12k_causal": (_D64_GQA_XCD, False),
    }
    for case_name, expected_policy in expected.items():
        batch, sq, skv, _hq, hkv, _d, causal = _D64_VALIDATION_SHAPES[case_name]
        assert causal
        assert _d64_gqa_grid_policy(batch, hkv, sq, skv, cu_count) == expected_policy

    cyclic = (8, 8, 16384, 16384, cu_count)
    assert _d64_gqa_grid_policy(*cyclic) == (_D64_GQA_XCD_N_FAST, True)

    # Negate each cyclic conjunct independently while retaining the others.
    assert _d64_gqa_grid_policy(8, 8, 8192, 16384, cu_count) == (
        _D64_GQA_XCD,
        False,
    )
    assert _d64_gqa_grid_policy(8, 8, 8192, 8192, cu_count) == (
        _D64_GQA_XCD,
        False,
    )
    assert _d64_gqa_grid_policy(1, 8, 16384, 16384, cu_count) == (
        _D64_GQA_XCD,
        False,
    )
    assert _d64_gqa_grid_policy(10, 7, 16384, 16384, cu_count) == (
        _D64_GQA_SPLIT_FAST,
        False,
    )

    boundary_cases = (
        ((1, 8, 4032, 4032, cu_count), (_D64_GQA_XCD, False)),
        ((1, 8, 4096, 4096, cu_count), (_D64_GQA_XCD, False)),
        ((4, 6, 8192, 8192, cu_count), (_D64_GQA_XCD, False)),
        ((4, 6, 8256, 8256, cu_count), (_D64_GQA_XCD, False)),
        ((4, 6, 4096, 16256, cu_count), (_D64_GQA_XCD, False)),
        ((4, 6, 4096, 16384, cu_count), (_D64_GQA_XCD, False)),
    )
    for arguments, expected_policy in boundary_cases:
        assert _d64_gqa_grid_policy(*arguments) == expected_policy


def test_d64_causal_gqa_grid_bijection_exhaustive():
    for batch, hkv, skv in (
        (2, 8, 256),
        (2, 8, 640),
        (4, 6, 384),
        (4, 6, 512),
        (4, 6, 640),
    ):
        assert (batch * hkv) % 8 == 0
        nt = triton.cdiv(skv, 128)
        total = batch * hkv * 4 * nt

        expected_orders = {
            _D64_GQA_SPLIT_FAST: [(batch_id, hkv_id, split, n)
                                  for batch_id in range(batch)
                                  for n in range(nt)
                                  for hkv_id in range(hkv)
                                  for split in range(4)],
            _D64_GQA_XCD: [(bkv // hkv, bkv % hkv, split, n)
                           for bkv_group in range(batch * hkv // 8)
                           for n in range(nt)
                           for split in range(4)
                           for xcd in range(8)
                           for bkv in (bkv_group * 8 + xcd, )],
            _D64_GQA_XCD_N_FAST: [(bkv // hkv, bkv % hkv, split, n)
                                  for bkv_group in range(batch * hkv // 8)
                                  for split in range(4)
                                  for n in range(nt)
                                  for xcd in range(8)
                                  for bkv in (bkv_group * 8 + xcd, )],
        }
        expected_coords = {(batch_id, hkv_id, split, n)
                           for batch_id in range(batch)
                           for hkv_id in range(hkv)
                           for split in range(4)
                           for n in range(nt)}
        assert len(expected_coords) == total

        for grid_mode, expected_order in expected_orders.items():
            decoded = [_d64_decode_gqa_pid(pid, batch, hkv, skv, grid_mode) for pid in range(total)]
            assert decoded == expected_order
            assert len(decoded) == len(set(decoded)) == total
            assert set(decoded) == expected_coords


def test_d64_causal_gqa_frontier_exhaustive():
    block_m, block_n = 64, 128
    for sq, skv in (
        (4096, 4096),
        (4096, 8192),
        (4096, 12288),
        (4096, 16384),
        (16384, 16384),
    ):
        diff = skv - sq
        m_blocks = triton.cdiv(sq, block_m)
        for n0 in range(0, skv, block_n):
            start_m_blk, masked = _d64_causal_physical_frontier(n0, sq, skv, block_m, block_n)
            expected_start = max((n0 - diff) // block_m, 0)
            expected_masked = tuple(m_blk for m_blk in range(expected_start, m_blocks)
                                    if n0 + block_n - 1 > m_blk * block_m + diff)
            assert start_m_blk == expected_start
            assert masked == expected_masked

            pair_start = (start_m_blk // 2) * 2
            batch_start = pair_start + (pair_start % 4)
            assert all(m_blk < batch_start + 2 for m_blk in masked if m_blk >= batch_start)

            # Every omitted block is wholly invalid, and every scheduled block
            # contains at least one valid (m, n) satisfying bottom-right causal.
            for m_blk in range(m_blocks):
                m0 = m_blk * block_m
                m_last = min(sq, m0 + block_m) - 1
                scheduled = m_blk >= start_m_blk
                assert scheduled is (n0 <= m_last + diff)
                tile_needs_mask = n0 + block_n - 1 > m0 + diff
                assert (m_blk in masked) is (scheduled and tile_needs_mask)
                if scheduled and not tile_needs_mask:
                    assert all(0 <= m < sq and 0 <= n < skv and n <= m + skv - sq
                               for m in range(m0, min(m0 + block_m, sq))
                               for n in range(n0, min(n0 + block_n, skv)))

        aligned_n0 = diff + block_n
        start_m_blk, masked = _d64_causal_physical_frontier(aligned_n0, sq, skv, block_m, block_n)
        assert start_m_blk > 0
        assert masked == (start_m_blk, start_m_blk + 1)

        if diff:
            zero_clamp_n0 = diff - block_n
            start_m_blk, masked = _d64_causal_physical_frontier(zero_clamp_n0, sq, skv, block_m, block_n)
            assert start_m_blk == 0
            assert masked == tuple(m_blk for m_blk in range(m_blocks)
                                   if zero_clamp_n0 + block_n - 1 > m_blk * block_m + diff)
            assert masked == ()


def test_d64_causal_dq_bulk_successor_in_range():
    for sq, skv in (
        (4096, 4096),
        (4096, 8192),
        (4096, 12288),
        (4096, 16384),
        (16384, 16384),
    ):
        for owner_rows in (192, 256):
            num_owners = triton.cdiv(sq, owner_rows)
            pad = num_owners * owner_rows - sq
            for physical_owner in range(num_owners):
                reverse_owner = num_owners - 1 - physical_owner
                raw_start = reverse_owner * owner_rows
                owner_start = max(raw_start - pad, 0)
                owner_end = raw_start + owner_rows - pad
                bulk_end_block = (owner_start + (skv - sq)) // 64
                end_n_block = min((owner_end - 1 + (skv - sq)) // 64 + 1, skv // 64)
                assert bulk_end_block < end_n_block


def test_d64_causal_mha_frontier_exhaustive():
    block_m, block_n = 32, 64
    for sq, skv in (
        (4096, 4096),
        (16384, 16384),
        (4096, 8192),
    ):
        diff = skv - sq
        m_blocks = triton.cdiv(sq, block_m)
        for n0 in range(0, skv, block_n):
            start_m_blk, masked = _d64_causal_physical_frontier(n0, sq, skv, block_m, block_n)
            expected_start = max((n0 - diff) // block_m, 0)
            expected_masked = tuple(m_blk for m_blk in range(expected_start, m_blocks)
                                    if n0 + block_n - 1 > m_blk * block_m + diff)
            assert start_m_blk == expected_start
            assert masked == expected_masked

            # Exhaust every physical BM32 block against bottom-right validity.
            for m_blk in range(m_blocks):
                m0 = m_blk * block_m
                m_last = min(sq, m0 + block_m) - 1
                any_valid = n0 <= m_last + diff
                all_valid = n0 + block_n - 1 <= m0 + diff
                scheduled = m_blk >= start_m_blk
                assert scheduled is any_valid
                assert (m_blk in masked) is (scheduled and not all_valid)

        aligned_n0 = diff + block_n
        start_m_blk, masked = _d64_causal_physical_frontier(aligned_n0, sq, skv, block_m, block_n)
        assert start_m_blk > 0
        assert masked == (start_m_blk, start_m_blk + 1)

        if diff:
            zero_clamp_n0 = diff - block_n
            start_m_blk, masked = _d64_causal_physical_frontier(zero_clamp_n0, sq, skv, block_m, block_n)
            assert start_m_blk == 0
            assert masked == tuple(m_blk for m_blk in range(m_blocks)
                                   if zero_clamp_n0 + block_n - 1 > m_blk * block_m + diff)
            assert masked == ()


def test_d64_causal_gqa_split_ownership():
    query_blocks = 13
    expected = {(head, m_blk) for head in range(8) for m_blk in range(query_blocks)}
    for cyclic in (False, True):
        by_split = [_d64_gqa_split_ownership(split, query_blocks, cyclic) for split in range(4)]
        for split, owned in enumerate(by_split):
            if cyclic:
                assert {head for head, _m_blk in owned} == set(range(8))
                assert all(m_blk % 4 == split for _head, m_blk in owned)
            else:
                assert {head
                        for head, _m_blk in owned} == {
                            2 * split,
                            2 * split + 1,
                        }
                assert all(0 <= m_blk < query_blocks for _head, m_blk in owned)
        flattened = [item for owned in by_split for item in owned]
        assert len(flattened) == len(set(flattened)) == len(expected)
        assert set(flattened) == expected


def test_d64_causal_gqa_lifetime_policy():
    for sq in (2048, 4096, 8192):
        assert _d64_gqa_lifetime(sq, sq) == _D64_GQA_DIRECT_D64
    for sq, skv in (
        (1024, 1152),
        (12288, 16384),
    ):
        assert _d64_gqa_lifetime(sq, skv) == _D64_GQA_INTERLEAVED_D32
    for sq, skv in (
        (4096, 8192),
        (4096, 12288),
        (4096, 16384),
    ):
        assert _d64_gqa_lifetime(sq, skv) == _D64_GQA_DIRECT_D64
    for sq in (1024, 12288, 16384):
        assert _d64_gqa_lifetime(sq, sq) == _D64_GQA_DIRECT_D64

    for sq in range(1024, 16385, 1024):
        assert _d64_gqa_lifetime(sq, sq) == _D64_GQA_DIRECT_D64
        assert _d64_gqa_lifetime(sq, sq + 128) == _D64_GQA_INTERLEAVED_D32
        assert _d64_gqa_lifetime(sq, 2 * sq - 128) == _D64_GQA_INTERLEAVED_D32
        assert _d64_gqa_lifetime(sq, 2 * sq) == _D64_GQA_DIRECT_D64
    with pytest.raises(ValueError, match="selected GQA8 shape has no lifetime mode"):
        _d64_gqa_lifetime(960, 960)


def test_d64_causal_gqa_d32_schedule_peels_odd_frontier_before_even_pairs():
    source = _d64_gqa8_d32_impl.src
    assert "pair_start = (start_m_blk // 2) * 2" not in source
    assert "peel_frontier = (start_m_blk % 2) != 0" in source
    assert "pair_start = start_m_blk + (start_m_blk % 2)" in source

    impl = ast.parse(source).body[0]
    scalar_assignments = {
        statement.targets[0].id: statement.value
        for statement in impl.body
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
    }

    def evaluate(name, start_m_blk):
        expression = ast.Expression(scalar_assignments[name])
        return eval(
            compile(ast.fix_missing_locations(expression), "<d32-jit>", "eval"),
            {},
            {"start_m_blk": start_m_blk},
        )

    def scheduled_groups(start_m_blk, num_m_blocks):
        groups = []
        if evaluate("peel_frontier", start_m_blk) and start_m_blk < num_m_blocks:
            groups.append((start_m_blk, ))
        pair_start = evaluate("pair_start", start_m_blk)
        for m_pair in range(pair_start // 2, num_m_blocks // 2):
            m_blk_a = m_pair * 2
            groups.append((m_blk_a, m_blk_a + 1))
        if num_m_blocks % 2 and pair_start < num_m_blocks:
            groups.append((num_m_blocks - 1, ))
        return tuple(groups)

    for num_m_blocks in (1, 2, 7, 17, 64):
        for start_m_blk in range(num_m_blocks + 1):
            groups = scheduled_groups(start_m_blk, num_m_blocks)
            scheduled = tuple(block for group in groups for block in group)
            assert scheduled == tuple(range(start_m_blk, num_m_blocks))
            if scheduled:
                assert scheduled[0] == start_m_blk
                assert len(scheduled) == len(set(scheduled))
            expected_groups = []
            next_m_blk = start_m_blk
            if next_m_blk % 2 and next_m_blk < num_m_blocks:
                expected_groups.append((next_m_blk, ))
                next_m_blk += 1
            while next_m_blk + 1 < num_m_blocks:
                expected_groups.append((next_m_blk, next_m_blk + 1))
                next_m_blk += 2
            if next_m_blk < num_m_blocks:
                expected_groups.append((next_m_blk, ))
            assert groups == tuple(expected_groups)
            assert all(len(group) == 1 or (len(group) == 2 and group[0] % 2 == 0) for group in groups)

    sq, skv, n0 = 1088, 1152, 128
    start_m_blk, _masked = _d64_causal_physical_frontier(n0, sq, skv, 64, 128)
    assert start_m_blk == 1
    assert scheduled_groups(start_m_blk, sq // 64) == (
        (1, ),
        (2, 3),
        (4, 5),
        (6, 7),
        (8, 9),
        (10, 11),
        (12, 13),
        (14, 15),
        (16, ),
    )


def test_d64_causal_gqa8_helper_ast_contract():

    def function_ast(jit_function):
        tree = ast.parse(jit_function.src)
        assert len(tree.body) == 1
        return tree.body[0]

    def dotted_name(call):
        value = call.func
        parts = []
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            parts.append(value.id)
        return ".".join(reversed(parts))

    def root_call_name(statement):
        if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.Expr)):
            value = statement.value
            if isinstance(value, ast.Call):
                return dotted_name(value)
        return None

    interesting = {
        "_d64_gqa8_issue_stage",
        "_d64_gqa8_d32_consume",
        "tlx.async_load_wait_group",
        "tl.debug_barrier",
    }

    def events(statements):
        return tuple(event for statement in statements if (event := root_call_name(statement)) in interesting)

    issue = function_ast(_d64_gqa8_issue_stage)
    load_tokens = []
    for statement in issue.body:
        if (isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call)
                and dotted_name(statement.value) == "tlx.buffer_load_to_local"):
            load_tokens.append(statement.targets[0].id)
    assert load_tokens == [
        "q_token",
        "do_token",
        "lse_token",
        "delta_token",
    ]
    commits = [
        node for node in ast.walk(issue)
        if isinstance(node, ast.Call) and dotted_name(node) == "tlx.async_load_commit_group"
    ]
    assert len(commits) == 1
    assert ast.unparse(commits[0].args[0]) == ("[q_token, do_token, lse_token, delta_token]")

    def assert_runtime_causal_mask(jit_function, step_name, mask_arg):
        consume = function_ast(jit_function)
        mask_assignment = next(node for node in consume.body if isinstance(node, ast.Assign) and len(node.targets) == 1
                               and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "apply_causal_mask")
        assert ast.unparse(mask_assignment.value) == ("n0 + BLOCK_N - 1 > m_blk * BLOCK_M + (SKV - SQ)")
        steps = [node for node in ast.walk(consume) if isinstance(node, ast.Call) and dotted_name(node) == step_name]
        assert len(steps) == 1
        assert ast.unparse(steps[0].args[mask_arg]) == "apply_causal_mask"
        assert not any(isinstance(node, ast.If) for node in consume.body)

    assert_runtime_causal_mask(_d64_gqa8_direct_consume, "_d64_gqa8_direct_d64_step", 17)
    assert_runtime_causal_mask(_d64_gqa8_d32_consume, "_d64_gqa8_d32_step", 19)

    signed_front = function_ast(_d64_gqa8_signed_front)
    vgpr_handoffs = [
        node for node in ast.walk(signed_front)
        if isinstance(node, ast.Call) and dotted_name(node) == "tlx.amd_register_handoff"
    ]
    assert len(vgpr_handoffs) == 1
    assert ast.unparse(vgpr_handoffs[0].args[0]) == "ds"
    assert ast.literal_eval(
        next(keyword.value for keyword in vgpr_handoffs[0].keywords if keyword.arg == "register_class")) == "vgpr"
    assert ast.unparse(
        next(keyword.value
             for keyword in vgpr_handoffs[0].keywords
             if keyword.arg == "registers_per_group")) == "handoff_group"
    handoff_assignment = next(
        node for node in signed_front.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "handoff_group")
    assert ast.unparse(handoff_assignment.annotation) == "tl.constexpr"
    assert ast.unparse(handoff_assignment.value) == "2 if LATE_DO_T else 1"

    def assignment_call(statement, target, call_name):
        return (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id == target
                and isinstance(statement.value, ast.Call) and dotted_name(statement.value) == call_name)

    early_do_index = next(index for index, statement in enumerate(signed_front.body)
                          if isinstance(statement, ast.If) and ast.unparse(statement.test) == "not LATE_DO_T")
    p_index = next(index for index, statement in enumerate(signed_front.body)
                   if assignment_call(statement, "p", "tlx.require_layout"))
    late_do_index = next(index for index, statement in enumerate(signed_front.body)
                         if isinstance(statement, ast.If) and ast.unparse(statement.test) == "LATE_DO_T")
    dp_dot_index = next(index for index, statement in enumerate(signed_front.body)
                        if assignment_call(statement, "dp", "tl.dot"))
    assert early_do_index < p_index < late_do_index < dp_dot_index
    for index in (early_do_index, late_do_index):
        assert any(assignment_call(statement, "do_t", "tlx.local_load") for statement in signed_front.body[index].body)

    for step_function, expected_late_do in (
        (_d64_gqa8_direct_d64_step, "True"),
        (_d64_gqa8_d32_step, "False"),
    ):
        signed_call = next(node for node in ast.walk(function_ast(step_function))
                           if isinstance(node, ast.Call) and dotted_name(node) == "_d64_gqa8_signed_front")
        assert ast.unparse(signed_call.args[16]) == expected_late_do

    step = function_ast(_d64_gqa8_d32_step)
    lifetime_ifs = [
        node for node in step.body if isinstance(node, ast.If) and ast.unparse(node.test) == "INTERLEAVED_D32"
    ]
    assert len(lifetime_ifs) == 1

    def dot_targets(statements):
        return [
            statement.targets[0].id for statement in statements if isinstance(statement, ast.Assign)
            and isinstance(statement.value, ast.Call) and dotted_name(statement.value) == "tl.dot"
        ]

    lifetime_if = lifetime_ifs[0]
    assert dot_targets(lifetime_if.body) == [
        "dv_d0",
        "dk_d0",
        "dv_d1",
        "dk_d1",
    ]
    assert dot_targets(lifetime_if.orelse) == [
        "dv_d0",
        "dv_d1",
        "dk_d0",
        "dk_d1",
    ]

    for loop_impl, expected_iters in (
        (_d64_gqa8_direct_d64_impl, {"range(0, 8)", "tl.static_range(0, 2)"}),
        (_d64_gqa8_d32_impl, {"tl.static_range(0, 2)"}),
    ):
        local_head_loops = [
            node for node in ast.walk(function_ast(loop_impl))
            if isinstance(node, ast.For) and ast.unparse(node.target) == "local_head"
        ]
        assert {ast.unparse(loop.iter) for loop in local_head_loops} == expected_iters

    kernel_impl = function_ast(_attn_bwd_dkdv_d64_causal_gqa8_kernel)
    qdo_stages = next(
        node for node in kernel_impl.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "qdo_stages")
    assert ast.unparse(qdo_stages.annotation) == "tl.constexpr"
    assert ast.unparse(qdo_stages.value) == "4 if BATCH_STATS4 else 2"

    stats4_impl = function_ast(_d64_gqa8_async_stats4_direct_d64_impl)
    stats4_quad_loop = next(node for node in ast.walk(stats4_impl)
                            if isinstance(node, ast.For) and ast.unparse(node.target) == "m_quad")
    assert ast.unparse(stats4_quad_loop.iter) == "range(first_quad, num_m_blocks // 4)"
    stats4_step_loop = next(node for node in ast.walk(stats4_quad_loop)
                            if isinstance(node, ast.For) and ast.unparse(node.target) == "step")
    stats4_assignments = {
        statement.targets[0].id: ast.unparse(statement.value)
        for statement in stats4_step_loop.body
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
    }
    assert stats4_assignments["current_slot"] == "step"
    assert stats4_assignments["apply_causal_mask"] == ("n0 + BLOCK_N - 1 > "
                                                       "(m0 + step) * BLOCK_M + (SKV - SQ)")
    next_slot_if = next(statement for statement in stats4_step_loop.body
                        if isinstance(statement, ast.If) and ast.unparse(statement.test) == "step + 1 < 4")
    assert any(
        isinstance(statement, ast.Assign) and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id == "next_slot"
        and ast.unparse(statement.value) == "step + 1" for statement in next_slot_if.body)
    barrier_guard = next(statement for statement in stats4_step_loop.body
                         if isinstance(statement, ast.If) and ast.unparse(statement.test) == "step % 2 == 1")
    assert events(barrier_guard.body) == ("tl.debug_barrier", )

    impl = function_ast(_d64_gqa8_d32_impl)
    head_loop = next(node for node in impl.body
                     if isinstance(node, ast.For) and ast.unparse(node.target) == "local_head")
    peel_if = next(node for node in head_loop.body
                   if isinstance(node, ast.If) and ast.unparse(node.test) == "peel_frontier")
    assert events(peel_if.body) == (
        "_d64_gqa8_issue_stage",
        "tlx.async_load_wait_group",
        "_d64_gqa8_d32_consume",
        "tl.debug_barrier",
    )
    pair_guard = next(node for node in head_loop.body
                      if isinstance(node, ast.If) and ast.unparse(node.test) == "pair_start < num_m_blocks")
    assert events(pair_guard.body[:1]) == ("_d64_gqa8_issue_stage", )
    pair_loop = next(node for node in pair_guard.body if isinstance(node, ast.For))
    assert ast.unparse(pair_loop.iter) == "range(pair_start // 2, full_pairs)"
    pair_assignments = {
        statement.targets[0].id: ast.unparse(statement.value)
        for statement in pair_loop.body
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
    }
    assert pair_assignments["m_blk_a"] == "m_pair * 2"
    assert pair_assignments["m_blk_b"] == "m_blk_a + 1"
    assert events(pair_loop.body) == (
        "_d64_gqa8_issue_stage",
        "tlx.async_load_wait_group",
        "_d64_gqa8_d32_consume",
        "_d64_gqa8_d32_consume",
    )
    following_if = next(node for node in pair_loop.body
                        if isinstance(node, ast.If) and ast.unparse(node.test) == "has_following")
    assert events(following_if.body) == (
        "tl.debug_barrier",
        "_d64_gqa8_issue_stage",
        "tlx.async_load_wait_group",
    )
    assert events(following_if.orelse) == ("tlx.async_load_wait_group", )
    tail_if = next(node for node in pair_guard.body if isinstance(node, ast.If) and ast.unparse(node.test) == "has_odd")
    assert events(tail_if.body)[:2] == (
        "tlx.async_load_wait_group",
        "_d64_gqa8_d32_consume",
    )


def test_d64_causal_gqa8_launch_mode_kwargs_same_shape(monkeypatch):

    class LaunchRecorder:

        def __init__(self):
            self.calls = []

        def __getitem__(self, grid):

            def record(*args, **kwargs):
                self.calls.append((grid, args, kwargs))

            return record

    recorder = LaunchRecorder()
    monkeypatch.setitem(vars(amd_fa_bwd), "_attn_bwd_dkdv_d64_causal_gqa8_kernel", recorder)
    q = torch.empty((1, 8, 1024, 64), device="meta", dtype=torch.bfloat16)
    k = torch.empty((1, 1, 1024, 64), device="meta", dtype=torch.bfloat16)
    stats = torch.empty((1, 8, 1024), device="meta", dtype=torch.float32)
    partial = torch.empty((1, 1, 4, 1024, 64), device="meta", dtype=torch.bfloat16)
    expected = (
        (
            _D64_GQA_SPLIT_FAST,
            False,
            False,
            False,
            _D64_GQA_INDEPENDENT_D32,
        ),
        (_D64_GQA_XCD, False, True, False, _D64_GQA_INDEPENDENT_D32),
        (
            _D64_GQA_XCD_N_FAST,
            False,
            True,
            True,
            _D64_GQA_INDEPENDENT_D32,
        ),
        (_D64_GQA_XCD_N_FAST, True, True, True, _D64_GQA_DIRECT_D64),
    )
    for grid_mode, cyclic, _use_xcd, _use_n_fast, lifetime in expected:
        dispatch = _D64Dispatch(
            "causal_scheduled_gqa8",
            owner_rows=192,
            key_rows=128,
            kv_splits=4,
            selected_causal=True,
            stat_mode=_D64_GQA_SIGNED,
            gqa_grid_mode=grid_mode,
            cyclic_query_split=cyclic,
            dkdv_lifetime=lifetime,
        )
        _launch_bwd_d64_causal_gqa8_dkdv(q, k, k, q, stats, stats, partial, partial, 0.125, dispatch)

    assert len(recorder.calls) == len(expected)
    for call, (grid_mode, cyclic, use_xcd, use_n_fast, lifetime) in zip(recorder.calls, expected):
        grid, _args, kwargs = call
        assert grid == (32, )
        assert kwargs["USE_GQA_XCD"] is use_xcd
        assert kwargs["USE_XCD_N_FAST"] is use_n_fast
        assert kwargs["CYCLIC_QUERY_SPLIT"] is cyclic
        assert kwargs["LIFETIME_MODE"] == lifetime
        assert kwargs["LSE_MODE"] == _D64_LSE_NEG_LOG2E
        assert kwargs["DELTA_MODE"] == _D64_DELTA_NEGATED


def test_d64_causal_gqa_workspace_contract(monkeypatch):
    batch, hq, hkv, sq, skv, head_dim = 4, 48, 6, 4096, 16384, 64
    q = torch.empty((batch, hq, sq, head_dim), device="meta", dtype=torch.bfloat16)
    k = torch.empty((batch, hkv, skv, head_dim), device="meta", dtype=torch.bfloat16)
    lse_term, dk_part, dv_part = _allocate_bwd_d64_causal_gqa8_workspaces(q, k)
    assert tuple(lse_term.shape) == (batch, hq, sq)
    assert lse_term.dtype == torch.float32
    assert lse_term.is_contiguous()
    partial_shape = (batch, hkv, 4, skv, head_dim)
    for partial in (dk_part, dv_part):
        assert tuple(partial.shape) == partial_shape
        assert partial.dtype == torch.bfloat16
        assert partial.is_contiguous()

    def forbidden_allocator(*_args, **_kwargs):
        raise AssertionError("MHA and generic routes must not allocate GQA workspaces")

    monkeypatch.setitem(
        vars(amd_fa_bwd),
        "_allocate_bwd_d64_causal_gqa8_workspaces",
        forbidden_allocator,
    )
    monkeypatch.setitem(vars(amd_fa_bwd), "_launch_bwd_d64_causal_dq", lambda *_args: None)
    monkeypatch.setitem(vars(amd_fa_bwd), "_launch_bwd_d64_causal_mha_dkdv", lambda *_args: None)
    monkeypatch.setitem(vars(amd_fa_bwd), "_launch_bwd_d64_dkdv", lambda *_args: None)
    monkeypatch.setitem(vars(amd_fa_bwd), "_run_bwd_preprocess", lambda *_args, **_kwargs: None)
    monkeypatch.setitem(vars(amd_fa_bwd), "_run_bwd_d64_direct", lambda *_args: None)

    def exercise_without_gqa_workspace(q_shape, k_shape, dispatch):
        q_meta = torch.empty(q_shape, device="meta", dtype=torch.bfloat16)
        k_meta = torch.empty(k_shape, device="meta", dtype=torch.bfloat16)
        stats = torch.empty(q_shape[:-1], device="meta", dtype=torch.float32)
        _run_bwd_d64(
            q_meta,
            k_meta,
            k_meta,
            q_meta,
            q_meta,
            stats,
            stats,
            q_meta,
            k_meta,
            k_meta,
            0.125,
            True,
            dispatch,
        )

    mha_shape = (4, 64, 4096, 64)
    mha = _select_d64_dispatch(
        mha_shape,
        mha_shape,
        True,
        arch="gfx950",
        cu_count=256,
        sm_scale=0.125,
        bases_aligned_16=True,
    )
    assert mha.family == "causal_scheduled_mha"
    exercise_without_gqa_workspace(mha_shape, mha_shape, mha)

    generic_q = (1, 8, 1024, 64)
    generic_k = (1, 1, 1024, 64)
    generic = _select_d64_dispatch(generic_q, generic_k, True)
    assert not generic.selected_causal
    exercise_without_gqa_workspace(generic_q, generic_k, generic)


def test_d64_causal_mha_workspace_and_launch_contract(monkeypatch):

    class LaunchRecorder:

        def __init__(self):
            self.calls = []

        def __getitem__(self, grid):

            def record(*args, **kwargs):
                self.calls.append((grid, args, kwargs))

            return record

    batch, heads, sq, skv, head_dim = 2, 32, 16384, 16384, 64
    q = torch.empty(
        (batch, heads, sq, head_dim),
        device="meta",
        dtype=torch.bfloat16,
    )
    k = torch.empty(
        (batch, heads, skv, head_dim),
        device="meta",
        dtype=torch.bfloat16,
    )
    v = torch.empty_like(k)
    o = torch.empty_like(q)
    do = torch.empty_like(q)
    lse = torch.empty((batch, heads, sq), device="meta", dtype=torch.float32)
    delta = torch.empty_like(lse)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dispatch = _select_d64_dispatch(
        tuple(q.shape),
        tuple(k.shape),
        True,
        arch="gfx950:sramecc+:xnack-",
        cu_count=256,
        sm_scale=0.125,
        bases_aligned_16=True,
    )
    assert dispatch.family == "causal_scheduled_mha"
    assert dispatch.stat_mode == _D64_MHA_POSITIVE
    assert dispatch.kv_splits == 1

    launches = []
    real_launcher = vars(amd_fa_bwd).get("_launch_bwd_d64_causal_mha_dkdv")

    def record_dq(*args):
        launches.append(("dq", args[7], args[6], args[8], args[-1]))

    def record_mha(*args):
        launches.append((
            "mha",
            args[4],
            args[5],
            args[6],
            args[7],
            args[-1],
        ))

    def reject_forbidden(*_args, **_kwargs):
        raise AssertionError("selected causal MHA must not preprocess, allocate partials, "
                             "reduce, convert, publish atomically, or use the retained producer")

    monkeypatch.setitem(vars(amd_fa_bwd), "_launch_bwd_d64_causal_dq", record_dq)
    monkeypatch.setitem(vars(amd_fa_bwd), "_launch_bwd_d64_causal_mha_dkdv", record_mha)
    for name in (
            "_run_bwd_preprocess",
            "_allocate_bwd_d64_causal_gqa8_workspaces",
            "_allocate_bwd_d64_kv_partials",
            "_launch_bwd_d64_causal_gqa8_dkdv",
            "_launch_bwd_d64_causal_gqa8_reduce",
            "_launch_bwd_d64_kv_reduce",
            "_launch_bwd_d64_fused_dq_convert",
            "_launch_bwd_d64_dkdv",
    ):
        monkeypatch.setitem(vars(amd_fa_bwd), name, reject_forbidden)

    _run_bwd_d64(
        q,
        k,
        v,
        o,
        do,
        lse,
        delta,
        dq,
        dk,
        dv,
        0.125,
        True,
        dispatch,
    )

    assert [launch[0] for launch in launches] == ["dq", "mha"]
    dq_launch, mha_launch = launches
    assert dq_launch[1] is None
    assert dq_launch[2] is delta is mha_launch[2]
    assert dq_launch[3] is dq
    assert mha_launch[1] is lse
    assert mha_launch[3] is dk
    assert mha_launch[4] is dv
    assert dq_launch[4] is mha_launch[5] is dispatch

    # Exercise the real launcher separately so this test covers both the
    # _run_bwd_d64 policy and the kernel launch ABI, not just a wished-for mock.
    assert callable(real_launcher)
    recorder = LaunchRecorder()
    monkeypatch.setitem(vars(amd_fa_bwd), "_attn_bwd_dkdv_d64_causal_mha_kernel", recorder)
    real_launcher(q, k, v, do, lse, delta, dk, dv, 0.125, dispatch)

    assert len(recorder.calls) == 1
    grid, args, kwargs = recorder.calls[0]
    assert grid == (batch * heads * triton.cdiv(skv, 64), )
    assert args == (q, k, v, do, lse, delta, dk, dv)
    assert kwargs["SM_SCALE"] == 0.125
    assert kwargs["HQ"] == kwargs["HKV"] == heads
    assert kwargs["SQ"] == sq
    assert kwargs["SKV"] == skv
    assert kwargs["D"] == head_dim
    assert kwargs["BLOCK_M"] == 32
    assert kwargs["BLOCK_N"] == 64
    assert kwargs["LSE_MODE"] == _D64_LSE_NATURAL_LOG
    assert kwargs["DELTA_MODE"] == _D64_DELTA_POSITIVE
    assert kwargs["num_warps"] == 2
    assert kwargs["matrix_instr_nonkdim"] == _matrix_instr_nonkdim()


def test_d64_causal_selected_dispatch_contract():
    arch = "gfx950:sramecc+:xnack-"
    cu_count = 256

    def is_eligible(q_shape, k_shape, **overrides):
        return _is_d64_scheduled_causal_eligible(
            q_shape,
            k_shape,
            True,
            arch=overrides.get("arch", arch),
            cu_count=overrides.get("cu_count", cu_count),
            sm_scale=overrides.get("sm_scale", 0.125),
            bases_aligned_16=overrides.get("bases_aligned_16", True),
        )

    def select(q_shape, k_shape):
        return _select_d64_dispatch(
            q_shape,
            k_shape,
            True,
            arch=arch,
            cu_count=cu_count,
            sm_scale=0.125,
            bases_aligned_16=True,
        )

    mha_m192 = ((4, 64, 4096, 64), (4, 64, 4096, 64))
    mha_m256 = ((2, 32, 16384, 64), (2, 32, 16384, 64))
    for q_shape, k_shape in (mha_m192, mha_m256):
        assert is_eligible(q_shape, k_shape)
        dispatch = select(q_shape, k_shape)
        assert dispatch.owner_rows == _d64_selected_causal_owner_rows(
            q_shape[2],
            k_shape[2],
            q_shape[1] // k_shape[1],
        )
        assert dispatch.family == "causal_scheduled_mha"
        assert dispatch.selected_causal
        assert dispatch.stat_mode == _D64_MHA_POSITIVE
        assert dispatch.key_rows == 64
        assert dispatch.dq_logical_n == 32
        assert dispatch.kv_splits == 1
        assert dispatch.dq_use_xcd is _d64_use_dq_xcd(q_shape[0], k_shape[1], q_shape[2], k_shape[2],
                                                      dispatch.owner_rows)
        assert dispatch.dq_launches == _d64_dq_launch_plan(
            q_shape[0],
            q_shape[1],
            k_shape[1],
            q_shape[2],
            k_shape[2],
            dispatch.owner_rows,
            cu_count,
            True,
        )
    assert select(*mha_m192).owner_rows == 192
    assert select(*mha_m256).owner_rows == 256

    peeled = select((4, 64, 8192, 64), (4, 64, 8192, 64))
    owners = triton.cdiv(8192, 192)
    assert peeled.dq_use_xcd
    assert peeled.dq_launches == (
        _D64DQLaunch(owners - 1, False, 0, owners - 1, 3, 0),
        _D64DQLaunch(1, False, owners - 1, 1, 2, 192),
    )

    gqa_cases = [
        ((2, 128, 1024, 64), (2, 16, 1024, 64)),
        *[(
            (batch, hq, sq, head_dim),
            (batch, hkv, skv, head_dim),
        )
          for name in _D64_CAUSAL_GQA8_VALIDATION_CASES
          for batch, sq, skv, hq, hkv, head_dim, causal in (_D64_VALIDATION_SHAPES[name], )
          if causal],
    ]
    for q_shape, k_shape in gqa_cases:
        assert q_shape[1] == 8 * k_shape[1]
        assert is_eligible(q_shape, k_shape)
        dispatch = select(q_shape, k_shape)
        assert dispatch.owner_rows == _d64_selected_causal_owner_rows(
            q_shape[2],
            k_shape[2],
            q_shape[1] // k_shape[1],
        )
        expected_grid, expected_cyclic = _d64_gqa_grid_policy(q_shape[0], k_shape[1], q_shape[2], k_shape[2], cu_count)
        assert dispatch.family == "causal_scheduled_gqa8"
        assert dispatch.selected_causal
        assert dispatch.stat_mode == _D64_GQA_SIGNED
        assert dispatch.key_rows == 128
        assert dispatch.kv_splits == 4
        assert dispatch.gqa_grid_mode == expected_grid
        assert dispatch.cyclic_query_split is expected_cyclic
        assert dispatch.dkdv_lifetime == _d64_gqa_lifetime(q_shape[2], k_shape[2])
        assert dispatch.dq_logical_n == _d64_selected_causal_logical_n(
            q_shape[2],
            k_shape[2],
            q_shape[1] // k_shape[1],
        )
        assert dispatch.dq_use_xcd is _d64_use_dq_xcd(q_shape[0], k_shape[1], q_shape[2], k_shape[2],
                                                      dispatch.owner_rows)
        assert dispatch.dq_launches == _d64_dq_launch_plan(
            q_shape[0],
            q_shape[1],
            k_shape[1],
            q_shape[2],
            k_shape[2],
            dispatch.owner_rows,
            cu_count,
            True,
        )
    assert [
        select(
            (batch, hq, sq, head_dim),
            (batch, hkv, skv, head_dim),
        ).owner_rows
        for case_name in _D64_CAUSAL_GQA8_VALIDATION_CASES
        for batch, sq, skv, hq, hkv, head_dim, _causal in (_D64_VALIDATION_SHAPES[case_name], )
    ] == [256, 256, 256, 256, 256]
    assert [
        select(
            (batch, hq, sq, head_dim),
            (batch, hkv, skv, head_dim),
        ).dkdv_lifetime
        for case_name in _D64_CAUSAL_GQA8_VALIDATION_CASES
        for batch, sq, skv, hq, hkv, head_dim, _causal in (_D64_VALIDATION_SHAPES[case_name], )
    ] == [
        _D64_GQA_DIRECT_D64,
        _D64_GQA_DIRECT_D64,
        _D64_GQA_DIRECT_D64,
        _D64_GQA_DIRECT_D64,
        _D64_GQA_DIRECT_D64,
    ]
    assert [
        _d64_causal_gqa8_batch_stats4(
            sq,
            skv,
            select(
                (batch, hq, sq, head_dim),
                (batch, hkv, skv, head_dim),
            ),
        )
        for case_name in _D64_CAUSAL_GQA8_VALIDATION_CASES
        for batch, sq, skv, hq, hkv, head_dim, _causal in (_D64_VALIDATION_SHAPES[case_name], )
    ] == [True, True, True, True, True]
    negative_cases = (
        (
            (4, 64, 4096, 64),
            (4, 64, 4096, 64),
            {"arch": "gfx942"},
        ),
        (
            (4, 64, 4096, 64),
            (4, 64, 4096, 64),
            {"bases_aligned_16": False},
        ),
        ((4, 64, 4160, 64), (4, 64, 4096, 64), {}),
        ((4, 64, 4096, 64), (4, 32, 4096, 64), {}),
        ((4, 64, 4096, 64), (4, 16, 4096, 64), {}),
        ((4, 64, 4032, 64), (4, 64, 4032, 64), {}),
        ((2, 128, 960, 64), (2, 16, 960, 64), {}),
        ((2, 128, 1024, 64), (2, 16, 1088, 64), {}),
        ((4, 64, 4097, 64), (4, 64, 4097, 64), {}),
        ((1, 8, 1024, 64), (1, 1, 16384, 64), {}),
        ((1, 8, 12288, 64), (1, 1, 12288, 64), {}),
    )
    for q_shape, k_shape, overrides in negative_cases:
        assert not is_eligible(q_shape, k_shape, **overrides)

    m192_neighbor = select((2, 32, 16320, 64), (2, 32, 16320, 64))
    m192_rectangular = select((2, 32, 16384, 64), (2, 32, 16448, 64))
    assert m192_neighbor.owner_rows == 192
    assert m192_neighbor.dq_logical_n == 32
    assert m192_rectangular.owner_rows == 192
    assert m192_rectangular.dq_logical_n == 64


def test_d64_causal_dispatch_allocation_order(monkeypatch):
    q_shape = (1, 8, 256, 64)
    k_shape = (1, 1, 256, 64)
    q = torch.empty(q_shape, dtype=torch.bfloat16)
    k = torch.empty(k_shape, dtype=torch.bfloat16)
    v = torch.empty_like(k)
    o = torch.empty_like(q)
    do = torch.empty_like(q)
    lse = torch.empty(q_shape[:-1], dtype=torch.float32)
    original_empty = torch.empty
    original_empty_like = torch.empty_like
    calls = []
    output_allocations = 0
    stat_allocations = 0
    active_dispatch = None

    def make_meta_inputs(test_q_shape, test_k_shape):
        test_q = torch.empty(test_q_shape, device="meta", dtype=torch.bfloat16)
        test_k = torch.empty(test_k_shape, device="meta", dtype=torch.bfloat16)
        return (
            test_q,
            test_k,
            torch.empty_like(test_k),
            torch.empty_like(test_q),
            torch.empty_like(test_q),
            torch.empty(test_q_shape[:-1], device="meta", dtype=torch.float32),
        )

    selected_mha_q_shape = (4, 64, 4096, 64)
    selected_mha_k_shape = (4, 64, 4096, 64)
    selected_mha_inputs = make_meta_inputs(selected_mha_q_shape, selected_mha_k_shape)
    selected_mha = _select_d64_dispatch(
        selected_mha_q_shape,
        selected_mha_k_shape,
        True,
        arch="gfx950:sramecc+:xnack-",
        cu_count=256,
        sm_scale=0.125,
        bases_aligned_16=True,
    )
    selected_gqa_q_shape = (8, 16, 1024, 64)
    selected_gqa_k_shape = (8, 2, 1024, 64)
    selected_gqa_inputs = make_meta_inputs(selected_gqa_q_shape, selected_gqa_k_shape)
    selected_gqa = _select_d64_dispatch(
        selected_gqa_q_shape,
        selected_gqa_k_shape,
        True,
        arch="gfx950:sramecc+:xnack-",
        cu_count=256,
        sm_scale=0.125,
        bases_aligned_16=True,
    )
    retained = _select_d64_dispatch(q_shape, k_shape, True)

    def record_validate(*_args):
        calls.append("validate")

    def record_dispatch(*args):
        calls.append("dispatch")
        assert args[7] is True
        return active_dispatch

    def record_empty_like(tensor, *args, **kwargs):
        nonlocal output_allocations
        if tensor.ndim == 4:
            output_allocations += 1
            if output_allocations == 1:
                calls.append("outputs")
        return original_empty_like(tensor, *args, **kwargs)

    def record_empty(*args, **kwargs):
        nonlocal stat_allocations
        shape = tuple(args[0]) if args else tuple(kwargs["size"])
        if len(shape) == 3:
            stat_allocations += 1
            calls.append("delta" if stat_allocations == 1 else "lse_term")
        elif len(shape) == 5:
            calls.append("partials")
        return original_empty(*args, **kwargs)

    monkeypatch.setitem(vars(amd_fa_bwd), "_validate_inputs", record_validate)
    monkeypatch.setitem(vars(amd_fa_bwd), "_select_d64_dispatch_for_device", record_dispatch)
    monkeypatch.setattr(torch, "empty_like", record_empty_like)
    monkeypatch.setattr(torch, "empty", record_empty)
    monkeypatch.setitem(
        vars(amd_fa_bwd),
        "_launch_bwd_d64_causal_dq",
        lambda *_args: calls.append("dq"),
    )
    monkeypatch.setitem(
        vars(amd_fa_bwd),
        "_launch_bwd_d64_causal_mha_dkdv",
        lambda *_args: calls.append("mha_producer"),
    )
    monkeypatch.setitem(
        vars(amd_fa_bwd),
        "_launch_bwd_d64_causal_gqa8_dkdv",
        lambda *_args: calls.append("gqa_producer"),
    )
    monkeypatch.setitem(
        vars(amd_fa_bwd),
        "_launch_bwd_d64_causal_gqa8_reduce",
        lambda *_args: calls.append("reducer"),
    )
    monkeypatch.setitem(vars(amd_fa_bwd), "_run_bwd_preprocess", lambda *_args: None)
    monkeypatch.setitem(
        vars(amd_fa_bwd),
        "_run_bwd_d64_direct",
        lambda *_args: calls.append("retained_generic"),
    )

    def exercise(inputs, dispatch, scale):
        nonlocal active_dispatch, output_allocations, stat_allocations
        active_dispatch = dispatch
        output_allocations = 0
        stat_allocations = 0
        calls.clear()
        result = fa_backward(*inputs, scale, True)
        assert len(result) == 3
        return tuple(calls)

    assert exercise(selected_mha_inputs, selected_mha, 0.125) == (
        "validate",
        "dispatch",
        "outputs",
        "delta",
        "dq",
        "mha_producer",
    )
    assert exercise(selected_gqa_inputs, selected_gqa, 0.125) == (
        "validate",
        "dispatch",
        "outputs",
        "delta",
        "lse_term",
        "partials",
        "dq",
        "gqa_producer",
        "reducer",
    )
    assert exercise((q, k, v, o, do, lse), retained, -0.125) == (
        "validate",
        "dispatch",
        "outputs",
        "delta",
        "retained_generic",
    )

    input_sentinels = tuple(tensor.view(torch.uint8).clone() for tensor in (q, k, v, o, do, lse))
    for invalid in (0.0, -0.0, float("nan"), float("inf"), -float("inf")):
        calls.clear()
        with pytest.raises(ValueError, match="^D64 sm_scale must be finite and nonzero$"):
            fa_backward(q, k, v, o, do, lse, invalid, True)
        assert calls == ["validate"]
        for tensor, sentinel in zip((q, k, v, o, do, lse), input_sentinels, strict=True):
            assert torch.equal(tensor.view(torch.uint8), sentinel)


def test_d64_causal_scale_classification_before_writes(monkeypatch):
    shape = (4, 64, 4096, 64)
    q = torch.empty(shape, device="meta", dtype=torch.bfloat16)
    k = torch.empty(shape, device="meta", dtype=torch.bfloat16)
    v = torch.empty_like(k)
    o = torch.empty_like(q)
    do = torch.empty_like(q)
    lse = torch.empty(shape[:-1], device="meta", dtype=torch.float32)

    class Properties:
        gcnArchName = "gfx950:sramecc+:xnack-"
        multi_processor_count = 256

    calls = []

    def record_empty_like(*_args, **_kwargs):
        calls.append("empty_like")
        return object()

    def record_empty(*_args, **_kwargs):
        calls.append("empty")
        return object()

    def record_run(*args):
        dispatch = args[-1]
        calls.append(("run", getattr(dispatch, "family", None)))

    def record_gqa_workspace(*_args, **_kwargs):
        calls.append("gqa_workspace")
        return object()

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device: Properties())
    monkeypatch.setattr(torch, "empty_like", record_empty_like)
    monkeypatch.setattr(torch, "empty", record_empty)
    monkeypatch.setitem(vars(amd_fa_bwd), "_run_bwd_d64", record_run)
    monkeypatch.setitem(
        vars(amd_fa_bwd),
        "_allocate_bwd_d64_causal_gqa8_workspaces",
        record_gqa_workspace,
    )

    for invalid in (
            0.0,
            -0.0,
            float("nan"),
            float("inf"),
            -float("inf"),
            10**10000,
    ):
        calls.clear()
        with pytest.raises(ValueError, match="^D64 sm_scale must be finite and nonzero$"):
            fa_backward(q, k, v, o, do, lse, invalid, True)
        assert calls == []

    calls.clear()
    fa_backward(q, k, v, o, do, lse, -0.125, True)
    assert ("run", "causal_m192") in calls
    assert "gqa_workspace" not in calls

    calls.clear()
    fa_backward(q, k, v, o, do, lse, 0.125, True)
    assert ("run", "causal_scheduled_mha") in calls
    assert "gqa_workspace" not in calls


def test_d64_causal_gqa8_tiny_scale_dispatch_before_workspace(monkeypatch):
    q_shape = (8, 16, 1024, 64)
    k_shape = (8, 2, 1024, 64)
    select_kwargs = {
        "arch": "gfx950:sramecc+:xnack-",
        "cu_count": 256,
        "bases_aligned_16": True,
    }
    selected = _select_d64_dispatch(
        q_shape,
        k_shape,
        True,
        sm_scale=0.125,
        **select_kwargs,
    )
    tiny_scale = _select_d64_dispatch(
        q_shape,
        k_shape,
        True,
        sm_scale=1e-38,
        **select_kwargs,
    )
    assert selected.family == "causal_scheduled_gqa8"
    assert tiny_scale.family == "causal_m192"

    class Properties:
        gcnArchName = select_kwargs["arch"]
        multi_processor_count = select_kwargs["cu_count"]

    q = torch.empty(q_shape, device="meta", dtype=torch.bfloat16)
    k = torch.empty(k_shape, device="meta", dtype=torch.bfloat16)
    stats = torch.empty(q_shape[:-1], device="meta", dtype=torch.float32)
    retained_dispatches = []

    def reject_selected_workspace(*_args, **_kwargs):
        raise AssertionError("tiny scale must fall back before GQA8 workspace allocation")

    def record_retained(*args):
        retained_dispatches.append(args[-1])

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device: Properties())
    monkeypatch.setitem(
        vars(amd_fa_bwd),
        "_allocate_bwd_d64_causal_gqa8_workspaces",
        reject_selected_workspace,
    )
    monkeypatch.setitem(vars(amd_fa_bwd), "_run_bwd_preprocess", lambda *_args: None)
    monkeypatch.setitem(vars(amd_fa_bwd), "_run_bwd_d64_direct", record_retained)

    outputs = fa_backward(q, k, k, q, q, stats, 1e-38, True)
    assert [dispatch.family for dispatch in retained_dispatches] == ["causal_m192"]
    assert all(output.device.type == "meta" for output in outputs)


def test_d64_causal_mha_tiny_scale_uses_retained_dispatch():
    shape = (4, 64, 4096, 64)
    select_kwargs = {
        "arch": "gfx950:sramecc+:xnack-",
        "cu_count": 256,
        "bases_aligned_16": True,
    }
    selected = _select_d64_dispatch(
        shape,
        shape,
        True,
        sm_scale=0.125,
        **select_kwargs,
    )
    tiny_scale = _select_d64_dispatch(
        shape,
        shape,
        True,
        sm_scale=1e-38,
        **select_kwargs,
    )
    assert selected.family == "causal_scheduled_mha"
    assert tiny_scale.family == "causal_m192"


def test_d64_causal_scale_oversized_integer_validation():
    with pytest.raises(ValueError, match=r"^D64 sm_scale must be finite and nonzero$"):
        _validate_d64_sm_scale(10**10000)


def test_d64_causal_scale_oversized_integer_eligibility():
    assert not _is_d64_scheduled_causal_eligible(
        (4, 64, 4096, 64),
        (4, 64, 4096, 64),
        True,
        arch="gfx950",
        cu_count=256,
        sm_scale=10**10000,
        bases_aligned_16=True,
    )


@pytest.mark.parametrize(
    ("q_shape", "k_shape", "causal", "family", "owner_rows", "key_rows"),
    [
        ((2, 32, 16384, 64), (2, 32, 16384, 64), False, "noncausal_direct_n256", 32, 256),
        ((2, 32, 16384, 64), (2, 32, 16384, 64), True, "causal_m256", 256, 32),
        ((2, 32, 16384, 64), (2, 4, 16384, 64), True, "causal_m256", 256, 32),
        ((4, 48, 4096, 64), (4, 6, 4096, 64), True, "causal_m192", 192, 32),
        ((4, 48, 4096, 64), (4, 6, 16384, 64), True, "causal_m192", 192, 64),
    ],
)
def test_d64_structural_dispatch(q_shape, k_shape, causal, family, owner_rows, key_rows):
    dispatch = _select_d64_dispatch(q_shape, k_shape, causal)
    assert (dispatch.family, dispatch.owner_rows, dispatch.key_rows) == (family, owner_rows, key_rows)


@pytest.mark.parametrize(
    ("q_shape", "k_shape", "causal", "arch", "cu_count", "expected"),
    [
        pytest.param(
            (2, 32, 16384, 64),
            (2, 32, 16384, 64),
            False,
            "gfx950",
            256,
            True,
            id="mha-square-16k",
        ),
        pytest.param(
            (2, 32, 16384, 64),
            (2, 4, 16384, 64),
            False,
            "gfx950:sramecc+:xnack-",
            256,
            True,
            id="gqa8-square-16k",
        ),
        pytest.param(
            (1, 1, 4096, 64),
            (1, 1, 4096, 64),
            False,
            "gfx950",
            256,
            False,
            id="insufficient-owner-grid",
        ),
        pytest.param(
            (1, 1, 4096, 64),
            (1, 1, 4096, 64),
            False,
            "gfx950",
            16,
            True,
            id="sufficient-owner-grid",
        ),
        pytest.param(
            (2, 32, 16384, 64),
            (2, 4, 16384, 64),
            True,
            "gfx950",
            256,
            False,
            id="causal",
        ),
        pytest.param(
            (2, 32, 16384, 64),
            (2, 8, 16384, 64),
            False,
            "gfx950",
            256,
            False,
            id="group4",
        ),
        pytest.param(
            (2, 32, 16384, 64),
            (2, 32, 16384, 64),
            False,
            "gfx942",
            256,
            False,
            id="wrong-arch",
        ),
        pytest.param(
            (1, 1, 256, 64),
            (1, 1, 256, 64),
            False,
            "gfx950",
            1,
            False,
            id="short",
        ),
        pytest.param(
            (1, 1, 4095, 64),
            (1, 1, 4096, 64),
            False,
            "gfx950",
            1,
            False,
            id="misaligned-sq",
        ),
        pytest.param(
            (1, 1, 4096, 64),
            (1, 1, 4095, 64),
            False,
            "gfx950",
            1,
            False,
            id="misaligned-skv",
        ),
        pytest.param(
            (2, 32, 16384, 64),
            (2, 32, 16384, 64),
            False,
            None,
            None,
            False,
            id="missing-device-metadata",
        ),
    ],
)
def test_d64_fused_n256_eligibility(q_shape, k_shape, causal, arch, cu_count, expected):
    assert _is_d64_fused_n256_eligible(q_shape, k_shape, causal, arch=arch, cu_count=cu_count) is expected


@pytest.mark.parametrize(
    ("q_shape", "k_shape", "causal", "arch", "cu_count", "family"),
    [
        pytest.param(
            (2, 32, 16384, 64),
            (2, 32, 16384, 64),
            False,
            "gfx950",
            256,
            "noncausal_fused_n256",
            id="mha-square-16k",
        ),
        pytest.param(
            (2, 32, 16384, 64),
            (2, 4, 16384, 64),
            False,
            "gfx950",
            256,
            "noncausal_fused_n256",
            id="gqa8-square-16k",
        ),
        pytest.param(
            (1, 1, 4096, 64),
            (1, 1, 4096, 64),
            False,
            "gfx950",
            256,
            "noncausal_direct_n256",
            id="insufficient-owner-grid",
        ),
        pytest.param(
            (2, 32, 16384, 64),
            (2, 4, 16384, 64),
            True,
            "gfx950",
            256,
            "causal_m256",
            id="causal",
        ),
        pytest.param(
            (2, 32, 16384, 64),
            (2, 8, 16384, 64),
            False,
            "gfx950",
            256,
            "noncausal_direct_n256",
            id="group4",
        ),
        pytest.param(
            (2, 32, 16384, 64),
            (2, 32, 16384, 64),
            False,
            "gfx942",
            256,
            "noncausal_direct_n256",
            id="wrong-arch",
        ),
        pytest.param(
            (1, 1, 256, 64),
            (1, 1, 256, 64),
            False,
            "gfx950",
            1,
            "noncausal_direct_n256",
            id="short",
        ),
        pytest.param(
            (2, 32, 16384, 64),
            (2, 32, 16384, 64),
            False,
            None,
            None,
            "noncausal_direct_n256",
            id="missing-device-metadata",
        ),
    ],
)
def test_d64_fused_dispatch_is_structural(q_shape, k_shape, causal, arch, cu_count, family):
    dispatch = _select_d64_dispatch(q_shape, k_shape, causal, arch=arch, cu_count=cu_count)
    assert dispatch.family == family


def test_d64_fused_dispatch_without_device_metadata_uses_direct_fallback():
    dispatch = _select_d64_dispatch((2, 32, 16384, 64), (2, 32, 16384, 64), False)
    assert dispatch.family == "noncausal_direct_n256"


def test_d64_fused_n256_uses_direct_score_and_output_layouts():

    def normalized_assignments(source, names):
        assignments = {name: [] for name in names}
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in assignments:
                assignments[target.id].append(ast.unparse(node.value))
        return assignments

    assert normalized_assignments(_attn_bwd_d64_fused_n256_update.src, ("p_nd", "ds_nd")) == {
        "p_nd": ["tlx.require_layout(p.to(tl.bfloat16), p_op0_nd, pin=False)"],
        "ds_nd": ["tlx.require_layout(ds_bf16, ds_op0_nd, pin=False)"],
    }
    assert normalized_assignments(_attn_bwd_d64_fused_n256_kernel.src, ("dk_out", "dv_out")) == {
        "dk_out": ["tlx.require_layout((dk * dk_scale).to(tl.bfloat16), kv_async_layout, pin=False)"],
        "dv_out": ["tlx.require_layout(dv.to(tl.bfloat16), kv_async_layout, pin=False)"],
    }


def test_d64_fused_workspace_contract():
    q = torch.empty((2, 32, 4096, 64), device="meta", dtype=torch.bfloat16)
    k_mha = torch.empty((2, 32, 4096, 64), device="meta", dtype=torch.bfloat16)
    k_gqa = torch.empty((2, 4, 4096, 64), device="meta", dtype=torch.bfloat16)
    mha = _D64Dispatch("noncausal_fused_n256", 32, 256, 1)
    gqa = _D64Dispatch("noncausal_fused_n256", 32, 256, 8)

    mha_acc, mha_dk, mha_dv = _allocate_bwd_d64_fused_workspaces(q, k_mha, mha)
    gqa_acc, gqa_dk, gqa_dv = _allocate_bwd_d64_fused_workspaces(q, k_gqa, gqa)

    assert mha_acc.shape == q.shape and mha_acc.dtype is torch.float32
    assert mha_acc.is_contiguous()
    assert mha_dk is mha_dv is None
    assert gqa_acc.shape == q.shape and gqa_acc.dtype is torch.float32
    assert gqa_acc.is_contiguous()
    assert gqa_dk.shape == gqa_dv.shape == (2, 4, 8, 4096, 64)
    assert gqa_dk.dtype is gqa_dv.dtype is torch.bfloat16
    assert gqa_dk.is_contiguous() and gqa_dv.is_contiguous()


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_d64_fused_preprocess_computes_delta_and_zeros_fp32_dq_gfx950():
    generator = torch.Generator(device="cuda")
    generator.manual_seed(47)
    o = torch.randn((1, 2, 256, 64), generator=generator, device="cuda", dtype=torch.bfloat16)
    do = torch.randn(o.shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    delta = torch.empty(o.shape[:-1], device="cuda", dtype=torch.float32)
    dq_acc = torch.full(o.shape, 7.0, device="cuda", dtype=torch.float32)
    _attn_bwd_preprocess_kernel.device_caches.clear()

    _run_bwd_preprocess(o, do, delta, dq_acc=dq_acc)

    expected = torch.sum(o.float() * do.float(), dim=-1)
    torch.testing.assert_close(delta, expected, rtol=1e-5, atol=1e-5)
    assert torch.count_nonzero(dq_acc).item() == 0

    device = torch.cuda.current_device()
    compiled = tuple(_attn_bwd_preprocess_kernel.device_caches[device][0].values())
    assert len(compiled) == 1
    obj = compiled[0]
    amdgcn = obj.asm["amdgcn"]
    private_segment = {
        int(value)
        for value in re.findall(r"(?:\.amdhsa_)?private_segment_fixed_size:?\s+(\d+)", amdgcn)
    }
    resources = {
        "n_spills": obj.n_spills,
        "global_scratch_bytes": obj.metadata.global_scratch_size,
        "private_segment_bytes": (private_segment.pop() if len(private_segment) == 1 else None),
        "scratch_load_instructions": len(re.findall(r"\bscratch_load", amdgcn)),
        "scratch_store_instructions": len(re.findall(r"\bscratch_store", amdgcn)),
    }
    assert resources == {
        "n_spills": 0,
        "global_scratch_bytes": 0,
        "private_segment_bytes": 0,
        "scratch_load_instructions": 0,
        "scratch_store_instructions": 0,
    }


_D64_ZERO_RESOURCE_FIELDS = (
    "n_spills",
    "global_scratch_bytes",
    "private_segment_bytes",
    "scratch_load_instructions",
    "scratch_store_instructions",
)


def _d64_code_object_resource(obj):
    amdgcn = obj.asm["amdgcn"]
    private_segments = {
        int(value)
        for value in re.findall(r"(?:\.amdhsa_)?private_segment_fixed_size:?\s+(\d+)", amdgcn)
    }
    vector_vgpr_counts = {int(value) for value in re.findall(r";\s+NumVgprs:\s+(\d+)", amdgcn)}
    agpr_counts = {int(value) for value in re.findall(r";\s+NumAgprs:\s+(\d+)", amdgcn)}
    return {
        "vgpr_count": obj.n_regs,
        "vector_vgpr_count": (vector_vgpr_counts.pop() if len(vector_vgpr_counts) == 1 else None),
        "agpr_count": agpr_counts.pop() if len(agpr_counts) == 1 else None,
        "unified_vgpr_count": obj.n_regs,
        "lds_bytes": obj.metadata.shared,
        "n_spills": obj.n_spills,
        "global_scratch_bytes": obj.metadata.global_scratch_size,
        "private_segment_bytes": (private_segments.pop() if len(private_segments) == 1 else None),
        "scratch_load_instructions": len(re.findall(r"\bscratch_load", amdgcn)),
        "scratch_store_instructions": len(re.findall(r"\bscratch_store", amdgcn)),
    }


def test_d64_structural_dispatch_codegen_resource_contract():

    class Metadata:
        shared = 41472
        global_scratch_size = 0

    class CompiledObject:
        n_regs = 268
        n_spills = 0
        metadata = Metadata()
        asm = {"amdgcn": "\n".join((
            "; NumVgprs: 254",
            "; NumAgprs: 12",
            ".private_segment_fixed_size: 0",
        ))}

    resource = _d64_code_object_resource(CompiledObject())
    assert resource["vgpr_count"] == 268
    assert resource["vector_vgpr_count"] == 254
    assert resource["agpr_count"] == 12
    assert resource["unified_vgpr_count"] == 268
    assert resource["unified_vgpr_count"] >= (resource["vector_vgpr_count"] + resource["agpr_count"])


def _assert_d64_code_object_scratch_free(name, obj):
    resource = _d64_code_object_resource(obj)
    for field in _D64_ZERO_RESOURCE_FIELDS:
        assert resource[field] == 0, (name, resource)
    return resource


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize(
    "shape, expected_logical_n, expected_use_xcd, seed",
    (
        pytest.param((1, 8, 1, 256, 256, 64), 32, False, 107, id="square_flat"),
        pytest.param((1, 8, 8, 256, 320, 64), 64, True, 108, id="rectangular_xcd"),
    ),
)
def test_d64_causal_common_dq_stat_modes_gfx950(shape, expected_logical_n, expected_use_xcd, seed):
    batch, hq, hkv, sq, skv, _head_dim = shape
    case = _make_d64_gqa_smoke_case(shape, causal=True, seed=seed)
    launches = _d64_dq_launch_plan(batch, hq, hkv, sq, skv, 192, 1, True)

    def dispatch(family, stat_mode):
        return _D64Dispatch(
            family,
            owner_rows=192,
            key_rows=64,
            kv_splits=1,
            selected_causal=True,
            stat_mode=stat_mode,
            dq_logical_n=_d64_selected_causal_logical_n(sq, skv, hq // hkv),
            dq_use_xcd=_d64_use_dq_xcd(batch, hkv, sq, skv, 192),
            dq_launches=launches,
        )

    mha_dispatch = dispatch("causal_scheduled_mha", _D64_MHA_POSITIVE)
    gqa_dispatch = dispatch("causal_scheduled_gqa8", _D64_GQA_SIGNED)
    for mode_dispatch in (mha_dispatch, gqa_dispatch):
        assert mode_dispatch.dq_logical_n == expected_logical_n
        assert mode_dispatch.dq_use_xcd is expected_use_xcd

    delta_mha = torch.empty_like(case.lse)
    delta_gqa = torch.empty_like(case.lse)
    lse_term_gqa = torch.empty_like(case.lse)
    dq_mha = torch.empty_like(case.q)
    dq_gqa = torch.empty_like(case.q)

    _launch_bwd_d64_causal_dq(
        case.q,
        case.k,
        case.v,
        case.o,
        case.do,
        case.lse,
        delta_mha,
        None,
        dq_mha,
        case.sm_scale,
        mha_dispatch,
    )
    _launch_bwd_d64_causal_dq(
        case.q,
        case.k,
        case.v,
        case.o,
        case.do,
        case.lse,
        delta_gqa,
        lse_term_gqa,
        dq_gqa,
        case.sm_scale,
        gqa_dispatch,
    )
    torch.cuda.synchronize()

    positive = torch.sum(case.o.float() * case.do.float(), dim=-1)
    torch.testing.assert_close(delta_mha, positive, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(delta_gqa, -positive, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        lse_term_gqa,
        -case.lse.float() * math.log2(math.e),
        rtol=1e-5,
        atol=1e-5,
    )
    for name, actual in (("mha", dq_mha), ("gqa", dq_gqa)):
        assert torch.isfinite(actual).all(), name
        relative_l2 = torch.linalg.vector_norm(actual.float() - case.grads[0]) / torch.linalg.vector_norm(case.grads[0])
        assert relative_l2.item() < 5e-3, (name, relative_l2.item())


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_d64_causal_common_dq_peeled_m128_accuracy_gfx950():
    shape = (1, 8, 8, 8192, 8192, 64)
    case = _make_d64_aten_case(shape, seed=137, causal=True)
    batch, hq, hkv, sq, skv, _head_dim = shape
    owner_rows = 192
    owners = triton.cdiv(sq, owner_rows)
    launches = _d64_dq_launch_plan(
        batch,
        hq,
        hkv,
        sq,
        skv,
        owner_rows,
        cu_count=8,
        host_skip_owner_tail=True,
    )
    assert launches == (
        _D64DQLaunch(owners - 1, False, 0, owners - 1, 3, 0),
        _D64DQLaunch(1, False, owners - 1, 1, 2, 192),
    )
    assert _d64_use_dq_xcd(batch, hkv, sq, skv, owner_rows)
    peeled_interval = _d64_causal_owner_interval(owners - 1, sq, owner_rows)
    full_interval = _d64_causal_owner_interval(owners - 2, sq, owner_rows)
    assert peeled_interval == (0, 128)
    assert full_interval == (128, 320)

    dispatch = _D64Dispatch(
        family="causal_scheduled_mha",
        owner_rows=owner_rows,
        key_rows=64,
        kv_splits=1,
        selected_causal=True,
        stat_mode=_D64_MHA_POSITIVE,
        dq_logical_n=32,
        dq_use_xcd=True,
        dq_launches=launches,
    )
    delta = torch.empty_like(case.lse)
    dq = torch.empty_like(case.q)
    _attn_bwd_dq_d64_causal_mha_kernel.device_caches.clear()
    _launch_bwd_d64_causal_dq(
        case.q,
        case.k,
        case.v,
        case.o,
        case.do,
        case.lse,
        delta,
        None,
        dq,
        case.sm_scale,
        dispatch,
    )
    torch.cuda.synchronize()

    positive = torch.sum(case.o.float() * case.do.float(), dim=-1)
    torch.testing.assert_close(delta, positive, rtol=1e-5, atol=1e-5)
    assert torch.isfinite(dq).all()
    expected_dq = case.grads[0].float()
    for region_name, row_interval in (
        ("peeled_m128", peeled_interval),
        ("full_m192", full_interval),
        ("all_dq", (0, sq)),
    ):
        row_begin, row_end = row_interval
        actual_region = dq[:, :, row_begin:row_end].float()
        expected_region = expected_dq[:, :, row_begin:row_end]
        relative_l2 = torch.linalg.vector_norm(actual_region -
                                               expected_region) / torch.linalg.vector_norm(expected_region)
        assert relative_l2.item() < 5e-3, (
            region_name,
            relative_l2.item(),
        )
    device = torch.cuda.current_device()
    objects = tuple(_attn_bwd_dq_d64_causal_mha_kernel.device_caches[device][0].values())
    assert len(objects) == 2


def test_d64_causal_common_dq_direct_load_ast_contract():
    helper = ast.parse(_attn_bwd_dq_d64_causal_load_q64.src).body[0]

    def dotted_name(call):
        value = call.func
        parts = []
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            parts.append(value.id)
        return ".".join(reversed(parts))

    direct_loads = [
        statement.targets[0].id for statement in helper.body if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name)
        and isinstance(statement.value, ast.Call) and dotted_name(statement.value) == "tlx.buffer_load"
    ]
    assert direct_loads == ["do", "o", "q"]

    stat_mode_branch = next(
        statement for statement in helper.body
        if isinstance(statement, ast.If) and ast.unparse(statement.test) == "STAT_MODE == _D64_MHA_POSITIVE_JIT")
    producer_lse_term = next(
        statement.value
        for statement in ast.walk(stat_mode_branch)
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id == "producer_lse_term")
    assert ast.unparse(producer_lse_term) == "-lse * 1.4426950408889634"
    lse_term_store = next(node for node in ast.walk(stat_mode_branch)
                          if isinstance(node, ast.Call) and dotted_name(node) == "tl.store"
                          and ast.unparse(node.args[0]) == "LSE_TERM + stats_base + rows")
    assert ast.unparse(lse_term_store.args[1]) == "producer_lse_term"

    q_scale = next(statement.value
                   for statement in ast.walk(stat_mode_branch)
                   if isinstance(statement, ast.Assign) and len(statement.targets) == 1
                   and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id == "q_scale")
    q_scale_fill = next(node for node in ast.walk(q_scale)
                        if isinstance(node, ast.Call) and dotted_name(node) == "tl.full")
    assert ast.unparse(q_scale_fill.args[1]) == "SM_SCALE * 1.4426950408889634"
    q_scale_products = [
        node for node in ast.walk(stat_mode_branch)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult) and ast.unparse(node.right) == "q_scale"
    ]
    assert len(q_scale_products) == 1
    assert ast.unparse(q_scale_products[0].left) == "q.to(tl.float32)"

    handoff = next(statement.value
                   for statement in helper.body
                   if isinstance(statement, ast.Assign) and len(statement.targets) == 1
                   and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id == "product"
                   and isinstance(statement.value, ast.Call) and dotted_name(statement.value) == "tlx.release_layout")
    assert ast.unparse(handoff.args[0]) == "product"

    reduction = next(statement.value
                     for statement in helper.body
                     if isinstance(statement, ast.Assign) and len(statement.targets) == 1
                     and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id == "positive")
    assert dotted_name(reduction) == "tl.sum"
    assert ast.unparse(reduction.args[0]) == "product"
    assert ast.literal_eval(reduction.keywords[0].value) == 1

    step = ast.parse(_attn_bwd_dq_d64_causal_step.src).body[0]
    n64_handoff = next(statement for statement in step.body
                       if isinstance(statement, ast.If) and ast.unparse(statement.test) == "BLOCK_N == 64")
    score_tie = next(
        statement.value
        for statement in n64_handoff.body
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id == "scores"
        and isinstance(statement.value, ast.Call) and dotted_name(statement.value) == "tlx.amd_register_handoff")
    assert ast.unparse(score_tie.args[0]) == "scores"
    assert ast.literal_eval(
        next(keyword.value for keyword in score_tie.keywords if keyword.arg == "registers_per_group")) == 2
    n32_handoff = next(statement for statement in step.body
                       if isinstance(statement, ast.If) and ast.unparse(statement.test) == "BLOCK_N == 32")
    assert not any(
        isinstance(node, ast.Call) and dotted_name(node) == "tlx.amd_register_handoff"
        for node in ast.walk(n32_handoff))
    assert any(
        isinstance(node, ast.Call) and dotted_name(node) == "tlx.require_layout"
        and ast.unparse(node.args[0]) == "scores" for node in ast.walk(n32_handoff))
    ds_handoff = next(
        statement.value for statement in step.body if isinstance(statement, ast.Assign) and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id == "ds"
        and isinstance(statement.value, ast.Call) and dotted_name(statement.value) == "tlx.amd_register_handoff")
    assert ast.unparse(ds_handoff.args[0]) == "p * dp"
    assert ast.literal_eval(
        next(keyword.value for keyword in ds_handoff.keywords if keyword.arg == "registers_per_group")) == 2
    assert not any(isinstance(node, ast.Call) and dotted_name(node) == "tlx.local_load" for node in ast.walk(step))
    score_scale_guard = next(statement for statement in step.body
                             if isinstance(statement, ast.If) and ast.unparse(statement.test) == "not SCORE_PRE_SCALED")
    assert any(
        isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult) and ast.unparse(node.left) == "scores"
        for node in ast.walk(score_scale_guard))

    score32 = ast.parse(_attn_bwd_dq_d64_causal_score32.src).body[0]
    score32_scale_guard = next(
        statement for statement in score32.body
        if isinstance(statement, ast.If) and ast.unparse(statement.test) == "not SCORE_PRE_SCALED")
    assert any(
        isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Mult) and ast.unparse(node.target) == "scores"
        for node in ast.walk(score32_scale_guard))
    split_score_handoffs = [
        node for node in ast.walk(score32)
        if isinstance(node, ast.Call) and dotted_name(node) == "tlx.amd_register_handoff"
    ]
    assert not split_score_handoffs

    finish32 = ast.parse(_attn_bwd_dq_d64_causal_finish32.src).body[0]
    finish32_handoff = next(
        statement.value
        for statement in finish32.body
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id == "ds"
        and isinstance(statement.value, ast.Call) and dotted_name(statement.value) == "tlx.amd_register_handoff")
    assert ast.unparse(finish32_handoff.args[0]) == "p * dp"
    assert ast.literal_eval(
        next(keyword.value for keyword in finish32_handoff.keywords if keyword.arg == "register_class")) == "vgpr"
    assert ast.literal_eval(
        next(keyword.value for keyword in finish32_handoff.keywords if keyword.arg == "registers_per_group")) == 2

    def positional_argument(call, function, name):
        parameter_names = [argument.arg for argument in function.args.args]
        return ast.unparse(call.args[parameter_names.index(name)])

    nslice = ast.parse(_attn_bwd_dq_d64_causal_nslice.src).body[0]
    shared_operand_assignments = [
        statement for statement in ast.walk(nslice)
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(
            statement.targets[0], ast.Name) and statement.targets[0].id in {"k_source", "kv_source", "v_source"}
    ]
    assert [statement.targets[0].id for statement in shared_operand_assignments] == [
        "k_source",
        "kv_source",
        "v_source",
    ]
    assert all(
        isinstance(statement.value, ast.Call) and dotted_name(statement.value) == "tlx.local_load"
        for statement in shared_operand_assignments)
    fragment_steps = [
        node for node in ast.walk(nslice)
        if isinstance(node, ast.Call) and dotted_name(node) == "_attn_bwd_dq_d64_causal_step"
    ]
    assert len(fragment_steps) == 11
    assert all([ast.unparse(arg)
                for arg in call.args[6:9]] == ["k_source", "v_source", "kv_source"]
               for call in fragment_steps)
    assert all(positional_argument(call, step, "SCORE_PRE_SCALED") == "SCORE_PRE_SCALED" for call in fragment_steps)

    m256 = ast.parse(_attn_bwd_dq_d64_causal_m256_unmasked_n32.src).body[0]
    ordered_assignments = [
        statement.targets[0].id for statement in m256.body if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id in {
            "kt",
            "k_nd",
            "scores0",
            "vt",
            "dp0",
            "dq0",
            "dq1",
            "dq3",
            "scores2",
            "dp2",
            "dq2",
        }
    ]
    assert ordered_assignments == [
        "kt",
        "k_nd",
        "scores0",
        "vt",
        "dp0",
        "dq0",
        "dq1",
        "dq3",
        "scores2",
        "dp2",
        "dq2",
    ]
    m256_fragment_steps = [
        statement.value for statement in m256.body if isinstance(statement, ast.Assign)
        and isinstance(statement.value, ast.Call) and dotted_name(statement.value) == "_attn_bwd_dq_d64_causal_step"
    ]
    assert [ast.unparse(call.args[0]) for call in m256_fragment_steps] == ["dq1", "dq3"]
    assert all(ast.unparse(call.args[15]) == "32" for call in m256_fragment_steps)
    assert all(ast.unparse(call.args[16]) == "False" for call in m256_fragment_steps)
    assert all(
        positional_argument(call, step, "SCORE_PRE_SCALED") == "SCORE_PRE_SCALED" for call in m256_fragment_steps)
    score32_calls = [
        statement.value for statement in m256.body if isinstance(statement, ast.Assign) and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id in {"scores0", "scores2"}
        and isinstance(statement.value, ast.Call) and dotted_name(statement.value) == "_attn_bwd_dq_d64_causal_score32"
    ]
    assert "PACK_TIE" not in {argument.arg for argument in score32.args.args}
    assert all(positional_argument(call, score32, "SCORE_PRE_SCALED") == "SCORE_PRE_SCALED" for call in score32_calls)

    impl = ast.parse(_attn_bwd_dq_d64_causal_impl.src).body[0]
    q3_residency = next(node for node in ast.walk(impl) if isinstance(node, ast.Assign) and len(node.targets) == 1
                        and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "q3"
                        and isinstance(node.value, ast.Call) and dotted_name(node.value) == "tlx.amd_register_resident")
    assert ast.unparse(q3_residency.value.args[0]) == "q3"
    assert ast.literal_eval(
        next(keyword.value for keyword in q3_residency.value.keywords if keyword.arg == "register_class")) == "agpr"
    assert ast.literal_eval(
        next(keyword.value for keyword in q3_residency.value.keywords if keyword.arg == "registers_per_group")) == 8
    assert not any(isinstance(node, ast.Name) and node.id == "q3_buffer" for node in ast.walk(impl))
    score_pre_scaled = next(statement for statement in impl.body if isinstance(statement, ast.AnnAssign)
                            and isinstance(statement.target, ast.Name) and statement.target.id == "score_pre_scaled")
    assert ast.unparse(score_pre_scaled.annotation) == "tl.constexpr"
    assert ast.unparse(score_pre_scaled.value) == "STAT_MODE == _D64_GQA_SIGNED_JIT"
    score_consumers = {
        "_attn_bwd_dq_d64_causal_m256_unmasked_n32": m256,
        "_attn_bwd_dq_d64_causal_nslice": nslice,
    }
    consumer_calls = [
        node for node in ast.walk(impl) if isinstance(node, ast.Call) and dotted_name(node) in score_consumers
    ]
    assert len(consumer_calls) == 8
    assert all(
        positional_argument(
            call,
            score_consumers[dotted_name(call)],
            "SCORE_PRE_SCALED",
        ) == "score_pre_scaled" for call in consumer_calls)

    store = ast.parse(_attn_bwd_dq_d64_causal_store_q64.src).body[0]
    offsets = next(statement.value
                   for statement in store.body
                   if isinstance(statement, ast.Assign) and len(statement.targets) == 1
                   and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id == "offsets")
    assert ast.unparse(offsets) == "(local_rows[:, None] * D + cols[None, :]).to(tl.int32)"
    store_call = next(node for node in ast.walk(store)
                      if isinstance(node, ast.Call) and dotted_name(node) == "tlx.buffer_store")
    assert ast.unparse(store_call.args[1]) == "DQ + q_base + row_start * D"


def test_d64_causal_common_dq_launch_order(monkeypatch):

    class LaunchRecorder:

        def __init__(self):
            self.calls = []

        def __getitem__(self, grid):

            def record(*args, **kwargs):
                self.calls.append((grid, args, kwargs))

            return record

    recorder = LaunchRecorder()
    monkeypatch.setitem(vars(amd_fa_bwd), "_attn_bwd_dq_d64_causal_mha_kernel", recorder)

    def tensors(batch, hq, hkv, sq, skv):
        q = torch.empty((batch, hq, sq, 64), device="meta", dtype=torch.bfloat16)
        k = torch.empty((batch, hkv, skv, 64), device="meta", dtype=torch.bfloat16)
        stats = torch.empty((batch, hq, sq), device="meta", dtype=torch.float32)
        return q, k, stats

    flat_q, flat_k, flat_stats = tensors(1, 8, 1, 256, 256)
    flat_launches = _d64_dq_launch_plan(1, 8, 1, 256, 256, 192, 1, True)
    flat = _D64Dispatch(
        "causal_scheduled_mha",
        192,
        64,
        1,
        True,
        _D64_MHA_POSITIVE,
        32,
        False,
        flat_launches,
    )
    _launch_bwd_d64_causal_dq(
        flat_q,
        flat_k,
        flat_k,
        flat_q,
        flat_q,
        flat_stats,
        flat_stats,
        None,
        flat_q,
        0.125,
        flat,
    )

    batch, hq, hkv, sq = 4, 64, 8, 8192
    xcd_q, xcd_k, xcd_stats = tensors(batch, hq, hkv, sq, sq)
    xcd_launches = _d64_dq_launch_plan(batch, hq, hkv, sq, sq, 192, 256, True)
    xcd = _D64Dispatch(
        "causal_scheduled_mha",
        192,
        64,
        1,
        True,
        _D64_MHA_POSITIVE,
        32,
        True,
        xcd_launches,
    )
    _launch_bwd_d64_causal_dq(
        xcd_q,
        xcd_k,
        xcd_k,
        xcd_q,
        xcd_q,
        xcd_stats,
        xcd_stats,
        None,
        xcd_q,
        0.125,
        xcd,
    )

    assert len(recorder.calls) == 1 + len(xcd_launches)
    flat_grid, _flat_args, flat_kwargs = recorder.calls[0]
    assert flat_grid == (1 * 8 * flat_launches[0].launch_tiles, )
    assert not flat_kwargs["USE_DQ_XCD"]
    assert flat_kwargs["KV_PIPELINE_STAGES"] == _D64_DQ_KV_STAGES

    owners = triton.cdiv(sq, 192)
    assert xcd_launches == (
        _D64DQLaunch(owners - 1, False, 0, owners - 1, 3, 0),
        _D64DQLaunch(1, False, owners - 1, 1, 2, 192),
    )
    for call, launch in zip(recorder.calls[1:], xcd_launches):
        grid, args, kwargs = call
        assert grid == (batch * hq * launch.launch_tiles, )
        assert len(args) == 8
        assert kwargs["USE_DQ_XCD"]
        assert kwargs["OWNER_PID_BASE"] == launch.owner_pid_base
        assert kwargs["LAUNCH_Q_TILES"] == launch.launch_q_tiles
        assert kwargs["OWNER_FRAGMENTS"] == launch.owner_fragments
        assert kwargs["GRID_OWNER_M"] == launch.grid_owner_m
        assert kwargs["KV_PIPELINE_STAGES"] == _D64_DQ_KV_STAGES

    decoded = []
    for launch_index, launch in enumerate(xcd_launches):
        for pid in range(batch * hq * launch.launch_tiles):
            coords = _d64_decode_dq_pid(
                pid,
                batch,
                hq,
                hkv,
                launch.launch_tiles,
                True,
                launch.owner_pid_base,
            )
            decoded.append((launch_index, coords))
    assert all(item[1][2] < owners - 1 for item in decoded if item[0] == 0)
    assert all(item[1][2] == owners - 1 for item in decoded if item[0] == 1)
    assert _d64_causal_owner_interval(0, sq, 192)[0] > _d64_causal_owner_interval(owners - 1, sq, 192)[0]
    assert _d64_causal_owner_interval(owners - 1, sq, 192) == (0, 128)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_d64_causal_common_dq_codegen_gfx950():

    def compile_variant(name, shape, owner_rows, launches, stat_mode, expected_lds, expected_private_segment):
        batch, hq, hkv, sq, skv, _head_dim = shape
        q = torch.zeros((batch, hq, sq, 64), device="cuda", dtype=torch.bfloat16)
        k = torch.zeros((batch, hkv, skv, 64), device="cuda", dtype=torch.bfloat16)
        stats = torch.zeros((batch, hq, sq), device="cuda", dtype=torch.float32)
        dq = torch.empty_like(q)
        signed = stat_mode == _D64_GQA_SIGNED
        kernel = (_attn_bwd_dq_d64_causal_gqa8_kernel if signed else _attn_bwd_dq_d64_causal_mha_kernel)
        kernel.device_caches.clear()
        dispatch = _D64Dispatch(
            family="causal_scheduled_gqa8" if signed else "causal_scheduled_mha",
            owner_rows=owner_rows,
            key_rows=64,
            kv_splits=1,
            selected_causal=True,
            stat_mode=stat_mode,
            dq_logical_n=_d64_selected_causal_logical_n(sq, skv, hq // hkv),
            dq_use_xcd=_d64_use_dq_xcd(batch, hkv, sq, skv, owner_rows),
            dq_launches=launches,
        )
        lse_term = torch.empty_like(stats) if signed else None
        _launch_bwd_d64_causal_dq(q, k, k, q, q, stats, stats, lse_term, dq, 0.125, dispatch)
        torch.cuda.synchronize()

        device = torch.cuda.current_device()
        objects = tuple(kernel.device_caches[device][0].values())
        assert len(objects) == 1, (name, len(objects))
        obj = objects[0]
        resource = _d64_code_object_resource(obj)
        if expected_private_segment == 0:
            for field in _D64_ZERO_RESOURCE_FIELDS:
                assert resource[field] == 0, (name, resource)
        else:
            expected_spill_resources = {
                264: {
                    "n_spills": 66,
                    "global_scratch_bytes": 0,
                    "private_segment_bytes": 264,
                    "scratch_load_instructions": 81,
                    "scratch_store_instructions": 54,
                },
                292: {
                    "n_spills": 73,
                    "global_scratch_bytes": 0,
                    "private_segment_bytes": 292,
                    "scratch_load_instructions": 73,
                    "scratch_store_instructions": 48,
                },
            }
            assert {
                "n_spills": resource["n_spills"],
                "global_scratch_bytes": resource["global_scratch_bytes"],
                "private_segment_bytes": resource["private_segment_bytes"],
                "scratch_load_instructions": resource["scratch_load_instructions"],
                "scratch_store_instructions": resource["scratch_store_instructions"],
            } == expected_spill_resources[expected_private_segment], (name, resource)
        assert resource["unified_vgpr_count"] == resource["vgpr_count"]
        assert resource["unified_vgpr_count"] >= (resource["vector_vgpr_count"] + resource["agpr_count"])
        assert resource["lds_bytes"] == expected_lds, (name, resource)
        amdgcn = obj.asm["amdgcn"]
        assert re.search(r"\bbuffer_store_dwordx4\b", amdgcn), name
        assert not re.search(r"\b\w*atomic\w*\b", amdgcn), name
        return resource

    square_sq = 8192
    square_owners = triton.cdiv(square_sq, 192)
    rectangular_shape = (1, 8, 8, 256, 320, 64)
    rectangular_launches = _d64_dq_launch_plan(1, 8, 8, 256, 320, 192, 1, True)
    deep_m192_gqa8_shape = (8, 8, 1, 1024, 2048, 64)
    deep_m192_gqa8_launches = _d64_dq_launch_plan(8, 8, 1, 1024, 2048, 192, 256, True)
    short_square_m256_gqa8_shape = (4, 48, 6, 4096, 4096, 64)
    short_square_m256_gqa8_launches = _d64_dq_launch_plan(4, 48, 6, 4096, 4096, 256, 256, True)
    square_m256_gqa8_shape = (1, 8, 1, 16384, 16384, 64)
    square_m256_gqa8_launches = _d64_dq_launch_plan(1, 8, 1, 16384, 16384, 256, 256, True)
    deep_m256_gqa8_shape = (8, 8, 1, 4096, 8192, 64)
    deep_m256_gqa8_launches = _d64_dq_launch_plan(8, 8, 1, 4096, 8192, 256, 256, True)
    long_m256_gqa8_shape = (8, 8, 1, 4096, 12288, 64)
    long_m256_gqa8_launches = _d64_dq_launch_plan(8, 8, 1, 4096, 12288, 256, 256, True)
    variants = (
        (
            "peeled_m128_square",
            (1, 8, 8, square_sq, square_sq, 64),
            192,
            (_D64DQLaunch(1, False, square_owners - 1, 1, 2, 192), ),
            _D64_MHA_POSITIVE,
            33536,
            0,
        ),
        (
            "m192_square",
            (1, 8, 8, square_sq, square_sq, 64),
            192,
            (_D64DQLaunch(
                square_owners - 1,
                False,
                0,
                square_owners - 1,
                3,
                0,
            ), ),
            _D64_MHA_POSITIVE,
            33536,
            0,
        ),
        (
            "m192_rectangular",
            rectangular_shape,
            192,
            rectangular_launches,
            _D64_MHA_POSITIVE,
            33536,
            0,
        ),
        (
            "m256_square",
            (1, 1, 1, 16384, 16384, 64),
            256,
            (_D64DQLaunch(64, True, 0, 0, 4, 0), ),
            _D64_MHA_POSITIVE,
            33536,
            0,
        ),
        (
            "gqa_signed_rectangular",
            rectangular_shape,
            192,
            rectangular_launches,
            _D64_GQA_SIGNED,
            33536,
            0,
        ),
        (
            "gqa_signed_deep_m192_n32",
            deep_m192_gqa8_shape,
            192,
            deep_m192_gqa8_launches,
            _D64_GQA_SIGNED,
            33536,
            0,
        ),
        (
            "gqa_signed_short_square_m256_n32",
            short_square_m256_gqa8_shape,
            256,
            short_square_m256_gqa8_launches,
            _D64_GQA_SIGNED,
            33536,
            0,
        ),
        (
            "gqa_signed_square_m256_n32",
            square_m256_gqa8_shape,
            256,
            square_m256_gqa8_launches,
            _D64_GQA_SIGNED,
            33536,
            0,
        ),
        (
            "gqa_signed_deep_m256_n32",
            deep_m256_gqa8_shape,
            256,
            deep_m256_gqa8_launches,
            _D64_GQA_SIGNED,
            33536,
            0,
        ),
        (
            "gqa_signed_long_m256_n32",
            long_m256_gqa8_shape,
            256,
            long_m256_gqa8_launches,
            _D64_GQA_SIGNED,
            33536,
            0,
        ),
    )
    resources = {
        name: compile_variant(
            name,
            shape,
            owner_rows,
            launches,
            stat_mode,
            expected_lds,
            expected_private_segment,
        )
        for (
            name,
            shape,
            owner_rows,
            launches,
            stat_mode,
            expected_lds,
            expected_private_segment,
        ) in variants
    }
    assert tuple(resources) == tuple(variant[0] for variant in variants)
    assert resources["gqa_signed_deep_m192_n32"]["unified_vgpr_count"] <= 224
    assert resources["gqa_signed_short_square_m256_n32"]["vector_vgpr_count"] == 245
    assert resources["gqa_signed_short_square_m256_n32"]["agpr_count"] == 8
    assert resources["gqa_signed_short_square_m256_n32"]["unified_vgpr_count"] == 256
    assert resources["gqa_signed_square_m256_n32"]["vector_vgpr_count"] == 245
    assert resources["gqa_signed_square_m256_n32"]["agpr_count"] == 8
    assert resources["gqa_signed_square_m256_n32"]["unified_vgpr_count"] == 256
    assert resources["gqa_signed_deep_m256_n32"]["vector_vgpr_count"] == 244
    assert resources["gqa_signed_deep_m256_n32"]["agpr_count"] == 8
    assert resources["gqa_signed_deep_m256_n32"]["unified_vgpr_count"] == 252
    assert resources["gqa_signed_long_m256_n32"]["vector_vgpr_count"] == 244
    assert resources["gqa_signed_long_m256_n32"]["agpr_count"] == 8
    assert resources["gqa_signed_long_m256_n32"]["unified_vgpr_count"] == 252


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_d64_causal_mha_accuracy_gfx950(monkeypatch):
    case_name = "mha_square_16k_causal"
    batch, sq, skv, hq, hkv, head_dim, causal = _D64_VALIDATION_SHAPES[case_name]
    assert causal and hq == hkv and sq == skv == 16384
    real_mha_launcher = vars(amd_fa_bwd).get("_launch_bwd_d64_causal_mha_dkdv")
    original_retained = _launch_bwd_d64_dkdv
    producer_calls = []

    def record_mha(*args):
        producer_calls.append(("mha", args[-1]))
        assert callable(real_mha_launcher)
        return real_mha_launcher(*args)

    def record_retained(*args, **kwargs):
        producer_calls.append(("retained", args[-1]))
        return original_retained(*args, **kwargs)

    monkeypatch.setitem(vars(amd_fa_bwd), "_launch_bwd_d64_causal_mha_dkdv", record_mha)
    monkeypatch.setitem(vars(amd_fa_bwd), "_launch_bwd_d64_dkdv", record_retained)

    case = _make_d64_aten_case(
        (batch, hq, hkv, sq, skv, head_dim),
        seed=223,
        causal=True,
    )
    actual = fa_backward(*case.kernel_args)

    assert [call[0] for call in producer_calls] == ["mha"]
    dispatch = producer_calls[0][1]
    assert dispatch.family == "causal_scheduled_mha"
    assert dispatch.stat_mode == _D64_MHA_POSITIVE
    for name, result, expected in zip(("dq", "dk", "dv"), actual, case.grads):
        assert torch.isfinite(result).all(), (case_name, name)
        expected_norm = torch.linalg.vector_norm(expected.float())
        assert torch.isfinite(expected_norm) and expected_norm.item() > 0.0
        error_norm = torch.linalg.vector_norm(result.float() - expected.float())
        assert torch.isfinite(error_norm), (case_name, name)
        relative_l2 = error_norm / expected_norm
        assert relative_l2.item() < 5e-3, (
            case_name,
            name,
            relative_l2.item(),
        )


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize(
    "shape",
    (
        pytest.param(
            (1, 24, 24, 4096, 4096, 64),
            id="square-sq4096-skv4096",
        ),
        pytest.param(
            (1, 24, 24, 4096, 8192, 64),
            id="rectangular-sq4096-skv8192",
        ),
    ),
)
def test_d64_causal_mha_direct_publication_gfx950(monkeypatch, shape):
    real_mha_launcher = vars(amd_fa_bwd).get("_launch_bwd_d64_causal_mha_dkdv")
    assert callable(real_mha_launcher)
    case = _make_d64_aten_case(shape, seed=227, causal=True)
    batch, hq, hkv, sq, skv, _head_dim = shape
    dispatch = _select_d64_dispatch(
        tuple(case.q.shape),
        tuple(case.k.shape),
        True,
        arch="gfx950:sramecc+:xnack-",
        cu_count=256,
        sm_scale=case.sm_scale,
        bases_aligned_16=True,
    )
    assert dispatch.family == "causal_scheduled_mha"
    delta = torch.sum(case.o.float() * case.do.float(), dim=-1)
    dq = torch.empty_like(case.q)
    dk = torch.full_like(case.k, float("nan"))
    dv = torch.full_like(case.v, float("nan"))
    producer_targets = []

    def record_publication(*args):
        producer_targets.append((args[6], args[7]))
        return real_mha_launcher(*args)

    def keep_precomputed_delta(*args):
        assert args[6] is delta
        assert args[7] is None

    def reject_partial_allocation(*_args, **_kwargs):
        raise AssertionError("direct causal MHA publication must not allocate partials")

    monkeypatch.setitem(vars(amd_fa_bwd), "_launch_bwd_d64_causal_dq", keep_precomputed_delta)
    monkeypatch.setitem(vars(amd_fa_bwd), "_launch_bwd_d64_causal_mha_dkdv", record_publication)
    monkeypatch.setitem(vars(amd_fa_bwd), "_allocate_bwd_d64_kv_partials", reject_partial_allocation)
    monkeypatch.setitem(
        vars(amd_fa_bwd),
        "_allocate_bwd_d64_causal_gqa8_workspaces",
        reject_partial_allocation,
    )
    monkeypatch.setitem(vars(amd_fa_bwd), "_launch_bwd_d64_dkdv", reject_partial_allocation)

    _run_bwd_d64(
        case.q,
        case.k,
        case.v,
        case.o,
        case.do,
        case.lse,
        delta,
        dq,
        dk,
        dv,
        case.sm_scale,
        True,
        dispatch,
    )
    torch.cuda.synchronize()

    assert len(producer_targets) == 1
    assert producer_targets[0][0] is dk
    assert producer_targets[0][1] is dv
    for name, result, expected in (
        ("dk", dk, case.grads[1]),
        ("dv", dv, case.grads[2]),
    ):
        assert torch.isfinite(result).all(), name
        relative_l2 = torch.linalg.vector_norm(result.float() - expected.float()) / torch.linalg.vector_norm(
            expected.float())
        assert relative_l2.item() < 5e-3, (name, relative_l2.item())


def _compile_d64_causal_mha_producer_variant(name, shape):
    batch, hq, hkv, sq, skv, head_dim = shape
    assert hq == hkv
    q = torch.empty((batch, hq, sq, head_dim), device="cuda", dtype=torch.bfloat16)
    k = torch.empty((batch, hkv, skv, head_dim), device="cuda", dtype=torch.bfloat16)
    v = torch.empty_like(k)
    do = torch.empty_like(q)
    lse = torch.empty((batch, hq, sq), device="cuda", dtype=torch.float32)
    delta = torch.empty_like(lse)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dispatch = _select_d64_dispatch(
        tuple(q.shape),
        tuple(k.shape),
        True,
        arch="gfx950:sramecc+:xnack-",
        cu_count=256,
        sm_scale=0.125,
        bases_aligned_16=True,
    )
    assert dispatch.family == "causal_scheduled_mha"
    kernel = vars(amd_fa_bwd).get("_attn_bwd_dkdv_d64_causal_mha_kernel")
    assert kernel is not None, name
    kernel.device_caches.clear()
    _launch_bwd_d64_causal_mha_dkdv(
        q,
        k,
        v,
        do,
        lse,
        delta,
        dk,
        dv,
        0.125,
        dispatch,
    )
    torch.cuda.synchronize()
    device = torch.cuda.current_device()
    objects = tuple(kernel.device_caches[device][0].values())
    assert len(objects) == 1, (name, len(objects))
    return dispatch, objects[0]


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize(
    "variant_name, shape",
    (
        pytest.param(
            "mha_bm32_bn64_square",
            (1, 24, 24, 4096, 4096, 64),
            id="square-sq4096-skv4096",
        ),
        pytest.param(
            "mha_bm32_bn64_rectangular",
            (1, 24, 24, 4096, 8192, 64),
            id="rectangular-sq4096-skv8192",
        ),
    ),
)
def test_d64_causal_mha_codegen_gfx950(variant_name, shape):
    dispatch, obj = _compile_d64_causal_mha_producer_variant(variant_name, shape)
    assert dispatch.family == "causal_scheduled_mha"
    resource = _assert_d64_code_object_scratch_free(variant_name, obj)
    assert resource["vgpr_count"] is not None
    assert resource["vector_vgpr_count"] is not None
    assert resource["agpr_count"] is not None
    assert resource["unified_vgpr_count"] == resource["vgpr_count"]
    assert resource["unified_vgpr_count"] >= (resource["vector_vgpr_count"] + resource["agpr_count"])
    assert resource["lds_bytes"] == 16896
    amdgcn = obj.asm["amdgcn"]
    assert not re.search(r"\b\w*atomic\w*\b", amdgcn)
    assert re.search(r"\bbuffer_store_dwordx4\b", amdgcn)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_d64_causal_mha_positive_compatibility_gfx950(monkeypatch):
    batch, heads, sq, head_dim = 1, 24, 4096, 64
    generator = torch.Generator(device="cuda")
    generator.manual_seed(109)

    def random(shape):
        return torch.randn(
            shape,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        ).contiguous()

    q = random((batch, heads, sq, head_dim))
    k = random(q.shape)
    v = random(q.shape)
    do = random(q.shape)
    sm_scale = head_dim**-0.5
    state = torch.ops.aten._scaled_dot_product_flash_attention.default(q, k, v, 0.0, True, False, scale=sm_scale)
    o, lse, cum_q, cum_k, max_q, max_k, rng, unused, _debug = state
    o = o.contiguous()
    lse = lse.contiguous()
    reference = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(
        do,
        q,
        k,
        v,
        o,
        lse,
        cum_q,
        cum_k,
        max_q,
        max_k,
        0.0,
        True,
        rng,
        unused,
        scale=sm_scale,
    )

    calls = []
    original_dq = _launch_bwd_d64_causal_dq
    original_dkdv = _launch_bwd_d64_causal_mha_dkdv

    def record_dq(*args, **kwargs):
        calls.append(("dq", args[7], args[6], args[-1]))
        return original_dq(*args, **kwargs)

    def record_dkdv(*args, **kwargs):
        calls.append(("dkdv", args[5]))
        return original_dkdv(*args, **kwargs)

    def reject_preprocess(*_args, **_kwargs):
        calls.append(("preprocess", ))
        raise AssertionError("selected causal MHA must not preprocess")

    monkeypatch.setitem(vars(amd_fa_bwd), "_launch_bwd_d64_causal_dq", record_dq)
    monkeypatch.setitem(vars(amd_fa_bwd), "_launch_bwd_d64_causal_mha_dkdv", record_dkdv)
    monkeypatch.setitem(vars(amd_fa_bwd), "_run_bwd_preprocess", reject_preprocess)

    actual = fa_backward(q, k, v, o, do, lse, sm_scale, True)

    assert [call[0] for call in calls] == ["dq", "dkdv"]
    assert calls[0][1] is None
    assert calls[0][2] is calls[1][1]
    dispatch = calls[0][3]
    assert dispatch.family == "causal_scheduled_mha"
    assert dispatch.stat_mode == _D64_MHA_POSITIVE
    for name, result, expected in zip(("dq", "dk", "dv"), actual, reference):
        assert torch.isfinite(result).all(), name
        relative_l2 = torch.linalg.vector_norm(result.float() - expected.float()) / torch.linalg.vector_norm(
            expected.float())
        assert relative_l2.item() < 5e-3, (name, relative_l2.item())


def test_d64_causal_gqa8_signed_pipeline_gfx950(monkeypatch):
    original_allocator = _allocate_bwd_d64_causal_gqa8_workspaces
    assert _launch_bwd_d64_causal_gqa8_dkdv is not None
    assert _launch_bwd_d64_causal_gqa8_reduce is not None

    class Properties:
        gcnArchName = "gfx950:sramecc+:xnack-"
        multi_processor_count = 256

    batch, hq, hkv, sq, skv, head_dim = 4, 48, 6, 4096, 16384, 64
    q = torch.empty((batch, hq, sq, head_dim), device="meta", dtype=torch.bfloat16)
    k = torch.empty((batch, hkv, skv, head_dim), device="meta", dtype=torch.bfloat16)
    v = torch.empty_like(k)
    o = torch.empty_like(q)
    do = torch.empty_like(q)
    lse = torch.empty((batch, hq, sq), device="meta", dtype=torch.float32)
    launches = []
    workspace = {}

    def allocate(q_arg, k_arg):
        lse_term, dk_part, dv_part = original_allocator(q_arg, k_arg)
        workspace.update(
            lse_term=lse_term,
            dk_part=dk_part,
            dv_part=dv_part,
        )
        return lse_term, dk_part, dv_part

    def launch_dq(*args):
        launches.append(("dq", args[6], args[7], args[-1]))

    def launch_producer(*args):
        launches.append(("producer", args[4], args[5], args[6], args[7], args[-1]))

    def launch_reducer(*args):
        launches.append(("reduce", args[0], args[1], args[2], args[3]))

    def reject_legacy(*_args, **_kwargs):
        raise AssertionError("selected signed GQA8 must not preprocess, convert, or use the retained producer")

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device: Properties())
    monkeypatch.setitem(vars(amd_fa_bwd), "_allocate_bwd_d64_causal_gqa8_workspaces", allocate)
    monkeypatch.setitem(vars(amd_fa_bwd), "_launch_bwd_d64_causal_dq", launch_dq)
    monkeypatch.setitem(vars(amd_fa_bwd), "_launch_bwd_d64_causal_gqa8_dkdv", launch_producer)
    monkeypatch.setitem(vars(amd_fa_bwd), "_launch_bwd_d64_causal_gqa8_reduce", launch_reducer)
    monkeypatch.setitem(vars(amd_fa_bwd), "_run_bwd_preprocess", reject_legacy)
    monkeypatch.setitem(vars(amd_fa_bwd), "_launch_bwd_d64_fused_dq_convert", reject_legacy)
    monkeypatch.setitem(vars(amd_fa_bwd), "_launch_bwd_d64_dkdv", reject_legacy)

    outputs = fa_backward(q, k, v, o, do, lse, 0.125, True)
    assert [launch[0] for launch in launches] == ["dq", "producer", "reduce"]
    dq_launch, producer_launch, reduce_launch = launches
    dispatch = dq_launch[-1]
    assert dispatch.family == "causal_scheduled_gqa8"
    assert dispatch.stat_mode == _D64_GQA_SIGNED
    assert dispatch.kv_splits == 4
    assert dq_launch[1] is producer_launch[2]
    assert dq_launch[2] is workspace["lse_term"] is producer_launch[1]
    assert producer_launch[3] is workspace["dk_part"] is reduce_launch[1]
    assert producer_launch[4] is workspace["dv_part"] is reduce_launch[2]
    partial_shape = (batch, hkv, _D64_CAUSAL_GQA8_KV_SPLITS, skv, head_dim)
    assert tuple(workspace["lse_term"].shape) == (batch, hq, sq)
    assert workspace["lse_term"].dtype == torch.float32
    for partial in (workspace["dk_part"], workspace["dv_part"]):
        assert tuple(partial.shape) == partial_shape
        assert partial.dtype == torch.bfloat16
        assert partial.is_contiguous()
    for output in outputs:
        assert output.dtype == torch.bfloat16
        assert output.is_contiguous()


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_d64_causal_gqa8_tiny_scale_retained_accuracy_gfx950(monkeypatch):
    shape = (8, 16, 2, 1024, 1024, 64)
    case = _make_d64_aten_case(
        shape,
        seed=307,
        causal=True,
        sm_scale=1e-38,
    )
    dispatches = []
    original_run = _run_bwd_d64

    def record_run(*args):
        dispatches.append(args[-1])
        return original_run(*args)

    def reject_selected_workspace(*_args, **_kwargs):
        raise AssertionError("tiny scale must use retained D64 workspaces")

    monkeypatch.setitem(vars(amd_fa_bwd), "_run_bwd_d64", record_run)
    monkeypatch.setitem(
        vars(amd_fa_bwd),
        "_allocate_bwd_d64_causal_gqa8_workspaces",
        reject_selected_workspace,
    )
    actual = fa_backward(*case.kernel_args)

    assert len(dispatches) == 1
    assert dispatches[0].family == "causal_m192"
    assert not dispatches[0].selected_causal
    for name, result, expected in zip(("dq", "dk", "dv"), actual, case.grads):
        assert torch.isfinite(result).all(), name
        error_norm = torch.linalg.vector_norm((result.float() - expected.float()).double())
        reference_norm = torch.linalg.vector_norm(expected.double())
        assert reference_norm.item() > 0.0, name
        relative_l2 = error_norm / reference_norm
        assert relative_l2.item() < 5e-3, (name, relative_l2.item())


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_d64_causal_gqa8_cyclic_analytic_accuracy_gfx950(monkeypatch):
    batch, hq, hkv, sequence, head_dim = 8, 64, 8, 16384, 64
    sm_scale = head_dim**-0.5
    counts = torch.arange(
        1,
        sequence + 1,
        device="cuda",
        dtype=torch.float64,
    )
    v_values = torch.linspace(
        -1.0,
        1.0,
        sequence,
        device="cuda",
        dtype=torch.float32,
    ).to(torch.bfloat16)
    prefix_mean = (torch.cumsum(v_values.double(), dim=0) / counts).to(torch.bfloat16)

    q = torch.zeros(
        (batch, hq, sequence, head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    k = torch.zeros(
        (batch, hkv, sequence, head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    v = torch.zeros_like(k)
    o = torch.zeros_like(q)
    do = torch.zeros_like(q)
    q[..., 0] = 1.0
    v[..., 0] = v_values
    o[..., 0] = prefix_mean
    do[..., 0] = 1.0
    lse = torch.log(counts).float()[None, None, :].expand(batch, hq, sequence).contiguous()

    dispatches = []
    producer_grids = []
    producer_kernel = _attn_bwd_dkdv_d64_causal_gqa8_kernel
    producer_kernel.device_caches.clear()
    original_select = _select_d64_dispatch_for_device

    def record_select(*args):
        dispatch = original_select(*args)
        dispatches.append(dispatch)
        return dispatch

    class RecordProducerGrid:

        def __getitem__(self, grid):
            producer_grids.append(grid)
            return producer_kernel[grid]

    monkeypatch.setitem(vars(amd_fa_bwd), "_select_d64_dispatch_for_device", record_select)
    monkeypatch.setitem(
        vars(amd_fa_bwd),
        "_attn_bwd_dkdv_d64_causal_gqa8_kernel",
        RecordProducerGrid(),
    )
    dq, dk, dv = fa_backward(q, k, v, o, do, lse, sm_scale, True)
    torch.cuda.synchronize()

    assert len(dispatches) == 1
    dispatch = dispatches[0]
    assert dispatch.family == "causal_scheduled_gqa8"
    assert dispatch.cyclic_query_split
    assert dispatch.gqa_grid_mode == _D64_GQA_XCD_N_FAST
    assert dispatch.dkdv_lifetime == _D64_GQA_DIRECT_D64
    expected_grid = batch * hkv * 4 * triton.cdiv(sequence, 128)
    assert producer_grids == [(expected_grid, )]

    for name, result in (("dq", dq), ("dk", dk), ("dv", dv)):
        assert torch.isfinite(result).all(), name
    assert torch.count_nonzero(dq).item() == 0
    assert torch.count_nonzero(dk[..., 1:]).item() == 0
    assert torch.count_nonzero(dv[..., 1:]).item() == 0

    inverse_counts = counts.reciprocal()
    harmonic_tail = torch.flip(
        torch.cumsum(torch.flip(inverse_counts, dims=(0, )), dim=0),
        dims=(0, ),
    )
    weighted_output_tail = torch.flip(
        torch.cumsum(
            torch.flip(prefix_mean.double() * inverse_counts, dims=(0, )),
            dim=0,
        ),
        dims=(0, ),
    )
    expected_dk = sm_scale * 8.0 * (v_values.double() * harmonic_tail - weighted_output_tail)
    expected_dv = 8.0 * harmonic_tail
    for name, result, expected in (
        ("dk", dk[..., 0], expected_dk),
        ("dv", dv[..., 0], expected_dv),
    ):
        expected = expected[None, None, :].expand_as(result)
        relative_l2 = torch.linalg.vector_norm(result.double() - expected)
        relative_l2 /= torch.linalg.vector_norm(expected)
        assert relative_l2.item() < 5e-3, (name, relative_l2.item())

    device = torch.cuda.current_device()
    objects = tuple(producer_kernel.device_caches[device][0].values())
    assert len(objects) == 1
    resources = _assert_d64_code_object_scratch_free("cyclic_analytic", objects[0])
    assert resources["lds_bytes"] == 36864
    assert not re.search(r"\b\w*atomic\w*\b", objects[0].asm["amdgcn"])


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize("case_name", _D64_CAUSAL_GQA8_VALIDATION_CASES)
def test_d64_causal_gqa8_accuracy_gfx950(case_name):
    assert _launch_bwd_d64_causal_gqa8_dkdv is not None
    batch, sq, skv, hq, hkv, head_dim, causal = _D64_VALIDATION_SHAPES[case_name]
    assert causal
    case = _make_d64_aten_case(
        (batch, hq, hkv, sq, skv, head_dim),
        seed=211 + _D64_CAUSAL_GQA8_VALIDATION_CASES.index(case_name),
        causal=True,
    )
    actual = fa_backward(*case.kernel_args)
    for name, result, expected in zip(("dq", "dk", "dv"), actual, case.grads):
        assert torch.isfinite(result).all(), (case_name, name)
        relative_l2 = torch.linalg.vector_norm(result.float() - expected.float()) / torch.linalg.vector_norm(
            expected.float())
        assert relative_l2.item() < 5e-3, (case_name, name, relative_l2.item())


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_d64_causal_gqa8_two_stage_m192_accuracy_gfx950():
    shape = (12, 8, 1, 1024, 3072, 64)
    case = _make_d64_aten_case(shape, seed=1203, causal=True)
    dispatch = _select_d64_dispatch_for_device(
        case.q,
        case.k,
        case.v,
        case.o,
        case.do,
        case.lse,
        case.sm_scale,
        True,
    )
    assert dispatch.family == "causal_scheduled_gqa8"
    assert dispatch.owner_rows == 192

    kernel = _attn_bwd_dq_d64_causal_gqa8_kernel
    kernel.device_caches.clear()
    actual = fa_backward(*case.kernel_args)
    for name, result, expected in zip(("dq", "dk", "dv"), actual, case.grads):
        assert torch.isfinite(result).all(), name
        relative_l2 = torch.linalg.vector_norm(result.float() - expected.float())
        relative_l2 /= torch.linalg.vector_norm(expected.float())
        assert relative_l2.item() < 5e-3, (name, relative_l2.item())

    device = torch.cuda.current_device()
    objects = tuple(kernel.device_caches[device][0].values())
    assert len(objects) == 1
    resources = _assert_d64_code_object_scratch_free("two_stage_m192", objects[0])
    assert resources["lds_bytes"] == 33536


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_d64_causal_gqa8_odd_frontier_accuracy_gfx950():
    shape = (8, 64, 8, 1088, 1152, 64)
    batch, hq, hkv, sq, skv, _head_dim = shape
    dispatch = _select_d64_dispatch(
        (batch, hq, sq, 64),
        (batch, hkv, skv, 64),
        True,
        arch="gfx950:sramecc+:xnack-",
        cu_count=256,
        sm_scale=0.125,
        bases_aligned_16=True,
    )
    assert dispatch.family == "causal_scheduled_gqa8"
    assert dispatch.dkdv_lifetime == _D64_GQA_INTERLEAVED_D32
    assert dispatch.gqa_grid_mode == _D64_GQA_XCD
    start_m_blk, _masked = _d64_causal_physical_frontier(128, sq, skv, 64, 128)
    assert start_m_blk == 1

    case = _make_d64_aten_case(shape, seed=293, causal=True)
    actual = fa_backward(*case.kernel_args)
    for name, result, expected in zip(("dq", "dk", "dv"), actual, case.grads):
        assert torch.isfinite(result).all(), name
        relative_l2 = torch.linalg.vector_norm(result.float() - expected.float()) / torch.linalg.vector_norm(
            expected.float())
        assert relative_l2.item() < 5e-3, (name, relative_l2.item())

    compiled_dispatch, obj = _compile_d64_causal_gqa8_producer_variant("odd_frontier", shape)
    assert compiled_dispatch == dispatch
    resources = _assert_d64_code_object_scratch_free("odd_frontier", obj)
    assert resources["lds_bytes"] == 33792
    assert not re.search(r"\b\w*atomic\w*\b", obj.asm["amdgcn"])


_D64_GQA8_COMPILED_VARIANTS = {}


def _compile_d64_causal_gqa8_producer_variant(name, shape, *, lifetime_mode=None):
    cache_key = (name, tuple(shape), lifetime_mode)
    cached = _D64_GQA8_COMPILED_VARIANTS.get(cache_key)
    if cached is not None:
        return cached
    batch, hq, hkv, sq, skv, head_dim = shape
    q = torch.empty((batch, hq, sq, head_dim), device="cuda", dtype=torch.bfloat16)
    k = torch.empty((batch, hkv, skv, head_dim), device="cuda", dtype=torch.bfloat16)
    v = torch.empty_like(k)
    do = torch.empty_like(q)
    lse_term, dk_part, dv_part = _allocate_bwd_d64_causal_gqa8_workspaces(q, k)
    delta = torch.empty_like(lse_term)
    dispatch = _select_d64_dispatch(
        tuple(q.shape),
        tuple(k.shape),
        True,
        arch="gfx950:sramecc+:xnack-",
        cu_count=256,
        sm_scale=0.125,
        bases_aligned_16=True,
    )
    assert dispatch.family == "causal_scheduled_gqa8"
    if lifetime_mode is not None:
        dispatch = dataclasses.replace(dispatch, dkdv_lifetime=lifetime_mode)
    amd_fa_bwd._attn_bwd_dkdv_d64_causal_gqa8_kernel.device_caches.clear()
    _launch_bwd_d64_causal_gqa8_dkdv(
        q,
        k,
        v,
        do,
        lse_term,
        delta,
        dk_part,
        dv_part,
        0.125,
        dispatch,
    )
    torch.cuda.synchronize()
    device = torch.cuda.current_device()
    objects = tuple(amd_fa_bwd._attn_bwd_dkdv_d64_causal_gqa8_kernel.device_caches[device][0].values())
    assert len(objects) == 1, (name, len(objects))
    result = dispatch, objects[0]
    _D64_GQA8_COMPILED_VARIANTS[cache_key] = result
    return result


def test_d64_causal_gqa8_compiled_variant_cache_identity(monkeypatch):
    compiled_objects = []

    class FakeProducerKernel:

        def __init__(self):
            self.device_caches = {}

        def __getitem__(self, _grid):

            def launch(*_args, **_kwargs):
                compiled = object()
                compiled_objects.append(compiled)
                self.device_caches[0] = ({"variant": compiled}, )

            return launch

    real_empty = torch.empty

    def meta_empty(*args, **kwargs):
        kwargs["device"] = "meta"
        return real_empty(*args, **kwargs)

    monkeypatch.setitem(globals(), "_D64_GQA8_COMPILED_VARIANTS", {})
    monkeypatch.setitem(
        vars(amd_fa_bwd),
        "_attn_bwd_dkdv_d64_causal_gqa8_kernel",
        FakeProducerKernel(),
    )
    monkeypatch.setattr(torch, "empty", meta_empty)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    first_shape = (4, 48, 6, 4096, 4096, 64)
    second_shape = (4, 48, 6, 4096, 8192, 64)
    first = _compile_d64_causal_gqa8_producer_variant("shared", first_shape)
    repeated = _compile_d64_causal_gqa8_producer_variant("shared", first_shape)
    different_shape = _compile_d64_causal_gqa8_producer_variant("shared", second_shape)
    different_lifetime = _compile_d64_causal_gqa8_producer_variant(
        "shared",
        first_shape,
        lifetime_mode=_D64_GQA_INTERLEAVED_D32,
    )

    assert repeated is first
    assert different_shape is not first
    assert different_lifetime is not first
    assert different_lifetime is not different_shape
    assert len(compiled_objects) == 3


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_d64_causal_gqa8_lifetime_codegen_gfx950():
    assert _attn_bwd_dkdv_d64_causal_gqa8_kernel is not None
    same_shape = (4, 48, 6, 4096, 4096, 64)
    d32_objects = {}
    for lifetime in (
            _D64_GQA_INDEPENDENT_D32,
            _D64_GQA_INTERLEAVED_D32,
    ):
        dispatch, obj = _compile_d64_causal_gqa8_producer_variant(
            f"same_shape_{lifetime}",
            same_shape,
            lifetime_mode=lifetime,
        )
        assert dispatch.dkdv_lifetime == lifetime
        assert dispatch.gqa_grid_mode == _D64_GQA_XCD
        assert not dispatch.cyclic_query_split
        ttgir = obj.asm.get("ttgir", "")
        assert ttgir, lifetime
        assert "tensor<128x32xf32" in ttgir, lifetime
        resource = _assert_d64_code_object_scratch_free(lifetime, obj)
        assert resource["lds_bytes"] == 33792
        d32_objects[lifetime] = obj
    independent = d32_objects[_D64_GQA_INDEPENDENT_D32]
    interleaved = d32_objects[_D64_GQA_INTERLEAVED_D32]
    assert independent.asm["ttgir"] != interleaved.asm["ttgir"]
    assert independent.asm["amdgcn"] != interleaved.asm["amdgcn"]

    direct_dispatch, direct = _compile_d64_causal_gqa8_producer_variant(
        "gqa8_square_16k_direct_d64",
        (2, 32, 4, 16384, 16384, 64),
    )
    assert direct_dispatch.dkdv_lifetime == _D64_GQA_DIRECT_D64
    assert _d64_causal_gqa8_batch_stats4(16384, 16384, direct_dispatch)
    assert "tensor<128x32xf32" not in direct.asm.get("ttgir", "")
    direct_resource = _assert_d64_code_object_scratch_free("direct", direct)
    assert direct_resource["lds_bytes"] == 70656

    source = _attn_bwd_dkdv_d64_causal_gqa8_kernel.src
    assert "_d64_gqa8_direct_d64_impl" in source
    assert "_d64_gqa8_d32_impl" in source
    assert "LIFETIME_MODE == _D64_GQA_DIRECT_D64_JIT" in source
    assert "INTERLEAVED_D32" in source


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_d64_causal_gqa8_reducer_order_left_associated_fp32_gfx950():
    dk_values_cpu = torch.tensor([0.5, 1.0, 2**24, -(2**24)], dtype=torch.bfloat16)
    dv_values_cpu = torch.tensor([0.5, 2**24, -1.0, -(2**24)], dtype=torch.bfloat16)
    assert dk_values_cpu[3].item() != 0
    assert dv_values_cpu[3].item() != 0

    def fp32_add(left, right):
        return torch.add(left.to(torch.float32), right.to(torch.float32))

    def reference_orders(values):
        left = fp32_add(
            fp32_add(fp32_add(values[0], values[1]), values[2]),
            values[3],
        )
        balanced = fp32_add(
            fp32_add(values[0], values[1]),
            fp32_add(values[2], values[3]),
        )
        alternate_pairs = fp32_add(
            fp32_add(values[0], values[2]),
            fp32_add(values[1], values[3]),
        )
        reordered = fp32_add(
            fp32_add(fp32_add(values[0], values[2]), values[1]),
            values[3],
        )
        reverse = fp32_add(
            fp32_add(fp32_add(values[3], values[2]), values[1]),
            values[0],
        )
        return left, (balanced, alternate_pairs, reordered, reverse)

    dk_expected, dk_forbidden = reference_orders(dk_values_cpu)
    dv_expected, dv_forbidden = reference_orders(dv_values_cpu)
    dk_expected = dk_expected.to(torch.bfloat16)
    dv_expected = dv_expected.to(torch.bfloat16)
    assert dk_expected.item() == 2.0
    assert dv_expected.item() == -1.0
    assert all(not torch.equal(dk_expected, result.to(torch.bfloat16)) for result in dk_forbidden)
    assert all(not torch.equal(dv_expected, result.to(torch.bfloat16)) for result in dv_forbidden)

    dk_values = dk_values_cpu.to(device="cuda")
    dv_values = dv_values_cpu.to(device="cuda")
    dk_part = (dk_values.view(1, 1, 4, 1, 1).expand(1, 1, 4, 128, 64).contiguous())
    dv_part = (dv_values.view(1, 1, 4, 1, 1).expand(1, 1, 4, 128, 64).contiguous())
    dk = torch.full((1, 1, 128, 64), 7, device="cuda", dtype=torch.bfloat16)
    dv = torch.full_like(dk, -7)
    _launch_bwd_d64_causal_gqa8_reduce(dk_part, dv_part, dk, dv)
    torch.cuda.synchronize()

    assert dk.dtype == dv.dtype == torch.bfloat16
    assert torch.equal(dk, torch.full_like(dk, dk_expected.item()))
    assert torch.equal(dv, torch.full_like(dv, dv_expected.item()))

    reducer_source = _attn_bwd_dkdv_d64_causal_gqa8_reduce_kernel.src
    reducer_ast = ast.parse(reducer_source).body[0]

    def accumulator_chain(name):
        return [
            ast.unparse(statement.value)
            for statement in reducer_ast.body
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id == name
        ]

    assert accumulator_chain("dk_acc") == [
        "tlx.zeros((BLOCK_N, D), tl.float32, layout=out_layout)",
    ]
    assert accumulator_chain("dv_acc") == [
        "tlx.zeros((BLOCK_N, D), tl.float32, layout=out_layout)",
    ]
    reduction_loops = [statement for statement in reducer_ast.body if isinstance(statement, ast.For)]
    assert len(reduction_loops) == 1
    reduction_loop = reduction_loops[0]
    assert ast.unparse(reduction_loop.target) == "split"
    assert ast.unparse(reduction_loop.iter) == "tl.static_range(0, KV_SPLITS)"
    accumulator_updates = [(ast.unparse(statement.target), ast.unparse(statement.value))
                           for statement in reduction_loop.body
                           if isinstance(statement, ast.AugAssign) and isinstance(statement.op, ast.Add)]
    assert accumulator_updates == [
        ("dk_acc", "dk_part.to(tl.float32)"),
        ("dv_acc", "dv_part.to(tl.float32)"),
    ]
    assert reducer_source.count("dk_acc.to(tl.bfloat16)") == 1
    assert reducer_source.count("dv_acc.to(tl.bfloat16)") == 1
    for split in range(4):
        assert f"dk_split{split}.to(tl.bfloat16)" not in reducer_source
        assert f"dv_split{split}.to(tl.bfloat16)" not in reducer_source


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_d64_causal_gqa8_reducer_determinism_gfx950():
    assert _launch_bwd_d64_causal_gqa8_reduce is not None
    generator = torch.Generator(device="cuda")
    generator.manual_seed(271)
    dk_part = torch.randn(
        (1, 2, 4, 256, 64),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    dv_part = torch.randn(
        (1, 2, 4, 256, 64),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    baseline = None
    for run in range(20):
        dk = torch.full(
            (1, 2, 256, 64),
            run + 1,
            device="cuda",
            dtype=torch.bfloat16,
        )
        dv = torch.full_like(dk, -(run + 1))
        _launch_bwd_d64_causal_gqa8_reduce(dk_part, dv_part, dk, dv)
        torch.cuda.synchronize()
        current = (dk.clone(), dv.clone())
        if baseline is None:
            baseline = current
        else:
            assert torch.equal(current[0], baseline[0]), run
            assert torch.equal(current[1], baseline[1]), run
    assert torch.isfinite(baseline[0]).all()
    assert torch.isfinite(baseline[1]).all()


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_d64_causal_gqa8_end_to_end_determinism_gfx950():
    assert _launch_bwd_d64_causal_gqa8_dkdv is not None
    case = _make_d64_aten_case((4, 48, 6, 4096, 4096, 64), seed=277, causal=True)
    baseline = None
    for run in range(5):
        _dq, dk, dv = fa_backward(*case.kernel_args)
        torch.cuda.synchronize()
        current = (dk.clone(), dv.clone())
        if baseline is None:
            baseline = current
        else:
            assert torch.equal(current[0], baseline[0]), run
            assert torch.equal(current[1], baseline[1]), run


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((2, 32, 4, 16384, 16384, 64), id="square"),
        pytest.param((4, 48, 6, 4096, 8192, 64), id="rectangular"),
    ],
)
def test_d64_causal_gqa8_stats4_producer_determinism_gfx950(shape):
    assert _launch_bwd_d64_causal_gqa8_dkdv is not None
    case = _make_d64_aten_case(shape, seed=281, causal=True)
    dispatch = _select_d64_dispatch_for_device(*case.kernel_args[:-2], case.sm_scale, case.causal)
    assert _d64_causal_gqa8_batch_stats4(case.q.shape[2], case.k.shape[2], dispatch)

    lse_term, dk_part, dv_part = _allocate_bwd_d64_causal_gqa8_workspaces(case.q, case.k)
    delta = torch.empty_like(lse_term)
    dq = torch.empty_like(case.q)
    _launch_bwd_d64_causal_dq(
        case.q,
        case.k,
        case.v,
        case.o,
        case.do,
        case.lse,
        delta,
        lse_term,
        dq,
        case.sm_scale,
        dispatch,
    )

    baseline = None
    for run in range(3):
        _launch_bwd_d64_causal_gqa8_dkdv(
            case.q,
            case.k,
            case.v,
            case.do,
            lse_term,
            delta,
            dk_part,
            dv_part,
            case.sm_scale,
            dispatch,
        )
        torch.cuda.synchronize()
        current = (dk_part.clone(), dv_part.clone())
        if baseline is None:
            baseline = current
        else:
            assert torch.equal(current[0], baseline[0]), run
            assert torch.equal(current[1], baseline[1]), run


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_d64_causal_gqa8_codegen_gfx950():
    assert _attn_bwd_dkdv_d64_causal_gqa8_kernel is not None
    assert _attn_bwd_dkdv_d64_causal_gqa8_reduce_kernel is not None
    producer_variants = {
        "square_xcd": (
            (2, 32, 4, 16384, 16384, 64),
            _D64_GQA_XCD,
            False,
            _D64_GQA_DIRECT_D64,
        ),
        "short_square_xcd": (
            (4, 48, 6, 4096, 4096, 64),
            _D64_GQA_XCD,
            False,
            _D64_GQA_DIRECT_D64,
        ),
        "rectangle_xcd": (
            (4, 48, 6, 4096, 16384, 64),
            _D64_GQA_XCD,
            False,
            _D64_GQA_DIRECT_D64,
        ),
        "cyclic": (
            (8, 64, 8, 16384, 16384, 64),
            _D64_GQA_XCD_N_FAST,
            True,
            _D64_GQA_DIRECT_D64,
        ),
    }
    resources = {}
    for name, (shape, expected_mode, expected_cyclic, expected_lifetime) in producer_variants.items():
        dispatch, obj = _compile_d64_causal_gqa8_producer_variant(name, shape)
        resources[name] = _assert_d64_code_object_scratch_free(name, obj)
        if _d64_causal_gqa8_batch_stats4(shape[3], shape[4], dispatch):
            expected_lds_bytes = 70656
        elif expected_cyclic:
            expected_lds_bytes = 36864
        else:
            expected_lds_bytes = 33792
        assert resources[name]["lds_bytes"] == expected_lds_bytes, (name, resources[name])
        if expected_lifetime == _D64_GQA_DIRECT_D64:
            assert resources[name]["agpr_count"] == 0, (name, resources[name])
        assert dispatch.gqa_grid_mode == expected_mode, name
        assert dispatch.cyclic_query_split is expected_cyclic, name
        assert dispatch.dkdv_lifetime == expected_lifetime, name
        amdgcn = obj.asm["amdgcn"]
        assert not re.search(r"\b\w*atomic\w*\b", amdgcn), name
        assert re.search(r"\bbuffer_store_dwordx4\b", amdgcn), name

    dk_part = torch.empty((1, 1, 4, 256, 64), device="cuda", dtype=torch.bfloat16)
    dv_part = torch.empty_like(dk_part)
    dk = torch.empty((1, 1, 256, 64), device="cuda", dtype=torch.bfloat16)
    dv = torch.empty_like(dk)
    _attn_bwd_dkdv_d64_causal_gqa8_reduce_kernel.device_caches.clear()
    _launch_bwd_d64_causal_gqa8_reduce(dk_part, dv_part, dk, dv)
    torch.cuda.synchronize()
    device = torch.cuda.current_device()
    reducer_objects = tuple(_attn_bwd_dkdv_d64_causal_gqa8_reduce_kernel.device_caches[device][0].values())
    assert len(reducer_objects) == 1
    reducer = reducer_objects[0]
    resources["reducer"] = _assert_d64_code_object_scratch_free("reducer", reducer)
    assert resources["reducer"]["lds_bytes"] == 0
    reducer_asm = reducer.asm["amdgcn"]
    assert not re.search(r"\b\w*atomic\w*\b", reducer_asm)
    assert re.search(r"\bbuffer_store_dwordx4\b", reducer_asm)

    producer_source = _attn_bwd_dkdv_d64_causal_gqa8_kernel.src
    consume_source = _d64_gqa8_d32_consume.src
    assert "tlx.buffer_load(K" in producer_source
    assert "tlx.buffer_load(V" in producer_source
    assert "qdo_stages: tl.constexpr = 4 if BATCH_STATS4 else 2" in producer_source
    assert "tlx.local_alloc((BLOCK_M, D), tl.bfloat16, qdo_stages" in producer_source
    assert "n0 + BLOCK_N - 1 > m_blk * BLOCK_M + (SKV - SQ)" in consume_source
    assert "LSE_MODE == _D64_LSE_NEG_LOG2E_JIT" in producer_source
    assert "DELTA_MODE == _D64_DELTA_NEGATED_JIT" in producer_source
    assert "atomic" not in producer_source.lower()

    reducer_source = _attn_bwd_dkdv_d64_causal_gqa8_reduce_kernel.src
    assert "for split in tl.static_range(0, KV_SPLITS)" in reducer_source
    assert "dk_acc += dk_part.to(tl.float32)" in reducer_source
    assert "dv_acc += dv_part.to(tl.float32)" in reducer_source
    assert "dk_acc.to(tl.bfloat16)" in reducer_source
    assert "dv_acc.to(tl.bfloat16)" in reducer_source


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_d64_fused_n256_gqa8_correctness_and_partials_gfx950(monkeypatch):
    case = _make_d64_aten_case((1, 16, 2, 4096, 4096, 64), seed=53)
    launches = []
    original_producer = _launch_bwd_d64_fused_n256
    original_convert = _launch_bwd_d64_fused_dq_convert
    original_reduce = _launch_bwd_d64_kv_reduce

    def record_producer(
        q,
        k,
        v,
        do,
        lse,
        delta,
        dq_acc,
        dk_owner,
        dv_owner,
        sm_scale,
        dispatch,
    ):
        launches.append((
            "producer",
            tuple(dk_owner.shape),
            tuple(dv_owner.shape),
            dk_owner.dtype,
            dv_owner.dtype,
            dispatch.kv_splits,
        ))
        return original_producer(
            q,
            k,
            v,
            do,
            lse,
            delta,
            dq_acc,
            dk_owner,
            dv_owner,
            sm_scale,
            dispatch,
        )

    def record_convert(dq_acc, dq):
        launches.append(("convert", dq_acc.dtype, dq.dtype))
        return original_convert(dq_acc, dq)

    def record_reduce(dk_part, dv_part, dk, dv, dispatch):
        launches.append((
            "reduce",
            tuple(dk_part.shape),
            tuple(dv_part.shape),
            dk_part.dtype,
            dv_part.dtype,
            dispatch.kv_splits,
        ))
        return original_reduce(dk_part, dv_part, dk, dv, dispatch)

    monkeypatch.setitem(vars(amd_fa_bwd), "_launch_bwd_d64_fused_n256", record_producer)
    monkeypatch.setitem(vars(amd_fa_bwd), "_launch_bwd_d64_fused_dq_convert", record_convert)
    monkeypatch.setitem(vars(amd_fa_bwd), "_launch_bwd_d64_kv_reduce", record_reduce)
    _attn_bwd_dq_d64_direct_kernel.device_caches.clear()
    _attn_bwd_dkdv_d64_direct_kernel.device_caches.clear()
    _attn_bwd_d64_fused_n256_kernel.device_caches.clear()
    _attn_bwd_d64_fused_dq_convert_kernel.device_caches.clear()
    _attn_bwd_dkdv_d64_reduce_kernel.device_caches.clear()

    actual = fa_backward(*case.kernel_args)

    partial_shape = (1, 2, 8, 4096, 64)
    assert launches == [
        (
            "producer",
            partial_shape,
            partial_shape,
            torch.bfloat16,
            torch.bfloat16,
            8,
        ),
        ("convert", torch.float32, torch.bfloat16),
        (
            "reduce",
            partial_shape,
            partial_shape,
            torch.bfloat16,
            torch.bfloat16,
            8,
        ),
    ]
    producer_source = _attn_bwd_d64_fused_n256_kernel.src
    assert "pid_split = pid_hq % group_size" in producer_source
    assert ("((pid_b * HKV + pid_hkv) * KV_SPLITS + pid_split)" in producer_source)

    for name, result, expected in zip(("dq", "dk", "dv"), actual, case.grads):
        assert torch.isfinite(result).all(), name
        relative_l2 = torch.linalg.vector_norm(result.float() - expected.float()) / torch.linalg.vector_norm(
            expected.float())
        assert relative_l2.item() < 5e-3, (name, relative_l2.item())

    device = torch.cuda.current_device()
    assert (device not in _attn_bwd_dq_d64_direct_kernel.device_caches
            or not _attn_bwd_dq_d64_direct_kernel.device_caches[device][0])
    assert (device not in _attn_bwd_dkdv_d64_direct_kernel.device_caches
            or not _attn_bwd_dkdv_d64_direct_kernel.device_caches[device][0])
    code_objects = {
        "producer": tuple(_attn_bwd_d64_fused_n256_kernel.device_caches[device][0].values()),
        "convert": tuple(_attn_bwd_d64_fused_dq_convert_kernel.device_caches[device][0].values()),
        "reduce": tuple(_attn_bwd_dkdv_d64_reduce_kernel.device_caches[device][0].values()),
    }
    assert all(len(objects) == 1 for objects in code_objects.values())
    for name, objects in code_objects.items():
        _assert_d64_code_object_scratch_free(name, objects[0])
    producer_asm = code_objects["producer"][0].asm["amdgcn"]
    assert re.search(r"\bbuffer_atomic_add_f32\b", producer_asm)
    assert "buffer_atomic_pk_add_bf16" not in producer_asm


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_d64_fused_n256_mha_correctness_and_codegen_gfx950():
    case = _make_d64_aten_case((1, 16, 16, 4096, 4096, 64), seed=47)
    _attn_bwd_dq_d64_direct_kernel.device_caches.clear()
    _attn_bwd_dkdv_d64_direct_kernel.device_caches.clear()
    _attn_bwd_d64_fused_n256_kernel.device_caches.clear()
    _attn_bwd_d64_fused_dq_convert_kernel.device_caches.clear()
    _attn_bwd_dkdv_d64_reduce_kernel.device_caches.clear()

    actual = fa_backward(*case.kernel_args)

    for name, result, expected in zip(("dq", "dk", "dv"), actual, case.grads):
        assert torch.isfinite(result).all(), name
        relative_l2 = torch.linalg.vector_norm(result.float() - expected.float()) / torch.linalg.vector_norm(
            expected.float())
        assert relative_l2.item() < 5e-3, (name, relative_l2.item())

    device = torch.cuda.current_device()
    assert (device not in _attn_bwd_dq_d64_direct_kernel.device_caches
            or not _attn_bwd_dq_d64_direct_kernel.device_caches[device][0])
    assert (device not in _attn_bwd_dkdv_d64_direct_kernel.device_caches
            or not _attn_bwd_dkdv_d64_direct_kernel.device_caches[device][0])
    assert (device not in _attn_bwd_dkdv_d64_reduce_kernel.device_caches
            or not _attn_bwd_dkdv_d64_reduce_kernel.device_caches[device][0])
    producer = tuple(_attn_bwd_d64_fused_n256_kernel.device_caches[device][0].values())
    convert = tuple(_attn_bwd_d64_fused_dq_convert_kernel.device_caches[device][0].values())
    assert len(producer) == len(convert) == 1
    for name, obj in (("producer", producer[0]), ("convert", convert[0])):
        _assert_d64_code_object_scratch_free(name, obj)
    producer_asm = producer[0].asm["amdgcn"]
    assert re.search(r"\bbuffer_atomic_add_f32\b", producer_asm)
    assert "buffer_atomic_pk_add_bf16" not in producer_asm


@pytest.mark.parametrize(
    ("q_shape", "k_shape", "causal", "expected"),
    [
        ((2, 32, 16384, 64), (2, 32, 16384, 64), False, 1),
        ((2, 32, 16384, 64), (2, 4, 16384, 64), False, 8),
        ((2, 32, 16384, 64), (2, 4, 16384, 64), True, 4),
        ((4, 48, 4096, 64), (4, 6, 16384, 64), True, 4),
    ],
)
def test_d64_gqa_kv_split_policy(q_shape, k_shape, causal, expected):
    assert _select_d64_dispatch(q_shape, k_shape, causal).kv_splits == expected


@pytest.mark.parametrize(
    ("owner_start", "owner_rows", "sq", "skv", "block_n", "expected"),
    [
        (0, 192, 4096, 4096, 32, 6),
        (192, 192, 4096, 4096, 32, 12),
        (0, 192, 4096, 16384, 64, 195),
        (3840, 192, 4096, 4096, 32, 126),
    ],
)
def test_d64_causal_dq_compact_key_frontier(owner_start, owner_rows, sq, skv, block_n, expected):
    assert _d64_causal_dq_key_blocks(owner_start, owner_rows, sq, skv, block_n) == expected


@pytest.mark.parametrize(
    ("key_start", "sq", "skv", "block_m", "expected"),
    [
        (0, 4096, 4096, 64, 0),
        (256, 4096, 4096, 64, 4),
        (12288, 4096, 16384, 64, 0),
        (16320, 4096, 16384, 64, 63),
    ],
)
def test_d64_causal_dkdv_compact_query_frontier(key_start, sq, skv, block_m, expected):
    assert _d64_causal_dkdv_first_query_block(key_start, sq, skv, block_m) == expected


@pytest.mark.parametrize(
    ("q_shape", "k_shape", "causal"),
    [
        ((2, 32, 16384, 64), (2, 32, 16384, 64), False),
        ((2, 32, 16384, 64), (2, 4, 16384, 64), True),
        ((4, 48, 4096, 64), (4, 6, 4096, 64), True),
        ((4, 48, 4096, 64), (4, 6, 16384, 64), True),
    ],
)
def test_d64_launch_uses_structural_dq_owner(monkeypatch, q_shape, k_shape, causal):

    class LaunchRecorder:

        def __init__(self):
            self.calls = []

        def __getitem__(self, grid):

            def record(*args, **kwargs):
                self.calls.append((grid, args, kwargs))

            return record

    dq_launch = LaunchRecorder()
    dkdv_launch = LaunchRecorder()
    reduce_launch = LaunchRecorder()
    monkeypatch.setitem(vars(amd_fa_bwd), "_attn_bwd_dq_d64_direct_kernel", dq_launch)
    monkeypatch.setitem(vars(amd_fa_bwd), "_attn_bwd_dkdv_d64_direct_kernel", dkdv_launch)
    monkeypatch.setitem(vars(amd_fa_bwd), "_attn_bwd_dkdv_d64_reduce_kernel", reduce_launch)
    q = torch.empty(q_shape, device="meta", dtype=torch.bfloat16)
    k = torch.empty(k_shape, device="meta", dtype=torch.bfloat16)
    dispatch = _select_d64_dispatch(q_shape, k_shape, causal)

    _run_bwd_d64_direct(q, k, k, q, object(), object(), q, k, k, 0.125, causal, dispatch)

    assert len(dq_launch.calls) == 1
    grid, _args, kwargs = dq_launch.calls[0]
    assert grid == (triton.cdiv(q_shape[2], dispatch.owner_rows), q_shape[1], q_shape[0])
    assert kwargs["OWNER_ROWS"] == dispatch.owner_rows
    assert kwargs["BLOCK_N"] == dispatch.key_rows
    assert len(dkdv_launch.calls) == 1
    dkdv_grid, _args, dkdv_kwargs = dkdv_launch.calls[0]
    assert dkdv_grid == (
        triton.cdiv(k_shape[2], 64),
        k_shape[1] * dispatch.kv_splits,
        k_shape[0],
    )
    assert dkdv_kwargs["KV_SPLITS"] == dispatch.kv_splits
    assert dkdv_kwargs["BLOCK_N"] == 64
    assert len(reduce_launch.calls) == (1 if dispatch.kv_splits > 1 else 0)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize(
    ("q_shape", "k_shape", "causal"),
    [
        pytest.param((1, 1, 256, 64), (1, 1, 256, 64), False, id="noncausal-n256-mha"),
        pytest.param((1, 8, 256, 64), (1, 1, 256, 64), False, id="noncausal-n256-gqa8"),
        pytest.param((1, 1, 256, 64), (1, 1, 256, 64), True, id="causal-m192-square"),
        pytest.param((1, 8, 256, 64), (1, 1, 256, 64), True, id="causal-m192-square-gqa8"),
        pytest.param((1, 8, 256, 64), (1, 1, 512, 64), True, id="causal-m192-rect-gqa8"),
        pytest.param((1, 1, 16384, 64), (1, 1, 16384, 64), True, id="causal-m256-deep"),
        pytest.param((1, 8, 16384, 64), (1, 1, 16384, 64), True, id="causal-m256-deep-gqa8"),
    ],
)
def test_d64_retained_specializations_are_scratch_free_gfx950(q_shape, k_shape, causal):
    q = torch.zeros(q_shape, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(k_shape, device="cuda", dtype=torch.bfloat16)
    lse = torch.zeros(q_shape[:-1], device="cuda", dtype=torch.float32)
    delta = torch.zeros_like(lse)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(k)
    _attn_bwd_dq_d64_direct_kernel.device_caches.clear()
    _attn_bwd_dkdv_d64_direct_kernel.device_caches.clear()
    _attn_bwd_dkdv_d64_reduce_kernel.device_caches.clear()

    dispatch = _select_d64_dispatch(q_shape, k_shape, causal)
    _run_bwd_d64_direct(q, k, k, q, lse, delta, dq, dk, dv, 0.125, causal, dispatch)
    torch.cuda.synchronize()

    device = torch.cuda.current_device()
    kernels = [
        ("dq", _attn_bwd_dq_d64_direct_kernel),
        ("dkdv", _attn_bwd_dkdv_d64_direct_kernel),
    ]
    if dispatch.kv_splits > 1:
        kernels.append(("reduce", _attn_bwd_dkdv_d64_reduce_kernel))
    for name, kernel in kernels:
        compiled = tuple(kernel.device_caches[device][0].values())
        assert len(compiled) == 1, (name, len(compiled))
        _assert_d64_code_object_scratch_free(name, compiled[0])


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize(
    ("shape", "causal", "family", "kernel_names"),
    [
        pytest.param(
            (1, 16, 16, 4096, 4096, 64),
            False,
            "noncausal_fused_n256",
            (
                "_attn_bwd_d64_fused_n256_kernel",
                "_attn_bwd_d64_fused_dq_convert_kernel",
            ),
            id="noncausal-mha",
        ),
        pytest.param(
            (1, 16, 2, 4096, 4096, 64),
            False,
            "noncausal_fused_n256",
            (
                "_attn_bwd_d64_fused_n256_kernel",
                "_attn_bwd_d64_fused_dq_convert_kernel",
                "_attn_bwd_dkdv_d64_reduce_kernel",
            ),
            id="noncausal-gqa8",
        ),
        pytest.param(
            (1, 24, 24, 4096, 4096, 64),
            True,
            "causal_scheduled_mha",
            (
                "_attn_bwd_dq_d64_causal_mha_kernel",
                "_attn_bwd_dkdv_d64_causal_mha_kernel",
            ),
            id="causal-mha",
        ),
        pytest.param(
            (4, 48, 6, 1024, 1024, 64),
            True,
            "causal_scheduled_gqa8",
            (
                "_attn_bwd_dq_d64_causal_gqa8_kernel",
                "_attn_bwd_dkdv_d64_causal_gqa8_kernel",
                "_attn_bwd_dkdv_d64_causal_gqa8_reduce_kernel",
            ),
            id="causal-gqa8",
        ),
        pytest.param(
            (4, 48, 6, 1024, 2048, 64),
            True,
            "causal_scheduled_gqa8",
            (
                "_attn_bwd_dq_d64_causal_gqa8_kernel",
                "_attn_bwd_dkdv_d64_causal_gqa8_kernel",
                "_attn_bwd_dkdv_d64_causal_gqa8_reduce_kernel",
            ),
            id="causal-gqa8-rectangular",
        ),
    ],
)
def test_d64_selected_routes_correct_and_scratch_free_gfx950(shape, causal, family, kernel_names, monkeypatch):
    from triton.language.extra.tlx.tutorials import amd_fa_bwd

    # Other tutorials imported by this module disable the post-RA scheduler
    # process-wide. Verify D64 with its default codegen configuration.
    monkeypatch.delenv("TRITON_DISABLE_POST_MISCHED", raising=False)
    case = _make_d64_aten_case(shape, seed=311, causal=causal)
    properties = torch.cuda.get_device_properties(case.q.device)
    dispatch = amd_fa_bwd._select_d64_dispatch(
        tuple(case.q.shape),
        tuple(case.k.shape),
        causal,
        arch=properties.gcnArchName,
        cu_count=properties.multi_processor_count,
        sm_scale=case.sm_scale,
        bases_aligned_16=True,
    )
    assert dispatch.family == family

    kernels = [getattr(amd_fa_bwd, name) for name in kernel_names]
    for kernel in kernels:
        kernel.device_caches.clear()

    actual = amd_fa_bwd.fa_backward(*case.kernel_args)
    for name, result, expected in zip(("dq", "dk", "dv"), actual, case.grads, strict=True):
        assert torch.isfinite(result).all(), name
        relative_l2 = torch.linalg.vector_norm(result.float() - expected.float()) / torch.linalg.vector_norm(
            expected.float())
        assert relative_l2.item() < 5e-3, (name, relative_l2.item())

    device = torch.cuda.current_device()
    for kernel_name, kernel in zip(kernel_names, kernels, strict=True):
        objects = tuple(kernel.device_caches[device][0].values())
        assert objects, kernel_name
        for index, obj in enumerate(objects):
            _assert_d64_code_object_scratch_free(f"{kernel_name}[{index}]", obj)
            if causal:
                assert not re.search(r"\b\w*atomic\w*\b", obj.asm["amdgcn"]), kernel_name
