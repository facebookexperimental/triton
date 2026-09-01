"""Focused integration tests for AMD FlashAttention backward routes."""

import inspect
import math
import re

import pytest
import torch
import triton

from triton.language.extra.tlx.tutorials import amd_fa_bwd, amd_fa_varlen_bwd
from triton.language.extra.tlx.tutorials.amd_fa_bwd import (
    ReferenceCase,
    _D64DQLaunch,
    _D64Dispatch,
    _D64_GQA_SIGNED,
    _D64_MHA_POSITIVE,
    _allocate_bwd_d64_causal_gqa8_workspaces,
    _allocate_bwd_d64_fused_workspaces,
    _d64_causal_dkdv_first_query_block,
    _d64_causal_dq_key_blocks,
    _d64_causal_stat_values,
    _d64_dq_launch_plan,
    _run_bwd_d64_direct,
    _select_d64_dispatch,
    _validate_d64_sm_scale,
    fa_backward,
    is_hip_cdna4,
)


def _make_d64_aten_case(shape, *, causal, seed):
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
    sm_scale = head_dim**-0.5
    state = torch.ops.aten._scaled_dot_product_flash_attention.default(
        q,
        k,
        v,
        0.0,
        causal,
        False,
        scale=sm_scale,
    )
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


def _assert_scratch_free(name, compiled):
    amdgcn = compiled.asm["amdgcn"]
    private_segments = {
        int(value)
        for value in re.findall(r"(?:\.amdhsa_)?private_segment_fixed_size:?\s+(\d+)", amdgcn)
    }
    resources = {
        "n_spills": compiled.n_spills,
        "global_scratch_bytes": compiled.metadata.global_scratch_size,
        "private_segment_bytes": private_segments.pop() if len(private_segments) == 1 else None,
        "scratch_loads": len(re.findall(r"\bscratch_load", amdgcn)),
        "scratch_stores": len(re.findall(r"\bscratch_store", amdgcn)),
    }
    assert resources == {
        "n_spills": 0,
        "global_scratch_bytes": 0,
        "private_segment_bytes": 0,
        "scratch_loads": 0,
        "scratch_stores": 0,
    }, (name, resources)


def test_d64_causal_stat_conventions_are_equivalent():
    generator = torch.Generator(device="cpu")
    generator.manual_seed(101)
    q = torch.randn((3, 4), generator=generator)
    k = torch.randn((5, 4), generator=generator)
    v = torch.randn((5, 4), generator=generator)
    o = torch.randn((3, 4), generator=generator)
    do = torch.randn((3, 4), generator=generator)
    lse = torch.randn((3, ), generator=generator)
    sm_scale = 0.5

    delta_mha, lse_mha = _d64_causal_stat_values(o, do, lse, sm_scale, _D64_MHA_POSITIVE)
    delta_gqa, lse_gqa = _d64_causal_stat_values(o, do, lse, sm_scale, _D64_GQA_SIGNED)

    expected_delta = torch.sum(o * do, dim=-1)
    expected_lse = -lse * math.log2(math.e)
    torch.testing.assert_close(delta_mha, expected_delta)
    torch.testing.assert_close(delta_gqa, -expected_delta)
    assert lse_mha is None
    torch.testing.assert_close(lse_gqa, expected_lse)

    scores = q @ k.mT
    p_mha = torch.exp2((scores * sm_scale - lse[..., None]) * math.log2(math.e))
    p_gqa = torch.exp2(scores * (sm_scale * math.log2(math.e)) + lse_gqa[..., None])
    ds_mha = p_mha * (do @ v.mT - delta_mha[..., None])
    ds_gqa = p_gqa * (do @ v.mT + delta_gqa[..., None])
    torch.testing.assert_close(p_mha, p_gqa)
    torch.testing.assert_close(ds_mha, ds_gqa)

    with pytest.raises(ValueError, match=r"^unknown D64 stat mode 2$"):
        _d64_causal_stat_values(o, do, lse, sm_scale, 2)


@pytest.mark.parametrize(
    "invalid_scale",
    [
        pytest.param(0.0, id="zero"),
        pytest.param(float("inf"), id="infinite"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(10**10000, id="overflowing-integer"),
    ],
)
def test_d64_scale_validation_and_scheduled_fallback(invalid_scale):
    with pytest.raises(ValueError, match=r"^D64 sm_scale must be finite and nonzero$"):
        _validate_d64_sm_scale(invalid_scale)

    shape = (4, 64, 4096, 64)
    dispatch = _select_d64_dispatch(
        shape,
        shape,
        True,
        arch="gfx950",
        cu_count=256,
        sm_scale=invalid_scale,
        bases_aligned_16=True,
    )
    assert dispatch.family == "causal_m192"


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        pytest.param(
            (4, 64, 8, 8192, 8192, 192, 256, True),
            (
                _D64DQLaunch(42, False, 0, 42, 3, 0),
                _D64DQLaunch(1, False, 42, 1, 2, 192),
            ),
            id="peeled-tail",
        ),
        pytest.param(
            (4, 64, 8, 4096, 4096, 192, 256, True),
            (_D64DQLaunch(22, True, 0, 0, 3, 0), ),
            id="single-launch",
        ),
        pytest.param(
            (4, 64, 8, 8192, 8192, 192, 256, False),
            (_D64DQLaunch(43, False, 0, 0, 3, 0), ),
            id="no-host-tail-skip",
        ),
    ],
)
def test_d64_causal_dq_launch_plan(args, expected):
    assert _d64_dq_launch_plan(*args) == expected


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


def test_d64_workspace_shapes():
    q = torch.empty((2, 32, 4096, 64), device="meta", dtype=torch.bfloat16)
    k_mha = torch.empty((2, 32, 4096, 64), device="meta", dtype=torch.bfloat16)
    k_gqa = torch.empty((2, 4, 4096, 64), device="meta", dtype=torch.bfloat16)

    lse_term, causal_dk, causal_dv = _allocate_bwd_d64_causal_gqa8_workspaces(q, k_gqa)
    assert lse_term.shape == (2, 32, 4096) and lse_term.dtype is torch.float32
    assert causal_dk.shape == causal_dv.shape == (2, 4, 4, 4096, 64)
    assert causal_dk.dtype is causal_dv.dtype is torch.bfloat16

    mha_dispatch = _D64Dispatch("noncausal_fused_n256", 32, 256, 1)
    gqa_dispatch = _D64Dispatch("noncausal_fused_n256", 32, 256, 8)
    mha_acc, mha_dk, mha_dv = _allocate_bwd_d64_fused_workspaces(q, k_mha, mha_dispatch)
    gqa_acc, gqa_dk, gqa_dv = _allocate_bwd_d64_fused_workspaces(q, k_gqa, gqa_dispatch)
    assert mha_acc.shape == gqa_acc.shape == q.shape
    assert mha_acc.dtype is gqa_acc.dtype is torch.float32
    assert mha_dk is mha_dv is None
    assert gqa_dk.shape == gqa_dv.shape == (2, 4, 8, 4096, 64)
    assert gqa_dk.dtype is gqa_dv.dtype is torch.bfloat16


def test_d64_direct_launch_uses_dispatch_ownership(monkeypatch):

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

    q_shape = (4, 48, 4096, 64)
    k_shape = (4, 6, 16384, 64)
    q = torch.empty(q_shape, device="meta", dtype=torch.bfloat16)
    k = torch.empty(k_shape, device="meta", dtype=torch.bfloat16)
    dispatch = _select_d64_dispatch(q_shape, k_shape, True)
    _run_bwd_d64_direct(q, k, k, q, object(), object(), q, k, k, 0.125, True, dispatch)

    assert len(dq_launch.calls) == 1
    dq_grid, _args, dq_kwargs = dq_launch.calls[0]
    assert dq_grid == (triton.cdiv(q_shape[2], 192), q_shape[1], q_shape[0])
    assert (dq_kwargs["OWNER_ROWS"], dq_kwargs["BLOCK_N"]) == (192, 64)

    assert len(dkdv_launch.calls) == 1
    dkdv_grid, _args, dkdv_kwargs = dkdv_launch.calls[0]
    assert dkdv_grid == (triton.cdiv(k_shape[2], 64), k_shape[1] * 4, k_shape[0])
    assert (dkdv_kwargs["KV_SPLITS"], dkdv_kwargs["BLOCK_N"]) == (4, 64)
    assert len(reduce_launch.calls) == 1


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize(
    ("shape", "causal", "family"),
    [
        pytest.param((1, 16, 2, 4096, 4096, 64), False, "noncausal_fused_n256", id="fused-gqa8"),
        pytest.param((1, 24, 24, 4096, 4096, 64), True, "causal_scheduled_mha", id="causal-mha"),
        pytest.param((4, 48, 6, 1024, 1024, 64), True, "causal_scheduled_gqa8", id="causal-gqa8"),
        pytest.param(
            (4, 48, 6, 1024, 2048, 64),
            True,
            "causal_scheduled_gqa8",
            id="bottom-right-rectangular-gqa8",
        ),
    ],
)
def test_d64_selected_route_correctness_gfx950(shape, causal, family, monkeypatch):
    monkeypatch.delenv("TRITON_DISABLE_POST_MISCHED", raising=False)
    case = _make_d64_aten_case(shape, causal=causal, seed=311)
    properties = torch.cuda.get_device_properties(case.q.device)
    dispatch = _select_d64_dispatch(
        tuple(case.q.shape),
        tuple(case.k.shape),
        causal,
        arch=properties.gcnArchName,
        cu_count=properties.multi_processor_count,
        sm_scale=case.sm_scale,
        bases_aligned_16=True,
    )
    assert dispatch.family == family

    actual = fa_backward(*case.kernel_args)
    for name, result, expected in zip(("dq", "dk", "dv"), actual, case.grads, strict=True):
        assert torch.isfinite(result).all(), name
        relative_l2 = torch.linalg.vector_norm(result.float() - expected.float()) / torch.linalg.vector_norm(
            expected.float())
        assert relative_l2.item() < 5e-3, (name, relative_l2.item())


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_d64_causal_gqa8_codegen_is_scratch_free_gfx950(monkeypatch):
    monkeypatch.delenv("TRITON_DISABLE_POST_MISCHED", raising=False)
    shape = (4, 48, 6, 1024, 1024, 64)
    batch, hq, hkv, sq, skv, head_dim = shape
    q = torch.zeros((batch, hq, sq, head_dim), device="cuda", dtype=torch.bfloat16)
    k = torch.zeros((batch, hkv, skv, head_dim), device="cuda", dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    o = torch.zeros_like(q)
    do = torch.zeros_like(q)
    lse = torch.zeros((batch, hq, sq), device="cuda", dtype=torch.float32)
    kernels = (
        amd_fa_bwd._attn_bwd_dq_d64_causal_gqa8_kernel,
        amd_fa_bwd._attn_bwd_dkdv_d64_causal_gqa8_kernel,
        amd_fa_bwd._attn_bwd_dkdv_d64_causal_gqa8_reduce_kernel,
    )
    for kernel in kernels:
        kernel.device_caches.clear()

    fa_backward(q, k, v, o, do, lse, 0.125, True)
    torch.cuda.synchronize()

    device = torch.cuda.current_device()
    for kernel in kernels:
        compiled_objects = tuple(kernel.device_caches[device][0].values())
        assert compiled_objects, kernel.fn.__name__
        for compiled in compiled_objects:
            _assert_scratch_free(kernel.fn.__name__, compiled)
            assert not re.search(r"\b\w*atomic\w*\b", compiled.asm["amdgcn"])


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_plan_owns_offsets_and_compact_schedules():
    cu_q = torch.tensor([0, 17, 48, 88], dtype=torch.int32, device="cuda")
    cu_kv = torch.tensor([0, 33, 162, 169], dtype=torch.int32, device="cuda")

    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv)

    assert plan.batch == 3
    assert plan.total_q == 88
    assert plan.total_kv == 169
    assert plan.max_q == 40
    assert plan.q_block_sequence.tolist() == [0, 0, 1, 1, 2, 2, 2]
    assert plan.q_block_start.tolist() == [0, 16, 0, 16, 0, 16, 32]
    assert plan.num_full_kv_blocks == 1
    assert plan.kv_block_sequence.tolist() == [1, 0, 1, 2]
    assert plan.kv_block_start.tolist() == [0, 0, 128, 0]
    assert plan.wide_kv_start.tolist() == [162, 33, 0]
    assert plan.wide_q_start.tolist() == [48, 17, 0]
    assert plan.wide_dq_start.tolist() == [78, 32, 0]
    assert plan.wide_q_len.tolist() == [40, 31, 17]
    assert plan.wide_kv_valid.tolist() == [7, 129, 33]

    cu_q.fill_(0)
    cu_kv.fill_(0)
    assert plan.cu_seqlens_q.tolist() == [0, 17, 48, 88]
    assert plan.cu_seqlens_k.tolist() == [0, 33, 162, 169]


def _make_varlen_d128_reference_case(q_lengths, kv_lengths, *, q_heads, kv_heads, seed):
    assert q_heads > 0 and kv_heads > 0 and q_heads % kv_heads == 0
    group_size = q_heads // kv_heads
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    total_q = sum(q_lengths)
    total_kv = sum(kv_lengths)
    scale = 128**-0.5
    q = torch.randn((total_q, q_heads, 128), dtype=torch.bfloat16, device="cuda", generator=generator)
    k = torch.randn((total_kv, kv_heads, 128), dtype=torch.bfloat16, device="cuda", generator=generator)
    v = torch.randn((total_kv, kv_heads, 128), dtype=torch.bfloat16, device="cuda", generator=generator)
    do = torch.randn((total_q, q_heads, 128), dtype=torch.bfloat16, device="cuda", generator=generator)
    out = torch.empty_like(q)
    lse = torch.empty((q_heads, total_q), dtype=torch.float32, device="cuda")
    expected_dq = torch.empty_like(q)
    expected_dk = torch.zeros_like(k, dtype=torch.float32)
    expected_dv = torch.zeros_like(v, dtype=torch.float32)

    q_begin = 0
    kv_begin = 0
    for q_length, kv_length in zip(q_lengths, kv_lengths, strict=True):
        q_end = q_begin + q_length
        kv_end = kv_begin + kv_length
        for q_head in range(q_heads):
            kv_head = q_head // group_size
            q_tile = q[q_begin:q_end, q_head].float()
            k_tile = k[kv_begin:kv_end, kv_head].float()
            v_tile = v[kv_begin:kv_end, kv_head].float()
            do_tile = do[q_begin:q_end, q_head].float()
            scores = q_tile @ k_tile.mT * scale
            lse_tile = torch.logsumexp(scores, dim=1)
            p = torch.exp(scores - lse_tile[:, None])
            out_tile = (p @ v_tile).to(torch.bfloat16)
            delta = torch.sum(out_tile.float() * do_tile, dim=1)
            dp = do_tile @ v_tile.mT
            ds = (p * (dp - delta[:, None])).to(torch.bfloat16).float()
            out[q_begin:q_end, q_head] = out_tile
            lse[q_head, q_begin:q_end] = lse_tile
            expected_dq[q_begin:q_end, q_head] = (ds @ k_tile * scale).to(torch.bfloat16)
            expected_dk[kv_begin:kv_end, kv_head].add_(ds.mT @ q_tile * scale)
            expected_dv[kv_begin:kv_end, kv_head].add_(p.to(torch.bfloat16).float().mT @ do_tile)
        q_begin = q_end
        kv_begin = kv_end

    cu_q = torch.tensor([0, *torch.tensor(q_lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    cu_kv = torch.tensor([0, *torch.tensor(kv_lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    return q, k, v, out, do, lse, cu_q, cu_kv, scale, (
        expected_dq,
        expected_dk.to(torch.bfloat16),
        expected_dv.to(torch.bfloat16),
    )


def _make_seeded_extend_attention_lengths(batch, max_context, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    prefix = torch.randint(1, max_context // 2, (batch, ), generator=generator)
    extend = torch.randint(1, max_context // 2, (batch, ), generator=generator)
    return extend.tolist(), (prefix + extend).tolist()


def test_varlen_d128_backward_api_is_noncausal():
    assert "causal" not in inspect.signature(amd_fa_varlen_bwd.fa_varlen_backward).parameters


def test_varlen_d128_seeded_extend_attention_lengths_are_reproducible():
    first = _make_seeded_extend_attention_lengths(batch=19, max_context=12331, seed=42)
    second = _make_seeded_extend_attention_lengths(batch=19, max_context=12331, seed=42)
    q_lengths, kv_lengths = first

    assert first == second
    assert len(q_lengths) == len(kv_lengths) == 19
    assert len(set(q_lengths)) > 1
    assert all(q_length > 0 for q_length in q_lengths)
    assert all(q_length < kv_length for q_length, kv_length in zip(q_lengths, kv_lengths, strict=True))


@pytest.mark.parametrize(
    ("max_q", "group_size", "expected"),
    (
        pytest.param(5460, 3, 3, id="long-gqa3"),
        pytest.param(4096, 4, 4, id="long-gqa4"),
        pytest.param(4096, 6, 3, id="long-gqa6"),
        pytest.param(2048, 8, 4, id="long-gqa8"),
        pytest.param(5456, 3, 1, id="short-gqa3"),
        pytest.param(2032, 8, 1, id="short-gqa8"),
        pytest.param(16384, 1, 1, id="mha"),
        pytest.param(16384, 5, 1, id="unsupported-gqa5"),
    ),
)
def test_varlen_d128_kv_split_selection(max_q, group_size, expected):
    assert amd_fa_varlen_bwd._select_varlen_kv_splits(max_q, group_size) == expected


@pytest.mark.parametrize(
    ("group_size", "kv_splits", "expected"),
    (
        pytest.param(3, 3, (32, 256), id="split-gqa3"),
        pytest.param(8, 4, (32, 256), id="split-gqa8"),
        pytest.param(1, 1, (16, 128), id="mha"),
        pytest.param(3, 1, (16, 128), id="unsplit-gqa"),
    ),
)
def test_varlen_d128_kernel_block_selection(group_size, kv_splits, expected):
    assert amd_fa_varlen_bwd._select_varlen_kernel_blocks(group_size, kv_splits) == expected


def test_varlen_d128_kv_partial_workspace_shapes():
    k = torch.empty((257, 2, 128), device="meta", dtype=torch.bfloat16)
    max_size_k = torch.empty((2**23, 1, 128), device="meta", dtype=torch.bfloat16)

    direct_dk, direct_dv = amd_fa_varlen_bwd._allocate_varlen_dkdv_partials(k, 1)
    split_dk, split_dv = amd_fa_varlen_bwd._allocate_varlen_dkdv_partials(k, 3)
    oversized_dk, oversized_dv = amd_fa_varlen_bwd._allocate_varlen_dkdv_partials(max_size_k, 4)

    assert direct_dk is direct_dv is None
    assert oversized_dk is oversized_dv is None
    assert split_dk.shape == split_dv.shape == (257, 2, 3, 128)
    assert split_dk.dtype is split_dv.dtype is torch.bfloat16
    assert split_dk.device.type == split_dv.device.type == "meta"


@pytest.mark.parametrize(
    ("q_lengths", "kv_lengths", "q_heads", "kv_heads"),
    (
        pytest.param([7, 31, 65], [33, 257, 7], 2, 2, id="mha-mixed-full-tail"),
        pytest.param([1, 17], [1, 127], 2, 2, id="mha-all-tail"),
        pytest.param([16, 32], [128, 256], 2, 2, id="mha-all-full"),
        pytest.param([7, 31, 65], [33, 257, 7], 6, 2, id="gqa3-mixed"),
        pytest.param([1, 17], [1, 129], 8, 1, id="gqa8-tail"),
        pytest.param([5460], [17], 3, 1, id="gqa3-long-split3"),
        pytest.param([5460], [17], 12, 4, id="gqa3-multi-kv-long-split3"),
        pytest.param([2048], [17], 8, 1, id="gqa8-long-split4"),
        pytest.param(
            *_make_seeded_extend_attention_lengths(batch=5, max_context=96, seed=443),
            12,
            4,
            id="gqa3-seeded-prefix-extend",
        ),
    ),
)
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_interleaved_lengths_gfx950(q_lengths, kv_lengths, q_heads, kv_heads):
    case = _make_varlen_d128_reference_case(
        q_lengths,
        kv_lengths,
        q_heads=q_heads,
        kv_heads=kv_heads,
        seed=431,
    )
    q, k, v, out, do, lse, cu_q, cu_kv, scale, expected = case
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv)

    actual = amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, scale)

    for name, result, reference in zip(("dq", "dk", "dv"), actual, expected, strict=True):
        assert torch.isfinite(result).all(), name
        relative_l2 = torch.linalg.vector_norm(result.float() - reference.float()) / torch.linalg.vector_norm(
            reference.float())
        assert relative_l2.item() < 1e-2, (name, relative_l2.item())


@pytest.mark.parametrize(
    ("q_lengths", "q_heads", "kv_heads"),
    (
        pytest.param([1, 17, 31, 32, 33, 5460], 3, 1, id="gqa3"),
        pytest.param([1, 17, 31, 32, 33, 2048], 8, 1, id="gqa8"),
    ),
)
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_bm32_boundaries_gfx950(q_lengths, q_heads, kv_heads):
    kv_lengths = [1, 255, 256, 257, 511, 17]
    case = _make_varlen_d128_reference_case(
        q_lengths,
        kv_lengths,
        q_heads=q_heads,
        kv_heads=kv_heads,
        seed=487 + q_heads,
    )
    q, k, v, out, do, lse, cu_q, cu_kv, scale, expected = case
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv)

    actual = amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, scale)

    for name, result, reference in zip(("dq", "dk", "dv"), actual, expected, strict=True):
        assert torch.isfinite(result).all(), name
        relative_l2 = torch.linalg.vector_norm(result.float() - reference.float()) / torch.linalg.vector_norm(
            reference.float())
        assert relative_l2.item() < 1e-2, (name, relative_l2.item())


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_runtime_totals_reuse_specialization_gfx950():
    kernels = (
        amd_fa_varlen_bwd._varlen_bwd_interleaved_kernel,
        amd_fa_varlen_bwd._varlen_dq_convert_kernel,
    )
    for kernel in kernels:
        kernel.device_caches.clear()

    for q_length in (17, 33):
        case = _make_varlen_d128_reference_case([q_length], [128], q_heads=1, kv_heads=1, seed=419 + q_length)
        q, k, v, out, do, lse, cu_q, cu_kv, scale, _expected = case
        plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv)

        amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, scale)
        torch.cuda.synchronize()

    device = torch.cuda.current_device()
    for kernel in kernels:
        assert len(kernel.device_caches[device][0]) == 1, kernel.fn.__name__


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_bm32_runtime_totals_reuse_specialization_gfx950():
    kernel = getattr(amd_fa_varlen_bwd, "_varlen_bwd_interleaved_bm32_kernel", None)
    assert kernel is not None
    kernel.device_caches.clear()

    # Keep Triton's ordinary integer divisibility specialization identical;
    # only the packed token count should differ.
    for q_length in (5460, 5476):
        case = _make_varlen_d128_reference_case([q_length], [17], q_heads=3, kv_heads=1, seed=503 + q_length)
        q, k, v, out, do, lse, cu_q, cu_kv, scale, _expected = case
        plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv)
        amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, scale)
        torch.cuda.synchronize()

    device = torch.cuda.current_device()
    assert len(kernel.device_caches[device][0]) == 1


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_plan_rejects_invalid_offsets():
    valid = torch.tensor([0, 16, 48], dtype=torch.int32, device="cuda")
    cases = (
        (torch.tensor([1, 17, 49], dtype=torch.int32, device="cuda"), valid, "must start at zero"),
        (torch.tensor([0, 16, 16], dtype=torch.int32, device="cuda"), valid, "must be strictly increasing"),
        (valid, torch.tensor([0, 32], dtype=torch.int32, device="cuda"), "must describe the same batch"),
    )
    for cu_q, cu_kv, message in cases:
        with pytest.raises(ValueError, match=message):
            amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv)


def test_varlen_d128_address_space_requires_i32_offsets():
    # One D128 BF16 head fits at most 2**30 elements in the signed i32
    # byte-offset range used by AMD buffer instructions.
    max_tokens = 2**23

    amd_fa_varlen_bwd._validate_i32_buffer_offsets(
        total_q=max_tokens - 15,
        total_kv=max_tokens,
        batch=1,
        q_heads=1,
        kv_heads=1,
    )

    with pytest.raises(ValueError, match="KV tensor size exceeds the signed 32-bit byte-offset range"):
        amd_fa_varlen_bwd._validate_i32_buffer_offsets(
            total_q=1,
            total_kv=max_tokens + 1,
            batch=1,
            q_heads=1,
            kv_heads=1,
        )
    with pytest.raises(ValueError, match="padded dQ size exceeds the signed 32-bit byte-offset range"):
        amd_fa_varlen_bwd._validate_i32_buffer_offsets(
            total_q=max_tokens - 14,
            total_kv=1,
            batch=1,
            q_heads=1,
            kv_heads=1,
        )


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_backward_rejects_unsupported_signature():
    case = _make_varlen_d128_reference_case([16], [128], q_heads=1, kv_heads=1, seed=407)
    q, k, v, out, do, lse, cu_q, cu_kv, scale, _expected = case
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv)

    with pytest.raises(ValueError, match="q must be contiguous bfloat16 THD"):
        amd_fa_varlen_bwd.fa_varlen_backward(q.float(), k, v, out, do, lse, plan, scale)
    with pytest.raises(ValueError, match="head dimension 128"):
        amd_fa_varlen_bwd.fa_varlen_backward(
            q[..., :64],
            k[..., :64],
            v[..., :64],
            out[..., :64],
            do[..., :64],
            lse,
            plan,
            scale,
        )
    with pytest.raises(ValueError, match="positive Q and KV head counts"):
        amd_fa_varlen_bwd.fa_varlen_backward(
            q[:, :0],
            k[:, :0],
            v[:, :0],
            out[:, :0],
            do[:, :0],
            lse[:0],
            plan,
            scale,
        )
    with pytest.raises(ValueError, match="positive Q and KV head counts"):
        amd_fa_varlen_bwd.fa_varlen_backward(
            q,
            k[:, :0],
            v[:, :0],
            out,
            do,
            lse,
            plan,
            scale,
        )
    with pytest.raises(ValueError, match="sm_scale must be finite"):
        amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, float("nan"))


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_backward_rejects_nondivisible_gqa():
    case = _make_varlen_d128_reference_case([16], [128], q_heads=4, kv_heads=2, seed=411)
    q, k, v, out, do, lse, cu_q, cu_kv, scale, _expected = case
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv)
    invalid_k = torch.empty((k.shape[0], 3, 128), dtype=k.dtype, device=k.device)
    invalid_v = torch.empty_like(invalid_k)

    with pytest.raises(ValueError, match="divisible"):
        amd_fa_varlen_bwd.fa_varlen_backward(q, invalid_k, invalid_v, out, do, lse, plan, scale)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_backward_rejects_noncontiguous_lse():
    case = _make_varlen_d128_reference_case([16], [128], q_heads=2, kv_heads=2, seed=413)
    q, k, v, out, do, lse, cu_q, cu_kv, scale, _expected = case
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv)
    lse_storage = torch.empty((2, 32), dtype=torch.float32, device="cuda")
    strided_lse = lse_storage[:, ::2]
    assert strided_lse.shape == lse.shape
    assert not strided_lse.is_contiguous()

    with pytest.raises(ValueError, match="lse must be contiguous FP32"):
        amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, strided_lse, plan, scale)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_interleaved_codegen_is_scratch_free_gfx950():
    kernels = (
        amd_fa_varlen_bwd._varlen_bwd_preprocess,
        amd_fa_varlen_bwd._varlen_bwd_interleaved_kernel,
        amd_fa_varlen_bwd._varlen_dq_convert_kernel,
    )
    for kernel in kernels:
        kernel.device_caches.clear()

    case = _make_varlen_d128_reference_case([17], [129], q_heads=3, kv_heads=1, seed=409)
    q, k, v, out, do, lse, cu_q, cu_kv, scale, _expected = case
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv)
    amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, scale)
    torch.cuda.synchronize()

    device = torch.cuda.current_device()
    expected_shared = (256, 64_640, 0)
    expected_specializations = (1, 2, 1)
    for kernel, shared, specialization_count in zip(kernels, expected_shared, expected_specializations, strict=True):
        compiled_objects = tuple(kernel.device_caches[device][0].values())
        assert len(compiled_objects) == specialization_count
        for compiled in compiled_objects:
            _assert_scratch_free(kernel.fn.__name__, compiled)
            assert compiled.metadata.num_warps == 4
            assert compiled.metadata.shared == shared

    for interleaved in kernels[1].device_caches[device][0].values():
        assert "buffer_atomic_pk_add_bf16" in interleaved.asm["amdgcn"]


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_split_codegen_is_scratch_free_gfx950():
    kernels = (
        amd_fa_varlen_bwd._varlen_bwd_interleaved_bm32_kernel,
        amd_fa_varlen_bwd._varlen_dkdv_reduce_kernel,
    )
    for kernel in kernels:
        kernel.device_caches.clear()

    for q_length, q_heads in ((5460, 3), (2048, 8)):
        case = _make_varlen_d128_reference_case([q_length], [129], q_heads=q_heads, kv_heads=1, seed=463 + q_heads)
        q, k, v, out, do, lse, cu_q, cu_kv, scale, _expected = case
        plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv)
        amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, scale)
    torch.cuda.synchronize()

    device = torch.cuda.current_device()
    for kernel, expected_shared, specialization_count in zip(kernels, (117_536, 0), (2, 2), strict=True):
        compiled_objects = tuple(kernel.device_caches[device][0].values())
        assert len(compiled_objects) == specialization_count
        for compiled in compiled_objects:
            _assert_scratch_free(kernel.fn.__name__, compiled)
            assert compiled.metadata.num_warps == 4
            assert compiled.metadata.shared == expected_shared
    for compiled in kernels[0].device_caches[device][0].values():
        assert "buffer_atomic_pk_add_bf16" in compiled.asm["amdgcn"]
