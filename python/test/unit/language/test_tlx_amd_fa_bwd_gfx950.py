"""TLX AMD tests -- CDNA4 (gfx950)."""
import re
import pytest
import torch
from triton.language.extra.tlx.tutorials import amd_fa_bwd, amd_fa_varlen_bwd
from triton.language.extra.tlx.tutorials.amd_fa_bwd import (
    ReferenceCase,
    _select_d64_dispatch,
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


def _make_varlen_d128_reference_case(q_lengths, kv_lengths, *, heads, seed):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    total_q = sum(q_lengths)
    total_kv = sum(kv_lengths)
    scale = 128**-0.5
    q = torch.randn((total_q, heads, 128), dtype=torch.bfloat16, device="cuda", generator=generator)
    k = torch.randn((total_kv, heads, 128), dtype=torch.bfloat16, device="cuda", generator=generator)
    v = torch.randn((total_kv, heads, 128), dtype=torch.bfloat16, device="cuda", generator=generator)
    do = torch.randn((total_q, heads, 128), dtype=torch.bfloat16, device="cuda", generator=generator)
    out = torch.empty_like(q)
    lse = torch.empty((heads, total_q), dtype=torch.float32, device="cuda")
    expected_dq = torch.empty_like(q)
    expected_dk = torch.empty_like(k)
    expected_dv = torch.empty_like(v)

    q_begin = 0
    kv_begin = 0
    for q_length, kv_length in zip(q_lengths, kv_lengths, strict=True):
        q_end = q_begin + q_length
        kv_end = kv_begin + kv_length
        for head in range(heads):
            q_tile = q[q_begin:q_end, head].float()
            k_tile = k[kv_begin:kv_end, head].float()
            v_tile = v[kv_begin:kv_end, head].float()
            do_tile = do[q_begin:q_end, head].float()
            scores = q_tile @ k_tile.mT * scale
            lse_tile = torch.logsumexp(scores, dim=1)
            p = torch.exp(scores - lse_tile[:, None])
            out_tile = (p @ v_tile).to(torch.bfloat16)
            delta = torch.sum(out_tile.float() * do_tile, dim=1)
            dp = do_tile @ v_tile.mT
            ds = (p * (dp - delta[:, None])).to(torch.bfloat16).float()
            out[q_begin:q_end, head] = out_tile
            lse[head, q_begin:q_end] = lse_tile
            expected_dq[q_begin:q_end, head] = (ds @ k_tile * scale).to(torch.bfloat16)
            expected_dk[kv_begin:kv_end, head] = (ds.mT @ q_tile * scale).to(torch.bfloat16)
            expected_dv[kv_begin:kv_end, head] = (p.to(torch.bfloat16).float().mT @ do_tile).to(torch.bfloat16)
        q_begin = q_end
        kv_begin = kv_end

    cu_q = torch.tensor([0, *torch.tensor(q_lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    cu_kv = torch.tensor([0, *torch.tensor(kv_lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    return q, k, v, out, do, lse, cu_q, cu_kv, scale, (expected_dq, expected_dk, expected_dv)


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

    cu_q.fill_(0)
    cu_kv.fill_(0)
    assert plan.cu_seqlens_q.tolist() == [0, 17, 48, 88]
    assert plan.cu_seqlens_k.tolist() == [0, 33, 162, 169]


@pytest.mark.parametrize(
    ("q_lengths", "kv_lengths"),
    (
        pytest.param([7, 31, 65], [33, 257, 7], id="mixed-full-tail"),
        pytest.param([1, 17], [1, 127], id="all-tail"),
        pytest.param([16, 32], [128, 256], id="all-full"),
    ),
)
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_interleaved_mixed_lengths_gfx950(q_lengths, kv_lengths):
    case = _make_varlen_d128_reference_case(q_lengths, kv_lengths, heads=2, seed=401)
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
        case = _make_varlen_d128_reference_case([q_length], [128], heads=1, seed=419 + q_length)
        q, k, v, out, do, lse, cu_q, cu_kv, scale, _expected = case
        plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv)

        amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, scale)
        torch.cuda.synchronize()

    device = torch.cuda.current_device()
    for kernel in kernels:
        assert len(kernel.device_caches[device][0]) == 1, kernel.fn.__name__


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


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_backward_rejects_unsupported_signature():
    case = _make_varlen_d128_reference_case([16], [128], heads=1, seed=407)
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
    with pytest.raises(ValueError, match="positive head count"):
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
    with pytest.raises(ValueError, match="sm_scale must be finite"):
        amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, float("nan"))


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_backward_rejects_gqa():
    case = _make_varlen_d128_reference_case([16], [128], heads=2, seed=411)
    q, k, v, out, do, lse, cu_q, cu_kv, scale, _expected = case
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv)
    gqa_k = k[:, :1].contiguous()
    gqa_v = v[:, :1].contiguous()

    with pytest.raises(ValueError, match="equal Q and KV head counts"):
        amd_fa_varlen_bwd.fa_varlen_backward(q, gqa_k, gqa_v, out, do, lse, plan, scale)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_backward_rejects_noncontiguous_lse():
    case = _make_varlen_d128_reference_case([16], [128], heads=2, seed=413)
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
    case = _make_varlen_d128_reference_case([16], [129], heads=1, seed=409)
    q, k, v, out, do, lse, cu_q, cu_kv, scale, _expected = case
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv)
    kernels = (
        amd_fa_varlen_bwd._varlen_bwd_preprocess,
        amd_fa_varlen_bwd._varlen_bwd_interleaved_kernel,
        amd_fa_varlen_bwd._varlen_dq_convert_kernel,
    )
    for kernel in kernels:
        kernel.device_caches.clear()

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
