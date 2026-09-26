"""TLX AMD tests -- CDNA4 (gfx950)."""

from dataclasses import replace
import ast
import inspect
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


def _assert_fp32_dq_wave8_pipeline(compiled):
    ttgir = compiled.asm["ttgir"]
    # The public route must load both full statistics planes once into shared
    # memory, then read BM16 slices instead of refetching statistics per tile.
    assert len(re.findall(r"ttg\.local_alloc[^\n]*!ttg\.memdesc<2x512xf32,", ttgir)) == 1
    cache_loads = [line for line in ttgir.splitlines()
                   if "amdg.buffer_load_to_local" in line and "-> <512xf32," in line]
    assert len(cache_loads) == 2
    assert all("mask =" in line and "other =" in line for line in cache_loads)
    assert len(re.findall(r"ttg\.memdesc_dynamic_subslice[^\n]*!ttg\.memdesc<16xf32,", ttgir)) >= 2

    # Follow the store's layout alias rather than depending on SSA numbering.
    # Its low three register bits are eight consecutive BF16 D elements; the
    # warp bits retain the native D32/N32 ownership of the accumulator.
    store_layouts = []
    for line in ttgir.splitlines():
        match = re.fullmatch(
            r"(#\w+) = #ttg\.linear<\{register = (\[.*\]), lane = (\[.*\]), "
            r"warp = (\[.*\]), block = \[\]\}>", line)
        if match and ast.literal_eval(match[2])[:3] == [[0, 1], [0, 2], [0, 4]]:
            if ast.literal_eval(match[4]) == [[0, 32], [32, 0]]:
                store_layouts.append(match[1])
    assert len(store_layouts) == 1
    store_type = f"tensor<64x128xbf16, {store_layouts[0]}>"
    assert sum("amdg.buffer_store" in line and store_type in line for line in ttgir.splitlines()) == 4
    assembly = compiled.asm["amdgcn"]
    assert "buffer_atomic_add_f32" in assembly
    assert "buffer_atomic_pk_add_bf16" not in assembly
    _assert_scratch_free(compiled.name, compiled)


def _capture_kernel_with_constexprs(monkeypatch, module, name, **constexprs):
    kernel = getattr(module, name)
    launches = []

    class CapturedKernel:

        def __getitem__(self, grid):

            def launch(*args, **kwargs):
                kwargs.update(constexprs)
                compiled = kernel[grid](*args, **kwargs)
                launches.append((kwargs, compiled))
                return compiled

            return launch

    monkeypatch.setattr(module, name, CapturedKernel())
    return launches


def _make_varlen_d128_reference_case(
    q_lengths,
    kv_lengths,
    *,
    q_heads,
    kv_heads,
    seed,
    causal=False,
    strided_v=False,
    sm_scale=None,
):
    assert q_heads > 0 and kv_heads > 0 and q_heads % kv_heads == 0
    if causal:
        assert q_lengths == kv_lengths
    group_size = q_heads // kv_heads
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    total_q = sum(q_lengths)
    total_kv = sum(kv_lengths)
    scale = 128**-0.5 if sm_scale is None else sm_scale
    q = torch.randn((total_q, q_heads, 128), dtype=torch.bfloat16, device="cuda", generator=generator)
    k = torch.randn((total_kv, kv_heads, 128), dtype=torch.bfloat16, device="cuda", generator=generator)
    if strided_v:
        v_storage = torch.randn(
            (total_kv, 3, kv_heads, 128),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        v = v_storage[:, 0]
    else:
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
            if causal:
                query_positions = torch.arange(q_length, device="cuda")
                key_positions = torch.arange(kv_length, device="cuda")
                scores = scores.masked_fill(key_positions[None, :] > query_positions[:, None], float("-inf"))
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
            assert not re.search(r"(?m)^[ \t]*\w*atomic\w*(?:[ \t]|$)", compiled.asm["amdgcn"])


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize("register_class", ("default", None, "vgpr", "agpr"), ids=("default", "none", "vgpr", "agpr"))
@pytest.mark.parametrize(
    ("shape", "family", "default_class"),
    (
        pytest.param((1, 8, 8, 16384, 16384, 64), "mha", None, id="mha"),
        pytest.param((1, 16, 2, 8192, 8192, 64), "gqa8", "agpr", id="gqa8"),
    ),
)
def test_d64_q3_register_class_tuning_gfx950(monkeypatch, register_class, shape, family, default_class):
    monkeypatch.delenv("TRITON_DISABLE_POST_MISCHED", raising=False)
    options = {} if register_class == "default" else {"Q3_REGISTER_CLASS": register_class}
    expected_class = default_class if register_class == "default" else register_class
    launches = _capture_kernel_with_constexprs(monkeypatch, amd_fa_bwd, f"_attn_bwd_dq_d64_causal_{family}_kernel",
                                               **options)
    case = _make_d64_aten_case(shape, causal=True, seed=3623)

    actual = fa_backward(*case.kernel_args)
    torch.cuda.synchronize()

    for name, result, expected in zip(("dq", "dk", "dv"), actual, case.grads, strict=True):
        assert torch.isfinite(result).all(), name
        relative_l2 = torch.linalg.vector_norm(result.float() - expected.float()) / torch.linalg.vector_norm(
            expected.float())
        assert relative_l2.item() < 5e-3, (name, relative_l2.item())

    assert len(launches) == 1
    kwargs, compiled = launches[0]
    assert kwargs["OWNER_FRAGMENTS"] == 4
    q3_hints = re.findall(
        r'amdg\.register_resident [^\n]*class "(\w+)" groups (\d+) : tensor<64x64xbf16,',
        compiled.asm["ttir"],
    )
    assert q3_hints == ([] if expected_class is None else [(expected_class, "8")])


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_device_compact_schedules():
    q_lengths = [17, 31, 40]
    kv_lengths = [33, 129, 7]
    cu_q = torch.tensor([0, 17, 48, 88], dtype=torch.int32, device="cuda")
    cu_kv = torch.tensor([0, 33, 162, 169], dtype=torch.int32, device="cuda")

    plan = amd_fa_varlen_bwd.prepare_varlen_backward(
        cu_q,
        cu_kv,
        sum(q_lengths),
        sum(kv_lengths),
        max(q_lengths),
        max(kv_lengths),
    )
    amd_fa_varlen_bwd.validate_varlen_backward_plan(plan)
    q_count, full_count, tail_count, wide_count, plan_error, causal_error = plan.task_counts.tolist()
    assert plan_error == 0
    assert causal_error == 1

    def tasks(sequences, starts, count):
        return set(zip(sequences[:count].tolist(), starts[:count].tolist()))

    assert plan.batch == 3
    assert plan.total_q == 88
    assert plan.total_kv == 169
    assert plan.max_q == 40
    assert tasks(plan.q_block_sequence, plan.q_block_start, q_count) == {(sequence, start)
                                                                         for sequence, length in enumerate(q_lengths)
                                                                         for start in range(0, length, 16)}
    assert tasks(
        plan.full_kv_block_sequence,
        plan.full_kv_block_start,
        full_count,
    ) == {(sequence, start)
          for sequence, length in enumerate(kv_lengths)
          for start in range(0, length // 128 * 128, 128)}
    assert tasks(
        plan.tail_kv_block_sequence,
        plan.tail_kv_block_start,
        tail_count,
    ) == {(sequence, length // 128 * 128)
          for sequence, length in enumerate(kv_lengths)
          if length % 128}
    actual_wide = set(
        zip(
            plan.wide_kv_start[:wide_count].tolist(),
            plan.wide_q_start[:wide_count].tolist(),
            plan.wide_dq_start[:wide_count].tolist(),
            plan.wide_q_len[:wide_count].tolist(),
            plan.wide_kv_valid[:wide_count].tolist(),
        ))
    expected_wide = set()
    q_start = 0
    kv_start = 0
    for sequence, (q_len, kv_len) in enumerate(zip(q_lengths, kv_lengths, strict=True)):
        for block_start in range(0, kv_len, 256):
            expected_wide.add((
                kv_start + block_start,
                q_start,
                q_start + sequence * 15,
                q_len,
                min(256, kv_len - block_start),
            ))
        q_start += q_len
        kv_start += kv_len
    assert actual_wide == expected_wide
    assert plan.cu_seqlens_q is cu_q
    assert plan.cu_seqlens_k is cu_kv
    assert plan.dq_full_kv_sequence is None
    assert plan.dq_full_kv_start is None
    assert plan.dq_tail_k96 is False


@pytest.mark.parametrize(
    ("q_lengths", "kv_lengths", "supply_metadata", "expected_capacities"),
    (
        pytest.param([16, 32], [128, 256], False, (3, 3, 0, 2), id="legacy-all-full"),
        pytest.param([1, 17], [1, 127], False, (3, 0, 2, 2), id="legacy-all-tail"),
        pytest.param([16, 16], [128, 128], True, (2, 2, 0, 2), id="metadata-uniform-all-full"),
        pytest.param([16, 16], [64, 96], True, (2, 0, 2, 3), id="metadata-all-tail"),
        pytest.param([17, 31, 40], [33, 129, 7], True, (9, 1, 3, 4), id="metadata-mixed"),
    ),
)
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_plan_tightens_provable_schedule_capacities(q_lengths, kv_lengths, supply_metadata,
                                                                expected_capacities):
    cu_q = torch.tensor([0, *torch.tensor(q_lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    cu_kv = torch.tensor([0, *torch.tensor(kv_lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    metadata = (sum(q_lengths), sum(kv_lengths), max(q_lengths), max(kv_lengths)) if supply_metadata else ()

    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv, *metadata)

    capacities = (
        plan.q_block_sequence.numel(),
        plan.full_kv_block_sequence.numel(),
        plan.tail_kv_block_sequence.numel(),
        plan.wide_kv_start.numel(),
    )
    assert capacities == expected_capacities
    amd_fa_varlen_bwd.validate_varlen_backward_plan(plan)


def _make_seeded_extend_attention_lengths(batch, max_context, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    prefix = torch.randint(1, max_context // 2, (batch, ), generator=generator)
    extend = torch.randint(1, max_context // 2, (batch, ), generator=generator)
    return extend.tolist(), (prefix + extend).tolist()


def test_varlen_d128_backward_api_defaults_to_noncausal():
    causal = inspect.signature(amd_fa_varlen_bwd.fa_varlen_backward).parameters["causal"]

    assert causal.default is False


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_plan_records_whether_offsets_match():
    shared = torch.tensor([0, 17, 48], dtype=torch.int32, device="cuda")
    different = torch.tensor([0, 17, 49], dtype=torch.int32, device="cuda")

    assert amd_fa_varlen_bwd.prepare_varlen_backward(shared, shared.clone()).qk_offsets_equal is True
    assert amd_fa_varlen_bwd.prepare_varlen_backward(shared, different).qk_offsets_equal is False

    device_plan = amd_fa_varlen_bwd.prepare_varlen_backward(
        shared,
        shared.clone(),
        48,
        48,
        31,
        31,
    )
    assert device_plan.qk_offsets_equal is None
    amd_fa_varlen_bwd.validate_varlen_backward_plan(device_plan, causal=True)

    device_mismatch = amd_fa_varlen_bwd.prepare_varlen_backward(
        shared,
        different,
        48,
        49,
        31,
        32,
    )
    assert device_mismatch.qk_offsets_equal is None
    amd_fa_varlen_bwd.validate_varlen_backward_plan(device_mismatch)
    with pytest.raises(ValueError, match="match between Q and KV"):
        amd_fa_varlen_bwd.validate_varlen_backward_plan(device_mismatch, causal=True)


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


def test_varlen_d128_fp32_dq_padding_falls_back_at_i32_boundary():
    batch = 1024
    q_heads = 3
    elements_per_row = q_heads * amd_fa_varlen_bwd._HEAD_DIM
    max_fp32_rows = amd_fa_varlen_bwd._I32_BUFFER_FP32_ELEMENTS // elements_per_row
    total_q = max_fp32_rows - batch * (amd_fa_varlen_bwd._BLOCK_M - 1)
    padded_bm16_elements = (total_q + batch * (amd_fa_varlen_bwd._BLOCK_M - 1)) * elements_per_row
    padded_bm32_elements = (total_q + batch * (amd_fa_varlen_bwd._WIDE_BLOCK_M - 1)) * elements_per_row

    assert padded_bm16_elements <= amd_fa_varlen_bwd._I32_BUFFER_FP32_ELEMENTS
    assert amd_fa_varlen_bwd._I32_BUFFER_FP32_ELEMENTS < padded_bm32_elements
    assert padded_bm32_elements <= amd_fa_varlen_bwd._I32_BUFFER_BF16_ELEMENTS

    fp32_pad_rows = amd_fa_varlen_bwd._select_varlen_dq_pad_rows(
        total_q=total_q,
        batch=batch,
        q_heads=q_heads,
        block_m=amd_fa_varlen_bwd._WIDE_BLOCK_M,
        block_n=amd_fa_varlen_bwd._WIDE_BLOCK_N,
        dq_atomic_fp32=True,
    )
    bf16_pad_rows = amd_fa_varlen_bwd._select_varlen_dq_pad_rows(
        total_q=total_q,
        batch=batch,
        q_heads=q_heads,
        block_m=amd_fa_varlen_bwd._WIDE_BLOCK_M,
        block_n=amd_fa_varlen_bwd._WIDE_BLOCK_N,
        dq_atomic_fp32=False,
    )

    assert fp32_pad_rows == amd_fa_varlen_bwd._BLOCK_M
    assert bf16_pad_rows == amd_fa_varlen_bwd._WIDE_BLOCK_M
    amd_fa_varlen_bwd._validate_i32_buffer_offsets(
        total_q=total_q,
        total_kv=1,
        batch=batch,
        q_heads=q_heads,
        kv_heads=1,
        dq_atomic_fp32=True,
        dq_pad_rows=fp32_pad_rows,
    )
    amd_fa_varlen_bwd._validate_i32_buffer_offsets(
        total_q=total_q,
        total_kv=1,
        batch=batch,
        q_heads=q_heads,
        kv_heads=1,
        dq_atomic_fp32=False,
        dq_pad_rows=bf16_pad_rows,
    )
    with pytest.raises(ValueError, match="padded dQ size"):
        amd_fa_varlen_bwd._validate_i32_buffer_offsets(
            total_q=total_q,
            total_kv=1,
            batch=batch,
            q_heads=q_heads,
            kv_heads=1,
            dq_atomic_fp32=True,
            dq_pad_rows=amd_fa_varlen_bwd._WIDE_BLOCK_M,
        )


def test_varlen_d128_kv_partial_workspace_shapes():
    k = torch.empty((257, 2, 128), device="meta", dtype=torch.bfloat16)
    max_size_k = torch.empty((2**20, 1, 128), device="meta", dtype=torch.bfloat16)
    oversized_k = torch.empty((2**20 + 1, 1, 128), device="meta", dtype=torch.bfloat16)

    direct_dk, direct_dv = amd_fa_varlen_bwd._allocate_varlen_dkdv_partials(k, 1)
    split_dk, split_dv = amd_fa_varlen_bwd._allocate_varlen_dkdv_partials(k, 3)
    max_size_dk, max_size_dv = amd_fa_varlen_bwd._allocate_varlen_dkdv_partials(max_size_k, 4)
    oversized_dk, oversized_dv = amd_fa_varlen_bwd._allocate_varlen_dkdv_partials(oversized_k, 4)

    assert direct_dk is direct_dv is None
    assert max_size_dk is not None and max_size_dv is not None
    assert oversized_dk is oversized_dv is None
    assert split_dk.shape == split_dv.shape == (257, 2, 3, 128)
    assert split_dk.dtype is split_dv.dtype is torch.float32
    assert split_dk.device.type == split_dv.device.type == "meta"


@pytest.mark.parametrize(
    ("q_lengths", "kv_lengths", "q_heads", "kv_heads", "qdo_offsets"),
    (
        pytest.param([7, 31, 65], [33, 257, 7], 1, 1, (0, 0), id="mha1-mixed-full-tail"),
        pytest.param([7, 31, 65], [33, 257, 7], 2, 2, (0, 0), id="mha-mixed-full-tail"),
        pytest.param([1, 17], [1, 127], 2, 2, (0, 0), id="mha-all-tail"),
        pytest.param([16, 32], [128, 256], 2, 2, (0, 0), id="mha-all-full"),
        pytest.param([1, 17], [1, 96], 4, 4, (0, 0), id="mha-k96-all-tail"),
        pytest.param([1, 17, 33], [96, 224, 128], 4, 4, (0, 0), id="mha-tail-96"),
        pytest.param([1, 17, 33], [97, 225, 128], 4, 4, (0, 0), id="mha-tail-97"),
        pytest.param([1, 17, 33], [127, 255, 128], 4, 4, (0, 0), id="mha-tail-127"),
        pytest.param([1, 17, 33], [224, 225, 128], 4, 4, (0, 0), id="mha-mixed-96-97"),
        pytest.param([1, 17, 33], [192, 193, 224], 4, 4, (0, 0), id="mha-mixed-64-65-96"),
        pytest.param([1, 16, 17, 511, 512], [1, 128, 129, 257, 256], 4, 4, (0, 0), id="mha-stats-cache-boundary"),
        pytest.param([1, 17, 513, 1025], [1, 129, 257, 256], 2, 2, (0, 0), id="mha-stats-cache-fallback"),
        pytest.param([17], [129], 2, 2, (1, 0), id="mha-unaligned-q"),
        pytest.param([17], [129], 2, 2, (0, 4), id="mha-unaligned-do"),
        pytest.param([7, 31, 65], [33, 257, 7], 6, 2, (0, 0), id="gqa3-mixed"),
        pytest.param([1, 17], [1, 129], 8, 1, (0, 0), id="gqa8-tail"),
        pytest.param([5460], [17], 3, 1, (0, 0), id="gqa3-long-split3"),
        pytest.param([5460], [17], 12, 4, (0, 0), id="gqa3-multi-kv-long-split3"),
        pytest.param([2048], [17], 8, 1, (0, 0), id="gqa8-long-split4"),
        pytest.param(
            *_make_seeded_extend_attention_lengths(batch=5, max_context=96, seed=443),
            12,
            4,
            (0, 0),
            id="gqa3-seeded-prefix-extend",
        ),
    ),
)
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_interleaved_lengths_gfx950(q_lengths, kv_lengths, q_heads, kv_heads, qdo_offsets):
    case = _make_varlen_d128_reference_case(
        q_lengths,
        kv_lengths,
        q_heads=q_heads,
        kv_heads=kv_heads,
        seed=431,
    )
    q, k, v, out, do, lse, cu_q, cu_kv, scale, expected = case
    shifted_inputs = []
    for tensor, offset in zip((q, do), qdo_offsets, strict=True):
        if offset:
            storage = torch.empty(tensor.numel() + offset, dtype=tensor.dtype, device=tensor.device)
            shifted = storage[offset:].view_as(tensor)
            shifted.copy_(tensor)
            assert shifted.is_contiguous() and shifted.data_ptr() % 16 != 0
            tensor = shifted
        shifted_inputs.append(tensor)
    q, do = shifted_inputs
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(
        cu_q,
        cu_kv,
        q.shape[0],
        k.shape[0],
        max(q_lengths),
        max(kv_lengths),
    )

    actual = amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, scale)

    for name, result, reference in zip(("dq", "dk", "dv"), actual, expected, strict=True):
        assert torch.isfinite(result).all(), name
        relative_l2 = torch.linalg.vector_norm(result.float() - reference.float()) / torch.linalg.vector_norm(
            reference.float())
        assert relative_l2.item() < 1e-2, (name, relative_l2.item())


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_causal_mha_boundaries_gfx950():
    lengths = [1, 15, 16, 17, 127, 128, 129, 255, 256, 257]
    case = _make_varlen_d128_reference_case(
        lengths,
        lengths,
        q_heads=4,
        kv_heads=4,
        seed=541,
        causal=True,
    )
    q, k, v, out, do, lse, cu_q, cu_kv, scale, expected = case
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv)

    actual = amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, scale, causal=True)

    for name, result, reference in zip(("dq", "dk", "dv"), actual, expected, strict=True):
        assert torch.isfinite(result).all(), name
        relative_l2 = torch.linalg.vector_norm(result.float() - reference.float()) / torch.linalg.vector_norm(
            reference.float())
        assert relative_l2.item() < 1e-2, (name, relative_l2.item())


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_causal_accepts_tritonbench_strided_v_gfx950():
    lengths = [17, 129, 257]
    case = _make_varlen_d128_reference_case(
        lengths,
        lengths,
        q_heads=4,
        kv_heads=4,
        seed=547,
        causal=True,
        strided_v=True,
    )
    q, k, v, out, do, lse, cu_q, cu_kv, scale, expected = case
    assert v.stride() == (3 * 4 * 128, 128, 1)
    assert not v.is_contiguous()
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv)

    actual = amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, scale, causal=True)

    assert actual[2].is_contiguous()
    for name, result, reference in zip(("dq", "dk", "dv"), actual, expected, strict=True):
        assert torch.isfinite(result).all(), name
        relative_l2 = torch.linalg.vector_norm(result.float() - reference.float()) / torch.linalg.vector_norm(
            reference.float())
        assert relative_l2.item() < 1e-2, (name, relative_l2.item())


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_causal_strided_v_rebases_high_token_offsets_gfx950():
    prefix_length = 912
    prefix_sequences = 767
    checked_length = 17
    checked_start = prefix_length * prefix_sequences
    total = checked_start + checked_length
    heads = 4
    dim = 128
    scale = dim**-0.5
    assert checked_start * (3 * heads * dim) > 2**30

    q = torch.zeros((total, heads, dim), dtype=torch.bfloat16, device="cuda")
    k = torch.zeros_like(q)
    v_storage = torch.zeros((total, 3, heads, dim), dtype=torch.bfloat16, device="cuda")
    v = v_storage[:, 0]
    out = torch.zeros_like(q)
    do = torch.zeros_like(q)
    lse = torch.empty((heads, total), dtype=torch.float32, device="cuda")
    prefix_lse = torch.arange(1, prefix_length + 1, dtype=torch.float32, device="cuda").log().repeat(prefix_sequences)
    lse[:, :checked_start] = prefix_lse

    generator = torch.Generator(device="cuda").manual_seed(549)
    checked = slice(checked_start, total)
    q[checked] = torch.randn((checked_length, heads, dim), dtype=torch.bfloat16, device="cuda", generator=generator)
    k[checked] = torch.randn((checked_length, heads, dim), dtype=torch.bfloat16, device="cuda", generator=generator)
    v[checked] = torch.randn((checked_length, heads, dim), dtype=torch.bfloat16, device="cuda", generator=generator)
    do[checked] = torch.randn((checked_length, heads, dim), dtype=torch.bfloat16, device="cuda", generator=generator)

    expected_dq = torch.empty((checked_length, heads, dim), dtype=torch.bfloat16, device="cuda")
    expected_dk = torch.empty_like(expected_dq)
    expected_dv = torch.empty_like(expected_dq)
    causal_mask = torch.ones((checked_length, checked_length), dtype=torch.bool, device="cuda").triu(1)
    for head in range(heads):
        q_tile = q[checked, head].float()
        k_tile = k[checked, head].float()
        v_tile = v[checked, head].float()
        do_tile = do[checked, head].float()
        scores = (q_tile @ k_tile.mT * scale).masked_fill(causal_mask, float("-inf"))
        lse_tile = torch.logsumexp(scores, dim=1)
        p = torch.exp(scores - lse_tile[:, None])
        out_tile = (p @ v_tile).to(torch.bfloat16)
        delta = torch.sum(out_tile.float() * do_tile, dim=1)
        ds = (p * (do_tile @ v_tile.mT - delta[:, None])).to(torch.bfloat16).float()
        out[checked, head] = out_tile
        lse[head, checked] = lse_tile
        expected_dq[:, head] = (ds @ k_tile * scale).to(torch.bfloat16)
        expected_dk[:, head] = (ds.mT @ q_tile * scale).to(torch.bfloat16)
        expected_dv[:, head] = (p.to(torch.bfloat16).float().mT @ do_tile).to(torch.bfloat16)

    lengths = [prefix_length] * prefix_sequences + [checked_length]
    cu = torch.tensor([0, *lengths], dtype=torch.int32, device="cuda").cumsum(0, dtype=torch.int32)
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu, cu)

    actual = amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, scale, causal=True)

    assert v.stride() == (3 * heads * dim, dim, 1)
    assert actual[2].is_contiguous()
    for name, result, reference in zip(
        ("dq", "dk", "dv"),
        (actual[0][checked], actual[1][checked], actual[2][checked]),
        (expected_dq, expected_dk, expected_dv),
            strict=True,
    ):
        assert torch.isfinite(result).all(), name
        relative_l2 = torch.linalg.vector_norm(result.float() - reference.float()) / torch.linalg.vector_norm(
            reference.float())
        assert relative_l2.item() < 1e-2, (name, relative_l2.item())


@pytest.mark.parametrize("sm_scale", [0.0, -0.125], ids=["zero-scale", "negative-scale"])
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_causal_masks_after_score_scaling_gfx950(sm_scale):
    lengths = [17, 129]
    case = _make_varlen_d128_reference_case(
        lengths,
        lengths,
        q_heads=4,
        kv_heads=4,
        seed=551,
        causal=True,
        sm_scale=sm_scale,
    )
    q, k, v, out, do, lse, cu_q, cu_kv, scale, expected = case
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv)

    actual = amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, scale, causal=True)

    for name, result, reference in zip(("dq", "dk", "dv"), actual, expected, strict=True):
        assert torch.isfinite(result).all(), name
        error = torch.linalg.vector_norm(result.float() - reference.float())
        scale_norm = torch.clamp(torch.linalg.vector_norm(reference.float()), min=1.0)
        assert (error / scale_norm).item() < 1e-2, (name, error.item(), scale_norm.item())


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
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(
        cu_q,
        cu_kv,
        q.shape[0],
        k.shape[0],
        max(q_lengths),
        max(kv_lengths),
    )

    actual = amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, scale)

    for name, result, reference in zip(("dq", "dk", "dv"), actual, expected, strict=True):
        assert torch.isfinite(result).all(), name
        relative_l2 = torch.linalg.vector_norm(result.float() - reference.float()) / torch.linalg.vector_norm(
            reference.float())
        assert relative_l2.item() < 1e-2, (name, relative_l2.item())


@pytest.mark.parametrize("register_class", ("default", None, "vgpr", "agpr"), ids=("default", "none", "vgpr", "agpr"))
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_prefix_register_class_tuning_gfx950(monkeypatch, register_class):
    options = {} if register_class == "default" else {"K_PREFIX_REGISTER_CLASS": register_class}
    expected_class = None if register_class == "default" else register_class
    launches = _capture_kernel_with_constexprs(monkeypatch, amd_fa_varlen_bwd, "_varlen_bwd_interleaved_bm32_kernel",
                                               **options)
    case = _make_varlen_d128_reference_case(
        [1, 17, 31, 32, 33, 5460],
        [1, 255, 256, 257, 511, 17],
        q_heads=3,
        kv_heads=1,
        seed=490,
    )
    q, k, v, out, do, lse, cu_q, cu_kv, scale, expected = case
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv)

    actual = amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, scale)
    torch.cuda.synchronize()

    for name, result, reference in zip(("dq", "dk", "dv"), actual, expected, strict=True):
        assert torch.isfinite(result).all(), name
        relative_l2 = torch.linalg.vector_norm(result.float() - reference.float()) / torch.linalg.vector_norm(
            reference.float())
        assert relative_l2.item() < 1e-2, (name, relative_l2.item())

    assert len(launches) == 1
    kwargs, compiled = launches[0]
    assert kwargs["KV_SPLITS"] == 3
    prefix_hints = re.findall(
        r'amdg\.register_resident [^\n]*class "(\w+)" groups (\d+) : tensor<256x32xbf16,',
        compiled.asm["ttir"],
    )
    assert prefix_hints == ([] if expected_class is None else [(expected_class, "4")])


@pytest.mark.parametrize(
    ("q_heads", "kv_heads", "kv_splits", "dq_atomic_fp32", "supply_metadata", "shifted_input"),
    (
        pytest.param(12, 4, 3, False, False, None, id="gqa3-bf16-dq"),
        pytest.param(12, 4, 3, True, False, None, id="gqa3-fp32-dq"),
        pytest.param(64, 8, 4, False, False, None, id="gqa8-bf16-dq"),
        pytest.param(64, 8, 4, True, False, None, id="gqa8-fp32-dq"),
        pytest.param(12, 4, 3, True, True, None, id="gqa3-fp32-metadata"),
        pytest.param(64, 8, 4, True, True, None, id="gqa8-fp32-metadata"),
        pytest.param(12, 4, 3, True, False, "q", id="gqa3-fp32-misaligned-q"),
        pytest.param(12, 4, 3, True, False, "k", id="gqa3-fp32-misaligned-k"),
        pytest.param(12, 4, 3, True, False, "v", id="gqa3-fp32-misaligned-v"),
        pytest.param(12, 4, 3, True, False, "do", id="gqa3-fp32-misaligned-do"),
        pytest.param(12, 4, 3, False, False, "q", id="gqa3-bf16-misaligned-q"),
        pytest.param(12, 4, 3, False, False, "k", id="gqa3-bf16-misaligned-k"),
        pytest.param(12, 4, 3, False, False, "v", id="gqa3-bf16-misaligned-v"),
        pytest.param(12, 4, 3, False, False, "do", id="gqa3-bf16-misaligned-do"),
        pytest.param(12, 4, 3, True, True, "q", id="gqa3-fp32-misaligned-q-metadata"),
    ),
)
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_legacy_prefix_public_backward_gfx950(monkeypatch, q_heads, kv_heads, kv_splits,
                                                         dq_atomic_fp32, supply_metadata, shifted_input):
    q_lengths, kv_lengths = _make_seeded_extend_attention_lengths(batch=19, max_context=12331, seed=42)
    check_reference = dq_atomic_fp32 or shifted_input is not None
    if check_reference:
        # Genuine exact-family offsets: the first three sequences end in
        # 1/15/16 rows after two full Q512 chunks, with KV tails 1/255/full.
        # Owner zero must accumulate a second nonzero chunk. The fourth
        # sequence leaves every query owner except owner zero empty.
        q_lengths = [1025, 1039, 1040, 321, 5662, *([3086] * 13), 1549]
        kv_lengths = [1281, 1279, 1280, 513, 10414, *([6248] * 7), *([6247] * 6), 4711]
    total_q, total_kv = sum(q_lengths), sum(kv_lengths)
    assert len(q_lengths) == len(kv_lengths) == 19
    assert all(0 < q_len <= kv_len for q_len, kv_len in zip(q_lengths, kv_lengths, strict=True))
    assert (total_q, total_kv, max(q_lengths), max(kv_lengths)) == (50754, 100696, 5662, 10414)
    cu_q = torch.tensor([0, *torch.tensor(q_lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    cu_kv = torch.tensor([0, *torch.tensor(kv_lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    # Legacy plans have a host-known wide count. Chunking also supports the
    # device-built plan, whose valid task count is available only on device.
    metadata = (total_q, total_kv, max(q_lengths), max(kv_lengths)) if supply_metadata else ()
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv, *metadata)
    if supply_metadata:
        assert plan.wide_task_count is None
    else:
        assert plan.wide_task_count is not None and plan.wide_task_count > 0

    # Inactive sequences have Q=0, V=O=1 and dO=0: uniform attention with
    # LSE=log(KV length), and all three gradients exactly zero.
    q = torch.zeros((total_q, q_heads, 128), dtype=torch.bfloat16, device="cuda")
    k = torch.ones((total_kv, kv_heads, 128), dtype=torch.bfloat16, device="cuda")
    v = torch.ones_like(k)
    out = torch.ones_like(q)
    do = torch.zeros_like(q)
    lse = torch.empty((q_heads, total_q), dtype=torch.float32, device="cuda")
    q_start = 0
    for q_length, kv_length in zip(q_lengths, kv_lengths, strict=True):
        lse[:, q_start:q_start + q_length].fill_(kv_length)
        q_start += q_length
    lse.log_()
    expected = None
    if check_reference:
        # Only four sequences need a dense independent oracle. Their inputs
        # and gradients are nonzero for every head; all other outputs are
        # checked against the analytical zero result, without dense matrices.
        checked_q_lengths, checked_kv_lengths = q_lengths[:4], kv_lengths[:4]
        checked = _make_varlen_d128_reference_case(
            checked_q_lengths, checked_kv_lengths, q_heads=q_heads, kv_heads=kv_heads, seed=2473)
        checked_q, checked_kv = sum(checked_q_lengths), sum(checked_kv_lengths)
        for full, small, count in zip((q, k, v, out, do), checked[:5],
                                      (checked_q, checked_kv, checked_kv, checked_q, checked_q), strict=True):
            full[:count].copy_(small)
        lse[:, :checked_q].copy_(checked[5])
        expected = checked[-1]
        del checked
    if shifted_input is not None:
        inputs = {"q": q, "k": k, "v": v, "do": do}
        original = inputs[shifted_input]
        storage = torch.empty(original.numel() + 1, dtype=original.dtype, device=original.device)
        shifted = storage[1:].view_as(original)
        shifted.copy_(original)
        assert shifted.is_contiguous() and shifted.data_ptr() % 16 != 0
        inputs[shifted_input] = shifted
        q, k, v, do = (inputs[name] for name in ("q", "k", "v", "do"))
    else:
        assert all(tensor.data_ptr() % 16 == 0 for tensor in (q, k, v, do))
    chunked = dq_atomic_fp32 and shifted_input is None
    expected_query_splits = 2 if chunked else 1
    allocate_partials = amd_fa_varlen_bwd._allocate_varlen_dkdv_partials
    partial_workspaces = []

    def capture_partials(k, splits):
        buffers = allocate_partials(k, splits)
        if chunked:
            # Missing stores, including empty owners, must not inherit zeros.
            for buffer in buffers:
                if buffer is not None:
                    buffer.fill_(float("nan"))
        partial_workspaces.append((splits, *buffers))
        return buffers

    monkeypatch.setattr(amd_fa_varlen_bwd, "_allocate_varlen_dkdv_partials", capture_partials)
    legacy_preprocess_launches = _capture_kernel_with_constexprs(
        monkeypatch, amd_fa_varlen_bwd, "_varlen_bwd_preprocess_dynamic_owner_queue")
    preprocess_launches = _capture_kernel_with_constexprs(
        monkeypatch, amd_fa_varlen_bwd, "_varlen_bwd_preprocess")
    rolling_launches = {
        splits: _capture_kernel_with_constexprs(
            monkeypatch, amd_fa_varlen_bwd, f"_varlen_bwd_interleaved_bm32_rolling_fp32_queue_s{splits}")
        for splits in (3, 4)
    }
    shared_launches = _capture_kernel_with_constexprs(
        monkeypatch, amd_fa_varlen_bwd, "_varlen_bwd_interleaved_bm16_bn256_fp32_kernel")
    convert_launches = _capture_kernel_with_constexprs(
        monkeypatch, amd_fa_varlen_bwd, "_varlen_mha_dq_convert_coalesced_fp32_swizzled_kernel")
    fallback_launches = _capture_kernel_with_constexprs(
        monkeypatch, amd_fa_varlen_bwd, "_varlen_bwd_interleaved_kernel")
    generic_launches = _capture_kernel_with_constexprs(
        monkeypatch, amd_fa_varlen_bwd, "_varlen_bwd_interleaved_bm32_kernel")
    reduce_launches = _capture_kernel_with_constexprs(
        monkeypatch, amd_fa_varlen_bwd, "_varlen_dkdv_reduce_kernel")
    actual = amd_fa_varlen_bwd.fa_varlen_backward(
        q, k, v, out, do, lse, plan, 128**-0.5, dq_atomic_fp32=dq_atomic_fp32)
    torch.cuda.synchronize()

    assert not generic_launches
    if shifted_input is None:
        assert not fallback_launches
    if chunked:
        assert len(partial_workspaces) == 1
        splits, dk_partial, dv_partial = partial_workspaces[0]
        assert splits == expected_query_splits
        for buffer in (dk_partial, dv_partial):
            assert buffer is not None and buffer.dtype is torch.float32
            assert buffer.shape == (total_kv, kv_heads, expected_query_splits, 128)
        assert len(preprocess_launches) == len(shared_launches) == len(convert_launches) == 1
        assert len(reduce_launches) == 1
        reduce_kwargs, _ = reduce_launches[0]
        assert reduce_kwargs["KV_SPLITS"] == expected_query_splits
        # NaN poisoning makes these exact-zero checks require an explicit
        # store for every empty query owner, including each sequence's KV tail.
        kv_start = 0
        for q_length, kv_length in zip(checked_q_lengths, checked_kv_lengths, strict=True):
            query_chunks = (q_length + 511) // 512
            if query_chunks < expected_query_splits:
                for buffer in (dk_partial, dv_partial):
                    empty = buffer[kv_start:kv_start + kv_length, :, query_chunks:]
                    assert torch.count_nonzero(empty).item() == 0, (q_length, query_chunks)
            kv_start += kv_length
        assert not legacy_preprocess_launches
        assert all(not launches for launches in rolling_launches.values())
        preprocess_kwargs, _ = preprocess_launches[0]
        core_kwargs, compiled = shared_launches[0]
        convert_kwargs, _ = convert_launches[0]
        assert preprocess_kwargs["ZERO_DQ"] is True
        assert preprocess_kwargs["PACK_STATS_MHA16"] is True and preprocess_kwargs["PACK_STATS"] is False
        assert preprocess_kwargs["DQ_PAD_ROWS"] == convert_kwargs["DQ_PAD_ROWS"] == 16
        assert core_kwargs["CHUNKED_Q"] is True
        assert core_kwargs["Q_SPLITS"] == expected_query_splits
        assert (core_kwargs["HQ"], core_kwargs["HKV"], core_kwargs["BLOCK_M"], core_kwargs["BLOCK_N"]) == (
            q_heads, kv_heads, 16, 256)
        assert "buffer_atomic_add_f32" in compiled.asm["amdgcn"]
        assert "buffer_atomic_pk_add_bf16" not in compiled.asm["amdgcn"]
    elif shifted_input is not None:
        assert len(partial_workspaces) == 1
        splits, dk_partial, dv_partial = partial_workspaces[0]
        assert splits == 1 and dk_partial is None and dv_partial is None
        assert len(preprocess_launches) == 1 and len(fallback_launches) == 2
        assert not legacy_preprocess_launches and not shared_launches and not convert_launches and not reduce_launches
        assert all(not launches for launches in rolling_launches.values())
        preprocess_kwargs, _ = preprocess_launches[0]
        assert preprocess_kwargs["ZERO_DQ"] is False and preprocess_kwargs["PACK_STATS"] is False
        assert preprocess_kwargs["DQ_PAD_ROWS"] == 16
        assert {kwargs["FULL_KV_TILE"] for kwargs, _ in fallback_launches} == {False, True}
        for core_kwargs, compiled in fallback_launches:
            assert (core_kwargs["KV_SPLITS"], core_kwargs["BLOCK_M"], core_kwargs["BLOCK_N"]) == (1, 16, 128)
            assert core_kwargs["QDO_ALIGNED"] is False and core_kwargs["CACHE_MHA_STATS"] is False
            atomic = "buffer_atomic_add_f32" if dq_atomic_fp32 else "buffer_atomic_pk_add_bf16"
            assert atomic in compiled.asm["amdgcn"]
    else:
        assert len(legacy_preprocess_launches) == len(rolling_launches[kv_splits]) == 1
        assert all(not launches for splits, launches in rolling_launches.items() if splits != kv_splits)
        assert not preprocess_launches and not shared_launches and not convert_launches and not reduce_launches
        preprocess_kwargs, _ = legacy_preprocess_launches[0]
        assert preprocess_kwargs["ZERO_DQ"] is True and preprocess_kwargs["PACK_STATS"] is True
        core_kwargs, compiled = rolling_launches[kv_splits][0]
        assert core_kwargs["KV_SPLITS"] == kv_splits
        atomic = "buffer_atomic_add_f32" if dq_atomic_fp32 else "buffer_atomic_pk_add_bf16"
        assert atomic in compiled.asm["amdgcn"]
    for index, (name, result, source) in enumerate(zip(("dq", "dk", "dv"), actual, (q, k, v), strict=True)):
        assert result.shape == source.shape and result.device == source.device, name
        assert result.dtype is torch.bfloat16, name
        if check_reference:
            reference = expected[index]
            count = reference.shape[0]
            assert torch.count_nonzero(reference).item() > 0, name
            assert torch.isfinite(result[:count]).all(), name
            relative_l2 = torch.linalg.vector_norm(result[:count].float() - reference.float()) / torch.linalg.vector_norm(
                reference.float())
            assert relative_l2.item() < 1e-2, (name, relative_l2.item())
            assert torch.count_nonzero(result[count:]).item() == 0, name
        else:
            assert torch.count_nonzero(result).item() == 0, name
    if check_reference:
        # Check each final chunk independently, so the two preceding Q512
        # chunks cannot conceal a missing or incorrectly rebased tail.
        q_start = 0
        for q_length in checked_q_lengths[:3]:
            tail = slice(q_start + 1024, q_start + q_length)
            reference = expected[0][tail].float()
            relative_l2 = torch.linalg.vector_norm(actual[0][tail].float() - reference) / torch.linalg.vector_norm(reference)
            assert relative_l2.item() < 1e-2, (q_length - 1024, relative_l2.item())
            q_start += q_length
        # Check the short sequence independently; aggregate error from the
        # three longer sequences must not conceal its missing contribution.
        short_q = slice(sum(checked_q_lengths[:3]), sum(checked_q_lengths))
        short_kv = slice(sum(checked_kv_lengths[:3]), sum(checked_kv_lengths))
        for name, result, reference, rows in zip(
            ("dq", "dk", "dv"), actual, expected, (short_q, short_kv, short_kv), strict=True,
        ):
            reference = reference[rows].float()
            assert torch.count_nonzero(reference).item() > 0, name
            relative_l2 = torch.linalg.vector_norm(result[rows].float() - reference) / torch.linalg.vector_norm(reference)
            assert relative_l2.item() < 1e-2, (name, "short-query", relative_l2.item())


@pytest.mark.parametrize("metadata", ("legacy", "missing_sequence", "missing_start", "default_k96"))
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_tail_finalizer_plan_defaults_gfx950(metadata):
    case = _make_varlen_d128_reference_case([1, 17, 33], [1, 225, 256], q_heads=4, kv_heads=4, seed=1901)
    q, k, v, out, do, lse, cu_q, cu_kv, scale, expected = case
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv)
    if metadata == "legacy":
        fields = {
            key: value
            for key, value in vars(plan).items()
            if key not in ("dq_full_kv_sequence", "dq_full_kv_start", "dq_tail_k96")
        }
        plan = amd_fa_varlen_bwd.VarlenBackwardPlan(**fields)
        assert plan.dq_full_kv_sequence is plan.dq_full_kv_start is None
    elif metadata == "missing_sequence":
        plan = replace(plan, dq_full_kv_sequence=None)
    elif metadata == "missing_start":
        plan = replace(plan, dq_full_kv_start=None)
    else:
        plan = amd_fa_varlen_bwd.VarlenBackwardPlan(
            **{key: value
               for key, value in vars(plan).items()
               if key != "dq_tail_k96"})
    assert plan.dq_tail_k96 is False

    actual = amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, scale)
    for name, result, reference in zip(("dq", "dk", "dv"), actual, expected, strict=True):
        assert torch.isfinite(result).all(), name
        relative_l2 = torch.linalg.vector_norm(result.float() - reference.float()) / torch.linalg.vector_norm(
            reference.float())
        assert relative_l2.item() < 1e-2, (name, relative_l2.item())


@pytest.mark.parametrize("kv_length", (128, 224))
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_runtime_totals_reuse_specialization_gfx950(kv_length):
    kernels = (
        amd_fa_varlen_bwd._varlen_bwd_interleaved_kernel,
        amd_fa_varlen_bwd._varlen_mha_dq_convert_coalesced_kernel,
    )
    for kernel in kernels:
        kernel.device_caches.clear()

    for q_length in (17, 33):
        case = _make_varlen_d128_reference_case([q_length], [kv_length], q_heads=1, kv_heads=1, seed=419 + q_length)
        q, k, v, out, do, lse, cu_q, cu_kv, scale, _expected = case
        plan = amd_fa_varlen_bwd.prepare_varlen_backward(
            cu_q,
            cu_kv,
            q.shape[0],
            k.shape[0],
            q_length,
            kv_length,
        )

        amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, scale)
        torch.cuda.synchronize()

    device = torch.cuda.current_device()
    # Uniform full-tile metadata proves the tail schedule is empty, so only
    # the full specialization is compiled for KV128.
    expected_counts = (1, 1) if kv_length == 128 else (2, 1)
    for kernel, expected in zip(kernels, expected_counts, strict=True):
        assert len(kernel.device_caches[device][0]) == expected, kernel.fn.__name__


@pytest.mark.parametrize(
    ("q_heads", "kv_heads", "long_q"),
    (
        pytest.param(1, 1, None, id="1"),
        pytest.param(4, 4, None, id="4"),
        pytest.param(3, 1, 5460, id="gqa3"),
        pytest.param(8, 1, 2048, id="gqa8"),
    ),
)
@pytest.mark.parametrize("tail_bound", (96, 127))
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_scratch_reset_graph_replay_gfx950(monkeypatch, q_heads, kv_heads, long_q, tail_bound):
    q_lengths = [1, 15, 16, 17, 31, 32, 33, 63, 64, 65]
    kv_lengths = [1, tail_bound, 128, 129, 128 + tail_bound, 256, 257, 33, 7, 384 + tail_bound]
    if long_q is not None:
        # Trigger split GQA while keeping the dense reference small.
        q_lengths.append(long_q)
        kv_lengths.append(17)
    case = _make_varlen_d128_reference_case(
        q_lengths,
        kv_lengths,
        q_heads=q_heads,
        kv_heads=kv_heads,
        seed=557 + q_heads,
    )
    q, k, v, out, do, lse, cu_q, cu_kv, scale, expected = case
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv)
    preprocess = amd_fa_varlen_bwd._varlen_bwd_preprocess

    class PoisonedPreprocess:

        def __getitem__(self, grid):

            def launch(o, do, delta, cu_q, dq_acc, total_q_padded, task_counts, **kwargs):
                # Valid dQ columns use the whole swizzled BM16 footprint,
                # including rows beyond the final logical query row.
                assert kwargs["ZERO_DQ"]
                dq_acc.fill_(float("nan"))
                return preprocess[grid](o, do, delta, cu_q, dq_acc, total_q_padded, task_counts, **kwargs)

            return launch

    monkeypatch.setattr(amd_fa_varlen_bwd, "_varlen_bwd_preprocess", PoisonedPreprocess())

    def backward():
        return amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, scale)

    def check(actual):
        for name, result, reference in zip(("dq", "dk", "dv"), actual, expected, strict=True):
            assert torch.isfinite(result).all(), name
            relative_l2 = torch.linalg.vector_norm(result.float() - reference.float()) / torch.linalg.vector_norm(
                reference.float())
            assert relative_l2.item() < 1e-2, (name, relative_l2.item())

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            actual = backward()
    torch.cuda.current_stream().wait_stream(stream)
    check(actual)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = backward()
    for _ in range(3):
        for result in actual:
            result.fill_(float("nan"))
        graph.replay()
        check(actual)


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
        plan = amd_fa_varlen_bwd.prepare_varlen_backward(
            cu_q,
            cu_kv,
            q.shape[0],
            k.shape[0],
            q_length,
            17,
        )
        amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, scale)
        torch.cuda.synchronize()

    device = torch.cuda.current_device()
    assert len(kernel.device_caches[device][0]) == 1


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_plan_rejects_invalid_offset_metadata():
    valid = torch.tensor([0, 16, 48], dtype=torch.int32, device="cuda")
    strided = torch.tensor(
        [0, -1, 16, -1, 48],
        dtype=torch.int32,
        device="cuda",
    )[::2]
    cases = (
        (valid.view(1, 3), valid, "rank-1 tensor"),
        (valid.to(torch.int64), valid, "dtype torch.int32"),
        (valid, torch.tensor([0, 32], dtype=torch.int32, device="cuda"), "same batch"),
        (
            strided,
            valid,
            "must be contiguous when token metadata is supplied",
        ),
    )
    for cu_q, cu_kv, message in cases:
        with pytest.raises(ValueError, match=message):
            amd_fa_varlen_bwd.prepare_varlen_backward(
                cu_q,
                cu_kv,
                48,
                48,
                32,
                32,
            )


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_legacy_plan_accepts_strided_offsets():
    cu_q = torch.tensor(
        [0, -1, 17, -1, 48],
        dtype=torch.int32,
        device="cuda",
    )[::2]
    cu_kv = torch.tensor(
        [0, -1, 33, -1, 162],
        dtype=torch.int32,
        device="cuda",
    )[::2]

    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv)

    assert plan.cu_seqlens_q.is_contiguous()
    assert plan.cu_seqlens_k.is_contiguous()
    assert plan.cu_seqlens_q.tolist() == [0, 17, 48]
    assert plan.cu_seqlens_k.tolist() == [0, 33, 162]

    cu_q.zero_()
    cu_kv.zero_()
    assert plan.cu_seqlens_q.tolist() == [0, 17, 48]
    assert plan.cu_seqlens_k.tolist() == [0, 33, 162]
    amd_fa_varlen_bwd.validate_varlen_backward_plan(plan)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize(
    ("q_offsets", "kv_offsets", "total_q", "total_kv", "max_q", "max_kv"),
    (
        pytest.param([1, 17, 49], [0, 16, 48], 48, 48, 32, 32, id="nonzero-start"),
        pytest.param([0, 16, 16], [0, 16, 48], 16, 48, 16, 32, id="empty-sequence"),
        pytest.param([0, 17, 16], [0, 16, 48], 16, 48, 17, 32, id="nonmonotonic"),
        pytest.param([0, 16, 48], [0, 16, 48], 47, 48, 32, 32, id="wrong-q-total"),
        pytest.param([0, 16], [0, 256], 16, 128, 16, 256, id="undersized-kv-total"),
        pytest.param(
            [0, 16, 32, 48, 64],
            [0, 128, 0, 128, 0],
            64,
            128,
            16,
            128,
            id="alternating-kv-offsets",
        ),
        pytest.param([0, 16, 48], [0, 16, 48], 48, 48, 31, 32, id="wrong-max"),
    ),
)
def test_varlen_d128_device_validation_rejects_invalid_values(q_offsets, kv_offsets, total_q, total_kv, max_q, max_kv):
    cu_q = torch.tensor(q_offsets, dtype=torch.int32, device="cuda")
    cu_kv = torch.tensor(kv_offsets, dtype=torch.int32, device="cuda")

    plan = amd_fa_varlen_bwd.prepare_varlen_backward(
        cu_q,
        cu_kv,
        total_q,
        total_kv,
        max_q,
        max_kv,
    )

    counts = plan.task_counts.tolist()
    assert counts[-2:] == [1, 1]
    assert counts[:-2] == [0, 0, 0, 0]
    with pytest.raises(ValueError, match="cu_seqlens must start at zero"):
        amd_fa_varlen_bwd.validate_varlen_backward_plan(plan)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_backward_rejects_unsupported_signature():
    case = _make_varlen_d128_reference_case([16], [128], q_heads=1, kv_heads=1, seed=407)
    q, k, v, out, do, lse, cu_q, cu_kv, scale, _expected = case
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv, q.shape[0], k.shape[0], 16, 128)

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
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv, q.shape[0], k.shape[0], 16, 128)
    invalid_k = torch.empty((k.shape[0], 3, 128), dtype=k.dtype, device=k.device)
    invalid_v = torch.empty_like(invalid_k)

    with pytest.raises(ValueError, match="divisible"):
        amd_fa_varlen_bwd.fa_varlen_backward(q, invalid_k, invalid_v, out, do, lse, plan, scale)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_causal_rejects_cross_attention_offsets():
    case = _make_varlen_d128_reference_case([16], [128], q_heads=1, kv_heads=1, seed=523)
    q, k, v, out, do, lse, cu_q, cu_kv, scale, _expected = case
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv)

    with pytest.raises(ValueError, match="identical Q and KV cumulative offsets"):
        amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, scale, causal=True)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_causal_rejects_gqa():
    case = _make_varlen_d128_reference_case([17], [17], q_heads=2, kv_heads=1, seed=527)
    q, k, v, out, do, lse, cu_q, cu_kv, scale, _expected = case
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv)

    with pytest.raises(ValueError, match="equal Q and KV head counts"):
        amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, scale, causal=True)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_causal_rejects_v_with_nondense_head_axes():
    case = _make_varlen_d128_reference_case([17], [17], q_heads=2, kv_heads=2, seed=531, causal=True)
    q, k, _v, out, do, lse, cu_q, cu_kv, scale, _expected = case
    storage = torch.empty((17, 2, 128, 2), dtype=torch.bfloat16, device="cuda")
    v = storage[..., 0]
    assert v.shape == k.shape
    assert v.stride(-1) == 2
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv)

    with pytest.raises(ValueError, match="dense head/D axes"):
        amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, scale, causal=True)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_noncausal_still_rejects_strided_v():
    case = _make_varlen_d128_reference_case(
        [17],
        [17],
        q_heads=2,
        kv_heads=2,
        seed=533,
        strided_v=True,
    )
    q, k, v, out, do, lse, cu_q, cu_kv, scale, _expected = case
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv)

    with pytest.raises(ValueError, match="v must be contiguous bfloat16 THD"):
        amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, scale)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_backward_rejects_noncontiguous_lse():
    case = _make_varlen_d128_reference_case([16], [128], q_heads=2, kv_heads=2, seed=413)
    q, k, v, out, do, lse, cu_q, cu_kv, scale, _expected = case
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv, q.shape[0], k.shape[0], 16, 128)
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
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv, q.shape[0], k.shape[0], 17, 129)
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
        ttir = interleaved.asm["ttir"]
        assert "amdg.rematerialized_range 0 to 128 identity 32" not in ttir
        assert "amdg.rematerialized_range 0 to 16 identity 33" not in ttir
        assert "arith.cmpi sle" not in ttir
        assert "buffer_atomic_pk_add_bf16" in interleaved.asm["amdgcn"]


@pytest.mark.parametrize(
    ("q_lengths", "kv_lengths", "q_heads", "kv_heads", "causal", "wide_split"),
    (
        pytest.param([17], [129], 2, 2, False, False, id="noncausal-mha"),
        pytest.param([33], [257], 3, 1, False, False, id="noncausal-gqa"),
        pytest.param([5460], [17], 3, 1, False, True, id="noncausal-gqa3-split"),
        pytest.param([17], [17], 2, 2, True, False, id="causal-mha"),
    ),
)
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_fp32_dq_atomics_generic_paths_gfx950(
    q_lengths,
    kv_lengths,
    q_heads,
    kv_heads,
    causal,
    wide_split,
):
    generic_core = amd_fa_varlen_bwd._varlen_bwd_interleaved_kernel
    wide_core = amd_fa_varlen_bwd._varlen_bwd_interleaved_bm32_kernel
    exact_core = amd_fa_varlen_bwd._varlen_bwd_interleaved_bm16_bn256_fp32_kernel
    generic_core.device_caches.clear()
    wide_core.device_caches.clear()
    exact_core.device_caches.clear()

    case = _make_varlen_d128_reference_case(
        q_lengths,
        kv_lengths,
        q_heads=q_heads,
        kv_heads=kv_heads,
        seed=2399 + q_heads,
        causal=causal,
    )
    q, k, v, out, do, lse, cu_q, cu_kv, scale, expected = case
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(
        cu_q,
        cu_kv,
        q.shape[0],
        k.shape[0],
        max(q_lengths),
        max(kv_lengths),
    )

    actual = amd_fa_varlen_bwd.fa_varlen_backward(
        q,
        k,
        v,
        out,
        do,
        lse,
        plan,
        scale,
        causal=causal,
        dq_atomic_fp32=True,
    )
    torch.cuda.synchronize()

    for name, result, reference in zip(("dq", "dk", "dv"), actual, expected, strict=True):
        assert result.dtype is torch.bfloat16, name
        assert torch.isfinite(result).all(), name
        relative_l2 = torch.linalg.vector_norm(result.float() - reference.float()) / torch.linalg.vector_norm(
            reference.float())
        assert relative_l2.item() < 1e-2, (name, relative_l2.item())

    device = torch.cuda.current_device()
    selected_core = wide_core if wide_split else generic_core
    unused_core = generic_core if wide_split else wide_core
    compiled_objects = tuple(selected_core.device_caches[device][0].values())
    assert compiled_objects
    assert not unused_core.device_caches.get(device)
    assert not exact_core.device_caches.get(device)
    atomic_assembly = "\n".join(compiled.asm["amdgcn"] for compiled in compiled_objects)
    assert "buffer_atomic_add_f32" in atomic_assembly
    assert "buffer_atomic_pk_add_bf16" not in atomic_assembly


@pytest.mark.parametrize(
    ("max_q", "q_heads", "kv_heads", "sm_scale"),
    (
        pytest.param(300, 4, 4, None, id="300"),
        pytest.param(400, 4, 4, None, id="400"),
        pytest.param(300, 12, 4, None, id="gqa3-q300"),
        pytest.param(400, 12, 4, None, id="gqa3-q400"),
        pytest.param(300, 12, 4, 0.125, id="gqa3-positive-scale"),
        pytest.param(400, 12, 4, -0.125, id="gqa3-negative-scale"),
    ),
)
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_fp32_dq_atomics_gfx950(monkeypatch, max_q, q_heads, kv_heads, sm_scale):
    kernels = (
        amd_fa_varlen_bwd._varlen_bwd_preprocess,
        amd_fa_varlen_bwd._varlen_bwd_interleaved_bm16_bn256_fp32_kernel,
        amd_fa_varlen_bwd._varlen_mha_dq_convert_coalesced_fp32_swizzled_kernel,
        amd_fa_varlen_bwd._varlen_bwd_interleaved_kernel,
        amd_fa_varlen_bwd._varlen_mha_dq_convert_coalesced_kernel,
    )
    for kernel in kernels:
        kernel.device_caches.clear()

    preprocess_launches = _capture_kernel_with_constexprs(
        monkeypatch, amd_fa_varlen_bwd, "_varlen_bwd_preprocess")
    exact_launches = _capture_kernel_with_constexprs(
        monkeypatch, amd_fa_varlen_bwd, "_varlen_bwd_interleaved_bm16_bn256_fp32_kernel")
    exact_convert_launches = _capture_kernel_with_constexprs(
        monkeypatch, amd_fa_varlen_bwd, "_varlen_mha_dq_convert_coalesced_fp32_swizzled_kernel")
    generic_launches = _capture_kernel_with_constexprs(
        monkeypatch, amd_fa_varlen_bwd, "_varlen_bwd_interleaved_kernel")
    generic_convert_launches = _capture_kernel_with_constexprs(
        monkeypatch, amd_fa_varlen_bwd, "_varlen_mha_dq_convert_coalesced_kernel")

    # Match the public route's batch/maxima while keeping the reference small.
    # Exercise short/partial BM16 tiles, both sides of Q256, and KV tails.
    # Q400 also exercises aligned runtime totals; Q300 keeps unaligned totals.
    last_q = 10 if max_q == 400 else 1
    q_lengths = [max_q, 1, 15, 16, 17, 255, 256, 257, max_q - 1, *([1] * 758), last_q]
    kv_lengths = [3200, 1, 127, 128, 129, 255, 256, 257, 257, *([1] * 759)]
    case = _make_varlen_d128_reference_case(
        q_lengths, kv_lengths, q_heads=q_heads, kv_heads=kv_heads, seed=2417, sm_scale=sm_scale)
    q, k, v, out, do, lse, cu_q, cu_kv, scale, expected = case
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(
        cu_q,
        cu_kv,
        q.shape[0],
        k.shape[0],
        max(q_lengths),
        max(kv_lengths),
    )
    assert plan.batch == len(q_lengths) == len(kv_lengths) == 768
    assert (plan.total_q, plan.total_kv, plan.max_q, plan.max_kv) == (
        sum(q_lengths), sum(kv_lengths), max_q, 3200)
    assert all(tensor.data_ptr() % 16 == 0 for tensor in (q, k, v, do))
    actual = amd_fa_varlen_bwd.fa_varlen_backward(
        q,
        k,
        v,
        out,
        do,
        lse,
        plan,
        scale,
        dq_atomic_fp32=True,
    )
    torch.cuda.synchronize()

    for name, result, reference in zip(("dq", "dk", "dv"), actual, expected, strict=True):
        assert result.dtype is torch.bfloat16, name
        assert torch.isfinite(result).all(), name
        relative_l2 = torch.linalg.vector_norm(result.float() - reference.float()) / torch.linalg.vector_norm(
            reference.float())
        assert relative_l2.item() < 1e-2, (name, relative_l2.item())

    assert len(preprocess_launches) == len(exact_launches) == len(exact_convert_launches) == 1
    assert generic_launches == generic_convert_launches == []
    wide_task_count = plan.task_counts[amd_fa_varlen_bwd._WIDE_KV_TASK_COUNT.value].item()
    assert wide_task_count == 782
    wide_q_starts = plan.wide_q_start[:wide_task_count].tolist()
    assert wide_q_starts.count(0) == 13
    assert wide_q_starts.count(max_q) == 1
    preprocess_kwargs, _ = preprocess_launches[0]
    exact_kwargs, compiled = exact_launches[0]
    assert preprocess_kwargs["PACK_STATS_MHA16"] is True
    assert preprocess_kwargs["PACK_STATS"] is False
    assert exact_kwargs["HQ"] == q_heads and exact_kwargs["HKV"] == kv_heads
    assert exact_kwargs["BLOCK_M"] == 16 and exact_kwargs["BLOCK_N"] == 256
    assert exact_kwargs["reverse_local_assignment"] is True
    assert exact_kwargs["enable_sched_group_barrier_scheduler"] is False
    assert exact_kwargs["llvm_fn_attrs"] == (("amdgpu-sched-strategy", "max-ilp"), )
    assert not ({
        "sink_insts_to_avoid_spills",
        "regclass_priority_trumps_globalness",
        "disable_unclustered_high_rp_reschedule",
    } & exact_kwargs.keys())
    assembly = compiled.asm["amdgcn"]
    assert "buffer_atomic_add_f32" in assembly
    assert "buffer_atomic_pk_add_bf16" not in assembly
    if q_heads == kv_heads:
        _assert_fp32_dq_wave8_pipeline(compiled)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_fp32_dq_wave8_state_graph_replay_gfx950(monkeypatch):
    q_lengths = [400, 17, *([1] * 766)]
    kv_lengths = [3200, 257, *([1] * 766)]
    original = _make_varlen_d128_reference_case(q_lengths, kv_lengths, q_heads=4, kv_heads=4, seed=2431)
    changed = _make_varlen_d128_reference_case(q_lengths, kv_lengths, q_heads=4, kv_heads=4, seed=2437)
    q, k, v, out, do, lse, cu_q, cu_kv, scale, expected = original
    original_inputs = tuple(tensor.clone() for tensor in original[:6])
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(
        cu_q, cu_kv, q.shape[0], k.shape[0], max(q_lengths), max(kv_lengths))
    preprocess = amd_fa_varlen_bwd._varlen_bwd_preprocess
    exact_launches = _capture_kernel_with_constexprs(
        monkeypatch, amd_fa_varlen_bwd, "_varlen_bwd_interleaved_bm16_bn256_fp32_kernel")

    class PoisonedPreprocess:

        def __getitem__(self, grid):

            def launch(o, do, delta, cu_q, dq_acc, total_q_padded, task_counts, **kwargs):
                assert kwargs["ZERO_DQ"] and kwargs["PACK_STATS_MHA16"]
                assert delta.dtype is dq_acc.dtype is torch.float32
                # Graph replay reuses these allocations. Both the first call
                # and every replay must initialize statistics and dQ scratch.
                delta.fill_(float("nan"))
                dq_acc.fill_(float("nan"))
                return preprocess[grid](o, do, delta, cu_q, dq_acc, total_q_padded, task_counts, **kwargs)

            return launch

    monkeypatch.setattr(amd_fa_varlen_bwd, "_varlen_bwd_preprocess", PoisonedPreprocess())

    def backward():
        return amd_fa_varlen_bwd.fa_varlen_backward(
            q, k, v, out, do, lse, plan, scale, dq_atomic_fp32=True)

    def check(actual, reference):
        for name, result, target in zip(("dq", "dk", "dv"), actual, reference, strict=True):
            assert result.dtype is torch.bfloat16, name
            assert torch.isfinite(result).all(), name
            relative_l2 = torch.linalg.vector_norm(result.float() - target.float()) / torch.linalg.vector_norm(
                target.float())
            assert relative_l2.item() < 1e-2, (name, relative_l2.item())

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        first = backward()
    torch.cuda.current_stream().wait_stream(stream)
    check(first, expected)
    assert len(exact_launches) == 1
    _assert_fp32_dq_wave8_pipeline(exact_launches[0][1])
    with torch.cuda.stream(stream):
        repeated = backward()
    torch.cuda.current_stream().wait_stream(stream)
    check(repeated, expected)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = backward()
    output_pointers = tuple(tensor.data_ptr() for tensor in actual)
    # Fresh forward output/LSE accompany the changed Q/K/V/dO. The replayed
    # launch keeps its input pointers, plan, scratch and output allocations.
    for inputs, reference in ((original_inputs, expected), (changed[:6], changed[-1]),
                              (changed[:6], changed[-1]), (original_inputs, expected)):
        for target, replacement in zip(original[:6], inputs, strict=True):
            target.copy_(replacement)
        for result in actual:
            result.fill_(float("nan"))
        graph.replay()
        check(actual, reference)
        assert tuple(tensor.data_ptr() for tensor in actual) == output_pointers


@pytest.mark.parametrize(
    ("max_q", "q_heads", "kv_heads", "dq_atomic_fp32", "shifted_input"),
    (
        pytest.param(401, 4, 4, True, None, id="q-above-cache-domain"),
        pytest.param(300, 4, 4, True, "q", id="misaligned-q"),
        pytest.param(300, 4, 4, True, "k", id="misaligned-k"),
        pytest.param(300, 4, 4, True, "v", id="misaligned-v"),
        pytest.param(300, 4, 4, True, "do", id="misaligned-do"),
        pytest.param(401, 12, 4, True, None, id="gqa3-q-above-cache-domain"),
        pytest.param(300, 12, 4, True, "q", id="gqa3-misaligned-q"),
        pytest.param(300, 12, 4, True, "k", id="gqa3-misaligned-k"),
        pytest.param(300, 12, 4, True, "v", id="gqa3-misaligned-v"),
        pytest.param(300, 12, 4, True, "do", id="gqa3-misaligned-do"),
        pytest.param(300, 12, 4, False, None, id="gqa3-bf16-dq-unchanged"),
    ),
)
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_fp32_dq_wave8_fallback_gfx950(monkeypatch, max_q, q_heads, kv_heads,
                                                dq_atomic_fp32, shifted_input):
    q_lengths = [max_q, 17, *([1] * 766)]
    kv_lengths = [3200, 257, *([1] * 766)]
    case = list(_make_varlen_d128_reference_case(
        q_lengths, kv_lengths, q_heads=q_heads, kv_heads=kv_heads, seed=2441))
    if shifted_input is not None:
        index = {"q": 0, "k": 1, "v": 2, "do": 4}[shifted_input]
        tensor = case[index]
        storage = torch.empty(tensor.numel() + 1, dtype=tensor.dtype, device=tensor.device)
        shifted = storage[1:].view(tensor.shape)
        shifted.copy_(tensor)
        assert shifted.is_contiguous() and shifted.data_ptr() % 16 != 0
        case[index] = shifted
    q, k, v, out, do, lse, cu_q, cu_kv, scale, expected = case
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(
        cu_q, cu_kv, q.shape[0], k.shape[0], max(q_lengths), max(kv_lengths))

    class UnsupportedExactKernel:

        def __getitem__(self, grid):
            # Fail at dispatch, before an unsafe direct-to-LDS launch can run.
            pytest.fail("The FP32 BM16/BN256 core must fall back outside its input domain")

    monkeypatch.setattr(amd_fa_varlen_bwd, "_varlen_bwd_interleaved_bm16_bn256_fp32_kernel",
                        UnsupportedExactKernel())
    generic_launches = _capture_kernel_with_constexprs(
        monkeypatch, amd_fa_varlen_bwd, "_varlen_bwd_interleaved_kernel")
    actual = amd_fa_varlen_bwd.fa_varlen_backward(
        q, k, v, out, do, lse, plan, scale, dq_atomic_fp32=dq_atomic_fp32)
    torch.cuda.synchronize()

    assert generic_launches
    for name, result, reference in zip(("dq", "dk", "dv"), actual, expected, strict=True):
        assert result.dtype is torch.bfloat16, name
        assert torch.isfinite(result).all(), name
        relative_l2 = torch.linalg.vector_norm(result.float() - reference.float()) / torch.linalg.vector_norm(
            reference.float())
        assert relative_l2.item() < 1e-2, (name, relative_l2.item())


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_causal_codegen_is_scratch_free_gfx950():
    kernel = amd_fa_varlen_bwd._varlen_bwd_interleaved_kernel
    kernel.device_caches.clear()

    case = _make_varlen_d128_reference_case(
        [129],
        [129],
        q_heads=4,
        kv_heads=4,
        seed=557,
        causal=True,
        strided_v=True,
    )
    q, k, v, out, do, lse, cu_q, cu_kv, scale, _expected = case
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv)
    amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, scale, causal=True)
    torch.cuda.synchronize()

    device = torch.cuda.current_device()
    compiled_objects = tuple(kernel.device_caches[device][0].values())
    assert len(compiled_objects) == 2
    for compiled in compiled_objects:
        _assert_scratch_free(kernel.fn.__name__, compiled)
        assert compiled.metadata.num_warps == 4
        assert compiled.metadata.shared == 64_640
        ttir = compiled.asm["ttir"]
        assert ttir.count("amdg.rematerialized_range 0 to 128 identity 32") == 1
        assert ttir.count("amdg.rematerialized_range 0 to 16 identity 33") == 1
        assert ttir.count("arith.cmpi sle") == 1
        assert "arith.select" in ttir
        assert re.search(r"tt\.addptr %V, %\w+ : !tt\.ptr<bf16>, i64", ttir)
        assert "buffer_atomic_pk_add_bf16" in compiled.asm["amdgcn"]


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
        plan = amd_fa_varlen_bwd.prepare_varlen_backward(
            cu_q,
            cu_kv,
            q.shape[0],
            k.shape[0],
            q_length,
            129,
        )
        amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, scale)
    torch.cuda.synchronize()

    device = torch.cuda.current_device()
    for kernel, expected_shared, specialization_count in zip(kernels, (118_048, 0), (2, 2), strict=True):
        compiled_objects = tuple(kernel.device_caches[device][0].values())
        assert len(compiled_objects) == specialization_count
        for compiled in compiled_objects:
            _assert_scratch_free(kernel.fn.__name__, compiled)
            assert compiled.metadata.num_warps == 4
            assert compiled.metadata.shared == expected_shared
    for compiled in kernels[0].device_caches[device][0].values():
        assert "buffer_atomic_pk_add_bf16" in compiled.asm["amdgcn"]
