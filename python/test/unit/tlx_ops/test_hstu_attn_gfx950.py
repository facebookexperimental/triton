"""gfx950 ``tlx.ops.hstu_attn_dev`` forward and backward correctness."""
import importlib
from pathlib import Path
import pytest
import torch
from triton._internal_testing import is_hip_cdna4
from triton.tlx.ops.kernels.hstu_attn._shapes import CORRECTNESS_SHAPES, inputs

GFX950_SHAPES = [
    (2, 128, 2, 128, 128),
    (4, 256, 4, 128, 128),
]

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
@pytest.mark.parametrize("Z,MAX_SEQ_LEN,H,HEAD_DIM,causal,dtype_name", CORRECTNESS_SHAPES)
def test_hstu_attn_common_shapes_gfx950(Z, MAX_SEQ_LEN, H, HEAD_DIM, causal, dtype_name):
    from triton.tlx.ops import hstu_attn_dev as tlx_hstu_attn
    from triton.tlx.ops.kernels.hstu_attn._reference import triton_hstu_mha

    dtype = DTYPES[dtype_name]
    q, k, v, offsets, attn_scale = inputs(Z, MAX_SEQ_LEN, H, HEAD_DIM, dtype)
    alpha = 1.0 / HEAD_DIM
    actual = tlx_hstu_attn(q, k, v, offsets, MAX_SEQ_LEN, attn_scale, alpha=alpha, causal=causal, space="smoke")
    expected = triton_hstu_mha(MAX_SEQ_LEN, alpha, q, k, v, offsets, attn_scale)
    precision = 1e-3 if dtype == torch.float16 else 8e-3
    torch.testing.assert_close(actual, expected, atol=precision * expected.abs().max().item(), rtol=precision)


def _gfx950_inputs(batch_size, max_seq_len, H, attn_dim, hidden_dim, dtype):
    device = torch.device("cuda")
    lengths = torch.linspace(max_seq_len // 2, max_seq_len, batch_size, device=device, dtype=torch.int32)
    offsets = torch.zeros((batch_size + 1, ), dtype=torch.int64, device=device)
    offsets[1:] = torch.cumsum(lengths.to(torch.int64), dim=0)
    total = int(offsets[-1].item())
    x = torch.empty((total, H, attn_dim * 2 + hidden_dim), dtype=dtype, device=device).uniform_(-0.01, 0.01)
    q, k, v = torch.split(x, [attn_dim, attn_dim, hidden_dim], dim=-1)
    num_targets = torch.clamp(lengths // 4, min=1)
    return q.contiguous(), k.contiguous(), v.contiguous(), offsets, num_targets


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
@pytest.mark.parametrize("batch_size, MAX_SEQ_LEN, H, ATTN_DIM, HIDDEN_DIM", GFX950_SHAPES)
def test_hstu_attn_gfx950(batch_size, MAX_SEQ_LEN, H, ATTN_DIM, HIDDEN_DIM):
    from triton.tlx.ops import hstu_attn_dev as tlx_hstu_attn
    from triton.tlx.ops.kernels.hstu_attn._reference import triton_hstu_mha

    torch.cuda.empty_cache()
    dtype = torch.bfloat16
    alpha = 10000.0 / ATTN_DIM
    q, k, v, offsets, num_targets = _gfx950_inputs(batch_size, MAX_SEQ_LEN, H, ATTN_DIM, HIDDEN_DIM, dtype)

    out = tlx_hstu_attn(q, k, v, offsets, MAX_SEQ_LEN, None, alpha=alpha, causal=True, num_targets=num_targets,
                        space="smoke")
    ref_attn_scale = torch.tensor(1.0 / MAX_SEQ_LEN, device=q.device, dtype=torch.float32)
    ref = triton_hstu_mha(
        MAX_SEQ_LEN,
        alpha,
        q,
        k,
        v,
        offsets,
        ref_attn_scale,
        num_targets=num_targets,
    )

    torch.testing.assert_close(out * MAX_SEQ_LEN, ref * MAX_SEQ_LEN, atol=1e-3, rtol=0)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
@pytest.mark.parametrize("variant", ["v3_ttgir", "v3_ttgir_fp32_pipeline_qdo", "split_softmax"])
@pytest.mark.parametrize("q_lengths,kv_lengths", [
    ([256] * 6, [127, 128, 129, 255, 256, 257]),
    ([256, 33, 31, 0], [129, 1, 128, 257]),
])
def test_hstu_cross_attn_gfx950_backward_tails(monkeypatch, variant, q_lengths, kv_lengths):
    tutorial = Path(__file__).resolve().parents[4] / "third_party/tlx/tutorials/hstu_cross_attn"
    monkeypatch.syspath_prepend(str(tutorial))
    xa = importlib.import_module("tlx_gfx950_cross_attention")
    config = xa.get_fwd_triton_spec_configs()[0]
    monkeypatch.setattr(xa._tlx_gfx950_cross_attn_fwd, "configs", [config])
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    torch.manual_seed(0)
    device = "cuda"
    q_offsets = torch.tensor([0] + q_lengths, device=device, dtype=torch.int64).cumsum(0)
    kv_offsets = torch.tensor([0] + kv_lengths, device=device, dtype=torch.int64).cumsum(0)
    q = torch.randn(sum(q_lengths), 1, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(sum(kv_lengths), 1, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    do = 0.1 * torch.randn_like(q)
    out = xa.tlx_gfx950_cross_attn_mha_wrapper(
        max_seq_len=max(kv_lengths),
        alpha=1.0 / 128,
        q=q,
        k=k,
        v=k,
        seq_offsets=kv_offsets,
        attn_scale=torch.tensor(1.0 / max(kv_lengths), device=device),
        max_q_len=256,
        seq_offsets_q=q_offsets,
        num_softmax_heads=1,
        num_targets=torch.tensor(q_lengths, device=device, dtype=torch.int64),
        causal=False,
        shared_kv=True,
        enable_tma=False,
        v3_ttgir_variant=variant,
    )
    out.backward(do)

    q_ref = q.detach().float().requires_grad_(True)
    k_ref = k.detach().float().requires_grad_(True)
    outputs = []
    q_start = kv_start = 0
    for q_len, kv_len in zip(q_lengths, kv_lengths):
        qs = q_ref[q_start:q_start + q_len, 0]
        ks = k_ref[kv_start:kv_start + kv_len, 0]
        p = torch.softmax((qs @ ks.T) / 128, dim=-1)
        outputs.append((p @ ks).unsqueeze(1))
        q_start += q_len
        kv_start += kv_len
    reference = torch.cat(outputs)
    reference.backward(do.float())
    for actual, expected in [(out, reference), (q.grad, q_ref.grad), (k.grad, k_ref.grad)]:
        assert torch.isfinite(actual).all()
        relative_error = torch.linalg.vector_norm(actual.float() - expected) / torch.linalg.vector_norm(expected)
        assert relative_error < 8e-3


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
@pytest.mark.parametrize("variant", ["v3_ttgir", "v3_ttgir_fp32_pipeline_qdo"])
@pytest.mark.parametrize("q_lengths", [[256, 256], [256, 33]])
@pytest.mark.parametrize("cap", [1, 128])
def test_hstu_cross_attn_gfx950_backward_empty_kv(monkeypatch, variant, q_lengths, cap):
    tutorial = Path(__file__).resolve().parents[4] / "third_party/tlx/tutorials/hstu_cross_attn"
    monkeypatch.syspath_prepend(str(tutorial))
    xa = importlib.import_module("tlx_gfx950_cross_attention")
    q = torch.randn(sum(q_lengths), 1, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.empty(0, 1, 128, device="cuda", dtype=torch.bfloat16)
    stats = torch.full((sum(q_lengths), 1), float("nan"), device="cuda", dtype=torch.float32)
    dq, dk, dv = xa.tlx_gfx950_cross_attn_bwd(
        dout=torch.randn_like(q),
        q=q,
        k=k,
        v=k,
        seq_offsets=torch.tensor([0, 0, 0], device="cuda", dtype=torch.int64),
        attn_scale=torch.tensor(1.0, device="cuda"),
        max_seq_len=cap,
        alpha=1.0 / 128,
        max_q_len=256,
        seq_offsets_q=torch.tensor([0] + q_lengths, device="cuda", dtype=torch.int64).cumsum(0),
        num_targets=torch.tensor(q_lengths, device="cuda", dtype=torch.int64),
        causal=False,
        shared_kv=True,
        G=1,
        num_softmax_heads=1,
        M=stats,
        Delta=stats,
        stride_mm=1,
        v3_ttgir_variant=variant,
    )
    torch.testing.assert_close(dq, torch.zeros_like(q), atol=0, rtol=0)
    assert dq.dtype == q.dtype
    assert dk.numel() == dv.numel() == 0


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
@pytest.mark.parametrize("variant", ["v3_ttgir", "v3_ttgir_fp32_pipeline_qdo"])
def test_hstu_cross_attn_gfx950_backward_query_dispatch(monkeypatch, variant):
    tutorial = Path(__file__).resolve().parents[4] / "third_party/tlx/tutorials/hstu_cross_attn"
    monkeypatch.syspath_prepend(str(tutorial))
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    xa = importlib.import_module("tlx_gfx950_cross_attention")
    torch.manual_seed(17)
    q_lengths = [255, 256, 257, 256, 33, 0]
    kv_lengths = [127, 128, 129, 255, 256, 257]
    q = torch.randn(sum(q_lengths), 1, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(sum(kv_lengths), 1, 128, device="cuda", dtype=torch.bfloat16)
    do = 0.1 * torch.randn_like(q)
    q_ref = q.float().requires_grad_(True)
    k_ref = k.float().requires_grad_(True)
    outputs, normalizers = [], []
    q_start = kv_start = 0
    for q_len, kv_len in zip(q_lengths, kv_lengths):
        qs = q_ref[q_start:q_start + q_len, 0]
        ks = k_ref[kv_start:kv_start + kv_len, 0]
        logits = (qs @ ks.T) / 128
        outputs.append((torch.softmax(logits, dim=-1) @ ks).unsqueeze(1))
        normalizers.append(torch.logsumexp(logits.detach(), dim=-1))
        q_start += q_len
        kv_start += kv_len
    reference = torch.cat(outputs)
    reference.backward(do.float())
    m = (torch.cat(normalizers) / torch.log(torch.tensor(2.0, device="cuda"))).unsqueeze(1)
    delta = (reference.detach() * do.float()).sum(dim=-1)
    # Test the device dispatch independently of the fixed forward grid.
    dq, dkv, _ = xa.tlx_gfx950_cross_attn_bwd(
        dout=do,
        q=q,
        k=k,
        v=k,
        seq_offsets=torch.tensor([0] + kv_lengths, device="cuda", dtype=torch.int64).cumsum(0),
        seq_offsets_q=torch.tensor([0] + q_lengths, device="cuda", dtype=torch.int64).cumsum(0),
        attn_scale=torch.tensor(1.0 / max(kv_lengths), device="cuda"),
        max_seq_len=max(kv_lengths),
        max_q_len=256,
        alpha=1.0 / 128,
        num_targets=torch.tensor(q_lengths, device="cuda", dtype=torch.int64),
        causal=False,
        shared_kv=True,
        G=1,
        num_softmax_heads=1,
        M=m,
        Delta=delta,
        stride_mm=1,
        v3_ttgir_variant=variant,
    )
    for actual, expected in [(dq, q_ref.grad), (dkv, k_ref.grad)]:
        assert torch.isfinite(actual).all()
        relative_error = torch.linalg.vector_norm(actual.float() - expected) / torch.linalg.vector_norm(expected)
        assert relative_error < 8e-3


def _cross_attention_fp32_reference(q, k, v, q_lengths, kv_lengths):
    outputs = []
    q_start = kv_start = 0
    for q_len, kv_len in zip(q_lengths, kv_lengths):
        qs = q[q_start:q_start + q_len].transpose(0, 1)
        ks = k[kv_start:kv_start + kv_len].transpose(0, 1)
        vs = v[kv_start:kv_start + kv_len].transpose(0, 1)
        probabilities = torch.softmax((qs @ ks.transpose(-1, -2)) / 128, dim=-1)
        outputs.append((probabilities @ vs).transpose(0, 1))
        q_start += q_len
        kv_start += kv_len
    return torch.cat(outputs)


def _assert_cross_attention_fp32_close(actual, expected):
    assert torch.isfinite(actual).all()
    difference = actual.float() - expected
    reference_norm = torch.linalg.vector_norm(expected)
    if reference_norm == 0:
        torch.testing.assert_close(actual.float(), expected, atol=0, rtol=0)
    else:
        assert torch.linalg.vector_norm(difference) / reference_norm < 8e-3
        assert difference.abs().max() / expected.abs().max() < 5e-3


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
@pytest.mark.parametrize("variant", [None, "v3_ttgir_fp32_pipeline_qdo"])
@pytest.mark.parametrize("heads,shared_kv,q_lengths,kv_lengths,coarse,partial", [
    pytest.param(heads, shared_kv, [0, 31, 256, 255], [257, 0, 1, 129], False, False,
                 id=f"small-h{heads}-{'shared' if shared_kv else 'separate'}")
    for heads in [1, 2]
    for shared_kv in [True, False]
] + [
    pytest.param(heads, True, [0, 31, 256, 255], [257, 0, 2048, 129], False, False, id=f"compact-long-h{heads}")
    for heads in [1, 2]
] + [
    pytest.param(2, True, [0, 31, 256, 255], [257, 0, 1024, 129], False, False, id="compact-eight-warps"),
    pytest.param(1, True, [0, 31, 256, 255, 1], [0, 1, 129, 2048, 257], True, False, id="coarse-ragged"),
])
def test_hstu_cross_attn_gfx950_fast_backward_reuse(monkeypatch, variant, heads, shared_kv, q_lengths, kv_lengths,
                                                    coarse, partial):
    tutorial = Path(__file__).resolve().parents[4] / "third_party/tlx/tutorials/hstu_cross_attn"
    monkeypatch.syspath_prepend(str(tutorial))
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    xa = importlib.import_module("tlx_gfx950_cross_attention")
    monkeypatch.setattr(xa._tlx_gfx950_cross_attn_fwd, "configs", [xa.get_fwd_triton_spec_configs()[0]])
    q_offsets = torch.tensor([0] + q_lengths, device="cuda", dtype=torch.int64).cumsum(0)
    kv_offsets = torch.tensor([0] + kv_lengths, device="cuda", dtype=torch.int64).cumsum(0)

    # A fresh context must not reuse tensors from the previous backward call.
    for context_index in range(2):
        torch.manual_seed(23 + context_index)
        q = torch.randn(sum(q_lengths), heads, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(sum(kv_lengths), heads, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        v = k if shared_kv else torch.randn_like(k, requires_grad=True)
        leaves = (q, k) if shared_kv else (q, k, v)
        q_ref = q.detach().float().requires_grad_(True)
        k_ref = k.detach().float().requires_grad_(True)
        v_ref = k_ref if shared_kv else v.detach().float().requires_grad_(True)
        reference_leaves = (q_ref, k_ref) if shared_kv else (q_ref, k_ref, v_ref)
        reference = _cross_attention_fp32_reference(q_ref, k_ref, v_ref, q_lengths, kv_lengths)
        out = xa.tlx_gfx950_cross_attn_mha_wrapper(
            max_seq_len=max(kv_lengths), alpha=1.0 / 128, q=q, k=k, v=v, seq_offsets=kv_offsets,
            attn_scale=torch.tensor(1.0 / max(kv_lengths), device="cuda"), max_q_len=256, seq_offsets_q=q_offsets,
            num_softmax_heads=heads, num_targets=torch.tensor(q_lengths, device="cuda", dtype=torch.int64),
            causal=False, shared_kv=shared_kv, enable_tma=False, v3_ttgir_variant=variant)
        assert out.grad_fn.use_small_backward
        assert out.grad_fn.use_coarse_backward == coarse
        assert out.grad_fn.use_partial_backward == partial
        assert out.grad_fn.use_compact_backward == (not coarse)
        _assert_cross_attention_fp32_close(out, reference)

        # Changed dO must replace all scratch values in the retained context.
        for _ in range(2):
            do = 0.1 * torch.randn_like(out)
            for leaf in leaves:
                leaf.grad = None
            out.backward(do, retain_graph=True)
            expected_grads = torch.autograd.grad(reference, reference_leaves, do.float(), retain_graph=True)
            for leaf, expected in zip(leaves, expected_grads):
                _assert_cross_attention_fp32_close(leaf.grad, expected)

            q_start = kv_start = 0
            for q_len, kv_len in zip(q_lengths, kv_lengths):
                if kv_len == 0:
                    torch.testing.assert_close(q.grad[q_start:q_start + q_len],
                                               torch.zeros_like(q.grad[q_start:q_start + q_len]), atol=0, rtol=0)
                if q_len == 0:
                    for leaf in leaves[1:]:
                        torch.testing.assert_close(leaf.grad[kv_start:kv_start + kv_len],
                                                   torch.zeros_like(leaf.grad[kv_start:kv_start + kv_len]), atol=0,
                                                   rtol=0)
                q_start += q_len
                kv_start += kv_len


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
@pytest.mark.parametrize("variant", ["v3_ttgir", "v3_ttgir_fp32_pipeline_qdo"])
def test_hstu_cross_attn_gfx950_fallback_noncontiguous_dout_reuse(monkeypatch, variant):
    """Normalize changed gradient strides before fallback preprocessing."""
    tutorial = Path(__file__).resolve().parents[4] / "third_party/tlx/tutorials/hstu_cross_attn"
    monkeypatch.syspath_prepend(str(tutorial))
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    xa = importlib.import_module("tlx_gfx950_cross_attention")
    monkeypatch.setattr(xa._tlx_gfx950_cross_attn_fwd, "configs", [xa.get_fwd_triton_spec_configs()[0]])
    batch, cap = 65, 128
    q_lengths = ([0, 1, 31, 32, 33, 127, 255, 256] * 9)[:batch]
    kv_lengths = ([1, 31, 32, 33, 63, 64, 127, 128] * 9)[:batch]
    q_offsets = torch.tensor([0] + q_lengths, device="cuda", dtype=torch.int64).cumsum(0)
    kv_offsets = torch.tensor([0] + kv_lengths, device="cuda", dtype=torch.int64).cumsum(0)
    for context_index in range(2):
        seed = 8 + context_index
        generator = torch.Generator(device="cuda").manual_seed(seed)
        full_q = torch.randn(batch * 256, 1, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
        q = full_q[:sum(q_lengths)].detach().requires_grad_(True)
        k = torch.randn(sum(kv_lengths), 1, 128, device="cuda", dtype=torch.bfloat16,
                        generator=generator).requires_grad_(True)
        torch.manual_seed(seed)
        first_do = (0.1 * torch.randn_like(full_q))[:sum(q_lengths)]
        q_ref = q.detach().float().requires_grad_(True)
        k_ref = k.detach().float().requires_grad_(True)
        reference = _cross_attention_fp32_reference(q_ref, k_ref, k_ref, q_lengths, kv_lengths)
        out = xa.tlx_gfx950_cross_attn_mha_wrapper(
            max_seq_len=cap, alpha=1.0 / 128, q=q, k=k, v=k, seq_offsets=kv_offsets,
            attn_scale=torch.tensor(1.0 / cap, device="cuda"), max_q_len=256, seq_offsets_q=q_offsets,
            num_softmax_heads=1, num_targets=torch.tensor(q_lengths, device="cuda", dtype=torch.int64), causal=False,
            shared_kv=True, enable_tma=False, v3_ttgir_variant=variant)
        assert not out.grad_fn.use_small_backward
        assert not out.grad_fn.use_retained_backward
        _assert_cross_attention_fp32_close(out, reference)
        for call in range(4):
            do = first_do if call == 0 else torch.randn(sum(q_lengths), 1, 256, device="cuda",
                                                        dtype=torch.bfloat16)[:, :, ::2]
            if call == 3:
                do = torch.ones((), device="cuda", dtype=torch.bfloat16).expand_as(out)
                assert do.stride() == (0, 0, 0)
            elif call:
                assert do.stride(-1) == 2 and not do.is_contiguous()
            actual = torch.autograd.grad(out, (q, k), do, retain_graph=True)
            expected = torch.autograd.grad(reference, (q_ref, k_ref), do.float(), retain_graph=True)
            for got, want in zip(actual, expected):
                _assert_cross_attention_fp32_close(got, want)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
@pytest.mark.parametrize("variant", [None, "v3_ttgir_fp32_pipeline_qdo"])
@pytest.mark.parametrize("cap", [383, 384, 512, 1024])
def test_hstu_cross_attn_gfx950_coarse_boundary_preservation(monkeypatch, variant, cap):
    """Preserve baseline dKV accuracy at the coarse dispatch boundary.

    The retained baseline has normalized dKV error 0.005061 on this fixture.
    This exceeds the separate strict oracle gate of 0.005.
    Require baseline agreement and no increase in that FP32 error.
    """
    tutorial = Path(__file__).resolve().parents[4] / "third_party/tlx/tutorials/hstu_cross_attn"
    monkeypatch.syspath_prepend(str(tutorial))
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    xa = importlib.import_module("tlx_gfx950_cross_attention")
    monkeypatch.setattr(xa._tlx_gfx950_cross_attn_fwd, "configs", [xa.get_fwd_triton_spec_configs()[0]])
    q_lengths = [0, 31, 256, 255, 1]
    kv_lengths = [1, 0, 129, cap, 257]
    q_offsets = torch.tensor([0] + q_lengths, device="cuda", dtype=torch.int64).cumsum(0)
    kv_offsets = torch.tensor([0] + kv_lengths, device="cuda", dtype=torch.int64).cumsum(0)
    for context_index in range(2):
        torch.manual_seed(23 + context_index)
        q = torch.randn(sum(q_lengths), 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(sum(kv_lengths), 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        q_baseline = q.detach().clone().requires_grad_(True)
        k_baseline = k.detach().clone().requires_grad_(True)
        q_ref = q.detach().float().requires_grad_(True)
        k_ref = k.detach().float().requires_grad_(True)
        reference = _cross_attention_fp32_reference(q_ref, k_ref, k_ref, q_lengths, kv_lengths)
        kwargs = dict(max_seq_len=cap, alpha=1.0 / 128, seq_offsets=kv_offsets,
                      attn_scale=torch.tensor(1.0 / cap, device="cuda"), max_q_len=256, seq_offsets_q=q_offsets,
                      num_softmax_heads=1, num_targets=torch.tensor(q_lengths, device="cuda", dtype=torch.int64),
                      causal=False, shared_kv=True, enable_tma=False)
        out = xa.tlx_gfx950_cross_attn_mha_wrapper(q=q, k=k, v=k, v3_ttgir_variant=variant, **kwargs)
        baseline = xa.tlx_gfx950_cross_attn_mha_wrapper(q=q_baseline, k=k_baseline, v=k_baseline,
                                                        v3_ttgir_variant="v3_ttgir", **kwargs)
        assert out.grad_fn.use_small_backward
        assert out.grad_fn.use_coarse_backward == (cap >= 384)
        assert not out.grad_fn.use_partial_backward
        assert not baseline.grad_fn.use_small_backward
        _assert_cross_attention_fp32_close(out, reference)
        _assert_cross_attention_fp32_close(baseline, reference)
        for _ in range(2):
            do = 0.1 * torch.randn_like(out)
            for leaf in (q, k, q_baseline, k_baseline):
                leaf.grad = None
            out.backward(do, retain_graph=True)
            baseline.backward(do, retain_graph=True)
            dq_ref, dkv_ref = torch.autograd.grad(reference, (q_ref, k_ref), do.float(), retain_graph=True)
            _assert_cross_attention_fp32_close(q.grad, dq_ref)
            _assert_cross_attention_fp32_close(q_baseline.grad, dq_ref)
            _assert_cross_attention_fp32_close(k.grad, k_baseline.grad.float())
            reference_max = dkv_ref.abs().max()
            candidate_error = (k.grad.float() - dkv_ref).abs().max() / reference_max
            baseline_error = (k_baseline.grad.float() - dkv_ref).abs().max() / reference_max
            assert candidate_error <= baseline_error
            q_start = kv_start = 0
            for q_len, kv_len in zip(q_lengths, kv_lengths):
                if kv_len == 0:
                    torch.testing.assert_close(q.grad[q_start:q_start + q_len],
                                               torch.zeros_like(q.grad[q_start:q_start + q_len]), atol=0, rtol=0)
                if q_len == 0:
                    torch.testing.assert_close(k.grad[kv_start:kv_start + kv_len],
                                               torch.zeros_like(k.grad[kv_start:kv_start + kv_len]), atol=0, rtol=0)
                q_start += q_len
                kv_start += kv_len


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
@pytest.mark.parametrize("variant", [None, "v3_ttgir_fp32_pipeline_qdo"])
@pytest.mark.parametrize("cap", [256, 512, 2048])
@pytest.mark.parametrize("alpha", [1.0 / 128, -1.0 / 128])
def test_hstu_cross_attn_gfx950_retained_backward_reuse(monkeypatch, variant, cap, alpha):
    tutorial = Path(__file__).resolve().parents[4] / "third_party/tlx/tutorials/hstu_cross_attn"
    monkeypatch.syspath_prepend(str(tutorial))
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    xa = importlib.import_module("tlx_gfx950_cross_attention")
    monkeypatch.setattr(xa._tlx_gfx950_cross_attn_fwd, "configs", [xa.get_fwd_triton_spec_configs()[0]])
    q_lengths = [0, 31, 256, 255, 1] + [0] * 2043
    kv_lengths = [min(257, cap), 0, 1, cap, 129] + [0] * 2043
    q_offsets = torch.tensor([0] + q_lengths, device="cuda", dtype=torch.int64).cumsum(0)
    kv_offsets = torch.tensor([0] + kv_lengths, device="cuda", dtype=torch.int64).cumsum(0)
    for context_index in range(2):
        torch.manual_seed(context_index)
        q = torch.randn(sum(q_lengths), 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(sum(kv_lengths), 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        q_ref = q.detach().float().requires_grad_(True)
        k_ref = k.detach().float().requires_grad_(True)
        outputs = []
        q_start = kv_start = 0
        # The remaining sequences contain no tokens.
        for q_len, kv_len in zip(q_lengths[:5], kv_lengths[:5]):
            qs = q_ref[q_start:q_start + q_len, 0]
            ks = k_ref[kv_start:kv_start + kv_len, 0]
            outputs.append((torch.softmax((qs @ ks.T) * alpha, dim=-1) @ ks).unsqueeze(1))
            q_start += q_len
            kv_start += kv_len
        reference = torch.cat(outputs)
        out = xa.tlx_gfx950_cross_attn_mha_wrapper(
            max_seq_len=cap, alpha=alpha, q=q, k=k, v=k, seq_offsets=kv_offsets,
            attn_scale=torch.tensor(1.0 / cap, device="cuda"), max_q_len=256, seq_offsets_q=q_offsets,
            num_softmax_heads=1, num_targets=torch.tensor(q_lengths, device="cuda", dtype=torch.int64), causal=False,
            shared_kv=True, enable_tma=False, v3_ttgir_variant=variant)
        assert out.grad_fn.use_retained_backward
        assert not out.grad_fn.use_small_backward
        _assert_cross_attention_fp32_close(out, reference)
        for _ in range(2):
            do = (0.1 * torch.randn(out.shape[0], 1, 256, device="cuda", dtype=torch.bfloat16))[:, :, ::2]
            assert not do.is_contiguous()
            actual = torch.autograd.grad(out, (q, k), do, retain_graph=True)
            expected = torch.autograd.grad(reference, (q_ref, k_ref), do.float(), retain_graph=True)
            for got, want in zip(actual, expected):
                _assert_cross_attention_fp32_close(got, want)
            torch.testing.assert_close(actual[0][:31], torch.zeros_like(actual[0][:31]), atol=0, rtol=0)
            torch.testing.assert_close(actual[1][:min(257, cap)], torch.zeros_like(actual[1][:min(257, cap)]), atol=0,
                                       rtol=0)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
@pytest.mark.parametrize("changes,expected", [
    ({}, True),
    ({"batch": 255}, False),
    ({"batch": 256}, True),
    ({"cap": 255}, False),
    ({"cap": 256}, True),
    ({"cap": 257}, False),
    ({"cap": 384}, False),
    ({"cap": 1536}, False),
    ({"cap": 2049}, False),
    ({"max_q": 257}, False),
    ({"heads": 2}, False),
    ({"shared": False}, False),
    ({"softmax_heads": 0}, False),
    ({"dtype": torch.float16}, False),
    ({"offset_dtype": torch.int32}, False),
    ({"causal": True}, False),
    ({"variant": "v3_ttgir"}, False),
])
def test_hstu_cross_attn_gfx950_retained_backward_dispatch(monkeypatch, changes, expected):
    tutorial = Path(__file__).resolve().parents[4] / "third_party/tlx/tutorials/hstu_cross_attn"
    monkeypatch.syspath_prepend(str(tutorial))
    xa = importlib.import_module("tlx_gfx950_cross_attention")
    options = dict(batch=2048, cap=512, max_q=256, heads=1, shared=True, softmax_heads=1, dtype=torch.bfloat16,
                   offset_dtype=torch.int64, causal=False, variant=None)
    options.update(changes)
    q = torch.ones(1, options["heads"], 128, device="cuda", dtype=options["dtype"], requires_grad=True)
    k = torch.ones_like(q, requires_grad=True)
    offsets = torch.ones(options["batch"] + 1, device="cuda", dtype=options["offset_dtype"])
    offsets[0] = 0

    def forward(**kwargs):
        q_input = kwargs["q"]
        return torch.zeros_like(q_input), torch.zeros(q_input.shape[:2], device=q_input.device, dtype=torch.float32)

    # Isolate metadata selection from forward kernel execution.
    monkeypatch.setattr(xa, "tlx_gfx950_cross_attn_fwd", forward)
    out = xa.tlx_gfx950_cross_attn_mha_wrapper(max_seq_len=options["cap"], alpha=1.0 / 128, q=q, k=k,
                                               v=k if options["shared"] else k.clone(), seq_offsets=offsets,
                                               attn_scale=torch.tensor(1.0, device="cuda"), max_q_len=options["max_q"],
                                               seq_offsets_q=offsets, num_softmax_heads=options["softmax_heads"],
                                               causal=options["causal"], shared_kv=options["shared"], enable_tma=False,
                                               v3_ttgir_variant=options["variant"])
    assert out.grad_fn.use_retained_backward == expected


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
@pytest.mark.parametrize("heads,shared_kv,cap", [
    pytest.param(heads, shared, cap, id=f"h{heads}-{'shared' if shared else 'separate'}-cap{cap}")
    for heads in [1, 2]
    for shared in [True, False]
    for cap in ([1, 1024, 2048] if shared else [1, 128, 1024])
])
def test_hstu_cross_attn_gfx950_compact_single_key_gradient(monkeypatch, heads, shared_kv, cap):
    tutorial = Path(__file__).resolve().parents[4] / "third_party/tlx/tutorials/hstu_cross_attn"
    monkeypatch.syspath_prepend(str(tutorial))
    xa = importlib.import_module("tlx_gfx950_cross_attention")
    small = importlib.import_module("tlx_gfx950_cross_attention_small")
    monkeypatch.setattr(xa._tlx_gfx950_cross_attn_fwd, "configs", [xa.get_fwd_triton_spec_configs()[0]])
    q_lengths = [0, 31, 255, 256]
    kv_lengths = [1, 0, 1, 0]
    query_begin = sum(q_lengths[:2])
    query_end = query_begin + q_lengths[2]
    key_index = sum(kv_lengths[:2])
    q_offsets = torch.tensor([0] + q_lengths, device="cuda", dtype=torch.int64).cumsum(0)
    kv_offsets = torch.tensor([0] + kv_lengths, device="cuda", dtype=torch.int64).cumsum(0)
    for context_index in range(2):
        torch.manual_seed(context_index)
        q = torch.randn(sum(q_lengths), heads, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(sum(kv_lengths), heads, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        v = k if shared_kv else torch.randn_like(k, requires_grad=True)
        leaves = (q, k) if shared_kv else (q, k, v)
        out = xa.tlx_gfx950_cross_attn_mha_wrapper(max_seq_len=cap, alpha=1.0 / 128, q=q, k=k, v=v,
                                                   seq_offsets=kv_offsets,
                                                   attn_scale=torch.tensor(1.0 / cap, device="cuda"), max_q_len=256,
                                                   seq_offsets_q=q_offsets, num_softmax_heads=heads,
                                                   shared_kv=shared_kv, causal=False)
        assert out.grad_fn.use_compact_backward
        expected_out = torch.zeros_like(out)
        expected_out[query_begin:query_end] = v.detach()[key_index]
        torch.testing.assert_close(out, expected_out, atol=0, rtol=0)
        poisoned_m = torch.full((q.shape[0], heads), float("nan"), device="cuda", dtype=torch.float32)
        poisoned_out = torch.full_like(out, float("nan"))
        for _ in range(2):
            do = (0.1 * torch.randn(q.shape[0], heads, 256, device="cuda", dtype=q.dtype))[:, :, ::2]
            # A single-key softmax has zero score gradients.
            expected_q = torch.zeros_like(q, dtype=torch.float32)
            expected_v = torch.zeros_like(v, dtype=torch.float32)
            expected_v[key_index] = do[query_begin:query_end].float().sum(dim=0)
            expected = ((expected_q, expected_v) if shared_kv else
                        (expected_q, torch.zeros_like(k, dtype=torch.float32), expected_v))
            public = torch.autograd.grad(out, leaves, do, retain_graph=True)
            direct = small.compact_softmax_backward(q, k, v, do, poisoned_m, poisoned_out, q_offsets, kv_offsets, cap,
                                                    1.0 / 128, shared_kv)
            direct = tuple(value for value in direct if value is not None)
            for actual in (public, direct):
                for value, reference in zip(actual, expected):
                    _assert_cross_attention_fp32_close(value, reference)
            for a, b in zip(public, direct):
                torch.testing.assert_close(a, b, atol=0, rtol=0)


def _load_gfx950_hstu_impl():
    """Load the production implementation for low-level schedule coverage."""
    from triton.tlx.ops.kernels.hstu_attn import gfx950

    return gfx950


def _load_gfx950_hstu_benchmark():
    """Load the benchmark helpers without requiring a GPU workload."""
    import sys

    kernel_dir = str(Path(__file__).resolve().parents[4] / "third_party" / "tlx" / "tutorials" / "hstu_self_attn")
    if kernel_dir not in sys.path:
        sys.path.insert(0, kernel_dir)
    import bench_gfx950_bwd as benchmark

    return benchmark


def test_hstu_gfx950_fixture_launch_coverage(tmp_path):
    benchmark = _load_gfx950_hstu_benchmark()

    # Issue #2005 intentionally has a length one row beyond N, but every
    # forward configuration still launches through row 1023.
    assert benchmark._validate_fixture_launch_coverage(996, torch.tensor([898, 997])) == 1024
    assert benchmark._validate_fixture_launch_coverage(1024, torch.tensor([1024])) == 1024

    with pytest.raises(ValueError, match=r"sequence 1 has length 1153.*N=1024.*at most 1024"):
        benchmark._validate_fixture_launch_coverage(1024, torch.tensor([1024, 1153]))

    invalid_fixture = {
        "N": 1024,
        "alpha": 1.0 / 128,
        "q_shape": (2177, 4, 128),
        "seq_offsets": torch.tensor([0, 1024, 2177]),
        "invalid_attn_mask_type": "lower_triangular",
        "num_targets": torch.tensor([20, 20]),
        "attn_bias": None,
        "seq2_offsets": None,
        "max_attn_len": 0,
        "contextual_seq_len": 0,
        "sort_by_length": False,
    }
    fixture_path = tmp_path / "invalid_hstu_fixture.pt"
    torch.save(invalid_fixture, fixture_path)
    with pytest.raises(ValueError, match=r"sequence 1 has length 1153.*N=1024.*at most 1024"):
        benchmark._make_input_fixture_workload(fixture_path)


def test_hstu_gfx950_sequence_xcd_padding_budget():
    hstu = _load_gfx950_hstu_impl()

    assert hstu._gfx950_fa_schedule_launch_sequences(7, 8) == (7, False)
    assert hstu._gfx950_fa_schedule_launch_sequences(9, 8) == (9, False)
    assert hstu._gfx950_fa_schedule_launch_sequences(13, 8) == (13, False)
    assert hstu._gfx950_fa_schedule_launch_sequences(14, 8) == (16, True)
    assert hstu._gfx950_fa_schedule_launch_sequences(512, 8) == (512, True)
    assert hstu._gfx950_fa_schedule_launch_sequences(513, 8) == (520, True)
    for num_xcds in (2, 4, 8):
        assert hstu._gfx950_fa_schedule_launch_sequences(15, num_xcds) == (16, True)


def _target_causal_hstu_ref(q, k, v, offsets, num_targets, max_seq_len, alpha):
    """Float reference for history-causal plus independent-target masking."""
    qf = q.float().detach().requires_grad_()
    kf = k.float().detach().requires_grad_()
    vf = v.float().detach().requires_grad_()
    outputs = []
    for z in range(offsets.numel() - 1):
        start, end = int(offsets[z]), int(offsets[z + 1])
        seq_len = end - start
        history_end = seq_len - int(num_targets[z])
        query = torch.arange(seq_len, device=q.device)[:, None]
        key = torch.arange(seq_len, device=q.device)[None, :]
        valid = (query == key) | ((key < history_end) & (key < query))
        scores = torch.einsum("qhd,khd->hqk", qf[start:end], kf[start:end]) * alpha
        weights = scores * torch.sigmoid(scores) / max_seq_len
        outputs.append(torch.einsum("hqk,khd->qhd", weights * valid[None], vf[start:end]))
    return torch.cat(outputs), qf, kf, vf


def _relative_l2(got, expected):
    return ((got.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-20)).item()


def _assert_per_sequence_close(name, got, expected, offsets, tolerance=8e-3, tail_tolerance=2e-2):
    """Keep short ragged segments and their last valid rows visible in the error."""
    for z in range(offsets.numel() - 1):
        start, end = int(offsets[z]), int(offsets[z + 1])
        error = _relative_l2(got[start:end], expected[start:end])
        assert error < tolerance, f"{name}[{z}] relative L2 {error:.3e} >= {tolerance:.3e}"
        tail_error = _relative_l2(got[end - 1], expected[end - 1])
        assert tail_error < tail_tolerance, f"{name}[{z}] last-row relative L2 {tail_error:.3e} >= {tail_tolerance:.3e}"


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
@pytest.mark.parametrize(
    "bwd_variant,sequence_xcd_case",
    [
        pytest.param("tlx.ops-general", "padded", id="tlx-ops-general"),
        pytest.param("kv_parallel_fa_schedule", "padded", id="fa-schedule"),
        pytest.param(
            "tlx.ops-production",
            "padded",
            id="tlx-ops-production",
        ),
        pytest.param(
            "kv_parallel_fa_schedule_bn256_direct_qdo_g2l",
            "padded",
            id="fa-schedule-bn256",
        ),
        pytest.param(
            "kv_parallel_fa_schedule_mask_peel_resident_k_dr_early_do_t",
            "unpadded",
            id="fa-schedule-production-unpadded-sequence-xcd",
        ),
    ],
)
def test_hstu_gfx950_backward_target_causal(bwd_variant, sequence_xcd_case):
    """Cover ragged tails and padded/unpadded XCD sequence scheduling."""
    hstu = _load_gfx950_hstu_impl()
    torch.manual_seed(7)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    max_seq_len, heads, head_dim = 512, 2, 128
    alpha = 1.0 / head_dim**0.5
    # The padded cases use fifteen sequences, which pad to sixteen on 2-, 4-,
    # and 8-XCD partitions while staying within the dummy-sequence budget.
    # Exercise a wholly invalid second Q/dO pipeline slot, both sides of the
    # 64-, 128-, and 256-row boundaries, partial final K/V tiles, and two
    # exact BN256 tiles.
    lengths_list = [1, 17, 33, 63, 65, 97, 127, 129, 193, 255, 256, 257, 321, 511, 512]
    if sequence_xcd_case == "unpadded":
        # A multiple of every supported gfx950 XCD count selects sequence-XCD
        # scheduling without compiling the padded-slot guard.
        lengths_list.insert(-2, 400)
    lengths = torch.tensor(lengths_list, device=device, dtype=torch.int64)
    offsets = torch.zeros(lengths.numel() + 1, device=device, dtype=torch.int64)
    offsets[1:] = torch.cumsum(lengths, dim=0)
    # Include a diagonal-only all-target sequence and history/target boundaries
    # that fall inside 64-, 128-, and 256-row tiles.
    num_targets_list = [1, 1, 5, 1, 65, 17, 20, 33, 5, 20, 17, 20, 17, 5, 20]
    if sequence_xcd_case == "unpadded":
        num_targets_list.insert(-2, 20)
        cu_count = torch.cuda.get_device_properties(device).multi_processor_count
        num_xcds = max(1, min(hstu.GFX950_MAX_XCDS, cu_count // hstu.GFX950_CUS_PER_XCD))
        scheduled_z, use_sequence_xcd = hstu._gfx950_fa_schedule_launch_sequences(len(lengths_list), num_xcds)
        assert use_sequence_xcd
        assert scheduled_z == len(lengths_list)
    num_targets = torch.tensor(num_targets_list, device=device, dtype=torch.int32)
    total = int(offsets[-1])

    q, k, v = (torch.empty((total, heads, head_dim), device=device, dtype=dtype).uniform_(-2.0, 2.0).requires_grad_()
               for _ in range(3))
    dout = torch.empty_like(q).uniform_(-1.0, 1.0)
    ref, q_ref, k_ref, v_ref = _target_causal_hstu_ref(
        q,
        k,
        v,
        offsets,
        num_targets,
        max_seq_len,
        alpha,
    )
    if bwd_variant.startswith("tlx.ops-"):
        from triton.tlx.ops import hstu_attn_dev

        attn_scale = (torch.tensor(1.0 / max_seq_len, device=device, dtype=torch.float32)
                      if bwd_variant == "tlx.ops-general" else None)
        out = hstu_attn_dev(
            q,
            k,
            v,
            offsets,
            max_seq_len,
            attn_scale,
            alpha=alpha,
            num_targets=num_targets,
            space="smoke",
        )
    else:
        out = hstu.tlx_gfx950_hstu_mha(
            max_seq_len,
            alpha,
            q,
            k,
            v,
            offsets,
            num_targets=num_targets,
            bwd_variant=bwd_variant,
        )
    out.backward(dout)
    ref.backward(dout.float())

    _assert_per_sequence_close("out", out, ref, offsets)
    for name, got, expected in (
        ("dq", q.grad, q_ref.grad),
        ("dk", k.grad, k_ref.grad),
        ("dv", v.grad, v_ref.grad),
    ):
        assert got is not None and expected is not None, name
        _assert_per_sequence_close(name, got, expected, offsets)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
def test_hstu_gfx950_backward_ticket_2005_layout():
    """Cover the interleaved QKV layout and N/length boundary from issue #2005."""
    hstu = _load_gfx950_hstu_impl()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    max_seq_len, heads, head_dim = 996, 4, 128
    alpha = 1.0 / head_dim**0.5
    lengths = torch.tensor([898, 997], device=device, dtype=torch.int64)
    offsets = torch.zeros(lengths.numel() + 1, device=device, dtype=torch.int64)
    offsets[1:] = torch.cumsum(lengths, dim=0)
    num_targets = lengths
    total = int(offsets[-1])

    value_gen = torch.Generator(device=device).manual_seed(1)
    backing = torch.empty((total, 4 * heads, head_dim), device=device,
                          dtype=dtype).uniform_(-2.0, 2.0, generator=value_gen)
    q, k, v, _ = torch.split(backing, [heads, heads, heads, heads], dim=1)
    q, k, v = (tensor.detach().requires_grad_() for tensor in (q, k, v))
    expected_stride = (4 * heads * head_dim, head_dim, 1)
    assert q.stride() == k.stride() == v.stride() == expected_stride
    dout_gen = torch.Generator(device=device).manual_seed(2)
    dout = torch.empty_like(q).uniform_(-1.0, 1.0, generator=dout_gen)
    assert dout.stride() == (heads * head_dim, head_dim, 1)

    ref, q_ref, k_ref, v_ref = _target_causal_hstu_ref(
        q,
        k,
        v,
        offsets,
        num_targets,
        max_seq_len,
        alpha,
    )
    out = hstu.tlx_gfx950_hstu_mha(
        max_seq_len,
        alpha,
        q,
        k,
        v,
        offsets,
        num_targets=num_targets,
        bwd_variant="kv_parallel_fa_schedule_mask_peel_resident_k_dr_early_do_t",
    )
    out.backward(dout)
    ref.backward(dout.float())

    _assert_per_sequence_close("out", out, ref, offsets)
    for name, got, expected in (
        ("dq", q.grad, q_ref.grad),
        ("dk", k.grad, k_ref.grad),
        ("dv", v.grad, v_ref.grad),
    ):
        assert got is not None and expected is not None, name
        _assert_per_sequence_close(name, got, expected, offsets)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
@pytest.mark.parametrize(
    "bwd_options",
    [
        pytest.param(
            {
                "native_mfma_dq_warps": 0,
                "kv_parallel": True,
            },
            id="aliased-dq-acc",
        ),
        pytest.param(
            {
                "native_mfma_dq_warps": 4,
                "kv_parallel": True,
            },
            id="separate-dq-acc",
        ),
        pytest.param(
            {
                "native_mfma_dq_warps": 4,
                "kv_parallel": True,
                "fa_schedule": True,
                "fa_schedule_direct_qdo_g2l": True,
                "fa_schedule_mask_peel": True,
                "fa_schedule_resident_k_score": True,
                "fa_schedule_dr_resident": True,
                "fa_schedule_early_do_t": True,
            },
            id="separate-dq-acc-fa-schedule",
        ),
    ],
)
def test_hstu_gfx950_backward_reuses_poisoned_outputs(bwd_options):
    """Repeated launches must clear aliased or separate dQ accumulation state."""
    hstu = _load_gfx950_hstu_impl()
    torch.manual_seed(11)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    max_seq_len, heads, head_dim = 256, 2, 128
    alpha = 1.0 / head_dim**0.5
    lengths = torch.tensor([65, 129, 193], device=device, dtype=torch.int64)
    offsets = torch.zeros(lengths.numel() + 1, device=device, dtype=torch.int64)
    offsets[1:] = torch.cumsum(lengths, dim=0)
    num_targets = torch.tensor([65, 20, 33], device=device, dtype=torch.int32)
    total = int(offsets[-1])

    q, k, v = (torch.empty((total, heads, head_dim), device=device, dtype=dtype).uniform_(-2.0, 2.0).requires_grad_()
               for _ in range(3))
    dout = torch.empty_like(q).uniform_(-1.0, 1.0)
    ref, q_ref, k_ref, v_ref = _target_causal_hstu_ref(
        q,
        k,
        v,
        offsets,
        num_targets,
        max_seq_len,
        alpha,
    )
    ref.backward(dout.float())
    expected = (q_ref.grad, k_ref.grad, v_ref.grad)
    assert all(tensor is not None for tensor in expected)

    dq, dk, dv = (torch.empty_like(q) for _ in range(3))
    first_result = None
    for poison in (float("nan"), 16384.0, -16384.0):
        for tensor in (dq, dk, dv):
            tensor.fill_(poison)
        hstu.tlx_gfx950_ragged_attention_bwd(
            dout=dout,
            q=q,
            k=k,
            v=v,
            dq=dq,
            dk=dk,
            dv=dv,
            seq_offsets=offsets,
            num_targets=num_targets,
            attn_scale=None,
            N=max_seq_len,
            alpha=alpha,
            max_attn_len=0,
            invalid_attn_mask_type="lower_triangular",
            contextual_seq_len=0,
            sort_by_length_indices=None,
            full_attn_size=0,
            **bwd_options,
        )

        result = (dq, dk, dv)
        for name, got, wanted in zip(("dq", "dk", "dv"), result, expected):
            assert torch.isfinite(got).all(), f"{name} retained poison {poison}"
            _assert_per_sequence_close(name, got, wanted, offsets)
        if first_result is None:
            first_result = tuple(tensor.clone() for tensor in result)
        else:
            for name, got, first in zip(("dq", "dk", "dv"), result, first_result):
                _assert_per_sequence_close(f"repeat-{name}", got, first, offsets, tolerance=2e-3)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
@pytest.mark.parametrize(
    "bwd_options",
    [
        pytest.param(
            {
                "native_mfma_dq_warps": 0,
                "kv_parallel": True,
            },
            id="aliased-dq-acc",
        ),
        pytest.param(
            {
                "native_mfma_dq_warps": 4,
                "kv_parallel": True,
                "fa_schedule": True,
            },
            id="separate-dq-acc-fa-schedule",
        ),
    ],
)
def test_hstu_gfx950_backward_graph_replay_resets_dq(bwd_options):
    """Captured backward launches must reset dQ accumulation on every replay."""
    hstu = _load_gfx950_hstu_impl()
    torch.manual_seed(13)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    max_seq_len, heads, head_dim = 256, 2, 128
    alpha = 1.0 / head_dim**0.5
    lengths = torch.tensor([65, 193], device=device, dtype=torch.int64)
    offsets = torch.zeros(lengths.numel() + 1, device=device, dtype=torch.int64)
    offsets[1:] = torch.cumsum(lengths, dim=0)
    num_targets = torch.tensor([20, 33], device=device, dtype=torch.int32)
    total = int(offsets[-1])
    q, k, v = (torch.empty((total, heads, head_dim), device=device, dtype=dtype).uniform_(-2.0, 2.0) for _ in range(3))
    dout = torch.empty_like(q).uniform_(-1.0, 1.0)
    dq, dk, dv = (torch.empty_like(q) for _ in range(3))

    def launch():
        hstu.tlx_gfx950_ragged_attention_bwd(
            dout=dout,
            q=q,
            k=k,
            v=v,
            dq=dq,
            dk=dk,
            dv=dv,
            seq_offsets=offsets,
            num_targets=num_targets,
            attn_scale=None,
            N=max_seq_len,
            alpha=alpha,
            max_attn_len=0,
            invalid_attn_mask_type="lower_triangular",
            contextual_seq_len=0,
            sort_by_length_indices=None,
            full_attn_size=0,
            **bwd_options,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        launch()
        launch()
    torch.cuda.current_stream().wait_stream(stream)
    expected_dq = dq.clone()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()

    for _ in range(3):
        dq.fill_(float("nan"))
        graph.replay()
        assert torch.isfinite(dq).all()
        _assert_per_sequence_close("dq", dq, expected_dq, offsets, tolerance=2e-3)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
@pytest.mark.parametrize("variant", [None, "v3_ttgir_fp32_pipeline_qdo"])
def test_hstu_cross_attn_gfx950_compact_kv_partition_reuse(monkeypatch, variant):
    """Check both KV partitions with query tails and repeated backward calls."""
    heads = 1
    q_lengths = [31, 256, 255, 1]
    kv_lengths = [1023, 1024, 1025, 2047]
    tutorial = Path(__file__).resolve().parents[4] / "third_party/tlx/tutorials/hstu_cross_attn"
    monkeypatch.syspath_prepend(str(tutorial))
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    xa = importlib.import_module("tlx_gfx950_cross_attention")
    monkeypatch.setattr(xa._tlx_gfx950_cross_attn_fwd, "configs", [xa.get_fwd_triton_spec_configs()[0]])
    q_offsets = torch.tensor([0] + q_lengths, device="cuda", dtype=torch.int64).cumsum(0)
    kv_offsets = torch.tensor([0] + kv_lengths, device="cuda", dtype=torch.int64).cumsum(0)

    # A fresh context must not reuse tensors from the previous backward call.
    for context_index in range(2):
        torch.manual_seed(23 + context_index)
        q = torch.randn(sum(q_lengths), heads, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(sum(kv_lengths), heads, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        leaves = (q, k)
        q_ref = q.detach().float().requires_grad_(True)
        k_ref = k.detach().float().requires_grad_(True)
        reference_leaves = (q_ref, k_ref)
        reference = _cross_attention_fp32_reference(q_ref, k_ref, k_ref, q_lengths, kv_lengths)
        out = xa.tlx_gfx950_cross_attn_mha_wrapper(
            max_seq_len=2048, alpha=1.0 / 128, q=q, k=k, v=k, seq_offsets=kv_offsets,
            attn_scale=torch.tensor(1.0 / 2048, device="cuda"), max_q_len=256, seq_offsets_q=q_offsets,
            num_softmax_heads=heads, num_targets=torch.tensor(q_lengths, device="cuda", dtype=torch.int64),
            causal=False, shared_kv=True, enable_tma=False, v3_ttgir_variant=variant)
        _assert_cross_attention_fp32_close(out, reference)

        # Changed dO must replace all scratch values in the retained context.
        for _ in range(2):
            do = 0.1 * torch.randn_like(out)
            for leaf in leaves:
                leaf.grad = None
            out.backward(do, retain_graph=True)
            expected_grads = torch.autograd.grad(reference, reference_leaves, do.float(), retain_graph=True)
            for leaf, expected in zip(leaves, expected_grads):
                _assert_cross_attention_fp32_close(leaf.grad, expected)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
@pytest.mark.parametrize("variant", ["v3_ttgir", "v3_ttgir_fp32_pipeline_qdo"])
@pytest.mark.parametrize("cap", [1, 128])
@pytest.mark.parametrize("alpha", [1.0 / 128, -1.0 / 128])
def test_hstu_cross_attn_gfx950_fallback_single_key_gradient(monkeypatch, variant, cap, alpha):
    """A one-key softmax has zero score derivatives for every query."""
    tutorial = Path(__file__).resolve().parents[4] / "third_party/tlx/tutorials/hstu_cross_attn"
    monkeypatch.syspath_prepend(str(tutorial))
    xa = importlib.import_module("tlx_gfx950_cross_attention")
    monkeypatch.setattr(xa._tlx_gfx950_cross_attn_fwd, "configs", [xa.get_fwd_triton_spec_configs()[0]])
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    q_lengths = ([0, 1, 31, 32, 33, 127, 255, 256] * 9)[:65]
    q_offsets = torch.tensor([0] + q_lengths, device="cuda", dtype=torch.int64).cumsum(0)
    kv_offsets = torch.arange(66, device="cuda", dtype=torch.int64)
    torch.manual_seed(8)
    q = torch.randn(sum(q_lengths), 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(65, 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = xa.tlx_gfx950_cross_attn_mha_wrapper(max_seq_len=cap, alpha=alpha, q=q, k=k, v=k, seq_offsets=kv_offsets,
                                               attn_scale=torch.tensor(1.0 / cap, device="cuda"), max_q_len=256,
                                               seq_offsets_q=q_offsets, num_softmax_heads=1, shared_kv=True,
                                               causal=False,
                                               num_targets=torch.tensor(q_lengths, device="cuda", dtype=torch.int64),
                                               enable_tma=False, v3_ttgir_variant=variant)
    assert not out.grad_fn.use_small_backward and not out.grad_fn.use_retained_backward
    for call in range(2):
        do = 0.1 * torch.randn_like(q) if call == 0 else (
            0.1 * torch.randn(q.shape[0], 1, 256, device="cuda", dtype=q.dtype))[..., ::2]
        dq, dkv = torch.autograd.grad(out, (q, k), do, retain_graph=True)
        expected_q = torch.zeros_like(q, dtype=torch.float32)
        expected_k = torch.zeros_like(k, dtype=torch.float32)
        start = 0
        for sequence, length in enumerate(q_lengths):
            expected_k[sequence] = do[start:start + length].float().sum(dim=0)
            start += length
        assert torch.count_nonzero(dq) == 0
        _assert_cross_attention_fp32_close(dq, expected_q)
        _assert_cross_attention_fp32_close(dkv, expected_k)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
@pytest.mark.parametrize("cap", [384, 512, 2048])
@pytest.mark.parametrize("alpha", [1.0 / 128, -1.0 / 128])
def test_hstu_cross_attn_gfx950_coarse_single_key_gradient(monkeypatch, cap, alpha):
    """A one-key softmax has zero score derivatives for every query."""
    tutorial = Path(__file__).resolve().parents[4] / "third_party/tlx/tutorials/hstu_cross_attn"
    monkeypatch.syspath_prepend(str(tutorial))
    xa = importlib.import_module("tlx_gfx950_cross_attention")
    monkeypatch.setattr(xa._tlx_gfx950_cross_attn_fwd, "configs", [xa.get_fwd_triton_spec_configs()[0]])
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    q_lengths = ([0, 1, 31, 32, 33, 127, 255, 256] * 9)[:64]
    q_offsets = torch.tensor([0] + q_lengths, device="cuda", dtype=torch.int64).cumsum(0)
    kv_offsets = torch.arange(65, device="cuda", dtype=torch.int64)
    torch.manual_seed(8)
    q = torch.randn(sum(q_lengths), 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(64, 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = xa.tlx_gfx950_cross_attn_mha_wrapper(max_seq_len=cap, alpha=alpha, q=q, k=k, v=k, seq_offsets=kv_offsets,
                                               attn_scale=torch.tensor(1.0 / cap, device="cuda"), max_q_len=256,
                                               seq_offsets_q=q_offsets, num_softmax_heads=1, shared_kv=True,
                                               causal=False,
                                               num_targets=torch.tensor(q_lengths, device="cuda", dtype=torch.int64),
                                               enable_tma=False, v3_ttgir_variant="v3_ttgir_fp32_pipeline_qdo")
    assert out.grad_fn.use_small_backward and out.grad_fn.use_coarse_backward
    assert not out.grad_fn.use_retained_backward
    for call in range(2):
        do = 0.1 * torch.randn_like(q) if call == 0 else (
            0.1 * torch.randn(q.shape[0], 1, 256, device="cuda", dtype=q.dtype))[..., ::2]
        dq, dkv = torch.autograd.grad(out, (q, k), do, retain_graph=True)
        expected_q = torch.zeros_like(q, dtype=torch.float32)
        expected_k = torch.zeros_like(k, dtype=torch.float32)
        start = 0
        for sequence, length in enumerate(q_lengths):
            expected_k[sequence] = do[start:start + length].float().sum(dim=0)
            start += length
        assert torch.count_nonzero(dq) == 0
        _assert_cross_attention_fp32_close(dq, expected_q)
        _assert_cross_attention_fp32_close(dkv, expected_k)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
@pytest.mark.parametrize("batch,cap,heads,shared_kv", [
    (64, 128, 1, True),
    (64, 256, 1, True),
    (4, 2048, 1, False),
    (64, 512, 2, False),
    (64, 512, 2, True),
])
@pytest.mark.parametrize("alpha", [1.0 / 128, -1.0 / 128])
def test_hstu_cross_attn_gfx950_small_single_key_gradient(monkeypatch, batch, cap, heads, shared_kv, alpha):
    """One-key owners produce zero score gradients and preserve value gradients."""
    tutorial = Path(__file__).resolve().parents[4] / "third_party/tlx/tutorials/hstu_cross_attn"
    monkeypatch.syspath_prepend(str(tutorial))
    xa = importlib.import_module("tlx_gfx950_cross_attention")
    monkeypatch.setattr(xa._tlx_gfx950_cross_attn_fwd, "configs", [xa.get_fwd_triton_spec_configs()[0]])
    q_lengths = ([0, 1, 31, 32, 33, 127, 255, 256] * ((batch + 7) // 8))[:batch]
    q_offsets = torch.tensor([0] + q_lengths, device="cuda", dtype=torch.int64).cumsum(0)
    kv_offsets = torch.arange(batch + 1, device="cuda", dtype=torch.int64)
    torch.manual_seed(8)
    q = torch.randn(sum(q_lengths), heads, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(batch, heads, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = k if shared_kv else torch.randn_like(k, requires_grad=True)
    out = xa.tlx_gfx950_cross_attn_mha_wrapper(max_seq_len=cap, alpha=alpha, q=q, k=k, v=v, seq_offsets=kv_offsets,
                                               attn_scale=torch.tensor(1.0 / cap, device="cuda"), max_q_len=256,
                                               seq_offsets_q=q_offsets, num_softmax_heads=heads, shared_kv=shared_kv,
                                               causal=False,
                                               num_targets=torch.tensor(q_lengths, device="cuda", dtype=torch.int64),
                                               enable_tma=False, v3_ttgir_variant="v3_ttgir_fp32_pipeline_qdo")
    assert out.grad_fn.use_small_backward
    assert not out.grad_fn.use_coarse_backward and not out.grad_fn.use_compact_backward
    assert not out.grad_fn.use_partial_backward and not out.grad_fn.use_retained_backward
    leaves = (q, k) if shared_kv else (q, k, v)
    for call in range(2):
        do = 0.1 * torch.randn_like(q) if call == 0 else (
            0.1 * torch.randn(q.shape[0], heads, 256, device="cuda", dtype=q.dtype))[..., ::2]
        gradients = torch.autograd.grad(out, leaves, do, retain_graph=True)
        expected_q = torch.zeros_like(q, dtype=torch.float32)
        expected_v = torch.zeros_like(v, dtype=torch.float32)
        start = 0
        for sequence, length in enumerate(q_lengths):
            expected_v[sequence] = do[start:start + length].float().sum(dim=0)
            start += length
        expected = (expected_q, expected_v) if shared_kv else (expected_q, torch.zeros_like(k, dtype=torch.float32),
                                                               expected_v)
        assert torch.count_nonzero(gradients[0]) == 0
        if not shared_kv:
            assert torch.count_nonzero(gradients[1]) == 0
        for actual, reference in zip(gradients, expected):
            _assert_cross_attention_fp32_close(actual, reference)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
@pytest.mark.parametrize("variant", [None, "v3_ttgir_fp32_pipeline_qdo"])
@pytest.mark.parametrize("cap", [256, 512, 1024, 2048])
@pytest.mark.parametrize("alpha", [0.0, 1.0 / 128, -1.0 / 128])
def test_hstu_cross_attn_gfx950_retained_single_key_reuse(monkeypatch, variant, cap, alpha):
    """Keep one-key score gradients zero and reduce shared dKV in FP32."""
    tutorial = Path(__file__).resolve().parents[4] / "third_party/tlx/tutorials/hstu_cross_attn"
    monkeypatch.syspath_prepend(str(tutorial))
    xa = importlib.import_module("tlx_gfx950_cross_attention")
    retained = importlib.import_module("tlx_gfx950_cross_attention_retained")
    monkeypatch.setattr(xa._tlx_gfx950_cross_attn_fwd, "configs", [xa.get_fwd_triton_spec_configs()[0]])
    q_lengths = [0, 1, 31, 32, 33, 127, 255, 256] * 32
    kv_lengths = [1, 0, 1, 1, 1, 1, 0, 1] * 32
    q_offsets = torch.tensor([0] + q_lengths, device="cuda", dtype=torch.int64).cumsum(0)
    kv_offsets = torch.tensor([0] + kv_lengths, device="cuda", dtype=torch.int64).cumsum(0)
    for context in range(2):
        torch.manual_seed(8 + context)
        q = torch.randn(sum(q_lengths), 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(sum(kv_lengths), 1, 128, device="cuda", dtype=q.dtype, requires_grad=True)
        out = xa.tlx_gfx950_cross_attn_mha_wrapper(
            max_seq_len=cap, alpha=alpha, q=q, k=k, v=k, seq_offsets=kv_offsets,
            attn_scale=torch.tensor(1.0 / cap, device="cuda"), max_q_len=256, seq_offsets_q=q_offsets,
            num_softmax_heads=1, num_targets=torch.tensor(q_lengths, device="cuda", dtype=torch.int64), causal=False,
            shared_kv=True, enable_tma=False, v3_ttgir_variant=variant)
        assert out.grad_fn.use_retained_backward
        for change in range(2):
            do = torch.randn(sum(q_lengths), 1, 256, device="cuda", dtype=q.dtype)[:, :, ::2]
            assert not do.is_contiguous()
            dq, dkv = torch.autograd.grad(out, (q, k), do, retain_graph=True)
            expected = torch.zeros_like(k, dtype=torch.float32)
            qs = ks = 0
            for q_len, kv_len in zip(q_lengths, kv_lengths):
                if kv_len:
                    expected[ks] = do[qs:qs + q_len].float().sum(0)
                qs += q_len
                ks += kv_len
            torch.testing.assert_close(dq, torch.zeros_like(dq), rtol=0, atol=0)
            _assert_cross_attention_fp32_close(dkv, expected)
            # One-key and empty-KV branches must not consume softmax statistics.
            poisoned_m = torch.full(q.shape[:2], float("nan"), device=q.device, dtype=torch.float32)
            poisoned_out = torch.full_like(q, float("nan"))
            direct_dq, direct_dkv = retained.retained_softmax_backward(q, k, do, poisoned_m, poisoned_out, q_offsets,
                                                                       kv_offsets, alpha)
            torch.testing.assert_close(direct_dq, dq, rtol=0, atol=0)
            torch.testing.assert_close(direct_dkv, dkv, rtol=0, atol=0)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
@pytest.mark.parametrize("qk_scale", [1.0, 1e20])
@pytest.mark.parametrize("alpha", [0.0, 1.0 / 128, -1.0 / 128, 3e38])
def test_hstu_cross_attn_gfx950_retained_single_key_finite_extremes(monkeypatch, qk_scale, alpha):
    """Keep finite one-key inputs independent of overflowing score products."""
    tutorial = Path(__file__).resolve().parents[4] / "third_party/tlx/tutorials/hstu_cross_attn"
    monkeypatch.syspath_prepend(str(tutorial))
    retained = importlib.import_module("tlx_gfx950_cross_attention_retained")
    q_lengths = [0, 1, 31, 256]
    kv_lengths = [1, 0, 1, 1]
    q_offsets = torch.tensor([0] + q_lengths, device="cuda", dtype=torch.int64).cumsum(0)
    kv_offsets = torch.tensor([0] + kv_lengths, device="cuda", dtype=torch.int64).cumsum(0)
    q = torch.full((sum(q_lengths), 1, 128), qk_scale, device="cuda", dtype=torch.bfloat16)
    k = torch.full((sum(kv_lengths), 1, 128), qk_scale, device="cuda", dtype=q.dtype)
    do = torch.full((sum(q_lengths), 1, 256), 0.25, device="cuda", dtype=q.dtype)[:, :, ::2]
    lse = torch.full(q.shape[:2], float("nan"), device="cuda", dtype=torch.float32)
    out = torch.full_like(q, float("nan"))
    dq, dkv = retained.retained_softmax_backward(q, k, do, lse, out, q_offsets, kv_offsets, alpha)
    expected = torch.tensor([0.0, 31 * 0.25, 256 * 0.25], device="cuda")[:, None, None].expand_as(k)
    torch.testing.assert_close(dq, torch.zeros_like(dq), rtol=0, atol=0)
    torch.testing.assert_close(dkv.float(), expected, rtol=0, atol=0)

    # Check public cap256 dispatch with the same poisoned statistics.
    xa = importlib.import_module("tlx_gfx950_cross_attention")
    q.requires_grad_(True)
    k.requires_grad_(True)
    q_offsets = torch.cat((q_offsets, q_offsets[-1:].expand(252)))
    kv_offsets = torch.cat((kv_offsets, kv_offsets[-1:].expand(252)))
    monkeypatch.setattr(xa, "tlx_gfx950_cross_attn_fwd", lambda **kwargs: (out, lse))
    public_out = xa.tlx_gfx950_cross_attn_mha_wrapper(
        max_seq_len=256, alpha=alpha, q=q, k=k, v=k, seq_offsets=kv_offsets,
        attn_scale=torch.tensor(1.0 / 256, device="cuda"), max_q_len=256, seq_offsets_q=q_offsets, num_softmax_heads=1,
        num_targets=q_offsets[1:] - q_offsets[:-1], causal=False, shared_kv=True, enable_tma=False,
        v3_ttgir_variant="v3_ttgir_fp32_pipeline_qdo")
    assert public_out.grad_fn.use_retained_backward
    public_dq, public_dkv = torch.autograd.grad(public_out, (q, k), do)
    torch.testing.assert_close(public_dq, torch.zeros_like(public_dq), rtol=0, atol=0)
    torch.testing.assert_close(public_dkv.float(), expected, rtol=0, atol=0)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
@pytest.mark.parametrize("cap", [384, 512, 1024, 2048])
@pytest.mark.parametrize("value", [1.0, 1e20])
@pytest.mark.parametrize("alpha", [0.0, 1.0 / 128, -1.0 / 128, 3e38])
def test_hstu_cross_attn_gfx950_coarse_single_key_finite_extreme(monkeypatch, cap, value, alpha):
    """One-key gradients do not depend on overflowing scores or saved statistics."""
    tutorial = Path(__file__).resolve().parents[4] / "third_party/tlx/tutorials/hstu_cross_attn"
    monkeypatch.syspath_prepend(str(tutorial))
    small = importlib.import_module("tlx_gfx950_cross_attention_small")
    lengths = [0, 1, 31, 256] * 16
    q_offsets = torch.tensor([0] + lengths, device="cuda", dtype=torch.int64).cumsum(0)
    kv_offsets = torch.arange(65, device="cuda", dtype=torch.int64)
    q = torch.full((sum(lengths), 1, 128), value, device="cuda", dtype=torch.bfloat16)
    k = torch.full((64, 1, 128), value, device="cuda", dtype=q.dtype)
    m = torch.full(q.shape[:2], float("nan"), device="cuda", dtype=torch.float32)
    out = torch.full_like(q, float("nan"))
    for do_value in [0.25, -0.125]:
        do = torch.full((1, 1, 128), do_value, device="cuda", dtype=q.dtype).expand_as(q)
        dq, dkv, _ = small.coarse_softmax_backward(q, k, do, m, out, q_offsets, kv_offsets, alpha, cache_delta=cap
                                                   > 1024, early_stats=cap == 1024)
        expected = torch.tensor(lengths, device="cuda", dtype=torch.float32)[:, None, None] * do_value
        expected = expected.expand_as(dkv)
        assert torch.count_nonzero(dq) == 0
        _assert_cross_attention_fp32_close(dq, torch.zeros_like(q, dtype=torch.float32))
        _assert_cross_attention_fp32_close(dkv, expected)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
@pytest.mark.parametrize("variant", [None, "v3_ttgir_fp32_pipeline_qdo"])
@pytest.mark.parametrize("cap", [256, 512, 1024, 2048])
@pytest.mark.parametrize("alpha", [1.0 / 128, -1.0 / 128])
def test_hstu_cross_attn_gfx950_retained_fixed_query_reuse(monkeypatch, variant, cap, alpha):
    tutorial = Path(__file__).resolve().parents[4] / "third_party/tlx/tutorials/hstu_cross_attn"
    monkeypatch.syspath_prepend(str(tutorial))
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    xa = importlib.import_module("tlx_gfx950_cross_attention")
    monkeypatch.setattr(xa._tlx_gfx950_cross_attn_fwd, "configs", [xa.get_fwd_triton_spec_configs()[0]])
    q_lengths = [256] * 256
    kv_lengths = [min(257, cap), 0, 1, cap, 129] + [0] * 251
    q_offsets = torch.tensor([0] + q_lengths, device="cuda", dtype=torch.int64).cumsum(0)
    kv_offsets = torch.tensor([0] + kv_lengths, device="cuda", dtype=torch.int64).cumsum(0)
    retained = importlib.import_module("tlx_gfx950_cross_attention_retained")
    selected = []
    original_run = retained._retained_softmax_backward.run

    def observe(*args, **kwargs):
        selected.append(kwargs.get("FIXED_Q"))
        return original_run(*args, **kwargs)

    monkeypatch.setattr(retained._retained_softmax_backward, "run", observe)
    for context_index in range(2):
        torch.manual_seed(context_index)
        q = torch.randn(sum(q_lengths), 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(sum(kv_lengths), 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        q_ref = q.detach().float().requires_grad_(True)
        k_ref = k.detach().float().requires_grad_(True)
        outputs = []
        q_start = kv_start = 0
        for q_len, kv_len in zip(q_lengths, kv_lengths):
            qs = q_ref[q_start:q_start + q_len, 0]
            ks = k_ref[kv_start:kv_start + kv_len, 0]
            outputs.append((torch.softmax((qs @ ks.T) * alpha, dim=-1) @ ks).unsqueeze(1))
            q_start += q_len
            kv_start += kv_len
        reference = torch.cat(outputs)
        out = xa.tlx_gfx950_cross_attn_mha_wrapper(
            max_seq_len=cap, alpha=alpha, q=q, k=k, v=k, seq_offsets=kv_offsets,
            attn_scale=torch.tensor(1.0 / cap, device="cuda"), max_q_len=256, seq_offsets_q=q_offsets,
            num_softmax_heads=1, num_targets=torch.tensor(q_lengths, device="cuda", dtype=torch.int64), causal=False,
            shared_kv=True, enable_tma=False, v3_ttgir_variant=variant)
        assert out.grad_fn.use_retained_backward
        assert not out.grad_fn.use_small_backward
        _assert_cross_attention_fp32_close(out, reference)
        for call in range(2):
            do = (0.1 * torch.randn(out.shape[0], 1, 256, device="cuda", dtype=torch.bfloat16))[:, :, ::2]
            if call == 1:
                do = do[:1, :, :1].expand_as(out)
                assert do.stride(0) == 0
            assert not do.is_contiguous()
            actual = torch.autograd.grad(out, (q, k), do, retain_graph=True)
            expected = torch.autograd.grad(reference, (q_ref, k_ref), do.float(), retain_graph=True)
            for got, want in zip(actual, expected):
                _assert_cross_attention_fp32_close(got, want)
            torch.testing.assert_close(actual[0][256:768], torch.zeros_like(actual[0][256:768]), atol=0, rtol=0)
    assert selected and all(selected)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
@pytest.mark.parametrize("cap", [128])
@pytest.mark.parametrize("alpha", [1.0 / 128, -1.0 / 128])
def test_hstu_cross_attn_gfx950_generic_packed_query_reuse(monkeypatch, cap, alpha):
    tutorial = Path(__file__).resolve().parents[4] / "third_party/tlx/tutorials/hstu_cross_attn"
    monkeypatch.syspath_prepend(str(tutorial))
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    xa = importlib.import_module("tlx_gfx950_cross_attention")
    monkeypatch.setattr(xa._tlx_gfx950_cross_attn_fwd, "configs", [xa.get_fwd_triton_spec_configs()[0]])
    q_lengths = [256] * 65
    kv_lengths = [cap, 0, 1, cap, 65] + [0] * 60
    q_offsets = torch.tensor([0] + q_lengths, device="cuda", dtype=torch.int64).cumsum(0)
    kv_offsets = torch.tensor([0] + kv_lengths, device="cuda", dtype=torch.int64).cumsum(0)
    retained = importlib.import_module("tlx_gfx950_cross_attention_v3_baseline")
    selected = []
    original_run = retained._tlx_gfx950_cross_attn_v3_ttgir_bwd.run

    def observe(*args, **kwargs):
        selected.append(kwargs.get("PACKED_FIXED_Q"))
        return original_run(*args, **kwargs)

    monkeypatch.setattr(retained._tlx_gfx950_cross_attn_v3_ttgir_bwd, "run", observe)
    for context_index in range(2):
        torch.manual_seed(context_index)
        q = torch.randn(sum(q_lengths), 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(sum(kv_lengths), 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        q_ref = q.detach().float().requires_grad_(True)
        k_ref = k.detach().float().requires_grad_(True)
        outputs = []
        q_start = kv_start = 0
        for q_len, kv_len in zip(q_lengths, kv_lengths):
            qs = q_ref[q_start:q_start + q_len, 0]
            ks = k_ref[kv_start:kv_start + kv_len, 0]
            outputs.append((torch.softmax((qs @ ks.T) * alpha, dim=-1) @ ks).unsqueeze(1))
            q_start += q_len
            kv_start += kv_len
        reference = torch.cat(outputs)
        out = xa.tlx_gfx950_cross_attn_mha_wrapper(
            max_seq_len=cap, alpha=alpha, q=q, k=k, v=k, seq_offsets=kv_offsets,
            attn_scale=torch.tensor(1.0 / cap, device="cuda"), max_q_len=256, seq_offsets_q=q_offsets,
            num_softmax_heads=1, num_targets=torch.tensor(q_lengths, device="cuda", dtype=torch.int64), causal=False,
            shared_kv=True, enable_tma=False, v3_ttgir_variant="v3_ttgir_fp32_pipeline_qdo")
        assert not out.grad_fn.use_retained_backward
        assert not out.grad_fn.use_small_backward
        _assert_cross_attention_fp32_close(out, reference)
        for call in range(2):
            do = (0.1 * torch.randn(out.shape[0], 1, 256, device="cuda", dtype=torch.bfloat16))[:, :, ::2]
            if call == 1:
                do = do[:1, :, :1].expand_as(out)
                assert do.stride(0) == 0
            assert not do.is_contiguous()
            actual = torch.autograd.grad(out, (q, k), do, retain_graph=True)
            assert selected and selected[-1] is True
            expected = torch.autograd.grad(reference, (q_ref, k_ref), do.float(), retain_graph=True)
            for got, want in zip(actual, expected):
                _assert_cross_attention_fp32_close(got, want)
            torch.testing.assert_close(actual[0][256:768], torch.zeros_like(actual[0][256:768]), atol=0, rtol=0)
    assert selected and all(selected)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
@pytest.mark.parametrize("ragged_q", [False, True])
@pytest.mark.parametrize("alpha", [1.0 / 128, -1.0 / 128])
def test_hstu_cross_attn_gfx950_native_query_view_reuse(monkeypatch, ragged_q, alpha):
    tutorial = Path(__file__).resolve().parents[4] / "third_party/tlx/tutorials/hstu_cross_attn"
    monkeypatch.syspath_prepend(str(tutorial))
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    xa = importlib.import_module("tlx_gfx950_cross_attention")
    monkeypatch.setattr(xa._tlx_gfx950_cross_attn_fwd, "configs", [xa.get_fwd_triton_spec_configs()[0]])
    cap = 384
    q_lengths = [0, 1, 31, 33, 65] + [256] * 60 if ragged_q else [256] * 65
    kv_lengths = [cap, 0, 1, cap, 65] + [0] * 60
    q_offsets = torch.tensor([0] + q_lengths, device="cuda", dtype=torch.int64).cumsum(0)
    kv_offsets = torch.tensor([0] + kv_lengths, device="cuda", dtype=torch.int64).cumsum(0)
    retained = importlib.import_module("tlx_gfx950_cross_attention_v3_baseline")
    selected = []
    original_run = retained._tlx_gfx950_cross_attn_v3_ttgir_bwd.run

    def observe(*args, **kwargs):
        selected.append((kwargs.get("PACKED_FIXED_Q"), kwargs.get("NATIVE_Q_SCORE")))
        return original_run(*args, **kwargs)

    monkeypatch.setattr(retained._tlx_gfx950_cross_attn_v3_ttgir_bwd, "run", observe)
    for context_index in range(2):
        torch.manual_seed(context_index)
        q = torch.randn(sum(q_lengths), 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(sum(kv_lengths), 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        q_ref = q.detach().float().requires_grad_(True)
        k_ref = k.detach().float().requires_grad_(True)
        outputs = []
        q_start = kv_start = 0
        for q_len, kv_len in zip(q_lengths, kv_lengths):
            qs = q_ref[q_start:q_start + q_len, 0]
            ks = k_ref[kv_start:kv_start + kv_len, 0]
            outputs.append((torch.softmax((qs @ ks.T) * alpha, dim=-1) @ ks).unsqueeze(1))
            q_start += q_len
            kv_start += kv_len
        reference = torch.cat(outputs)
        out = xa.tlx_gfx950_cross_attn_mha_wrapper(
            max_seq_len=cap, alpha=alpha, q=q, k=k, v=k, seq_offsets=kv_offsets,
            attn_scale=torch.tensor(1.0 / cap, device="cuda"), max_q_len=256, seq_offsets_q=q_offsets,
            num_softmax_heads=1, num_targets=torch.tensor(q_lengths, device="cuda", dtype=torch.int64), causal=False,
            shared_kv=True, enable_tma=False, v3_ttgir_variant="v3_ttgir_fp32_pipeline_qdo")
        assert not out.grad_fn.use_retained_backward
        assert not out.grad_fn.use_small_backward
        _assert_cross_attention_fp32_close(out, reference)
        for call in range(2):
            do = (0.1 * torch.randn(out.shape[0], 1, 256, device="cuda", dtype=torch.bfloat16))[:, :, ::2]
            if call == 1:
                do = do[:1, :, :1].expand_as(out)
                assert do.stride(0) == 0
            assert not do.is_contiguous()
            actual = torch.autograd.grad(out, (q, k), do, retain_graph=True)
            assert selected and selected[-1] == (not ragged_q, not ragged_q)
            expected = torch.autograd.grad(reference, (q_ref, k_ref), do.float(), retain_graph=True)
            for got, want in zip(actual, expected):
                _assert_cross_attention_fp32_close(got, want)
            torch.testing.assert_close(actual[0][q_lengths[0]:sum(q_lengths[:3])],
                                       torch.zeros_like(actual[0][q_lengths[0]:sum(q_lengths[:3])]), atol=0, rtol=0)
    assert selected and all(flags == (not ragged_q, not ragged_q) for flags in selected)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
@pytest.mark.parametrize("variant", [None, "v3_ttgir_fp32_pipeline_qdo"])
@pytest.mark.parametrize("alpha", [1.0 / 128, -1.0 / 128])
def test_hstu_cross_attn_gfx950_cached_delta_1024_reuse(monkeypatch, variant, alpha):
    tutorial = Path(__file__).resolve().parents[4] / "third_party/tlx/tutorials/hstu_cross_attn"
    monkeypatch.syspath_prepend(str(tutorial))
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    xa = importlib.import_module("tlx_gfx950_cross_attention")
    monkeypatch.setattr(xa._tlx_gfx950_cross_attn_fwd, "configs", [xa.get_fwd_triton_spec_configs()[0]])
    cap = 1024
    q_lengths = [0, 1, 31, 33, 65, 127, 129, 255, 256] + [256] * 55
    kv_lengths = [1024, 0, 1, 1023, 513, 512, 257, 128, 65] + [0] * 55
    q_offsets = torch.tensor([0] + q_lengths, device="cuda", dtype=torch.int64).cumsum(0)
    kv_offsets = torch.tensor([0] + kv_lengths, device="cuda", dtype=torch.int64).cumsum(0)
    for context_index in range(2):
        torch.manual_seed(context_index)
        q = torch.randn(sum(q_lengths), 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(sum(kv_lengths), 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        q_ref = q.detach().float().requires_grad_(True)
        k_ref = k.detach().float().requires_grad_(True)
        outputs = []
        q_start = kv_start = 0
        for q_len, kv_len in zip(q_lengths, kv_lengths):
            qs = q_ref[q_start:q_start + q_len, 0]
            ks = k_ref[kv_start:kv_start + kv_len, 0]
            outputs.append((torch.softmax((qs @ ks.T) * alpha, dim=-1) @ ks).unsqueeze(1))
            q_start += q_len
            kv_start += kv_len
        reference = torch.cat(outputs)
        out = xa.tlx_gfx950_cross_attn_mha_wrapper(
            max_seq_len=cap, alpha=alpha, q=q, k=k, v=k, seq_offsets=kv_offsets,
            attn_scale=torch.tensor(1.0 / cap, device="cuda"), max_q_len=256, seq_offsets_q=q_offsets,
            num_softmax_heads=1, num_targets=torch.tensor(q_lengths, device="cuda", dtype=torch.int64), causal=False,
            shared_kv=True, enable_tma=False, v3_ttgir_variant=variant)
        assert not out.grad_fn.use_retained_backward
        assert out.grad_fn.use_small_backward and out.grad_fn.use_coarse_backward
        assert out.grad_fn.cache_coarse_delta
        _assert_cross_attention_fp32_close(out, reference)
        for call in range(2):
            do = (0.1 * torch.randn(out.shape[0], 1, 256, device="cuda", dtype=torch.bfloat16))[:, :, ::2]
            if call == 1:
                do = do[:1, :, :1].expand_as(out)
                assert do.stride(0) == 0
            assert not do.is_contiguous()
            actual = torch.autograd.grad(out, (q, k), do, retain_graph=True)
            expected = torch.autograd.grad(reference, (q_ref, k_ref), do.float(), retain_graph=True)
            for got, want in zip(actual, expected):
                _assert_cross_attention_fp32_close(got, want)
            torch.testing.assert_close(actual[0][q_lengths[0]:sum(q_lengths[:3])],
                                       torch.zeros_like(actual[0][q_lengths[0]:sum(q_lengths[:3])]), atol=0, rtol=0)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
@pytest.mark.parametrize("ragged_q", [False, True])
@pytest.mark.parametrize("ragged_k", [False, True])
@pytest.mark.parametrize("alpha", [1.0 / 128, -1.0 / 128])
def test_hstu_cross_attn_gfx950_packed_kv384_reuse(monkeypatch, ragged_q, ragged_k, alpha):
    tutorial = Path(__file__).resolve().parents[4] / "third_party/tlx/tutorials/hstu_cross_attn"
    monkeypatch.syspath_prepend(str(tutorial))
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    xa = importlib.import_module("tlx_gfx950_cross_attention")
    monkeypatch.setattr(xa._tlx_gfx950_cross_attn_fwd, "configs", [xa.get_fwd_triton_spec_configs()[0]])
    cap = 384
    q_lengths = [0, 1, 31, 33, 65] + [256] * 60 if ragged_q else [256] * 65
    kv_lengths = [cap, 0, 1, cap, 65] + [0] * 60 if ragged_k else [cap] * 65
    q_offsets = torch.tensor([0] + q_lengths, device="cuda", dtype=torch.int64).cumsum(0)
    kv_offsets = torch.tensor([0] + kv_lengths, device="cuda", dtype=torch.int64).cumsum(0)
    retained = importlib.import_module("tlx_gfx950_cross_attention_v3_baseline")
    selected = []
    original_run = retained._tlx_gfx950_cross_attn_v3_ttgir_bwd.run

    def observe(*args, **kwargs):
        selected.append((kwargs.get("PACKED_FIXED_Q"), kwargs.get("NATIVE_Q_SCORE"), kwargs.get("FIXED_KV")))
        return original_run(*args, **kwargs)

    monkeypatch.setattr(retained._tlx_gfx950_cross_attn_v3_ttgir_bwd, "run", observe)
    for context_index in range(2):
        torch.manual_seed(context_index)
        q = torch.randn(sum(q_lengths), 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(sum(kv_lengths), 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        q_ref = q.detach().float().requires_grad_(True)
        k_ref = k.detach().float().requires_grad_(True)
        outputs = []
        q_start = kv_start = 0
        for q_len, kv_len in zip(q_lengths, kv_lengths):
            qs = q_ref[q_start:q_start + q_len, 0]
            ks = k_ref[kv_start:kv_start + kv_len, 0]
            outputs.append((torch.softmax((qs @ ks.T) * alpha, dim=-1) @ ks).unsqueeze(1))
            q_start += q_len
            kv_start += kv_len
        reference = torch.cat(outputs)
        out = xa.tlx_gfx950_cross_attn_mha_wrapper(
            max_seq_len=cap, alpha=alpha, q=q, k=k, v=k, seq_offsets=kv_offsets,
            attn_scale=torch.tensor(1.0 / cap, device="cuda"), max_q_len=256, seq_offsets_q=q_offsets,
            num_softmax_heads=1, num_targets=torch.tensor(q_lengths, device="cuda", dtype=torch.int64), causal=False,
            shared_kv=True, enable_tma=False, v3_ttgir_variant="v3_ttgir_fp32_pipeline_qdo")
        assert not out.grad_fn.use_retained_backward
        assert not out.grad_fn.use_small_backward
        _assert_cross_attention_fp32_close(out, reference)
        for call in range(2):
            do = (0.1 * torch.randn(out.shape[0], 1, 256, device="cuda", dtype=torch.bfloat16))[:, :, ::2]
            if call == 1:
                do = do[:1, :, :1].expand_as(out)
                assert do.stride(0) == 0
            assert not do.is_contiguous()
            actual = torch.autograd.grad(out, (q, k), do, retain_graph=True)
            assert selected and selected[-1] == (not ragged_q, not ragged_q,
                                                 384 if not ragged_q and not ragged_k else 0)
            expected = torch.autograd.grad(reference, (q_ref, k_ref), do.float(), retain_graph=True)
            for got, want in zip(actual, expected):
                _assert_cross_attention_fp32_close(got, want)
            if ragged_k:
                torch.testing.assert_close(actual[0][q_lengths[0]:sum(q_lengths[:3])],
                                           torch.zeros_like(actual[0][q_lengths[0]:sum(q_lengths[:3])]), atol=0, rtol=0)

    assert selected and all(flags == (not ragged_q, not ragged_q, 384 if not ragged_q and not ragged_k else 0)
                            for flags in selected)
