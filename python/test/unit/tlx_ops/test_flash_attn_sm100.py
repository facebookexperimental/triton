"""L1 correctness for `tlx.ops.flash_attn`.

No compile-time cap as in test_mm.py: this kernel has no shape heuristic, so the
correctness caller autotunes a small space and wall-clock says nothing about
compile time.

TODO: cover the config variants dropped with the tutorial copy of this kernel --
RESCALE_OPT/USE_WHERE, the FAST_FIXED opt-out, NUM_CTAS=2, and USE_WARP_BARRIER.
Nothing tests them today.
"""
import pytest
import torch
from triton._internal_testing import is_blackwell
from triton.tlx.ops.kernels.flash_attn._shapes import CORRECTNESS_SHAPES, SYNTHETIC

pytestmark = pytest.mark.skipif(not is_blackwell(), reason="tlx.ops.flash_attn is sm100-only today")

torch.manual_seed(0)

REL_PRECISION = {torch.float16: 1e-3, torch.bfloat16: 8e-3}
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
FWD_SHAPES = tuple(dict.fromkeys((*CORRECTNESS_SHAPES, *(shape._replace(dtype="bf16") for shape in SYNTHETIC))))


def _qkv(Z, H, N_CTX, HEAD_DIM, dtype, requires_grad=False):
    return [
        torch.randn((Z, H, N_CTX, HEAD_DIM), device="cuda", dtype=dtype).requires_grad_(requires_grad) for _ in range(3)
    ]


def _sdpa(q, k, v, causal):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=q.shape[-1]**-0.5)


def _sdpa_fp32_grads(q, k, v, do, causal):
    grads = [torch.empty(q.shape, device=q.device) for _ in range(3)]
    for b in range(q.shape[0]):
        for h in range(q.shape[1]):
            inputs = [x[b, h].detach().float().requires_grad_() for x in (q, k, v)]
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                out = _sdpa(*inputs, causal)
            for grad, value in zip(grads, torch.autograd.grad(out, inputs, do[b, h].float())):
                grad[b, h] = value
    return grads


def _assert_close(out, ref, dtype):
    precision = REL_PRECISION[dtype]
    torch.testing.assert_close(out, ref, atol=precision * ref.abs().max().item(), rtol=precision)


@pytest.mark.parametrize("Z,H,N_CTX,HEAD_DIM,causal,dtype_name", FWD_SHAPES)
def test_flash_attn_fwd(Z, H, N_CTX, HEAD_DIM, causal, dtype_name):
    from triton.tlx.ops import flash_attn as tlx_flash_attn

    if N_CTX % 128 or HEAD_DIM not in (64, 128):
        pytest.skip(f"sm100 takes N_CTX % 128 == 0 and HEAD_DIM 64/128, got {N_CTX}x{HEAD_DIM}")
    dtype = DTYPES[dtype_name]
    q, k, v = _qkv(Z, H, N_CTX, HEAD_DIM, dtype)
    out = tlx_flash_attn(q, k, v, causal=causal, space="smoke")
    _assert_close(out, _sdpa(q, k, v, causal), dtype)


@pytest.mark.parametrize("Z,H,N_CTX,HEAD_DIM,causal,dtype_name", CORRECTNESS_SHAPES)
def test_flash_attn_bwd(Z, H, N_CTX, HEAD_DIM, causal, dtype_name):
    """The kernel carries the full backward path even though the op defers it."""
    from triton.tlx.ops import flash_attn as tlx_flash_attn

    if N_CTX % 128 or HEAD_DIM not in (64, 128):
        pytest.skip(f"sm100 takes N_CTX % 128 == 0 and HEAD_DIM 64/128, got {N_CTX}x{HEAD_DIM}")
    dtype = DTYPES[dtype_name]
    q, k, v = _qkv(Z, H, N_CTX, HEAD_DIM, dtype, requires_grad=True)
    rq, rk, rv = (t.detach().clone().requires_grad_() for t in (q, k, v))
    do = torch.randn_like(q)

    tlx_flash_attn(q, k, v, causal=causal, space="smoke").backward(do)
    _sdpa(rq, rk, rv, causal).backward(do)

    for got, want in ((q.grad, rq.grad), (k.grad, rk.grad), (v.grad, rv.grad)):
        _assert_close(got, want, dtype)


@pytest.mark.parametrize("N_CTX", [512, 768, 1280])
def test_flash_attn_fwd_2cta_grid(N_CTX):
    """A 2-CTA cluster covers 2 * BLOCK_M rows; the grid must launch one cluster per such tile."""
    from triton.tlx.ops.kernels.flash_attn import sm100

    config = next(c for c in sm100.configs if c.kwargs.get("NUM_CTAS", 1) == 2).kwargs
    q, k, v = _qkv(2, 4, N_CTX, 128, torch.bfloat16)
    out = sm100.attention(q, k, v, 128**-0.5, False, config=dict(config))
    _assert_close(out, _sdpa(q, k, v, False), torch.bfloat16)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("N_CTX", [384, 640])
def test_flash_attn_partial_last_m_block(N_CTX, causal):
    """N_CTX not a multiple of the forward BLOCK_M: the last M-block must not spill into the next head."""
    from triton.tlx.ops import flash_attn as tlx_flash_attn

    dtype = torch.bfloat16
    q, k, v = _qkv(2, 4, N_CTX, 128, dtype, requires_grad=True)
    rq, rk, rv = (t.detach().clone().requires_grad_() for t in (q, k, v))
    do = torch.randn_like(q)

    out = tlx_flash_attn(q, k, v, causal=causal, space="smoke")
    ref = _sdpa(rq, rk, rv, causal)
    _assert_close(out, ref, dtype)
    out.backward(do)
    ref.backward(do)
    for got, want in ((q.grad, rq.grad), (k.grad, rk.grad), (v.grad, rv.grad)):
        _assert_close(got, want, dtype)


@pytest.mark.parametrize("Z, H, N_CTX, HEAD_DIM", [[2, 4, 1024, 128], [1, 16, 4096, 128], [2, 32, 2048, 64]])
def test_flash_attn_bwd_bf16_accuracy(Z, H, N_CTX, HEAD_DIM):
    from triton.tlx.ops import flash_attn as tlx_flash_attn

    q, k, v = _qkv(Z, H, N_CTX, HEAD_DIM, torch.bfloat16, requires_grad=True)
    rq, rk, rv = (t.detach().clone().requires_grad_() for t in (q, k, v))
    do = torch.randn_like(q)

    tlx_flash_attn(q, k, v, causal=False, space="smoke").backward(do)
    _sdpa(rq, rk, rv, causal=False).backward(do)

    for got, want in ((q.grad, rq.grad), (k.grad, rk.grad), (v.grad, rv.grad)):
        rel_l2 = torch.linalg.vector_norm(got.float() - want.float()) / torch.linalg.vector_norm(want.float())
        assert rel_l2 < REL_PRECISION[torch.bfloat16]


@pytest.mark.parametrize("num_ctas", [1, 2])
def test_flash_attn_bwd_rms_error_matches_sdpa(num_ctas, monkeypatch):
    """Gradients carry no more rounding error than BF16 SDPA's backward from the same inputs.

    Rounding a scaled operand to BF16 before an MMA, such as folding sm_scale / ln(2) into K,
    raises the RMS error 1.2-1.5x while nearly every input still passes `_assert_close`.
    """
    import triton
    from triton.tlx.ops import flash_attn as tlx_flash_attn
    from triton.tlx.ops.kernels.flash_attn import sm100

    config = (sm100.configs_bwd_1cta if num_ctas == 1 else sm100.configs_bwd_2cta)[0]

    def tuned_bwd(space):
        return triton.autotune(configs=[config], key=["N_CTX", "HEAD_DIM"])(sm100._attn_bwd_ws)

    monkeypatch.setattr(sm100, "_tuned_bwd", tuned_bwd)
    q, k, v = _qkv(2, 4, 1024, 64, torch.bfloat16, requires_grad=True)
    rq, rk, rv = (t.detach().clone().requires_grad_() for t in (q, k, v))
    do = torch.randn_like(q)

    tlx_flash_attn(q, k, v, causal=False, space="smoke").backward(do)
    _sdpa(rq, rk, rv, causal=False).backward(do)

    exact = _sdpa_fp32_grads(q, k, v, do, causal=False)
    for name, got, want, ref in zip("qkv", (q.grad, k.grad, v.grad), (rq.grad, rk.grad, rv.grad), exact):
        rms_got, rms_want = ((x.float() - ref).square().mean().sqrt().item() for x in (got, want))
        assert rms_got < 1.1 * rms_want, f"d{name} RMS error {rms_got:.3e} vs BF16 SDPA {rms_want:.3e}"


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("n_ctx", [31, 127, 128, 129, 200, 256, 400, 600])
def test_flash_attn_d256(n_ctx, dtype, causal):
    """Exercise public dispatch and tails across multiple batch/head boundaries."""
    from triton.tlx.ops import flash_attn as tlx_flash_attn

    torch.manual_seed(n_ctx)
    q, k, v = _qkv(2, 2, n_ctx, 256, dtype, requires_grad=True)
    rq, rk, rv = (t.detach().clone().requires_grad_() for t in (q, k, v))
    do = torch.randn_like(q)

    out = tlx_flash_attn(q, k, v, causal=causal, space="smoke")
    ref = _sdpa(rq, rk, rv, causal)
    assert out.shape == q.shape
    assert out.is_contiguous()
    assert torch.isfinite(out).all()
    _assert_close(out, ref, dtype)
    out.backward(do, retain_graph=True)
    ref.backward(do)
    first_grads = tuple(t.grad.clone() for t in (q, k, v))
    for got, want in zip(first_grads, (rq.grad, rk.grad, rv.grad)):
        assert torch.isfinite(got).all()
        _assert_close(got, want, dtype)
        rel_l2 = torch.linalg.vector_norm(got.float() - want.float()) / torch.linalg.vector_norm(
            want.float()).clamp_min(1e-12)
        assert rel_l2 < REL_PRECISION[dtype]

    # Repeated backward must not retain accumulators from earlier invocations.
    for t in (q, k, v):
        t.grad = None
    out.backward(do)
    for t, expected in zip((q, k, v), first_grads):
        _assert_close(t.grad, expected, dtype)


@pytest.mark.parametrize("n_ctx", [200, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flash_attn_d256_strided(n_ctx, dtype):
    from triton.tlx.ops import flash_attn as tlx_flash_attn

    packed = torch.randn((2, 2, n_ctx, 3 * 256), device="cuda", dtype=dtype)
    q, k, v = (t.requires_grad_() for t in packed.chunk(3, dim=-1))
    assert not q.is_contiguous()
    rq, rk, rv = (t.detach().clone().requires_grad_() for t in (q, k, v))
    out = tlx_flash_attn(q, k, v, space="smoke")
    ref = _sdpa(rq, rk, rv, False)
    _assert_close(out, ref, dtype)
    # sum() supplies a broadcast, non-contiguous upstream gradient.
    out.sum().backward()
    ref.sum().backward()
    for got, want in zip((q.grad, k.grad, v.grad), (rq.grad, rk.grad, rv.grad)):
        _assert_close(got, want, dtype)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("scale", [0.0, -0.0625, 0.03125])
@pytest.mark.parametrize("space", ["smoke", "full"])
def test_flash_attn_d256_scale(scale, causal, space):
    from triton.tlx.ops import flash_attn as tlx_flash_attn

    dtype = torch.bfloat16
    q, k, v = _qkv(2, 2, 129, 256, dtype, requires_grad=True)
    rq, rk, rv = (t.detach().clone().requires_grad_() for t in (q, k, v))
    do = torch.randn_like(q)
    out = tlx_flash_attn(q, k, v, causal=causal, sm_scale=scale, space=space)
    from torch.nn.attention import sdpa_kernel, SDPBackend

    # FlashAttention returns NaNs at scale=0; use the explicit math reference.
    with sdpa_kernel(SDPBackend.MATH):
        ref = torch.nn.functional.scaled_dot_product_attention(rq.float(), rk.float(), rv.float(), is_causal=causal,
                                                               scale=scale)
    assert torch.isfinite(out).all()
    _assert_close(out.float(), ref, dtype)
    out.backward(do)
    ref.backward(do)
    for got, want in zip((q.grad, k.grad, v.grad), (rq.grad, rk.grad, rv.grad)):
        _assert_close(got, want, dtype)


@pytest.mark.parametrize("space", ["smoke", "full"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flash_attn_d256_resources(space, dtype, monkeypatch):
    from triton.tlx.ops import flash_attn as tlx_flash_attn
    from triton.tlx.ops.kernels.flash_attn import sm100

    import triton

    q, k, v = _qkv(2, 2, 200, 256, dtype, requires_grad=True)
    warm = tlx_flash_attn(q, k, v, space=space)
    torch.autograd.grad(warm, (q, k, v), torch.randn_like(warm))
    names = ("_attn_fwd_ws_hd256_kernel", "_attn_bwd_preprocess", "_attn_bwd_dkdv_hd256_kernel",
             "_attn_bwd_dq_hd256_kernel")
    compiled = []
    # The C dispatcher bypasses Python .run wrappers unless a launch hook is active.
    monkeypatch.setattr(triton.knobs.runtime, "launch_enter_hook", lambda *args: None)
    for name in names:
        kernel = getattr(sm100, name)
        run = kernel.run

        def recording_run(*args, _run=run, _name=name, **kwargs):
            result = _run(*args, **kwargs)
            compiled.append((_name, result))
            return result

        monkeypatch.setattr(kernel, "run", recording_run)

    out = tlx_flash_attn(q, k, v, space=space)
    do = torch.randn_like(out)
    out.backward(do)
    assert [name for name, _ in compiled] == list(names)
    for name, kernel in compiled:
        assert kernel.metadata.shared <= 232448
        assert kernel.metadata.tmem_size <= 512
        expected_cluster = (1, 1, 1) if name == "_attn_bwd_preprocess" else (2, 1, 1)
        assert tuple(kernel.metadata.cluster_dims) == expected_cluster
    spill_stats = {name: {"spills": kernel.n_spills, "registers": kernel.n_regs} for name, kernel in compiled}
    # Bound the measured forward spill footprint; backward must remain spill-free.
    assert all(stats["spills"] <= (2 if name == "_attn_fwd_ws_hd256_kernel" else 0)
               for name, stats in spill_stats.items()), spill_stats
    assert torch.isfinite(out).all()
    for t in (q, k, v):
        assert torch.isfinite(t.grad).all()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    rq, rk, rv = (t.detach().clone().requires_grad_() for t in (q, k, v))
    with sdpa_kernel(SDPBackend.MATH):
        ref = torch.nn.functional.scaled_dot_product_attention(rq.float(), rk.float(), rv.float(), scale=0.0625)
    ref.backward(do.float())
    _assert_close(out.float(), ref, dtype)
    for got, want in zip((q.grad, k.grad, v.grad), (rq.grad, rk.grad, rv.grad)):
        _assert_close(got, want, dtype)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("n_ctx", [129, 200, 400, 600])
def test_flash_attn_d256_cudagraph(n_ctx, causal):
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from triton.testing import cuda_graph_without_gc
    from triton.tlx.ops import flash_attn as tlx_flash_attn

    dtype = torch.bfloat16
    q, k, v = _qkv(2, 2, n_ctx, 256, dtype, requires_grad=True)
    do = torch.randn_like(q)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            warm = tlx_flash_attn(q, k, v, causal=causal, space="smoke")
            torch.autograd.grad(warm, (q, k, v), do)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with cuda_graph_without_gc(graph, stream=stream):
        out = tlx_flash_attn(q, k, v, causal=causal, space="smoke")
        grads = torch.autograd.grad(out, (q, k, v), do)
    # Captured padding and TMA descriptors must follow updates, not reuse old data.
    for _ in range(3):
        with torch.no_grad():
            for tensor in (q, k, v, do):
                tensor.normal_()
        graph.replay()
        torch.cuda.synchronize()
        rq, rk, rv = (tensor.detach().float().requires_grad_() for tensor in (q, k, v))
        with sdpa_kernel(SDPBackend.MATH):
            ref = _sdpa(rq, rk, rv, causal)
        ref_grads = torch.autograd.grad(ref, (rq, rk, rv), do.float())
        assert out.shape == q.shape
        for actual, expected in zip((out, *grads), (ref, *ref_grads)):
            assert torch.isfinite(actual).all()
            _assert_close(actual.float(), expected, dtype)


@pytest.mark.parametrize("space", ["smoke", "full"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flash_attn_d256_s200_batch_bounds(dtype, space, monkeypatch):
    """Bounded TMA must not copy or read the following batch/head's tail."""
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from triton.tlx.ops import flash_attn as tlx_flash_attn

    q, k, v = _qkv(2, 2, 200, 256, dtype, requires_grad=True)
    with torch.no_grad():
        v[0, 0].add_(0.5)
        v[0, 1].sub_(0.5)
        v[1, 0].add_(1.0)
        v[1, 1].sub_(1.0)
    do = torch.randn_like(q)
    pads = []
    original_pad = F.pad

    def recording_pad(tensor, padding, *args, **kwargs):
        pads.append(tensor.data_ptr())
        return original_pad(tensor, padding, *args, **kwargs)

    monkeypatch.setattr(F, "pad", recording_pad)
    out = tlx_flash_attn(q, k, v, space=space)
    assert not pads
    grads = torch.autograd.grad(out, (q, k, v), do)
    assert pads == [do.data_ptr()]
    assert out.is_contiguous()
    assert all(grad.shape == q.shape and grad.is_contiguous() for grad in grads)
    rq, rk, rv = (tensor.detach().float().requires_grad_() for tensor in (q, k, v))
    with sdpa_kernel(SDPBackend.MATH):
        ref = _sdpa(rq, rk, rv, False)
    ref_grads = torch.autograd.grad(ref, (rq, rk, rv), do.float())
    for actual, expected in zip((out, *grads), (ref, *ref_grads)):
        assert torch.isfinite(actual).all()
        _assert_close(actual.float(), expected, dtype)


@pytest.mark.parametrize("causal,scale", [(True, 0.0625), (False, 0.0), (False, -0.0625)])
def test_flash_attn_d256_s200_fallback(causal, scale, monkeypatch):
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from triton.tlx.ops import flash_attn as tlx_flash_attn

    q, k, v = _qkv(2, 2, 200, 256, torch.bfloat16, requires_grad=True)
    do = torch.randn_like(q)
    pads = []
    original_pad = F.pad

    def recording_pad(tensor, padding, *args, **kwargs):
        pads.append(tensor.data_ptr())
        return original_pad(tensor, padding, *args, **kwargs)

    monkeypatch.setattr(F, "pad", recording_pad)
    out = tlx_flash_attn(q, k, v, causal=causal, sm_scale=scale, space="smoke")
    assert pads == [tensor.data_ptr() for tensor in (q, k, v)]
    grads = torch.autograd.grad(out, (q, k, v), do)
    assert pads[-1] == do.data_ptr() and len(pads) == 4
    rq, rk, rv = (tensor.detach().float().requires_grad_() for tensor in (q, k, v))
    with sdpa_kernel(SDPBackend.MATH):
        ref = F.scaled_dot_product_attention(rq, rk, rv, is_causal=causal, scale=scale)
    ref_grads = torch.autograd.grad(ref, (rq, rk, rv), do.float())
    for actual, expected in zip((out, *grads), (ref, *ref_grads)):
        assert torch.isfinite(actual).all()
        _assert_close(actual.float(), expected, torch.bfloat16)


@pytest.mark.parametrize("n_ctx", [200, 400, 600])
def test_flash_attn_d256_batch1024(n_ctx):
    """Validate the full traced batch size, not only small correctness inputs."""
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from triton.tlx.ops import flash_attn as tlx_flash_attn

    dtype = torch.bfloat16
    q, k, v = _qkv(1024, 1, n_ctx, 256, dtype, requires_grad=True)
    rq, rk, rv = (t.detach().clone().requires_grad_() for t in (q, k, v))
    do = torch.randn_like(q)
    out = tlx_flash_attn(q, k, v, causal=False, space="smoke")
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        ref = _sdpa(rq, rk, rv, False)
    _assert_close(out, ref, dtype)
    got = torch.autograd.grad(out, (q, k, v), do)
    want = torch.autograd.grad(ref, (rq, rk, rv), do)
    for grad, ref_grad in zip(got, want):
        assert torch.isfinite(grad).all()
        _assert_close(grad, ref_grad, dtype)
        rel_l2 = torch.linalg.vector_norm(grad.float() - ref_grad.float()) / torch.linalg.vector_norm(ref_grad.float())
        assert rel_l2 < REL_PRECISION[dtype]


@pytest.mark.parametrize("dtype_name", ["fp16", "bf16"])
@pytest.mark.parametrize("HEAD_DIM", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("num_ctas", [1, 2])
def test_flash_attn_bwd_pinned_config(num_ctas, causal, HEAD_DIM, dtype_name, monkeypatch):
    """The other backward tests run only the config autotuning picks, so pin each CTA count here.

    Spills pass every correctness check, so also read local memory off the compiled kernel.
    The 2-CTA kernel has regressed this way before: losing its single-warp-specialize
    dispatch left values live across setmaxnreg, which ptxas spilled.
    """
    import triton
    from triton.tlx.ops import flash_attn as tlx_flash_attn
    from triton.tlx.ops.kernels.flash_attn import sm100

    config = (sm100.configs_bwd_1cta if num_ctas == 1 else sm100.configs_bwd_2cta)[0]
    compiled = []

    def tuned_bwd_recording(space):
        tuner = triton.autotune(configs=[config], key=["N_CTX", "HEAD_DIM"])(sm100._attn_bwd_ws)
        run = tuner.run

        def recording_run(*args, **kwargs):
            compiled.append(run(*args, **kwargs))
            return compiled[-1]

        tuner.run = recording_run
        return tuner

    monkeypatch.setattr(sm100, "_tuned_bwd", tuned_bwd_recording)
    dtype = DTYPES[dtype_name]
    q, k, v = _qkv(2, 4, 1024, HEAD_DIM, dtype, requires_grad=True)
    rq, rk, rv = (t.detach().clone().requires_grad_() for t in (q, k, v))
    do = torch.randn_like(q)

    tlx_flash_attn(q, k, v, causal=causal, space="smoke").backward(do)
    _sdpa(rq, rk, rv, causal).backward(do)

    assert len(compiled) == 1
    for got, want in ((q.grad, rq.grad), (k.grad, rk.grad), (v.grad, rv.grad)):
        rel_l2 = torch.linalg.vector_norm(got.float() - want.float()) / torch.linalg.vector_norm(want.float())
        assert rel_l2 < REL_PRECISION[dtype]
    assert compiled[0].n_spills == 0, f"_attn_bwd_ws uses {4 * compiled[0].n_spills} bytes of local memory per thread"
