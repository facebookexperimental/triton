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
