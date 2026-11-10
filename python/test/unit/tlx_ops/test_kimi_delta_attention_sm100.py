"""L1 correctness for `tlx.ops.kimi_delta_attention`."""
import pytest
import torch
import torch.nn.functional as F
from triton._internal_testing import is_blackwell
from triton.tlx.ops.kernels.kda._shapes import FOCUS, SYNTHETIC, inputs as shape_inputs

pytestmark = pytest.mark.skipif(not is_blackwell(), reason="tlx.ops.kimi_delta_attention is sm100-only today")

torch.manual_seed(0)

REL_PRECISION = {torch.bfloat16: 8e-3}


@pytest.mark.parametrize("B,T,H,HEAD_DIM,dtype_name", FOCUS.all_shapes())
def test_kimi_delta_attention_focus_shapes_run(B, T, H, HEAD_DIM, dtype_name):
    from triton.tlx.ops import kimi_delta_attention as tlx_kda

    dtype = {"bf16": torch.bfloat16}[dtype_name]
    q, k, v, g, beta, cu_seqlens = shape_inputs(B, T, H, HEAD_DIM, dtype)
    out, aux = tlx_kda(q, k, v, g, beta, scale=1.0, cu_seqlens=cu_seqlens)
    assert aux is None
    assert torch.isfinite(out).all()


def _reference(q, k, v, g, beta, cu_seqlens, scale, chunk_size=64):
    qf, kf, vf = q.float(), k.float(), v.float()
    # The TLX kernel stages `g` as bf16 before computing the chunk cumsum.
    gf = g.to(torch.bfloat16).float()
    betaf = beta.float()
    _, _, H, D = q.shape
    seq_outs = []
    for sid in range(cu_seqlens.numel() - 1):
        seq_beg = int(cu_seqlens[sid])
        seq_end = int(cu_seqlens[sid + 1])
        head_outs = []
        for h in range(H):
            state = torch.zeros((D, D), device=q.device, dtype=torch.float32)
            chunk_outs = []
            for start in range(seq_beg, seq_end, chunk_size):
                stop = min(start + chunk_size, seq_end)
                qc = qf[:, start:stop, h, :].squeeze(0)
                kc = kf[:, start:stop, h, :].squeeze(0)
                vc = vf[:, start:stop, h, :].squeeze(0)
                gc = gf[:, start:stop, h, :].squeeze(0)
                bc = betaf[:, start:stop, h].squeeze(0)

                gamma = torch.cumsum(gc, dim=0)
                k_hat = kc * torch.exp(-gamma)
                k_bar = kc * torch.exp(gamma)
                q_bar = scale * qc * torch.exp(gamma)
                scores = k_bar @ k_hat.T
                a = torch.tril(scores * bc[None, :], diagonal=-1)
                tinv = torch.linalg.solve_triangular(
                    torch.eye(stop - start, device=q.device, dtype=torch.float32) + a,
                    torch.eye(stop - start, device=q.device, dtype=torch.float32),
                    upper=False,
                )
                rhs = vc - k_bar @ state
                u = tinv @ rhs
                qk = torch.tril(q_bar @ k_hat.T)
                chunk_outs.append(q_bar @ state + qk @ (bc[:, None] * u))

                decay = torch.exp(gamma[-1])
                k_tilde = kc * torch.exp(gamma[-1][None, :] - gamma)
                state = decay[:, None] * state + k_tilde.T @ (bc[:, None] * u)
            head_outs.append(torch.cat(chunk_outs, dim=0))
        seq_outs.append(torch.stack(head_outs, dim=1))
    return torch.cat(seq_outs, dim=0).unsqueeze(0)


def _inputs(dtype, requires_grad=False):
    n, t, h, d, _ = SYNTHETIC[0]
    gen = torch.Generator(device="cuda").manual_seed(0)

    def rn(*shape):
        return torch.randn(*shape, generator=gen, device="cuda", dtype=torch.float32)

    total = n * t
    q = F.normalize(rn(1, total, h, d), dim=-1).to(dtype).requires_grad_(requires_grad)
    k = F.normalize(rn(1, total, h, d), dim=-1).to(dtype).requires_grad_(requires_grad)
    v = rn(1, total, h, d).to(dtype).requires_grad_(requires_grad)
    g = (-F.softplus(rn(1, total, h, d))).requires_grad_(requires_grad)
    beta = torch.sigmoid(rn(1, total, h)).requires_grad_(requires_grad)
    cu_seqlens = torch.tensor([0, t, 2 * t], device="cuda", dtype=torch.int64)
    return q, k, v, g, beta, cu_seqlens


@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
def test_kimi_delta_attention_fwd(dtype):
    from triton.tlx.ops import kimi_delta_attention as tlx_kda

    q, k, v, g, beta, cu_seqlens = _inputs(dtype)
    out, aux = tlx_kda(q, k, v, g, beta, scale=1.0, cu_seqlens=cu_seqlens)

    assert aux is None
    ref = _reference(q, k, v, g, beta, cu_seqlens, scale=1.0).to(out.dtype)
    precision = REL_PRECISION[dtype]
    torch.testing.assert_close(out, ref, atol=precision * ref.abs().max().item(), rtol=precision)


def _recurrent_reference(q, k, v, g, beta, cu_seqlens, scale):
    # Token-by-token delta rule in fp32: S <- diag(e^g) S; S += beta k (v - S^T k)^T.
    # It never exponentiates a cumulative gate, so it is valid for any decay.
    qf, kf, vf = (x[0].float() for x in (q, k, v))
    gf = g[0].to(torch.bfloat16).float()
    betaf = beta[0].float()
    _, H, D = qf.shape
    out = torch.empty_like(vf)
    for sid in range(cu_seqlens.numel() - 1):
        state = torch.zeros((H, D, D), device=q.device, dtype=torch.float32)
        for t in range(int(cu_seqlens[sid]), int(cu_seqlens[sid + 1])):
            state = torch.exp(gf[t])[:, :, None] * state
            u = vf[t] - torch.einsum("hk,hkv->hv", kf[t], state)
            state = state + betaf[t][:, None, None] * kf[t][:, :, None] * u[:, None, :]
            out[t] = scale * torch.einsum("hk,hkv->hv", qf[t], state)
    return out.unsqueeze(0)


def _stress_inputs(case, dtype):
    gen = torch.Generator(device="cuda").manual_seed(1)
    h, d = 2, 128
    # Two sequences, both ending in a partial chunk.
    cu_seqlens = torch.tensor([0, 300, 300 + 517], device="cuda", dtype=torch.int64)
    total = int(cu_seqlens[-1])

    def rn(*shape):
        return torch.randn(*shape, generator=gen, device="cuda", dtype=torch.float32)

    q = F.normalize(rn(1, total, h, d), dim=-1)
    k = F.normalize(rn(1, total, h, d), dim=-1)
    v = rn(1, total, h, d)
    beta = torch.sigmoid(rn(1, total, h))
    if case == "const2":
        g = torch.full((1, total, h, d), -2.0, device="cuda")
    elif case == "const5":
        g = torch.full((1, total, h, d), -5.0, device="cuda")
    elif case == "bounded5":
        # FLA's lower-bound gate: g = -5 * sigmoid(.), g in (-5, 0).
        g = -5.0 * torch.sigmoid(rn(1, total, h, d))
    else:
        assert case == "correlated"
        # Near-parallel keys (cos ~0.9) with beta ~1 and slow decay make
        # ||A|| >> 1, where a truncated Neumann inverse breaks down.
        k = F.normalize(rn(1, 1, h, d) + 0.3 * rn(1, total, h, d), dim=-1)
        beta = torch.sigmoid(rn(1, total, h) + 4.0)
        g = -0.05 * F.softplus(rn(1, total, h, d))
    return q.to(dtype), k.to(dtype), v.to(dtype), g, beta, cu_seqlens


@pytest.mark.parametrize("case", ["const2", "const5", "bounded5", "correlated"])
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
def test_kimi_delta_attention_fwd_stress(case, dtype):
    from triton.tlx.ops.kernels.kda.sm100 import kimi_delta_attention_ws

    q, k, v, g, beta, cu_seqlens = _stress_inputs(case, dtype)
    out, _ = kimi_delta_attention_ws(q, k, v, g, beta, scale=1.0, cu_seqlens=cu_seqlens)

    assert torch.isfinite(out).all()
    ref = _recurrent_reference(q, k, v, g, beta, cu_seqlens, scale=1.0).to(out.dtype)
    # Near-collinear keys with beta ~ 1 make (I + A) ill-conditioned, so bf16
    # rounding of the inverse is amplified; the bound still rules out overflow.
    precision = 2e-2 if case == "correlated" else REL_PRECISION[dtype]
    torch.testing.assert_close(out, ref, atol=precision * ref.abs().max().item(), rtol=precision)


@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
def test_kimi_delta_attention_bwd(dtype):
    from triton.tlx.ops import kimi_delta_attention as tlx_kda

    q, k, v, g, beta, cu_seqlens = _inputs(dtype, requires_grad=True)
    rq, rk, rv = (x.detach().clone().requires_grad_() for x in (q, k, v))
    rg = g.detach().clone().requires_grad_()
    rbeta = beta.detach().clone().requires_grad_()
    do = torch.randn_like(q)

    out, aux = tlx_kda(q, k, v, g, beta, scale=1.0, cu_seqlens=cu_seqlens)
    ref = _reference(rq, rk, rv, rg, rbeta, cu_seqlens, scale=1.0)
    out.backward(do)
    ref.backward(do.float())

    assert aux is None
    for got, want in ((q.grad, rq.grad), (k.grad, rk.grad), (v.grad, rv.grad)):
        torch.testing.assert_close(got, want.to(got.dtype), atol=5e-2 * want.abs().max().item(), rtol=5e-2)
    for got, want in ((g.grad, rg.grad), (beta.grad, rbeta.grad)):
        torch.testing.assert_close(got, want, atol=5e-2 * want.abs().max().item(), rtol=5e-2)
