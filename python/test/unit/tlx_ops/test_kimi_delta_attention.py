"""L1 correctness for `tlx.ops.kimi_delta_attention`."""

import platform

import pytest
import torch
import torch.nn.functional as F

from triton._internal_testing import is_cuda


def _torch_cuda_version():
    try:
        major, minor, *_ = (torch.version.cuda or "").split(".")
        return (int(major), int(minor))
    except ValueError:
        # Version undetectable; let the run proceed and fail loudly rather
        # than silently skipping coverage.
        return (99, 99)


def _is_supported_device():
    # B200 and GB200 are sm100; GB300 is sm103. The Grace pairings
    # (GB200/GB300, i.e. aarch64 hosts) need CUDA >= 13 at runtime. The b300a
    # codegen arch for GB300 is selected by the build's arch generation, not
    # by this test (which always pins the sm100 kernel entry below).
    if not is_cuda():
        return False
    if torch.cuda.get_device_capability()[:2] not in ((10, 0), (10, 3)):
        return False
    return platform.machine() != "aarch64" or _torch_cuda_version() >= (13, 0)


pytestmark = pytest.mark.skipif(
    not _is_supported_device(),
    reason="tlx.ops.kimi_delta_attention needs B200/GB200 (sm100) or GB300 (sm103); GB200/GB300 need CUDA >= 13",
)

torch.manual_seed(0)

ARCH = "sm100"
REL_PRECISION = {torch.bfloat16: 8e-3}


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
    n, t, h, d = (2, 64, 2, 128)
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
    out, aux = tlx_kda(q, k, v, g, beta, scale=1.0, cu_seqlens=cu_seqlens, arch=ARCH)

    assert aux is None
    ref = _reference(q, k, v, g, beta, cu_seqlens, scale=1.0).to(out.dtype)
    precision = REL_PRECISION[dtype]
    torch.testing.assert_close(out, ref, atol=precision * ref.abs().max().item(), rtol=precision)


@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
def test_kimi_delta_attention_bwd(dtype):
    from triton.tlx.ops import kimi_delta_attention as tlx_kda

    q, k, v, g, beta, cu_seqlens = _inputs(dtype, requires_grad=True)
    rq, rk, rv = (x.detach().clone().requires_grad_() for x in (q, k, v))
    rg = g.detach().clone().requires_grad_()
    rbeta = beta.detach().clone().requires_grad_()
    do = torch.randn_like(q)

    out, aux = tlx_kda(q, k, v, g, beta, scale=1.0, cu_seqlens=cu_seqlens, arch=ARCH)
    ref = _reference(rq, rk, rv, rg, rbeta, cu_seqlens, scale=1.0)
    out.backward(do)
    ref.backward(do.float())

    assert aux is None
    for got, want in ((q.grad, rq.grad), (k.grad, rk.grad), (v.grad, rv.grad)):
        torch.testing.assert_close(got, want.to(got.dtype), atol=5e-2 * want.abs().max().item(), rtol=5e-2)
    for got, want in ((g.grad, rg.grad), (beta.grad, rbeta.grad)):
        torch.testing.assert_close(got, want, atol=5e-2 * want.abs().max().item(), rtol=5e-2)
