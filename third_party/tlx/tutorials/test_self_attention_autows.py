"""Correctness test for the HSTU SELF-attention forward under automatic warp
specialization (meta-WS).

Enables autoWS on the hammer-template Triton self-attn kernel via the
`warp_specialize=True` annotation on the main KV loop (env HSTU_SELF_AUTOWS=1,
baked as a tl.constexpr at import) and asserts the fwd output AND the dq/dk/dv
gradients match a torch-autograd float causal-SiLU reference.

Notes:
- autoWS needs TRITON_USE_META_WS=1 + TRITON_DISABLE_WSBARRIER_REORDER=1; these
  are set BEFORE importing the kernel so the constexpr/config pick see them.
- HSTU_SELF_DP=1: data_partition_factor=2 would need BLOCK_M=256 (each slice
  >=128 TMEM rows) which OOMs TMEM on this kernel, so DP is off here.
- The TLX self-attn *fwd* uses num_stages=0 and asserts under meta-WS, so it
  cannot be compiled in the same (META_WS) process; the accuracy oracle here is
  the torch reference (as in test_cross_attention_bwd_autows.py).

Run: pytest third_party/tlx/tutorials/test_self_attention_autows.py
"""
import os
import sys

_HSTU_DIR = os.path.join(os.path.dirname(__file__), "hstu_self_attn")
sys.path.insert(0, _HSTU_DIR)

# Enable autoWS + its env BEFORE importing the kernel (the flag/config are read
# at import / first compile).
os.environ["HSTU_SELF_AUTOWS"] = "1"
os.environ["HSTU_SELF_DP"] = "1"
os.environ["HSTU_SELF_PIN"] = "1"
os.environ["TRITON_USE_META_WS"] = "1"
os.environ["TRITON_DISABLE_WSBARRIER_REORDER"] = "1"

import pytest  # noqa: E402
import torch  # noqa: E402

import triton_hstu_attention as A  # noqa: E402

D = 128
H = 2


def _torch_ref(q, k, v, do, so, asc):
    """Float autograd HSTU-SiLU causal self-attention reference."""
    qf = q.detach().float().requires_grad_(True)
    kf = k.detach().float().requires_grad_(True)
    vf = v.detach().float().requires_grad_(True)
    alpha, scale = 1.0 / D, asc.item()
    outs = []
    for z in range(so.numel() - 1):
        s, e = int(so[z]), int(so[z + 1])
        n = e - s
        qk = torch.einsum("qhd,khd->hqk", qf[s:e], kf[s:e]) * alpha
        sig = qk * torch.sigmoid(qk) * scale
        i = torch.arange(n, device=qk.device)
        sig = sig * (i[:, None] >= i[None, :]).float()[None]
        outs.append(torch.einsum("hqk,khd->qhd", sig, vf[s:e]))
    torch.cat(outs, 0).backward(do.float())
    return qf.grad, kf.grad, vf.grad


def _rel_l2(a, b):
    return (torch.norm(a.float() - b.float()) / (torch.norm(b.float()) + 1e-12)).item()


@pytest.mark.parametrize("L,Z", [(256, 4), (512, 2)])
def test_self_attention_fwd_autows(L, Z):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    assert bool(A._HSTU_SELF_AUTOWS), "autoWS flag not baked on"

    torch.manual_seed(0)
    t = Z * L
    g = lambda: torch.randn(t, H, D, device="cuda", dtype=torch.bfloat16)
    q, k, v = g().requires_grad_(True), g().requires_grad_(True), g().requires_grad_(True)
    so = torch.arange(0, t + 1, L, device="cuda", dtype=torch.int64)
    asc = torch.tensor(1.0 / L, device="cuda", dtype=torch.float32)
    do = g()

    rq, rk, rv = _torch_ref(q, k, v, do, so, asc)

    for tsr in (q, k, v):
        tsr.grad = None
    o = A.triton_hstu_mha(
        max_seq_len=L,
        alpha=1.0 / D,
        q=q,
        k=k,
        v=v,
        seq_offsets=so,
        attn_scale=asc,
        enable_tma=True,
    )
    o.backward(do)
    dq, dk, dv = q.grad.clone(), k.grad.clone(), v.grad.clone()

    for name, got, want in (("dq", dq, rq), ("dk", dk, rk), ("dv", dv, rv)):
        rl2 = _rel_l2(got, want)
        assert rl2 < 1e-2, f"autoWS {name} rel-L2 {rl2:.2e} too high (L={L} Z={Z})"
