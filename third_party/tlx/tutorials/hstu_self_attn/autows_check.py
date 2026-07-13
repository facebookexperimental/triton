"""Check: does the hammer Triton fwd, with HSTU_SELF_AUTOWS=1, actually
warp-specialize (autoWS) and stay correct? Run with:
  HSTU_SELF_AUTOWS=1 HSTU_SELF_PIN=1 TRITON_USE_META_WS=1 \
  TRITON_DISABLE_WSBARRIER_REORDER=1 python autows_check.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

import triton_hstu_attention as A  # noqa: E402

print("HSTU_SELF_AUTOWS baked =", A._HSTU_SELF_AUTOWS)
Z, L, H, D = 2, 512, 2, 128
t = Z * L
torch.manual_seed(0)
q = torch.randn(t, H, D, dtype=torch.bfloat16, device="cuda")
k = torch.randn_like(q)
v = torch.randn_like(q)
so = torch.arange(0, t + 1, L, device="cuda", dtype=torch.int64)
asc = torch.tensor(1.0 / L, device="cuda", dtype=torch.float32)
alpha, scale = 1.0 / D, 1.0 / L

outs = []
for z in range(Z):
    s, e = z * L, (z + 1) * L
    n = e - s
    qk = torch.einsum("qhd,khd->hqk", q[s:e].float(), k[s:e].float()) * alpha
    sig = qk * torch.sigmoid(qk) * scale
    i = torch.arange(n, device=qk.device)
    sig = sig * (i[:, None] >= i[None, :]).float()[None]
    outs.append(torch.einsum("hqk,khd->qhd", sig, v[s:e].float()))
oref = torch.cat(outs, 0)

o = A.triton_hstu_mha(
    max_seq_len=L, alpha=alpha, q=q, k=k, v=v, seq_offsets=so,
    attn_scale=asc, enable_tma=True,
).detach()
rel = (torch.norm(o.float() - oref) / torch.norm(oref)).item()
print(f"fwd rel_l2 vs torch-causal: {rel:.3e}")
