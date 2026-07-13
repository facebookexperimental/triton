"""Isolate the fwd-output discrepancy: compare triton vs tlx vs torch-float
causal HSTU-SiLU forward directly (no backward)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.pop("TRITON_USE_META_WS", None)
import torch  # noqa: E402

import triton_hstu_attention as A  # noqa: E402
import tlx_bw_hstu_attention as T  # noqa: E402

D = 128
H = 2
L = 256
Z = 2


def rel(a, b):
    return (torch.norm(a.float() - b.float()) / (torch.norm(b.float()) + 1e-12)).item()


torch.manual_seed(0)
t = Z * L
g = lambda: torch.randn(t, H, D, device="cuda", dtype=torch.bfloat16)
q, k, v = g(), g(), g()
so = torch.arange(0, t + 1, L, device="cuda", dtype=torch.int64)
asc = torch.tensor(1.0 / L, device="cuda", dtype=torch.float32)
alpha = 1.0 / D
scale = asc.item()

# torch-float causal HSTU-SiLU forward reference
outs = []
for z in range(Z):
    s, e = z * L, (z + 1) * L
    n = e - s
    qk = torch.einsum("qhd,khd->hqk", q[s:e].float(), k[s:e].float()) * alpha
    sig = qk * torch.sigmoid(qk) * scale
    i = torch.arange(n, device=qk.device)
    sig = sig * (i[:, None] >= i[None, :]).float()[None]
    outs.append(torch.einsum("hqk,khd->qhd", sig, v[s:e].float()))
o_ref = torch.cat(outs, 0)

o_t = A.triton_hstu_mha(max_seq_len=L, alpha=alpha, q=q, k=k, v=v, seq_offsets=so,
                        attn_scale=asc, enable_tma=True).detach()
o_x = T.tlx_bw_hstu_mha(max_seq_len=L, alpha=alpha, q=q, k=k, v=v, seq_offsets=so,
                        attn_scale=asc, num_softmax_heads=0, causal=True).detach()

print(f"shapes: ref={tuple(o_ref.shape)} triton={tuple(o_t.shape)} tlx={tuple(o_x.shape)}")
print(f"norms:  ref={o_ref.float().norm():.3f} triton={o_t.float().norm():.3f} tlx={o_x.float().norm():.3f}")
print(f"rel_l2 triton vs torch-causal: {rel(o_t, o_ref):.3e}")
print(f"rel_l2 tlx    vs torch-causal: {rel(o_x, o_ref):.3e}")
print(f"rel_l2 tlx    vs triton:       {rel(o_x, o_t):.3e}")
# element ratio (where triton is non-tiny) to detect a constant scale factor
m = o_t.float().abs() > 1e-3
if m.any():
    ratio = (o_x.float()[m] / o_t.float()[m])
    print(f"tlx/triton elementwise ratio: median={ratio.median():.4f} mean={ratio.mean():.4f} std={ratio.std():.4f}")
# per-row (query) L2 to see if a specific row range (e.g. first block, or targets) differs
rn_t = o_t.float().norm(dim=(1, 2))
rn_x = o_x.float().norm(dim=(1, 2))
diff = (o_x.float() - o_t.float()).norm(dim=(1, 2))
worst = torch.topk(diff, 8).indices.tolist()
print(f"rows with largest tlx-vs-triton diff (idx: triton_norm, tlx_norm, diff):")
for r in sorted(worst):
    print(f"  row {r} (in-seq {r % L}): {rn_t[r]:.3f} {rn_x[r]:.3f} {diff[r]:.3f}")
