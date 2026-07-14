"""Accuracy harness for the ported HSTU SELF-attention kernels (MetaMain2 OSS).

Variants:
  * triton - hammer-template Triton self-attn (triton_hstu_mha), fwd+bwd. Trusted ref.
  * tlx    - hand-written TLX warp-specialized self-attn (tlx_bw_hstu_mha), Blackwell.

Checks fwd output + dq/dk/dv grads:
  - triton vs a torch-float HSTU-SiLU reference (both causal and non-causal, to
    identify which masking the kernel uses),
  - tlx vs triton (the byte-ish correctness check; same hammer math).

Usage (from this dir, on a Blackwell GPU):
  CUDA_VISIBLE_DEVICES=1 ~/.conda/envs/metamain2/bin/python bench_self.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import triton  # noqa: E402

import triton_hstu_attention as A  # noqa: E402
import tlx_bw_hstu_attention as T  # noqa: E402

D = 128
H = 2


def make(L, Z, dtype=torch.bfloat16):
    t = Z * L
    g = lambda: torch.randn(t, H, D, device="cuda", dtype=dtype)
    q = g().requires_grad_(True)
    k = g().requires_grad_(True)
    v = g().requires_grad_(True)
    so = torch.arange(0, t + 1, L, device="cuda", dtype=torch.int64)
    asc = torch.tensor(1.0 / L, device="cuda", dtype=torch.float32)
    do = g()
    return q, k, v, do, so, asc


def torch_ref(q, k, v, do, so, asc, causal):
    """Float autograd HSTU-SiLU self-attention reference."""
    qf = q.detach().float().requires_grad_(True)
    kf = k.detach().float().requires_grad_(True)
    vf = v.detach().float().requires_grad_(True)
    alpha = 1.0 / D
    scale = asc.item()
    outs = []
    for z in range(so.numel() - 1):
        s, e = int(so[z]), int(so[z + 1])
        n = e - s
        qk = torch.einsum("qhd,khd->hqk", qf[s:e], kf[s:e]) * alpha
        sig = qk * torch.sigmoid(qk) * scale  # SiLU(qk) * attn_scale
        if causal:
            i = torch.arange(n, device=qk.device)
            valid = (i[:, None] >= i[None, :]).float()  # lower-tri incl diagonal
            sig = sig * valid[None]
        outs.append(torch.einsum("hqk,khd->qhd", sig, vf[s:e]))
    torch.cat(outs, 0).backward(do.float())
    return qf.grad, kf.grad, vf.grad


def rel_l2(a, b):
    return (torch.norm(a.float() - b.float()) / (torch.norm(b.float()) + 1e-12)).item()


def run_triton(q, k, v, so, L, asc):
    return A.triton_hstu_mha(
        max_seq_len=L, alpha=1.0 / D, q=q, k=k, v=v, seq_offsets=so,
        attn_scale=asc, num_targets=None, max_attn_len=0, contextual_seq_len=0,
        enable_tma=True,
    )


def run_tlx(q, k, v, so, L, asc, causal=True):
    return T.tlx_bw_hstu_mha(
        max_seq_len=L, alpha=1.0 / D, q=q, k=k, v=v, seq_offsets=so,
        attn_scale=asc, num_softmax_heads=0, num_targets=None,
        max_attn_len=0, contextual_seq_len=0, causal=causal,
    )


def grads(run, q, k, v, so, L, asc, do, **kw):
    for t in (q, k, v):
        t.grad = None
    # TLX and plain Triton are not compiler-warp-specialized: force meta-WS off
    # (and the WS-barrier reorder off, which TLX lowering needs) through a knobs
    # scope so the override is auto-isolated instead of leaking into os.environ.
    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = False
        triton.knobs.nvidia.disable_wsbarrier_reorder = True
        out = run(q, k, v, so, L, asc, **kw)
        out.backward(do)
    return out.detach(), q.grad.clone(), k.grad.clone(), v.grad.clone()


def main():
    for (L, Z) in [(256, 4), (512, 2)]:
        print(f"\n=== L={L} Z={Z} H={H} D={D} ===")
        torch.manual_seed(0)
        q, k, v, do, so, asc = make(L, Z)
        rc = torch_ref(q, k, v, do, so, asc, causal=True)
        rn = torch_ref(q, k, v, do, so, asc, causal=False)

        try:
            o_t, dq_t, dk_t, dv_t = grads(run_triton, q, k, v, so, L, asc, do)
            print("  triton fwd vs torch:  causal dq/dk/dv relL2 = "
                  f"{rel_l2(dq_t, rc[0]):.2e}/{rel_l2(dk_t, rc[1]):.2e}/{rel_l2(dv_t, rc[2]):.2e}"
                  f"   noncausal = {rel_l2(dq_t, rn[0]):.2e}/{rel_l2(dk_t, rn[1]):.2e}/{rel_l2(dv_t, rn[2]):.2e}")
        except Exception as e:
            print(f"  triton ERROR {type(e).__name__}: {str(e)[:120]}")
            o_t = None

        for causal in (True, False):
            try:
                o_x, dq_x, dk_x, dv_x = grads(run_tlx, q, k, v, so, L, asc, do, causal=causal)
                line = (f"  tlx(causal={causal}) vs torch-{'causal' if causal else 'noncausal'}: "
                        f"dq/dk/dv relL2 = "
                        f"{rel_l2(dq_x, (rc if causal else rn)[0]):.2e}/"
                        f"{rel_l2(dk_x, (rc if causal else rn)[1]):.2e}/"
                        f"{rel_l2(dv_x, (rc if causal else rn)[2]):.2e}")
                if o_t is not None:
                    line += (f"   vs triton: fwd {rel_l2(o_x, o_t):.2e} "
                             f"dq {rel_l2(dq_x, dq_t):.2e} dk {rel_l2(dk_x, dk_t):.2e} dv {rel_l2(dv_x, dv_t):.2e}")
                print(line)
            except Exception as e:
                print(f"  tlx(causal={causal}) ERROR {type(e).__name__}: {str(e)[:120]}")


if __name__ == "__main__":
    main()
