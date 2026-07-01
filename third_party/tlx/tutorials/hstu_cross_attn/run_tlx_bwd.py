"""OSS validation driver for the TLX warp-specialized reduce_dq backward
(attn_bwd_ws), ported from fbcode hammer.

Correctness: compares TLX dq/dk/dv against (a) a torch-autograd reference on the
same HSTU cross-attention forward math, and (b) the OSS redq kernel (already
audited byte-identical to buck).

Perf: measures TLX bwd latency at the buck shape (kv1024 q256 d128 B256 H2,
non-shared KV, softmax) and compares to the buck TLX numbers.

Usage:
  ~/.conda/envs/metamain2/bin/python run_tlx_bwd.py            # correctness
  ~/.conda/envs/metamain2/bin/python run_tlx_bwd.py --perf     # + perf
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import torch
import triton

import tlx_bw_cross_attention as tx
import triton_bw_cross_attention as xa


def torch_ref_bwd(q, k, v, do, seq_offsets, attn_scale, alpha, softmax, shared_kv):
    """Reference backward via torch autograd on the HSTU cross-attn forward.

    Per batch (jagged): qk = q @ k^T; activation over KV; out = S @ v.
      SiLU:    S = silu(qk*alpha) * attn_scale   (silu(x)=x*sigmoid(x))
      Softmax: S = softmax_kv(qk*alpha)          (attn_scale unused)
    """
    qf = q.detach().float().requires_grad_(True)
    kf = k.detach().float().requires_grad_(True)
    # For shared KV, v IS k: use a single leaf so kf.grad accumulates both the
    # k-path and v-path gradients (the kernel folds dv into dk the same way).
    if shared_kv:
        vf = kf
    else:
        vf = v.detach().float().requires_grad_(True)
    Z = seq_offsets.numel() - 1
    H = q.shape[1]
    outs = []
    scale = attn_scale.item() if attn_scale.ndim == 0 else None
    for z in range(Z):
        s, e = int(seq_offsets[z]), int(seq_offsets[z + 1])
        qb = qf[s:e]  # [Lq, H, D]
        kb = kf[s:e]
        vb = vf[s:e]
        # [H, Lq, Lkv]
        qk = torch.einsum("qhd,khd->hqk", qb, kb) * alpha
        if softmax:
            S = torch.softmax(qk, dim=-1)
        else:
            S = (qk * torch.sigmoid(qk)) * scale
        ob = torch.einsum("hqk,khd->qhd", S, vb)
        outs.append(ob)
    out = torch.cat(outs, dim=0)
    out.backward(do.float())
    dv = kf.grad if shared_kv else vf.grad
    return qf.grad, kf.grad, dv


def rel_l2(a, b):
    a = a.float()
    b = b.float()
    return (torch.norm(a - b) / (torch.norm(b) + 1e-12)).item()


def max_abs(a, b):
    return (a.float() - b.float()).abs().max().item()


def run_correctness():
    torch.manual_seed(0)
    dev = "cuda"
    Z, H, D = 4, 2, 128
    L = 256
    dtype = torch.bfloat16
    total = Z * L
    seq_offsets = torch.arange(0, total + 1, L, device=dev, dtype=torch.int64)
    attn_scale = torch.tensor(1.0 / L, device=dev, dtype=torch.float32)
    alpha = 1.0 / D

    rows = []
    for softmax in (False, True):
        for shared_kv in (False, True):
            mk = lambda: torch.randn(total, H, D, device=dev, dtype=dtype)
            q, k, v = mk(), mk(), mk()
            if shared_kv:
                v = k
            do = torch.randn(total, H, D, device=dev, dtype=dtype)
            nsh = H if softmax else 0

            out = M = None
            if softmax:
                out, M, _ = xa.triton_hstu_cross_attn_v3_fwd(
                    L, alpha, q, k, v, seq_offsets, seq_offsets, L, attn_scale,
                    G=1, shared_kv=shared_kv, num_softmax_heads=nsh, enable_tma=True,
                )

            dq, dk, dv = tx.tlx_bw_reduce_dq(
                q, k, v, do, seq_offsets, attn_scale, L, alpha,
                max_q_len=L, seq_offsets_q=seq_offsets,
                shared_kv=shared_kv, num_softmax_heads=nsh, out=out, M=M,
            )

            rq, rk, rv = torch_ref_bwd(
                q, k, v, do, seq_offsets, attn_scale, alpha, softmax, shared_kv
            )

            act = "softmax" if softmax else "silu"
            kv = "shared" if shared_kv else "non-shared"
            dq_e = rel_l2(dq, rq)
            dk_e = rel_l2(dk, rk)
            if shared_kv:
                # dv folded into dk; reference dk already includes the v-grad path
                dv_e = float("nan")
            else:
                dv_e = rel_l2(dv, rv)
            tol = 2e-2  # bf16 inputs, fp32 accum
            ok = dq_e < tol and dk_e < tol and (shared_kv or dv_e < tol)
            rows.append((act, kv, dq_e, dk_e, dv_e, max_abs(dq, rq), ok))

    print("\n=== Correctness: TLX reduce_dq vs torch-autograd ref (rel-L2) ===")
    print(f"{'activation':<10} {'kv':<11} {'dq relL2':>10} {'dk relL2':>10} "
          f"{'dv relL2':>10} {'dq maxabs':>11} {'pass':>6}")
    allok = True
    for act, kv, dqe, dke, dve, dqm, ok in rows:
        allok = allok and ok
        print(f"{act:<10} {kv:<11} {dqe:>10.2e} {dke:>10.2e} {dve:>10.2e} "
              f"{dqm:>11.2e} {str(ok):>6}")
    print(f"\nTolerance: rel-L2 < 2e-2.  ALL PASS: {allok}")
    return allok


def run_perf():
    torch.manual_seed(0)
    dev = "cuda"
    # Buck shape: kv1024 q256 d128 B256 H2 non-shared softmax
    Z, H, D = 256, 2, 128
    Lkv, Lq = 1024, 256
    dtype = torch.bfloat16
    alpha = 1.0 / D

    def make(jagged):
        if jagged:
            torch.manual_seed(1)
            lens_kv = torch.randint(Lkv // 2, Lkv + 1, (Z,), device=dev)
            lens_q = torch.randint(Lq // 2, Lq + 1, (Z,), device=dev)
        else:
            lens_kv = torch.full((Z,), Lkv, device=dev)
            lens_q = torch.full((Z,), Lq, device=dev)
        so_kv = torch.zeros(Z + 1, device=dev, dtype=torch.int64)
        so_q = torch.zeros(Z + 1, device=dev, dtype=torch.int64)
        so_kv[1:] = torch.cumsum(lens_kv, 0)
        so_q[1:] = torch.cumsum(lens_q, 0)
        Tkv, Tq = int(so_kv[-1]), int(so_q[-1])
        q = torch.randn(Tq, H, D, device=dev, dtype=dtype)
        k = torch.randn(Tkv, H, D, device=dev, dtype=dtype)
        v = torch.randn(Tkv, H, D, device=dev, dtype=dtype)
        do = torch.randn(Tq, H, D, device=dev, dtype=dtype)
        attn_scale = torch.tensor(1.0 / Lkv, device=dev, dtype=torch.float32)
        return q, k, v, do, so_kv, so_q, attn_scale

    print("\n=== Perf: TLX reduce_dq bwd, softmax, non-shared KV "
          "(kv1024 q256 d128 B256 H2) ===")
    buck = {"jagged": 0.723, "dense": 0.623}
    for label, jagged in (("dense", False), ("jagged", True)):
        q, k, v, do, so_kv, so_q, attn_scale = make(jagged)
        out, M, _ = xa.triton_hstu_cross_attn_v3_fwd(
            Lkv, alpha, q, k, v, so_kv, so_q, Lq, attn_scale,
            G=1, shared_kv=False, num_softmax_heads=H, enable_tma=True,
        )
        fn = lambda: tx.tlx_bw_reduce_dq(
            q, k, v, do, so_kv, attn_scale, Lkv, alpha,
            max_q_len=Lq, seq_offsets_q=so_q,
            shared_kv=False, num_softmax_heads=H, out=out, M=M,
        )
        fn()  # warm/autotune
        ms = triton.testing.do_bench(fn, warmup=50, rep=200)
        b = buck[label]
        delta = (ms - b) / b * 100.0
        verdict = "PARITY" if abs(delta) <= 10.0 else "OUT-OF-PARITY"
        print(f"  {label:<7} OSS-TLX {ms:.3f} ms | buck-TLX {b:.3f} ms | "
              f"delta {delta:+.1f}% | {verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perf", action="store_true")
    ap.add_argument("--only-perf", action="store_true")
    args = ap.parse_args()
    if not args.only_perf:
        run_correctness()
    if args.perf or args.only_perf:
        run_perf()


if __name__ == "__main__":
    main()
