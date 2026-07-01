"""Combined accuracy + perf benchmark for the HSTU cross-attention backward
(reduce_dq), across the three variants:

  * redq   - triton non-WS reduce_dq (trusted reference kernel)
  * autows - triton automatic warp specialization (meta-WS)
  * tlx    - hand-written TLX warp-specialized reduce_dq (attn_bwd_ws)

Accuracy: rel-L2 of dq/dk/dv vs a torch-autograd float reference, plus the
count of dq rows that diverge from redq grouped into Q-blocks (this isolates
the WS-specific bug: autoWS corrupts the last min(KV_blocks-1, num_stages)
Q-blocks for KV>=2; redq/tlx match the reference).

Perf: backward latency per variant (autograd backward on a prebuilt fwd graph),
reported with speedup relative to redq.

Usage (from this directory):
  ~/.conda/envs/metamain2/bin/python bench_bwd.py           # accuracy + perf
  ~/.conda/envs/metamain2/bin/python bench_bwd.py --acc     # accuracy only
  ~/.conda/envs/metamain2/bin/python bench_bwd.py --perf    # perf only
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
os.environ.pop("TRITON_USE_META_WS", None)
import torch
import triton

import triton_bw_cross_attention as xa

D = 128
H = 2
BLOCK = 64  # BLOCK_M == BLOCK_N used by the benchmark configs

VARIANTS = [
    ("redq", xa.BwdVariant.TRITON_REDQ, "0"),
    ("autows", xa.BwdVariant.TRITON_AUTOWS, "1"),
    ("tlx", xa.BwdVariant.TLX, "0"),
]


def force(ns=2, bm=BLOCK, bn=BLOCK):
    """Pin fwd num_stages>=1 (autoWS asserts on 0) and the bwd block sizes, and
    shrink the autotune space to one config for fast, deterministic runs."""
    for fn in (xa._attn_fwd_triton,):
        if hasattr(fn, "configs"):
            c = fn.configs[0]
            c.num_stages = max(getattr(c, "num_stages", 1), 1)
            fn.configs = [c]
    c = xa._hstu_attn_bwd_redq.configs[0]
    c.num_stages = ns
    c.kwargs["BLOCK_M"] = bm
    c.kwargs["BLOCK_N"] = bn
    xa._hstu_attn_bwd_redq.configs = [c]
    xa.set_fwd_variant(xa.FwdVariant.TRITON)


def make(Lq, Lkv, Z):
    tq, tk = Z * Lq, Z * Lkv
    g = lambda n: torch.randn(n, H, D, device="cuda", dtype=torch.bfloat16)
    q = g(tq).requires_grad_(True)
    k = g(tk).requires_grad_(True)
    v = g(tk).requires_grad_(True)
    so_kv = torch.arange(0, tk + 1, Lkv, device="cuda", dtype=torch.int64)
    so_q = torch.arange(0, tq + 1, Lq, device="cuda", dtype=torch.int64)
    asc = torch.tensor(1.0 / Lkv, device="cuda", dtype=torch.float32)
    do = g(tq)
    return q, k, v, do, so_kv, so_q, asc


def torch_ref(q, k, v, do, so_kv, so_q, asc, softmax=True):
    """Float autograd reference on the HSTU cross-attn forward (asymmetric Q/KV)."""
    qf = q.detach().float().requires_grad_(True)
    kf = k.detach().float().requires_grad_(True)
    vf = v.detach().float().requires_grad_(True)
    scale = asc.item()
    outs = []
    for z in range(so_kv.numel() - 1):
        qs, qe = int(so_q[z]), int(so_q[z + 1])
        ks, ke = int(so_kv[z]), int(so_kv[z + 1])
        qk = torch.einsum("qhd,khd->hqk", qf[qs:qe], kf[ks:ke]) * (1.0 / D)
        S = torch.softmax(qk, -1) if softmax else (qk * torch.sigmoid(qk)) * scale
        outs.append(torch.einsum("hqk,khd->qhd", S, vf[ks:ke]))
    torch.cat(outs, 0).backward(do.float())
    return qf.grad, kf.grad, vf.grad


def rel_l2(a, b):
    return (torch.norm(a.float() - b.float()) / (torch.norm(b.float()) + 1e-12)).item()


def max_abs(a, b):
    return (a.float() - b.float()).abs().max().item()


def bad_qblocks(test, ref, Lq, tol=5e-3):
    pr = (test.float() - ref.float()).abs().amax(dim=(1, 2))
    rm = torch.arange(pr.numel(), device=pr.device) % Lq
    bad = pr > tol
    return int(bad.sum()), sorted(set((rm[bad] // BLOCK).tolist()))


def _fwd(var, ws, q, k, v, so_kv, so_q, Lkv, Lq, asc):
    xa.set_bwd_variant(var)
    os.environ["TRITON_USE_META_WS"] = ws
    return xa.triton_bw_hstu_mha_wrapper(
        max_seq_len=Lkv, alpha=1.0 / D, q=q, k=k, v=v, seq_offsets=so_kv,
        attn_scale=asc, max_q_len=Lq, seq_offsets_q=so_q, num_softmax_heads=H,
        shared_kv=False, enable_tma=True,
    )


def run_accuracy(configs, ns=2):
    print("\n=== Accuracy: dq/dk/dv rel-L2 vs torch-float ref; dq bad-Q-blocks vs redq ===")
    for Lq, Lkv, Z in configs:
        force(ns)
        torch.manual_seed(0)
        q, k, v, do, so_kv, so_q, asc = make(Lq, Lkv, Z)
        rq, rk, rv = torch_ref(q, k, v, do, so_kv, so_q, asc)
        print(f"\n  Lq={Lq}({Lq // BLOCK}blk) Lkv={Lkv}({Lkv // BLOCK}blk) Z={Z} ns={ns}")
        print(f"  {'variant':<8}{'dq relL2':>10}{'dk relL2':>10}{'dv relL2':>10}"
              f"{'dq maxabs':>11}  dq-vs-redq bad-Qblk")
        ref = None
        for name, var, ws in VARIANTS:
            for t in (q, k, v):
                t.grad = None
            try:
                out = _fwd(var, ws, q, k, v, so_kv, so_q, Lkv, Lq, asc)
                out.backward(do)
                dq, dk, dv = q.grad.clone(), k.grad.clone(), v.grad.clone()
                if name == "redq":
                    ref = dq.clone()
                n, bm = bad_qblocks(dq, ref, Lq)
                print(f"  {name:<8}{rel_l2(dq, rq):>10.2e}{rel_l2(dk, rk):>10.2e}"
                      f"{rel_l2(dv, rv):>10.2e}{max_abs(dq, rq):>11.2e}  {n} rows Qblk{bm}")
            except Exception as e:
                print(f"  {name:<8} ERROR {type(e).__name__}: {str(e)[:50]}")


def run_perf(shapes, ns=2):
    print("\n=== Perf: backward latency per variant (autograd bwd on prebuilt fwd graph) ===")
    for Lq, Lkv, Z in shapes:
        force(ns)
        torch.manual_seed(0)
        q, k, v, do, so_kv, so_q, asc = make(Lq, Lkv, Z)
        print(f"\n  Lq={Lq} Lkv={Lkv}({Lkv // BLOCK}blk) Z={Z} H={H} D={D} ns={ns}")
        print(f"  {'variant':<8}{'bwd ms':>10}{'vs redq':>10}")
        base = None
        for name, var, ws in VARIANTS:
            try:
                out = _fwd(var, ws, q, k, v, so_kv, so_q, Lkv, Lq, asc)

                def fn():
                    for t in (q, k, v):
                        t.grad = None
                    out.backward(do, retain_graph=True)

                fn()  # warm / compile / autotune
                ms = triton.testing.do_bench(fn, warmup=25, rep=100)
                if name == "redq":
                    base = ms
                spd = f"{base / ms:.2f}x" if base else "-"
                print(f"  {name:<8}{ms:>10.3f}{spd:>10}")
            except Exception as e:
                print(f"  {name:<8} ERROR {type(e).__name__}: {str(e)[:50]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acc", action="store_true", help="accuracy only")
    ap.add_argument("--perf", action="store_true", help="perf only")
    ap.add_argument("--ns", type=int, default=2, help="bwd num_stages")
    args = ap.parse_args()
    do_acc = args.acc or not args.perf
    do_perf = args.perf or not args.acc
    # (Lq, Lkv, Z): KV=1 correct; KV>=2 exposes the autoWS bug.
    acc_cfgs = [(256, 64, 2), (256, 128, 2), (256, 192, 2)]
    perf_shapes = [(256, 256, 4), (256, 1024, 4)]
    if do_acc:
        run_accuracy(acc_cfgs, ns=args.ns)
    if do_perf:
        run_perf(perf_shapes, ns=args.ns)


if __name__ == "__main__":
    main()
