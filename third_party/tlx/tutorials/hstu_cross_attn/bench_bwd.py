"""Combined accuracy + perf benchmark for the HSTU cross-attention backward
(reduce_dq), across the variants:

  * redq    - triton non-WS reduce_dq (trusted reference kernel)
  * autows  - triton automatic warp specialization (meta-WS)
  * tlx     - hand-written TLX warp-specialized reduce_dq (attn_bwd_ws)
  * tlx_2kv - TLX 2-KV-block data-partitioned reduce_dq (attn_bwd_ws_2kv);
              two independent MMA groups per program. SHARED-KV ONLY (V aliases
              K), so it only runs under --shared-kv.

Accuracy: rel-L2 of dq/dk(/dv) vs a torch-autograd float reference, plus the
count of dq rows that diverge from a reference kernel grouped into Q-blocks.

Perf: backward latency per variant (autograd backward on a prebuilt fwd graph),
reported with speedup relative to redq.

Usage (from this directory):
  ~/.conda/envs/metamain2/bin/python bench_bwd.py                 # acc + perf
  ~/.conda/envs/metamain2/bin/python bench_bwd.py --acc
  ~/.conda/envs/metamain2/bin/python bench_bwd.py --perf
  # 2-KV-block DP variant (shared-KV): compare tlx_2kv vs tlx
  ~/.conda/envs/metamain2/bin/python bench_bwd.py --shared-kv --variants tlx,tlx_2kv,redq
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
BLOCK = 64  # BLOCK_M == BLOCK_N used by the redq/autows benchmark configs

VARIANTS = [
    ("redq", xa.BwdVariant.TRITON_REDQ, "0"),
    ("autows", xa.BwdVariant.TRITON_AUTOWS, "1"),
    ("tlx", xa.BwdVariant.TLX, "0"),
    ("tlx_2kv", xa.BwdVariant.TLX_2KV, "0"),
]
# Variants that require shared-KV (V aliases K); only run under --shared-kv.
SHARED_KV_ONLY = {"tlx_2kv"}


def force(ns=2, bm=BLOCK, bn=BLOCK):
    """Pin fwd num_stages>=1 (autoWS asserts on 0) and the bwd block sizes, and
    shrink the autotune space to one config for fast, deterministic runs. Only
    touches redq/autows (_hstu_attn_bwd_redq) + the fwd; the TLX kernels keep
    their own autotune configs."""
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
    if hasattr(xa, "_hstu_attn_bwd_redq_2kv"):
        c2 = xa._hstu_attn_bwd_redq_2kv.configs[0]
        c2.num_stages = ns
        c2.kwargs["BLOCK_M"] = bm
        c2.kwargs["BLOCK_N"] = bn
        xa._hstu_attn_bwd_redq_2kv.configs = [c2]
    xa.set_fwd_variant(xa.FwdVariant.TRITON)


def make(Lq, Lkv, Z, shared=False):
    tq, tk = Z * Lq, Z * Lkv
    g = lambda n: torch.randn(n, H, D, device="cuda", dtype=torch.bfloat16)
    q = g(tq).requires_grad_(True)
    k = g(tk).requires_grad_(True)
    # shared-KV: V aliases K (same tensor), so k.grad accumulates dk + dv.
    v = k if shared else g(tk).requires_grad_(True)
    so_kv = torch.arange(0, tk + 1, Lkv, device="cuda", dtype=torch.int64)
    so_q = torch.arange(0, tq + 1, Lq, device="cuda", dtype=torch.int64)
    asc = torch.tensor(1.0 / Lkv, device="cuda", dtype=torch.float32)
    do = g(tq)
    return q, k, v, do, so_kv, so_q, asc


def torch_ref(q, k, v, do, so_kv, so_q, asc, softmax=True, shared=False):
    """Float autograd reference on the HSTU cross-attn forward (asymmetric Q/KV).
    When shared, K and V are one leaf so the returned dk = dk + dv (dv is None)."""
    qf = q.detach().float().requires_grad_(True)
    kf = k.detach().float().requires_grad_(True)
    vf = kf if shared else v.detach().float().requires_grad_(True)
    scale = asc.item()
    outs = []
    for z in range(so_kv.numel() - 1):
        qs, qe = int(so_q[z]), int(so_q[z + 1])
        ks, ke = int(so_kv[z]), int(so_kv[z + 1])
        qk = torch.einsum("qhd,khd->hqk", qf[qs:qe], kf[ks:ke]) * (1.0 / D)
        S = torch.softmax(qk, -1) if softmax else (qk * torch.sigmoid(qk)) * scale
        outs.append(torch.einsum("hqk,khd->qhd", S, vf[ks:ke]))
    torch.cat(outs, 0).backward(do.float())
    return qf.grad, kf.grad, (None if shared else vf.grad)


def rel_l2(a, b):
    return (torch.norm(a.float() - b.float()) / (torch.norm(b.float()) + 1e-12)).item()


def max_abs(a, b):
    return (a.float() - b.float()).abs().max().item()


def bad_qblocks(test, ref, Lq, tol=5e-3):
    pr = (test.float() - ref.float()).abs().amax(dim=(1, 2))
    rm = torch.arange(pr.numel(), device=pr.device) % Lq
    bad = pr > tol
    return int(bad.sum()), sorted(set((rm[bad] // BLOCK).tolist()))


def _fwd(var, ws, q, k, v, so_kv, so_q, Lkv, Lq, asc, shared=False):
    xa.set_bwd_variant(var)
    os.environ["TRITON_USE_META_WS"] = ws
    return xa.triton_bw_hstu_mha_wrapper(
        max_seq_len=Lkv, alpha=1.0 / D, q=q, k=k, v=v, seq_offsets=so_kv,
        attn_scale=asc, max_q_len=Lq, seq_offsets_q=so_q, num_softmax_heads=H,
        shared_kv=shared, enable_tma=True,
    )


def run_accuracy(configs, ns=2, ref_name="redq", variants=None, shared=False):
    """rel-L2 is vs the torch-float ref (dv skipped under shared-KV); the dq
    bad-Q-block count is vs the byte-reference kernel `ref_name`."""
    sel = VARIANTS if variants is None else [x for x in VARIANTS if x[0] in variants]
    ref_var, ref_ws = next((v, w) for n, v, w in VARIANTS if n == ref_name)
    mode = "shared-KV" if shared else "separate-KV"
    print(f"\n=== Accuracy ({mode}): dq/dk/dv rel-L2 vs torch-float ref; "
          f"dq bad-Q-blocks vs {ref_name} ===")
    for Lq, Lkv, Z in configs:
        force(ns)
        torch.manual_seed(0)
        q, k, v, do, so_kv, so_q, asc = make(Lq, Lkv, Z, shared=shared)
        rq, rk, rv = torch_ref(q, k, v, do, so_kv, so_q, asc, shared=shared)
        for t in (q, k, v):
            t.grad = None
        _fwd(ref_var, ref_ws, q, k, v, so_kv, so_q, Lkv, Lq, asc,
             shared=shared).backward(do)
        ref = q.grad.clone()
        print(f"\n  Lq={Lq}({Lq // BLOCK}blk) Lkv={Lkv}({Lkv // BLOCK}blk) Z={Z} ns={ns}")
        print(f"  {'variant':<8}{'dq relL2':>10}{'dk relL2':>10}{'dv relL2':>10}"
              f"{'dq maxabs':>11}  dq-vs-{ref_name} bad-Qblk")
        for name, var, ws in sel:
            for t in (q, k, v):
                t.grad = None
            try:
                out = _fwd(var, ws, q, k, v, so_kv, so_q, Lkv, Lq, asc, shared=shared)
                out.backward(do)
                dq, dk = q.grad.clone(), k.grad.clone()
                dv_s = "-" if (shared or rv is None) else f"{rel_l2(v.grad, rv):>10.2e}"
                n, bm = bad_qblocks(dq, ref, Lq)
                print(f"  {name:<8}{rel_l2(dq, rq):>10.2e}{rel_l2(dk, rk):>10.2e}"
                      f"{dv_s:>10}{max_abs(dq, rq):>11.2e}  {n} rows Qblk{bm}")
            except Exception as e:
                print(f"  {name:<8} ERROR {type(e).__name__}: {str(e)[:50]}")


def run_perf(shapes, ns=2, variants=None, shared=False):
    sel = VARIANTS if variants is None else [x for x in VARIANTS if x[0] in variants]
    mode = "shared-KV" if shared else "separate-KV"
    print(f"\n=== Perf ({mode}): backward latency per variant "
          f"(autograd bwd on prebuilt fwd graph) ===")
    for Lq, Lkv, Z in shapes:
        force(ns)
        torch.manual_seed(0)
        q, k, v, do, so_kv, so_q, asc = make(Lq, Lkv, Z, shared=shared)
        print(f"\n  Lq={Lq} Lkv={Lkv}({Lkv // BLOCK}blk) Z={Z} H={H} D={D} ns={ns}")
        print(f"  {'variant':<8}{'bwd ms':>10}{'vs redq':>10}")
        base = None
        for name, var, ws in sel:
            try:
                out = _fwd(var, ws, q, k, v, so_kv, so_q, Lkv, Lq, asc, shared=shared)

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
    ap.add_argument("--shared-kv", action="store_true", dest="shared_kv",
                    help="run shared-KV (V aliases K); required for tlx_2kv")
    ap.add_argument("--ref", choices=["redq", "tlx"], default="redq",
                    help="byte-reference kernel for the dq bad-Q-block count")
    known = [n for n, _, _ in VARIANTS]
    ap.add_argument("--variants", default=",".join(known),
                    help=f"comma-separated subset (default all: {','.join(known)})")
    args = ap.parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    bad = [v for v in variants if v not in known]
    if bad:
        ap.error(f"unknown variant(s) {bad}; choose from {known}")

    shared = args.shared_kv
    needs_shared = [v for v in variants if v in SHARED_KV_ONLY]
    if needs_shared and not shared:
        print(f"note: {needs_shared} require shared-KV; enabling --shared-kv")
        shared = True
    if shared:
        # tlx as the byte-reference (proven shared-KV kernel) unless overridden.
        ref = "tlx" if args.ref == "redq" else args.ref
    else:
        ref = args.ref
        variants = [v for v in variants if v not in SHARED_KV_ONLY]

    do_acc = args.acc or not args.perf
    do_perf = args.perf or not args.acc
    if shared:
        # 2kv uses BLOCK_N=128 (2*BLOCK_N=256 per pair); use Lkv multiples of 128,
        # incl. an odd number of KV blocks (384) to exercise the partial tail pair.
        acc_cfgs = [(256, 256, 2), (256, 384, 2), (256, 512, 2)]
        perf_shapes = [(256, 256, 4), (256, 512, 4), (256, 1024, 4)]
    else:
        acc_cfgs = [(256, 64, 2), (256, 128, 2), (256, 192, 2)]
        perf_shapes = [(256, 256, 4), (256, 1024, 4)]
    if do_acc:
        run_accuracy(acc_cfgs, ns=args.ns, ref_name=ref, variants=variants, shared=shared)
    if do_perf:
        run_perf(perf_shapes, ns=args.ns, variants=variants, shared=shared)


if __name__ == "__main__":
    main()
