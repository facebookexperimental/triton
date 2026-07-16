"""Accuracy + perf harness for the ported HSTU SELF-attention kernels (MetaMain2 OSS).

Variants:
  * triton - hammer-template Triton self-attn (triton_hstu_mha), fwd+bwd. Trusted ref.
  * tlx    - hand-written TLX warp-specialized self-attn (tlx_bw_hstu_mha), Blackwell.
  * autows - the SAME triton_hstu_mha compiled under meta-WS (HSTU_SELF_AUTOWS=1) +
             the TLX-matching dq-reduce bwd config (BM=BN=128, ns=2, TMEM reuse,
             mem-plan search).

Accuracy (--acc, default): checks fwd output + dq/dk/dv grads
  - triton vs a torch-float HSTU-SiLU reference (both causal and non-causal),
  - tlx vs triton (the byte-ish correctness check; same hammer math).

Perf (--perf): fwd + bwd latency per variant via triton.testing.do_bench, N-rep
  mean/std, and speedup vs TLX (mirrors hstu_cross_attn/bench_bwd.py run_perf).
  Because autoWS (meta-WS on; HSTU_SELF_AUTOWS is a tl.constexpr baked at import)
  cannot share a process with TLX (TLX asserts under meta-WS), EACH variant is
  timed in its OWN subprocess with the right env, and this parent tabulates.

Usage (from this dir, on a Blackwell GPU):
  CUDA_VISIBLE_DEVICES=1 ~/.conda/envs/metamain2/bin/python bench_self.py            # accuracy
  CUDA_VISIBLE_DEVICES=1 ~/.conda/envs/metamain2/bin/python bench_self.py --perf --nrep 5
  ... --perf --variants autows,tlx --seqlens 512,1024,4096 --batch 2
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import triton  # noqa: E402

import triton_hstu_attention as A  # noqa: E402
import tlx_bw_hstu_attention as T  # noqa: E402

D = 128
# heads/sparsity are env-driven so they flow to the per-variant perf subprocesses
# (D102650040 production shape: heads=4, seq=4096, sparsity=0.95, batch=512).
H = int(os.environ.get("BENCH_HEADS", "2"))


def generate_sparse_seq_len(size, max_seq_len, sparsity, device, generator=None):
    """Port of generative_recommenders.common.generate_sparse_seq_len (the
    fbsource HSTU bench / D102650040 shape). `sparsity` is a DENSITY knob --
    higher => LONGER sequences. Uniform: sparsity>=0.5 -> U[(2s-1)*max, max);
    sparsity<0.5 -> U[0, 2s*max). Clamped >=1 so every segment is non-empty."""
    if sparsity == 0.0:
        return torch.ones(size, device=device, dtype=torch.int64)
    if sparsity == 1.0:
        return torch.full((size,), max_seq_len, device=device, dtype=torch.int64)
    if sparsity >= 0.5:
        lo, hi = int((2 * sparsity - 1.0) * max_seq_len), max_seq_len
    else:
        lo, hi = 0, int(2 * sparsity * max_seq_len)
    lo, hi = max(lo, 1), max(hi, 2)
    return torch.randint(lo, hi, (size,), device=device, dtype=torch.int64, generator=generator)


def make(L, Z, dtype=torch.bfloat16, sparsity=None, seed=1001):
    """Build (q,k,v,do,so,asc). Default (sparsity=None) = uniform dense segments
    (every batch item seq-len = L). sparsity set = jagged variable-length segments
    (perf-only; torch_ref assumes uniform L). L is the per-item MAX seq-len."""
    dev = "cuda"
    if sparsity is None:
        lens = torch.full((Z,), L, device=dev, dtype=torch.int64)
    else:
        gen = torch.Generator(device=dev).manual_seed(seed)
        lens = generate_sparse_seq_len(Z, L, sparsity, dev, gen)
    so = torch.zeros(Z + 1, device=dev, dtype=torch.int64)
    torch.cumsum(lens, 0, out=so[1:])
    t = int(so[-1])
    g = lambda: torch.randn(t, H, D, device=dev, dtype=dtype)  # noqa: E731
    q = g().requires_grad_(True)
    k = g().requires_grad_(True)
    v = g().requires_grad_(True)
    asc = torch.tensor(1.0 / L, device=dev, dtype=torch.float32)
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


def run_accuracy(shapes):
    for (L, Z) in shapes:
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


# ---- perf ---------------------------------------------------------------
# Per-variant env applied BEFORE the child imports the kernel (autoWS flags are
# tl.constexpr baked at import). "autows" uses the TLX-matching dq-reduce config.
_PERF_ENV = {
    "triton": {},
    "tlx": {},
    "autows": {
        "HSTU_SELF_AUTOWS": "1", "HSTU_SELF_DQ_REDUCE": "1", "HSTU_SELF_DQ_REUSE": "1",
        "HSTU_SELF_DP": "1", "HSTU_SELF_AUTOWS_BWD_BM": "128", "HSTU_SELF_AUTOWS_BWD_BN": "128",
        "HSTU_SELF_AUTOWS_BWD_STAGES": "2", "HSTU_SELF_AUTOWS_WARPS": "4",
        "HSTU_SELF_PIN": "1", "HSTU_SELF_DQ_ITERS": "4", "TRITON_WS_SMEM_PLAN_SEARCH": "1",
    },
}


def _bench_one(variant, L, Z, nrep):
    """Time fwd + bwd of ONE variant in THIS process (env already set by parent).
    Prints a machine-parseable PERF line."""
    import statistics
    torch.manual_seed(0)
    _sp = os.environ.get("BENCH_SPARSITY")
    q, k, v, do, so, asc = make(L, Z, sparsity=float(_sp) if _sp else None)
    run = run_tlx if variant == "tlx" else run_triton  # triton & autows share triton_hstu_mha
    kw = {"causal": True} if variant == "tlx" else {}
    meta_ws = variant == "autows"
    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = meta_ws
        triton.knobs.nvidia.disable_wsbarrier_reorder = True
        fwd = lambda: run(q, k, v, so, L, asc, **kw)  # noqa: E731
        out = fwd()  # warm / compile / autotune
        torch.cuda.synchronize()
        fwd_s = [triton.testing.do_bench(fwd, warmup=25, rep=100) for _ in range(nrep)]

        def bwd():
            for t in (q, k, v):
                t.grad = None
            out.backward(do, retain_graph=True)

        bwd()  # warm
        bwd_s = [triton.testing.do_bench(bwd, warmup=25, rep=100) for _ in range(nrep)]
    fm, bm = statistics.mean(fwd_s), statistics.mean(bwd_s)
    fsd = statistics.stdev(fwd_s) if len(fwd_s) > 1 else 0.0
    bsd = statistics.stdev(bwd_s) if len(bwd_s) > 1 else 0.0
    print(f"PERF {variant} L={L} Z={Z} fwd={fm:.4f} fwd_sd={fsd:.4f} bwd={bm:.4f} bwd_sd={bsd:.4f}")


def run_perf(shapes, variants, nrep=1, heads=None, sparsity=None, timeout=3600):
    """Time each variant's fwd+bwd in its own subprocess and tabulate (speedup vs tlx).
    heads/sparsity flow to the children via env (BENCH_HEADS / BENCH_SPARSITY)."""
    import re
    import subprocess

    hh = heads if heads is not None else H
    inp = "uniform-dense" if sparsity is None else f"jagged sparsity={sparsity}"
    print(f"\n=== Perf (self-attn fwd/bwd latency, nrep={nrep}, heads={hh}, {inp}) "
          f"— each variant in its own process ===")
    for (L, Z) in shapes:
        print(f"\n  maxL={L} Z={Z} H={hh} D={D}")
        hdr = f"  {'variant':<10}{'fwd ms':>10}{'bwd ms':>10}"
        hdr += f"{'fwd sd':>9}{'bwd sd':>9}" if nrep > 1 else f"{'fwd/tlx':>9}{'bwd/tlx':>9}"
        print(hdr)
        rows = []
        for var in variants:
            env = dict(os.environ)
            env.update(_PERF_ENV[var])
            if heads is not None:
                env["BENCH_HEADS"] = str(heads)
            if sparsity is not None:
                env["BENCH_SPARSITY"] = str(sparsity)
            try:
                r = subprocess.run(
                    [sys.executable, os.path.abspath(__file__), "--bench-one", var,
                     str(L), str(Z), str(nrep)],
                    env=env, capture_output=True, text=True, timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                print(f"  {var:<10} TIMEOUT/HANG")
                continue
            m = re.search(r"PERF \S+ L=\d+ Z=\d+ fwd=(\S+) fwd_sd=(\S+) bwd=(\S+) bwd_sd=(\S+)",
                          r.stdout)
            if not m:
                tail = (r.stdout + r.stderr).strip().splitlines()[-1:] or ["(no output)"]
                print(f"  {var:<10} ERROR: {tail[0][:70]}")
                continue
            fm, fsd, bm, bsd = map(float, m.groups())
            rows.append((var, fm, fsd, bm, bsd))
        tlx = next((x for x in rows if x[0] == "tlx"), None)
        for var, fm, fsd, bm, bsd in rows:
            if nrep > 1:
                print(f"  {var:<10}{fm:>10.4f}{bm:>10.4f}{fsd:>9.4f}{bsd:>9.4f}")
            else:
                fx = f"{tlx[1] / fm:.2f}x" if tlx and fm else "-"
                bx = f"{tlx[3] / bm:.2f}x" if tlx and bm else "-"
                print(f"  {var:<10}{fm:>10.3f}{bm:>10.3f}{fx:>9}{bx:>9}")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--perf", action="store_true", help="run the fwd/bwd latency benchmark")
    ap.add_argument("--acc", action="store_true", help="run the accuracy check (default)")
    ap.add_argument("--nrep", type=int, default=1,
                    help="do_bench repetitions per point; >1 reports mean + std")
    ap.add_argument("--variants", default="autows,tlx,triton",
                    help="comma subset of autows,tlx,triton (perf)")
    ap.add_argument("--seqlens", default=None, help="comma L list, e.g. 256,512,1024,4096")
    ap.add_argument("--batch", type=int, default=None, help="batch Z (default 2)")
    ap.add_argument("--heads", type=int, default=None, help="num heads H (perf; default 2)")
    ap.add_argument("--sparsity", type=float, default=None,
                    help="jagged density knob (0.95 = D102650040); omit for uniform-dense")
    # hidden per-variant subprocess entry: --bench-one <variant> <L> <Z> <nrep>
    ap.add_argument("--bench-one", nargs=4, default=None, help=argparse.SUPPRESS,
                    metavar=("VARIANT", "L", "Z", "NREP"))
    args = ap.parse_args()

    if args.bench_one:
        var, L, Z, nrep = args.bench_one
        _bench_one(var, int(L), int(Z), int(nrep))
        return

    Z = args.batch if args.batch is not None else 2
    if args.seqlens:
        shapes = [(int(x), Z) for x in args.seqlens.split(",")]
    else:
        shapes = [(256, 4), (512, 2)] if args.batch is None else [(256, Z), (512, Z)]

    do_acc = args.acc or not args.perf
    do_perf = args.perf or not args.acc
    if do_acc:
        run_accuracy(shapes)
    if do_perf:
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]
        run_perf(shapes, variants, nrep=args.nrep, heads=args.heads, sparsity=args.sparsity)


if __name__ == "__main__":
    main()
