"""Self-contained FA4 timing worker for bench_bars — run under the FA4 venv:

    env -u LD_LIBRARY_PATH /projects/kzhou6/hwu27/baselines/.venv-fa4/bin/python \
        bench/fa4_worker.py --mode fwd --b 4 --h 32 --s 4096 --d 128

Uses flash_attn.cute.interface.flash_attn_func; layout is (B, S, H, D).
Correctness is gated vs torch SDPA in fp32 (fwd rel < 1e-2, grad rel < 3e-2).
Prints exactly one JSON line to stdout:

    {"ms_median": ..., "ms_lo": ..., "ms_hi": ..., "ok": ...[, "reason": ...]}

bwd mode times the full fwd+bwd and fwd alone (graph still built) and reports
(total - fwd median); quantiles are those of the fwd+bwd distribution shifted
by the fwd median.  See bench/README.md.
"""

import argparse
import json
import math
import sys

import torch

QUANTILES = [0.5, 0.2, 0.8]


def bench(fn, warmup, rep, quantiles=QUANTILES):
    try:
        from triton.testing import do_bench
        ms = do_bench(fn, warmup=warmup, rep=rep, quantiles=quantiles)
        # do_bench returns a bare float for a single quantile
        return list(ms) if isinstance(ms, (list, tuple)) else [ms]
    except ImportError:
        pass
    # minimal clone of do_bench (times in ms, L2 flushed between reps)
    fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        fn()
    end.record()
    torch.cuda.synchronize()
    est = start.elapsed_time(end) / 5
    n_warmup = max(1, int(warmup / est))
    n_rep = max(1, int(rep / est))
    for _ in range(n_warmup):
        fn()
    cache = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda")
    times = []
    for _ in range(n_rep):
        cache.zero_()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    t = torch.tensor(times, dtype=torch.float64)
    return [torch.quantile(t, q).item() for q in quantiles]


def rel_err(a, b):
    return (a.float() - b.float()).abs().max().item() / max(
        b.float().abs().max().item(), 1e-9)


def main():
    ap = argparse.ArgumentParser(prog="fa4_worker")
    ap.add_argument("--mode", choices=["fwd", "bwd"], default="fwd")
    ap.add_argument("--b", type=int, default=4)
    ap.add_argument("--h", type=int, default=32)
    ap.add_argument("--s", type=int, required=True)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--rep", type=int, default=500)
    args = ap.parse_args()

    out = {"ms_median": 0.0, "ms_lo": 0.0, "ms_hi": 0.0, "ok": False}
    try:
        # The CuTe DSL inside flash_attn parses sys.argv at import time (a
        # "Process diagnostic status" argparse) and dies on our worker flags;
        # our own args are already parsed, so hide them from the import.
        sys.argv = sys.argv[:1]
        from flash_attn.cute.interface import flash_attn_func

        torch.manual_seed(0)
        scale = 1.0 / math.sqrt(args.d)
        q, k, v = (torch.randn(args.b, args.s, args.h, args.d, device="cuda",
                               dtype=torch.float16) for _ in range(3))

        def fwd():
            r = flash_attn_func(q, k, v, softmax_scale=scale, causal=False)
            return r[0] if isinstance(r, tuple) else r

        o = fwd()
        torch.cuda.synchronize()
        # fp32 SDPA reference ((B, H, S, D) layout)
        ref = torch.nn.functional.scaled_dot_product_attention(
            q.detach().transpose(1, 2).float(),
            k.detach().transpose(1, 2).float(),
            v.detach().transpose(1, 2).float(), scale=scale)
        rel = rel_err(o.transpose(1, 2), ref)
        del o, ref

        if args.mode == "fwd":
            if rel < 1e-2:
                med, lo, hi = bench(fwd, args.warmup, args.rep)
                out.update(ms_median=med, ms_lo=lo, ms_hi=hi, ok=True)
            else:
                out["reason"] = f"fwd rel {rel:.2e} >= 1e-2"
        else:
            for t in (q, k, v):
                t.requires_grad_(True)
            do = torch.randn_like(q)

            def total():
                q.grad = k.grad = v.grad = None
                fwd().backward(do)

            def fwd_only():
                # graph still built (inputs require grad) so total - fwd
                # isolates the backward
                fwd()

            total()
            torch.cuda.synchronize()
            q32, k32, v32 = (t.detach().transpose(1, 2).float().
                             requires_grad_(True) for t in (q, k, v))
            o32 = torch.nn.functional.scaled_dot_product_attention(
                q32, k32, v32, scale=scale)
            o32.backward(do.transpose(1, 2).float())
            rels = [
                rel_err(t.grad.transpose(1, 2), r.grad)
                for t, r in ((q, q32), (k, k32), (v, v32))
            ]
            del o32, q32, k32, v32
            if rel < 1e-2 and max(rels) < 3e-2:
                med_t, lo_t, hi_t = bench(total, args.warmup, args.rep)
                fwd_med = bench(fwd_only, args.warmup, args.rep,
                                quantiles=[0.5])[0]
                out.update(ms_median=med_t - fwd_med, ms_lo=lo_t - fwd_med,
                           ms_hi=hi_t - fwd_med, fwd_ms=fwd_med, ok=True)
            else:
                out["reason"] = ("gate: fwd rel {:.2e}, grad rel "
                                 "dQ={:.2e} dK={:.2e} dV={:.2e}".format(
                                     rel, *rels))
    except Exception as e:
        out["reason"] = f"{type(e).__name__}: {e}"
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
