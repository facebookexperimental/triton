"""SKC Phase B timing worker — fa4_worker protocol + shim binding.

Run under the FA4 venv with FA_DISABLE_2CTA set in the environment BEFORE
Python starts (utils.py reads it at config time):

    env -u LD_LIBRARY_PATH FA_DISABLE_2CTA=1 <fa4-venv-python> \
        bench/skc_cute_worker.py --mode fwd --s 4096 --binding fwd_1cta \
        --regs solver_liveness

--binding '' (empty) = identity shim (M2 gate: must tie fa4_1cta).
--regs selects a candidate from the binding's reg_candidates (E2);
--split-p / --kv-clamp / --q-clamp are E3 perturbation knobs.
One binding per process (compile_cache is override-blind).
Emits one JSON line incl. binding_hash, audit, and effective cluster shape.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch

PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG))

QUANTILES = [0.5, 0.2, 0.8]


def bench(fn, warmup, rep, quantiles=QUANTILES):
    from triton.testing import do_bench
    ms = do_bench(fn, warmup=warmup, rep=rep, quantiles=quantiles)
    return list(ms) if isinstance(ms, (list, tuple)) else [ms]


def rel_err(a, b):
    return (a.float() - b.float()).abs().max().item() / max(
        b.float().abs().max().item(), 1e-9)


def build_shim_binding(spec, args):
    """Translate a binder JSON + CLI selections into the shim's flat dict."""
    if spec is None:
        return None
    b = {}
    if spec["kind"] == "fwd":
        b["verify_kv_stage"] = spec["verifies"]["kv_stage"]["expect"]
        b["verify_q_stage"] = spec["verifies"]["q_stage"]["expect"]
        if args.regs:
            b["regs"] = spec["overrides"]["reg_candidates"][args.regs]
        if args.split_p is not None:
            b["split_P_arrive"] = args.split_p
        if args.kv_clamp is not None:
            b["kv_stage_clamp"] = args.kv_clamp
            b.pop("verify_kv_stage")
    else:
        b["verify_Q_stage"] = spec["verifies"]["Q_stage"]["expect"]
        if args.regs:
            b["regs"] = spec["overrides"]["reg_candidates"][args.regs]
        if args.q_clamp is not None:
            b["Q_stage_clamp"] = args.q_clamp
            b.pop("verify_Q_stage")
    return b


def main():
    ap = argparse.ArgumentParser(prog="skc_cute_worker")
    ap.add_argument("--mode", choices=["fwd", "bwd"], default="fwd")
    ap.add_argument("--b", type=int, default=4)
    ap.add_argument("--h", type=int, default=32)
    ap.add_argument("--s", type=int, required=True)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--rep", type=int, default=500)
    ap.add_argument("--binding", default="",
                    help="binding name under skc_cute/bindings/ (no .json); "
                         "empty = identity shim")
    ap.add_argument("--regs", default="", help="reg_candidates key (E2)")
    ap.add_argument("--split-p", type=int, default=None)
    ap.add_argument("--kv-clamp", type=int, default=None)
    ap.add_argument("--q-clamp", type=int, default=None)
    args = ap.parse_args()

    out = {"ms_median": 0.0, "ms_lo": 0.0, "ms_hi": 0.0, "ok": False}
    try:
        import os
        out["fa_disable_2cta"] = os.environ.get("FA_DISABLE_2CTA", "")
        # CuTe DSL argparses sys.argv at import — hide our flags.
        sys.argv = sys.argv[:1]

        spec = None
        if args.binding:
            spec = json.loads(
                (PKG / "skc_cute" / "bindings" / f"{args.binding}.json")
                .read_text())
            assert spec["kind"] == args.mode
        shim_binding = build_shim_binding(spec, args)

        from skc_cute import driver
        info = driver.install(args.mode, shim_binding)
        out.update(info)

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

        if args.mode == "fwd":
            from skc_cute.shim_fwd import SKCForwardSm100
            out["audit"] = SKCForwardSm100._skc_audit
        else:
            from skc_cute.shim_bwd import SKCBackwardSm100
            out["audit"] = SKCBackwardSm100._skc_audit

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
                fwd()

            total()
            torch.cuda.synchronize()
            q32, k32, v32 = (t.detach().transpose(1, 2).float()
                             .requires_grad_(True) for t in (q, k, v))
            o32 = torch.nn.functional.scaled_dot_product_attention(
                q32, k32, v32, scale=scale)
            o32.backward(do.transpose(1, 2).float())
            rels = [rel_err(t.grad.transpose(1, 2), r.grad)
                    for t, r in ((q, q32), (k, k32), (v, v32))]
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
    print(json.dumps(out, default=str), flush=True)


if __name__ == "__main__":
    main()
