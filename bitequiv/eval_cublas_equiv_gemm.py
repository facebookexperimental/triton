"""Fuzz evaluation: is the cuBLAS-equivalent Triton GEMM really bit-identical to cuBLAS?

A timeout-bounded infinite loop. Each round:
  1. draw a random matmul shape (M, N, K) + dtype;
  2. draw R random input tensors for it;
  3. feed each input to BOTH our API and cuBLAS run directly, and check the two
     outputs are byte-for-byte equal.

Every mismatch (our API returned a result that is NOT bit-identical to cuBLAS -- a
soundness bug) is written to a temp file for debugging. A shape the API declines
(`CublasUnsupportedShape`) is counted separately: that is an honest "no
reconstruction", NOT a mismatch. The bit-consistency rate is reported periodically.

IMPORTANT: the fuzz inputs use seeds disjoint from the ones the API calibrates on,
so this tests inputs the reconstruction was NOT tuned for.

Run:
  python -m bitequiv.eval_cublas_equiv_gemm --timeout 300
  python -m bitequiv.eval_cublas_equiv_gemm --timeout 3600 --dtypes fp16,fp8 --R 5
"""
import argparse
import json
import os
import random
import time

import torch

from bitequiv import cublas_equiv_gemm as _G
from bitequiv.cublas_equiv_gemm import (
    CublasNeedRuntimeMatch,
    CublasUnsupportedPlatform,
    CublasUnsupportedShape,
    cublas_equivalent_gemm,
    cublas_equivalent_scaled_mm,
    cublas_matmul,
)

DEVICE = "cuda"
_F8 = torch.float8_e4m3fn


def _bits_eq(x, y):
    return torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8))


def random_shape(rng, dtypes, mode, max_dim, max_mn, max_k, fp8_round16):
    """Draw a shape. Two modes, both free of the rounding and regime buckets an earlier
    version had (those made ~98% of draws aligned, when only 1.6% of random shapes are, and
    hid every hard family).

      random  -- M, N, K independent uniform draws. This is the "any shape at all" case.
      extreme -- M, N small and K large: the skinny+deep corner where cuBLAS reaches for
                 split-K, and where its SIMT gemv fallbacks live. Uniform sampling barely
                 visits it (only ~1.2% of random draws have a dim under 100), so it needs its
                 own mode to be measured at all.

    `max_dim` / `max_mn` / `max_k` bound memory and run time; they are not restrictions on the
    shape class. fp8 is the one place a rounding is justified: cuBLAS itself refuses fp8 unless
    every dim is a multiple of 16, so with `fp8_round16` we sample uniformly among the shapes
    it can actually run rather than spending the whole budget on shapes with no reference."""
    dtype = rng.choice(dtypes)
    if mode == "extreme":
        M, N, K = rng.randint(1, max_mn), rng.randint(1, max_mn), rng.randint(1, max_k)
    else:
        M, N, K = rng.randint(1, max_dim), rng.randint(1, max_dim), rng.randint(1, max_k)
    if dtype == "fp8" and fp8_round16:
        M, N, K = (max(16, (v // 16) * 16) for v in (M, N, K))
    return M, N, K, dtype


def operand_bytes(M, N, K, dtype):
    """Device memory the operands plus the fp16 output need, ignoring any split-K workspace."""
    elem = 1 if dtype == "fp8" else 2
    return (M * K + K * N) * elem + M * N * 2


def _is_oom(exc):
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def make_inputs(M, N, K, dtype, seed, mode):
    """Random inputs. `order` mode = magnitude spread + alternating sign along K (stresses
    the K reduction order -- the hardest test for a split-K reconstruction); `plain` mode =
    ordinary small-scale gaussians. Magnitudes stay in range so the output is finite (fp8
    values must fit e4m3, max 448). fp8 returns b as [K,N] column-major."""
    torch.manual_seed(seed)
    sign = torch.where(torch.arange(K, device=DEVICE) % 2 == 0, 1.0, -1.0).unsqueeze(0)
    if mode == "order":
        lo, hi = (-1, 1) if dtype == "fp8" else (-3, 3)  # fp8: 0.1..10 (no overflow)
        scale = torch.logspace(lo, hi, K, device=DEVICE, dtype=torch.float32).unsqueeze(0)
        af = torch.randn(M, K, device=DEVICE) * scale * sign
        if dtype == "fp8":
            bf = torch.randn(N, K, device=DEVICE) * 0.25
        else:
            bf = torch.randn(K, N, device=DEVICE) * 0.05
    else:
        af = torch.randn(M, K, device=DEVICE) * 0.3
        bf = (torch.randn(N, K, device=DEVICE) if dtype == "fp8" else torch.randn(K, N, device=DEVICE)) * 0.3

    if dtype == "fp8":
        return af.to(_F8), bf.to(_F8).t()  # b -> [K,N] column-major
    dt = torch.bfloat16 if dtype == "bf16" else torch.float16
    return af.to(dt), bf.to(dt)


def dtype_kind(dtype):
    """The key the plan cache uses for this eval dtype."""
    return "fp8" if dtype == "fp8" else ("bf16" if dtype == "bf16" else "fp16")


def our_api(a, b, dtype, out_dtype, enable_rt):
    if dtype == "fp8":
        return cublas_equivalent_scaled_mm(a, b, 1.0, 1.0, out_dtype, enable_runtime_match=enable_rt)
    return cublas_equivalent_gemm(a, b, out_dtype, enable_runtime_match=enable_rt)


def load_done(path):
    """Resume support: the (M,N,K,dtype) keys already recorded in a shape log."""
    done = set()
    if not path or not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done.add((r["M"], r["N"], r["K"], r["dtype"]))
            except Exception:
                pass  # a torn last line from a kill; the shape just gets redrawn
    return done


def run(timeout, report_every, R, dtypes, seed, enable_rt, mode, max_dim, max_mn, max_k, fp8_round16, max_bytes,
        shape_log):
    rng = random.Random(seed)
    mm_path = f"/tmp/cublas_equiv_mismatches_{os.getpid()}.jsonl"
    mm_file = open(mm_path, "a")
    done = load_done(shape_log)
    sf = open(shape_log, "a") if shape_log else None

    st = {"drawn": 0, "cmp": 0, "match": 0, "mismatch": 0, "no_ref": 0, "nonfinite": 0, "error": 0,
          "too_big": 0, "oom": 0, "dup": 0}
    # distinct shapes by outcome: static (heuristic only) / runtime (byte-compare) / need-rt
    # (skipped in static-only mode) / unsupported (no reconstruction even with a runtime match)
    seen_static, seen_pseudo, seen_runtime, seen_needrt, seen_unsup = set(), set(), set(), set(), set()
    t0 = time.time()
    last = t0

    try:
        prof = _G._platform()
    except CublasUnsupportedPlatform as e:
        print(f"cannot run here: {e}")
        return
    print(f"device: {torch.cuda.get_device_name()} | platform: {prof.name} | cuBLASLt "
          f"{'.'.join(map(str, _G._cublaslt_version()))} from {_G._load_lt()._name} | dtypes={dtypes} R={R} "
          f"timeout={timeout}s enable_runtime_match={enable_rt}")
    if mode == "extreme":
        print(f"EXTREME shapes: M,N ~ uniform[1,{max_mn}], K ~ uniform[1,{max_k}] -- the skinny+deep "
              f"split-K corner")
    else:
        print(f"UNRESTRICTED shapes: M,N ~ uniform[1,{max_dim}], K ~ uniform[1,{max_k}], independent, "
              f"no rounding, no regimes")
    if "fp8" in dtypes and fp8_round16:
        print("fp8 draws rounded to multiples of 16 -- cuBLAS itself will not run fp8 otherwise")
    print(f"memory cap {max_bytes / 2**30:.1f} GiB/shape | resumed with {len(done)} shapes already done")
    print(f"mismatches -> {mm_path} | shape log -> {shape_log}\n", flush=True)

    def report(tag):
        el = time.time() - t0
        scmp = st["match"] + st["mismatch"]
        rate = (100.0 * st["match"] / scmp) if scmp else 100.0
        recon = len(seen_static) + len(seen_pseudo) + len(seen_runtime)
        classified = recon + len(seen_needrt) + len(seen_unsup)
        cov = (100.0 * recon / classified) if classified else 0.0
        print(f"[{tag} {el:6.0f}s] drawn={st['drawn']} classified={classified} "
              f"RECONSTRUCTABLE={cov:5.1f}% ({recon}/{classified}) | "
              f"static={len(seen_static)} pseudo={len(seen_pseudo)} runtime={len(seen_runtime)} "
              f"needRT={len(seen_needrt)} unsup={len(seen_unsup)} | bit-consistent={rate:6.2f}% ({st['match']}/{scmp}) "
              f"mismatch={st['mismatch']} | no_ref={st['no_ref']} too_big={st['too_big']} "
              f"oom={st['oom']} nonfinite={st['nonfinite']} err={st['error']}", flush=True)

    def log_shape(M, N, K, dtype, outcome, bit_ok, nsplit=None):
        if sf is None:
            return
        sf.write(json.dumps({"M": M, "N": N, "K": K, "dtype": dtype, "outcome": outcome,
                             "bit_ok": bit_ok, "nsplit": nsplit}) + "\n")
        sf.flush()
        os.fsync(sf.fileno())

    while time.time() - t0 < timeout:
        M, N, K, dtype = random_shape(rng, dtypes, mode, max_dim, max_mn, max_k, fp8_round16)
        key = (M, N, K, dtype)
        st["drawn"] += 1
        if key in done:
            st["dup"] += 1
            continue
        done.add(key)
        if operand_bytes(M, N, K, dtype) > max_bytes:
            st["too_big"] += 1  # resource skip, not a shape-class skip
            continue
        out_dtype = torch.float16 if dtype == "fp8" else (torch.bfloat16 if dtype == "bf16" else torch.float16)
        use_rt = None  # per-shape mode, decided on the first comparable input
        outcome, bit_ok = None, None
        try:
            for r in range(R):
                s = 100003 + st["drawn"] * 131 + r  # disjoint from the API's calibration seeds
                imode = "order" if (r % 2 == 0) else "plain"
                try:
                    a, b = make_inputs(M, N, K, dtype, s, imode)
                except Exception as e:
                    if _is_oom(e):
                        raise
                    st["error"] += 1
                    outcome = "error"
                    break
                try:
                    ref = cublas_matmul(a, b, out_dtype)
                except Exception as e:
                    if _is_oom(e):
                        raise
                    st["no_ref"] += 1  # cuBLAS itself has no algo for this shape
                    outcome = "no_ref"
                    break
                if not torch.isfinite(ref.float()).all():
                    st["nonfinite"] += 1  # overflow -> byte-compare is meaningless, skip this input
                    continue
                try:
                    if use_rt is None:  # first comparable input: let the API pick a tier
                        try:
                            out = our_api(a, b, dtype, out_dtype, False)
                            use_rt = False
                        except CublasNeedRuntimeMatch:
                            if not enable_rt:
                                seen_needrt.add(key)
                                outcome = "needrt"
                                break  # static-only mode: count it, do not compare
                            out = our_api(a, b, dtype, out_dtype, True)
                            use_rt = True
                        # which tier actually resolved it is recorded in the plan cache
                        origin = _G.plan_origin(M, N, K, dtype_kind(dtype), out_dtype)
                        outcome = origin
                        {"static": seen_static, "pseudo-static": seen_pseudo,
                         "runtime": seen_runtime}.get(origin, seen_runtime).add(key)
                    else:
                        out = our_api(a, b, dtype, out_dtype, use_rt)
                except CublasUnsupportedShape:
                    seen_unsup.add(key)
                    outcome = "unsupported"
                    break
                except Exception as e:
                    if _is_oom(e):
                        raise
                    st["error"] += 1
                    outcome = "error"
                    mm_file.write(json.dumps({"M": M, "N": N, "K": K, "dtype": dtype, "seed": s, "mode": imode,
                                              "error": f"{type(e).__name__}: {str(e)[:120]}"}) + "\n")
                    mm_file.flush()
                    break
                st["cmp"] += 1
                if _bits_eq(out, ref):
                    st["match"] += 1
                    bit_ok = True if bit_ok is None else bit_ok
                else:
                    st["mismatch"] += 1
                    bit_ok = False
                    diff = (out.float() - ref.float()).abs()
                    mm_file.write(json.dumps({
                        "M": M, "N": N, "K": K, "dtype": dtype, "seed": s, "mode": imode,
                        "class": outcome, "max_abs_diff": float(diff.max()),
                        "n_diff": int((diff > 0).sum()), "out_dtype": str(out_dtype)}) + "\n")
                    mm_file.flush()
        except Exception as e:  # OOM anywhere in the shape: drop it and keep the sweep alive
            if not _is_oom(e):
                raise
            st["oom"] += 1
            outcome = "oom"
            torch.cuda.empty_cache()
        log_shape(M, N, K, dtype, outcome, bit_ok)
        if time.time() - last >= report_every:
            report("run")
            last = time.time()

    mm_file.close()
    if sf is not None:
        sf.close()
    print()
    report("DONE")
    recon = len(seen_static) + len(seen_pseudo) + len(seen_runtime)
    classified = recon + len(seen_needrt) + len(seen_unsup)
    scmp = st["match"] + st["mismatch"]
    print(f"\nRECONSTRUCTABLE (a bit-identical Triton GEMM exists and we found it): "
          f"{recon}/{classified} = {(100.0 * recon / classified) if classified else 0.0:.2f}% "
          f"of the random shapes cuBLAS can run")
    print(f"  static        (heuristic alone, nothing executed):    {len(seen_static)}")
    print(f"  pseudo-static (recipe read off the launched kernel):  {len(seen_pseudo)}")
    print(f"  runtime       (recipe found by byte-compare search):  {len(seen_runtime)}")
    print(f"  need-runtime  (skipped in static-only mode):          {len(seen_needrt)}")
    print(f"  DECLINED      (no reconstruction found):              {len(seen_unsup)}")
    print(f"bit-consistent on the reconstructed ones: {st['match']}/{scmp} = "
          f"{(100.0 * st['match'] / scmp) if scmp else 100.0:.3f}%")
    print(f"excluded: no_ref={st['no_ref']} (cuBLAS itself cannot run it) too_big={st['too_big']} "
          f"oom={st['oom']} err={st['error']}")
    if st["mismatch"]:
        print(f"{st['mismatch']} mismatches ({{static, runtime}} that were not bit-identical) -> {mm_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=300, help="run this many seconds, then stop")
    ap.add_argument("--report-every", type=float, default=15, help="print the running rate every N seconds")
    ap.add_argument("--R", type=int, default=5, help="random input tensors per shape")
    ap.add_argument("--dtypes", type=str, default="fp16,fp8", help="comma list: fp16,fp8,bf16")
    ap.add_argument("--seed", type=int, default=0, help="shape RNG seed (reproducible)")
    ap.add_argument("--enable-runtime-match", action="store_true",
                    help="also byte-compare the need-runtime shapes (default: static only -> they are "
                         "counted as need-runtime and skipped)")
    ap.add_argument("--shape-mode", choices=("random", "extreme"), default="random",
                    help="random: M,N,K independent uniform draws. extreme: M,N small and K large, the "
                         "skinny+deep corner where cuBLAS uses split-K and its gemv fallbacks")
    ap.add_argument("--max-dim", type=int, default=8192,
                    help="bound on M and N in random mode. Bounds memory and run time only -- there is "
                         "no rounding or alignment restriction on the shape")
    ap.add_argument("--max-mn", type=int, default=256, help="bound on M and N in extreme mode")
    ap.add_argument("--max-k", type=int, default=0,
                    help="bound on K; 0 means max-dim in random mode and 400000 in extreme mode")
    ap.add_argument("--no-fp8-round16", action="store_true",
                    help="do NOT round fp8 draws to multiples of 16. cuBLAS refuses fp8 otherwise, so "
                         "nearly every draw becomes no_ref; useful only to measure that rate")
    ap.add_argument("--max-gib", type=float, default=6.0,
                    help="skip a shape whose operands+output exceed this many GiB (resource skip)")
    ap.add_argument("--cublaslt", type=str, default="",
                    help="path to the libcublasLt to match (default: auto-detect the newest installed)")
    ap.add_argument("--shape-log", type=str, default="",
                    help="append one jsonl row per distinct shape here; also used to resume")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("no CUDA GPU; this eval needs one.")
        return
    dtypes = [d.strip() for d in args.dtypes.split(",") if d.strip()]
    max_k = args.max_k or (400000 if args.shape_mode == "extreme" else args.max_dim)
    if args.cublaslt:
        _G.set_cublaslt(args.cublaslt)
    run(args.timeout, args.report_every, args.R, dtypes, args.seed, args.enable_runtime_match, args.shape_mode,
        args.max_dim, args.max_mn, max_k, not args.no_fp8_round16, int(args.max_gib * 2**30), args.shape_log)


if __name__ == "__main__":
    main()
