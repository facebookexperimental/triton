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

from bitequiv.cublas_equiv_gemm import (
    CublasNeedRuntimeMatch,
    CublasUnsupportedShape,
    cublas_equivalent_gemm,
    cublas_equivalent_scaled_mm,
    cublas_matmul,
)

DEVICE = "cuda"
_F8 = torch.float8_e4m3fn


def _bits_eq(x, y):
    return torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8))


def _round_to(x, m):
    return max(m, (x // m) * m)


def random_shape(rng, dtypes):
    """A random, aligned matmul shape + dtype. Mixes plain (square/rect) and split-K
    (skinny + deep) regimes, plus an occasional non-aligned fp16 to exercise the decline
    path. fp16/bf16 need K%8==0 & N%8==0; fp8 needs all dims %16."""
    dtype = rng.choice(dtypes)
    regime = rng.choices(["splitk", "plain", "nonaligned"], weights=[0.6, 0.35, 0.05], k=1)[0]

    if regime == "splitk":  # skinny + deep -> cuBLAS tends to split K
        small = rng.choice([16, 24, 32, 48, 64, 80, 96, 112, 128])
        other = rng.choice([16, 32, 48, 64, 96, 128, 192, 256])
        M, N = (small, other) if rng.random() < 0.5 else (other, small)
        K = rng.choice([8192, 12288, 16384, 24576, 32768, 49152, 65536, 98304, 131072])
    elif regime == "plain":  # larger output -> cuBLAS runs plain
        M = rng.choice([256, 512, 768, 1024, 2048, 4096])
        N = rng.choice([256, 512, 768, 1024, 2048, 4096])
        K = rng.choice([256, 512, 1024, 2048, 4096, 8192])
    else:  # deliberately non-aligned fp16 (odd N or K) -> cuBLAS uses CUTLASS; API should decline
        dtype = "fp16"
        M = rng.choice([16, 64, 512])
        N = rng.choice([65, 129, 257, 513])
        K = rng.choice([4096, 4097, 8192, 8195])

    if dtype == "fp8":
        M, N, K = _round_to(M, 16), _round_to(N, 16), _round_to(K, 16)
    elif regime != "nonaligned":
        N, K = _round_to(N, 8), _round_to(K, 8)
    return M, N, K, dtype


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


def our_api(a, b, dtype, out_dtype, enable_rt):
    if dtype == "fp8":
        return cublas_equivalent_scaled_mm(a, b, 1.0, 1.0, out_dtype, enable_runtime_match=enable_rt)
    return cublas_equivalent_gemm(a, b, out_dtype, enable_runtime_match=enable_rt)


def run(timeout, report_every, R, dtypes, seed, enable_rt):
    rng = random.Random(seed)
    mm_path = f"/tmp/cublas_equiv_mismatches_{os.getpid()}.jsonl"
    mm_file = open(mm_path, "w")

    st = {"shapes": 0, "cmp": 0, "match": 0, "mismatch": 0, "no_ref": 0, "nonfinite": 0, "error": 0}
    # distinct shapes by class: static (heuristic only) / runtime (byte-compare) / need-rt (skipped in
    # static-only mode) / unsupported (no reconstruction even with a runtime match)
    seen_static, seen_runtime, seen_needrt, seen_unsup = set(), set(), set(), set()
    t0 = time.time()
    last = t0

    print(f"device: {torch.cuda.get_device_name()} | dtypes={dtypes} R={R} timeout={timeout}s "
          f"enable_runtime_match={enable_rt}")
    print(f"mismatches -> {mm_path}\n", flush=True)

    def report(tag):
        el = time.time() - t0
        scmp = st["match"] + st["mismatch"]
        rate = (100.0 * st["match"] / scmp) if scmp else 100.0
        classified = len(seen_static) + len(seen_runtime) + len(seen_needrt) + len(seen_unsup)
        sfrac = (100.0 * len(seen_static) / classified) if classified else 0.0
        print(f"[{tag} {el:6.0f}s] shapes={st['shapes']} cmp={st['cmp']} "
              f"bit-consistent={rate:6.2f}% ({st['match']}/{scmp}) mismatch={st['mismatch']} | "
              f"static={len(seen_static)} runtime={len(seen_runtime)} needRT={len(seen_needrt)} "
              f"unsup={len(seen_unsup)} static-frac={sfrac:5.1f}% | "
              f"no_ref={st['no_ref']} nonfinite={st['nonfinite']} err={st['error']}", flush=True)

    while time.time() - t0 < timeout:
        M, N, K, dtype = random_shape(rng, dtypes)
        key = (M, N, K, dtype)
        out_dtype = torch.float16 if dtype == "fp8" else (torch.bfloat16 if dtype == "bf16" else torch.float16)
        st["shapes"] += 1
        use_rt = None  # per-shape mode, decided on the first comparable input
        for r in range(R):
            s = 100003 + st["shapes"] * 131 + r  # disjoint from the API's calibration seeds
            imode = "order" if (r % 2 == 0) else "plain"
            try:
                a, b = make_inputs(M, N, K, dtype, s, imode)
            except Exception:
                st["error"] += 1
                break
            try:
                ref = cublas_matmul(a, b, out_dtype)
            except (CublasUnsupportedShape, Exception):
                st["no_ref"] += 1  # cuBLAS itself has no algo (e.g. unrunnable shape)
                break
            if not torch.isfinite(ref.float()).all():
                st["nonfinite"] += 1  # overflow -> byte-compare is meaningless, skip this input
                continue
            try:
                if use_rt is None:  # classify the shape: static, else need-runtime
                    try:
                        out = our_api(a, b, dtype, out_dtype, False)
                        use_rt = False
                        seen_static.add(key)
                    except CublasNeedRuntimeMatch:
                        if not enable_rt:
                            seen_needrt.add(key)
                            break  # static-only mode: count it, do not compare
                        out = our_api(a, b, dtype, out_dtype, True)
                        use_rt = True
                        seen_runtime.add(key)
                else:
                    out = our_api(a, b, dtype, out_dtype, use_rt)
            except CublasUnsupportedShape:
                seen_unsup.add(key)
                break
            except Exception as e:
                st["error"] += 1
                mm_file.write(json.dumps({"M": M, "N": N, "K": K, "dtype": dtype, "seed": s, "mode": imode,
                                          "error": f"{type(e).__name__}: {str(e)[:120]}"}) + "\n")
                mm_file.flush()
                break
            st["cmp"] += 1
            if _bits_eq(out, ref):
                st["match"] += 1
            else:
                st["mismatch"] += 1
                diff = (out.float() - ref.float()).abs()
                mm_file.write(json.dumps({
                    "M": M, "N": N, "K": K, "dtype": dtype, "seed": s, "mode": imode,
                    "class": "runtime" if use_rt else "static", "max_abs_diff": float(diff.max()),
                    "n_diff": int((diff > 0).sum()), "out_dtype": str(out_dtype)}) + "\n")
                mm_file.flush()
        if time.time() - last >= report_every:
            report("run")
            last = time.time()

    mm_file.close()
    print()
    report("DONE")
    classified = len(seen_static) + len(seen_runtime) + len(seen_needrt) + len(seen_unsup)
    sfrac = (100.0 * len(seen_static) / classified) if classified else 0.0
    scmp = st["match"] + st["mismatch"]
    print(f"\nstatic-reconstructable (cuBLAS heuristic alone, no GEMM run): {len(seen_static)}/{classified} = "
          f"{sfrac:.1f}% of shapes")
    print(f"  runtime-match: {len(seen_runtime)} | need-runtime(skipped): {len(seen_needrt)} | "
          f"unsupported: {len(seen_unsup)}")
    print(f"bit-consistent (static + runtime): {st['match']}/{scmp} = "
          f"{(100.0 * st['match'] / scmp) if scmp else 100.0:.3f}%")
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
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("no CUDA GPU; this eval needs one.")
        return
    dtypes = [d.strip() for d in args.dtypes.split(",") if d.strip()]
    run(args.timeout, args.report_every, args.R, dtypes, args.seed, args.enable_runtime_match)


if __name__ == "__main__":
    main()
