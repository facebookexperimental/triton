"""PTX-direct per-candidate worker (the isolated objective body).

Given a FIXED kernel.ptx + a launch spec + (optionally) an ACF, this:
  1. ptxas-assembles the PTX (baseline) -> cubin -> driver-launch -> snapshot outputs (reference),
  2. ptxas-assembles the PTX WITH --apply-controls=<acf> -> cubin -> driver-launch -> snapshot,
  3. checks self-consistency (the ACF must not change results vs no-ACF), and
  4. benchmarks the ACF launch.

No Triton frontend recompile -- only ptxas + the CUDA driver. Run in its own spawn subprocess so
a candidate ACF that wedges/crashes the GPU can only kill this child (the parent kills it at a
timeout and scores the candidate INVALID).

Usage: python ptx_bench_one.py <kernel.ptx> <spec.json> <acf_path|NONE> [warmup] [rep]
Prints exactly one line: "MS <float>" on success, else "INVALID".
"""
import json
import os
import sys

import torch
import triton

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the compile_iq pkg root
from compile_iq import ptx_launch as L

REL_TOL = 1e-2
PTXAS = None  # resolved in main()


def _run_and_snapshot(ptx, spec, acf_path):
    """ptxas(ptx[,acf]) -> cubin -> launch with fresh (deterministic) tensors -> return
    (kernel, tensors, kernel_args, tensormaps). `tensormaps` (the PyCUtensorMap objects backing any TMA
    descriptors) MUST be kept alive by the caller while it benchmarks via k.launch(...)."""
    cubin = L.ptxas_compile(ptx, PTXAS, arch=spec["arch"], acf_path=acf_path)
    k = L.load_cubin(cubin, spec["entry"], spec["shared"])
    tensors = L.alloc_tensors(spec, seed=0)
    tms, addrs = L.build_tensormaps(spec, tensors)  # [] for non-TMA kernels
    ka = L.kernel_args_from_spec(spec, tensors, addrs)
    k.launch(spec["grid"], spec["block"], ka)
    torch.cuda.synchronize()
    return k, tensors, ka, tms


def main():
    global PTXAS
    ptx_file, spec_file, acf = sys.argv[1], sys.argv[2], sys.argv[3]
    warmup = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    rep = int(sys.argv[5]) if len(sys.argv) > 5 else 200
    acf_path = None if acf == "NONE" else acf

    with open(ptx_file) as f:
        ptx = f.read()
    with open(spec_file) as f:
        spec = json.load(f)
    PTXAS = spec["ptxas"]

    try:
        # 1) baseline reference (no ACF).
        _, ref_tensors, _, _tms0 = _run_and_snapshot(ptx, spec, None)
        ref = [t.detach().clone() for t in ref_tensors]
        # 2) candidate run (with ACF, or baseline again when acf is NONE). Keep `tms` alive: the TMA
        #    descriptors it holds are referenced by `ka` for the duration of the benchmark below.
        k, tensors, ka, tms = _run_and_snapshot(ptx, spec, acf_path)
        # 3) self-consistency: the ACF must not change any buffer beyond tolerance.
        for r, cur in zip(ref, tensors):
            denom = max(r.float().abs().max().item(), 1e-9)
            d = (cur.float() - r.float()).abs().max().item() / denom
            if not (d == d and d <= REL_TOL):  # NaN-safe
                print("INVALID")
                return
        # 4) benchmark the candidate launch.
        # TODO(compile_iq perf, item 1): this runs at UNLOCKED clocks with a short do_bench, so the
        # per-candidate ms is noisy (~+-2-4%). Lock the GPU to a sustainable clock for trustworthy
        # search ranking (see ws_ab.py on branch daohang/compile_iq_perf_harness).
        ms = triton.testing.do_bench(lambda: k.launch(spec["grid"], spec["block"], ka), warmup=warmup, rep=rep,
                                     return_mode="mean")
        del tms
        print(f"MS {ms}")
    except Exception as e:
        sys.stderr.write(f"[ptx_bench_one] {type(e).__name__}: {e}\n")
        print("INVALID")


if __name__ == "__main__":
    main()
