"""Correctness + pool-vs-sum A/B for case8 multi-phase GEMM (general path).

pool = full requested ring depths, phases share one pooled SMEM backing.
sum  = joint-sum budget accounting: depths trimmed / IIs raised to fit
       without sharing. Same join structure both sides — the delta isolates
       depths/II, i.e. what buffer reuse buys.
"""

from __future__ import annotations

import importlib.util
import sys

import torch
import triton

SHAPES = [(2048, 2048, 2048), (4096, 4096, 4096)]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def alloc_fn(size, alignment, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


def main() -> int:
    triton.set_allocator(alloc_fn)
    torch.manual_seed(0)
    pool = _load("generated.py", "gen_pool")
    summ = _load("generated_sum.py", "gen_sum")
    failed = 0

    for M, N, K in SHAPES:
        print(f"== M={M} N={N} K={K}")
        a1 = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b1 = torch.randn(K, N, device="cuda", dtype=torch.float16)
        a2 = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b2 = torch.randn(K, N, device="cuda", dtype=torch.float16)
        a3 = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        b3 = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
        c1 = torch.empty(M, N, device="cuda", dtype=torch.float16)
        c2 = torch.empty(M, N, device="cuda", dtype=torch.float16)
        c3 = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        refs = [torch.matmul(a.float(), b.float())
                for a, b in ((a1, b1), (a2, b2), (a3, b3))]
        grid = (M // 128, N // 128)
        flops = 3 * 2 * M * N * K
        ms_by = {}
        for name, mod in (("pool", pool), ("sum", summ)):
            def run(m=mod):
                m.triple_gemm_nows[grid](
                    a1, b1, c1, a2, b2, c2, a3, b3, c3, M, N, K,
                    num_warps=4, num_ctas=1, num_stages=2,
                )
            run()
            torch.cuda.synchronize()
            for out, ref, lbl, tol in (
                (c1, refs[0], "C1", 5e-3), (c2, refs[1], "C2", 5e-3),
                (c3, refs[2], "C3", 2e-2),
            ):
                nan = int(torch.isnan(out.float()).sum().item())
                rel = (out.float() - ref).abs().max().item() / ref.abs().max().item()
                ok = nan == 0 and rel < tol
                print(f"  [{name} {lbl}] nan={nan} rel={rel:.3e} "
                      f"{'PASS' if ok else 'FAIL'}")
                failed += 0 if ok else 1
            ms = triton.testing.do_bench(run, warmup=50, rep=200, quantiles=[0.5])
            ms = float(ms[0] if isinstance(ms, (list, tuple)) else ms)
            ms_by[name] = ms
            print(f"  {name}: {ms:.4f} ms  {flops / (ms * 1e-3) / 1e12:.1f} TF")
        print(f"  pool speedup vs sum: {ms_by['sum'] / ms_by['pool']:.3f}x")

    for name, mod in (("pool", pool), ("sum", summ)):
        k = mod.triple_gemm_nows
        smem = None
        try:
            dk = list(k.device_caches.values())[0][0]
            smem = list(dk.values())[0].metadata.shared
        except Exception:
            pass
        print(f"{name} kernel SMEM bytes: {smem}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
