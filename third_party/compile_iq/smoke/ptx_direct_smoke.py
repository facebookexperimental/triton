"""PTX-direct prototype (proof for design (b)): tune a ptxas ACF by assembling a FIXED kernel.ptx
and launching the cubin via the CUDA driver API -- NO Triton frontend recompile from source.

Stages:
  1. compile the naive matmul once via Triton -> emit kernel.ptx + spec.json (the self-contained
     launch description a source-free run needs; previews the (a) collector schema).
  2. BASELINE: prove ptxas(kernel.ptx) -> cubin -> driver-launch matches torch + Triton, and times.
  3. SEARCH (optional, needs CompileIQ engine + a search-space bin): drive the CIQ engine with the
     PTX-direct objective, each candidate isolated in a spawn subprocess (ptx_bench_one.py); store
     the best validated ACF.
  4. APPLY: prove ptxas(kernel.ptx, --apply-controls=<acf>) -> cubin -> driver-launch runs correctly.

Usage:
  python ptx_direct_smoke.py [--search-space-bin PATH] [--generations N] [--pool-size N]
                             [--shape M N K] [--work-dir DIR] [--ptxas PATH]
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the compile_iq pkg root
from compile_iq import ptx_launch as L
from compile_iq import store as ciq_store

HERE = pathlib.Path(__file__).resolve().parent
BENCH_ONE = str(HERE / "ptx_bench_one.py")
DEFAULT_PTXAS = os.environ.get("TRITON_PTXAS_BLACKWELL_PATH", "ptxas")  # env/PATH; override with --ptxas


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """Plain tiled matmul (fp32 accumulate). No warp specialization / async pipeline."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = acc.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _dtype_str(dt):
    return str(dt).replace("torch.", "")


def compile_and_spec(M, N, K, ptxas, dtype=torch.float16, device="cuda"):
    """Compile the matmul once via Triton; return (ptx, spec, a, b, c) where spec is the source-free
    launch description. Applies equal_to_1 specialization to derive the final kernel-param order and
    asserts it matches the PTX `.entry` param count."""
    BLOCK_M = BLOCK_N = 64
    BLOCK_K = 32
    torch.manual_seed(0)
    a = torch.randn((M, K), device=device, dtype=dtype)
    b = torch.randn((K, N), device=device, dtype=dtype)
    c = torch.empty((M, N), device=device, dtype=dtype)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    h = matmul_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0),
                            c.stride(1), BLOCK_M, BLOCK_N, BLOCK_K)
    torch.cuda.synchronize()
    ptx = h.asm["ptx"]
    # Same source-free spec builder the collector uses: classify the call args in bound order, then
    # build_spec applies equal_to_1 / constexpr specialization and the trailing null scratch params.
    ds = _dtype_str(dtype)
    ordered = [("tensor", {"shape": [M, K], "dtype": ds, "strides": list(a.stride())}),
               ("tensor", {"shape": [K, N], "dtype": ds, "strides": list(b.stride())}),
               ("tensor", {"shape": [M, N], "dtype": ds, "strides": list(c.stride())}), ("scalar", M), ("scalar", N),
               ("scalar", K), ("scalar", a.stride(0)), ("scalar", a.stride(1)), ("scalar", b.stride(0)),
               ("scalar", b.stride(1)), ("scalar", c.stride(0)), ("scalar", c.stride(1)), ("constexpr", ),
               ("constexpr", ), ("constexpr", ),  # BLOCK_M, BLOCK_N, BLOCK_K
               ]
    spec = L.build_spec(ptx, h.metadata, tuple(grid), ordered, ptxas)
    spec["ptx_sha256"] = ciq_store.ptx_sha256(ptx)
    return ptx, spec, a, b, c


def isolated_bench(ptx_file, spec_file, acf, timeout, warmup, rep, ptxas_for_log=""):
    """Run one candidate in a spawn subprocess via ptx_bench_one.py. Returns ms (float) or None."""
    cmd = [sys.executable, BENCH_ONE, ptx_file, spec_file, acf if acf else "NONE", str(warmup), str(rep)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    for line in out.stdout.splitlines():
        if line.startswith("MS "):
            try:
                return float(line.split()[1])
            except Exception:
                return None
    if out.stderr.strip():
        sys.stderr.write(out.stderr)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-space-bin", default=os.environ.get("COMPILE_IQ_SEARCH_SPACE_BIN"))
    ap.add_argument("--generations", type=int, default=1)
    ap.add_argument("--pool-size", type=int, default=6)
    ap.add_argument("--shape", type=int, nargs=3, default=[256, 256, 256], metavar=("M", "N", "K"))
    ap.add_argument("--work-dir", default=os.path.expanduser("~/.compile_iq/ptx_smoke"))
    ap.add_argument("--ptxas", default=os.environ.get("TRITON_PTXAS_BLACKWELL_PATH", DEFAULT_PTXAS))
    ap.add_argument("--per-cand-timeout", type=int, default=60)
    ap.add_argument(
        "--acf-hex", default=None, help="path to a hex-encoded ACF (e.g. minted by ptx_evo_search.py); apply it via "
        "PTX-direct driver launch and report correctness + speedup (skips the CIQ search)")
    a = ap.parse_args()

    M, N, K = a.shape
    os.makedirs(a.work_dir, exist_ok=True)
    ptx_file = os.path.join(a.work_dir, "kernel.ptx")
    spec_file = os.path.join(a.work_dir, "spec.json")

    print(f"[ptx-smoke] ptxas: {a.ptxas}")
    if not os.path.exists(a.ptxas):
        raise FileNotFoundError(f"ptxas not found: {a.ptxas}")

    # ---- Stage 1: compile once -> kernel.ptx + spec.json -------------------------------------
    ptx, spec, ta, tb, tc = compile_and_spec(M, N, K, a.ptxas)
    with open(ptx_file, "w") as f:
        f.write(ptx)
    with open(spec_file, "w") as f:
        json.dump(spec, f, indent=2)
    print(f"[ptx-smoke] stage1: emitted {ptx_file} + spec.json  entry={spec['entry']} arch={spec['arch']} "
          f"shared={spec['shared']} block={spec['block']} grid={spec['grid']} ptx_sha={spec['ptx_sha256'][:16]}")

    # ---- Stage 2: baseline driver launch (in-process) matches torch + Triton -----------------
    ref_triton = torch.empty((M, N), device="cuda", dtype=torch.float16)
    matmul_kernel[(triton.cdiv(M, 64), triton.cdiv(N, 64))](ta, tb, ref_triton, M, N, K, ta.stride(0), ta.stride(1),
                                                            tb.stride(0), tb.stride(1), ref_triton.stride(0),
                                                            ref_triton.stride(1), 64, 64, 32)
    torch.cuda.synchronize()
    cubin = L.ptxas_compile(ptx, a.ptxas, arch=spec["arch"])
    k = L.load_cubin(cubin, spec["entry"], spec["shared"])
    out = torch.empty((M, N), device="cuda", dtype=torch.float16)
    ka = L.kernel_args_from_spec(spec, [ta, tb, out])  # tensors order matches spec["tensors"] = [a,b,c]
    k.launch(spec["grid"], spec["block"], ka)
    torch.cuda.synchronize()
    ref = torch.matmul(ta.float(), tb.float())
    e_torch = (out.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-9)
    e_triton = (out.float() - ref_triton.float()).abs().max().item() / max(ref_triton.float().abs().max().item(), 1e-9)
    print(f"[ptx-smoke] stage2 baseline: rel_err vs torch={e_torch:.2e} vs triton={e_triton:.2e}")
    assert e_torch < 1e-2 and e_triton == 0.0, "baseline driver launch INCORRECT"

    base_ms = isolated_bench(ptx_file, spec_file, None, max(a.per_cand_timeout, 120), 100, 200)
    print(f"[ptx-smoke] stage2 baseline (isolated): {base_ms} ms")
    assert base_ms is not None, "baseline isolated bench failed"

    # ---- Stage 4 (direct): apply a provided (e.g. EVO-minted) ACF and report -----------------
    if a.acf_hex:
        import tempfile
        with open(a.acf_hex) as f:
            acf_bytes = bytes.fromhex(f.read().strip())
        acf_path = os.path.join(a.work_dir, "applied.acf")
        with open(acf_path, "wb") as f:
            f.write(acf_bytes)
        ms = isolated_bench(ptx_file, spec_file, acf_path, max(a.per_cand_timeout, 120), 100, 200)
        if ms is None:
            print(f"[ptx-smoke] stage4: APPLY FAILED -- the ACF ({len(acf_bytes)}B) did not run under "
                  "PTX-direct driver launch (diverged/wedged, or ptxas/driver version mismatch).")
            sys.exit(1)
        print(f"[ptx-smoke] stage4: APPLY OK -- ptxas(kernel.ptx, --apply-controls) driver-launched, {ms} ms "
              f"({(base_ms / ms - 1) * 100:+.2f}% vs baseline {base_ms})  acf={len(acf_bytes)}B")
        return

    # ---- Stage 3: drive the CIQ search with the PTX-direct objective --------------------------
    if not a.search_space_bin or not os.path.exists(a.search_space_bin):
        print("[ptx-smoke] stage3: SKIPPED (no --search-space-bin / COMPILE_IQ_SEARCH_SPACE_BIN). "
              "Baseline PTX-direct launch is proven; supply a search-space bin to mint+apply a real ACF.")
        return
    best_acf_path = run_search(ptx_file, spec_file, spec, a, base_ms)
    if best_acf_path is None:
        print("[ptx-smoke] stage3: no ACF stored.")
        return

    # ---- Stage 4: apply the minted ACF via driver launch -------------------------------------
    ms = isolated_bench(ptx_file, spec_file, best_acf_path, max(a.per_cand_timeout, 120), 100, 200)
    if ms is None:
        print(f"[ptx-smoke] stage4: APPLY FAILED -- the stored ACF did not run under driver launch "
              f"(driver CUDA may be < 13.3; ptxas accepts --apply-controls but the cubin won't load).")
    else:
        print(f"[ptx-smoke] stage4: APPLY OK -- ptxas(kernel.ptx, --apply-controls) launched, {ms} ms "
              f"({(base_ms / ms - 1) * 100:+.2f}% vs baseline {base_ms})")


def run_search(ptx_file, spec_file, spec, a, base_ms):
    try:
        from compileiq.ciq import Search
        from compileiq.search_spaces.compilers import LocalSearchSpaceBin
        from compileiq.types import INVALID_SCORE, SearchConfiguration
        from compileiq.utils.gpu import gpu_benchmark_mode
        from compileiq.utils.helpers import save_compiler_config
    except ImportError as e:
        print(f"[ptx-smoke] stage3: CompileIQ engine not importable ({e}) -- skipping search.")
        return None

    import tempfile

    def objective(acf_hex):
        with tempfile.NamedTemporaryFile(suffix=".acf", delete=True) as f:
            save_compiler_config(f.name, acf_hex)
            ms = isolated_bench(ptx_file, spec_file, f.name, a.per_cand_timeout, 50, 100)
            return ms if ms is not None else INVALID_SCORE

    print(f"[ptx-smoke] stage3: CIQ search gen={a.generations} pool={a.pool_size} space={a.search_space_bin}")
    cfg = SearchConfiguration(problem_type="min", generations=a.generations, pool_size=a.pool_size)
    tuner = Search(objective_function=objective, search_space=LocalSearchSpaceBin(a.search_space_bin),
                   search_config=cfg, exit_on_failure=False)
    with gpu_benchmark_mode(clock_mhz=1965, raise_on_failure=False):
        results = tuner.start()
    df = results.get_results()
    n = len(df) if df is not None else 0
    try:
        best = results.get_best_result()
        best_ms = best.get("score_1", best.get("score")) if best else None
    except Exception:
        best, best_ms = None, None
    if not isinstance(best_ms, (int, float)):
        print(f"[ptx-smoke] stage3: no valid candidate among {n} (all wedged/diverged). If ALL fail, the "
              "driver likely can't RUN --apply-controls cubins (needs CUDA >= 13.3 driver).")
        return None
    print(f"[ptx-smoke] stage3: radius={n} best={best_ms} ms ({(base_ms / best_ms - 1) * 100:+.2f}% vs baseline)")
    import compileiq.utils.helpers as H
    acf_bytes = bytes.fromhex(best["params"])
    meta = {
        "ptx_sha256": spec["ptx_sha256"], "arch": spec["arch"], "baseline_ms": base_ms, "best_ms": best_ms, "source":
        "ptx_direct_smoke"
    }
    p = ciq_store.write_acf(spec["ptx_sha256"], spec["arch"], acf_bytes, meta)
    print(f"[ptx-smoke] stage3: stored ACF -> {p}")
    return p


if __name__ == "__main__":
    main()
