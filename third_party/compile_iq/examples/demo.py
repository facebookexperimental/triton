"""compile_iq exploration demo: one run per kernel produces, under an output dir:
  - ptx_<target>/    every PTX the kernel compiles (autotune variants included)
  - acfs_<target>/   every candidate ACF the search evaluated (best-first)
  - results_<target>.csv, best_<target>.acf

This drives the CompileIQ search directly (not the productized factory) and dumps everything, so it
is handy for eyeballing the search space + the SASS an ACF actually changes. For the normal
collect -> factory -> consume flow, use `run_e2e.sh` / `python -m triton.compile_iq.factory`.

NOTE: each ACF is applied via the `ptx_options` launch kwarg (matching the factory/consume path), but
this demo has NO per-candidate isolation -- an ACF that wedges the GPU will HANG it (the factory
isolates each candidate in a killable subprocess). Run it where ACFs are runnable (ptxas >= 13.3
assembler AND a >= 13.3 GPU driver), or use the factory for robust tuning.

Targets:
  --target naive   (default)  in-file naive Triton matmul (no autotune, 1 PTX)
  --target ws                 fbtriton warp-specialized Blackwell GEMM (autotuned)

Requires (no hardcoded paths):
  TRITON_PTXAS_BLACKWELL_PATH   ptxas >= 13.3 (or a 13.3 ptxas on PATH)
  COMPILE_IQ_SEARCH_SPACE_BIN   the ptxas13.3 search-space .bin
Optional: ACF_DEMO_OUT (output dir; default a fresh temp dir).

Usage:
  TRITON_PTXAS_BLACKWELL_PATH=/path/to/ptxas COMPILE_IQ_SEARCH_SPACE_BIN=/path/to/ss.bin \
    python third_party/compile_iq/examples/demo.py --target naive
"""

import os
import sys
import tempfile

# ---- set ptxas before importing triton (cache dir / ptxas must be pre-set) ----
HERE = os.path.dirname(os.path.abspath(__file__))


def _argv(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


TARGET = os.environ.get("PTX_ACF_TARGET") or _argv("--target", "naive")
DUMP_PTX_ONLY = "--dump-ptx-only" in sys.argv

PTXAS = os.environ.get("TRITON_PTXAS_BLACKWELL_PATH", "")
if PTXAS:
    os.environ["TRITON_PTXAS_PATH"] = PTXAS
os.environ.setdefault("CIQ_PROCESS_MODE", "spawn")

OUT_DIR = os.environ.get("ACF_DEMO_OUT") or tempfile.mkdtemp(prefix="acf_demo_")

if DUMP_PTX_ONLY:
    os.environ["TRITON_CACHE_DIR"] = os.path.join(OUT_DIR, f"ptx_{TARGET}")
    os.environ["TRITON_ALWAYS_COMPILE"] = "0"
else:
    os.environ.setdefault("TRITON_ALWAYS_COMPILE", "1")  # recompile per candidate so each ACF applies

import glob
import re
import shutil
import subprocess

import torch
import triton
import triton.language as tl
from compileiq.ciq import Search
from compileiq.search_spaces.compilers import LocalSearchSpaceBin
from compileiq.types import INVALID_SCORE, SearchConfiguration
from compileiq.utils.gpu import gpu_benchmark_mode
from compileiq.utils.helpers import save_compiler_config

DEVICE = triton.runtime.driver.active.get_active_torch_device()
SEARCH_SPACE_BIN = os.environ.get("COMPILE_IQ_SEARCH_SPACE_BIN")

M = N = K = 2048
DTYPE = torch.float16
REL_TOL = 1e-2


def _make_inputs():
    torch.manual_seed(0)
    a = torch.randn((M, K), device=DEVICE, dtype=DTYPE)
    b = torch.randn((K, N), device=DEVICE, dtype=DTYPE)
    return a, b


@triton.jit
def _naive_mm_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
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


def naive_triton_matmul(a, b, ptx_options=None):
    Mx, Kx = a.shape
    Kx, Nx = b.shape
    c = torch.empty((Mx, Nx), device=a.device, dtype=a.dtype)
    BM = BN = 64
    BK = 32
    grid = (triton.cdiv(Mx, BM), triton.cdiv(Nx, BN))
    kw = {"ptx_options": ptx_options} if ptx_options else {}
    _naive_mm_kernel[grid](a, b, c, Mx, Nx, Kx, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0),
                           c.stride(1), BM, BN, BK, **kw)
    return c


def target_matmul(a, b, ptx_options=None):
    """Kernel under test, chosen by --target. The ACF (when given) is applied via the ptx_options
    launch kwarg -> ptxas, matching the factory/consume path."""
    if (os.environ.get("PTX_ACF_TARGET") or TARGET) == "naive":
        return naive_triton_matmul(a, b, ptx_options)
    from triton.language.extra.tlx.tutorials.blackwell_gemm_ws import matmul as ws_matmul  # lazy: WS only
    return ws_matmul(a, b)  # NOTE: the WS wrapper does not forward ptx_options (WS is a later diff)


def reference(a, b):
    if os.environ.get("ACF_REF", "triton") == "torch":
        return torch.matmul(a, b)
    return naive_triton_matmul(a, b)


def objective(acf: str) -> float:
    a, b = _make_inputs()
    ref = reference(a, b)  # ACF-free reference
    with tempfile.NamedTemporaryFile(suffix=".acf", delete=True) as f:
        save_compiler_config(f.name, acf)
        acf_opt = f"--apply-controls={f.name}"
        try:
            out = target_matmul(a, b, ptx_options=acf_opt)
            rel = (out.float() - ref.float()).abs().max() / ref.float().abs().max()
            if not torch.isfinite(rel) or rel.item() > REL_TOL:
                return INVALID_SCORE
            return triton.testing.do_bench(lambda: target_matmul(a, b, ptx_options=acf_opt), warmup=25, rep=100,
                                           return_mode="mean")
        except Exception as e:
            print(f"[objective] candidate failed: {type(e).__name__}: {e}")
            return INVALID_SCORE


def baseline_ms() -> float:
    a, b = _make_inputs()
    target_matmul(a, b)
    return triton.testing.do_bench(lambda: target_matmul(a, b), warmup=25, rep=100, return_mode="mean")


def _ptxas_version() -> str:
    ptxas = PTXAS or shutil.which("ptxas")
    out = subprocess.run([ptxas, "--version"], capture_output=True, text=True).stdout
    return re.search(r"release (\d+\.\d+)", out).group(1)


def _dump_ptx_only():
    """Compile the target once (clean cache) and report how many PTX it produced."""
    cache = os.environ["TRITON_CACHE_DIR"]
    a, b = _make_inputs()
    target_matmul(a, b)
    torch.cuda.synchronize()
    ptxs = glob.glob(f"{cache}/**/*.ptx", recursive=True)
    print(f"[ptx-dump target={TARGET}] {len(ptxs)} .ptx in {cache}")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["ws", "naive"], default="naive")
    ap.add_argument("--ref", choices=["triton", "torch"], default="triton")
    ap.add_argument("--dump-ptx-only", action="store_true")
    args = ap.parse_args()

    if args.dump_ptx_only:
        _dump_ptx_only()
        return

    assert SEARCH_SPACE_BIN and os.path.exists(SEARCH_SPACE_BIN), \
        "set COMPILE_IQ_SEARCH_SPACE_BIN to the ptxas13.3 search-space .bin"
    ver = _ptxas_version()
    assert float(ver) >= 13.3, f"need ptxas >= 13.3, got {ver} (set TRITON_PTXAS_BLACKWELL_PATH)"

    os.environ["PTX_ACF_TARGET"] = args.target
    os.environ["ACF_REF"] = args.ref
    ptx_dir = os.path.join(OUT_DIR, f"ptx_{args.target}")
    acf_dir = os.path.join(OUT_DIR, f"acfs_{args.target}")
    res_csv = os.path.join(OUT_DIR, f"results_{args.target}.csv")
    best_acf = os.path.join(OUT_DIR, f"best_{args.target}.acf")
    for p in (ptx_dir, acf_dir):
        shutil.rmtree(p, ignore_errors=True)
    print(f"=== target={args.target} ref={args.ref} ptxas={ver} out={OUT_DIR} ===")

    # 1. PTX dump (separate clean process: ALWAYS_COMPILE off + dedicated cache dir)
    env = dict(os.environ)
    env["TRITON_CACHE_DIR"] = ptx_dir
    env["TRITON_ALWAYS_COMPILE"] = "0"
    env["ACF_DEMO_OUT"] = OUT_DIR
    subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--target", args.target, "--dump-ptx-only"], env=env, check=True)

    # 2. baseline + ACF search
    base = baseline_ms()
    print(f"baseline (no ACF): {base:.4f} ms")
    config = SearchConfiguration(problem_type="min", generations=2, pool_size=8)
    tuner = Search(objective_function=objective, search_space=LocalSearchSpaceBin(SEARCH_SPACE_BIN),
                   search_config=config)
    with gpu_benchmark_mode(clock_mhz=1965, raise_on_failure=False):
        results = tuner.start()

    # 3. dump results + every candidate ACF
    df = results.get_results()
    sort_col = results.score_columns[0]
    view = df.sort_values(sort_col)
    view.to_csv(res_csv, index=False)
    best = results.get_best_result()
    save_compiler_config(best_acf, best["params"])
    os.makedirs(acf_dir, exist_ok=True)
    for rank, (idx, row) in enumerate(view.iterrows()):
        s = row[sort_col]
        tag = f"{s:.6f}" if isinstance(s, (int, float)) else "INVALID"
        save_compiler_config(os.path.join(acf_dir, f"{rank:02d}_idx{idx}_{tag}.acf"), row["params"])

    n_ptx = len(glob.glob(f"{ptx_dir}/**/*.ptx", recursive=True))
    print("\n===== SUMMARY =====")
    print(f"target          : {args.target}")
    print(f"PTX files       : {n_ptx}   (in {ptx_dir})")
    print(f"candidate ACFs  : {len(view)}   (in {acf_dir})")
    print(f"best ACF        : {best_acf}  (runtime {best.get('score_1')} ms, baseline {base:.4f})")


if __name__ == "__main__":
    main()
