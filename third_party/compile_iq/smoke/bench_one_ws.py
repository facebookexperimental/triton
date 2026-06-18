"""WS per-candidate evaluator (base Python 3.13) for the EVO bridge -- the warp-specialized
counterpart of bench_one.py.

Naive kernels are replayed from reconstructed args; WS/TMA kernels can't be (the reconstructed
launch produces different PTX), so we use the REAL launch: import the kernel's `matmul` wrapper
and run it, applying the candidate ACF via PTXAS_OPTIONS (which Triton's knobs read once at import,
so it must be set BEFORE importing triton). Correctness is checked against torch (cuBLAS), which is
the right oracle for a GEMM. This makes the validated PTX identical to collect/consume's real launch
(parity for free).

Usage: python bench_one_ws.py <acf_hex_file | NONE>   ; shape via WS_GEMM_SIZE env (default 2048)
Prints exactly one line: "MS <float>" on success, or "INVALID" (diverged/crashed/wedged).
"""
import os
import sys
import tempfile

REL_TOL = 1e-2

acf_arg = sys.argv[1]
if acf_arg != "NONE":
    with open(acf_arg) as f:
        acf_hex = f.read().strip()
    t = tempfile.NamedTemporaryFile(suffix=".acf", delete=False)
    t.write(bytes.fromhex(acf_hex))
    t.close()
    os.environ["PTXAS_OPTIONS"] = f"--apply-controls={t.name}"  # MUST precede triton import
os.environ["TRITON_ALWAYS_COMPILE"] = "1"  # force a fresh compile so the ACF is applied

import torch  # noqa: E402
import triton  # noqa: E402
from triton.language.extra.tlx.tutorials.blackwell_gemm_ws import matmul  # noqa: E402


def main():
    s = int(os.environ.get("WS_GEMM_SIZE", "2048"))
    dev = triton.runtime.driver.active.get_active_torch_device()
    torch.manual_seed(0)
    a = torch.randn((s, s), device=dev, dtype=torch.float16)
    b = torch.randn((s, s), device=dev, dtype=torch.float16)
    try:
        c = matmul(a, b)
        torch.cuda.synchronize()
        ref = torch.matmul(a, b)
        denom = max(ref.float().abs().max().item(), 1e-9)
        rel = (c.float() - ref.float()).abs().max().item() / denom
        if not (rel == rel and rel <= REL_TOL):  # NaN-safe
            print("INVALID")
            return
        ms = triton.testing.do_bench(lambda: matmul(a, b), warmup=100, rep=200, return_mode="mean")
        print(f"MS {ms}")
    except Exception:
        import traceback
        traceback.print_exc()
        print("INVALID")


if __name__ == "__main__":
    main()
