"""Flattened, simplified ws-GEMM sample run — the collect -> factory -> consume target.

This is a self-contained, single-file equivalent of:

    third_party/tlx/denoise.sh \
        python third_party/tlx/tutorials/testing/test_blackwell_gemm_perf.py --version ws

with two deliberate simplifications so it's a clean compile_iq target:
  (1) ONE shape         — M=N=K=8192 (constants below), instead of [2048,4096,8192].
  (2) ONE autotune cfg  — TLX_GEMM_USE_HEURISTIC=1 makes `matmul` pick a single
                          shape-dependent heuristic config and launch it directly
                          (no autotuning). This is what makes collection dump ONE
                          task instead of one-per-config (hundreds).

Like examples/user_kernel.py, this file has ZERO compile_iq references — collection
and consumption are driven purely by environment variables, with no code change:

    # collect a task for this kernel/shape/config:
    TRITON_COMPILE_IQ_COLLECT=1 COMPILE_IQ_TASK_DIR=/tmp/ciq_tasks \
        python third_party/compile_iq/examples/blackwell_gemm_ws_sample.py

    # ...run the factory on the one task it produced, then consume (the stored ACF is applied
    # in-memory at load -- no TRITON_ALWAYS_COMPILE needed, even on a warm compile cache):
    TRITON_COMPILE_IQ_APPLY=1 COMPILE_IQ_STORE=/tmp/ciq_store \
    TRITON_COMPILE_IQ_DEBUG=1 \
        python third_party/compile_iq/examples/blackwell_gemm_ws_sample.py

denoise.sh notes: its clock/power lock (`sudo nvidia-smi -lgc/-pm/--power-limit`) and
`numactl -m 0 -c 0` pinning need root / a process wrapper, so they aren't done here.
For locked-clock numbers, still run this script *under* denoise.sh. What this file does
replicate is denoise-grade measurement: a fixed device and do_bench(warmup=2000, rep=2000).
"""

import os

# Must be set before CUDA init / before `matmul` is called.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")  # mirrors denoise.sh default
os.environ.setdefault("TLX_GEMM_USE_HEURISTIC", "1")  # (2) single heuristic config, no autotune

import torch

import triton

from triton.language.extra.tlx.tutorials.blackwell_gemm_ws import matmul

try:
    from triton._internal_testing import is_blackwell
except Exception:  # pragma: no cover - fallback if helper moves

    def is_blackwell():
        return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


DEVICE = triton.runtime.driver.active.get_active_torch_device()

# (1) ONE shape. 8192 overflows B200 smem with the current heuristic config; the compile_iq
# e2e sets WS_GEMM_SIZE=2048 (a shape whose heuristic config fits) -- override as needed.
M = N = K = int(os.environ.get("WS_GEMM_SIZE", "8192"))
DTYPE = torch.float16  # matches `--version ws` default (no --dtype)
REL_TOL = 1e-2


def _tflops(ms):
    return 2 * M * N * K * 1e-12 / (ms * 1e-3)


def main():
    if not is_blackwell():
        print("Skipping: no Blackwell GPU found.")
        return

    torch.manual_seed(0)
    a = torch.randn((M, K), device=DEVICE, dtype=DTYPE)
    b = torch.randn((K, N), device=DEVICE, dtype=DTYPE)

    # Correctness vs torch (tolerant — fp16 accumulation).
    c = matmul(a, b)
    torch.cuda.synchronize()
    ref = torch.matmul(a, b)
    rel = (c.float() - ref.float()).abs().max() / ref.float().abs().max()
    assert torch.isfinite(rel) and rel.item() <= REL_TOL, f"correctness failed: rel-err {rel.item():.3e}"

    # Denoise-grade measurement of the WS kernel (no cuBLAS baseline -- not needed for the e2e).
    ws_ms = triton.testing.do_bench(lambda: matmul(a, b), warmup=2000, rep=2000, return_mode="median")

    print(f"shape M=N=K={M}  dtype={DTYPE}  heuristic-config (single)")
    print(f"  ws     : {ws_ms:.4f} ms  =  {_tflops(ws_ms):.1f} TFLOPS  (rel-err {rel.item():.2e})")


if __name__ == "__main__":
    main()
