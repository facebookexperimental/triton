import argparse
import os
import statistics
import subprocess
import threading
import time

import torch

import triton

from triton.language.extra.tlx.tutorials.blackwell_gemm_2cta import (
    matmul as _tlx_matmul_2cta, )
from triton.language.extra.tlx.tutorials.blackwell_gemm_clc import (
    matmul as _tlx_matmul_clc, )
from triton.language.extra.tlx.tutorials.blackwell_gemm_pipelined import (
    matmul as _tlx_matmul_pipelined, )
from triton.language.extra.tlx.tutorials.blackwell_gemm_ws import (
    matmul as _tlx_matmul_ws, )

from triton._internal_testing import is_blackwell

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# Registry of available matmul implementations
MATMUL_METHODS = {
    "ws": _tlx_matmul_ws,
    "clc": _tlx_matmul_clc,
    "pipelined": _tlx_matmul_pipelined,
    "2cta": _tlx_matmul_2cta,
}

ref_lib = "cuBLAS"

# --- Benchmark methodology knobs (CUTLASS GEMM perf guidelines) ------------
# Run under third_party/tlx/denoise.sh, which locks the clock and caps power
# (a "fixed-frequency" architectural test). For that test type CUTLASS says to
# fill inputs with ZEROS (low entropy -> low power -> the locked clock is not
# power-throttled). Use --fill uniform for a "fixed-power" test (no clock lock).
WARMUP_S = 3.0     # >= 3 s of continuous execution so clocks/power settle
REP_S = 2.0        # timed window; yields ~1000-4000 iters depending on kernel
COOLDOWN_S = 1.0   # idle between (provider, shape) cells

"""
This script is used for benchmarking the performance of TLX tutorial kernels.
It's recommended to run with `third_party/tlx/denoise.sh third_party/tlx/tutorials/testing/test_blackwell_gemm_perf.py`

Facebook: If you are developing in fbsource, use tritonbench instead to collect perf numbers.
"""


def _phys_gpu_index():
    """Physical GPU id for nvidia-smi (denoise.sh exports CUDA_VISIBLE_DEVICES)."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd:
        first = cvd.split(",")[0].strip()
        if first.isdigit():
            return first
    return "0"


class GpuMonitor:
    """Sample SM clock (MHz), power (W), temperature (C) and active throttle
    reasons via nvidia-smi on a background thread, so we can verify the GPU was
    not silently throttled during timing (CUTLASS: always co-measure clock and
    power). pynvml is unavailable here; nvidia-smi queries need no sudo and the
    background thread issues no work on the compute stream. NOTE: L2 hit-rate is
    not exposed by NVML/nvidia-smi (needs ncu), so cold-cache behavior is instead
    documented via the rotation-buffer footprint vs 2x L2 in the per-cell log."""

    QUERY = ("clocks.sm,power.draw,temperature.gpu,"
             "clocks_throttle_reasons.active")

    def __init__(self, interval=0.1):
        self.interval = interval
        self.gpu = _phys_gpu_index()
        self._stop = threading.Event()
        self._thread = None
        self.sm_clocks = []
        self.powers = []
        self.temps = []
        self.throttles = set()

    def _poll(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", f"--query-gpu={self.QUERY}",
                     "--format=csv,noheader,nounits", "-i", self.gpu],
                    stderr=subprocess.DEVNULL, timeout=1.0).decode().strip()
                parts = [p.strip() for p in out.split(",")]
                self.sm_clocks.append(float(parts[0]))
                self.powers.append(float(parts[1]))
                self.temps.append(float(parts[2]))
                self.throttles.add(parts[3])
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def summary(self):
        def stats(xs):
            if not xs:
                return (float("nan"), float("nan"), float("nan"))
            return (min(xs), statistics.mean(xs), max(xs))

        active = sorted(t for t in self.throttles
                        if t and t not in ("Not Active", ""))
        return {
            "sm_clock": stats(self.sm_clocks),
            "power": stats(self.powers),
            "temp": stats(self.temps),
            "throttle": active,
            "n": len(self.sm_clocks),
        }


def _elem_size(dtype):
    return torch.empty((), dtype=dtype).element_size()


def _rotations_for(M, N, K, dtype):
    """Number of (a, b) copies whose footprint totals >= 2x L2, forcing cold
    caches (DRAM fetch) every iteration. At least 2 so buffers actually rotate."""
    l2 = int(torch.cuda.get_device_properties(DEVICE).L2_cache_size)
    per = (M * K + K * N) * _elem_size(dtype)
    n = max(2, -(-(2 * l2) // per))  # ceil(2*L2 / per), >= 2
    return n, l2, per


def _make_inputs(M, N, K, dtype, fill, n_copies):
    bufs = []
    for _ in range(n_copies):
        if fill == "zeros":
            a = torch.zeros((M, K), device=DEVICE, dtype=dtype)
            b = torch.zeros((K, N), device=DEVICE, dtype=dtype)
        elif fill == "uniform":
            a = torch.empty((M, K), device=DEVICE, dtype=dtype).uniform_(-1.0, 1.0)
            b = torch.empty((K, N), device=DEVICE, dtype=dtype).uniform_(-1.0, 1.0)
        else:  # "randn" (legacy behavior)
            a = torch.randn((M, K), device=DEVICE, dtype=dtype)
            b = torch.randn((K, N), device=DEVICE, dtype=dtype)
        bufs.append((a, b))
    return bufs


def _bench_rotating(call, bufs, quantiles):
    """Time `call(a, b)` rotating through `bufs`. Warm up for >= WARMUP_S, then
    time n_rep iterations with NO per-iteration allocation or cache memset
    (zero inter-launch overhead); cold L2 comes from rotating buffers whose
    total footprint exceeds 2x L2. Clocks/power are monitored throughout."""
    n = len(bufs)
    call(*bufs[0])
    torch.cuda.synchronize()

    # Estimate per-iteration cost to size warmup / rep iteration counts.
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for i in range(5):
        call(*bufs[i % n])
    e.record()
    torch.cuda.synchronize()
    est_ms = max(s.elapsed_time(e) / 5, 1e-3)

    n_warmup = max(1, int(WARMUP_S * 1000 / est_ms))
    n_rep = max(50, int(REP_S * 1000 / est_ms))

    for i in range(n_warmup):
        call(*bufs[i % n])
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_rep)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_rep)]
    with GpuMonitor() as mon:
        for i in range(n_rep):
            starts[i].record()
            call(*bufs[i % n])
            ends[i].record()
        torch.cuda.synchronize()
    times = sorted(st.elapsed_time(en) for st, en in zip(starts, ends))

    def q(frac):
        idx = min(len(times) - 1, max(0, int(frac * len(times))))
        return times[idx]

    med, lo_t, hi_t = q(quantiles[0]), q(quantiles[1]), q(quantiles[2])
    return med, lo_t, hi_t, n_rep, mon.summary()


def create_benchmark(versions, dtype=torch.float16, fill="zeros"):
    line_vals = [ref_lib.lower()] + versions
    line_names = [ref_lib] + versions
    dtype_name = {torch.float16: "fp16", torch.bfloat16: "bf16"}[dtype]

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],
            x_vals=[2048, 4096, 8192],
            line_arg="provider",
            line_vals=line_vals,
            line_names=line_names,
            ylabel="TFLOPS",
            plot_name=f"matmul-performance-{dtype_name}",
            args={},
        ))
    def benchmark(M, N, K, provider):
        n_copies, l2, per = _rotations_for(M, N, K, dtype)
        bufs = _make_inputs(M, N, K, dtype, fill, n_copies)
        quantiles = [0.5, 0.2, 0.8]

        if provider == ref_lib.lower():
            call = lambda a, b: torch.matmul(a, b)  # noqa: E731
        else:
            matmul = MATMUL_METHODS[provider]
            call = lambda a, b: matmul(a, b)  # noqa: E731

        med, lo_t, hi_t, n_rep, mon = _bench_rotating(call, bufs, quantiles)

        def tflops(ms):
            return 2 * M * N * K * 1e-12 / (ms * 1e-3)

        sm, pw = mon["sm_clock"], mon["power"]
        tot_mb = n_copies * per / 1024 / 1024
        print(
            f"[monitor] {provider:>9} M=N=K={M:<5} fill={fill} "
            f"rot={n_copies}x({tot_mb:.0f}MB>=2*L2={2 * l2 / 1024 / 1024:.0f}MB) "
            f"iters={n_rep} "
            f"clk[min/mean/max]={sm[0]:.0f}/{sm[1]:.0f}/{sm[2]:.0f}MHz "
            f"pwr={pw[0]:.0f}/{pw[1]:.0f}/{pw[2]:.0f}W "
            f"throttle={mon['throttle'] or 'none'} "
            f"-> {tflops(med):.1f} TFLOPS",
            flush=True)

        time.sleep(COOLDOWN_S)  # cooldown before next cell
        return tflops(med), tflops(hi_t), tflops(lo_t)

    return benchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark TLX Blackwell GEMM implementations")
    parser.add_argument(
        "--version",
        type=str,
        nargs="+",
        choices=list(MATMUL_METHODS.keys()),
        help=f"Run only the specified version(s). Choices: {list(MATMUL_METHODS.keys())}",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16"],
        help="Data type for the benchmark (default: fp16)",
    )
    parser.add_argument(
        "--fill",
        type=str,
        default="zeros",
        choices=["zeros", "uniform", "randn"],
        help="Input fill: zeros (fixed-frequency, matches denoise.sh clock lock), "
        "uniform [-1,1] (fixed-power), or randn (legacy). Default: zeros.",
    )
    args = parser.parse_args()

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    if is_blackwell():
        versions = args.version if args.version else list(MATMUL_METHODS.keys())
        print(f"Running benchmarks for: {versions} (dtype={args.dtype}, fill={args.fill})")
        benchmark = create_benchmark(versions, dtype=dtype, fill=args.fill)
        benchmark.run(print_data=True)
    else:
        print("Skipping benchmarks, no Blackwell GPU found.")
