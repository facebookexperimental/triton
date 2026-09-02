"""Shared measurement harness for the gfx942 (MI300X) perf scripts.

Used by ``test_amd_{gemm,addmm,bmm}_gfx942_perf.py`` so the three stay in sync.
The method mirrors ``test_blackwell_gemm_perf.py``: warm up for long enough that
clocks and power settle, then take a single continuous timed window and report
quantiles over the per-iteration times.

That is deliberately *not* what ``triton.testing.do_bench`` does, and the
difference matters here. ``do_bench`` warms for 25 ms and times ~``rep`` ms of
calls; wrapping it in short bursts separated by sleeps -- the obvious way to
"avoid throttling" -- actively prevents the clocks from ever settling, and the
resulting burst-to-burst variation was measured at 14-20% on this part for
1024^3 addmm. Warming continuously for ``WARMUP_S`` and then measuring one
``REP_S`` window instead removes that failure mode: every timed iteration runs at
the same steady-state clock.

Reported statistics are the median and the p20/p80 band over the individual
iteration timings (thousands of them, not ten burst means), so the band describes
the real per-call distribution rather than drift between sampling windows.

Note ``do_bench``'s 256 MB L2 flush between calls is also gone with it, so each
iteration now runs against whatever the previous one left in cache. For these
shapes that is a warm-cache number. Buffer rotation to force cold caches is a
possible follow-up; it needs to target MI300X's 256 MB Infinity Cache rather than
the 4 MB ``L2_cache_size`` torch reports, or it will under-rotate badly.

Dropping that flush also changes what the short shapes measure, and the change is
not cosmetic. The memset used to keep the GPU busy while Python prepared the next
launch, so back-to-back kernels pipelined and the event pair around each call saw
only kernel time. Without it the host is the bottleneck for small tiles: a TLX
call pays the autotuner wrapper plus a Python launch (the C dispatcher is
NVIDIA-only, so gfx942 never gets one), which ``torch.matmul`` does not. Measured
at 1024^3: aten 16.8 us/call against TLX 48.1 us, where the old flush-driven
method reported 19 us and 17 us. The kernels did not change -- the earlier numbers
simply hid launch overhead behind the flush. Treat any row whose per-call time is
within an order of magnitude of ~40 us as launch-bound, and read it as end-to-end
cost rather than as a statement about the kernel.

For clock stability wrap the run in ``third_party/tlx/denoise.sh``, which on
MI300X pins sclk via ``rocm-smi --setperfdeterminism`` and applies a power
overdrive.
"""

import argparse
import os
import time
from typing import Callable, NamedTuple

# Set before the first kernel launch. Triton reads its knobs lazily (env_bool
# resolves os.environ on access), so importing this module is early enough.
#
# TRITON_DISABLE_POST_MISCHED keeps the LLVM post-RA scheduler from re-ordering
# the hand-written load / MFMA / local_store order in the hot loop.
os.environ.setdefault("TRITON_DISABLE_POST_MISCHED", "1")
# TRITON_USE_C_DISPATCHER defaults on, but the C dispatcher needs the NVIDIA
# backend's asm["launch_metadata"], which the AMD backend never emits -- so on
# gfx942 it can never be built and every launch warns. Turning it off skips a
# branch that could not have succeeded; the launch path is unchanged.
os.environ.setdefault("TRITON_USE_C_DISPATCHER", "0")

import torch  # noqa: E402
import triton  # noqa: E402

from triton._internal_testing import is_hip_cdna3  # noqa: E402

DEVICE = triton.runtime.driver.active.get_active_torch_device()

REF = "aten"

# Idle between shape rows, so one row does not heat the part into the next.
COOLDOWN_S = 1.0

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}

# Continuous execution before timing starts, so clocks and power settle.
DEFAULT_WARMUP_S = 3.0
# Length of the single timed window.
DEFAULT_REP_S = 2.0
# Floor on timed iterations, for shapes slow enough that REP_S yields few.
MIN_ITERS = 50
# Ceiling on timed iterations. Each one costs a CUDA event pair, and at these
# shapes (tens of microseconds per call, versus ~1 ms for the Blackwell GEMM this
# method is taken from) an uncapped REP_S window asks for tens of thousands of
# events, whose record overhead lands inside the measured interval: 1024^3 GEMM
# measured 55 us/iter at 37k iters against 19 us/iter capped. Blackwell's own
# window lands at ~1000-4000 iters, so cap here to stay in that range.
MAX_ITERS = 4000
# Quantiles reported: median, then the low/high band edges.
QUANTILES = (0.5, 0.2, 0.8)


class Measurement(NamedTuple):
    """Result of :func:`measure`.

    ``ms``    -- median per-iteration time.
    ``lo_ms`` -- p20 (fast edge).
    ``hi_ms`` -- p80 (slow edge).
    ``iters`` -- timed iterations in the window.
    """

    ms: float
    lo_ms: float
    hi_ms: float
    iters: int

    @property
    def band_pct(self) -> float:
        """(p80 - p20) / median, as a percent -- how tight the distribution is."""
        return 100.0 * (self.hi_ms - self.lo_ms) / self.ms if self.ms > 0 else float("nan")


def measure_samples(fn, warmup_s=DEFAULT_WARMUP_S, rep_s=DEFAULT_REP_S) -> list[float]:
    """Per-iteration times (ms, in execution order) from one continuous window.

    The raw form of :func:`measure`, split out so callers that want the whole
    distribution -- the kernel-optimization harness computes its own median and
    coefficient of variation -- get it from the same settling method rather than
    reimplementing it and drifting.
    """
    fn()
    torch.cuda.synchronize()

    # Estimate per-iteration cost to size the warmup and timed iteration counts.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        fn()
    end.record()
    torch.cuda.synchronize()
    est_ms = max(start.elapsed_time(end) / 5, 1e-3)

    n_warmup = max(1, int(warmup_s * 1000 / est_ms))
    n_iters = min(MAX_ITERS, max(MIN_ITERS, int(rep_s * 1000 / est_ms)))

    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]
    for i in range(n_iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()

    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def measure(fn, warmup_s=DEFAULT_WARMUP_S, rep_s=DEFAULT_REP_S) -> Measurement:
    """Time ``fn`` over one continuous window after a settling warmup."""
    times = sorted(measure_samples(fn, warmup_s=warmup_s, rep_s=rep_s))

    def q(frac):
        return times[min(len(times) - 1, max(0, int(frac * len(times))))]

    return Measurement(ms=q(QUANTILES[0]), lo_ms=q(QUANTILES[1]), hi_ms=q(QUANTILES[2]), iters=len(times))


def add_measurement_args(parser):
    """Add the knobs this harness understands to an ``argparse`` parser."""
    parser.add_argument("--warmup-s", type=float, default=DEFAULT_WARMUP_S,
                        help=f"continuous warmup before timing, seconds (default {DEFAULT_WARMUP_S})")
    parser.add_argument("--rep-s", type=float, default=DEFAULT_REP_S,
                        help=f"timed window, seconds (default {DEFAULT_REP_S})")
    return parser


def measurement_kwargs(args):
    """Pull the harness knobs back out of parsed args."""
    return {"warmup_s": args.warmup_s, "rep_s": args.rep_s}


class OpSpec(NamedTuple):
    """Everything that differs between the gemm / addmm / bmm perf scripts.

    ``make_inputs`` takes the shape tuple plus a dtype and returns the argument
    tuple that ``ref`` and every provider are called with, so an op that needs a
    bias or a batch dimension just returns more tensors.
    """

    name: str  # e.g. "amd_addmm_gfx942", used in the table title and plot name
    axes: tuple  # shape axis labels, e.g. ("M", "N", "K")
    shapes: list  # list of tuples, each matching `axes`
    make_inputs: Callable  # (shape, dtype) -> tuple of tensors
    ref: Callable  # (*tensors) -> out, the aten baseline
    providers: dict  # provider name -> (*tensors) -> out
    flops: Callable  # (shape) -> total flops for one call

    def tflops(self, shape, ms):
        return self.flops(shape) * 1e-12 / (ms * 1e-3)


def _shape_header(spec):
    return " ".join(f"{axis:>6}" for axis in spec.axes)


def _shape_cells(spec, shape):
    return " ".join(f"{dim:>6}" for dim in shape)


def run_speedup_table(spec, versions, dtype, mkw):
    """Print TFLOPS, the p20/p80 band, and the speedup of each version over aten."""
    dtype_name = {v: k for k, v in DTYPES.items()}[dtype]
    header = f"{_shape_header(spec)} {REF + ' TFLOPS':>14} {'iters':>7} {'band':>7}"
    for v in versions:
        header += f" {v + ' TFLOPS':>16} {'vs ' + REF:>10} {'band':>7}"
    print(f"\n=== {spec.name} vs {REF} ({dtype_name}) ===")
    print(header)
    for shape in spec.shapes:
        tensors = spec.make_inputs(shape, dtype)
        ref = measure(lambda: spec.ref(*tensors), **mkw)
        row = (f"{_shape_cells(spec, shape)} {spec.tflops(shape, ref.ms):>14.1f} "
               f"{ref.iters:>7} {ref.band_pct:>6.1f}%")
        for v in versions:
            fn = spec.providers[v]
            m = measure(lambda: fn(*tensors), **mkw)
            row += (f" {spec.tflops(shape, m.ms):>16.1f} {ref.ms / m.ms:>9.2f}x "
                    f"{m.band_pct:>6.1f}%")
        print(row, flush=True)
        time.sleep(COOLDOWN_S)


def create_benchmark(spec, versions, dtype, mkw):
    """The same numbers via triton.testing.perf_report, for plotting."""
    line_vals = [REF] + versions
    dtype_name = {v: k for k, v in DTYPES.items()}[dtype]

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=list(spec.axes),
            x_vals=spec.shapes,
            line_arg="provider",
            line_vals=line_vals,
            line_names=line_vals,
            ylabel="TFLOPS",
            plot_name=f"{spec.name}-performance-{dtype_name}",
            args={},
        ))
    def benchmark(provider, **dims):
        # perf_report calls fn(**x_args, provider=...), so the axes arrive by
        # keyword; rebuild the shape tuple in spec.axes order.
        shape = tuple(dims[axis] for axis in spec.axes)
        tensors = spec.make_inputs(shape, dtype)
        call = spec.ref if provider == REF else spec.providers[provider]
        m = measure(lambda: call(*tensors), **mkw)
        return spec.tflops(shape, m.ms)

    return benchmark


def main(spec):
    """argparse + arch gate + dispatch, shared by the three perf scripts."""
    parser = argparse.ArgumentParser(description=f"Benchmark {spec.name} against aten on MI300X")
    parser.add_argument("--version", type=str, nargs="+", choices=list(spec.providers),
                        help=f"Run only the specified version(s). Choices: {list(spec.providers)}")
    parser.add_argument("--dtype", type=str, default="fp16", choices=list(DTYPES))
    parser.add_argument("--table", action="store_true", help="Print a TFLOPS + speedup-vs-aten table")
    add_measurement_args(parser)
    args = parser.parse_args()
    mkw = measurement_kwargs(args)
    dtype = DTYPES[args.dtype]

    if not is_hip_cdna3():
        print("Skipping benchmarks: this script targets AMD gfx942 (MI300X / CDNA3).")
        return
    versions = args.version if args.version else list(spec.providers)
    print(f"Running benchmarks for: {versions} (dtype={args.dtype})")
    if args.table:
        run_speedup_table(spec, versions, dtype, mkw)
    else:
        create_benchmark(spec, versions, dtype, mkw).run(print_data=True)
