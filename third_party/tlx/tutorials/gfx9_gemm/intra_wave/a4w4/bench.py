"""Correctness and timing harness for the TLX gfx950 MXFP4 GEMM."""

import argparse
import concurrent.futures
import multiprocessing
from pathlib import Path
import statistics
import time

import torch
import triton
from triton import knobs
from triton.runtime.jit import MockTensor

try:
    from .matmul_kernel import _a4w4_kernel
except ImportError:
    from matmul_kernel import _a4w4_kernel

SCALE_GROUP_SIZE = 32
BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 256
GROUP_SIZE_M = 4
NUM_XCDS = 8
DEFAULT_SIZES = [
    (4096, 4096, 1024),
    (4096, 4096, 2048),
    (4096, 4096, 4096),
    (4096, 4096, 8192),
    (4096, 4096, 16384),
    (4096, 4096, 32768),
]
DEFAULT_COMPILE_WORKERS = max(1, min(8, multiprocessing.cpu_count()))
TIMING_MODES = ("batched", "triton")
DEFAULT_WARMUP_LAUNCHES = 25
DEFAULT_TIMED_LAUNCHES = 1000
DEFAULT_TIMING_REPEATS = 7


class StridedMockTensor(MockTensor):

    def __init__(self, dtype, shape, strides):
        super().__init__(dtype, shape)
        self._strides = tuple(int(stride) for stride in strides)

    def stride(self):
        return self._strides


def active_torch_device():
    return triton.runtime.driver.active.get_active_torch_device()


def mxfp4_to_f32(x):
    x = x.repeat_interleave(2, dim=1)
    x[:, ::2] = x[:, ::2] & 0xF
    x[:, 1::2] = x[:, 1::2] >> 4
    values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
        device=x.device,
    )
    return values[x.long()]


def e8m0_to_f32(x):
    return torch.pow(2.0, x.to(torch.int16).to(torch.float32) - 127.0)


def generate_mxfp4_inputs(M, N, K, seed=42):
    device = active_torch_device()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    a_low = torch.randint(0, 16, (M, K // 2), dtype=torch.uint8, device=device, generator=generator)
    a_high = torch.randint(0, 16, (M, K // 2), dtype=torch.uint8, device=device, generator=generator)
    a = (a_high << 4) | a_low

    b_low = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device=device, generator=generator)
    b_high = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device=device, generator=generator)
    b = (b_high << 4) | b_low

    k_scales = K // SCALE_GROUP_SIZE
    m_pad = triton.cdiv(M, BLOCK_M) * BLOCK_M
    a_scales = torch.randint(124, 128, (k_scales, m_pad), dtype=torch.uint8, device=device, generator=generator).T[:M]
    b_scales = torch.randint(124, 128, (k_scales, N), dtype=torch.uint8, device=device, generator=generator).T
    return a, b, a_scales, b_scales


def torch_reference(a, b, a_scales, b_scales):
    a_f32 = mxfp4_to_f32(a) * e8m0_to_f32(a_scales).repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    b_f32 = mxfp4_to_f32(b) * e8m0_to_f32(b_scales).repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    return torch.mm(a_f32, b_f32.T).to(torch.bfloat16)


def nonnegative_int(text):
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {text!r}")
    return value


def positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}")
    return value


def parse_shape(text):
    parts = text.replace("x", ",").split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"shape must be MxNxK or M,N,K, got {text!r}")
    try:
        shape = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"shape must contain integer M/N/K values, got {text!r}") from exc
    M, N, K = shape
    if M <= 0 or N <= 0 or K <= 0:
        raise argparse.ArgumentTypeError(f"shape dimensions must be positive: {text!r}")
    if M % 256 or N % 256:
        raise argparse.ArgumentTypeError("TLX a4w4 requires M and N to be multiples of 256")
    if K < 1024 or K % 512:
        raise argparse.ArgumentTypeError("TLX a4w4 requires K >= 1024 and K to be a multiple of 512")
    return shape


def tflops(ms, M, N, K):
    return 2 * M * N * K * 1e-12 / (ms * 1e-3)


def launch_matmul(a, b, a_scales, b_scales, out=None):
    M = a.shape[0]
    K_packed = a.shape[1]
    K = K_packed * 2
    N = b.shape[0]
    c = out if out is not None else torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    grid_mn = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    compile_kwargs = {
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_K": BLOCK_K,
        "GROUP_SIZE_M": GROUP_SIZE_M,
        "NUM_XCDS": NUM_XCDS,
        "GRID_MN": grid_mn,
        "num_warps": 4,
        "num_stages": 1,
        "matrix_instr_nonkdim": 16,
    }
    _a4w4_kernel[(grid_mn, )](
        a,
        b,
        c,
        a_scales,
        b_scales,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        a_scales.stride(0),
        a_scales.stride(1),
        b_scales.stride(0),
        b_scales.stride(1),
        **compile_kwargs,
    )
    return c


def benchmark_matmul(a, b, a_scales, b_scales, out=None):
    return launch_matmul(
        a,
        b,
        a_scales,
        b_scales,
        out=out,
    )


def do_bench_batched(
    fn,
    *,
    warmup_launches=DEFAULT_WARMUP_LAUNCHES,
    timed_launches=DEFAULT_TIMED_LAUNCHES,
    repeats=DEFAULT_TIMING_REPEATS,
    device_interface=None,
):
    """Time one event span per batch and return the median per-launch time."""
    if warmup_launches < 0:
        raise ValueError("warmup_launches must be non-negative")
    if timed_launches <= 0:
        raise ValueError("timed_launches must be positive")
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    di = device_interface or triton.runtime.driver.active.get_device_interface()
    samples = []
    for _ in range(repeats):
        for _ in range(warmup_launches):
            fn()
        di.synchronize()

        start_event = di.Event(enable_timing=True)
        end_event = di.Event(enable_timing=True)
        start_event.record()
        for _ in range(timed_launches):
            fn()
        end_event.record()
        di.synchronize()
        samples.append(start_event.elapsed_time(end_event) / timed_launches)
    return statistics.median(samples)


def measure_matmul(args, fn):
    if args.timing_mode == "triton":
        return triton.testing.do_bench(
            fn,
            warmup=args.warmup,
            rep=args.rep,
            return_mode="median",
        )
    return do_bench_batched(
        fn,
        warmup_launches=args.warmup_launches,
        timed_launches=args.timed_launches,
        repeats=args.timing_repeats,
    )


def shape_cache_dir(cache_root, M, N, K):
    if cache_root is None:
        return None
    return Path(cache_root) / f"M{M}_N{N}_K{K}"


def compile_shape(shape, cache_root):
    M, N, K = shape
    cache_dir = shape_cache_dir(cache_root, M, N, K)
    a = MockTensor(torch.uint8, [M, K // 2])
    b = MockTensor(torch.uint8, [N, K // 2])
    c = MockTensor(torch.bfloat16, [M, N])
    k_scales = K // SCALE_GROUP_SIZE
    m_pad = triton.cdiv(M, BLOCK_M) * BLOCK_M
    a_scales = StridedMockTensor(torch.uint8, [M, k_scales], (1, m_pad))
    b_scales = StridedMockTensor(torch.uint8, [N, k_scales], (1, N))
    a_strides = a.stride()
    b_strides = b.stride()
    c_strides = c.stride()
    a_scale_strides = a_scales.stride()
    b_scale_strides = b_scales.stride()
    grid_mn = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    with knobs.cache.scope():
        if cache_dir is not None:
            knobs.cache.dir = str(cache_dir)
        compile_kwargs = {
            "BLOCK_M": BLOCK_M,
            "BLOCK_N": BLOCK_N,
            "BLOCK_K": BLOCK_K,
            "GROUP_SIZE_M": GROUP_SIZE_M,
            "NUM_XCDS": NUM_XCDS,
            "GRID_MN": grid_mn,
            "num_warps": 4,
            "num_stages": 1,
            "matrix_instr_nonkdim": 16,
            "grid": (grid_mn, ),
        }
        _a4w4_kernel.warmup(
            a,
            b,
            c,
            a_scales,
            b_scales,
            M,
            N,
            K,
            a_strides[0],
            a_strides[1],
            b_strides[0],
            b_strides[1],
            c_strides[0],
            c_strides[1],
            a_scale_strides[0],
            a_scale_strides[1],
            b_scale_strides[0],
            b_scale_strides[1],
            **compile_kwargs,
        )


def compile_shape_worker(shape, cache_root):
    start = time.monotonic()
    compile_shape(shape, cache_root)
    return shape, time.monotonic() - start


def precompile_shapes(sizes, cache_root, workers):
    if workers == 0:
        return
    workers = min(workers, len(sizes))
    print(f"\nPrecompiling {len(sizes)} shape(s) with {workers} worker(s):", flush=True)
    if workers == 1:
        for shape in sizes:
            compiled_shape, elapsed = compile_shape_worker(shape, cache_root)
            M, N, K = compiled_shape
            print(f"  compiled {M}x{N}x{K} in {elapsed:.1f}s", flush=True)
        return

    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = [executor.submit(compile_shape_worker, shape, cache_root) for shape in sizes]
        for future in concurrent.futures.as_completed(futures):
            compiled_shape, elapsed = future.result()
            M, N, K = compiled_shape
            print(f"  compiled {M}x{N}x{K} in {elapsed:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description="TLX gfx950 MXFP4 GEMM benchmark")
    parser.add_argument("--shape", action="append", type=parse_shape, default=None)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument(
        "--timing-mode",
        choices=TIMING_MODES,
        default="batched",
        help=("batched records one event span around back-to-back launches; triton records "
              "each launch and clears L2; default: batched"),
    )
    parser.add_argument(
        "--rep",
        "--rep-ms",
        dest="rep",
        type=positive_int,
        default=200,
        help="timed duration in milliseconds for --timing-mode=triton; default: 200",
    )
    parser.add_argument(
        "--warmup",
        "--warmup-ms",
        dest="warmup",
        type=nonnegative_int,
        default=25,
        help="warmup duration in milliseconds for --timing-mode=triton; default: 25",
    )
    parser.add_argument(
        "--warmup-launches",
        type=nonnegative_int,
        default=DEFAULT_WARMUP_LAUNCHES,
        help=("warmup launches before each batched timing repeat; "
              f"default: {DEFAULT_WARMUP_LAUNCHES}"),
    )
    parser.add_argument(
        "--timed-launches",
        type=positive_int,
        default=DEFAULT_TIMED_LAUNCHES,
        help=f"launches in each batched event span; default: {DEFAULT_TIMED_LAUNCHES}",
    )
    parser.add_argument(
        "--timing-repeats",
        type=positive_int,
        default=DEFAULT_TIMING_REPEATS,
        help=f"batched timing samples whose median is reported; default: {DEFAULT_TIMING_REPEATS}",
    )
    parser.add_argument("--atol", type=float, default=1e-1)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--compile-workers",
        type=nonnegative_int,
        default=DEFAULT_COMPILE_WORKERS,
        help=f"parallel workers for the precompile phase; 0 disables it; default: {DEFAULT_COMPILE_WORKERS}",
    )
    parser.add_argument("--no-bench", action="store_true")
    args = parser.parse_args()

    sizes = args.shape or DEFAULT_SIZES
    if args.K is not None:
        sizes = [(m, n, k) for m, n, k in sizes if k == args.K]
    if not sizes:
        raise SystemExit("no shapes selected")
    precompile_shapes(sizes, args.cache_dir, args.compile_workers)

    if args.timing_mode == "triton":
        timing_summary = f"triton median, {args.warmup}ms warmup/{args.rep}ms timed"
    else:
        timing_summary = (f"batched median, {args.timing_repeats}x{args.timed_launches} timed launches, "
                          f"{args.warmup_launches} warmups/repeat")
    print(f"\nTLX a4w4 gfx950 ({timing_summary}):")
    print(f"{'M':>6s} {'N':>6s} {'K':>6s}  {'status':>10s}  {'max_err':>10s}  {'TLX':>17s}")
    for M, N, K in sizes:
        a, b, a_scales, b_scales = generate_mxfp4_inputs(M, N, K)
        cache_dir = shape_cache_dir(args.cache_dir, M, N, K)
        with knobs.cache.scope():
            if cache_dir is not None:
                knobs.cache.dir = str(cache_dir)
            c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
            benchmark_matmul(a, b, a_scales, b_scales, out=c)
            torch.cuda.synchronize()
            ref = torch_reference(a, b, a_scales, b_scales)
            max_err = (c - ref).abs().max().item()
            ok = torch.allclose(c, ref, atol=args.atol, rtol=args.rtol)
            if ok and not args.no_bench:
                ms = measure_matmul(
                    args,
                    lambda: benchmark_matmul(a, b, a_scales, b_scales, out=c),
                )
                perf = f"{tflops(ms, M, N, K):8.1f}T/{ms:6.3f}ms"
            else:
                perf = "-"
        status = "ok" if ok else "FAIL"
        print(f"{M:6d} {N:6d} {K:6d}  {status:>10s}  {max_err:10.4f}  {perf:>17s}", flush=True)


if __name__ == "__main__":
    main()
