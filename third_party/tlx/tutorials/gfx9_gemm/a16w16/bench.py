"""Benchmark harness for the gfx9 TLX GEMM tutorial kernels."""

import argparse
import concurrent.futures
from contextlib import contextmanager
import importlib.util
import multiprocessing
import os
from pathlib import Path
import statistics
import time

import torch
import triton
from triton import knobs
from triton.runtime.jit import MockTensor

VERSION_MAP = {
    0: "v0_naive",
    1: "v1_buffer_load",
    2: "v2_async_copy",
    3: "v3_lds",
    4: "v4_global_prefetch",
    5: "v5_local_prefetch",
    6: "v6_loop_unroll",
    7: "v7_slice",
    8: "v8_warp_pipeline",
    9: "v9_beyond_hotloop",
    10: "wave_8wave",
    11: "wave_4wave_specialized",
}

PROVIDER_LABELS = {
    "rocblas": "rocBLAS",
    "tlx": "TLX",
    "wave": "Wave",
}

BENCH_DIR = Path(__file__).resolve().parent
TILE_M = 256
TILE_N = 256
TILE_K = 64
TWO_STAGE_K = 2 * TILE_K
TUTORIAL_PROVIDERS = frozenset({"tlx", "wave"})
TWO_STAGE_K_VERSIONS = frozenset(range(5, 12))
UNTILED_K_VERSIONS = frozenset({0, 1})
WAVE_STRUCTURED_VERSIONS = frozenset({"wave_8wave", "wave_4wave_specialized"})
GROUPED_PID_VERSIONS = frozenset({"v9_beyond_hotloop", *WAVE_STRUCTURED_VERSIONS})
EIGHT_WARP_VERSIONS = frozenset({"v8_warp_pipeline", "v9_beyond_hotloop", "wave_8wave"})
MULTI_WAVE_SPECIALIZED_VERSIONS = frozenset({"wave_4wave_specialized"})
DEFAULT_COMPILE_WORKERS = max(1, min(8, os.cpu_count() or 1))
INPUT_MODES = ("normal", "hpl", "rand-int", "zero", "ones")
TIMING_MODES = ("triton", "batched")
DEFAULT_INPUT_SEED = 0
DEFAULT_WARMUP_LAUNCHES = 25
DEFAULT_TIMED_LAUNCHES = 1000
DEFAULT_TIMING_REPEATS = 7
_INPUT_INIT_CHUNK_ELEMENTS = 4 * 1024 * 1024
_HIPBLASLT_RNG_SEED_STRIDE = 0x9E3779B9
_UINT32_MAX = (1 << 32) - 1
_LOGICAL_RSHIFT_17_MASK = (1 << (64 - 17)) - 1


class StridedMockTensor(MockTensor):

    def __init__(self, dtype, shape, strides):
        super().__init__(dtype, shape)
        self._strides = tuple(int(stride) for stride in strides)

    def stride(self):
        return self._strides


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


def get_x_vals():
    return [
        (4096, 4096, 1024),
        (4096, 4096, 2048),
        (4096, 4096, 4096),
        (4096, 4096, 8192),
    ]


def parse_shape(text):
    parts = text.replace("x", ",").split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"shape must be MxNxK or M,N,K, got {text!r}")
    try:
        shape = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"shape must contain integer M/N/K values, got {text!r}") from exc
    if any(dim <= 0 for dim in shape):
        raise argparse.ArgumentTypeError(f"shape dimensions must be positive: {text!r}")
    return shape


def validate_shape_for_providers(shape, version, providers):
    if not TUTORIAL_PROVIDERS.intersection(providers):
        return
    M, N, K = shape
    if M % TILE_M:
        raise argparse.ArgumentTypeError(f"tutorial kernels require M to be a multiple of {TILE_M}, got {M}")
    if N % TILE_N:
        raise argparse.ArgumentTypeError(f"tutorial kernels require N to be a multiple of {TILE_N}, got {N}")
    if version not in UNTILED_K_VERSIONS and K % TILE_K:
        raise argparse.ArgumentTypeError(f"tutorial kernels v{version} require K to be a multiple of "
                                         f"{TILE_K}, got {K}")
    if version in TWO_STAGE_K_VERSIONS and (K < TWO_STAGE_K or K % TWO_STAGE_K):
        raise argparse.ArgumentTypeError(f"tutorial kernels v{version} prefetch two {TILE_K}-wide K tiles; "
                                         f"K must be at least {TWO_STAGE_K} and a multiple of {TWO_STAGE_K}, "
                                         f"got {K}")


def validate_shapes_for_providers(shapes, version, providers):
    for shape in shapes:
        validate_shape_for_providers(shape, version, providers)


def load_matmul_module(version_dir, suffix):
    kernel_path = BENCH_DIR / version_dir / "matmul_kernel.py"
    spec = importlib.util.spec_from_file_location(
        f"_tlx_gfx9_gemm_{version_dir}_{suffix}_{time.time_ns()}",
        kernel_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import benchmark kernel from {kernel_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_driver(provider):
    if provider == "tlx":
        from triton.backends.amd import driver as amd_driver

        return amd_driver.HIPDriver()
    if provider == "wave":
        from triton.backends.tlx_wave import driver as tlx_wave_driver

        return tlx_wave_driver.TLXWaveDriver()
    return None


@contextmanager
def active_driver(driver):
    if driver is None:
        yield
        return
    previous_driver = triton.runtime.driver.active
    triton.runtime.driver.set_active(driver)
    try:
        yield
    finally:
        triton.runtime.driver.set_active(previous_driver)


def provider_defaults(version):
    if version in (10, 11):
        return ["wave"]
    if version == 9:
        return ["tlx", "wave"]
    return ["rocblas", "tlx"]


def _hipblaslt_random_u32(indices):
    """Return hipBLASLt's deterministic pseudo_random_device value per index."""
    state = indices * 1664525 + 1013904223
    for _ in range(3):
        logical_rshift_17 = (state >> 17) & _LOGICAL_RSHIFT_17_MASK
        state = state ^ (state << 13) ^ logical_rshift_17 ^ (state << 5)
    return state & _UINT32_MAX


def _make_hipblaslt_input(rows, cols, device, input_mode, seed, *, checkerboard_sign=False):
    """Build exact seed-zero hipBLASLt HPL/rand_int data without large temporaries."""
    result = torch.empty((rows, cols), device=device, dtype=torch.float16)
    chunk_rows = max(1, _INPUT_INIT_CHUNK_ELEMENTS // cols)
    seed_offset = seed * _HIPBLASLT_RNG_SEED_STRIDE
    col_ids = None
    if checkerboard_sign:
        col_ids = torch.arange(cols, device=device, dtype=torch.int64)[None, :]

    for row_begin in range(0, rows, chunk_rows):
        row_end = min(rows, row_begin + chunk_rows)
        flat_begin = row_begin * cols
        flat_end = row_end * cols
        indices = torch.arange(flat_begin, flat_end, device=device, dtype=torch.int64)
        random_u32 = _hipblaslt_random_u32(indices + seed_offset)
        if input_mode == "hpl":
            values = random_u32.to(torch.float64) / float(_UINT32_MAX) - 0.5
        else:
            values = random_u32.remainder(5) - 2
            if checkerboard_sign:
                row_ids = torch.arange(row_begin, row_end, device=device, dtype=torch.int64)[:, None]
                negate = ((row_ids ^ col_ids) & 1) == 0
                values = values.reshape(row_end - row_begin, cols)
                values = torch.where(negate, -values, values)
        result[row_begin:row_end].copy_(values.reshape(row_end - row_begin, cols))
    return result


def _make_input_storage(rows, cols, device, input_mode, seed, generator, *, is_b=False):
    if input_mode == "normal":
        return torch.randn(
            (rows, cols),
            device=device,
            dtype=torch.float16,
            generator=generator,
        )
    if input_mode in {"hpl", "rand-int"}:
        return _make_hipblaslt_input(
            rows,
            cols,
            device,
            input_mode,
            seed,
            checkerboard_sign=is_b and input_mode == "rand-int",
        )
    if input_mode == "zero":
        return torch.zeros((rows, cols), device=device, dtype=torch.float16)
    if input_mode == "ones":
        return torch.ones((rows, cols), device=device, dtype=torch.float16)
    raise ValueError(f"unsupported input mode: {input_mode}")


def make_inputs(M, N, K, device, b_layout, input_mode="normal", seed=DEFAULT_INPUT_SEED):
    generator = None
    if input_mode == "normal":
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    a = _make_input_storage(M, K, device, input_mode, seed, generator)
    if b_layout == "contiguous":
        b = _make_input_storage(K, N, device, input_mode, seed, generator, is_b=True)
    else:
        b_storage = _make_input_storage(N, K, device, input_mode, seed, generator, is_b=True)
        b = b_storage.T
    return a, b


def launch_tutorial_matmul(module, version_dir, a, b, out=None, extra_compile_options=None):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    if version_dir in WAVE_STRUCTURED_VERSIONS:
        assert b.stride(0) == 1, f"{version_dir} expects a K-contiguous transposed-B view"
    M, K = a.shape
    K, N = b.shape
    if out is None:
        out = torch.empty((M, N), device=a.device, dtype=a.dtype)
    else:
        assert out.shape == (M, N), "Output shape must match the GEMM result"
        assert out.device == a.device, "Output must be on the same device as the inputs"
        assert out.dtype == a.dtype, "Output dtype must match the input dtype"
    BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 64
    grid_m = triton.cdiv(M, BLOCK_M)
    grid_n = triton.cdiv(N, BLOCK_N)
    grid_mn = grid_m * grid_n
    grid = (grid_m, grid_n) if version_dir in WAVE_STRUCTURED_VERSIONS else (grid_mn, )
    compile_kwargs = {
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_K": BLOCK_K,
        "num_warps": 8 if version_dir in EIGHT_WARP_VERSIONS else 4,
        "num_stages": 1,
        "matrix_instr_nonkdim": 16,
    }
    if version_dir in GROUPED_PID_VERSIONS:
        compile_kwargs.update({
            "GROUP_SIZE_M": 4,
            "NUM_XCDS": 8,
            "GRID_MN": grid_mn,
        })
    if extra_compile_options:
        compile_kwargs.update(extra_compile_options)

    kernel = getattr(module, version_dir)
    kernel[grid](
        a,
        b,
        out,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        **compile_kwargs,
    )
    return out


def provider_matmul(args, provider, module, version_dir, a, b, out=None):
    if provider == "rocblas":
        return torch.matmul(a, b, out=out)
    extra_compile_options = {}
    if provider == "wave" and args.wave_split_barriers:
        extra_compile_options["tlx_wave_enable_split_barriers"] = True
    if provider == "wave" and version_dir in MULTI_WAVE_SPECIALIZED_VERSIONS:
        extra_compile_options["tlx_wave_enable_multi_wave_specialize"] = True
    return launch_tutorial_matmul(
        module,
        version_dir,
        a,
        b,
        out=out,
        extra_compile_options=extra_compile_options or None,
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


def measure_provider(args, fn):
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


def benchmark_provider(args, provider, version_dir, a, b, ref, M, N, K):
    module = None
    if provider != "rocblas":
        module = load_matmul_module(version_dir, provider)
    driver = make_driver(provider)
    cache_dir = compile_cache_dir(args.cache_dir, version_dir, provider, M, N, K)

    with active_driver(driver), knobs.cache.scope(), knobs.runtime.scope():
        if cache_dir is not None:
            knobs.cache.dir = str(cache_dir)
        if args.arch is not None:
            knobs.runtime.override_arch = args.arch
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
        provider_matmul(args, provider, module, version_dir, a, b, out=c)
        torch.cuda.synchronize()
        ok = torch.allclose(c, ref, atol=args.atol, rtol=args.rtol)
        max_err = (c - ref).abs().max().item()
        if not ok:
            bad = int((~torch.isclose(c, ref, atol=args.atol, rtol=args.rtol)).sum().item())
            return {
                "ok": False,
                "max_err": max_err,
                "bad": bad,
                "ms": None,
                "tflops": None,
            }
        ms = measure_provider(
            args,
            lambda: provider_matmul(args, provider, module, version_dir, a, b, out=c),
        )
    return {
        "ok": True,
        "max_err": max_err,
        "bad": 0,
        "ms": ms,
        "tflops": tflops(ms, M, N, K),
    }


def tflops(ms, M, N, K):
    return 2 * M * N * K * 1e-12 / (ms * 1e-3)


def make_mock_inputs(M, N, K, b_layout):
    a = MockTensor(torch.float16, [M, K])
    if b_layout == "contiguous":
        b = MockTensor(torch.float16, [K, N])
    else:
        b = StridedMockTensor(torch.float16, [K, N], (1, K))
    c = MockTensor(torch.float16, [M, N])
    return a, b, c


def compile_cache_dir(cache_root, version_dir, provider, M, N, K):
    if cache_root is None:
        return None
    return Path(cache_root) / version_dir / provider / f"M{M}_N{N}_K{K}"


def compile_provider_shape(provider, version_dir, shape, b_layout, cache_root, arch, wave_split_barriers=False):
    if provider == "rocblas":
        return shape, provider, 0.0

    M, N, K = shape
    module = load_matmul_module(version_dir, f"compile_{provider}")
    kernel = getattr(module, version_dir)
    driver = make_driver(provider)
    cache_dir = compile_cache_dir(cache_root, version_dir, provider, M, N, K)
    a, b, c = make_mock_inputs(M, N, K, b_layout)
    BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 64
    grid_m = triton.cdiv(M, BLOCK_M)
    grid_n = triton.cdiv(N, BLOCK_N)
    grid_mn = grid_m * grid_n
    grid = (grid_m, grid_n) if version_dir in WAVE_STRUCTURED_VERSIONS else (grid_mn, )
    compile_kwargs = {
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_K": BLOCK_K,
        "num_warps": 8 if version_dir in EIGHT_WARP_VERSIONS else 4,
        "num_stages": 1,
        "matrix_instr_nonkdim": 16,
        "grid": grid,
    }
    if version_dir in GROUPED_PID_VERSIONS:
        compile_kwargs.update({
            "GROUP_SIZE_M": 4,
            "NUM_XCDS": 8,
            "GRID_MN": grid_mn,
        })
    if provider == "wave" and wave_split_barriers:
        compile_kwargs["tlx_wave_enable_split_barriers"] = True
    if provider == "wave" and version_dir in MULTI_WAVE_SPECIALIZED_VERSIONS:
        compile_kwargs["tlx_wave_enable_multi_wave_specialize"] = True

    start = time.monotonic()
    with active_driver(driver), knobs.cache.scope(), knobs.runtime.scope():
        if cache_dir is not None:
            knobs.cache.dir = str(cache_dir)
        if arch is not None:
            knobs.runtime.override_arch = arch
        a_strides = a.stride()
        b_strides = b.stride()
        c_strides = c.stride()
        kernel.warmup(
            a,
            b,
            c,
            M,
            N,
            K,
            a_strides[0],
            a_strides[1],
            b_strides[0],
            b_strides[1],
            c_strides[0],
            c_strides[1],
            **compile_kwargs,
        )
    return shape, provider, time.monotonic() - start


def precompile_shapes(args, version_dir, providers, sizes):
    if args.compile_workers == 0:
        return

    jobs = [
        (provider, shape)
        for shape in sizes
        for provider in providers
        if provider != "rocblas"
    ]
    if not jobs:
        return

    workers = min(args.compile_workers, len(jobs))
    print(f"\nPrecompiling {len(jobs)} provider/shape job(s) with {workers} worker(s):", flush=True)

    if workers == 1:
        for provider, shape in jobs:
            compiled_shape, compiled_provider, elapsed = compile_provider_shape(
                provider, version_dir, shape, args.b_layout, args.cache_dir, args.arch, args.wave_split_barriers)
            M, N, K = compiled_shape
            print(f"  compiled {compiled_provider} {M}x{N}x{K} in {elapsed:.1f}s", flush=True)
        return

    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = [
            executor.submit(
                compile_provider_shape,
                provider,
                version_dir,
                shape,
                args.b_layout,
                args.cache_dir,
                args.arch,
                args.wave_split_barriers,
            )
            for provider, shape in jobs
        ]
        for future in concurrent.futures.as_completed(futures):
            compiled_shape, compiled_provider, elapsed = future.result()
            M, N, K = compiled_shape
            print(f"  compiled {compiled_provider} {M}x{N}x{K} in {elapsed:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description="TLX GEMM benchmark")
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--version", type=int, default=0, choices=range(0, 12))
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=tuple(PROVIDER_LABELS),
        default=None,
        help=("providers to benchmark. Defaults to rocblas tlx, except v9 defaults "
              "to tlx wave and the Wave-derived variants default to wave."),
    )
    parser.add_argument(
        "--shape",
        action="append",
        type=parse_shape,
        default=None,
        help=("custom shape as MxNxK or M,N,K. Can be repeated. TLX/Wave "
              "providers require tutorial tile-compatible shapes."),
    )
    parser.add_argument(
        "--b-layout",
        choices=("transposed", "contiguous"),
        default="transposed",
        help="layout used for B input; transposed matches the tutorial benchmark.",
    )
    parser.add_argument(
        "--input-mode",
        choices=INPUT_MODES,
        default="normal",
        help=("input distribution; hpl and rand-int reproduce hipBLASLt/Wave "
              "seed-zero data, and rand-int applies their alternating sign to B"),
    )
    parser.add_argument(
        "--seed",
        type=nonnegative_int,
        default=DEFAULT_INPUT_SEED,
        help=f"deterministic input seed; default: {DEFAULT_INPUT_SEED}",
    )
    parser.add_argument(
        "--timing-mode",
        choices=TIMING_MODES,
        default="triton",
        help=("triton records every launch and clears L2; batched records one event span "
              "around many back-to-back launches"),
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
    # The f16 tutorial reductions can differ from torch by one or two fp16 ulps
    # on both TLX/LLVM and Wave for larger K. Keep the default tolerance aligned
    # with the observed backend-independent drift so perf sweeps do not fail
    # numerically identical TLX/Wave results.
    parser.add_argument("--atol", type=float, default=3e-1)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--arch", default=None, help="optional Triton runtime arch override")
    parser.add_argument("--cache-dir", default=None, help="optional Triton cache root")
    parser.add_argument("--wave-opt", default=None, help="optional path to wave-opt")
    parser.add_argument(
        "--wave-split-barriers",
        action="store_true",
        help="compile Wave provider kernels with tlx_wave_enable_split_barriers=True",
    )
    parser.add_argument(
        "--compile-workers",
        type=nonnegative_int,
        default=DEFAULT_COMPILE_WORKERS,
        help=f"parallel workers for a precompile phase; 0 disables it; default: {DEFAULT_COMPILE_WORKERS}",
    )
    args = parser.parse_args()

    if args.wave_opt:
        os.environ["TRITON_WAVE_OPT"] = args.wave_opt

    version_dir = VERSION_MAP[args.version]
    providers = (list(args.providers) if args.providers is not None else provider_defaults(args.version))
    sizes = list(args.shape) if args.shape is not None else get_x_vals()
    if args.K:
        sizes = [(m, n, k) for m, n, k in sizes if k == args.K]
    if not sizes:
        raise SystemExit("no shapes selected")
    try:
        validate_shapes_for_providers(sizes, args.version, providers)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    precompile_shapes(args, version_dir, providers, sizes)

    device = triton.runtime.driver.active.get_active_torch_device()

    if args.timing_mode == "triton":
        timing_summary = f"triton median, {args.warmup}ms warmup/{args.rep}ms timed"
    else:
        timing_summary = (f"batched median, {args.timing_repeats}x"
                          f"{args.timed_launches} timed launches, "
                          f"{args.warmup_launches} warmups/repeat")
    print(
        f"\n{version_dir} ({args.b_layout} B, input={args.input_mode}, seed={args.seed}; "
        f"{timing_summary}):"
    )
    header = f"{'M':>6s} {'N':>6s} {'K':>6s}"
    for provider in providers:
        label = PROVIDER_LABELS[provider]
        header += f"  {label:>17s}"
    if "tlx" in providers and "wave" in providers:
        header += f"  {'Wave/TLX':>9s}"
    print(header)

    for M, N, K in sizes:
        a, b = make_inputs(
            M,
            N,
            K,
            device,
            args.b_layout,
            input_mode=args.input_mode,
            seed=args.seed,
        )
        ref = torch.matmul(a, b)
        torch.cuda.synchronize()

        row = f"{M:6d} {N:6d} {K:6d}"
        results = {}
        for provider in providers:
            try:
                result = benchmark_provider(args, provider, version_dir, a, b, ref, M, N, K)
            except Exception as exc:
                result = {
                    "ok": False,
                    "max_err": None,
                    "bad": None,
                    "ms": None,
                    "tflops": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            results[provider] = result
            if result["ok"]:
                row += f"  {result['tflops']:8.1f}T/{result['ms']:6.3f}ms"
            else:
                row += f"  {'FAIL':>17s}"
                if "error" in result:
                    print(f"[{PROVIDER_LABELS[provider]}] M={M} N={N} K={K} failed: "
                          f"{result['error']}")
                else:
                    print(f"[{PROVIDER_LABELS[provider]}] M={M} N={N} K={K} failed "
                          f"correctness: max_err={result['max_err']}, bad={result['bad']}")
        if ("tlx" in results and "wave" in results and results["tlx"]["ok"] and results["wave"]["ok"]):
            ratio = results["wave"]["tflops"] / results["tlx"]["tflops"]
            row += f"  {ratio:8.3f}x"
        print(row, flush=True)


if __name__ == "__main__":
    main()
