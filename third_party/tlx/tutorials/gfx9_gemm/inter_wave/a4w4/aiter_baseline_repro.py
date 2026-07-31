#!/usr/bin/env python3
"""Benchmark the TLX MXFP4 (A4W4) GEMM against the AITER baseline on gfx950.

The **baseline is AITER** (`aiter.ops.gemm_op_a4w4.gemm_a4w4`, its tuned assembly
`_ZN...` kernel with pre-shuffled B). This script runs the *unchanged* public TLX
A4W4 matmul next to that AITER baseline on the same inputs, checks both against a
FP32-dequantized reference, and reports TLX latency relative to the AITER
baseline. It is fully self-contained: no external product, serving, or framework
dependencies.

We are sharing this to ask for help closing the gap to the AITER baseline: on
large shapes the TLX kernel is materially slower, and we would like guidance on
what the AITER assembly kernels do that a Triton/TLX kernel can replicate (MFMA
selection, pipeline depth, B pre-shuffle, scheduling).

Requirements: a single gfx950 (CDNA4) GPU, ROCm PyTorch, Triton with TLX, and an
AITER build that has tuned A4W4 configs for the benchmarked shapes.

Example:

    HIP_VISIBLE_DEVICES=0 python \
      third_party/tlx/tutorials/gfx9_gemm/inter_wave/a4w4/aiter_baseline_repro.py \
      --tlx-inter-wave-source third_party/tlx/tutorials/gfx9_gemm/inter_wave/a4w4/matmul_kernel.py \
      --tlx-intra-wave-source third_party/tlx/tutorials/gfx9_gemm/intra_wave/a4w4/matmul_kernel.py \
      --json /tmp/a4w4_vs_aiter.json

Use --tlx-inter-wave-source to load matmul_kernel.py directly from a checkout. The source
hash is checked by default so the kernel under test cannot silently drift; pass
--allow-source-mismatch to benchmark a modified kernel.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Protocol

import torch
import triton

# Kernel under test: the in-repo inter_wave matmul_kernel.py shipped in this change
# (co-located under this tutorials tree, including the skinny-gate dispatch tweak
# that routes grid_256 <= NUM_CU//4 to the 128-tile split-K path). EXPECTED_TLX_SHA256
# pins that exact file so the benchmarked kernel cannot silently drift; pass
# --allow-source-mismatch to benchmark a modified copy. AITER is the baseline.
EXPECTED_TLX_SHA256 = "5808240cc31ce9c30238678cfbad437eedae56745a2a830342b788e0f93f350d"
TLX_MODULE = (
    "triton.language.extra.tlx.tutorials.gfx9_gemm.inter_wave.a4w4.matmul_kernel"
)
# Second TLX variant: the 4-wave intra_wave kernel (same public matmul() ABI).
# Not hash-pinned, so it can track the checked-in intra_wave kernel.
TLX_INTRA_MODULE = (
    "triton.language.extra.tlx.tutorials.gfx9_gemm.intra_wave.a4w4.matmul_kernel"
)

# AITER baseline: the public gemm_a4w4 assembly path with pre-shuffled B.
AITER_BASE_COMMIT = "3ccc08d20c465c2617b393177d45405245032528"
AITER_BASE_URL = f"https://github.com/ROCm/aiter/commit/{AITER_BASE_COMMIT}"

SCALE_GROUP_SIZE = 32
DEFAULT_SHAPES = (
    # small / occupancy-starved shapes
    (256, 4096, 4096),
    (256, 8192, 4096),
    (512, 4096, 4096),
    (512, 8192, 4096),
    # large / well-filled shapes (the gap to the AITER baseline is largest here)
    (2048, 4096, 8192),
    (2048, 8192, 4096),
    (2048, 8192, 8192),
)


@dataclass(frozen=True)
class Timing:
    median_us: float
    mad_us: float
    p10_us: float
    p90_us: float
    mean_us: float
    stdev_us: float
    cv_percent: float
    minimum_us: float
    maximum_us: float
    samples: int


class Runner(Protocol):
    name: str

    def __call__(self) -> torch.Tensor: ...


class L2Flusher:
    def __init__(self, mib: int) -> None:
        self.buffer = torch.zeros(
            mib * 1024 * 1024 // 4,
            dtype=torch.int32,
            device="cuda",
        )

    def flush(self) -> None:
        self.buffer.add_(1)


class TlxRunner:
    def __init__(self, name, module, a, b, a_scales, b_scales) -> None:
        self.name = name
        self.module = module
        self.a = a
        self.b = b
        self.a_scales = a_scales
        self.b_scales = b_scales

    def __call__(self) -> torch.Tensor:
        return self.module.matmul(
            self.a,
            self.b,
            self.a_scales,
            self.b_scales,
        )


class AiterBaselineRunner:
    name = "aiter_baseline_gemm_a4w4"

    def __init__(self, a, b, a_scales, b_scales) -> None:
        from aiter.ops.gemm_op_a4w4 import gemm_a4w4, get_GEMM_config
        from aiter.ops.shuffle import shuffle_scale, shuffle_weight

        self.gemm = gemm_a4w4
        self.a = a
        self.b = shuffle_weight(b)
        self.a_scales = shuffle_scale(a_scales.contiguous())
        self.b_scales = shuffle_scale(b_scales.contiguous())
        self.config = get_GEMM_config(a.shape[0], b.shape[0], a.shape[1] * 2)
        if self.config is None:
            raise RuntimeError(
                "AITER baseline has no tuned configuration for "
                f"M={a.shape[0]}, N={b.shape[0]}, K={a.shape[1] * 2}"
            )
        if "_ZN" not in self.config["kernelName"]:
            raise RuntimeError(
                f"Expected a tuned AITER assembly kernel, got {self.config}"
            )

    def __call__(self) -> torch.Tensor:
        return self.gemm(
            self.a,
            self.b,
            self.a_scales,
            self.b_scales,
            bpreshuffle=True,
        )


def parse_shape(raw: str) -> tuple[int, int, int]:
    pieces = raw.lower().replace("x", ",").split(",")
    if len(pieces) != 3:
        raise argparse.ArgumentTypeError(f"Expected MxNxK, got {raw!r}")
    shape = tuple(int(piece) for piece in pieces)
    if any(value <= 0 for value in shape):
        raise argparse.ArgumentTypeError(f"Shape must be positive, got {shape}")
    return shape


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark the TLX A4W4 matmul against the AITER baseline."
    )
    parser.add_argument(
        "--shape",
        action="append",
        type=parse_shape,
        dest="shapes",
        help="M x N x K shape; repeat for multiple shapes",
    )
    parser.add_argument(
        "--tlx-inter-wave-source",
        type=Path,
        help="load the inter_wave matmul_kernel.py directly instead of the installed module",
    )
    parser.add_argument(
        "--tlx-intra-wave-source",
        type=Path,
        help="load the intra_wave matmul_kernel.py to also benchmark it (optional)",
    )
    parser.add_argument(
        "--no-intra",
        action="store_true",
        help="skip the intra_wave variant even if it is available",
    )
    parser.add_argument(
        "--allow-source-mismatch",
        action="store_true",
        help="allow a TLX source hash other than the pinned kernel under test",
    )
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--timing-rounds", type=int, default=3)
    parser.add_argument("--flush-mib", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    args.shapes = args.shapes or list(DEFAULT_SHAPES)
    for name in ("warmups", "samples", "timing_rounds", "flush_mib"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def package_version(*names: str) -> str:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


def runtime_metadata() -> dict:
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("This repro requires ROCm PyTorch")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Expose exactly one GPU with HIP_VISIBLE_DEVICES; got "
            f"{torch.cuda.device_count()} visible GPUs"
        )
    properties = torch.cuda.get_device_properties(0)
    if not properties.gcnArchName.startswith("gfx950"):
        raise RuntimeError(
            f"The kernel under test requires gfx950, got {properties.gcnArchName}"
        )
    metadata = {
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        "device_name": properties.name,
        "arch": properties.gcnArchName,
        "compute_units": properties.multi_processor_count,
        "vram_bytes": properties.total_memory,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "triton": triton.__version__,
        "aiter": package_version("amd-aiter", "aiter"),
        "python": sys.version.split()[0],
        "argv": sys.argv,
    }
    print(f"runtime={json.dumps(metadata, sort_keys=True)}", flush=True)
    return metadata


def load_tlx_module(
    source_arg: Path | None,
    allow_source_mismatch: bool,
    module_name: str,
    expected_sha256: str | None,
    unique_name: str,
) -> tuple[ModuleType, Path, str]:
    if source_arg is None:
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"Could not find installed TLX module {module_name}")
        source = Path(spec.origin).resolve()
    else:
        source = source_arg.resolve()
        if not source.is_file():
            raise FileNotFoundError(source)

    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if (
        expected_sha256 is not None
        and digest != expected_sha256
        and not allow_source_mismatch
    ):
        raise RuntimeError(
            f"TLX source SHA256 mismatch: expected {expected_sha256}, "
            f"got {digest} at {source}. Use --allow-source-mismatch only when "
            "intentionally testing a modified kernel."
        )

    if source_arg is None:
        module = importlib.import_module(module_name)
    else:
        spec = importlib.util.spec_from_file_location(unique_name, source)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load TLX source {source}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

    print(f"tlx_source={source} sha256={digest} expected={expected_sha256}", flush=True)
    return module, source, digest


def mxfp4_to_f32(packed: torch.Tensor) -> torch.Tensor:
    unpacked = packed.repeat_interleave(2, dim=1)
    unpacked[:, ::2] &= 0xF
    unpacked[:, 1::2] >>= 4
    values = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
        device=packed.device,
    )
    return values[unpacked.long()]


def e8m0_to_f32(scales: torch.Tensor) -> torch.Tensor:
    exponent = scales.to(torch.int16).to(torch.float32) - 127.0
    return torch.pow(2.0, exponent)


def make_inputs(m: int, n: int, k: int, seed: int):
    if k % (2 * SCALE_GROUP_SIZE) != 0:
        raise ValueError(f"K must be divisible by {2 * SCALE_GROUP_SIZE}, got {k}")
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed + m * 17 + n)
    a = torch.randint(
        0,
        256,
        (m, k // 2),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    b = torch.randint(
        0,
        256,
        (n, k // 2),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )

    # TLX requires row scales to be contiguous in the M/N dimension.
    padded_m = triton.cdiv(m, 256) * 256
    a_scales = torch.randint(
        124,
        128,
        (k // SCALE_GROUP_SIZE, padded_m),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    ).T[:m]
    b_scales = torch.randint(
        124,
        128,
        (k // SCALE_GROUP_SIZE, n),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    ).T
    if a_scales.stride(0) != 1 or b_scales.stride(0) != 1:
        raise RuntimeError("TLX scale layout construction failed")
    return a, b, a_scales, b_scales


def reference(a, b, a_scales, b_scales) -> torch.Tensor:
    a_f32 = mxfp4_to_f32(a)
    a_f32 *= e8m0_to_f32(a_scales).repeat_interleave(
        SCALE_GROUP_SIZE,
        dim=1,
    )
    b_f32 = mxfp4_to_f32(b)
    b_f32 *= e8m0_to_f32(b_scales).repeat_interleave(
        SCALE_GROUP_SIZE,
        dim=1,
    )
    output = torch.mm(a_f32, b_f32.T).to(torch.bfloat16)
    del a_f32, b_f32
    return output


def validate_runner(runner: Runner, expected: torch.Tensor) -> dict:
    first = runner().clone()
    second = runner().clone()
    torch.cuda.synchronize()
    deterministic = torch.equal(first, second)
    exact = torch.equal(first, expected)
    difference = first.float() - expected.float()
    metrics = {
        "deterministic": deterministic,
        "exact": exact,
        "max_abs_error": float(difference.abs().max().item()),
        "mean_abs_error": float(difference.abs().mean().item()),
        "nan_count": int(torch.isnan(first).sum().item()),
        "inf_count": int(torch.isinf(first).sum().item()),
    }
    del first, second, difference
    if not deterministic:
        raise RuntimeError(f"{runner.name} is not bitwise deterministic")
    if not exact:
        raise RuntimeError(
            f"{runner.name} does not exactly match the BF16 reference: {metrics}"
        )
    print(f"correctness name={runner.name} {json.dumps(metrics, sort_keys=True)}")
    return metrics


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def summarize(values: list[float]) -> Timing:
    median = statistics.median(values)
    mean = statistics.mean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return Timing(
        median_us=median,
        mad_us=statistics.median(abs(value - median) for value in values),
        p10_us=percentile(values, 0.10),
        p90_us=percentile(values, 0.90),
        mean_us=mean,
        stdev_us=stdev,
        cv_percent=100.0 * stdev / mean,
        minimum_us=min(values),
        maximum_us=max(values),
        samples=len(values),
    )


def benchmark_interleaved(
    runners: list[Runner],
    flusher: L2Flusher,
    warmups: int,
    samples: int,
    rounds: int,
) -> dict[str, dict]:
    for runner in runners:
        for _ in range(warmups):
            output = runner()
            del output
    torch.cuda.synchronize()

    all_samples = {runner.name: [] for runner in runners}
    round_medians = {runner.name: [] for runner in runners}
    for round_index in range(rounds):
        round_samples = {runner.name: [] for runner in runners}
        for sample_index in range(samples):
            order = runners if (round_index + sample_index) % 2 == 0 else runners[::-1]
            for runner in order:
                flusher.flush()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                output = runner()
                end.record()
                end.synchronize()
                elapsed_us = start.elapsed_time(end) * 1000.0
                del output
                round_samples[runner.name].append(elapsed_us)
                all_samples[runner.name].append(elapsed_us)
        for runner in runners:
            round_medians[runner.name].append(
                statistics.median(round_samples[runner.name])
            )

    return {
        runner.name: {
            "timing": asdict(summarize(all_samples[runner.name])),
            "round_medians_us": round_medians[runner.name],
        }
        for runner in runners
    }


def tlx_dispatch(module, m: int, n: int) -> dict:
    # Best-effort dispatch prediction; intra_wave may not expose these attrs.
    block_m = getattr(module, "BLOCK_M", 256)
    block_n = getattr(module, "BLOCK_N", 256)
    grid_256 = triton.cdiv(m, block_m) * triton.cdiv(n, block_n)
    info = {"grid_256": grid_256}
    num_cu = getattr(module, "NUM_CU", None)
    if num_cu is not None and hasattr(module, "SKINNY_BLOCK_M"):
        skinny_threshold = num_cu // 4
        info["skinny_threshold"] = skinny_threshold
        info["expected_selected_dispatch"] = (
            "skinny" if grid_256 <= skinny_threshold else "256tile"
        )
    else:
        info["expected_selected_dispatch"] = "256tile"
    return info


def run_shape(tlx_variants, shape, args, flusher) -> dict:
    m, n, k = shape
    print(f"\nshape={m}x{n}x{k}", flush=True)
    a, b, a_scales, b_scales = make_inputs(m, n, k, args.seed)
    try:
        aiter_runner = AiterBaselineRunner(a, b, a_scales, b_scales)
    except RuntimeError as exc:
        # This AITER build has no tuned A4W4 baseline for this shape; skip it
        # rather than abort the whole sweep. Try a build with more tuned configs
        # to benchmark these shapes against AITER.
        print(f"skip shape={m}x{n}x{k} reason={exc}", flush=True)
        del a, b, a_scales, b_scales
        gc.collect()
        torch.cuda.empty_cache()
        return {"shape": list(shape), "skipped": str(exc)}

    tlx_runners = [
        TlxRunner(name, module, a, b, a_scales, b_scales)
        for name, module in tlx_variants
    ]
    dispatch = {name: tlx_dispatch(module, m, n) for name, module in tlx_variants}
    # AITER first: it is the baseline every TLX number is reported against.
    runners: list[Runner] = [aiter_runner, *tlx_runners]

    expected = reference(a, b, a_scales, b_scales)
    torch.cuda.synchronize()
    correctness = {runner.name: validate_runner(runner, expected) for runner in runners}
    del expected
    torch.cuda.empty_cache()

    timings = benchmark_interleaved(
        runners,
        flusher,
        args.warmups,
        args.samples,
        args.timing_rounds,
    )
    aiter_us = timings[aiter_runner.name]["timing"]["median_us"]
    # speedup = AITER / TLX  ->  > 1.0 means TLX is FASTER (a win); < 1.0 is a regression.
    speedup = {
        r.name: aiter_us / timings[r.name]["timing"]["median_us"] for r in tlx_runners
    }
    result = {
        "shape": list(shape),
        "aiter_baseline_us": aiter_us,
        "aiter_baseline_config": aiter_runner.config,
        "tlx_dispatch": dispatch,
        "correctness": correctness,
        "timings": timings,
        "tlx_speedup_over_aiter_baseline": speedup,
    }
    print(f"result={json.dumps(result, sort_keys=True)}", flush=True)

    del runners, tlx_runners, aiter_runner, a, b, a_scales, b_scales
    gc.collect()
    torch.cuda.empty_cache()
    return result


def print_table(results: list[dict], variant_names: list[str]) -> None:
    print("\nTLX vs AITER baseline (cold L2, GPU-event median)")
    print("Speedup = AITER / TLX:  > 1.0 means TLX is FASTER (win); < 1.0 is a regression.")
    header = f"{'M x N x K':>18} {'AITER us':>9}"
    for name in variant_names:
        header += f" {name + ' us':>{len(name) + 4}} {name + ' x':>{len(name) + 3}}"
    print(header)
    for result in results:
        shape = "x".join(str(value) for value in result["shape"])
        if "skipped" in result:
            print(f"{shape:>18} {'skipped (no AITER baseline config)':>44}")
            continue
        aiter_us = result["aiter_baseline_us"]
        row = f"{shape:>18} {aiter_us:9.2f}"
        for name in variant_names:
            uw, xw = len(name) + 4, len(name) + 3
            if name in result["timings"]:
                tlx_us = result["timings"][name]["timing"]["median_us"]
                spd = result["tlx_speedup_over_aiter_baseline"][name]
                row += f" {tlx_us:>{uw}.2f} {spd:>{xw}.3f}"
            else:
                row += f" {'--':>{uw}} {'--':>{xw}}"
        print(row)


def main() -> None:
    args = parse_args()
    runtime = runtime_metadata()

    # inter_wave: the pinned, hash-checked primary kernel under test.
    inter_module, inter_source, inter_digest = load_tlx_module(
        args.tlx_inter_wave_source,
        args.allow_source_mismatch,
        TLX_MODULE,
        EXPECTED_TLX_SHA256,
        "tlx_a4w4_inter_kernel",
    )
    tlx_variants = [("inter_wave_TLX", inter_module)]
    kernels_under_test = {
        "inter_wave_TLX": {
            "path": str(inter_source),
            "sha256": inter_digest,
            "expected_sha256": EXPECTED_TLX_SHA256,
        }
    }
    # intra_wave: optional second variant (not hash-pinned).
    if not args.no_intra:
        try:
            intra_module, intra_source, intra_digest = load_tlx_module(
                args.tlx_intra_wave_source,
                True,  # no pinned hash for intra_wave
                TLX_INTRA_MODULE,
                None,
                "tlx_a4w4_intra_kernel",
            )
            tlx_variants.append(("intra_wave_TLX", intra_module))
            kernels_under_test["intra_wave_TLX"] = {
                "path": str(intra_source),
                "sha256": intra_digest,
            }
        except Exception as exc:  # noqa: BLE001 - optional variant
            print(f"intra_wave variant unavailable: {exc}", flush=True)

    variant_names = [name for name, _ in tlx_variants]
    protocol = {
        "baseline": "AITER gemm_a4w4 (tuned assembly, pre-shuffled B)",
        "warmups": args.warmups,
        "samples_per_round": args.samples,
        "timing_rounds": args.timing_rounds,
        "flush_mib_per_sample": args.flush_mib,
        "seed": args.seed,
        "timing": "GPU events around unchanged public API calls",
        "runner_order": "alternating",
        "correctness": "exact BF16 equality against FP32-dequantized torch.mm",
        "layout_conversion_timed": False,
        "output_allocation_timed": True,
        "input_layouts": {
            "tlx": "raw packed A/B; scales contiguous in M/N dimension",
            "aiter": "raw A; pre-shuffled B and A/B scales",
        },
        "scope": (
            "public-wrapper device elapsed time, not preallocated kernel-only latency"
        ),
    }
    print(f"protocol={json.dumps(protocol, sort_keys=True)}", flush=True)
    flusher = L2Flusher(args.flush_mib)
    results = [run_shape(tlx_variants, shape, args, flusher) for shape in args.shapes]
    payload = {
        "runtime": runtime,
        "tlx_kernels_under_test": kernels_under_test,
        "protocol": protocol,
        "aiter_baseline": {
            "api": "aiter.ops.gemm_op_a4w4.gemm_a4w4",
            "installed_build": runtime["aiter"],
            "public_base_commit": AITER_BASE_COMMIT,
            "public_base_url": AITER_BASE_URL,
            "note": "Absolute latency can differ with another AITER build.",
        },
        "results": results,
    }
    print_table(results, variant_names)
    print(f"\nsummary={json.dumps(payload, sort_keys=True)}", flush=True)
    if args.json is not None:
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
