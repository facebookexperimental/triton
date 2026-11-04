#!/usr/bin/env python3
"""Reproduce production gfx950 TLX MXFP8 attention versus AITER ASM.

The public BF16 API is packed varlen, H4/D128, non-causal, dropout-free.
Internally TLX prepares E4M3 payloads and E8M0 scales before its scaled-MFMA
ping-pong core. The default portfolio contains all 20 unique cross-attention
aggregate shapes observed in the paired control and candidate production
traces: ten short-KV and ten long-KV shapes, each called three times in its
trace window. The traces do not retain offset values, so every case uses a
deterministic long-tail sequence distribution with the same batch, totals, and
maxima.

Example from the fbsource root:

  HIP_VISIBLE_DEVICES=0 \
  third-party/triton/beta/triton/third_party/tlx/denoise.sh python \
    third-party/triton/beta/triton/third_party/tlx/tutorials/\
amd_mxfp8_varlen_attention/aiter_baseline_repro.py \
    --json /tmp/mxfp8_long_kv_vs_aiter.json

Use ``--allow-source-mismatch`` only when intentionally tuning a modified
kernel. The two expected SHA256 values pin the exact implementation under test.
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

EXPECTED_CORE_SHA256 = "bff4a76cb97ca346fdba5793b79ecf0eb289eba050c3f48e161197f8e1b4ee46"
EXPECTED_FORWARD_SHA256 = "baf23510224bd884aa5a12ff6a4d747e416e3d7981a33c9d2f571fdac2f9d849"
EXPECTED_BACKWARD_SHA256 = "7afbf15c487cce00d55c0b1c8a2a01e53f84b6886ad157cbc9afaf939004b594"

BATCH = 432
HEADS = 4
HEAD_DIM = 128
SOFTMAX_SCALE = HEAD_DIM**-0.5


@dataclass(frozen=True)
class Timing:
    median_ms: float
    mad_ms: float
    p10_ms: float
    p90_ms: float
    mean_ms: float
    stdev_ms: float
    cv_percent: float
    minimum_ms: float
    maximum_ms: float
    samples: int


@dataclass(frozen=True)
class ShapeCase:
    name: str
    family: str
    total_q: int
    total_kv: int
    max_q: int
    max_kv: int
    trace_calls: int = 3


PRODUCTION_SHAPES = (
    ShapeCase("control_short_0", "short", 24_688, 264_119, 192, 1_004),
    ShapeCase("control_short_1", "short", 27_248, 264_237, 160, 828),
    ShapeCase("control_short_2", "short", 29_232, 264_686, 240, 748),
    ShapeCase("control_short_3", "short", 31_856, 263_664, 224, 788),
    ShapeCase("control_short_4", "short", 32_736, 263_726, 240, 748),
    ShapeCase("candidate_short_0", "short", 26_672, 259_407, 224, 964),
    ShapeCase("candidate_short_1", "short", 27_200, 264_751, 224, 748),
    ShapeCase("candidate_short_2", "short", 30_576, 264_690, 240, 748),
    ShapeCase("candidate_short_3", "short", 30_928, 264_436, 224, 748),
    ShapeCase("candidate_short_4", "short", 32_784, 264_520, 240, 748),
    ShapeCase("control_long_0", "long", 24_688, 3_135_104, 192, 32_738),
    ShapeCase("control_long_1", "long", 27_248, 3_142_170, 160, 21_144),
    ShapeCase("control_long_2", "long", 29_232, 3_161_784, 240, 16_354),
    ShapeCase("control_long_3", "long", 31_856, 3_146_362, 224, 18_851),
    ShapeCase("control_long_4", "long", 32_736, 3_139_507, 240, 16_354),
    ShapeCase("candidate_long_0", "long", 26_672, 3_117_994, 224, 29_926),
    ShapeCase("candidate_long_1", "long", 27_200, 3_164_011, 224, 16_354),
    ShapeCase("candidate_long_2", "long", 30_576, 3_164_659, 240, 16_354),
    ShapeCase("candidate_long_3", "long", 30_928, 3_180_031, 224, 16_354),
    ShapeCase("candidate_long_4", "long", 32_784, 3_156_939, 240, 16_354),
)


class Runner(Protocol):
    name: str

    def __call__(self) -> object: ...


class L2Flusher:
    def __init__(self, mib: int) -> None:
        self.buffer = torch.zeros(
            mib * 1024 * 1024 // 4,
            dtype=torch.int32,
            device="cuda",
        )

    def flush(self) -> None:
        self.buffer.add_(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    directory = Path(__file__).resolve().parent
    parser.add_argument(
        "--tlx-core-source",
        type=Path,
        default=directory / "mxfp8_attention_core_gfx950.py",
    )
    parser.add_argument(
        "--tlx-forward-source",
        type=Path,
        default=directory / "mxfp8_varlen_pingpong_gfx950.py",
    )
    parser.add_argument(
        "--tlx-backward-source",
        type=Path,
        default=directory / "mxfp8_varlen_backward_gfx950.py",
    )
    parser.add_argument("--allow-source-mismatch", action="store_true")
    parser.add_argument(
        "--allow-correctness-failures",
        action="store_true",
        help="report and time failing cases instead of exiting nonzero after the suite",
    )
    parser.add_argument(
        "--suite",
        choices=("production", "short", "long"),
        default="production",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(case.name for case in PRODUCTION_SHAPES),
        dest="cases",
        help="run one named production shape; repeat for multiple shapes",
    )
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--timing-rounds", type=int, default=3)
    parser.add_argument("--flush-mib", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
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


def _load_source(
    source: Path,
    expected_sha256: str,
    module_name: str,
    allow_source_mismatch: bool,
) -> tuple[ModuleType, str]:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != expected_sha256 and not allow_source_mismatch:
        raise RuntimeError(
            f"TLX source SHA256 mismatch: expected {expected_sha256}, got "
            f"{digest} at {source}. Use --allow-source-mismatch only when "
            "intentionally testing modified sources."
        )
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load TLX source {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    print(
        f"tlx_source={source} sha256={digest} expected={expected_sha256}",
        flush=True,
    )
    return module, digest


def load_tlx_sources(
    args: argparse.Namespace,
) -> tuple[ModuleType, ModuleType, dict]:
    package_name = (
        "triton.language.extra.tlx.tutorials.amd_mxfp8_varlen_attention"
    )
    package = sys.modules.setdefault(package_name, ModuleType(package_name))
    package.__path__ = [str(Path(__file__).resolve().parent)]
    core_name = f"{package_name}.mxfp8_attention_core_gfx950"
    core, core_digest = _load_source(
        args.tlx_core_source,
        EXPECTED_CORE_SHA256,
        core_name,
        args.allow_source_mismatch,
    )
    setattr(package, "mxfp8_attention_core_gfx950", core)
    forward_name = f"{package_name}.mxfp8_varlen_pingpong_gfx950"
    forward, forward_digest = _load_source(
        args.tlx_forward_source,
        EXPECTED_FORWARD_SHA256,
        forward_name,
        args.allow_source_mismatch,
    )
    setattr(package, "mxfp8_varlen_pingpong_gfx950", forward)
    backward_name = f"{package_name}.mxfp8_varlen_backward_gfx950"
    backward, backward_digest = _load_source(
        args.tlx_backward_source,
        EXPECTED_BACKWARD_SHA256,
        backward_name,
        args.allow_source_mismatch,
    )
    return forward, backward, {
        "core": {
            "path": str(args.tlx_core_source.resolve()),
            "sha256": core_digest,
            "expected_sha256": EXPECTED_CORE_SHA256,
        },
        "forward": {
            "path": str(args.tlx_forward_source.resolve()),
            "sha256": forward_digest,
            "expected_sha256": EXPECTED_FORWARD_SHA256,
        },
        "backward": {
            "path": str(args.tlx_backward_source.resolve()),
            "sha256": backward_digest,
            "expected_sha256": EXPECTED_BACKWARD_SHA256,
        },
    }


def lengths(count: int, total: int, maximum: int, seed: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    weights = -torch.log(torch.rand(count, generator=generator).clamp_min(1e-6))
    values = torch.clamp((weights / weights.sum() * total).to(torch.int64), 1, maximum)
    values[0] = maximum
    result = values.tolist()
    delta = total - sum(result)
    while delta:
        if delta > 0:
            for index in range(1, count):
                added = min(maximum - result[index], delta)
                result[index] += added
                delta -= added
                if not delta:
                    break
        else:
            for index in range(1, count):
                removed = min(result[index] - 1, -delta)
                result[index] -= removed
                delta += removed
                if not delta:
                    break
    return result


def cumulative(length_values: list[int]) -> torch.Tensor:
    offsets = torch.zeros(
        len(length_values) + 1,
        dtype=torch.int32,
        device="cuda",
    )
    offsets[1:] = torch.cumsum(
        torch.tensor(length_values, dtype=torch.int32, device="cuda"),
        dim=0,
    )
    return offsets


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
        median_ms=median,
        mad_ms=statistics.median(abs(value - median) for value in values),
        p10_ms=percentile(values, 0.10),
        p90_ms=percentile(values, 0.90),
        mean_ms=mean,
        stdev_ms=stdev,
        cv_percent=100.0 * stdev / mean,
        minimum_ms=min(values),
        maximum_ms=max(values),
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
                elapsed = start.elapsed_time(end)
                del output
                round_samples[runner.name].append(elapsed)
                all_samples[runner.name].append(elapsed)
        for runner in runners:
            round_medians[runner.name].append(
                statistics.median(round_samples[runner.name])
            )
    return {
        runner.name: {
            "timing": asdict(summarize(all_samples[runner.name])),
            "round_medians_ms": round_medians[runner.name],
        }
        for runner in runners
    }


def run_case(
    case: ShapeCase,
    args: argparse.Namespace,
    forward: ModuleType,
    backward: ModuleType,
    aiter: ModuleType,
    flusher: L2Flusher,
) -> dict:
    print(f"\ncase={case.name} family={case.family}", flush=True)
    torch.manual_seed(args.seed)
    q_lengths = [
        16 * value
        for value in lengths(BATCH, case.total_q // 16, case.max_q // 16, seed=3)
    ]
    kv_lengths = lengths(BATCH, case.total_kv, case.max_kv, seed=5)
    cu_q = cumulative(q_lengths)
    cu_kv = cumulative(kv_lengths)
    q = torch.randn(
        (case.total_q, HEADS, HEAD_DIM), device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn(
        (case.total_kv, HEADS, HEAD_DIM), device="cuda", dtype=torch.bfloat16
    )
    v = torch.randn_like(k)
    dout = torch.randn_like(q)
    metadata = forward._metadata(
        cu_q, cu_kv, case.total_q, case.total_kv, 128
    )

    class TlxComplete:
        name = "tlx_complete"

        def __call__(self):
            return forward.forward(
                q,
                k,
                v,
                cu_q,
                cu_kv,
                SOFTMAX_SCALE,
                block_m=128,
            )

    class TlxPrepare:
        name = "tlx_prepare"

        def __call__(self):
            return forward.prepare_kv(k, v, cu_kv, metadata, backward=False)

    state = TlxPrepare()()

    class TlxCore:
        name = "tlx_core"

        def __call__(self):
            return forward.forward(
                q,
                k,
                v,
                cu_q,
                cu_kv,
                SOFTMAX_SCALE,
                kv_state=state,
                block_m=128,
            )

    class AiterForward:
        name = "aiter_forward"

        def __call__(self):
            out, lse, _, _ = aiter.fmha_v3_varlen_fwd(
                q,
                k,
                v,
                cu_q,
                cu_kv,
                case.max_q,
                case.max_kv,
                0,
                0.0,
                SOFTMAX_SCALE,
                0.0,
                False,
                False,
                -1,
                -1,
                True,
                False,
                0,
            )
            return out, lse

    tlx_out, tlx_lse, _ = TlxComplete()()
    aiter_out, aiter_lse = AiterForward()()
    tlx_train_out, tlx_train_lse, tlx_train_state = forward.forward(
        q,
        k,
        v,
        cu_q,
        cu_kv,
        SOFTMAX_SCALE,
        backward_state=True,
        block_m=128,
    )

    class TlxBackward:
        name = "tlx_backward"

        def __call__(self):
            return backward.backward(
                dout,
                q,
                k,
                v,
                tlx_train_out,
                tlx_train_lse,
                cu_q,
                cu_kv,
                SOFTMAX_SCALE,
                kv_state=tlx_train_state,
                kv_shape=k.shape,
                max_seqlen_q=case.max_q,
                max_seqlen_k=case.max_kv,
            )

    class AiterBackward:
        name = "aiter_backward_fp32_atomic"

        def __call__(self):
            dq, dk, dv, _ = aiter.fmha_v3_varlen_bwd(
                dout,
                q,
                k,
                v,
                aiter_out,
                aiter_lse,
                cu_q,
                cu_kv,
                max_seqlen_q=case.max_q,
                max_seqlen_k=case.max_kv,
                dropout_p=0.0,
                softmax_scale=SOFTMAX_SCALE,
                zero_tensors=False,
                is_causal=False,
                window_size_left=-1,
                window_size_right=-1,
                deterministic=False,
                is_v3_atomic_fp32=True,
                how_v3_bf16_cvt=1,
            )
            return dq, dk, dv

    if not all(
        bool(torch.isfinite(tensor).all())
        for tensor in (tlx_out, tlx_lse, aiter_out, aiter_lse)
    ):
        raise RuntimeError(f"{case.name}: non-finite attention output")
    output_error = float(
        (tlx_out.float() - aiter_out.float()).norm()
        / aiter_out.float().norm().clamp_min(1e-30)
    )
    lse_error = float((tlx_lse - aiter_lse).abs().max())
    tlx_grads = TlxBackward()()
    aiter_grads = AiterBackward()()
    if not all(
        bool(torch.isfinite(tensor).all())
        for tensor in (*tlx_grads, *aiter_grads)
    ):
        raise RuntimeError(f"{case.name}: non-finite attention gradient")
    gradient_errors = {
        name: float(
            (actual.float() - expected.float()).norm()
            / expected.float().norm().clamp_min(1e-30)
        )
        for name, actual, expected in zip(
            ("dq", "dk", "dv"), tlx_grads, aiter_grads
        )
    }
    passed = (
        output_error < 0.08
        and lse_error < 0.10
        and all(error < 0.10 for error in gradient_errors.values())
    )
    correctness = {
        "passed": passed,
        "limits": {
            "output_relative_l2": 0.08,
            "lse_max_abs": 0.10,
            "gradient_relative_l2": 0.10,
        },
        "output_relative_l2": output_error,
        "lse_max_abs": lse_error,
        "gradient_relative_l2": gradient_errors,
    }
    print(f"correctness={json.dumps(correctness, sort_keys=True)}", flush=True)

    runners: list[Runner] = [
        AiterForward(),
        TlxComplete(),
        TlxPrepare(),
        TlxCore(),
        AiterBackward(),
        TlxBackward(),
    ]
    timings = benchmark_interleaved(
        runners,
        flusher,
        args.warmups,
        args.samples,
        args.timing_rounds,
    )
    aiter_forward_ms = timings["aiter_forward"]["timing"]["median_ms"]
    tlx_complete_ms = timings["tlx_complete"]["timing"]["median_ms"]
    tlx_core_ms = timings["tlx_core"]["timing"]["median_ms"]
    aiter_backward_ms = timings["aiter_backward_fp32_atomic"]["timing"][
        "median_ms"
    ]
    tlx_backward_ms = timings["tlx_backward"]["timing"]["median_ms"]
    result = {
        "case": asdict(case),
        "correctness": correctness,
        "timings": timings,
        "aiter_over_tlx_complete": aiter_forward_ms / tlx_complete_ms,
        "aiter_over_tlx_core": aiter_forward_ms / tlx_core_ms,
        "aiter_over_tlx_backward": aiter_backward_ms / tlx_backward_ms,
    }
    print(
        f"result case={case.name} AITER-forward={aiter_forward_ms:.3f} ms "
        f"TLX-complete={tlx_complete_ms:.3f} ms "
        f"TLX-prepare={timings['tlx_prepare']['timing']['median_ms']:.3f} ms "
        f"TLX-core={tlx_core_ms:.3f} ms "
        f"AITER/TLX-complete={aiter_forward_ms / tlx_complete_ms:.3f}x "
        f"AITER/TLX-core={aiter_forward_ms / tlx_core_ms:.3f}x "
        f"AITER-backward={aiter_backward_ms:.3f} ms "
        f"TLX-backward={tlx_backward_ms:.3f} ms "
        f"AITER/TLX-backward={aiter_backward_ms / tlx_backward_ms:.3f}x",
        flush=True,
    )
    del (
        state,
        tlx_train_state,
        q,
        k,
        v,
        dout,
        tlx_out,
        tlx_lse,
        tlx_train_out,
        tlx_train_lse,
        aiter_out,
        aiter_lse,
        tlx_grads,
        aiter_grads,
    )
    gc.collect()
    torch.cuda.empty_cache()
    return result


def geometric_mean(values: list[float]) -> float:
    return math.exp(statistics.mean(math.log(value) for value in values))


def main() -> None:
    args = parse_args()
    runtime = runtime_metadata()
    forward, backward, sources = load_tlx_sources(args)
    aiter = importlib.import_module("aiter_")
    if args.cases:
        by_name = {case.name: case for case in PRODUCTION_SHAPES}
        selected = [by_name[name] for name in args.cases]
    else:
        selected = [
            case
            for case in PRODUCTION_SHAPES
            if args.suite == "production" or case.family == args.suite
        ]
    protocol = {
        "portfolio": [asdict(case) for case in selected],
        "semantics": {
            "dtype": "bfloat16",
            "packed_varlen": True,
            "causal": False,
            "dropout_p": 0.0,
            "window": [-1, -1],
            "softmax_scale": SOFTMAX_SCALE,
            "sequence_distribution": "deterministic long-tail proxy",
        },
        "implementations": {
            "tlx_prepare": "_prepare_kv",
            "tlx_core": "_mxfp8_varlen_kernel",
            "tlx_backward": "_prepare_q + _dq_kernel + _dkdv_stream_kernel",
            "aiter_forward": "fmha_fwd_hd128_bf16_group",
            "aiter_backward": "fmha_bwd_hd128_bf16_a32_psskddv_group with FP32 atomics",
        },
        "warmups": args.warmups,
        "samples_per_round": args.samples,
        "timing_rounds": args.timing_rounds,
        "flush_mib_per_sample": args.flush_mib,
        "runner_order": "alternating",
        "timing": "cold-L2 GPU-event device elapsed time",
        "output_allocation_timed": True,
    }
    print(f"protocol={json.dumps(protocol, sort_keys=True)}", flush=True)
    flusher = L2Flusher(args.flush_mib)
    results = [
        run_case(case, args, forward, backward, aiter, flusher)
        for case in selected
    ]
    forward_ratios = [result["aiter_over_tlx_complete"] for result in results]
    core_ratios = [result["aiter_over_tlx_core"] for result in results]
    backward_ratios = [result["aiter_over_tlx_backward"] for result in results]
    summary = {
        "runtime": runtime,
        "tlx_sources": sources,
        "protocol": protocol,
        "results": results,
        "portfolio": {
            "cases": len(results),
            "correctness_failures": [
                result["case"]["name"]
                for result in results
                if not result["correctness"]["passed"]
            ],
            "forward_geomean": geometric_mean(forward_ratios),
            "forward_worst": min(forward_ratios),
            "core_geomean": geometric_mean(core_ratios),
            "backward_geomean": geometric_mean(backward_ratios),
            "backward_worst": min(backward_ratios),
        },
    }
    print(f"\nsummary={json.dumps(summary, sort_keys=True)}", flush=True)
    print(
        f"portfolio cases={len(results)} "
        f"AITER/TLX-complete-geomean={summary['portfolio']['forward_geomean']:.3f}x "
        f"AITER/TLX-complete-worst={summary['portfolio']['forward_worst']:.3f}x "
        f"AITER/TLX-core-geomean={summary['portfolio']['core_geomean']:.3f}x "
        f"AITER/TLX-backward-geomean={summary['portfolio']['backward_geomean']:.3f}x "
        f"AITER/TLX-backward-worst={summary['portfolio']['backward_worst']:.3f}x"
    )
    if args.json is not None:
        args.json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    failures = summary["portfolio"]["correctness_failures"]
    if failures and not args.allow_correctness_failures:
        raise RuntimeError(
            f"{len(failures)} production cases failed correctness: {failures}"
        )


if __name__ == "__main__":
    main()
