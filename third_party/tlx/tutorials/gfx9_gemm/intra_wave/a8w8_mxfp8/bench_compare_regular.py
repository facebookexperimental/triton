"""Compare regular (not grouped) MXFP8 GEMMs on gfx950 square problems.

TLX, AITER Triton, and Primus-Turbo FlyDSL all compute A @ B.T from E4M3
operands and per-row, per-32 E8M0 scales. Report TLX both with cached scales
and with scale packing on every call; Primus includes its fused scale packing.
"""

import argparse
import gc
import importlib.metadata
import json
import os
import statistics
import subprocess
import warnings

import torch

from triton.language.extra.tlx.tutorials.gfx9_gemm.intra_wave.a8w8_mxfp8.bench_upstream import regular_inputs
from triton.language.extra.tlx.tutorials.gfx9_gemm.intra_wave.a8w8_mxfp8.regular_upstream_kernel import (
    pack_regular_mxfp8_scales,
    regular_mxfp8_square,
)


SIZES = (1024, 2048, 4096, 8192, 16384)


def _capture(fn, unroll):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(unroll):
            fn()
    torch.cuda.synchronize()
    return graph


def _time_graph(graph, iterations, unroll):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / (iterations * unroll)


def _check(name, actual, a, b, a_scale, b_scale, size):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(41)
    if size <= 2048:
        rows = cols = torch.arange(size, device="cuda")
    else:
        rows = torch.randperm(size, device="cuda", generator=generator)[:128]
        cols = torch.randperm(size, device="cuda", generator=generator)[:128]
    a_ref = a[rows].float() * torch.exp2(a_scale[rows].float() - 127).repeat_interleave(32, dim=1)
    b_ref = b[cols].float() * torch.exp2(b_scale[cols].float() - 127).repeat_interleave(32, dim=1)
    expected = a_ref @ b_ref.T
    got = actual[rows[:, None], cols[None, :]].float()
    finite = bool(torch.isfinite(got).all())
    max_abs = float((got - expected).abs().max()) if finite else float("inf")
    correct = finite and bool(torch.allclose(got, expected, rtol=0.03, atol=0.5))
    if not correct:
        raise AssertionError(f"{name} size={size}: finite={finite}, max_abs={max_abs}")
    return {"checked_elements": rows.numel() * cols.numel(), "max_abs": max_abs}


def benchmark(size, samples, iterations, unroll):
    from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale_group32 import gemm_a8w8_blockscale_group32
    from primus_turbo.flydsl.gemm.gemm_mxfp8_kernel import gemm_mxfp8_flydsl_kernel

    a, b, a_scale, b_scale = regular_inputs(size)
    packed_a, packed_b = pack_regular_mxfp8_scales(a_scale, b_scale)
    out_tlx = torch.empty((size, size), device="cuda", dtype=torch.bfloat16)
    out_packed = torch.empty_like(out_tlx)
    out_aiter = torch.empty_like(out_tlx)
    out_flydsl = torch.empty_like(out_tlx)
    outputs = {}

    def tlx_cached():
        return regular_mxfp8_square(a, b, packed_a, packed_b, out=out_tlx)

    def tlx_pack_each_call():
        sa, sb = pack_regular_mxfp8_scales(a_scale, b_scale)
        return regular_mxfp8_square(a, b, sa, sb, out=out_packed)

    def aiter_triton():
        return gemm_a8w8_blockscale_group32(a, b, a_scale, b_scale, dtype=torch.bfloat16,
                                            y=out_aiter, weight_group_rows=1)

    def primus_flydsl():
        return gemm_mxfp8_flydsl_kernel(a, a_scale, b, b_scale, out=out_flydsl)

    functions = {
        "tlx_cached": tlx_cached,
        "tlx_pack_each_call": tlx_pack_each_call,
        "aiter_triton": aiter_triton,
        "primus_flydsl": primus_flydsl,
    }
    excluded = {}
    for name, fn in functions.items():
        if name == "primus_flydsl":
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", RuntimeWarning)
                output = fn()
            if any("no reproducible config" in str(w.message) for w in caught):
                excluded[name] = "Primus autotuner found no reproducible configuration"
                continue
            outputs[name] = output
        else:
            outputs[name] = fn()
        torch.cuda.synchronize()
    for name in excluded:
        del functions[name]
    if "primus_flydsl" in functions:
        # The Primus autotuner checks exact output reproducibility after a cold
        # cache flush. Confirm the selected path here before publishing a time.
        first = outputs["primus_flydsl"].clone()
        cache_flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
        for _ in range(3):
            cache_flush.zero_()
            functions["primus_flydsl"]()
            torch.cuda.synchronize()
            if not torch.equal(first, outputs["primus_flydsl"]):
                excluded["primus_flydsl"] = "full output changed across identical cold runs"
                del functions["primus_flydsl"]
                del outputs["primus_flydsl"]
                break
        del first, cache_flush
    checks = {name: _check(name, output, a, b, a_scale, b_scale, size)
              for name, output in outputs.items()}
    graphs = {name: _capture(fn, unroll) for name, fn in functions.items()}
    measurements = {name: [] for name in functions}
    names = list(functions)
    for sample in range(samples):
        # Rotate ordering to reduce systematic clock/temperature bias.
        for name in names[sample % len(names):] + names[:sample % len(names)]:
            measurements[name].append(_time_graph(graphs[name], iterations, unroll))
    for name, graph in graphs.items():
        graph.replay()
        torch.cuda.synchronize()
        checks[name]["post_replay"] = _check(name, outputs[name], a, b, a_scale, b_scale, size)
    timing = {
        name: {
            "ms": statistics.median(values),
            "tflops": 2 * size**3 / (statistics.median(values) * 1e9),
            "samples_ms": values,
        }
        for name, values in measurements.items()
    }
    result = {"size": size, "correctness": checks, "timings": timing, "excluded": excluded}
    del graphs, outputs, a, b, a_scale, b_scale, packed_a, packed_b
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", action="append", type=int, choices=SIZES)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--unroll", type=int, default=2)
    args = parser.parse_args()
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                           stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = os.environ.get("TLX_REVISION", "unknown")
    print(json.dumps({
        "kind": "environment", "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__, "tlx_revision": revision,
        "aiter_version": importlib.metadata.version("amd-aiter"),
        "flydsl_version": importlib.metadata.version("flydsl"),
        "primus_version": importlib.metadata.version("primus-turbo"),
        "dtype": "E4M3 x E4M3, E8M0 1x32 A/B, BF16 output",
        "metric": "CUDA/HIP graph replay median operation time; output buffers preallocated",
        "scope": "TLX cached=GEMM only; TLX pack=two scale packs+GEMM; AITER=raw-scale Triton; Primus=raw-scale fused preshuffle+GEMM",
        "clock_policy": "not fixed",
        "aiter_tuning": "package defaults (no square-shape group32 config rows)",
        "primus_tuning": "first-call autotuned per shape",
        "samples": args.samples, "iterations": args.iterations, "unroll": args.unroll,
    }), flush=True)
    for size in args.size or SIZES:
        print(json.dumps({"kind": "result", **benchmark(size, args.samples, args.iterations, args.unroll)}),
              flush=True)


if __name__ == "__main__":
    main()
