"""Correctness and kernel-only timing for the regular MXFP8 GEMM on current TLX."""

import argparse
import json
import re
import statistics

import torch

from triton.language.extra.tlx.tutorials.gfx9_gemm.intra_wave.a8w8_mxfp8.regular_upstream_kernel import (
    pack_regular_mxfp8_scales,
    regular_mxfp8_generic,
    regular_mxfp8_lds_scales,
    regular_mxfp8_square,
)


def regular_inputs(size, seed=42):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    a = torch.randint(-2, 3, (size, size), dtype=torch.int8, device="cuda", generator=generator).to(
        torch.float8_e4m3fn)
    b = torch.randint(-2, 3, (size, size), dtype=torch.int8, device="cuda", generator=generator).to(
        torch.float8_e4m3fn)
    a_scales = torch.randint(125, 128, (size, size // 32), dtype=torch.uint8,
                             device="cuda", generator=generator)
    b_scales = torch.randint(125, 128, (size, size // 32), dtype=torch.uint8,
                             device="cuda", generator=generator)
    return a, b, a_scales, b_scales


def _time_batch(fn, iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def run(size: int, samples: int, iterations: int, unroll: int, variant: str,
        bm: int, bn: int, stages: int, warps: int, xcd_chunk: int = 0) -> dict:
    a, b, raw_sa, raw_sb = regular_inputs(size)
    sa, sb = pack_regular_mxfp8_scales(raw_sa, raw_sb)
    out = torch.empty((size, size), device="cuda", dtype=torch.bfloat16)

    def launch(capture=False):
        if variant == "auto":
            return regular_mxfp8_square(a, b, sa, sb, out=out, return_kernel=capture)
        if variant == "lds":
            return regular_mxfp8_lds_scales(a, b, sa, sb, out=out,
                                            return_kernel=capture, xcd_chunk=xcd_chunk)
        if variant == "generic":
            return regular_mxfp8_generic(a, b, sa, sb, out=out, return_kernel=capture,
                                         bm=bm, bn=bn, stages=stages, warps=warps)
        raise ValueError(f"unknown variant: {variant}")

    _, kernel = launch(capture=True)
    torch.cuda.synchronize()

    # Check full output at 1K/2K; sample a deterministic 128x128 submatrix at
    # larger sizes to keep independent FP32 dequantization reasonably cheap.
    if size <= 2048:
        rows = cols = torch.arange(size, device="cuda")
    else:
        gen = torch.Generator(device="cuda")
        gen.manual_seed(41)
        rows = torch.randperm(size, device="cuda", generator=gen)[:128]
        cols = torch.randperm(size, device="cuda", generator=gen)[:128]
    a_ref = a[rows].float() * torch.exp2(raw_sa[rows].float() - 127).repeat_interleave(32, dim=1)
    b_ref = b[cols].float() * torch.exp2(raw_sb[cols].float() - 127).repeat_interleave(32, dim=1)
    expected = a_ref @ b_ref.T
    actual = out[rows[:, None], cols[None, :]].float()
    finite = bool(torch.isfinite(actual).all())
    max_abs = float((actual - expected).abs().max())
    correct = finite and bool(torch.allclose(actual, expected, rtol=0.03, atol=0.5))
    if not correct:
        raise AssertionError(f"size={size}: finite={finite}, max_abs={max_abs}")

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(unroll):
            launch()
    torch.cuda.synchronize()
    timings = [_time_batch(graph.replay, iterations) / unroll for _ in range(samples)]
    ms = statistics.median(timings)
    asm = kernel.asm.get("amdgcn", "")
    return {
        "size": size,
        "variant": variant,
        "correct": correct,
        "checked_elements": rows.numel() * cols.numel(),
        "max_abs": max_abs,
        "ms": ms,
        "tflops": 2 * size**3 / (ms * 1e9),
        "vgprs": kernel.n_regs,
        "spills": kernel.n_spills,
        "lds_bytes": kernel.metadata.shared,
        "scratch_loads": len(re.findall(r"^\s*scratch_load", asm, re.MULTILINE)),
        "scratch_stores": len(re.findall(r"^\s*scratch_store", asm, re.MULTILINE)),
        "samples_ms": timings,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", action="append", type=int, choices=(1024, 2048, 4096, 8192, 16384))
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--unroll", type=int, default=2)
    parser.add_argument("--variant", choices=("lds", "generic", "auto"), default="auto")
    parser.add_argument("--bm", type=int, default=128)
    parser.add_argument("--bn", type=int, default=256)
    parser.add_argument("--stages", type=int, default=2)
    parser.add_argument("--warps", type=int, default=8)
    parser.add_argument("--xcd-chunk", type=int, default=0)
    args = parser.parse_args()
    for size in args.size or (1024, 2048, 4096, 8192, 16384):
        print(json.dumps(run(size, args.samples, args.iterations, args.unroll, args.variant,
                             args.bm, args.bn, args.stages, args.warps,
                             args.xcd_chunk)), flush=True)


if __name__ == "__main__":
    main()
