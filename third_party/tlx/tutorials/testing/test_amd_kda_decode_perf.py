"""Performance harness for the gfx950 TLX KDA recurrent decode operator."""

from __future__ import annotations

import argparse
import time

import torch
import triton
from triton.tlx.ops import kda_recurrent_decode

DEVICE = triton.runtime.driver.active.get_active_torch_device()
DIM = 128
DECODE_HEADS = (4, 12)
DECODE_BATCHES = (1, 2, 4, 8, 16, 32)
DECODE_SHAPES = tuple((batch, heads) for heads in DECODE_HEADS for batch in DECODE_BATCHES)
DECODE_PROVIDERS = ("tlx",)


def build_decode_inputs(batch: int, heads: int):
    """Build deterministic one-token decode inputs and a 2..32-slot pool."""
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(29 + batch + heads)
    shape = (1, batch, heads, DIM)
    q = torch.nn.functional.normalize(
        torch.randn(shape, generator=generator, device=DEVICE, dtype=torch.float32),
        dim=-1,
    ).to(torch.bfloat16)
    k = torch.nn.functional.normalize(
        torch.randn(shape, generator=generator, device=DEVICE, dtype=torch.float32),
        dim=-1,
    ).to(torch.bfloat16)
    v = torch.randn(shape, generator=generator, device=DEVICE, dtype=torch.bfloat16)
    g = -torch.nn.functional.softplus(
        torch.randn(shape, generator=generator, device=DEVICE, dtype=torch.float32))
    beta = torch.sigmoid(torch.randn(
        (1, batch, heads),
        generator=generator,
        device=DEVICE,
        dtype=torch.float32,
    ))
    slots = max(2, 2 * batch)
    state_pool = 0.1 * torch.randn(
        (slots, heads, DIM, DIM),
        generator=generator,
        device=DEVICE,
        dtype=torch.float32,
    )
    read_indices = torch.arange(batch, device=DEVICE, dtype=torch.int32)
    write_indices = read_indices + batch
    cu_seqlens = torch.arange(batch + 1, device=DEVICE, dtype=torch.int64)
    return q, k, v, g, beta, state_pool, read_indices, write_indices, cu_seqlens


def make_decode_runner(provider: str, inputs):
    q, k, v, g, beta, state_pool, read_indices, write_indices, cu_seqlens = inputs
    if provider != "tlx":
        return None

    def run():
        return kda_recurrent_decode(
            q,
            k,
            v,
            g,
            beta,
            state_pool=state_pool,
            read_indices=read_indices,
            write_indices=write_indices,
            cu_seqlens=cu_seqlens,
        )

    return run


def first_call_ms(function) -> float:
    """Measure a synchronized first call (compile plus launch when cache-cold)."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    function()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e3


def create_benchmark(providers=DECODE_PROVIDERS):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["BATCH", "HEADS"],
            x_vals=list(DECODE_SHAPES),
            line_arg="provider",
            line_vals=list(providers),
            line_names=list(providers),
            ylabel="microseconds",
            plot_name="gfx950-kda-decode-bf16-h4-h12-d128",
            args={},
        ))
    def benchmark(BATCH, HEADS, provider):
        inputs = build_decode_inputs(BATCH, HEADS)
        function = make_decode_runner(provider, inputs)
        if function is None:
            return float("nan"), float("nan"), float("nan")
        median_ms, min_ms, max_ms = triton.testing.do_bench(
            function,
            quantiles=[0.5, 0.2, 0.8],
            warmup=100,
            rep=200,
        )
        return median_ms * 1e3, min_ms * 1e3, max_ms * 1e3

    return benchmark


benchmark = create_benchmark()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark gfx950 TLX KDA decode")
    parser.add_argument(
        "--version",
        nargs="+",
        choices=list(DECODE_PROVIDERS),
        default=list(DECODE_PROVIDERS),
    )
    parser.add_argument(
        "--compile-time",
        action="store_true",
        help="Report synchronized first-call time for batch 32",
    )
    args = parser.parse_args()
    if args.compile_time:
        for heads in DECODE_HEADS:
            compile_inputs = build_decode_inputs(32, heads)
            for selected_provider in args.version:
                selected_function = make_decode_runner(selected_provider, compile_inputs)
                print(f"{selected_provider}-h{heads}: first_call_ms={first_call_ms(selected_function):.3f}")
    create_benchmark(tuple(args.version)).run(show_plots=False, print_data=True)
