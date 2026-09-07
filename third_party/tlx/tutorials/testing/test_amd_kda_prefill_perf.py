"""Performance harness for the gfx950 TLX KDA prefill tutorial."""

from __future__ import annotations

import argparse
import time

import torch
import triton
from triton.language.extra.tlx.tutorials.amd_kda_prefill import kda_paged_prefill_tlx

DEVICE = triton.runtime.driver.active.get_active_torch_device()
DIM = 128
PREFILL_HEADS = (4, 12)
PREFILL_BASE_SHAPES = (
    (4096, 1),
    (4096, 4),
    (131072, 1),
    (131072, 8),
)
PREFILL_SHAPES = tuple((*shape, heads) for heads in PREFILL_HEADS for shape in PREFILL_BASE_SHAPES)
PREFILL_PROVIDERS = ("tlx",)


def _packed_lengths(total_tokens: int, sequences: int) -> list[int]:
    if sequences == 1:
        return [total_tokens]
    weights = [sequences - index for index in range(sequences)]
    denominator = sum(weights)
    lengths = [max(1, total_tokens * weight // denominator) for weight in weights]
    lengths[-1] += total_tokens - sum(lengths)
    return lengths


def build_prefill_inputs(total_tokens: int, sequences: int, heads: int):
    """Build deterministic BF16 KDA inputs with nonuniform packed lengths."""
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(17 + total_tokens + sequences + heads)
    shape = (1, total_tokens, heads, DIM)
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
        (1, total_tokens, heads),
        generator=generator,
        device=DEVICE,
        dtype=torch.float32,
    ))
    initial_state = torch.zeros(
        (sequences, heads, DIM, DIM),
        device=DEVICE,
        dtype=torch.float32,
    )
    lengths = _packed_lengths(total_tokens, sequences)
    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)
    cu_seqlens = torch.tensor(boundaries, device=DEVICE, dtype=torch.int64)
    return q, k, v, g, beta, initial_state, cu_seqlens


def make_prefill_runner(provider: str, inputs):
    q, k, v, g, beta, initial_state, cu_seqlens = inputs
    if provider != "tlx":
        return None

    def run():
        return kda_paged_prefill_tlx(
            q,
            k,
            v,
            g,
            beta,
            initial_state=initial_state,
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


def create_benchmark(providers=PREFILL_PROVIDERS):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["TOTAL_TOKENS", "SEQUENCES", "HEADS"],
            x_vals=list(PREFILL_SHAPES),
            line_arg="provider",
            line_vals=list(providers),
            line_names=list(providers),
            ylabel="million tokens/s",
            plot_name="gfx950-kda-prefill-bf16-h4-h12-d128",
            args={},
        ))
    def benchmark(TOTAL_TOKENS, SEQUENCES, HEADS, provider):
        inputs = build_prefill_inputs(TOTAL_TOKENS, SEQUENCES, HEADS)
        function = make_prefill_runner(provider, inputs)
        if function is None:
            return float("nan"), float("nan"), float("nan")
        warmup, repetitions = (5, 20) if TOTAL_TOKENS >= 131072 else (20, 50)
        median_ms, min_ms, max_ms = triton.testing.do_bench(
            function,
            quantiles=[0.5, 0.2, 0.8],
            warmup=warmup,
            rep=repetitions,
        )

        def throughput(milliseconds):
            return TOTAL_TOKENS / (milliseconds * 1e-3) / 1e6

        return throughput(median_ms), throughput(max_ms), throughput(min_ms)

    return benchmark


benchmark = create_benchmark()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark gfx950 TLX KDA prefill")
    parser.add_argument(
        "--version",
        nargs="+",
        choices=list(PREFILL_PROVIDERS),
        default=list(PREFILL_PROVIDERS),
    )
    parser.add_argument(
        "--compile-time",
        action="store_true",
        help="Report synchronized first-call time for the 4K packed shape",
    )
    args = parser.parse_args()
    if args.compile_time:
        for heads in PREFILL_HEADS:
            compile_inputs = build_prefill_inputs(4096, 4, heads)
            for selected_provider in args.version:
                selected_function = make_prefill_runner(selected_provider, compile_inputs)
                print(f"{selected_provider}-h{heads}: first_call_ms={first_call_ms(selected_function):.3f}")
    create_benchmark(tuple(args.version)).run(show_plots=False, print_data=True)
