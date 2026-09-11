"""Perf and compile-time reporting for ``tlx.ops.kda_paged_prefill``.

There is no dependency-free reference with the same prepared-input and
state-carry contract, so this benchmark reports TLX throughput and latency
without a speedup gate. Correctness is covered by the TLX ops unit tests.
"""

from __future__ import annotations

import importlib
import pathlib
import sys

import torch
import torch.nn.functional as F
from triton.tlx.ops.kernels.kda._shapes import FLOOR_TFLOPS, GFX950_PREFILL_SYNTHETIC, flops

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _harness import Case, Prepared, driver

OP = "kda_paged_prefill"
REF_NAME = ""
DEFAULT_SPACE = "heuristic"
EXTRA_COLUMNS = (("latency us", "latency_us"), ("Mtok/s", "mtokens_per_s"))
COLD_COMPILE = "first"

DTYPES = {"bf16": torch.bfloat16}


def shapes(synthetic: bool = False) -> list[list]:
    if synthetic:
        return list(GFX950_PREFILL_SYNTHETIC)
    module = importlib.import_module(f"triton.tlx.ops.kernels.kda.{driver.arch()}_prefill")
    return list(module.PERF_SHAPES)


def _label(total_tokens: int, sequences: int, heads: int, key_dim: int, value_dim: int, dtype: str) -> str:
    return (f"((), {{'dtype': '{dtype}', 'tokens': '{total_tokens}', 'sequences': '{sequences}', "
            f"'H': '{heads}', 'K': '{key_dim}', 'V': '{value_dim}'}})")


def cases(synthetic: bool = False) -> list[Case]:
    return [
        Case(
            op=OP,
            arch=driver.arch(),
            dtype=str(DTYPES[entry[5]]).removeprefix("torch."),
            shape=tuple(entry[:5]),
            label=_label(*entry),
        )
        for entry in shapes(synthetic)
    ]


def _packed_lengths(total_tokens: int, sequences: int) -> list[int]:
    if sequences == 1:
        return [total_tokens]
    weights = [sequences - index for index in range(sequences)]
    denominator = sum(weights)
    lengths = [max(1, total_tokens * weight // denominator) for weight in weights]
    lengths[-1] += total_tokens - sum(lengths)
    return lengths


def prepare(case: Case, space: str) -> Prepared:
    from triton.tlx.ops import kda_paged_prefill

    del space  # This implementation has one fixed launch policy.
    total_tokens, sequences, heads, key_dim, value_dim = case.shape
    dtype = getattr(torch, case.dtype)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(17 + total_tokens + sequences + heads)
    shape = (1, total_tokens, heads, key_dim)

    def randn(*tensor_shape):
        return torch.randn(*tensor_shape, generator=generator, device="cuda", dtype=torch.float32)

    q = F.normalize(randn(*shape), dim=-1).to(dtype)
    k = F.normalize(randn(*shape), dim=-1).to(dtype)
    v = randn(1, total_tokens, heads, value_dim).to(dtype)
    g = -F.softplus(randn(*shape))
    beta = torch.sigmoid(randn(1, total_tokens, heads))
    initial_state = torch.zeros(
        sequences,
        heads,
        value_dim,
        key_dim,
        device="cuda",
        dtype=torch.float32,
    )
    boundaries = [0]
    for length in _packed_lengths(total_tokens, sequences):
        boundaries.append(boundaries[-1] + length)
    cu_seqlens = torch.tensor(boundaries, device="cuda", dtype=torch.int64)

    tlx_fn = lambda: kda_paged_prefill(
        q,
        k,
        v,
        g,
        beta,
        scale=1.0,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        arch=driver.arch(),
    )
    return Prepared(
        tlx_fn=tlx_fn,
        ref_fn=None,
        flop_count=flops(1, total_tokens, heads, key_dim),
        floor_tflops=FLOOR_TFLOPS,
        check=None,
        extra={"tokens": total_tokens},
    )


def annotate(result) -> None:
    if result.tlx and result.flop_count:
        latency_s = result.flop_count / (result.tlx.mean * 1e12)
        result.extra["latency_us"] = latency_s * 1e6
        result.extra["mtokens_per_s"] = result.extra["tokens"] / latency_s / 1e6


supported, default_json, run, main = driver.bind(sys.modules[__name__])

if __name__ == "__main__":
    raise SystemExit(main())
