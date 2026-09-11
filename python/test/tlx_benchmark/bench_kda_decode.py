"""Perf and compile-time reporting for ``tlx.ops.kda_recurrent_decode``.

The indexed state-pool contract has no dependency-free reference in this
suite, so the benchmark reports TLX latency without a speedup gate.
Correctness is covered by the TLX ops unit tests.
"""

from __future__ import annotations

import importlib
import pathlib
import sys

import torch
import torch.nn.functional as F
from triton.tlx.ops.kernels.kda._shapes import FLOOR_TFLOPS, GFX950_DECODE_SYNTHETIC

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _harness import Case, Prepared, driver

OP = "kda_recurrent_decode"
REF_NAME = ""
DEFAULT_SPACE = "heuristic"
EXTRA_COLUMNS = (("latency us", "latency_us"), )
COLD_COMPILE = "first"

DTYPES = {"bf16": torch.bfloat16}


def shapes(synthetic: bool = False) -> list[list]:
    if synthetic:
        return list(GFX950_DECODE_SYNTHETIC)
    module = importlib.import_module(f"triton.tlx.ops.kernels.kda.{driver.arch()}_decode")
    return list(module.PERF_SHAPES)


def _label(batch: int, heads: int, key_dim: int, value_dim: int, dtype: str) -> str:
    return (f"((), {{'dtype': '{dtype}', 'batch': '{batch}', 'H': '{heads}', "
            f"'K': '{key_dim}', 'V': '{value_dim}'}})")


def cases(synthetic: bool = False) -> list[Case]:
    return [
        Case(
            op=OP,
            arch=driver.arch(),
            dtype=str(DTYPES[entry[4]]).removeprefix("torch."),
            shape=tuple(entry[:4]),
            label=_label(*entry),
        )
        for entry in shapes(synthetic)
    ]


def _flops(batch: int, heads: int, key_dim: int, value_dim: int) -> int:
    # State decay plus prediction, rank-1 update, and output matrix-vector products.
    return batch * heads * (7 * key_dim * value_dim + 2 * value_dim)


def prepare(case: Case, space: str) -> Prepared:
    from triton.tlx.ops import kda_recurrent_decode

    del space  # This implementation has one fixed launch policy.
    batch, heads, key_dim, value_dim = case.shape
    dtype = getattr(torch, case.dtype)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(29 + batch + heads)

    def randn(*tensor_shape):
        return torch.randn(*tensor_shape, generator=generator, device="cuda", dtype=torch.float32)

    q = F.normalize(randn(1, batch, heads, key_dim), dim=-1).to(dtype)
    k = F.normalize(randn(1, batch, heads, key_dim), dim=-1).to(dtype)
    v = randn(1, batch, heads, value_dim).to(dtype)
    g = -F.softplus(randn(1, batch, heads, key_dim))
    beta = torch.sigmoid(randn(1, batch, heads))
    state_pool = 0.1 * randn(2 * batch, heads, value_dim, key_dim)
    read_indices = torch.arange(batch, device="cuda", dtype=torch.int32)
    write_indices = read_indices + batch
    cu_seqlens = torch.arange(batch + 1, device="cuda", dtype=torch.int64)

    tlx_fn = lambda: kda_recurrent_decode(
        q,
        k,
        v,
        g,
        beta,
        scale=1.0,
        state_pool=state_pool,
        read_indices=read_indices,
        write_indices=write_indices,
        cu_seqlens=cu_seqlens,
        arch=driver.arch(),
    )
    return Prepared(
        tlx_fn=tlx_fn,
        ref_fn=None,
        flop_count=_flops(batch, heads, key_dim, value_dim),
        floor_tflops=FLOOR_TFLOPS,
        check=None,
    )


def annotate(result) -> None:
    if result.tlx and result.flop_count:
        latency_s = result.flop_count / (result.tlx.mean * 1e12)
        result.extra["latency_us"] = latency_s * 1e6


supported, default_json, run, main = driver.bind(sys.modules[__name__])

if __name__ == "__main__":
    raise SystemExit(main())
