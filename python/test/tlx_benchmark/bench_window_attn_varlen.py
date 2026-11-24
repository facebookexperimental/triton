"""Perf and compile-time reporting for ``tlx.ops.window_attn_varlen``.

The reference is PyTorch's FlashAttention-2 varlen forward,
``torch.ops.aten._flash_attention_forward`` with the same ``window_size_left``
and ``window_size_right``: the kernel a PyTorch caller runs for packed
sliding-window self-attention today.
"""

from __future__ import annotations

import pathlib
import sys

import torch
from triton.tlx.ops.kernels.window_attn_varlen._shapes import FOCUS as SHAPE_SUITES
from triton.tlx.ops.kernels.window_attn_varlen._shapes import SYNTHETIC, flops, inputs, label, seqlens

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _harness import Case, Prepared, close_enough, driver  # noqa: E402

OP = "window_attn_varlen"
REF_NAME = "torch.ops.aten._flash_attention_forward (FlashAttention-2, windowed)"
DEFAULT_SPACE = "heuristic"
EXTRA_COLUMNS = (("tokens", "tokens"), )
COLD_COMPILE = "first"

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
REL_PRECISION = {"float16": 1e-3, "bfloat16": 8e-3}


def shapes(synthetic: bool = False, suites=None) -> list:
    return list(SYNTHETIC if synthetic else SHAPE_SUITES.shapes(driver.arch(), suites))


def cases(synthetic: bool = False, suites=None) -> list[Case]:
    return [
        Case(op=OP, arch=driver.arch(), dtype=str(DTYPES[entry[8]]).removeprefix("torch."), shape=tuple(entry[:8]),
             label=label(*entry)) for entry in shapes(synthetic, suites)
    ]


def prepare(case: Case, space: str) -> Prepared:
    from triton.tlx.ops import window_attn_varlen

    del space  # One fixed launch config.
    batch, min_seqlen, max_seqlen, heads, kv_heads, head_dim, left, right = case.shape
    lengths = seqlens(batch, min_seqlen, max_seqlen)
    q, k, v, cu_seqlens = inputs(lengths, heads, kv_heads, head_dim, getattr(torch, case.dtype))
    longest = max(lengths)
    tlx_fwd = lambda: window_attn_varlen(q, k, v, cu_seqlens, (left, right))  # noqa: E731
    ref_fwd = lambda: torch.ops.aten._flash_attention_forward(  # noqa: E731
        q, k, v, cu_seqlens, cu_seqlens, longest, longest, 0.0, False, False, window_size_left=left, window_size_right=
        right)[0]

    return Prepared(
        tlx_fn=tlx_fwd,
        ref_fn=ref_fwd,
        flop_count=flops(lengths, heads, head_dim, left, right),
        check=lambda: close_enough(tlx_fwd(), ref_fwd(), REL_PRECISION[case.dtype]),
        extra={"tokens": sum(lengths)},
    )


supported, default_json, run, main = driver.bind(sys.modules[__name__])

if __name__ == "__main__":
    raise SystemExit(main())
