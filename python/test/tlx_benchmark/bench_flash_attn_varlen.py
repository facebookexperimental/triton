"""Perf and compile-time reporting for ``tlx.ops.flash_attn_varlen``.

The reference is torch SDPA over the batch padded to its longest sequence,
with a boolean mask for the padding (and the causal triangle): what a
torch-only caller runs without a varlen kernel. It computes the padded
positions too, so it is the baseline to replace rather than a peer kernel.
"""

from __future__ import annotations

import pathlib
import sys

import torch
import torch.nn.functional as F
from triton.tlx.ops.kernels.flash_attn_varlen._shapes import FOCUS as SHAPE_SUITES
from triton.tlx.ops.kernels.flash_attn_varlen._shapes import SYNTHETIC, flops, inputs, label, seqlens

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _harness import Case, Prepared, close_enough, driver  # noqa: E402

OP = "flash_attn_varlen"
REF_NAME = "torch.nn.functional.scaled_dot_product_attention (padded, masked)"
DEFAULT_SPACE = "heuristic"
EXTRA_COLUMNS = (("tokens", "tokens"), )
COLD_COMPILE = "first"

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
REL_PRECISION = {"float16": 1e-3, "bfloat16": 8e-3}


def shapes(synthetic: bool = False, suites=None) -> list:
    return list(SYNTHETIC if synthetic else SHAPE_SUITES.shapes(driver.arch(), suites))


def cases(synthetic: bool = False, suites=None) -> list[Case]:
    return [
        Case(op=OP, arch=driver.arch(), dtype=str(DTYPES[entry[6]]).removeprefix("torch."), shape=tuple(entry[:6]),
             label=label(*entry)) for entry in shapes(synthetic, suites)
    ]


def prepare(case: Case, space: str) -> Prepared:
    from triton.tlx.ops import flash_attn_varlen

    del space  # One fixed launch config.
    batch, min_seqlen, max_seqlen, heads, head_dim, causal = case.shape
    lengths = seqlens(batch, min_seqlen, max_seqlen)
    q, k, v, cu_seqlens = inputs(lengths, heads, head_dim, getattr(torch, case.dtype))
    longest = max(lengths)

    pos = torch.arange(longest, device=q.device)
    valid = pos[None, :] < torch.tensor(lengths, device=q.device)[:, None]
    mask = valid[:, None, None, :] & valid[:, None, :, None]
    if causal:
        mask = mask & torch.ones(longest, longest, dtype=torch.bool, device=q.device).tril()
    # Padding query rows attend everywhere, since SDPA returns NaN for a row
    # with no key; they are never compared.
    mask = mask | ~valid[:, None, :, None]

    def padded(x):
        out = x.new_zeros(batch, longest, heads, head_dim)
        out[valid] = x
        return out.transpose(1, 2)

    qp, kp, vp = padded(q), padded(k), padded(v)

    tlx_fwd = lambda: flash_attn_varlen(  # noqa: E731
        q, k, v, cu_seqlens, cu_seqlens, longest, longest, causal=causal)
    ref_fwd = lambda: F.scaled_dot_product_attention(qp, kp, vp, attn_mask=mask)  # noqa: E731

    return Prepared(
        tlx_fn=tlx_fwd,
        ref_fn=ref_fwd,
        flop_count=flops(lengths, heads, head_dim, causal),
        check=lambda: close_enough(tlx_fwd(),
                                   ref_fwd().transpose(1, 2)[valid], REL_PRECISION[case.dtype]),
        extra={"tokens": sum(lengths)},
    )


supported, default_json, run, main = driver.bind(sys.modules[__name__])

if __name__ == "__main__":
    raise SystemExit(main())
