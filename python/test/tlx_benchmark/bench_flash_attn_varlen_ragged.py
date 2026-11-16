"""Perf and compile-time reporting for ``tlx.ops.flash_attn_varlen`` with key/value lengths that differ from the
query lengths, as in cross-attention over a user history.

The reference is torch SDPA over the batch padded to its longest query and key sequences, with a boolean mask for
the padding (and the causal triangle): what a torch-only caller runs without a varlen kernel. It computes the padded
positions too, so it is the baseline to replace rather than a peer kernel.
"""

from __future__ import annotations

import pathlib
import sys

import torch
import torch.nn.functional as F
from triton.tlx.ops.kernels.flash_attn_varlen._shapes import FOCUS as SHAPE_SUITES
from triton.tlx.ops.kernels.flash_attn_varlen._shapes import (SYNTHETIC, FlashAttentionVarlenShape, cross_flops,
                                                              cross_inputs, is_cross_attention, label, lengths)

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
    found = SYNTHETIC if synthetic else SHAPE_SUITES.shapes(driver.arch(), suites)
    return [entry for entry in found if is_cross_attention(entry)]


def cases(synthetic: bool = False, suites=None) -> list[Case]:
    return [
        Case(op=OP, arch=driver.arch(), dtype=str(DTYPES[entry[6]]).removeprefix("torch."),
             shape=(*entry[:6], *entry[7:]), label=label(*entry)) for entry in shapes(synthetic, suites)
    ]


def prepare(case: Case, space: str) -> Prepared:
    from triton.tlx.ops import flash_attn_varlen

    del space  # One fixed launch config.
    batch, _, _, heads, head_dim, causal = case.shape[:6]
    lens_q, lens_k = lengths(FlashAttentionVarlenShape(*case.shape[:6], case.dtype, *case.shape[6:]))
    q, k, v, cu_q, cu_k = cross_inputs(lens_q, lens_k, heads, head_dim, getattr(torch, case.dtype))
    longest_q, longest_k = max(lens_q), max(lens_k)

    device = q.device
    sizes_q = torch.tensor(lens_q, device=device)
    sizes_k = torch.tensor(lens_k, device=device)
    valid_q = torch.arange(longest_q, device=device)[None, :] < sizes_q[:, None]
    valid_k = torch.arange(longest_k, device=device)[None, :] < sizes_k[:, None]
    mask = valid_q[:, None, :, None] & valid_k[:, None, None, :]
    if causal:
        # Bottom-right aligned: row i sees keys j <= i + k_len - q_len.
        rows = torch.arange(longest_q, device=device)[None, :, None]
        cols = torch.arange(longest_k, device=device)[None, None, :]
        mask = mask & (cols <= rows + (sizes_k - sizes_q)[:, None, None])[:, None]
    # Padding query rows attend everywhere, since SDPA returns NaN for a row
    # with no key; they are never compared.
    mask = mask | ~valid_q[:, None, :, None]

    def padded(x, valid, longest):
        out = x.new_zeros(batch, longest, heads, head_dim)
        out[valid] = x
        return out.transpose(1, 2)

    qp, kp, vp = padded(q, valid_q, longest_q), padded(k, valid_k, longest_k), padded(v, valid_k, longest_k)

    tlx_fwd = lambda: flash_attn_varlen(  # noqa: E731
        q, k, v, cu_q, cu_k, longest_q, longest_k, causal=causal)
    ref_fwd = lambda: F.scaled_dot_product_attention(qp, kp, vp, attn_mask=mask)  # noqa: E731

    return Prepared(
        tlx_fn=tlx_fwd,
        ref_fn=ref_fwd,
        flop_count=cross_flops(lens_q, lens_k, heads, head_dim, causal),
        check=lambda: close_enough(tlx_fwd(),
                                   ref_fwd().transpose(1, 2)[valid_q], REL_PRECISION[case.dtype]),
        extra={"tokens": sum(lens_q)},
    )


supported, default_json, run, main = driver.bind(sys.modules[__name__])

if __name__ == "__main__":
    raise SystemExit(main())
