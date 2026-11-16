from __future__ import annotations

import math
from typing import NamedTuple

from .._shape_suites import FocusRegistry, FocusSuite


class MLAWindowAttnShape(NamedTuple):
    tokens: int
    seqs: int
    d_model: int
    heads: int
    ranks: tuple[int, int, int]
    window: tuple[int, int]
    dtype: str


SYNTHETIC: tuple[MLAWindowAttnShape, ...] = (
    MLAWindowAttnShape(1205, 3, 256, 2, (64, 32, 64), (4, 4), "fp16"),
    # Unequal q and gate ranks: separate down products.
    MLAWindowAttnShape(1205, 3, 128, 2, (32, 64, 16), (4, 4), "fp16"),
    # Four heads: the per-block kernel.
    MLAWindowAttnShape(1205, 3, 256, 4, (16, 16, 32), (4, 4), "fp16"),
)

# The self-attention block of a production ranking model: the smallest, quartile and largest of the 88 calls in its
# trace, as (tokens, sequences); no sequence is longer than 5120 tokens.
SM90_1 = FocusSuite(
    name="sm90_1",
    op="mla_window_attn",
    shapes=tuple(
        MLAWindowAttnShape(tokens, seqs, 256, 2, (64, 32, 64), (4, 4), "fp16")
        for tokens, seqs in ((20480, 4), (32304, 7), (35988, 8), (40960, 8), (55800, 11))),
)

FOCUS_SUITES = (SM90_1, )
DEFAULT_SUITES = {"sm90": ("sm90_1", )}
FOCUS = FocusRegistry("mla_window_attn", FOCUS_SUITES, DEFAULT_SUITES)
CORRECTNESS_SHAPES = tuple(dict.fromkeys((*SYNTHETIC, *FOCUS.all_shapes())))


def label(tokens, seqs, d_model, heads, ranks, window, dtype) -> str:
    return (f"(({tokens}, {d_model}), {{'dtype': '{dtype}', 'seqs': {seqs}, 'heads': {heads}, 'ranks': {ranks}, "
            f"'window': {window}}})")


def lengths(tokens: int, seqs: int) -> list[int]:
    """Sequence lengths as equal as possible."""
    return [tokens // seqs + (i < tokens % seqs) for i in range(seqs)]


def flops(tokens, seqs, d_model, heads, ranks, window) -> int:
    """The projections plus q @ k^T and p @ v over the window (keys outside a sequence counted as well)."""
    q_rank, kv_rank, gate_rank = ranks
    down = 2 * d_model * (q_rank + kv_rank + gate_rank)
    up = 2 * d_model * (q_rank + 2 * kv_rank + gate_rank)
    attention = 4 * (window[0] + window[1] + 1) * d_model
    return tokens * (down + up + attention)


def inputs(tokens, seqs, d_model, heads, ranks, window, dtype, device="cuda"):
    """`x`, `cu_seqlens` and the weights as a dict: `norm`, and `q`, `kv`, `gate` as `(down, norm, up)`."""
    import torch

    gen = torch.Generator(device=device).manual_seed(0)

    def uniform(*shape, bound):
        return ((torch.rand(shape, generator=gen, device=device) * 2 - 1) * bound).to(dtype)

    def norm(n):
        return (0.5 + torch.rand(n, generator=gen, device=device)).to(dtype)

    def lora(rank, out):
        return (uniform(rank, d_model, bound=math.sqrt(1 / (d_model + rank))), norm(rank),
                uniform(out, rank, bound=math.sqrt(1 / (rank + out))))

    q_rank, kv_rank, gate_rank = ranks
    weights = dict(norm=norm(d_model), q=lora(q_rank, d_model), kv=lora(kv_rank, 2 * d_model),
                   gate=lora(gate_rank, d_model))
    x = torch.randn(tokens, d_model, generator=gen, device=device).to(dtype)
    starts = [0]
    for n in lengths(tokens, seqs):
        starts.append(starts[-1] + n)
    return x, torch.tensor(starts, device=device, dtype=torch.int32), weights


def window_block_mask(cu_seqlens, tokens: int, window: tuple[int, int]):
    """FlexAttention block mask: each query sees keys `i - left .. i + right` of its own sequence."""
    import torch
    from torch.nn.attention.flex_attention import create_block_mask

    seq = torch.searchsorted(cu_seqlens[1:].long(), torch.arange(tokens, device=cu_seqlens.device), right=True)
    left, right = window

    def mask_mod(b, h, q_idx, kv_idx):
        return (seq[q_idx] == seq[kv_idx]) & (kv_idx >= q_idx - left) & (kv_idx <= q_idx + right)

    return create_block_mask(mask_mod, None, None, tokens, tokens, device=cu_seqlens.device, _compile=True)


def reference(x, weights, heads, block_mask, norm_eps=1e-6, lora_eps=1e-6):
    """The eager module, with the windowed attention as FlexAttention over `block_mask` (see `window_block_mask`)."""
    import torch
    from torch.nn.attention.flex_attention import flex_attention

    def rms(t, w, eps):
        tf = t.float()
        return (w.float() * (tf * torch.rsqrt(tf.square().mean(-1, keepdim=True) + eps))).to(t.dtype)

    tokens, d_model = x.shape
    head_dim = d_model // heads
    h = rms(x, weights["norm"], norm_eps)

    def lora(p):
        down, norm, up = p
        return rms(h @ down.t(), norm, lora_eps) @ up.t()

    q = lora(weights["q"]).view(tokens, heads, head_dim).transpose(0, 1)[None]
    kv = lora(weights["kv"]).view(tokens, heads, 2, head_dim)
    k, v = (kv[:, :, i].transpose(0, 1)[None] for i in range(2))
    gate = torch.sigmoid(lora(weights["gate"]))
    attn = flex_attention(q, k, v, block_mask=block_mask, scale=head_dim**-0.5)
    return x + attn[0].transpose(0, 1).reshape(tokens, d_model) * gate
