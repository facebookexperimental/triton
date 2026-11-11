"""Shapes for ``tlx.ops.hstu_cross_attn``: the production inference and training calls it was tuned on.

All shapes are jagged. Inference shapes are H=1, D=128 fp16 calls from four
recommendation models (``model_a``..``model_d``; per-sequence lengths are
synthesized around the known totals), a serving trace of ``model_a`` at 11-16
sequences per call, and an H=4, D=64 bf16 serving trace with num_targets.
Training shapes are a recommendation training layout as the kernel sees it:
batch 1024, MQA with its 2 q heads folded into the sequence (H=1, q per
sequence = 2 x targets), shared K/V, D=128, bf16, KV in [0.9, 1.0) x max KV.
"""

from __future__ import annotations

from typing import NamedTuple

from .._shape_suites import FocusRegistry, FocusSuite

#: How per-sequence lengths are drawn from the shape fields:
#:   split -- `batch` lengths summing exactly to `total_q` / `total_kv`, capped at
#:            `max_q` / `max_kv` when nonzero.
#:   train -- q = 2 x randint(1, max_q // 2), kv = randint(0.9 x max_kv, max_kv).
#:   train_full -- train with every q at max_q (the no-padding ceiling).
#:   serving -- the H=4 serving trace: kv in [27/32, 1] x max_kv, q in
#:            [64, 2/3 max_q), first sequence at both maxima, num_targets <= 128,
#:            q/k/v as row-stride-8-head views.
DRAWS = ("split", "train", "train_full", "serving")


class HSTUCrossAttentionShape(NamedTuple):
    model: str
    draw: str
    batch: int
    total_q: int
    total_kv: int
    max_q: int
    max_kv: int
    heads: int
    head_dim: int
    num_softmax_heads: int
    alpha: float
    dtype: str


def _split(model, z, tq, tkv, mq, mkv, alpha, softmax):
    return HSTUCrossAttentionShape(model, "split", z, tq, tkv, mq, mkv, 1, 128, 1 if softmax else 0, alpha, "fp16")


def _train(model, draw, targets, kv_cap, softmax):
    return HSTUCrossAttentionShape(model, draw, 1024, 0, 0, 2 * targets, kv_cap, 1, 128, 1 if softmax else 0, 128**-0.5,
                                   "bf16")


SYNTHETIC: tuple[HSTUCrossAttentionShape, ...] = (
    HSTUCrossAttentionShape("synthetic", "split", 4, 512, 2048, 0, 0, 1, 128, 1, 128**-0.5, "fp16"),
    HSTUCrossAttentionShape("synthetic", "split", 4, 512, 2048, 0, 0, 2, 128, 0, 128**-0.5, "bf16"),
    HSTUCrossAttentionShape("synthetic", "split", 8, 1024, 4096, 0, 0, 4, 64, 4, 64**-0.5, "bf16"),
)

INFERENCE = tuple(
    _split(*args, softmax=softmax) for softmax in (True, False) for args in (
        ("model_a", 9, 37_024, 121_885, 0, 0, 128**0.5),
        ("model_b", 8, 95_744, 55_694, 16_896, 11_000, 11.31),
        ("model_c", 2, 34_176, 8_192, 0, 0, 128**0.5),
        ("model_d", 21, 110_432, 197_016, 11_000, 0, 25.30),
        # model_a serving trace: 11-16 sequences per call, ~4.1K q and ~13.5K KV each.
        ("model_a_trace_z11", 11, 11 * 4114, 11 * 13_543, 6350, 0, 128**-0.5),
        ("model_a_trace_z13", 13, 13 * 4114, 13 * 13_543, 6350, 0, 128**-0.5),
        ("model_a_trace_z16", 16, 16 * 4114, 16 * 13_543, 6350, 0, 128**-0.5),
    )) + tuple(
        HSTUCrossAttentionShape("serving_h4", "serving", z, 0, 0, 480, 1024, 4, 64, 4, 0.125, "bf16")
        for z in (8, 16, 37, 40, 43, 46))

TRAINING = tuple(
    _train(f"train_t{targets}_kv{kv_cap}", "train", targets, kv_cap, softmax)
    for softmax in (True, False)
    for kv_cap in (4096, 8192)
    for targets in (32, 128, 160, 256)) + tuple(
        _train("train_full_t256_kv4096", "train_full", 256, 4096, softmax) for softmax in (True, False))

FOCUS_SUITES = (
    FocusSuite(name="gfx950_1", op="hstu_cross_attn", shapes=INFERENCE),
    FocusSuite(name="gfx950_2", op="hstu_cross_attn", shapes=TRAINING),
    FocusSuite(name="gfx950_all", op="hstu_cross_attn", shapes=(), includes=("gfx950_1", "gfx950_2")),
)

DEFAULT_SUITES = {"gfx950": ("gfx950_all", )}
FOCUS = FocusRegistry("hstu_cross_attn", FOCUS_SUITES, DEFAULT_SUITES)
CORRECTNESS_SHAPES = tuple(dict.fromkeys((*SYNTHETIC, *FOCUS.all_shapes())))


def _split_total(total, z, cap, gen):
    import torch

    weights = 0.5 + torch.rand(z, generator=gen)
    lengths = (weights / weights.sum() * total).long()
    lengths[-1] += total - int(lengths.sum())
    if cap:
        while int(lengths.max()) > cap:
            over = (lengths - cap).clamp(min=0)
            lengths -= over
            room = (cap - lengths).clamp(min=0)
            lengths += (room.double() / room.sum() * int(over.sum())).floor().long()
            lengths[int(torch.argmin(lengths))] += total - int(lengths.sum())
    return lengths


def _lengths(shape, gen):
    import torch

    z = shape.batch
    if shape.draw == "split":
        return _split_total(shape.total_q, z, shape.max_q, gen), _split_total(shape.total_kv, z, shape.max_kv, gen)
    if shape.draw in ("train", "train_full"):
        kv = torch.randint(int(0.9 * shape.max_kv), shape.max_kv, (z, ), generator=gen)
        if shape.draw == "train_full":
            return torch.full((z, ), shape.max_q, dtype=torch.int64), kv
        return torch.randint(1, shape.max_q // 2 + 1, (z, ), generator=gen) * 2, kv
    if shape.draw == "serving":
        kv = torch.randint(shape.max_kv * 27 // 32, shape.max_kv + 1, (z, ), generator=gen)
        q = torch.randint(64, shape.max_q * 2 // 3, (z, ), generator=gen)
        kv[0], q[0] = shape.max_kv, shape.max_q
        return q, kv
    raise ValueError(f"unknown draw {shape.draw!r}")


def _offsets(lengths, device):
    import torch

    offsets = torch.zeros((lengths.numel() + 1, ), dtype=torch.int64, device=device)
    offsets[1:] = torch.cumsum(lengths.to(device), dim=0)
    return offsets


def inputs(shape: HSTUCrossAttentionShape, requires_grad=False, device="cuda", seed=0) -> dict:
    """Keyword arguments for `tlx.ops.hstu_cross_attn`, plus the observed lengths.

    Training shapes share K/V (`v is k`), as the training model does; inference
    shapes pass distinct K and V.
    """
    import torch

    gen = torch.Generator().manual_seed(seed)
    lengths_q, lengths_kv = _lengths(shape, gen)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[shape.dtype]
    total_q, total_kv = int(lengths_q.sum()), int(lengths_kv.sum())
    dgen = torch.Generator(device=device).manual_seed(seed)

    def rand(total, row_heads):
        base = torch.empty((total, row_heads, shape.head_dim), dtype=dtype, device=device)
        base.uniform_(-0.1, 0.1, generator=dgen)
        return base[:, :shape.heads, :]

    serving = shape.draw == "serving"
    shared_kv = shape.draw in ("train", "train_full")
    row_heads = 8 if serving else shape.heads
    q = rand(total_q, row_heads)
    k = rand(total_kv, row_heads)
    v = k if shared_kv else rand(total_kv, row_heads)
    if requires_grad:
        # Gradients need leaf tensors, so the strided serving views become dense.
        q = q.detach().contiguous().requires_grad_()
        k = k.detach().contiguous().requires_grad_()
        v = k if shared_kv else v.detach().contiguous().requires_grad_()
    max_q = shape.max_q if shape.draw == "split" and shape.max_q else int(lengths_q.max())
    max_kv = shape.max_kv if shape.max_kv else int(lengths_kv.max())
    if serving:
        num_targets = torch.minimum(torch.randint(1, 129, (shape.batch, ), generator=gen), lengths_q).to(device)
        attn_scale = torch.tensor(1.0 / shape.max_kv, device=device)
    else:
        num_targets = None
        attn_scale = torch.tensor(1.0 / (max_q + max_kv), device=device)
    return dict(q=q, k=k, v=v, seq_offsets=_offsets(lengths_kv, device), max_seq_len=max_kv, attn_scale=attn_scale,
                alpha=shape.alpha, seq_offsets_q=_offsets(lengths_q, device), max_q_len=max_q,
                num_softmax_heads=shape.num_softmax_heads, num_targets=num_targets, shared_kv=shared_kv,
                lengths_q=lengths_q, lengths_kv=lengths_kv)


def flops(shape: HSTUCrossAttentionShape, lengths_q, lengths_kv, direction="fwd") -> int:
    """QK^T plus PV over the actual (unpadded) lengths; backward counts 2.5x."""
    pairs = int((lengths_q.to("cpu").double() * lengths_kv.to("cpu").double()).sum())
    total = 2.0 * shape.heads * (shape.head_dim + shape.head_dim) * pairs
    return int(total * (2.5 if direction == "bwd" else 1.0))


def label(shape: HSTUCrossAttentionShape, direction="fwd") -> str:
    act = "softmax" if shape.num_softmax_heads else "silu"
    return (f"((), {{'model': '{shape.model}', 'act': '{act}', 'dtype': '{shape.dtype}', 'dir': '{direction}', "
            f"'Z': '{shape.batch}', 'H': '{shape.heads}', 'HEAD_DIM': '{shape.head_dim}'}})")
