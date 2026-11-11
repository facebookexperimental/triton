"""INT4 table-batched embedding lookup without pooling for ``tlx.ops.int4_embedding``.

Output row ``p`` is row ``indices[p]`` of the table ``t`` whose rows
``offsets[t * B] .. offsets[(t + 1) * B]`` contain ``p``, dequantized to FP16.
A weight row is ``[fp16 scale, fp16 shift]`` followed by ``D / 2`` bytes of
4-bit values, padded to a multiple of 16 bytes. The values are permuted within
each 32-bit word: nibble ``j`` (counting from the low bits) holds column
``8 * w + [0, 4, 2, 6, 1, 5, 3, 7][j]``. Each value is
``fma(q, scale, shift)`` rounded once to FP16. Rows with a negative index are
zeros.

The kernel is plain Triton so TritonCC can compile it ahead of time.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..._catalog import InvalidInput

BLOCK_P = 32
NUM_WARPS = 2
NUM_STAGES = 1
SUPPORTED_D = (64, )
MAX_TABLES = 128


def row_bytes(D: int) -> int:
    return (4 + D // 2 + 15) // 16 * 16


def t_pad(T: int) -> int:
    return max(8, triton.next_power_of_2(T))


@triton.jit
def _fma_f16(q, scale, shift):
    return tl.inline_asm_elementwise(
        "fma.rn.f16 $0, $1, $2, $3;",
        "=h,h,h,h",
        [q, scale, shift],
        dtype=tl.float16,
        is_pure=True,
        pack=1,
    )


@triton.jit
def int4_embedding_kernel(
    W,
    weights_offsets,
    indices,
    offsets,
    out,
    N,
    B,
    T,
    D: tl.constexpr,
    T_PAD: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    ROW_BYTES: tl.constexpr = (4 + D // 2 + 15) // 16 * 16
    WORDS: tl.constexpr = D // 8

    p = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
    in_range = p < N

    # Table of each row: the last table whose first row is at or before p.
    tt = tl.arange(0, T_PAD)
    starts = tl.load(offsets + tt * B, mask=(tt >= 1) & (tt < T), other=2147483647)
    t = tl.sum((p[:, None] >= starts[None, :]).to(tl.int32), axis=1)

    idx = tl.load(indices + p, mask=in_range, other=-1)
    valid = in_range & (idx >= 0)
    table_base = tl.load(weights_offsets + t, mask=in_range, other=0)
    row = (W + table_base + idx.to(tl.int64) * ROW_BYTES).to(tl.pointer_type(tl.int32))

    header = tl.load(row, mask=valid, other=0)
    words = tl.load(row[:, None] + 1 + tl.arange(0, WORDS)[None, :], mask=valid[:, None], other=0)
    scale = (header & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
    shift = ((header >> 16) & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)

    n0 = words & 0xF
    n1 = (words >> 4) & 0xF
    n2 = (words >> 8) & 0xF
    n3 = (words >> 12) & 0xF
    n4 = (words >> 16) & 0xF
    n5 = (words >> 20) & 0xF
    n6 = (words >> 24) & 0xF
    n7 = (words >> 28) & 0xF
    # [P, WORDS, 8] -> [P, D]. Each join adds a minor dimension, so the nesting
    # orders a word's nibbles 0, 4, 2, 6, 1, 5, 3, 7: the row layout's permutation.
    q = tl.join(tl.join(tl.join(n0, n1), tl.join(n2, n3)), tl.join(tl.join(n4, n5), tl.join(n6, n7)))
    q = tl.reshape(q, (BLOCK_P, D)).to(tl.float16)
    vals = _fma_f16(q, scale[:, None], shift[:, None])
    vals = tl.where(valid[:, None], vals, 0.0)

    cols = tl.arange(0, D)
    tl.store(out + p.to(tl.int64)[:, None] * D + cols[None, :], vals, mask=in_range[:, None])


def int4_embedding(weights, weights_offsets, indices, offsets, D, *, out=None, space="heuristic"):
    if space != "heuristic":
        raise InvalidInput(f"tlx.ops.int4_embedding does not provide space={space!r}")
    if D not in SUPPORTED_D:
        raise InvalidInput(f"tlx.ops.int4_embedding supports D in {SUPPORTED_D}, got {D}")
    T = weights_offsets.numel()
    if not 1 <= T <= MAX_TABLES:
        raise InvalidInput(f"tlx.ops.int4_embedding supports 1 to {MAX_TABLES} tables, got {T}")
    if (offsets.numel() - 1) % T != 0:
        raise InvalidInput("tlx.ops.int4_embedding needs T * B + 1 offsets")
    N = indices.numel()
    B = (offsets.numel() - 1) // T
    if out is None:
        out = torch.empty((N, D), dtype=torch.float16, device=weights.device)
    elif out.shape != (N, D) or out.dtype != torch.float16 or not out.is_contiguous():
        raise InvalidInput(f"tlx.ops.int4_embedding out must be a contiguous FP16 [{N}, {D}] tensor")
    if N == 0:
        return out
    int4_embedding_kernel[(triton.cdiv(N, BLOCK_P), )](
        weights,
        weights_offsets,
        indices,
        offsets,
        out,
        N,
        B,
        T,
        D=D,
        T_PAD=t_pad(T),
        BLOCK_P=BLOCK_P,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return out
