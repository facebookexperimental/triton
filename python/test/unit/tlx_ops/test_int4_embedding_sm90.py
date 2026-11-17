"""Hopper correctness for the INT4 embedding lookup without pooling."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from triton._internal_testing import is_hopper
from triton.tlx.ops import InvalidInput, int4_embedding
from triton.tlx.ops.kernels.int4_embedding._shapes import CORRECTNESS_SHAPES

pytestmark = pytest.mark.skipif(not is_hopper(), reason="Requires sm90")

_PERM = [0, 4, 2, 6, 1, 5, 3, 7]


def _row_bytes(D):
    return (4 + D // 2 + 15) // 16 * 16


def _tables(T, E, D, scale=None, shift=None):
    """Random INT4 tables in the kernel's row layout, one contiguous uint8 buffer."""
    rb = _row_bytes(D)
    q = torch.randint(0, 16, (T, E, D), dtype=torch.uint8)
    if scale is None:
        scale = torch.empty(T, E).uniform_(-0.05, 0.05).half()
    if shift is None:
        shift = torch.empty(T, E).uniform_(-0.05, 0.05).half()
    rows = torch.zeros(T, E, rb, dtype=torch.uint8)
    rows[..., 0:2] = scale.view(torch.uint8).view(T, E, 2)
    rows[..., 2:4] = shift.view(torch.uint8).view(T, E, 2)
    # INT4 rows are permuted: nibble j of each 32-bit word holds column 8 * w + _PERM[j].
    nibbles = q.view(T, E, D // 8, 8)[..., _PERM].reshape(T, E, D)
    rows[..., 4:4 + D // 2] = nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)
    weights_offsets = torch.arange(T, dtype=torch.int64) * E * rb
    return rows.flatten().cuda(), weights_offsets.cuda(), q, scale, shift


def _requests(T, lengths, E, pruned=0.0):
    offsets = torch.zeros(lengths.numel() + 1, dtype=torch.int32)
    offsets[1:] = torch.cumsum(lengths.flatten(), 0)
    indices = torch.randint(0, E, (int(offsets[-1]), ), dtype=torch.int32)
    if pruned:
        indices[torch.rand(indices.numel()) < pruned] = -1
    return indices, offsets


def _reference(q, scale, shift, indices, offsets, T):
    """fma(q, scale, shift) rounded once to fp16: exact in float64, then one rounding."""
    B = (offsets.numel() - 1) // T
    table = torch.searchsorted(offsets[torch.arange(1, T) * B], torch.arange(indices.numel()), right=True)
    idx = indices.long().clamp(min=0)
    vals = q[table, idx].double() * scale[table, idx].double()[:, None] + shift[table, idx].double()[:, None]
    vals[indices < 0] = 0
    # numpy rounds float64 to float16 directly; torch goes through float32, rounding twice.
    return torch.from_numpy(vals.numpy().astype(np.float16))


def _check(T, lengths, E, D=64, pruned=0.0, scale=None, shift=None):
    weights, weights_offsets, q, scale, shift = _tables(T, E, D, scale, shift)
    indices, offsets = _requests(T, lengths, E, pruned)
    out = int4_embedding(weights, weights_offsets, indices.cuda(), offsets.cuda(), D)
    expected = _reference(q, scale, shift, indices, offsets, T)
    assert out.shape == expected.shape and out.dtype == torch.float16
    assert torch.equal(out.cpu().view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("tables, batch, rows_per_bag, dim, rows_per_table", CORRECTNESS_SHAPES)
def test_lookup(tables, batch, rows_per_bag, dim, rows_per_table):
    torch.manual_seed(0)
    _check(tables, torch.full((tables * batch, ), rows_per_bag), min(rows_per_table, 200_000), dim)


def test_skewed_and_empty_bags():
    torch.manual_seed(1)
    lengths = torch.randint(0, 3400, (24 * 6, ))
    lengths[::3] = 0
    _check(24, lengths, 50_000)


def test_pruned_indices():
    torch.manual_seed(2)
    _check(5, torch.full((5 * 4, ), 300), 10_000, pruned=0.3)


def test_no_indices():
    _check(4, torch.zeros(4 * 2, dtype=torch.long), 100)


def test_rounding():
    # Scale and shift far apart in magnitude, and subnormal scales.
    torch.manual_seed(3)
    T, E = 2, 3000
    n = T * E
    scale = torch.cat([
        torch.full((n // 3, ), 2.0**-20),
        torch.empty(n // 3).uniform_(-64, 64),
        torch.empty(n - 2 * (n // 3)).uniform_(-1e-3, 1e-3),
    ]).half().view(T, E)
    shift = torch.cat([torch.empty(n // 2).uniform_(-1000, 1000),
                       torch.empty(n - n // 2).uniform_(-1e-4, 1e-4)])[torch.randperm(n)].half().view(T, E)
    _check(T, torch.full((T * 4, ), 3000), E, scale=scale, shift=shift)


def test_invalid_inputs():
    weights, weights_offsets, *_ = _tables(2, 10, 64)
    indices, offsets = _requests(2, torch.full((2 * 2, ), 3), 10)
    indices, offsets = indices.cuda(), offsets.cuda()
    with pytest.raises(InvalidInput):
        int4_embedding(weights, weights_offsets, indices, offsets, 128)
    with pytest.raises(InvalidInput):
        int4_embedding(weights, weights_offsets, indices.long(), offsets, 64)
    with pytest.raises(InvalidInput):
        int4_embedding(weights, weights_offsets, indices, offsets, 64, space="full")
    with pytest.raises(InvalidInput):
        int4_embedding(weights, weights_offsets, indices, offsets[:-1], 64)
    misaligned = torch.cat([indices.new_zeros(1), indices])[1:]
    with pytest.raises(InvalidInput):
        int4_embedding(weights, weights_offsets, misaligned, offsets, 64)
