import inspect
import os

import pytest
import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton._internal_testing import is_blackwell

ROWS = 128
COLS = 64
ROWS_PER_WARP = 32


@triton.jit
def _rescale_tmem(acc, alpha):
    tlx.local_store(acc, tlx.local_load(acc) * alpha[:, None])


@triton.jit
def _per_warp_rescale_kernel(x_ptr, alpha_ptr, y_ptr, ROWS: tl.constexpr, COLS: tl.constexpr):
    rows = tl.arange(0, ROWS)
    offsets = rows[:, None] * COLS + tl.arange(0, COLS)[None, :]
    alpha = tl.load(alpha_ptr + rows)
    acc = tlx.local_alloc((ROWS, COLS), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)[0]
    tlx.local_store(acc, tl.load(x_ptr + offsets))
    # Each warp decides for its own rows; the warps need not agree.
    tlx.warp_predicate(tlx.warp_any(alpha < 1.0), (), _rescale_tmem, args=(acc, alpha), wave_uniform=True)
    tl.store(y_ptr + offsets, tlx.local_load(acc))


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_warp_predicate_per_warp_tmem_rescale(device):
    torch.manual_seed(0)
    x = torch.randn((ROWS, COLS), dtype=torch.float32, device=device)
    alpha = torch.full((ROWS, ), 2.0, dtype=torch.float32, device=device)
    # Warps 0 and 2 have a row below 1, so they rescale all their rows; warps 1 and 3 skip the region.
    alpha[5] = 0.5
    alpha[2 * ROWS_PER_WARP + 7] = 0.25
    y = torch.empty_like(x)
    _per_warp_rescale_kernel[(1, )](x, alpha, y, ROWS, COLS, num_warps=4)

    rescaled = (alpha < 1.0).view(-1, ROWS_PER_WARP).any(dim=1).repeat_interleave(ROWS_PER_WARP)
    expected = torch.where(rescaled[:, None], x * alpha[:, None], x)
    torch.testing.assert_close(y, expected)


@triton.jit
def _named_barrier_body():
    tlx.named_barrier_wait(9, 128)


@triton.jit
def _named_barrier_kernel(x_ptr, ROWS: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, ROWS))
    tlx.warp_predicate(tlx.warp_any(x < 0.0), (), _named_barrier_body, wave_uniform=True)


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_warp_predicate_rejects_named_barrier(device, capfd):
    x = torch.randn((ROWS, ), dtype=torch.float32, device=device)
    with pytest.raises(Exception, match="is not yet supported in predicated regions on NVIDIA"):
        _named_barrier_kernel[(1, )](x, ROWS, num_warps=4)
    # The diagnostic points at the tlx.warp_predicate call.
    lines, first = inspect.getsourcelines(_named_barrier_kernel.fn)
    call_line = first + next(i for i, line in enumerate(lines) if "tlx.warp_predicate" in line)
    assert f"{os.path.basename(__file__)}:{call_line}:" in capfd.readouterr().err


@triton.jit
def _lane_predicated_tmem_kernel(x_ptr, alpha_ptr, y_ptr, ROWS: tl.constexpr, COLS: tl.constexpr):
    rows = tl.arange(0, ROWS)
    offsets = rows[:, None] * COLS + tl.arange(0, COLS)[None, :]
    alpha = tl.load(alpha_ptr + rows)
    acc = tlx.local_alloc((ROWS, COLS), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)[0]
    tlx.local_store(acc, tl.load(x_ptr + offsets))
    tlx.warp_predicate(alpha < 1.0, (), _rescale_tmem, args=(acc, alpha))
    tl.store(y_ptr + offsets, tlx.local_load(acc))


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_warp_predicate_rejects_non_wave_uniform(device):
    x = torch.randn((ROWS, COLS), dtype=torch.float32, device=device)
    alpha = torch.rand((ROWS, ), dtype=torch.float32, device=device)
    y = torch.empty_like(x)
    with pytest.raises(
            Exception,
            match="per-lane and non-wave-uniform predicates are not yet supported in predicated regions on NVIDIA"):
        _lane_predicated_tmem_kernel[(1, )](x, alpha, y, ROWS, COLS, num_warps=4)
