"""Correctness tests for TMA TensorDescriptor with C fast cache + _TritonDispatcher.

Verifies that kernels using TMA-path TensorDescriptors (nvTmaDesc) produce
correct results when launched via:
  1. C fast cache (c_cache=True) — cache key correctly built
  2. _TritonDispatcher (TRITON_USE_TRITON_DISPATCHER=1) — nvTmaDesc dispatched
  3. Repeated calls hit cache and still produce correct results
"""

import os
from unittest import TestCase
from unittest.mock import patch

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


def _get_device():
    """Get the active torch device (deferred to avoid module-scope side effects)."""
    return triton.runtime.driver.active.get_active_torch_device()


@triton.jit(c_cache=True)
def _tma_load_kernel(out_ptr, desc, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
    """Load a block from TensorDescriptor (TMA) and store to output."""
    block = desc.load([0, 0])
    idx = tl.arange(0, M_BLOCK)[:, None] * N_BLOCK + tl.arange(0, N_BLOCK)[None, :]
    tl.store(out_ptr + idx, block)


@triton.jit(c_cache=False)
def _tma_load_kernel_nocache(out_ptr, desc, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
    """Same kernel without c_cache — used as correctness reference."""
    block = desc.load([0, 0])
    idx = tl.arange(0, M_BLOCK)[:, None] * N_BLOCK + tl.arange(0, N_BLOCK)[None, :]
    tl.store(out_ptr + idx, block)


@triton.jit(c_cache=True)
def _tma_load_offset_kernel(out_ptr, desc, row_off, col_off, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
    """Load a block at a given offset."""
    block = desc.load([row_off, col_off])
    idx = tl.arange(0, M_BLOCK)[:, None] * N_BLOCK + tl.arange(0, N_BLOCK)[None, :]
    tl.store(out_ptr + idx, block)


class TestTmaTensorDescCorrectness(TestCase):
    """Verify TMA TensorDescriptor produces correct results via fast cache + dispatcher."""

    def setUp(self):
        patcher = patch.dict(os.environ, {"TRITON_USE_TRITON_DISPATCHER": "1"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_basic_load_correctness(self):
        """TMA load through fast cache + dispatcher produces correct output."""
        device = _get_device()
        M_BLOCK, N_BLOCK = 8, 32
        M, N = M_BLOCK * 3, N_BLOCK * 4

        # Input tensor with known values
        src = torch.arange(M * N, device=device, dtype=torch.float16).reshape(M, N)
        out = torch.zeros(M_BLOCK, N_BLOCK, device=device, dtype=torch.float16)
        desc = TensorDescriptor(src, shape=src.shape, strides=src.stride(), block_shape=[M_BLOCK, N_BLOCK])

        _tma_load_kernel[(1, )](out, desc, M, N, M_BLOCK, N_BLOCK)

        # The kernel loads block at [0,0] — should be top-left M_BLOCK x N_BLOCK
        expected = src[:M_BLOCK, :N_BLOCK]
        torch.testing.assert_close(out, expected)

    def test_matches_nocache_reference(self):
        """Fast cache + dispatcher result matches no-cache Python path exactly."""
        device = _get_device()
        M_BLOCK, N_BLOCK = 8, 32
        M, N = M_BLOCK * 2, N_BLOCK * 2

        src = torch.randn(M, N, device=device, dtype=torch.float16)
        desc = TensorDescriptor(src, shape=src.shape, strides=src.stride(), block_shape=[M_BLOCK, N_BLOCK])

        out_cached = torch.zeros(M_BLOCK, N_BLOCK, device=device, dtype=torch.float16)
        out_nocache = torch.zeros(M_BLOCK, N_BLOCK, device=device, dtype=torch.float16)

        _tma_load_kernel[(1, )](out_cached, desc, M, N, M_BLOCK, N_BLOCK)
        _tma_load_kernel_nocache[(1, )](out_nocache, desc, M, N, M_BLOCK, N_BLOCK)

        torch.testing.assert_close(out_cached, out_nocache)

    def test_repeated_calls_still_correct(self):
        """Multiple cached calls with different data produce correct (not stale) results."""
        device = _get_device()
        M_BLOCK, N_BLOCK = 8, 32
        M, N = M_BLOCK * 2, N_BLOCK * 2

        for i in range(5):
            src = torch.full((M, N), fill_value=float(i + 1), device=device, dtype=torch.float16)
            out = torch.zeros(M_BLOCK, N_BLOCK, device=device, dtype=torch.float16)
            desc = TensorDescriptor(
                src,
                shape=src.shape,
                strides=src.stride(),
                block_shape=[M_BLOCK, N_BLOCK],
            )

            _tma_load_kernel[(1, )](out, desc, M, N, M_BLOCK, N_BLOCK)

            expected = torch.full(
                (M_BLOCK, N_BLOCK),
                fill_value=float(i + 1),
                device=device,
                dtype=torch.float16,
            )
            torch.testing.assert_close(out, expected, msg=f"Failed on iteration {i}")

    def test_different_tensors_same_desc_shape(self):
        """Different backing tensors with same desc shape use cache but read correct data."""
        device = _get_device()
        M_BLOCK, N_BLOCK = 8, 32
        M, N = M_BLOCK * 2, N_BLOCK * 2

        src_a = torch.ones(M, N, device=device, dtype=torch.float16) * 42.0
        src_b = torch.ones(M, N, device=device, dtype=torch.float16) * 7.0

        desc_a = TensorDescriptor(
            src_a,
            shape=src_a.shape,
            strides=src_a.stride(),
            block_shape=[M_BLOCK, N_BLOCK],
        )
        desc_b = TensorDescriptor(
            src_b,
            shape=src_b.shape,
            strides=src_b.stride(),
            block_shape=[M_BLOCK, N_BLOCK],
        )

        # First call — populates cache
        @triton.jit(c_cache=True)
        def _load_kernel(out_ptr, desc, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
            block = desc.load([0, 0])
            idx = (tl.arange(0, M_BLOCK)[:, None] * N_BLOCK + tl.arange(0, N_BLOCK)[None, :])
            tl.store(out_ptr + idx, block)

        out_a = torch.zeros(M_BLOCK, N_BLOCK, device=device, dtype=torch.float16)
        out_b = torch.zeros(M_BLOCK, N_BLOCK, device=device, dtype=torch.float16)

        _load_kernel[(1, )](out_a, desc_a, M, N, M_BLOCK, N_BLOCK)
        _load_kernel[(1, )](out_b, desc_b, M, N, M_BLOCK, N_BLOCK)

        # Both should read their own data, NOT stale cached values
        torch.testing.assert_close(out_a, torch.full_like(out_a, 42.0))
        torch.testing.assert_close(out_b, torch.full_like(out_b, 7.0))

    def test_offset_load_correctness(self):
        """TMA load at non-zero offset produces correct block."""
        device = _get_device()
        M_BLOCK, N_BLOCK = 8, 32
        M, N = M_BLOCK * 4, N_BLOCK * 4

        src = torch.arange(M * N, device=device, dtype=torch.float16).reshape(M, N)
        desc = TensorDescriptor(src, shape=src.shape, strides=src.stride(), block_shape=[M_BLOCK, N_BLOCK])

        out = torch.zeros(M_BLOCK, N_BLOCK, device=device, dtype=torch.float16)
        row_off = M_BLOCK  # load second block row
        col_off = N_BLOCK  # load second block col

        _tma_load_offset_kernel[(1, )](out, desc, row_off, col_off, M, N, M_BLOCK, N_BLOCK)

        expected = src[row_off:row_off + M_BLOCK, col_off:col_off + N_BLOCK]
        torch.testing.assert_close(out, expected)


class TestTmaCacheKeyDiscrimination(TestCase):
    """Verify the C fast cache correctly distinguishes TMA TensorDescriptor specializations."""

    def setUp(self):
        patcher = patch.dict(os.environ, {"TRITON_USE_TRITON_DISPATCHER": "1"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_different_dtype_different_compilation(self):
        """fp16 vs fp32 TMA TensorDescriptors must compile different kernels."""
        device = _get_device()
        M_BLOCK, N_BLOCK = 8, 32
        M, N = M_BLOCK * 2, N_BLOCK * 2

        @triton.jit(c_cache=True)
        def _dtype_kernel(out_ptr, desc, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
            block = desc.load([0, 0])
            idx = (tl.arange(0, M_BLOCK)[:, None] * N_BLOCK + tl.arange(0, N_BLOCK)[None, :])
            tl.store(out_ptr + idx, block)

        src_fp16 = torch.ones(M, N, device=device, dtype=torch.float16) * 3.0
        src_fp32 = torch.ones(M, N, device=device, dtype=torch.float32) * 5.0

        desc_fp16 = TensorDescriptor(
            src_fp16,
            shape=src_fp16.shape,
            strides=src_fp16.stride(),
            block_shape=[M_BLOCK, N_BLOCK],
        )
        desc_fp32 = TensorDescriptor(
            src_fp32,
            shape=src_fp32.shape,
            strides=src_fp32.stride(),
            block_shape=[M_BLOCK, N_BLOCK],
        )

        out_fp16 = torch.zeros(M_BLOCK, N_BLOCK, device=device, dtype=torch.float16)
        out_fp32 = torch.zeros(M_BLOCK, N_BLOCK, device=device, dtype=torch.float32)

        _dtype_kernel[(1, )](out_fp16, desc_fp16, M, N, M_BLOCK, N_BLOCK)
        _dtype_kernel[(1, )](out_fp32, desc_fp32, M, N, M_BLOCK, N_BLOCK)

        # Each should produce correct output for its dtype
        torch.testing.assert_close(out_fp16, torch.full_like(out_fp16, 3.0))
        torch.testing.assert_close(out_fp32, torch.full_like(out_fp32, 5.0))

    def test_different_block_shape_different_compilation(self):
        """Different block_shapes must compile different kernels (different constexpr)."""
        device = _get_device()

        src = torch.arange(64 * 128, device=device, dtype=torch.float16).reshape(64, 128)

        @triton.jit(c_cache=True)
        def _block_kernel(out_ptr, desc, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
            block = desc.load([0, 0])
            idx = (tl.arange(0, M_BLOCK)[:, None] * N_BLOCK + tl.arange(0, N_BLOCK)[None, :])
            tl.store(out_ptr + idx, block)

        # Block shape 8x32
        desc_8x32 = TensorDescriptor(src, shape=src.shape, strides=src.stride(), block_shape=[8, 32])
        out_8x32 = torch.zeros(8, 32, device=device, dtype=torch.float16)
        _block_kernel[(1, )](out_8x32, desc_8x32, 64, 128, 8, 32)
        torch.testing.assert_close(out_8x32, src[:8, :32])

        # Block shape 16x64
        desc_16x64 = TensorDescriptor(src, shape=src.shape, strides=src.stride(), block_shape=[16, 64])
        out_16x64 = torch.zeros(16, 64, device=device, dtype=torch.float16)
        _block_kernel[(1, )](out_16x64, desc_16x64, 64, 128, 16, 64)
        torch.testing.assert_close(out_16x64, src[:16, :64])
