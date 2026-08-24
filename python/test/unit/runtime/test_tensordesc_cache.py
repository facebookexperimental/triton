# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-ignore-all-errors
"""Test that the C fast cache correctly differentiates TensorDescriptor specializations.

If fc_build_key doesn't properly key on (dtype, block_shape), the cache would
return a stale compiled kernel when switching between different TensorDescriptor
types — a silent correctness bug.
"""

import os
from unittest import TestCase
from unittest.mock import patch

import torch
import triton
from triton.tools.tensor_descriptor import TensorDescriptor


def _get_device():
    """Get the active torch device (deferred to avoid module-scope side effects)."""
    return triton.runtime.driver.active.get_active_torch_device()


@triton.jit(c_cache=True)
def nop_tensordesc_kernel(desc, out_ptr):
    """Kernel that takes a tensordesc and a scalar output."""
    pass


def _get_kernel_cache(jit_fn):
    """Get the Python-level kernel compilation cache dict for the current device."""
    device = triton.runtime.driver.active.get_current_device()
    kernel_cache, _, _, _, _ = jit_fn.device_caches[device]
    return kernel_cache


class TestTensorDescCacheKey(TestCase):
    """Verify the C fast cache distinguishes different TensorDescriptor types."""

    def setUp(self):
        """Ensure c_cache (fast path) is active without leaking env state."""
        patcher = patch.dict(os.environ, {"TRITON_USE_C_DISPATCHER": "1"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_different_dtype_different_cache_entry(self):
        """fp16 and fp32 TensorDescriptors must produce different cache keys."""

        fp16_tensor = torch.zeros(32, dtype=torch.float16, device=_get_device())
        fp32_tensor = torch.zeros(32, dtype=torch.float32, device=_get_device())
        out = torch.zeros(1, device=_get_device())

        desc_fp16 = TensorDescriptor(fp16_tensor, [32], [1], [32])
        desc_fp32 = TensorDescriptor(fp32_tensor, [32], [1], [32])

        # First call — compiles for fp16
        nop_tensordesc_kernel[(1, )](desc_fp16, out)
        cache = _get_kernel_cache(nop_tensordesc_kernel)
        cache_size_after_fp16 = len(cache)
        self.assertEqual(cache_size_after_fp16, 1)

        # Second call — should compile a NEW kernel for fp32 (cache miss)
        nop_tensordesc_kernel[(1, )](desc_fp32, out)
        cache_size_after_fp32 = len(cache)
        self.assertEqual(
            cache_size_after_fp32,
            2,
            "C fast cache should distinguish fp16 vs fp32 TensorDescriptor — "
            "got same cache size, indicates false cache hit",
        )

    def test_different_block_shape_different_cache_entry(self):
        """Different block_shapes must produce different cache keys."""

        tensor = torch.zeros(64, dtype=torch.float16, device=_get_device())
        out = torch.zeros(1, device=_get_device())

        desc_32 = TensorDescriptor(tensor, [64], [1], [32])
        desc_64 = TensorDescriptor(tensor, [64], [1], [64])

        @triton.jit(c_cache=True)
        def nop_block_shape_kernel(desc, out_ptr):
            pass

        # First call with block_shape=[32]
        nop_block_shape_kernel[(1, )](desc_32, out)
        cache = _get_kernel_cache(nop_block_shape_kernel)
        self.assertEqual(len(cache), 1)

        # Second call with block_shape=[64] — must be a different entry
        nop_block_shape_kernel[(1, )](desc_64, out)
        self.assertEqual(
            len(cache),
            2,
            "C fast cache should distinguish different block_shapes — "
            "got same cache size, indicates false cache hit",
        )

    def test_same_tensordesc_type_hits_cache(self):
        """Same dtype + block_shape should HIT the cache (no recompile)."""

        tensor_a = torch.ones(32, dtype=torch.float16, device=_get_device())
        tensor_b = torch.zeros(32, dtype=torch.float16, device=_get_device())
        out = torch.zeros(1, device=_get_device())

        # Two different tensors, same dtype + block_shape
        desc_a = TensorDescriptor(tensor_a, [32], [1], [32])
        desc_b = TensorDescriptor(tensor_b, [32], [1], [32])

        # Use a fresh kernel to avoid state from other tests
        @triton.jit(c_cache=True)
        def nop_cache_hit_kernel(desc, out_ptr):
            pass

        nop_cache_hit_kernel[(1, )](desc_a, out)
        cache = _get_kernel_cache(nop_cache_hit_kernel)
        self.assertEqual(len(cache), 1)

        # Should reuse the same compiled kernel
        nop_cache_hit_kernel[(1, )](desc_b, out)
        self.assertEqual(
            len(cache),
            1,
            "Same dtype + block_shape should hit cache, not trigger recompile",
        )
