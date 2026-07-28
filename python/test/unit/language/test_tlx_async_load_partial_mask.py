# Owner(s): ["module: inductor"]
# Regression (gfx950 / AMD MI350X): a partially masked tlx.async_load must lower and
# preserve the valid values. The synchronous tail-load fallback remains covered too.
import unittest

import torch
import triton  # @manual
import triton.language as tl  # @manual

# Use "cuda" directly instead of torch.testing._internal.inductor_utils.GPU_TYPE,
# which is not available in the py_unit_language_tests dep graph.
GPU_TYPE = "cuda"


def has_tlx() -> bool:
    try:
        import triton.language.extra.tlx  # noqa: F401  # @manual

        return True
    except ImportError:
        return False


def is_gfx950() -> bool:
    if torch.version.hip is None:
        return False
    try:
        return "gfx95" in torch.cuda.get_device_properties(0).gcnArchName
    except Exception:
        return False


if has_tlx():
    import triton.language.extra.tlx as tlx  # @manual

    @triton.jit
    def _async_load_kernel(
        a_ptr,
        out_ptr,
        VALID_K: tl.constexpr,
        USE_MASK: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        offs_m = tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, BLOCK_K)
        offs = offs_m[:, None] * BLOCK_K + offs_k[None, :]
        smem = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_ptr), 1)
        if USE_MASK:
            tok = tlx.async_load(
                a_ptr + offs,
                tlx.local_view(smem, 0),
                mask=offs_k[None, :] < VALID_K,
                other=0.0,
            )
        else:
            tok = tlx.async_load(a_ptr + offs, tlx.local_view(smem, 0))
        tlx.async_load_commit_group([tok])
        tlx.async_load_wait_group(0)
        t = tlx.local_load(tlx.local_view(smem, 0))
        tl.store(out_ptr + offs, t)

    @triton.jit
    def _sync_load_kernel(
        a_ptr,
        out_ptr,
        VALID_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        # The "sync-load the tail" fix: a masked *synchronous* tl.load lowers fine, so a
        # partial K-tile is handled by tl.load instead of tlx.async_load.
        offs_m = tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, BLOCK_K)
        offs = offs_m[:, None] * BLOCK_K + offs_k[None, :]
        t = tl.load(a_ptr + offs, mask=offs_k[None, :] < VALID_K, other=0.0)
        tl.store(out_ptr + offs, t)


@unittest.skipIf(not is_gfx950(), "Need AMD MI350X (gfx950)")
@unittest.skipIf(not has_tlx(), "TLX not available")
class TlxAsyncLoadPartialMaskTest(unittest.TestCase):
    BLOCK_M = 128
    BLOCK_K = 64

    def _a_out(self):
        a = torch.randn(self.BLOCK_M, self.BLOCK_K, device=GPU_TYPE, dtype=torch.float16)
        return a, torch.full_like(a, float("nan"))

    def test_async_load_no_mask_ok(self):
        a, out = self._a_out()
        _async_load_kernel[(1, )](
            a,
            out,
            VALID_K=self.BLOCK_K,
            USE_MASK=False,
            BLOCK_M=self.BLOCK_M,
            BLOCK_K=self.BLOCK_K,
            num_warps=4,
        )
        torch.testing.assert_close(out, a)

    def test_async_load_all_true_mask_ok(self):
        # mask present but all-true (like an ALIGNED K, where every K-tile is full).
        a, out = self._a_out()
        _async_load_kernel[(1, )](
            a,
            out,
            VALID_K=self.BLOCK_K,
            USE_MASK=True,
            BLOCK_M=self.BLOCK_M,
            BLOCK_K=self.BLOCK_K,
            num_warps=4,
        )
        torch.testing.assert_close(out, a)

    def test_async_load_partial_mask_ok(self):
        a, out = self._a_out()
        _async_load_kernel[(1, )](
            a,
            out,
            VALID_K=5,
            USE_MASK=True,
            BLOCK_M=self.BLOCK_M,
            BLOCK_K=self.BLOCK_K,
            num_warps=4,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(out[:, :5], a[:, :5])
        torch.testing.assert_close(out[:, 5:], torch.zeros_like(out[:, 5:]))

    def test_sync_load_partial_mask_fix_ok(self):
        # THE FIX: the same partial mask via a synchronous tl.load compiles and is correct.
        a, out = self._a_out()
        _sync_load_kernel[(1, )](
            a,
            out,
            VALID_K=5,
            BLOCK_M=self.BLOCK_M,
            BLOCK_K=self.BLOCK_K,
            num_warps=4,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(out[:, :5], a[:, :5])
        torch.testing.assert_close(out[:, 5:], torch.zeros_like(out[:, 5:]))


if __name__ == "__main__":
    unittest.main()
