# Owner(s): ["module: inductor"]
# Repro (gfx950 / AMD MI350X, T280910119): an UNMASKED, full-tile tlx.async_load whose global ROW
# STRIDE is not 16-byte aligned fails to legalize -> "failed to translate module to LLVM IR"
# (builtin.unrealized_conversion_cast). The fp16/bf16 padded direct-to-LDS path lowers to 128-bit
# (16-byte) transactions, so each row start must be 16-byte aligned: the row stride K*itemsize bytes
# must be a multiple of 16, i.e. K % 8 == 0 for fp16/bf16. An ODD K (e.g. 2309) has a row stride of
# K*2 bytes that is never 16-byte aligned, so it fails for ANY config. (The exact aligned threshold
# can be config-dependent -- a wider vectorization may need more than 16-byte -- so this test only
# asserts the config-independent odd-K failure, not a specific K%8-but-not-K%16 value.)
#
# This is DISTINCT from the K % BLOCK_K partial-mask failure fixed by the "sync-load the tail"
# change (see test_tlx_async_load_partial_mask.py). There the last K-tile is partial and the
# MASKED async_load can't lower; the sync-tail replaces it with a tl.load. Here every FULL tile
# fails because of the row-stride alignment, so the sync-tail does NOT help -- the fix is an
# AMD-backend async_copy path for non-16-aligned strides (T280910119). It is also why the TLX
# addmm/bmm heuristics gate on (K*itemsize) % 16 == 0 (K % 8 for fp16) and decline others to rocBLAS.
import unittest

import torch
import triton  # @manual
import triton.language as tl  # @manual
from torch.testing._internal.inductor_utils import GPU_TYPE


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
    def _row_stride_async_load(
        a_ptr,
        out_ptr,
        stride_am,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        # Load a full (BLOCK_M, BLOCK_K) tile from rows strided by stride_am (= the matrix's K
        # for a row-major [M, K] operand). No mask: this is a FULL, in-bounds tile.
        offs_m = tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, BLOCK_K)
        offs = offs_m[:, None] * stride_am + offs_k[None, :]
        smem = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_ptr), 1)
        tok = tlx.async_load(a_ptr + offs, tlx.local_view(smem, 0))
        tlx.async_load_commit_group([tok])
        tlx.async_load_wait_group(0)
        t = tlx.local_load(tlx.local_view(smem, 0))
        tl.store(out_ptr + offs_m[:, None] * BLOCK_K + offs_k[None, :], t)


@unittest.skipIf(not is_gfx950(), "Need AMD MI350X (gfx950)")
@unittest.skipIf(not has_tlx(), "TLX not available")
class TlxAsyncLoadRowStrideTest(unittest.TestCase):
    BLOCK_M = 128
    BLOCK_K = 64

    def _launch(self, K):
        a = torch.randn(self.BLOCK_M, K, device=GPU_TYPE, dtype=torch.float16)
        out = torch.zeros(self.BLOCK_M, self.BLOCK_K, device=GPU_TYPE, dtype=torch.float16)
        _row_stride_async_load[(1,)](
            a, out, a.stride(0), BLOCK_M=self.BLOCK_M, BLOCK_K=self.BLOCK_K, num_warps=4
        )
        torch.cuda.synchronize()
        return a, out

    def test_row_stride_aligned_ok(self):
        # K=2560 (16-byte aligned row stride) -> async_load lowers + is correct.
        a, out = self._launch(2560)
        torch.testing.assert_close(out, a[:, : self.BLOCK_K])

    def test_row_stride_odd_fails_to_lower(self):
        # THE REPRO: K=2309 (odd, the production compression bmm's K) -> row stride K*2 bytes is
        # never 16-byte aligned -> unmasked async_load fails to translate to LLVM IR (for any
        # config). Flip to assert_close once the AMD-backend async_copy alignment fix (T280910119)
        # lands. NOTE: the addmm/bmm heuristics gate on (K*itemsize) % 16 == 0 (K % 8 for fp16), so
        # in the real template such shapes are declined and fall back to rocBLAS rather than reaching
        # this failure.
        with self.assertRaises(Exception):
            self._launch(2309)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
