"""TRITON_USE_META_WS only changes kernels that request warp specialization."""

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_cuda


def _is_blackwell():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


@triton.jit
def _matmul(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
            WARP_SPECIALIZE: tl.constexpr):
    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_M, BLOCK_K])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[K, N], strides=[N, 1], block_shape=[BLOCK_K, BLOCK_N])
    c_desc = tl.make_tensor_descriptor(c_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N])
    off_m = tl.program_id(0) * BLOCK_M
    off_n = tl.program_id(1) * BLOCK_N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, K, BLOCK_K, warp_specialize=WARP_SPECIALIZE):
        acc = tl.dot(a_desc.load([off_m, k]), b_desc.load([k, off_n]), acc)
    c_desc.store([off_m, off_n], acc.to(tl.float16))


def _compile(monkeypatch, use_meta_ws, warp_specialize):
    monkeypatch.setenv("TRITON_USE_META_WS", "1" if use_meta_ws else "0")
    M = N = K = 512
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.empty(M, N, device="cuda", dtype=torch.float16)
    triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda"))
    # The in-memory kernel cache key does not include the knob.
    _matmul.device_caches.clear()
    with triton.knobs.compilation.scope():
        triton.knobs.compilation.always_compile = True
        kernel = _matmul.warmup(a, b, c, M, N, K, BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
                                WARP_SPECIALIZE=warp_specialize, num_warps=4, num_stages=3, grid=(1, ))
    return kernel.asm["ttgir"]


@pytest.mark.skipif(not _is_blackwell(), reason="Meta AutoWS pipeline under test is Blackwell")
def test_non_ws_kernel_compiles_as_without_meta_ws(monkeypatch):
    assert _compile(monkeypatch, True, False) == _compile(monkeypatch, False, False)

