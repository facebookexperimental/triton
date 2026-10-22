# FMA tt.dot with a non-power-of-2 K (TRITON_ALLOW_NPOT=1): the FMA lowering must unpack
# each operand with the PowerOf2Ceil(K) registers of its modular K digit. fp16 K24 takes
# FMA because no MMA K tile divides it; fp32 with input_precision="ieee" always does.

import pytest
import torch

import triton
import triton.language as tl
from triton import knobs

from triton._internal_testing import is_cuda

requires_npot = pytest.mark.skipif(not knobs.language.allow_npot, reason="requires TRITON_ALLOW_NPOT=1")
requires_gpu = pytest.mark.skipif(not is_cuda(), reason="requires CUDA GPU")


@triton.jit
def _dot_kernel(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, PREC: tl.constexpr):
    m = tl.arange(0, M)
    n = tl.arange(0, N)
    k = tl.arange(0, K)
    a = tl.load(A + m[:, None] * K + k[None, :])
    b = tl.load(B + k[:, None] * N + n[None, :])
    tl.store(C + m[:, None] * N + n[None, :], tl.dot(a, b, input_precision=PREC))


@requires_gpu
@requires_npot
@pytest.mark.parametrize("K, dtype, prec", [(24, torch.float16, None), (24, torch.float32, "ieee")])
def test_npot_fma_dot_k(K, dtype, prec):
    M, N = 64, 64
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype)
    c = torch.empty(M, N, device="cuda", dtype=torch.float32)
    h = _dot_kernel[(1, )](a, b, c, M=M, N=N, K=K, PREC=prec)
    ttgir = h.asm["ttgir"]
    assert not any(mma in ttgir for mma in ("nvidia_mma", "warp_group_dot", "tc_gen5_mma")), "expected the FMA path"
    torch.testing.assert_close(c, (a.double() @ b.double()).float(), atol=1e-3, rtol=1e-3)
