# NVIDIA tt.dot with a non-power-of-2 result M or N (TRITON_ALLOW_NPOT=1), compile
# only: accelerate-matmul rejects it by name instead of aborting in a later pass.

import pytest

import triton
import triton.language as tl
from triton import knobs
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from triton._internal_testing import is_cuda

requires_npot = pytest.mark.skipif(not knobs.language.allow_npot, reason="requires TRITON_ALLOW_NPOT=1")
requires_gpu = pytest.mark.skipif(not is_cuda(), reason="requires CUDA GPU")


@triton.jit
def _dot_kernel(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    m = tl.arange(0, M)
    n = tl.arange(0, N)
    k = tl.arange(0, K)
    a = tl.load(A + m[:, None] * K + k[None, :])
    b = tl.load(B + k[:, None] * N + n[None, :])
    tl.store(C + m[:, None] * N + n[None, :], tl.dot(a, b))


def _compile_dot(M, N, K, arch, num_ctas):
    signature = {"A": "*fp16", "B": "*fp16", "C": "*fp32", "M": "constexpr", "N": "constexpr", "K": "constexpr"}
    src = ASTSource(_dot_kernel, signature, {"M": M, "N": N, "K": K})
    return triton.compile(src, target=GPUTarget("cuda", arch, 32), options={"num_warps": 4, "num_ctas": num_ctas})


@requires_gpu
@requires_npot
def test_npot_nvidia_dot_result_rejected():
    with pytest.raises(RuntimeError, match="NPOT NVIDIA dot: no MMA or FMA lowering admits a result M or N extent"):
        _compile_dot(64, 96, 64, 90, 1)


@requires_gpu
@requires_npot
def test_pow2_control_nvidia_dot_compiles():
    assert "ttng.warp_group_dot" in _compile_dot(64, 64, 64, 90, 1).asm["ttgir"]
