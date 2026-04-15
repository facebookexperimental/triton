"""Verify modulo scheduling on a simple GEMM kernel.

Each test runs in a subprocess to avoid JIT cache sharing between
different scheduling algorithms.
"""
import os
import subprocess
import sys

import pytest
import torch

KERNEL_SCRIPT = '''
import sys
import torch
import triton
import triton.language as tl

@triton.jit
def gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)

M, N, K = 256, 256, 256
torch.manual_seed(0)
a = torch.randn(M, K, device="cuda", dtype=torch.float16)
b = torch.randn(K, N, device="cuda", dtype=torch.float16)
c = torch.zeros(M, N, device="cuda", dtype=torch.float32)
grid = (M // 128, N // 128)
gemm_kernel[grid](
    a, b, c, M, N, K,
    a.stride(0), a.stride(1),
    b.stride(0), b.stride(1),
    c.stride(0), c.stride(1),
    BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
)
ref = a.float() @ b.float()
max_err = (c - ref).abs().max().item()
print(f"MaxErr: {max_err}")
sys.exit(0 if max_err < 1.0 else 1)
'''


def _run_in_subprocess(algo=""):
    """Run GEMM kernel in a fresh subprocess with the given schedule algo."""
    import tempfile
    env = os.environ.copy()
    env["TRITON_USE_MODULO_SCHEDULE"] = "1"
    env["TRITON_USE_META_WS"] = "1"
    env["TRITON_ALWAYS_COMPILE"] = "1"
    if algo:
        env["TRITON_MODULO_SCHEDULE_ALGO"] = algo
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(KERNEL_SCRIPT)
        f.flush()
        result = subprocess.run(
            [sys.executable, f.name],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
    os.unlink(f.name)
    return result


@pytest.mark.skipif(
    torch.cuda.get_device_capability()[0] < 10,
    reason="Modulo scheduling requires Blackwell (SM100+)",
)
class TestModuloSchedule:

    def test_sms(self):
        result = _run_in_subprocess(algo="sms")
        assert result.returncode == 0, (f"SMS failed:\nstdout: {result.stdout}\nstderr: {result.stderr[-500:]}")

    @pytest.mark.xfail(reason="Rau IMS produces incorrect schedule for tl.load-based GEMM (pre-existing)")
    def test_rau(self):
        result = _run_in_subprocess(algo="")
        assert result.returncode == 0, (f"Rau IMS failed:\nstdout: {result.stdout}\nstderr: {result.stderr[-500:]}")
