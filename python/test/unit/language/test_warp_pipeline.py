"""Tests for TLX warp-pipeline support (tlx.warp_pipeline_stage)."""

import pytest
import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


# --- Runtime test: simple GEMM with warp pipeline ---

@triton.jit
def _gemm_warp_pipeline_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in tl.range(0, tl.cdiv(K, BLOCK_K)):
        with tlx.warp_pipeline_stage("load"):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K)
        with tlx.warp_pipeline_stage("compute"):
            acc += tl.dot(a, b, allow_tf32=False)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


@pytest.mark.skipif(not is_hip(), reason="warp pipeline is AMD-only")
def test_gemm_warp_pipeline():
    """Runtime correctness test: small GEMM with warp pipeline stages."""
    M, N, K = 128, 128, 128
    BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 32

    torch.manual_seed(42)
    a = torch.randn((M, K), dtype=torch.float16, device="cuda")
    b = torch.randn((K, N), dtype=torch.float16, device="cuda")
    c = torch.zeros((M, N), dtype=torch.float32, device="cuda")

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _gemm_warp_pipeline_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4,
    )

    ref = (a.float() @ b.float())
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)


# --- IR test: verify border markers in raw IR ---

@triton.jit
def _simple_pipeline_kernel(OUT, K: tl.constexpr):
    y = 0
    for i in tl.range(0, K):
        with tlx.warp_pipeline_stage("stage0", priority=1):
            x = i + 1
        with tlx.warp_pipeline_stage("stage1", priority=0):
            y = x * 1
    tl.store(OUT + tl.program_id(0), y)


@pytest.mark.skipif(not is_hip(), reason="warp pipeline is AMD-only")
def test_warp_pipeline_ir():
    """Verify that warp_pipeline_stage emits border markers in the IR."""
    from triton.compiler import ASTSource
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget

    target = GPUTarget("hip", "gfx950", 64)
    backend = triton.compiler.compiler.make_backend(target)
    opts = backend.parse_options({})

    src = ASTSource(
        fn=_simple_pipeline_kernel,
        signature={"OUT": "*fp32"},
        constexprs={"K": 10},
    )
    ctx = ir.context()
    backend.load_dialects(ctx)
    codegen_fns = backend.get_codegen_implementation(opts)
    raw_ir = src.make_ir(target, opts, codegen_fns, {}, ctx)
    ir_str = str(raw_ir)
    assert "triton.warp_pipeline.border" in ir_str, (
        "Expected warp_pipeline border markers in IR, got:\n" + ir_str
    )
    assert '"stage0"' in ir_str
    assert '"stage1"' in ir_str
    assert 'triton.warp_pipeline.priority = 1' in ir_str
    assert 'triton.warp_pipeline.priority = 0' in ir_str


# --- IR test: verify warp pipeline lowering forms execute_region clusters ---

@triton.jit
def _two_stage_kernel(OUT, IN, N: tl.constexpr, K: tl.constexpr):
    offs = tl.program_id(0) * N + tl.arange(0, N)
    acc = tl.zeros((N,), dtype=tl.float32)
    for i in tl.range(0, K):
        with tlx.warp_pipeline_stage("load"):
            x = tl.load(IN + offs)
        with tlx.warp_pipeline_stage("compute"):
            acc += x
    tl.store(OUT + offs, acc)


@triton.jit
def _smem_warp_pipeline_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
):
    """GEMM with shared memory + warp pipeline to trigger full lowering."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    K_ITERS = tl.cdiv(K, BLOCK_K)
    buffers_A = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_ptr), NUM_BUFFERS)
    buffers_B = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(b_ptr), NUM_BUFFERS)
    for i in tl.range(0, NUM_BUFFERS - 1, loop_unroll_factor=NUM_BUFFERS - 1):
        tok_a = tlx.async_load(a_ptrs, tlx.local_view(buffers_A, i),
                               mask=offs_k[None, :] < K - i * BLOCK_K)
        tok_b = tlx.async_load(b_ptrs, tlx.local_view(buffers_B, i),
                               mask=offs_k[:, None] < K - i * BLOCK_K)
        tlx.async_load_commit_group([tok_a, tok_b])
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    tlx.async_load_wait_group(NUM_BUFFERS - 2)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(NUM_BUFFERS - 1, K_ITERS):
        consumer = (k - (NUM_BUFFERS - 1)) % NUM_BUFFERS
        producer = k % NUM_BUFFERS
        with tlx.warp_pipeline_stage("lds_load", priority=1):
            a_tile = tlx.local_load(tlx.local_view(buffers_A, consumer))
            b_tile = tlx.local_load(tlx.local_view(buffers_B, consumer))
        tlx.async_load_wait_group(0)
        with tlx.warp_pipeline_stage("compute_and_load", priority=0):
            tok_a = tlx.async_load(a_ptrs, tlx.local_view(buffers_A, producer),
                                   mask=offs_k[None, :] < K - k * BLOCK_K)
            tok_b = tlx.async_load(b_ptrs, tlx.local_view(buffers_B, producer),
                                   mask=offs_k[:, None] < K - k * BLOCK_K)
            tlx.async_load_commit_group([tok_a, tok_b])
            acc = tl.dot(a_tile, b_tile, acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
    tlx.async_load_wait_group(0)
    for i in tl.range(0, NUM_BUFFERS - 1, loop_unroll_factor=NUM_BUFFERS - 1):
        consumer = (K_ITERS - (NUM_BUFFERS - 1) + i) % NUM_BUFFERS
        a_tile = tlx.local_load(tlx.local_view(buffers_A, consumer))
        b_tile = tlx.local_load(tlx.local_view(buffers_B, consumer))
        acc = tl.dot(a_tile, b_tile, acc)
    c = acc.to(tlx.dtype_of(c_ptr))
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@pytest.mark.skipif(not is_hip(), reason="warp pipeline is AMD-only")
def test_warp_pipeline_lowering():
    """Verify that shared-memory GEMM with warp pipeline produces barriers in assembly."""
    torch.manual_seed(0)
    M = N = K = 192
    a = torch.randn((M, K), dtype=torch.float16, device="cuda")
    b = torch.randn((K, N), dtype=torch.float16, device="cuda")
    c = torch.zeros((M, N), dtype=torch.float32, device="cuda")
    BM, BN, BK = 64, 64, 32
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)

    kernel = _smem_warp_pipeline_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, NUM_BUFFERS=3, num_warps=8,
    )

    # Check correctness
    ref = a.float() @ b.float()
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)

    # Verify warp pipeline artifacts in compiled assembly
    asm = kernel.asm["amdgcn"]
    assert "s_barrier" in asm, "Expected s_barrier from warp pipeline lowering"
    assert "s_setprio" in asm, "Expected s_setprio from warp pipeline priority hints"


# --- Validation tests ---

def test_warp_pipeline_stage_label_validation():
    """Label must be string or None."""
    with pytest.raises(ValueError, match="label must be a string"):
        tlx.warp_pipeline_stage(123)


def test_warp_pipeline_stage_priority_validation():
    """Priority must be 0-3."""
    with pytest.raises(ValueError, match="priority must be 0-3"):
        tlx.warp_pipeline_stage("test", priority=5)
    with pytest.raises(ValueError, match="priority must be 0-3"):
        tlx.warp_pipeline_stage("test", priority=-1)


def test_warp_pipeline_stage_valid_construction():
    """Valid constructions should not raise."""
    s1 = tlx.warp_pipeline_stage("load", priority=0)
    assert s1.label == "load" and s1.priority == 0
    s2 = tlx.warp_pipeline_stage(priority=3)
    assert s2.label is None and s2.priority == 3
    s3 = tlx.warp_pipeline_stage()
    assert s3.label is None and s3.priority is None


if __name__ == "__main__":
    test_warp_pipeline_stage_label_validation()
    test_warp_pipeline_stage_priority_validation()
    test_warp_pipeline_stage_valid_construction()
    print("PASSED: validation tests")
    if is_hip():
        test_warp_pipeline_ir()
        print("PASSED: test_warp_pipeline_ir")
        test_warp_pipeline_lowering()
        print("PASSED: test_warp_pipeline_lowering")
        test_gemm_warp_pipeline()
        print("PASSED: test_gemm_warp_pipeline")
    else:
        print("Skipped: HIP tests (not running on AMD)")
