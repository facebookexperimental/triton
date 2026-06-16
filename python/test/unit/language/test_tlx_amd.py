"""
Tests for TLX AMD support (async_load, local_load, async_token in loops,
TDM descriptor load/store/prefetch for gfx1250).

These tests compile kernels targeting gfx950/gfx1250 via triton.compile() with
an explicit GPUTarget and verify the generated TTGIR/AMDGCN. No AMD hardware is
required for the compilation checks. Correctness checks (actual execution) run
only when the corresponding hardware is available.
"""
import re

import pytest
import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton._internal_testing import is_hip, is_hip_cdna4, is_hip_gfx1250
from triton.compiler.compiler import ASTSource, compile as triton_compile
from triton.compiler.errors import CompilationError
from triton.backends.compiler import GPUTarget
from triton.language.extra.tlx.tutorials.amd_tdm_gemm_pipelined import (
    matmul_tdm_pipelined_kernel as _amd_tdm_gemm_kernel, )

# Skip the entire module if no HIP runtime is available.
pytestmark = pytest.mark.skipif(not is_hip(), reason="Requires HIP runtime")

GFX950 = GPUTarget("hip", "gfx950", 64)
GFX1250 = GPUTarget("hip", "gfx1250", 32)


def compile_for_gfx950(fn, signature, constexprs):
    """Compile a TLX kernel for gfx950 and return the compiled object."""
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs)
    return triton_compile(src, target=GFX950)


# ---------------------------------------------------------------------------
# Test: async_load compiles on gfx950 and produces the expected ops.
# ---------------------------------------------------------------------------


@triton.jit
def _async_load_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    buffers = tlx.local_alloc((BLOCK_SIZE, ), tl.float32, 2)

    buf0 = tlx.local_view(buffers, 0)
    buf1 = tlx.local_view(buffers, 1)
    tok_x = tlx.async_load(x_ptr + offs, buf0, mask=mask)
    tok_y = tlx.async_load(y_ptr + offs, buf1, mask=mask)
    tlx.async_load_commit_group([tok_x, tok_y])
    tlx.async_load_wait_group(0)

    x = tlx.local_load(buf0)
    y = tlx.local_load(buf1)
    tl.store(output_ptr + offs, x + y, mask=mask)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_async_load_compiles_gfx950(device):
    """async_load should produce async_copy_global_to_local in TTGIR on gfx950."""
    compiled = compile_for_gfx950(
        _async_load_kernel,
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "output_ptr": "*fp32", "n_elements": "i32"},
        constexprs={"BLOCK_SIZE": 64},
    )
    ttgir = compiled.asm["ttgir"]
    assert "async_copy_global_to_local" in ttgir or "buffer_load_to_local" in ttgir
    assert "async_commit_group" in ttgir
    assert "async_wait" in ttgir
    assert "local_load" in ttgir

    # Verify the kernel compiled all the way to AMDGCN.
    assert "amdgcn" in compiled.asm
    assert len(compiled.asm["amdgcn"]) > 0


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_async_load_correctness(device):
    """async_load produces correct results on gfx950 hardware."""
    size = 256
    x = torch.rand(size, dtype=torch.float32, device=device)
    y = torch.rand(size, dtype=torch.float32, device=device)
    output = torch.empty_like(x)
    grid = (triton.cdiv(size, 64), )
    _async_load_kernel[grid](x, y, output, size, BLOCK_SIZE=64)
    torch.testing.assert_close(x + y, output)


# ---------------------------------------------------------------------------
# Test: local_load after async_wait compiles and runs correctly.
# ---------------------------------------------------------------------------


@triton.jit
def _local_load_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    buf = tlx.local_alloc((BLOCK_SIZE, ), tl.float32, 1)
    buf0 = tlx.local_view(buf, 0)
    tok = tlx.async_load(x_ptr + offs, buf0, mask=mask)
    tlx.async_load_commit_group([tok])
    tlx.async_load_wait_group(0)

    x = tlx.local_load(buf0)
    tl.store(output_ptr + offs, x, mask=mask)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_local_load_compiles_gfx950(device):
    """local_load after async_wait should compile and produce local_load in TTGIR."""
    compiled = compile_for_gfx950(
        _local_load_kernel,
        signature={"x_ptr": "*fp32", "output_ptr": "*fp32", "n_elements": "i32"},
        constexprs={"BLOCK_SIZE": 64},
    )
    ttgir = compiled.asm["ttgir"]
    assert "local_load" in ttgir


@triton.jit
def _local_load_with_token_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    buf = tlx.local_alloc((BLOCK_SIZE, ), tl.float32, 1)
    buf0 = tlx.local_view(buf, 0)
    tok = tlx.async_load(x_ptr + offs, buf0, mask=mask)
    tlx.async_load_commit_group([tok])
    wait_tok = tlx.async_load_wait_group(0)

    x = tlx.local_load(buf0, token=wait_tok)
    tl.store(output_ptr + offs, x, mask=mask)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_local_load_with_token_compiles_gfx950(device):
    """local_load with a wait token should set syncedViaAsyncWait in TTGIR."""
    compiled = compile_for_gfx950(
        _local_load_with_token_kernel,
        signature={"x_ptr": "*fp32", "output_ptr": "*fp32", "n_elements": "i32"},
        constexprs={"BLOCK_SIZE": 64},
    )
    ttgir = compiled.asm["ttgir"]
    assert "local_load" in ttgir
    assert re.search(r'ttg\.local_load .* \{ttg\.amdg\.syncedViaAsyncWait = true\}', ttgir, re.MULTILINE)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_local_load_correctness(device):
    """local_load after async_wait produces correct results on gfx950 hardware."""
    size = 256
    x = torch.rand(size, dtype=torch.float32, device=device)
    output = torch.empty_like(x)
    grid = (triton.cdiv(size, 64), )
    _local_load_kernel[grid](x, output, size, BLOCK_SIZE=64)
    torch.testing.assert_close(x, output)


# ---------------------------------------------------------------------------
# Test: async_token survives in scope around tl.range without crashing.
# ---------------------------------------------------------------------------


@triton.jit
def _token_in_loop_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    NUM_ITERS: tl.constexpr,
):
    """async_token from async_load_commit_group is live when tl.range is
    entered. If async_token._flatten_ir is broken, the code generator
    crashes with NotImplementedError when collecting carries."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    buf = tlx.local_alloc((BLOCK_SIZE, ), tl.float32, 1)
    buf0 = tlx.local_view(buf, 0)

    tok = tlx.async_load(x_ptr + offs, buf0, mask=mask)
    tlx.async_load_commit_group([tok])

    acc = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)

    # tok is in scope here -- that's what we're testing.
    for i in tl.range(0, NUM_ITERS, num_stages=0):
        tlx.async_load_wait_group(0)
        x = tlx.local_load(buf0)
        acc += x

    tl.store(output_ptr + offs, acc, mask=mask)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_async_token_loop_compiles_gfx950(device):
    """async_token in scope around tl.range should compile without crashing."""
    compiled = compile_for_gfx950(
        _token_in_loop_kernel,
        signature={"x_ptr": "*fp32", "output_ptr": "*fp32", "n_elements": "i32"},
        constexprs={"BLOCK_SIZE": 64, "NUM_ITERS": 4},
    )
    ttgir = compiled.asm["ttgir"]
    assert "local_load" in ttgir
    assert "async_wait" in ttgir


# ---------------------------------------------------------------------------
# Test: loop-carried dot operands do not fall back through tensor local_alloc.
# ---------------------------------------------------------------------------


@triton.jit
def _loop_carried_dot_layout_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_ITERS: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * (BLOCK_K * K_ITERS) + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * BLOCK_N + offs_n[None, :]

    a_buffers = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.float16, 2)
    b_buffers = tlx.local_alloc((BLOCK_K, BLOCK_N), tl.float16, 2)

    a_buf = tlx.local_view(a_buffers, 0)
    b_buf = tlx.local_view(b_buffers, 0)
    tlx.local_store(a_buf, tl.load(a_ptrs))
    tlx.local_store(b_buf, tl.load(b_ptrs))

    a_reg = tlx.local_load(a_buf)
    b_reg = tlx.local_load(b_buf)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in tl.range(0, K_ITERS - 1, num_stages=0):
        acc = tl.dot(a_reg, b_reg, acc)
        next_slot = (k + 1) % 2
        next_a = tlx.local_view(a_buffers, next_slot)
        next_b = tlx.local_view(b_buffers, next_slot)
        tlx.local_store(next_a, tl.load(a_ptrs + (k + 1) * BLOCK_K))
        tlx.local_store(next_b, tl.load(b_ptrs + (k + 1) * BLOCK_K * BLOCK_N))
        a_reg = tlx.local_load(next_a)
        b_reg = tlx.local_load(next_b)

    acc = tl.dot(a_reg, b_reg, acc)
    c_ptrs = c_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :]
    tl.store(c_ptrs, acc)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_loop_carried_dot_layout_cleanup_compiles_gfx950(device):
    """Full AMD pipeline should remove late dot operand local_alloc fallbacks."""
    compiled = compile_for_gfx950(
        _loop_carried_dot_layout_kernel,
        signature={"a_ptr": "*fp16", "b_ptr": "*fp16", "c_ptr": "*fp32"},
        constexprs={"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "K_ITERS": 3},
    )
    ttgir = compiled.asm["ttgir"]
    assert "ttg.local_alloc %" not in ttgir
    assert "tt.dot" in ttgir
    assert "amdgcn" in compiled.asm
    assert len(compiled.asm["amdgcn"]) > 0


# ---------------------------------------------------------------------------
# gfx1250 TDM tests
#
# Compile-only tests use is_hip() (not is_hip_gfx1250()) because
# triton_compile() with GPUTarget("hip", "gfx1250", 32) only needs the
# HIP compiler toolchain, not actual gfx1250 hardware. This lets them
# run on gfx950 CI. Correctness tests that launch kernels on GPU still
# require is_hip_gfx1250().
# ---------------------------------------------------------------------------


def compile_for_gfx1250(fn, signature, constexprs):
    """Compile a TLX kernel for gfx1250 and return the compiled object."""
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs)
    return triton_compile(src, target=GFX1250)


@triton.jit
def _async_amd_desc_load_kernel(
    x_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    desc = tl.make_tensor_descriptor(x_ptr, [M, N], [N, 1], [M, N])
    buf = tlx.local_alloc((M, N), tl.float16, 1)
    buf0 = tlx.local_view(buf, 0)
    tlx.async_amd_descriptor_load(desc, buf0, [0, 0])
    tlx.async_amd_descriptor_wait(pendings=0)
    x = tlx.local_load(buf0)
    tl.store(output_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], x)


@pytest.mark.skipif(not is_hip(), reason="Requires HIP runtime")
def test_async_amd_desc_load_compiles_gfx1250(device):
    """async_amd_descriptor_load should produce TDM ops in TTGIR."""
    compiled = compile_for_gfx1250(
        _async_amd_desc_load_kernel,
        signature={"x_ptr": "*fp16", "output_ptr": "*fp16"},
        constexprs={"M": 32, "N": 32},
    )
    ttgir = compiled.asm["ttgir"]
    assert "async_tdm_copy_global_to_local" in ttgir
    assert "async_tdm_wait" in ttgir
    assert "local_load" in ttgir
    assert "amdgcn" in compiled.asm
    assert len(compiled.asm["amdgcn"]) > 0


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250 hardware")
@pytest.mark.parametrize("M, N", [(32, 32), (64, 128)])
def test_async_amd_desc_load_correctness_gfx1250(device, M, N):
    """async_amd_descriptor_load produces correct results on gfx1250."""
    x = torch.randn(M, N, dtype=torch.float16, device=device)
    output = torch.empty_like(x)
    _async_amd_desc_load_kernel[(1, )](x, output, M=M, N=N)
    torch.testing.assert_close(x, output)


@triton.jit
def _async_amd_desc_load_with_token_kernel(
    x_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    desc = tl.make_tensor_descriptor(x_ptr, [M, N], [N, 1], [M, N])
    buf = tlx.local_alloc((M, N), tl.float16, 1)
    buf0 = tlx.local_view(buf, 0)
    tok = tlx.async_amd_descriptor_load(desc, buf0, [0, 0])
    tlx.async_amd_descriptor_wait(tokens=[tok])
    x = tlx.local_load(buf0)
    tl.store(output_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], x)


@pytest.mark.skipif(not is_hip(), reason="Requires HIP runtime")
def test_async_amd_desc_load_with_token_compiles_gfx1250(device):
    """async_amd_descriptor_load with token-threaded wait compiles."""
    compiled = compile_for_gfx1250(
        _async_amd_desc_load_with_token_kernel,
        signature={"x_ptr": "*fp16", "output_ptr": "*fp16"},
        constexprs={"M": 32, "N": 32},
    )
    ttgir = compiled.asm["ttgir"]
    assert "async_tdm_copy_global_to_local" in ttgir
    assert "async_tdm_wait" in ttgir


@triton.jit
def _async_amd_desc_load_pred_kernel(
    x_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    desc = tl.make_tensor_descriptor(x_ptr, [M, N], [N, 1], [M, N])
    buf = tlx.local_alloc((M, N), tl.float16, 1)
    buf0 = tlx.local_view(buf, 0)
    pred = tl.program_id(0) == 0
    tlx.async_amd_descriptor_load(desc, buf0, [0, 0], pred=pred)
    tlx.async_amd_descriptor_wait(pendings=0)
    x = tlx.local_load(buf0)
    tl.store(output_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], x)


@pytest.mark.skipif(not is_hip(), reason="Requires HIP runtime")
def test_async_amd_desc_load_pred_compiles_gfx1250(device):
    """async_amd_descriptor_load with i1 pred extends to i32."""
    compiled = compile_for_gfx1250(
        _async_amd_desc_load_pred_kernel,
        signature={"x_ptr": "*fp16", "output_ptr": "*fp16"},
        constexprs={"M": 32, "N": 32},
    )
    ttgir = compiled.asm["ttgir"]
    assert "async_tdm_copy_global_to_local" in ttgir


@triton.jit
def _async_amd_desc_store_kernel(
    x_ptr,
    y_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    desc_in = tl.make_tensor_descriptor(x_ptr, [M, N], [N, 1], [M, N])
    desc_out = tl.make_tensor_descriptor(y_ptr, [M, N], [N, 1], [M, N])
    # Separate buffers for load vs store — they get different encodings
    # (padded for load, swizzled for store) and can't share a buffer
    # until alignTDMDescriptorEncodings is ported.
    load_buf = tlx.local_alloc((M, N), tl.float16, 1)
    store_buf = tlx.local_alloc((M, N), tl.float16, 1)
    load_view = tlx.local_view(load_buf, 0)
    store_view = tlx.local_view(store_buf, 0)
    tlx.async_amd_descriptor_load(desc_in, load_view, [0, 0])
    tlx.async_amd_descriptor_wait(pendings=0)
    data = tlx.local_load(load_view)
    tlx.local_store(store_view, data)
    tlx.async_amd_descriptor_store(desc_out, store_view, [0, 0])
    tlx.async_amd_descriptor_wait(pendings=0)


@pytest.mark.skipif(not is_hip(), reason="Requires HIP runtime")
def test_async_amd_desc_store_compiles_gfx1250(device):
    """async_amd_descriptor_store produces TDM store ops in TTGIR."""
    compiled = compile_for_gfx1250(
        _async_amd_desc_store_kernel,
        signature={"x_ptr": "*fp16", "y_ptr": "*fp16"},
        constexprs={"M": 32, "N": 32},
    )
    ttgir = compiled.asm["ttgir"]
    assert "async_tdm_copy_global_to_local" in ttgir
    assert "async_tdm_copy_local_to_global" in ttgir


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250 hardware")
@pytest.mark.parametrize("M, N", [(32, 32), (64, 128)])
def test_async_amd_desc_store_correctness_gfx1250(device, M, N):
    """TDM load → store round-trip produces correct results on gfx1250."""
    x = torch.randn(M, N, dtype=torch.float16, device=device)
    y = torch.zeros_like(x)
    _async_amd_desc_store_kernel[(1, )](x, y, M=M, N=N)
    torch.testing.assert_close(x, y)


@triton.jit
def _amd_desc_prefetch_kernel(
    x_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    desc = tl.make_tensor_descriptor(x_ptr, [M, N], [N, 1], [M, N])
    tlx.amd_descriptor_prefetch_tensor(desc, [0, 0])
    buf = tlx.local_alloc((M, N), tl.float16, 1)
    buf0 = tlx.local_view(buf, 0)
    tlx.async_amd_descriptor_load(desc, buf0, [0, 0])
    tlx.async_amd_descriptor_wait(pendings=0)
    x = tlx.local_load(buf0)
    tl.store(output_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], x)


@pytest.mark.skipif(not is_hip(), reason="Requires HIP runtime")
def test_amd_desc_prefetch_compiles_gfx1250(device):
    """amd_descriptor_prefetch_tensor produces tdm_prefetch in TTGIR."""
    compiled = compile_for_gfx1250(
        _amd_desc_prefetch_kernel,
        signature={"x_ptr": "*fp16", "output_ptr": "*fp16"},
        constexprs={"M": 32, "N": 32},
    )
    ttgir = compiled.asm["ttgir"]
    assert "tdm_prefetch" in ttgir


@triton.jit
def _amd_desc_prefetch_speculative_kernel(
    x_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    desc = tl.make_tensor_descriptor(x_ptr, [M, N], [N, 1], [M, N])
    pred = tl.program_id(0) == 0
    tlx.amd_descriptor_prefetch_tensor(desc, [0, 0], pred=pred, speculative=True)
    # A TDM load on the same descriptor so it gets a valid encoding
    # during lowering (prefetch alone doesn't assign one).
    buf = tlx.local_alloc((M, N), tl.float16, 1)
    buf0 = tlx.local_view(buf, 0)
    tlx.async_amd_descriptor_load(desc, buf0, [0, 0])
    tlx.async_amd_descriptor_wait(pendings=0)
    x = tlx.local_load(buf0)
    tl.store(output_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], x)


@pytest.mark.skipif(not is_hip(), reason="Requires HIP runtime")
def test_amd_desc_prefetch_speculative_compiles_gfx1250(device):
    """amd_descriptor_prefetch_tensor with speculative=True compiles."""
    compiled = compile_for_gfx1250(
        _amd_desc_prefetch_speculative_kernel,
        signature={"x_ptr": "*fp16", "output_ptr": "*fp16"},
        constexprs={"M": 32, "N": 32},
    )
    ttgir = compiled.asm["ttgir"]
    assert "tdm_prefetch" in ttgir


@pytest.mark.skipif(not is_hip(), reason="Requires HIP runtime")
def test_async_descriptor_load_rejects_amd(device):
    """NV-only async_descriptor_load raises NotImplementedError on AMD."""

    @triton.jit
    def _kernel(x_ptr, M: tl.constexpr, N: tl.constexpr):
        desc = tl.make_tensor_descriptor(x_ptr, [M, N], [N, 1], [M, N])
        barrier = tlx.alloc_barriers(1)
        buf = tlx.local_alloc((M, N), tl.float16, 1)
        buf0 = tlx.local_view(buf, 0)
        tlx.async_descriptor_load(desc, buf0, [0, 0], barrier)

    with pytest.raises(CompilationError, match="NV-only"):
        compile_for_gfx1250(
            _kernel,
            signature={"x_ptr": "*fp16"},
            constexprs={"M": 32, "N": 32},
        )


@pytest.mark.skipif(not is_hip(), reason="Requires HIP runtime")
def test_async_descriptor_store_rejects_amd(device):
    """NV-only async_descriptor_store raises NotImplementedError on AMD."""

    @triton.jit
    def _kernel(x_ptr, M: tl.constexpr, N: tl.constexpr):
        desc = tl.make_tensor_descriptor(x_ptr, [M, N], [N, 1], [M, N])
        buf = tlx.local_alloc((M, N), tl.float16, 1)
        buf0 = tlx.local_view(buf, 0)
        tlx.async_descriptor_store(desc, buf0, [0, 0])

    with pytest.raises(CompilationError, match="NV-only"):
        compile_for_gfx1250(
            _kernel,
            signature={"x_ptr": "*fp16"},
            constexprs={"M": 32, "N": 32},
        )


@pytest.mark.skipif(not is_hip(), reason="Requires HIP runtime")
def test_async_descriptor_prefetch_rejects_amd(device):
    """NV-only async_descriptor_prefetch_tensor raises NotImplementedError on AMD."""

    @triton.jit
    def _kernel(x_ptr, M: tl.constexpr, N: tl.constexpr):
        desc = tl.make_tensor_descriptor(x_ptr, [M, N], [N, 1], [M, N])
        tlx.async_descriptor_prefetch_tensor(desc, [0, 0])

    with pytest.raises(CompilationError, match="NV-only"):
        compile_for_gfx1250(
            _kernel,
            signature={"x_ptr": "*fp16"},
            constexprs={"M": 32, "N": 32},
        )


@pytest.mark.skipif(not is_hip(), reason="Requires HIP runtime")
def test_padded_layout_local_alloc_compiles_gfx1250(device):
    """local_alloc with an explicit padded_shared_layout_encoding compiles."""

    @triton.jit
    def _kernel(x_ptr, output_ptr, M: tl.constexpr, N: tl.constexpr):
        layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for([(N, 128 // 16)], [M, N])
        buf = tlx.local_alloc((M, N), tl.float16, 1, layout=layout)
        buf0 = tlx.local_view(buf, 0)
        x = tlx.local_load(buf0)
        tl.store(output_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], x)

    compiled = compile_for_gfx1250(
        _kernel,
        signature={"x_ptr": "*fp16", "output_ptr": "*fp16"},
        constexprs={"M": 32, "N": 32},
    )
    ttgir = compiled.asm["ttgir"]
    assert "padded_shared" in ttgir


@pytest.mark.skipif(not is_hip(), reason="Requires HIP runtime")
def test_async_amd_desc_load_auto_propagates_padded_layout_gfx1250(device):
    """Default local_alloc + async_amd_descriptor_load auto-propagates padded encoding."""
    compiled = compile_for_gfx1250(
        _async_amd_desc_load_kernel,
        signature={"x_ptr": "*fp16", "output_ptr": "*fp16"},
        constexprs={"M": 32, "N": 32},
    )
    ttgir = compiled.asm["ttgir"]
    assert "padded_shared" in ttgir


# ---------------------------------------------------------------------------
# TDM GEMM tutorial compile test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not is_hip(), reason="Requires HIP runtime")
def test_amd_tdm_gemm_pipelined_compiles_gfx1250(device):
    """Compile-only: validates TDM GEMM tutorial produces TDM ops + padded encoding."""
    compiled = compile_for_gfx1250(
        _amd_tdm_gemm_kernel,
        signature={
            "a_ptr": "*fp16",
            "b_ptr": "*fp16",
            "c_ptr": "*fp16",
            "M": "i32",
            "N": "i32",
            "K": "i32",
        },
        constexprs={"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
    )
    ttgir = compiled.asm["ttgir"]
    assert "amdg.async_tdm_copy_global_to_local" in ttgir
    assert "amdg.tdm_prefetch" in ttgir
    assert "ttg.padded_shared" in ttgir, "expected propagated padded encoding"
    amdgcn = compiled.asm["amdgcn"]
    assert "tensor_load_to_lds" in amdgcn or "tensor.load.to.lds" in amdgcn
