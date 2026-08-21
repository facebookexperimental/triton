"""
Tests for TLX AMD support (async_load, local_load, async_token in loops,
TDM descriptor load/store/prefetch for gfx1250).

These tests compile kernels targeting gfx950/gfx1250 via triton.compile() with
an explicit GPUTarget and verify the generated TTGIR/AMDGCN. No AMD hardware is
required for the compilation checks. Correctness checks (actual execution) run
only when the corresponding hardware is available.
"""
import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton import knobs
from triton._internal_testing import is_hip, is_hip_cdna4, is_hip_gfx1250
from triton.compiler.compiler import ASTSource, compile as triton_compile
from triton.compiler.errors import CompilationError
from triton.backends.amd import compiler as amd_compiler
from triton.backends.compiler import GPUTarget
from triton.runtime.jit import MockTensor
from triton.language.extra.tlx.tutorials.amd_tdm_gemm_pipelined import (
    matmul_tdm_pipelined_kernel as _amd_tdm_gemm_kernel, )
from triton.language.extra.tlx.tutorials.amd_mxfp_gemm_tdm_pipelined import (
    mxgemm_tdm_pipelined_kernel as _amd_mxfp_gemm_kernel, )
from triton.language.extra.tlx.tutorials.gfx9_gemm.intra_wave.a4w4.bench import (
    compile_shape as _compile_a4w4_shape,
    generate_mxfp4_inputs as _generate_a4w4_inputs,
    launch_matmul as _launch_a4w4,
    torch_reference as _a4w4_reference,
)
from triton.language.extra.tlx.tutorials.gfx9_gemm.inter_wave.a4w4.matmul_kernel import (
    BLOCK_K as _A4W4_INTER_WAVE_BLOCK_K,
    BLOCK_M as _A4W4_INTER_WAVE_BLOCK_M,
    BLOCK_N as _A4W4_INTER_WAVE_BLOCK_N,
    MIN_K as _A4W4_INTER_WAVE_MIN_K,
    _a4w4_8wave_kernel as _a4w4_inter_wave_256tile_kernel,
    _a4w4_8wave_merged_scales_kernel as _a4w4_inter_wave_merged_scales_kernel,
    _a4w4_8wave_preshuffled_scales_kernel as _a4w4_inter_wave_preshuffled_scales_kernel,
    _A4W4_8WAVE_LLVM_FN_ATTRS,
    matmul as _a4w4_inter_wave_matmul,
    matmul_merged_scales as _a4w4_inter_wave_matmul_merged_scales,
    matmul_preshuffled as _a4w4_inter_wave_matmul_preshuffled,
    preshuffle_mxfp4_a_scales as _preshuffle_a4w4_a_scales,
    preshuffle_mxfp4_b_scales as _preshuffle_a4w4_b_scales,
    preshuffle_mxfp4_scales as _preshuffle_a4w4_scales,
    select_matmul_path as _select_a4w4_inter_wave_path,
)

# Skip the entire module if no HIP runtime is available.
pytestmark = pytest.mark.skipif(not is_hip(), reason="Requires HIP runtime")

GFX950 = GPUTarget("hip", "gfx950", 64)
GFX942 = GPUTarget("hip", "gfx942", 64)
GFX1250 = GPUTarget("hip", "gfx1250", 32)


def test_amd_ttgir_schedule_env_is_cache_keyed_and_overridable(monkeypatch):
    backend = amd_compiler.HIPBackend(GFX950)
    monkeypatch.delenv("TRITON_AMD_TTGIR_SCHEDULE", raising=False)
    baseline = backend.parse_options({})

    monkeypatch.setenv("TRITON_AMD_TTGIR_SCHEDULE", "1")
    env_enabled = backend.parse_options({})
    config_disabled = backend.parse_options({"enable_sched_group_barrier_scheduler": False})

    assert not baseline.enable_sched_group_barrier_scheduler
    assert env_enabled.enable_sched_group_barrier_scheduler
    assert env_enabled.hash() != baseline.hash()
    assert not config_disabled.enable_sched_group_barrier_scheduler
    assert config_disabled.hash() == baseline.hash()


def test_amd_regalloc_codegen_options_are_cache_keyed():
    baseline = amd_compiler.HIPOptions(arch="gfx950")
    tuned = amd_compiler.HIPOptions(
        arch="gfx950",
        reverse_local_assignment=True,
        sink_insts_to_avoid_spills=True,
        regclass_priority_trumps_globalness=True,
        disable_unclustered_high_rp_reschedule=True,
    )

    assert tuned.hash() != baseline.hash()
    assert amd_compiler._get_codegen_flags(baseline) == []
    assert amd_compiler._get_codegen_flags(tuned) == [
        "greedy-reverse-local-assignment",
        "sink-insts-to-avoid-spills",
        "greedy-regclass-priority-trumps-globalness",
        "amdgpu-disable-unclustered-high-rp-reschedule",
    ]


def test_amd_sched_group_barrier_options_are_cache_keyed_and_validated():
    baseline = amd_compiler.HIPOptions(arch="gfx950")
    tuned = amd_compiler.HIPOptions(
        arch="gfx950",
        enable_sched_group_barrier_scheduler=True,
        sched_group_barrier_mfma_per_dwordx4=2,
        sched_group_barrier_required_region_count=4,
    )

    assert tuned.hash() != baseline.hash()
    assert tuned.enable_sched_group_barrier_scheduler
    assert tuned.sched_group_barrier_mfma_per_dwordx4 == 2
    assert tuned.sched_group_barrier_required_region_count == 4
    with pytest.raises(ValueError, match="sched_group_barrier_mfma_per_dwordx4 must be positive"):
        amd_compiler.HIPOptions(arch="gfx950", sched_group_barrier_mfma_per_dwordx4=0)
    with pytest.raises(ValueError, match="sched_group_barrier_required_region_count must be non-negative"):
        amd_compiler.HIPOptions(arch="gfx950", sched_group_barrier_required_region_count=-1)


def compile_for_target(fn, signature, constexprs, target):
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs)
    return triton_compile(src, target=target)


def compile_for_gfx950(fn, signature, constexprs):
    """Compile a TLX kernel for gfx950 and return the compiled object."""
    return compile_for_target(fn, signature, constexprs, GFX950)


@triton.jit
def _shared_concrete_helper(values):
    return values


@triton.jit
def _shared_concrete_helper_kernel(x_ptr, y_ptr):
    layout_m: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    layout_n: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    rows = tl.arange(0, 64)
    cols = tl.arange(0, 64)
    offsets = rows[:, None] * 64 + cols[None, :]
    values = tl.load(x_ptr + offsets)
    values_m = tlx.require_layout(values, layout_m, pin=False)
    values_n = tlx.require_layout(values, layout_n, pin=False)
    result_m = _shared_concrete_helper(values_m)
    result_n = _shared_concrete_helper(values_n)
    output_m = tlx.require_layout(y_ptr + offsets, layout_m, pin=False)
    output_n = tlx.require_layout(y_ptr + 4096 + offsets, layout_n, pin=False)
    tl.store(output_m, result_m)
    tl.store(output_n, result_n)


def test_shared_helper_accepts_distinct_concrete_layouts_gfx950():
    compiled = compile_for_gfx950(
        _shared_concrete_helper_kernel,
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32"},
        constexprs={},
    )
    assert "amdgcn" in compiled.asm


@triton.jit
def _mixed_helper_results(values, condition, LAYOUT: tl.constexpr):
    # The frontend emits an encoding-free return for the else path and the
    # trailing unreachable block.  Fixup must bridge only result 0; result 1
    # intentionally remains encoding-free.
    if condition:
        concrete = tlx.require_layout(values, LAYOUT, pin=False)
        return concrete, values
    return values, values


@triton.jit
def _mixed_helper_results_kernel(x_ptr, y_ptr, condition):
    value_layout: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    rows = tl.arange(0, 16)
    cols = tl.arange(0, 64)
    offsets = rows[:, None] * 64 + cols[None, :]
    values = tl.load(x_ptr + offsets)
    concrete, deferred = _mixed_helper_results(values, condition, value_layout)
    concrete_offsets = tlx.require_layout(y_ptr + offsets, value_layout, pin=False)
    # Consume the siblings under different ABIs.  If fixup retypes the shared
    # producer, the encoding-free store below becomes invalid.
    tl.store(concrete_offsets, concrete)
    tl.store(y_ptr + 1024 + offsets, deferred)


def test_mixed_helper_result_abi_compiles_gfx950():
    compiled = compile_for_gfx950(
        _mixed_helper_results_kernel,
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "condition": "i1"},
        constexprs={},
    )
    assert "amdgcn" in compiled.asm


@triton.jit
def _placeholder_mixed_results(values):
    zeros = tl.zeros(values.shape, tl.float32)
    combined = values + zeros
    reduced = tl.sum(values, axis=1)
    return combined, reduced, zeros


@triton.jit
def _placeholder_mixed_results_kernel(x_ptr, y_ptr):
    value_layout: tl.constexpr = tlx.layout(
        shape=((64, 4), (4, )),
        stride=((4, 256), (1, )),
    )
    rows = tl.arange(0, 16)
    cols = tl.arange(0, 64)
    offsets = rows[:, None] * 64 + cols[None, :]
    values = tl.load(x_ptr + offsets)
    values = tlx.require_layout(values, value_layout)
    combined, reduced, zeros = _placeholder_mixed_results(values)
    tl.store(y_ptr + offsets, combined + zeros)
    tl.store(y_ptr + 1024 + rows, reduced)


def test_placeholder_mixed_and_constant_helper_results_compile_gfx950():
    compiled = compile_for_gfx950(
        _placeholder_mixed_results_kernel,
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32"},
        constexprs={},
    )
    assert "amdgcn" in compiled.asm
    assert "#tlx.user_layout" not in compiled.asm["ttgir"]
    assert "#tlx.no_verify_layout" not in compiled.asm["ttgir"]


@triton.jit
def _buffer_load_contiguity_kernel(x_ptr, y_ptr):
    load_layout: tl.constexpr = tlx.layout(
        shape=((64, 4), (4, )),
        stride=((4, 256), (1, )),
    )
    offsets = tl.arange(0, 1024).to(tl.int32)
    offsets = tlx.require_layout(offsets, load_layout, pin=False)
    values = tlx.buffer_load(x_ptr, offsets, contiguity=4)
    tl.store(y_ptr + offsets, values)


def test_buffer_load_contiguity_vectorizes_gfx950():
    compiled = compile_for_gfx950(
        _buffer_load_contiguity_kernel,
        signature={"x_ptr": "*bf16", "y_ptr": "*bf16"},
        constexprs={},
    )

    ttgir = compiled.asm["ttgir"]
    assert "amdg.buffer_load" in ttgir
    assert "contiguity = 4" in ttgir
    assert "buffer_load_dwordx2" in compiled.asm["amdgcn"]


@triton.jit
def _buffer_atomic_contiguity_layout_anchor_kernel(x_ptr, atomic_ptr, y_ptr):
    contiguous_layout: tl.constexpr = tlx.layout(
        shape=((64, 4), (4, )),
        stride=((4, 256), (1, )),
    )
    competing_layout: tl.constexpr = tlx.layout(
        shape=((64, 4), (4, )),
        stride=((1, 256), (64, )),
    )
    offsets = tl.arange(0, 1024).to(tl.int32)
    offsets = tlx.require_layout(offsets, contiguous_layout, pin=False)
    values = tl.load(x_ptr + offsets)
    values = tlx.require_layout(values, contiguous_layout, pin=False)
    previous = tlx.buffer_atomic_add(
        atomic_ptr,
        offsets,
        values,
        sem="relaxed",
        contiguity=2,
    )
    previous = tlx.require_layout(previous, competing_layout)
    output_offsets = tlx.require_layout(y_ptr + offsets, competing_layout)
    tl.store(output_offsets, previous)


def test_buffer_atomic_contiguity_preserves_layout_gfx950():
    compiled = compile_for_gfx950(
        _buffer_atomic_contiguity_layout_anchor_kernel,
        signature={"x_ptr": "*bf16", "atomic_ptr": "*bf16", "y_ptr": "*bf16"},
        constexprs={},
    )

    ttgir = compiled.asm["ttgir"]
    assert "amdg.buffer_atomic_rmw" in ttgir
    assert "contiguity = 2" in ttgir
    assert "tlx.preserve_layout" in ttgir
    atomic = re.search(
        r"(?P<result>%[\w.]+) = amdg\.buffer_atomic_rmw.*"
        r"tlx\.preserve_layout.*: tensor<1024xbf16, (?P<layout>#[\w.]+)>",
        ttgir,
    )
    assert atomic is not None
    conversion = re.search(
        rf"ttg\.convert_layout {re.escape(atomic.group('result'))} : "
        rf"tensor<1024xbf16, {re.escape(atomic.group('layout'))}> -> "
        r"tensor<1024xbf16, (?P<layout>#[\w.]+)>",
        ttgir,
    )
    assert conversion is not None
    assert conversion.group("layout") != atomic.group("layout")
    assert "buffer_atomic_pk_add_bf16" in compiled.asm["amdgcn"]


@triton.jit
def _masked_buffer_atomic_contiguity_kernel(
    x_ptr,
    atomic_ptr,
    MASK_BOUNDARY: tl.constexpr,
):
    contiguous_layout: tl.constexpr = tlx.layout(
        shape=((64, 4), (4, )),
        stride=((4, 256), (1, )),
    )
    offsets = tl.arange(0, 1024).to(tl.int32)
    offsets = tlx.require_layout(offsets, contiguous_layout, pin=False)
    values = tl.load(x_ptr + offsets)
    values = tlx.require_layout(values, contiguous_layout, pin=False)
    tlx.buffer_atomic_add(
        atomic_ptr,
        offsets,
        values,
        mask=offsets < MASK_BOUNDARY,
        sem="relaxed",
        contiguity=2,
    )


def test_masked_buffer_atomic_contiguity_vectorizes_gfx950():
    compiled = compile_for_gfx950(
        _masked_buffer_atomic_contiguity_kernel,
        signature={
            "x_ptr": "*bf16",
            "atomic_ptr": "*bf16",
            "MASK_BOUNDARY": "constexpr",
        },
        constexprs={"MASK_BOUNDARY": 512},
    )
    assert "buffer_atomic_pk_add_bf16" in compiled.asm["amdgcn"]


def test_masked_buffer_atomic_rejects_scalar_bf16_gfx950(capfd):
    with pytest.raises(RuntimeError):
        compile_for_gfx950(
            _masked_buffer_atomic_contiguity_kernel,
            signature={
                "x_ptr": "*bf16",
                "atomic_ptr": "*bf16",
                "MASK_BOUNDARY": "constexpr",
            },
            constexprs={"MASK_BOUNDARY": 511},
        )
    assert ("16-bit buffer atomics require two contiguous elements" in capfd.readouterr().err)


@triton.jit
def _unsupported_i16_buffer_atomic_kernel(atomic_ptr):
    offsets = tl.arange(0, 64).to(tl.int32)
    values = tl.zeros((64, ), tl.int16)
    tlx.buffer_atomic_add(atomic_ptr, offsets, values)


def test_buffer_atomic_rejects_unsupported_i16_gfx950():
    with pytest.raises(CompilationError, match="buffer_atomic_add supports only"):
        compile_for_gfx950(
            _unsupported_i16_buffer_atomic_kernel,
            signature={"atomic_ptr": "*i16"},
            constexprs={},
        )


def test_pinned_buffer_load_layout_survives_optimization_gfx950():
    from triton.language.extra.tlx.tutorials.amd_fa_bwd import (
        _attn_bwd_dq_native_convert_kernel, )

    compiled = compile_for_gfx950(
        _attn_bwd_dq_native_convert_kernel,
        signature={"DQ_ACC": "*bf16", "DQ": "*bf16"},
        constexprs={"N": 128, "D": 128, "BLOCK_M": 128},
    )

    ttgir = compiled.asm["ttgir"]
    amdgcn = compiled.asm["amdgcn"]
    assert "contiguity = 4" in ttgir
    assert "ttg.convert_layout" in ttgir
    assert amdgcn.count("buffer_load_dwordx2") == 16
    assert amdgcn.count("v_permlane16_swap_b32") == 16


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_gqa_oversized_batches_rebase_buffer_offsets_gfx950(causal):
    from triton.language.extra.tlx.tutorials.amd_fa_bwd import (
        _attn_bwd_dkdv_dq_d128_gqa_kernel, )

    # At N=16384 and D=128, 512 BF16 heads exactly fill the signed 32-bit
    # byte-offset range; 520 exercises the per-tile 64-bit pointer rebasing
    # path without allocating multi-gigabyte test tensors.
    block_m = 16
    block_n = 256
    compiled = compile_for_gfx950(
        _attn_bwd_dkdv_dq_d128_gqa_kernel,
        signature={
            "Q": "*bf16",
            "K": "*bf16",
            "V": "*bf16",
            "DO": "*bf16",
            "LSE": "*fp32",
            "Delta": "*fp32",
            "DQ_ACC": "*bf16",
            "DK": "*bf16",
            "DV": "*bf16",
        },
        constexprs={
            "SM_SCALE": 0.125,
            "IS_CAUSAL": causal,
            "HQ": 520,
            "HK": 520,
            "N": 16384,
            "D": 128,
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
        },
    )

    ttir = compiled.asm["ttir"]
    ttgir = compiled.asm["ttgir"]
    dummy_clamps = re.findall(r"arith\.maxsi %dq_step, (?P<floor>%[\w_]+)", ttir)
    assert len(dummy_clamps) == 1
    if causal:
        assert dummy_clamps[0] != "%c0_i32"
        first_active_stride = block_n // block_m
        assert re.search(
            rf"^\s*{re.escape(dummy_clamps[0])} = arith\.muli "
            rf"%pid_n(?:_\d+)?, %c{first_active_stride}_i32",
            ttir,
            re.MULTILINE,
        )
    else:
        assert dummy_clamps[0] == "%c0_i32"
    assert ttgir.count("tlx.rematerialize_coordinates_group = 21 : i32") == (9 if causal else 0)
    assert "amdgcn" in compiled.asm


def test_gqa_oversized_head_rebases_native_conversion_gfx950():
    from triton.language.extra.tlx.tutorials.amd_fa_bwd import (
        _attn_bwd_dq_native_convert_kernel, )

    compiled = compile_for_gfx950(
        _attn_bwd_dq_native_convert_kernel,
        signature={"DQ_ACC": "*bf16", "DQ": "*bf16"},
        constexprs={
            "N": (1 << 23) + 256,
            "D": 128,
            "BLOCK_M": 128,
        },
    )

    assert "amdgcn" in compiled.asm


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_pinned_buffer_load_layout_correctness_gfx950(device):
    from triton.language.extra.tlx.tutorials.amd_fa_bwd import (
        _attn_bwd_dq_native_convert_kernel, )

    m = torch.arange(128, device=device, dtype=torch.int64)[:, None]
    d = torch.arange(128, device=device, dtype=torch.int64)[None, :]
    expected = ((m * 131 + d * 7) % 2048).to(torch.bfloat16)
    local_m = m & 15
    tile_m = m - local_m
    d_swizzled = ((d & 1) | ((d & 2) << 6) | ((d & 12) << 3) | ((d & 48) << 5) | ((d & 64) << 2))
    physical = tile_m * 128 + (local_m << 1) + d_swizzled
    native = torch.empty(128 * 128, device=device, dtype=torch.bfloat16)
    native[physical.flatten()] = expected.flatten()
    actual = torch.empty_like(expected)

    _attn_bwd_dq_native_convert_kernel[(1, 1)](
        native,
        actual,
        N=128,
        D=128,
        BLOCK_M=128,
        num_warps=4,
        matrix_instr_nonkdim=16,
    )

    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def _load_tlx_gfx9_gemm_bench_module(module_name="_tlx_amd_test_gfx9_bench"):
    repo_root = Path(__file__).resolve().parents[4]
    bench_path = (repo_root / "third_party" / "tlx" / "tutorials" / "gfx9_gemm" / "a16w16" / "bench.py")
    spec = importlib.util.spec_from_file_location(module_name, bench_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_tlx_gfx9_inter_wave_bench_module(module_name="_tlx_amd_test_gfx9_inter_wave_bench"):
    repo_root = Path(__file__).resolve().parents[4]
    bench_path = (repo_root / "third_party" / "tlx" / "tutorials" / "gfx9_gemm" / "inter_wave" / "a16w16" / "bench.py")
    previous_kernel_module = sys.modules.get("matmul_kernel")
    try:
        sys.modules["matmul_kernel"] = SimpleNamespace(
            matmul=lambda _a, _b: None,
            MIN_K=128,
            KERNEL_NAME="a16w16_8wave",
        )
        spec = importlib.util.spec_from_file_location(module_name, bench_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous_kernel_module is None:
            sys.modules.pop("matmul_kernel", None)
        else:
            sys.modules["matmul_kernel"] = previous_kernel_module


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


@pytest.mark.skipif(not is_hip(), reason="Requires HIP runtime")
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
# Test: aligned AMD register slicing preserves an MFMA operand layout.
# ---------------------------------------------------------------------------


@triton.jit
def _extract_slice_kernel(x_ptr, y_ptr):
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    dot0: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=8)
    rows = tl.arange(0, 16)
    cols = tl.arange(0, 256)
    values = tl.load(x_ptr + rows[:, None] * 256 + cols[None, :])
    values = tlx.require_layout(values, dot0, pin=False)
    band = tlx.extract_slice(values, [16, 32], [0, 64])
    band_cols = tl.arange(0, 32)
    out_ptrs = y_ptr + rows[:, None] * 32 + band_cols[None, :]
    out_ptrs = tlx.require_layout(out_ptrs, dot0, pin=False)
    tl.store(out_ptrs, band)


def test_extract_slice_compiles_gfx950():
    compiled = compile_for_gfx950(
        _extract_slice_kernel,
        signature={"x_ptr": "*bf16", "y_ptr": "*bf16"},
        constexprs={},
    )
    assert "amdg.extract_slice" in compiled.asm["ttir"]
    assert "amdgcn" in compiled.asm


@triton.jit
def _extract_slice_dot1_kernel(
    x_ptr,
    y_ptr,
    ROW_OFFSET: tl.constexpr,
    COL_OFFSET: tl.constexpr,
):
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    dot1: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=8)
    rows = tl.arange(0, 256)
    cols = tl.arange(0, 128)
    values = tl.load(x_ptr + rows[:, None] * 128 + cols[None, :])
    values = tlx.require_layout(values, dot1, pin=False)
    band = tlx.extract_slice(values, [32, 64], [ROW_OFFSET, COL_OFFSET])
    band_rows = tl.arange(0, 32)
    band_cols = tl.arange(0, 64)
    out_ptrs = y_ptr + band_rows[:, None] * 64 + band_cols[None, :]
    out_ptrs = tlx.require_layout(out_ptrs, dot1, pin=False)
    tl.store(out_ptrs, band)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_extract_slice_dot1_correct_gfx950():
    x = torch.arange(256 * 128, device="cuda", dtype=torch.float32).reshape(256, 128).to(torch.bfloat16)
    actual = torch.empty((32, 64), device="cuda", dtype=torch.bfloat16)
    for row in range(0, 256, 32):
        for col in (0, 64):
            _extract_slice_dot1_kernel[(1, )](
                x,
                actual,
                ROW_OFFSET=row,
                COL_OFFSET=col,
                num_warps=4,
                matrix_instr_nonkdim=16,
            )
            torch.testing.assert_close(actual, x[row:row + 32, col:col + 64])


@triton.jit
def _extract_slice_mfma_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    BAND: tl.constexpr,
):
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    dot0: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=8)
    dot1: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=8)
    rows = tl.arange(0, 16)
    reduction = tl.arange(0, 256)
    cols = tl.arange(0, 64)
    a = tl.load(a_ptr + rows[:, None] * 256 + reduction[None, :])
    b = tl.load(b_ptr + reduction[:, None] * 64 + cols[None, :])
    a = tlx.require_layout(a, dot0, pin=False)
    b = tlx.require_layout(b, dot1, pin=False)
    a_band = tlx.extract_slice(a, [16, 32], [0, BAND * 32])
    b_band = tlx.extract_slice(b, [32, 64], [BAND * 32, 0])
    acc = tlx.zeros((16, 64), tl.float32, layout=mma)
    result = tl.dot(a_band, b_band, acc)
    output_offsets = output_ptr + rows[:, None] * 64 + cols[None, :]
    output_offsets = tlx.require_layout(output_offsets, mma, pin=False)
    tl.store(output_offsets, result)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_extract_slice_mfma_correct_gfx950():
    torch.manual_seed(0)
    a = torch.randn((16, 256), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((256, 64), device="cuda", dtype=torch.bfloat16)
    actual = torch.empty((16, 64), device="cuda", dtype=torch.float32)
    for band in range(8):
        _extract_slice_mfma_kernel[(1, )](
            a,
            b,
            actual,
            BAND=band,
            num_warps=4,
            matrix_instr_nonkdim=16,
        )
        expected = (a[:, band * 32:(band + 1) * 32].float() @ b[band * 32:(band + 1) * 32].float())
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)


@triton.jit
def _rematerialized_range_kernel(x_ptr, y_ptr):
    load_rows = tlx.rematerialized_range(0, 64, identity=0)
    load_cols = tlx.rematerialized_range(0, 64, identity=1)
    values = tl.load(x_ptr + load_rows[:, None] * 64 + load_cols[None, :])

    store_rows = tlx.rematerialized_range(0, 64, identity=2)
    store_cols = tlx.rematerialized_range(0, 64, identity=3)
    tl.store(y_ptr + store_rows[:, None] * 64 + store_cols[None, :], values)


def test_rematerialized_range_compiles_gfx950():
    compiled = compile_for_gfx950(
        _rematerialized_range_kernel,
        signature={"x_ptr": "*bf16", "y_ptr": "*bf16"},
        constexprs={},
    )
    assert compiled.asm["ttir"].count("amdg.rematerialized_range") == 4
    assert compiled.asm["ttgir"].count("amdg.rematerialized_range") == 4
    # Each range layout depends on one distributed coordinate; do not anchor
    # the zero-basis lane/warp dimension.
    assert compiled.asm["llir"].count('asm sideeffect "", "=v,0"') == 4
    assert "amdg.rematerialized_range" not in compiled.asm["llir"]
    assert "amdgcn" in compiled.asm


@triton.jit
def _amd_late_address_compute_kernel(x_ptr, y_ptr):
    src_mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    dst_mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    src_dot0: tl.constexpr = tlx.dot_operand_layout(0, src_mma, k_width=8)
    src_dot1: tl.constexpr = tlx.dot_operand_layout(1, src_mma, k_width=8)
    rows = tl.arange(0, 64)
    reduction = tl.arange(0, 32)
    cols = tl.arange(0, 64)
    a = tlx.require_layout(
        tl.load(x_ptr + rows[:, None] * 32 + reduction[None, :]),
        src_dot0,
        pin=False,
    )
    b = tlx.require_layout(
        tl.load(x_ptr + reduction[:, None] * 64 + cols[None, :]),
        src_dot1,
        pin=False,
    )
    values = tl.dot(
        a,
        b,
        tlx.zeros((64, 64), tl.float32, layout=src_mma),
    )
    values = tlx.require_layout(
        values,
        dst_mma,
        late_address_compute=True,
    )
    offsets = rows[:, None] * 64 + cols[None, :]
    output_offsets = tlx.require_layout(y_ptr + offsets, dst_mma, pin=False)
    tl.store(output_offsets, values)


def test_amd_late_address_compute_compiles_gfx950():
    compiled = compile_for_gfx950(
        _amd_late_address_compute_kernel,
        signature={"x_ptr": "*bf16", "y_ptr": "*bf16"},
        constexprs={},
    )
    assert "tlx.rematerialize_coordinates" in compiled.asm["ttir"]
    assert "tlx.rematerialize_coordinates" in compiled.asm["ttgir"]
    assert compiled.asm["llir"].count('asm sideeffect "", "=v,0"') >= 2
    assert "amdgcn" in compiled.asm


@triton.jit
def _amd_scheduled_mfma_kernel(a_ptr, b_ptr, output_ptr):
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    dot0: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=8)
    dot1: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=8)
    rows = tl.arange(0, 16)
    reduction = tl.arange(0, 32)
    cols = tl.arange(0, 64)
    a = tl.load(a_ptr + rows[:, None] * 32 + reduction[None, :])
    b = tl.load(b_ptr + reduction[:, None] * 64 + cols[None, :])
    a = tlx.require_layout(a, dot0, pin=False)
    b = tlx.require_layout(b, dot1, pin=False)
    b = tlx.amd_register_resident(b, register_class="agpr", registers_per_group=4)
    acc = tl.full((16, 64), 7.0, tl.float32)
    acc = tlx.require_layout(acc, mma, pin=False)
    result = tlx.amd_scheduled_mfma(
        a,
        b,
        acc,
        resident_operand=1,
        accumulator_role="transient",
        initialize=True,
    )
    result, _ = tlx.amd_mfma_commit(result, b)
    output_offsets = output_ptr + rows[:, None] * 64 + cols[None, :]
    output_offsets = tlx.require_layout(output_offsets, mma, pin=False)
    tl.store(output_offsets, result)


def test_amd_scheduled_mfma_compiles_gfx950():
    compiled = compile_for_gfx950(
        _amd_scheduled_mfma_kernel,
        signature={"a_ptr": "*bf16", "b_ptr": "*bf16", "output_ptr": "*fp32"},
        constexprs={},
    )
    assert "amdg.register_resident" in compiled.asm["ttir"]
    assert 'class "agpr" groups 4' in compiled.asm["ttir"]
    assert "amdg.scheduled_mfma" in compiled.asm["ttir"]
    assert "amdg.mfma_commit" in compiled.asm["ttir"]
    assert "=a,0" in compiled.asm["llir"]
    assert "@llvm.amdgcn.mfma.f32.16x16x32.bf16" in compiled.asm["llir"]
    assert 'asm sideeffect "v_mfma' not in compiled.asm["llir"]
    assert "v_mfma_f32_16x16x32_bf16" in compiled.asm["amdgcn"]
    assert "s_nop 5" in compiled.asm["llir"]


def test_amd_scheduled_mfma_rejects_non_cdna4():
    with pytest.raises(RuntimeError, match=r"scheduled_mfma.*only on CDNA4"):
        compile_for_target(
            _amd_scheduled_mfma_kernel,
            signature={
                "a_ptr": "*bf16",
                "b_ptr": "*bf16",
                "output_ptr": "*fp32",
            },
            constexprs={},
            target=GFX942,
        )


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_amd_scheduled_mfma_initialize_discards_acc_gfx950():
    torch.manual_seed(0)
    a = torch.randn((16, 32), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((32, 64), device="cuda", dtype=torch.bfloat16)
    actual = torch.empty((16, 64), device="cuda", dtype=torch.float32)
    _amd_scheduled_mfma_kernel[(1, )](
        a,
        b,
        actual,
        num_warps=4,
        matrix_instr_nonkdim=16,
    )
    torch.testing.assert_close(actual, a.float() @ b.float(), atol=2e-4, rtol=2e-4)


@triton.jit
def _amd_scheduled_mfma_chain_kernel(a_ptr, b_ptr, output_ptr, BANDS: tl.constexpr):
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    dot0: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=8)
    dot1: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=8)
    rows = tl.arange(0, 16)
    reduction = tl.arange(0, 256)
    cols = tl.arange(0, 64)
    a = tl.load(a_ptr + rows[:, None] * 256 + reduction[None, :])
    b = tl.load(b_ptr + reduction[:, None] * 64 + cols[None, :])
    a = tlx.require_layout(a, dot0, pin=False)
    b = tlx.require_layout(b, dot1, pin=False)
    b = tlx.amd_register_resident(b, register_class="agpr", registers_per_group=32)
    acc = tlx.zeros((16, 64), tl.float32, layout=mma)

    for band in tl.static_range(BANDS):
        a_band = tlx.extract_slice(a, [16, 32], [0, band * 32])
        b_band = tlx.extract_slice(b, [32, 64], [band * 32, 0])
        acc = tlx.amd_scheduled_mfma(
            a_band,
            b_band,
            acc,
            resident_operand=1,
            accumulator_role="transient",
            initialize=band == 0,
        )
    acc, _ = tlx.amd_mfma_commit(acc, b_band)
    output_offsets = output_ptr + rows[:, None] * 64 + cols[None, :]
    output_offsets = tlx.require_layout(output_offsets, mma, pin=False)
    tl.store(output_offsets, acc)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_amd_scheduled_mfma_chain_correct_gfx950():
    torch.manual_seed(0)
    a = torch.randn((16, 256), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((256, 64), device="cuda", dtype=torch.bfloat16)
    actual = torch.empty((16, 64), device="cuda", dtype=torch.float32)
    _amd_scheduled_mfma_chain_kernel[(1, )](
        a,
        b,
        actual,
        BANDS=8,
        num_warps=4,
        matrix_instr_nonkdim=16,
    )
    expected = a.float() @ b.float()
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)


@triton.jit
def _amd_scheduled_mfma_persistent_acc_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    USE_VGPR: tl.constexpr,
):
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    dot0: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=8)
    dot1: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=8)
    rows = tl.arange(0, 16)
    reduction = tl.arange(0, 64)
    cols = tl.arange(0, 64)
    a = tl.load(a_ptr + rows[:, None] * 64 + reduction[None, :])
    b = tl.load(b_ptr + reduction[:, None] * 64 + cols[None, :])
    a = tlx.require_layout(a, dot0, pin=False)
    b = tlx.require_layout(b, dot1, pin=False)
    a0 = tlx.extract_slice(a, [16, 32], [0, 0])
    b0 = tlx.extract_slice(b, [32, 64], [0, 0])
    acc = tlx.zeros((16, 64), tl.float32, layout=mma)
    acc = tlx.amd_scheduled_mfma(
        a0,
        b0,
        acc,
        accumulator_role="persistent",
        accumulator_register_class="vgpr" if USE_VGPR else None,
        initialize=True,
    )
    a1 = tlx.extract_slice(a, [16, 32], [0, 32])
    b1 = tlx.extract_slice(b, [32, 64], [32, 0])
    acc = tlx.amd_scheduled_mfma(
        a1,
        b1,
        acc,
        accumulator_role="persistent",
        accumulator_register_class="vgpr" if USE_VGPR else None,
    )
    output_offsets = output_ptr + rows[:, None] * 64 + cols[None, :]
    output_offsets = tlx.require_layout(output_offsets, mma, pin=False)
    tl.store(output_offsets, acc)


@pytest.mark.parametrize("elem_ty", ["bf16", "fp16"])
def test_amd_scheduled_mfma_persistent_acc_lowering_gfx950(elem_ty):
    compiled = compile_for_gfx950(
        _amd_scheduled_mfma_persistent_acc_kernel,
        signature={
            "a_ptr": f"*{elem_ty}",
            "b_ptr": f"*{elem_ty}",
            "output_ptr": "*fp32",
            "USE_VGPR": "constexpr",
        },
        constexprs={"USE_VGPR": False},
    )
    llir = compiled.asm["llir"]
    asm_ty = "f16" if elem_ty == "fp16" else elem_ty
    # A persistent accumulator lowers to inline asm so the accumulator's
    # register class is preserved. AMDGPU's hazard recognizer cannot see an
    # MFMA inside `asm sideeffect`, so the required wait states are part of the
    # asm: `s_nop 3` before the MFMA covers a preceding VALU write of srcA/srcB
    # or EXEC. Without it the MFMA reads stale operands and the kernel silently
    # computes garbage.
    assert f'asm sideeffect "s_nop 3\\0Av_mfma_f32_16x16x32_{asm_ty}' in llir
    assert '"=a,v,v"' in llir
    assert f"@llvm.amdgcn.mfma.f32.16x16x32.{asm_ty}" not in llir
    # ...and the chain is drained (18 wait states) before anything reads the
    # accumulator back.
    assert 'asm sideeffect "s_nop 15\\0As_nop 1"' in llir


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_amd_scheduled_mfma_persistent_acc_correct_gfx950():
    torch.manual_seed(0)
    a = torch.randn((16, 64), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((64, 64), device="cuda", dtype=torch.bfloat16)
    actual = torch.empty((16, 64), device="cuda", dtype=torch.float32)
    expected = a.float() @ b.float()
    for use_vgpr in (False, True):
        _amd_scheduled_mfma_persistent_acc_kernel[(1, )](
            a,
            b,
            actual,
            USE_VGPR=use_vgpr,
            num_warps=4,
            matrix_instr_nonkdim=16,
        )
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)


@triton.jit
def _amd_scheduled_mfma_32x32_kernel(a_ptr, b_ptr, output_ptr):
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[32, 32, 16],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    dot0: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=8)
    dot1: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=8)
    rows = tl.arange(0, 128)
    reduction = tl.arange(0, 16)
    cols = tl.arange(0, 32)
    a = tlx.require_layout(
        tl.load(a_ptr + rows[:, None] * 16 + reduction[None, :]),
        dot0,
        pin=False,
    )
    b = tlx.require_layout(
        tl.load(b_ptr + reduction[:, None] * 32 + cols[None, :]),
        dot1,
        pin=False,
    )
    acc = tlx.zeros((128, 32), tl.float32, layout=mma)
    result = tlx.amd_scheduled_mfma(
        a,
        b,
        acc,
        accumulator_role="transient",
        initialize=True,
    )
    result, _ = tlx.amd_mfma_commit(result, b)
    offsets = tlx.require_layout(
        output_ptr + rows[:, None] * 32 + cols[None, :],
        mma,
        pin=False,
    )
    tl.store(offsets, result)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_amd_scheduled_mfma_32x32_correct_gfx950():
    torch.manual_seed(0)
    a = torch.randn((128, 16), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((16, 32), device="cuda", dtype=torch.bfloat16)
    actual = torch.empty((128, 32), device="cuda", dtype=torch.float32)
    _amd_scheduled_mfma_32x32_kernel[(1, )](
        a,
        b,
        actual,
        num_warps=4,
        matrix_instr_nonkdim=16,
    )
    torch.testing.assert_close(actual, a.float() @ b.float(), atol=2e-4, rtol=2e-4)


@triton.jit
def _amd_scheduled_mfma_fragmented_nd_kernel(a_ptr, b_ptr, output_ptr):
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[32, 32, 16],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    dot0: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=8)
    dot1: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=8)
    rows = tl.arange(0, 256)
    reduction = tl.arange(0, 16)
    cols = tl.arange(0, 128)
    a = tl.load(a_ptr + rows[:, None] * 16 + reduction[None, :])
    b = tl.load(b_ptr + reduction[:, None] * 128 + cols[None, :])
    a = tlx.require_layout(a, dot0, pin=False)
    b = tlx.require_layout(b, dot1, pin=False)

    a_lo = tlx.extract_slice(a, [128, 16], [0, 0])
    a_hi = tlx.extract_slice(a, [128, 16], [128, 0])
    b0 = tlx.extract_slice(b, [16, 32], [0, 0])
    b1 = tlx.extract_slice(b, [16, 32], [0, 32])
    b2 = tlx.extract_slice(b, [16, 32], [0, 64])
    b3 = tlx.extract_slice(b, [16, 32], [0, 96])
    acc = tlx.zeros((256, 128), tl.float32, layout=mma)
    c00 = tlx.extract_slice(acc, [128, 32], [0, 0])
    c10 = tlx.extract_slice(acc, [128, 32], [128, 0])
    c01 = tlx.extract_slice(acc, [128, 32], [0, 32])
    c11 = tlx.extract_slice(acc, [128, 32], [128, 32])
    c02 = tlx.extract_slice(acc, [128, 32], [0, 64])
    c12 = tlx.extract_slice(acc, [128, 32], [128, 64])
    c03 = tlx.extract_slice(acc, [128, 32], [0, 96])
    c13 = tlx.extract_slice(acc, [128, 32], [128, 96])

    tl.debug_barrier()
    c00 = tlx.amd_scheduled_mfma(a_lo, b0, c00, accumulator_role="transient", initialize=True)
    c10 = tlx.amd_scheduled_mfma(a_hi, b0, c10, accumulator_role="transient", initialize=True)
    c01 = tlx.amd_scheduled_mfma(a_lo, b1, c01, accumulator_role="transient", initialize=True)
    c11 = tlx.amd_scheduled_mfma(a_hi, b1, c11, accumulator_role="transient", initialize=True)
    c02 = tlx.amd_scheduled_mfma(a_lo, b2, c02, accumulator_role="transient", initialize=True)
    c12 = tlx.amd_scheduled_mfma(a_hi, b2, c12, accumulator_role="transient", initialize=True)
    c03 = tlx.amd_scheduled_mfma(a_lo, b3, c03, accumulator_role="transient", initialize=True)
    c13 = tlx.amd_scheduled_mfma(a_hi, b3, c13, accumulator_role="transient", initialize=True)
    c00, _ = tlx.amd_mfma_commit(c00, b3)
    c10, _ = tlx.amd_mfma_commit(c10, b3)
    c01, _ = tlx.amd_mfma_commit(c01, b3)
    c11, _ = tlx.amd_mfma_commit(c11, b3)
    c02, _ = tlx.amd_mfma_commit(c02, b3)
    c12, _ = tlx.amd_mfma_commit(c12, b3)
    c03, _ = tlx.amd_mfma_commit(c03, b3)
    c13, _ = tlx.amd_mfma_commit(c13, b3)

    fragment_rows = tl.arange(0, 128)
    fragment_cols = tl.arange(0, 32)
    out00 = output_ptr + fragment_rows[:, None] * 128 + fragment_cols[None, :]
    out10 = out00 + 128 * 128
    out01 = out00 + 32
    out11 = out10 + 32
    out02 = out00 + 64
    out12 = out10 + 64
    out03 = out00 + 96
    out13 = out10 + 96
    tl.store(tlx.require_layout(out00, mma, pin=False), c00)
    tl.store(tlx.require_layout(out10, mma, pin=False), c10)
    tl.store(tlx.require_layout(out01, mma, pin=False), c01)
    tl.store(tlx.require_layout(out11, mma, pin=False), c11)
    tl.store(tlx.require_layout(out02, mma, pin=False), c02)
    tl.store(tlx.require_layout(out12, mma, pin=False), c12)
    tl.store(tlx.require_layout(out03, mma, pin=False), c03)
    tl.store(tlx.require_layout(out13, mma, pin=False), c13)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_amd_scheduled_mfma_fragmented_nd_correct_gfx950():
    torch.manual_seed(0)
    a = torch.randn((256, 16), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((16, 128), device="cuda", dtype=torch.bfloat16)
    actual = torch.empty((256, 128), device="cuda", dtype=torch.float32)
    _amd_scheduled_mfma_fragmented_nd_kernel[(1, )](
        a,
        b,
        actual,
        num_warps=4,
        matrix_instr_nonkdim=16,
    )
    expected = a.float() @ b.float()
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)


@triton.jit
def _amd_scheduled_mfma_fragmented_nd_update_kernel(
    a0_ptr,
    b0_ptr,
    a1_ptr,
    b1_ptr,
    output_ptr,
    DIRECT_STORE: tl.constexpr,
):
    """Match the GQA dK path: a full update, then eight persistent updates."""
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[32, 32, 16],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    dot0: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=8)
    dot1: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=8)
    rows = tl.arange(0, 256)
    reduction = tl.arange(0, 16)
    cols = tl.arange(0, 128)
    a0 = tlx.require_layout(
        tl.load(a0_ptr + rows[:, None] * 16 + reduction[None, :]),
        dot0,
        pin=False,
    )
    b0 = tlx.require_layout(
        tl.load(b0_ptr + reduction[:, None] * 128 + cols[None, :]),
        dot1,
        pin=False,
    )
    a1 = tlx.require_layout(
        tl.load(a1_ptr + rows[:, None] * 16 + reduction[None, :]),
        dot0,
        pin=False,
    )
    b1 = tlx.require_layout(
        tl.load(b1_ptr + reduction[:, None] * 128 + cols[None, :]),
        dot1,
        pin=False,
    )

    acc = tlx.zeros((256, 128), tl.float32, layout=mma)
    acc = tl.dot(a0, b0, acc)
    tl.debug_barrier()

    lhs0 = tlx.extract_slice(a1, [128, 16], [0, 0])
    lhs1 = tlx.extract_slice(a1, [128, 16], [128, 0])
    rhs0 = tlx.extract_slice(b1, [16, 32], [0, 0])
    rhs1 = tlx.extract_slice(b1, [16, 32], [0, 32])
    rhs2 = tlx.extract_slice(b1, [16, 32], [0, 64])
    rhs3 = tlx.extract_slice(b1, [16, 32], [0, 96])
    c00 = tlx.extract_slice(acc, [128, 32], [0, 0])
    c10 = tlx.extract_slice(acc, [128, 32], [128, 0])
    c01 = tlx.extract_slice(acc, [128, 32], [0, 32])
    c11 = tlx.extract_slice(acc, [128, 32], [128, 32])
    c02 = tlx.extract_slice(acc, [128, 32], [0, 64])
    c12 = tlx.extract_slice(acc, [128, 32], [128, 64])
    c03 = tlx.extract_slice(acc, [128, 32], [0, 96])
    c13 = tlx.extract_slice(acc, [128, 32], [128, 96])

    c00 = tlx.amd_scheduled_mfma(lhs0, rhs0, c00, accumulator_role="persistent")
    c10 = tlx.amd_scheduled_mfma(lhs1, rhs0, c10, accumulator_role="persistent")
    c01 = tlx.amd_scheduled_mfma(lhs0, rhs1, c01, accumulator_role="persistent")
    c11 = tlx.amd_scheduled_mfma(lhs1, rhs1, c11, accumulator_role="persistent")
    c02 = tlx.amd_scheduled_mfma(lhs0, rhs2, c02, accumulator_role="persistent")
    c12 = tlx.amd_scheduled_mfma(lhs1, rhs2, c12, accumulator_role="persistent")
    c03 = tlx.amd_scheduled_mfma(lhs0, rhs3, c03, accumulator_role="persistent")
    c13 = tlx.amd_scheduled_mfma(lhs1, rhs3, c13, accumulator_role="persistent")

    if DIRECT_STORE:
        fragment_rows = tl.arange(0, 128)
        fragment_cols = tl.arange(0, 32)
        out00 = output_ptr + fragment_rows[:, None] * 128 + fragment_cols[None, :]
        out10 = out00 + 128 * 128
        out01 = out00 + 32
        out11 = out10 + 32
        out02 = out00 + 64
        out12 = out10 + 64
        out03 = out00 + 96
        out13 = out10 + 96
        tl.store(tlx.require_layout(out00, mma, pin=False), c00)
        tl.store(tlx.require_layout(out10, mma, pin=False), c10)
        tl.store(tlx.require_layout(out01, mma, pin=False), c01)
        tl.store(tlx.require_layout(out11, mma, pin=False), c11)
        tl.store(tlx.require_layout(out02, mma, pin=False), c02)
        tl.store(tlx.require_layout(out12, mma, pin=False), c12)
        tl.store(tlx.require_layout(out03, mma, pin=False), c03)
        tl.store(tlx.require_layout(out13, mma, pin=False), c13)
    else:
        row0 = tl.cat(
            tl.cat(c00, c01, dim=1),
            tl.cat(c02, c03, dim=1),
            dim=1,
        )
        row1 = tl.cat(
            tl.cat(c10, c11, dim=1),
            tl.cat(c12, c13, dim=1),
            dim=1,
        )
        result = tlx.require_layout(tl.cat(row0, row1, dim=0), mma, pin=False)
        offsets = tlx.require_layout(
            output_ptr + rows[:, None] * 128 + cols[None, :],
            mma,
            pin=False,
        )
        tl.store(offsets, result)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize("direct_store", [True, False])
def test_amd_scheduled_mfma_fragmented_nd_update_correct_gfx950(direct_store, ):
    torch.manual_seed(0)
    a0 = torch.randn((256, 16), device="cuda", dtype=torch.bfloat16)
    b0 = torch.randn((16, 128), device="cuda", dtype=torch.bfloat16)
    a1 = torch.randn((256, 16), device="cuda", dtype=torch.bfloat16)
    b1 = torch.randn((16, 128), device="cuda", dtype=torch.bfloat16)
    actual = torch.empty((256, 128), device="cuda", dtype=torch.float32)
    _amd_scheduled_mfma_fragmented_nd_update_kernel[(1, )](
        a0,
        b0,
        a1,
        b1,
        actual,
        DIRECT_STORE=direct_store,
        num_warps=4,
        matrix_instr_nonkdim=16,
    )
    expected = a0.float() @ b0.float() + a1.float() @ b1.float()
    torch.testing.assert_close(actual, expected, atol=4e-4, rtol=4e-4)


@triton.jit
def _amd_scheduled_mfma_interleaved_chains_kernel(a_ptr, b_ptr, output_ptr, BANDS: tl.constexpr):
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    dot0: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=8)
    dot1: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=8)
    rows = tl.arange(0, 16)
    reduction = tl.arange(0, 256)
    cols = tl.arange(0, 128)
    a = tl.load(a_ptr + rows[:, None] * 256 + reduction[None, :])
    b = tl.load(b_ptr + reduction[:, None] * 128 + cols[None, :])
    a = tlx.require_layout(a, dot0, pin=False)
    b = tlx.require_layout(b, dot1, pin=False)
    b = tlx.amd_register_resident(b, register_class="agpr", registers_per_group=32)
    acc0 = tlx.zeros((16, 64), tl.float32, layout=mma)
    acc1 = tlx.zeros((16, 64), tl.float32, layout=mma)

    for band in tl.static_range(BANDS):
        a_band = tlx.extract_slice(a, [16, 32], [0, band * 32])
        b0 = tlx.extract_slice(b, [32, 64], [band * 32, 0])
        b1 = tlx.extract_slice(b, [32, 64], [band * 32, 64])
        acc0 = tlx.amd_scheduled_mfma(
            a_band,
            b0,
            acc0,
            resident_operand=1,
            accumulator_role="transient",
            initialize=band == 0,
        )
        acc1 = tlx.amd_scheduled_mfma(
            a_band,
            b1,
            acc1,
            resident_operand=1,
            accumulator_role="transient",
            initialize=band == 0,
        )
    acc0, acc1, b1 = tlx.amd_mfma_commit((acc0, acc1), b1)
    half_cols = tl.arange(0, 64)
    output_offsets0 = output_ptr + rows[:, None] * 128 + half_cols[None, :]
    output_offsets1 = output_offsets0 + 64
    output_offsets0 = tlx.require_layout(output_offsets0, mma, pin=False)
    output_offsets1 = tlx.require_layout(output_offsets1, mma, pin=False)
    tl.store(output_offsets0, acc0)
    tl.store(output_offsets1, acc1)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_amd_scheduled_mfma_interleaved_chains_correct_gfx950():
    torch.manual_seed(0)
    a = torch.randn((16, 256), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((256, 128), device="cuda", dtype=torch.bfloat16)
    actual = torch.empty((16, 128), device="cuda", dtype=torch.float32)
    _amd_scheduled_mfma_interleaved_chains_kernel[(1, )](
        a,
        b,
        actual,
        BANDS=8,
        num_warps=4,
        matrix_instr_nonkdim=16,
    )
    expected = a.float() @ b.float()
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)


@triton.jit
def _amd_scheduled_mfma_split_resident_chains_kernel(
    a_ptr,
    b_ptr,
    v_ptr,
    output_ptr,
    v_output_ptr,
    USE_LOCAL: tl.constexpr,
    EXACT_LOCAL_LAYOUT: tl.constexpr,
    FULL_COMMIT: tl.constexpr,
):
    """Match dQ's 128+64+32+32 resident-K decomposition."""
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    v_mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    dot0: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=8)
    dot1: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=8)
    v_layout: tl.constexpr = tlx.dot_operand_layout(0, v_mma, k_width=8)
    rows = tl.arange(0, 16)
    reduction = tl.arange(0, 256)
    cols = tl.arange(0, 128)
    a = tlx.require_layout(
        tl.load(a_ptr + rows[:, None] * 256 + reduction[None, :]),
        dot0,
        pin=False,
    )
    b = tl.load(b_ptr + reduction[:, None] * 128 + cols[None, :])
    if USE_LOCAL:
        if EXACT_LOCAL_LAYOUT:
            b_smem_layout: tl.constexpr = (tlx.shared_linear_layout_encoding(
                offset_bases=[
                    [0, 1],
                    [0, 2],
                    [0, 4],
                    [0, 8],
                    [0, 64],
                    [1, 0],
                    [2, 0],
                    [4, 0],
                    [8, 64],
                    [0, 16],
                    [0, 32],
                    [16, 0],
                    [32, 0],
                    [64, 0],
                    [128, 0],
                ],
                block_bases=[],
                alignment=16,
            ))
            b_buffers = tlx.local_alloc(
                (256, 128),
                tl.bfloat16,
                1,
                layout=b_smem_layout,
            )
        else:
            b_buffers = tlx.local_alloc((256, 128), tl.bfloat16, 1)
        b_buffer = tlx.local_view(b_buffers, 0)
        tlx.local_store(b_buffer, b)
        tl.debug_barrier()
        b_lo = tlx.local_load(
            tlx.local_slice(b_buffer, [0, 0], [128, 128]),
            layout=dot1,
            relaxed=True,
        )
        b_mid = tlx.local_load(
            tlx.local_slice(b_buffer, [128, 0], [64, 128]),
            layout=dot1,
            relaxed=True,
        )
        b6 = tlx.local_load(
            tlx.local_slice(b_buffer, [192, 0], [32, 128]),
            layout=dot1,
            relaxed=True,
        )
        b7 = tlx.local_load(
            tlx.local_slice(b_buffer, [224, 0], [32, 128]),
            layout=dot1,
            relaxed=True,
        )
    else:
        b = tlx.require_layout(b, dot1, pin=False)
        b_lo = tlx.extract_slice(b, [128, 128], [0, 0])
        b_mid = tlx.extract_slice(b, [64, 128], [128, 0])
        b6 = tlx.extract_slice(b, [32, 128], [192, 0])
        b7 = tlx.extract_slice(b, [32, 128], [224, 0])

    acc0 = tlx.zeros((16, 64), tl.float32, layout=mma)
    acc1 = tlx.zeros((16, 64), tl.float32, layout=mma)
    for band in tl.static_range(4):
        a_band = tlx.extract_slice(a, [16, 32], [0, band * 32])
        b0 = tlx.extract_slice(b_lo, [32, 64], [band * 32, 0])
        b1 = tlx.extract_slice(b_lo, [32, 64], [band * 32, 64])
        acc0 = tlx.amd_scheduled_mfma(
            a_band,
            b0,
            acc0,
            resident_operand=1,
            accumulator_role="transient",
            initialize=band == 0,
        )
        acc1 = tlx.amd_scheduled_mfma(
            a_band,
            b1,
            acc1,
            resident_operand=1,
            accumulator_role="transient",
            initialize=band == 0,
        )
    for band in tl.static_range(2):
        a_band = tlx.extract_slice(a, [16, 32], [0, (band + 4) * 32])
        b0 = tlx.extract_slice(b_mid, [32, 64], [band * 32, 0])
        b1 = tlx.extract_slice(b_mid, [32, 64], [band * 32, 64])
        acc0 = tlx.amd_scheduled_mfma(
            a_band,
            b0,
            acc0,
            resident_operand=1,
            accumulator_role="transient",
        )
        acc1 = tlx.amd_scheduled_mfma(
            a_band,
            b1,
            acc1,
            resident_operand=1,
            accumulator_role="transient",
        )
    a_band6 = tlx.extract_slice(a, [16, 32], [0, 192])
    b60 = tlx.extract_slice(b6, [32, 64], [0, 0])
    b61 = tlx.extract_slice(b6, [32, 64], [0, 64])
    acc0 = tlx.amd_scheduled_mfma(
        a_band6,
        b60,
        acc0,
        resident_operand=1,
        accumulator_role="transient",
    )
    acc1 = tlx.amd_scheduled_mfma(
        a_band6,
        b61,
        acc1,
        resident_operand=1,
        accumulator_role="transient",
    )
    a_band7 = tlx.extract_slice(a, [16, 32], [0, 224])
    b70 = tlx.extract_slice(b7, [32, 64], [0, 0])
    b71 = tlx.extract_slice(b7, [32, 64], [0, 64])
    acc0 = tlx.amd_scheduled_mfma(
        a_band7,
        b70,
        acc0,
        resident_operand=1,
        accumulator_role="transient",
    )
    acc1 = tlx.amd_scheduled_mfma(
        a_band7,
        b71,
        acc1,
        resident_operand=1,
        accumulator_role="transient",
    )
    if FULL_COMMIT:
        v_resident = tlx.require_layout(
            tl.load(v_ptr + reduction[:, None] * 128 + cols[None, :]),
            v_layout,
            pin=False,
        )
        acc0, acc1, v_resident = tlx.amd_mfma_commit((acc0, acc1), v_resident)
        v_offsets = tlx.require_layout(
            v_output_ptr + reduction[:, None] * 128 + cols[None, :],
            v_layout,
            pin=False,
        )
        tl.store(v_offsets, v_resident)
    else:
        acc0, acc1, b71 = tlx.amd_mfma_commit((acc0, acc1), b71)
    half_cols = tl.arange(0, 64)
    output_offsets0 = tlx.require_layout(
        output_ptr + rows[:, None] * 128 + half_cols[None, :],
        mma,
        pin=False,
    )
    output_offsets1 = tlx.require_layout(
        output_ptr + rows[:, None] * 128 + half_cols[None, :] + 64,
        mma,
        pin=False,
    )
    tl.store(output_offsets0, acc0)
    tl.store(output_offsets1, acc1)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize(
    "exact_local_layout,full_commit",
    [(False, False), (True, False), (True, True)],
)
def test_amd_scheduled_mfma_split_resident_chains_correct_gfx950(
    exact_local_layout,
    full_commit,
):
    torch.manual_seed(0)
    a = torch.randn((16, 256), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((256, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn((256, 128), device="cuda", dtype=torch.bfloat16)
    actual = torch.empty((16, 128), device="cuda", dtype=torch.float32)
    v_actual = torch.empty_like(v)
    _amd_scheduled_mfma_split_resident_chains_kernel[(1, )](
        a,
        b,
        v,
        actual,
        v_actual,
        USE_LOCAL=True,
        EXACT_LOCAL_LAYOUT=exact_local_layout,
        FULL_COMMIT=full_commit,
        num_warps=4,
        matrix_instr_nonkdim=16,
    )
    expected = a.float() @ b.float()
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
    if full_commit:
        torch.testing.assert_close(v_actual, v)


# ---------------------------------------------------------------------------
# Reproducer: restructuring a loaded value must not release its pinned offsets.
# ---------------------------------------------------------------------------


@triton.jit
def _load_then_restructure(base, offsets):
    value = tlx.buffer_load(base, offsets)
    value = tl.reshape(value, [4, 4, 16, 2, 2])
    value = tl.trans(value, (0, 4, 2, 3, 1))
    return tl.reshape(value, [128, 8])


@triton.jit
def _pinned_load_helper_kernel(src, dst, PHYSICAL: tl.constexpr):
    row = tl.arange(0, 4)[:, None]
    col = tl.arange(0, 256)[None, :]
    offsets = tlx.require_layout(row * 256 + col, PHYSICAL)
    value = _load_then_restructure(src, offsets)
    out_row = tl.arange(0, 128)[:, None]
    out_col = tl.arange(0, 8)[None, :]
    tl.store(dst + out_row * 8 + out_col, value)


def test_load_helper_preserves_pinned_offset_layout_gfx950():
    physical = tlx.layout(
        shape=((16, 4, 4), (2, 2, 4)),
        stride=((4, 64, 0), (1, 2, 256)),
    )
    compiled = compile_for_gfx950(
        _pinned_load_helper_kernel,
        signature={"src": "*fp8e4nv", "dst": "*fp8e4nv"},
        constexprs={"PHYSICAL": physical},
    )

    # The helper keeps the offset pin and derives layouts for the loaded value.
    assert "tlx.release_layout" not in compiled.asm["ttir"]


# ---------------------------------------------------------------------------
# Test: warp-pipelined batched matmul (bmm) with a partial-K tail on gfx950.
#
# Models the production "compression bmm" (batch, M, prime K=2309, N).
# The kernel mirrors the AMD warp-pipe addmm template (async_load prefetch
# into multi-buffered LDS, tlx.warp_pipeline_stage mfma/mem stages, B fed [N, K]
# K-contiguous + local_trans) plus a batch dimension addressed with a genuine
# 64-bit base (bid.to(tl.int64) * stride), as the real bmm requires (A can exceed
# 2**31 elements).
#
# Partial-K (K not a multiple of BLOCK_K) makes the async_load masked, which forces
# the async src blocked layout to sizePerThread=[1,1] (vec=1). fp16 x vec1 = 16-bit
# direct-to-LDS, which CDNA4 supports only at {32, 128} bits, so canLoadDirectToLDS()
# (third_party/amd/lib/TritonAMDGPUToLLVM/Utility.cpp) returns false and both
# async-copy conversion patterns bail. With no async_copy -> load+local_store
# fallback, ttg.async_copy_global_to_local is left unlowered and make_llir aborts:
#   error: LLVM Translation failed for operation: builtin.unrealized_conversion_cast
#   RuntimeError: failed to translate module to LLVM IR
# Aligned K (K % BLOCK_K == 0) coalesces to 128-bit and compiles + runs fine.
# ---------------------------------------------------------------------------


@triton.jit
def _warp_pipe_bmm_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    sab,
    sam,
    sak,
    sbb,
    sbn,
    sbk,
    scb,
    scm,
    scn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
):
    """C[b] = A[b] @ B[b]; B fed [b, N, K] (K-contiguous) + local_trans; 64-bit batch base."""
    bid = tl.program_id(1)
    pid = tl.program_id(0)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    # 64-bit base: batch offset can exceed 2**31 for the production shape.
    a_base = bid.to(tl.int64) * sab + offs_m[:, None].to(tl.int64) * sam
    b_base = bid.to(tl.int64) * sbb + offs_n[:, None].to(tl.int64) * sbn
    K_ITERS = tl.cdiv(K, BLOCK_K)

    smemA = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(A), NUM_BUFFERS)
    smemB = tlx.local_alloc((BLOCK_N, BLOCK_K), tlx.dtype_of(B), NUM_BUFFERS)

    for i in tl.range(0, NUM_BUFFERS, loop_unroll_factor=NUM_BUFFERS):
        ks = i * BLOCK_K
        m = offs_k[None, :] < K - ks  # partial-K mask (folds away when K % BLOCK_K == 0)
        ta = tlx.async_load(A + a_base + (ks + offs_k[None, :]) * sak, tlx.local_view(smemA, i), mask=m, other=0.0)
        tb = tlx.async_load(B + b_base + (ks + offs_k[None, :]) * sbk, tlx.local_view(smemB, i), mask=m, other=0.0)
        tlx.async_load_commit_group([ta, tb])

    tlx.async_load_wait_group(NUM_BUFFERS - 2)
    a_tile = tlx.local_load(tlx.local_view(smemA, 0))
    b_tile = tlx.local_load(tlx.local_trans(tlx.local_view(smemB, 0)))
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for tile_id in tl.range(0, K_ITERS - NUM_BUFFERS):
        pf = (tile_id % NUM_BUFFERS).to(tl.int32)
        nb = ((tile_id + 1) % NUM_BUFFERS).to(tl.int32)
        kpf = (tile_id + NUM_BUFFERS) * BLOCK_K
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)
        with tlx.warp_pipeline_stage("mem", priority=1):
            m = offs_k[None, :] < K - kpf
            ta = tlx.async_load(A + a_base + (kpf + offs_k[None, :]) * sak, tlx.local_view(smemA, pf), mask=m,
                                other=0.0)
            tb = tlx.async_load(B + b_base + (kpf + offs_k[None, :]) * sbk, tlx.local_view(smemB, pf), mask=m,
                                other=0.0)
            tlx.async_load_commit_group([ta, tb])
            a_tile = tlx.local_load(tlx.local_view(smemA, nb))
            b_tile = tlx.local_load(tlx.local_trans(tlx.local_view(smemB, nb)))
        tlx.async_load_wait_group(NUM_BUFFERS - 2)

    acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)
    tlx.async_load_wait_group(0)
    for i in tl.range(0, NUM_BUFFERS - 1, loop_unroll_factor=NUM_BUFFERS - 1):
        buf = ((K_ITERS - (NUM_BUFFERS - 1) + i) % NUM_BUFFERS).to(tl.int32)
        a_tile = tlx.local_load(tlx.local_view(smemA, buf))
        b_tile = tlx.local_load(tlx.local_trans(tlx.local_view(smemB, buf)))
        acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)

    ocm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    ocn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptr = C + bid.to(tl.int64) * scb + scm * ocm[:, None].to(tl.int64) + scn * ocn[None, :]
    tl.store(c_ptr, acc.to(tlx.dtype_of(C)), mask=(ocm[:, None] < M) & (ocn[None, :] < N))


def _run_warp_pipe_bmm(device, bt, M, N, K):
    """Build fp16 operands and launch the warp-pipe bmm (B fed [bt, N, K] for local_trans)."""
    BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS = 128, 64, 64, 2
    a = torch.randn((bt, M, K), device=device, dtype=torch.float16) * 0.1
    b = torch.randn((bt, K, N), device=device, dtype=torch.float16) * 0.1
    bT = b.transpose(1, 2).contiguous()  # [bt, N, K], K-contiguous
    c = torch.empty((bt, M, N), device=device, dtype=torch.float16)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), bt)
    _warp_pipe_bmm_kernel[grid](
        a,
        bT,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        bT.stride(0),
        bT.stride(1),
        bT.stride(2),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        NUM_BUFFERS=NUM_BUFFERS,
        num_warps=8,
        num_stages=1,
        matrix_instr_nonkdim=16,
    )
    return a, b, c


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_warp_pipe_bmm_aligned_k_gfx950(device):
    """Warp-pipe bmm with K a multiple of BLOCK_K compiles + runs correctly (positive control)."""
    a, b, c = _run_warp_pipe_bmm(device, bt=8, M=256, N=256, K=2560)  # 2560 % 64 == 0
    torch.testing.assert_close(c.float(), torch.bmm(a.float(), b.float()), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_warp_pipe_bmm_partial_k_gfx950(device):
    """Warp-pipe bmm with a partial-K tail (K not a multiple of BLOCK_K).

    Same kernel and config as the aligned-K positive control; only K differs (prime 2309, the
    production compression-bmm K). The partial-K mask makes the async_load un-lowerable as a
    direct-to-LDS copy on CDNA4 (vec=1 -> 16-bit); CoalesceAsyncCopy now falls back to a
    synchronous tt.load + ttg.local_store so it compiles and runs correctly.
    Previously this aborted make_llir with an unrealized_conversion_cast.
    """
    a, b, c = _run_warp_pipe_bmm(device, bt=8, M=256, N=256, K=2309)  # 2309 % 64 == 5
    torch.testing.assert_close(c.float(), torch.bmm(a.float(), b.float()), atol=2e-2, rtol=2e-2)


# ---------------------------------------------------------------------------
# Test: unmasked full-tile async_load with a non-16-aligned global row stride.
# ---------------------------------------------------------------------------


@triton.jit
def _row_stride_async_load_kernel(a_ptr, out_ptr, stride_am, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs = offs_m[:, None] * stride_am + offs_k[None, :]
    smem = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_ptr), 1)
    tok = tlx.async_load(a_ptr + offs, tlx.local_view(smem, 0))  # unmasked -- full tile
    tlx.async_load_commit_group([tok])
    tlx.async_load_wait_group(0)
    t = tlx.local_load(tlx.local_view(smem, 0))
    tl.store(out_ptr + offs_m[:, None] * BLOCK_K + offs_k[None, :], t)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize("K", [2320, 2309, 2312, 1956])
def test_async_load_row_stride_gfx950(device, K):
    """Unmasked full-tile async_load with a non-16-aligned global row stride (T280910119).

    A row stride not a multiple of 16 elements collapses the direct-to-LDS vector width
    below a supported bitwidth (fp16 -> 16-bit) on CDNA4, so the copy cannot be lowered as
    a direct-to-LDS load (its swizzled dst hits loadContig == 0). CoalesceAsyncCopy now
    falls back to a synchronous tt.load + ttg.local_store for both swizzled and padded
    dsts, so it compiles and runs correctly. Previously K % 16 != 0 aborted make_llir with
    an unrealized_conversion_cast. K=2320 (% 16 == 0) is the positive control and keeps the
    fast direct-to-LDS path.
    """
    BLOCK_M, BLOCK_K = 128, 64
    a = torch.randn((BLOCK_M, K), device=device, dtype=torch.float16)
    out = torch.empty((BLOCK_M, BLOCK_K), device=device, dtype=torch.float16)
    _row_stride_async_load_kernel[(1, )](a, out, a.stride(0), BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K)
    torch.testing.assert_close(out, a[:, :BLOCK_K])


# ---------------------------------------------------------------------------
# Test: non-contiguous gather-pointer async_load (bf16).
# ---------------------------------------------------------------------------


@triton.jit
def _noncontiguous_gather_async_load_kernel(V, out_ptr, stride_b, stride_po, stride_d, stride_x, N: tl.constexpr,
                                            HEAD_DIM: tl.constexpr, PAGE: tl.constexpr):
    n = tl.arange(0, N)
    d = tl.arange(0, HEAD_DIM)
    page = n // PAGE
    token = n % PAGE
    # V is laid out [block, page // 8, head_dim, 8]. Reconstructing the logical
    # [token, head_dim] tile makes the async-load pointer tensor non-contiguous
    # (a gather: sizePerThread=[1,1]).
    ptrs = (V + page[:, None] * stride_b + (token[:, None] // 8) * stride_po + d[None, :] * stride_d +
            (token[:, None] % 8) * stride_x)
    smem = tlx.local_alloc((N, HEAD_DIM), tlx.dtype_of(V), 2)
    tok = tlx.async_load(ptrs, tlx.local_view(smem, 0))
    tlx.async_load_commit_group([tok])
    tlx.async_load_wait_group(0)
    value = tlx.local_load(tlx.local_view(smem, 0))
    tl.store(out_ptr + n[:, None] * HEAD_DIM + d[None, :], value)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_async_load_noncontiguous_gather_gfx950(device):
    """Non-contiguous gather-pointer async_load in bf16 (P2440272260).

    A third way (besides a partial-K mask or a non-16-aligned row stride) to collapse the
    direct-to-LDS vector width to 16-bit on CDNA4: a genuinely non-contiguous pointer tensor.
    The gather offsets force the async src blocked layout to sizePerThread=[1,1] (vec=1), so
    bf16 -> 16-bit, canLoadDirectToLDS() rejects it (loadContig == 0), and CoalesceAsyncCopy
    falls back to a synchronous tt.load + ttg.local_store. Previously this aborted make_llir
    with an unrealized_conversion_cast. Uses bfloat16 -- the other 16-bit dtype; the mask and
    row-stride tests cover fp16.
    """
    N, HEAD_DIM, PAGE = 128, 64, 64
    v = torch.randn((2, 8, HEAD_DIM, 8), device=device, dtype=torch.bfloat16)
    out = torch.empty((N, HEAD_DIM), device=device, dtype=torch.bfloat16)
    _noncontiguous_gather_async_load_kernel[(1, )](v, out, *v.stride(), N=N, HEAD_DIM=HEAD_DIM, PAGE=PAGE, num_warps=4)
    n = torch.arange(N, device=device)
    page = n // PAGE
    token = n % PAGE
    ref = v[page, token // 8, :, token % 8]
    torch.testing.assert_close(out, ref)


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


@pytest.mark.skipif(not is_hip(), reason="Requires HIP runtime")
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


@pytest.mark.skipif(not is_hip(), reason="Requires HIP runtime")
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


@triton.jit
def _local_load_rematerialized_coordinates_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    buf = tlx.local_alloc((BLOCK_SIZE, ), tl.float32, 1)
    buf0 = tlx.local_view(buf, 0)
    tlx.local_store(buf0, tl.load(x_ptr + offs, mask=mask, other=0.0))
    tl.debug_barrier()
    values = tlx.local_load(buf0, rematerialize_coordinates=True)
    grouped_values = tlx.local_load(buf0, rematerialize_coordinates_group=3)
    tl.store(output_ptr + offs, values + grouped_values, mask=mask)


def test_local_load_rematerialized_coordinates_compiles_gfx950():
    compiled = compile_for_gfx950(
        _local_load_rematerialized_coordinates_kernel,
        signature={
            "x_ptr": "*fp32",
            "output_ptr": "*fp32",
            "n_elements": "i32",
        },
        constexprs={"BLOCK_SIZE": 256},
    )
    assert "tlx.rematerialize_coordinates" in compiled.asm["ttgir"]
    assert "tlx.rematerialize_coordinates_group = 3 : i32" in compiled.asm["ttgir"]
    assert 'asm sideeffect "", "=v,0"' in compiled.asm["llir"]


@triton.jit
def _padded_local_slice_transposed_load_kernel(x_ptr, rhs_ptr, output_ptr):
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    dot0: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=8)
    dot1: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=8)
    smem_layout: tl.constexpr = (tlx.padded_shared_layout_encoding.with_bases(
        [(512, 16)],
        [
            [1, 0],
            [2, 0],
            [0, 1],
            [0, 2],
            [4, 0],
            [0, 8],
            [8, 0],
            [0, 32],
            [0, 16],
            [0, 4],
            [0, 64],
            [0, 128],
        ],
        [16, 256],
    ))
    rows = tl.arange(0, 16)
    cols = tl.arange(0, 256)
    values = tl.load(x_ptr + rows[:, None] * 256 + cols[None, :])
    buffers = tlx.local_alloc((16, 256), tl.bfloat16, 1, layout=smem_layout)
    buffer = tlx.local_view(buffers, 0)
    tlx.local_store(buffer, values)
    tl.debug_barrier()
    band = tlx.local_load(
        tlx.local_slice(buffer, [0, 32], [16, 32]),
        layout=dot0,
        relaxed=True,
    )
    reduction = tl.arange(0, 32)
    output_cols = tl.arange(0, 64)
    rhs = tl.load(rhs_ptr + reduction[:, None] * 64 + output_cols[None, :])
    rhs = tlx.require_layout(rhs, dot1, pin=False)
    accumulator = tlx.zeros((16, 64), tl.float32, layout=mma)
    result = tlx.amd_scheduled_mfma(
        band,
        rhs,
        accumulator,
        resident_operand=1,
        accumulator_role="transient",
        initialize=True,
    )
    output_offsets = rows[:, None] * 64 + output_cols[None, :]
    output_ptrs = output_ptr + output_offsets
    output_ptrs = tlx.require_layout(output_ptrs, mma, pin=False)
    tl.store(output_ptrs, result)


def test_padded_local_slice_uses_transposed_lds_read_gfx950():
    """A padded dS-style subslice should retain the CDNA4 transposed load."""
    compiled = compile_for_gfx950(
        _padded_local_slice_transposed_load_kernel,
        signature={
            "x_ptr": "*bf16",
            "rhs_ptr": "*bf16",
            "output_ptr": "*fp32",
        },
        constexprs={},
    )
    amdgcn = compiled.asm["amdgcn"]
    assert "ds_read_b64_tr_b16" in amdgcn
    assert "ds_read_u16" not in amdgcn


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
    for i in tl.range(0, NUM_ITERS, num_stages=1):
        tlx.async_load_wait_group(0)
        x = tlx.local_load(buf0)
        acc += x

    tl.store(output_ptr + offs, acc, mask=mask)


@pytest.mark.skipif(not is_hip(), reason="Requires HIP runtime")
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

    for k in tl.range(0, K_ITERS - 1, num_stages=1):
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


@pytest.mark.skipif(not is_hip(), reason="Requires HIP runtime")
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
    assert "clamp_bounds" in ttgir
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
def _async_amd_desc_load_fused_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    a_desc = tl.make_tensor_descriptor(a_ptr, [M, N], [N, 1], [M, N])
    b_desc = tl.make_tensor_descriptor(b_ptr, [M, N], [N, 1], [M, N])
    a_buf = tlx.local_alloc((M, N), tl.float16, 1)
    b_buf = tlx.local_alloc((M, N), tl.float16, 1)
    a_smem = tlx.local_view(a_buf, 0)
    b_smem = tlx.local_view(b_buf, 0)
    a_desc = tlx.update_tensor_descriptor(a_desc, add_offsets=[0, 0], pred=True, clamp_bounds=True)
    b_desc = tlx.update_tensor_descriptor(b_desc, add_offsets=[0, 0], pred=True, clamp_bounds=True)
    token = tlx.async_amd_descriptor_load_fused([
        (a_desc, a_smem, 0b0011),
        (b_desc, b_smem, 0b1100),
    ])
    tlx.async_amd_descriptor_wait(tokens=[token])
    result = tlx.local_load(a_smem) + tlx.local_load(b_smem)
    offsets = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    tl.store(output_ptr + offsets, result)


@pytest.mark.skipif(not is_hip(), reason="Requires HIP runtime")
def test_async_amd_desc_load_fused_compiles_gfx1250(device):
    """Two positioned TLX descriptors lower to one fused TDM instruction."""
    compiled = compile_for_gfx1250(
        _async_amd_desc_load_fused_kernel,
        signature={"a_ptr": "*fp16", "b_ptr": "*fp16", "output_ptr": "*fp16"},
        constexprs={"M": 16, "N": 32},
    )
    ttgir = compiled.asm["ttgir"]
    assert "amdg.async_tdm_fused_copy_global_to_local" in ttgir
    assert "warp_used_hints = array<i32: 3, 12>" in ttgir
    assert len(re.findall(r"tensor_load_to_lds|tensor\.load\.to\.lds", compiled.asm["amdgcn"])) == 1


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250 hardware")
def test_async_amd_desc_load_fused_correctness_gfx1250(device):
    rows, cols = 16, 32
    a = torch.randn((rows, cols), device=device, dtype=torch.float16)
    b = torch.randn((rows, cols), device=device, dtype=torch.float16)
    output = torch.empty_like(a)
    _async_amd_desc_load_fused_kernel[(1, )](a, b, output, M=rows, N=cols)
    torch.testing.assert_close(output, a + b)


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
    assert ttgir.count("clamp_bounds") == 2


@triton.jit
def _update_tensor_descriptor_store_kernel(
    x_ptr,
    y_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    desc_in = tl.make_tensor_descriptor(x_ptr, [M, N], [N, 1], [M, N])
    desc_out = tl.make_tensor_descriptor(y_ptr, [M, N], [N, 1], [M, N])
    load_buf = tlx.local_alloc((M, N), tl.float16, 1)
    store_buf = tlx.local_alloc((M, N), tl.float16, 1)
    load_view = tlx.local_view(load_buf, 0)
    store_view = tlx.local_view(store_buf, 0)

    pred = tl.program_id(0) == 0
    desc_in = tlx.update_tensor_descriptor(desc_in, set_bounds=[M, N], pred=pred)
    offset_m = desc_in.shape[0] - M
    offset_n = (desc_in.strides[1] - 1).to(tl.int32)
    desc_in = tlx.update_tensor_descriptor(desc_in, add_offsets=[offset_m, offset_n])
    tlx.async_amd_descriptor_load(desc_in, load_view)
    tlx.async_amd_descriptor_wait(0)
    tlx.local_store(store_view, tlx.local_load(load_view))

    desc_out = tlx.update_tensor_descriptor(desc_out, add_offsets=[0, 0], clamp_bounds=True)
    tlx.async_amd_descriptor_store(desc_out, store_view)
    tlx.async_amd_descriptor_wait(0)


@pytest.mark.skipif(not is_hip(), reason="Requires HIP runtime")
def test_update_tensor_descriptor_store_compiles_gfx1250(device):
    compiled = compile_for_gfx1250(
        _update_tensor_descriptor_store_kernel,
        signature={"x_ptr": "*fp16", "y_ptr": "*fp16"},
        constexprs={"M": 32, "N": 32},
    )
    ttgir = compiled.asm["ttgir"]
    assert "amdg.update_tensor_descriptor" in ttgir
    assert "set_bounds =" in ttgir
    assert "pred =" in ttgir
    assert "clamp_bounds" in ttgir
    assert "amdg.async_tdm_copy_local_to_global" in ttgir
    assert ttgir.count("clamp_bounds") == 1
    # Two explicit input updates and one output update are the only descriptor
    # mutations. Neither pre-positioned copy may add a no-op update.
    assert ttgir.count("amdg.update_tensor_descriptor") == 3


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250 hardware")
def test_update_tensor_descriptor_roundtrip_gfx1250(device):
    x = torch.randn((32, 32), dtype=torch.float16, device=device)
    y = torch.empty_like(x)
    _update_tensor_descriptor_store_kernel[(1, )](x, y, M=32, N=32)
    torch.testing.assert_close(x, y)


@triton.jit
def _invalid_update_tensor_descriptor_kernel(x_ptr, MODE: tl.constexpr):
    desc = tl.make_tensor_descriptor(x_ptr, [16, 16], [16, 1], [16, 16])
    if MODE == 0:
        desc = tlx.update_tensor_descriptor(desc)
    elif MODE == 1:
        desc = tlx.update_tensor_descriptor(desc, pred=True, clamp_bounds=True)
    elif MODE == 2:
        desc = tlx.update_tensor_descriptor(
            desc,
            add_offsets=[0, 0],
            set_bounds=[16, 16],
            clamp_bounds=True,
        )
    elif MODE == 3:
        desc = tlx.update_tensor_descriptor(desc, add_offsets=[0])
    else:
        desc = tlx.update_tensor_descriptor(desc, add_offsets=[0, 0])


@pytest.mark.parametrize(
    "mode, error",
    [
        (0, "requires at least one"),
        (1, "clamp_bounds requires add_offsets"),
        (2, "clamp_bounds and set_bounds are mutually exclusive"),
        (3, "add_offsets must have length 2"),
    ],
)
def test_update_tensor_descriptor_rejects_invalid_gfx1250(mode, error):
    with pytest.raises(CompilationError, match=error):
        compile_for_gfx1250(
            _invalid_update_tensor_descriptor_kernel,
            signature={"x_ptr": "*fp16"},
            constexprs={"MODE": mode},
        )


def test_update_tensor_descriptor_rejects_unsupported_target():
    with pytest.raises(CompilationError, match="only available on AMD TDM-capable targets"):
        compile_for_gfx950(
            _invalid_update_tensor_descriptor_kernel,
            signature={"x_ptr": "*fp16"},
            constexprs={"MODE": 4},
        )


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


@triton.jit
def _pinned_tdm_memdesc_view_kernel(
    input_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    VIEW: tl.constexpr,
):
    smem_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for([(N, 128 // 16)], [M, N])
    desc = tl.make_tensor_descriptor(input_ptr, [M, N], [N, 1], [M, N])
    buffers = tlx.local_alloc((M, N), tl.float16, 1, layout=smem_layout)
    full = tlx.local_view(buffers, 0)

    token = tlx.async_amd_descriptor_load(desc, full, [0, 0])
    tlx.async_amd_descriptor_wait(tokens=[token])

    if VIEW == 0:
        view = tlx.local_slice(full, [0, 32], [M, 32])
        rows = tl.arange(0, M)
        cols = tl.arange(0, 32)
        width: tl.constexpr = 32
    else:
        view = tlx.local_reshape(full, [N, M])
        rows = tl.arange(0, N)
        cols = tl.arange(0, M)
        width: tl.constexpr = M

    values = tlx.local_load(view)
    tl.store(output_ptr + rows[:, None] * width + cols[None, :], values)


@pytest.mark.skipif(not is_hip(), reason="Requires HIP runtime")
@pytest.mark.parametrize(
    "view, expected_op",
    [(0, "ttg.memdesc_subslice"), (1, "ttg.memdesc_reshape")],
    ids=["slice", "reshape"],
)
def test_pinned_tdm_memdesc_views_compile_gfx1250(device, view, expected_op):
    compiled = compile_for_gfx1250(
        _pinned_tdm_memdesc_view_kernel,
        signature={"input_ptr": "*fp16", "output_ptr": "*fp16"},
        constexprs={"M": 32, "N": 128, "VIEW": view},
    )
    ttgir = compiled.asm["ttgir"]
    assert ttgir.count("amdg.async_tdm_copy_global_to_local") == 1
    assert expected_op in ttgir
    assert "#ttg.padded_shared<[128:+8]" in ttgir
    assert "#tlx.user_layout" not in ttgir
    assert "#tlx.no_verify_layout" not in ttgir
    assert "tlx.require_layout" not in ttgir
    amdgcn = compiled.asm["amdgcn"]
    assert "tensor_load_to_lds" in amdgcn or "tensor.load.to.lds" in amdgcn


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


# ---------------------------------------------------------------------------
# Test: tlx.local_reshape reinterprets a flat LDS buffer as a 2D tile.
# ---------------------------------------------------------------------------


@triton.jit
def _local_reshape_kernel(
    input_ptr,
    output_ptr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    offsets = tl.arange(0, ROWS * COLS)
    values = tl.load(input_ptr + offsets)

    flat_buffers = tlx.local_alloc((ROWS * COLS, ), tl.float32, 1)
    flat = tlx.local_view(flat_buffers, 0)
    tlx.local_store(flat, values)

    reshaped = tlx.local_reshape(flat, [ROWS, COLS])
    result = tlx.local_load(reshaped)

    offs_m = tl.arange(0, ROWS)
    offs_n = tl.arange(0, COLS)
    output_offsets = offs_m[:, None] * COLS + offs_n[None, :]
    tl.store(output_ptr + output_offsets, result)


def test_local_reshape_compiles_gfx1250(device):
    """tlx.local_reshape should lower to ttg.memdesc_reshape and compile."""
    compiled = compile_for_gfx1250(
        _local_reshape_kernel,
        signature={"input_ptr": "*fp32", "output_ptr": "*fp32"},
        constexprs={"ROWS": 8, "COLS": 8},
    )
    ttgir = compiled.asm["ttgir"]
    assert "ttg.memdesc_reshape" in ttgir, ("expected memdesc_reshape in TTGIR, got:\n" + ttgir)
    assert "amdgcn" in compiled.asm
    assert len(compiled.asm["amdgcn"]) > 0


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250 hardware")
def test_local_reshape_correctness_gfx1250(device):
    """End-to-end: local_reshape reinterprets a flat LDS buffer as a 2D tile."""
    rows, cols = 8, 8
    inp = torch.arange(rows * cols, dtype=torch.float32, device=device)
    out = torch.empty((rows, cols), dtype=torch.float32, device=device)
    _local_reshape_kernel[(1, )](inp, out, ROWS=rows, COLS=cols)
    torch.testing.assert_close(out, inp.reshape(rows, cols))


# ---------------------------------------------------------------------------
# Test: mxfp TDM-pipelined GEMM compiles on gfx1250 with TDM + dot_scaled + WMMA.
# ---------------------------------------------------------------------------


@triton.jit
def _dot_scaled_tiles_per_warp_kernel(
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    c_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
    TILES_PER_WARP: tl.constexpr,
):
    block_k_scale: tl.constexpr = BLOCK_K // SCALE_BLOCK
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_ks = tl.arange(0, block_k_scale)

    a = tl.load(a_ptr + offs_m[:, None] * BLOCK_K + offs_k[None, :])
    b = tl.load(b_ptr + offs_k[:, None] * BLOCK_N + offs_n[None, :])
    a_scale = tl.load(a_scale_ptr + offs_m[:, None] * block_k_scale + offs_ks[None, :])
    b_scale = tl.load(b_scale_ptr + offs_n[:, None] * block_k_scale + offs_ks[None, :])

    acc = tlx.dot_scaled(a, a_scale, "e5m2", b, b_scale, "e5m2", tiles_per_warp=TILES_PER_WARP)
    tl.store(c_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :], acc)


def _compile_dot_scaled_tiles_per_warp(tiles_per_warp):
    src = ASTSource(
        fn=_dot_scaled_tiles_per_warp_kernel,
        signature={
            "a_ptr": "*fp8e5",
            "b_ptr": "*fp8e5",
            "a_scale_ptr": "*i8",
            "b_scale_ptr": "*i8",
            "c_ptr": "*fp32",
        },
        constexprs={
            "BLOCK_M": 256,
            "BLOCK_N": 256,
            "BLOCK_K": 128,
            "SCALE_BLOCK": 32,
            "TILES_PER_WARP": tiles_per_warp,
        },
    )
    return triton_compile(src, target=GPUTarget("hip", "gfx1250", 32))


def test_dot_scaled_tiles_per_warp_attr_gfx1250():
    compiled = _compile_dot_scaled_tiles_per_warp((2, 2))
    assert "amdg.wmma_tiles_per_warp = array<i32: 2, 2>" in compiled.asm["ttir"]
    assert "#ttg.amd_wmma" in compiled.asm["ttgir"]


@pytest.mark.parametrize(
    "tiles_per_warp, error",
    [
        ((1, ), "tiles_per_warp requires 2 entries"),
        ((0, 1), "tiles_per_warp entries must be positive"),
    ],
)
def test_dot_scaled_tiles_per_warp_rejects_invalid_gfx1250(tiles_per_warp, error):
    with pytest.raises(CompilationError, match=error):
        _compile_dot_scaled_tiles_per_warp(tiles_per_warp)


@triton.jit
def _require_amd_wmma_layout_kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
    offs_m = tl.arange(0, BLOCK)
    offs_n = tl.arange(0, BLOCK)
    offsets = offs_m[:, None] * BLOCK + offs_n[None, :]
    values = tl.load(x_ptr + offsets)
    values = tlx.require_amd_wmma_layout(
        values,
        version=3,
        transposed=True,
        warp_bases=((0, 2), (2, 0)),
        reg_bases=((0, 1), (1, 0)),
        instr_shape=(16, 16, 128),
    )
    offsets = tlx.require_amd_wmma_layout(
        offsets,
        version=3,
        transposed=True,
        warp_bases=((0, 2), (2, 0)),
        reg_bases=((0, 1), (1, 0)),
        instr_shape=(16, 16, 128),
    )
    tl.store(y_ptr + offsets, values)


def test_require_amd_wmma_layout_compiles_gfx1250():
    compiled = compile_for_gfx1250(
        _require_amd_wmma_layout_kernel,
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32"},
        constexprs={"BLOCK": 256},
    )
    assert "#ttg.amd_wmma" in compiled.asm["ttgir"]


def test_mxgemm_tdm_pipelined_compiles_gfx1250(device):
    """The mxfp GEMM tutorial kernel should lower to TDM + dot_scaled + WMMA."""
    compiled = compile_for_gfx1250(
        _amd_mxfp_gemm_kernel,
        signature={
            "a_ptr": "*fp8e5",
            "b_ptr": "*fp8e5",
            "c_ptr": "*fp32",
            "a_scale": "*i8",
            "b_scale": "*i8",
            "M": "i32",
            "N": "i32",
            "K": "i32",
            "stride_am": "i64",
            "stride_ak": "i64",
            "stride_bk": "i64",
            "stride_bn": "i64",
            "stride_cm": "i64",
            "stride_cn": "i64",
            "stride_scale": "i64",
        },
        constexprs={
            "DTYPE_A": "e5m2",
            "DTYPE_B": "e5m2",
            "SCALE_BLOCK": 32,
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 128,
            "GROUP_SIZE_M": 8,
            "TRANSPOSE_B": True,
            "NUM_BUFFERS": 2,
        },
    )
    ttgir = compiled.asm["ttgir"]
    amdgcn = compiled.asm["amdgcn"]
    assert "amdg.async_tdm_copy_global_to_local" in ttgir
    assert "tt.dot_scaled" in ttgir
    assert "tensor_load_to_lds" in amdgcn or "tensor.load.to.lds" in amdgcn
    assert "wmma" in amdgcn


def test_tlx_gfx9_gemm_bench_parses_shapes_and_defaults():
    bench = _load_tlx_gfx9_gemm_bench_module()

    assert not hasattr(bench, "DEVICE")
    assert set(bench.VERSION_MAP) == set(range(10))
    assert set(bench.PROVIDER_LABELS) == {"rocblas", "tlx"}
    assert bench.provider_defaults(9) == ["rocblas", "tlx"]
    assert bench.provider_defaults(0) == ["rocblas", "tlx"]
    assert bench.parse_shape("128x256x64") == (128, 256, 64)
    assert bench.parse_shape("128,256,64") == (128, 256, 64)
    with pytest.raises(Exception, match="shape dimensions must be positive"):
        bench.parse_shape("128x0x64")
    with pytest.raises(Exception, match="shape must be MxNxK"):
        bench.parse_shape("128x256")
    bench.validate_shape_for_providers((256, 256, 64), 0, ["tlx"])
    bench.validate_shape_for_providers((128, 128, 64), 9, ["rocblas"])
    with pytest.raises(Exception, match="M to be a multiple of 256"):
        bench.validate_shape_for_providers((128, 256, 64), 9, ["tlx"])
    with pytest.raises(Exception, match="N to be a multiple of 256"):
        bench.validate_shape_for_providers((256, 128, 64), 9, ["tlx"])
    with pytest.raises(Exception, match="K to be a multiple of 64"):
        bench.validate_shape_for_providers((256, 256, 96), 2, ["tlx"])
    with pytest.raises(Exception, match="prefetch two 64-wide K tiles"):
        bench.validate_shape_for_providers((256, 256, 64), 9, ["tlx"])
    bench.validate_shape_for_providers((256, 256, 128), 9, ["tlx"])


def test_tlx_gfx9_gemm_bench_input_modes_are_deterministic():
    bench = _load_tlx_gfx9_gemm_bench_module("_tlx_amd_test_gfx9_bench_inputs")
    inter_wave = _load_tlx_gfx9_inter_wave_bench_module("_tlx_amd_test_gfx9_inter_wave_bench_inputs")
    assert inter_wave.INPUT_MODES == bench.INPUT_MODES
    normal_seed_zero = None

    for input_mode in bench.INPUT_MODES:
        a, b = bench.make_inputs(
            2,
            4,
            8,
            torch.device("cpu"),
            "transposed",
            input_mode=input_mode,
            seed=0,
        )
        repeat_a, repeat_b = bench.make_inputs(
            2,
            4,
            8,
            torch.device("cpu"),
            "transposed",
            input_mode=input_mode,
            seed=0,
        )
        torch.testing.assert_close(a, repeat_a)
        torch.testing.assert_close(b, repeat_b)
        inter_wave_a, inter_wave_b = inter_wave.make_inputs(
            2,
            4,
            8,
            torch.device("cpu"),
            "transposed",
            input_mode=input_mode,
            seed=0,
        )
        torch.testing.assert_close(inter_wave_a, a)
        torch.testing.assert_close(inter_wave_b, b)
        assert b.shape == (8, 4)
        assert b.stride() == (1, 8)
        if input_mode == "normal":
            normal_seed_zero = a

    normal_a, _ = bench.make_inputs(2, 4, 8, "cpu", "transposed", input_mode="normal", seed=1)
    assert not torch.equal(normal_seed_zero, normal_a)


def test_tlx_gfx9_gemm_bench_reproduces_hipblaslt_rand_int_inputs():
    bench = _load_tlx_gfx9_gemm_bench_module("_tlx_amd_test_gfx9_bench_rand_int")

    a, b = bench.make_inputs(
        2,
        4,
        8,
        torch.device("cpu"),
        "transposed",
        input_mode="rand-int",
        seed=0,
    )

    expected_a = torch.tensor(
        [
            [-2, -2, 0, 0, 1, 0, 1, 2],
            [-1, -2, 0, -1, -2, -2, 0, -1],
        ],
        dtype=torch.float16,
    )
    expected_b_storage = torch.tensor(
        [
            [2, -2, 0, 0, -1, 0, -1, 2],
            [-1, 2, 0, 1, -2, 2, 0, 1],
            [2, 0, -2, -1, 1, 2, 0, 2],
            [2, -1, -1, 1, 2, 2, 1, 2],
        ],
        dtype=torch.float16,
    )
    torch.testing.assert_close(a, expected_a)
    torch.testing.assert_close(b.T, expected_b_storage)


def test_tlx_gfx9_gemm_bench_launch_reuses_output():
    bench = _load_tlx_gfx9_gemm_bench_module("_tlx_amd_test_gfx9_bench_output")
    call = {}

    class FakeKernel:

        def __getitem__(self, grid):
            call["grid"] = grid

            def launch(*args, **kwargs):
                call["args"] = args
                call["kwargs"] = kwargs

            return launch

    module = SimpleNamespace(v9_beyond_hotloop=FakeKernel())
    a = torch.empty((256, 128), dtype=torch.float16)
    b = torch.empty((128, 256), dtype=torch.float16)
    out = torch.empty((256, 256), dtype=torch.float16)

    result = bench.launch_tutorial_matmul(module, "v9_beyond_hotloop", a, b, out=out)

    assert result is out
    assert call["args"][2] is out
    assert call["grid"] == (1, )


def test_tlx_gfx9_gemm_bench_batched_timing_uses_one_event_span_per_repeat():
    bench = _load_tlx_gfx9_gemm_bench_module("_tlx_amd_test_gfx9_bench_timing")
    state = {"launches": 0, "synchronizes": 0, "events": 0}

    class FakeEvent:

        def __init__(self):
            self.launch = None

        def record(self):
            self.launch = state["launches"]

        def elapsed_time(self, other):
            return (other.launch - self.launch) * 0.25

    class FakeDeviceInterface:

        def Event(self, *, enable_timing):
            assert enable_timing
            state["events"] += 1
            return FakeEvent()

        def synchronize(self):
            state["synchronizes"] += 1

    def launch():
        state["launches"] += 1

    ms = bench.do_bench_batched(
        launch,
        warmup_launches=2,
        timed_launches=4,
        repeats=3,
        device_interface=FakeDeviceInterface(),
    )

    assert ms == 0.25
    assert state == {"launches": 18, "synchronizes": 6, "events": 6}


def test_tlx_gfx9_gemm_bench_triton_timing_reports_median(monkeypatch):
    bench = _load_tlx_gfx9_gemm_bench_module("_tlx_amd_test_gfx9_bench_median")
    call = {}

    def do_bench(fn, **kwargs):
        call["fn"] = fn
        call["kwargs"] = kwargs
        return 0.75

    monkeypatch.setattr(bench.triton.testing, "do_bench", do_bench)
    fn = lambda: None
    ms = bench.measure_provider(
        SimpleNamespace(timing_mode="triton", warmup=13, rep=29),
        fn,
    )

    assert ms == 0.75
    assert call == {
        "fn": fn,
        "kwargs": {"warmup": 13, "rep": 29, "return_mode": "median"},
    }


def test_tlx_gfx9_gemm_bench_loads_modules_without_import_leaks():
    bench = _load_tlx_gfx9_gemm_bench_module("_tlx_amd_test_gfx9_bench_imports")
    before_path = list(sys.path)

    module = bench.load_matmul_module("v0_naive", "test")

    assert hasattr(module, "matmul")
    assert list(sys.path) == before_path
    assert module.__name__ not in sys.modules


def test_a4w4_shape_stride_layouts_compile_gfx950(device, tmp_path):
    with knobs.runtime.scope():
        knobs.runtime.override_arch = "gfx950"
        _compile_a4w4_shape((256, 256, 1024), tmp_path)
        _compile_a4w4_shape((256, 256, 1536), tmp_path)

    ttgir_files = list(tmp_path.rglob("_a4w4_kernel.ttgir"))
    amdgcn_files = list(tmp_path.rglob("_a4w4_kernel.amdgcn"))
    assert len(ttgir_files) == 1
    assert len(amdgcn_files) == 1
    ttgir = ttgir_files[0].read_text()
    amdgcn = amdgcn_files[0].read_text()
    assert ttgir.count("tt.dot_scaled") == 8
    assert "#tlx.user_layout" not in ttgir
    assert "#tlx.no_verify_layout" not in ttgir
    assert amdgcn.count("v_mfma_scale_f32_16x16x128_f8f6f4") == 512
    # Narrow in the accumulator layout before redistributing for the store.
    # A wide f32 epilogue redistribution adds 32 writes and 32 reads here.
    assert amdgcn.count("ds_write") == 44
    assert amdgcn.count("ds_read") == 176
    assert "buffer_store_dwordx4" in amdgcn


def _compile_a4w4_inter_wave_256tile(m, n, k, preshuffled_scales=False):
    grid_mn = triton.cdiv(m, _A4W4_INTER_WAVE_BLOCK_M) * triton.cdiv(n, _A4W4_INTER_WAVE_BLOCK_N)
    a = MockTensor(torch.uint8, (m, k // 2))
    b = MockTensor(torch.uint8, (n, k // 2))
    c = MockTensor(torch.bfloat16, (m, n))
    a_scales = MockTensor(torch.uint8, (m * k // 32, ) if preshuffled_scales else (m, k // 32))
    b_scales = MockTensor(torch.uint8, (n * k // 32, ) if preshuffled_scales else (n, k // 32))
    kernel = (_a4w4_inter_wave_preshuffled_scales_kernel if preshuffled_scales else _a4w4_inter_wave_256tile_kernel)
    scale_strides = () if preshuffled_scales else (1, m, 1, n)

    with knobs.runtime.scope():
        knobs.runtime.override_arch = "gfx950"
        return kernel.warmup(
            a,
            b,
            c,
            c,
            a_scales,
            b_scales,
            m,
            n,
            k,
            k // _A4W4_INTER_WAVE_BLOCK_K,
            k // 2,
            1,
            k // 2,
            1,
            n,
            1,
            *scale_strides,
            BLOCK_M=_A4W4_INTER_WAVE_BLOCK_M,
            BLOCK_N=_A4W4_INTER_WAVE_BLOCK_N,
            BLOCK_K=_A4W4_INTER_WAVE_BLOCK_K,
            GROUP_SIZE_M=4,
            NUM_XCDS=8,
            GRID_MN=grid_mn,
            SPLIT_K=1,
            grid=(grid_mn, ),
            num_warps=8,
            num_stages=1,
            matrix_instr_nonkdim=16,
            llvm_fn_attrs=_A4W4_8WAVE_LLVM_FN_ATTRS,
        )


def _compile_a4w4_inter_wave_merged_scales(m, n, k):
    grid_mn = triton.cdiv(m, _A4W4_INTER_WAVE_BLOCK_M) * triton.cdiv(n, _A4W4_INTER_WAVE_BLOCK_N)
    a = MockTensor(torch.uint8, (m, k // 2))
    b = MockTensor(torch.uint8, (n, k // 2))
    c = MockTensor(torch.bfloat16, (m, n))
    scales = MockTensor(torch.uint8, ((m + n) * k // 32, ))

    with knobs.runtime.scope():
        knobs.runtime.override_arch = "gfx950"
        return _a4w4_inter_wave_merged_scales_kernel.warmup(
            a,
            b,
            c,
            c,
            scales,
            m,
            n,
            k,
            k // _A4W4_INTER_WAVE_BLOCK_K,
            k // 2,
            1,
            k // 2,
            1,
            n,
            1,
            BLOCK_M=_A4W4_INTER_WAVE_BLOCK_M,
            BLOCK_N=_A4W4_INTER_WAVE_BLOCK_N,
            BLOCK_K=_A4W4_INTER_WAVE_BLOCK_K,
            GROUP_SIZE_M=4,
            NUM_XCDS=8,
            GRID_MN=grid_mn,
            SPLIT_K=1,
            grid=(grid_mn, ),
            num_warps=8,
            num_stages=1,
            matrix_instr_nonkdim=16,
            llvm_fn_attrs=_A4W4_8WAVE_LLVM_FN_ATTRS,
        )


def test_a4w4_inter_wave_256tile_codegen_gfx950(device, fresh_triton_cache):
    """Check the performance-sensitive structure of the compiled 256-tile path."""
    compiled = _compile_a4w4_inter_wave_256tile(768, 768, 1536)

    ttgir = compiled.asm["ttgir"]
    amdgcn = compiled.asm["amdgcn"]

    # All source layout anchors must be resolved. The scale swizzles use the
    # generic linear representation and lower directly into shared memory.
    assert "#tlx.user_layout" not in ttgir
    assert "#tlx.no_verify_layout" not in ttgir
    assert "#ttg.generic_linear" in ttgir
    assert "#ttg.shared_linear" in ttgir
    assert "ttg.memdesc_reinterpret" not in ttgir
    assert "arith.xori" not in ttgir
    assert ttgir.count('triton.warp_pipeline.stage = "mfma"') == 8
    assert ttgir.count('triton.warp_pipeline.stage = "mem"') == 8
    assert ttgir.count("rocdl.sched.barrier 0") == 8
    assert ttgir.count("tt.dot_scaled") == 16
    assert ttgir.count("amdg.buffer_load_to_local") == 28
    assert "contiguity" not in ttgir

    assert len(re.findall(r"^\s*v_mfma_scale_f32_16x16x128_f8f6f4\b", amdgcn, re.MULTILINE)) == 256
    assert len(re.findall(r"^\s*buffer_load_[^\n]*\blds\s*$", amdgcn, re.MULTILINE)) == 44
    assert len(re.findall(r"^\s*ds_read_b64_tr_b8\b", amdgcn, re.MULTILINE)) == 12
    assert len(re.findall(r"^\s*ds_read_b128\b", amdgcn, re.MULTILINE)) == 112
    assert len(re.findall(r"^\s*ds_write_b128\b", amdgcn, re.MULTILINE)) == 16
    assert len(re.findall(r"^\s*ds_read", amdgcn, re.MULTILINE)) == 124
    assert len(re.findall(r"^\s*ds_write", amdgcn, re.MULTILINE)) == 16
    assert "ds_read_u8" not in amdgcn
    assert "ds_bpermute" not in amdgcn
    assert "ds_permute" not in amdgcn
    assert "v_mov_b32_dpp" not in amdgcn
    assert "v_permlane" not in amdgcn
    # These are deliberate static goldens for the grid-9, K=1536 specialization.
    assert len(re.findall(r"^\s*s_barrier\s*$", amdgcn, re.MULTILINE)) == 41
    assert len(re.findall(r"^\s*s_waitcnt\b", amdgcn, re.MULTILINE)) == 52
    assert compiled.metadata.shared == 143232
    assert compiled.metadata.global_scratch_size == 0
    assert tuple(map(tuple, compiled.metadata.llvm_fn_attrs)) == _A4W4_8WAVE_LLVM_FN_ATTRS
    assert '"amdgpu-post-sched-strategy"="nop"' in compiled.asm["llir"]
    assert ".private_segment_fixed_size: 8" in amdgcn
    assert ".sgpr_spill_count: 0" in amdgcn
    assert ".vgpr_spill_count: 1" in amdgcn
    assert ".agpr_count:     0" in amdgcn

    unrelated = compile_for_gfx950(
        _amd_sched_barrier_kernel,
        signature={"x_ptr": "*bf16", "y_ptr": "*bf16", "BLOCK": "constexpr"},
        constexprs={"BLOCK": 64},
    )
    assert tuple(map(tuple, unrelated.metadata.llvm_fn_attrs)) == ()
    assert "amdgpu-post-sched-strategy" not in unrelated.asm["llir"]


def test_a4w4_inter_wave_256tile_single_trip_codegen_gfx950(device, fresh_triton_cache):
    """K=1024 retains the loop pipeline for its single main step."""
    compiled = _compile_a4w4_inter_wave_256tile(768, 768, 1024)
    ttgir = compiled.asm["ttgir"]
    amdgcn = compiled.asm["amdgcn"]

    assert ttgir.count('triton.warp_pipeline.stage = "mfma"') == 8
    assert ttgir.count('triton.warp_pipeline.stage = "mem"') == 8
    assert ttgir.count("scf.execute_region") == 16
    assert ttgir.count("scf.for") == 1
    assert len(re.findall(r"^\s*v_mfma_scale_f32_16x16x128_f8f6f4\b", amdgcn, re.MULTILINE)) == 256
    assert len(re.findall(r"^\s*buffer_load_[^\n]*\blds\s*$", amdgcn, re.MULTILINE)) == 44
    assert len(re.findall(r"^\s*s_barrier\s*$", amdgcn, re.MULTILINE)) == 41
    assert len(re.findall(r"^\s*s_waitcnt\b", amdgcn, re.MULTILINE)) == 52
    assert "s_trap" not in amdgcn
    assert compiled.metadata.shared == 143232
    assert compiled.metadata.global_scratch_size == 0
    assert ".private_segment_fixed_size: 8" in amdgcn
    assert ".sgpr_spill_count: 0" in amdgcn
    assert ".vgpr_spill_count: 1" in amdgcn


def test_a4w4_inter_wave_preshuffled_scale_codegen_gfx950(device, fresh_triton_cache):
    """The fastest prepacked ABI coalesces both A halves into one b128 read."""
    compiled = _compile_a4w4_inter_wave_256tile(768, 768, 1536, preshuffled_scales=True)
    ttgir = compiled.asm["ttgir"]
    amdgcn = compiled.asm["amdgcn"]

    assert "#tlx.user_layout" not in ttgir
    assert "#tlx.no_verify_layout" not in ttgir
    assert ttgir.count("amdg.buffer_load_to_local") == 24
    assert "buffer_load_dwordx2" not in amdgcn
    assert len(re.findall(r"^\s*buffer_load_[^\n]*\blds\s*$", amdgcn, re.MULTILINE)) == 40
    assert "ds_read_b64_tr_b8" not in amdgcn
    assert len(re.findall(r"^\s*ds_read_b64\b", amdgcn, re.MULTILINE)) == 4
    assert len(re.findall(r"^\s*ds_read_b128\b", amdgcn, re.MULTILINE)) == 116
    assert len(re.findall(r"^\s*ds_read", amdgcn, re.MULTILINE)) == 120
    assert compiled.metadata.shared == 143232
    assert compiled.metadata.global_scratch_size == 0
    assert ".private_segment_fixed_size: 0" in amdgcn
    assert ".vgpr_spill_count: 0" in amdgcn


def test_a4w4_inter_wave_merged_scale_codegen_gfx950(device, fresh_triton_cache):
    """The merged ABI combines wide scale DMA with conflict-free A b128 reads."""
    compiled = _compile_a4w4_inter_wave_merged_scales(768, 768, 1536)
    ttgir = compiled.asm["ttgir"]
    amdgcn = compiled.asm["amdgcn"]

    assert "#tlx.user_layout" not in ttgir
    assert "#tlx.no_verify_layout" not in ttgir
    assert ttgir.count("amdg.buffer_load_to_local") == 18
    assert "ttg.memdesc_reinterpret" in ttgir
    assert len(re.findall(r"^\s*buffer_load_[^\n]*\blds\s*$", amdgcn, re.MULTILINE)) == 34
    assert len(re.findall(r"^\s*buffer_load_dwordx4[^\n]*\blds\s*$", amdgcn, re.MULTILINE)) == 34
    assert len(re.findall(r"^\s*ds_read_b64_tr_b8\b", amdgcn, re.MULTILINE)) == 4
    assert "ds_read_b64 " not in amdgcn
    assert len(re.findall(r"^\s*ds_read_b128\b", amdgcn, re.MULTILINE)) == 116
    assert len(re.findall(r"^\s*ds_read", amdgcn, re.MULTILINE)) == 120
    assert "v_perm_b32" not in amdgcn
    # Refilling immediately after the second-half reads starts the next pair one
    # stage earlier. This deliberately pays a RAW-to-refill wait/barrier; moving
    # the copy across the next existing barrier shortens DMA latency hiding and
    # regresses both measured benchmark shapes.
    assert len(re.findall(r"^\s*s_waitcnt\b", amdgcn, re.MULTILINE)) == 66
    assert len(re.findall(r"^\s*s_barrier\s*$", amdgcn, re.MULTILINE)) == 42
    assert compiled.metadata.shared == 143232
    assert compiled.metadata.global_scratch_size == 0
    assert ".private_segment_fixed_size: 0" in amdgcn
    assert ".sgpr_count:     58" in amdgcn
    assert ".sgpr_spill_count: 0" in amdgcn
    assert ".vgpr_spill_count: 0" in amdgcn


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_a4w4_shape_stride_layouts_correctness_gfx950(device):
    m = n = 256
    for k in (1024, 1536):
        a, b, a_scales, b_scales = _generate_a4w4_inputs(m, n, k)
        actual = _launch_a4w4(a, b, a_scales, b_scales)
        expected = _a4w4_reference(a, b, a_scales, b_scales)
        torch.testing.assert_close(actual, expected, atol=0.1, rtol=0.0)


@pytest.mark.parametrize("k, split_k", [(1024, 1), (4096, 2)])
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_a4w4_inter_wave_preshuffled_scale_correctness_gfx950(device, k, split_k):
    m = n = 256
    a, b, a_scales, b_scales = _generate_a4w4_inputs(m, n, k)
    a_scales_preshuffled = _preshuffle_a4w4_a_scales(a_scales)
    b_scales_preshuffled = _preshuffle_a4w4_b_scales(b_scales)
    actual = _a4w4_inter_wave_matmul_preshuffled(a, b, a_scales_preshuffled, b_scales_preshuffled, SPLIT_K=split_k)
    expected = _a4w4_reference(a, b, a_scales, b_scales)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("k, split_k", [(1024, 1), (4096, 2)])
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_a4w4_inter_wave_merged_scale_correctness_gfx950(device, k, split_k):
    m = n = 256
    a, b, a_scales, b_scales = _generate_a4w4_inputs(m, n, k)
    scales = _preshuffle_a4w4_scales(a_scales, b_scales)
    actual = _a4w4_inter_wave_matmul_merged_scales(a, b, scales, SPLIT_K=split_k)
    expected = _a4w4_reference(a, b, a_scales, b_scales)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "k, expected_path",
    [(1536, "intra_wave_256x256"), (2048, "inter_wave_256x256")],
)
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_a4w4_inter_wave_large_grid_dispatch_correctness_gfx950(device, k, expected_path):
    # A 2x33 grid exceeds the skinny threshold. K=1536 selects the measured
    # lower-overhead 4-wave path; K=2048 selects the 8-wave pipeline.
    m, n = 512, 8448
    assert _select_a4w4_inter_wave_path(m, n, k) == expected_path
    a, b, a_scales, b_scales = _generate_a4w4_inputs(m, n, k)
    actual = _a4w4_inter_wave_matmul(a, b, a_scales, b_scales)
    expected = _a4w4_reference(a, b, a_scales, b_scales)
    torch.testing.assert_close(actual, expected, atol=0.1, rtol=0.0)


@pytest.mark.parametrize("m, n", [(256, 16640), (512, 8448)])
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_a4w4_short_k_dispatch_stress_gfx950(device, m, n):
    # Both public shapes exceed the skinny threshold by the smallest possible
    # grid margins. K=1024 must dispatch to the measured 4-wave path.
    k = _A4W4_INTER_WAVE_MIN_K
    grid_mn = triton.cdiv(m, _A4W4_INTER_WAVE_BLOCK_M) * triton.cdiv(n, _A4W4_INTER_WAVE_BLOCK_N)
    assert grid_mn in (65, 66)
    assert _select_a4w4_inter_wave_path(m, n, k) == "intra_wave_256x256"
    assert k == 1024

    a, b, a_scales, b_scales = _generate_a4w4_inputs(m, n, k)
    expected = _a4w4_reference(a, b, a_scales, b_scales)
    for launch in range(500):
        actual = _a4w4_inter_wave_matmul(a, b, a_scales, b_scales)
        torch.testing.assert_close(actual, expected, atol=0.1, rtol=0.0, msg=f"failed on launch {launch}")


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_a4w4_inter_wave_skinny_correctness_gfx950(device):
    # 512x256x1536 -> 256-tile grid = 2*1 = 2 <= NUM_CU/32, so the dispatcher takes
    # the occupancy-starved 128x128 + split-K TLX path (and its fp32 reduce).
    m = 512
    n = 256
    k = 1536
    a, b, a_scales, b_scales = _generate_a4w4_inputs(m, n, k)
    actual = _a4w4_inter_wave_matmul(a, b, a_scales, b_scales)
    expected = _a4w4_reference(a, b, a_scales, b_scales)
    torch.testing.assert_close(actual, expected, atol=0.1, rtol=0.0)


# ---------------------------------------------------------------------------
# Test: tlx.assume_uniform marks a scalar wave-uniform for the AMD backend.
# ---------------------------------------------------------------------------


@triton.jit
def _assume_uniform_ptr_kernel(ptr_array, out_ptr, BLOCK: tl.constexpr):
    # A pointer loaded from memory is not provably uniform, so the backend would
    # otherwise waterfall every buffer access built on it.
    base = tl.load(ptr_array).to(tl.pointer_type(tl.float32))
    base = tlx.assume_uniform(base)
    offs = tl.arange(0, BLOCK).to(tl.int32)
    tlx.buffer_store(tlx.buffer_load(base, offs), out_ptr, offs)


@triton.jit
def _assume_uniform_ptr_kernel_no_hint(ptr_array, out_ptr, BLOCK: tl.constexpr):
    base = tl.load(ptr_array).to(tl.pointer_type(tl.float32))
    offs = tl.arange(0, BLOCK).to(tl.int32)
    tlx.buffer_store(tlx.buffer_load(base, offs), out_ptr, offs)


@triton.jit
def _assume_uniform_scalar_kernel(in_ptr, out_ptr, BLOCK: tl.constexpr):
    v = tlx.assume_uniform(tl.load(in_ptr))
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.zeros((BLOCK, ), tl.float32) + v.to(tl.float32))


@pytest.mark.skipif(not is_hip(), reason="Requires HIP runtime")
def test_assume_uniform_compiles_gfx950(device):
    """assume_uniform on a loaded pointer produces amdg.assume_uniform in TTIR/TTGIR."""
    compiled = compile_for_gfx950(
        _assume_uniform_ptr_kernel,
        signature={"ptr_array": "*i64", "out_ptr": "*fp32"},
        constexprs={"BLOCK": 64},
    )
    assert "amdg.assume_uniform" in compiled.asm["ttir"]
    assert "amdg.assume_uniform" in compiled.asm["ttgir"]
    assert "amdgcn" in compiled.asm
    assert len(compiled.asm["amdgcn"]) > 0


@pytest.mark.skipif(not is_hip(), reason="Requires HIP runtime")
def test_assume_uniform_emits_readfirstlane_gfx950(device):
    """assume_uniform lowers to readfirstlane beyond what the backend emits anyway.

    The buffer resource descriptor already forces one readfirstlane, so the count
    is compared against the same kernel without the hint rather than asserted
    absolutely.
    """
    signature = {"ptr_array": "*i64", "out_ptr": "*fp32"}
    with_hint = compile_for_gfx950(_assume_uniform_ptr_kernel, signature, {"BLOCK": 64})
    without_hint = compile_for_gfx950(_assume_uniform_ptr_kernel_no_hint, signature, {"BLOCK": 64})
    assert with_hint.asm["llir"].count("readfirstlane") > without_hint.asm["llir"].count("readfirstlane")


@pytest.mark.skipif(not is_hip(), reason="Requires HIP runtime")
@pytest.mark.parametrize("dtype", ["i16", "i32", "i64", "fp16", "bf16", "fp32"])
def test_assume_uniform_scalar_types_compiles_gfx950(device, dtype):
    """assume_uniform accepts every 16/32/64-bit scalar type."""
    compiled = compile_for_gfx950(
        _assume_uniform_scalar_kernel,
        signature={"in_ptr": f"*{dtype}", "out_ptr": "*fp32"},
        constexprs={"BLOCK": 64},
    )
    assert "amdg.assume_uniform" in compiled.asm["ttgir"]
    assert len(compiled.asm["amdgcn"]) > 0


@pytest.mark.skipif(not is_hip(), reason="Requires HIP runtime")
def test_assume_uniform_rejects_narrow_type_gfx950(device):
    """readfirstlane has no sub-16-bit form, so narrower scalars are rejected."""
    with pytest.raises(CompilationError, match="16/32/64-bit"):
        compile_for_gfx950(
            _assume_uniform_scalar_kernel,
            signature={"in_ptr": "*i8", "out_ptr": "*fp32"},
            constexprs={"BLOCK": 64},
        )


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_assume_uniform_correctness_gfx950(device):
    """assume_uniform returns its argument unchanged."""
    size = 64
    x = torch.rand(size, dtype=torch.float32, device=device)
    out = torch.empty_like(x)
    ptr_array = torch.tensor([x.data_ptr()], dtype=torch.int64, device=device)
    _assume_uniform_ptr_kernel[(1, )](ptr_array, out, BLOCK=size)
    torch.testing.assert_close(out, x)


@triton.jit
def _amd_sched_barrier_kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    values = tl.load(x_ptr + offsets)
    tlx.amd_sched_barrier()
    tl.store(y_ptr + offsets, values)


def test_amd_sched_barrier_compiles_gfx950():
    compiled = compile_for_gfx950(
        _amd_sched_barrier_kernel,
        signature={"x_ptr": "*bf16", "y_ptr": "*bf16", "BLOCK": "constexpr"},
        constexprs={"BLOCK": 64},
    )
    assert "llvm.amdgcn.sched.barrier" in compiled.asm["llir"]
