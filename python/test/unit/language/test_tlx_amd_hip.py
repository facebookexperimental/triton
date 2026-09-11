"""TLX AMD tests -- any AMD arch (HIP runtime).

These go through JITFunction.warmup, which validates kwargs against the active
backend, so AMD-only options (waves_per_eu, matrix_instr_nonkdim) need HIP to be
the active driver even though the compile targets an explicit gfx arch. The
compile-only siblings that build an ASTSource directly need no driver and live
in test_tlx_codegen.py.
"""
import re
import pytest
import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton import knobs
from triton._internal_testing import is_hip
from triton.compiler.compiler import ASTSource, compile as triton_compile
from triton.backends.compiler import GPUTarget
from triton.runtime.jit import MockTensor
from triton.language.extra.tlx.tutorials import amd_fa_cluster as _amd_fa_cluster_module
from triton.language.extra.tlx.tutorials.amd_bmm_shared_a import (
    _bmm_register_staged,
    _MT144X256_MI16_KERNEL_SPEC,
    _MT64X256_MI32_KERNEL_SPEC,
    _MT224X160_MI16_KERNEL_SPEC,
)
from triton.language.extra.tlx.tutorials.gfx9_gemm.intra_wave.a4w4.bench import (
    compile_shape as _compile_a4w4_shape, )
from triton.language.extra.tlx.tutorials.gfx9_gemm.intra_wave.a4w4.matmul_kernel import (
    _a4w4_kernel as _a4w4_intra_wave_kernel, )
from triton.language.extra.tlx.tutorials.gfx9_gemm.inter_wave.a4w4.matmul_kernel import (
    BLOCK_K as _A4W4_INTER_WAVE_BLOCK_K,
    BLOCK_M as _A4W4_INTER_WAVE_BLOCK_M,
    BLOCK_N as _A4W4_INTER_WAVE_BLOCK_N,
    _a4w4_8wave_kernel as _a4w4_inter_wave_256tile_kernel,
    _a4w4_8wave_merged_scales_kernel as _a4w4_inter_wave_merged_scales_kernel,
    _a4w4_8wave_preshuffled_scales_kernel as _a4w4_inter_wave_preshuffled_scales_kernel,
    _A4W4_8WAVE_LLVM_FN_ATTRS,
)

pytestmark = pytest.mark.skipif(not is_hip(), reason="Requires HIP runtime")

GFX950 = GPUTarget("hip", "gfx950", 64)


def compile_for_target(fn, signature, constexprs, target):
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs)
    return triton_compile(src, target=target)


def compile_for_gfx950(fn, signature, constexprs):
    """Compile a TLX kernel for gfx950 and return the compiled object."""
    return compile_for_target(fn, signature, constexprs, GFX950)


def _compile_register_staged_bmm_gfx950(m, n, k, kernel_spec):
    batch = 1
    block_m, block_n = kernel_spec.block_m, kernel_spec.block_n
    instr_shape = kernel_spec.instr_shape
    warps_per_cta = kernel_spec.warps_per_cta
    n_blocks = triton.cdiv(n, block_n)
    tiles_per_batch = triton.cdiv(m, block_m) * n_blocks
    a = MockTensor(torch.float16, (batch, m, k))
    b = MockTensor(torch.float16, (batch, k, n))
    c = MockTensor(torch.float16, (batch, m, n))
    strides = (0, k, 1, k * n, n, 1, m * n, n, 1)

    with knobs.runtime.scope():
        knobs.runtime.override_arch = "gfx950"
        return _bmm_register_staged.warmup(
            a,
            b,
            c,
            m,
            n,
            k,
            *strides,
            KERNEL_SPEC=kernel_spec,
            EVEN_M=m % block_m == 0,
            EVEN_N=n % block_n == 0,
            HAS_K_TAIL=k % 32 != 0,
            N_BLOCKS=n_blocks,
            BATCH_GROUP=8,
            GMN=tiles_per_batch,
            NT=tiles_per_batch,
            grid=(tiles_per_batch,),
            num_warps=warps_per_cta[0] * warps_per_cta[1],
            num_stages=1,
            matrix_instr_nonkdim=instr_shape[0],
        )


def test_shared_a_bmm_tuned_dispatches_keep_wide_unaligned_loads_gfx950():
    compiled = [
        _compile_register_staged_bmm_gfx950(
            40, 256, 1956, _MT64X256_MI32_KERNEL_SPEC
        ),
        _compile_register_staged_bmm_gfx950(
            262, 256, 294, _MT144X256_MI16_KERNEL_SPEC
        ),
        _compile_register_staged_bmm_gfx950(
            448, 160, 931, _MT224X160_MI16_KERNEL_SPEC
        ),
    ]
    amdgcn = "\n".join(kernel.asm["amdgcn"] for kernel in compiled)

    assert "buffer_load_dwordx2" in amdgcn
    assert "buffer_load_dwordx4" in amdgcn


@pytest.fixture(scope="module")
def amd_fa_cluster_long_bf16_codegen_gfx950():
    """Compile the long BF16 object shared by its focused codegen checks."""
    batch, heads, n_ctx, head_dim = 1, 64, 16384, 128
    tensor = MockTensor(torch.bfloat16, (batch, heads, n_ctx, head_dim))
    q_strides = (heads * n_ctx * 257, n_ctx * 257, 257, 1)
    k_strides = (heads * n_ctx * 263, n_ctx * 263, 263, 1)
    v_strides = (heads * n_ctx * 269, n_ctx * 269, 269, 1)
    o_strides = q_strides

    with knobs.runtime.scope():
        knobs.runtime.override_arch = "gfx950"
        compiled = _amd_fa_cluster_module._attn_fwd_cluster_pipeline.warmup(
            tensor,
            tensor,
            tensor,
            tensor,
            *q_strides,
            *k_strides,
            *v_strides,
            *o_strides,
            batch,
            H=heads,
            N_CTX=n_ctx,
            sm_scale=1.0 / head_dim**0.5,
            BLOCK_M=256,
            BLOCK_N=64,
            BUF_DEPTH=2,
            HEAD_DIM=head_dim,
            USE_DIRECT_LOAD=False,
            IS_CAUSAL=False,
            STATIC_STRIDE_KN=k_strides[2],
            grid=(n_ctx // 256, heads, batch),
            num_warps=8,
            num_stages=3,
            waves_per_eu=2,
            enable_sched_group_barrier_scheduler=False,
            llvm_fn_attrs=_amd_fa_cluster_module._CLUSTER_VGPR_ONLY_LLVM_FN_ATTRS,
        )
    return compiled


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


@triton.jit
def _amd_sched_barrier_kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    values = tl.load(x_ptr + offsets)
    tlx.amd_sched_barrier()
    tl.store(y_ptr + offsets, values)


def test_amd_fa_cluster_one_tile_prefix_handoff_codegen_gfx950():
    """The heavy N512 class hands prefetched diagonal slots to the short tail."""
    batch, heads, n_ctx, head_dim = 1, 64, 512, 128
    tensor = MockTensor(torch.float16, (batch, heads, n_ctx, head_dim))
    strides = (heads * n_ctx * head_dim, n_ctx * head_dim, head_dim, 1)

    with knobs.runtime.scope():
        knobs.runtime.override_arch = "gfx950"
        compiled = _amd_fa_cluster_module._attn_fwd_cluster_short_causal_pipeline.warmup(
            tensor,
            tensor,
            tensor,
            tensor,
            *strides,
            *strides,
            *strides,
            *strides,
            batch,
            H=heads,
            N_CTX=n_ctx,
            sm_scale=1.0 / head_dim**0.5,
            BLOCK_M=128,
            BLOCK_N=64,
            BUF_DEPTH=2,
            HEAD_DIM=head_dim,
            USE_DIRECT_LOAD=False,
            IS_CAUSAL=True,
            SPECIALIZE_QUERY_CLASSES=True,
            grid=(heads, n_ctx // 128, batch),
            num_warps=4,
            num_stages=3,
            waves_per_eu=0,
            enable_sched_group_barrier_scheduler=False,
            llvm_fn_attrs=_amd_fa_cluster_module._CLUSTER_SHORT_N512_LLVM_FN_ATTRS,
        )

    ttir = compiled.asm["ttir"]
    # The dedicated dense short-class route specializes physical strides, so
    # no stride remains a runtime scalar anywhere in the generated module.
    assert not any(
        f"%stride_{suffix}" in ttir
        for suffix in ("qz", "qh", "qm", "qk", "kz", "kh", "kn", "kk", "vz", "vh", "vn", "vk", "oz", "oh", "om", "ok"))
    # The four class-specialized bodies retain five full CTA rendezvous.  The
    # two pipelined prefixes use result-dependent inline-asm barriers instead;
    # an extra full barrier means the diagonal handoff was broken.
    assert ttir.count("ttg.barrier all") == 5
    amdgcn = compiled.asm["amdgcn"]
    assert ".private_segment_fixed_size: 0" in amdgcn
    assert ".vgpr_spill_count: 0" in amdgcn


def test_amd_fa_cluster_persistent_reuse_waits_for_lds_consumers_gfx950():
    """Persistent tiles rendezvous before their LDS slots are reused."""
    batch, heads, n_ctx, head_dim = 2, 9, 1024, 128
    tensor = MockTensor(torch.bfloat16, (batch, heads, n_ctx, head_dim))
    strides = (heads * n_ctx * head_dim, n_ctx * head_dim, head_dim, 1)

    with knobs.runtime.scope():
        knobs.runtime.override_arch = "gfx950"
        compiled = _amd_fa_cluster_module._attn_fwd_cluster_persistent_pipeline.warmup(
            tensor,
            tensor,
            tensor,
            tensor,
            *strides,
            *strides,
            *strides,
            *strides,
            batch,
            H=heads,
            N_CTX=n_ctx,
            sm_scale=1.0 / head_dim**0.5,
            BLOCK_M=256,
            BLOCK_N=64,
            BUF_DEPTH=2,
            HEAD_DIM=head_dim,
            USE_DIRECT_LOAD=False,
            IS_CAUSAL=True,
            NUM_M_BLOCKS=n_ctx // 256,
            NUM_SMS=16,
            NUM_XCDS=4,
            grid=(16, ),
            num_warps=8,
            num_stages=3,
            waves_per_eu=2,
            enable_sched_group_barrier_scheduler=False,
        )

    ttgir = compiled.asm["ttgir"]
    output_stores = re.findall(r"amdg\.buffer_store[^\n]*\n\s+ttg\.barrier all", ttgir)
    # A causal persistent work unit statically contains two tile bodies.
    assert len(output_stores) == 2


def test_amd_fa_cluster_n1024_fp16_qk_handoff_uses_result_barrier_gfx950():
    """The long BM128 prefix anchors both QK groups before reusing K0."""
    batch, heads, n_ctx, head_dim = 1, 64, 1024, 128
    tensor = MockTensor(torch.float16, (batch, heads, n_ctx, head_dim))
    strides = (heads * n_ctx * head_dim, n_ctx * head_dim, head_dim, 1)

    with knobs.runtime.scope():
        knobs.runtime.override_arch = "gfx950"
        compiled = _amd_fa_cluster_module._attn_fwd_cluster_short_causal_pipeline.warmup(
            tensor,
            tensor,
            tensor,
            tensor,
            *strides,
            *strides,
            *strides,
            *strides,
            batch,
            H=heads,
            N_CTX=n_ctx,
            sm_scale=1.0 / head_dim**0.5,
            BLOCK_M=128,
            BLOCK_N=64,
            BUF_DEPTH=2,
            HEAD_DIM=head_dim,
            USE_DIRECT_LOAD=False,
            IS_CAUSAL=True,
            SPECIALIZE_QUERY_CLASSES=False,
            grid=(heads, n_ctx // 128, batch),
            num_warps=4,
            num_stages=3,
            waves_per_eu=0,
            enable_sched_group_barrier_scheduler=False,
            llvm_fn_attrs=_amd_fa_cluster_module._CLUSTER_SHORT_N1024_LLVM_FN_ATTRS,
        )

    qk_constraints = 'constraints = "=s,=s,=s,=s,v,v,v,v,v,v,v,v"'
    assert compiled.asm["ttir"].count(qk_constraints) == 1


def test_amd_fa_cluster_static_k_row_stride_codegen_gfx950(amd_fa_cluster_long_bf16_codegen_gfx950):
    """A selected K-row override removes the dynamic stride from pointer arithmetic."""
    compiled = amd_fa_cluster_long_bf16_codegen_gfx950

    ttir = compiled.asm["ttir"]
    # Runtime kernel parameters remain in the public ABI, but a specialized
    # stride has no use beyond that declaration. The unselected Q stride still
    # participates in its assumption and address-vector construction.
    kernel_body = "\n".join(ttir.split("tt.func public @_attn_fwd_cluster_pipeline", 1)[1].splitlines()[1:])
    assert "%stride_kn" not in kernel_body
    assert "%stride_qm" in kernel_body


def test_amd_fa_cluster_row_reduction_preserves_mfma_slices_gfx950(amd_fa_cluster_long_bf16_codegen_gfx950):
    """The four row-reduction slices retain their parent MFMA layout."""
    compiled = amd_fa_cluster_long_bf16_codegen_gfx950
    ttgir = compiled.asm["ttgir"]
    reduction_slices = re.findall(
        r"amdg\.extract_slice .*tensor<256x64xf32, #mma> to tensor<256x16xf32, #mma>",
        ttgir,
    )
    assert len(reduction_slices) >= 4


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_amd_fa_cluster_n2048_four_slot_prefix_handoff_codegen_gfx950(dtype):
    """The N2048 prefix supplies all four diagonal slots without reloading them."""
    batch, heads, n_ctx, head_dim = 1, 64, 2048, 128
    tensor = MockTensor(dtype, (batch, heads, n_ctx, head_dim))
    strides = (heads * n_ctx * head_dim, n_ctx * head_dim, head_dim, 1)

    with knobs.runtime.scope():
        knobs.runtime.override_arch = "gfx950"
        compiled = _amd_fa_cluster_module._attn_fwd_cluster_pipeline.warmup(
            tensor,
            tensor,
            tensor,
            tensor,
            *strides,
            *strides,
            *strides,
            *strides,
            batch,
            H=heads,
            N_CTX=n_ctx,
            sm_scale=1.0 / head_dim**0.5,
            BLOCK_M=256,
            BLOCK_N=64,
            BUF_DEPTH=2,
            HEAD_DIM=head_dim,
            USE_DIRECT_LOAD=False,
            IS_CAUSAL=True,
            grid=(heads, n_ctx // 256, batch),
            num_warps=8,
            num_stages=3,
            waves_per_eu=2,
            enable_sched_group_barrier_scheduler=False,
            llvm_fn_attrs=_amd_fa_cluster_module._CLUSTER_VGPR_ONLY_LLVM_FN_ATTRS,
        )

    ttir = compiled.asm["ttir"]
    # One object serves both query tiles with a prefix and the early tiles
    # without one.  It therefore retains eight fallback diagonal copies in
    # addition to five prologue, four paired-loop, two odd-tail, and five
    # prefix-handoff copies.  The old three-tile drain has 20 sites instead.
    assert ttir.count("ttg.async_copy_global_to_local") == 24
    # The diagonal decision is all-or-none within each wave.  Preserve the
    # wave vote + uniform-if lowering instead of reintroducing EXEC masking.
    assert ttir.count("ttg.warp_vote") == 2
    assert ttir.count("ttg.warp_predicate") == 3
    amdgcn = compiled.asm["amdgcn"]
    assert ".private_segment_fixed_size: 0" in amdgcn
    assert ".vgpr_spill_count: 0" in amdgcn


def test_a4w4_shape_stride_layouts_compile_gfx950(device, tmp_path):
    # Both shapes must reach the compiler, so drop any in-process binary another
    # test left behind. Clear once: the single artifact below is the assertion
    # that K is a runtime argument, so the two shapes share one cache key.
    _a4w4_intra_wave_kernel.device_caches.clear()
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
    assert ttgir.count("rocdl.sched.barrier none") == 8
    assert ttgir.count("tt.dot_scaled") == 16
    assert ttgir.count("amdg.buffer_load_to_local") == 28
    assert ttgir.count("contiguity =") == 28

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
