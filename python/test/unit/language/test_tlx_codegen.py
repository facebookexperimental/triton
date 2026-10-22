import re
from triton.compiler.errors import CompilationError

import torch
import math
from triton.language.extra.tlx.tutorials import amd_fa_bwd, amd_fa_varlen_bwd
from triton.language.extra.tlx.tutorials.amd_fa_bwd import (
    _D64DQLaunch,
    _D64Dispatch,
    _D64_GQA_SIGNED,
    _D64_MHA_POSITIVE,
    _allocate_bwd_d64_causal_gqa8_workspaces,
    _allocate_bwd_d64_fused_workspaces,
    _d64_causal_dkdv_first_query_block,
    _d64_causal_dq_key_blocks,
    _d64_causal_stat_values,
    _d64_dq_launch_plan,
    _run_bwd_d64_direct,
    _select_d64_dispatch,
    _validate_d64_sm_scale,
)

import pytest
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource, compile as triton_compile

GFX950 = GPUTarget("hip", "gfx950", 64)
GFX942 = GPUTarget("hip", "gfx942", 64)
GFX1250 = GPUTarget("hip", "gfx1250", 32)


def make_ir_for_target(fn, signature, constexprs, target):
    backend = triton.compiler.compiler.make_backend(target)
    options = backend.parse_options({})
    context = ir.context()
    backend.load_dialects(context)
    codegen_fns = backend.get_codegen_implementation(options)
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs)
    return src.make_ir(target, options, codegen_fns, {}, context)


def compile_for_target(fn, signature, constexprs, target):
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs)
    return triton_compile(src, target=target)


def compile_for_gfx950(fn, signature, constexprs):
    """Compile a TLX kernel for gfx950 and return the compiled object."""
    return compile_for_target(fn, signature, constexprs, GFX950)


@triton.jit
def _concrete_dot_loop_helper(lhs, rhs, acc):
    for _ in tl.range(0, 2, num_stages=1):
        acc = tl.dot(lhs, rhs, acc=acc, out_dtype=tl.float32)
    return acc


@triton.jit
def _concrete_dot_loop_helper_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    out_ptr,
    CAST_BEFORE_RELEASE: tl.constexpr,
    RELAXED_RELEASE: tl.constexpr,
):
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    consumer_layout: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    dot0: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=8)
    dot1: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=8)
    m = tl.arange(0, 64)
    n = tl.arange(0, 64)
    k = tl.arange(0, 32)
    lhs = tl.load(a_ptr + m[:, None] * 32 + k[None, :])
    rhs = tl.load(b_ptr + k[:, None] * 64 + n[None, :])
    lhs = tlx.require_layout(lhs, dot0, pin=False)
    rhs = tlx.require_layout(rhs, dot1, pin=False)
    acc = tlx.zeros([64, 64], dtype=tl.float32, layout=mma)
    result = _concrete_dot_loop_helper(lhs, rhs, acc)
    offsets = m[:, None] * 64 + n[None, :]
    bias = tlx.require_layout(tl.load(bias_ptr + offsets), mma, pin=False)
    result += bias
    if CAST_BEFORE_RELEASE:
        result = result.to(tl.bfloat16)
    result = tlx.release_layout(result, relaxed=RELAXED_RELEASE)
    result = tlx.require_layout(result, consumer_layout, pin=False)
    output = tlx.require_layout(out_ptr + offsets, consumer_layout, pin=False)
    tl.store(output, result)


def test_concrete_dot_loop_helper_result_layout_compiles_gfx950():
    compiled = compile_for_gfx950(
        _concrete_dot_loop_helper_kernel,
        signature={
            "a_ptr": "*fp16",
            "b_ptr": "*fp16",
            "bias_ptr": "*fp32",
            "out_ptr": "*fp32",
        },
        constexprs={"CAST_BEFORE_RELEASE": False, "RELAXED_RELEASE": False},
    )
    assert "amdgcn" in compiled.asm
    assert "scf.for" in compiled.asm["ttir"]
    assert "#ttg.amd_mfma" in compiled.asm["ttgir"]
    assert "#tlx.no_verify_layout" not in compiled.asm["ttgir"]


@pytest.mark.parametrize("relaxed", [False, True])
def test_release_layout_accepts_cast_helper_result_gfx950(relaxed):
    module = make_ir_for_target(
        _concrete_dot_loop_helper_kernel,
        signature={
            "a_ptr": "*fp16",
            "b_ptr": "*fp16",
            "bias_ptr": "*fp32",
            "out_ptr": "*bf16",
        },
        constexprs={"CAST_BEFORE_RELEASE": True, "RELAXED_RELEASE": relaxed},
        target=GFX950,
    )
    ttir = str(module)
    assert "tt.call" in ttir
    assert "arith.addf" in ttir
    assert "arith.truncf" in ttir
    release_line = next(line for line in ttir.splitlines() if "tlx.release_layout" in line)
    assert ("relaxed = true" in release_line) == relaxed
    assert ttir.index("arith.truncf") < ttir.index("tlx.release_layout")


@triton.jit
def _unencoded_release_layout_kernel(x_ptr, y_ptr):
    offsets = tl.arange(0, 64)
    values = tl.load(x_ptr + offsets)
    values = values.to(tl.float16).to(tl.float32)
    values = tlx.release_layout(values)
    tl.store(y_ptr + offsets, values)


def test_release_layout_accepts_unencoded_source_gfx950():
    compiled = compile_for_gfx950(
        _unencoded_release_layout_kernel,
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32"},
        constexprs={},
    )
    assert "tlx.release_layout" not in compiled.asm["ttir"]
    assert "amdgcn" in compiled.asm


SM80 = GPUTarget("cuda", 80, 32)
SM90 = GPUTarget("cuda", 90, 32)
SM100 = GPUTarget("cuda", 100, 32)
GFX942 = GPUTarget("hip", "gfx942", 64)
GFX1250 = GPUTarget("hip", "gfx1250", 32)


@triton.jit
def _async_local_slice_dot_kernel(q_ptr, k_ptr, output_ptr):
    rows = tl.arange(0, 128)
    cols = tl.arange(0, 32)
    reduction = tl.arange(0, 128)
    q = tl.load(q_ptr + rows[:, None] * 128 + reduction[None, :])

    k_rows = tl.arange(0, 64)
    k_ptrs = k_ptr + k_rows[:, None] * 128 + reduction[None, :]
    k_buffers = tlx.local_alloc((64, 128), tl.float16, 1)
    k_view = tlx.local_view(k_buffers, 0)
    token = tlx.async_load(k_ptrs, k_view)
    tlx.async_load_commit_group([token])
    wait = tlx.async_load_wait_group(0)

    k_lo = tlx.local_slice(k_view, [0, 0], [32, 128])
    kt = tlx.local_load(tlx.local_trans(k_lo), token=wait, relaxed=True)
    result = tl.dot(q, kt)
    tl.store(output_ptr + rows[:, None] * 32 + cols[None, :], result)


@pytest.mark.parametrize(
    "target,expected_warp_size",
    [
        (SM80, 32),
        (SM90, 32),
        (SM100, 32),
        (GFX942, 64),
        (GFX950, 64),
        (GFX1250, 32),
    ],
    ids=["sm80", "sm90", "sm100", "gfx942", "gfx950", "gfx1250"],
)
def test_raw_ttir_uses_target_warp_size(target, expected_warp_size):
    backend = triton.compiler.compiler.make_backend(target)
    options = backend.parse_options({})
    assert options.warp_size == expected_warp_size
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    source = ASTSource(
        fn=_async_local_slice_dot_kernel,
        signature={
            "q_ptr": "*fp16",
            "k_ptr": "*fp16",
            "output_ptr": "*fp32",
        },
        constexprs={},
    )

    module = source.make_ir(
        target,
        options,
        backend.get_codegen_implementation(options),
        backend.get_module_map(),
        context,
    )

    assert module.get_int_attr("ttg.threads-per-warp") == expected_warp_size


def test_d64_causal_stat_conventions_are_equivalent():
    generator = torch.Generator(device="cpu")
    generator.manual_seed(101)
    q = torch.randn((3, 4), generator=generator)
    k = torch.randn((5, 4), generator=generator)
    v = torch.randn((5, 4), generator=generator)
    o = torch.randn((3, 4), generator=generator)
    do = torch.randn((3, 4), generator=generator)
    lse = torch.randn((3, ), generator=generator)
    sm_scale = 0.5

    delta_mha, lse_mha = _d64_causal_stat_values(o, do, lse, sm_scale, _D64_MHA_POSITIVE)
    delta_gqa, lse_gqa = _d64_causal_stat_values(o, do, lse, sm_scale, _D64_GQA_SIGNED)

    expected_delta = torch.sum(o * do, dim=-1)
    expected_lse = -lse * math.log2(math.e)
    torch.testing.assert_close(delta_mha, expected_delta)
    torch.testing.assert_close(delta_gqa, -expected_delta)
    assert lse_mha is None
    torch.testing.assert_close(lse_gqa, expected_lse)

    scores = q @ k.mT
    p_mha = torch.exp2((scores * sm_scale - lse[..., None]) * math.log2(math.e))
    p_gqa = torch.exp2(scores * (sm_scale * math.log2(math.e)) + lse_gqa[..., None])
    ds_mha = p_mha * (do @ v.mT - delta_mha[..., None])
    ds_gqa = p_gqa * (do @ v.mT + delta_gqa[..., None])
    torch.testing.assert_close(p_mha, p_gqa)
    torch.testing.assert_close(ds_mha, ds_gqa)

    with pytest.raises(ValueError, match=r"^unknown D64 stat mode 2$"):
        _d64_causal_stat_values(o, do, lse, sm_scale, 2)


@pytest.mark.parametrize(
    "invalid_scale",
    [
        pytest.param(0.0, id="zero"),
        pytest.param(float("inf"), id="infinite"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(10**10000, id="overflowing-integer"),
    ],
)
def test_d64_scale_validation_and_scheduled_fallback(invalid_scale):
    with pytest.raises(ValueError, match=r"^D64 sm_scale must be finite and nonzero$"):
        _validate_d64_sm_scale(invalid_scale)

    shape = (4, 64, 4096, 64)
    dispatch = _select_d64_dispatch(
        shape,
        shape,
        True,
        arch="gfx950",
        cu_count=256,
        sm_scale=invalid_scale,
        bases_aligned_16=True,
    )
    assert dispatch.family == "causal_m192"


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        pytest.param(
            (4, 64, 8, 8192, 8192, 192, 256, True),
            (
                _D64DQLaunch(42, False, 0, 42, 3, 0),
                _D64DQLaunch(1, False, 42, 1, 2, 192),
            ),
            id="peeled-tail",
        ),
        pytest.param(
            (4, 64, 8, 4096, 4096, 192, 256, True),
            (_D64DQLaunch(22, True, 0, 0, 3, 0), ),
            id="single-launch",
        ),
        pytest.param(
            (4, 64, 8, 8192, 8192, 192, 256, False),
            (_D64DQLaunch(43, False, 0, 0, 3, 0), ),
            id="no-host-tail-skip",
        ),
    ],
)
def test_d64_causal_dq_launch_plan(args, expected):
    assert _d64_dq_launch_plan(*args) == expected


@pytest.mark.parametrize(
    ("owner_start", "owner_rows", "sq", "skv", "block_n", "expected"),
    [
        (0, 192, 4096, 4096, 32, 6),
        (192, 192, 4096, 4096, 32, 12),
        (0, 192, 4096, 16384, 64, 195),
        (3840, 192, 4096, 4096, 32, 126),
    ],
)
def test_d64_causal_dq_compact_key_frontier(owner_start, owner_rows, sq, skv, block_n, expected):
    assert _d64_causal_dq_key_blocks(owner_start, owner_rows, sq, skv, block_n) == expected


@pytest.mark.parametrize(
    ("key_start", "sq", "skv", "block_m", "expected"),
    [
        (0, 4096, 4096, 64, 0),
        (256, 4096, 4096, 64, 4),
        (12288, 4096, 16384, 64, 0),
        (16320, 4096, 16384, 64, 63),
    ],
)
def test_d64_causal_dkdv_compact_query_frontier(key_start, sq, skv, block_m, expected):
    assert _d64_causal_dkdv_first_query_block(key_start, sq, skv, block_m) == expected


def test_d64_workspace_shapes():
    q = torch.empty((2, 32, 4096, 64), device="meta", dtype=torch.bfloat16)
    k_mha = torch.empty((2, 32, 4096, 64), device="meta", dtype=torch.bfloat16)
    k_gqa = torch.empty((2, 4, 4096, 64), device="meta", dtype=torch.bfloat16)

    lse_term, causal_dk, causal_dv = _allocate_bwd_d64_causal_gqa8_workspaces(q, k_gqa)
    assert lse_term.shape == (2, 32, 4096) and lse_term.dtype is torch.float32
    assert causal_dk.shape == causal_dv.shape == (2, 4, 4, 4096, 64)
    assert causal_dk.dtype is causal_dv.dtype is torch.bfloat16

    mha_dispatch = _D64Dispatch("noncausal_fused_n256", 32, 256, 1)
    gqa_dispatch = _D64Dispatch("noncausal_fused_n256", 32, 256, 8)
    mha_acc, mha_dk, mha_dv = _allocate_bwd_d64_fused_workspaces(q, k_mha, mha_dispatch)
    gqa_acc, gqa_dk, gqa_dv = _allocate_bwd_d64_fused_workspaces(q, k_gqa, gqa_dispatch)
    assert mha_acc.shape == gqa_acc.shape == q.shape
    assert mha_acc.dtype is gqa_acc.dtype is torch.float32
    assert mha_dk is mha_dv is None
    assert gqa_dk.shape == gqa_dv.shape == (2, 4, 8, 4096, 64)
    assert gqa_dk.dtype is gqa_dv.dtype is torch.bfloat16


def test_d64_direct_launch_uses_dispatch_ownership(monkeypatch):

    class LaunchRecorder:

        def __init__(self):
            self.calls = []

        def __getitem__(self, grid):

            def record(*args, **kwargs):
                self.calls.append((grid, args, kwargs))

            return record

    dq_launch = LaunchRecorder()
    dkdv_launch = LaunchRecorder()
    reduce_launch = LaunchRecorder()
    monkeypatch.setitem(vars(amd_fa_bwd), "_attn_bwd_dq_d64_direct_kernel", dq_launch)
    monkeypatch.setitem(vars(amd_fa_bwd), "_attn_bwd_dkdv_d64_direct_kernel", dkdv_launch)
    monkeypatch.setitem(vars(amd_fa_bwd), "_attn_bwd_dkdv_d64_reduce_kernel", reduce_launch)

    q_shape = (4, 48, 4096, 64)
    k_shape = (4, 6, 16384, 64)
    q = torch.empty(q_shape, device="meta", dtype=torch.bfloat16)
    k = torch.empty(k_shape, device="meta", dtype=torch.bfloat16)
    dispatch = _select_d64_dispatch(q_shape, k_shape, True)
    _run_bwd_d64_direct(q, k, k, q, object(), object(), q, k, k, 0.125, True, dispatch)

    assert len(dq_launch.calls) == 1
    dq_grid, _args, dq_kwargs = dq_launch.calls[0]
    assert dq_grid == (triton.cdiv(q_shape[2], 192), q_shape[1], q_shape[0])
    assert (dq_kwargs["OWNER_ROWS"], dq_kwargs["BLOCK_N"]) == (192, 64)

    assert len(dkdv_launch.calls) == 1
    dkdv_grid, _args, dkdv_kwargs = dkdv_launch.calls[0]
    assert dkdv_grid == (triton.cdiv(k_shape[2], 64), k_shape[1] * 4, k_shape[0])
    assert (dkdv_kwargs["KV_SPLITS"], dkdv_kwargs["BLOCK_N"]) == (4, 64)
    assert len(reduce_launch.calls) == 1


def test_varlen_d128_address_space_requires_i32_offsets():
    # One D128 BF16 head fits at most 2**30 elements in the signed i32
    # byte-offset range used by AMD buffer instructions.
    max_tokens = 2**23

    amd_fa_varlen_bwd._validate_i32_buffer_offsets(
        total_q=max_tokens - 15,
        total_kv=max_tokens,
        batch=1,
        q_heads=1,
        kv_heads=1,
        dq_atomic_fp32=False,
    )

    with pytest.raises(ValueError, match="KV tensor size exceeds the signed 32-bit byte-offset range"):
        amd_fa_varlen_bwd._validate_i32_buffer_offsets(
            total_q=1,
            total_kv=max_tokens + 1,
            batch=1,
            q_heads=1,
            kv_heads=1,
            dq_atomic_fp32=False,
        )
    with pytest.raises(ValueError, match="padded dQ size exceeds the signed 32-bit byte-offset range"):
        amd_fa_varlen_bwd._validate_i32_buffer_offsets(
            total_q=max_tokens - 14,
            total_kv=1,
            batch=1,
            q_heads=1,
            kv_heads=1,
            dq_atomic_fp32=False,
        )

    # FP32 dQ scratch reaches the same signed byte-offset limit at half as
    # many elements as the BF16 path.
    max_fp32_tokens = 2**22
    amd_fa_varlen_bwd._validate_i32_buffer_offsets(
        total_q=max_fp32_tokens - 15,
        total_kv=1,
        batch=1,
        q_heads=1,
        kv_heads=1,
        dq_atomic_fp32=True,
    )
    with pytest.raises(ValueError, match="padded dQ size exceeds the signed 32-bit byte-offset range"):
        amd_fa_varlen_bwd._validate_i32_buffer_offsets(
            total_q=max_fp32_tokens - 14,
            total_kv=1,
            batch=1,
            q_heads=1,
            kv_heads=1,
            dq_atomic_fp32=True,
        )
    amd_fa_varlen_bwd._validate_i32_buffer_offsets(
        total_q=max_fp32_tokens - 31,
        total_kv=1,
        batch=1,
        q_heads=1,
        kv_heads=1,
        dq_atomic_fp32=True,
        dq_pad_rows=32,
    )
    with pytest.raises(ValueError, match="padded dQ size exceeds the signed 32-bit byte-offset range"):
        amd_fa_varlen_bwd._validate_i32_buffer_offsets(
            total_q=max_fp32_tokens - 30,
            total_kv=1,
            batch=1,
            q_heads=1,
            kv_heads=1,
            dq_atomic_fp32=True,
            dq_pad_rows=32,
        )


@triton.jit
def _amd_iglp_opt_kernel(x_ptr, y_ptr, VARIANT: tl.constexpr):
    offsets = tl.arange(0, 64)
    values = tl.load(x_ptr + offsets)
    tlx.amd_iglp_opt(VARIANT)
    tl.store(y_ptr + offsets, values)


@triton.jit
def _amd_iglp_opt_dynamic_kernel(variant):
    tlx.amd_iglp_opt(variant)


def test_amd_iglp_opt_compiles_gfx950():
    compiled = compile_for_gfx950(
        _amd_iglp_opt_kernel,
        signature={"x_ptr": "*bf16", "y_ptr": "*bf16", "VARIANT": "constexpr"},
        constexprs={"VARIANT": 3},
    )
    calls = re.findall(r"call void @llvm\.amdgcn\.iglp\.opt\(i32 3\)", compiled.asm["llir"])
    assert len(calls) == 1


@pytest.mark.parametrize("variant", [False, 3.0, "3", -1, 4, 1 << 32])
def test_amd_iglp_opt_rejects_invalid_variant(variant):
    # Exercise frontend validation even when a string and integer constant
    # have the same textual ASTSource cache key.
    with triton.knobs.compilation.scope():
        triton.knobs.compilation.always_compile = True
        with pytest.raises(CompilationError, match="variant must be"):
            compile_for_gfx950(
                _amd_iglp_opt_kernel,
                signature={"x_ptr": "*bf16", "y_ptr": "*bf16", "VARIANT": "constexpr"},
                constexprs={"VARIANT": variant},
            )


def test_amd_iglp_opt_rejects_runtime_variant():
    with pytest.raises(CompilationError, match="variant must be a constexpr integer"):
        compile_for_gfx950(_amd_iglp_opt_dynamic_kernel, signature={"variant": "i32"}, constexprs={})


def test_amd_iglp_opt_rejects_cuda_backend():
    with pytest.raises(CompilationError, match="only supported on AMD"):
        compile_for_target(
            _amd_iglp_opt_kernel,
            signature={"x_ptr": "*bf16", "y_ptr": "*bf16", "VARIANT": "constexpr"},
            constexprs={"VARIANT": 3},
            target=GPUTarget("cuda", 90, 32),
        )

@triton.jit
def _explicit_mfma_reduce_add(a, b):
    return a + b


@triton.jit
def _explicit_mfma_batch_reduce_kernel(Input, Output, VERSION: tl.constexpr, INSTR_K: tl.constexpr,
                                       COLUMN_WARPS: tl.constexpr = 4):
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=VERSION,
        instr_shape=[16, 16, INSTR_K],
        transposed=True,
        warps_per_cta=[2, 1, COLUMN_WARPS],
    )
    batch = tl.arange(0, 2)
    rows = tl.arange(0, 16)
    cols = tl.arange(0, 64)
    offsets = batch[:, None, None] * 1024 + rows[None, :, None] * 64 + cols[None, None, :]
    values = tlx.require_layout(tl.load(Input + offsets), mma, pin=False)
    total = tl.reduce(values, 0, _explicit_mfma_reduce_add)
    output_offsets = tlx.require_layout(rows[:, None] * 64 + cols[None, :], tlx.slice_layout(mma, 0), pin=False)
    tl.store(Output + output_offsets, total)


def _lower_explicit_mfma_reduction_layouts(module, backend, options):
    from triton._C.libtriton import passes, tlx as tlx_ir

    backend.make_ttir(module, {}, options)
    pm = ir.pass_manager(module.context)
    pm.enable_debug()
    passes.ttir.add_convert_to_ttgpuir(pm, backend.get_target_name(options), options.num_warps, options.warp_size,
                                       options.num_ctas)
    tlx_ir.tlx_passes.add_tlx_resolve_placeholder_layouts(pm)
    pm.run(module, "explicit_mfma_reduction_layouts")


@pytest.mark.parametrize(
    "target,version,instr_k,num_warps,error",
    [
        pytest.param(GFX942, 3, 16, 8, None, id="gfx942"),
        pytest.param(GFX950, 4, 32, 8, None, id="gfx950"),
        pytest.param(GFX950, 4, 32, 4, "Layout has 8 warps per CTA, but the context requires 4 warps per CTA",
                     id="invalid-warps"),
        pytest.param(GFX1250, 4, 32, 8, "Layout has 64 threads per warp, but the module specifies 32 threads per warp",
                     id="invalid-wave-size"),
    ],
)
def test_explicit_mfma_reduction_deferred_layout_validation(target, version, instr_k, num_warps, error, capfd):
    from triton.compiler.code_generator import ast_to_ttir

    backend = triton.compiler.compiler.make_backend(target)
    options = backend.parse_options({"num_warps": num_warps})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    source = ASTSource(
        fn=_explicit_mfma_batch_reduce_kernel,
        signature={"Input": "*fp32", "Output": "*fp32"},
        constexprs={"VERSION": version, "INSTR_K": instr_k, "COLUMN_WARPS": 4},
    )
    codegen_fns = backend.get_codegen_implementation(options)
    # Layout verification must wait until target and execution context are known.
    module = ast_to_ttir(source.fn, source, context=context, options=options, codegen_fns=codegen_fns,
                         module_map=backend.get_module_map())
    codegen_fns["post_ast_lowering"](module)
    assert module.verify()
    if error is not None:
        with pytest.raises(RuntimeError):
            _lower_explicit_mfma_reduction_layouts(module, backend, options)
        assert error in capfd.readouterr().err
        return
    _lower_explicit_mfma_reduction_layouts(module, backend, options)
    assert module.verify()
    assert "tt.reduce" in str(module)
    assert "#ttg.slice" in str(module)
    assert "#tlx.no_verify_layout" not in str(module)


@triton.jit
def _explicit_mfma_batch_reduce_ws_kernel(Input, Output, VERSION: tl.constexpr, INSTR_K: tl.constexpr,
                                          PARTITION_WARPS: tl.constexpr):
    with tlx.async_tasks():
        with tlx.async_task("default"):
            _ = tl.arange(0, 1)
        with tlx.async_task(num_warps=PARTITION_WARPS):
            _explicit_mfma_batch_reduce_kernel(Input, Output, VERSION, INSTR_K, COLUMN_WARPS=2)


@pytest.mark.parametrize("target,version,instr_k", [(GFX942, 3, 16), (GFX950, 4, 32)])
@pytest.mark.parametrize("partition_warps", [4, 2])
def test_explicit_mfma_reduction_helper_partition_warp_count(target, version, instr_k, partition_warps, capfd):
    backend = triton.compiler.compiler.make_backend(target)
    options = backend.parse_options({"num_warps": 8})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    source = ASTSource(
        fn=_explicit_mfma_batch_reduce_ws_kernel,
        signature={"Input": "*fp32", "Output": "*fp32"},
        constexprs={"VERSION": version, "INSTR_K": instr_k, "PARTITION_WARPS": partition_warps},
    )
    module = source.make_ir(target, options, backend.get_codegen_implementation(options), backend.get_module_map(),
                            context)
    assert module.verify()
    if partition_warps == 2:
        with pytest.raises(RuntimeError):
            _lower_explicit_mfma_reduction_layouts(module, backend, options)
        assert "Layout has 4 warps per CTA, but the context requires 2 warps per CTA" in capfd.readouterr().err
        return
    _lower_explicit_mfma_reduction_layouts(module, backend, options)
    assert module.verify()
    module_text = str(module)
    assert "tt.call" not in module_text
    assert "tt.reduce" in module_text
    assert "num_warps(4)" in module_text
    assert "#tlx.no_verify_layout" not in module_text


@triton.jit
def _amd_scheduled_mfma_split_commit_kernel(
    a_ptr,
    k_ptr,
    p_ptr,
    do_ptr,
    v_ptr,
    output_ptr,
    REVERSE_COMMITS: tl.constexpr,
    INTERVENING_MFMA: tl.constexpr,
):
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[32, 32, 16],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    dot0: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=8)
    dot1: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=8)
    rows = tl.arange(0, 32)
    reduction = tl.arange(0, 16)
    cols = tl.arange(0, 128)
    a = tlx.require_layout(
        tl.load(a_ptr + rows[:, None] * 16 + reduction[None, :]),
        dot0,
        pin=False,
    )
    k = tlx.require_layout(
        tl.load(k_ptr + reduction[:, None] * 128 + cols[None, :]),
        dot1,
        pin=False,
    )
    p = tlx.require_layout(
        tl.load(p_ptr + rows[:, None] * 16 + reduction[None, :]),
        dot0,
        pin=False,
    )
    do = tlx.require_layout(
        tl.load(do_ptr + reduction[:, None] * 128 + cols[None, :]),
        dot1,
        pin=False,
    )
    v_live = tlx.require_layout(
        tl.load(v_ptr + reduction[:, None] * 128 + cols[None, :]),
        dot1,
        pin=False,
    )
    v_live = tlx.amd_register_resident(
        v_live,
        register_class="agpr",
        registers_per_group=4,
    )
    dk = tlx.zeros((32, 128), tl.float32, layout=mma)
    dv = tlx.zeros((32, 128), tl.float32, layout=mma)
    dk = tlx.amd_scheduled_mfma(
        a,
        k,
        dk,
        accumulator_role="persistent",
        accumulator_register_class="agpr",
        initialize=True,
    )
    dv = tlx.amd_scheduled_mfma(
        p,
        do,
        dv,
        accumulator_role="persistent",
        accumulator_register_class="vgpr",
        initialize=True,
    )
    if REVERSE_COMMITS:
        dv, v_live = tlx.amd_mfma_commit(dv, preserve=v_live)
        dk = tlx.amd_mfma_commit(dk)
    else:
        dk = tlx.amd_mfma_commit(dk)
        if INTERVENING_MFMA:
            dv = tlx.amd_scheduled_mfma(
                p,
                do,
                dv,
                accumulator_role="persistent",
                accumulator_register_class="vgpr",
            )
        dv, v_live = tlx.amd_mfma_commit(dv, preserve=v_live)
    output_offsets = tlx.require_layout(
        output_ptr + rows[:, None] * 128 + cols[None, :],
        mma,
        pin=False,
    )
    tl.store(output_offsets, dk + dv)


@triton.jit
def _amd_scheduled_mfma_short_first_split_commit_kernel(
    short_a_ptr,
    short_b_ptr,
    long_a_ptr,
    long_b_ptr,
    live_ptr,
    short_output_ptr,
    long_output_ptr,
):
    short_mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    short_dot0: tl.constexpr = tlx.dot_operand_layout(0, short_mma, k_width=8)
    short_dot1: tl.constexpr = tlx.dot_operand_layout(1, short_mma, k_width=8)
    short_rows = tl.arange(0, 16)
    short_reduction = tl.arange(0, 32)
    short_cols = tl.arange(0, 64)
    short_a = tlx.require_layout(
        tl.load(short_a_ptr + short_rows[:, None] * 32 + short_reduction[None, :]),
        short_dot0,
        pin=False,
    )
    short_b = tlx.require_layout(
        tl.load(short_b_ptr + short_reduction[:, None] * 64 + short_cols[None, :]),
        short_dot1,
        pin=False,
    )
    short_result = tlx.zeros((16, 64), tl.float32, layout=short_mma)
    short_result = tlx.amd_scheduled_mfma(
        short_a,
        short_b,
        short_result,
        accumulator_role="persistent",
        accumulator_register_class="agpr",
        initialize=True,
    )

    long_mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[32, 32, 16],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    long_dot0: tl.constexpr = tlx.dot_operand_layout(0, long_mma, k_width=8)
    long_dot1: tl.constexpr = tlx.dot_operand_layout(1, long_mma, k_width=8)
    long_rows = tl.arange(0, 32)
    long_reduction = tl.arange(0, 16)
    long_cols = tl.arange(0, 128)
    long_a = tlx.require_layout(
        tl.load(long_a_ptr + long_rows[:, None] * 16 + long_reduction[None, :]),
        long_dot0,
        pin=False,
    )
    long_b = tlx.require_layout(
        tl.load(long_b_ptr + long_reduction[:, None] * 128 + long_cols[None, :]),
        long_dot1,
        pin=False,
    )
    live = tlx.require_layout(
        tl.load(live_ptr + long_reduction[:, None] * 128 + long_cols[None, :]),
        long_dot1,
        pin=False,
    )
    live = tlx.amd_register_resident(
        live,
        register_class="agpr",
        registers_per_group=4,
    )
    long_result = tlx.zeros((32, 128), tl.float32, layout=long_mma)
    long_result = tlx.amd_scheduled_mfma(
        long_a,
        long_b,
        long_result,
        accumulator_role="persistent",
        accumulator_register_class="vgpr",
        initialize=True,
    )

    short_result = tlx.amd_mfma_commit(short_result)
    long_result, live = tlx.amd_mfma_commit(long_result, preserve=live)
    short_offsets = tlx.require_layout(
        short_output_ptr + short_rows[:, None] * 64 + short_cols[None, :],
        short_mma,
        pin=False,
    )
    long_offsets = tlx.require_layout(
        long_output_ptr + long_rows[:, None] * 128 + long_cols[None, :],
        long_mma,
        pin=False,
    )
    tl.store(short_offsets, short_result)
    tl.store(long_offsets, long_result)


















@triton.jit
def _amd_scheduled_mfma_zero_batch_warps_kernel():
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[0, 1, 1],
    )
    lhs: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=8)
    rhs: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=8)
    a = tlx.zeros((2, 16, 32), tl.bfloat16, layout=lhs)
    b = tlx.zeros((2, 32, 16), tl.bfloat16, layout=rhs)
    acc = tlx.zeros((2, 16, 16), tl.float32, layout=mma)
    tlx.amd_scheduled_mfma(a, b, acc, accumulator_role="persistent", initialize=True)


def test_amd_scheduled_mfma_zero_batch_warps_rejected():
    # Build the layout through the DSL to exercise unchecked attribute builders,
    # which do not run the textual parser's attribute validation.
    # Use one warp so earlier layout checks do not mask the MFMA zero-warp check.
    source = ASTSource(fn=_amd_scheduled_mfma_zero_batch_warps_kernel, signature={}, constexprs={})
    with pytest.raises(RuntimeError, match="MFMA warpsPerCTA entries must be positive"):
        triton_compile(source, target=GFX950, options={"num_warps": 1})








def test_amd_scheduled_mfma_vgpr_results_only_commit_is_conservative_gfx950():
    compiled = compile_for_gfx950(
        _amd_scheduled_mfma_persistent_acc_kernel,
        signature={
            "a_ptr": "*bf16",
            "b_ptr": "*bf16",
            "output_ptr": "*fp32",
            "USE_VGPR": "constexpr",
            "COMMIT": "constexpr",
            "INITIALIZE": "constexpr",
            "FULL_K": "constexpr",
        },
        constexprs={"USE_VGPR": True, "COMMIT": True, "INITIALIZE": True, "FULL_K": False},
    )
    llir = compiled.asm["llir"]
    assert "; triton_amd_scheduled_mfma\\0A" not in llir
    assert not re.search(r'asm sideeffect "[^"\n]*v_mfma_', llir)
    assert len(re.findall(r'call <4 x float> @llvm\.amdgcn\.mfma\.f32\.16x16x32\.bf16\(', llir)) == 2
    assert 'asm sideeffect "", "=v,0"' in llir
    assert 'asm sideeffect "s_nop 11", "=a,0,~{memory}"' in llir


@triton.jit
def _amd_scheduled_mfma_output_subtiles_atomic_kernel(a_ptr, b_ptr, atomic_ptr, output_ptr):
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
    cols = tl.arange(0, 128)
    a = tlx.require_layout(
        tl.load(a_ptr + rows[:, None] * 32 + reduction[None, :]),
        dot0,
        pin=False,
    )
    b = tlx.require_layout(
        tl.load(b_ptr + reduction[:, None] * 128 + cols[None, :]),
        dot1,
        pin=False,
    )
    b_left = tlx.extract_slice(b, [32, 64], [0, 0])
    b_right = tlx.extract_slice(b, [32, 64], [0, 64])
    acc_left = tlx.zeros((16, 64), tl.float32, layout=mma)
    acc_right = tlx.zeros((16, 64), tl.float32, layout=mma)
    acc_left = tlx.amd_scheduled_mfma(
        a,
        b_left,
        acc_left,
        accumulator_role="persistent",
        initialize=True,
    )
    tlx.amd_sched_barrier(0)
    atomic_layout: tl.constexpr = tlx.layout(
        shape=((64, 4), ()),
        stride=((1, 0), ()),
    )
    atomic_offsets = tl.arange(0, 64).to(tl.int32)
    atomic_offsets = tlx.require_layout(atomic_offsets, atomic_layout, pin=False)
    atomic_value = tl.full((64, ), 1.0, tl.float32)
    atomic_value = tlx.require_layout(atomic_value, atomic_layout, pin=False)
    tlx.buffer_atomic_add(
        atomic_ptr,
        atomic_offsets,
        atomic_value,
        mask=atomic_offsets < 32,
        sem="relaxed",
    )
    tlx.amd_sched_barrier(0)
    acc_right = tlx.amd_scheduled_mfma(
        a,
        b_right,
        acc_right,
        accumulator_role="persistent",
        initialize=True,
    )
    acc_left, acc_right = tlx.amd_mfma_commit((acc_left, acc_right))
    acc = tlx.require_layout(tl.cat(acc_left, acc_right, dim=1), mma, pin=False)
    output_offsets = tlx.require_layout(
        output_ptr + rows[:, None] * 128 + cols[None, :],
        mma,
        pin=False,
    )
    tl.store(output_offsets, acc)


def test_amd_scheduled_mfma_output_subtiles_order_atomic_gfx950():
    compiled = compile_for_gfx950(
        _amd_scheduled_mfma_output_subtiles_atomic_kernel,
        signature={
            "a_ptr": "*bf16",
            "b_ptr": "*bf16",
            "atomic_ptr": "*fp32",
            "output_ptr": "*fp32",
        },
        constexprs={},
    )
    ttir = compiled.asm["ttir"]
    assert "output_fragment" not in ttir
    subtiles = [line for line in ttir.splitlines() if "amdg.scheduled_mfma" in line]
    assert len(subtiles) == 2
    assert all("tensor<16x64xf32" in line for line in subtiles)
    assert ttir.count("amdg.mfma_commit") == 1

    mnemonic = "v_mfma_f32_16x16x32_bf16"
    instructions = [
        "mfma" if mnemonic in line else "atomic"
        for line in compiled.asm["amdgcn"].splitlines()
        if mnemonic in line or "buffer_atomic_add_f32" in line
    ]
    assert instructions == ["mfma", "atomic", "mfma"]


@triton.jit
def _amd_scheduled_mfma_persistent_acc_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    USE_VGPR: tl.constexpr,
    COMMIT: tl.constexpr,
    INITIALIZE: tl.constexpr = True,
    FULL_K: tl.constexpr = False,
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
    acc = tl.full((16, 64), 7.0, tl.float32)
    acc = tlx.require_layout(acc, mma, pin=False)
    if FULL_K:
        # Exercise both K fragments inside one scheduled operation.
        acc = tlx.amd_scheduled_mfma(
            a,
            b,
            acc,
            accumulator_role="persistent",
            accumulator_register_class="vgpr" if USE_VGPR else None,
            initialize=INITIALIZE,
        )
    else:
        a0 = tlx.extract_slice(a, [16, 32], [0, 0])
        b0 = tlx.extract_slice(b, [32, 64], [0, 0])
        acc = tlx.amd_scheduled_mfma(
            a0,
            b0,
            acc,
            accumulator_role="persistent",
            accumulator_register_class="vgpr" if USE_VGPR else None,
            initialize=INITIALIZE,
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
    if COMMIT:
        acc = tlx.amd_mfma_commit(acc)
    output_offsets = output_ptr + rows[:, None] * 64 + cols[None, :]
    output_offsets = tlx.require_layout(output_offsets, mma, pin=False)
    tl.store(output_offsets, acc)




@triton.jit
def _buffer_store_mfma_layout_kernel(
    out_ptr,
    mask_boundary,
    CONTIGUITY: tl.constexpr,
):
    store_layout: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[2, 2],
    )
    rows = tl.arange(0, 64)
    cols = tl.arange(0, 32)
    offsets = (rows[:, None] * 32 + cols[None, :]).to(tl.int32)
    values = tl.full((64, 32), 1.0, tl.bfloat16)
    values = tlx.require_layout(values, store_layout)
    mask_boundary = mask_boundary - mask_boundary % CONTIGUITY
    tlx.buffer_store(
        values,
        out_ptr,
        offsets,
        mask=offsets < mask_boundary,
        contiguity=CONTIGUITY,
    )


@pytest.mark.parametrize(
    "contiguity,opcode",
    [(2, "buffer_store_dword"), (4, "buffer_store_dwordx2")],
    ids=["width2", "width4"],
)
def test_buffer_store_mfma_layout_vectorizes_gfx950(contiguity, opcode):
    compiled = compile_for_gfx950(
        _buffer_store_mfma_layout_kernel,
        signature={
            "out_ptr": "*bf16",
            "mask_boundary": "i32",
            "CONTIGUITY": "constexpr",
        },
        constexprs={"CONTIGUITY": contiguity},
    )

    ttir = compiled.asm["ttir"]
    ttgir = compiled.asm["ttgir"]
    amdgcn = compiled.asm["amdgcn"]
    assert ttir.count("tlx.require_layout") == 3
    assert "#ttg.amd_mfma" in ttgir
    assert "isTransposed = true" in ttgir
    assert "amdg.buffer_store" in ttgir
    assert f"contiguity = {contiguity}" in ttgir
    assert "tlx.preserve_layout" not in ttgir
    assert re.search(rf"^\s*{opcode}\b", amdgcn, re.MULTILINE)
    assert "buffer_store_short" not in amdgcn


@pytest.mark.parametrize("contiguity", [0, 3, True], ids=["zero", "non-power-of-two", "bool"])
def test_buffer_store_rejects_invalid_contiguity_gfx950(contiguity):
    with pytest.raises(CompilationError, match="contiguity must be a positive power of two"):
        compile_for_gfx950(
            _buffer_store_mfma_layout_kernel,
            signature={
                "out_ptr": "*bf16",
                "mask_boundary": "i32",
                "CONTIGUITY": "constexpr",
            },
            constexprs={"CONTIGUITY": contiguity},
        )


def test_buffer_store_rejects_contiguity_wider_than_layout_gfx950():
    with pytest.raises(RuntimeError, match="exceeds physical layout contiguity"):
        compile_for_gfx950(
            _buffer_store_mfma_layout_kernel,
            signature={
                "out_ptr": "*bf16",
                "mask_boundary": "i32",
                "CONTIGUITY": "constexpr",
            },
            constexprs={"CONTIGUITY": 8},
        )


@triton.jit
def _buffer_store_without_layout_kernel(out_ptr, CONTIGUITY: tl.constexpr):
    offsets = tl.arange(0, 128).to(tl.int32)
    values = tl.full((128, ), 1.0, tl.bfloat16)
    tlx.buffer_store(values, out_ptr, offsets, contiguity=CONTIGUITY)


def test_buffer_store_rejects_contiguity_without_layout_gfx950():
    with pytest.raises(
            CompilationError,
            match="contiguity > 1 requires stored_value to be pinned with tlx.require_layout",
    ):
        compile_for_gfx950(
            _buffer_store_without_layout_kernel,
            signature={"out_ptr": "*bf16", "CONTIGUITY": "constexpr"},
            constexprs={"CONTIGUITY": 2},
        )


@triton.jit
def _buffer_store_overwide_hint_kernel(in_ptr, out_ptr):
    store_layout: tl.constexpr = tlx.layout(
        shape=((64, 4), (16, )),
        stride=((16, 1024), (1, )),
    )
    offsets = tl.arange(0, 4096).to(tl.int32)
    values = tl.load(in_ptr + offsets)
    values = tlx.require_layout(values, store_layout)
    tlx.buffer_store(
        values,
        out_ptr,
        offsets,
        contiguity=16,
    )


def test_buffer_store_caps_overwide_hint_to_128_bits_gfx950():
    compiled = compile_for_gfx950(
        _buffer_store_overwide_hint_kernel,
        signature={"in_ptr": "*bf16", "out_ptr": "*bf16"},
        constexprs={},
    )

    assert "contiguity = 16" in compiled.asm["ttgir"]
    assert "buffer_store_dwordx4" in compiled.asm["amdgcn"]
    assert "buffer_store_dwordx8" not in compiled.asm["amdgcn"]




def compile_for_gfx942(fn, signature, constexprs):
    """Compile a TLX kernel for gfx942 and return the compiled object."""
    return compile_for_target(fn, signature, constexprs, GFX942)


@triton.jit
def _amd_scheduled_mfma_gfx942_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    PERSISTENT: tl.constexpr,
    TRANSPOSED: tl.constexpr,
    INSTR_K: tl.constexpr,
):
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=3,
        instr_shape=[16, 16, INSTR_K],
        transposed=TRANSPOSED,
        warps_per_cta=[1, 4],
    )
    dot0: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=4)
    dot1: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=4)
    rows = tl.arange(0, 32)
    reduction = tl.arange(0, 32)
    cols = tl.arange(0, 128)
    a = tlx.require_layout(
        tl.load(a_ptr + rows[:, None] * 32 + reduction[None, :]),
        dot0,
        pin=False,
    )
    b = tlx.require_layout(
        tl.load(b_ptr + reduction[:, None] * 128 + cols[None, :]),
        dot1,
        pin=False,
    )
    acc = tl.full((32, 128), 7.0, tl.float32)
    acc = tlx.require_layout(acc, mma, pin=False)
    if PERSISTENT:
        result = tlx.amd_scheduled_mfma(
            a,
            b,
            acc,
            accumulator_role="persistent",
            # On CDNA3 the compiler-generated AGPR read is not ordered
            # against the MFMA drain, so the accumulator stays in VGPRs.
            accumulator_register_class="vgpr",
            initialize=True,
        )
    else:
        result = tlx.amd_scheduled_mfma(
            a,
            b,
            acc,
            accumulator_role="transient",
            initialize=True,
        )
        result, _ = tlx.amd_mfma_commit(result, b)
    offsets = output_ptr + rows[:, None] * 128 + cols[None, :]
    offsets = tlx.require_layout(offsets, mma, pin=False)
    tl.store(offsets, result)


@triton.jit
def _amd_scheduled_mfma_opaque_consumer_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    iterations,
    VERSION: tl.constexpr,
    INSTR_K: tl.constexpr,
    CONSUMER: tl.constexpr,
):
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=VERSION,
        instr_shape=[16, 16, INSTR_K],
        transposed=False,
        warps_per_cta=[1, 1],
    )
    dot0: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=INSTR_K // 4)
    dot1: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=INSTR_K // 4)
    rows = tl.arange(0, 16)
    reduction = tl.arange(0, INSTR_K)
    cols = tl.arange(0, 16)
    a = tlx.require_layout(tl.load(a_ptr + rows[:, None] * INSTR_K + reduction[None, :]), dot0, pin=False)
    b = tlx.require_layout(tl.load(b_ptr + reduction[:, None] * 16 + cols[None, :]), dot1, pin=False)
    acc = tlx.zeros((16, 16), tl.float32, layout=mma)
    if CONSUMER == "if":
        if iterations != 0:
            acc = tlx.amd_scheduled_mfma(a, b, acc, accumulator_role="persistent", accumulator_register_class="vgpr",
                                         initialize=True)
    elif CONSUMER == "loop":
        for _ in range(iterations):
            acc = tlx.amd_scheduled_mfma(a, b, acc, accumulator_role="persistent", accumulator_register_class="vgpr")
    else:
        acc = tlx.amd_scheduled_mfma(a, b, acc, accumulator_role="persistent", accumulator_register_class="vgpr",
                                     initialize=True)

    if CONSUMER == "layout":
        acc = tlx.require_layout(tlx.release_layout(acc), mma, pin=False)
    elif CONSUMER == "reshape":
        acc = tl.reshape(tlx.release_layout(acc), [256])

    if CONSUMER == "native":
        acc = acc + 1.0
    else:
        acc = tl.inline_asm_elementwise("v_add_f32 $0, $1, 1.0", constraints="=v,v", args=[acc], dtype=tl.float32,
                                        is_pure=False, pack=1)

    if CONSUMER == "reshape":
        tl.store(output_ptr + tl.arange(0, 256), acc)
    else:
        output_offsets = tlx.require_layout(output_ptr + rows[:, None] * 16 + cols[None, :], mma, pin=False)
        tl.store(output_offsets, acc)


@triton.jit
def _amd_scheduled_mfma_dead_result_writer_kernel(
    output_ptr,
    VERSION: tl.constexpr,
    INSTR_M: tl.constexpr,
    INSTR_K: tl.constexpr,
    K_WIDTH: tl.constexpr,
    WRITER: tl.constexpr,
):
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=VERSION,
        instr_shape=[INSTR_M, INSTR_M, INSTR_K],
        transposed=False,
        warps_per_cta=[1, 1],
    )
    dot0: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=K_WIDTH)
    dot1: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=K_WIDTH)
    BLOCK_N: tl.constexpr = INSTR_M * (2 if WRITER == "partial" else 1)
    a = tlx.require_layout(tl.full((INSTR_M, INSTR_K), 1.0, tl.bfloat16), dot0, pin=False)
    b = tl.full((INSTR_K, BLOCK_N), 1.0, tl.bfloat16)
    if WRITER == "partial":
        # Distinct tiles prevent LLVM from merging the two native MFMAs.
        b = (b + (tl.arange(0, BLOCK_N)[None, :] >= INSTR_M)).to(tl.bfloat16)
    b = tlx.require_layout(b, dot1, pin=False)
    zero = tlx.zeros((INSTR_M, BLOCK_N), tl.float32, layout=mma)
    result = tlx.amd_scheduled_mfma(a, b, zero, accumulator_role="persistent", accumulator_register_class="vgpr",
                                    initialize=True)
    if WRITER == "commit":
        tlx.amd_mfma_commit(result)

    # Unused result fragments let the writer reuse their physical registers.
    # It deliberately has no SSA dependency on the MFMA destination.
    if WRITER == "native":
        written = zero + 1.0
    else:
        written = tl.inline_asm_elementwise("v_mov_b32 $0, 1.0", constraints="=v,v", args=[zero], dtype=tl.float32,
                                            is_pure=WRITER == "pure", pack=1)
    offsets = output_ptr + tl.arange(0, INSTR_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    tl.store(tlx.require_layout(offsets, mma, pin=False), written)
    if WRITER == "partial":
        # Keep only the first native fragment live across the opaque writer.
        live = tlx.extract_slice(result, [INSTR_M, INSTR_M], [0, 0])
        live_offsets = (output_ptr + INSTR_M * BLOCK_N + tl.arange(0, INSTR_M)[:, None] * INSTR_M +
                        tl.arange(0, INSTR_M)[None, :])
        tl.store(tlx.require_layout(live_offsets, mma, pin=False), live)


@pytest.mark.parametrize("elem_ty", ["bf16", "fp16"])
@pytest.mark.parametrize("persistent", [False, True])
def test_amd_scheduled_mfma_compiles_gfx942(elem_ty, persistent):
    compiled = compile_for_gfx942(
        _amd_scheduled_mfma_gfx942_kernel,
        signature={
            "a_ptr": f"*{elem_ty}",
            "b_ptr": f"*{elem_ty}",
            "output_ptr": "*fp32",
            "PERSISTENT": "constexpr",
            "TRANSPOSED": "constexpr",
            "INSTR_K": "constexpr",
        },
        constexprs={"PERSISTENT": persistent, "TRANSPOSED": True, "INSTR_K": 16},
    )
    asm_ty = "f16" if elem_ty == "fp16" else "bf16"
    assert "amdg.scheduled_mfma" in compiled.asm["ttir"]
    assert f"v_mfma_f32_16x16x16_{asm_ty}" in compiled.asm["amdgcn"]
    intrinsic_ty = "f16" if elem_ty == "fp16" else "bf16.1k"
    assert f"@llvm.amdgcn.mfma.f32.16x16x16{intrinsic_ty}" in compiled.asm["llir"]
    assert not re.search(r'asm sideeffect "[^"\n]*v_mfma_', compiled.asm["llir"])
    if persistent:
        assert 'asm sideeffect "", "=v,0"' in compiled.asm["llir"]
        # Global loads do not prove shared-load provenance. Operand pins stay
        # pure while persistent accumulator placement remains ordered.
        class_pins = re.findall(r'asm( sideeffect)? "", "=[av],0"\(', compiled.asm["llir"])
        assert class_pins
        assert not all(class_pins), "global-load operand pins must remain pure"
        assert "ttg.amdg.scheduled_mfma_operand_pin" not in compiled.asm["llir"]
    else:
        # A live-dependency commit still needs the full CDNA3 result drain.
        assert 'asm sideeffect "s_nop 10"' in compiled.asm["llir"]


def test_amd_scheduled_mfma_native_grid_gfx942():
    compiled = compile_for_gfx942(
        _amd_scheduled_mfma_gfx942_kernel,
        signature={
            "a_ptr": "*fp16",
            "b_ptr": "*fp16",
            "output_ptr": "*fp32",
            "PERSISTENT": "constexpr",
            "TRANSPOSED": "constexpr",
            "INSTR_K": "constexpr",
        },
        constexprs={"PERSISTENT": True, "TRANSPOSED": True, "INSTR_K": 16},
    )
    mnemonic = "v_mfma_f32_16x16x16_f16"
    mfmas = [line.strip() for line in compiled.asm["amdgcn"].splitlines() if mnemonic in line]

    assert len(mfmas) == 8
    # M,N,K = 32,128,32, warps_per_cta = [1,4]
    # each warp computes 32x32 => 2 M-reps x 2 N-reps
    # K steps => 32/16 = 2

    # Native MFMA instructions retain the eight grid updates while LLVM may
    # reschedule them; source-level K/N/M ordering is checked in lit.
    assert len(re.findall(r'call <4 x float> @llvm\.amdgcn\.mfma\.f32\.16x16x16f16\(', compiled.asm["llir"])) == 8


@pytest.mark.parametrize("elem_ty", ["bf16", "fp16"])
@pytest.mark.parametrize("use_vgpr", [False, True])
def test_amd_scheduled_mfma_persistent_acc_lowering_gfx950(elem_ty, use_vgpr):
    compiled = compile_for_gfx950(
        _amd_scheduled_mfma_persistent_acc_kernel,
        signature={
            "a_ptr": f"*{elem_ty}",
            "b_ptr": f"*{elem_ty}",
            "output_ptr": "*fp32",
            "USE_VGPR": "constexpr",
            "COMMIT": "constexpr",
            "INITIALIZE": "constexpr",
            "FULL_K": "constexpr",
        },
        constexprs={"USE_VGPR": use_vgpr, "COMMIT": False, "INITIALIZE": True, "FULL_K": False},
    )
    llir = compiled.asm["llir"]
    intrinsic_ty = "f16" if elem_ty == "fp16" else elem_ty
    assert not re.search(r'asm sideeffect "[^"\n]*v_mfma_', llir)
    results = re.findall(
        r'(%[\w.]+) = (?:tail )?call <4 x float> @llvm\.amdgcn\.mfma\.f32\.16x16x32\.' + intrinsic_ty + r'\(', llir)
    assert len(results) == 2
    # These operands originate from global loads. Their class pins stay pure;
    # the C/D anchors below must still retain side-effect ordering.
    class_pins = re.findall(r'asm( sideeffect)? "", "=[av],0"\(', llir)
    assert class_pins
    assert not all(class_pins), "global-load operand pins must remain pure"
    assert "ttg.amdg.scheduled_mfma_operand_pin" not in llir
    register_class = "v" if use_vgpr else "a"
    for result in results:
        packed = re.search(r'(%[\w.]+) = bitcast <4 x float> ' + re.escape(result) + r' to <4 x i32>', llir)
        assert packed is not None
        assert re.search(
            r'asm sideeffect "", "=' + register_class + r',0"\(<4 x i32> ' + re.escape(packed.group(1)) + r'\)', llir)
    assert 'asm sideeffect "s_nop' not in llir


@pytest.mark.parametrize("target,version,instr_k,wait_states", [(GFX942, 3, 16, 11), (GFX950, 4, 32, 12)],
                         ids=["gfx942", "gfx950"])
@pytest.mark.parametrize("consumer", ["direct", "layout", "reshape", "if", "loop", "native"])
def test_amd_scheduled_mfma_opaque_consumer_completion(target, version, instr_k, wait_states, consumer):
    src = ASTSource(
        fn=_amd_scheduled_mfma_opaque_consumer_kernel,
        signature={
            "a_ptr": "*bf16",
            "b_ptr": "*bf16",
            "output_ptr": "*fp32",
            "iterations": "i32",
            "VERSION": "constexpr",
            "INSTR_K": "constexpr",
            "CONSUMER": "constexpr",
        },
        constexprs={"VERSION": version, "INSTR_K": instr_k, "CONSUMER": consumer},
    )
    compiled = triton_compile(src, target=target, options={"num_warps": 1})
    ttgir = compiled.asm["ttgir"]
    assert "amdg.mfma_commit" not in compiled.asm["ttir"]
    if consumer == "if":
        assert "scf.if" in ttgir
    elif consumer == "loop":
        assert "scf.for" in ttgir
    llir = compiled.asm["llir"]
    mfma = re.search(r'call <4 x float> @llvm\.amdgcn\.mfma\.f32\.16x16x' + str(instr_k), llir)
    assert mfma is not None
    assert not re.search(r'asm sideeffect "[^"\n]*v_mfma_', llir)
    # Inspect the tied completion boundaries separately from the waits now
    # prefixed to the opaque consumer's own assembly.
    waits = list(re.finditer(r'asm sideeffect "(s_nop \d+(?:\\0As_nop \d+)*)", "([^"\n]*)"', llir))
    if consumer == "native":
        assert not re.search(r'asm(?: sideeffect)? "[^"\n]*s_nop', llir)
        return
    opaque_consumer = re.search(r'asm sideeffect "[^"\n]*v_add_f32', llir)
    assert opaque_consumer is not None
    opaque_consumer = opaque_consumer.start()
    if consumer in ("layout", "reshape"):
        # The alias analysis conservatively follows LDS reloads too, even if
        # this particular layout transfer survives LLVM optimization.
        after_mfma = ttgir[ttgir.index("amdg.scheduled_mfma"):]
        assert "ttg.convert_layout" in after_mfma[:after_mfma.index("tt.elementwise_inline_asm")]
        native_transfer = llir[mfma.end():opaque_consumer]
        assert re.search(r'store <\d+ x float>.*ptr addrspace\(3\)', native_transfer)
        assert re.search(r'load <\d+ x float>, ptr addrspace\(3\)', native_transfer)

    assert waits
    for wait in waits:
        # Check the completion duration and SSA dependency, independent of how
        # LLVM schedules the final MFMA and consumer instructions.
        assert sum(int(count) + 1 for count in re.findall(r's_nop (\d+)', wait.group(1))) >= wait_states
        assert wait.group(2).startswith("=")
        assert "0" in wait.group(2).split(",")
    if consumer == "direct":
        assert mfma.start() < waits[0].start() < opaque_consumer


@pytest.mark.parametrize(
    "target,version,instr_m,instr_k,k_width,wait_states",
    [(GFX942, 3, 16, 16, 4, 11), (GFX942, 3, 32, 8, 4, 19), (GFX950, 4, 16, 32, 8, 12), (GFX950, 4, 32, 16, 8, 20)],
    ids=["gfx942_16x16", "gfx942_32x32", "gfx950_16x16", "gfx950_32x32"],
)
@pytest.mark.parametrize("writer", ["opaque", "pure", "partial", "commit", "native"])
def test_amd_scheduled_mfma_dead_result_writer_completion(target, version, instr_m, instr_k, k_width, wait_states,
                                                          writer):
    src = ASTSource(
        fn=_amd_scheduled_mfma_dead_result_writer_kernel,
        signature={
            "output_ptr": "*fp32",
            "VERSION": "constexpr",
            "INSTR_M": "constexpr",
            "INSTR_K": "constexpr",
            "K_WIDTH": "constexpr",
            "WRITER": "constexpr",
        },
        constexprs={"VERSION": version, "INSTR_M": instr_m, "INSTR_K": instr_k, "K_WIDTH": k_width, "WRITER": writer},
    )
    compiled = triton_compile(src, target=target, options={"num_warps": 1, "matrix_instr_nonkdim": instr_m})
    llir = compiled.asm["llir"]
    registers = instr_m * instr_m // 64
    mfmas = list(
        re.finditer(
            rf'(%[\w.]+) = (?:tail )?call <{registers} x float> @llvm\.amdgcn\.mfma\.f32\.{instr_m}x{instr_m}x{instr_k}',
            llir,
        ))
    assert len(mfmas) == (2 if writer == "partial" else 1)
    assert not re.search(r'asm sideeffect "[^"\n]*v_mfma_', llir)
    writers = list(re.finditer(r'asm( sideeffect)? "([^"\n]*v_mov_b32[^"\n]*)", "=v,v"', llir))
    if writer == "native":
        assert not writers
        assert not re.search(r'asm(?: sideeffect)? "[^"\n]*s_nop', llir)
        return

    assert writers
    for opaque_writer in writers:
        # Put the delay inside the writer so pure assembly cannot be moved
        # away from it, even when the MFMA result has no SSA consumer.
        prefix, instruction = opaque_writer.group(2).split("v_mov_b32", 1)
        assert instruction == " $0, 1.0"
        assert re.fullmatch(r"(?:s_nop \d+\\0A)+", prefix)
        assert sum(int(count) + 1 for count in re.findall(r"s_nop (\d+)", prefix)) >= wait_states
        assert bool(opaque_writer.group(1)) == (writer != "pure")
    if writer == "commit":
        # The explicit commit remains a separate tied completion boundary.
        assert re.search(r'asm sideeffect "s_nop \d+(?:\\0As_nop \d+)*", "=a,0,~\{memory\}"', llir)


def _amd_scheduled_mfma_memory_forward_ir(target, version, instr_k, k_width, memory, consumer):
    if memory == "global":
        pointer_ir = """%scratch_ptr = builtin.unrealized_conversion_cast %scratch : !tt.ptr<f32> to !llvm.ptr<1>
    %slot = llvm.getelementptr %scratch_ptr[%lane] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32"""
        pointer_type = "!llvm.ptr<1>"
    else:
        pointer_ir = "%slot = llvm.alloca %count x f32 : (i32) -> !llvm.ptr<5>"
        pointer_type = "!llvm.ptr<5>"
        if memory == "private_cast":
            pointer_ir = """%allocation = llvm.alloca %count x f32 : (i32) -> !llvm.ptr<5>
    %slot = llvm.addrspacecast %allocation : !llvm.ptr<5> to !llvm.ptr"""
            pointer_type = "!llvm.ptr"
    overwrite = "llvm.store %zero_f32, %slot : f32, !llvm.ptr<1>" if memory == "global" else ""
    commit = "%committed = amdg.mfma_commit %result : tensor<16x16xf32, #mma>" if consumer == "commit" else ""
    result = "%committed" if consumer == "commit" else "%result"
    read = "%read = llvm.fadd %reloaded, %one_f32 : f32" if consumer == "native" else (
        '%read = llvm.inline_asm has_side_effects "v_add_f32 $0, $1, 1.0", "=v,v" %reloaded : (f32) -> f32')
    return f"""
#mma = #ttg.amd_mfma<{{version = {version}, warpsPerCTA = [1, 1], instrShape = [16, 16, {instr_k}], isTransposed = false}}>
#lhs = #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {k_width}}}>
#rhs = #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {k_width}}}>
module attributes {{"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:{target.arch}", "ttg.threads-per-warp" = 64 : i32}} {{
  tt.func public @memory_forward(%out: !tt.ptr<f32>, %scratch: !tt.ptr<f32>) {{
    %a = arith.constant dense<1.0> : tensor<16x{instr_k}xbf16, #lhs>
    %b = arith.constant dense<1.0> : tensor<{instr_k}x16xbf16, #rhs>
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent" register_class "vgpr" initialize true
        : tensor<16x{instr_k}xbf16, #lhs>, tensor<{instr_k}x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    {commit}
    %pack = builtin.unrealized_conversion_cast {result} : tensor<16x16xf32, #mma> to !llvm.struct<(f32, f32, f32, f32)>
    %x = llvm.extractvalue %pack[0] : !llvm.struct<(f32, f32, f32, f32)>
    %count = llvm.mlir.constant(1 : i32) : i32
    %zero_f32 = llvm.mlir.constant(0.0 : f32) : f32
    %one_f32 = llvm.mlir.constant(1.0 : f32) : f32
    %lane = rocdl.workitem.id.x : i32
    {pointer_ir}
    llvm.store %x, %slot : f32, {pointer_type}
    %reloaded = llvm.load %slot : {pointer_type} -> f32
    {overwrite}
    {read}
    %ptr = builtin.unrealized_conversion_cast %out : !tt.ptr<f32> to !llvm.ptr<1>
    %dst = llvm.getelementptr %ptr[%lane] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    llvm.store %read, %dst : f32, !llvm.ptr<1>
    tt.return
  }}
}}
"""


@pytest.mark.parametrize("target,version,instr_k,k_width,expected_wait", [(GFX942, 3, 16, 4, "s_nop 10"),
                                                                          (GFX950, 4, 32, 8, "s_nop 11")],
                         ids=["gfx942", "gfx950"])
@pytest.mark.parametrize("memory,consumer", [("private", "opaque"), ("private_cast", "opaque"), ("global", "opaque"),
                                             ("private", "native"), ("private", "commit")])
def test_amd_scheduled_mfma_memory_forward_completion(tmp_path, target, version, instr_k, k_width, expected_wait,
                                                      memory, consumer):
    source = tmp_path / "memory_forward.ttgir"
    source.write_text(_amd_scheduled_mfma_memory_forward_ir(target, version, instr_k, k_width, memory, consumer))
    compiled = triton_compile(str(source), target=target, options={"num_warps": 1})
    llir = compiled.asm["llir"]
    # LLVM O3 must remove the store/load forwarding edge that hid the opaque
    # consumer during Triton lowering. A retained reload would miss the bug.
    assert not re.search(r"\balloca\b|\bload float\b", llir)
    stores = re.findall(r"^\s*store float [^\n]+", llir, re.MULTILINE)
    assert len(stores) == (2 if memory == "global" else 1)
    if memory == "global":
        assert any("store float 0.000000e+00" in store for store in stores)
    mfma = re.search(r"call <4 x float> @llvm\.amdgcn\.mfma\.f32\.16x16x" + str(instr_k), llir)
    assert mfma is not None
    assert not re.search(r'asm sideeffect "[^"\n]*v_mfma_', llir)
    waits = list(re.finditer(r'asm sideeffect "(s_nop \d+(?:\\0As_nop \d+)*)", "([^"\n]*)"', llir))
    if consumer == "native":
        assert "fadd float" in llir
        assert not re.search(r'asm(?: sideeffect)? "[^"\n]*s_nop', llir)
        return
    assert len(waits) == 1
    wait = waits[0]
    assert wait.group(1) == expected_wait
    opaque_consumer = re.search(r'asm sideeffect "[^"\n]*v_add_f32', llir)
    assert opaque_consumer is not None
    assert mfma.start() < wait.start() < opaque_consumer.start()
    if consumer == "commit":
        # A full explicit commit supplies the wait; the producer pin stays empty.
        assert wait.group(2) == "=a,0,~{memory}"
        assert 'asm sideeffect "", "=v,0"' in llir
    else:
        assert wait.group(2) == "=v,0"


def test_amd_scheduled_mfma_persistent_acc_native_commit_gfx950():
    compiled = compile_for_gfx950(
        _amd_scheduled_mfma_persistent_acc_kernel,
        signature={
            "a_ptr": "*fp16",
            "b_ptr": "*fp16",
            "output_ptr": "*fp32",
            "USE_VGPR": "constexpr",
            "COMMIT": "constexpr",
            "INITIALIZE": "constexpr",
            "FULL_K": "constexpr",
        },
        constexprs={"USE_VGPR": False, "COMMIT": True, "INITIALIZE": True, "FULL_K": False},
    )
    llir = compiled.asm["llir"]
    marker = "; triton_amd_scheduled_mfma\\0A"
    assert marker not in llir
    assert "@llvm.amdgcn.mfma.f32.16x16x32.f16" in llir
    assert 'asm sideeffect "s_nop 3\\0Av_mfma' not in llir
    # CDNA4 16x16x32 has 8 passes, so its result-read drain is 8 + 3 + 1.
    assert llir.count('asm sideeffect "s_nop 11"') == 1


def test_amd_scheduled_mfma_split_commit_shares_anchored_wait_gfx950():
    compiled = compile_for_gfx950(
        _amd_scheduled_mfma_split_commit_kernel,
        signature={
            "a_ptr": "*bf16",
            "k_ptr": "*bf16",
            "p_ptr": "*bf16",
            "do_ptr": "*bf16",
            "v_ptr": "*bf16",
            "output_ptr": "*fp32",
            "REVERSE_COMMITS": "constexpr",
            "INTERVENING_MFMA": "constexpr",
        },
        constexprs={"REVERSE_COMMITS": False, "INTERVENING_MFMA": False},
    )
    llir = compiled.asm["llir"]
    assert not re.search(r'asm sideeffect "[^"\n]*v_mfma_', llir)
    assert len(re.findall(r'call <16 x float> @llvm\.amdgcn\.mfma\.f32\.32x32x16\.bf16\(', llir)) == 2
    assert 'asm sideeffect "", "=a,0"' in llir
    assert 'asm sideeffect "", "=v,0"' in llir

    # Carry the complete second tuple through the first full delay. Its native
    # producers cannot sink past completion, and its register classes stay fixed.
    assert llir.count('asm sideeffect "s_nop 15\\0As_nop 3"') == 1
    combined = re.search(
        r'(%[\w.]+) = (?:tail )?call .* asm sideeffect "s_nop 15\\0As_nop 3", '
        r'"=a,=v,=a,0,1,2,~\{memory\}"', llir)
    assert combined is not None
    suffix = []
    for index in (1, 2):
        extracted = re.search(rf'(%[\w.]+) = extractvalue .* {re.escape(combined.group(1))}, {index}\b', llir)
        assert extracted is not None
        suffix.append(extracted.group(1))
    second = re.search(
        r'asm sideeffect "", "=v,=a,0,1,~\{memory\}"'
        rf'\(<16 x i32> {re.escape(suffix[0])}, <4 x i32> {re.escape(suffix[1])}\)', llir)
    assert second is not None
    assert combined.start() < second.start()


@pytest.mark.parametrize(
    ("reverse_commits", "intervening_mfma"),
    [
        pytest.param(True, False, id="reversed"),
        pytest.param(False, True, id="intervening-mfma"),
    ],
)
def test_amd_scheduled_mfma_split_commit_retains_waits_when_reordered_gfx950(
    reverse_commits,
    intervening_mfma,
):
    compiled = compile_for_gfx950(
        _amd_scheduled_mfma_split_commit_kernel,
        signature={
            "a_ptr": "*bf16",
            "k_ptr": "*bf16",
            "p_ptr": "*bf16",
            "do_ptr": "*bf16",
            "v_ptr": "*bf16",
            "output_ptr": "*fp32",
            "REVERSE_COMMITS": "constexpr",
            "INTERVENING_MFMA": "constexpr",
        },
        constexprs={
            "REVERSE_COMMITS": reverse_commits,
            "INTERVENING_MFMA": intervening_mfma,
        },
    )
    llir = compiled.asm["llir"]
    assert llir.count('asm sideeffect "s_nop 15\\0As_nop 3"') == 2
    assert 'asm sideeffect "s_nop 15\\0As_nop 3", "=a,0,~{memory}"' in llir
    assert 'asm sideeffect "s_nop 15\\0As_nop 3", "=v,=a,0,1,~{memory}"' in llir
    assert 'asm sideeffect "", "=v,=a,0,1,~{memory}"' not in llir


def test_amd_scheduled_mfma_short_first_split_commit_keeps_native_waits_gfx950():
    compiled = compile_for_gfx950(
        _amd_scheduled_mfma_short_first_split_commit_kernel,
        signature={
            "short_a_ptr": "*bf16",
            "short_b_ptr": "*bf16",
            "long_a_ptr": "*bf16",
            "long_b_ptr": "*bf16",
            "live_ptr": "*bf16",
            "short_output_ptr": "*fp32",
            "long_output_ptr": "*fp32",
        },
        constexprs={},
    )
    llir = compiled.asm["llir"]
    assert llir.count('asm sideeffect "s_nop 11"') == 1
    assert 'asm sideeffect "s_nop 11", "=a,0,~{memory}"' in llir
    # The later 32x32 result needs 20 states even after a 12-state commit.
    assert 'asm sideeffect "s_nop 15\\0As_nop 3", "=v,=a,0,1,~{memory}"' in llir
    assert 'asm sideeffect "", "=v,=a,0,1,~{memory}"' not in llir


@triton.jit
def _amd_scheduled_mfma_bypassed_commit_kernel(a_ptr, b_ptr, output_ptr, take_commit):
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
    a = tlx.require_layout(
        tl.load(a_ptr + rows[:, None] * 32 + reduction[None, :]),
        dot0,
        pin=False,
    )
    b = tlx.require_layout(
        tl.load(b_ptr + reduction[:, None] * 64 + cols[None, :]),
        dot1,
        pin=False,
    )
    acc = tlx.zeros((16, 64), tl.float32, layout=mma)
    acc = tlx.amd_scheduled_mfma(
        a,
        b,
        acc,
        accumulator_role="persistent",
        initialize=True,
    )
    if take_commit != 0:
        committed = tlx.amd_mfma_commit(acc)
        output_offsets = tlx.require_layout(
            output_ptr + rows[:, None] * 64 + cols[None, :],
            mma,
            pin=False,
        )
        tl.store(output_offsets, committed)


def test_amd_scheduled_mfma_bypassed_commit_is_native_gfx950():
    compiled = compile_for_gfx950(
        _amd_scheduled_mfma_bypassed_commit_kernel,
        signature={
            "a_ptr": "*bf16",
            "b_ptr": "*bf16",
            "output_ptr": "*fp32",
            "take_commit": "i32",
        },
        constexprs={},
    )
    assert "scf.if" in compiled.asm["ttgir"]
    llir = compiled.asm["llir"]
    assert "; triton_amd_scheduled_mfma\\0A" not in llir
    assert 'asm sideeffect "s_nop 3\\0Av_mfma' not in llir
    assert "@llvm.amdgcn.mfma.f32.16x16x32.bf16" in llir


@triton.jit
def _amd_scheduled_mfma_unproven_chain_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    SECOND_TRANSIENT: tl.constexpr,
    SECOND_VGPR: tl.constexpr,
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
        initialize=True,
    )
    a1 = tlx.extract_slice(a, [16, 32], [0, 32])
    b1 = tlx.extract_slice(b, [32, 64], [32, 0])
    acc = tlx.amd_scheduled_mfma(
        a1,
        b1,
        acc,
        accumulator_role=("transient" if SECOND_TRANSIENT else "persistent"),
        accumulator_register_class="vgpr" if SECOND_VGPR else None,
    )
    acc = tlx.amd_mfma_commit(acc)
    output_offsets = output_ptr + rows[:, None] * 64 + cols[None, :]
    output_offsets = tlx.require_layout(output_offsets, mma, pin=False)
    tl.store(output_offsets, acc)


@pytest.mark.parametrize(
    "second_transient,second_vgpr",
    [
        (True, False),
        (False, True),
    ],
)
def test_amd_scheduled_mfma_mixed_chains_are_native_gfx950(second_transient, second_vgpr):
    signature = {
        "a_ptr": "*bf16",
        "b_ptr": "*bf16",
        "output_ptr": "*fp32",
        "SECOND_TRANSIENT": "constexpr",
        "SECOND_VGPR": "constexpr",
    }
    constexprs = {
        "SECOND_TRANSIENT": second_transient,
        "SECOND_VGPR": second_vgpr,
    }
    compiled = compile_for_gfx950(
        _amd_scheduled_mfma_unproven_chain_kernel,
        signature=signature,
        constexprs=constexprs,
    )
    llir = compiled.asm["llir"]
    assert "; triton_amd_scheduled_mfma\\0A" not in llir
    assert 'asm sideeffect "s_nop 3\\0Av_mfma' not in llir
    assert "@llvm.amdgcn.mfma.f32.16x16x32.bf16" in llir


@triton.jit
def _amd_scheduled_mfma_forked_chain_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
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
    reduction = tl.arange(0, 128)
    cols = tl.arange(0, 64)
    a = tlx.require_layout(
        tl.load(a_ptr + rows[:, None] * 128 + reduction[None, :]),
        dot0,
        pin=False,
    )
    b = tlx.require_layout(
        tl.load(b_ptr + reduction[:, None] * 64 + cols[None, :]),
        dot1,
        pin=False,
    )
    a0 = tlx.extract_slice(a, [16, 32], [0, 0])
    a1 = tlx.extract_slice(a, [16, 32], [0, 32])
    a2 = tlx.extract_slice(a, [16, 32], [0, 64])
    b0 = tlx.extract_slice(b, [32, 64], [0, 0])
    b1 = tlx.extract_slice(b, [32, 64], [32, 0])
    b2 = tlx.extract_slice(b, [32, 64], [64, 0])
    zero = tlx.zeros((16, 64), tl.float32, layout=mma)
    root = tlx.amd_scheduled_mfma(
        a0,
        b0,
        zero,
        accumulator_role="persistent",
        initialize=True,
    )
    left = tlx.amd_scheduled_mfma(
        a1,
        b1,
        root,
        accumulator_role="persistent",
    )
    right = tlx.amd_scheduled_mfma(
        a2,
        b2,
        root,
        accumulator_role="persistent",
    )
    left, right = tlx.amd_mfma_commit((left, right))
    output_offsets = tlx.require_layout(
        output_ptr + rows[:, None] * 64 + cols[None, :],
        mma,
        pin=False,
    )
    tl.store(output_offsets, left)
    tl.store(output_offsets + 16 * 64, right)


def test_amd_scheduled_mfma_forked_chain_is_native_gfx950():
    compiled = compile_for_gfx950(
        _amd_scheduled_mfma_forked_chain_kernel,
        signature={
            "a_ptr": "*bf16",
            "b_ptr": "*bf16",
            "output_ptr": "*fp32",
        },
        constexprs={},
    )
    llir = compiled.asm["llir"]
    # All three arithmetic operations stay visible to LLVM across the fork.
    assert "; triton_amd_scheduled_mfma\\0A" not in llir
    assert len(re.findall(r"call <4 x float> @llvm\.amdgcn\.mfma\.f32\.16x16x32\.bf16\(", llir)) == 3
    assert 'asm sideeffect "s_nop 3\\0Av_mfma' not in llir
    assert "@llvm.amdgcn.mfma.f32.16x16x32.bf16" in llir
    assert llir.count('asm sideeffect "s_nop 11"') == 1


@triton.jit
def _amd_scheduled_mfma_lds_loop_kernel(a_ptr, b_ptr, output_ptr, iterations, OPERAND_ORIGIN: tl.constexpr):
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
    a_local = tlx.local_alloc((16, 32), tl.bfloat16, 1)
    b_local = tlx.local_alloc((32, 64), tl.bfloat16, 1)
    tlx.local_store(tlx.local_view(a_local, 0), a)
    tlx.local_store(tlx.local_view(b_local, 0), b)
    tl.debug_barrier()
    a = tlx.local_load(tlx.local_view(a_local, 0), layout=dot0)
    b = tlx.local_load(tlx.local_view(b_local, 0), layout=dot1)
    if OPERAND_ORIGIN == "computed":
        a = a + a
    elif OPERAND_ORIGIN == "resident":
        a = tlx.amd_register_resident(a, register_class="vgpr", registers_per_group=4)
    acc = tlx.zeros((16, 64), tl.float32, layout=mma)
    for _ in tl.range(0, iterations, num_stages=1):
        acc = tlx.amd_scheduled_mfma(
            a,
            b,
            acc,
            accumulator_role="persistent",
        )
    acc = tlx.amd_scheduled_mfma(
        a,
        b,
        acc,
        accumulator_role="persistent",
    )
    acc = tlx.amd_mfma_commit(acc)
    output_offsets = output_ptr + rows[:, None] * 64 + cols[None, :]
    output_offsets = tlx.require_layout(output_offsets, mma, pin=False)
    tl.store(output_offsets, acc)


@pytest.mark.parametrize("operand_origin", ["shared", "computed", "resident"])
def test_amd_scheduled_mfma_lds_loop_is_native_gfx950(operand_origin):
    compiled = compile_for_gfx950(
        _amd_scheduled_mfma_lds_loop_kernel,
        signature={
            "a_ptr": "*bf16",
            "b_ptr": "*bf16",
            "output_ptr": "*fp32",
            "iterations": "i32",
            "OPERAND_ORIGIN": "constexpr",
        },
        constexprs={"OPERAND_ORIGIN": operand_origin},
    )
    assert "scf.for" in compiled.asm["ttgir"]
    llir = compiled.asm["llir"]
    assert "; triton_amd_scheduled_mfma\\0A" not in llir
    assert "@llvm.amdgcn.mfma.f32.16x16x32.bf16" in llir
    assert 'asm sideeffect "s_nop 3\\0Av_mfma' not in llir
    class_pins = re.findall(r'asm( sideeffect)? "", "=[av],0"\(', llir)
    assert class_pins
    if operand_origin == "shared":
        assert all(class_pins), "shared-load operand pins must be ordered"
    else:
        assert not all(class_pins), "computed or explicitly resident inputs must keep pure operand pins"
        # C/D placement is AGPR-only here, so any ordered empty VGPR pin
        # would incorrectly promote an A/B operand of this excluded pair.
        assert not re.search(r'asm sideeffect "", "=v,0"\(', llir)
    assert "ttg.amdg.scheduled_mfma_operand_pin" not in llir
    # CDNA4 16x16x32 has 8 passes, so its result-read drain is 8 + 3 + 1.
    assert llir.count('asm sideeffect "s_nop 11"') == 1

