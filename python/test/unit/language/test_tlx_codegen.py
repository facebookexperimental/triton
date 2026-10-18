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
from triton.backends.amd import amdgc_hazard_repair
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


def test_scheduled_mfma_hazard_repair_accepts_vgpr_result():
    assembly = """kernel:
; triton_amd_scheduled_mfma
v_mfma_f32_32x32x16_bf16 v[0:15], v[32:35], a[0:3], 0
v_add_f32_e32 v16, v0, v17
"""

    repaired = amdgc_hazard_repair.insert_scheduled_mfma_hazard_nops(assembly, "gfx950")

    assert "s_nop 15\ns_nop 3\nv_add_f32_e32 v16, v0, v17" in repaired


@pytest.mark.parametrize(
    "existing_wait,expected_wait",
    [
        pytest.param("", "s_nop 1\n", id="zero-wait-states"),
        pytest.param("s_nop 0\n", "s_nop 0\n", id="one-wait-state"),
        pytest.param("s_nop 1\n", "", id="two-wait-states"),
    ],
)
def test_scheduled_mfma_hazard_repair_delays_tied_vgpr_accumulator(existing_wait, expected_wait):
    # Legacy VALU writes need two wait states before every explicit MFMA VGPR
    # use, including srcC (LLVM GCNHazardRecognizer::checkMAIHazards90A).
    marker = "; triton_amd_scheduled_mfma\n"
    assembly = f"""kernel:
s_nop 3
v_mov_b32_e32 v0, 1
{existing_wait}{marker}v_mfma_f32_16x16x32_f16 v[0:3], v[8:11], a[4:7], v[0:3]
"""

    repaired = amdgc_hazard_repair.insert_scheduled_mfma_hazard_nops(assembly, "gfx950")

    assert repaired == assembly.replace(marker, marker + expected_wait)


@pytest.mark.parametrize("register_class", ["v", "a"])
def test_scheduled_mfma_hazard_repair_allows_full_accumulator_forwarding(register_class):
    accumulator = f"{register_class}[0:3]"
    assembly = f"""kernel:
s_nop 3
; triton_amd_scheduled_mfma
v_mfma_f32_16x16x32_f16 {accumulator}, v[8:11], a[4:7], 0
; triton_amd_scheduled_mfma
v_mfma_f32_16x16x32_f16 {accumulator}, v[8:11], a[4:7], {accumulator}
"""

    repaired = amdgc_hazard_repair.insert_scheduled_mfma_hazard_nops(assembly, "gfx950")

    assert repaired == assembly


def test_scheduled_mfma_hazard_repair_drains_vgpr_result_overwrite():
    assembly = """kernel:
s_nop 3
; triton_amd_scheduled_mfma
v_mfma_f32_32x32x16_bf16 v[0:15], v[32:35], a[0:3], 0
v_add_f32_e32 v0, v16, v17
"""

    repaired = amdgc_hazard_repair.insert_scheduled_mfma_hazard_nops(assembly, "gfx950")

    assert "s_nop 15\ns_nop 3\nv_add_f32_e32 v0, v16, v17" in repaired


def test_scheduled_mfma_hazard_repair_allows_unrelated_vgpr_definition():
    assembly = """kernel:
s_nop 3
; triton_amd_scheduled_mfma
v_mfma_f32_32x32x16_bf16 v[0:15], v[32:35], a[0:3], 0
v_add_f32_e32 v20, v16, v17
"""

    repaired = amdgc_hazard_repair.insert_scheduled_mfma_hazard_nops(assembly, "gfx950")

    assert repaired == assembly


def test_scheduled_mfma_hazard_repair_merges_mixed_results_through_diamond():
    assembly = """kernel:
s_cbranch_scc1 .LBB0_2
.LBB0_1:
s_nop 3
; triton_amd_scheduled_mfma
v_mfma_f32_16x16x32_f16 a[0:3], v[24:27], v[28:31], 0
s_branch .LBB0_3
.LBB0_2:
s_nop 3
; triton_amd_scheduled_mfma
v_mfma_f32_32x32x16_f16 v[8:23], v[32:35], a[4:7], 0
s_branch .LBB0_3
.LBB0_3:
v_accvgpr_read_b32 v24, a0
v_mov_b32_e32 v8, v25
"""

    repaired = amdgc_hazard_repair.insert_scheduled_mfma_hazard_nops(assembly, "gfx950")

    assert ".LBB0_3:\ns_nop 10\nv_accvgpr_read_b32 v24, a0\ns_nop 6\nv_mov_b32_e32 v8, v25" in repaired


def test_scheduled_mfma_hazard_repair_propagates_mixed_results_through_loop_backedge():
    assembly = """kernel:
s_branch .LBB0_2
.LBB0_1:
v_accvgpr_read_b32 v24, a0
v_mov_b32_e32 v8, v25
.LBB0_2:
s_nop 3
; triton_amd_scheduled_mfma
v_mfma_f32_16x16x32_f16 a[0:3], v[24:27], v[28:31], 0
; triton_amd_scheduled_mfma
v_mfma_f32_32x32x16_f16 v[8:23], v[32:35], a[4:7], 0
s_branch .LBB0_1
"""

    repaired = amdgc_hazard_repair.insert_scheduled_mfma_hazard_nops(assembly, "gfx950")

    assert ".LBB0_1:\ns_nop 9\nv_accvgpr_read_b32 v24, a0\ns_nop 7\nv_mov_b32_e32 v8, v25" in repaired


@pytest.mark.parametrize(
    ("destination", "accumulator"),
    [
        ("v[0:15]", "a[0:15]"),
        ("a[0:15]", "v[0:15]"),
    ],
)
def test_scheduled_mfma_hazard_repair_rejects_mixed_destination_accumulator_classes(destination, accumulator):
    assembly = f"""kernel:
; triton_amd_scheduled_mfma
v_mfma_f32_32x32x16_bf16 {destination}, v[32:35], a[0:3], {accumulator}
"""

    with pytest.raises(ValueError, match="marked MFMA srcC must be tied to its destination"):
        amdgc_hazard_repair.insert_scheduled_mfma_hazard_nops(assembly, "gfx950")


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


def test_amd_scheduled_mfma_split_commit_defers_vgpr_chain_drain_gfx950():
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
    marker = "; triton_amd_scheduled_mfma\\0A"
    mfmas = [line for line in llir.splitlines() if marker in line]
    assert len(mfmas) == 2
    assert '"=a,v,v"' in mfmas[0]
    assert '"=&v,v,v"' in mfmas[1]
    assert "s_nop 3\\0Av_mfma" not in mfmas[0]
    assert "s_nop 3\\0Av_mfma" not in mfmas[1]

    # The first full dK drain also covers the older dV result. The second
    # side-effecting boundary keeps its tied VGPR/AGPR allocation constraints
    # without emitting another native wait instruction.
    assert llir.count('asm sideeffect "s_nop 15\\0As_nop 3"') == 1
    assert 'asm sideeffect "s_nop 15\\0As_nop 3", "=a,0,~{memory}"' in llir
    assert 'asm sideeffect "s_nop 5", "=v,=a,0,1,~{memory}"' not in llir
    assert 'asm sideeffect "", "=v,=a,0,1,~{memory}"' in llir


@pytest.mark.parametrize(
    ("reverse_commits", "intervening_mfma"),
    [
        pytest.param(True, False, id="reversed"),
        pytest.param(False, True, id="intervening-mfma"),
    ],
)
def test_amd_scheduled_mfma_split_commit_requires_adjacent_full_drain_first_gfx950(
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
    assert llir.count('asm sideeffect "s_nop 15\\0As_nop 3"') == 1
    assert 'asm sideeffect "s_nop 15\\0As_nop 3", "=a,0,~{memory}"' in llir
    assert 'asm sideeffect "s_nop 5", "=v,=a,0,1,~{memory}"' in llir
    assert 'asm sideeffect "", "=v,=a,0,1,~{memory}"' not in llir


def test_amd_scheduled_mfma_split_commit_rejects_short_first_drain_gfx950():
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
    assert 'asm sideeffect "s_nop 5", "=v,=a,0,1,~{memory}"' in llir
    assert 'asm sideeffect "", "=v,=a,0,1,~{memory}"' not in llir


def test_amd_scheduled_mfma_vgpr_results_only_commit_is_conservative_gfx950():
    compiled = compile_for_gfx950(
        _amd_scheduled_mfma_persistent_acc_kernel,
        signature={
            "a_ptr": "*bf16",
            "b_ptr": "*bf16",
            "output_ptr": "*fp32",
            "USE_VGPR": "constexpr",
            "COMMIT": "constexpr",
        },
        constexprs={"USE_VGPR": True, "COMMIT": True},
    )
    llir = compiled.asm["llir"]
    assert "; triton_amd_scheduled_mfma\\0A" not in llir
    assert 'asm sideeffect "s_nop 3\\0Av_mfma' in llir


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
    if COMMIT:
        acc = tlx.amd_mfma_commit(acc)
    output_offsets = output_ptr + rows[:, None] * 64 + cols[None, :]
    output_offsets = tlx.require_layout(output_offsets, mma, pin=False)
    tl.store(output_offsets, acc)


@pytest.mark.parametrize(
    "prefix,expected_nop",
    [
        (
            "s_mov_b32 s0, 0\ns_mov_b32 s1, 0\ns_mov_b32 s2, 0\n"
            "v_mov_b32_e32 v0, 1",
            "s_nop 1",
        ),
        (
            "s_mov_b32 s0, 0\ns_mov_b32 s1, 0\n"
            "v_mov_b32_e32 v0, 1\n"
            "v_mfma_f32_16x16x32_f16 a[4:7], v[8:11], v[12:15], 0",
            "s_nop 0",
        ),
        (
            "s_mov_b32 s0, 0\n"
            "v_mov_b32_e32 v0, 1\n"
            "v_mfma_f32_16x16x32_f16 a[4:7], v[8:11], v[12:15], 0\n"
            "v_mfma_f32_16x16x32_f16 a[8:11], v[16:19], v[20:23], 0",
            None,
        ),
        ("v_cmpx_ne_u32_e32 vcc, v24, 0", "s_nop 3"),
        (
            "s_mov_b32 s0, 0\ns_mov_b32 s1, 0\ns_mov_b32 s2, 0\n"
            "v_accvgpr_write_b32 a4, v20",
            "s_nop 2",
        ),
        (
            "s_mov_b32 s0, 0\ns_mov_b32 s1, 0\ns_mov_b32 s2, 0\n"
            "v_accvgpr_write_b32 a0, v20",
            "s_nop 0",
        ),
        (
            "s_mov_b32 s0, 0\ns_mov_b32 s1, 0\ns_mov_b32 s2, 0\n"
            "v_accvgpr_write_b32 a0, v20\ns_nop 0",
            None,
        ),
        (
            "s_mov_b32 s0, 0\ns_mov_b32 s1, 0\ns_mov_b32 s2, 0\n"
            "v_swap_b32 v20, v0",
            "s_nop 1",
        ),
        (
            "s_mov_b32 s0, 0\ns_mov_b32 s1, 0\ns_mov_b32 s2, 0\n"
            "v_permlane16_swap_b32_e32 v20, v0",
            "s_nop 1",
        ),
        (
            "s_mov_b32 s0, 0\ns_mov_b32 s1, 0\ns_mov_b32 s2, 0\n"
            "v_permlane32_swap_b32_e32 v20, v0",
            "s_nop 1",
        ),
    ],
)
def test_scheduled_mfma_hazard_nop_insertion(prefix, expected_nop):
    marker = "; triton_amd_scheduled_mfma"
    source_a = "a[4:7]" if "a4" in prefix else "v[0:3]"
    source_c = "a[0:3]" if "a0" in prefix else "0"
    destination = source_c if source_c != "0" else "a[8:11]"
    mfma = ("v_mfma_f32_16x16x32_f16 "
            f"{destination}, {source_a}, v[4:7], {source_c}")
    assembly = f"kernel:\n{prefix}\n{marker}\n{mfma}\n"

    result = amdgc_hazard_repair.insert_scheduled_mfma_hazard_nops(assembly, "gfx950")

    if expected_nop is None:
        assert f"{marker}\n{mfma}" in result
    else:
        assert f"{marker}\n{expected_nop}\n{mfma}" in result


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


