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
    )

    with pytest.raises(ValueError, match="KV tensor size exceeds the signed 32-bit byte-offset range"):
        amd_fa_varlen_bwd._validate_i32_buffer_offsets(
            total_q=1,
            total_kv=max_tokens + 1,
            batch=1,
            q_heads=1,
            kv_heads=1,
        )
    with pytest.raises(ValueError, match="padded dQ size exceeds the signed 32-bit byte-offset range"):
        amd_fa_varlen_bwd._validate_i32_buffer_offsets(
            total_q=max_tokens - 14,
            total_kv=1,
            batch=1,
            q_heads=1,
            kv_heads=1,
        )



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


