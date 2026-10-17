import re
from triton.compiler.errors import CompilationError
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
    result = tlx.release_layout(result)
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
        constexprs={"CAST_BEFORE_RELEASE": False},
    )
    assert "amdgcn" in compiled.asm
    assert "scf.for" in compiled.asm["ttir"]
    assert "#ttg.amd_mfma" in compiled.asm["ttgir"]
    assert "#tlx.no_verify_layout" not in compiled.asm["ttgir"]


def test_release_layout_accepts_cast_helper_result_gfx950():
    module = make_ir_for_target(
        _concrete_dot_loop_helper_kernel,
        signature={
            "a_ptr": "*fp16",
            "b_ptr": "*fp16",
            "bias_ptr": "*fp32",
            "out_ptr": "*bf16",
        },
        constexprs={"CAST_BEFORE_RELEASE": True},
        target=GFX950,
    )
    ttir = str(module)
    assert "tt.call" in ttir
    assert "arith.addf" in ttir
    assert "arith.truncf" in ttir
    assert "tlx.release_layout" in ttir
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


