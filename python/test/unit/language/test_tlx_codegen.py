import re
from triton.compiler.errors import CompilationError
import pytest
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource, compile as triton_compile

GFX950 = GPUTarget("hip", "gfx950", 64)


def compile_for_target(fn, signature, constexprs, target):
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs)
    return triton_compile(src, target=target)


def compile_for_gfx950(fn, signature, constexprs):
    """Compile a TLX kernel for gfx950 and return the compiled object."""
    return compile_for_target(fn, signature, constexprs, GFX950)

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


