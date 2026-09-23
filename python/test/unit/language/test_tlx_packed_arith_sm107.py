"""Compile-only coverage for Rubin TLX packed x4 arithmetic."""

import pytest

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.compiler.compiler import make_backend

_RESULT_SIZE = tl.constexpr(512)
_RESULT_LAYOUT = tlx.layout(
    shape=((32, 4), (4, )),
    stride=((4, 128), (1, )),
)
_PACKED_FP4_LAYOUT = tlx.layout(
    shape=((32, 4), (2, )),
    stride=((2, 64), (1, )),
)


@triton.jit
def _packed_fp8_kernel(a_ptr, b_ptr, c_ptr, output_ptr, OP: tl.constexpr, DTYPE: tl.constexpr,
                       RESULT_LAYOUT: tl.constexpr):
    offsets = tlx.require_layout(tl.arange(0, _RESULT_SIZE), RESULT_LAYOUT)
    a = tlx.require_layout(tl.load(a_ptr + offsets), RESULT_LAYOUT)
    b = tlx.require_layout(tl.load(b_ptr + offsets), RESULT_LAYOUT)
    c = tlx.require_layout(tl.load(c_ptr + offsets), RESULT_LAYOUT)
    if OP == "add":
        result = tlx.add4(a, b, dtype=DTYPE)
    elif OP == "sub":
        result = tlx.sub4(a, b, dtype=DTYPE)
    elif OP == "mul":
        result = tlx.mul4(a, b, dtype=DTYPE)
    else:
        result = tlx.fma4(a, b, c, dtype=DTYPE)
    tl.store(output_ptr + offsets, result)


@triton.jit
def _packed_fp4_kernel(fp4_a_ptr, fp4_b_ptr, fp8_ptr, output_ptr, OP: tl.constexpr, DTYPE: tl.constexpr,
                       FP4_SIZE: tl.constexpr, RESULT_LAYOUT: tl.constexpr, FP4_LAYOUT: tl.constexpr):
    offsets = tlx.require_layout(tl.arange(0, _RESULT_SIZE), RESULT_LAYOUT)
    packed_offsets = tlx.require_layout(tl.arange(0, FP4_SIZE), FP4_LAYOUT)
    fp4_a = tlx.require_layout(tl.load(fp4_a_ptr + packed_offsets), FP4_LAYOUT)
    fp4_b = tlx.require_layout(tl.load(fp4_b_ptr + packed_offsets), FP4_LAYOUT)
    fp8 = tlx.require_layout(tl.load(fp8_ptr + offsets), RESULT_LAYOUT)
    if OP == "add":
        result = tlx.add4(fp4_a, fp8, dtype=DTYPE)
    elif OP == "sub":
        result = tlx.sub4(fp4_a, fp8, dtype=DTYPE)
    elif OP == "mul":
        result = tlx.mul4(fp4_a, fp8, dtype=DTYPE)
    elif OP == "fma":
        result = tlx.fma4(fp4_a, fp8, fp8, dtype=DTYPE)
    else:
        result = tlx.mul4(fp4_a, fp4_b, dtype=DTYPE)
    tl.store(output_ptr + offsets, result)


@triton.jit
def _packed_fp4_missing_dtype_kernel(fp4_a_ptr, fp4_b_ptr, FP4_LAYOUT: tl.constexpr):
    packed_offsets = tlx.require_layout(tl.arange(0, 256), FP4_LAYOUT)
    fp4_a = tlx.require_layout(tl.load(fp4_a_ptr + packed_offsets), FP4_LAYOUT)
    fp4_b = tlx.require_layout(tl.load(fp4_b_ptr + packed_offsets), FP4_LAYOUT)
    tlx.mul4(fp4_a, fp4_b)


def _compile_to_ptx(kernel, signature, constexprs, capability=107):
    # The bundled Blackwell ptxas does not recognize sm_107a, so run the normal
    # compiler pipeline through PTX and stop before cubin assembly.
    source = ASTSource(fn=kernel, signature=signature, constexprs=constexprs)
    target = GPUTarget("cuda", capability, 32)
    backend = make_backend(target)
    options = backend.parse_options({"num_warps": 4, **source.parse_options()})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    module = source.make_ir(
        target,
        options,
        backend.get_codegen_implementation(options),
        backend.get_module_map(),
        context,
    )
    metadata = {"target": target, **options.__dict__}
    stages = {}
    backend.add_stages(stages, options, source.language)
    artifacts = {}
    for stage_name, compile_stage in stages.items():
        module = compile_stage(module, metadata)
        artifacts[stage_name] = module if isinstance(module, str) else str(module)
        if stage_name == "ptx":
            break
    return artifacts


@pytest.mark.parametrize(
    "operation,signature,result_dtype,ptx_instruction",
    [
        (
            "add",
            {"a_ptr": "*fp8e5", "b_ptr": "*fp8e4nv", "c_ptr": "*fp8e4nv", "output_ptr": "*fp8e4nv"},
            None,
            "add.e4m3x4.e5m2x4",
        ),
        (
            "sub",
            {"a_ptr": "*fp8e5", "b_ptr": "*fp8e4nv", "c_ptr": "*fp8e4nv", "output_ptr": "*fp8e4nv"},
            tl.float8e4nv,
            "sub.e4m3x4.e5m2x4",
        ),
        (
            "mul",
            {"a_ptr": "*fp8e5", "b_ptr": "*fp8e4nv", "c_ptr": "*fp8e4nv", "output_ptr": "*fp8e4nv"},
            tl.float8e4nv,
            "mul.e4m3x4.e5m2x4.e4m3x4",
        ),
        (
            "fma",
            {"a_ptr": "*fp8e4nv", "b_ptr": "*fp8e5", "c_ptr": "*fp8e5", "output_ptr": "*fp8e5"},
            tl.float8e5,
            "fma.e5m2x4.e4m3x4.e5m2x4",
        ),
    ],
)
def test_packed_fp8_x4_lowers_to_rubin_ptx(operation, signature, result_dtype, ptx_instruction):
    compiled = _compile_to_ptx(
        _packed_fp8_kernel,
        signature,
        {"OP": operation, "DTYPE": result_dtype, "RESULT_LAYOUT": _RESULT_LAYOUT},
    )
    assert f"ttng.packed_arith {operation}" in compiled["ttgir"]
    assert ptx_instruction in compiled["ptx"]


@pytest.mark.parametrize(
    "operation,ptx_instruction",
    [
        ("add", "add.e4m3x4.e2m1x4"),
        ("sub", "sub.e4m3x4.e2m1x4"),
        ("mul", "mul.e4m3x4.e2m1x4.e4m3x4"),
        ("fma", "fma.e4m3x4.e2m1x4.e4m3x4"),
        ("pure_fp4", "mul.e4m3x4.e2m1x4.e2m1x4"),
    ],
)
def test_packed_fp4_x4_lowers_to_rubin_ptx(operation, ptx_instruction):
    compiled = _compile_to_ptx(
        _packed_fp4_kernel,
        {"fp4_a_ptr": "*i8", "fp4_b_ptr": "*i8", "fp8_ptr": "*fp8e4nv", "output_ptr": "*fp8e4nv"},
        {
            "OP": operation,
            "DTYPE": tl.float8e4nv,
            "FP4_SIZE": 256,
            "RESULT_LAYOUT": _RESULT_LAYOUT,
            "FP4_LAYOUT": _PACKED_FP4_LAYOUT,
        },
    )
    expected_operation = "mul" if operation == "pure_fp4" else operation
    assert f"ttng.packed_arith {expected_operation}" in compiled["ttgir"]
    assert ptx_instruction in compiled["ptx"]


def test_packed_x4_rejects_blackwell():
    signature = {
        "a_ptr": "*fp8e4nv",
        "b_ptr": "*fp8e4nv",
        "c_ptr": "*fp8e4nv",
        "output_ptr": "*fp8e4nv",
    }
    with pytest.raises(triton.CompilationError, match="requires compute capability >= 107"):
        _compile_to_ptx(
            _packed_fp8_kernel,
            signature,
            {"OP": "add", "DTYPE": tl.float8e4nv, "RESULT_LAYOUT": _RESULT_LAYOUT},
            capability=100,
        )


def test_packed_x4_rejects_unsupported_operand_type():
    signature = {
        "a_ptr": "*fp16",
        "b_ptr": "*fp8e4nv",
        "c_ptr": "*fp8e4nv",
        "output_ptr": "*fp8e4nv",
    }
    with pytest.raises(triton.CompilationError, match="only supports FP8 E4M3/E5M2 or packed FP4"):
        _compile_to_ptx(
            _packed_fp8_kernel,
            signature,
            {"OP": "add", "DTYPE": tl.float8e4nv, "RESULT_LAYOUT": _RESULT_LAYOUT},
        )


def test_packed_x4_rejects_unsupported_result_type():
    signature = {
        "a_ptr": "*fp8e4nv",
        "b_ptr": "*fp8e4nv",
        "c_ptr": "*fp8e4nv",
        "output_ptr": "*fp16",
    }
    with pytest.raises(triton.CompilationError, match="result must be FP8 E4M3 or E5M2"):
        _compile_to_ptx(
            _packed_fp8_kernel,
            signature,
            {"OP": "add", "DTYPE": tl.float16, "RESULT_LAYOUT": _RESULT_LAYOUT},
        )


def test_packed_x4_rejects_mismatched_addend_type():
    signature = {
        "a_ptr": "*fp8e4nv",
        "b_ptr": "*fp8e4nv",
        "c_ptr": "*fp8e4nv",
        "output_ptr": "*fp8e5",
    }
    with pytest.raises(triton.CompilationError, match="second operand type to match result type"):
        _compile_to_ptx(
            _packed_fp8_kernel,
            signature,
            {"OP": "add", "DTYPE": tl.float8e5, "RESULT_LAYOUT": _RESULT_LAYOUT},
        )


def test_packed_x4_requires_result_type_for_pure_fp4():
    with pytest.raises(triton.CompilationError, match="packed FP4 operands require an explicit FP8 result dtype"):
        _compile_to_ptx(
            _packed_fp4_missing_dtype_kernel,
            {"fp4_a_ptr": "*i8", "fp4_b_ptr": "*i8"},
            {"FP4_LAYOUT": _PACKED_FP4_LAYOUT},
        )


def test_packed_x4_rejects_fp4_shape_mismatch():
    with pytest.raises(triton.CompilationError, match="with exactly one dimension halved"):
        _compile_to_ptx(
            _packed_fp4_kernel,
            {"fp4_a_ptr": "*i8", "fp4_b_ptr": "*i8", "fp8_ptr": "*fp8e4nv", "output_ptr": "*fp8e4nv"},
            {
                "OP": "add",
                "DTYPE": tl.float8e4nv,
                "FP4_SIZE": 512,
                "RESULT_LAYOUT": _RESULT_LAYOUT,
                "FP4_LAYOUT": _RESULT_LAYOUT,
            },
        )
