"""SM103+ compile and runtime coverage for PTX 9.4 TLX packed x4 arithmetic."""

import pytest

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton._C.libtriton import ir, nvidia
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.compiler.compiler import make_backend

_RESULT_BYTES = 512
_RESULT_SIZE = tl.constexpr(_RESULT_BYTES)
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
        result = tlx.packed_add4(a, b, dtype=DTYPE)
    elif OP == "sub":
        result = tlx.packed_sub4(a, b, dtype=DTYPE)
    elif OP == "mul":
        result = tlx.packed_mul4(a, b, dtype=DTYPE)
    else:
        result = tlx.packed_fma4(a, b, c, dtype=DTYPE)
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
        result = tlx.packed_add4(fp4_a, fp8, dtype=DTYPE)
    elif OP == "sub":
        result = tlx.packed_sub4(fp4_a, fp8, dtype=DTYPE)
    elif OP == "mul":
        result = tlx.packed_mul4(fp4_a, fp8, dtype=DTYPE)
    elif OP == "fma":
        result = tlx.packed_fma4(fp4_a, fp8, fp8, dtype=DTYPE)
    else:
        result = tlx.packed_mul4(fp4_a, fp4_b, dtype=DTYPE)
    tl.store(output_ptr + offsets, result)


@triton.jit
def _packed_fp4_missing_dtype_kernel(fp4_a_ptr, fp4_b_ptr, FP4_LAYOUT: tl.constexpr):
    packed_offsets = tlx.require_layout(tl.arange(0, 256), FP4_LAYOUT)
    fp4_a = tlx.require_layout(tl.load(fp4_a_ptr + packed_offsets), FP4_LAYOUT)
    fp4_b = tlx.require_layout(tl.load(fp4_b_ptr + packed_offsets), FP4_LAYOUT)
    tlx.packed_mul4(fp4_a, fp4_b)


def _compile(kernel, signature, constexprs, capability=103, stop_after=None):
    # Exercise SM103 through cubin while allowing unsupported future targets to stop at PTX.
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
        artifacts[stage_name] = module if isinstance(module, (str, bytes)) else str(module)
        if stage_name == stop_after:
            break
    return artifacts


class _DevicePtr:

    def __init__(self, ptr, dtype):
        self.ptr = ptr
        self.dtype = dtype

    def data_ptr(self):
        return self.ptr


def _require_sm103_or_newer():
    try:
        device = triton.runtime.driver.active.get_current_device()
        triton.runtime.driver.active.set_current_device(device)
        target = triton.runtime.driver.active.get_current_target()
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    if target.backend != "cuda" or target.arch < 103:
        pytest.skip("requires an NVIDIA SM103 or newer GPU")


def _allocate_repeated_byte(value, size, dtype):
    ptr = nvidia.device_malloc(size)
    nvidia.copy_host_to_device(ptr, bytes([value]) * size)
    return ptr, _DevicePtr(ptr, dtype)


def _assert_repeated_byte(ptr, expected, size):
    actual = nvidia.copy_device_to_host(ptr, size)
    assert actual == bytes([expected]) * size


@pytest.mark.parametrize(
    "capability,stop_after,ptx_target",
    [(103, None, "sm_103a"), (107, "ptx", "sm_107a")],
)
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
def test_packed_fp8_x4_lowers_on_sm103up(capability, stop_after, ptx_target, operation, signature, result_dtype,
                                         ptx_instruction):
    compiled = _compile(
        _packed_fp8_kernel,
        signature,
        {"OP": operation, "DTYPE": result_dtype, "RESULT_LAYOUT": _RESULT_LAYOUT},
        capability=capability,
        stop_after=stop_after,
    )
    assert f"ttng.packed_arith {operation}" in compiled["ttgir"]
    assert ".version 9.4" in compiled["ptx"]
    assert f".target {ptx_target}" in compiled["ptx"]
    assert ptx_instruction in compiled["ptx"]
    if stop_after is None:
        assert compiled["cubin"]
    else:
        assert "cubin" not in compiled


@pytest.mark.parametrize(
    "capability,stop_after,ptx_target",
    [(103, None, "sm_103a"), (107, "ptx", "sm_107a")],
)
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
def test_packed_fp4_x4_lowers_on_sm103up(capability, stop_after, ptx_target, operation, ptx_instruction):
    compiled = _compile(
        _packed_fp4_kernel,
        {"fp4_a_ptr": "*i8", "fp4_b_ptr": "*i8", "fp8_ptr": "*fp8e4nv", "output_ptr": "*fp8e4nv"},
        {
            "OP": operation,
            "DTYPE": tl.float8e4nv,
            "FP4_SIZE": 256,
            "RESULT_LAYOUT": _RESULT_LAYOUT,
            "FP4_LAYOUT": _PACKED_FP4_LAYOUT,
        },
        capability=capability,
        stop_after=stop_after,
    )
    expected_operation = "mul" if operation == "pure_fp4" else operation
    assert f"ttng.packed_arith {expected_operation}" in compiled["ttgir"]
    assert ".version 9.4" in compiled["ptx"]
    assert f".target {ptx_target}" in compiled["ptx"]
    assert ptx_instruction in compiled["ptx"]
    if stop_after is None:
        assert compiled["cubin"]
    else:
        assert "cubin" not in compiled


@pytest.mark.parametrize(
    "operation,a_dtype,a_byte,b_dtype,b_byte,c_dtype,c_byte,output_dtype,result_dtype,expected_byte",
    [
        pytest.param("add", tl.float8e5, 0x3C, tl.float8e4nv, 0x38, tl.float8e4nv, 0x00, tl.float8e4nv, None, 0x40,
                     id="add-e5m2-e4m3"),
        pytest.param("sub", tl.float8e5, 0x40, tl.float8e4nv, 0x38, tl.float8e4nv, 0x00, tl.float8e4nv, tl.float8e4nv,
                     0x38, id="sub-e5m2-e4m3"),
        pytest.param("mul", tl.float8e5, 0x40, tl.float8e4nv, 0x40, tl.float8e4nv, 0x00, tl.float8e4nv, tl.float8e4nv,
                     0x48, id="mul-e5m2-e4m3"),
        pytest.param("fma", tl.float8e4nv, 0x38, tl.float8e5, 0x3C, tl.float8e5, 0x3C, tl.float8e5, tl.float8e5, 0x40,
                     id="fma-e4m3-e5m2"),
    ],
)
def test_packed_fp8_x4_executes_on_sm103up(operation, a_dtype, a_byte, b_dtype, b_byte, c_dtype, c_byte, output_dtype,
                                           result_dtype, expected_byte):
    _require_sm103_or_newer()
    allocations = []
    try:
        a_ptr, a = _allocate_repeated_byte(a_byte, _RESULT_BYTES, a_dtype)
        allocations.append(a_ptr)
        b_ptr, b = _allocate_repeated_byte(b_byte, _RESULT_BYTES, b_dtype)
        allocations.append(b_ptr)
        c_ptr, c = _allocate_repeated_byte(c_byte, _RESULT_BYTES, c_dtype)
        allocations.append(c_ptr)
        output_ptr, output = _allocate_repeated_byte(0, _RESULT_BYTES, output_dtype)
        allocations.append(output_ptr)
        _packed_fp8_kernel[(1, )](
            a,
            b,
            c,
            output,
            OP=operation,
            DTYPE=result_dtype,
            RESULT_LAYOUT=_RESULT_LAYOUT,
            num_warps=4,
        )
        nvidia.synchronize()
        _assert_repeated_byte(output_ptr, expected_byte, _RESULT_BYTES)
    finally:
        for ptr in reversed(allocations):
            nvidia.device_free(ptr)


@pytest.mark.parametrize(
    "operation,fp4_a_byte,fp4_b_byte,fp8_byte,expected_byte",
    [
        pytest.param("add", 0x22, 0x00, 0x38, 0x40, id="add-e2m1-e4m3"),
        pytest.param("sub", 0x44, 0x00, 0x38, 0x38, id="sub-e2m1-e4m3"),
        pytest.param("mul", 0x44, 0x00, 0x40, 0x48, id="mul-e2m1-e4m3"),
        pytest.param("fma", 0x22, 0x00, 0x38, 0x40, id="fma-e2m1-e4m3"),
        pytest.param("pure_fp4", 0x44, 0x44, 0x00, 0x48, id="mul-e2m1-e2m1"),
    ],
)
def test_packed_fp4_x4_executes_on_sm103up(operation, fp4_a_byte, fp4_b_byte, fp8_byte, expected_byte):
    _require_sm103_or_newer()
    allocations = []
    try:
        fp4_a_ptr, fp4_a = _allocate_repeated_byte(fp4_a_byte, _RESULT_BYTES // 2, tl.int8)
        allocations.append(fp4_a_ptr)
        fp4_b_ptr, fp4_b = _allocate_repeated_byte(fp4_b_byte, _RESULT_BYTES // 2, tl.int8)
        allocations.append(fp4_b_ptr)
        fp8_ptr, fp8 = _allocate_repeated_byte(fp8_byte, _RESULT_BYTES, tl.float8e4nv)
        allocations.append(fp8_ptr)
        output_ptr, output = _allocate_repeated_byte(0, _RESULT_BYTES, tl.float8e4nv)
        allocations.append(output_ptr)
        _packed_fp4_kernel[(1, )](
            fp4_a,
            fp4_b,
            fp8,
            output,
            OP=operation,
            DTYPE=tl.float8e4nv,
            FP4_SIZE=_RESULT_BYTES // 2,
            RESULT_LAYOUT=_RESULT_LAYOUT,
            FP4_LAYOUT=_PACKED_FP4_LAYOUT,
            num_warps=4,
        )
        nvidia.synchronize()
        _assert_repeated_byte(output_ptr, expected_byte, _RESULT_BYTES)
    finally:
        for ptr in reversed(allocations):
            nvidia.device_free(ptr)


def test_packed_x4_rejects_pre_sm103():
    signature = {
        "a_ptr": "*fp8e4nv",
        "b_ptr": "*fp8e4nv",
        "c_ptr": "*fp8e4nv",
        "output_ptr": "*fp8e4nv",
    }
    with pytest.raises(triton.CompilationError, match="requires compute capability >= 103"):
        _compile(
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
        _compile(
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
        _compile(
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
        _compile(
            _packed_fp8_kernel,
            signature,
            {"OP": "add", "DTYPE": tl.float8e5, "RESULT_LAYOUT": _RESULT_LAYOUT},
        )


def test_packed_x4_requires_result_type_for_pure_fp4():
    with pytest.raises(triton.CompilationError, match="packed FP4 operands require an explicit FP8 result dtype"):
        _compile(
            _packed_fp4_missing_dtype_kernel,
            {"fp4_a_ptr": "*i8", "fp4_b_ptr": "*i8"},
            {"FP4_LAYOUT": _PACKED_FP4_LAYOUT},
        )


def test_packed_x4_rejects_fp4_shape_mismatch():
    with pytest.raises(triton.CompilationError, match="with exactly one dimension halved"):
        _compile(
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
