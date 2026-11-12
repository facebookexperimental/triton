import triton
import triton.language as tl

import torch
import math
import pytest

_BLOCK_SIZE = 16


@triton.jit
def add_helper(x, y):
    return x + y


@triton.jit
def add_kernel(
    in_ptr0,
    in_ptr1,
    n_elements,
    out_ptr,
    BLOCK_SIZE: "tl.constexpr",
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    y = tl.load(in_ptr1 + offsets, mask=mask)
    x2d = x[None, :]
    x1d = tl.reshape(x2d, [BLOCK_SIZE])
    output = add_helper(x1d, y)
    tl.store(out_ptr + offsets, output, mask=mask)


def test_module_walk(device):
    """
    Test the MLIR bindings exposed for the out-of-tree walk.
    """

    def walk_fn(op):
        name = op.get_name()
        for i in range(op.get_num_results()):
            op.get_result(i).id()
        for i in range(op.get_num_operands()):
            op.get_operand(i).id()
        for i in range(op.get_num_regions()):
            op.get_region(i).id()
        block = op.get_block()
        if block is not None:
            block.id()
            for i in range(block.get_num_arguments()):
                block.get_argument(i)
        if name == "tt.func":
            op.get_str_attr("sym_name")
        if name == "tt.call":
            op.get_flat_symbol_ref_attr("callee")
        if name == "tt.make_range":
            assert 0 == op.get_int_attr("start")
            assert _BLOCK_SIZE == op.get_int_attr("end")
        if name == "arith.constant":
            val = op.get_constant_value()
            assert isinstance(val, int)
        if name == "tt.expand_dims":
            shape = op.get_result(0).get_shape()
            assert shape == [1, _BLOCK_SIZE]
        if name == "tt.reshape":
            in_shape = op.get_operand(0).get_shape()
            out_shape = op.get_result(0).get_shape()
            assert in_shape == [1, _BLOCK_SIZE]
            assert out_shape == [_BLOCK_SIZE]

    kernel = add_kernel
    args = [
        torch.empty((32, 32), device=device),  # in_ptr0
        torch.empty((32, 32), device=device),  # in_ptr1
        1024,  # n_elements
        torch.empty((32, 32), device=device),  # out_ptr
        _BLOCK_SIZE,  # BLOCK_SIZE
    ]
    target = triton.runtime.driver.active.get_current_target()
    backend = triton.compiler.compiler.make_backend(target)
    src = triton.compiler.compiler.ASTSource(
        fn=kernel,
        signature={kernel.arg_names[i]: triton.runtime.jit.mangle_type(arg)
                   for i, arg in enumerate(args)},
        constexprs={kernel.arg_names[i]: arg
                    for i, arg in enumerate(args)
                    if not isinstance(arg, torch.Tensor)},
    )

    context = triton._C.libtriton.ir.context()
    options = backend.parse_options(dict())
    codegen_fns = dict()
    module_map = backend.get_module_map()
    triton._C.libtriton.ir.load_dialects(context)
    backend.load_dialects(context)

    ttir_module = src.make_ir(target, options, codegen_fns, module_map, context)
    ttir_module.walk(walk_fn)


def test_python_func_in_visit_call(device):

    @triton.jit
    def test_py_call_const_kernel(
        in_ptr0,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        log2e: tl.constexpr = math.log2(math.e)
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        output = x * log2e
        tl.store(out_ptr + offsets, output, mask=mask)

    x = torch.randn(4, device=device)
    out = torch.zeros_like(x)
    test_py_call_const_kernel[(4, )](x, out, 4, 4)


@pytest.mark.parametrize("scalarize", [False, True])
def test_scalarize_packed_fops_llvm_pipeline(tmp_path, scalarize):
    from triton._C.libtriton import ir, llvm

    # NVIDIA requests this pass explicitly: its module does not yet have a
    # target triple when optimize_module runs. Other backends retain the default.
    source = tmp_path / "fma.mlir"
    source.write_text("""
module {
  llvm.func @fma_lane(%a: vector<2xf32>, %b: vector<2xf32>, %c: vector<2xf32>) -> f32 {
    %v = llvm.call_intrinsic "llvm.fma"(%a, %b, %c) : (vector<2xf32>, vector<2xf32>, vector<2xf32>) -> vector<2xf32>
    %index = llvm.mlir.constant(1 : i32) : i32
    %r = llvm.extractelement %v[%index : i32] : vector<2xf32>
    llvm.return %r : f32
  }
}
""")
    context = ir.context()
    ir.load_dialects(context)
    module = ir.parse_mlir_module(str(source), context)
    llvm_context = llvm.context()
    llvm_module = llvm.to_module(module, llvm_context)
    options = {"scalarize_packed_fops": True} if scalarize else {}
    llvm.optimize_module(llvm_module, llvm.OPTIMIZE_O3, **options)
    expected = "call float @llvm.fma.f32" if scalarize else "call <2 x float> @llvm.fma.v2f32"
    assert expected in str(llvm_module)


@pytest.mark.parametrize("disable_slp", [False, True])
@pytest.mark.parametrize("disable_slp_env", [False, True])
def test_disable_slp_vectorizer(tmp_path, monkeypatch, disable_slp, disable_slp_env):
    from triton._C.libtriton import ir, llvm

    monkeypatch.setenv("TRITON_DISABLE_SLPVECTORIZATION", str(int(disable_slp_env)))
    lines = [
        "module {",
        "llvm.func @products(%x: !llvm.ptr, %y: !llvm.ptr, %out: !llvm.ptr) -> f32 {",
    ]
    for i in range(4):
        lines += [
            f"%px{i} = llvm.getelementptr %x[{i}] : (!llvm.ptr) -> !llvm.ptr, f32",
            f"%py{i} = llvm.getelementptr %y[{i}] : (!llvm.ptr) -> !llvm.ptr, f32",
            f"%a{i} = llvm.load %px{i} : !llvm.ptr -> f32",
            f"%b{i} = llvm.load %py{i} : !llvm.ptr -> f32",
            f"%v{i} = llvm.fmul %a{i}, %b{i} : f32",
        ]
    for i in range(4):
        lines += [
            f"%po{i} = llvm.getelementptr %out[{i}] : (!llvm.ptr) -> !llvm.ptr, f32",
            f"llvm.store %v{i}, %po{i} : f32, !llvm.ptr",
        ]
    lines += [
        "%r0 = llvm.fadd %v2, %v3 : f32",
        "%r1 = llvm.fadd %r0, %v1 : f32",
        "%r2 = llvm.fadd %r1, %v0 : f32",
        "llvm.return %r2 : f32", "}", "}",
    ]
    source = tmp_path / "products.mlir"
    source.write_text("\n".join(lines))
    context = ir.context()
    ir.load_dialects(context)
    module = ir.parse_mlir_module(str(source), context)
    llvm_context = llvm.context()
    llvm_module = llvm.to_module(module, llvm_context)
    llvm.optimize_module(llvm_module, llvm.OPTIMIZE_O3, disable_slp_vectorizer=disable_slp)
    assert ("fmul <4 x float>" in str(llvm_module)) == (not disable_slp and not disable_slp_env)
