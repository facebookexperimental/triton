"""Test AMD packed f32 instruction selection.

Compile-only tests; no target hardware required.
Compiles small Gluon kernels for CDNA5 and checks that:
  - LLVM IR contains <2 x float> fmul/fsub/fadd (from packed conversion)
  - ASM contains v_pk_fma_f32 (from ISel contraction of packed fmul+fsub)
  - gl.fma lowers to packed FMA instructions
Compiles attn_fwd.ttir and checks that:
  - LLVM IR contains <2 x float> fmul/fsub/fadd (from packed conversion + VectorCombine)
  - CDNA5 ASM contains v_pk_fma_f32
  - gfx950 can suppress packed f32 arithmetic per kernel
"""

import re
from functools import lru_cache
from pathlib import Path

import pytest
import triton
from triton._C.libtriton import llvm
from triton.backends.amd import compiler as amd_compiler
from triton.backends.compiler import GPUTarget
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

GFX950_TARGET = GPUTarget("hip", "gfx950", 64)
CDNA5_TARGET = GPUTarget("hip", "gfx1250", 32)
TTIR_PATH = str(Path(__file__).parent / "attn_fwd.ttir")
BLOCK = 1024
# Each thread owns 8 elements, so a fully packed elementwise op lowers to 4
# packed instructions per thread.
NUM_PACKED = 4


@gluon.jit
def gluon_binary_kernel(x_ptr, y_ptr, out_ptr, OP: gl.constexpr, BLOCK: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([8], [32], [4], [0])
    offs = gl.arange(0, BLOCK, layout)
    x = gl.load(x_ptr + offs)
    y = gl.load(y_ptr + offs)
    if OP == "add":
        out = x + y
    elif OP == "sub":
        out = x - y
    else:
        out = x * y
    gl.store(out_ptr + offs, out)


@gluon.jit
def gluon_mul_sub_kernel(x_ptr, y_ptr, z_ptr, out_ptr, BLOCK: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([8], [32], [4], [0])
    offs = gl.arange(0, BLOCK, layout)
    x = gl.load(x_ptr + offs)
    y = gl.load(y_ptr + offs)
    z = gl.load(z_ptr + offs)
    gl.store(out_ptr + offs, x * y - z)


@gluon.jit
def gluon_fma_kernel(x_ptr, y_ptr, z_ptr, out_ptr, BLOCK: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([8], [32], [4], [0])
    offs = gl.arange(0, BLOCK, layout)
    x = gl.load(x_ptr + offs)
    y = gl.load(y_ptr + offs)
    z = gl.load(z_ptr + offs)
    gl.store(out_ptr + offs, gl.fma(x, y, z))


def compile_gluon(kernel, signature, constexprs, options=None, aligned=True):
    # With aligned pointers the loads are vectorized, as in real kernels.
    # Otherwise LLVM scalarizes packed fadd/fsub/fmul whose operands come from
    # scalar loads.
    attrs = {}
    if aligned:
        attrs = {(i, ): [["tt.divisibility", 16]] for i, ty in enumerate(signature.values()) if ty.startswith("*")}
    src = gluon.GluonASTSource(kernel, signature, constexprs, attrs)
    return triton.compile(src, target=CDNA5_TARGET, options=options)


@lru_cache(maxsize=None)
def compile_for_target(target):
    return triton.compile(TTIR_PATH, target=target, options={"enable_fp_fusion": True})


def get_func_body_llir(llir):
    func_body = re.findall(
        r"define amdgpu_kernel void .*? \{(.* ret void.*?)}",
        llir,
        flags=re.DOTALL,
    )
    assert len(func_body) >= 1, "couldn't find kernel body in LLVM IR"
    return func_body[0]


def get_func_body_asm(amdgcn, kernel_name="attn_fwd"):
    pattern = rf"^{kernel_name}:(.*); -- End function"
    body = re.findall(pattern, amdgcn, flags=re.DOTALL | re.MULTILINE)
    assert len(body) >= 1, f"couldn't find {kernel_name} body in asm"
    return body[0]


def count_asm_instructions(func_body):
    return {
        "v_pk_fma_f32": func_body.count("v_pk_fma_f32"),
        "v_pk_mul_f32": func_body.count("v_pk_mul_f32"),
        "v_pk_add_f32": func_body.count("v_pk_add_f32"),
        "v_fma_f32": func_body.count("v_fma_f32"),
        "s_fmac_f32": func_body.count("s_fmac_f32"),
    }


@pytest.fixture(scope="module")
def cdna5_kernel():
    return compile_for_target(CDNA5_TARGET)


# There is no v_pk_sub_f32; packed fsub selects v_pk_add_f32 with a neg modifier.
@pytest.mark.parametrize("op, packed_inst", [("add", "v_pk_add_f32"), ("sub", "v_pk_add_f32"), ("mul", "v_pk_mul_f32")])
def test_cdna5_packed_f32_arith(op, packed_inst):
    """f32 fadd/fsub/fmul on CDNA5 should lower to packed <2 x float> ops."""
    kernel = compile_gluon(
        gluon_binary_kernel,
        {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "OP": "constexpr", "BLOCK": "constexpr"},
        {"OP": op, "BLOCK": BLOCK},
    )
    llir = get_func_body_llir(kernel.asm["llir"])
    assert len(re.findall(rf"\bf{op} (?:\w+ )*<2 x float>", llir)) == NUM_PACKED, llir
    assert not re.search(rf"\bf{op} (?:\w+ )*float ", llir), llir

    asm = get_func_body_asm(kernel.asm["amdgcn"], "gluon_binary_kernel")
    assert asm.count(packed_inst) == NUM_PACKED, asm


def test_cdna5_packed_mul_sub_contracts_to_v_pk_fma_f32():
    """Packed fmul+fsub should be contracted into v_pk_fma_f32 when fp fusion is enabled."""
    kernel = compile_gluon(
        gluon_mul_sub_kernel,
        {"x_ptr": "*fp32", "y_ptr": "*fp32", "z_ptr": "*fp32", "out_ptr": "*fp32", "BLOCK": "constexpr"},
        {"BLOCK": BLOCK},
        options={"enable_fp_fusion": True},
    )
    llir = get_func_body_llir(kernel.asm["llir"])
    assert re.search(r"\bfmul (?:\w+ )*<2 x float>", llir), llir
    assert re.search(r"\bfsub (?:\w+ )*<2 x float>", llir), llir

    asm = get_func_body_asm(kernel.asm["amdgcn"], "gluon_mul_sub_kernel")
    assert asm.count("v_pk_fma_f32") == NUM_PACKED, asm
    assert "v_fma_f32" not in asm
    assert "v_dual_fma" not in asm


@pytest.mark.parametrize("intrinsic", ["mfma", "smfmac", "wmma"])
def test_gfx950_matrix_intrinsic_packed_f32_option(intrinsic):
    llvm_ir = f"call void @llvm.amdgcn.{intrinsic}.test()"

    assert amd_compiler.get_amdgpu_codegen_features("gfx950", llvm_ir) == ""
    assert amd_compiler.get_amdgpu_codegen_features("gfx950", llvm_ir, True) == "-packed-fp32-ops"
    assert amd_compiler.get_amdgpu_codegen_features("gfx942", llvm_ir, True) == ""
    assert amd_compiler.get_amdgpu_codegen_features("gfx1100", llvm_ir, True) == ""
    assert amd_compiler.get_amdgpu_codegen_features("gfx950", "fadd float %a, %b", True) == ""


def test_vector_combine_policy_reaches_llvm(monkeypatch):
    optimize_module = llvm.optimize_module
    observed = []

    def record_optimize_module(*args, **kwargs):
        observed.append(kwargs["disable_vector_combine"])
        return optimize_module(*args, **kwargs)

    monkeypatch.setattr(llvm, "optimize_module", record_optimize_module)
    monkeypatch.setattr(triton.knobs.compilation, "always_compile", True)
    triton.compile(TTIR_PATH, target=GFX950_TARGET)
    triton.compile(TTIR_PATH, target=GFX950_TARGET, options={"disable_vector_combine": True})
    triton.compile(TTIR_PATH, target=CDNA5_TARGET)

    assert observed == [False, True, True]
    baseline = amd_compiler.HIPOptions(arch="gfx950")
    disabled = amd_compiler.HIPOptions(arch="gfx950", disable_vector_combine=True)
    assert baseline.hash() != disabled.hash()


def test_gfx950_packed_f32_suppression_is_per_kernel():
    baseline = triton.compile(TTIR_PATH, target=GFX950_TARGET)
    suppressed = triton.compile(TTIR_PATH, target=GFX950_TARGET, options={"disable_packed_fp32_ops": True})
    baseline_body = get_func_body_asm(baseline.asm["amdgcn"])
    suppressed_body = get_func_body_asm(suppressed.asm["amdgcn"])
    packed_f32 = re.compile(r"\bv_pk_(?:add|sub|mul|fma)_f32\b")

    assert "v_mfma" in baseline_body
    assert packed_f32.search(baseline_body)
    assert not packed_f32.search(suppressed_body)
    assert baseline.metadata.disable_packed_fp32_ops is False
    assert suppressed.metadata.disable_packed_fp32_ops is True


def test_cdna5_packed_f32_in_llir(cdna5_kernel):
    """CDNA5 should produce packed <2 x float> fmul/fsub in LLVM IR."""
    llir = cdna5_kernel.asm["llir"]
    func_body = get_func_body_llir(llir)

    packed_fop = re.compile(r"f(mul|sub|add)(?:\s+[a-z]+)* <2 x float>")
    assert packed_fop.search(func_body), ("Expected packed <2 x float> fmul/fsub/fadd in LLVM IR for CDNA5")

    # Both fmul and fsub must exist for ISel FMA contraction
    assert re.search(r"fmul.*<2 x float>", func_body), ("Expected packed fmul <2 x float> in LLVM IR")
    assert re.search(r"fsub.*<2 x float>", func_body), ("Expected packed fsub <2 x float> in LLVM IR")


def test_cdna5_v_pk_fma_f32_in_asm(cdna5_kernel):
    """CDNA5 ASM should contain v_pk_fma_f32 from ISel contraction."""
    amdgcn = cdna5_kernel.asm["amdgcn"]
    func_body = get_func_body_asm(amdgcn)
    counts = count_asm_instructions(func_body)

    assert counts["v_pk_fma_f32"] > 100, (f"Expected a substantial number of v_pk_fma_f32 instructions, got "
                                          f"{counts['v_pk_fma_f32']}")
    assert counts["v_fma_f32"] < 20, (f"Expected scalar v_fma_f32 instructions to stay low, got "
                                      f"{counts['v_fma_f32']}")


@pytest.mark.parametrize("dtype, packed_inst", [("fp32", "v_pk_fma_f32"), ("bf16", "v_pk_fma_bf16")])
def test_cdna5_gl_fma_packed(dtype, packed_inst):
    """gl.fma on CDNA5 should lower to packed FMA instructions."""
    ptr_ty = f"*{dtype}"
    kernel = compile_gluon(
        gluon_fma_kernel,
        {"x_ptr": ptr_ty, "y_ptr": ptr_ty, "z_ptr": ptr_ty, "out_ptr": ptr_ty, "BLOCK": "constexpr"},
        {"BLOCK": BLOCK},
        # With vectorized loads LLVM's SLP vectorizer packs scalar FMAs on its
        # own. Use scalar loads so the packing must come from the lowering.
        aligned=False,
    )
    asm = get_func_body_asm(kernel.asm["amdgcn"], "gluon_fma_kernel")
    assert asm.count(packed_inst) == NUM_PACKED, asm
    assert "v_fma_f32" not in asm
    assert "v_dual_fma" not in asm
