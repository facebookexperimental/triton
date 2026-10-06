"""Test AMD packed f32 instruction selection.

Compile-only tests; no target hardware required.
Compiles attn_fwd.ttir and checks that:
  - LLVM IR contains <2 x float> fmul/fsub/fadd (from packed conversion + VectorCombine)
  - gfx1250 ASM contains v_pk_fma_f32
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

GFX950_TARGET = GPUTarget("hip", "gfx950", 64)
GFX1250_TARGET = GPUTarget("hip", "gfx1250", 32)
TTIR_PATH = str(Path(__file__).parent / "attn_fwd.ttir")


@lru_cache(maxsize=None)
def compile_for_target(target):
    return triton.compile(TTIR_PATH, target=target)


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
def gfx1250_kernel():
    return compile_for_target(GFX1250_TARGET)


@pytest.mark.parametrize("intrinsic", ["mfma", "smfmac", "wmma"])
def test_gfx950_matrix_intrinsic_packed_f32_option(intrinsic):
    llvm_ir = f"call void @llvm.amdgcn.{intrinsic}.test()"

    assert amd_compiler.get_amdgpu_codegen_features("gfx950", llvm_ir) == ""
    assert amd_compiler.get_amdgpu_codegen_features("gfx950", llvm_ir, True) == "-packed-fp32-ops"
    assert amd_compiler.get_amdgpu_codegen_features("gfx942", llvm_ir, True) == ""
    assert amd_compiler.get_amdgpu_codegen_features("gfx1100", llvm_ir, True) == "-real-true16"
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
    triton.compile(TTIR_PATH, target=GFX1250_TARGET)

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


def test_gfx1250_packed_f32_in_llir(gfx1250_kernel):
    """GFX1250 should produce packed <2 x float> fmul/fsub in LLVM IR."""
    llir = gfx1250_kernel.asm["llir"]
    func_body = get_func_body_llir(llir)

    packed_fop = re.compile(r"f(mul|sub|add) <2 x float>")
    assert packed_fop.search(func_body), ("Expected packed <2 x float> fmul/fsub/fadd in LLVM IR for gfx1250")

    # Both fmul and fsub must exist for ISel FMA contraction
    assert re.search(r"fmul.*<2 x float>", func_body), ("Expected packed fmul <2 x float> in LLVM IR")
    assert re.search(r"fsub.*<2 x float>", func_body), ("Expected packed fsub <2 x float> in LLVM IR")


def test_gfx1250_v_pk_fma_f32_in_asm(gfx1250_kernel):
    """GFX1250 ASM should contain v_pk_fma_f32 from ISel contraction."""
    amdgcn = gfx1250_kernel.asm["amdgcn"]
    func_body = get_func_body_asm(amdgcn)
    counts = count_asm_instructions(func_body)

    assert counts["v_pk_fma_f32"] > 100, (f"Expected a substantial number of v_pk_fma_f32 instructions, got "
                                          f"{counts['v_pk_fma_f32']}")
    assert counts["v_fma_f32"] < 20, (f"Expected scalar v_fma_f32 instructions to stay low, got "
                                      f"{counts['v_fma_f32']}")
