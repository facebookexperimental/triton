"""Tests for Level 0 launch metadata schema generation.

Validates that the Triton compiler emits a versioned, machine-readable
launch metadata JSON alongside the cubin, and that the schema fields
are consistent with the existing metadata bag.
"""

import json

import pytest
import triton
import triton.language as tl
from triton.compiler.compiler import ASTSource, compile as triton_compile


@triton.jit
def add_kernel(X, Y, OUT, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    y = tl.load(Y + offs, mask=mask)
    tl.store(OUT + offs, x + y, mask=mask)


@triton.jit
def kernel_with_constant(X, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    tl.store(X + offs, x + 1, mask=mask)


def _compile_kernel(fn, signature, constexprs=None, attrs=None):
    """Helper to compile a kernel and return the CompiledKernel."""
    target = triton.runtime.driver.active.get_current_target()
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs, attrs=attrs)
    return triton_compile(src, target=target)


@pytest.mark.parametrize("dtype", ["*fp32"])
def test_launch_metadata_exists(dtype):
    """asm['launch_metadata'] should exist and be valid JSON."""
    compiled = _compile_kernel(
        add_kernel,
        signature={"X": dtype, "Y": dtype, "OUT": dtype, "N": "i32"},
        constexprs={"BLOCK": 1024},
    )
    assert "launch_metadata" in compiled.asm, "launch_metadata not found in asm dict"
    schema = json.loads(compiled.asm["launch_metadata"])
    assert isinstance(schema, dict)


def test_abi_version():
    """abi_version should be 1."""
    compiled = _compile_kernel(
        add_kernel,
        signature={"X": "*fp32", "Y": "*fp32", "OUT": "*fp32", "N": "i32"},
        constexprs={"BLOCK": 1024},
    )
    schema = compiled.launch_metadata_schema
    assert schema is not None
    assert schema["abi_version"] == 1


def test_entry_name_matches():
    """entry_name in schema should match the kernel name from ptx."""
    compiled = _compile_kernel(
        add_kernel,
        signature={"X": "*fp32", "Y": "*fp32", "OUT": "*fp32", "N": "i32"},
        constexprs={"BLOCK": 1024},
    )
    schema = compiled.launch_metadata_schema
    assert schema["entry_name"] == compiled.metadata.name


def test_launch_fields_match_metadata():
    """Launch-critical fields should match the existing metadata."""
    compiled = _compile_kernel(
        add_kernel,
        signature={"X": "*fp32", "Y": "*fp32", "OUT": "*fp32", "N": "i32"},
        constexprs={"BLOCK": 1024},
    )
    schema = compiled.launch_metadata_schema
    md = compiled.metadata

    assert schema["num_warps"] == md.num_warps
    assert schema["num_ctas"] == md.num_ctas
    assert schema["shared_mem"] == md.shared
    assert schema["launch_cooperative_grid"] == md.launch_cooperative_grid
    assert schema["launch_pdl"] == md.launch_pdl
    assert schema["global_scratch_size"] == md.global_scratch_size
    assert schema["global_scratch_align"] == md.global_scratch_align
    assert schema["profile_scratch_size"] == md.profile_scratch_size
    assert schema["profile_scratch_align"] == md.profile_scratch_align


def test_constants_excluded_from_args():
    """Compile-time constants (constexprs) should appear in 'constants', not 'args'."""
    compiled = _compile_kernel(
        add_kernel,
        signature={"X": "*fp32", "Y": "*fp32", "OUT": "*fp32", "N": "i32"},
        constexprs={"BLOCK": 1024},
    )
    schema = compiled.launch_metadata_schema

    arg_names = [a["name"] for a in schema["args"]]
    assert "BLOCK" not in arg_names, "constexpr BLOCK should not be in args"
    assert len(schema["constants"]) > 0, "constants dict should not be empty"

    # The runtime args should be X, Y, OUT, N
    assert "X" in arg_names
    assert "Y" in arg_names
    assert "OUT" in arg_names
    assert "N" in arg_names


def test_args_types():
    """Each arg should have correct type information."""
    compiled = _compile_kernel(
        add_kernel,
        signature={"X": "*fp32", "Y": "*fp32", "OUT": "*fp32", "N": "i32"},
        constexprs={"BLOCK": 1024},
    )
    schema = compiled.launch_metadata_schema
    args_by_name = {a["name"]: a for a in schema["args"]}

    assert args_by_name["X"]["type"] == "*fp32"
    assert args_by_name["Y"]["type"] == "*fp32"
    assert args_by_name["OUT"]["type"] == "*fp32"
    assert args_by_name["N"]["type"] == "i32"


def test_args_have_index():
    """Each arg should have a positional index."""
    compiled = _compile_kernel(
        add_kernel,
        signature={"X": "*fp32", "Y": "*fp32", "OUT": "*fp32", "N": "i32"},
        constexprs={"BLOCK": 1024},
    )
    schema = compiled.launch_metadata_schema
    for arg in schema["args"]:
        assert "index" in arg, f"arg {arg['name']} missing index"
        assert isinstance(arg["index"], int)


def test_pointer_divisibility():
    """Pointer args with divisibility hints should have divisible_by in schema."""
    compiled = _compile_kernel(
        add_kernel,
        signature={"X": "*fp32", "Y": "*fp32", "OUT": "*fp32", "N": "i32"},
        constexprs={"BLOCK": 1024},
        attrs={(0, ): [("tt.divisibility", 16)], (1, ): [("tt.divisibility", 16)], (2, ): [("tt.divisibility", 16)]},
    )
    schema = compiled.launch_metadata_schema
    args_by_name = {a["name"]: a for a in schema["args"]}

    assert args_by_name["X"].get("divisible_by") == 16
    assert args_by_name["Y"].get("divisible_by") == 16
    assert args_by_name["OUT"].get("divisible_by") == 16
    # N is a scalar, should not have divisible_by
    assert "divisible_by" not in args_by_name["N"]


def test_schema_required_fields():
    """All required fields should be present in the schema."""
    compiled = _compile_kernel(
        add_kernel,
        signature={"X": "*fp32", "Y": "*fp32", "OUT": "*fp32", "N": "i32"},
        constexprs={"BLOCK": 1024},
    )
    schema = compiled.launch_metadata_schema
    required_fields = [
        "abi_version",
        "entry_name",
        "num_warps",
        "num_ctas",
        "shared_mem",
        "cluster_dims",
        "preferred_cluster_dims",
        "launch_cooperative_grid",
        "launch_pdl",
        "global_scratch_size",
        "global_scratch_align",
        "profile_scratch_size",
        "profile_scratch_align",
        "tmem_size",
        "args",
        "constants",
        "tensordesc_meta",
    ]
    for field in required_fields:
        assert field in schema, f"Missing required field: {field}"


def test_cluster_dims_is_list():
    """cluster_dims and preferred_cluster_dims should be JSON-serializable lists."""
    compiled = _compile_kernel(
        add_kernel,
        signature={"X": "*fp32", "Y": "*fp32", "OUT": "*fp32", "N": "i32"},
        constexprs={"BLOCK": 1024},
    )
    schema = compiled.launch_metadata_schema
    assert isinstance(schema["cluster_dims"], list)
    assert isinstance(schema["preferred_cluster_dims"], list)
    assert len(schema["cluster_dims"]) == 3
    assert len(schema["preferred_cluster_dims"]) == 3


def test_launch_metadata_schema_property():
    """CompiledKernel.launch_metadata_schema should return parsed dict."""
    compiled = _compile_kernel(
        add_kernel,
        signature={"X": "*fp32", "Y": "*fp32", "OUT": "*fp32", "N": "i32"},
        constexprs={"BLOCK": 1024},
    )
    assert compiled.launch_metadata_schema is not None
    assert isinstance(compiled.launch_metadata_schema, dict)
    assert compiled.launch_metadata_schema["abi_version"] == 1


# =========================================================================
# Level 1: Standalone launcher source (asm["launcher_src"])
# =========================================================================


def test_launcher_src_exists():
    """asm['launcher_src'] should exist and be a non-empty string."""
    compiled = _compile_kernel(
        add_kernel,
        signature={"X": "*fp32", "Y": "*fp32", "OUT": "*fp32", "N": "i32"},
        constexprs={"BLOCK": 1024},
    )
    assert "launcher_src" in compiled.asm, "launcher_src not found in asm dict"
    src = compiled.asm["launcher_src"]
    assert isinstance(src, str)
    assert len(src) > 0


def test_launcher_src_includes_launch_h():
    """Generated C source should include triton/runtime/launch.h."""
    compiled = _compile_kernel(
        add_kernel,
        signature={"X": "*fp32", "Y": "*fp32", "OUT": "*fp32", "N": "i32"},
        constexprs={"BLOCK": 1024},
    )
    src = compiled.asm["launcher_src"]
    assert '#include "triton/runtime/launch.h"' in src


def test_launcher_src_no_python_h():
    """Generated C source must NOT depend on Python.h."""
    compiled = _compile_kernel(
        add_kernel,
        signature={"X": "*fp32", "Y": "*fp32", "OUT": "*fp32", "N": "i32"},
        constexprs={"BLOCK": 1024},
    )
    src = compiled.asm["launcher_src"]
    assert "Python.h" not in src
    assert "PyObject" not in src
    assert "PyArg_ParseTuple" not in src


def test_launcher_src_has_launch_function():
    """Generated C source should contain a triton_launch_<kernel> function."""
    compiled = _compile_kernel(
        add_kernel,
        signature={"X": "*fp32", "Y": "*fp32", "OUT": "*fp32", "N": "i32"},
        constexprs={"BLOCK": 1024},
    )
    src = compiled.asm["launcher_src"]
    assert "CUresult triton_launch_" in src


def test_launcher_src_has_args_struct():
    """Generated C source should define a typed args struct."""
    compiled = _compile_kernel(
        add_kernel,
        signature={"X": "*fp32", "Y": "*fp32", "OUT": "*fp32", "N": "i32"},
        constexprs={"BLOCK": 1024},
    )
    src = compiled.asm["launcher_src"]
    assert "_args_t;" in src
    assert "CUdeviceptr X;" in src
    assert "CUdeviceptr Y;" in src
    assert "CUdeviceptr OUT;" in src
    assert "int32_t N;" in src


def test_launcher_src_bakes_constants():
    """Compile-time constants (num_warps, shared_mem) should be baked in."""
    compiled = _compile_kernel(
        add_kernel,
        signature={"X": "*fp32", "Y": "*fp32", "OUT": "*fp32", "N": "i32"},
        constexprs={"BLOCK": 1024},
    )
    src = compiled.asm["launcher_src"]
    md = compiled.metadata
    assert f"/*num_warps=*/{md.num_warps}" in src
    assert f"/*shared_mem=*/{md.shared}u" in src


def test_launcher_src_has_abi_version_comment():
    """Generated source should contain the ABI version as a comment."""
    compiled = _compile_kernel(
        add_kernel,
        signature={"X": "*fp32", "Y": "*fp32", "OUT": "*fp32", "N": "i32"},
        constexprs={"BLOCK": 1024},
    )
    src = compiled.asm["launcher_src"]
    assert "ABI version: 1" in src
