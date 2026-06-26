"""Tests for Level 0 launch metadata schema generation.

Validates that the Triton compiler emits a versioned, machine-readable
launch metadata JSON alongside the cubin, and that the schema fields
are consistent with the existing metadata bag.
"""

import json

import pytest
import torch
import triton
import triton.language as tl
from triton import knobs
from triton.backends.nvidia.driver import (
    build_kernel_signature_from_schema,
    expand_signature,
    make_kernel_signature,
)
from triton.compiler.compiler import ASTSource, compile as triton_compile, make_backend
from triton.knobs import HookChain


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
    """Generated C source should include nvidia/backend/launch.h."""
    compiled = _compile_kernel(
        add_kernel,
        signature={"X": "*fp32", "Y": "*fp32", "OUT": "*fp32", "N": "i32"},
        constexprs={"BLOCK": 1024},
    )
    src = compiled.asm["launcher_src"]
    assert '#include "nvidia/backend/launch.h"' in src


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
    assert f".num_warps = {md.num_warps}," in src
    assert f".shared_mem = {md.shared}u," in src
    # Param offsets use offsetof (C compiler owns layout, not Python).
    assert "offsetof(" in src


def test_launcher_src_emits_cluster_dims():
    """The descriptor should carry the explicit multi-dim cluster_dims so the
    shared core can build CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION for the
    ctas_per_cga path (converged onto triton_launch_kernel, not a separate
    launcher)."""
    compiled = _compile_kernel(
        add_kernel,
        signature={"X": "*fp32", "Y": "*fp32", "OUT": "*fp32", "N": "i32"},
        constexprs={"BLOCK": 1024},
    )
    src = compiled.asm["launcher_src"]
    assert ".cluster_dims = {" in src
    # Sanity: still emits the 1-D num_ctas path inputs too (both are present;
    # the core gives num_ctas priority over cluster_dims).
    assert ".num_ctas = " in src


def test_launcher_src_has_abi_version_comment():
    """Generated source should contain the ABI version as a comment."""
    compiled = _compile_kernel(
        add_kernel,
        signature={"X": "*fp32", "Y": "*fp32", "OUT": "*fp32", "N": "i32"},
        constexprs={"BLOCK": 1024},
    )
    src = compiled.asm["launcher_src"]
    assert "ABI version: 1" in src


def test_launcher_src_compiles_with_gcc():
    """Generated C should compile with the system C compiler (validates struct
    layout, offsetof usage, and launch.h compatibility)."""
    import os
    import shutil
    import subprocess
    import tempfile
    import triton.backends.nvidia

    @triton.jit
    def _gcc_test_add(X, Y, OUT, N, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        tl.store(OUT + offs, tl.load(X + offs, mask=mask) + tl.load(Y + offs, mask=mask), mask=mask)

    # Force a fresh triton compilation cache so we always run make_launcher_src
    # (the disk cache keys on source hash + options, not compiler.py version).
    with tempfile.TemporaryDirectory() as fresh_cache:
        prev_cache = os.environ.get("TRITON_CACHE_DIR")
        os.environ["TRITON_CACHE_DIR"] = fresh_cache
        try:
            compiled = _compile_kernel(
                _gcc_test_add,
                signature={"X": "*fp32", "Y": "*fp32", "OUT": "*fp32", "N": "i32"},
                constexprs={"BLOCK": 1024},
            )
        finally:
            if prev_cache is None:
                os.environ.pop("TRITON_CACHE_DIR", None)
            else:
                os.environ["TRITON_CACHE_DIR"] = prev_cache

    src = compiled.asm.get("launcher_src")
    if not src:
        # Diagnostic: understand WHY launcher_src is missing on this environment
        target = triton.runtime.driver.active.get_current_target()
        from triton.compiler.compiler import make_backend
        backend = make_backend(target)
        diag = (f"launcher_src not in asm. Diagnostics:\n"
                f"  target: {target}\n"
                f"  backend type: {type(backend).__name__}\n"
                f"  has make_launcher_src: {hasattr(backend, 'make_launcher_src')}\n"
                f"  asm keys: {sorted(compiled.asm.keys())}\n"
                f"  TRITON_CACHE_DIR: {os.environ.get('TRITON_CACHE_DIR', '<not set>')}")
        pytest.fail(diag)

    # Find launch.h: try the canonical source location, then the packaged one.
    backend_dir = os.path.dirname(os.path.realpath(triton.backends.nvidia.__file__))
    candidates = [
        os.path.join(backend_dir, "launch.h"),
        os.path.join(backend_dir, "..", "..", "..", "python", "triton", "runtime", "launch.h"),
    ]
    launch_h = next((p for p in candidates if os.path.exists(p)), None)
    if launch_h is None:
        pytest.skip("launch.h not found for compile test")

    # Find a C compiler
    cc = shutil.which("gcc") or shutil.which("clang") or shutil.which("cc")
    if cc is None:
        pytest.skip("No C compiler available")

    # Find cuda.h include dir
    cuda_include = os.path.join(backend_dir, "include")
    if not os.path.exists(os.path.join(cuda_include, "cuda.h")):
        pytest.skip("cuda.h not found")

    with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
        # Inline launch.h into the source (same as the JIT runtime compile does
        # via driver.py) so we only need -I for cuda.h — avoids fragile
        # triton_include path resolution in buck link-tree environments.
        from pathlib import Path
        launch_h_content = Path(launch_h).read_text()
        test_src = src.replace('#include "nvidia/backend/launch.h"', launch_h_content)
        f.write(test_src)
        f.flush()
        try:
            result = subprocess.run(
                [cc, "-fsyntax-only", "-c", f.name, f"-I{cuda_include}"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert result.returncode == 0, (f"Generated launcher_src failed to compile:\n"
                                            f"stdout: {result.stdout}\nstderr: {result.stderr}")
        finally:
            os.unlink(f.name)


# =============================================================================
# Tests for schema-driven kernel_signature derivation
# =============================================================================


@triton.jit
def multi_type_kernel(ptr_fp32, ptr_fp16, scalar_i32, scalar_i64, scalar_fp32, N, BLOCK: tl.constexpr):
    """Kernel with diverse arg types to test schema-driven signature derivation."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(ptr_fp32 + offs, tl.load(ptr_fp32 + offs, mask=offs < N), mask=offs < N)


@pytest.mark.parametrize("kernel,signature,constexprs", [
    (add_kernel, {"X": "*fp32", "Y": "*fp32", "OUT": "*fp32", "N": "i32"}, {"BLOCK": 1024}),
    (kernel_with_constant, {"X": "*fp16", "N": "i64"}, {"BLOCK": 512}),
    (multi_type_kernel, {
        "ptr_fp32": "*fp32", "ptr_fp16": "*fp16", "scalar_i32": "i32", "scalar_i64": "i64", "scalar_fp32": "fp32", "N":
        "i32"
    }, {"BLOCK": 256}),
])
def test_schema_derived_signature_matches_legacy(kernel, signature, constexprs):
    """kernel_signature from Level 0 schema must match legacy expand_signature path.

    This validates that build_kernel_signature_from_schema() produces the exact
    same byte sequence as the old make_kernel_signature(expand_signature(...)) path.
    """
    compiled = _compile_kernel(kernel, signature=signature, constexprs=constexprs)
    src = compiled.src
    md = compiled.metadata

    # Legacy path: expand_signature → make_kernel_signature
    sig = {idx: value for idx, value in src.signature.items()}
    tensordesc_meta = getattr(md, "tensordesc_meta", None)
    expanded = expand_signature(sig.values(), tensordesc_meta)
    legacy_signature = make_kernel_signature(expanded)

    # Schema path: make_launch_metadata → build_kernel_signature_from_schema
    backend = make_backend(md.target)
    schema = backend.make_launch_metadata(md._asdict(), src)
    schema_signature = build_kernel_signature_from_schema(schema)

    assert schema_signature == legacy_signature, (f"Schema-derived signature differs from legacy!\n"
                                                  f"  schema args: {[a['type'] for a in schema['args']]}\n"
                                                  f"  expanded sig: {expanded}\n"
                                                  f"  schema bytes: {list(schema_signature)}\n"
                                                  f"  legacy bytes: {list(legacy_signature)}")


@pytest.mark.parametrize("tensordesc_type,tensordesc_meta,other_args", [
    # Host TMA path (meta is None): 2D tensor descriptor
    ("tensordesc<fp32[128, 64]>", [], [{"name": "N", "type": "i32", "index": 1}]),
    # Device TMA path: 2D tensor descriptor with device TMA metadata
    ("tensordesc<fp16[64, 64]>", [{"use_device_tma": True}], [{"name": "N", "type": "i32", "index": 1}]),
    # Host TMA path: 1D tensor descriptor
    ("tensordesc<fp32[256]>", [], []),
    # Device TMA path: 1D tensor descriptor
    ("tensordesc<bf16[128]>", [{"use_device_tma": True}], []),
    # Mixed: tensordesc + regular pointer args
    ("tensordesc<fp16[32, 32]>", [], [
        {"name": "out_ptr", "type": "*fp16", "index": 1},
        {"name": "N", "type": "i32", "index": 2},
    ]),
])
def test_schema_derived_signature_tensordesc(tensordesc_type, tensordesc_meta, other_args):
    """build_kernel_signature_from_schema handles tensordesc args (host and device TMA paths).

    This directly constructs a schema dict to test tensordesc expansion logic
    without requiring GPU compilation of a TMA kernel.
    """
    schema = {
        "args": [{"name": "desc", "type": tensordesc_type, "index": 0}] + other_args,
        "tensordesc_meta": tensordesc_meta,
    }

    # Schema path
    schema_signature = build_kernel_signature_from_schema(schema)

    # Legacy path: build equivalent flat signature list
    sig_values = [tensordesc_type] + [a["type"] for a in other_args]
    expanded = expand_signature(sig_values, tensordesc_meta or None)
    legacy_signature = make_kernel_signature(expanded)

    assert schema_signature == legacy_signature, (f"Schema-derived signature differs from legacy for tensordesc!\n"
                                                  f"  tensordesc_type: {tensordesc_type}\n"
                                                  f"  tensordesc_meta: {tensordesc_meta}\n"
                                                  f"  expanded sig: {expanded}\n"
                                                  f"  schema bytes: {list(schema_signature)}\n"
                                                  f"  legacy bytes: {list(legacy_signature)}")


# =========================================================================
# HookChain.__bool__, _TritonDispatcher, _TritonJITRunner
# =========================================================================


def test_hookchain_bool_empty():
    """Empty HookChain should evaluate as False."""
    chain = HookChain()
    assert not chain
    assert bool(chain) is False


def test_hookchain_bool_nonempty():
    """HookChain with hooks should evaluate as True."""
    chain = HookChain()
    chain.add(lambda: None)
    assert chain
    assert bool(chain) is True


def test_hookchain_bool_after_remove():
    """HookChain should become False again after removing all hooks."""
    chain = HookChain()
    fn = lambda: None
    chain.add(fn)
    assert bool(chain) is True
    chain.remove(fn)
    assert bool(chain) is False


def test_dispatcher_created_with_flag(monkeypatch):
    """CompiledKernel._dispatcher should be set when use_triton_dispatcher is enabled."""
    monkeypatch.setenv("TRITON_USE_C_DISPATCHER", "1")
    compiled = _compile_kernel(
        add_kernel,
        signature={"X": "*fp32", "Y": "*fp32", "OUT": "*fp32", "N": "i32"},
        constexprs={"BLOCK": 1024},
    )
    compiled._init_handles()
    assert compiled._dispatcher is not None


def test_dispatcher_not_created_without_flag(monkeypatch):
    """CompiledKernel._dispatcher should be None without the flag."""
    monkeypatch.setenv("TRITON_USE_C_DISPATCHER", "0")
    compiled = _compile_kernel(
        add_kernel,
        signature={"X": "*fp32", "Y": "*fp32", "OUT": "*fp32", "N": "i32"},
        constexprs={"BLOCK": 1024},
    )
    compiled._init_handles()
    assert compiled._dispatcher is None


def test_dispatcher_is_callable(monkeypatch):
    """_TritonDispatcher should be callable when created."""
    monkeypatch.setenv("TRITON_USE_C_DISPATCHER", "1")
    compiled = _compile_kernel(
        add_kernel,
        signature={"X": "*fp32", "Y": "*fp32", "OUT": "*fp32", "N": "i32"},
        constexprs={"BLOCK": 1024},
    )
    compiled._init_handles()
    assert compiled._dispatcher is not None, "_dispatcher was not created"
    assert callable(compiled._dispatcher)


def test_launch_metadata_returns_none_without_hooks():
    """launch_metadata should return None when no enter hooks are registered."""
    compiled = _compile_kernel(
        add_kernel,
        signature={"X": "*fp32", "Y": "*fp32", "OUT": "*fp32", "N": "i32"},
        constexprs={"BLOCK": 1024},
    )
    if knobs.runtime.launch_enter_hook:
        pytest.skip("launch_enter_hook is registered, precondition not met")
    result = compiled.launch_metadata((1, 1, 1), 0)
    assert result is None


def test_dispatcher_e2e(monkeypatch):
    """End-to-end: _TritonDispatcher dispatches correctly on GPU."""
    monkeypatch.setenv("TRITON_USE_C_DISPATCHER", "1")
    compiled = _compile_kernel(
        add_kernel,
        signature={"X": "*fp32", "Y": "*fp32", "OUT": "*fp32", "N": "i32"},
        constexprs={"BLOCK": 1024},
    )
    compiled._init_handles()
    if compiled._dispatcher is None:
        pytest.skip("Dispatcher not available")

    N = 1024
    x = torch.randn(N, device="cuda", dtype=torch.float32)
    y = torch.randn(N, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)

    stream = torch.cuda.current_stream().cuda_stream
    compiled._dispatcher(1, 1, 1, stream, x.data_ptr(), y.data_ptr(), out.data_ptr(), N)
    torch.cuda.synchronize()
    assert torch.allclose(out, x + y, atol=1e-5), f"max diff: {(out - x - y).abs().max()}"


def test_dispatcher_getitem_jit_runner(monkeypatch):
    """CompiledKernel.__getitem__ should return a _TritonJITRunner when dispatcher is available."""
    monkeypatch.setenv("TRITON_USE_C_DISPATCHER", "1")
    compiled = _compile_kernel(
        add_kernel,
        signature={"X": "*fp32", "Y": "*fp32", "OUT": "*fp32", "N": "i32"},
        constexprs={"BLOCK": 1024},
    )

    N = 1024
    x = torch.randn(N, device="cuda", dtype=torch.float32)
    y = torch.randn(N, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)

    runner = compiled[(1, 1, 1)]
    runner(x.data_ptr(), y.data_ptr(), out.data_ptr(), N)
    torch.cuda.synchronize()
    assert torch.allclose(out, x + y, atol=1e-5), f"max diff: {(out - x - y).abs().max()}"
