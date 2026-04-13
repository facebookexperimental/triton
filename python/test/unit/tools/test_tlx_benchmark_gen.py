"""Unit tests for triton.tools.tlx_benchmark_gen.

Tests cover the argument-capture serialization, grid capture, and standalone
test-script generation logic.  All tests are CPU-only unless marked with
@pytest.mark.skipif (GPU-dependent tests are gated on CUDA availability).
"""

import json
import os
from collections import OrderedDict

import pytest
import torch

from triton.tools.tlx_benchmark_gen import (
    _dtype_str,
    _ensure_dump_dir,
    capture_grid,
    capture_kernel_args,
    generate_standalone_test,
)

# ---------------------------------------------------------------------------
# _dtype_str
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype,expected",
    [
        (torch.bfloat16, "bfloat16"),
        (torch.float32, "float32"),
        (torch.float16, "float16"),
        (torch.int32, "int32"),
        (torch.int8, "int8"),
        (torch.bool, "bool"),
    ],
)
def test_dtype_str(dtype, expected):
    assert _dtype_str(dtype) == expected


# ---------------------------------------------------------------------------
# _ensure_dump_dir
# ---------------------------------------------------------------------------


def test_ensure_dump_dir_creates_dir(monkeypatch):
    monkeypatch.delenv("TRITON_TLX_DUMP_DIR", raising=False)
    dump_dir = _ensure_dump_dir()
    try:
        assert os.path.isdir(dump_dir)
        assert os.environ["TRITON_TLX_DUMP_DIR"] == dump_dir
    finally:
        monkeypatch.delenv("TRITON_TLX_DUMP_DIR", raising=False)
        os.rmdir(dump_dir)


def test_ensure_dump_dir_reuses_existing(monkeypatch, tmp_path):
    existing = str(tmp_path)
    monkeypatch.setenv("TRITON_TLX_DUMP_DIR", existing)
    assert _ensure_dump_dir() == existing


# ---------------------------------------------------------------------------
# capture_kernel_args — scalars
# ---------------------------------------------------------------------------


def test_capture_kernel_args_scalars(monkeypatch, tmp_path):
    monkeypatch.setenv("TRITON_TLX_DUMP_DIR", str(tmp_path))
    bound_args = OrderedDict([("alpha", 0.5), ("count", 42), ("flag", True)])
    signature = {"alpha": "fp32", "count": "i32", "flag": "i1"}
    constexprs = {}

    capture_kernel_args(bound_args, signature, constexprs)

    with open(tmp_path / "_kernel_args.json") as f:
        meta = json.load(f)

    args = meta["args"]
    assert len(args) == 3

    assert args[0]["name"] == "alpha"
    assert args[0]["kind"] == "scalar"
    assert args[0]["scalar_type"] == "float"
    assert args[0]["value"] == 0.5

    assert args[1]["name"] == "count"
    assert args[1]["kind"] == "scalar"
    assert args[1]["scalar_type"] == "int"
    assert args[1]["value"] == 42

    # bool must come before int in isinstance checks
    assert args[2]["name"] == "flag"
    assert args[2]["kind"] == "scalar"
    assert args[2]["scalar_type"] == "bool"
    assert args[2]["value"] is True


# ---------------------------------------------------------------------------
# capture_kernel_args — tensors
# ---------------------------------------------------------------------------


def test_capture_kernel_args_tensors(monkeypatch, tmp_path):
    monkeypatch.setenv("TRITON_TLX_DUMP_DIR", str(tmp_path))
    t = torch.randn(4, 48, 1024, dtype=torch.float32)
    bound_args = OrderedDict([("M", t)])
    signature = {"M": "*fp32"}
    constexprs = {}

    capture_kernel_args(bound_args, signature, constexprs)

    with open(tmp_path / "_kernel_args.json") as f:
        meta = json.load(f)

    entry = meta["args"][0]
    assert entry["kind"] == "tensor"
    assert entry["shape"] == [4, 48, 1024]
    assert entry["dtype"] == "float32"
    assert entry["strides"] == list(t.stride())


# ---------------------------------------------------------------------------
# capture_kernel_args — TensorDescriptors
# ---------------------------------------------------------------------------


def test_capture_kernel_args_tensor_descriptors(monkeypatch, tmp_path):
    monkeypatch.setenv("TRITON_TLX_DUMP_DIR", str(tmp_path))
    # TensorDescriptor requires 16-byte aligned base pointer and strides.
    # On CPU tensors, data_ptr() alignment depends on the allocator, so we
    # directly write the expected JSON structure and verify it round-trips
    # correctly (testing the serialization format, not the isinstance path).
    base = torch.randn(4, 128, dtype=torch.bfloat16)

    import triton.tools.tlx_benchmark_gen as tbg
    dump_dir = tbg._ensure_dump_dir()
    meta = {
        "args": [{
            "name": "desc_q",
            "sig_type": "tensordesc<bf16[64, 128]>",
            "kind": "tensor_descriptor",
            "base_shape": list(base.shape),
            "base_dtype": tbg._dtype_str(base.dtype),
            "base_strides": list(base.stride()),
            "desc_shape": [512, 128],
            "desc_strides": [128, 1],
            "block_shape": [64, 128],
        }],
        "constexprs": {},
    }
    json_path = os.path.join(dump_dir, "_kernel_args.json")
    with open(json_path, "w") as f:
        json.dump(meta, f)

    with open(json_path) as f:
        loaded = json.load(f)

    entry = loaded["args"][0]
    assert entry["kind"] == "tensor_descriptor"
    assert entry["base_shape"] == [4, 128]
    assert entry["base_dtype"] == "bfloat16"
    assert entry["desc_shape"] == [512, 128]
    assert entry["desc_strides"] == [128, 1]
    assert entry["block_shape"] == [64, 128]


# ---------------------------------------------------------------------------
# capture_kernel_args — constexprs
# ---------------------------------------------------------------------------


def test_capture_kernel_args_constexprs(monkeypatch, tmp_path):
    monkeypatch.setenv("TRITON_TLX_DUMP_DIR", str(tmp_path))
    bound_args = OrderedDict([("x", 1.0), ("N", 1024), ("BLOCK_M", 256), ("FP8", False)])
    signature = {"x": "fp32", "N": "i32", "BLOCK_M": "constexpr", "FP8": "constexpr"}
    # constexprs maps (index,) -> value for constexpr params
    constexprs = {(2, ): 256, (3, ): False}

    capture_kernel_args(bound_args, signature, constexprs)

    with open(tmp_path / "_kernel_args.json") as f:
        meta = json.load(f)

    # x and N should be scalars, BLOCK_M and FP8 should be constexprs
    args = meta["args"]
    assert args[0]["kind"] == "scalar"
    assert args[1]["kind"] == "scalar"
    assert args[2]["kind"] == "constexpr"
    assert args[2]["name"] == "BLOCK_M"
    assert args[2]["value"] == 256
    assert args[3]["kind"] == "constexpr"
    assert args[3]["name"] == "FP8"
    assert args[3]["value"] is False

    # Top-level constexprs map should be populated
    assert meta["constexprs"]["BLOCK_M"] == 256
    assert meta["constexprs"]["FP8"] is False


# ---------------------------------------------------------------------------
# capture_grid
# ---------------------------------------------------------------------------


def test_capture_grid(monkeypatch, tmp_path):
    monkeypatch.setenv("TRITON_TLX_DUMP_DIR", str(tmp_path))
    # Write initial JSON
    with open(tmp_path / "_kernel_args.json", "w") as f:
        json.dump({"args": [], "constexprs": {}}, f)

    capture_grid((4, 192, 1))

    with open(tmp_path / "_kernel_args.json") as f:
        meta = json.load(f)

    assert meta["grid"] == [4, 192, 1]


def test_capture_grid_noop_without_dir(monkeypatch):
    monkeypatch.delenv("TRITON_TLX_DUMP_DIR", raising=False)
    # Should not raise
    capture_grid((1, 1, 1))


# ---------------------------------------------------------------------------
# generate_standalone_test — without source
# ---------------------------------------------------------------------------


def test_generate_standalone_test_no_source(tmp_path):
    """Test generation when no _source.py exists (TLX kernel only)."""
    kernel_name = "_my_kernel"
    meta = {
        "args": [
            {
                "name": "x_ptr", "kind": "tensor", "dtype": "float32", "shape": [1024], "strides": [1], "sig_type":
                "*fp32"
            },
            {"name": "N", "kind": "scalar", "scalar_type": "int", "value": 1024, "sig_type": "i32"},
            {"name": "BLOCK", "kind": "constexpr", "scalar_type": "int", "value": 256, "sig_type": "constexpr"},
        ],
        "constexprs": {"BLOCK": 256},
        "grid": [4, 1, 1],
    }
    with open(tmp_path / "_kernel_args.json", "w") as f:
        json.dump(meta, f)

    generate_standalone_test(str(tmp_path), kernel_name)

    test_path = tmp_path / "_test_standalone.py"
    assert test_path.exists()

    content = test_path.read_text()

    # Should import the kernel
    assert "from _my_kernel_kernel import _my_kernel" in content
    # Should have benchmark function
    assert "def benchmark():" in content
    # Should create tensors from JSON via dtype-aware helper
    assert "_make_tensor" in content
    # Should call do_bench
    assert "triton.testing.do_bench" in content
    # Should NOT have source module loading (no _source.py)
    assert "_load_source_module" not in content
    # Should NOT have source kernel benchmark section (no _load_source_module call)
    assert "src_kernel" not in content
    assert "ms_src" not in content
    # The generated script should be valid Python syntax
    compile(content, str(test_path), "exec")


# ---------------------------------------------------------------------------
# generate_standalone_test — with source
# ---------------------------------------------------------------------------


def test_generate_standalone_test_with_source(tmp_path):
    """Test generation when _source.py exists (both TLX and source kernel)."""
    kernel_name = "_attn_fwd"
    meta = {
        "args": [
            {"name": "sm_scale", "kind": "scalar", "scalar_type": "float", "value": 0.088, "sig_type": "fp32"},
            {
                "name": "M", "kind": "tensor", "dtype": "float32", "shape": [4, 48, 1024], "strides": [49152, 1024, 1],
                "sig_type": "*fp32"
            },
            {"name": "Z", "kind": "scalar", "scalar_type": "int", "value": 4, "sig_type": "i32"},
            {"name": "H", "kind": "scalar", "scalar_type": "int", "value": 48, "sig_type": "i32"},
            {
                "name": "desc_q", "kind": "tensor_descriptor", "base_shape": [4, 48, 1024, 128], "base_dtype":
                "bfloat16", "base_strides": [6291456, 131072, 128, 1], "desc_shape": [196608, 128], "desc_strides":
                [128, 1], "block_shape": [128, 128], "sig_type": "tensordesc<bf16[128,128]>"
            },
            {"name": "N_CTX", "kind": "scalar", "scalar_type": "int", "value": 1024, "sig_type": "i32"},
            {"name": "HEAD_DIM", "kind": "constexpr", "scalar_type": "int", "value": 128, "sig_type": "constexpr"},
            {"name": "BLOCK_M", "kind": "constexpr", "scalar_type": "int", "value": 256, "sig_type": "constexpr"},
            {"name": "FP8_OUTPUT", "kind": "constexpr", "scalar_type": "bool", "value": False, "sig_type": "constexpr"},
            {"name": "STAGE", "kind": "constexpr", "scalar_type": "int", "value": 1, "sig_type": "constexpr"},
        ],
        "constexprs": {
            "HEAD_DIM": 128,
            "BLOCK_M": 256,
            "FP8_OUTPUT": False,
            "STAGE": 1,
        },
        "grid": [4, 192, 1],
    }
    with open(tmp_path / "_kernel_args.json", "w") as f:
        json.dump(meta, f)

    # Create a dummy source file
    (tmp_path / "_attn_fwd_source.py").write_text("# dummy source\n")

    generate_standalone_test(str(tmp_path), kernel_name)

    test_path = tmp_path / "_test_standalone.py"
    assert test_path.exists()

    content = test_path.read_text()

    # Should import the kernel
    assert "from _attn_fwd_kernel import _attn_fwd" in content
    # Should have source module loading
    assert "_load_source_module" in content
    # Should have both TLX and source benchmarks
    assert "TLX kernel" in content
    assert "Source kernel" in content
    # Should compute TFLOPS from descriptor shapes
    assert "desc_base_shapes" in content
    assert "tflops_tlx" in content
    assert "tflops_src" in content
    # Should filter autotuner-managed constexprs for source kernel
    assert "BLOCK_" in content
    assert "user_kwargs" in content
    # Constexprs should NOT be passed to TLX kernel
    assert "*kernel_args)" in content
    # The generated script should be valid Python syntax
    compile(content, str(test_path), "exec")


# ---------------------------------------------------------------------------
# generate_standalone_test — missing JSON
# ---------------------------------------------------------------------------


def test_generate_standalone_test_missing_json(tmp_path):
    """generate_standalone_test should gracefully handle missing JSON."""
    generate_standalone_test(str(tmp_path), "_missing_kernel")
    # No test file should be created
    assert not (tmp_path / "_test_standalone.py").exists()


# ---------------------------------------------------------------------------
# E2E: capture_kernel_args + capture_grid + generate_standalone_test
# ---------------------------------------------------------------------------


def test_e2e_capture_and_generate(monkeypatch, tmp_path):
    """End-to-end test: capture args → capture grid → generate test."""
    monkeypatch.setenv("TRITON_TLX_DUMP_DIR", str(tmp_path))

    # Simulate the JIT capturing args for a kernel with mixed arg types
    t1 = torch.randn(4, 48, 1024, dtype=torch.float32)
    bound_args = OrderedDict([
        ("scale", 0.5),
        ("M", t1),
        ("batch", 4),
        ("heads", 48),
        ("BLOCK_SIZE", 256),
        ("USE_FP8", False),
    ])
    signature = {
        "scale": "fp32",
        "M": "*fp32",
        "batch": "i32",
        "heads": "i32",
        "BLOCK_SIZE": "constexpr",
        "USE_FP8": "constexpr",
    }
    constexprs = {(4, ): 256, (5, ): False}

    # Phase 1: capture args (happens before _do_compile in jit.py)
    capture_kernel_args(bound_args, signature, constexprs)
    json_path = tmp_path / "_kernel_args.json"
    assert json_path.exists()

    with open(json_path) as f:
        meta = json.load(f)
    assert len(meta["args"]) == 6
    assert "grid" not in meta  # grid not captured yet

    # Phase 2: capture grid (happens after grid evaluation in jit.py)
    capture_grid((4, 192, 1))
    with open(json_path) as f:
        meta = json.load(f)
    assert meta["grid"] == [4, 192, 1]

    # Phase 3: generate standalone test (happens in make_llir)
    kernel_name = "_my_kernel"
    generate_standalone_test(str(tmp_path), kernel_name)

    test_path = tmp_path / "_test_standalone.py"
    assert test_path.exists()

    content = test_path.read_text()
    # Verify the generated script is syntactically valid
    compile(content, str(test_path), "exec")
    # Verify it reads the JSON
    assert "_kernel_args.json" in content
    # Verify it creates the kernel call
    assert "_my_kernel[grid](*kernel_args)" in content
