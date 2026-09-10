"""Compile-only TLX checks: no device required.

The Python counterpart of the LIT suite in test/ -- compile for an
explicit target and assert on the emitted IR, which lets one kernel be
parametrized across targets and dtypes.
"""
import pytest
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
import triton.language.extra.tlx as tlx
from triton._filecheck import run_parser
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language._core import builtin as gluon_builtin
import dataclasses
import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
from triton.compiler.compiler import ASTSource, compile as triton_compile
from triton.compiler.errors import CompilationError
from triton.backends.amd import compiler as amd_compiler
from triton.language.extra.tlx.tutorials import amd_fa_cluster as _amd_fa_cluster_module
from triton.language.extra.tlx.tutorials.amd_mxfp_gemm_tdm_pipelined import (
    mxgemm_tdm_pipelined_kernel as _amd_mxfp_gemm_kernel, )
from triton.language.extra.tlx.tutorials.amd_tdm_gemm_pipelined import (
    matmul_tdm_pipelined_kernel as _amd_tdm_gemm_kernel, )
from triton.language.extra.tlx.tutorials.amd_fa_cluster import (
    _validate_cluster_inputs as _validate_amd_fa_cluster_inputs,
    _validate_cluster_tiles as _validate_amd_fa_cluster_tiles,
)
import math
from triton.language.extra.tlx.tutorials import amd_fa_bwd, amd_fa_varlen_bwd
from triton.language.extra.tlx.tutorials.amd_fa_bwd import (
    _D64DQLaunch,
    _D64Dispatch,
    _D64_GQA_SIGNED,
    _D64_MHA_POSITIVE,
    _allocate_bwd_d64_causal_gqa8_workspaces,
    _allocate_bwd_d64_fused_workspaces,
    _d64_causal_dkdv_first_query_block,
    _d64_causal_dq_key_blocks,
    _d64_causal_stat_values,
    _d64_dq_launch_plan,
    _run_bwd_d64_direct,
    _select_d64_dispatch,
    _validate_d64_sm_scale,
)


@triton.jit
def _clc_default_multi_ctas_kernel(NUM_CONSUMERS: tl.constexpr):
    clc_context = tlx.clc_create_context(NUM_CONSUMERS)
    tlx.clc_producer(clc_context, 1)
    tlx.clc_consumer(clc_context, 0)


@triton.jit
def _clc_explicit_multi_ctas_kernel(NUM_CONSUMERS: tl.constexpr, MULTI_CTAS: tl.constexpr):
    clc_context = tlx.clc_create_context(NUM_CONSUMERS)
    tlx.clc_producer(clc_context, 1, multi_ctas=MULTI_CTAS)
    tlx.clc_consumer(clc_context, 0, multi_ctas=MULTI_CTAS)


@pytest.mark.parametrize(
    "ctas_per_cga,num_consumers,multi_ctas,expect_remote",
    [
        ((1, 1, 1), 1, None, False),
        ((2, 1, 1), 2, None, True),
        ((2, 1, 1), 1, False, False),
    ],
)
def test_cluster_launch_control_multi_ctas_frontend(ctas_per_cga, num_consumers, multi_ctas, expect_remote):
    if multi_ctas is None:
        src = triton.compiler.ASTSource(
            fn=_clc_default_multi_ctas_kernel,
            signature={"NUM_CONSUMERS": "constexpr"},
            constexprs={"NUM_CONSUMERS": num_consumers},
        )
    else:
        src = triton.compiler.ASTSource(
            fn=_clc_explicit_multi_ctas_kernel,
            signature={"NUM_CONSUMERS": "constexpr", "MULTI_CTAS": "constexpr"},
            constexprs={"NUM_CONSUMERS": num_consumers, "MULTI_CTAS": multi_ctas},
        )
    kernel = triton.compile(
        src,
        target=GPUTarget("cuda", 100, 32),
        options={"ctas_per_cga": ctas_per_cga},
    )
    ttir = kernel.asm["ttir"]
    ttgir = kernel.asm["ttgir"]
    ptx = kernel.asm["ptx"]

    assert ttir.count("nvg.cluster_id") == 1
    expected_equalities = 2 if expect_remote else 1
    assert ttir.count("arith.cmpi eq") == expected_equalities

    if expect_remote:
        assert "ttng.map_to_remote_buffer" in ttgir
        assert "mapa.shared::cluster" in ptx
    else:
        assert "ttng.map_to_remote_buffer" not in ttgir
        assert "mapa.shared::cluster" not in ptx

    assert "multicast::cluster::all" in ptx


def test_clc_response_type_mangle():
    """`clc_response_type` must not share a mangled name with `mbarrier_type`.

    Both carry a placeholder `tl.int64` element type (Triton has no 128-bit
    dtype), so the inherited `buffered_tensor_type.mangle` collapses them to the
    same string for a given shape. A clc_response lowers to a `ui128` memdesc
    and an mbarrier to an `i64` one, so that collision would let the `tt.func`
    generated for one be reused from the JIT cache for the other.
    """
    for num in (0, 1):
        clc_mangle = tlx.clc_response_type(num, None).mangle()
        mbar_mangle = tlx.mbarrier_type(num, None, tlx.storage_kind.smem).mangle()
        assert clc_mangle != mbar_mangle, f"num={num}: {clc_mangle}"


HIP_TARGET_CDNA4 = GPUTarget("hip", "gfx950", 64)


@gluon_builtin
def _semantic_dot_with_acc_layout(a, b, acc, _semantic=None):
    """Expose semantic dot for the accumulator-layout propagation test."""
    return _semantic.dot(
        a,
        b,
        acc,
        input_precision=None,
        max_num_imprecise_acc=None,
        out_dtype=acc.dtype,
    )


@pytest.mark.parametrize("target", [HIP_TARGET_CDNA4])
def test_dot_propagates_accumulator_layout(target):
    """A semantic dot with an explicit accumulator keeps its distributed type."""

    @gluon.jit
    def kernel():
        mfma_layout: ttgl.constexpr = ttgl.amd.AMDMFMALayout(version=4, warps_per_cta=[4, 1], instr_shape=[16, 16, 32],
                                                             transposed=True)
        a_layout: ttgl.constexpr = ttgl.DotOperandLayout(0, mfma_layout, 8)
        b_layout: ttgl.constexpr = ttgl.DotOperandLayout(1, mfma_layout, 8)
        a = ttgl.full([16, 32], 1.0, ttgl.bfloat16, layout=a_layout)
        b = ttgl.full([32, 16], 1.0, ttgl.bfloat16, layout=b_layout)
        acc = ttgl.full([16, 16], 0.0, ttgl.float32, layout=mfma_layout)
        result = _semantic_dot_with_acc_layout(a, b, acc)
        ttgl.static_assert(result.type.layout == mfma_layout)

    module = run_parser(kernel, target=target)
    assert "#ttg.amd_mfma" in module.str_nodebug()


_SEPARABLE_QK_SHAPE = ((32, 4, 2), (32, 2))

_SEPARABLE_QK_STRIDE = ((128, 4096, 32), (1, 64))


def _separable_qk_layout():
    return tlx.layout(
        shape=_SEPARABLE_QK_SHAPE,  # (thread, value)
        stride=_SEPARABLE_QK_STRIDE,
    )


_A16W16_SHARED_INTERVALS = [(512, 16)]

_A16W16_SHARED_OFFSET_BASES = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [64, 0], [1, 0],
                               [2, 0], [4, 0], [8, 0]]

_A16W16_TILE = [128, 64]


def test_amd_mfma_tiles_per_warp_uses_warp_rank():
    """MFMA tile factors follow Gluon's two-axis warp configuration."""
    default = tlx.amd_mfma_layout(4, [32, 32, 16], True, [4, 1])
    tiled = tlx.amd_mfma_layout(4, [32, 32, 16], True, [4, 1], tiles_per_warp=[2, 2])
    assert default.tiles_per_warp == [1, 1]
    assert tiled.tiles_per_warp == [2, 2]


def test_slice_layout_rejects_negative_dimension():
    mma = tlx.amd_mfma_layout(4, [32, 32, 16], True, [4, 1])
    with pytest.raises(ValueError, match="dim must be non-negative"):
        tlx.slice_layout(mma, dim=-1)


def test_swizzled_layout_cute_mapping():
    """`tlx.swizzled_layout(B, M, S)` is the CuTe Swizzle<B,M,S> (positional args).
    It resolves to Triton's (vec, perPhase, maxPhase) for a given contiguous extent,
    per the inverse of DumpLayout's emitCuteSwizzle. Pure-Python, no GPU."""

    # vec = 2**M, maxPhase = 2**B, perPhase = 2**(S+M) // numContig.
    # Mirror the SwizzledSharedEncoding doc examples (order=[1,0], numContig = shape[1]):
    #   vec=1, perPhase=1, maxPhase=4 over a width-4 tile -> Swizzle<2,0,2>
    enc = tlx.swizzled_layout(2, 0, 2, order=[1, 0])._to_encoding(shape=[4, 4])
    assert (enc.vectorSize, enc.perPhase, enc.maxPhase) == (1, 1, 4)
    #   vec=1, perPhase=2, maxPhase=4 over a width-4 tile -> Swizzle<2,0,3>
    enc = tlx.swizzled_layout(2, 0, 3, order=[1, 0])._to_encoding(shape=[4, 4])
    assert (enc.vectorSize, enc.perPhase, enc.maxPhase) == (1, 2, 4)
    #   vec=2, perPhase=1, maxPhase=4 over a width-8 tile -> Swizzle<2,1,2>
    enc = tlx.swizzled_layout(2, 1, 2, order=[1, 0])._to_encoding(shape=[4, 8])
    assert (enc.vectorSize, enc.perPhase, enc.maxPhase) == (2, 1, 4)

    # A Swizzle that would give perPhase < 1 for the extent is rejected.
    with pytest.raises(AssertionError):
        tlx.swizzled_layout(2, 0, 0, order=[1, 0])._to_encoding(shape=[4, 8])


def test_swizzled_layout_vs_register_layout():
    """`swizzled_layout` is a shared-memory layout; the shape/stride `tlx.layout`
    stays a register layout. `tlx.layout(swizzled_layout(...))` also accepts one
    (eagerly resolving the trivial default). Pure-Python, no GPU."""

    # The no-swizzle default is shape-independent -> tlx.layout resolves it eagerly.
    a = tlx.layout(tlx.swizzled_layout.make_default(rank=2))
    assert type(a) is tlx.swizzled_shared_layout_encoding  # exact type -> `type() is` checks hold
    assert (a.vectorSize, a.perPhase, a.maxPhase, a.order) == (1, 1, 1, [1, 0])

    # A real swizzle is deferred (needs the buffer shape): tlx.layout returns it as-is.
    atom = tlx.layout(tlx.swizzled_layout(2, 0, 2, order=[1, 0]))
    assert isinstance(atom, tlx.swizzled_layout)

    # the shape/stride form is unchanged: a register layout
    r = _separable_qk_layout()
    assert isinstance(r, tlx.layout) and not isinstance(r, tlx.shared_layout_encoding)

    # tlx.layout() with neither a swizzled_layout nor shape/stride is rejected
    with pytest.raises(AssertionError):
        tlx.layout()


def test_with_bases_builds_swizzled_padded_encoding():
    """`padded_shared_layout_encoding.with_bases` records the explicit linear
    (offset) component instead of the identity {order, shape}. Pure-Python."""
    enc = tlx.padded_shared_layout_encoding.with_bases(_A16W16_SHARED_INTERVALS, _A16W16_SHARED_OFFSET_BASES,
                                                       _A16W16_TILE)
    assert enc.intervals == [512]
    assert enc.paddings == [16]
    assert enc.order == [1, 0]  # reversed(range(rank))
    assert enc.offset_bases == _A16W16_SHARED_OFFSET_BASES
    assert enc.block_bases == []
    assert enc.shape == _A16W16_TILE


def test_shared_linear_layout_records_gluon_k_tile_mapping():
    """TLX exposes Gluon's explicit row-major shared-memory mapping."""
    bases = [
        [0, 1],
        [0, 2],
        [0, 4],
        [0, 8],
        [1, 0],
        [2, 0],
        [4, 0],
        [8, 0],
        [0, 16],
        [0, 32],
        [0, 64],
        [16, 0],
        [32, 0],
    ]
    layout = tlx.shared_linear_layout_encoding(offset_bases=bases, block_bases=[], alignment=16)
    assert layout.offset_bases == bases
    assert layout.block_bases == []
    assert layout.alignment == 16


def test_nv_mma_tile_to_shape_builds_shared_linear():
    """`tile_to_shape` exposes CuTe's atom-layout tiled-to-shape spelling."""
    layout = tlx.nv_mma_shared_layout_encoding(
        (64, 64),
        [1, 0],
        tl.bfloat16,
        [1, 1],
        [1, 1],
        [1, 0],
        False,
        True,
    ).tile_to_shape((64, 128))

    assert isinstance(layout, tlx.shared_linear_layout_encoding)
    assert layout.tile_shape == [64, 128]


@pytest.mark.parametrize("kwargs", [{"num_regs": 128}, {"registers": 128}])
def test_default_task_accepts_registers(kwargs):
    task = tlx.async_task("default", **kwargs)
    assert task.num_regs == 128


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_regs": 25},
        {"registers": 25},
    ],
)
@pytest.mark.parametrize("task_args", [(), ("default", )])
def test_async_task_rejects_unaligned_registers(kwargs, task_args):
    with pytest.raises(ValueError, match="divisible by 8"):
        if task_args:
            tlx.async_task(*task_args, **kwargs)
        else:
            tlx.async_task(num_warps=1, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minRegAutoWS": 25},
        {"maxRegAutoWS": 153},
    ],
)
def test_config_rejects_unaligned_reg_auto_ws(kwargs):
    with pytest.raises(ValueError, match="divisible by 8"):
        triton.Config({}, **kwargs)


def _make_test_buffered_tensor(storage: tlx.storage_kind = tlx.storage_kind.smem):
    """Helper to create a buffered_tensor for testing reuse_group."""
    layout = tlx.swizzled_shared_layout_encoding.make_default(rank=2)
    return tlx.buffered_tensor(
        handle=None,
        element_ty=tl.float32,
        shape=[64, 64],
        num=2,
        storage=storage,
        layout=layout,
    )


class TestStorageKind:
    """Tests for tlx.storage_kind enum."""

    def test_storage_kind_values(self):
        assert tlx.storage_kind.smem.value == "smem"
        assert tlx.storage_kind.tmem.value == "tmem"
        assert tlx.storage_kind.smemCluster.value == "smemCluster"


class TestStorageAliasSpecType:
    """Tests for storage_alias_spec_type class."""

    def test_type_smem_unsized(self):
        ty = tlx.storage_alias_spec_type(tlx.storage_kind.smem)
        assert ty.storage == tlx.storage_kind.smem
        assert ty.buffer_size_bytes is None

    def test_type_tmem_unsized(self):
        ty = tlx.storage_alias_spec_type(tlx.storage_kind.tmem)
        assert ty.storage == tlx.storage_kind.tmem
        assert ty.buffer_size_bytes is None

    def test_type_smem_sized(self):
        ty = tlx.storage_alias_spec_type(tlx.storage_kind.smem, 16384)
        assert ty.storage == tlx.storage_kind.smem
        assert ty.buffer_size_bytes == 16384

    def test_type_tmem_sized(self):
        ty = tlx.storage_alias_spec_type(tlx.storage_kind.tmem, 32768)
        assert ty.storage == tlx.storage_kind.tmem
        assert ty.buffer_size_bytes == 32768

    def test_type_equality_same(self):
        ty1 = tlx.storage_alias_spec_type(tlx.storage_kind.smem, 16384)
        ty2 = tlx.storage_alias_spec_type(tlx.storage_kind.smem, 16384)
        assert ty1 == ty2

    def test_type_equality_different_storage(self):
        ty1 = tlx.storage_alias_spec_type(tlx.storage_kind.smem, 16384)
        ty2 = tlx.storage_alias_spec_type(tlx.storage_kind.tmem, 16384)
        assert ty1 != ty2

    def test_type_equality_different_size(self):
        ty1 = tlx.storage_alias_spec_type(tlx.storage_kind.smem, 16384)
        ty2 = tlx.storage_alias_spec_type(tlx.storage_kind.smem, 32768)
        assert ty1 != ty2

    def test_type_equality_sized_vs_unsized(self):
        ty1 = tlx.storage_alias_spec_type(tlx.storage_kind.smem, 16384)
        ty2 = tlx.storage_alias_spec_type(tlx.storage_kind.smem)
        assert ty1 != ty2

    def test_type_repr_unsized(self):
        ty = tlx.storage_alias_spec_type(tlx.storage_kind.smem)
        assert "smem" in repr(ty)
        assert "size" not in repr(ty)

    def test_type_repr_sized(self):
        ty = tlx.storage_alias_spec_type(tlx.storage_kind.tmem, 16384)
        assert "tmem" in repr(ty)
        assert "16384" in repr(ty)

    def test_type_mangle_unsized(self):
        ty = tlx.storage_alias_spec_type(tlx.storage_kind.smem)
        mangle = ty.mangle()
        assert "storage_alias_spec" in mangle
        assert "smem" in mangle

    def test_type_mangle_sized(self):
        ty = tlx.storage_alias_spec_type(tlx.storage_kind.tmem, 8192)
        mangle = ty.mangle()
        assert "storage_alias_spec" in mangle
        assert "tmem" in mangle
        assert "8192" in mangle


class TestStorageAliasSpecClass:
    """Tests for the storage_alias_spec value class (not the builtin function)."""

    def test_class_smem_unsized(self):
        buf = tlx.storage_alias_spec_type_class(
            handle=None,
            storage=tlx.storage_kind.smem,
        )
        assert buf.storage == tlx.storage_kind.smem
        assert buf.buffer_size_bytes is None
        assert buf.handle is None

    def test_class_tmem_sized(self):
        buf = tlx.storage_alias_spec_type_class(
            handle=None,
            storage=tlx.storage_kind.tmem,
            buffer_size_bytes=32768,
        )
        assert buf.storage == tlx.storage_kind.tmem
        assert buf.buffer_size_bytes == 32768

    def test_class_rejects_smem_cluster(self):
        with pytest.raises(ValueError, match="smemCluster"):
            tlx.storage_alias_spec_type_class(
                handle=None,
                storage=tlx.storage_kind.smemCluster,
            )

    def test_class_type_attribute(self):
        buf = tlx.storage_alias_spec_type_class(
            handle=None,
            storage=tlx.storage_kind.smem,
            buffer_size_bytes=4096,
        )
        assert isinstance(buf.type, tlx.storage_alias_spec_type)
        assert buf.type.storage == tlx.storage_kind.smem
        assert buf.type.buffer_size_bytes == 4096

    def test_class_immutability_storage(self):
        buf = tlx.storage_alias_spec_type_class(
            handle=None,
            storage=tlx.storage_kind.smem,
        )
        with pytest.raises(AttributeError):
            buf.storage = tlx.storage_kind.tmem

    def test_class_immutability_buffer_size(self):
        buf = tlx.storage_alias_spec_type_class(
            handle=None,
            storage=tlx.storage_kind.smem,
            buffer_size_bytes=1024,
        )
        with pytest.raises(AttributeError):
            buf.buffer_size_bytes = 2048

    def test_class_repr_unsized(self):
        buf = tlx.storage_alias_spec_type_class(
            handle=None,
            storage=tlx.storage_kind.smem,
        )
        r = repr(buf)
        assert "storage_alias_spec" in r
        assert "smem" in r

    def test_class_repr_sized(self):
        buf = tlx.storage_alias_spec_type_class(
            handle=None,
            storage=tlx.storage_kind.tmem,
            buffer_size_bytes=65536,
        )
        r = repr(buf)
        assert "storage_alias_spec" in r
        assert "tmem" in r
        assert "65536" in r


class TestLocalAllocWithStorageAliasSpec:
    """Tests for local_alloc accepting storage_alias_spec in reuse parameter."""

    def test_local_alloc_reuse_type_check_buffered_tensor(self):
        """Verify local_alloc accepts buffered_tensor in reuse (legacy behavior)."""
        # This is a type-level test - we can't fully test without a kernel context
        # but we verify the type annotation allows buffered_tensor
        import inspect
        from triton.language.extra.tlx.mem_ops import local_alloc as local_alloc_func

        sig = inspect.signature(local_alloc_func)
        reuse_param = sig.parameters["reuse"]
        # The annotation should include Union or | with both types
        annotation_str = str(reuse_param.annotation)
        assert "buffered_tensor" in annotation_str or "tlx.buffered_tensor" in annotation_str

    def test_local_alloc_reuse_type_check_storage_alias_spec(self):
        """Verify local_alloc accepts storage_alias_spec in reuse (new behavior)."""
        import inspect
        from triton.language.extra.tlx.mem_ops import local_alloc as local_alloc_func

        sig = inspect.signature(local_alloc_func)
        reuse_param = sig.parameters["reuse"]
        # The annotation should include Union or | with both types
        annotation_str = str(reuse_param.annotation)
        assert "storage_alias_spec" in annotation_str or "tlx.storage_alias_spec" in annotation_str

    def test_reuse_storage_mismatch_error_message(self):
        """Verify helpful error message when storage kinds don't match."""
        # Create a storage_alias_spec with smem storage
        buf = tlx.storage_alias_spec_type_class(
            handle=None,
            storage=tlx.storage_kind.smem,
        )
        # The error should mention both storage kinds when there's a mismatch
        # We can't fully test the error without a kernel context, but we can
        # verify the storage_alias_spec's storage property is accessible
        assert buf.storage == tlx.storage_kind.smem


class TestReuseGroupType:
    """Tests for tlx.reuse_group_type enum."""

    def test_reuse_group_type_values(self):
        assert tlx.reuse_group_type.shared.value == "shared"
        assert tlx.reuse_group_type.distinct.value == "distinct"

    def test_reuse_group_type_enum_members(self):
        # Verify all expected members exist
        members = list(tlx.reuse_group_type)
        assert len(members) == 2
        assert tlx.reuse_group_type.shared in members
        assert tlx.reuse_group_type.distinct in members


class TestReuseGroup:
    """Tests for tlx.reuse_group class."""

    def test_reuse_group_basic_shared(self):
        """Test basic reuse_group creation with shared type."""
        elem1 = _make_test_buffered_tensor()
        elem2 = _make_test_buffered_tensor()
        group = tlx.reuse_group(
            elem1,
            elem2,
            group_type=tlx.reuse_group_type.shared,
        )
        assert group.args == (elem1, elem2)
        assert group.group_type == tlx.reuse_group_type.shared

    def test_reuse_group_basic_distinct(self):
        """Test basic reuse_group creation with distinct type."""
        elem1 = _make_test_buffered_tensor()
        elem2 = _make_test_buffered_tensor()
        group = tlx.reuse_group(
            elem1,
            elem2,
            group_type=tlx.reuse_group_type.distinct,
        )
        assert group.args == (elem1, elem2)
        assert group.group_type == tlx.reuse_group_type.distinct

    def test_reuse_group_single_element(self):
        """Test reuse_group with a single element."""
        elem = _make_test_buffered_tensor()
        group = tlx.reuse_group(
            elem,
            group_type=tlx.reuse_group_type.shared,
        )
        assert len(group.args) == 1
        assert group.args[0] is elem

    def test_reuse_group_multiple_elements(self):
        """Test reuse_group with more than 2 elements."""
        elems = tuple(_make_test_buffered_tensor() for _ in range(4))
        group = tlx.reuse_group(
            *elems,
            group_type=tlx.reuse_group_type.distinct,
        )
        assert group.args == elems
        assert len(group.args) == 4

    def test_reuse_group_nested(self):
        """Test nested reuse_group (Flash Attention pattern)."""
        # Inner group: distinct elements
        p = _make_test_buffered_tensor()
        alpha = _make_test_buffered_tensor()
        inner_group = tlx.reuse_group(
            p,
            alpha,
            group_type=tlx.reuse_group_type.distinct,
        )

        # Outer group: shared with inner group
        qk = _make_test_buffered_tensor()
        outer_group = tlx.reuse_group(
            qk,
            inner_group,
            group_type=tlx.reuse_group_type.shared,
        )

        assert outer_group.group_type == tlx.reuse_group_type.shared
        assert len(outer_group.args) == 2
        assert outer_group.args[0] is qk
        assert outer_group.args[1] is inner_group
        assert inner_group.group_type == tlx.reuse_group_type.distinct

    def test_reuse_group_deeply_nested(self):
        """Test 3-level nested reuse_group."""
        # Level 3 (innermost)
        c = _make_test_buffered_tensor()
        d = _make_test_buffered_tensor()
        inner = tlx.reuse_group(
            c,
            d,
            group_type=tlx.reuse_group_type.shared,
        )

        # Level 2
        b = _make_test_buffered_tensor()
        middle = tlx.reuse_group(
            b,
            inner,
            group_type=tlx.reuse_group_type.distinct,
        )

        # Level 1 (outermost)
        a = _make_test_buffered_tensor()
        outer = tlx.reuse_group(
            a,
            middle,
            group_type=tlx.reuse_group_type.shared,
        )

        assert outer.group_type == tlx.reuse_group_type.shared
        assert outer.args[1].group_type == tlx.reuse_group_type.distinct
        assert outer.args[1].args[1].group_type == tlx.reuse_group_type.shared

    def test_reuse_group_empty_args_raises_error(self):
        """Test reuse_group raises error with empty args tuple."""
        with pytest.raises(ValueError, match="at least one element"):
            tlx.reuse_group(group_type=tlx.reuse_group_type.shared, )

    def test_reuse_group_invalid_element_type_raises_error(self):
        """Test that invalid element types raise TypeError."""
        with pytest.raises(TypeError, match="must be buffered_tensor or reuse_group"):
            tlx.reuse_group(
                "invalid",
                group_type=tlx.reuse_group_type.shared,
            )


GFX950 = GPUTarget("hip", "gfx950", 64)

GFX942 = GPUTarget("hip", "gfx942", 64)

GFX1250 = GPUTarget("hip", "gfx1250", 32)


def compile_for_target(fn, signature, constexprs, target):
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs)
    return triton_compile(src, target=target)


def compile_for_gfx950(fn, signature, constexprs):
    """Compile a TLX kernel for gfx950 and return the compiled object."""
    return compile_for_target(fn, signature, constexprs, GFX950)


def compile_for_gfx942(fn, signature, constexprs):
    """Compile a TLX kernel for gfx942 and return the compiled object."""
    return compile_for_target(fn, signature, constexprs, GFX942)


@triton.jit
def _warp_predicate_update(lhs, rhs, increment, side_ptr, offsets):
    tl.store(side_ptr + offsets, lhs)
    return lhs + increment, rhs - increment


@triton.jit
def _warp_predicate_kernel(x_ptr, lhs_ptr, rhs_ptr, side_ptr, size: tl.constexpr):
    offsets = tl.arange(0, size)
    lhs = tl.load(x_ptr + offsets)
    rhs = lhs * 2.0
    predicate = (offsets >= 64) & (offsets < 128) & (offsets % 5 < 2)
    lhs, rhs = tlx.warp_predicate(
        predicate,
        (lhs, rhs),
        _warp_predicate_update,
        args=(3.0, side_ptr, offsets),
    )
    tl.store(lhs_ptr + offsets, lhs)
    tl.store(rhs_ptr + offsets, rhs)


@triton.jit
def _nested_warp_predicate_inner(value):
    return value + 1.0


@triton.jit
def _nested_warp_predicate_outer(value, predicate):
    return tlx.warp_predicate(predicate, value, _nested_warp_predicate_inner)


@triton.jit
def _nested_warp_predicate_kernel(x_ptr, output_ptr):
    offsets = tl.arange(0, 256)
    value = tl.load(x_ptr + offsets)
    predicate = offsets % 3 == 0
    value = tlx.warp_predicate(
        predicate,
        value,
        _nested_warp_predicate_outer,
        args=(predicate, ),
    )
    tl.store(output_ptr + offsets, value)


@triton.jit
def _warp_predicate_cross_wave_reduce(value):
    return value + tl.sum(value, axis=0)


@triton.jit
def _warp_predicate_cross_wave_reduce_kernel(x_ptr, output_ptr, size: tl.constexpr):
    offsets = tl.arange(0, size)
    value = tl.load(x_ptr + offsets)
    wave = tlx.thread_id(0) // 64
    predicate = wave >= 2
    value = tlx.warp_predicate(predicate, value, _warp_predicate_cross_wave_reduce, wave_uniform=True)
    tl.store(output_ptr + offsets, value)


@triton.jit
def _warp_predicate_warp_local_reduce(value):
    row_sum = tl.sum(value, axis=1)
    return value + row_sum[:, None]


@triton.jit
def _warp_predicate_warp_local_reduce_kernel(x_ptr, output_ptr):
    offsets = tl.arange(0, 256)
    value = tl.reshape(tl.load(x_ptr + offsets), (4, 64))
    wave = tlx.thread_id(0) // 64
    predicate = wave >= 2
    value = tlx.warp_predicate(predicate, value, _warp_predicate_warp_local_reduce, wave_uniform=True)
    tl.store(output_ptr + offsets, tl.reshape(value, (256, )))


@triton.jit
def _warp_predicate_lane_divergent_reduce_kernel(x_ptr, output_ptr):
    offsets = tl.arange(0, 256)
    value = tl.reshape(tl.load(x_ptr + offsets), (4, 64))
    predicate = tlx.thread_id(0) % 2 == 0
    value = tlx.warp_predicate(predicate, value, _warp_predicate_warp_local_reduce)
    tl.store(output_ptr + offsets, tl.reshape(value, (256, )))


@triton.jit
def _async_local_slice_dot_kernel(q_ptr, k_ptr, output_ptr):
    rows = tl.arange(0, 128)
    cols = tl.arange(0, 32)
    reduction = tl.arange(0, 128)
    q = tl.load(q_ptr + rows[:, None] * 128 + reduction[None, :])

    k_rows = tl.arange(0, 64)
    k_ptrs = k_ptr + k_rows[:, None] * 128 + reduction[None, :]
    k_buffers = tlx.local_alloc((64, 128), tl.float16, 1)
    k_view = tlx.local_view(k_buffers, 0)
    token = tlx.async_load(k_ptrs, k_view)
    tlx.async_load_commit_group([token])
    wait = tlx.async_load_wait_group(0)

    k_lo = tlx.local_slice(k_view, [0, 0], [32, 128])
    kt = tlx.local_load(tlx.local_trans(k_lo), token=wait, relaxed=True)
    result = tl.dot(q, kt)
    tl.store(output_ptr + rows[:, None] * 32 + cols[None, :], result)


@triton.jit
def _warp_vote_kernel(x_ptr, all_ptr, any_ptr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    predicate = tl.load(x_ptr + offsets) != 0
    all_value = tlx.warp_all(predicate).to(tl.int32)
    any_value = tlx.warp_any(predicate).to(tl.int32)
    tl.store(all_ptr + offsets, all_value)
    tl.store(any_ptr + offsets, any_value)


@triton.jit
def _warp_vote_scalar_predicate_kernel(output):
    predicate = tl.program_id(0) == 0
    tl.store(output, tlx.warp_all(predicate).to(tl.int32))


@triton.jit
def _shared_concrete_helper(values):
    return values


@triton.jit
def _shared_concrete_helper_kernel(x_ptr, y_ptr):
    layout_m: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    layout_n: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    rows = tl.arange(0, 64)
    cols = tl.arange(0, 64)
    offsets = rows[:, None] * 64 + cols[None, :]
    values = tl.load(x_ptr + offsets)
    values_m = tlx.require_layout(values, layout_m, pin=False)
    values_n = tlx.require_layout(values, layout_n, pin=False)
    result_m = _shared_concrete_helper(values_m)
    result_n = _shared_concrete_helper(values_n)
    output_m = tlx.require_layout(y_ptr + offsets, layout_m, pin=False)
    output_n = tlx.require_layout(y_ptr + 4096 + offsets, layout_n, pin=False)
    tl.store(output_m, result_m)
    tl.store(output_n, result_n)


@triton.jit
def _mixed_helper_results(values, condition, LAYOUT: tl.constexpr):
    # The frontend emits an encoding-free return for the else path and the
    # trailing unreachable block.  Fixup must bridge only result 0; result 1
    # intentionally remains encoding-free.
    if condition:
        concrete = tlx.require_layout(values, LAYOUT, pin=False)
        return concrete, values
    return values, values


@triton.jit
def _mixed_helper_results_kernel(x_ptr, y_ptr, condition):
    value_layout: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    rows = tl.arange(0, 16)
    cols = tl.arange(0, 64)
    offsets = rows[:, None] * 64 + cols[None, :]
    values = tl.load(x_ptr + offsets)
    concrete, deferred = _mixed_helper_results(values, condition, value_layout)
    concrete_offsets = tlx.require_layout(y_ptr + offsets, value_layout, pin=False)
    # Consume the siblings under different ABIs.  If fixup retypes the shared
    # producer, the encoding-free store below becomes invalid.
    tl.store(concrete_offsets, concrete)
    tl.store(y_ptr + 1024 + offsets, deferred)


@triton.jit
def _concrete_layout_while_kernel(x_ptr, y_ptr, count):
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    rows = tl.arange(0, 16)
    cols = tl.arange(0, 64)
    offsets = rows[:, None] * 64 + cols[None, :]
    values = tl.load(x_ptr + offsets)
    i = 0
    while i < count:
        values = tlx.require_layout(values, mma, pin=False)
        values += 1.0
        i += 1
    tl.store(y_ptr + offsets, values)


@triton.jit
def _slice_layout_validation_kernel(output, layout: tl.constexpr):
    values = tlx.zeros([16], tl.float32, layout=layout)
    tl.store(output + tl.arange(0, 16), values)


@triton.jit
def _concrete_predicate_scale(value, scale):
    return value * scale


@triton.jit
def _concrete_dot_control_flow_helper(a, b, condition, predicate, MMA: tl.constexpr, DOT0: tl.constexpr,
                                      DOT1: tl.constexpr):
    a = tlx.require_layout(a, DOT0, pin=False)
    b = tlx.require_layout(b, DOT1, pin=False)
    acc = tlx.require_layout(tl.zeros((16, 64), tl.float32), MMA, pin=False)
    result = tl.dot(a, b, acc)
    if condition:
        result = result * 2.0
    else:
        result = result + 1.0
    return tlx.warp_predicate(predicate, result, _concrete_predicate_scale, args=(0.5, ))


@triton.jit
def _concrete_helper_release_kernel(a_ptr, b_ptr, output_ptr, condition):
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    dot0: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=8)
    dot1: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=8)
    rows = tl.arange(0, 16)
    reduction = tl.arange(0, 32)
    cols = tl.arange(0, 64)
    a = tl.load(a_ptr + rows[:, None] * 32 + reduction[None, :])
    b = tl.load(b_ptr + reduction[:, None] * 64 + cols[None, :])
    predicate = (rows[:, None] < 8) & (cols[None, :] >= 0)
    concrete = _concrete_dot_control_flow_helper(a, b, condition, predicate, mma, dot0, dot1)

    offsets = rows[:, None] * 64 + cols[None, :]
    # Fixup must specialize this pointer use when the helper result acquires
    # its concrete MFMA layout.
    tl.store(output_ptr + offsets, concrete)
    # The call result is still encoding-free while the Python frontend builds
    # this operation. The release remains as a deliberate layout-domain edge
    # after helper-ABI specialization and lets this store choose a fresh layout.
    generic = tlx.release_layout(concrete)
    tl.store(output_ptr + 1024 + offsets, generic)


@triton.jit
def _placeholder_mixed_results(values):
    zeros = tl.zeros(values.shape, tl.float32)
    combined = values + zeros
    reduced = tl.sum(values, axis=1)
    return combined, reduced, zeros


@triton.jit
def _placeholder_mixed_results_kernel(x_ptr, y_ptr):
    value_layout: tl.constexpr = tlx.layout(
        shape=((64, 4), (4, )),
        stride=((4, 256), (1, )),
    )
    rows = tl.arange(0, 16)
    cols = tl.arange(0, 64)
    offsets = rows[:, None] * 64 + cols[None, :]
    values = tl.load(x_ptr + offsets)
    values = tlx.require_layout(values, value_layout)
    combined, reduced, zeros = _placeholder_mixed_results(values)
    tl.store(y_ptr + offsets, combined + zeros)
    tl.store(y_ptr + 1024 + rows, reduced)


@triton.jit
def _buffer_load_contiguity_kernel(x_ptr, y_ptr):
    load_layout: tl.constexpr = tlx.layout(
        shape=((64, 4), (4, )),
        stride=((4, 256), (1, )),
    )
    offsets = tl.arange(0, 1024).to(tl.int32)
    offsets = tlx.require_layout(offsets, load_layout, pin=False)
    values = tlx.buffer_load(x_ptr, offsets, contiguity=4)
    tl.store(y_ptr + offsets, values)


@triton.jit
def _buffer_atomic_contiguity_layout_anchor_kernel(x_ptr, atomic_ptr, y_ptr):
    contiguous_layout: tl.constexpr = tlx.layout(
        shape=((64, 4), (4, )),
        stride=((4, 256), (1, )),
    )
    competing_layout: tl.constexpr = tlx.layout(
        shape=((64, 4), (4, )),
        stride=((1, 256), (64, )),
    )
    offsets = tl.arange(0, 1024).to(tl.int32)
    offsets = tlx.require_layout(offsets, contiguous_layout, pin=False)
    values = tl.load(x_ptr + offsets)
    values = tlx.require_layout(values, contiguous_layout, pin=False)
    previous = tlx.buffer_atomic_add(
        atomic_ptr,
        offsets,
        values,
        sem="relaxed",
        contiguity=2,
    )
    previous = tlx.require_layout(previous, competing_layout)
    output_offsets = tlx.require_layout(y_ptr + offsets, competing_layout)
    tl.store(output_offsets, previous)


@triton.jit
def _masked_buffer_atomic_contiguity_kernel(
    x_ptr,
    atomic_ptr,
    MASK_BOUNDARY: tl.constexpr,
):
    contiguous_layout: tl.constexpr = tlx.layout(
        shape=((64, 4), (4, )),
        stride=((4, 256), (1, )),
    )
    offsets = tl.arange(0, 1024).to(tl.int32)
    offsets = tlx.require_layout(offsets, contiguous_layout, pin=False)
    values = tl.load(x_ptr + offsets)
    values = tlx.require_layout(values, contiguous_layout, pin=False)
    tlx.buffer_atomic_add(
        atomic_ptr,
        offsets,
        values,
        mask=offsets < MASK_BOUNDARY,
        sem="relaxed",
        contiguity=2,
    )


@triton.jit
def _unsupported_i16_buffer_atomic_kernel(atomic_ptr):
    offsets = tl.arange(0, 64).to(tl.int32)
    values = tl.zeros((64, ), tl.int16)
    tlx.buffer_atomic_add(atomic_ptr, offsets, values)


def _load_tlx_gfx9_gemm_bench_module(module_name="_tlx_amd_test_gfx9_bench"):
    repo_root = Path(__file__).resolve().parents[4]
    bench_path = (repo_root / "third_party" / "tlx" / "tutorials" / "gfx9_gemm" / "a16w16" / "bench.py")
    spec = importlib.util.spec_from_file_location(module_name, bench_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_tlx_gfx9_inter_wave_bench_module(module_name="_tlx_amd_test_gfx9_inter_wave_bench"):
    repo_root = Path(__file__).resolve().parents[4]
    bench_path = (repo_root / "third_party" / "tlx" / "tutorials" / "gfx9_gemm" / "inter_wave" / "a16w16" / "bench.py")
    previous_kernel_module = sys.modules.get("matmul_kernel")
    try:
        sys.modules["matmul_kernel"] = SimpleNamespace(
            matmul=lambda _a, _b: None,
            streamk_matmul=lambda _a, _b: None,
            MIN_K=128,
            KERNEL_NAME="a16w16_8wave",
        )
        spec = importlib.util.spec_from_file_location(module_name, bench_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous_kernel_module is None:
            sys.modules.pop("matmul_kernel", None)
        else:
            sys.modules["matmul_kernel"] = previous_kernel_module


@triton.jit
def _extract_slice_kernel(x_ptr, y_ptr):
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    dot0: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=8)
    rows = tl.arange(0, 16)
    cols = tl.arange(0, 256)
    values = tl.load(x_ptr + rows[:, None] * 256 + cols[None, :])
    values = tlx.require_layout(values, dot0, pin=False)
    band = tlx.extract_slice(values, [16, 32], [0, 64])
    band_cols = tl.arange(0, 32)
    out_ptrs = y_ptr + rows[:, None] * 32 + band_cols[None, :]
    out_ptrs = tlx.require_layout(out_ptrs, dot0, pin=False)
    tl.store(out_ptrs, band)


@triton.jit
def _rematerialized_range_kernel(x_ptr, y_ptr):
    load_rows = tlx.rematerialized_range(0, 64, identity=0)
    load_cols = tlx.rematerialized_range(0, 64, identity=1)
    values = tl.load(x_ptr + load_rows[:, None] * 64 + load_cols[None, :])

    store_rows = tlx.rematerialized_range(0, 64, identity=2)
    store_cols = tlx.rematerialized_range(0, 64, identity=3)
    tl.store(y_ptr + store_rows[:, None] * 64 + store_cols[None, :], values)


@triton.jit
def _amd_late_address_compute_kernel(x_ptr, y_ptr):
    src_mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    dst_mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    src_dot0: tl.constexpr = tlx.dot_operand_layout(0, src_mma, k_width=8)
    src_dot1: tl.constexpr = tlx.dot_operand_layout(1, src_mma, k_width=8)
    rows = tl.arange(0, 64)
    reduction = tl.arange(0, 32)
    cols = tl.arange(0, 64)
    a = tlx.require_layout(
        tl.load(x_ptr + rows[:, None] * 32 + reduction[None, :]),
        src_dot0,
        pin=False,
    )
    b = tlx.require_layout(
        tl.load(x_ptr + reduction[:, None] * 64 + cols[None, :]),
        src_dot1,
        pin=False,
    )
    values = tl.dot(
        a,
        b,
        tlx.zeros((64, 64), tl.float32, layout=src_mma),
    )
    values = tlx.require_layout(
        values,
        dst_mma,
        late_address_compute=True,
    )
    offsets = rows[:, None] * 64 + cols[None, :]
    output_offsets = tlx.require_layout(y_ptr + offsets, dst_mma, pin=False)
    tl.store(output_offsets, values)


@triton.jit
def _release_dot_layout_reduce_kernel(x_ptr, y_ptr):
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    dot0: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=8)
    rows = tl.arange(0, 64)
    cols = tl.arange(0, 64)
    values = tl.load(x_ptr + rows[:, None] * 64 + cols[None, :]).to(tl.float32)
    values = tlx.require_layout(values, dot0, pin=False)
    values = tlx.release_layout(values)
    reduced = tl.sum(values, axis=1)
    tl.store(y_ptr + rows, reduced)


@triton.jit
def _invalid_release_layout_kernel(x_ptr, y_ptr):
    offsets = tl.arange(0, 64)
    values = tl.load(x_ptr + offsets)
    values = tlx.release_layout(values)
    tl.store(y_ptr + offsets, values)


@triton.jit
def _amd_scheduled_mfma_kernel(a_ptr, b_ptr, output_ptr, K_WIDTH: tl.constexpr):
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    dot0: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=K_WIDTH)
    dot1: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=K_WIDTH)
    rows = tl.arange(0, 16)
    reduction = tl.arange(0, 32)
    cols = tl.arange(0, 64)
    a = tl.load(a_ptr + rows[:, None] * 32 + reduction[None, :])
    b = tl.load(b_ptr + reduction[:, None] * 64 + cols[None, :])
    a = tlx.require_layout(a, dot0, pin=False)
    b = tlx.require_layout(b, dot1, pin=False)
    b = tlx.amd_register_resident(b, register_class="agpr", registers_per_group=4)
    acc = tl.full((16, 64), 7.0, tl.float32)
    acc = tlx.require_layout(acc, mma, pin=False)
    result = tlx.amd_scheduled_mfma(
        a,
        b,
        acc,
        resident_operand=1,
        accumulator_role="transient",
        initialize=True,
    )
    result, _ = tlx.amd_mfma_commit(result, b)
    output_offsets = output_ptr + rows[:, None] * 64 + cols[None, :]
    output_offsets = tlx.require_layout(output_offsets, mma, pin=False)
    tl.store(output_offsets, result)


@triton.jit
def _amd_scheduled_mfma_gfx942_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    PERSISTENT: tl.constexpr,
    TRANSPOSED: tl.constexpr,
    INSTR_K: tl.constexpr,
):
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=3,
        instr_shape=[16, 16, INSTR_K],
        transposed=TRANSPOSED,
        warps_per_cta=[1, 4],
    )
    dot0: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=4)
    dot1: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=4)
    rows = tl.arange(0, 32)
    reduction = tl.arange(0, 32)
    cols = tl.arange(0, 128)
    a = tlx.require_layout(
        tl.load(a_ptr + rows[:, None] * 32 + reduction[None, :]),
        dot0,
        pin=False,
    )
    b = tlx.require_layout(
        tl.load(b_ptr + reduction[:, None] * 128 + cols[None, :]),
        dot1,
        pin=False,
    )
    acc = tl.full((32, 128), 7.0, tl.float32)
    acc = tlx.require_layout(acc, mma, pin=False)
    if PERSISTENT:
        result = tlx.amd_scheduled_mfma(
            a,
            b,
            acc,
            accumulator_role="persistent",
            # On CDNA3 the compiler-generated AGPR read is not ordered
            # against the MFMA drain, so the accumulator stays in VGPRs.
            accumulator_register_class="vgpr",
            initialize=True,
        )
    else:
        result = tlx.amd_scheduled_mfma(
            a,
            b,
            acc,
            accumulator_role="transient",
            initialize=True,
        )
        result, _ = tlx.amd_mfma_commit(result, b)
    offsets = output_ptr + rows[:, None] * 128 + cols[None, :]
    offsets = tlx.require_layout(offsets, mma, pin=False)
    tl.store(offsets, result)


@triton.jit
def _amd_scheduled_mfma_32x32_gfx942_kernel(a_ptr, b_ptr, output_ptr, PERSISTENT: tl.constexpr):
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=3,
        instr_shape=[32, 32, 8],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    dot0: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=4)
    dot1: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=4)
    rows = tl.arange(0, 32)
    reduction = tl.arange(0, 16)
    cols = tl.arange(0, 128)
    a = tlx.require_layout(
        tl.load(a_ptr + rows[:, None] * 16 + reduction[None, :]),
        dot0,
        pin=False,
    )
    b = tlx.require_layout(
        tl.load(b_ptr + reduction[:, None] * 128 + cols[None, :]),
        dot1,
        pin=False,
    )
    acc = tlx.zeros((32, 128), tl.float32, layout=mma)
    if PERSISTENT:
        result = tlx.amd_scheduled_mfma(
            a,
            b,
            acc,
            accumulator_role="persistent",
            # On CDNA3 the compiler-generated AGPR read is not ordered
            # against the MFMA drain, so the accumulator stays in VGPRs.
            accumulator_register_class="vgpr",
            initialize=True,
        )
    else:
        result = tlx.amd_scheduled_mfma(
            a,
            b,
            acc,
            accumulator_role="transient",
            initialize=True,
        )
    offsets = output_ptr + rows[:, None] * 128 + cols[None, :]
    offsets = tlx.require_layout(offsets, mma, pin=False)
    tl.store(offsets, result)


@triton.jit
def _amd_scheduled_mfma_regclass_gfx942_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    ACC_CLASS: tl.constexpr,
):
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=3,
        instr_shape=[16, 16, 16],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    dot0: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=4)
    dot1: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=4)
    rows = tl.arange(0, 32)
    reduction = tl.arange(0, 32)
    cols = tl.arange(0, 128)
    a = tlx.require_layout(tl.load(a_ptr + rows[:, None] * 32 + reduction[None, :]), dot0, pin=False)
    b = tlx.require_layout(tl.load(b_ptr + reduction[:, None] * 128 + cols[None, :]), dot1, pin=False)
    acc = tlx.require_layout(tl.zeros((32, 128), tl.float32), mma, pin=False)
    result = tlx.amd_scheduled_mfma(
        a,
        b,
        acc,
        accumulator_role="persistent",
        accumulator_register_class=ACC_CLASS,
        initialize=True,
    )
    tl.store(output_ptr + rows[:, None] * 128 + cols[None, :], result)


@triton.jit
def _amd_scheduled_mfma_persistent_acc_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    USE_VGPR: tl.constexpr,
):
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    dot0: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=8)
    dot1: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=8)
    rows = tl.arange(0, 16)
    reduction = tl.arange(0, 64)
    cols = tl.arange(0, 64)
    a = tl.load(a_ptr + rows[:, None] * 64 + reduction[None, :])
    b = tl.load(b_ptr + reduction[:, None] * 64 + cols[None, :])
    a = tlx.require_layout(a, dot0, pin=False)
    b = tlx.require_layout(b, dot1, pin=False)
    a0 = tlx.extract_slice(a, [16, 32], [0, 0])
    b0 = tlx.extract_slice(b, [32, 64], [0, 0])
    acc = tlx.zeros((16, 64), tl.float32, layout=mma)
    acc = tlx.amd_scheduled_mfma(
        a0,
        b0,
        acc,
        accumulator_role="persistent",
        accumulator_register_class="vgpr" if USE_VGPR else None,
        initialize=True,
    )
    a1 = tlx.extract_slice(a, [16, 32], [0, 32])
    b1 = tlx.extract_slice(b, [32, 64], [32, 0])
    acc = tlx.amd_scheduled_mfma(
        a1,
        b1,
        acc,
        accumulator_role="persistent",
        accumulator_register_class="vgpr" if USE_VGPR else None,
    )
    output_offsets = output_ptr + rows[:, None] * 64 + cols[None, :]
    output_offsets = tlx.require_layout(output_offsets, mma, pin=False)
    tl.store(output_offsets, acc)


@triton.jit
def _load_then_restructure(base, offsets):
    value = tlx.buffer_load(base, offsets)
    value = tl.reshape(value, [4, 4, 16, 2, 2])
    value = tl.trans(value, (0, 4, 2, 3, 1))
    return tl.reshape(value, [128, 8])


@triton.jit
def _pinned_load_helper_kernel(src, dst, PHYSICAL: tl.constexpr):
    row = tl.arange(0, 4)[:, None]
    col = tl.arange(0, 256)[None, :]
    offsets = tlx.require_layout(row * 256 + col, PHYSICAL)
    value = _load_then_restructure(src, offsets)
    out_row = tl.arange(0, 128)[:, None]
    out_col = tl.arange(0, 8)[None, :]
    tl.store(dst + out_row * 8 + out_col, value)


@triton.jit
def _local_load_rematerialized_coordinates_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    buf = tlx.local_alloc((BLOCK_SIZE, ), tl.float32, 1)
    buf0 = tlx.local_view(buf, 0)
    tlx.local_store(buf0, tl.load(x_ptr + offs, mask=mask, other=0.0))
    tl.debug_barrier()
    values = tlx.local_load(buf0, rematerialize_coordinates=True)
    grouped_values = tlx.local_load(buf0, rematerialize_coordinates_group=3)
    tl.store(output_ptr + offs, values + grouped_values, mask=mask)


@triton.jit
def _local_slice_runtime_offset_kernel(x_ptr, output_ptr, row):
    value_layout: tl.constexpr = tlx.layout(
        shape=((8, 32), (2, )),
        stride=((64, 2), (1, )),
    )
    smem_layout: tl.constexpr = tlx.shared_linear_layout_encoding(
        offset_bases=[
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [1, 0],
            [2, 8],
            [4, 16],
        ],
        block_bases=[],
        alignment=8,
    )
    rows = tl.arange(0, 8)
    cols = tl.arange(0, 64)
    offsets = rows[:, None] * 64 + cols[None, :]
    offsets = tlx.require_layout(offsets, value_layout, pin=False)
    values = tl.load(x_ptr + offsets)
    buffers = tlx.local_alloc((8, 64), tl.float32, 1, layout=smem_layout)
    buffer = tlx.local_view(buffers, 0)
    tlx.local_store(buffer, values)
    tl.debug_barrier()
    view = tlx.local_slice(buffer, [row, 0], [1, 64])
    selected = tl.reshape(tlx.local_load(view, relaxed=True), (64, ))
    tl.store(output_ptr + cols, selected)


@triton.jit
def _padded_local_slice_transposed_load_kernel(x_ptr, rhs_ptr, output_ptr):
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    dot0: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=8)
    dot1: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=8)
    smem_layout: tl.constexpr = (tlx.padded_shared_layout_encoding.with_bases(
        [(512, 16)],
        [
            [1, 0],
            [2, 0],
            [0, 1],
            [0, 2],
            [4, 0],
            [0, 8],
            [8, 0],
            [0, 32],
            [0, 16],
            [0, 4],
            [0, 64],
            [0, 128],
        ],
        [16, 256],
    ))
    rows = tl.arange(0, 16)
    cols = tl.arange(0, 256)
    values = tl.load(x_ptr + rows[:, None] * 256 + cols[None, :])
    buffers = tlx.local_alloc((16, 256), tl.bfloat16, 1, layout=smem_layout)
    buffer = tlx.local_view(buffers, 0)
    tlx.local_store(buffer, values)
    tl.debug_barrier()
    band = tlx.local_load(
        tlx.local_slice(buffer, [0, 32], [16, 32]),
        layout=dot0,
        relaxed=True,
    )
    reduction = tl.arange(0, 32)
    output_cols = tl.arange(0, 64)
    rhs = tl.load(rhs_ptr + reduction[:, None] * 64 + output_cols[None, :])
    rhs = tlx.require_layout(rhs, dot1, pin=False)
    accumulator = tlx.zeros((16, 64), tl.float32, layout=mma)
    result = tlx.amd_scheduled_mfma(
        band,
        rhs,
        accumulator,
        resident_operand=1,
        accumulator_role="transient",
        initialize=True,
    )
    output_offsets = rows[:, None] * 64 + output_cols[None, :]
    output_ptrs = output_ptr + output_offsets
    output_ptrs = tlx.require_layout(output_ptrs, mma, pin=False)
    tl.store(output_ptrs, result)


def compile_for_gfx1250(fn, signature, constexprs):
    """Compile a TLX kernel for gfx1250 and return the compiled object."""
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs)
    return triton_compile(src, target=GFX1250)


@triton.jit
def _invalid_update_tensor_descriptor_kernel(x_ptr, MODE: tl.constexpr):
    desc = tl.make_tensor_descriptor(x_ptr, [16, 16], [16, 1], [16, 16])
    if MODE == 0:
        desc = tlx.update_tensor_descriptor(desc)
    elif MODE == 1:
        desc = tlx.update_tensor_descriptor(desc, pred=True, clamp_bounds=True)
    elif MODE == 2:
        desc = tlx.update_tensor_descriptor(
            desc,
            add_offsets=[0, 0],
            set_bounds=[16, 16],
            clamp_bounds=True,
        )
    elif MODE == 3:
        desc = tlx.update_tensor_descriptor(desc, add_offsets=[0])
    else:
        desc = tlx.update_tensor_descriptor(desc, add_offsets=[0, 0])


@triton.jit
def _local_reshape_kernel(
    input_ptr,
    output_ptr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    offsets = tl.arange(0, ROWS * COLS)
    values = tl.load(input_ptr + offsets)

    flat_buffers = tlx.local_alloc((ROWS * COLS, ), tl.float32, 1)
    flat = tlx.local_view(flat_buffers, 0)
    tlx.local_store(flat, values)

    reshaped = tlx.local_reshape(flat, [ROWS, COLS])
    result = tlx.local_load(reshaped)

    offs_m = tl.arange(0, ROWS)
    offs_n = tl.arange(0, COLS)
    output_offsets = offs_m[:, None] * COLS + offs_n[None, :]
    tl.store(output_ptr + output_offsets, result)


@triton.jit
def _dot_scaled_tiles_per_warp_kernel(
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    c_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
    TILES_PER_WARP: tl.constexpr,
):
    block_k_scale: tl.constexpr = BLOCK_K // SCALE_BLOCK
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_ks = tl.arange(0, block_k_scale)

    a = tl.load(a_ptr + offs_m[:, None] * BLOCK_K + offs_k[None, :])
    b = tl.load(b_ptr + offs_k[:, None] * BLOCK_N + offs_n[None, :])
    a_scale = tl.load(a_scale_ptr + offs_m[:, None] * block_k_scale + offs_ks[None, :])
    b_scale = tl.load(b_scale_ptr + offs_n[:, None] * block_k_scale + offs_ks[None, :])

    acc = tlx.dot_scaled(a, a_scale, "e5m2", b, b_scale, "e5m2", tiles_per_warp=TILES_PER_WARP)
    tl.store(c_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :], acc)


def _compile_dot_scaled_tiles_per_warp(tiles_per_warp):
    src = ASTSource(
        fn=_dot_scaled_tiles_per_warp_kernel,
        signature={
            "a_ptr": "*fp8e5",
            "b_ptr": "*fp8e5",
            "a_scale_ptr": "*i8",
            "b_scale_ptr": "*i8",
            "c_ptr": "*fp32",
        },
        constexprs={
            "BLOCK_M": 256,
            "BLOCK_N": 256,
            "BLOCK_K": 128,
            "SCALE_BLOCK": 32,
            "TILES_PER_WARP": tiles_per_warp,
        },
    )
    return triton_compile(src, target=GPUTarget("hip", "gfx1250", 32))


@triton.jit
def _require_amd_wmma_layout_kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
    offs_m = tl.arange(0, BLOCK)
    offs_n = tl.arange(0, BLOCK)
    offsets = offs_m[:, None] * BLOCK + offs_n[None, :]
    values = tl.load(x_ptr + offsets)
    values = tlx.require_amd_wmma_layout(
        values,
        version=3,
        transposed=True,
        warp_bases=((0, 2), (2, 0)),
        reg_bases=((0, 1), (1, 0)),
        instr_shape=(16, 16, 128),
    )
    offsets = tlx.require_amd_wmma_layout(
        offsets,
        version=3,
        transposed=True,
        warp_bases=((0, 2), (2, 0)),
        reg_bases=((0, 1), (1, 0)),
        instr_shape=(16, 16, 128),
    )
    tl.store(y_ptr + offsets, values)


@triton.jit
def _amd_sched_barrier_kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    values = tl.load(x_ptr + offsets)
    tlx.amd_sched_barrier()
    tl.store(y_ptr + offsets, values)


def test_amd_ttgir_schedule_env_is_cache_keyed_and_overridable(monkeypatch):
    backend = amd_compiler.HIPBackend(GFX950)
    monkeypatch.delenv("TRITON_AMD_TTGIR_SCHEDULE", raising=False)
    baseline = backend.parse_options({})

    monkeypatch.setenv("TRITON_AMD_TTGIR_SCHEDULE", "1")
    env_enabled = backend.parse_options({})
    config_disabled = backend.parse_options({"enable_sched_group_barrier_scheduler": False})

    assert not baseline.enable_sched_group_barrier_scheduler
    assert env_enabled.enable_sched_group_barrier_scheduler
    assert env_enabled.hash() != baseline.hash()
    assert not config_disabled.enable_sched_group_barrier_scheduler
    assert config_disabled.hash() == baseline.hash()


def test_amd_regalloc_codegen_options_are_cache_keyed():
    baseline = amd_compiler.HIPOptions(arch="gfx950")
    tuned = amd_compiler.HIPOptions(
        arch="gfx950",
        reverse_local_assignment=True,
        sink_insts_to_avoid_spills=True,
        regclass_priority_trumps_globalness=True,
        disable_unclustered_high_rp_reschedule=True,
    )

    assert tuned.hash() != baseline.hash()
    assert amd_compiler._get_codegen_flags(baseline) == []
    assert amd_compiler._get_codegen_flags(tuned) == [
        "greedy-reverse-local-assignment",
        "sink-insts-to-avoid-spills",
        "greedy-regclass-priority-trumps-globalness",
        "amdgpu-disable-unclustered-high-rp-reschedule",
    ]


def test_amd_sched_group_barrier_options_are_cache_keyed_and_validated():
    baseline = amd_compiler.HIPOptions(arch="gfx950")
    tuned = amd_compiler.HIPOptions(
        arch="gfx950",
        enable_sched_group_barrier_scheduler=True,
        sched_group_barrier_mfma_per_dwordx4=2,
        sched_group_barrier_required_region_count=4,
    )

    assert tuned.hash() != baseline.hash()
    assert tuned.enable_sched_group_barrier_scheduler
    assert tuned.sched_group_barrier_mfma_per_dwordx4 == 2
    assert tuned.sched_group_barrier_required_region_count == 4
    with pytest.raises(ValueError, match="sched_group_barrier_mfma_per_dwordx4 must be positive"):
        amd_compiler.HIPOptions(arch="gfx950", sched_group_barrier_mfma_per_dwordx4=0)
    with pytest.raises(ValueError, match="sched_group_barrier_required_region_count must be non-negative"):
        amd_compiler.HIPOptions(arch="gfx950", sched_group_barrier_required_region_count=-1)


@pytest.mark.parametrize(
    ("sq", "skv", "supported"),
    [
        pytest.param(8_388_544, 16_777_152, True, id="largest-aligned-safe"),
        pytest.param(8_388_608, 16_777_152, False, id="fp32-dq-span-overflow"),
        pytest.param(8_388_544, 16_777_216, False, id="bf16-kv-span-overflow"),
    ],
)
def test_d64_buffer_span_boundaries_guard_dispatch(sq, skv, supported):
    from triton.language.extra.tlx.tutorials import amd_fa_bwd

    assert amd_fa_bwd._AMD_BUFFER_MAX_ADDRESSABLE_BYTES == (1 << 31) - 1
    assert amd_fa_bwd._D64_MAX_QUERY_SEQUENCE == 8_388_544
    assert amd_fa_bwd._D64_MAX_KV_SEQUENCE == 16_777_152
    assert amd_fa_bwd._D64_MAX_QUERY_SEQUENCE * 64 * torch.float32.itemsize <= (1 << 31) - 1
    assert (amd_fa_bwd._D64_MAX_QUERY_SEQUENCE + 64) * 64 * torch.float32.itemsize > (1 << 31) - 1
    assert amd_fa_bwd._D64_MAX_KV_SEQUENCE * 64 * torch.bfloat16.itemsize <= (1 << 31) - 1
    assert (amd_fa_bwd._D64_MAX_KV_SEQUENCE + 64) * 64 * torch.bfloat16.itemsize > (1 << 31) - 1

    q_shape = (1, 8, sq, 64)
    k_shape = (1, 1, skv, 64)
    assert amd_fa_bwd._is_supported_d64_shape(q_shape, k_shape) is supported
    if supported:
        dispatch = amd_fa_bwd._select_d64_dispatch(q_shape, k_shape, False)
        assert dispatch.family == "noncausal_direct_n256"
    else:
        with pytest.raises(ValueError, match="unsupported D64 dispatch shapes"):
            amd_fa_bwd._select_d64_dispatch(q_shape, k_shape, False)


@pytest.mark.parametrize(
    ("q_shape", "k_shape", "causal", "dispatch_kwargs", "message"),
    [
        pytest.param(
            (1, 1, 256, 128),
            (1, 1, 256, 128),
            False,
            {
                "family": "noncausal_direct_n256",
                "owner_rows": 32,
                "key_rows": 256,
                "kv_splits": 1,
            },
            "unsupported D64 dispatch shapes",
            id="shape",
        ),
        pytest.param(
            (1, 1, 4096, 64),
            (1, 1, 4096, 64),
            True,
            {
                "family": "noncausal_direct_n256",
                "owner_rows": 32,
                "key_rows": 256,
                "kv_splits": 1,
            },
            "requires noncausal attention",
            id="causal-family",
        ),
        pytest.param(
            (1, 8, 4096, 64),
            (1, 1, 4096, 64),
            True,
            {
                "family": "causal_scheduled_gqa8",
                "owner_rows": 256,
                "key_rows": 128,
                "kv_splits": 1,
                "selected_causal": True,
                "stat_mode": 1,
                "dq_logical_n": 32,
            },
            "kv_splits",
            id="gqa-splits",
        ),
        pytest.param(
            (1, 1, 4096, 64),
            (1, 1, 4096, 64),
            True,
            {
                "family": "causal_scheduled_mha",
                "owner_rows": 192,
                "key_rows": 64,
                "kv_splits": 1,
                "selected_causal": True,
                "stat_mode": 1,
                "dq_logical_n": 32,
            },
            "stat_mode",
            id="mha-stat-mode",
        ),
        pytest.param(
            (1, 1, 4096, 64),
            (1, 1, 4096, 64),
            False,
            {
                "family": "unknown",
                "owner_rows": 32,
                "key_rows": 256,
                "kv_splits": 1,
            },
            "unknown D64 dispatch family",
            id="unknown-family",
        ),
    ],
)
def test_d64_dispatch_validation_rejects_invalid_contracts(q_shape, k_shape, causal, dispatch_kwargs, message):
    from triton.language.extra.tlx.tutorials import amd_fa_bwd

    assert hasattr(amd_fa_bwd,
                   "_validate_d64_dispatch"), ("D64 dispatch validation must not depend on removable Python asserts")
    dispatch = amd_fa_bwd._D64Dispatch(**dispatch_kwargs)

    with pytest.raises(ValueError, match=message):
        amd_fa_bwd._validate_d64_dispatch(q_shape, k_shape, causal, dispatch)


def test_d64_dispatch_validation_rejects_incomplete_dq_launch_plan():
    from triton.language.extra.tlx.tutorials import amd_fa_bwd

    q_shape = (4, 48, 4096, 64)
    k_shape = (4, 6, 4096, 64)
    dispatch = amd_fa_bwd._select_d64_dispatch(
        q_shape,
        k_shape,
        True,
        arch="gfx950:sramecc+:xnack-",
        cu_count=256,
        sm_scale=0.125,
        bases_aligned_16=True,
    )
    malformed = dataclasses.replace(
        dispatch,
        dq_launches=(amd_fa_bwd._D64DQLaunch(1, False, 0, 0, 3, 0), ),
    )

    with pytest.raises(ValueError, match="dq_launches must match"):
        amd_fa_bwd._validate_d64_dispatch(q_shape, k_shape, True, malformed)


@pytest.mark.parametrize(
    ("q_shape", "k_shape", "changes", "message"),
    [
        pytest.param(
            (1, 25, 4096, 64),
            (1, 25, 4096, 64),
            {"dq_use_xcd": True},
            "dq_use_xcd",
            id="dq-xcd",
        ),
        pytest.param(
            (4, 40, 4096, 64),
            (4, 5, 4096, 64),
            {"gqa_grid_mode": "xcd"},
            "GQA XCD grid requires",
            id="gqa-xcd-grid",
        ),
        pytest.param(
            (4, 48, 1024, 64),
            (4, 6, 2048, 64),
            {"dkdv_lifetime": "independent_d32"},
            "dkdv_lifetime",
            id="gqa-lifetime",
        ),
        pytest.param(
            (4, 48, 4096, 64),
            (4, 6, 4096, 64),
            {"cyclic_query_split": True},
            "cyclic_query_split",
            id="gqa-cyclic",
        ),
    ],
)
def test_d64_dispatch_validation_rejects_incompatible_selected_modes(q_shape, k_shape, changes, message):
    from triton.language.extra.tlx.tutorials import amd_fa_bwd

    dispatch = amd_fa_bwd._select_d64_dispatch(
        q_shape,
        k_shape,
        True,
        arch="gfx950:sramecc+:xnack-",
        cu_count=256,
        sm_scale=0.125,
        bases_aligned_16=True,
    )
    assert dispatch.selected_causal
    malformed = dataclasses.replace(dispatch, **changes)

    with pytest.raises(ValueError, match=message):
        amd_fa_bwd._validate_d64_dispatch(q_shape, k_shape, True, malformed)


@pytest.mark.parametrize(
    ("q_shape", "k_shape", "causal", "family"),
    [
        pytest.param(
            (2, 32, 16384, 64),
            (2, 32, 16384, 64),
            False,
            "noncausal_fused_n256",
            id="mha-square-16k-noncausal",
        ),
        pytest.param(
            (2, 32, 16384, 64),
            (2, 32, 16384, 64),
            True,
            "causal_scheduled_mha",
            id="mha-square-16k-causal",
        ),
        pytest.param(
            (2, 32, 16384, 64),
            (2, 4, 16384, 64),
            False,
            "noncausal_fused_n256",
            id="gqa8-square-16k-noncausal",
        ),
        pytest.param(
            (2, 32, 16384, 64),
            (2, 4, 16384, 64),
            True,
            "causal_scheduled_gqa8",
            id="gqa8-square-16k-causal",
        ),
        pytest.param(
            (4, 48, 4096, 64),
            (4, 6, 4096, 64),
            True,
            "causal_scheduled_gqa8",
            id="gqa8-square-4k-causal",
        ),
        pytest.param(
            (4, 48, 4096, 64),
            (4, 6, 16384, 64),
            True,
            "causal_scheduled_gqa8",
            id="gqa8-rect-4k-16k-causal",
        ),
        pytest.param(
            (4, 48, 4096, 64),
            (4, 6, 8192, 64),
            True,
            "causal_scheduled_gqa8",
            id="gqa8-rect-4k-8k-causal",
        ),
        pytest.param(
            (4, 48, 4096, 64),
            (4, 6, 12288, 64),
            True,
            "causal_scheduled_gqa8",
            id="gqa8-rect-4k-12k-causal",
        ),
    ],
)
def test_d64_dispatch_contract_is_ci_discovered(q_shape, k_shape, causal, family):
    from triton.language.extra.tlx.tutorials import amd_fa_bwd

    dispatch = amd_fa_bwd._select_d64_dispatch(
        q_shape,
        k_shape,
        causal,
        arch="gfx950:sramecc+:xnack-",
        cu_count=256,
        sm_scale=0.125,
        bases_aligned_16=True,
    )

    assert dispatch.family == family
    assert dispatch.selected_causal is causal
    amd_fa_bwd._validate_d64_dispatch(q_shape, k_shape, causal, dispatch)


def test_amd_fa_cluster_rejects_unsupported_inputs():
    q = torch.empty((1, 1, 8, 64), dtype=torch.float16)
    with pytest.raises(ValueError, match="same shape"):
        _validate_amd_fa_cluster_inputs(q, torch.empty((1, 1, 7, 64), dtype=q.dtype), q)
    with pytest.raises(ValueError, match="only FP16/BF16"):
        _validate_amd_fa_cluster_inputs(q.float(), q.float(), q.float())
    with pytest.raises(ValueError, match="BLOCK_M"):
        _validate_amd_fa_cluster_tiles(64, 64)
    with pytest.raises(ValueError, match="BLOCK_N"):
        _validate_amd_fa_cluster_tiles(256, 128)


@pytest.mark.parametrize(
    ("dtype", "n_ctx", "head_dim", "causal", "config", "expected"),
    [
        pytest.param(torch.float16, 2048, 128, False, {}, -1, id="short-row"),
        pytest.param(torch.float16, 4096, 128, False, {}, 263, id="n4096"),
        pytest.param(torch.float16, 8192, 128, False, {}, 263, id="n8192"),
        pytest.param(torch.bfloat16, 4096, 128, False, {}, -1, id="bf16-n4096"),
        pytest.param(torch.bfloat16, 16384, 128, False, {}, 263, id="bf16-n16384"),
        pytest.param(torch.float16, 4096, 128, True, {}, -1, id="causal"),
        pytest.param(torch.float16, 4096, 64, False, {}, -1, id="d64"),
        pytest.param(torch.float16, 4096, 128, False, {"use_autotune": False}, -1, id="explicit-config"),
        pytest.param(torch.float16, 4096, 128, False, {"block_m": 128}, -1, id="bm128"),
        pytest.param(torch.float16, 4096, 128, False, {"block_n": 32}, -1, id="bn32"),
        pytest.param(torch.float16, 4096, 128, False, {"waves_per_eu": 0}, -1, id="wpe0"),
    ],
)
def test_amd_fa_cluster_selects_static_k_row_stride(dtype, n_ctx, head_dim, causal, config, expected):
    """The regular kernel specializes only the measured long noncausal K rows."""
    q_strides = (64 * n_ctx * 257, n_ctx * 257, 257, 1)
    k_strides = (64 * n_ctx * 263, n_ctx * 263, 263, 1)
    q = SimpleNamespace(shape=(1, 64, n_ctx, head_dim), dtype=dtype, stride=lambda dim: q_strides[dim])
    k = SimpleNamespace(shape=(1, 64, n_ctx, head_dim), dtype=dtype, stride=lambda dim: k_strides[dim])
    launch = {
        "use_autotune": True,
        "block_m": 256,
        "block_n": 64,
        "num_warps": 8,
        "waves_per_eu": 2,
        **config,
    }

    assert _amd_fa_cluster_module._cluster_static_k_row_stride(q, k, causal, **launch) == expected


@pytest.mark.parametrize(
    ("dtype", "n_ctx", "expected"),
    [
        pytest.param(torch.float16, 4096, 263, id="selected-fp16"),
        pytest.param(torch.bfloat16, 4096, -1, id="dynamic-bf16-n4096"),
        pytest.param(torch.bfloat16, 16384, 263, id="selected-bf16-n16384"),
    ],
)
def test_amd_fa_cluster_launch_forwards_static_k_row_stride(monkeypatch, dtype, n_ctx, expected):
    """The public wrapper forwards the selected stride to the compiled kernel."""
    strides = (64 * n_ctx * 263, n_ctx * 263, 263, 1)
    tensor = SimpleNamespace(shape=(1, 64, n_ctx, 128), dtype=dtype, stride=lambda dim: strides[dim])
    captured = {}

    class CaptureKernel:

        def __getitem__(self, grid):
            captured["grid"] = grid

            def launch(*args, **kwargs):
                captured["kwargs"] = kwargs

            return launch

    kernel = CaptureKernel()
    monkeypatch.setattr(_amd_fa_cluster_module, "_validate_cluster_inputs", lambda q, k, v: None)
    monkeypatch.setattr(_amd_fa_cluster_module.torch, "empty_like", lambda q: q)
    monkeypatch.setattr(_amd_fa_cluster_module, "_attn_fwd_cluster_pipeline_autotuned", kernel)

    out = _amd_fa_cluster_module.flash_attn_cluster_pipeline(tensor, tensor, tensor, 1.3, False)

    assert out is tensor
    assert captured["grid"] == (n_ctx // 256, 64, 1)
    assert captured["kwargs"]["STATIC_STRIDE_KN"] == expected
    assert captured["kwargs"]["enable_sched_group_barrier_scheduler"] is False


def test_amd_fa_result_war_barriers_depend_on_all_mfma_groups_gfx950():
    """Each lightweight slot handoff remains data-dependent on its last consumers."""
    qk_barrier = _amd_fa_cluster_module._attn_qk_war_barrier_relaxed
    pv_barrier = _amd_fa_cluster_module._attn_pv_war_barrier_relaxed

    @triton.jit
    def result_war_barriers(qk_ptr, acc_ptr):
        rows = tl.arange(0, 128)
        qk_cols = tl.arange(0, 64)
        acc_cols = tl.arange(0, 128)
        qk_offsets = rows[:, None] * 64 + qk_cols[None, :]
        acc_offsets = rows[:, None] * 128 + acc_cols[None, :]
        qk = tl.load(qk_ptr + qk_offsets)
        acc = tl.load(acc_ptr + acc_offsets)
        qk_barrier(qk)
        pv_barrier(acc)
        tl.store(qk_ptr + qk_offsets, qk)
        tl.store(acc_ptr + acc_offsets, acc)

    compiled = compile_for_gfx950(
        result_war_barriers,
        signature={"qk_ptr": "*fp32", "acc_ptr": "*fp32"},
        constexprs={},
    )
    barriers = re.findall(r'tt\.elementwise_inline_asm "[^"\n]*s_barrier"[^\n]+', compiled.asm["ttir"])
    assert len(barriers) == 2
    qk_constraints = 'constraints = "=s,=s,=s,=s,v,v,v,v,v,v,v,v"'
    pv_constraints = 'constraints = "=s,=s,=s,=s,' + ','.join(["v"] * 16) + '"'
    assert sum(qk_constraints in barrier for barrier in barriers) == 1
    assert sum(pv_constraints in barrier for barrier in barriers) == 1
    assert all("s_waitcnt lgkmcnt(0)" not in barrier for barrier in barriers)
    assert all("packed_element = 4 : i32" in barrier for barrier in barriers)


def test_warp_predicate_lowers_to_amd_exec_mask_gfx950():
    compiled = compile_for_gfx950(
        _warp_predicate_kernel,
        signature={
            "x_ptr": "*fp32",
            "lhs_ptr": "*fp32",
            "rhs_ptr": "*fp32",
            "side_ptr": "*fp32",
        },
        constexprs={"size": 256},
    )
    assert "ttg.warp_predicate" in compiled.asm["ttgir"]
    amdgcn = compiled.asm["amdgcn"]
    assert "s_and_saveexec_b64" in amdgcn
    assert "s_cbranch_execz" in amdgcn


def test_nested_warp_predicate_lowers_to_amd_exec_mask_gfx950():
    compiled = compile_for_gfx950(
        _nested_warp_predicate_kernel,
        signature={"x_ptr": "*fp32", "output_ptr": "*fp32"},
        constexprs={},
    )
    assert compiled.asm["ttgir"].count("ttg.warp_predicate") == 2
    assert "amdgcn" in compiled.asm


def test_warp_predicate_rejects_cross_wave_reduce_gfx950():
    with pytest.raises(RuntimeError, match="region reduction axis must be warp-local"):
        compile_for_gfx950(
            _warp_predicate_cross_wave_reduce_kernel,
            signature={"x_ptr": "*fp32", "output_ptr": "*fp32"},
            constexprs={"size": 256},
        )


def test_warp_predicate_accepts_scalar_warp_local_reduce_gfx950():
    compiled = compile_for_gfx950(
        _warp_predicate_warp_local_reduce_kernel,
        signature={"x_ptr": "*fp32", "output_ptr": "*fp32"},
        constexprs={},
    )
    assert "ttg.warp_predicate" in compiled.asm["ttgir"]
    assert "s_barrier" not in compiled.asm["amdgcn"]


def test_warp_predicate_rejects_lane_divergent_reduce_gfx950():
    with pytest.raises(RuntimeError, match="cross-lane operation tt.reduce requires a wave-uniform predicate"):
        compile_for_gfx950(
            _warp_predicate_lane_divergent_reduce_kernel,
            signature={"x_ptr": "*fp32", "output_ptr": "*fp32"},
            constexprs={},
        )


def test_async_local_slice_dot_compiles_gfx950():
    compiled = compile_for_gfx950(
        _async_local_slice_dot_kernel,
        signature={"q_ptr": "*fp16", "k_ptr": "*fp16", "output_ptr": "*fp32"},
        constexprs={},
    )
    assert "ttg.memdesc_subslice" in compiled.asm["ttgir"]
    assert "v_mfma" in compiled.asm["amdgcn"]


def test_amd_warp_votes_lower_without_public_ballot_gfx950():
    compiled = compile_for_gfx950(
        _warp_vote_kernel,
        signature={
            "x_ptr": "*i32",
            "all_ptr": "*i32",
            "any_ptr": "*i32",
            "BLOCK": "constexpr",
        },
        constexprs={"BLOCK": 64},
    )
    assert 'ttg.warp_vote' in compiled.asm["ttgir"]
    assert '"all"' in compiled.asm["ttgir"]
    assert '"any"' in compiled.asm["ttgir"]
    assert "warp_ballot" not in compiled.asm["ttgir"]
    assert "llvm.amdgcn.ballot" in compiled.asm["llir"]


def test_amd_warp_vote_rejects_scalar_predicate():
    with pytest.raises(CompilationError, match="warp_all expects a distributed tensor predicate"):
        compile_for_gfx950(
            _warp_vote_scalar_predicate_kernel,
            signature={"output": "*i32"},
            constexprs={},
        )


def test_amd_warp_vote_rejects_multiple_elements_per_lane_gfx950():
    with pytest.raises(RuntimeError, match="predicate must distribute exactly one element per lane"):
        compile_for_gfx950(
            _warp_vote_kernel,
            signature={
                "x_ptr": "*i32",
                "all_ptr": "*i32",
                "any_ptr": "*i32",
                "BLOCK": "constexpr",
            },
            constexprs={"BLOCK": 512},
        )


def test_shared_helper_accepts_distinct_concrete_layouts_gfx950():
    compiled = compile_for_gfx950(
        _shared_concrete_helper_kernel,
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32"},
        constexprs={},
    )
    assert "amdgcn" in compiled.asm


def test_mixed_helper_result_abi_compiles_gfx950():
    compiled = compile_for_gfx950(
        _mixed_helper_results_kernel,
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "condition": "i1"},
        constexprs={},
    )
    assert "amdgcn" in compiled.asm


def test_concrete_layout_while_compiles_gfx950():
    """Fixup synchronizes both carried-value domains of a dynamic while."""
    compiled = compile_for_gfx950(
        _concrete_layout_while_kernel,
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "count": "i32"},
        constexprs={},
    )
    assert "amdgcn" in compiled.asm


def test_slice_layout_rejects_out_of_range_dimension_gfx950():
    mma = tlx.amd_mfma_layout(4, [16, 16, 32], True, [1, 4])
    rank_one = tlx.slice_layout(mma, dim=1)
    cases = [
        (tlx.slice_layout(mma, dim=2), r"slice dim=2 must be less than the parent rank=2"),
        (tlx.slice_layout(rank_one, dim=0), r"parent layout must have at least rank >= 2"),
    ]
    for invalid, error in cases:
        with pytest.raises(CompilationError, match=error):
            compile_for_gfx950(
                _slice_layout_validation_kernel,
                signature={"output": "*fp32"},
                constexprs={"layout": invalid},
            )


def test_concrete_helper_control_flow_release_compiles_gfx950():
    compiled = compile_for_gfx950(
        _concrete_helper_release_kernel,
        signature={
            "a_ptr": "*fp16",
            "b_ptr": "*fp16",
            "output_ptr": "*fp32",
            "condition": "i1",
        },
        constexprs={},
    )
    assert "tlx.release_layout" in compiled.asm["ttir"]
    ttgir = compiled.asm["ttgir"]
    assert "ttg.warp_predicate" in ttgir
    assert ": (tensor<16x64xi1, #mma>, tensor<16x64xf32, #mma>)" in ttgir
    assert "v_mfma" in compiled.asm["amdgcn"]


def test_placeholder_mixed_and_constant_helper_results_compile_gfx950():
    compiled = compile_for_gfx950(
        _placeholder_mixed_results_kernel,
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32"},
        constexprs={},
    )
    assert "amdgcn" in compiled.asm
    assert "#tlx.user_layout" not in compiled.asm["ttgir"]
    assert "#tlx.no_verify_layout" not in compiled.asm["ttgir"]


def test_buffer_load_contiguity_vectorizes_gfx950():
    compiled = compile_for_gfx950(
        _buffer_load_contiguity_kernel,
        signature={"x_ptr": "*bf16", "y_ptr": "*bf16"},
        constexprs={},
    )

    ttgir = compiled.asm["ttgir"]
    assert "amdg.buffer_load" in ttgir
    assert "contiguity = 4" in ttgir
    assert "buffer_load_dwordx2" in compiled.asm["amdgcn"]


@triton.jit
def _plain_unaligned_vector_load_kernel(
    x_ptr,
    y_ptr,
    OFFSET: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    values = tl.load(x_ptr + OFFSET + offsets)
    tl.store(y_ptr + offsets, values)


@pytest.mark.parametrize(
    "pointer_type,block_size,expected_load",
    [
        ("*fp16", 1024, "buffer_load_dwordx2"),
        ("*fp16", 2048, "buffer_load_dwordx4"),
        ("*i8", 4096, "buffer_load_dwordx4"),
    ],
)
def test_plain_unaligned_vector_load_is_automatic_gfx950(pointer_type, block_size, expected_load):
    """A plain gfx950 load vectorizes without a per-kernel opt-in."""
    compiled = compile_for_gfx950(
        _plain_unaligned_vector_load_kernel,
        signature={"x_ptr": pointer_type, "y_ptr": pointer_type},
        constexprs={"OFFSET": 1, "BLOCK_SIZE": block_size},
    )

    assert expected_load in compiled.asm["amdgcn"]


def test_buffer_atomic_contiguity_preserves_layout_gfx950():
    compiled = compile_for_gfx950(
        _buffer_atomic_contiguity_layout_anchor_kernel,
        signature={"x_ptr": "*bf16", "atomic_ptr": "*bf16", "y_ptr": "*bf16"},
        constexprs={},
    )

    ttgir = compiled.asm["ttgir"]
    assert "amdg.buffer_atomic_rmw" in ttgir
    assert "contiguity = 2" in ttgir
    assert "tlx.preserve_layout" in ttgir
    atomic = re.search(
        r"(?P<result>%[\w.]+) = amdg\.buffer_atomic_rmw.*"
        r"tlx\.preserve_layout.*: tensor<1024xbf16, (?P<layout>#[\w.]+)>",
        ttgir,
    )
    assert atomic is not None
    conversion = re.search(
        rf"ttg\.convert_layout {re.escape(atomic.group('result'))} : "
        rf"tensor<1024xbf16, {re.escape(atomic.group('layout'))}> -> "
        r"tensor<1024xbf16, (?P<layout>#[\w.]+)>",
        ttgir,
    )
    assert conversion is not None
    assert conversion.group("layout") != atomic.group("layout")
    assert "buffer_atomic_pk_add_bf16" in compiled.asm["amdgcn"]


def test_masked_buffer_atomic_contiguity_vectorizes_gfx950():
    compiled = compile_for_gfx950(
        _masked_buffer_atomic_contiguity_kernel,
        signature={
            "x_ptr": "*bf16",
            "atomic_ptr": "*bf16",
            "MASK_BOUNDARY": "constexpr",
        },
        constexprs={"MASK_BOUNDARY": 512},
    )
    assert "buffer_atomic_pk_add_bf16" in compiled.asm["amdgcn"]


def test_masked_buffer_atomic_rejects_scalar_bf16_gfx950(capfd):
    with pytest.raises(RuntimeError):
        compile_for_gfx950(
            _masked_buffer_atomic_contiguity_kernel,
            signature={
                "x_ptr": "*bf16",
                "atomic_ptr": "*bf16",
                "MASK_BOUNDARY": "constexpr",
            },
            constexprs={"MASK_BOUNDARY": 511},
        )
    assert ("16-bit buffer atomics require two contiguous elements" in capfd.readouterr().err)


def test_buffer_atomic_rejects_unsupported_i16_gfx950():
    with pytest.raises(CompilationError, match="buffer_atomic_add supports only"):
        compile_for_gfx950(
            _unsupported_i16_buffer_atomic_kernel,
            signature={"atomic_ptr": "*i16"},
            constexprs={},
        )


def test_pinned_buffer_load_layout_survives_optimization_gfx950():
    from triton.language.extra.tlx.tutorials.amd_fa_bwd import (
        _attn_bwd_dq_native_convert_kernel, )

    compiled = compile_for_gfx950(
        _attn_bwd_dq_native_convert_kernel,
        signature={"DQ_ACC": "*bf16", "DQ": "*bf16"},
        constexprs={"N": 128, "D": 128, "BLOCK_M": 128},
    )

    ttgir = compiled.asm["ttgir"]
    amdgcn = compiled.asm["amdgcn"]
    assert "contiguity = 4" in ttgir
    assert "ttg.convert_layout" in ttgir
    assert amdgcn.count("buffer_load_dwordx2") == 16
    assert amdgcn.count("v_permlane16_swap_b32") == 16


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_gqa_oversized_batches_rebase_buffer_offsets_gfx950(causal):
    from triton.language.extra.tlx.tutorials.amd_fa_bwd import (
        _attn_bwd_dkdv_dq_d128_gqa_kernel, )

    # At N=16384 and D=128, 512 BF16 heads exactly fill the signed 32-bit
    # byte-offset range; 520 exercises the per-tile 64-bit pointer rebasing
    # path without allocating multi-gigabyte test tensors.
    block_m = 16
    block_n = 256
    compiled = compile_for_gfx950(
        _attn_bwd_dkdv_dq_d128_gqa_kernel,
        signature={
            "Q": "*bf16",
            "K": "*bf16",
            "V": "*bf16",
            "DO": "*bf16",
            "LSE": "*fp32",
            "Delta": "*fp32",
            "DQ_ACC": "*bf16",
            "DK": "*bf16",
            "DV": "*bf16",
        },
        constexprs={
            "SM_SCALE": 0.125,
            "IS_CAUSAL": causal,
            "HQ": 520,
            "HK": 520,
            "N": 16384,
            "D": 128,
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
        },
    )

    ttir = compiled.asm["ttir"]
    ttgir = compiled.asm["ttgir"]
    dummy_clamps = re.findall(r"arith\.maxsi %dq_step, (?P<floor>%[\w_]+)", ttir)
    assert len(dummy_clamps) == 1
    if causal:
        assert dummy_clamps[0] != "%c0_i32"
        first_active_stride = block_n // block_m
        assert re.search(
            rf"^\s*{re.escape(dummy_clamps[0])} = arith\.muli "
            rf"%pid_n(?:_\d+)?, %c{first_active_stride}_i32",
            ttir,
            re.MULTILINE,
        )
    else:
        assert dummy_clamps[0] == "%c0_i32"
    assert ttgir.count("tlx.rematerialize_coordinates_group = 21 : i32") == (9 if causal else 0)
    assert "amdgcn" in compiled.asm


def test_gqa_oversized_head_rebases_native_conversion_gfx950():
    from triton.language.extra.tlx.tutorials.amd_fa_bwd import (
        _attn_bwd_dq_native_convert_kernel, )

    compiled = compile_for_gfx950(
        _attn_bwd_dq_native_convert_kernel,
        signature={"DQ_ACC": "*bf16", "DQ": "*bf16"},
        constexprs={
            "N": (1 << 23) + 256,
            "D": 128,
            "BLOCK_M": 128,
        },
    )

    assert "amdgcn" in compiled.asm


def test_extract_slice_compiles_gfx950():
    compiled = compile_for_gfx950(
        _extract_slice_kernel,
        signature={"x_ptr": "*bf16", "y_ptr": "*bf16"},
        constexprs={},
    )
    assert "amdg.extract_slice" in compiled.asm["ttir"]
    assert "amdgcn" in compiled.asm


def test_rematerialized_range_compiles_gfx950():
    compiled = compile_for_gfx950(
        _rematerialized_range_kernel,
        signature={"x_ptr": "*bf16", "y_ptr": "*bf16"},
        constexprs={},
    )
    assert compiled.asm["ttir"].count("amdg.rematerialized_range") == 4
    assert compiled.asm["ttgir"].count("amdg.rematerialized_range") == 4
    # Each range layout depends on one distributed coordinate; do not anchor
    # the zero-basis lane/warp dimension.
    assert compiled.asm["llir"].count('asm sideeffect "", "=v,0"') == 4
    assert "amdg.rematerialized_range" not in compiled.asm["llir"]
    assert "amdgcn" in compiled.asm


def test_amd_late_address_compute_compiles_gfx950():
    compiled = compile_for_gfx950(
        _amd_late_address_compute_kernel,
        signature={"x_ptr": "*bf16", "y_ptr": "*bf16"},
        constexprs={},
    )
    assert "tlx.rematerialize_coordinates" in compiled.asm["ttir"]
    assert "tlx.rematerialize_coordinates" in compiled.asm["ttgir"]
    assert compiled.asm["llir"].count('asm sideeffect "", "=v,0"') >= 2
    assert "amdgcn" in compiled.asm


@triton.jit
def _amd_register_class_anchor_kernel(x_ptr, y_ptr, REGISTER_CLASS: tl.constexpr):
    offsets = tl.arange(0, 2048)
    values = tl.load(x_ptr + offsets)
    values = tlx.amd_register_class_anchor(values, register_class=REGISTER_CLASS)
    tl.store(y_ptr + offsets, values)


@pytest.mark.parametrize(
    ("register_class", "element_type"),
    [
        pytest.param("vgpr", "fp32", id="vgpr-fp32"),
        pytest.param("vgpr", "fp16", id="vgpr-fp16"),
        pytest.param("agpr", "fp32", id="agpr-fp32"),
    ],
)
def test_amd_register_class_anchor_compiles_gfx950(register_class, element_type):
    compiled = compile_for_gfx950(
        _amd_register_class_anchor_kernel,
        signature={"x_ptr": f"*{element_type}", "y_ptr": f"*{element_type}"},
        constexprs={"REGISTER_CLASS": register_class},
    )
    ttir = compiled.asm["ttir"]
    assert ttir.count("amdg.register_class_anchor") == 1
    assert f'class "{register_class}"' in ttir
    assert "groups" not in ttir
    assert "tt.elementwise_inline_asm" not in ttir
    assert "amdg.register_resident" not in ttir
    llir = compiled.asm["llir"]
    assert "amdg.register_class_anchor" not in llir
    register_constraint = "a" if register_class == "agpr" else "v"
    constraint = f'"={register_constraint},0"'
    anchor_asm = [line for line in llir.splitlines() if constraint in line]
    expected_asm = 4 if element_type == "fp16" else 8
    assert len(anchor_asm) == expected_asm
    assert all("sideeffect" in line for line in anchor_asm)


@triton.jit
def _invalid_amd_register_class_anchor_kernel(
    x_ptr,
    y_ptr,
    REGISTER_CLASS: tl.constexpr,
):
    offsets = tl.arange(0, 1024)
    values = tl.load(x_ptr + offsets)
    values = tlx.amd_register_class_anchor(
        values,
        register_class=REGISTER_CLASS,
    )
    tl.store(y_ptr + offsets, values)


@pytest.mark.parametrize(
    ("register_class", "element_type", "message"),
    [
        pytest.param("sgpr", "fp32", 'register_class must be either "agpr" or "vgpr"', id="register-class"),
        pytest.param("vgpr", "i8", "value elements must be 16 or 32 bits", id="element-width"),
    ],
)
def test_amd_register_class_anchor_rejects_invalid_contract(register_class, element_type, message):
    with pytest.raises(CompilationError, match=message):
        compile_for_gfx950(
            _invalid_amd_register_class_anchor_kernel,
            signature={"x_ptr": f"*{element_type}", "y_ptr": f"*{element_type}"},
            constexprs={
                "REGISTER_CLASS": register_class,
            },
        )


def test_release_dot_layout_reduce_compiles_gfx950():
    compiled = compile_for_gfx950(
        _release_dot_layout_reduce_kernel,
        signature={"x_ptr": "*bf16", "y_ptr": "*fp32"},
        constexprs={},
    )
    assert "tlx.release_layout" in compiled.asm["ttir"]
    assert "tt.reduce" in compiled.asm["ttgir"]
    assert "amdgcn" in compiled.asm


def test_release_layout_rejects_unencoded_source():
    with pytest.raises(CompilationError, match="release_layout requires an explicit source layout"):
        compile_for_gfx950(
            _invalid_release_layout_kernel,
            signature={"x_ptr": "*fp32", "y_ptr": "*fp32"},
            constexprs={},
        )


def test_amd_scheduled_mfma_compiles_gfx950():
    compiled = compile_for_gfx950(
        _amd_scheduled_mfma_kernel,
        signature={
            "a_ptr": "*bf16",
            "b_ptr": "*bf16",
            "output_ptr": "*fp32",
            "K_WIDTH": "constexpr",
        },
        constexprs={"K_WIDTH": 8},
    )
    assert "amdg.register_resident" in compiled.asm["ttir"]
    assert 'class "agpr" groups 4' in compiled.asm["ttir"]
    assert "amdg.scheduled_mfma" in compiled.asm["ttir"]
    assert "amdg.mfma_commit" in compiled.asm["ttir"]
    assert "=a,0" in compiled.asm["llir"]
    assert "@llvm.amdgcn.mfma.f32.16x16x32.bf16" in compiled.asm["llir"]
    assert 'asm sideeffect "v_mfma' not in compiled.asm["llir"]
    assert "v_mfma_f32_16x16x32_bf16" in compiled.asm["amdgcn"]
    assert "s_nop 5" in compiled.asm["llir"]


@pytest.mark.parametrize("elem_ty", ["bf16", "fp16"])
@pytest.mark.parametrize("persistent", [False, True])
def test_amd_scheduled_mfma_compiles_gfx942(elem_ty, persistent):
    compiled = compile_for_gfx942(
        _amd_scheduled_mfma_gfx942_kernel,
        signature={
            "a_ptr": f"*{elem_ty}",
            "b_ptr": f"*{elem_ty}",
            "output_ptr": "*fp32",
            "PERSISTENT": "constexpr",
            "TRANSPOSED": "constexpr",
            "INSTR_K": "constexpr",
        },
        constexprs={"PERSISTENT": persistent, "TRANSPOSED": True, "INSTR_K": 16},
    )
    asm_ty = "f16" if elem_ty == "fp16" else "bf16"
    assert "amdg.scheduled_mfma" in compiled.asm["ttir"]
    assert f"v_mfma_f32_16x16x16_{asm_ty}" in compiled.asm["amdgcn"]
    if persistent:
        assert f'asm sideeffect "s_nop 3\\0Av_mfma_f32_16x16x16_{asm_ty}' in compiled.asm["llir"]
        # 8 passes + 3 = 11 wait states for a CDNA3 16x16x16 result read.
        assert 'asm sideeffect "s_nop 10"' in compiled.asm["llir"]
        # gfx942 has to pin the accumulator to VGPRs: with AGPRs, LLVM emits
        # v_accvgpr_read of the asm result ahead of the source-level drain,
        # reading it before the MFMA retires.
        assert '"=&v,v,v"' in compiled.asm["llir"]
        assert '"=&a,v,v"' not in compiled.asm["llir"]
    else:
        intrinsic_ty = "f16" if elem_ty == "fp16" else "bf16.1k"
        assert f"@llvm.amdgcn.mfma.f32.16x16x16{intrinsic_ty}" in compiled.asm["llir"]
        assert 'asm sideeffect "v_mfma' not in compiled.asm["llir"]


@pytest.mark.parametrize("elem_ty", ["bf16", "fp16"])
@pytest.mark.parametrize("persistent", [False, True])
def test_amd_scheduled_mfma_32x32_compiles_gfx942(elem_ty, persistent):
    compiled = compile_for_gfx942(
        _amd_scheduled_mfma_32x32_gfx942_kernel,
        signature={
            "a_ptr": f"*{elem_ty}",
            "b_ptr": f"*{elem_ty}",
            "output_ptr": "*fp32",
            "PERSISTENT": "constexpr",
        },
        constexprs={"PERSISTENT": persistent},
    )
    asm_ty = "f16" if elem_ty == "fp16" else "bf16"
    assert f"v_mfma_f32_32x32x8_{asm_ty}" in compiled.asm["amdgcn"]


def test_amd_scheduled_mfma_rejects_target_version_mismatch():
    with pytest.raises(RuntimeError, match=r"scheduled_mfma.*target requires version 3"):
        compile_for_target(
            _amd_scheduled_mfma_kernel,
            signature={
                "a_ptr": "*bf16",
                "b_ptr": "*bf16",
                "output_ptr": "*fp32",
                "K_WIDTH": "constexpr",
            },
            constexprs={"K_WIDTH": 8},
            target=GFX942,
        )


def test_amd_scheduled_mfma_rejects_non_native_gfx942_shape():
    with pytest.raises(RuntimeError, match=r"version 3 supports only its native 32x32x8 and 16x16x16 shapes"):
        compile_for_gfx942(
            _amd_scheduled_mfma_gfx942_kernel,
            signature={
                "a_ptr": "*fp16",
                "b_ptr": "*fp16",
                "output_ptr": "*fp32",
                "PERSISTENT": "constexpr",
                "TRANSPOSED": "constexpr",
                "INSTR_K": "constexpr",
            },
            constexprs={"PERSISTENT": False, "TRANSPOSED": True, "INSTR_K": 32},
            # 16x16x32 is CDNA4-native, so the v3 verifier rejects it and
            # nothing is emitted
        )


@pytest.mark.parametrize("acc_class", ["agpr", None], ids=["explicit_agpr", "default"])
def test_amd_scheduled_mfma_rejects_agpr_accumulator_gfx942(acc_class):
    reported = acc_class if acc_class is not None else "auto"
    with pytest.raises(RuntimeError, match=f'accumulator_register_class "{reported}" is not yet supported on CDNA3'):
        compile_for_gfx942(
            _amd_scheduled_mfma_regclass_gfx942_kernel,
            signature={
                "a_ptr": "*fp16",
                "b_ptr": "*fp16",
                "output_ptr": "*fp32",
                "ACC_CLASS": "constexpr",
            },
            constexprs={"ACC_CLASS": acc_class},
        )


def test_amd_scheduled_mfma_accepts_explicit_vgpr_gfx942():
    """The rejection is specific to AGPRs; an explicit VGPR class still works."""
    compiled = compile_for_gfx942(
        _amd_scheduled_mfma_regclass_gfx942_kernel,
        signature={
            "a_ptr": "*fp16",
            "b_ptr": "*fp16",
            "output_ptr": "*fp32",
            "ACC_CLASS": "constexpr",
        },
        constexprs={"ACC_CLASS": "vgpr"},
    )
    assert "v_mfma_f32_16x16x16_f16" in compiled.asm["amdgcn"]


def test_amd_scheduled_mfma_round_robin_order_gfx942():
    compiled = compile_for_gfx942(
        _amd_scheduled_mfma_gfx942_kernel,
        signature={
            "a_ptr": "*fp16",
            "b_ptr": "*fp16",
            "output_ptr": "*fp32",
            "PERSISTENT": "constexpr",
            "TRANSPOSED": "constexpr",
            "INSTR_K": "constexpr",
        },
        constexprs={"PERSISTENT": True, "TRANSPOSED": True, "INSTR_K": 16},
    )
    mnemonic = "v_mfma_f32_16x16x16_f16"
    mfmas = [line.strip() for line in compiled.asm["amdgcn"].splitlines() if mnemonic in line]

    assert len(mfmas) == 8
    # M,N,K = 32,128,32, warps_per_cta = [1,4]
    # each warp computes 32x32 => 2 M-reps x 2 N-reps
    # K steps => 32/16 = 2

    destinations = [line.split(mnemonic, 1)[1].strip().split(",", 1)[0] for line in mfmas]
    assert len(set(destinations[:4])) == 4
    assert destinations[4:] == destinations[:4]


@pytest.mark.parametrize("elem_ty", ["bf16", "fp16"])
def test_amd_scheduled_mfma_persistent_acc_lowering_gfx950(elem_ty):
    compiled = compile_for_gfx950(
        _amd_scheduled_mfma_persistent_acc_kernel,
        signature={
            "a_ptr": f"*{elem_ty}",
            "b_ptr": f"*{elem_ty}",
            "output_ptr": "*fp32",
            "USE_VGPR": "constexpr",
        },
        constexprs={"USE_VGPR": False},
    )
    llir = compiled.asm["llir"]
    asm_ty = "f16" if elem_ty == "fp16" else elem_ty
    assert f'asm sideeffect "s_nop 3\\0Av_mfma_f32_16x16x32_{asm_ty}' in llir
    assert '"=a,v,v"' in llir
    assert f"@llvm.amdgcn.mfma.f32.16x16x32.{asm_ty}" not in llir
    # 8 passes + 3 + 1 = 12 wait states for a CDNA4 16x16x32 result read.
    assert 'asm sideeffect "s_nop 11"' in llir


def test_load_helper_preserves_pinned_offset_layout_gfx950():
    physical = tlx.layout(
        shape=((16, 4, 4), (2, 2, 4)),
        stride=((4, 64, 0), (1, 2, 256)),
    )
    compiled = compile_for_gfx950(
        _pinned_load_helper_kernel,
        signature={"src": "*fp8e4nv", "dst": "*fp8e4nv"},
        constexprs={"PHYSICAL": physical},
    )

    # The helper keeps the offset pin and derives layouts for the loaded value.
    assert "tlx.release_layout" not in compiled.asm["ttir"]


def test_local_load_rematerialized_coordinates_compiles_gfx950():
    compiled = compile_for_gfx950(
        _local_load_rematerialized_coordinates_kernel,
        signature={
            "x_ptr": "*fp32",
            "output_ptr": "*fp32",
            "n_elements": "i32",
        },
        constexprs={"BLOCK_SIZE": 256},
    )
    assert "tlx.rematerialize_coordinates" in compiled.asm["ttgir"]
    assert "tlx.rematerialize_coordinates_group = 3 : i32" in compiled.asm["ttgir"]
    assert 'asm sideeffect "", "=v,0"' in compiled.asm["llir"]


def test_local_slice_runtime_offset_compiles_gfx950():
    compiled = compile_for_gfx950(
        _local_slice_runtime_offset_kernel,
        signature={"x_ptr": "*fp32", "output_ptr": "*fp32", "row": "i32"},
        constexprs={},
    )
    assert "ttg.memdesc_dynamic_subslice" in compiled.asm["ttgir"]
    assert "ttg.memdesc_dynamic_subslice" not in compiled.asm["llir"]


def test_padded_local_slice_uses_transposed_lds_read_gfx950():
    """A padded dS-style subslice should retain the CDNA4 transposed load."""
    compiled = compile_for_gfx950(
        _padded_local_slice_transposed_load_kernel,
        signature={
            "x_ptr": "*bf16",
            "rhs_ptr": "*bf16",
            "output_ptr": "*fp32",
        },
        constexprs={},
    )
    assert "ttg.memdesc_subslice" in compiled.asm["ttgir"]
    assert "ttg.memdesc_dynamic_subslice" not in compiled.asm["ttgir"]
    amdgcn = compiled.asm["amdgcn"]
    assert "ds_read_b64_tr_b16" in amdgcn
    assert "ds_read_u16" not in amdgcn


@pytest.mark.parametrize(
    "mode, error",
    [
        (0, "requires at least one"),
        (1, "clamp_bounds requires add_offsets"),
        (2, "clamp_bounds and set_bounds are mutually exclusive"),
        (3, "add_offsets must have length 2"),
    ],
)
def test_update_tensor_descriptor_rejects_invalid_gfx1250(mode, error):
    with pytest.raises(CompilationError, match=error):
        compile_for_gfx1250(
            _invalid_update_tensor_descriptor_kernel,
            signature={"x_ptr": "*fp16"},
            constexprs={"MODE": mode},
        )


def test_update_tensor_descriptor_rejects_unsupported_target():
    with pytest.raises(CompilationError, match="only available on AMD TDM-capable targets"):
        compile_for_gfx950(
            _invalid_update_tensor_descriptor_kernel,
            signature={"x_ptr": "*fp16"},
            constexprs={"MODE": 4},
        )


def test_local_reshape_compiles_gfx1250(device):
    """tlx.local_reshape should lower to ttg.memdesc_reshape and compile."""
    compiled = compile_for_gfx1250(
        _local_reshape_kernel,
        signature={"input_ptr": "*fp32", "output_ptr": "*fp32"},
        constexprs={"ROWS": 8, "COLS": 8},
    )
    ttgir = compiled.asm["ttgir"]
    assert "ttg.memdesc_reshape" in ttgir, ("expected memdesc_reshape in TTGIR, got:\n" + ttgir)
    assert "amdgcn" in compiled.asm
    assert len(compiled.asm["amdgcn"]) > 0


def test_dot_scaled_tiles_per_warp_attr_gfx1250():
    compiled = _compile_dot_scaled_tiles_per_warp((2, 2))
    assert "amdg.wmma_tiles_per_warp = array<i32: 2, 2>" in compiled.asm["ttir"]
    assert "#ttg.amd_wmma" in compiled.asm["ttgir"]


@pytest.mark.parametrize(
    "tiles_per_warp, error",
    [
        ((1, ), "tiles_per_warp requires 2 entries"),
        ((0, 1), "tiles_per_warp entries must be positive"),
    ],
)
def test_dot_scaled_tiles_per_warp_rejects_invalid_gfx1250(tiles_per_warp, error):
    with pytest.raises(CompilationError, match=error):
        _compile_dot_scaled_tiles_per_warp(tiles_per_warp)


def test_require_amd_wmma_layout_compiles_gfx1250():
    compiled = compile_for_gfx1250(
        _require_amd_wmma_layout_kernel,
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32"},
        constexprs={"BLOCK": 256},
    )
    assert "#ttg.amd_wmma" in compiled.asm["ttgir"]


def test_mxgemm_tdm_pipelined_compiles_gfx1250(device):
    """The mxfp GEMM tutorial kernel should lower to TDM + dot_scaled + WMMA."""
    compiled = compile_for_gfx1250(
        _amd_mxfp_gemm_kernel,
        signature={
            "a_ptr": "*fp8e5",
            "b_ptr": "*fp8e5",
            "c_ptr": "*fp32",
            "a_scale": "*i8",
            "b_scale": "*i8",
            "M": "i32",
            "N": "i32",
            "K": "i32",
            "stride_am": "i64",
            "stride_ak": "i64",
            "stride_bk": "i64",
            "stride_bn": "i64",
            "stride_cm": "i64",
            "stride_cn": "i64",
            "stride_scale": "i64",
        },
        constexprs={
            "DTYPE_A": "e5m2",
            "DTYPE_B": "e5m2",
            "SCALE_BLOCK": 32,
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 128,
            "GROUP_SIZE_M": 8,
            "TRANSPOSE_B": True,
            "NUM_BUFFERS": 2,
        },
    )
    ttgir = compiled.asm["ttgir"]
    amdgcn = compiled.asm["amdgcn"]
    assert "amdg.async_tdm_copy_global_to_local" in ttgir
    assert "tt.dot_scaled" in ttgir
    assert "tensor_load_to_lds" in amdgcn or "tensor.load.to.lds" in amdgcn
    assert "wmma" in amdgcn


def test_tlx_gfx9_gemm_bench_parses_shapes_and_defaults():
    bench = _load_tlx_gfx9_gemm_bench_module()

    assert not hasattr(bench, "DEVICE")
    assert set(bench.VERSION_MAP) == set(range(10))
    assert set(bench.PROVIDER_LABELS) == {"rocblas", "tlx"}
    assert bench.provider_defaults(9) == ["rocblas", "tlx"]
    assert bench.provider_defaults(0) == ["rocblas", "tlx"]
    assert bench.parse_shape("128x256x64") == (128, 256, 64)
    assert bench.parse_shape("128,256,64") == (128, 256, 64)
    with pytest.raises(Exception, match="shape dimensions must be positive"):
        bench.parse_shape("128x0x64")
    with pytest.raises(Exception, match="shape must be MxNxK"):
        bench.parse_shape("128x256")
    bench.validate_shape_for_providers((256, 256, 64), 0, ["tlx"])
    bench.validate_shape_for_providers((128, 128, 64), 9, ["rocblas"])
    with pytest.raises(Exception, match="M to be a multiple of 256"):
        bench.validate_shape_for_providers((128, 256, 64), 9, ["tlx"])
    with pytest.raises(Exception, match="N to be a multiple of 256"):
        bench.validate_shape_for_providers((256, 128, 64), 9, ["tlx"])
    with pytest.raises(Exception, match="K to be a multiple of 64"):
        bench.validate_shape_for_providers((256, 256, 96), 2, ["tlx"])
    with pytest.raises(Exception, match="prefetch two 64-wide K tiles"):
        bench.validate_shape_for_providers((256, 256, 64), 9, ["tlx"])
    bench.validate_shape_for_providers((256, 256, 128), 9, ["tlx"])


def test_tlx_gfx9_gemm_bench_input_modes_are_deterministic():
    bench = _load_tlx_gfx9_gemm_bench_module("_tlx_amd_test_gfx9_bench_inputs")
    inter_wave = _load_tlx_gfx9_inter_wave_bench_module("_tlx_amd_test_gfx9_inter_wave_bench_inputs")
    assert inter_wave.INPUT_MODES == bench.INPUT_MODES
    normal_seed_zero = None

    for input_mode in bench.INPUT_MODES:
        a, b = bench.make_inputs(
            2,
            4,
            8,
            torch.device("cpu"),
            "transposed",
            input_mode=input_mode,
            seed=0,
        )
        repeat_a, repeat_b = bench.make_inputs(
            2,
            4,
            8,
            torch.device("cpu"),
            "transposed",
            input_mode=input_mode,
            seed=0,
        )
        torch.testing.assert_close(a, repeat_a)
        torch.testing.assert_close(b, repeat_b)
        inter_wave_a, inter_wave_b = inter_wave.make_inputs(
            2,
            4,
            8,
            torch.device("cpu"),
            "transposed",
            input_mode=input_mode,
            seed=0,
        )
        torch.testing.assert_close(inter_wave_a, a)
        torch.testing.assert_close(inter_wave_b, b)
        assert b.shape == (8, 4)
        assert b.stride() == (1, 8)
        if input_mode == "normal":
            normal_seed_zero = a

    normal_a, _ = bench.make_inputs(2, 4, 8, "cpu", "transposed", input_mode="normal", seed=1)
    assert not torch.equal(normal_seed_zero, normal_a)


def test_tlx_gfx9_gemm_bench_reproduces_hipblaslt_rand_int_inputs():
    bench = _load_tlx_gfx9_gemm_bench_module("_tlx_amd_test_gfx9_bench_rand_int")

    a, b = bench.make_inputs(
        2,
        4,
        8,
        torch.device("cpu"),
        "transposed",
        input_mode="rand-int",
        seed=0,
    )

    expected_a = torch.tensor(
        [
            [-2, -2, 0, 0, 1, 0, 1, 2],
            [-1, -2, 0, -1, -2, -2, 0, -1],
        ],
        dtype=torch.float16,
    )
    expected_b_storage = torch.tensor(
        [
            [2, -2, 0, 0, -1, 0, -1, 2],
            [-1, 2, 0, 1, -2, 2, 0, 1],
            [2, 0, -2, -1, 1, 2, 0, 2],
            [2, -1, -1, 1, 2, 2, 1, 2],
        ],
        dtype=torch.float16,
    )
    torch.testing.assert_close(a, expected_a)
    torch.testing.assert_close(b.T, expected_b_storage)


def test_tlx_gfx9_gemm_bench_launch_reuses_output():
    bench = _load_tlx_gfx9_gemm_bench_module("_tlx_amd_test_gfx9_bench_output")
    call = {}

    class FakeKernel:

        def __getitem__(self, grid):
            call["grid"] = grid

            def launch(*args, **kwargs):
                call["args"] = args
                call["kwargs"] = kwargs

            return launch

    module = SimpleNamespace(v9_beyond_hotloop=FakeKernel())
    a = torch.empty((256, 128), dtype=torch.float16)
    b = torch.empty((128, 256), dtype=torch.float16)
    out = torch.empty((256, 256), dtype=torch.float16)

    result = bench.launch_tutorial_matmul(module, "v9_beyond_hotloop", a, b, out=out)

    assert result is out
    assert call["args"][2] is out
    assert call["grid"] == (1, )


def test_tlx_gfx9_gemm_bench_batched_timing_uses_one_event_span_per_repeat():
    bench = _load_tlx_gfx9_gemm_bench_module("_tlx_amd_test_gfx9_bench_timing")
    state = {"launches": 0, "synchronizes": 0, "events": 0}

    class FakeEvent:

        def __init__(self):
            self.launch = None

        def record(self):
            self.launch = state["launches"]

        def elapsed_time(self, other):
            return (other.launch - self.launch) * 0.25

    class FakeDeviceInterface:

        def Event(self, *, enable_timing):
            assert enable_timing
            state["events"] += 1
            return FakeEvent()

        def synchronize(self):
            state["synchronizes"] += 1

    def launch():
        state["launches"] += 1

    ms = bench.do_bench_batched(
        launch,
        warmup_launches=2,
        timed_launches=4,
        repeats=3,
        device_interface=FakeDeviceInterface(),
    )

    assert ms == 0.25
    assert state == {"launches": 18, "synchronizes": 6, "events": 6}


def test_tlx_gfx9_gemm_bench_triton_timing_reports_median(monkeypatch):
    bench = _load_tlx_gfx9_gemm_bench_module("_tlx_amd_test_gfx9_bench_median")
    call = {}

    def do_bench(fn, **kwargs):
        call["fn"] = fn
        call["kwargs"] = kwargs
        return 0.75

    monkeypatch.setattr(bench.triton.testing, "do_bench", do_bench)
    fn = lambda: None
    ms = bench.measure_provider(
        SimpleNamespace(timing_mode="triton", warmup=13, rep=29),
        fn,
    )

    assert ms == 0.75
    assert call == {
        "fn": fn,
        "kwargs": {"warmup": 13, "rep": 29, "return_mode": "median"},
    }


def test_tlx_gfx9_gemm_bench_loads_modules_without_import_leaks():
    bench = _load_tlx_gfx9_gemm_bench_module("_tlx_amd_test_gfx9_bench_imports")
    before_path = list(sys.path)

    module = bench.load_matmul_module("v0_naive", "test")

    assert hasattr(module, "matmul")
    assert list(sys.path) == before_path
    assert module.__name__ not in sys.modules


def test_amd_sched_barrier_compiles_gfx950():
    compiled = compile_for_gfx950(
        _amd_sched_barrier_kernel,
        signature={"x_ptr": "*bf16", "y_ptr": "*bf16", "BLOCK": "constexpr"},
        constexprs={"BLOCK": 64},
    )
    assert "llvm.amdgcn.sched.barrier" in compiled.asm["llir"]


def test_d64_causal_stat_conventions_are_equivalent():
    generator = torch.Generator(device="cpu")
    generator.manual_seed(101)
    q = torch.randn((3, 4), generator=generator)
    k = torch.randn((5, 4), generator=generator)
    v = torch.randn((5, 4), generator=generator)
    o = torch.randn((3, 4), generator=generator)
    do = torch.randn((3, 4), generator=generator)
    lse = torch.randn((3, ), generator=generator)
    sm_scale = 0.5

    delta_mha, lse_mha = _d64_causal_stat_values(o, do, lse, sm_scale, _D64_MHA_POSITIVE)
    delta_gqa, lse_gqa = _d64_causal_stat_values(o, do, lse, sm_scale, _D64_GQA_SIGNED)

    expected_delta = torch.sum(o * do, dim=-1)
    expected_lse = -lse * math.log2(math.e)
    torch.testing.assert_close(delta_mha, expected_delta)
    torch.testing.assert_close(delta_gqa, -expected_delta)
    assert lse_mha is None
    torch.testing.assert_close(lse_gqa, expected_lse)

    scores = q @ k.mT
    p_mha = torch.exp2((scores * sm_scale - lse[..., None]) * math.log2(math.e))
    p_gqa = torch.exp2(scores * (sm_scale * math.log2(math.e)) + lse_gqa[..., None])
    ds_mha = p_mha * (do @ v.mT - delta_mha[..., None])
    ds_gqa = p_gqa * (do @ v.mT + delta_gqa[..., None])
    torch.testing.assert_close(p_mha, p_gqa)
    torch.testing.assert_close(ds_mha, ds_gqa)

    with pytest.raises(ValueError, match=r"^unknown D64 stat mode 2$"):
        _d64_causal_stat_values(o, do, lse, sm_scale, 2)


@pytest.mark.parametrize(
    "invalid_scale",
    [
        pytest.param(0.0, id="zero"),
        pytest.param(float("inf"), id="infinite"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(10**10000, id="overflowing-integer"),
    ],
)
def test_d64_scale_validation_and_scheduled_fallback(invalid_scale):
    with pytest.raises(ValueError, match=r"^D64 sm_scale must be finite and nonzero$"):
        _validate_d64_sm_scale(invalid_scale)

    shape = (4, 64, 4096, 64)
    dispatch = _select_d64_dispatch(
        shape,
        shape,
        True,
        arch="gfx950",
        cu_count=256,
        sm_scale=invalid_scale,
        bases_aligned_16=True,
    )
    assert dispatch.family == "causal_m192"


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        pytest.param(
            (4, 64, 8, 8192, 8192, 192, 256, True),
            (
                _D64DQLaunch(42, False, 0, 42, 3, 0),
                _D64DQLaunch(1, False, 42, 1, 2, 192),
            ),
            id="peeled-tail",
        ),
        pytest.param(
            (4, 64, 8, 4096, 4096, 192, 256, True),
            (_D64DQLaunch(22, True, 0, 0, 3, 0), ),
            id="single-launch",
        ),
        pytest.param(
            (4, 64, 8, 8192, 8192, 192, 256, False),
            (_D64DQLaunch(43, False, 0, 0, 3, 0), ),
            id="no-host-tail-skip",
        ),
    ],
)
def test_d64_causal_dq_launch_plan(args, expected):
    assert _d64_dq_launch_plan(*args) == expected


@pytest.mark.parametrize(
    ("owner_start", "owner_rows", "sq", "skv", "block_n", "expected"),
    [
        (0, 192, 4096, 4096, 32, 6),
        (192, 192, 4096, 4096, 32, 12),
        (0, 192, 4096, 16384, 64, 195),
        (3840, 192, 4096, 4096, 32, 126),
    ],
)
def test_d64_causal_dq_compact_key_frontier(owner_start, owner_rows, sq, skv, block_n, expected):
    assert _d64_causal_dq_key_blocks(owner_start, owner_rows, sq, skv, block_n) == expected


@pytest.mark.parametrize(
    ("key_start", "sq", "skv", "block_m", "expected"),
    [
        (0, 4096, 4096, 64, 0),
        (256, 4096, 4096, 64, 4),
        (12288, 4096, 16384, 64, 0),
        (16320, 4096, 16384, 64, 63),
    ],
)
def test_d64_causal_dkdv_compact_query_frontier(key_start, sq, skv, block_m, expected):
    assert _d64_causal_dkdv_first_query_block(key_start, sq, skv, block_m) == expected


def test_d64_workspace_shapes():
    q = torch.empty((2, 32, 4096, 64), device="meta", dtype=torch.bfloat16)
    k_mha = torch.empty((2, 32, 4096, 64), device="meta", dtype=torch.bfloat16)
    k_gqa = torch.empty((2, 4, 4096, 64), device="meta", dtype=torch.bfloat16)

    lse_term, causal_dk, causal_dv = _allocate_bwd_d64_causal_gqa8_workspaces(q, k_gqa)
    assert lse_term.shape == (2, 32, 4096) and lse_term.dtype is torch.float32
    assert causal_dk.shape == causal_dv.shape == (2, 4, 4, 4096, 64)
    assert causal_dk.dtype is causal_dv.dtype is torch.bfloat16

    mha_dispatch = _D64Dispatch("noncausal_fused_n256", 32, 256, 1)
    gqa_dispatch = _D64Dispatch("noncausal_fused_n256", 32, 256, 8)
    mha_acc, mha_dk, mha_dv = _allocate_bwd_d64_fused_workspaces(q, k_mha, mha_dispatch)
    gqa_acc, gqa_dk, gqa_dv = _allocate_bwd_d64_fused_workspaces(q, k_gqa, gqa_dispatch)
    assert mha_acc.shape == gqa_acc.shape == q.shape
    assert mha_acc.dtype is gqa_acc.dtype is torch.float32
    assert mha_dk is mha_dv is None
    assert gqa_dk.shape == gqa_dv.shape == (2, 4, 8, 4096, 64)
    assert gqa_dk.dtype is gqa_dv.dtype is torch.bfloat16


def test_d64_direct_launch_uses_dispatch_ownership(monkeypatch):

    class LaunchRecorder:

        def __init__(self):
            self.calls = []

        def __getitem__(self, grid):

            def record(*args, **kwargs):
                self.calls.append((grid, args, kwargs))

            return record

    dq_launch = LaunchRecorder()
    dkdv_launch = LaunchRecorder()
    reduce_launch = LaunchRecorder()
    monkeypatch.setitem(vars(amd_fa_bwd), "_attn_bwd_dq_d64_direct_kernel", dq_launch)
    monkeypatch.setitem(vars(amd_fa_bwd), "_attn_bwd_dkdv_d64_direct_kernel", dkdv_launch)
    monkeypatch.setitem(vars(amd_fa_bwd), "_attn_bwd_dkdv_d64_reduce_kernel", reduce_launch)

    q_shape = (4, 48, 4096, 64)
    k_shape = (4, 6, 16384, 64)
    q = torch.empty(q_shape, device="meta", dtype=torch.bfloat16)
    k = torch.empty(k_shape, device="meta", dtype=torch.bfloat16)
    dispatch = _select_d64_dispatch(q_shape, k_shape, True)
    _run_bwd_d64_direct(q, k, k, q, object(), object(), q, k, k, 0.125, True, dispatch)

    assert len(dq_launch.calls) == 1
    dq_grid, _args, dq_kwargs = dq_launch.calls[0]
    assert dq_grid == (triton.cdiv(q_shape[2], 192), q_shape[1], q_shape[0])
    assert (dq_kwargs["OWNER_ROWS"], dq_kwargs["BLOCK_N"]) == (192, 64)

    assert len(dkdv_launch.calls) == 1
    dkdv_grid, _args, dkdv_kwargs = dkdv_launch.calls[0]
    assert dkdv_grid == (triton.cdiv(k_shape[2], 64), k_shape[1] * 4, k_shape[0])
    assert (dkdv_kwargs["KV_SPLITS"], dkdv_kwargs["BLOCK_N"]) == (4, 64)
    assert len(reduce_launch.calls) == 1


def test_varlen_d128_address_space_requires_i32_offsets():
    # One D128 BF16 head fits at most 2**30 elements in the signed i32
    # byte-offset range used by AMD buffer instructions.
    max_tokens = 2**23

    amd_fa_varlen_bwd._validate_i32_buffer_offsets(
        total_q=max_tokens - 15,
        total_kv=max_tokens,
        batch=1,
        q_heads=1,
        kv_heads=1,
    )

    with pytest.raises(ValueError, match="KV tensor size exceeds the signed 32-bit byte-offset range"):
        amd_fa_varlen_bwd._validate_i32_buffer_offsets(
            total_q=1,
            total_kv=max_tokens + 1,
            batch=1,
            q_heads=1,
            kv_heads=1,
        )
    with pytest.raises(ValueError, match="padded dQ size exceeds the signed 32-bit byte-offset range"):
        amd_fa_varlen_bwd._validate_i32_buffer_offsets(
            total_q=max_tokens - 14,
            total_kv=1,
            batch=1,
            q_heads=1,
            kv_heads=1,
        )


@triton.jit
def _async_load_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    buffers = tlx.local_alloc((BLOCK_SIZE, ), tl.float32, 2)

    buf0 = tlx.local_view(buffers, 0)
    buf1 = tlx.local_view(buffers, 1)
    tok_x = tlx.async_load(x_ptr + offs, buf0, mask=mask)
    tok_y = tlx.async_load(y_ptr + offs, buf1, mask=mask)
    tlx.async_load_commit_group([tok_x, tok_y])
    tlx.async_load_wait_group(0)

    x = tlx.local_load(buf0)
    y = tlx.local_load(buf1)
    tl.store(output_ptr + offs, x + y, mask=mask)


@triton.jit
def _local_load_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    buf = tlx.local_alloc((BLOCK_SIZE, ), tl.float32, 1)
    buf0 = tlx.local_view(buf, 0)
    tok = tlx.async_load(x_ptr + offs, buf0, mask=mask)
    tlx.async_load_commit_group([tok])
    tlx.async_load_wait_group(0)

    x = tlx.local_load(buf0)
    tl.store(output_ptr + offs, x, mask=mask)


@triton.jit
def _local_load_with_token_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    buf = tlx.local_alloc((BLOCK_SIZE, ), tl.float32, 1)
    buf0 = tlx.local_view(buf, 0)
    tok = tlx.async_load(x_ptr + offs, buf0, mask=mask)
    tlx.async_load_commit_group([tok])
    wait_tok = tlx.async_load_wait_group(0)

    x = tlx.local_load(buf0, token=wait_tok)
    tl.store(output_ptr + offs, x, mask=mask)


@triton.jit
def _token_in_loop_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    NUM_ITERS: tl.constexpr,
):
    """async_token from async_load_commit_group is live when tl.range is
    entered. If async_token._flatten_ir is broken, the code generator
    crashes with NotImplementedError when collecting carries."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    buf = tlx.local_alloc((BLOCK_SIZE, ), tl.float32, 1)
    buf0 = tlx.local_view(buf, 0)

    tok = tlx.async_load(x_ptr + offs, buf0, mask=mask)
    tlx.async_load_commit_group([tok])

    acc = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)

    # tok is in scope here -- that's what we're testing.
    for i in tl.range(0, NUM_ITERS, num_stages=1):
        tlx.async_load_wait_group(0)
        x = tlx.local_load(buf0)
        acc += x

    tl.store(output_ptr + offs, acc, mask=mask)


@triton.jit
def _loop_carried_dot_layout_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_ITERS: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * (BLOCK_K * K_ITERS) + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * BLOCK_N + offs_n[None, :]

    a_buffers = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.float16, 2)
    b_buffers = tlx.local_alloc((BLOCK_K, BLOCK_N), tl.float16, 2)

    a_buf = tlx.local_view(a_buffers, 0)
    b_buf = tlx.local_view(b_buffers, 0)
    tlx.local_store(a_buf, tl.load(a_ptrs))
    tlx.local_store(b_buf, tl.load(b_ptrs))

    a_reg = tlx.local_load(a_buf)
    b_reg = tlx.local_load(b_buf)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in tl.range(0, K_ITERS - 1, num_stages=1):
        acc = tl.dot(a_reg, b_reg, acc)
        next_slot = (k + 1) % 2
        next_a = tlx.local_view(a_buffers, next_slot)
        next_b = tlx.local_view(b_buffers, next_slot)
        tlx.local_store(next_a, tl.load(a_ptrs + (k + 1) * BLOCK_K))
        tlx.local_store(next_b, tl.load(b_ptrs + (k + 1) * BLOCK_K * BLOCK_N))
        a_reg = tlx.local_load(next_a)
        b_reg = tlx.local_load(next_b)

    acc = tl.dot(a_reg, b_reg, acc)
    c_ptrs = c_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :]
    tl.store(c_ptrs, acc)


@triton.jit
def _async_amd_desc_load_kernel(
    x_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    desc = tl.make_tensor_descriptor(x_ptr, [M, N], [N, 1], [M, N])
    buf = tlx.local_alloc((M, N), tl.float16, 1)
    buf0 = tlx.local_view(buf, 0)
    tlx.async_amd_descriptor_load(desc, buf0, [0, 0])
    tlx.async_amd_descriptor_wait(pendings=0)
    x = tlx.local_load(buf0)
    tl.store(output_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], x)


@triton.jit
def _async_amd_desc_load_fused_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    a_desc = tl.make_tensor_descriptor(a_ptr, [M, N], [N, 1], [M, N])
    b_desc = tl.make_tensor_descriptor(b_ptr, [M, N], [N, 1], [M, N])
    a_buf = tlx.local_alloc((M, N), tl.float16, 1)
    b_buf = tlx.local_alloc((M, N), tl.float16, 1)
    a_smem = tlx.local_view(a_buf, 0)
    b_smem = tlx.local_view(b_buf, 0)
    a_desc = tlx.update_tensor_descriptor(a_desc, add_offsets=[0, 0], pred=True, clamp_bounds=True)
    b_desc = tlx.update_tensor_descriptor(b_desc, add_offsets=[0, 0], pred=True, clamp_bounds=True)
    token = tlx.async_amd_descriptor_load_fused([
        (a_desc, a_smem, 0b0011),
        (b_desc, b_smem, 0b1100),
    ])
    tlx.async_amd_descriptor_wait(tokens=[token])
    result = tlx.local_load(a_smem) + tlx.local_load(b_smem)
    offsets = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    tl.store(output_ptr + offsets, result)


@triton.jit
def _async_amd_desc_load_with_token_kernel(
    x_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    desc = tl.make_tensor_descriptor(x_ptr, [M, N], [N, 1], [M, N])
    buf = tlx.local_alloc((M, N), tl.float16, 1)
    buf0 = tlx.local_view(buf, 0)
    tok = tlx.async_amd_descriptor_load(desc, buf0, [0, 0])
    tlx.async_amd_descriptor_wait(tokens=[tok])
    x = tlx.local_load(buf0)
    tl.store(output_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], x)


@triton.jit
def _async_amd_desc_load_pred_kernel(
    x_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    desc = tl.make_tensor_descriptor(x_ptr, [M, N], [N, 1], [M, N])
    buf = tlx.local_alloc((M, N), tl.float16, 1)
    buf0 = tlx.local_view(buf, 0)
    pred = tl.program_id(0) == 0
    tlx.async_amd_descriptor_load(desc, buf0, [0, 0], pred=pred)
    tlx.async_amd_descriptor_wait(pendings=0)
    x = tlx.local_load(buf0)
    tl.store(output_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], x)


@triton.jit
def _async_amd_desc_store_kernel(
    x_ptr,
    y_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    desc_in = tl.make_tensor_descriptor(x_ptr, [M, N], [N, 1], [M, N])
    desc_out = tl.make_tensor_descriptor(y_ptr, [M, N], [N, 1], [M, N])
    # Separate buffers for load vs store — they get different encodings
    # (padded for load, swizzled for store) and can't share a buffer
    # until alignTDMDescriptorEncodings is ported.
    load_buf = tlx.local_alloc((M, N), tl.float16, 1)
    store_buf = tlx.local_alloc((M, N), tl.float16, 1)
    load_view = tlx.local_view(load_buf, 0)
    store_view = tlx.local_view(store_buf, 0)
    tlx.async_amd_descriptor_load(desc_in, load_view, [0, 0])
    tlx.async_amd_descriptor_wait(pendings=0)
    data = tlx.local_load(load_view)
    tlx.local_store(store_view, data)
    tlx.async_amd_descriptor_store(desc_out, store_view, [0, 0])
    tlx.async_amd_descriptor_wait(pendings=0)


@triton.jit
def _update_tensor_descriptor_store_kernel(
    x_ptr,
    y_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    desc_in = tl.make_tensor_descriptor(x_ptr, [M, N], [N, 1], [M, N])
    desc_out = tl.make_tensor_descriptor(y_ptr, [M, N], [N, 1], [M, N])
    load_buf = tlx.local_alloc((M, N), tl.float16, 1)
    store_buf = tlx.local_alloc((M, N), tl.float16, 1)
    load_view = tlx.local_view(load_buf, 0)
    store_view = tlx.local_view(store_buf, 0)

    pred = tl.program_id(0) == 0
    desc_in = tlx.update_tensor_descriptor(desc_in, set_bounds=[M, N], pred=pred)
    offset_m = desc_in.shape[0] - M
    offset_n = (desc_in.strides[1] - 1).to(tl.int32)
    desc_in = tlx.update_tensor_descriptor(desc_in, add_offsets=[offset_m, offset_n])
    tlx.async_amd_descriptor_load(desc_in, load_view)
    tlx.async_amd_descriptor_wait(0)
    tlx.local_store(store_view, tlx.local_load(load_view))

    desc_out = tlx.update_tensor_descriptor(desc_out, add_offsets=[0, 0], clamp_bounds=True)
    tlx.async_amd_descriptor_store(desc_out, store_view)
    tlx.async_amd_descriptor_wait(0)


@triton.jit
def _amd_desc_prefetch_kernel(
    x_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    desc = tl.make_tensor_descriptor(x_ptr, [M, N], [N, 1], [M, N])
    tlx.amd_descriptor_prefetch_tensor(desc, [0, 0])
    buf = tlx.local_alloc((M, N), tl.float16, 1)
    buf0 = tlx.local_view(buf, 0)
    tlx.async_amd_descriptor_load(desc, buf0, [0, 0])
    tlx.async_amd_descriptor_wait(pendings=0)
    x = tlx.local_load(buf0)
    tl.store(output_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], x)


@triton.jit
def _amd_desc_prefetch_speculative_kernel(
    x_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    desc = tl.make_tensor_descriptor(x_ptr, [M, N], [N, 1], [M, N])
    pred = tl.program_id(0) == 0
    tlx.amd_descriptor_prefetch_tensor(desc, [0, 0], pred=pred, speculative=True)
    # A TDM load on the same descriptor so it gets a valid encoding
    # during lowering (prefetch alone doesn't assign one).
    buf = tlx.local_alloc((M, N), tl.float16, 1)
    buf0 = tlx.local_view(buf, 0)
    tlx.async_amd_descriptor_load(desc, buf0, [0, 0])
    tlx.async_amd_descriptor_wait(pendings=0)
    x = tlx.local_load(buf0)
    tl.store(output_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], x)


@triton.jit
def _pinned_tdm_memdesc_view_kernel(
    input_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    VIEW: tl.constexpr,
):
    smem_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for([(N, 128 // 16)], [M, N])
    desc = tl.make_tensor_descriptor(input_ptr, [M, N], [N, 1], [M, N])
    buffers = tlx.local_alloc((M, N), tl.float16, 1, layout=smem_layout)
    full = tlx.local_view(buffers, 0)

    token = tlx.async_amd_descriptor_load(desc, full, [0, 0])
    tlx.async_amd_descriptor_wait(tokens=[token])

    if VIEW == 0:
        view = tlx.local_slice(full, [0, 32], [M, 32])
        rows = tl.arange(0, M)
        cols = tl.arange(0, 32)
        width: tl.constexpr = 32
    else:
        view = tlx.local_reshape(full, [N, M])
        rows = tl.arange(0, N)
        cols = tl.arange(0, M)
        width: tl.constexpr = M

    values = tlx.local_load(view)
    tl.store(output_ptr + rows[:, None] * width + cols[None, :], values)


@triton.jit
def _assume_uniform_ptr_kernel(ptr_array, out_ptr, BLOCK: tl.constexpr):
    # A pointer loaded from memory is not provably uniform, so the backend would
    # otherwise waterfall every buffer access built on it.
    base = tl.load(ptr_array).to(tl.pointer_type(tl.float32))
    base = tlx.assume_uniform(base)
    offs = tl.arange(0, BLOCK).to(tl.int32)
    tlx.buffer_store(tlx.buffer_load(base, offs), out_ptr, offs)


@triton.jit
def _assume_uniform_ptr_kernel_no_hint(ptr_array, out_ptr, BLOCK: tl.constexpr):
    base = tl.load(ptr_array).to(tl.pointer_type(tl.float32))
    offs = tl.arange(0, BLOCK).to(tl.int32)
    tlx.buffer_store(tlx.buffer_load(base, offs), out_ptr, offs)


@triton.jit
def _assume_uniform_scalar_kernel(in_ptr, out_ptr, BLOCK: tl.constexpr):
    v = tlx.assume_uniform(tl.load(in_ptr))
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.zeros((BLOCK, ), tl.float32) + v.to(tl.float32))


def test_async_load_compiles_gfx950(device):
    """async_load should produce async_copy_global_to_local in TTGIR on gfx950."""
    compiled = compile_for_gfx950(
        _async_load_kernel,
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "output_ptr": "*fp32", "n_elements": "i32"},
        constexprs={"BLOCK_SIZE": 64},
    )
    ttgir = compiled.asm["ttgir"]
    assert "async_copy_global_to_local" in ttgir or "buffer_load_to_local" in ttgir
    assert "async_commit_group" in ttgir
    assert "async_wait" in ttgir
    assert "local_load" in ttgir

    # Verify the kernel compiled all the way to AMDGCN.
    assert "amdgcn" in compiled.asm
    assert len(compiled.asm["amdgcn"]) > 0


def test_local_load_compiles_gfx950(device):
    """local_load after async_wait should compile and produce local_load in TTGIR."""
    compiled = compile_for_gfx950(
        _local_load_kernel,
        signature={"x_ptr": "*fp32", "output_ptr": "*fp32", "n_elements": "i32"},
        constexprs={"BLOCK_SIZE": 64},
    )
    ttgir = compiled.asm["ttgir"]
    assert "local_load" in ttgir


def test_local_load_with_token_compiles_gfx950(device):
    """local_load with a wait token should set syncedViaAsyncWait in TTGIR."""
    compiled = compile_for_gfx950(
        _local_load_with_token_kernel,
        signature={"x_ptr": "*fp32", "output_ptr": "*fp32", "n_elements": "i32"},
        constexprs={"BLOCK_SIZE": 64},
    )
    ttgir = compiled.asm["ttgir"]
    assert "local_load" in ttgir
    assert re.search(r'ttg\.local_load .* \{ttg\.amdg\.syncedViaAsyncWait = true\}', ttgir, re.MULTILINE)


def test_async_token_loop_compiles_gfx950(device):
    """async_token in scope around tl.range should compile without crashing."""
    compiled = compile_for_gfx950(
        _token_in_loop_kernel,
        signature={"x_ptr": "*fp32", "output_ptr": "*fp32", "n_elements": "i32"},
        constexprs={"BLOCK_SIZE": 64, "NUM_ITERS": 4},
    )
    ttgir = compiled.asm["ttgir"]
    assert "local_load" in ttgir
    assert "async_wait" in ttgir


def test_loop_carried_dot_layout_cleanup_compiles_gfx950(device):
    """Full AMD pipeline should remove late dot operand local_alloc fallbacks."""
    compiled = compile_for_gfx950(
        _loop_carried_dot_layout_kernel,
        signature={"a_ptr": "*fp16", "b_ptr": "*fp16", "c_ptr": "*fp32"},
        constexprs={"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "K_ITERS": 3},
    )
    ttgir = compiled.asm["ttgir"]
    assert "ttg.local_alloc %" not in ttgir
    assert "tt.dot" in ttgir
    assert "amdgcn" in compiled.asm
    assert len(compiled.asm["amdgcn"]) > 0


def test_async_amd_desc_load_compiles_gfx1250(device):
    """async_amd_descriptor_load should produce TDM ops in TTGIR."""
    compiled = compile_for_gfx1250(
        _async_amd_desc_load_kernel,
        signature={"x_ptr": "*fp16", "output_ptr": "*fp16"},
        constexprs={"M": 32, "N": 32},
    )
    ttgir = compiled.asm["ttgir"]
    assert "async_tdm_copy_global_to_local" in ttgir
    assert "clamp_bounds" in ttgir
    assert "async_tdm_wait" in ttgir
    assert "local_load" in ttgir
    assert "amdgcn" in compiled.asm
    assert len(compiled.asm["amdgcn"]) > 0


def test_async_amd_desc_load_fused_compiles_gfx1250(device):
    """Two positioned TLX descriptors lower to one fused TDM instruction."""
    compiled = compile_for_gfx1250(
        _async_amd_desc_load_fused_kernel,
        signature={"a_ptr": "*fp16", "b_ptr": "*fp16", "output_ptr": "*fp16"},
        constexprs={"M": 16, "N": 32},
    )
    ttgir = compiled.asm["ttgir"]
    assert "amdg.async_tdm_fused_copy_global_to_local" in ttgir
    assert "warp_used_hints = array<i32: 3, 12>" in ttgir
    assert len(re.findall(r"tensor_load_to_lds|tensor\.load\.to\.lds", compiled.asm["amdgcn"])) == 1


def test_async_amd_desc_load_with_token_compiles_gfx1250(device):
    """async_amd_descriptor_load with token-threaded wait compiles."""
    compiled = compile_for_gfx1250(
        _async_amd_desc_load_with_token_kernel,
        signature={"x_ptr": "*fp16", "output_ptr": "*fp16"},
        constexprs={"M": 32, "N": 32},
    )
    ttgir = compiled.asm["ttgir"]
    assert "async_tdm_copy_global_to_local" in ttgir
    assert "async_tdm_wait" in ttgir


def test_async_amd_desc_load_pred_compiles_gfx1250(device):
    """async_amd_descriptor_load with i1 pred extends to i32."""
    compiled = compile_for_gfx1250(
        _async_amd_desc_load_pred_kernel,
        signature={"x_ptr": "*fp16", "output_ptr": "*fp16"},
        constexprs={"M": 32, "N": 32},
    )
    ttgir = compiled.asm["ttgir"]
    assert "async_tdm_copy_global_to_local" in ttgir


def test_async_amd_desc_store_compiles_gfx1250(device):
    """async_amd_descriptor_store produces TDM store ops in TTGIR."""
    compiled = compile_for_gfx1250(
        _async_amd_desc_store_kernel,
        signature={"x_ptr": "*fp16", "y_ptr": "*fp16"},
        constexprs={"M": 32, "N": 32},
    )
    ttgir = compiled.asm["ttgir"]
    assert "async_tdm_copy_global_to_local" in ttgir
    assert "async_tdm_copy_local_to_global" in ttgir
    assert ttgir.count("clamp_bounds") == 2


def test_update_tensor_descriptor_store_compiles_gfx1250(device):
    compiled = compile_for_gfx1250(
        _update_tensor_descriptor_store_kernel,
        signature={"x_ptr": "*fp16", "y_ptr": "*fp16"},
        constexprs={"M": 32, "N": 32},
    )
    ttgir = compiled.asm["ttgir"]
    assert "amdg.update_tensor_descriptor" in ttgir
    assert "set_bounds =" in ttgir
    assert "pred =" in ttgir
    assert "clamp_bounds" in ttgir
    assert "amdg.async_tdm_copy_local_to_global" in ttgir
    assert ttgir.count("clamp_bounds") == 1
    # Two explicit input updates and one output update are the only descriptor
    # mutations. Neither pre-positioned copy may add a no-op update.
    assert ttgir.count("amdg.update_tensor_descriptor") == 3


def test_amd_desc_prefetch_compiles_gfx1250(device):
    """amd_descriptor_prefetch_tensor produces tdm_prefetch in TTGIR."""
    compiled = compile_for_gfx1250(
        _amd_desc_prefetch_kernel,
        signature={"x_ptr": "*fp16", "output_ptr": "*fp16"},
        constexprs={"M": 32, "N": 32},
    )
    ttgir = compiled.asm["ttgir"]
    assert "tdm_prefetch" in ttgir


def test_amd_desc_prefetch_speculative_compiles_gfx1250(device):
    """amd_descriptor_prefetch_tensor with speculative=True compiles."""
    compiled = compile_for_gfx1250(
        _amd_desc_prefetch_speculative_kernel,
        signature={"x_ptr": "*fp16", "output_ptr": "*fp16"},
        constexprs={"M": 32, "N": 32},
    )
    ttgir = compiled.asm["ttgir"]
    assert "tdm_prefetch" in ttgir


def test_async_descriptor_load_rejects_amd(device):
    """NV-only async_descriptor_load raises NotImplementedError on AMD."""

    @triton.jit
    def _kernel(x_ptr, M: tl.constexpr, N: tl.constexpr):
        desc = tl.make_tensor_descriptor(x_ptr, [M, N], [N, 1], [M, N])
        barrier = tlx.alloc_barriers(1)
        buf = tlx.local_alloc((M, N), tl.float16, 1)
        buf0 = tlx.local_view(buf, 0)
        tlx.async_descriptor_load(desc, buf0, [0, 0], barrier)

    with pytest.raises(CompilationError, match="NV-only"):
        compile_for_gfx1250(
            _kernel,
            signature={"x_ptr": "*fp16"},
            constexprs={"M": 32, "N": 32},
        )


def test_async_descriptor_store_rejects_amd(device):
    """NV-only async_descriptor_store raises NotImplementedError on AMD."""

    @triton.jit
    def _kernel(x_ptr, M: tl.constexpr, N: tl.constexpr):
        desc = tl.make_tensor_descriptor(x_ptr, [M, N], [N, 1], [M, N])
        buf = tlx.local_alloc((M, N), tl.float16, 1)
        buf0 = tlx.local_view(buf, 0)
        tlx.async_descriptor_store(desc, buf0, [0, 0])

    with pytest.raises(CompilationError, match="NV-only"):
        compile_for_gfx1250(
            _kernel,
            signature={"x_ptr": "*fp16"},
            constexprs={"M": 32, "N": 32},
        )


def test_async_descriptor_prefetch_rejects_amd(device):
    """NV-only async_descriptor_prefetch_tensor raises NotImplementedError on AMD."""

    @triton.jit
    def _kernel(x_ptr, M: tl.constexpr, N: tl.constexpr):
        desc = tl.make_tensor_descriptor(x_ptr, [M, N], [N, 1], [M, N])
        tlx.async_descriptor_prefetch_tensor(desc, [0, 0])

    with pytest.raises(CompilationError, match="NV-only"):
        compile_for_gfx1250(
            _kernel,
            signature={"x_ptr": "*fp16"},
            constexprs={"M": 32, "N": 32},
        )


def test_padded_layout_local_alloc_compiles_gfx1250(device):
    """local_alloc with an explicit padded_shared_layout_encoding compiles."""

    @triton.jit
    def _kernel(x_ptr, output_ptr, M: tl.constexpr, N: tl.constexpr):
        layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for([(N, 128 // 16)], [M, N])
        buf = tlx.local_alloc((M, N), tl.float16, 1, layout=layout)
        buf0 = tlx.local_view(buf, 0)
        x = tlx.local_load(buf0)
        tl.store(output_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], x)

    compiled = compile_for_gfx1250(
        _kernel,
        signature={"x_ptr": "*fp16", "output_ptr": "*fp16"},
        constexprs={"M": 32, "N": 32},
    )
    ttgir = compiled.asm["ttgir"]
    assert "padded_shared" in ttgir


def test_async_amd_desc_load_auto_propagates_padded_layout_gfx1250(device):
    """Default local_alloc + async_amd_descriptor_load auto-propagates padded encoding."""
    compiled = compile_for_gfx1250(
        _async_amd_desc_load_kernel,
        signature={"x_ptr": "*fp16", "output_ptr": "*fp16"},
        constexprs={"M": 32, "N": 32},
    )
    ttgir = compiled.asm["ttgir"]
    assert "padded_shared" in ttgir


@pytest.mark.parametrize(
    "view, expected_op",
    [(0, "ttg.memdesc_subslice"), (1, "ttg.memdesc_reshape")],
    ids=["slice", "reshape"],
)
def test_pinned_tdm_memdesc_views_compile_gfx1250(device, view, expected_op):
    compiled = compile_for_gfx1250(
        _pinned_tdm_memdesc_view_kernel,
        signature={"input_ptr": "*fp16", "output_ptr": "*fp16"},
        constexprs={"M": 32, "N": 128, "VIEW": view},
    )
    ttgir = compiled.asm["ttgir"]
    assert ttgir.count("amdg.async_tdm_copy_global_to_local") == 1
    assert expected_op in ttgir
    assert "#ttg.padded_shared<[128:+8]" in ttgir
    assert "#tlx.user_layout" not in ttgir
    assert "#tlx.no_verify_layout" not in ttgir
    assert "tlx.require_layout" not in ttgir
    amdgcn = compiled.asm["amdgcn"]
    assert "tensor_load_to_lds" in amdgcn or "tensor.load.to.lds" in amdgcn


def test_amd_tdm_gemm_pipelined_compiles_gfx1250(device):
    """Compile-only: validates TDM GEMM tutorial produces TDM ops + padded encoding."""
    compiled = compile_for_gfx1250(
        _amd_tdm_gemm_kernel,
        signature={
            "a_ptr": "*fp16",
            "b_ptr": "*fp16",
            "c_ptr": "*fp16",
            "M": "i32",
            "N": "i32",
            "K": "i32",
        },
        constexprs={"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
    )
    ttgir = compiled.asm["ttgir"]
    assert "amdg.async_tdm_copy_global_to_local" in ttgir
    assert "amdg.tdm_prefetch" in ttgir
    assert "ttg.padded_shared" in ttgir, "expected propagated padded encoding"
    amdgcn = compiled.asm["amdgcn"]
    assert "tensor_load_to_lds" in amdgcn or "tensor.load.to.lds" in amdgcn


def test_assume_uniform_compiles_gfx950(device):
    """assume_uniform on a loaded pointer produces amdg.assume_uniform in TTIR/TTGIR."""
    compiled = compile_for_gfx950(
        _assume_uniform_ptr_kernel,
        signature={"ptr_array": "*i64", "out_ptr": "*fp32"},
        constexprs={"BLOCK": 64},
    )
    assert "amdg.assume_uniform" in compiled.asm["ttir"]
    assert "amdg.assume_uniform" in compiled.asm["ttgir"]
    assert "amdgcn" in compiled.asm
    assert len(compiled.asm["amdgcn"]) > 0


def test_assume_uniform_emits_readfirstlane_gfx950(device):
    """assume_uniform lowers to readfirstlane beyond what the backend emits anyway.

    The buffer resource descriptor already forces one readfirstlane, so the count
    is compared against the same kernel without the hint rather than asserted
    absolutely.
    """
    signature = {"ptr_array": "*i64", "out_ptr": "*fp32"}
    with_hint = compile_for_gfx950(_assume_uniform_ptr_kernel, signature, {"BLOCK": 64})
    without_hint = compile_for_gfx950(_assume_uniform_ptr_kernel_no_hint, signature, {"BLOCK": 64})
    assert with_hint.asm["llir"].count("readfirstlane") > without_hint.asm["llir"].count("readfirstlane")


@pytest.mark.parametrize("dtype", ["i16", "i32", "i64", "fp16", "bf16", "fp32"])
def test_assume_uniform_scalar_types_compiles_gfx950(device, dtype):
    """assume_uniform accepts every 16/32/64-bit scalar type."""
    compiled = compile_for_gfx950(
        _assume_uniform_scalar_kernel,
        signature={"in_ptr": f"*{dtype}", "out_ptr": "*fp32"},
        constexprs={"BLOCK": 64},
    )
    assert "amdg.assume_uniform" in compiled.asm["ttgir"]
    assert len(compiled.asm["amdgcn"]) > 0


def test_assume_uniform_rejects_narrow_type_gfx950(device):
    """readfirstlane has no sub-16-bit form, so narrower scalars are rejected."""
    with pytest.raises(CompilationError, match="16/32/64-bit"):
        compile_for_gfx950(
            _assume_uniform_scalar_kernel,
            signature={"in_ptr": "*i8", "out_ptr": "*fp32"},
            constexprs={"BLOCK": 64},
        )
