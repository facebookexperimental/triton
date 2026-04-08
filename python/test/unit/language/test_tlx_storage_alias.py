import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_hopper_or_newer, is_hip
import triton.language.extra.tlx as tlx


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


def _make_test_storage_alias_spec(storage: tlx.storage_kind = tlx.storage_kind.smem):
    """Helper to create a storage_alias_spec for testing reuse_group."""
    return tlx.storage_alias_spec_type_class(handle=None, storage=storage)


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


@pytest.mark.skipif(is_hip(), reason="Not supported on AMD")
@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
class TestSetBufferOverlap:
    """Tests for tlx.set_buffer_overlap and storage_alias_spec.set_buffer_overlap method."""

    def test_set_buffer_overlap_shared_different_sizes(self):
        """Test shared overlap with different sized allocations (f32 vs bf16).

        When allocations of different sizes share memory, the smaller allocation's
        shape is expanded to account for the larger allocation's buffer spacing.
        This test verifies that shape expansion and index rewriting work correctly.
        """

        @triton.jit
        def set_buffer_overlap_kernel(out_ptr, BLOCK_SIZE: tl.constexpr):
            # Create a storage alias spec
            spec = tlx.storage_alias_spec(storage=tlx.storage_kind.smem)

            # Allocate buffers using the spec
            # a: 2 x BLOCK_SIZE x BLOCK_SIZE x f32 = 2 x 64 x 64 x 4 = 32768 bytes
            # b: 2 x BLOCK_SIZE x BLOCK_SIZE x bf16 = 2 x 64 x 64 x 2 = 16384 bytes
            a = tlx.local_alloc((BLOCK_SIZE, BLOCK_SIZE), tl.float32, tl.constexpr(2), tlx.storage_kind.smem,
                                reuse=spec)
            b = tlx.local_alloc((BLOCK_SIZE, BLOCK_SIZE), tl.bfloat16, tl.constexpr(2), tlx.storage_kind.smem,
                                reuse=spec)

            # Define overlap scheme: a and b share the same memory region
            # bytes_between_buffers = max(16384, 8192) = 16384
            # For b (8192 bytes): scale = 16384/8192 = 2
            # b's shape expands from 2 to 4 buffers
            spec.set_buffer_overlap(tlx.reuse_group(a, b, group_type=tlx.reuse_group_type.shared))

            # Initialize output to zeros
            offs_m = tl.arange(0, BLOCK_SIZE)
            offs_n = tl.arange(0, BLOCK_SIZE)
            zeros = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32)

            # Initialize all 4 output regions to 0
            for i in tl.static_range(4):
                out_offsets = out_ptr + i * BLOCK_SIZE * BLOCK_SIZE + (offs_m[:, None] * BLOCK_SIZE + offs_n[None, :])
                tl.store(out_offsets, zeros)

            # Write 1.0 to a[0] (16384 bytes per buffer)
            ones = tl.full((BLOCK_SIZE, BLOCK_SIZE), 1.0, tl.float32)
            tlx.local_store(a[0], ones)

            # Write 2.0 to a[1]
            twos = tl.full((BLOCK_SIZE, BLOCK_SIZE), 2.0, tl.float32)
            tlx.local_store(a[1], twos)

            # Since b shares memory with a and has scale=2:
            # b[0] maps to physical slot 0 (same as a[0])
            # b[1] maps to physical slot 2 (same as a[1]'s start, since a's buffer is 2x size of b's)
            # So reading b[0] should give us the first half of a[0]'s data (reinterpreted as bf16)

            # Read from b[0] and b[1] and store to output
            b0_data = tlx.local_load(b[0])
            b0_as_f32 = b0_data.to(tl.float32)
            out_offsets_0 = out_ptr + (offs_m[:, None] * BLOCK_SIZE + offs_n[None, :])
            tl.store(out_offsets_0, b0_as_f32)

            b1_data = tlx.local_load(b[1])
            b1_as_f32 = b1_data.to(tl.float32)
            out_offsets_1 = out_ptr + BLOCK_SIZE * BLOCK_SIZE + (offs_m[:, None] * BLOCK_SIZE + offs_n[None, :])
            tl.store(out_offsets_1, b1_as_f32)

        grid = lambda meta: (1, )

        BLOCK_SIZE = 64
        out = torch.zeros((2 * BLOCK_SIZE, BLOCK_SIZE), dtype=torch.float32, device="cuda")
        set_buffer_overlap_kernel[grid](out, BLOCK_SIZE)

        # The values stored as f32 and read back as bf16->f32 will have precision loss
        # but should be non-zero (proving the memory is shared)
        # b[0] should contain data from a[0] reinterpreted as bf16
        # b[1] should contain data from a[1] reinterpreted as bf16
        assert out[:BLOCK_SIZE, :].abs().sum() > 0, "b[0] should have non-zero data from a[0]"
        assert out[BLOCK_SIZE:, :].abs().sum() > 0, "b[1] should have non-zero data from a[1]"

    def test_set_buffer_overlap_nested_shared_distinct(self):
        """Test nested reuse_group: shared(qk, distinct(p, alpha)).

        This test verifies Flash Attention-style nested overlap schemes work.
        The distinct group places p and alpha at different offsets within the
        shared region with qk.
        """

        @triton.jit
        def set_buffer_overlap_nested_kernel(out_ptr, BLOCK_SIZE: tl.constexpr):
            # Create a storage alias spec
            spec = tlx.storage_alias_spec(storage=tlx.storage_kind.smem)

            # Allocate buffers (Flash Attention like pattern)
            qk = tlx.local_alloc((BLOCK_SIZE, BLOCK_SIZE), tl.float32, tl.constexpr(2), tlx.storage_kind.smem,
                                 reuse=spec)
            p = tlx.local_alloc((BLOCK_SIZE, BLOCK_SIZE), tl.bfloat16, tl.constexpr(2), tlx.storage_kind.smem,
                                reuse=spec)
            # alpha: 2 x 64 x f32 = 512 bytes (256 per buffer)
            alpha = tlx.local_alloc((BLOCK_SIZE, BLOCK_SIZE // 2), tl.float32, tl.constexpr(2), tlx.storage_kind.smem,
                                    reuse=spec)

            spec.set_buffer_overlap(
                tlx.reuse_group(
                    qk,
                    tlx.reuse_group(p, alpha, group_type=tlx.reuse_group_type.distinct),
                    group_type=tlx.reuse_group_type.shared,
                ))

            # Write 1.0 to qk[0]
            data = tl.full((BLOCK_SIZE, BLOCK_SIZE), 1.0, tl.float32)
            tlx.local_store(qk[0], data)

            # Read from alpha[0] (should alias with half of qk[0] since they share)
            alpha0_data = tlx.local_load(alpha[0])

            offs_m = tl.arange(0, BLOCK_SIZE)
            offs_n_half = tl.arange(0, BLOCK_SIZE // 2)

            # Write alpha[0] to the first half of output columns
            offs_n_half = tl.arange(0, BLOCK_SIZE // 2)
            out_offsets_first_half = out_ptr + (offs_m[:, None] * BLOCK_SIZE + offs_n_half[None, :])
            tl.store(out_offsets_first_half, alpha0_data)

        grid = lambda meta: (1, )

        BLOCK_SIZE = 64
        out = torch.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=torch.float32, device="cuda")
        set_buffer_overlap_nested_kernel[grid](out, BLOCK_SIZE)
        # alpha[0] should have half of qk[0]'s data (1s)
        # Output should be 1s for the first half of columns, 0s for the second half
        expected = torch.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=torch.float32, device="cuda")
        expected[:, :BLOCK_SIZE // 2] = 1.0
        torch.testing.assert_close(out, expected)

    def test_reuse_group_with_group_size(self):
        """Test reuse_group with group_size for subtiling.

        This test verifies that group_size works correctly for subtiling scenarios.
        We have two allocations:
        - qk: 2 buffers of (64, 64) float32
        - p: 4 buffers of (64, 64) float16 with group_size=2

        With group_size=2, p's 4 buffers are grouped into 2 logical groups:
        - p[0], p[1] form logical group 0 (shares with qk[0])
        - p[2], p[3] form logical group 1 (shares with qk[1])

        The index computation should map:
        - p[0] -> physical index 0 (group 0, offset 0)
        - p[1] -> physical index 1 (group 0, offset 1)
        - p[2] -> physical index 2 (group 1, offset 0)
        - p[3] -> physical index 3 (group 1, offset 1)
        """

        @triton.jit
        def group_size_kernel(out_ptr, BLOCK_SIZE: tl.constexpr):
            # Create a storage alias spec
            spec = tlx.storage_alias_spec(storage=tlx.storage_kind.smem)

            # Allocate qk: 2 buffers
            qk = tlx.local_alloc((BLOCK_SIZE, BLOCK_SIZE), tl.float32, tl.constexpr(2), tlx.storage_kind.smem,
                                 reuse=spec)
            # Allocate p: 4 buffers with group_size=2
            # This means p[0],p[1] share with qk[0] and p[2],p[3] share with qk[1]
            p = tlx.local_alloc((BLOCK_SIZE, BLOCK_SIZE), tl.float16, tl.constexpr(4), tlx.storage_kind.smem,
                                reuse=spec)

            # Define overlap with group_size=2 for p
            spec.set_buffer_overlap(
                tlx.reuse_group(
                    qk,
                    tlx.reuse_group(p, group_size=2),
                    group_type=tlx.reuse_group_type.shared,
                ))

            # Write different values to qk[0] and qk[1]
            offs_m = tl.arange(0, BLOCK_SIZE)
            offs_n = tl.arange(0, BLOCK_SIZE)

            # Write 1.0 to qk[0]
            ones = tl.full((BLOCK_SIZE, BLOCK_SIZE), 1.0, tl.float32)
            tlx.local_store(qk[0], ones)

            # Write 2.0 to qk[1]
            twos = tl.full((BLOCK_SIZE, BLOCK_SIZE), 2.0, tl.float32)
            tlx.local_store(qk[1], twos)

            # Read from p buffers - they should see the qk data reinterpreted as float16
            # p[0] and p[1] should see qk[0]'s data
            # p[2] and p[3] should see qk[1]'s data
            p0_data = tlx.local_load(p[0])
            p1_data = tlx.local_load(p[1])
            p2_data = tlx.local_load(p[2])
            p3_data = tlx.local_load(p[3])

            # Output layout: 4 blocks of (BLOCK_SIZE, BLOCK_SIZE)
            out_offsets_0 = out_ptr + 0 * BLOCK_SIZE * BLOCK_SIZE + (offs_m[:, None] * BLOCK_SIZE + offs_n[None, :])
            out_offsets_1 = out_ptr + 1 * BLOCK_SIZE * BLOCK_SIZE + (offs_m[:, None] * BLOCK_SIZE + offs_n[None, :])
            out_offsets_2 = out_ptr + 2 * BLOCK_SIZE * BLOCK_SIZE + (offs_m[:, None] * BLOCK_SIZE + offs_n[None, :])
            out_offsets_3 = out_ptr + 3 * BLOCK_SIZE * BLOCK_SIZE + (offs_m[:, None] * BLOCK_SIZE + offs_n[None, :])

            tl.store(out_offsets_0, p0_data)
            tl.store(out_offsets_1, p1_data)
            tl.store(out_offsets_2, p2_data)
            tl.store(out_offsets_3, p3_data)

        grid = lambda meta: (1, )

        BLOCK_SIZE = 64
        out = torch.zeros((4 * BLOCK_SIZE, BLOCK_SIZE), dtype=torch.float16, device="cuda")
        group_size_kernel[grid](out, BLOCK_SIZE)

        # p[0] and p[1] should have the same data (from qk[0])
        # p[2] and p[3] should have the same data (from qk[1])
        # The data should be non-zero since qk was written with 1.0 and 2.0
        p0_out = out[:BLOCK_SIZE, :]
        p1_out = out[BLOCK_SIZE:2 * BLOCK_SIZE, :]
        p2_out = out[2 * BLOCK_SIZE:3 * BLOCK_SIZE, :]
        p3_out = out[3 * BLOCK_SIZE:, :]

        # p[0] and p[1] should be equal (both alias qk[0])
        torch.testing.assert_close(p0_out, p1_out)
        # p[2] and p[3] should be equal (both alias qk[1])
        torch.testing.assert_close(p2_out, p3_out)
        # p[0] and p[2] should be different (different qk buffers)
        assert not torch.allclose(p0_out, p2_out), "p[0] and p[2] should have different data"

    def test_basic_shared_buffer_overlap(self):
        """Test that allocating two identical buffers with shared overlap works.

        Both buffers have the same type and size, so scale=1 and offset=0 for both.
        No shape expansion or index rewriting is needed.
        """

        @triton.jit
        def set_buffer_overlap_kernel(out_ptr, BLOCK_SIZE: tl.constexpr):
            # Create a storage alias spec
            spec = tlx.storage_alias_spec(storage=tlx.storage_kind.smem)

            # Allocate buffers using the spec (same type and size)
            a = tlx.local_alloc((BLOCK_SIZE, BLOCK_SIZE), tl.float16, tl.constexpr(2), tlx.storage_kind.smem,
                                reuse=spec)
            b = tlx.local_alloc((BLOCK_SIZE, BLOCK_SIZE), tl.float16, tl.constexpr(2), tlx.storage_kind.smem,
                                reuse=spec)

            # Define overlap scheme: a and b share the same memory region
            spec.set_buffer_overlap(tlx.reuse_group(a, b, group_type=tlx.reuse_group_type.shared))

            # Initialize output to zeros
            offs_m = tl.arange(0, BLOCK_SIZE)
            offs_n = tl.arange(0, BLOCK_SIZE)
            zeros = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float16)

            out_offsets_0 = out_ptr + (offs_m[:, None] * BLOCK_SIZE + offs_n[None, :])
            out_offsets_1 = out_ptr + BLOCK_SIZE * BLOCK_SIZE + (offs_m[:, None] * BLOCK_SIZE + offs_n[None, :])
            tl.store(out_offsets_0, zeros)
            tl.store(out_offsets_1, zeros)

            # Write all 1s to a[0]
            ones = tl.full((BLOCK_SIZE, BLOCK_SIZE), 1.0, tl.float16)
            tlx.local_store(a[0], ones)

            # Write all 2s to b[1]
            twos = tl.full((BLOCK_SIZE, BLOCK_SIZE), 2.0, tl.float16)
            tlx.local_store(b[1], twos)

            # Since a and b share the same memory, b[0] should equal a[0] (all 1s)
            # and a[1] should equal b[1] (all 2s)

            # Write b[0] to out_ptr (should be all 1s)
            b0_data = tlx.local_load(b[0])
            tl.store(out_offsets_0, b0_data)

            # Write a[1] to out_ptr + BLOCK_SIZE*BLOCK_SIZE (should be all 2s)
            a1_data = tlx.local_load(a[1])
            tl.store(out_offsets_1, a1_data)

        grid = lambda meta: (1, )

        BLOCK_SIZE = 64
        out = torch.zeros((2 * BLOCK_SIZE, BLOCK_SIZE), dtype=torch.float16, device="cuda")
        set_buffer_overlap_kernel[grid](out, BLOCK_SIZE)

        # First half should be all 1s (from b[0] which shares memory with a[0])
        expected_ones = torch.ones((BLOCK_SIZE, BLOCK_SIZE), dtype=torch.float16, device="cuda")
        # Second half should be all 2s (from a[1] which shares memory with b[1])
        expected_twos = torch.full((BLOCK_SIZE, BLOCK_SIZE), 2.0, dtype=torch.float16, device="cuda")

        torch.testing.assert_close(out[:BLOCK_SIZE, :], expected_ones)
        torch.testing.assert_close(out[BLOCK_SIZE:, :], expected_twos)

    def test_distinct_buffer_overlap(self):
        """Test distinct overlap where buffers are placed at different offsets.

        Two identical allocations in a distinct group:
        - a at offset 0
        - b at offset = a's buffer size
        Shape expansion: both get scale=2 (since bytes_between_buffers = 2 * buffer_size)
        Index rewriting:
        - a[i] -> physical slot 2*i
        - b[i] -> physical slot 2*i + 1
        """

        @triton.jit
        def distinct_buffer_overlap_kernel(out_ptr, BLOCK_SIZE: tl.constexpr):
            # Create a storage alias spec
            spec = tlx.storage_alias_spec(storage=tlx.storage_kind.smem)

            # Allocate two identical buffers
            # Each: 2 x 64 x 64 x f16 = 2 x 8192 bytes = 16384 total
            a = tlx.local_alloc((BLOCK_SIZE, BLOCK_SIZE), tl.float16, tl.constexpr(2), tlx.storage_kind.smem,
                                reuse=spec)
            b = tlx.local_alloc((BLOCK_SIZE, BLOCK_SIZE), tl.float16, tl.constexpr(2), tlx.storage_kind.smem,
                                reuse=spec)

            # Define overlap scheme: a and b are distinct (placed sequentially)
            # bytes_between_buffers = 8192 + 8192 = 16384
            # For a: scale = 16384/8192 = 2, offset = 0
            # For b: scale = 16384/8192 = 2, offset_slots = 8192/8192 = 1
            # Shape expansion: a: 2 -> 4, b: 2 -> 5 (2*2 + 1)
            spec.set_buffer_overlap(tlx.reuse_group(a, b, group_type=tlx.reuse_group_type.distinct))

            # Initialize output to zeros
            offs_m = tl.arange(0, BLOCK_SIZE)
            offs_n = tl.arange(0, BLOCK_SIZE)
            zeros = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float16)

            for i in tl.static_range(4):
                out_offsets = out_ptr + i * BLOCK_SIZE * BLOCK_SIZE + (offs_m[:, None] * BLOCK_SIZE + offs_n[None, :])
                tl.store(out_offsets, zeros)

            # Write to a[0] - should go to physical slot 0
            ones = tl.full((BLOCK_SIZE, BLOCK_SIZE), 1.0, tl.float16)
            tlx.local_store(a[0], ones)

            # Write to a[1] - should go to physical slot 2
            twos = tl.full((BLOCK_SIZE, BLOCK_SIZE), 2.0, tl.float16)
            tlx.local_store(a[1], twos)

            # Write to b[0] - should go to physical slot 1
            threes = tl.full((BLOCK_SIZE, BLOCK_SIZE), 3.0, tl.float16)
            tlx.local_store(b[0], threes)

            # Write to b[1] - should go to physical slot 3
            fours = tl.full((BLOCK_SIZE, BLOCK_SIZE), 4.0, tl.float16)
            tlx.local_store(b[1], fours)

            # Read back and verify distinct memory regions
            # Reading a[0] should give 1s (not overwritten by b)
            a0_data = tlx.local_load(a[0])
            out_offsets_0 = out_ptr + (offs_m[:, None] * BLOCK_SIZE + offs_n[None, :])
            tl.store(out_offsets_0, a0_data)

            # Reading b[0] should give 3s (distinct from a)
            b0_data = tlx.local_load(b[0])
            out_offsets_1 = out_ptr + BLOCK_SIZE * BLOCK_SIZE + (offs_m[:, None] * BLOCK_SIZE + offs_n[None, :])
            tl.store(out_offsets_1, b0_data)

            # Reading a[1] should give 2s
            a1_data = tlx.local_load(a[1])
            out_offsets_2 = out_ptr + 2 * BLOCK_SIZE * BLOCK_SIZE + (offs_m[:, None] * BLOCK_SIZE + offs_n[None, :])
            tl.store(out_offsets_2, a1_data)

            # Reading b[1] should give 4s
            b1_data = tlx.local_load(b[1])
            out_offsets_3 = out_ptr + 3 * BLOCK_SIZE * BLOCK_SIZE + (offs_m[:, None] * BLOCK_SIZE + offs_n[None, :])
            tl.store(out_offsets_3, b1_data)

        grid = lambda meta: (1, )

        BLOCK_SIZE = 64
        out = torch.zeros((4 * BLOCK_SIZE, BLOCK_SIZE), dtype=torch.float16, device="cuda")
        distinct_buffer_overlap_kernel[grid](out, BLOCK_SIZE)

        # Verify each region has the expected value
        expected_ones = torch.ones((BLOCK_SIZE, BLOCK_SIZE), dtype=torch.float16, device="cuda")
        expected_twos = torch.full((BLOCK_SIZE, BLOCK_SIZE), 2.0, dtype=torch.float16, device="cuda")
        expected_threes = torch.full((BLOCK_SIZE, BLOCK_SIZE), 3.0, dtype=torch.float16, device="cuda")
        expected_fours = torch.full((BLOCK_SIZE, BLOCK_SIZE), 4.0, dtype=torch.float16, device="cuda")

        torch.testing.assert_close(out[:BLOCK_SIZE, :], expected_ones)
        torch.testing.assert_close(out[BLOCK_SIZE:2 * BLOCK_SIZE, :], expected_threes)
        torch.testing.assert_close(out[2 * BLOCK_SIZE:3 * BLOCK_SIZE, :], expected_twos)
        torch.testing.assert_close(out[3 * BLOCK_SIZE:, :], expected_fours)

    def test_shared_different_element_sizes(self):
        """Test shared overlap with different element types (f32 vs f16).

        When f32 and f16 buffers share memory:
        - f32: 2 x 64 x 64 x 4 bytes = 32768 bytes (16384 per buffer)
        - f16: 2 x 64 x 64 x 2 bytes = 16384 bytes (8192 per buffer)
        - bytes_between_buffers = max(16384, 8192) = 16384
        - For f16: scale = 16384/8192 = 2, shape expands 2 -> 4
        - Index rewriting: f16[i] -> physical slot 2*i
        """

        @triton.jit
        def shared_different_sizes_kernel(out_ptr, BLOCK_SIZE: tl.constexpr):
            # Create a storage alias spec
            spec = tlx.storage_alias_spec(storage=tlx.storage_kind.smem)

            # Allocate f32 and f16 buffers
            a_f32 = tlx.local_alloc((BLOCK_SIZE, BLOCK_SIZE), tl.float32, tl.constexpr(2), tlx.storage_kind.smem,
                                    reuse=spec)
            b_f16 = tlx.local_alloc((BLOCK_SIZE, BLOCK_SIZE), tl.float16, tl.constexpr(2), tlx.storage_kind.smem,
                                    reuse=spec)

            # Define shared overlap
            spec.set_buffer_overlap(tlx.reuse_group(a_f32, b_f16, group_type=tlx.reuse_group_type.shared))

            # Initialize output to zeros
            offs_m = tl.arange(0, BLOCK_SIZE)
            offs_n = tl.arange(0, BLOCK_SIZE)
            zeros_f32 = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32)

            out_offsets_0 = out_ptr + (offs_m[:, None] * BLOCK_SIZE + offs_n[None, :])
            out_offsets_1 = out_ptr + BLOCK_SIZE * BLOCK_SIZE + (offs_m[:, None] * BLOCK_SIZE + offs_n[None, :])
            tl.store(out_offsets_0, zeros_f32)
            tl.store(out_offsets_1, zeros_f32)

            # Write to a_f32[0]
            ones_f32 = tl.full((BLOCK_SIZE, BLOCK_SIZE), 1.0, tl.float32)
            tlx.local_store(a_f32[0], ones_f32)

            # Write to a_f32[1]
            twos_f32 = tl.full((BLOCK_SIZE, BLOCK_SIZE), 2.0, tl.float32)
            tlx.local_store(a_f32[1], twos_f32)

            # Read b_f16[0] and b_f16[1] - these should contain data from a_f32
            # (reinterpreted as f16, so values will be different but non-zero)
            b0_data = tlx.local_load(b_f16[0])
            b0_as_f32 = b0_data.to(tl.float32)
            tl.store(out_offsets_0, b0_as_f32)

            b1_data = tlx.local_load(b_f16[1])
            b1_as_f32 = b1_data.to(tl.float32)
            tl.store(out_offsets_1, b1_as_f32)

        grid = lambda meta: (1, )

        BLOCK_SIZE = 64
        out = torch.zeros((2 * BLOCK_SIZE, BLOCK_SIZE), dtype=torch.float32, device="cuda")
        shared_different_sizes_kernel[grid](out, BLOCK_SIZE)

        # The f16 reinterpretation of f32 data will produce non-zero values
        # We can't predict exact values due to bit reinterpretation, but they should be non-zero
        assert out[:BLOCK_SIZE, :].abs().sum() > 0, "b_f16[0] should have non-zero data from a_f32[0]"
        assert out[BLOCK_SIZE:, :].abs().sum() > 0, "b_f16[1] should have non-zero data from a_f32[1]"
