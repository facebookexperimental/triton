"""
Unit tests for the storage_alias_spec API.

These tests validate the Python-level interface for storage_alias_spec
without requiring full compilation, as the lowering pipeline is not yet complete.
"""

import pytest
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
