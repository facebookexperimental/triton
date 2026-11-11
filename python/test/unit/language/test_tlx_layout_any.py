"""TLX layout tests -- any GPU arch."""
import copy
import pickle

import pytest
import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
import triton.language.extra.tlx.types as tlx_types

LAYOUT_BUILDERS = {
    "swizzled_shared": lambda v: tlx.swizzled_shared_layout_encoding(v, 1, 1, [1, 0], [1, 1], [1, 1], [1, 1], [1, 0]),
    "padded_shared": lambda v: tlx.padded_shared_layout_encoding([v], [4], [1, 0], [64, 64], [1, 1], [1, 1], [1, 0]),
    "shared_linear": lambda v: tlx.shared_linear_layout_encoding([[0, v], [1, 0]]),
    "nv_mma": lambda v: tlx.nv_mma_layout([v, 1], [16, 64, 16]),
    "amd_mfma": lambda v: tlx.amd_mfma_layout(4, [16, 16, 32], True, [v, 1]),
    "dot_operand": lambda v: tlx.dot_operand_layout(0, tlx.amd_mfma_layout(4, [16, 16, 32], True, [v, 1])),
    "tensor_memory": lambda v: tlx.tensor_memory_layout_encoding(128, 64 * v, 1, 1, 1),
    "tensor_memory_scales": lambda v: tlx.tensor_memory_scales_layout_encoding(v, 1),
    "nv_mma_shared": lambda v: tlx.nv_mma_shared_layout_encoding.make_default([64, 64 * v], tl.float16),
}


@pytest.mark.parametrize("name", LAYOUT_BUILDERS)
def test_layout_equality_follows_fields(name):
    build = LAYOUT_BUILDERS[name]
    a, b, other = build(2), build(2), build(4)

    assert a == b and hash(a) == hash(b) and repr(a) == repr(b)
    assert a != other and repr(a) != repr(other)
    assert name in repr(a)


def test_layout_equality_normalizes_constexpr_and_sequences():
    a = tlx.swizzled_shared_layout_encoding(1, 1, 1, [1, 0], [1, 1], [1, 1], [1, 1], [1, 0])
    b = tlx.swizzled_shared_layout_encoding(tl.constexpr(1), 1, 1, (1, 0), [1, 1], [1, 1], [1, 1], [1, 0])

    assert a == b and hash(a) == hash(b) and repr(a) == repr(b)
    assert a.make_permute([0, 1]) == a


def test_layout_is_frozen_once_hashed():
    a = LAYOUT_BUILDERS["padded_shared"](2)
    a.paddings = [8]  # Not hashed yet: construction-time edits are allowed.
    h = hash(a)
    with pytest.raises(AttributeError, match="immutable"):
        a.paddings = [4]

    pinned = copy.copy(a)
    pinned._tlx_user_pinned = True
    assert pinned != a and hash(a) == h

    restored = pickle.loads(pickle.dumps(a))
    assert restored == a and repr(restored) == repr(a)


def _count_normalizations(monkeypatch):
    calls = []
    freeze = tlx_types._freeze_layout_value

    def counting_freeze(value):
        calls.append(value)
        return freeze(value)

    monkeypatch.setattr(tlx_types, "_freeze_layout_value", counting_freeze)
    return calls


def test_layout_fields_are_normalized_once(monkeypatch):
    calls = _count_normalizations(monkeypatch)
    a, b = LAYOUT_BUILDERS["dot_operand"](2), LAYOUT_BUILDERS["dot_operand"](2)
    assert a == b
    normalized = len(calls)

    for _ in range(100):
        assert hash(a) == hash(b) and a == b
    assert len(calls) == normalized


@triton.jit
def _layout_constexpr_kernel(out_ptr, LAYOUT: tl.constexpr):
    buf = tlx.local_alloc((16, 16), tl.float32, tl.constexpr(1), layout=LAYOUT)
    v = tlx.local_view(buf, 0)
    tlx.local_store(v, tl.zeros((16, 16), tl.float32))
    offs = tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]
    tl.store(out_ptr + offs, tlx.local_load(v))


def test_rebuilt_layout_constexpr_reuses_kernel_cache(device):
    """A layout rebuilt per launch must hit the same JIT cache entry rather than
    add one per launch, and a different layout must not reuse its kernel."""
    out = torch.empty((16, 16), dtype=torch.float32, device=device)
    _layout_constexpr_kernel.device_caches.clear()

    for _ in range(3):
        _layout_constexpr_kernel.warmup(out, tlx.swizzled_shared_layout_encoding.make_default(rank=2), grid=(1, ))
    kernel_cache, kernel_key_cache, *_ = next(iter(_layout_constexpr_kernel.device_caches.values()))
    assert len(kernel_key_cache) == 1 and len(kernel_cache) == 1

    swizzled = tlx.swizzled_shared_layout_encoding(4, 2, 4, [1, 0], [1, 1], [1, 1], [1, 1], [1, 0])
    _layout_constexpr_kernel.warmup(out, swizzled, grid=(1, ))
    assert len(kernel_key_cache) == 2 and len(kernel_cache) == 2


def test_kernel_cache_hit_does_not_renormalize_layout(device, monkeypatch):
    """Steady state: relaunching with the same layout object reuses its
    memoized key instead of re-normalizing the fields on every lookup."""
    out = torch.empty((16, 16), dtype=torch.float32, device=device)
    layout = tlx.swizzled_shared_layout_encoding(4, 2, 4, [1, 0], [1, 1], [1, 1], [1, 1], [1, 0])
    _layout_constexpr_kernel.warmup(out, layout, grid=(1, ))

    calls = _count_normalizations(monkeypatch)
    for _ in range(10):
        _layout_constexpr_kernel.warmup(out, layout, grid=(1, ))
    assert not calls
