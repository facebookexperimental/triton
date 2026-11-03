"""Architecture-independent coverage for the shared TLX SM holdout.

The count comes from the process-wide PyTorch carveout; 0 (or no API) keeps
historical behavior. No GPU is touched except where noted: the carveout
stream test degrades to a no-op off HIP.
"""
import pytest
import torch
from triton.tlx.ops import InvalidInput
from triton.tlx.ops._holdout import (
    _device_sm_count_cached,
    cap_grid,
    carveout_stream,
    cu_mask_drop_first,
    cu_mask_words,
    device_sm_count,
    green_sm_count,
    holdout_sms,
    num_sms,
)
from triton.tlx.ops.kernels.flash_attn import gfx950 as flash_attn_gfx950
from triton.tlx.ops.kernels.flash_attn_varlen import gfx950 as flash_attn_varlen_gfx950
from triton.tlx.ops.kernels.flash_attn_varlen import sm90 as flash_attn_varlen_sm90
from triton.tlx.ops.kernels.grouped_gemm import sm100 as grouped_gemm_sm100
from triton.tlx.ops.kernels.hstu_cross_attn import gfx950 as hstu_cross_attn_gfx950
from triton.tlx.ops.kernels.kda import sm100 as kda_sm100
from triton.tlx.ops.kernels.mm import sm100 as mm_sm100
from triton.tlx.ops.kernels.mm_mxfp8 import sm100 as mm_mxfp8_sm100


@pytest.mark.parametrize("torch_value,expected", [(None, 0), (0, 0), (2, 2), (32, 32)])
def test_holdout_sms_torch_carveout(monkeypatch, torch_value, expected):
    monkeypatch.setattr(torch._C, "_get_sm_carveout_experimental", lambda: torch_value)
    assert holdout_sms() == expected


def test_holdout_sms_without_torch_api(monkeypatch):
    monkeypatch.delattr(torch._C, "_get_sm_carveout_experimental")
    assert holdout_sms() == 0


@pytest.mark.parametrize("torch_value", [-1, "16", True])
def test_holdout_sms_rejects_bad_torch_value(monkeypatch, torch_value):
    monkeypatch.setattr(torch._C, "_get_sm_carveout_experimental", lambda: torch_value)
    with pytest.raises(InvalidInput, match="torch SM carveout"):
        holdout_sms()


@pytest.mark.parametrize("carveout,grid,expected", [
    (None, 256, 256),
    (32, 256, 224),
    (16, 148, 132),
    (256, 256, 1),
    (240, 224, 1),
])
def test_cap_grid(monkeypatch, carveout, grid, expected):
    monkeypatch.setattr(torch._C, "_get_sm_carveout_experimental", lambda: carveout)
    assert cap_grid(grid) == expected


@pytest.mark.parametrize("holdout,total,expected", [
    (0, 256, (0, ) * 8),
    (8, 256, (0xFF, ) + (0, ) * 7),
    (32, 256, (0xFFFFFFFF, ) + (0, ) * 7),
    (48, 256, (0xFFFFFFFF, 0xFFFF) + (0, ) * 6),
    (256, 256, (0xFFFFFFFF, ) * 8),
    (16, 148, (0xFFFF, 0, 0, 0, 0)),
])
def test_cu_mask_words(holdout, total, expected):
    assert cu_mask_words(holdout, total) == expected


@pytest.mark.parametrize("holdout,total,expected", [
    (0, 256, (0xFFFFFFFF, ) * 8),
    (32, 256, (0, ) + (0xFFFFFFFF, ) * 7),
    (48, 256, (0, 0xFFFF0000) + (0xFFFFFFFF, ) * 6),
    (256, 256, (0, ) * 8),
    (16, 148, (0xFFFF0000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFF)),
])
def test_cu_mask_drop_first(holdout, total, expected):
    assert cu_mask_drop_first(holdout, total) == expected


def test_get_num_sms_tracks_carveout_changes(monkeypatch):
    """Per-op SM counts must see carveout changes; only device props are cached."""
    from types import SimpleNamespace

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *args, **kwargs: SimpleNamespace(multi_processor_count=148),
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    carveout = {"value": 0}
    monkeypatch.setattr(torch._C, "_get_sm_carveout_experimental", lambda: carveout["value"])
    _device_sm_count_cached.cache_clear()
    try:
        assert num_sms() == 148
        assert mm_sm100._get_num_sms() == 148
        carveout["value"] = 16
        assert num_sms() == 132
        assert mm_sm100._get_num_sms() == 132
    finally:
        _device_sm_count_cached.cache_clear()


def test_get_num_sms_device_index_tracks_carveout_changes(monkeypatch):
    """Device-keyed SM counts must also see carveout changes."""
    from types import SimpleNamespace

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *args, **kwargs: SimpleNamespace(multi_processor_count=148),
    )
    carveout = {"value": 0}
    monkeypatch.setattr(torch._C, "_get_sm_carveout_experimental", lambda: carveout["value"])
    _device_sm_count_cached.cache_clear()
    try:
        assert num_sms(0) == 148
        assert grouped_gemm_sm100._get_num_sms(0) == 148
        carveout["value"] = 16
        assert num_sms(0) == 132
        assert grouped_gemm_sm100._get_num_sms(0) == 132
    finally:
        _device_sm_count_cached.cache_clear()


def test_num_sms_accepts_up_to_sm_count_minus_one(monkeypatch):
    """Holdout may reserve all but one SM; reserving every SM is rejected."""
    from types import SimpleNamespace

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *args, **kwargs: SimpleNamespace(multi_processor_count=148),
    )
    carveout = {"value": 147}
    monkeypatch.setattr(torch._C, "_get_sm_carveout_experimental", lambda: carveout["value"])
    _device_sm_count_cached.cache_clear()
    try:
        assert num_sms(0) == 1
        carveout["value"] = 148
        with pytest.raises(InvalidInput, match="leaves no SMs"):
            num_sms(0)
    finally:
        _device_sm_count_cached.cache_clear()


@pytest.mark.parametrize("module,helper", [
    (mm_mxfp8_sm100, "_num_sms"),
    (flash_attn_varlen_gfx950, "_num_cus"),
    (flash_attn_varlen_sm90, "_num_sms"),
    (hstu_cross_attn_gfx950, "_num_cus"),
])
def test_aliased_sm_helpers_track_carveout_changes(monkeypatch, module, helper):
    """Per-op aliases of the shared `num_sms` must see carveout changes."""
    from types import SimpleNamespace

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *args, **kwargs: SimpleNamespace(multi_processor_count=256),
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    carveout = {"value": 0}
    monkeypatch.setattr(torch._C, "_get_sm_carveout_experimental", lambda: carveout["value"])
    _device_sm_count_cached.cache_clear()
    try:
        assert device_sm_count(0) == 256
        assert num_sms(torch.device("cuda", 0)) == 256
        assert getattr(module, helper)(0) == 256
        carveout["value"] = 32
        assert num_sms(0) == 224
        assert getattr(module, helper)(0) == 224
    finally:
        _device_sm_count_cached.cache_clear()


def _fake_holdout_streams(monkeypatch):
    """Stub the HIP/CUDA surface `carveout_stream` needs (no GPU touched)."""
    import contextlib
    from types import SimpleNamespace

    from triton.tlx.ops import _holdout

    monkeypatch.setattr(torch._C, "_get_sm_carveout_experimental", lambda: 32)
    monkeypatch.setattr(torch.version, "hip", "7.0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *args, **kwargs: SimpleNamespace(multi_processor_count=256),
    )

    waits = []

    class FakeStream:

        def __init__(self, handle):
            self.cuda_stream = handle

        def wait_stream(self, other):
            waits.append((self.cuda_stream, other.cuda_stream))

    masked = FakeStream(123)
    ambient = FakeStream(456)
    monkeypatch.setattr(_holdout, "_masked_stream", lambda *a, **k: masked)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a, **k: ambient)
    monkeypatch.setattr(torch.cuda, "stream", lambda *a, **k: contextlib.nullcontext())
    return _holdout, masked, ambient, waits


def test_carveout_stream_syncs_when_body_raises(monkeypatch):
    """A raising body must still post-sync the ambient stream (no cross-stream race)."""
    _holdout, masked, ambient, waits = _fake_holdout_streams(monkeypatch)

    with pytest.raises(RuntimeError, match="boom"):
        with carveout_stream():
            raise RuntimeError("boom")
    # pre-sync (masked waits on ambient) + post-sync (ambient waits on
    # masked), both via PyTorch's Stream.wait_stream helper.
    assert waits == [(masked.cuda_stream, ambient.cuda_stream), (ambient.cuda_stream, masked.cuda_stream)]


def test_carveout_stream_noop_when_already_masked(monkeypatch):
    """Nested entry while already on the masked stream skips any re-sync."""
    _holdout, masked, ambient, waits = _fake_holdout_streams(monkeypatch)

    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a, **k: masked)

    def fail_stream(*a, **k):
        raise AssertionError("must not switch streams when already masked")

    monkeypatch.setattr(torch.cuda, "stream", fail_stream)

    ran = []
    with carveout_stream():
        ran.append(True)
    assert ran == [True]
    assert waits == []


def test_carveout_stream_noop_without_carveout(monkeypatch):
    monkeypatch.setattr(torch._C, "_get_sm_carveout_experimental", lambda: None)
    with carveout_stream():
        pass


def test_carveout_stream_noop_when_cuda_unavailable(monkeypatch):
    """CPU-only: must yield without resolving the current CUDA device."""
    monkeypatch.setattr(torch._C, "_get_sm_carveout_experimental", lambda: 32)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def _no_cuda():
        raise AssertionError("CUDA device resolved while CUDA is unavailable")

    monkeypatch.setattr(torch.cuda, "current_device", _no_cuda)
    ran = []
    with carveout_stream():
        ran.append(True)
    assert ran == [True]


def test_destroy_masked_streams_never_raises():
    """Teardown swallows FFI errors and still clears the stream cache."""
    import ctypes

    from triton.tlx.ops import _holdout

    class _BadLib:

        def hipStreamDestroy(self, handle):
            raise ctypes.ArgumentError("interpreter shutting down")

    _holdout._MASKED_STREAMS[(0, 32, 256)] = (object(), 123, _BadLib())
    _holdout._destroy_masked_streams()
    assert _holdout._MASKED_STREAMS == {}


def test_hip_lib_is_cached(monkeypatch):
    """The HIP library handle is loaded through PyTorch once, not per stream."""
    from unittest import mock

    from triton.tlx.ops import _holdout

    fake = mock.MagicMock()
    calls = []

    def _load():
        calls.append(1)
        return fake

    monkeypatch.setattr("torch.cuda._utils._get_hip_runtime_library", _load)
    _holdout._hip_lib.cache_clear()
    try:
        assert _holdout._hip_lib() is fake
        assert _holdout._hip_lib() is fake
        assert len(calls) == 1
    finally:
        _holdout._hip_lib.cache_clear()


def _fake_green_driver(sm_count=64):
    from types import SimpleNamespace

    class _Drv:
        CUdevResourceType = SimpleNamespace(CU_DEV_RESOURCE_TYPE_SM="sm")

        @staticmethod
        def cuStreamGetGreenCtx(handle):
            return (0, "green-ctx")

        @staticmethod
        def cuGreenCtxGetDevResource(green, resource_type):
            return (0, SimpleNamespace(sm=SimpleNamespace(smCount=sm_count)))

    return _Drv


def test_stream_green_sms_via_driver(monkeypatch):
    """Stream -> green context -> SM resource yields the partition size."""
    from types import SimpleNamespace

    from triton.tlx.ops import _holdout

    monkeypatch.setattr(_holdout, "_green_driver", _fake_green_driver)
    monkeypatch.setattr("torch.cuda._utils._check_cuda_bindings", lambda result: result[1])
    assert _holdout._stream_green_sms(SimpleNamespace(cuda_stream=999)) == 64


def test_stream_green_sms_degrades_without_green(monkeypatch):
    """Non-green streams and missing bindings both degrade to None."""
    from types import SimpleNamespace

    from triton.tlx.ops import _holdout

    monkeypatch.setattr(_holdout, "_green_driver", lambda: None)
    assert _holdout._stream_green_sms(SimpleNamespace(cuda_stream=999)) is None

    class _NotGreen:
        CUdevResourceType = SimpleNamespace(CU_DEV_RESOURCE_TYPE_SM="sm")

        @staticmethod
        def cuStreamGetGreenCtx(handle):
            raise RuntimeError("not a green stream")

    monkeypatch.setattr(_holdout, "_green_driver", lambda: _NotGreen)
    monkeypatch.setattr("torch.cuda._utils._check_cuda_bindings", lambda result: result[1])
    assert _holdout._stream_green_sms(SimpleNamespace(cuda_stream=999)) is None


def test_green_sm_count_skips_without_cuda(monkeypatch):
    """HIP/CPU launches never query the current stream for a partition."""
    monkeypatch.setattr(torch.version, "hip", "7.0")

    def _boom(*args, **kwargs):
        raise AssertionError("stream queried on a non-CUDA launch")

    monkeypatch.setattr(torch.cuda, "current_stream", _boom)
    assert green_sm_count() is None


def test_cap_grid_skips_stream_query_by_default(monkeypatch):
    """Without check_stream, no stream is queried on any launch."""
    from triton.tlx.ops import _holdout

    def _boom(device=None):
        raise AssertionError("stream queried without check_stream")

    monkeypatch.setattr(torch._C, "_get_sm_carveout_experimental", lambda: 16)
    monkeypatch.setattr(_holdout, "green_sm_count", _boom)
    assert cap_grid(256) == 240
    assert cap_grid(256, device=0) == 240


def test_cap_grid_queries_launch_device_green_partition(monkeypatch):
    """The green cap comes from the launch device's stream, not the current one."""
    from triton.tlx.ops import _holdout

    seen = []
    monkeypatch.setattr(torch._C, "_get_sm_carveout_experimental", lambda: 0)
    monkeypatch.setattr(_holdout, "green_sm_count", lambda device=None: seen.append(device) or 64)
    assert cap_grid(256, device=3, check_stream=True) == 64
    assert seen == [3]


def test_num_sms_queries_launch_device_green_partition(monkeypatch):
    from types import SimpleNamespace

    from triton.tlx.ops import _holdout

    seen = []
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *args, **kwargs: SimpleNamespace(multi_processor_count=148),
    )
    monkeypatch.setattr(torch._C, "_get_sm_carveout_experimental", lambda: 0)
    monkeypatch.setattr(_holdout, "green_sm_count", lambda device=None: seen.append(device) or None)
    _device_sm_count_cached.cache_clear()
    try:
        assert num_sms(2, check_stream=True) == 148
        assert seen == [2]
    finally:
        _device_sm_count_cached.cache_clear()


def test_stream_green_sms_caches_partition_per_context(monkeypatch):
    """A green context's partition is queried once, not per launch."""
    from types import SimpleNamespace

    from triton.tlx.ops import _holdout

    calls = {"resource": 0}

    class _Drv:
        CUdevResourceType = SimpleNamespace(CU_DEV_RESOURCE_TYPE_SM="sm")

        @staticmethod
        def cuStreamGetGreenCtx(handle):
            return (0, "green-ctx-cache-test")

        @staticmethod
        def cuGreenCtxGetDevResource(green, resource_type):
            calls["resource"] += 1
            return (0, SimpleNamespace(sm=SimpleNamespace(smCount=48)))

    monkeypatch.setattr(_holdout, "_green_driver", lambda: _Drv)
    monkeypatch.setattr("torch.cuda._utils._check_cuda_bindings", lambda result: result[1])
    _holdout._GREEN_SM_COUNTS.clear()
    try:
        assert _holdout._stream_green_sms(SimpleNamespace(cuda_stream=7)) == 48
        assert _holdout._stream_green_sms(SimpleNamespace(cuda_stream=7)) == 48
        assert calls["resource"] == 1
    finally:
        _holdout._GREEN_SM_COUNTS.clear()


def test_cap_grid_caps_at_green_partition(monkeypatch):
    """A green partition below the carved physical count caps the grid."""
    from triton.tlx.ops import _holdout

    monkeypatch.setattr(torch._C, "_get_sm_carveout_experimental", lambda: 16)
    monkeypatch.setattr(_holdout, "green_sm_count", lambda device=None: 64)
    assert cap_grid(256, check_stream=True) == 64
    monkeypatch.setattr(_holdout, "green_sm_count", lambda device=None: 512)
    assert cap_grid(256, check_stream=True) == 240


def _cluster_fake_tensor():
    from types import SimpleNamespace

    return SimpleNamespace(
        ndim=4,
        shape=(1, 1, 256, 128),
        dtype=torch.bfloat16,
        device=torch.device("cuda", 0),
        is_cuda=True,
        stride=lambda dim: 128 if dim == 2 else 1,
    )


def _cluster_launch(monkeypatch, capped_sms, **kw):
    captured = {}

    class _FakeKernel:

        def __getitem__(self, grid):
            captured["grid"] = grid
            return lambda *args, **kwargs: captured.update(kwargs)

    monkeypatch.setattr(flash_attn_gfx950, "_num_sms", lambda device: capped_sms)
    monkeypatch.setattr(flash_attn_gfx950, "_attn_fwd_cluster_persistent_pipeline", _FakeKernel())
    monkeypatch.setattr(torch, "empty_like", lambda tensor: tensor)
    q = _cluster_fake_tensor()
    out = flash_attn_gfx950.flash_attn_cluster_persistent_pipeline(q, q, q, 0.5, causal=False, USE_DIRECT_LOAD=False,
                                                                   num_warps=8, **kw)
    assert out is q
    return captured


def test_kda_sequence_work_order_uses_launch_sms(monkeypatch):
    """KDA sorts work against the capped launch count, not the physical count."""
    offsets = torch.tensor([0, 128, 256], dtype=torch.int32)
    monkeypatch.setattr(kda_sm100, "_num_sms", lambda device: 256)
    assert kda_sm100._sequence_work_order(offsets, 100, torch.device("cpu")) is None
    monkeypatch.setattr(kda_sm100, "_num_sms", lambda device: 100)
    order = kda_sm100._sequence_work_order(offsets, 100, torch.device("cpu"))
    assert order.tolist() == [0, 1]


def test_cluster_persistent_grid_rounds_down_to_xcd_multiple(monkeypatch):
    """The capped count is rounded down to an XCD multiple, never up."""
    captured = _cluster_launch(monkeypatch, 255)
    assert captured["grid"] == (248, )
    assert "NUM_SMS" not in captured


def test_cluster_persistent_rejects_num_sms_kwarg(monkeypatch):
    with pytest.raises(ValueError, match="NUM_SMS is not supported"):
        _cluster_launch(monkeypatch, 256, NUM_SMS=128)


def test_cluster_persistent_rejects_cap_below_one_program_per_xcd(monkeypatch):
    """A cap below NUM_XCDS raises instead of rounding up past the cap."""
    with pytest.raises(ValueError, match="cannot provide one program per XCD"):
        _cluster_launch(monkeypatch, 4)


def test_carveout_stream_uses_masked_stream(monkeypatch):
    if torch.version.hip is None or not torch.cuda.is_available():
        pytest.skip("HIP masked streams only")
    monkeypatch.setattr(torch._C, "_get_sm_carveout_experimental", lambda: 32)
    ambient = torch.cuda.current_stream()
    with carveout_stream():
        assert torch.cuda.current_stream().cuda_stream != ambient.cuda_stream
