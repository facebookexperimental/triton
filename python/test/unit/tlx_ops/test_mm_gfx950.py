"""gfx950 correctness coverage for ``tlx.ops.mm``."""

import time

import pytest
import torch
import triton
from triton._internal_testing import is_hip_cdna4
from triton.tlx import ops as tlx_ops
from triton.tlx.ops import BitwiseInvarianceSpec, InvalidInput, UnsupportedOp
from triton.tlx.ops.kernels.mm._shapes import CORRECTNESS_SHAPES, operand
from triton.tlx.ops.kernels.mm import gfx950 as _gfx950

from mm_test_utils import run_mm_invariance_case

pytestmark = pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950")

MAX_SECONDS_PER_CASE = 60

# TODO: Re-enable shapes here when their direct-TLX correctness failures are fixed.
FAILED_SHAPES = {
    (384, 1152, 2536160, (1, 384), (1152, 1), "bf16"),
    (384, 1152, 2617290, (1, 384), (1152, 1), "bf16"),
    (384, 1152, 2701258, (1, 384), (1152, 1), "bf16"),
    (2536160, 384, 1152, (1152, 1), (1, 1152), "bf16"),
    (2617290, 384, 1152, (1152, 1), (1, 1152), "bf16"),
    (2701258, 384, 1152, (1152, 1), (1, 1152), "bf16"),
}


def _assert_range_result(actual, expected):
    assert actual.shape == expected.shape
    tolerance = 1e-2 if actual.dtype == torch.bfloat16 else 1e-3
    amax = max(expected[row:row + 64].abs().max().item() for row in range(0, expected.shape[0], 64))
    for row in range(0, expected.shape[0], 64):
        torch.testing.assert_close(actual[row:row + 64], expected[row:row + 64], atol=tolerance * amax, rtol=tolerance)


@pytest.mark.parametrize("m,n,k,dtype_name,b_step", [
    (24, 1024, 768, "bf16", 1),
    (96, 1280, 1024, "bf16", 1),
    (3072, 1280, 960, "bf16", 1),
    (1536, 1184, 997, "bf16", 1),
    (3072, 896, 1024, "bf16", 2),
    (16384, 1024, 1280, "bf16", 1),
    (6144, 1792, 1280, "bf16", 1),
    (4608, 1536, 1792, "bf16", 1),
    (6144, 128, 1024, "bf16", 1),
    (6144, 192, 1024, "bf16", 1),
    (1280, 1536, 1024, "fp16", 1),
    (992, 1008, 1024, "fp16", 1),
    (992, 1008, 1024, "bf16", 1),
    (256, 256, 256, "bf16", 1),
    (12288, 1152, 1008, "bf16", 1),
    (96, 16384, 1024, "bf16", 1),
    (8192, 128, 512, "bf16", 1),
    (1536, 640, 768, "bf16", 2),
])
def test_mm_range_family_numerics(m, n, k, dtype_name, b_step):
    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16
    torch.manual_seed(23)
    a = torch.randn((m, k), device="cuda", dtype=dtype)
    b = torch.randn((k, n * b_step), device="cuda", dtype=dtype)[:, ::b_step]
    candidates = _gfx950._range_dispatch_candidates(m, n, k, dtype, a.element_size(), a.stride(), b.stride())
    assert candidates
    out = torch.empty((m, n * 2), device="cuda", dtype=dtype)[:, ::2]
    expected = a @ b
    for dispatch in candidates:
        if dispatch[1] is not None:
            assert dispatch[1].get("split_k", 1) == 1
            assert dispatch[1].get("streamk_tail_split", 0) == 0
        assert _gfx950._launch_dispatch(a, b, out, dispatch) is out
        _assert_range_result(out, expected)
    assert tlx_ops.mm(a, b, out=out, space="heuristic") is out
    _assert_range_result(out, expected)


@pytest.mark.parametrize("space", ["heuristic", "full"])
@pytest.mark.parametrize("offset,row_padding", [(0, 0), (1, 0), (0, 1)])
def test_mm_paired_strided_b_is_batch_invariant(space, offset, row_padding):
    m, n, k = 1536, 640, 768
    torch.manual_seed(53)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    row_stride = n * 2 + row_padding
    size = offset + (k - 1) * row_stride + (n - 1) * 2 + 1
    storage = torch.randn(size, device="cuda", dtype=a.dtype)
    b = torch.as_strided(storage, (k, n), (row_stride, 2), storage_offset=offset)
    out = torch.full((m, n * 2), float("nan"), device="cuda", dtype=a.dtype)[:, ::2]
    assert tlx_ops.mm(a, b, out=out, space=space) is out
    _assert_range_result(out, a @ b)
    dispatch = _gfx950._range_dispatch_for(a, b)
    assert dispatch[1].get("PACK_B_PAIRS", False) == (offset == 0 and row_padding == 0)
    for prefix in (1, 127, 1023, 1024):
        actual = tlx_ops.mm(a[:prefix], b, space=space)
        assert torch.equal(actual.view(torch.int16), out[:prefix].view(torch.int16)), (space, prefix)


@pytest.mark.parametrize("space", ["heuristic", "full"])
@pytest.mark.parametrize("offset,padding", [(0, 0), (0, 1), (1, 1), (2, 1)])
def test_mm_padded_strided_b_storage_and_prefix(space, offset, padding):
    m, n, k = 1536, 640, 768
    torch.manual_seed(71)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    size = offset + (k - 1) * n * 2 + (n - 1) * 2 + 1
    storage = torch.randn(size + padding, device="cuda", dtype=a.dtype)
    b = torch.as_strided(storage, (k, n), (n * 2, 2), storage_offset=offset)
    dispatch = _gfx950._range_dispatch_for(a, b)
    assert dispatch[1].get("B_PAIRS_PADDED", False) == (padding == 1 and offset % 2 == 0)
    backing = torch.full((m, n * 2), float("nan"), device="cuda", dtype=a.dtype)
    out = backing[:, ::2]
    assert tlx_ops.mm(a, b, out=out, space=space) is out
    _assert_range_result(out, a @ b)
    assert backing[:, 1::2].isnan().all()
    for prefix in (1, 127, 1023, 1024, 1535):
        part = tlx_ops.mm(a[:prefix], b, space=space)
        assert torch.equal(part.view(torch.int16), out[:prefix].view(torch.int16)), (space, prefix)


def test_mm_padded_strided_b_cache_rebinds_storage():
    m, n, k = 1536, 640, 768
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((m, n), device="cuda", dtype=a.dtype)
    config = dict(_gfx950._fixed_register_plan(128, 64, 128, 4, 8, 4, 2, matrix_instr_nonkdim=32), PACK_B_PAIRS=True,
                  B_PAIRS_PADDED=True)
    cache = _gfx950._range_register_compiled_cache()
    cache.clear()
    keys = []
    try:
        for padding in (1, 0, 1):
            size = (k - 1) * n * 2 + (n - 1) * 2 + 1 + padding
            storage = torch.randn(size, device="cuda", dtype=a.dtype)
            b = torch.as_strided(storage, (k, n), (n * 2, 2))
            keys.append(_gfx950._range_tuned_plan_cache_key(a, b, out))
            assert _gfx950._range_register_launch_options(a, b, config)["B_PAIRS_PADDED"] == bool(padding)
            assert _gfx950._launch_range_register_plan(a, b, config=config, out=out) is out
            _assert_range_result(out, a @ b)
        assert keys[0] == keys[2] and keys[0] != keys[1]
        assert len(cache) == 2
    finally:
        cache.clear()


def test_mm_padded_strided_b_partial_n_falls_back():
    m, n, k = 1536, 641, 768
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((k, n * 2), device="cuda", dtype=a.dtype)[:, ::2]
    candidates = _gfx950._range_dispatch_candidates(m, n, k, a.dtype, a.element_size(), a.stride(), b.stride())
    assert candidates
    assert not any(plan.get("B_PAIRS_PADDED", False) for _, plan in candidates)
    for space in ("heuristic", "full"):
        _assert_range_result(tlx_ops.mm(a, b, space=space), a @ b)


@pytest.mark.parametrize("m,n,k,b_padding", [
    (17, 65536, 768, 0),
    (31, 65536, 1152, 0),
    (64, 32768, 1152, 0),
    (96, 65536, 1152, 8),
    (40, 147585, 2304, 0),
    (40, 262144, 1664, 0),
    (40, 131072, 2304, 8),
    (2848, 1152, 2048, 0),
    (9216, 1920, 1792, 0),
    (65536, 1536, 2048, 0),
    (1920, 131072, 1536, 0),
])
def test_mm_column_major_range_numerics(m, n, k, b_padding):
    torch.manual_seed(37)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((n, k + b_padding), device="cuda", dtype=a.dtype)
    b = weight[:, :k].T
    assert b.stride() == (1, k + b_padding)
    candidates = _gfx950._range_dispatch_candidates(m, n, k, a.dtype, a.element_size(), a.stride(), b.stride())
    assert candidates
    out = torch.empty((m, n * 2), device="cuda", dtype=a.dtype)[:, ::2]
    expected = a @ b
    for dispatch in candidates:
        path, plan = dispatch
        if path == "lds":
            assert plan[2] == 1
        else:
            assert plan.get("split_k", 1) == plan.get("SPLIT_K", 1) == 1
            assert plan.get("streamk_tail_split", 0) == 0
        out.fill_(float("nan"))
        assert _gfx950._launch_dispatch(a, b, out, dispatch) is out
        _assert_range_result(out, expected)
    for space in ("heuristic", "full"):
        out.fill_(float("nan"))
        assert tlx_ops.mm(a, b, out=out, space=space) is out
        _assert_range_result(out, expected)


@pytest.mark.parametrize("unaligned_operand", ["a", "b"])
def test_mm_column_major_unaligned_input_falls_back(unaligned_operand):
    m, n, k = 2304, 1408, 1152
    for space in ("heuristic", "full"):
        a_offset = int(unaligned_operand == "a")
        b_offset = int(unaligned_operand == "b")
        a = torch.randn(m * k + a_offset, device="cuda", dtype=torch.bfloat16)[a_offset:].view(m, k)
        b = torch.randn(n * k + b_offset, device="cuda", dtype=a.dtype)[b_offset:].view(n, k).T
        out = torch.full((m, n * 2), float("nan"), device="cuda", dtype=a.dtype)[:, ::2]
        assert _gfx950._range_dispatch_for(a, b)[0] == "range_register"
        assert tlx_ops.mm(a, b, out=out, space=space) is out
        _assert_range_result(out, a @ b)


def test_mm_column_major_cg_compiled_cache_rebinds_inputs():
    m, n, k = 17, 65536, 1152
    cache = _gfx950._range_register_compiled_cache()
    cache.clear()
    inputs = [(torch.randn(
        (m, k), device="cuda", dtype=torch.bfloat16), torch.randn(
            (n, k), device="cuda", dtype=torch.bfloat16).T, torch.empty((m, n), device="cuda", dtype=torch.bfloat16))
              for _ in range(2)]
    try:
        for a, b, out in inputs:
            dispatch = _gfx950._range_dispatch_for(a, b)
            assert dispatch[0] == "range_register" and dispatch[1]["CACHE_B_CG"]
            out.fill_(float("nan"))
            assert tlx_ops.mm(a, b, out=out) is out
            _assert_range_result(out, a @ b)
            assert len(cache) == 1
        out = torch.full((m, n * 2), float("nan"), device="cuda", dtype=a.dtype)[:, ::2]
        assert tlx_ops.mm(a, b, out=out) is out
        _assert_range_result(out, a @ b)
        assert len(cache) == 2
    finally:
        cache.clear()


@pytest.mark.parametrize("n,k", [(147585, 2304), (262144, 1664)])
@pytest.mark.parametrize("space", ["heuristic", "full"])
def test_mm_streamed_column_major_is_batch_invariant(n, k, space):
    torch.manual_seed(53)
    a = torch.randn((96, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=a.dtype).T
    full = torch.full((96, n), float("nan"), device="cuda", dtype=a.dtype)
    assert tlx_ops.mm(a, b, out=full, space=space) is full
    _assert_range_result(full, a @ b)
    for prefix in (1, 17, 32, 33, 40, 64, 65):
        out = torch.full((prefix, n), float("nan"), device="cuda", dtype=a.dtype)
        assert tlx_ops.mm(a[:prefix], b, out=out, space=space) is out
        assert torch.equal(out.view(torch.int16), full[:prefix].view(torch.int16)), (space, prefix)


@pytest.mark.parametrize("operand", ["a", "b"])
@pytest.mark.parametrize("offset", [1, 8])
def test_mm_streamed_column_major_offset_inputs(operand, offset):
    m, n, k = 40, 131072, 2304
    a_offset = offset if operand == "a" else 0
    b_offset = offset if operand == "b" else 0
    a = torch.randn(m * k + a_offset, device="cuda", dtype=torch.bfloat16)[a_offset:].view(m, k)
    b = torch.randn(n * k + b_offset, device="cuda", dtype=a.dtype)[b_offset:].view(n, k).T
    expected = a @ b
    out = torch.full((m, n), float("nan"), device="cuda", dtype=a.dtype)
    for space in ("heuristic", "full"):
        out.fill_(float("nan"))
        assert tlx_ops.mm(a, b, out=out, space=space) is out
        _assert_range_result(out, expected)
        for prefix in (17, 33):
            part = torch.full((prefix, n), float("nan"), device=a.device, dtype=a.dtype)
            assert tlx_ops.mm(a[:prefix], b, out=part, space=space) is part
            assert torch.equal(part.view(torch.int16), out[:prefix].view(torch.int16)), (space, prefix)


@pytest.mark.parametrize("m,n,k,b_padding", [(9216, 1920, 1792, 8), (10240, 1664, 1536, 0)])
@pytest.mark.parametrize("space", ["heuristic", "full"])
def test_mm_column_major_range_is_batch_invariant(m, n, k, b_padding, space):
    torch.manual_seed(47)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((n, k + b_padding), device="cuda", dtype=a.dtype)
    b = weight[:, :k].T
    assert b.stride() == (1, k + b_padding)
    full = torch.full((m, n), float("nan"), device="cuda", dtype=a.dtype)
    assert tlx_ops.mm(a, b, out=full, space=space) is full
    _assert_range_result(full, a @ b)
    for prefix in (1025, 4097, 6145):
        out = torch.full((prefix, n), float("nan"), device="cuda", dtype=a.dtype)
        assert tlx_ops.mm(a[:prefix], b, out=out, space=space) is out
        assert torch.equal(out, full[:prefix]), (space, prefix)


@pytest.mark.parametrize("a_strides,b_strides,k", [
    ((1, 768), (1280, 1), 1024),
    ((1024, 1), (1280, 1), 8192),
    ((2**30, 1), (1280, 1), 1024),
])
def test_mm_range_family_rejects_unsupported_geometry(a_strides, b_strides, k):
    assert not _gfx950._range_dispatch_candidates(768, 1280, k, torch.bfloat16, 2, a_strides, b_strides)


@pytest.mark.parametrize("m,n,k,block_m", [(96, 1280, 1024, 32), (8192, 640, 1024, 256)])
def test_mm_range_compiled_cache_rebinds_operands_and_output(m, n, k, block_m):
    cache = _gfx950._range_register_compiled_cache()
    cache.clear()
    try:
        for _ in range(2):
            a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
            b = torch.randn((k, n), device="cuda", dtype=torch.bfloat16)
            candidates = _gfx950._range_dispatch_candidates(m, n, k, a.dtype, a.element_size(), a.stride(), b.stride())
            dispatch = next(item for item in candidates
                            if item[0] == "range_register" and item[1]["BLOCK_M"] == block_m)
            out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
            assert _gfx950._launch_dispatch(a, b, out, dispatch) is out
            _assert_range_result(out, a @ b)
            assert len(cache) == 1
        out = torch.empty((m, n * 2), device="cuda", dtype=torch.bfloat16)[:, ::2]
        assert _gfx950._launch_dispatch(a, b, out, dispatch) is out
        _assert_range_result(out, a @ b)
        assert len(cache) == 2
    finally:
        cache.clear()


def test_mm_range_full_space_reuses_selection(monkeypatch):
    from triton import knobs, testing

    _gfx950._RANGE_TUNED_PLAN_CACHE.clear()
    measured = 0
    original = testing.do_bench_cudagraph

    def observed(*args, **kwargs):
        nonlocal measured
        measured += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(testing, "do_bench_cudagraph", observed)
    try:
        for _ in range(2):
            a = torch.randn((96, 1024), device="cuda", dtype=torch.bfloat16)
            b = torch.randn((1024, 1280), device="cuda", dtype=torch.bfloat16)
            out = torch.empty((96, 1280), device="cuda", dtype=torch.bfloat16)
            assert tlx_ops.mm(a, b, out=out, space="full") is out
            _assert_range_result(out, a @ b)
            assert len(_gfx950._RANGE_TUNED_PLAN_CACHE) == 1
            assert measured == 9
        launches = 0

        def on_launch(metadata):
            nonlocal launches
            launches += 1

        knobs.runtime.launch_enter_hook.add(on_launch)
        try:
            assert tlx_ops.mm(a, b, out=out, space="full") is out
            _assert_range_result(out, a @ b)
            assert launches > 0
        finally:
            knobs.runtime.launch_enter_hook.remove(on_launch)
    finally:
        _gfx950._RANGE_TUNED_PLAN_CACHE.clear()


@pytest.mark.parametrize("b_contiguous_n", [True, False])
def test_mm_wave_grid_compiled_cache_rebinds_operands_and_output(b_contiguous_n):
    cache = _gfx950._WAVE_GRID_COMPILED_CACHE
    cache.clear()
    try:
        m, n, k = 8192, 128, 1024
        for _ in range(2):
            a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
            b = (torch.randn((k, n), device="cuda", dtype=a.dtype) if b_contiguous_n else torch.randn(
                (n, k), device="cuda", dtype=a.dtype).T)
            dispatch = next(
                item
                for item in _gfx950._range_dispatch_candidates(m, n, k, a.dtype, a.element_size(), a.stride(), (n, 1))
                if item[0] == "wave_grid" and item[1]["kind"] == "regular_mi16_wave_grid")
            out = torch.empty((m, n), device="cuda", dtype=a.dtype)
            assert _gfx950._launch_dispatch(a, b, out, dispatch) is out
            _assert_range_result(out, a @ b)
            assert len(cache) == 1
        out = torch.empty((m, n * 2), device="cuda", dtype=a.dtype)[:, ::2]
        assert _gfx950._launch_dispatch(a, b, out, dispatch) is out
        _assert_range_result(out, a @ b)
        assert len(cache) == 2
        out = torch.empty(m * n + 1, device="cuda", dtype=a.dtype)[1:].view(m, n)
        assert _gfx950._launch_dispatch(a, b, out, dispatch) is out
        _assert_range_result(out, a @ b)
        assert len(cache) == 3
    finally:
        cache.clear()


@pytest.mark.parametrize("instrumentation", ["pre_run_hook", "always_compile"])
def test_mm_wave_grid_compiled_cache_preserves_instrumentation(monkeypatch, instrumentation):
    cache = _gfx950._WAVE_GRID_COMPILED_CACHE
    cache.clear()
    try:
        a = torch.randn((8192, 1024), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((1024, 128), device="cuda", dtype=a.dtype)
        out = torch.empty((8192, 128), device="cuda", dtype=a.dtype)
        tlx_ops.mm(a, b, out=out)
        assert len(cache) == 1
        kernel = _gfx950._wg_kernel_regular_mi16_wave_grid
        original_run = kernel.run
        calls = []

        def observed(*args, **kwargs):
            calls.append(True)
            return original_run(*args, **kwargs)

        monkeypatch.setattr(kernel, "run", observed)
        if instrumentation == "pre_run_hook":
            hook_calls = []
            monkeypatch.setattr(kernel, "pre_run_hooks", [lambda *args, **kwargs: hook_calls.append(True)])
        else:
            monkeypatch.setattr(triton.knobs.compilation, "always_compile", True)
        assert tlx_ops.mm(a, b, out=out) is out
        assert len(calls) == 1
        if instrumentation == "pre_run_hook":
            assert len(hook_calls) == 1
        _assert_range_result(out, a @ b)
    finally:
        cache.clear()


def test_mm_wave_grid_compiled_cache_rejects_changed_globals(monkeypatch):
    cache = _gfx950._WAVE_GRID_COMPILED_CACHE
    cache.clear()
    try:
        a = torch.randn((8192, 1024), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((1024, 128), device="cuda", dtype=a.dtype)
        out = torch.empty((8192, 128), device="cuda", dtype=a.dtype)
        tlx_ops.mm(a, b, out=out)
        assert len(cache) == 1
        kernel = _gfx950._wg_kernel_regular_mi16_wave_grid
        name, namespace = next((name, namespace)
                               for (name, _), (_, namespace) in kernel.used_global_vals.items()
                               if name.startswith("_wg_B_BASES"))
        monkeypatch.setitem(namespace, name, None)
        with pytest.raises(RuntimeError, match="Global variable .* has changed"):
            tlx_ops.mm(a, b, out=out)
    finally:
        cache.clear()


@pytest.mark.parametrize("space", ["heuristic", "full"])
@pytest.mark.parametrize("n", [240, 256])
@pytest.mark.parametrize("m,k", [(25601, 1280), (30721, 1536)])
def test_mm_narrow_direct_range_is_batch_invariant(m, k, n, space):
    torch.manual_seed(71)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((k, n), device="cuda", dtype=a.dtype)
    path, plan = _gfx950._range_dispatch_for(a, b)
    assert path == "wave_grid" and plan["direct_to_lds"]
    assert plan["split_k"] == 1 and plan["streamk_tail_split"] == 0
    full = torch.full((m, n), float("nan"), device=a.device, dtype=a.dtype)
    assert tlx_ops.mm(a, b, out=full, space=space) is full
    _assert_range_result(full, a @ b)
    for prefix_m in (1, 159, 160, 161, 20320, 20321, 30560, 30561, m - 1):
        if prefix_m >= m:
            continue
        prefix = torch.full((prefix_m, n), float("nan"), device=a.device, dtype=a.dtype)
        assert tlx_ops.mm(a[:prefix_m], b, out=prefix, space=space) is prefix
        assert torch.equal(prefix.view(torch.int16), full[:prefix_m].view(torch.int16)), prefix_m


def test_mm_narrow_direct_range_boundaries():
    for m, n, k, a_padding, b_padding, dtype, direct in (
        (20320, 256, 1280, 0, 0, torch.bfloat16, False),
        (20321, 256, 1280, 0, 0, torch.bfloat16, True),
        (40960, 256, 1280, 0, 0, torch.bfloat16, True),
        (40961, 256, 1280, 0, 0, torch.bfloat16, False),
        (25600, 232, 1280, 0, 0, torch.bfloat16, False),
        (25600, 240, 1280, 0, 0, torch.bfloat16, True),
        (25600, 264, 1280, 0, 0, torch.bfloat16, False),
        (25600, 256, 896, 0, 0, torch.bfloat16, False),
        (25600, 256, 1024, 0, 0, torch.bfloat16, True),
        (25600, 256, 1088, 0, 0, torch.bfloat16, False),
        (25600, 256, 4096, 0, 0, torch.bfloat16, True),
        (25600, 256, 4224, 0, 0, torch.bfloat16, False),
        (25600, 256, 1280, 8, 0, torch.bfloat16, False),
        (25600, 256, 1280, 0, 8, torch.bfloat16, False),
        (25600, 256, 1280, 0, 0, torch.float16, False),
    ):
        candidates = _gfx950._range_family_candidates(m, n, k, dtype, 2, (k + a_padding, 1), (n + b_padding, 1))
        first = candidates[0] if candidates else (None, None)
        is_narrow_direct = (first[0] == "wave_grid" and first[1]["direct_to_lds"] and first[1]["mi_wave_tile_m"] == 5)
        assert is_narrow_direct == direct, (m, n, k, a_padding, b_padding, dtype)


@pytest.mark.parametrize("warps_n,direct_chunk", [(2, 64), (4, 64), (4, 128)])
def test_mm_direct_a_tail_preserves_four_and_eight_wave_loads(warps_n, direct_chunk):
    m, n, k = 319, 248, 256
    torch.manual_seed(73)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((k, n), device="cuda", dtype=a.dtype)
    plan = _gfx950._wg_regular_wave_grid_plan(5, 16 // warps_n, 2, warps_n, 64, local_stages=2, pgr2_operands=True,
                                              direct_to_lds=True, direct_chunk=direct_chunk, row_major_b_lds=True,
                                              row_wise_epilogue=True, waves_per_eu=1)
    out = torch.full((m, n), float("nan"), device=a.device, dtype=a.dtype)
    assert _gfx950._launch_dispatch(a, b, out, ("wave_grid", plan)) is out
    _assert_range_result(out, a @ b)


def _cases():
    return [entry for entry in CORRECTNESS_SHAPES if tuple(entry) not in FAILED_SHAPES]


def _assert_strides(tensor, wanted):
    for dim, (got, expected) in enumerate(zip(tensor.stride(), wanted)):
        if tensor.shape[dim] != 1:
            assert got == expected, (f"dim {dim}: stride {got}, recorded {expected}")


@pytest.mark.parametrize("m,n,k,a_strides,b_strides,dtype_name", _cases())
def test_mm(m, n, k, a_strides, b_strides, dtype_name):
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[dtype_name]
    from triton.tlx.ops import mm as tlx_mm

    a = operand(m, k, a_strides, dtype)
    b = operand(k, n, b_strides, dtype)
    _assert_strides(a, a_strides)
    _assert_strides(b, b_strides)

    torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        out = tlx_mm(a, b, space="heuristic")
    except (InvalidInput, UnsupportedOp) as declined:
        pytest.skip(f"gfx950 does not support this shape: {declined}")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    assert elapsed < MAX_SECONDS_PER_CASE, (f"mm({m}x{n}x{k}, {dtype}) took {elapsed:.1f}s, "
                                            f"over the {MAX_SECONDS_PER_CASE}s budget")

    expected = torch.matmul(a, b)
    _assert_range_result(out, expected)


@pytest.mark.parametrize(
    "m,n,k,dtype",
    [
        (279, 256, 4096, torch.float16),
        (1024, 4096, 800, torch.bfloat16),
    ],
    ids=["intermediate-fp16", "full-grid-bf16"],
)
def test_mm_register_fallback(m, n, k, dtype):
    from triton.tlx.ops import mm as tlx_mm

    a = torch.randn((m, k), device="cuda", dtype=dtype)
    b = torch.randn((n, k), device="cuda", dtype=dtype).T

    out = tlx_mm(a, b, space="heuristic")
    expected = torch.matmul(a, b)
    torch.testing.assert_close(
        out,
        expected,
        atol=1e-2 * expected.abs().max().item(),
        rtol=1e-2,
    )


@pytest.mark.parametrize(
    "k,split_k",
    [(256, 1), (320, 1), (1280, 4)],
    ids=["even-k64-blocks", "odd-k64-blocks", "odd-k64-blocks-split-k"],
)
def test_mm_wave_grid_pgr2_double_buffered_k64_tail(k, split_k):
    """Keep both parity tails in bounds and numerically correct."""
    m = n = 256
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)
    plan = _gfx950._wg_regular_wave_grid_plan(
        8,
        8,
        2,
        2,
        64,
        local_stages=2,
        pgr2_operands=True,
        row_major_b_lds=True,
        row_wise_epilogue=True,
        wide_epilogue=True,
        split_k=split_k,
    )

    out = _gfx950._wg_matmul(a, b, _candidate_plan=plan)
    expected = torch.matmul(a, b)
    torch.testing.assert_close(
        out,
        expected,
        atol=1e-3 * expected.abs().max().item(),
        rtol=1e-3,
    )


@pytest.mark.parametrize("dtype_name,b_transposed,m,n", [
    ("bf16", False, 256, 256),
    ("bf16", False, 257, 352),
    ("bf16", True, 257, 352),
    ("fp16", False, 257, 352),
    ("fp16", True, 257, 352),
])
def test_mm_wave_grid_row_major_direct_to_lds(dtype_name, b_transposed, m, n):
    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16
    k = 256
    torch.manual_seed(23)
    a = torch.randn((m, k), device="cuda", dtype=dtype)
    b = torch.randn((n, k), device="cuda", dtype=dtype).T if b_transposed else torch.randn(
        (k, n), device="cuda", dtype=dtype)
    plan = _gfx950._wg_regular_wave_grid_plan(
        8,
        5 if b_transposed else 8,
        2,
        2,
        64,
        local_stages=2,
        pgr2_operands=True,
        direct_to_lds=True,
        direct_chunk=32 if b_transposed else 128,
        pack_direct_chunks=b_transposed,
        row_major_b_lds=True,
        row_wise_epilogue=True,
        wide_epilogue=not b_transposed,
        num_xcds=1,
        workgroup_mapping=1,
        reverse_local_assignment=False,
        sink_insts_to_avoid_spills=True,
        disable_unclustered_high_rp_reschedule=True,
    )

    out = _gfx950._wg_matmul(a, b, _candidate_plan=plan)
    expected = torch.matmul(a, b)
    _assert_range_result(out, expected)


@pytest.mark.parametrize("k", [257, 294, 319])
def test_mm_wave_grid_pgr2_arbitrary_k_tail(k):
    """Append a masked partial K64 block without spilling accumulators."""
    m = n = 256
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)
    plan = _gfx950._wg_regular_wave_grid_plan(
        8,
        8,
        2,
        2,
        64,
        local_stages=2,
        pgr2_operands=True,
        row_major_b_lds=True,
        row_wise_epilogue=True,
        wide_epilogue=True,
    )

    out = _gfx950._wg_matmul(a, b, _candidate_plan=plan)
    expected = torch.matmul(a, b)
    torch.testing.assert_close(
        out,
        expected,
        atol=1e-3 * expected.abs().max().item(),
        rtol=1e-3,
    )


@pytest.mark.parametrize("k,split_k", [(448, 1), (1792, 4)])
def test_mm_wave_grid_pgr2_k128_with_k64_tail(k, split_k):
    """Keep the physical K64 banks and final K64 tail numerically correct."""
    m = n = 128
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((n, k), device="cuda", dtype=torch.float16).T
    plan = _gfx950._wg_regular_wave_grid_plan(
        4,
        4,
        2,
        2,
        128,
        local_stages=1,
        pgr2_operands=True,
        row_wise_epilogue=True,
        split_k=split_k,
        reduce_tile=(16, 64),
        reduce_warps=4,
    )

    out = _gfx950._wg_matmul(a, b, _candidate_plan=plan)
    expected = torch.matmul(a, b)
    torch.testing.assert_close(
        out,
        expected,
        atol=1e-3 * expected.abs().max().item(),
        rtol=1e-3,
    )


@pytest.mark.parametrize("m", [256, 257])
def test_mm_small_square_register_plan(m):
    from triton.tlx.ops import mm as tlx_mm

    n, k = 257, 4096
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((n, k), device="cuda", dtype=torch.float16).T
    assert _gfx950._dispatch_for(a, b)[0] == "register"
    out = tlx_mm(a, b, space="heuristic")
    expected = torch.matmul(a, b)
    torch.testing.assert_close(
        out,
        expected,
        atol=1e-3 * expected.abs().max().item(),
        rtol=1e-3,
    )


def test_mm_tuned_rectangular_register_plans():
    expected = {
        (677, 2048, 4096): (64, 128, 128, 2, 8, 8, 3, 32),
        (279, 4096, 4352): (128, 64, 128, 8, 8, 8, 3, 16),
        (4096, 4096, 2048): (256, 256, 64, 2, 1, 8, 2, 32),
        (61440, 3840, 4096): (128, 256, 32, 16, 8, 4, 2, 16),
        (61440, 5120, 2048): (128, 256, 32, 8, 8, 4, 2, 16),
        (61440, 5120, 7744): (128, 256, 32, 16, 8, 4, 2, 16),
        (4096, 242432, 1894): (256, 256, 64, 8, 8, 8, 2, 32),
    }
    for shape, wanted in expected.items():
        plan = _gfx950._register_plan_for_shape(*shape, torch.float16)
        assert (
            plan["BLOCK_M"],
            plan["BLOCK_N"],
            plan["BLOCK_K"],
            plan["GROUP_M"],
            plan["NUM_XCDS"],
            plan["num_warps"],
            plan["num_stages"],
            plan["matrix_instr_nonkdim"],
        ) == wanted


def test_mm_new_production_register_plans():
    expected = {
        (128, 2688, 4416): (32, 64, 256, 16, 8, 2, 3, 1),
        (128, 448, 384): (64, 32, 128, 1, 1, 4, 3, 2),
        (128, 10304, 1152): (64, 64, 128, 16, 1, 4, 2, 0),
        (128, 34272, 384): (128, 128, 64, 16, 1, 4, 2, 0),
        (128, 896, 576): (32, 32, 256, 16, 1, 4, 2, 0),
        (128, 70770, 192): (128, 64, 64, 4, 8, 4, 2, 0),
        (128, 1152, 512): (32, 32, 256, 16, 1, 4, 2, 2),
        (128, 1344, 4032): (16, 16, 256, 8, 8, 4, 2, 0),
    }
    for shape, wanted in expected.items():
        plan = _gfx950._register_plan_for_shape(*shape, torch.float16)
        assert (
            plan["BLOCK_M"],
            plan["BLOCK_N"],
            plan["BLOCK_K"],
            plan["GROUP_M"],
            plan["NUM_XCDS"],
            plan["num_warps"],
            plan["num_stages"],
            plan["waves_per_eu"],
        ) == wanted


@pytest.mark.parametrize(
    "shape,wanted",
    [
        ((128, 1536, 2688), (32, 32, 256, 4, 1, 4, 3, 2, 16)),
        ((128, 896, 2016), (32, 16, 256, 4, 1, 2, 3, 2, 16)),
        ((128, 2304, 2560), (32, 64, 256, 4, 1, 8, 3, 2, 16)),
        ((128, 3456, 2560), (64, 32, 256, 4, 1, 8, 3, 2, 16)),
        ((128, 56768, 64), (64, 128, 64, 1, 1, 4, 2, 0, 32)),
        ((128, 65462, 96), (128, 128, 32, 1, 8, 8, 3, 0, 32)),
        ((25600, 256, 256), (128, 128, 64, 16, 4, 4, 2, 0, 32)),
        ((128, 8064, 1728), (64, 64, 128, 2, 1, 4, 3, 2, 32)),
    ],
)
def test_mm_production_uses_transposed_register_plan(shape, wanted):
    path, plan = _gfx950._dispatch_plan(*shape, torch.float16, 2)
    assert path == "transposed_register"
    assert (
        plan["BLOCK_M"],
        plan["BLOCK_N"],
        plan["BLOCK_K"],
        plan["GROUP_M"],
        plan["NUM_XCDS"],
        plan["num_warps"],
        plan["num_stages"],
        plan["waves_per_eu"],
        plan["matrix_instr_nonkdim"],
    ) == wanted


def test_mm_production_deep_k_uses_split_wave_grid():
    path, plan = _gfx950._dispatch_plan(128, 224, 20926, torch.float16, 2)
    assert path == "wave_grid"
    assert plan["mi_wave_tile_m"] == 1
    assert plan["mi_wave_tile_n"] == 2
    assert plan["warps_m"] == 2
    assert plan["warps_n"] == 2
    assert plan["block_k"] == 128
    assert plan["split_k"] == 16
    assert plan["pair_balanced_split_k"]


def test_mm_production_large_split_k_uses_fp16_workspace():
    path, plan = _gfx950._dispatch_plan(128, 10304, 20926, torch.float16, 2)
    assert path == "wave_grid"
    assert plan["split_k"] == 3
    assert plan["pair_balanced_split_k"]
    assert plan["workspace_dtype"] == torch.float16


def test_mm_production_ranking_split_k_uses_fp16_workspace():
    path, plan = _gfx950._dispatch_plan(128, 9216, 7392, torch.float16, 2)
    assert path == "wave_grid"
    assert plan["split_k"] == 3
    assert not plan["pair_balanced_split_k"]
    assert plan["workspace_dtype"] == torch.float16
    assert plan["waves_per_eu"] == 1


@pytest.mark.parametrize("m", [196608, 245760])
def test_mm_production_fragmented_register_plan(m):
    path, plan = _gfx950._dispatch_plan(m, 80, 80, torch.float16, 2)
    assert path == "fragmented_register"
    assert dict(plan) == {
        "M_HEAD": 64,
        "M_TAIL": 16,
        "BLOCK_N": 256,
        "BLOCK_K": 64,
        "matrix_instr_nonkdim": 16,
        "num_warps": 4,
        "num_stages": 2,
    }


def test_mm_production_short_k_uses_exact_wave_grid():
    path, plan = _gfx950._dispatch_plan(128, 34272, 384, torch.float16, 2)
    assert path == "transposed_wave_grid"
    assert plan["pgr2_operands"]
    assert (
        plan["mi_wave_tile_m"],
        plan["mi_wave_tile_n"],
        plan["warps_m"],
        plan["warps_n"],
    ) == (5, 4, 2, 2)
    assert plan["num_xcds"] == 8
    assert plan["workgroup_mapping"] == 6


def test_mm_production_deep_reduction_uses_exact_wave_grid():
    path, plan = _gfx950._dispatch_plan(128, 10304, 1728, torch.float16, 2)
    assert path == "transposed_wave_grid"
    assert plan["pgr2_operands"]
    assert plan["num_xcds"] == 8
    assert plan["workgroup_mapping"] == 6
    assert (
        plan["mi_wave_tile_m"],
        plan["mi_wave_tile_n"],
        plan["warps_m"],
        plan["warps_n"],
    ) == (3, 2, 2, 2)


def test_mm_production_ranking_shape_uses_full_read_wave_grid():
    path, plan = _gfx950._dispatch_plan(128, 10752, 2560, torch.float16, 2)
    assert path == "transposed_wave_grid"
    assert plan["pgr2_operands"]
    assert plan["read_cover_sixteenths"] == 16
    assert plan["reverse_local_assignment"]
    assert plan["num_xcds"] == 8
    assert plan["workgroup_mapping"] == 6
    assert (
        plan["mi_wave_tile_m"],
        plan["mi_wave_tile_n"],
        plan["warps_m"],
        plan["warps_n"],
    ) == (3, 2, 2, 2)


@pytest.mark.parametrize(
    "m,n,k",
    [
        (128, 2688, 4416),
        (128, 10304, 1728),
        (128, 34272, 384),
        (128, 10752, 2560),
    ],
)
def test_mm_production_reuses_compiled_wave_grid_launcher(monkeypatch, m, n, k):
    from triton.tlx.ops import mm as tlx_mm

    _gfx950._production_wave_grid_launcher.cache_clear()
    compile_kernel = _gfx950._compile_regular_wave_grid
    compiled = []

    def record_compile(*args, **kwargs):
        launcher = compile_kernel(*args, **kwargs)
        compiled.append(launcher)
        return launcher

    monkeypatch.setattr(_gfx950, "_compile_regular_wave_grid", record_compile)
    operands = [(
        torch.randn((m, k), device="cuda", dtype=torch.float16),
        torch.randn((n, k), device="cuda", dtype=torch.float16).T,
        torch.empty((m, n), device="cuda", dtype=torch.float16),
    ) for _ in range(2)]
    for a, b, out in operands:
        assert tlx_mm(a, b, out=out, space="heuristic") is out
        expected = torch.matmul(a, b)
        torch.testing.assert_close(
            out,
            expected,
            atol=1e-3 * expected.abs().max().item(),
            rtol=1e-3,
        )
    assert len(compiled) == 1


@pytest.mark.parametrize("m,n,k", [(384, 384, 48), (319, 385, 63)])
def test_mm_short_k_register_is_batch_invariant(m, n, k):
    from triton.tlx.ops import mm as tlx_mm

    torch.manual_seed(17)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((k, n), device="cuda", dtype=torch.bfloat16)

    out = tlx_mm(a, b, space="heuristic")
    sliced = tlx_mm(a[:256], b, space="heuristic")
    expected = torch.matmul(a, b)
    assert torch.equal(out[:256], sliced)
    torch.testing.assert_close(
        out,
        expected,
        atol=1e-2 * expected.abs().max().item(),
        rtol=1e-2,
    )


def test_mm_caches_public_catalog_lookup(monkeypatch):
    a = torch.empty((16, 16), device="cuda", dtype=torch.bfloat16)
    b = torch.empty_like(a)
    out = torch.empty_like(a)
    _, spec = tlx_ops.impl_for("mm", device=a.device)
    calls = 0

    def counting_impl_for(op, *, device):
        nonlocal calls
        calls += 1
        assert op == "mm"
        assert device == a.device
        return (lambda _a, _b, *, out, space, invariance=None: out), spec

    monkeypatch.setattr(tlx_ops, "impl_for", counting_impl_for)
    tlx_ops._mm_impl_for_device.cache_clear()
    try:
        assert tlx_ops.mm(a, b, out=out) is out
        assert tlx_ops.mm(a, b, out=out) is out
        assert calls == 1
    finally:
        tlx_ops._mm_impl_for_device.cache_clear()


def test_mm_reuses_out_tensor():
    from triton.tlx.ops import mm as tlx_mm

    a = torch.randn((128, 128), device="cuda", dtype=torch.float16)
    b = torch.randn((128, 128), device="cuda", dtype=torch.float16)
    out = torch.empty((128, 128), device="cuda", dtype=torch.float16)

    actual = tlx_mm(a, b, out=out, space="heuristic")
    expected = torch.matmul(a, b)

    assert actual is out
    torch.testing.assert_close(
        actual,
        expected,
        atol=1e-3 * expected.abs().max().item(),
        rtol=1e-3,
    )


def test_mm_row_major_uses_tuned_register_plan():
    path, plan = _gfx950.heuristic_config(
        61440,
        5120,
        2048,
        torch.float16,
        2,
        (2048, 1),
        (5120, 1),
    )
    assert path == "register"
    assert (plan["BLOCK_M"], plan["BLOCK_N"], plan["BLOCK_K"]) == (
        128,
        256,
        32,
    )


def test_mm_register_fallback_for_row_major_b(monkeypatch):
    from triton.tlx.ops.kernels.mm.gfx950 import mm

    a = torch.randn((64, 128), device="cuda", dtype=torch.float16)
    b = torch.randn((128, 32), device="cuda", dtype=torch.float16)
    expected = torch.empty((64, 32), device="cuda", dtype=torch.float16)

    def launch_register_plan(actual_a, actual_b, *, config, out, _validated):
        assert actual_a is a
        assert actual_b is b
        assert config == _gfx950._intermediate_register_config(64, 32, 128)
        assert _validated
        return expected

    monkeypatch.setattr(_gfx950, "_launch_register_plan", launch_register_plan)
    assert mm(a, b, space="heuristic") is expected


def test_mm_register_fallback_for_strided_b(monkeypatch):
    from triton.tlx.ops.kernels.mm.gfx950 import mm

    a = torch.randn((64, 128), device="cuda", dtype=torch.float16)
    b = torch.randn((128, 64), device="cuda", dtype=torch.float16)[:, ::2]
    expected = torch.empty((64, 32), device="cuda", dtype=torch.float16)

    def launch_register_plan(actual_a, actual_b, *, config, out, _validated):
        assert actual_a is a
        assert actual_b is b
        assert config == _gfx950._intermediate_register_config(64, 32, 128)
        assert _validated
        return expected

    monkeypatch.setattr(_gfx950, "_launch_register_plan", launch_register_plan)
    assert mm(a, b, space="heuristic") is expected


@pytest.mark.parametrize(
    "m,n,k,expected_path",
    [
        (48, 131072, 256, "wave_grid"),
        (98304, 80, 512, "register"),
        (4320, 5168, 4096, "transposed_wave_grid"),
        (192, 147472, 8192, "wave_grid"),
    ],
    ids=[
        "fragmented-register",
        "rectangular-pgr2",
        "deep-k-direct",
        "bounded-streamk-tail",
    ],
)
def test_mm_promoted_dispatch(m, n, k, expected_path):
    """Exercise promoted shape families through public ``tlx.ops.mm``."""
    from triton.tlx.ops import mm as tlx_mm

    torch.manual_seed(m + n + k)
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((n, k), device="cuda", dtype=torch.float16).T
    assert _gfx950._dispatch_for(a, b)[0] == expected_path

    actual = tlx_mm(a, b, space="heuristic")
    expected = torch.matmul(a, b)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    "m,n,k,expected_path",
    [
        (240, 196608, 2048, "inter_wave"),
        (4096, 4352, 8192, "streamk"),
        (800, 76800, 512, "m192n256"),
    ],
)
def test_mm_promoted_inter_wave_dispatch(m, n, k, expected_path):
    from triton.tlx.ops import mm as tlx_mm

    torch.manual_seed(m + n + k)
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((n, k), device="cuda", dtype=torch.float16).T
    assert _gfx950._dispatch_for(a, b)[0] == expected_path

    actual = tlx_mm(a, b, space="heuristic")
    expected = torch.matmul(a, b)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_mm_promoted_bf16_dispatch():
    assert _gfx950._dispatch_plan(2048, 25408, 10240, torch.bfloat16, 2) == ("inter_wave", None)
    assert _gfx950._dispatch_plan(1024, 6144, 20480, torch.bfloat16, 2) == ("lds", (256, 256, 2))


@pytest.mark.parametrize(
    "m,n,k,expected_path",
    [
        (1024, 6144, 4096, "persistent"),
        (1024, 20480, 6144, "persistent"),
        (208, 106496, 6144, "hybrid_n160"),
    ],
)
def test_mm_promoted_persistent_dispatch(m, n, k, expected_path):
    """Keep large persistent families in dispatch without allocating inputs."""
    assert _gfx950._dispatch_plan(m, n, k, torch.float16, 2)[0] == expected_path


def test_mm_row_major_k4096_persistent_correctness():
    """Exercise the MT224x128 specialization with its partial M tile."""
    from triton.tlx.ops import mm as tlx_mm

    m, n, k = 1024, 6144, 4096
    torch.manual_seed(m + n + k)
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)
    assert _gfx950._dispatch_for(a, b) == (
        "persistent",
        "mt224x128_k4096",
    )

    actual = tlx_mm(a, b, space="heuristic")
    expected = torch.matmul(a, b)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("n", [2560])
def test_mm_deep_k_avoids_excess_m192_work(n):
    """Preserve the measured D120380534 LDS winners for deep-K M2032."""
    assert _gfx950._dispatch_plan(2032, n, 18688, torch.float16, 2)[0] == "lds"


def test_mm_m2032_thin_n_uses_k128_pgr2_split_k():
    """Use the deep register-prefetch window and its final K64 tail."""
    path, plan = _gfx950._dispatch_plan(2032, 512, 18688, torch.float16, 2)
    assert path == "wave_grid"
    assert not plan["pair_balanced_split_k"]
    assert plan["block_k"] == 128
    assert plan["local_stages"] == 1
    assert plan["pgr2_operands"]
    assert plan["split_k"] == 4
    split_k = 18688 // plan["split_k"]
    assert split_k // plan["block_k"] == 36
    assert split_k % plan["block_k"] == 64


def test_mm_m2032_k32_tail_uses_ordered_pgr2_wave_grid():
    """Keep the K64 main loop plus final ordered K32 on the PGR2 path."""
    path, plan = _gfx950._dispatch_plan(2032, 2048, 7072, torch.float16, 2)
    assert path == "wave_grid"
    assert plan["block_k"] == 64
    assert plan["local_stages"] == 2
    assert plan["pgr2_operands"]
    assert plan["pgr2_late_read_count"] == 2
    assert plan["wide_epilogue"]
    assert plan["waves_per_eu"] == 1
    assert 7072 % plan["block_k"] == 32
    assert plan["split_k"] == 1


def test_mm_shallow_k_uses_m2032_transposed_wave_grid():
    path, plan = _gfx950._dispatch_plan(2032, 18688, 512, torch.float16, 2)
    assert path == "transposed_wave_grid"
    assert plan["pgr2_operands"]
    assert plan["direct_to_lds"]
    assert plan["pgr2_late_read_count"] == 2
    assert plan["read_cover_sixteenths"] == 12
    assert plan["row_major_b_lds"]


def test_mm_large_2d_streamk_grid_uses_wide_locality_band():
    assert _gfx950._iw_streamk_mapping(40144, 51792, 8192, 256, 256) == (16, 2)
    # Do not apply the large-grid mapping to the tall/narrow or smaller
    # Stream-K families where it was measured to regress.
    assert _gfx950._iw_streamk_mapping(896, 117136, 8192, 256, 256) == (4, 8)
    assert _gfx950._iw_streamk_mapping(7088, 5552, 32768, 256, 256) == (4, 8)


@pytest.mark.parametrize(
    "m,n,k,expected_path",
    [
        (80, 17456, 1024, "wave_grid"),
        (80, 65552, 6144, "wave_grid"),
        (1072, 1584, 32768, "wave_grid"),
        (224, 106496, 6144, "wave_grid"),
        (224, 172032, 6144, "wave_grid"),
        (400, 73744, 4096, "wave_grid"),
        (279, 2048, 4096, "wave_grid"),
        (279, 4096, 4352, "wave_grid"),
        (279, 8192, 2048, "wave_grid"),
        (677, 2048, 4096, "wave_grid"),
        (272, 12272, 6144, "wave_grid"),
        (2032, 2048, 7072, "wave_grid"),
        (2032, 512, 18688, "wave_grid"),
        (677, 4096, 2048, "wave_grid"),
        (144, 147472, 768, "transposed_wave_grid"),
        (896, 65552, 4096, "transposed_wave_grid"),
        (1136, 90128, 4096, "transposed_wave_grid"),
        (1136, 117136, 1024, "transposed_wave_grid"),
        (2032, 18688, 512, "transposed_wave_grid"),
        (12272, 1328, 8192, "transposed_wave_grid"),
        (73744, 224, 4096, "wave_grid"),
        (528, 73744, 256, "wave_grid"),
        (528, 147472, 8192, "wave_grid"),
        (192, 147472, 8192, "wave_grid"),
        (106512, 896, 256, "inter_wave"),
    ],
)
def test_mm_sota77_promoted_dispatch(m, n, k, expected_path):
    """Keep measured SOTA77 winners on their validated algorithm family."""
    assert _gfx950._dispatch_plan(m, n, k, torch.float16, 2)[0] == expected_path


@pytest.mark.parametrize("n", [106496, 172032])
def test_mm_m224_streaming_b_uses_exact_wave_grid(n):
    """Keep the one-M-tile family on its measured MT224x224 PGR2 path."""
    plan = _gfx950._wave_grid_plan_for_shape(224, n, 6144, torch.float16)
    assert plan is not None
    assert plan["pgr2_operands"]
    assert not plan["direct_to_lds"]
    assert 16 * plan["mi_wave_tile_m"] * plan["warps_m"] == 224
    assert 16 * plan["mi_wave_tile_n"] * plan["warps_n"] == 224


def test_mm_m279_deep_k_wave_grid_saturates_all_compute_units():
    plan = _gfx950._wave_grid_plan_for_shape(279, 4096, 4352, torch.float16)
    assert plan["split_k"] == 2
    assert plan["num_xcds"] == 4
    assert plan["workgroup_mapping"] == 4
    assert plan["loop_unroll_factor"] == 2
    assert (
        plan["mi_wave_tile_m"],
        plan["mi_wave_tile_n"],
        plan["warps_m"],
        plan["warps_n"],
    ) == (5, 2, 1, 4)


def test_mm_m279_medium_n_uses_saturating_k128_split_k():
    plan = _gfx950._wave_grid_plan_for_shape(279, 2048, 4096, torch.float16)
    assert plan["block_k"] == 128
    assert plan["split_k"] == 4
    assert plan["num_xcds"] == 4
    assert plan["workgroup_mapping"] == 4
    assert (
        plan["mi_wave_tile_m"],
        plan["mi_wave_tile_n"],
        plan["warps_m"],
        plan["warps_n"],
    ) == (5, 2, 1, 4)


def test_mm_m7_medium_n_uses_cancellation_safe_local_split_u_chain():
    path, plan = _gfx950._dispatch_plan(7, 2048, 4096, torch.float16, 2)
    assert path == "local_split_u"
    assert plan == _gfx950._Plan(
        tile_m=16,
        tile_n=16,
        local_split_u=4,
        wave_k=256,
        k_width=8,
    )
    assert not _gfx950._uses_local_split_u_lds_for_shape(7, 2048, 4096)


def test_mm_m7_wide_n_uses_four_wave_direct_local_split_u():
    path, plan = _gfx950._dispatch_plan(7, 8192, 2048, torch.float16, 2)
    assert path == "local_split_u"
    assert plan == _gfx950._Plan(
        tile_m=16,
        tile_n=32,
        local_split_u=4,
        wave_k=128,
        k_width=8,
    )
    assert _gfx950._uses_local_split_u_lds_for_shape(7, 8192, 2048)


def test_mm_m677_medium_n_uses_saturating_k128_split_k():
    plan = _gfx950._wave_grid_plan_for_shape(677, 2048, 4096, torch.float16)
    assert plan["block_k"] == 128
    assert plan["split_k"] == 2
    assert plan["num_xcds"] == 1
    assert plan["workgroup_mapping"] == 1
    assert (
        plan["mi_wave_tile_m"],
        plan["mi_wave_tile_n"],
        plan["warps_m"],
        plan["warps_n"],
    ) == (3, 4, 2, 2)


def test_mm_m677_wide_n_uses_k128_direct_refills():
    plan = _gfx950._wave_grid_plan_for_shape(677, 4096, 2048, torch.float16)
    assert plan["block_k"] == 128
    assert plan["split_k"] == 1
    assert plan["num_xcds"] == 4
    assert plan["workgroup_mapping"] == 4
    assert (
        plan["mi_wave_tile_m"],
        plan["mi_wave_tile_n"],
        plan["warps_m"],
        plan["warps_n"],
    ) == (3, 4, 2, 2)


def test_mm_m1072_deep_k_overlaps_late_pgr2_reads():
    plan = _gfx950._wave_grid_plan_for_shape(1072, 1584, 32768, torch.float16)
    assert plan["split_k"] == 4
    assert plan["pgr2_late_read_count"] == 8
    assert plan["workgroup_mapping"] == 8
    assert (
        plan["mi_wave_tile_m"],
        plan["mi_wave_tile_n"],
        plan["warps_m"],
        plan["warps_n"],
    ) == (5, 7, 2, 2)


def test_mm_m528_short_k_uses_wide_locality_band():
    plan = _gfx950._wave_grid_plan_for_shape(528, 73744, 256, torch.float16)
    assert (
        plan["mi_wave_tile_m"],
        plan["mi_wave_tile_n"],
        plan["warps_m"],
        plan["warps_n"],
    ) == (6, 4, 2, 2)
    assert plan["workgroup_mapping"] == 8


def test_mm_large_short_k_uses_wide_epilogue():
    plan = _gfx950._wave_grid_plan_for_shape(15440, 19696, 256, torch.float16)
    assert plan["wide_epilogue"]
    assert (
        plan["mi_wave_tile_m"],
        plan["mi_wave_tile_n"],
        plan["warps_m"],
        plan["warps_n"],
    ) == (7, 4, 2, 2)


def test_mm_m279_wide_n_uses_cold_cache_k64_plan():
    plan = _gfx950._wave_grid_plan_for_shape(279, 8192, 2048, torch.float16)
    assert plan["block_k"] == 64
    assert (
        plan["mi_wave_tile_m"],
        plan["mi_wave_tile_n"],
        plan["warps_m"],
        plan["warps_n"],
    ) == (3, 4, 2, 2)
    assert plan["num_xcds"] == 4
    assert plan["workgroup_mapping"] == 4
    assert plan["loop_unroll_factor"] == 2


def test_mm_m192_uses_bounded_streamk_tail():
    plan = _gfx950._wave_grid_plan_for_shape(192, 147472, 8192, torch.float16)
    assert plan["streamk_tail_split"] == 32
    assert _gfx950._wg_streamk_tail_schedule(192, 147472, 8192, plan) == (3, 147456, 32)

    unsupported = dict(plan)
    with pytest.raises(ValueError, match="one logical M tile"):
        _gfx950._wg_streamk_tail_schedule(384, 147472, 8192, unsupported)


def test_mm_m272_deep_k_uses_saturating_wave_grid():
    plan = _gfx950._wave_grid_plan_for_shape(272, 12272, 6144, torch.float16)
    assert plan["split_k"] == 1
    assert plan["num_xcds"] == 8
    assert plan["workgroup_mapping"] == 2
    assert (
        plan["mi_wave_tile_m"],
        plan["mi_wave_tile_n"],
        plan["warps_m"],
        plan["warps_n"],
    ) == (5, 3, 1, 4)


def test_mm_m1136_shallow_k_uses_transposed_mt256x192():
    plan = _gfx950._wg_transposed_wave_grid_plan(1136, 117136, 1024)
    assert plan["local_stages"] == 2
    assert plan["pgr2_operands"]
    assert not plan["direct_to_lds"]
    assert plan["num_xcds"] == 8
    assert plan["workgroup_mapping"] == 1
    assert (
        plan["mi_wave_tile_m"],
        plan["mi_wave_tile_n"],
        plan["warps_m"],
        plan["warps_n"],
    ) == (8, 6, 2, 2)


def test_mm_tall_narrow_k256_pairs_transposed_tiles():
    plan = _gfx950._wg_transposed_wave_grid_plan(73744, 192, 256)
    assert plan["tiles_per_program"] == 2
    assert (
        plan["mi_wave_tile_m"],
        plan["mi_wave_tile_n"],
        plan["warps_m"],
        plan["warps_n"],
    ) == (6, 6, 2, 2)


def test_mm_m1136_deep_k_uses_packed_direct_transposed_grid():
    plan = _gfx950._wg_transposed_wave_grid_plan(1136, 90128, 4096)
    assert plan["local_stages"] == 2
    assert plan["pgr2_operands"]
    assert plan["direct_to_lds"]
    assert plan["direct_chunk"] == 32
    assert plan["pack_direct_chunks"]
    assert plan["loop_unroll_factor"] == 2
    assert (
        plan["mi_wave_tile_m"],
        plan["mi_wave_tile_n"],
        plan["warps_m"],
        plan["warps_n"],
    ) == (8, 8, 2, 2)


def test_mm_m80_k1024_uses_direct_rectangular_wave_grid():
    plan = _gfx950._wave_grid_plan_for_shape(80, 17456, 1024, torch.float16)
    assert plan["direct_to_lds"]
    assert plan["direct_chunk"] == 128
    assert (
        plan["mi_wave_tile_m"],
        plan["mi_wave_tile_n"],
        plan["warps_m"],
        plan["warps_n"],
    ) == (4, 3, 2, 2)


def test_mm_short_k_specialized_plans():
    bf16_plan = _gfx950._wg_BF16_WAVE_GRID_SPECIALIZATIONS[(819200, 1024, 192)]
    row_major_plan = _gfx950._ROW_MAJOR_WAVE_GRID_PLANS[(819200, 192, 1024)]
    balanced_row_major_plan = _gfx950._ROW_MAJOR_WAVE_GRID_PLANS[(4096, 4096, 2048)]
    large_row_major_plan = _gfx950._ROW_MAJOR_WAVE_GRID_PLANS[(61440, 5120, 2048)]
    deep_row_major_plan = _gfx950._ROW_MAJOR_WAVE_GRID_PLANS[(61440, 5120, 7744)]
    very_deep_row_major_plan = _gfx950._ROW_MAJOR_WAVE_GRID_PLANS[(2048, 10240, 25408)]
    arbitrary_tail_plan = _gfx950._ROW_MAJOR_WAVE_GRID_PLANS[(4096, 242432, 1894)]
    assert bf16_plan["kind"] == "regular_mi32_wave_grid"
    assert (
        bf16_plan["mi_wave_tile_m"],
        bf16_plan["mi_wave_tile_n"],
        bf16_plan["warps_m"],
        bf16_plan["warps_n"],
    ) == (4, 2, 1, 4)
    assert not bf16_plan["reverse_local_assignment"]
    assert bf16_plan["sink_insts_to_avoid_spills"]
    assert bf16_plan["disable_unclustered_high_rp_reschedule"]
    assert row_major_plan["wide_epilogue"]
    assert row_major_plan["row_major_b_lds"]
    assert not bf16_plan["direct_to_lds"]
    assert not row_major_plan["direct_to_lds"]
    assert row_major_plan["local_stages"] == 2
    assert (
        row_major_plan["mi_wave_tile_m"],
        row_major_plan["mi_wave_tile_n"],
        row_major_plan["warps_m"],
        row_major_plan["warps_n"],
    ) == (10, 6, 2, 2)
    assert balanced_row_major_plan["local_stages"] == 2
    assert balanced_row_major_plan["pgr2_operands"]
    assert balanced_row_major_plan["row_major_b_lds"]
    assert balanced_row_major_plan["row_wise_epilogue"]
    assert balanced_row_major_plan["wide_epilogue"]
    assert balanced_row_major_plan["num_xcds"] == 1
    assert balanced_row_major_plan["workgroup_mapping"] == 4
    assert (
        balanced_row_major_plan["mi_wave_tile_m"],
        balanced_row_major_plan["mi_wave_tile_n"],
        balanced_row_major_plan["warps_m"],
        balanced_row_major_plan["warps_n"],
    ) == (8, 8, 2, 2)
    assert large_row_major_plan["local_stages"] == 2
    assert large_row_major_plan["pgr2_operands"]
    assert large_row_major_plan["row_major_b_lds"]
    assert large_row_major_plan["row_wise_epilogue"]
    assert large_row_major_plan["wide_epilogue"]
    assert large_row_major_plan["num_xcds"] == 1
    assert large_row_major_plan["workgroup_mapping"] == 1
    assert (
        large_row_major_plan["mi_wave_tile_m"],
        large_row_major_plan["mi_wave_tile_n"],
        large_row_major_plan["warps_m"],
        large_row_major_plan["warps_n"],
    ) == (8, 8, 2, 2)
    assert deep_row_major_plan["local_stages"] == 2
    assert deep_row_major_plan["pgr2_operands"]
    assert deep_row_major_plan["row_major_b_lds"]
    assert deep_row_major_plan["row_wise_epilogue"]
    assert deep_row_major_plan["wide_epilogue"]
    assert deep_row_major_plan["num_xcds"] == 1
    assert deep_row_major_plan["workgroup_mapping"] == 1
    assert (
        deep_row_major_plan["mi_wave_tile_m"],
        deep_row_major_plan["mi_wave_tile_n"],
        deep_row_major_plan["warps_m"],
        deep_row_major_plan["warps_n"],
    ) == (8, 8, 2, 2)
    assert very_deep_row_major_plan["local_stages"] == 2
    assert very_deep_row_major_plan["pgr2_operands"]
    assert very_deep_row_major_plan["row_major_b_lds"]
    assert very_deep_row_major_plan["row_wise_epilogue"]
    assert very_deep_row_major_plan["wide_epilogue"]
    assert very_deep_row_major_plan["num_xcds"] == 1
    assert very_deep_row_major_plan["workgroup_mapping"] == 1
    assert (
        very_deep_row_major_plan["mi_wave_tile_m"],
        very_deep_row_major_plan["mi_wave_tile_n"],
        very_deep_row_major_plan["warps_m"],
        very_deep_row_major_plan["warps_n"],
    ) == (6, 8, 2, 2)
    assert 25408 // very_deep_row_major_plan["block_k"] == 397
    assert arbitrary_tail_plan["local_stages"] == 2
    assert arbitrary_tail_plan["pgr2_operands"]
    assert arbitrary_tail_plan["row_major_b_lds"]
    assert arbitrary_tail_plan["wide_epilogue"]
    assert arbitrary_tail_plan["split_k"] == 1
    assert 1894 % arbitrary_tail_plan["block_k"] == 38


def test_mm_rejects_invalid_rank():
    from triton.tlx.ops import mm as tlx_mm

    a = torch.randn((16, ), device="cuda", dtype=torch.float16)
    b = torch.randn((16, 16), device="cuda", dtype=torch.float16)
    with pytest.raises(InvalidInput, match="rank-2"):
        tlx_mm(a, b)


def test_mm_rejects_mismatched_reduction_dimensions():
    from triton.tlx.ops import mm as tlx_mm

    a = torch.randn((8, 16), device="cuda", dtype=torch.float16)
    b = torch.randn((17, 8), device="cuda", dtype=torch.float16)
    with pytest.raises(InvalidInput, match="reduction dimensions"):
        tlx_mm(a, b)


def test_mm_rejects_mismatched_dtype():
    from triton.tlx.ops import mm as tlx_mm

    a = torch.randn((8, 16), device="cuda", dtype=torch.float16)
    b = torch.randn((16, 8), device="cuda", dtype=torch.float32)
    with pytest.raises(InvalidInput, match="same dtype and device"):
        tlx_mm(a, b)


def test_mm_rejects_mismatched_device():
    from triton.tlx.ops import mm as tlx_mm

    a = torch.randn((8, 16), device="cuda", dtype=torch.float16)
    b = torch.randn((16, 8), device="cpu", dtype=torch.float16)
    with pytest.raises(InvalidInput, match="same dtype and device"):
        tlx_mm(a, b)


def test_mm_accepts_full_space(monkeypatch):
    from triton.tlx.ops import mm as tlx_mm

    a = torch.randn((7, 2048), device="cuda", dtype=torch.float16)
    b = torch.randn((8192, 2048), device="cuda", dtype=torch.float16).T
    expected = torch.empty((7, 8192), device="cuda", dtype=torch.float16)

    def launch_register(actual_a, actual_b, *, out):
        assert actual_a is a
        assert actual_b is b
        assert out.shape == expected.shape
        return expected

    monkeypatch.setattr(_gfx950, "_launch_register", launch_register)
    assert tlx_mm(a, b, out=expected, space="full") is expected


def test_mm_full_space_passes_complete_register_signature(monkeypatch):
    from triton.tlx.ops import mm as tlx_mm

    a = torch.randn((16, 64), device="cuda", dtype=torch.float16)
    b = torch.randn((64, 32), device="cuda", dtype=torch.float16)
    out = torch.empty((16, 32), device="cuda", dtype=torch.float16)
    captured = {}

    class FakeTuner:

        def __getitem__(self, grid):
            captured["grid"] = grid

            def launch(*args, **kwargs):
                captured["args"] = args
                captured["kwargs"] = kwargs

            return launch

    monkeypatch.setattr(_gfx950, "_register_kernel", FakeTuner())
    assert tlx_mm(a, b, out=out, space="full") is out
    args = captured["args"]
    assert args[0] is a
    assert args[1] is b
    assert all(arg is out for arg in args[2:6])
    assert args[6:9] == (16, 32, 64)
    assert captured["kwargs"]["ADD_BIAS"] is False
    assert captured["kwargs"]["WRITE_STATS"] is False
    assert captured["kwargs"]["IS_RMS_NORM"] is False


def test_mm_rejects_invalid_space():
    from triton.tlx.ops.kernels.mm.gfx950 import mm

    a = torch.randn((7, 2048), device="cuda", dtype=torch.float16)
    b = torch.randn((8192, 2048), device="cuda", dtype=torch.float16).T
    with pytest.raises(InvalidInput, match="unknown gfx950 mm search space"):
        mm(a, b, space="bogus")


def test_mm_rejects_unsupported_operands():
    from triton.tlx.ops.kernels.mm.gfx950 import matmul, mm, supports

    a = torch.randn((7, 2048), device="cuda", dtype=torch.float16)
    b = torch.randn((8192, 2048), device="cuda", dtype=torch.float16).T
    unsupported_a = torch.randn((17, 64), device="cuda", dtype=torch.float16)
    unsupported_b = torch.randn((8192, 64), device="cuda", dtype=torch.float16).T
    assert supports(a, b)
    assert not supports(unsupported_a, unsupported_b)
    assert not supports(a.to(torch.float32), b.to(torch.float32))
    assert not supports(a, b.contiguous())
    with pytest.raises(InvalidInput, match="does not support"):
        mm(unsupported_a, unsupported_b)
    with pytest.raises(InvalidInput, match="does not support"):
        matmul(unsupported_a, unsupported_b)


def test_mm_lds_respects_output_strides():
    from triton.tlx.ops import mm as tlx_mm

    m, n, k = 2048, 512, 2048
    assert _gfx950._dispatch_plan(m, n, k, torch.float16, 2) == (
        "lds",
        (128, 128, 2),
    )
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((n, k), device="cuda", dtype=torch.float16).T
    out = torch.empty((n, m), device="cuda", dtype=torch.float16).T

    actual = tlx_mm(a, b, out=out)
    expected = torch.matmul(a, b)

    assert actual is out
    assert out.stride() == (1, m)
    torch.testing.assert_close(
        actual,
        expected,
        atol=1e-3 * expected.abs().max().item(),
        rtol=1e-3,
    )


def test_mm_lds_dynamic_m_reuses_compiled_kernel():
    device_id = torch.cuda.current_device()
    kernel_cache = _gfx950.a16w16_8wave.device_caches[device_id][0]
    kernel_cache.clear()

    dtype = torch.bfloat16
    k = n = 512
    b = torch.randn((n, k), device="cuda", dtype=dtype).T
    for m in (768, 1280):
        a = torch.randn((m, k), device="cuda", dtype=dtype)
        bias = torch.randn((m, n), device="cuda", dtype=dtype)
        actual = _gfx950._launch_lds(
            a,
            b,
            bias=bias,
            SPLIT_K=1,
            TILE=(256, 256),
        )
        expected = torch.addmm(bias, a, b)
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

    assert len(kernel_cache) == 1


def test_mm_offset_width_selection():
    i32_max_element = (1 << 30) - 1
    within_i32 = torch.empty((i32_max_element + 1, ), device="meta", dtype=torch.float16)
    beyond_i32 = torch.empty((i32_max_element + 2, ), device="meta", dtype=torch.float16)

    assert not _gfx950._needs_i64_offsets(within_i32)
    assert _gfx950._needs_i64_offsets(beyond_i32)


def test_mm_output_offset_width_selection(monkeypatch):
    launches = []

    class FakeKernel:

        def __getitem__(self, grid):

            def launch(*args, **kwargs):
                launches.append((grid, kwargs["USE_I64_C_OFFSETS"]))

            return launch

    monkeypatch.setattr(_gfx950, "a16w16_8wave", FakeKernel())
    for m, n in [(256, 256), (925210, 4096)]:
        a = torch.empty((m, 128), device="meta", dtype=torch.float16)
        b = torch.empty((128, n), device="meta", dtype=torch.float16)
        _gfx950._launch_lds(a, b, SPLIT_K=1, TILE=(256, 256))

    assert [use_i64_c_offsets for _, use_i64_c_offsets in launches] == [
        False,
        True,
    ]


def test_mm_input_offset_width_selection(monkeypatch):
    launches = []

    class FakeKernel:

        def __getitem__(self, grid):

            def launch(*args, **kwargs):
                launches.append((
                    kwargs["USE_I64_A_OFFSETS"],
                    kwargs["USE_I64_B_OFFSETS"],
                    kwargs["HAS_M_TAIL"],
                    kwargs["HAS_N_TAIL"],
                ))

            return launch

    monkeypatch.setattr(_gfx950, "a16w16_8wave", FakeKernel())
    cases = [
        ((256, 256, 4096), (False, False, False, False)),
        ((257, 256, 4096), (False, False, True, False)),
        ((256, 257, 4096), (False, False, False, True)),
        ((262400, 256, 4096), (True, False, False, False)),
        ((256, 262400, 4096), (False, True, False, False)),
    ]
    for (m, n, k), _ in cases:
        a = torch.empty((m, k), device="meta", dtype=torch.float16)
        b = torch.empty((k, n), device="meta", dtype=torch.float16)
        _gfx950._launch_lds(a, b, SPLIT_K=1, TILE=(256, 256))

    assert launches == [expected for _, expected in cases]


def test_mm_lds_scheduler_policy(monkeypatch):
    launches = []

    class FakeKernel:

        def __getitem__(self, grid):

            def launch(*args, **kwargs):
                launches.append(kwargs["enable_sched_group_barrier_scheduler"])

            return launch

    monkeypatch.setattr(_gfx950, "a16w16_8wave", FakeKernel())
    for dtype, shape in [
        (torch.bfloat16, (1024, 6144, 20480)),
        (torch.float16, (1024, 6144, 20480)),
        (torch.bfloat16, (1024, 6144, 8192)),
    ]:
        m, n, k = shape
        a = torch.empty((m, k), device="meta", dtype=dtype)
        b = torch.empty((k, n), device="meta", dtype=dtype)
        _gfx950._launch_lds(a, b, SPLIT_K=1, TILE=(256, 256))

    assert launches == [False, True, True]


def test_mm_lds_grid_mapping(monkeypatch):
    launches = []

    class FakeKernel:

        def __getitem__(self, grid):

            def launch(*args, **kwargs):
                launches.append((kwargs["GROUP_SIZE_M"], kwargs["NUM_XCDS"]))

            return launch

    monkeypatch.setattr(_gfx950, "a16w16_8wave", FakeKernel())
    for dtype, shape in [
        (torch.bfloat16, (4096, 1894, 242432)),
        (torch.float16, (4096, 1894, 242432)),
        (torch.float16, (4096, 4096, 8192)),
        (torch.float16, (1024, 16384, 8192)),
    ]:
        m, n, k = shape
        a = torch.empty((m, k), device="meta", dtype=dtype)
        b = torch.empty((k, n), device="meta", dtype=dtype)
        _gfx950._launch_lds(a, b, SPLIT_K=1, TILE=(256, 256))

    assert launches == [(8, 4), (4, 8), (4, 1), (2, 8)]


def test_mm_irregular_shape_policy():
    assert _gfx950._lds_plan_for_shape(677, 4096, 8192) == (256, 256, 4)
    assert _gfx950._strong_lds_plan(677, 4096, 8192) == (192, 256, 4)
    common_square = _gfx950._register_plan_for_shape(2041, 2041, 2048, torch.bfloat16)
    assert (
        common_square["BLOCK_M"],
        common_square["BLOCK_N"],
        common_square["BLOCK_K"],
        common_square["GROUP_M"],
        common_square["NUM_XCDS"],
        common_square["num_warps"],
    ) == (128, 128, 128, 16, 8, 8)
    common_narrow = _gfx950._register_plan_for_shape(2048, 256, 1024)
    assert (
        common_narrow["BLOCK_M"],
        common_narrow["BLOCK_N"],
        common_narrow["BLOCK_K"],
        common_narrow["GROUP_M"],
        common_narrow["NUM_XCDS"],
        common_narrow["num_warps"],
    ) == (64, 64, 256, 4, 8, 8)
    for m in (256, 257):
        small_square = _gfx950._register_plan_for_shape(m, 257, 4096, torch.float16)
        assert (
            small_square["BLOCK_M"],
            small_square["BLOCK_N"],
            small_square["BLOCK_K"],
            small_square["num_warps"],
        ) == (32, 16, 256, 2)

    expected_register_tiles = {
        (2048, 256, 1024): (64, 64, 256, 8, 2),
        (272, 3072, 4608): (64, 64, 256, 8, 2),
    }
    for shape, expected in expected_register_tiles.items():
        plan = _gfx950._register_plan_for_shape(*shape, dtype=torch.float16)
        assert (
            plan["BLOCK_M"],
            plan["BLOCK_N"],
            plan["BLOCK_K"],
            plan["num_warps"],
            plan["num_stages"],
        ) == expected

    # Cached dispatch results must remain immutable: several shapes can share
    # one tuned plan, and same-shape cache hits reuse the returned object.
    with pytest.raises(TypeError):
        common_narrow["BLOCK_M"] = 1
    assert _gfx950._register_plan_for_shape(272, 3072, 4608, torch.float16)["BLOCK_M"] == 64

    deep_k = _gfx950._intermediate_register_config(677, 2048, 4096)
    assert (
        deep_k["BLOCK_M"],
        deep_k["BLOCK_N"],
        deep_k["BLOCK_K"],
        deep_k["matrix_instr_nonkdim"],
        deep_k["num_warps"],
        deep_k["num_stages"],
    ) == (128, 64, 128, 32, 8, 3)

    high_padding = _gfx950._intermediate_register_config(279, 2048, 4096)
    assert (
        high_padding["BLOCK_M"],
        high_padding["BLOCK_N"],
        high_padding["matrix_instr_nonkdim"],
    ) == (64, 32, 16)


def test_mm_rejects_large_workspace():
    m, n, k = 262145, 2048, 256
    a = torch.empty((m, k), device="meta", dtype=torch.float16)
    b = torch.empty((k, n), device="meta", dtype=torch.float16)

    with pytest.raises(
            ValueError,
            match="FP32 workspace exceeds signed-i32 byte offsets",
    ):
        _gfx950._launch_lds(
            a,
            b,
            SPLIT_K=2,
            TILE=(256, 256),
        )


def test_mm_validated_register_launch_still_checks_bias():
    a = torch.empty((279, 4096), device="meta", dtype=torch.float16)
    b = torch.empty((4096, 256), device="meta", dtype=torch.float16)
    bias = torch.empty((278, 256), device="meta", dtype=torch.float16)

    with pytest.raises(ValueError, match="Bias must expand"):
        _gfx950._launch_register_plan(
            a,
            b,
            config=_gfx950._register_plan_for_shape(279, 256, 4096),
            bias=bias,
            _validated=True,
        )


@pytest.mark.parametrize("failure", ["type", "shape", "dtype", "device"])
def test_mm_rejects_invalid_output(failure):
    from triton.tlx.ops import mm as tlx_mm

    a = torch.randn((7, 2048), device="cuda", dtype=torch.float16)
    b = torch.randn((8192, 2048), device="cuda", dtype=torch.float16).T
    if failure == "type":
        out, match = object(), "torch.Tensor"
    elif failure == "shape":
        out = torch.empty((7, 8191), device="cuda", dtype=torch.float16)
        match = "output shape"
    elif failure == "dtype":
        out = torch.empty((7, 8192), device="cuda", dtype=torch.float32)
        match = "output dtype"
    else:
        out = torch.empty((7, 8192), device="cpu", dtype=torch.float16)
        match = "output device"
    with pytest.raises(InvalidInput, match=match):
        tlx_mm(a, b, out=out)


def test_mm_rejects_plan_that_does_not_cover_m(monkeypatch):
    import triton.tlx.ops.kernels.mm.gfx950 as gfx950

    a = torch.randn((17, 32), device="cuda", dtype=torch.float16)
    b = torch.randn((16, 32), device="cuda", dtype=torch.float16).T
    monkeypatch.setitem(
        gfx950._KNOWN_PLANS,
        (17, 16, 32),
        gfx950._Plan(16, 16, 2, 16, 4),
    )
    with pytest.raises(InvalidInput, match="at most 16 rows"):
        gfx950.matmul(a, b)


def test_mm_supports_unaligned_contiguous_k_views():
    from triton.tlx.ops.kernels.mm.gfx950 import matmul

    a = torch.randn((7, 2049), device="cuda", dtype=torch.float16)[:, 1:]
    b = torch.randn((8192, 2049), device="cuda", dtype=torch.float16)[:, 1:].T
    actual = matmul(a, b)
    expected = torch.matmul(a, b)
    torch.testing.assert_close(
        actual,
        expected,
        atol=1e-3 * expected.abs().max().item(),
        rtol=1e-3,
    )


@pytest.mark.parametrize(
    "n,k,pattern_period,segment_k",
    [(8192, 2048, 512, 128), (2048, 4096, 1024, 256)],
)
def test_mm_matches_aten_for_cancellation(n, k, pattern_period, segment_k):
    """Exercise cancellation-sensitive ordered partial reduction."""
    from triton.tlx.ops.kernels.mm.gfx950 import matmul

    values = torch.zeros(k, device="cuda", dtype=torch.float16)
    for base in range(0, k, pattern_period):
        values[base:base + segment_k] = 65504.0
        values[base + segment_k:base + 2 * segment_k] = 0.001
        values[base + 2 * segment_k:base + 3 * segment_k] = -65504.0
        values[base + 3 * segment_k:base + 4 * segment_k] = 0.001
    a = torch.ones((7, k), device="cuda", dtype=torch.float16)
    b = values[None, :].repeat(n, 1).contiguous().T

    actual = matmul(a, b)
    expected = torch.matmul(a, b)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


_INVARIANCE_SPECS = [
    BitwiseInvarianceSpec.dims("M"),
    BitwiseInvarianceSpec.dims("K"),
    BitwiseInvarianceSpec.all(),
]

_INVARIANCE_SHAPES = [
    (64, 1920, 1536, (1536, 1), (1920, 1), "bf16"),
    (2048, 1024, 1000, (1000, 1), (1024, 1), "bf16"),
    (512, 512, 48, (48, 1), (512, 1), "bf16"),
    (7, 8192, 2048, (2048, 1), (8192, 1), "fp16"),
    # TN persistent specialization: full-M run uses the persistent kernel.
    (1024, 6144, 4096, (4096, 1), (1, 4096), "fp16"),
    # Unaligned K keeps a sequential range fallback instead of raising.
    (2048, 512, 1001, (1001, 1), (1, 1001), "bf16"),
]


@pytest.mark.parametrize("spec", _INVARIANCE_SPECS)
@pytest.mark.parametrize("m,n,k,a_strides,b_strides,dtype_name", _INVARIANCE_SHAPES)
def test_mm_invariance(spec, m, n, k, a_strides, b_strides, dtype_name):
    """A non-empty spec gives exact M-slice equality (public `tlx.ops.mm`)."""
    run_mm_invariance_case("gfx950", m, n, k, a_strides, b_strides, dtype_name, spec)


def _assert_sequential_dispatch(dispatch, fixed=1):
    """Reject every K-splitting plan a fixed schedule must prune."""
    assert dispatch is not None
    path, plan = dispatch
    assert path not in ("local_split_u", "streamk", "ragged_n", "m192n256")
    if path == "lds":
        assert plan[2] == fixed
    if path == "inter_wave":
        assert plan is not None and plan["SPLIT_K"] == fixed
    if path == "wave_grid":
        assert _gfx950._wave_grid_is_sequential(plan, fixed)


def test_heuristic_prunes_split_k():
    """A fixed schedule prunes K-splitting plans from the dispatch (no launch)."""
    invariance = BitwiseInvarianceSpec.dims("K")
    # (7, 8192, 2048) fp16 selects a LocalSplitU plan by default.
    default = _gfx950._dispatch_plan(7, 8192, 2048, torch.float16, 2)
    assert default[0] == "local_split_u"
    pruned = _gfx950._dispatch_plan(7, 8192, 2048, torch.float16, 2, invariance=invariance)
    _assert_sequential_dispatch(pruned)
    # (677, 4096, 8192) bf16 selects a split-K LDS plan by default.
    default = _gfx950._dispatch_plan(677, 4096, 8192, torch.bfloat16, 2)
    assert default == ("lds", (192, 256, 4))
    pruned = _gfx950._dispatch_plan(677, 4096, 8192, torch.bfloat16, 2, invariance=invariance)
    _assert_sequential_dispatch(pruned)
    # Persistent shapes stay eligible: the kernel reduces K sequentially
    # within one program per output tile.
    pruned = _gfx950._dispatch_plan(1024, 6144, 4096, torch.float16, 2, invariance=invariance)
    assert pruned == ("persistent", "mt224x128_k4096")


def test_heuristic_unaligned_k_keeps_sequential_fallback():
    """Unaligned K keeps a sequential range fallback under the spec (no launch)."""
    strides_a, strides_b = (1001, 1), (1, 1001)
    default = _gfx950.heuristic_config(2048, 512, 1001, torch.bfloat16, 2, strides_a, strides_b)
    assert default[0] == "range_register"
    pruned = _gfx950.heuristic_config(2048, 512, 1001, torch.bfloat16, 2, strides_a, strides_b,
                                      invariance=BitwiseInvarianceSpec.dims("M"))
    assert pruned == default


def test_heuristic_split_k_inter_wave_promotion_stays_sequential(monkeypatch):
    """A split-K inter-wave promotion stays sequential under the spec (no launch)."""
    shape = (2048, 25408, 10240)
    monkeypatch.setitem(_gfx950._BF16_INTER_WAVE_PROMOTIONS, shape, {"TILE": (256, 256), "SPLIT_K": 4})
    _gfx950._dispatch_plan.cache_clear()
    try:
        pruned = _gfx950._dispatch_plan(*shape, torch.bfloat16, 2, invariance=BitwiseInvarianceSpec.dims("K"))
    finally:
        _gfx950._dispatch_plan.cache_clear()
    _assert_sequential_dispatch(pruned)


def test_full_space_fixed_takes_sequential_register(monkeypatch):
    """`space="full"` under a fixed schedule launches the register path (no launch)."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    a = SimpleNamespace(is_cuda=True, shape=(64, 128), dtype=torch.bfloat16, device="cuda:0")
    a.stride = lambda: (128, 1)
    b = SimpleNamespace(shape=(128, 64), dtype=torch.bfloat16, device="cuda:0")
    b.stride = lambda: (64, 1)
    # `mm` validates `out` is a `torch.Tensor` before the mocked launch;
    # use a tensor-spec mock so this GPU-free test passes validation.
    out = MagicMock(spec=torch.Tensor)
    out.shape = (64, 64)
    out.dtype = torch.bfloat16
    out.device = "cuda:0"
    sentinel = object()
    monkeypatch.setattr(_gfx950, "_launch_register", lambda x, y, out=None: sentinel)
    monkeypatch.setattr(
        _gfx950, "_launch_range_autotuned", lambda *args:
        (_ for _ in ()).throw(AssertionError("full+fixed must not autotune range candidates")))
    assert _gfx950.mm(a, b, out=out, space="full", invariance=BitwiseInvarianceSpec.dims("K")) is sentinel


def test_mm_invariance_full_space():
    """End-to-end M-slice equality through `Space("full", ...)` (public `tlx.ops.mm`)."""
    from triton.tlx.ops import Space, mm as tlx_mm

    spec = BitwiseInvarianceSpec.dims("M")
    space = Space("full", invariance=spec)
    a = operand(64, 1536, (1536, 1), torch.bfloat16)
    b = operand(1536, 1920, (1920, 1), torch.bfloat16)
    full = tlx_mm(a, b, space=space)
    rerun = tlx_mm(a[:32], b, space=space)
    assert torch.equal(full[:32], rerun)
    ref = torch.matmul(a, b)
    torch.testing.assert_close(full, ref, atol=8e-3 * ref.abs().max().item(), rtol=8e-3)
    torch.cuda.empty_cache()


def _dyadic_gemm_operands(m, n, k, dtype, transposed_b):
    torch.manual_seed(97)
    a = torch.randint(-2, 3, (m, k), device="cuda", dtype=torch.int16).to(dtype) * 0.125
    shape = (n, k) if transposed_b else (k, n)
    b = torch.randint(-2, 3, shape, device="cuda", dtype=torch.int16).to(dtype) * 0.125
    return a, b.T if transposed_b else b


@pytest.mark.parametrize("m,n,k", [
    (511, 2048, 4096),
    (677, 2048, 4096),
    (704, 2048, 4096),
    (896, 1536, 3072),
    (1023, 2048, 4096),
    (677, 2048, 4224),
    (677, 2049, 4352),
    (2048, 3840, 2048),
    (2048, 4096, 960),
    (2048, 4096, 1024),
    (2048, 4096, 4096),
    (2048, 4096, 4160),
    (3072, 5120, 2048),
    (3073, 5120, 2112),
    (2032, 2048, 7072),
])
def test_mm_tn_boundaries_preserve_values_and_prefix(m, n, k):
    a, b = _dyadic_gemm_operands(m, n, k, torch.float16, True)
    actual = tlx_ops.mm(a, b, space="heuristic")
    torch.testing.assert_close(actual, a @ b, rtol=0, atol=0)
    for space in ("heuristic", "full"):
        prefix = tlx_ops.mm(a[:16], b, space=space)
        assert torch.equal(prefix.view(torch.int16), actual[:16].view(torch.int16))


@pytest.mark.parametrize("m,n,k", [(677, 2048, 4096), (3072, 5120, 2048), (2032, 2048, 7072)])
@pytest.mark.parametrize("step,padding,offset", [(1, 0, 0), (2, 0, 0), (1, 1, 1), (1, 350000, 0)])
def test_mm_tn_preserves_output_storage_and_rebinding(m, n, k, step, padding, offset):
    a, b = _dyadic_gemm_operands(m, n, k, torch.float16, True)
    expected = a @ b
    row_stride = n * step + padding
    for space in ("heuristic", "full"):
        storage = torch.full((offset + m * row_stride + 8, ), float("nan"), device="cuda", dtype=a.dtype)
        out = torch.as_strided(storage, (m, n), (row_stride, step), storage_offset=offset)
        assert tlx_ops.mm(a, b, out=out, space=space) is out
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        body = storage[offset:offset + m * row_stride].view(m, row_stride)
        if step == 2:
            assert body[:, 1:n * step:2].isnan().all()
        if padding:
            assert body[:, n * step:].isnan().all()
        assert storage[:offset].isnan().all() and storage[-8:].isnan().all()
        rebound = torch.full((m, n), float("nan"), device="cuda", dtype=a.dtype)
        assert tlx_ops.mm(a, b, out=rebound, space=space) is rebound
        torch.testing.assert_close(rebound, expected, rtol=0, atol=0)


@pytest.mark.parametrize("a_padding,b_padding", [
    (0, 32),
    (8, 8),
    (16, 16),
    (24, 56),
    (32, 64),
    (64, 96),
    (128, 160),
])
def test_mm_tn_long_k_padding_and_prefix(a_padding, b_padding):
    m, n, k = 2032, 2048, 7072
    values_a, values_b = _dyadic_gemm_operands(m, n, k, torch.float16, True)
    a = torch.empty_strided((m, k), (k + a_padding, 1), dtype=values_a.dtype, device="cuda")
    b = torch.empty_strided((k, n), (1, k + b_padding), dtype=values_b.dtype, device="cuda")
    a.copy_(values_a)
    b.copy_(values_b)
    expected = a @ b
    row_stride = n + 600000
    storage = torch.full((m * row_stride + 8, ), float("nan"), device="cuda", dtype=a.dtype)
    out = torch.as_strided(storage, (m, n), (row_stride, 1))
    assert tlx_ops.mm(a, b, out=out, space="heuristic") is out
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    assert storage[:m * row_stride].view(m, row_stride)[:, n:].isnan().all()
    assert storage[-8:].isnan().all()
    rebound = torch.full((m, n), float("nan"), device="cuda", dtype=a.dtype)
    assert tlx_ops.mm(a, b, out=rebound, space="heuristic") is rebound
    torch.testing.assert_close(rebound, expected, rtol=0, atol=0)
    for space in ("heuristic", "full"):
        prefix = tlx_ops.mm(a[:16], b, space=space)
        assert torch.equal(prefix.view(torch.int16), out[:16].view(torch.int16))


@pytest.mark.parametrize("k,a_padding,b_padding", [(196864, 64, 96), (242432, 0, 0), (242944, 32, 64)])
def test_mm_bf16_long_k_preserves_output_storage_and_prefix(k, a_padding, b_padding):
    m, n = 4096, 1894
    values_a, values_b = _dyadic_gemm_operands(m, n, k, torch.bfloat16, True)
    a = torch.empty_strided((m, k), (k + a_padding, 1), dtype=values_a.dtype, device="cuda")
    b = torch.empty_strided((k, n), (1, k + b_padding), dtype=values_b.dtype, device="cuda")
    a.copy_(values_a)
    b.copy_(values_b)
    del values_a, values_b
    expected = a @ b
    actual = tlx_ops.mm(a, b, space="heuristic")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for step in (1, 2):
        storage = torch.full((m, n * step + 16), float("nan"), device="cuda", dtype=a.dtype)
        out = storage[:, 8:8 + n * step:step]
        assert tlx_ops.mm(a, b, out=out, space="heuristic") is out
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        assert storage[:, :8].isnan().all()
        assert storage[:, 8 + n * step:].isnan().all()
        if step == 2:
            assert storage[:, 9:8 + n * step:step].isnan().all()
    for space in ("heuristic", "full"):
        prefix = tlx_ops.mm(a[:16], b, space=space)
        assert torch.equal(prefix.view(torch.int16), actual[:16].view(torch.int16))


@pytest.mark.parametrize("dtype_name,m,n,k", [
    ("bf16", 524288, 512, 192),
    ("fp16", 262144, 256, 256),
])
@pytest.mark.parametrize("step,padding,offset", [(1, 0, 0), (2, 0, 0), (1, 1, 1), (1, 4096, 0)])
def test_mm_cached_b_preserves_output_storage_and_prefix(dtype_name, m, n, k, step, padding, offset):
    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16
    a, b = _dyadic_gemm_operands(m, n, k, dtype, dtype_name == "bf16")
    row_stride = n * step + padding
    storage = torch.full((offset + m * row_stride + 8, ), float("nan"), device="cuda", dtype=dtype)
    out = torch.as_strided(storage, (m, n), (row_stride, step), storage_offset=offset)
    assert tlx_ops.mm(a, b, out=out, space="heuristic") is out
    torch.testing.assert_close(out, a @ b, rtol=0, atol=0)
    body = storage[offset:offset + m * row_stride].view(m, row_stride)
    if step == 2:
        assert body[:, 1:n * step:2].isnan().all()
    if padding:
        assert body[:, n * step:].isnan().all()
    assert storage[:offset].isnan().all() and storage[-8:].isnan().all()
    for space in ("heuristic", "full"):
        prefix = tlx_ops.mm(a[:16], b, space=space)
        assert torch.equal(prefix.view(torch.int16), out[:16].view(torch.int16))


def test_mm_cached_b_rejects_i32_offset_overflow():
    m, n, k = 524288, 512, 192
    dense = _gfx950._optimized_geometry_config(m, n, k, torch.bfloat16, 2, (k, 1), (n, 1))
    assert dense is not None and dense[0] == "cached_b_lds"
    b_strides = (2**26, 1)
    assert _gfx950._optimized_geometry_config(m, n, k, torch.bfloat16, 2, (k, 1), b_strides) is None
    assert _gfx950.heuristic_config(m, n, k, torch.bfloat16, 2, (k, 1), b_strides)[0] == "register"


@pytest.mark.parametrize("m,n,k", [
    (32768, 4096, 1536),
    (32768, 4352, 1664),
    (32769, 4352, 1664),
    (32832, 5120, 1792),
    (61632, 5120, 2176),
])
def test_mm_grouped_xcd_covers_partial_grid(m, n, k):
    a, b = _dyadic_gemm_operands(m, n, k, torch.float16, False)
    out = torch.full((m, n), float("nan"), device="cuda", dtype=a.dtype)
    assert tlx_ops.mm(a, b, out=out, space="heuristic") is out
    torch.testing.assert_close(out, a @ b, rtol=0, atol=0)


@pytest.mark.parametrize("m,n,k", [(33024, 5120, 1792), (819200, 192, 1024)])
@pytest.mark.parametrize("step,padding,offset", [(1, 0, 0), (2, 0, 0), (1, 1, 1), (1, 35000, 0)])
def test_mm_row_major_output_storage_and_prefix(m, n, k, step, padding, offset):
    a, b = _dyadic_gemm_operands(m, n, k, torch.float16, False)
    expected = a @ b
    row_stride = n * step + padding
    for space in ("heuristic", "full"):
        storage = torch.full((offset + m * row_stride + 8, ), float("nan"), device="cuda", dtype=a.dtype)
        out = torch.as_strided(storage, (m, n), (row_stride, step), storage_offset=offset)
        assert tlx_ops.mm(a, b, out=out, space=space) is out
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        body = storage[offset:offset + m * row_stride].view(m, row_stride)
        if step == 2:
            assert body[:, 1:n * step:2].isnan().all()
        if padding:
            assert body[:, n * step:].isnan().all()
        assert storage[:offset].isnan().all() and storage[-8:].isnan().all()
        rebound = torch.full((m, n), float("nan"), device="cuda", dtype=a.dtype)
        assert tlx_ops.mm(a, b, out=rebound, space=space) is rebound
        torch.testing.assert_close(rebound, expected, rtol=0, atol=0)
        prefix = tlx_ops.mm(a[:16], b, space=space)
        assert torch.equal(prefix.view(torch.int16), out[:16].view(torch.int16))


@pytest.mark.parametrize("m,n,k", [(33024, 5120, 1792), (819200, 192, 1024)])
@pytest.mark.parametrize("a_padding,b_padding", [(8, 16), (16, 0), (24, 0), (32, 32), (64, 64), (128, 256)])
def test_mm_row_major_padded_operands(m, n, k, a_padding, b_padding):
    values_a, values_b = _dyadic_gemm_operands(m, n, k, torch.float16, False)
    a = torch.empty_strided((m, k), (k + a_padding, 1), dtype=values_a.dtype, device="cuda")
    b = torch.empty_strided((k, n), (n + b_padding, 1), dtype=values_b.dtype, device="cuda")
    a.copy_(values_a)
    b.copy_(values_b)
    expected = a @ b
    for space in ("heuristic", "full"):
        actual = tlx_ops.mm(a, b, space=space)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        prefix = tlx_ops.mm(a[:16], b, space=space)
        assert torch.equal(prefix.view(torch.int16), actual[:16].view(torch.int16))


@pytest.mark.parametrize("m,n,k", [
    (63, 32768, 192),
    (64, 32768, 192),
    (127, 33153, 192),
    (128, 33153, 256),
    (129, 32768, 192),
    (128, 32767, 192),
    (128, 32768, 128),
    (128, 32768, 320),
])
def test_mm_short_k_transposed_family_boundaries(m, n, k):
    a, b = _dyadic_gemm_operands(m, n, k, torch.float16, True)
    actual = tlx_ops.mm(a, b, space="heuristic")
    torch.testing.assert_close(actual, a @ b, rtol=0, atol=0)
    for space in ("heuristic", "full"):
        prefix = tlx_ops.mm(a[:16], b, space=space)
        assert torch.equal(prefix.view(torch.int16), actual[:16].view(torch.int16))
