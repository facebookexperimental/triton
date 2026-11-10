"""gfx950 correctness coverage for ``tlx.ops.mm``."""

import time

import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_hip_cdna4
from triton.tlx import ops as tlx_ops
from triton.tlx.ops import BitwiseInvarianceSpec, InvalidInput, Space, UnsupportedOp
from triton.tlx.ops.kernels.mm._shapes import CORRECTNESS_SHAPES, FOCUS, operand
from triton.tlx.ops.kernels.mm import gfx950 as _gfx950

from mm_test_utils import run_mm_invariance_case

pytestmark = pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950")

MAX_SECONDS_PER_CASE = 60

# TODO: Re-enable shapes here when their direct-TLX correctness failures are fixed.
FAILED_SHAPES = set()


def _assert_range_result(actual, expected):
    assert actual.shape == expected.shape
    tolerance = 1e-2 if actual.dtype == torch.bfloat16 else 1e-3
    amax = max(expected[row:row + 64].abs().max().item() for row in range(0, expected.shape[0], 64))
    for row in range(0, expected.shape[0], 64):
        torch.testing.assert_close(actual[row:row + 64], expected[row:row + 64], atol=tolerance * amax, rtol=tolerance)


def _split_register_test_plan(split_k):
    return _gfx950._normalize_split_register_plan(
        dict(_gfx950._fixed_register_plan(16, 64, 128, 4, 1, 4, 2), SPLIT_K=split_k, GROUP_M=4, NUM_XCDS=1,
             CACHE_B=False, ACCUMULATE_IN_DOT=False, SERIAL_REDUCE=True, K_TAIL="mask"))


def _launch_split_register_test(a, b, plan, *, out=None, workspace=None):
    if out is None:
        out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
    return _gfx950._launch_split_register(a, b, out, plan, workspace=workspace)


@pytest.mark.parametrize("m,n,k,split_k", [(1, 73, 260, 2), (27, 129, 516, 4), (31, 257, 1024, 8)])
def test_mm_split_register_fp32_partials_and_tails(m, n, k, split_k):
    torch.manual_seed(107)
    a = torch.randint(-2, 3, (m, k), device="cuda").to(torch.bfloat16)
    b = torch.randint(-2, 3, (n, k), device="cuda").to(torch.bfloat16).T
    workspace = torch.empty((split_k * m, n), device="cuda", dtype=torch.float32)
    out = torch.empty((m, n * 2), device="cuda", dtype=a.dtype)[:, ::2]
    assert _launch_split_register_test(a, b, _split_register_test_plan(split_k), out=out, workspace=workspace) is out
    ks = k // split_k
    for split in range(split_k):
        expected = a[:, split * ks:(split + 1) * ks].float() @ b[split * ks:(split + 1) * ks].float()
        torch.testing.assert_close(workspace.view(split_k, m, n)[split], expected, atol=0, rtol=0)
    torch.testing.assert_close(out, (a.float() @ b.float()).to(a.dtype), atol=0, rtol=0)


@pytest.mark.parametrize("m", [1, 27])
def test_mm_split_register_catalog_candidates_accuracy(m):
    n, k = 4096, 8192
    torch.manual_seed(113)
    a = torch.randint(-2, 3, (m, k), device="cuda").to(torch.bfloat16)
    b = torch.randint(-2, 3, (n, k), device="cuda").to(torch.bfloat16).T
    expected = (a.float() @ b.float()).to(a.dtype)
    candidates = _gfx950._full_dispatch_candidates(m, n, k, a.dtype, 2, a.stride(), b.stride())
    split_candidates = [dispatch for dispatch in candidates if dispatch[0] == "split_register"]
    assert split_candidates
    executable = set()
    for dispatch in split_candidates:
        try:
            actual = _launch_split_register_test(a, b, dispatch[1])
        except triton.OutOfResources:
            continue
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        executable.add((dispatch[1]["BLOCK_K"], dispatch[1]["num_stages"]))
    assert {(128, 2), (128, 3), (256, 2)} <= executable


@pytest.mark.parametrize("invalid", ["output", "workspace", "unequal_slices", "dtype"])
def test_mm_split_register_rejects_invalid_input(invalid):
    a = torch.ones((27, 516), device="cuda", dtype=torch.bfloat16)
    b = torch.ones((129, 516), device="cuda", dtype=a.dtype).T
    config = _split_register_test_plan(4)
    kwargs = {}
    if invalid == "output":
        kwargs["out"] = torch.empty((27, 129), device="cuda", dtype=torch.float32)
    elif invalid == "workspace":
        kwargs["workspace"] = torch.empty((108, 129), device="cuda", dtype=torch.bfloat16)
    elif invalid == "unequal_slices":
        config = _gfx950._normalize_split_register_plan(dict(config, SPLIT_K=5, K_TAIL="mask"))
    else:
        a = torch.ones((27, 516), device="cuda", dtype=torch.float32)
        b = torch.ones((129, 516), device="cuda", dtype=a.dtype).T
    with pytest.raises(InvalidInput):
        _launch_split_register_test(a, b, config, **kwargs)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mm_split_register_supported_dtypes(dtype):
    torch.manual_seed(109)
    m, n, k, split_k = 27, 129, 516, 4
    a = torch.randint(-2, 3, (m, k), device="cuda").to(dtype)
    b = torch.randint(-2, 3, (n, k), device="cuda").to(dtype).T
    out = _launch_split_register_test(a, b, _split_register_test_plan(split_k))
    torch.testing.assert_close(out, (a.float() @ b.float()).to(dtype), atol=0, rtol=0)


def test_mm_split_register_graph_replay():
    a = torch.ones((27, 516), device="cuda", dtype=torch.bfloat16)
    b = torch.ones((129, 516), device="cuda", dtype=a.dtype).T
    out = torch.empty((27, 129), device="cuda", dtype=a.dtype)
    workspace = torch.empty((108, 129), device="cuda", dtype=torch.float32)
    config = _split_register_test_plan(4)
    for _ in range(3):
        _launch_split_register_test(a, b, config, out=out, workspace=workspace)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _launch_split_register_test(a, b, config, out=out)
    a.fill_(2)
    for _ in range(3):
        graph.replay()
    torch.testing.assert_close(out, (a.float() @ b.float()).to(a.dtype), atol=0, rtol=0)


def test_mm_split_register_cached_public_dispatch():
    a = torch.ones((27, 516), device="cuda", dtype=torch.bfloat16)
    b = torch.ones((129, 516), device="cuda", dtype=a.dtype).T
    out = torch.empty((27, 129), device="cuda", dtype=a.dtype)
    key = _gfx950._range_tuned_plan_cache_key(a, b, out)
    selected = ("split_register", _split_register_test_plan(4))
    previous = _gfx950._RANGE_TUNED_PLAN_CACHE.get(key)
    try:
        _gfx950._RANGE_TUNED_PLAN_CACHE[key] = selected
        assert tlx_ops.mm(a, b, out=out, space="full") is out
        assert _gfx950._RANGE_TUNED_PLAN_CACHE[key] == selected
        torch.testing.assert_close(out, (a.float() @ b.float()).to(a.dtype), atol=0, rtol=0)
    finally:
        if previous is None:
            _gfx950._RANGE_TUNED_PLAN_CACHE.pop(key, None)
        else:
            _gfx950._RANGE_TUNED_PLAN_CACHE[key] = previous


def test_mm_split_register_single_canonical_route():
    torch.manual_seed(117)
    m, n, k, split_k = 27, 129, 516, 4
    a = torch.randint(-2, 3, (m, k), device="cuda").to(torch.bfloat16)
    b = torch.randint(-2, 3, (n, k), device="cuda").to(torch.bfloat16).T
    new_plan = _split_register_test_plan(split_k)
    out_new = _launch_split_register_test(a, b, new_plan)
    torch.testing.assert_close(out_new, (a.float() @ b.float()).to(a.dtype), atol=0, rtol=0)
    candidates = _gfx950._full_dispatch_candidates(m, n, k, a.dtype, 2, a.stride(), b.stride())
    assert all(path != "split_k_register" for path, _ in candidates)
    assert not hasattr(_gfx950, "_split_k_workspace_kernel")
    assert not hasattr(_gfx950, "_launch_split_register_plan")


def test_mm_split_register_qkv_launch_options(monkeypatch):
    torch.manual_seed(131)
    m, n, k = 27, 2304, 8192
    a = torch.randint(-2, 3, (m, k), device="cuda").to(torch.bfloat16)
    b = torch.randint(-2, 3, (n, k), device="cuda").to(torch.bfloat16).T
    out = torch.empty((m, n), device="cuda", dtype=a.dtype)
    seen = {}
    kernel = _gfx950._split_register_kernel

    class ProducerRecorder:

        def __getitem__(self, grid):

            def launch(*args, **kwargs):
                seen.update(kwargs)
                return kernel.run(*args, grid=grid, warmup=False, **kwargs)

            return launch

    monkeypatch.setattr(_gfx950, "_split_register_kernel", ProducerRecorder())
    assert _gfx950._launch_split_register(a, b, out, _gfx950._BF16_M27_QKV_SPLIT_REGISTER_PLAN) is out
    assert seen["BLOCK_M"] == 32 and seen["BLOCK_N"] == 32 and seen["BLOCK_K"] == 64 and seen["num_warps"] == 2
    assert seen["ACCUMULATE_IN_DOT"] is True and seen["CACHE_B_CG"] is True
    assert seen["GROUP_M"] == 1 and seen["NUM_XCDS"] == 1
    assert seen["matrix_instr_nonkdim"] == 16 and seen["waves_per_eu"] == 0
    assert seen["kpack"] == 1 and seen["regclass_priority_trumps_globalness"] is False


def test_mm_split_register_high_input_offsets_use_i64_flags(monkeypatch):
    torch.manual_seed(137)
    m, n, k, split_k = 27, 129, 516, 4
    a = torch.ones((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.ones((n, k), device="cuda", dtype=a.dtype).T
    out = torch.empty((m, n), device="cuda", dtype=a.dtype)
    seen = {}
    kernel = _gfx950._split_register_kernel

    class ProducerRecorder:

        def __getitem__(self, grid):

            def launch(*args, **kwargs):
                seen.update(kwargs)
                return kernel.run(*args, grid=grid, warmup=False, **kwargs)

            return launch

    monkeypatch.setattr(_gfx950, "_split_register_kernel", ProducerRecorder())
    monkeypatch.setattr(_gfx950, "_needs_i64_offsets", lambda tensor: tensor.dtype != torch.float32)
    with pytest.raises(InvalidInput):
        _gfx950._launch_split_register(a, b, out, _split_register_test_plan(split_k))
    # Output/i64-bound views still reject; small input offsets take the i32 path.
    monkeypatch.setattr(_gfx950, "_needs_i64_offsets", lambda tensor: False)
    big_a_strides = ((2**31 - 1) // max(m - 1, 1), 1)
    big_b_strides = ((2**31 - 1) // max(k - 1, 1), 1)
    seen.clear()
    assert _gfx950._launch_split_register(a, b, out, _split_register_test_plan(split_k)) is out
    assert seen["USE_I64_A_OFFSETS"] is False and seen["USE_I64_B_OFFSETS"] is False
    assert _gfx950._offsets_fit_signed_i32(m, n, k, 2, big_a_strides, big_b_strides) is False


def test_mm_split_register_no_duplicate_recipes():
    torch.manual_seed(119)
    a = torch.ones((27, 8192), device="cuda", dtype=torch.bfloat16)
    b = torch.ones((4096, 8192), device="cuda", dtype=a.dtype).T
    candidates = _gfx950._full_dispatch_candidates(27, 4096, 8192, a.dtype, 2, a.stride(), b.stride())
    keys = [
        _gfx950._split_register_recipe_key(_gfx950._normalize_split_register_plan(dict(plan, SPLIT_K=plan["SPLIT_K"])))
        for path, plan in candidates
        if path == "split_register"
    ]
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize("k_tail,expect", [("forbid", "fallback"), ("mask", "exact")])
def test_mm_split_register_block_k_tail_policy(k_tail, expect, monkeypatch):
    torch.manual_seed(123)
    m, n, k, split_k = 27, 129, 260, 2
    a = torch.randint(-2, 3, (m, k), device="cuda").to(torch.bfloat16)
    b = torch.randint(-2, 3, (n, k), device="cuda").to(torch.bfloat16).T
    plan = _gfx950._normalize_split_register_plan(dict(_split_register_test_plan(split_k), K_TAIL=k_tail))
    out = torch.empty((m, n), device="cuda", dtype=a.dtype)
    if expect == "fallback":
        seen = []
        real = _gfx950._launch_range_register_plan

        def record(aa, bb, *, config, out):
            seen.append(config)
            return real(aa, bb, config=config, out=out)

        monkeypatch.setattr(_gfx950, "_launch_range_register_plan", record)
        assert _gfx950._launch_split_register(a, b, out, plan) is out
        assert seen
    else:
        assert _launch_split_register_test(a, b, plan, out=out) is out
    torch.testing.assert_close(out, (a.float() @ b.float()).to(a.dtype), atol=0, rtol=0)


def test_mm_split_register_mask_rejects_unequal_slices():
    a = torch.ones((27, 518), device="cuda", dtype=torch.bfloat16)
    b = torch.ones((129, 518), device="cuda", dtype=a.dtype).T
    with pytest.raises(InvalidInput):
        _launch_split_register_test(a, b, _split_register_test_plan(4))
    legacy_plan = _gfx950._normalize_split_register_plan(dict(_split_register_test_plan(4), K_TAIL="forbid"))
    actual = _launch_split_register_test(a, b, legacy_plan)
    torch.testing.assert_close(actual, (a.float() @ b.float()).to(a.dtype), atol=0, rtol=0)


def test_mm_split_register_mask_rejects_short_slices():
    a = torch.ones((27, 60), device="cuda", dtype=torch.bfloat16)
    b = torch.ones((129, 60), device="cuda", dtype=a.dtype).T
    with pytest.raises(InvalidInput):
        _launch_split_register_test(a, b, _split_register_test_plan(2))


@pytest.mark.parametrize("shape", [(4, 27, 129), (108, 129)])
def test_mm_split_register_caller_workspace_shapes(shape):
    torch.manual_seed(127)
    m, n, k, split_k = 27, 129, 516, 4
    a = torch.randint(-2, 3, (m, k), device="cuda").to(torch.bfloat16)
    b = torch.randint(-2, 3, (n, k), device="cuda").to(torch.bfloat16).T
    workspace = torch.empty(shape, device="cuda", dtype=torch.float32)
    out = torch.empty((m, n), device="cuda", dtype=a.dtype)
    assert _launch_split_register_test(a, b, _split_register_test_plan(split_k), out=out, workspace=workspace) is out
    torch.testing.assert_close(out, (a.float() @ b.float()).to(a.dtype), atol=0, rtol=0)


def test_mm_full_split_candidates_skip_i64_output(monkeypatch):
    from triton import testing

    a = torch.ones((27, 512), device="cuda", dtype=torch.bfloat16)
    b = torch.ones((128, 512), device="cuda", dtype=a.dtype).T
    out = torch.empty_strided((27, 128), (2**31 // 26 + 1, 1), device="cuda", dtype=a.dtype)
    key = _gfx950._range_tuned_plan_cache_key(a, b, out)
    selected = ("range_register", _gfx950._fixed_register_plan(16, 16, 256, 4, 1, 4, 2))
    visited = []
    monkeypatch.setattr(_gfx950, "_launch_dispatch", lambda x, y, z, dispatch: visited.append(dispatch) or z)
    monkeypatch.setattr(testing, "do_bench_cudagraph", lambda fn, **kwargs: fn() is not None)
    previous = _gfx950._RANGE_TUNED_PLAN_CACHE.pop(key, None)
    try:
        candidates = (selected, ("split_register", _split_register_test_plan(4)), ("lds", (128, 128, 4)))
        assert _gfx950._launch_range_autotuned(a, b, out, candidates) is out
        assert visited and all(dispatch == selected for dispatch in visited)
    finally:
        _gfx950._RANGE_TUNED_PLAN_CACHE.pop(key, None)
        if previous is not None:
            _gfx950._RANGE_TUNED_PLAN_CACHE[key] = previous


def test_mm_split_register_is_pruned_from_fixed_schedule():
    assert not _gfx950._range_candidate_is_sequential("split_register", _split_register_test_plan(4), 1)


@pytest.mark.parametrize("m,n,k", [(1, 8192, 4096), (27, 2304, 8192)])
def test_mm_full_space_autotunes_split_k_candidates(m, n, k):
    torch.manual_seed(m)
    a = torch.randint(-2, 3, (m, k), device="cuda").to(torch.bfloat16)
    b = torch.randint(-2, 3, (n, k), device="cuda").to(torch.bfloat16).T
    paths = {d[0] for d in _gfx950._full_dispatch_candidates(m, n, k, a.dtype, 2, a.stride(), b.stride())}
    assert "split_register" in paths
    assert "split_k_register" not in paths
    torch.testing.assert_close(tlx_ops.mm(a, b, space="full"), (a.float() @ b.float()).to(a.dtype), atol=0, rtol=0)


@triton.jit
def _dual_b_i64_test_kernel(a, b, b2, out, out2, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                            stride_bk: tl.constexpr, stride_bn: tl.constexpr):
    rows = tl.arange(0, 16)
    cols = tl.program_id(0) * 16 + tl.arange(0, 16)
    valid_cols = tl.where(cols < N, cols, 0)
    acc, acc2 = _gfx950._register_matmul_core(
        a,
        b,
        b2,
        rows,
        valid_cols,
        rows < M,
        N,
        K,
        0,
        K,
        1,
        stride_bk,
        stride_bn,
        16,
        16,
        128,
        MASK_ROWS=True,
        DUAL_B=True,
        USE_I64_B_OFFSETS=True,
    )
    offsets = rows[:, None] * N + cols[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    tl.store(out + offsets, acc, mask=mask)
    tl.store(out2 + offsets, acc2, mask=mask)


def test_mm_shared_core_dual_b_i64_full_and_tail():
    m, n, k = 16, 129, 255
    stride = 2**31 // (n - 1) + 1
    a = torch.ones((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.empty_strided((k, n), (1, stride), device="cuda", dtype=a.dtype)
    b2 = torch.empty_strided((k, n), (1, stride), device="cuda", dtype=a.dtype)
    b.fill_(1)
    b2.fill_(2)
    out = torch.empty((m, n), device="cuda", dtype=torch.float32)
    out2 = torch.empty_like(out)
    _dual_b_i64_test_kernel[(triton.cdiv(n, 16), )](a, b, b2, out, out2, m, n, k, *b.stride(), num_warps=4,
                                                    matrix_instr_nonkdim=16)
    torch.testing.assert_close(out, torch.full_like(out, k), atol=0, rtol=0)
    torch.testing.assert_close(out2, torch.full_like(out2, 2 * k), atol=0, rtol=0)


@pytest.mark.parametrize("m", [1, 27])
def test_mm_full_dense_tn_preserves_attention_hot_candidate(m):
    candidates = _gfx950._full_dispatch_candidates(m, 8192, 1024, torch.bfloat16, 2, (1024, 1), (1, 1024))
    expected = {
        "BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 256, "GROUP_M": 4, "NUM_XCDS": 8, "num_warps": 4, "num_stages": 3,
        "waves_per_eu": 2, "CACHE_B_CG": False
    }
    assert any(path == "range_register" and all(plan.get(key) == value
                                                for key, value in expected.items())
               for path, plan in candidates)


@pytest.mark.parametrize("m,n,k,split", [
    (1, 8192, 1024, None),
    (27, 8192, 1024, None),
    (1920, 4096, 8192, 2),
    (1920, 8192, 4096, 1),
])
@pytest.mark.parametrize("dtype,contiguous_out,promoted", [
    (torch.bfloat16, True, True),
    (torch.bfloat16, False, False),
    (torch.float16, True, False),
])
def test_mm_full_dense_tn_preserves_hot_incumbent(monkeypatch, m, n, k, split, dtype, contiguous_out, promoted):
    a = torch.empty((m, k), device="cuda", dtype=dtype)
    b = torch.empty((n, k), device="cuda", dtype=dtype).T
    out = torch.empty_strided((m, n), (n if contiguous_out else n * 2, 1 if contiguous_out else 2), device="cuda",
                              dtype=dtype)
    assert out.is_contiguous() == contiguous_out
    heuristic = ("gemv", {})
    legacy = (("lds", (256, 256, split)) if split is not None else
              ("range_register", dict(_gfx950._BF16_M27_REGISTER_CONFIGS[(n, k)], CACHE_B_CG=False, waves_per_eu=2)))
    observed = []
    monkeypatch.setattr(_gfx950, "_can_use_range_register_compiled_cache", lambda: False)
    monkeypatch.setattr(_gfx950, "_dispatch_for", lambda *args: heuristic)
    monkeypatch.setattr(_gfx950, "_optimized_dispatch_for", lambda *args: heuristic)
    monkeypatch.setattr(_gfx950, "_full_dispatch_candidates", lambda *args: [heuristic, legacy])
    monkeypatch.setattr(_gfx950, "_launch_gemv", lambda *args: out)

    def autotune(a, b, out, candidates):
        observed.extend(candidates)
        return out

    monkeypatch.setattr(_gfx950, "_launch_range_autotuned", autotune)
    assert _gfx950.mm(a, b, out=out, space="full") is out
    # The uncached-B (8192, 1024) plan stays a candidate but is not pinned ahead of the heuristic.
    assert observed[0] == (legacy if promoted and split is not None else heuristic)
    assert observed.count(legacy) == 1
    assert heuristic in observed
    observed.clear()
    assert _gfx950.mm(a, b, out=out) is out
    assert not observed


@pytest.mark.parametrize("m", [1, 192])
@pytest.mark.parametrize("use_cache,winner_timings,paired,cold,expected,event_count", [
    (True, [0.8, 1.0, 0.85], [0.9] * 9, [1.0] * 9, True, 36),
    (True, [0.8, 1.0, 0.85], [0.9] * 9, [1.02] * 9, True, 36),
    (True, [0.8, 1.0, 0.85], [0.9] * 9, [1.0] * 4 + [1.2] * 5, False, 36),
    (True, [0.8, 1.0, 0.85], [0.9] * 3 + [1.1] * 3 + [0.9] * 3, [], False, 18),
    (True, [1.0] * 3, [0.9] * 9, [], False, 0),
    (False, [0.8, 1.0, 0.85], [0.9] * 9, [], False, 0),
])
def test_mm_dense_full_winner_overlap_uses_paired_recheck(monkeypatch, m, use_cache, winner_timings, paired, cold,
                                                          expected, event_count):
    a = torch.ones((m, 128), device="cuda", dtype=torch.bfloat16)
    b = torch.ones((128, 128), device="cuda", dtype=a.dtype).T
    out = torch.full((m, 128), 128, device="cuda", dtype=a.dtype)
    candidates = [("range_register", _gfx950._fixed_register_plan(16, 16, bk, 4, 1, 4, 2)) for bk in (32, 64)]
    measurements = []
    for round_index, value in enumerate(paired):
        order = (round_index % 2, (round_index + 1) % 2)
        measurements.extend(64 * (1.0 if index == 0 else value) for index in order)
    for round_index, value in enumerate(cold):
        order = (round_index % 2, (round_index + 1) % 2)
        measurements.extend(1.0 if index == 0 else value for index in order)
    values = iter(measurements)
    observed = []

    def elapsed_time(start, end):
        # Only the clock is synthetic; graph capture and kernel launches remain real.
        value = next(values)
        observed.append(value)
        return value

    monkeypatch.setattr(torch.cuda.Event, "elapsed_time", elapsed_time)
    actual = _gfx950._range_winner_is_stable(a, b, out, candidates, [[1.0, 1.05, 0.9], winner_timings], 1, use_cache)
    torch.cuda.synchronize()
    assert actual is expected
    assert len(observed) == event_count
    assert torch.equal(out, torch.full_like(out, 128))


@pytest.mark.parametrize("m", [1, 27, 1920, 3840])
@pytest.mark.parametrize("n,k", [(2304, 8192), (8192, 1024), (4096, 8192), (8192, 4096)])
def test_mm_full_dense_tn_has_multiple_candidates(m, n, k):
    candidates = _gfx950._full_dispatch_candidates(m, n, k, torch.bfloat16, 2, (k, 1), (1, k))
    assert len(candidates) > 1
    assert any(path == "range_register" for path, _ in candidates)
    assert any(path == "lds" for path, _ in candidates)
    assert all(candidate not in candidates[:index] for index, candidate in enumerate(candidates))


@pytest.mark.parametrize("dtype,a_strides,b_strides", [
    (torch.float16, (8192, 1), (1, 8192)),
    (torch.bfloat16, (8192, 1), (128, 1)),
    (torch.bfloat16, (16384, 2), (1, 8192)),
    (torch.bfloat16, (8192, 1), (1, 8193)),
])
def test_mm_full_dense_tn_preserves_other_domains(dtype, a_strides, b_strides):
    args = (27, 128, 8192, dtype, 2, a_strides, b_strides)
    assert _gfx950._full_dispatch_candidates(*args) == _gfx950._range_dispatch_candidates(*args)


@pytest.mark.parametrize("m,n,k", [(1, 128, 8192), (27, 128, 8192), (1920, 128, 1024), (3840, 128, 8192)])
def test_mm_full_dense_tn_candidates_accuracy(m, n, k):
    torch.manual_seed(101)
    a = torch.randint(-2, 3, (m, k), device="cuda").to(torch.bfloat16)
    b = torch.randint(-2, 3, (n, k), device="cuda").to(torch.bfloat16).T
    expected = (a.float() @ b.float()).to(a.dtype)
    out = torch.empty_strided((m, n), (n * 2, 2), device="cuda", dtype=a.dtype)
    candidates = _gfx950._full_dispatch_candidates(m, n, k, a.dtype, a.element_size(), a.stride(), b.stride())
    assert len(candidates) > 1
    for dispatch in candidates:
        try:
            assert _gfx950._launch_dispatch(a, b, out, dispatch) is out
        except triton.OutOfResources:
            continue
        torch.testing.assert_close(out, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "m,n,k",
    [
        (3072, 2048, 797),
        (3072, 128, 44536),
        (3072, 1024, 2048),
        (3072, 4096, 2048),
        (4096, 192, 2048),
        (3072, 1536, 5888),
        (3072, 5888, 1536),
        (3072, 5888, 2304),
        (3072, 40440, 256),
        (3072, 44032, 128),
        (3072, 2304, 5120),
        (4096, 2048, 6144),
        (3072, 2048, 6144),
        (3072, 2048, 6640),
        (128, 9216, 7392),
        (128, 10304, 20926),
        (128, 2688, 4416),
    ],
)
def test_mm_fp16_promoted_exact_shape_accuracy(m, n, k):
    torch.manual_seed(m + n + k)
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((n, k), device="cuda", dtype=torch.float16).T
    actual = tlx_ops.mm(a, b)
    _assert_range_result(actual, a @ b)


def test_mm_fp16_lsr_k160_production_shape():
    torch.manual_seed(83)
    a = torch.randn((2496, 160), device="cuda", dtype=torch.float16)
    b = torch.randn((22528, 160), device="cuda", dtype=torch.float16).T
    actual = tlx_ops.mm(a, b)
    _assert_range_result(actual, a @ b)


def test_mm_bf16_odd_k_production_shape():
    torch.manual_seed(79)
    a = torch.randn((1024, 6145), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((4096, 6145), device="cuda", dtype=torch.bfloat16).T
    path, plan = _gfx950._dispatch_plan(1024, 4096, 6145, torch.bfloat16, 2)
    assert path == "transposed_wave_grid"
    assert plan["direct_to_lds"]
    assert plan["local_stages"] == 2
    actual = tlx_ops.mm(a, b)
    _assert_range_result(actual, a @ b)


def test_mm_split_k_production_plans_keep_fp32_workspace():
    tables = (
        _gfx950._wg_REGULAR_WAVE_GRID_SPECIALIZATIONS,
        _gfx950._wg_PRODUCTION_TRANSPOSED_WAVE_GRID_PROMOTIONS,
        _gfx950._wg_SOTA77_TRANSPOSED_WAVE_GRID_PROMOTIONS,
    )
    for table in tables:
        for plan in table.values():
            assert "workspace_dtype" not in plan


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mm_n192_uses_bk128_register_plan(dtype):
    torch.manual_seed(103)
    a = torch.randn((4096, 2048), device="cuda", dtype=dtype)
    b = torch.randn((192, 2048), device="cuda", dtype=dtype).T
    path, plan = _gfx950._optimized_geometry_config(
        4096,
        192,
        2048,
        dtype,
        2,
        a.stride(),
        b.stride(),
    )
    assert path == "register"
    assert plan["BLOCK_M"] == 64
    assert plan["BLOCK_N"] == 64
    assert plan["BLOCK_K"] == 128
    assert plan["num_stages"] == 3
    actual = tlx_ops.mm(a, b)
    _assert_range_result(actual, a @ b)


def test_mm_fp16_m256_n2048_uses_untransposed_register_tile():
    torch.manual_seed(107)
    a = torch.randn((256, 359), device="cuda", dtype=torch.float16)
    b = torch.randn((2048, 359), device="cuda", dtype=torch.float16).T
    path, plan = _gfx950._optimized_geometry_config(
        256,
        2048,
        359,
        torch.float16,
        2,
        a.stride(),
        b.stride(),
    )
    assert path == "register"
    assert plan["BLOCK_M"] == 64
    assert plan["BLOCK_N"] == 32
    assert plan["BLOCK_K"] == 64
    actual = tlx_ops.mm(a, b)
    _assert_range_result(actual, a @ b)


def test_mm_fp16_m1024_square_uses_bk128_register_plan():
    torch.manual_seed(113)
    a = torch.randn((1024, 864), device="cuda", dtype=torch.float16)
    b = torch.randn((1024, 864), device="cuda", dtype=torch.float16).T
    path, plan = _gfx950._optimized_geometry_config(
        1024,
        1024,
        864,
        torch.float16,
        2,
        a.stride(),
        b.stride(),
    )
    assert path == "register"
    assert plan["BLOCK_M"] == 64
    assert plan["BLOCK_N"] == 64
    assert plan["BLOCK_K"] == 128
    assert plan["NUM_XCDS"] == 1
    assert plan["num_stages"] == 3
    actual = tlx_ops.mm(a, b)
    _assert_range_result(actual, a @ b)


def test_mm_fp16_wide_output_k128_uses_three_stage_register_plan():
    path, plan = _gfx950._dispatch_plan(3072, 44536, 128, torch.float16, 2)
    assert path == "register"
    assert plan["BLOCK_M"] == 128
    assert plan["BLOCK_N"] == 256
    assert plan["BLOCK_K"] == 128
    assert plan["NUM_XCDS"] == 1
    assert plan["waves_per_eu"] == 0
    assert plan["num_stages"] == 3


def test_mm_optimized_geometry_cache_returns_fresh_plan():
    _gfx950._cached_optimized_geometry_config.cache_clear()
    first_path, first_plan = _gfx950._optimized_geometry_config(
        3072,
        2304,
        4608,
        torch.float16,
        2,
        (4608, 1),
        (1, 4608),
    )
    first_plan["workgroup_mapping"] = 999
    second_path, second_plan = _gfx950._optimized_geometry_config(
        3072,
        2304,
        4608,
        torch.float16,
        2,
        (4608, 1),
        (1, 4608),
    )
    assert first_path == second_path
    assert second_plan["workgroup_mapping"] != 999


def test_mm_fp16_m3072_n2048_uses_rectangular_wave_grid():
    torch.manual_seed(109)
    a = torch.randn((3072, 6144), device="cuda", dtype=torch.float16)
    b = torch.randn((2048, 6144), device="cuda", dtype=torch.float16).T
    path, plan = _gfx950._optimized_geometry_config(
        3072,
        2048,
        6144,
        torch.float16,
        2,
        a.stride(),
        b.stride(),
    )
    assert path == "wave_grid"
    assert plan["mi_wave_tile_m"] == 6
    assert plan["mi_wave_tile_n"] == 4
    assert plan["num_warps"] == 4
    assert plan["direct_to_lds"]
    assert plan["workgroup_mapping"] == 4
    assert plan["reverse_local_assignment"]
    assert plan["dot_mode"] == "logical"
    assert (3072, 2048, 6144) in _gfx950._PRODUCTION_COMPILED_WAVE_GRID_SHAPES
    actual = tlx_ops.mm(a, b)
    _assert_range_result(actual, a @ b)


def test_mm_fp16_addmm_derived_deep_k_shape():
    torch.manual_seed(83)
    a = torch.randn((1024, 6144), device="cuda", dtype=torch.float16)
    b = torch.randn((4096, 6144), device="cuda", dtype=torch.float16).T
    path, plan = _gfx950._dispatch_plan(1024, 4096, 6144, torch.float16, 2)
    assert path == "transposed_wave_grid"
    assert plan["reverse_local_assignment"]
    assert plan["workgroup_mapping"] == 4
    actual = tlx_ops.mm(a, b)
    _assert_range_result(actual, a @ b)


def test_mm_addmm_promotion_does_not_capture_row_major_b():
    tn_path, _ = _gfx950._optimized_geometry_config(
        61440,
        5120,
        2048,
        torch.float16,
        2,
        (2048, 1),
        (1, 2048),
    )
    row_major_path, row_major_plan = _gfx950._optimized_geometry_config(
        61440,
        5120,
        2048,
        torch.float16,
        2,
        (2048, 1),
        (5120, 1),
    )
    assert tn_path == "inter_wave"
    assert row_major_path == "wave_grid"
    assert row_major_plan["row_major_b_lds"]


def test_mm_fp16_addmm_derived_split_k_shape():
    torch.manual_seed(89)
    a = torch.randn((3072, 44536), device="cuda", dtype=torch.float16)
    b = torch.randn((5888, 44536), device="cuda", dtype=torch.float16).T
    path, plan = _gfx950._optimized_geometry_config(
        3072,
        5888,
        44536,
        torch.float16,
        2,
        a.stride(),
        b.stride(),
    )
    assert path == "transposed_wave_grid"
    assert plan["split_k"] == 4
    actual = tlx_ops.mm(a, b)
    _assert_range_result(actual, a @ b)


def test_mm_fp16_addmm_derived_pair_balanced_split_k_shape():
    torch.manual_seed(97)
    a = torch.randn((3072, 44536), device="cuda", dtype=torch.float16)
    b = torch.randn((128, 44536), device="cuda", dtype=torch.float16).T
    path, plan = _gfx950._optimized_geometry_config(
        3072,
        128,
        44536,
        torch.float16,
        2,
        a.stride(),
        b.stride(),
    )
    assert path == "transposed_wave_grid"
    assert plan["mi_wave_tile_n"] == 3
    assert plan["split_k"] == 8
    assert plan["pair_balanced_split_k"]
    actual = tlx_ops.mm(a, b)
    _assert_range_result(actual, a @ b)


def test_mm_fp16_addmm_derived_inter_wave_dispatch():
    path, plan = _gfx950._dispatch_plan(61440, 5120, 7744, torch.float16, 2)
    assert path == "inter_wave"
    assert plan["TILE"] == (256, 256)
    assert plan["SPLIT_K"] == 1
    assert plan["NUM_XCDS_OVERRIDE"] == 1
    assert plan["STORE_CACHE"] == ".cs"
    path, plan = _gfx950._dispatch_plan(61440, 5120, 2048, torch.float16, 2)
    assert path == "inter_wave"
    assert plan["TILE"] == (256, 256)
    assert plan["SPLIT_K"] == 1
    assert plan["STORE_CACHE"] == ".cs"
    for n in (19584, 39936, 65536):
        path, plan = _gfx950._optimized_geometry_config(
            3072,
            n,
            768,
            torch.float16,
            2,
            (768, 1),
            (1, 768),
        )
        assert path == "inter_wave"
        assert plan["TILE"] == (256, 256)
        assert plan["SPLIT_K"] == 1
        assert plan["STORE_CACHE"] == ".cs"
        assert plan["GROUP_M_OVERRIDE"] == (1 if n == 65536 else None)
    for n in (19584, 19968, 22240):
        path, plan = _gfx950._optimized_geometry_config(
            3072,
            n,
            512,
            torch.float16,
            2,
            (512, 1),
            (1, 512),
        )
        assert path == "inter_wave"
        assert plan["TILE"] == (256, 256)
        assert plan["SPLIT_K"] == 1
        assert plan["STORE_CACHE"] == ".cs"


def test_mm_bf16_wide_k768_uses_streaming_stores():
    for m, n in ((2048, 40192), (1024, 47232)):
        path, plan = _gfx950._optimized_geometry_config(
            m,
            n,
            768,
            torch.bfloat16,
            2,
            (768, 1),
            (1, 768),
        )
        assert path == "inter_wave"
        assert plan["TILE"] == (256, 256)
        assert plan["SPLIT_K"] == 1
        assert plan["STORE_CACHE"] == ".cs"
        assert plan.get("GROUP_M_OVERRIDE") == (6 if (m, n) == (2048, 40192) else None)


def test_mm_bf16_deep_square_uses_eight_wave_reducer():
    assert _gfx950._LDS_REDUCE_CONFIGS[(2048, 2048, 4, torch.bfloat16)] == (
        64,
        256,
        8,
    )
    assert (2048, 2048, 4, torch.float16) not in _gfx950._LDS_REDUCE_CONFIGS


def test_mm_bf16_addmm_derived_m2048_n9216_uses_transient_wgm4():
    path, plan = _gfx950._optimized_geometry_config(
        2048,
        9216,
        2304,
        torch.bfloat16,
        2,
        (2304, 1),
        (1, 2304),
    )
    assert path == "transposed_wave_grid"
    assert plan["workgroup_mapping"] == 4
    assert plan["dot_mode"] == "transient"

    torch.manual_seed(2048 + 9216 + 2304)
    a = torch.randn((2048, 2304), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((9216, 2304), device="cuda", dtype=torch.bfloat16).T
    _assert_range_result(tlx_ops.mm(a, b), a @ b)


def test_mm_bf16_addmm_derived_odd_k_uses_short_read_cover():
    path, plan = _gfx950._optimized_geometry_config(
        1024,
        4096,
        6145,
        torch.bfloat16,
        2,
        (6145, 1),
        (1, 6145),
    )
    assert path == "transposed_wave_grid"
    assert plan["read_cover_sixteenths"] == 4
    assert plan["dot_mode"] == "logical"
    assert plan["workgroup_mapping"] == 4

    torch.manual_seed(1024 + 4096 + 6145)
    a = torch.randn((1024, 6145), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((4096, 6145), device="cuda", dtype=torch.bfloat16).T
    _assert_range_result(tlx_ops.mm(a, b), a @ b)


def test_mm_fp16_addmm_derived_odd_k_uses_four_xcds():
    path, plan = _gfx950._optimized_geometry_config(
        3072,
        38848,
        607,
        torch.float16,
        2,
        (607, 1),
        (1, 607),
    )
    assert path == "transposed_wave_grid"
    assert plan["num_xcds"] == 4
    assert not plan["exact_lds"]
    assert plan["local_stages"] == 2


def test_mm_fp16_addmm_derived_short_k_uses_transposed_wave_grid():
    path, plan = _gfx950._optimized_geometry_config(
        64000,
        256,
        256,
        torch.float16,
        2,
        (256, 1),
        (1, 256),
    )
    assert path == "transposed_wave_grid"
    assert plan["workgroup_mapping"] == 4
    assert plan["matrix_instr_nonkdim"] == 16
    assert plan["dot_mode"] == "logical"


def test_mm_fp16_addmm_derived_m192n160_uses_exact_wave_grid():
    path, plan = _gfx950._optimized_geometry_config(
        3072,
        2304,
        4608,
        torch.float16,
        2,
        (4608, 1),
        (1, 4608),
    )
    assert path == "wave_grid"
    assert plan["mi_wave_tile_m"] == 5
    assert plan["mi_wave_tile_n"] == 6
    assert plan["warps_m"] == 2
    assert plan["warps_n"] == 2
    assert plan["workgroup_mapping"] == 8
    assert plan["direct_to_lds"]
    assert plan["local_stages"] == 2
    assert plan["dot_mode"] == "transient"
    assert (3072, 2304, 4608) in _gfx950._PRODUCTION_COMPILED_WAVE_GRID_SHAPES


def test_mm_fp16_addmm_derived_m160n192_uses_exact_wave_grid():
    path, plan = _gfx950._optimized_geometry_config(
        3072,
        2304,
        5120,
        torch.float16,
        2,
        (5120, 1),
        (1, 5120),
    )
    assert path == "wave_grid"
    assert plan["mi_wave_tile_m"] == 5
    assert plan["mi_wave_tile_n"] == 6
    assert plan["warps_m"] == 2
    assert plan["warps_n"] == 2
    assert plan["direct_to_lds"]
    assert plan["local_stages"] == 2
    assert plan["num_xcds"] == 8
    assert plan["workgroup_mapping"] == 8
    assert plan["dot_mode"] == "transient"
    assert (3072, 2304, 5120) in _gfx950._PRODUCTION_COMPILED_WAVE_GRID_SHAPES


def test_mm_fp16_addmm_derived_m128n256_uses_buffered_wave_grid():
    path, plan = _gfx950._optimized_geometry_config(
        4096,
        2048,
        6144,
        torch.float16,
        2,
        (6144, 1),
        (1, 6144),
    )
    assert path == "wave_grid"
    assert plan["mi_wave_tile_m"] == 4
    assert plan["mi_wave_tile_n"] == 8
    assert plan["warps_m"] == 2
    assert plan["warps_n"] == 2
    assert plan["local_stages"] == 2
    assert plan["pgr2_operands"]
    assert plan["pgr2_late_read_count"] == 4
    assert plan["workgroup_mapping"] == 2
    assert (4096, 2048, 6144) in _gfx950._PRODUCTION_COMPILED_WAVE_GRID_SHAPES


def test_mm_fp16_addmm_derived_k6640_uses_exact_wave_grid():
    path, plan = _gfx950._optimized_geometry_config(
        3072,
        2048,
        6640,
        torch.float16,
        2,
        (6640, 1),
        (1, 6640),
    )
    assert path == "wave_grid"
    assert plan["mi_wave_tile_m"] == 6
    assert plan["mi_wave_tile_n"] == 4
    assert plan["warps_m"] == 2
    assert plan["warps_n"] == 2
    assert plan["block_k"] == 64
    assert plan["exact_lds"]
    assert plan["pgr2_operands"]
    assert plan["dot_mode"] == "transient"
    assert (3072, 2048, 6640) in _gfx950._PRODUCTION_COMPILED_WAVE_GRID_SHAPES


def test_mm_fp16_n5888_uses_two_xcds():
    for k in (1536, 2304):
        path, plan = _gfx950._dispatch_plan(3072, 5888, k, torch.float16, 2)
        assert path == "transposed_wave_grid"
        assert plan["num_xcds"] == 2
        assert plan["dot_mode"] == "transient"


def test_mm_fp16_n1536_k5888_uses_combined_k128_pipeline():
    path, plan = _gfx950._dispatch_plan(3072, 1536, 5888, torch.float16, 2)
    assert path == "transposed_wave_grid"
    assert not plan["disable_unclustered_high_rp_reschedule"]
    assert plan["dot_mode"] == "persistent_6x3"


def test_mm_fp16_m3072_n4096_k2048_uses_two_column_mapping():
    path, plan = _gfx950._optimized_geometry_config(
        3072,
        4096,
        2048,
        torch.float16,
        2,
        (2048, 1),
        (1, 2048),
    )
    assert path == "inter_wave"
    assert plan["WORKGROUP_MAPPING"] == 2


def test_mm_fp16_m1024_n6144_k4096_uses_four_xcds():
    path, plan = _gfx950._dispatch_plan(1024, 6144, 4096, torch.float16, 2)
    assert path == "transposed_wave_grid"
    assert plan["num_xcds"] == 4
    assert plan["dot_mode"] == "transient"


def test_mm_fp16_m3072_n44032_k128_uses_group_one():
    path, plan = _gfx950._dispatch_plan(3072, 44032, 128, torch.float16, 2)
    assert path == "inter_wave"
    assert plan["GROUP_M_OVERRIDE"] == 1
    assert plan["NUM_XCDS_OVERRIDE"] == 2


def test_mm_fp16_addmm_derived_wide_n_uses_eight_xcds():
    path, plan = _gfx950._optimized_geometry_config(
        4096,
        242432,
        1894,
        torch.float16,
        2,
        (1894, 1),
        (1, 1894),
    )
    assert path == "transposed_wave_grid"
    assert plan["num_xcds"] == 8


def test_mm_fp16_addmm_derived_narrow_output_uses_two_waves():
    torch.manual_seed(97)
    storage = torch.randn((3072, 30720), device="cuda", dtype=torch.float16)
    a = storage[:, :1024]
    b = torch.randn((3, 1024), device="cuda", dtype=torch.float16).T
    path, plan = _gfx950._dispatch_for(a, b)
    assert path == "register"
    assert plan["num_warps"] == 2
    assert plan["GROUP_M"] == 32
    assert plan["NUM_XCDS"] == 1
    assert plan["CACHE_A_CG"]
    actual = tlx_ops.mm(a, b)
    _assert_range_result(actual, a @ b)


def test_mm_fp16_addmm_derived_n80_uses_fragmented_register():
    path, plan = _gfx950._dispatch_plan(4718592, 80, 80, torch.float16, 2)
    assert path == "fragmented_register"
    assert plan["M_HEAD"] == 64
    assert plan["M_TAIL"] == 16


def test_mm_fp16_addmm_derived_wide_output_uses_default_grid_mapping():
    path, plan = _gfx950._dispatch_plan(61440, 10000, 1024, torch.float16, 2)
    assert path == "inter_wave"
    assert plan["TILE"] == (256, 256)
    assert plan["SPLIT_K"] == 1
    assert "GROUP_M_OVERRIDE" not in plan
    assert "NUM_XCDS_OVERRIDE" not in plan
    assert plan["STORE_CACHE"] == ".cs"


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


@pytest.mark.parametrize("m", [1, 16, 27, 32])
@pytest.mark.parametrize("n", [1024, 8192])
@pytest.mark.parametrize("k", [4160, 7168, 8192])
def test_mm_small_m_deep_k_preserves_incumbent_dispatch(m, n, k):
    args = (m, n, k, torch.bfloat16, 2, (k, 1), (1, k))
    assert not _gfx950._range_family_candidates(*args)
    assert _gfx950.heuristic_config(*args) == _gfx950._incumbent_heuristic_config(*args)


@pytest.mark.parametrize("m,n,k", [
    (1, 1024, 512),
    (16, 8192, 512),
    (16, 8192, 4096),
    (27, 1536, 1024),
    (27, 8192, 512),
    (32, 4096, 4096),
])
def test_mm_small_m_generic_candidate_order(m, n, k):
    expected = (
        ("range_register", _gfx950._fixed_register_plan(16, 16, 256, 1, 4, 2, 3)),
        ("range_register", _gfx950._fixed_register_plan(32, 32, 256, 4, 8, 4, 3, waves_per_eu=2)),
    )
    assert _gfx950._range_family_candidates(m, n, k, torch.bfloat16, 2, (k, 1), (1, k)) == expected
    assert _gfx950.heuristic_config(m, n, k, torch.bfloat16, 2, (k, 1), (1, k)) == expected[0]


@pytest.mark.parametrize("path",
                         ["wave_grid", "transposed_wave_grid", "transposed_inter_wave", "lds", "split_register"])
@pytest.mark.parametrize("a_offset,b_offset", [(1, 0), (0, 1)])
def test_mm_range_selectors_skip_unaligned_candidates(path, a_offset, b_offset, monkeypatch):
    m, n, k = 16, 128, 512
    a = torch.empty(m * k + a_offset, device="cuda", dtype=torch.bfloat16)[a_offset:].view(m, k)
    b = torch.empty(n * k + b_offset, device="cuda", dtype=a.dtype)[b_offset:].view(n, k).T
    out = torch.empty((m, n), device="cuda", dtype=a.dtype)
    fallback = ("range_register", _gfx950._fixed_register_plan(16, 16, 256, 1, 4, 2, 3))
    plan = (128, 128, 1) if path == "lds" else {}
    candidates = ((path, plan), fallback)
    launches = []

    def launch(a, b, out, dispatch):
        launches.append(dispatch)
        return out

    def measure(fn, **kwargs):
        fn()
        return 1.0

    monkeypatch.setattr(_gfx950, "_range_dispatch_candidates", lambda *args, **kwargs: candidates)
    monkeypatch.setattr(_gfx950, "_can_use_range_register_compiled_cache", lambda: False)
    monkeypatch.setattr(_gfx950, "_launch_dispatch", launch)
    monkeypatch.setattr("triton.testing.do_bench", measure)
    assert _gfx950._range_dispatch_for(a, b) == fallback
    assert _gfx950._launch_range_autotuned(a, b, out, candidates) is out
    assert launches and all(dispatch == fallback for dispatch in launches)


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


@pytest.mark.parametrize("b_layout,launcher", [("row", "plan"), ("column", "plan"), ("row", "range")])
def test_mm_register_plan_int64_operand_offsets(b_layout, launcher):
    # Padded leading strides put row 127 of each operand past 2**31 elements, in
    # the full K tile and the K tail. The column case uses the dense TN launcher.
    m = n = k = 136
    stride = 2**31 // 127 + 1
    storage = torch.empty(135 * stride + 4 * k, device="cuda", dtype=torch.bfloat16)
    if b_layout == "row":
        a = storage.as_strided((m, k), (1, stride))
        b = storage.as_strided((k, n), (stride, 1), 2 * k)
    else:
        a = storage.as_strided((m, k), (stride, 1))
        b = storage.as_strided((k, n), (1, stride), 2 * k)
    torch.manual_seed(97)
    a_values = torch.randint(-2, 3, (m, k), device="cuda").to(torch.bfloat16)
    b_values = torch.randint(-2, 3, (k, n), device="cuda").to(torch.bfloat16)
    a.copy_(a_values)
    b.copy_(b_values)
    plan = _gfx950._fixed_register_plan(128, 64, 128, 4, 1, 8, 3, matrix_instr_nonkdim=32)
    launch = _gfx950._launch_register_plan if launcher == "plan" else _gfx950._launch_range_register_plan
    out = launch(a, b, config=plan)
    expected = (a_values.float() @ b_values.float()).to(torch.bfloat16)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


def test_mm_tuned_rectangular_register_plans():
    expected = {
        (677, 2048, 4096): (64, 128, 128, 2, 8, 8, 3, 32),
        (279, 4096, 4352): (128, 64, 128, 8, 8, 8, 3, 16),
        (4096, 4096, 2048): (256, 256, 64, 2, 1, 8, 2, 32),
        (61440, 3840, 4096): (128, 256, 32, 16, 8, 4, 2, 16),
        (61440, 5120, 2048): (128, 256, 32, 8, 8, 4, 2, 16),
        (61440, 5120, 7744): (128, 256, 32, 16, 8, 4, 2, 16),
        (3072, 40440, 256): (128, 256, 32, 32, 1, 4, 2, 16),
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
    torch.manual_seed(101)
    a = torch.randn((128, 20926), device="cuda", dtype=torch.float16)
    b = torch.randn((224, 20926), device="cuda", dtype=torch.float16).T
    actual = tlx_ops.mm(a, b)
    _assert_range_result(actual, a @ b)


def test_mm_production_large_split_k_has_fixed_workspace_type():
    path, plan = _gfx950._dispatch_plan(128, 10304, 20926, torch.float16, 2)
    assert path == "wave_grid"
    assert plan["split_k"] == 3
    assert plan["pair_balanced_split_k"]
    assert "workspace_dtype" not in plan
    assert plan["reverse_local_assignment"]


def test_mm_production_ranking_split_k_has_fixed_workspace_type():
    path, plan = _gfx950._dispatch_plan(128, 9216, 7392, torch.float16, 2)
    assert path == "wave_grid"
    assert plan["split_k"] == 3
    assert not plan["pair_balanced_split_k"]
    assert "workspace_dtype" not in plan
    assert plan["waves_per_eu"] == 1
    assert plan["dot_mode"] == "logical"
    assert plan["reverse_local_assignment"]


def test_mm_production_fp32_split_k_uses_narrow_wave_grid():
    path, plan = _gfx950._dispatch_plan(128, 2688, 4416, torch.float16, 2)
    assert path == "transposed_wave_grid"
    assert plan["split_k"] == 4
    assert plan["pair_balanced_split_k"]
    assert plan["reduce_tile"] == (16, 32)
    assert plan["reduce_warps"] == 2
    assert (
        plan["mi_wave_tile_m"],
        plan["mi_wave_tile_n"],
        plan["warps_m"],
        plan["warps_n"],
    ) == (3, 2, 2, 2)


@pytest.mark.parametrize("m", [196608, 245760, 4718592])
def test_mm_production_fragmented_register_plan(m):
    path, plan = _gfx950._dispatch_plan(m, 80, 80, torch.float16, 2)
    assert path == "fragmented_register"
    assert dict(plan) == {
        "M_HEAD": 64,
        "M_TAIL": 16,
        "BLOCK_N": 128,
        "BLOCK_K": 128,
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


def test_mm_bf16_production_launcher_preserves_codegen_options(monkeypatch):
    shape = (3840, 2304, 8192)
    plan = dict(_gfx950._BF16_WAVE_GRID_PROMOTIONS[shape], regclass_priority_trumps_globalness=False)
    monkeypatch.setitem(_gfx950._BF16_WAVE_GRID_PROMOTIONS, shape, plan)
    _gfx950._dispatch_plan.cache_clear()
    _gfx950._production_wave_grid_launcher.cache_clear()
    kernel = _gfx950._wg_kernel_regular_mi16_wave_grid
    warmup = kernel.warmup
    options = []

    def record_warmup(*args, **kwargs):
        options.append(kwargs["regclass_priority_trumps_globalness"])
        return warmup(*args, **kwargs)

    monkeypatch.setattr(kernel, "warmup", record_warmup)
    try:
        a = torch.randn((3840, 8192), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((2304, 8192), device="cuda", dtype=a.dtype).T
        out = torch.empty((3840, 2304), device="cuda", dtype=a.dtype)
        assert tlx_ops.mm(a, b, out=out, space="heuristic") is out
        expected = torch.matmul(a, b)
        torch.testing.assert_close(out, expected, atol=8e-3 * expected.abs().max().item(), rtol=8e-3)
        assert options == [False]
    finally:
        _gfx950._dispatch_plan.cache_clear()
        _gfx950._production_wave_grid_launcher.cache_clear()


@pytest.mark.parametrize("instrumentation", ["pre_run_hook", "always_compile"])
def test_mm_bf16_production_launcher_preserves_instrumentation(monkeypatch, instrumentation):
    _gfx950._production_wave_grid_launcher.cache_clear()
    try:
        a = torch.randn((3840, 8192), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((2304, 8192), device="cuda", dtype=a.dtype).T
        out = torch.empty((3840, 2304), device="cuda", dtype=a.dtype)
        assert tlx_ops.mm(a, b, out=out, space="heuristic") is out
        assert _gfx950._production_wave_grid_launcher.cache_info().currsize == 1
        kernel = _gfx950._wg_kernel_regular_mi16_wave_grid
        run = kernel.run
        calls = []
        hooks = []

        def record_run(*args, **kwargs):
            calls.append(True)
            return run(*args, **kwargs)

        monkeypatch.setattr(kernel, "run", record_run)
        if instrumentation == "pre_run_hook":
            monkeypatch.setattr(kernel, "pre_run_hooks", [lambda *args, **kwargs: hooks.append(True)])
        else:
            monkeypatch.setattr(triton.knobs.compilation, "always_compile", True)
        assert tlx_ops.mm(a, b, out=out, space="heuristic") is out
        assert calls == [True]
        if instrumentation == "pre_run_hook":
            assert hooks == [True]
        expected = torch.matmul(a, b)
        torch.testing.assert_close(out, expected, atol=8e-3 * expected.abs().max().item(), rtol=8e-3)
    finally:
        _gfx950._production_wave_grid_launcher.cache_clear()


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


def _bf16_projection_operands(m, n, k, pattern, *, seed, offset=0):
    torch.manual_seed(seed)
    a = torch.randn(m * k + offset, device="cuda", dtype=torch.bfloat16)[offset:].view(m, k)
    weight = torch.randn(n * k + offset, device="cuda", dtype=a.dtype)[offset:].view(n, k)
    if pattern == "cancellation":
        a.fill_(1)
        values = torch.tensor([16.0, 1 / 16, -16.0, 1 / 16], device="cuda", dtype=a.dtype).repeat(k // 4)
        weight.copy_(values.expand(n, k))
    return a, weight.T


@pytest.mark.parametrize("space", ["heuristic", "full"])
@pytest.mark.parametrize("n,k", [(2304, 8192), (4096, 8192), (8192, 4096)])
@pytest.mark.parametrize("pattern", ["random", "cancellation"])
@pytest.mark.parametrize("offset", [0, 1, 2])
def test_mm_bf16_m1_gemv_numerics_and_output_strides(monkeypatch, space, n, k, pattern, offset):
    a, b = _bf16_projection_operands(1, n, k, pattern, seed=83, offset=offset)
    backing = torch.full((1, n * 2), float("nan"), device="cuda", dtype=a.dtype)
    out = backing[:, ::2]
    calls = []

    def record_canonical_launch(*args, **kwargs):
        calls.append(("canonical", kwargs["num_warps"]))

    def record_packed_launch(*args, **kwargs):
        calls.append(("packed", kwargs["num_warps"]))

    monkeypatch.setattr(_gfx950._gemv_kernel, "pre_run_hooks", [record_canonical_launch])
    monkeypatch.setattr(_gfx950._gemv_packed_dot2_kernel, "pre_run_hooks", [record_packed_launch])
    assert tlx_ops.mm(a, b, out=out, space=space) is out
    expected = torch.matmul(a.float(), b.float()).to(a.dtype)
    torch.testing.assert_close(out, expected, atol=8e-3 * expected.abs().max().item(), rtol=8e-3)
    packed = k == 8192 and n in (2304, 4096) and offset != 1
    expected_warps = 4 if (n, k) == (8192, 4096) else 8
    expected_call = ("packed" if packed else "canonical", expected_warps)
    if space == "heuristic":
        assert calls == [expected_call]
    else:
        # Full search may benchmark GEMV repeatedly or reuse a non-GEMV winner.
        assert all(call == expected_call for call in calls)
    assert backing[:, 1::2].isnan().all()


@pytest.mark.parametrize("a_offset,b_offset", [(1, 0), (0, 1)])
@pytest.mark.parametrize("n", [2304, 4096])
def test_mm_bf16_m1_gemv_one_input_unaligned(monkeypatch, n, a_offset, b_offset):
    k = 8192
    torch.manual_seed(87)
    a = torch.randn(k + a_offset, device="cuda", dtype=torch.bfloat16)[a_offset:].view(1, k)
    weight = torch.randn(n * k + b_offset, device="cuda", dtype=a.dtype)[b_offset:].view(n, k)
    out = torch.empty((1, n), device="cuda", dtype=a.dtype)
    calls = []

    def record_canonical_launch(*args, **kwargs):
        calls.append(kwargs["num_warps"])

    def reject_packed_launch(*args, **kwargs):
        raise AssertionError("Packed GEMV requires both inputs to be 4-byte aligned")

    monkeypatch.setattr(_gfx950._gemv_kernel, "pre_run_hooks", [record_canonical_launch])
    monkeypatch.setattr(_gfx950._gemv_packed_dot2_kernel, "run", reject_packed_launch)
    monkeypatch.setattr(_gfx950._gemv_packed_dot2_kernel, "pre_run_hooks", [reject_packed_launch])
    assert tlx_ops.mm(a, weight.T, out=out, space="heuristic") is out
    expected = torch.matmul(a.float(), weight.T.float()).to(a.dtype)
    torch.testing.assert_close(out, expected, atol=8e-3 * expected.abs().max().item(), rtol=8e-3)
    assert calls == [8]


def _record_gemv_launches(monkeypatch):
    calls = []

    class Kernel:

        def __init__(self, kind):
            self.kind = kind

        def __getitem__(self, grid):
            return lambda *args, **kwargs: calls.append((self.kind, grid, kwargs))

    monkeypatch.setattr(_gfx950, "_gemv_kernel", Kernel("canonical"))
    monkeypatch.setattr(_gfx950, "_gemv_packed_dot2_kernel", Kernel("packed"))
    return calls


@pytest.mark.parametrize("packed", [False, True])
def test_mm_bf16_m1_gemv_rejects_insufficient_reduction_tile(monkeypatch, packed):
    n, k = 130, 257
    plan = {"BLOCK_N": 4, "BLOCK_K": 256, "num_warps": 8, "PACKED": packed}
    monkeypatch.setitem(_gfx950._GEMV_CONFIGS, (n, k), plan)
    calls = _record_gemv_launches(monkeypatch)
    a = torch.empty((1, k), device="cuda", dtype=torch.bfloat16)
    b = torch.empty((n, k), device="cuda", dtype=a.dtype).T
    out = torch.empty((1, n), device="cuda", dtype=a.dtype)
    _gfx950._range_family_candidates.cache_clear()
    try:
        with pytest.raises(AssertionError, match="GEMV BLOCK_K must cover K"):
            tlx_ops.mm(a, b, out=out, space="heuristic")
        assert calls == []
    finally:
        _gfx950._range_family_candidates.cache_clear()


@pytest.mark.parametrize("n,packed,block_n,warps,k,a_offset,kind", [
    (130, None, 4, 8, 256, 0, "canonical"),
    (134, False, 4, 8, 256, 0, "canonical"),
    (138, True, 2, 4, 256, 0, "packed"),
    (142, True, 2, 4, 256, 1, "canonical"),
    (146, True, 2, 4, 192, 0, "packed"),
    (150, True, 2, 4, 255, 0, "canonical"),
])
def test_mm_bf16_m1_gemv_uses_explicit_packed_plan(monkeypatch, n, packed, block_n, warps, k, a_offset, kind):
    plan = {"BLOCK_N": block_n, "BLOCK_K": 256, "num_warps": warps}
    if packed is not None:
        plan["PACKED"] = packed
    original_plan = dict(plan)
    monkeypatch.setitem(_gfx950._GEMV_CONFIGS, (n, k), plan)
    calls = _record_gemv_launches(monkeypatch)
    a = torch.empty(k + a_offset, device="cuda", dtype=torch.bfloat16)[a_offset:].view(1, k)
    b = torch.empty((n, k), device="cuda", dtype=a.dtype).T
    out = torch.empty((1, n), device="cuda", dtype=a.dtype)
    assert tlx_ops.mm(a, b, out=out, space="heuristic") is out
    assert len(calls) == 1
    actual_kind, grid, kwargs = calls[0]
    assert (actual_kind, grid, kwargs["num_warps"]) == (kind, (triton.cdiv(n, block_n), ), warps)
    assert "PACKED" not in kwargs
    assert kwargs["HAS_K_TAIL"] is (k < 256)
    assert kwargs["HAS_N_TAIL"] is (n % block_n != 0)
    assert kwargs["DENSE_B_K"] is (k == 256)
    assert plan == original_plan


def _assert_graph_replay_and_rebinding(m, n, k, *, seed, space="heuristic", padded=False, capture_calls=1):
    torch.manual_seed(seed)
    a_pad, b_pad, c_pad, offset = (1, 8, 8, 8) if padded else (0, 0, 0, 0)
    a_storage = torch.randn((m, k + a_pad), device="cuda", dtype=torch.bfloat16)
    w_storage = torch.randn((n, k + b_pad), device="cuda", dtype=a_storage.dtype)
    a, weight = a_storage[:, :k], w_storage[:, :k]
    pitch = 2 * n + c_pad
    backing = torch.full((m * pitch + 2 * offset, ), float("nan"), device="cuda", dtype=a.dtype)
    out = torch.as_strided(backing, (m, n), (pitch, 2), storage_offset=offset)

    def assert_gaps(buffer):
        body = buffer[offset:offset + m * pitch].view(m, pitch)
        assert body[:, 1:2 * n:2].isnan().all()
        if c_pad:
            assert body[:, 2 * n:].isnan().all()
        if offset:
            assert buffer[:offset].isnan().all() and buffer[-offset:].isnan().all()

    assert tlx_ops.mm(a, weight.T, out=out, space=space) is out
    _assert_runtime_cache_mm_result(out, a, weight.T)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(capture_calls):
            tlx_ops.mm(a, weight.T, out=out, space=space)
    original = out.clone()
    if padded:
        a.mul_(0.5)
    else:
        a.add_(0.25)
    weight.neg_()
    out.fill_(float("nan"))
    graph.replay()
    _assert_runtime_cache_mm_result(out, a, weight.T)
    assert not torch.equal(out, original)
    assert_gaps(backing)

    other_a = torch.full_like(a_storage, float("nan"))[:, :k]
    other_weight = torch.full_like(w_storage, float("nan"))[:, :k]
    other_a.copy_(a).mul_(0.5)
    other_weight.copy_(weight).mul_(0.25)
    other_backing = torch.full_like(backing, float("nan"))
    other_out = torch.as_strided(other_backing, (m, n), (pitch, 2), storage_offset=offset)
    assert tlx_ops.mm(other_a, other_weight.T, out=other_out, space=space) is other_out
    _assert_runtime_cache_mm_result(other_out, other_a, other_weight.T)
    assert_gaps(other_backing)


@pytest.mark.parametrize("n,k", [(2304, 8192), (4096, 8192), (8192, 4096)])
def test_mm_bf16_m1_gemv_graph_replay_and_rebinding(n, k):
    _assert_graph_replay_and_rebinding(1, n, k, seed=89)


@pytest.mark.parametrize("n,k", [(2304, 8192), (4096, 8192), (8192, 4096)])
def test_mm_bf16_m1_gemv_invariance_keeps_fixed_schedule(monkeypatch, n, k):
    from triton.tlx.ops import Space

    def reject_gemv(*args, **kwargs):
        raise AssertionError("GEMV must not serve a fixed reduction schedule")

    for kernel in (_gfx950._gemv_kernel, _gfx950._gemv_packed_dot2_kernel):
        monkeypatch.setattr(kernel, "run", reject_gemv)
        monkeypatch.setattr(kernel, "pre_run_hooks", [reject_gemv])
    a = torch.randn((2, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=a.dtype).T
    space = Space("heuristic", invariance=BitwiseInvarianceSpec.dims("M"))
    full = tlx_ops.mm(a, b, space=space)
    prefix = tlx_ops.mm(a[:1], b, space=space)
    assert torch.equal(prefix, full[:1])
    expected = torch.matmul(a, b)
    torch.testing.assert_close(full, expected, atol=8e-3 * expected.abs().max().item(), rtol=8e-3)


@pytest.mark.parametrize("layout", ["row_padding", "column_stride"])
def test_mm_regular_wave_grid_odd_mfma_count_numerics(layout):
    m, n, k = 27, 192, 2048
    torch.manual_seed(211)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((n, k), device="cuda", dtype=a.dtype)
    b = weight.T
    pitch = n + 1 if layout == "row_padding" else n * 2
    backing = torch.full((m, pitch), float("nan"), device="cuda", dtype=a.dtype)
    out = backing[:, :n] if layout == "row_padding" else backing[:, ::2]
    expected = torch.matmul(a.float(), b.float()).to(a.dtype)
    plan = _gfx950._wg_regular_wave_grid_plan(
        1,
        3,
        2,
        2,
        64,
        local_stages=2,
        pgr2_operands=True,
        direct_to_lds=True,
        direct_chunk=128,
        direct_b_cache_cg=True,
        dot_mode="logical",
        split_k=8,
    )
    assert _gfx950._launch_wave_grid(a, b, out=out, _candidate_plan=plan) is out
    tolerance = 8e-3
    torch.testing.assert_close(out, expected, rtol=tolerance, atol=tolerance * expected.abs().max().item())
    untouched = backing[:, n:] if layout == "row_padding" else backing[:, 1::2]
    assert untouched.isnan().all().item()


@pytest.mark.parametrize("space", ["heuristic", "full"])
@pytest.mark.parametrize("pattern", ["random", "cancellation"])
@pytest.mark.parametrize("m,n,k,offset,seed,expected_options", [
    (27, 2304, 8192, 1, 97, (32, 16, 512, True, 4, 8, 2, 3, False)),
    (27, 4096, 8192, 1, 97, (32, 16, 512, True, 4, 8, 2, 3, False)),
    (27, 8192, 4096, 1, 97, (32, 32, 256, True, 4, 8, 4, 3, False)),
    (27, 8192, 1024, 1, 97, (32, 32, 256, True, 4, 8, 4, 3, False)),
    (1, 8192, 1024, 0, 127, (16, 32, 256, True, 1, 4, 2, 3, True)),
    (1, 8192, 1024, 1, 127, (16, 32, 256, True, 1, 4, 2, 3, True)),
])
def test_mm_bf16_projection_numerics_and_launch_options(monkeypatch, space, pattern, m, n, k, offset, seed,
                                                        expected_options):
    a, b = _bf16_projection_operands(m, n, k, pattern, seed=seed, offset=offset)
    backing = torch.full((m, n * 2), float("nan"), device="cuda", dtype=a.dtype)
    out = backing[:, ::2]
    options = []

    def record_launch(*args, **kwargs):
        options.append((kwargs["BLOCK_M"], kwargs["BLOCK_N"], kwargs["BLOCK_K"], kwargs["CACHE_B_CG"],
                        kwargs["GROUP_M"], kwargs["NUM_XCDS"], kwargs["num_warps"], kwargs["num_stages"],
                        kwargs.get("regclass_priority_trumps_globalness", True)))

    if space == "heuristic":
        monkeypatch.setattr(_gfx950._register_kernel_impl, "pre_run_hooks", [record_launch])
    assert tlx_ops.mm(a, b, out=out, space=space) is out
    expected = torch.matmul(a, b)
    torch.testing.assert_close(out, expected, atol=8e-3 * expected.abs().max().item(), rtol=8e-3)
    if space == "heuristic":
        if m == 1:
            assert options == [expected_options]
        else:
            assert options and all(option == expected_options for option in options)
    assert backing[:, 1::2].isnan().all()


@pytest.mark.parametrize("space", ["heuristic", "full"])
@pytest.mark.parametrize("layout", ["contiguous", "row_pitch", "column_stride", "offset"])
@pytest.mark.parametrize("pattern", ["random", "cancellation"])
def test_mm_bf16_qkv_m27_split_register_strided_output(space, layout, pattern):
    m, n, k = 27, 2304, 8192
    a, b = _bf16_projection_operands(m, n, k, pattern, seed=151)
    pitch = n + 8 if layout == "row_pitch" else n * 2 if layout == "column_stride" else n
    offset = int(layout == "offset")
    backing = torch.full((m * pitch + offset + 1, ), float("nan"), device="cuda", dtype=a.dtype)
    out = torch.as_strided(backing, (m, n), (pitch, 2 if layout == "column_stride" else 1), offset)
    assert tlx_ops.mm(a, b, out=out, space=space) is out
    expected = torch.matmul(a.float(), b.float()).to(a.dtype)
    torch.testing.assert_close(out, expected, atol=8e-3 * expected.abs().max().item(), rtol=8e-3)
    path, plan = _gfx950._dispatch_for(a, b)
    assert path == "split_register" and plan["SPLIT_K"] == 8
    if layout == "row_pitch":
        assert backing[:m * pitch].view(m, pitch)[:, n:].isnan().all()
    elif layout == "column_stride":
        assert backing[:m * pitch].view(m, pitch)[:, 1::2].isnan().all()
    assert backing[-1].isnan()
    if offset:
        assert backing[0].isnan()


@pytest.mark.parametrize("m", [27, 59])
def test_mm_split_register_wrapped_input_vector_axes(m):
    n, k, split_k = 73, 2048, 8
    torch.manual_seed(203)
    a = torch.randn((k, m), device="cuda", dtype=torch.bfloat16).T
    b = torch.randn((k, n), device="cuda", dtype=a.dtype)
    partials = torch.empty((split_k, m, n), device="cuda", dtype=torch.float32)
    grid = (triton.cdiv(m, 32) * triton.cdiv(n, 64), split_k)
    _gfx950._split_register_kernel[grid](
        a,
        b,
        partials,
        m,
        n,
        k,
        *a.stride(),
        *b.stride(),
        BLOCK_M=32,
        BLOCK_N=64,
        BLOCK_K=64,
        SPLIT_K=split_k,
        VALID_M_ROWS=m if m <= 32 else 0,
        num_warps=4,
        num_stages=3,
    )
    for split in range(split_k):
        start, end = split * (k // split_k), (split + 1) * (k // split_k)
        expected = a[:, start:end].float() @ b[start:end].float()
        torch.testing.assert_close(partials[split], expected, atol=2e-4, rtol=2e-4)
    out = torch.empty((m, n), device="cuda", dtype=torch.float32)
    _gfx950._split_register_reduce_kernel[(triton.cdiv(m, 32), triton.cdiv(n, 32))](
        partials,
        out,
        m,
        n,
        *out.stride(),
        SPLIT_K=split_k,
        BLOCK_M=32,
        BLOCK_N=32,
        num_warps=4,
        num_stages=1,
    )
    torch.testing.assert_close(out, a.float() @ b.float(), atol=5e-4, rtol=2e-4)


def test_mm_split_register_generic_reducer_dense_and_pitched_reuse(monkeypatch):
    m, n, k = 27, 128, 2048
    torch.manual_seed(204)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=a.dtype).T
    padded = torch.full((m, n + 16), float("nan"), device="cuda", dtype=a.dtype)
    outputs = [torch.empty((m, n), device="cuda", dtype=a.dtype), padded[:, :n]]
    reducers = []
    kernel = _gfx950._split_register_reduce_kernel

    class ReducerRecorder:

        def __getitem__(self, grid):

            def launch(*args, **kwargs):
                compiled = kernel.run(*args, grid=grid, warmup=False, **kwargs)
                reducers.append(compiled)
                return compiled

            return launch

    monkeypatch.setattr(_gfx950, "_split_register_reduce_kernel", ReducerRecorder())
    expected = (a.float() @ b.float()).to(a.dtype)
    for out in outputs:
        assert _gfx950._launch_split_register(a, b, out, _gfx950._BF16_M27_QKV_SPLIT_REGISTER_PLAN) is out
        torch.testing.assert_close(out, expected, atol=8e-3 * expected.abs().max().item(), rtol=8e-3)
    assert len(reducers) == 2
    assert reducers[0] is reducers[1]
    assert reducers[0].hash == reducers[1].hash
    assert padded[:, n:].isnan().all()


@pytest.mark.parametrize("k", [1024, 2056, 2057])
def test_mm_split_register_reused_plan_fallback(k):
    m, n = 59, 192
    torch.manual_seed(205)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=a.dtype).T
    backing = torch.full((m, n + 16), float("nan"), device="cuda", dtype=a.dtype)
    out = backing[:, :n]
    assert _gfx950._launch_split_register(a, b, out, _gfx950._BF16_M27_QKV_SPLIT_REGISTER_PLAN) is out
    expected = (a.float() @ b.float()).to(a.dtype)
    torch.testing.assert_close(out, expected, atol=8e-3 * expected.abs().max().item(), rtol=8e-3)
    assert backing[:, n:].isnan().all()


@pytest.mark.parametrize("m,split_k,expected_path", [(27, 8, "split_register"), (1920, 2, "wave_grid")])
def test_mm_bf16_qkv_split_k_preserves_fp32_partials(m, split_k, expected_path):
    n, k = 2304, 8192
    a = torch.ones((m, k), device="cuda", dtype=torch.bfloat16)
    weight = torch.zeros((n, k), device="cuda", dtype=a.dtype)
    weight[:, 0] = 256
    weight[:, 1] = 0.125
    weight[:, k // split_k] = -256
    b = weight.T
    path, plan = _gfx950._dispatch_for(a, b)
    assert path == expected_path
    assert plan.get("SPLIT_K", plan.get("split_k", 1)) == split_k
    actual = tlx_ops.mm(a, b, space="heuristic")
    assert actual.dtype == torch.bfloat16
    assert torch.equal(actual, torch.full_like(actual, 0.125))


@pytest.mark.parametrize("m,n,k,path,step,input_offset", [
    (1, 2304, 8192, "gemv", 1, 0),
    (27, 2304, 8192, "split_register", 1, 0),
    (1920, 8192, 1024, "transposed_inter_wave", 1, 0),
    (1920, 8192, 1024, "transposed_inter_wave", 2, 0),
    (1920, 8192, 1024, "transposed_inter_wave", 1, 1),
])
def test_mm_bf16_promoted_dispatch_preserves_bias(m, n, k, path, step, input_offset):
    torch.manual_seed(211)
    a_storage = torch.zeros((m * k + input_offset, ), device="cuda", dtype=torch.bfloat16)
    aligned_a = a_storage[:m * k].view(m, k)
    a = a_storage[input_offset:].view(m, k)
    b = torch.ones((n, k), device="cuda", dtype=a.dtype).T
    dispatch = _gfx950._dispatch_for(aligned_a, b)
    assert dispatch[0] == path
    bias = torch.randn((m, n), device="cuda", dtype=a.dtype)
    backing = torch.full((m, n * step), float("nan"), device="cuda", dtype=a.dtype)
    out = backing[:, ::step]
    assert _gfx950._launch_dispatch(a, b, out, dispatch, bias=bias) is out
    assert torch.equal(out, bias)
    if step == 2:
        assert backing[:, 1::2].isnan().all()


def test_mm_bf16_dense_register_launcher_preserves_hook_after_cache_capture(monkeypatch):
    m, n, k = 27, 8192, 1024
    torch.manual_seed(137)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=a.dtype).T
    config = _gfx950._BF16_M27_REGISTER_CONFIGS[(n, k)]
    backing = torch.full((m, n * 2), float("nan"), device="cuda", dtype=a.dtype)
    out = backing[:, ::2]
    options = []

    def record_launch(*args, **kwargs):
        assert args[3] is out
        assert (args[15], args[16]) == out.stride()
        options.append((kwargs["BLOCK_M"], kwargs["BLOCK_N"], kwargs["BLOCK_K"], kwargs["CACHE_B_CG"]))

    _gfx950._dense_tn_register_launcher.cache_clear()
    try:
        assert _gfx950._launch_register_plan(a, b, config=config, out=out) is out
        assert _gfx950._dense_tn_register_launcher.cache_info().currsize == 1
        monkeypatch.setattr(_gfx950._register_kernel_impl, "pre_run_hooks", [record_launch])
        a.mul_(0.5)
        b.neg_()
        out.fill_(float("nan"))
        assert _gfx950._launch_register_plan(a, b, config=config, out=out) is out
        expected = torch.matmul(a, b)
        torch.testing.assert_close(out, expected, atol=8e-3 * expected.abs().max().item(), rtol=8e-3)
        assert options == [(32, 32, 256, True)]
        assert backing[:, 1::2].isnan().all()
    finally:
        _gfx950._dense_tn_register_launcher.cache_clear()


@pytest.mark.parametrize("m,n,k,reject_iw", [
    (27, 2304, 8192, False),
    (27, 4096, 8192, False),
    (27, 8192, 4096, False),
    (27, 8192, 1024, False),
    (1920, 2304, 8192, False),
    (3840, 2304, 8192, False),
    (3840, 4096, 8192, True),
    (1920, 8192, 4096, True),
    (3840, 8192, 4096, True),
])
def test_mm_bf16_projection_m_invariance(monkeypatch, m, n, k, reject_iw):
    if reject_iw:

        def reject_promotion(*args, **kwargs):
            raise AssertionError("The expert IW promotion must not serve a fixed reduction schedule")

        monkeypatch.setattr(_gfx950._iw_a16w16_8wave, "run", reject_promotion)
    run_mm_invariance_case("gfx950", m, n, k, (k, 1), (1, k), "bf16", BitwiseInvarianceSpec.dims("M"))


@pytest.mark.parametrize("space_name", ["heuristic", "full"])
def test_mm_bf16_attention_output_m1_invariance(space_name):
    from triton.tlx.ops import Space

    torch.manual_seed(129)
    a = torch.randn((2, 1024), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((8192, 1024), device="cuda", dtype=a.dtype).T
    space = Space(space_name, invariance=BitwiseInvarianceSpec.dims("M"))
    full = tlx_ops.mm(a, b, space=space)
    prefix = tlx_ops.mm(a[:1], b, space=space)
    assert torch.equal(prefix, full[:1])
    expected = torch.matmul(a, b)
    torch.testing.assert_close(full, expected, atol=8e-3 * expected.abs().max().item(), rtol=8e-3)


@pytest.mark.parametrize("padded_operand", ["a", "b"])
@pytest.mark.parametrize("m,n,k,padding", [
    (27, 2304, 8192, 8),
    (27, 4096, 8192, 8),
    (1, 8192, 1024, 1),
    (1, 8192, 1024, 8),
    (27, 8192, 1024, 1),
    (27, 8192, 1024, 8),
])
def test_mm_bf16_projection_padded_layout_fallback(monkeypatch, m, n, k, padding, padded_operand):
    if k == 1024:
        torch.manual_seed(131)
    a_padding = padding if padded_operand == "a" else 0
    b_padding = padding if padded_operand == "b" else 0
    a = torch.randn((m, k + a_padding), device="cuda", dtype=torch.bfloat16)[:, :k]
    b = torch.randn((n, k + b_padding), device="cuda", dtype=a.dtype)[:, :k].T
    options = []

    def record_launch(*args, **kwargs):
        options.append((kwargs["BLOCK_K"], kwargs.get("CACHE_B_CG", False)))

    monkeypatch.setattr(_gfx950._register_kernel_impl, "pre_run_hooks", [record_launch])
    actual = tlx_ops.mm(a, b, space="heuristic")
    expected = torch.matmul(a, b)
    torch.testing.assert_close(actual, expected, atol=8e-3 * expected.abs().max().item(), rtol=8e-3)
    if k == 8192:
        fallback = _gfx950.heuristic_config(m, n, k, a.dtype, a.element_size(), a.stride(), b.stride())
        assert fallback != ("range_register", _gfx950._BF16_M27_REGISTER_CONFIGS[(n, k)])
        assert fallback[0] != "split_register"
        assert all(block_k != 512 for block_k, _ in options)
    else:
        assert options and not any(cache_b_cg for _, cache_b_cg in options)


@pytest.mark.parametrize("m", [1920, 3840])
def test_mm_bf16_attention_transposed_route_scope(m):
    n, k = 8192, 1024
    path, plan = _gfx950.heuristic_config(m, n, k, torch.bfloat16, 2, (k, 1), (1, k))
    assert path == "transposed_inter_wave" and plan["SPLIT_K"] == 1
    for a_strides, b_strides in (((k + 8, 1), (1, k)), ((k, 1), (1, k + 8))):
        fallback = _gfx950.heuristic_config(m, n, k, torch.bfloat16, 2, a_strides, b_strides)
        assert fallback[0] != "transposed_inter_wave"
    for dim in ("M", "N", "K"):
        fixed = _gfx950.heuristic_config(m, n, k, torch.bfloat16, 2, (k, 1), (1, k),
                                         invariance=BitwiseInvarianceSpec.dims(dim))
        assert fixed[0] != "transposed_inter_wave"


@pytest.mark.parametrize("m,step,padding,offset", [(1920, 1, 0, 0), (3840, 1, 0, 0), (1920, 1, 1, 1), (3840, 2, 0, 0)])
def test_mm_bf16_attention_transposed_graph_contract(monkeypatch, m, step, padding, offset):
    n, k = 8192, 1024
    torch.manual_seed(163)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((n, k), device="cuda", dtype=a.dtype)
    pitch = n * step + padding
    storage = torch.full((offset + m * pitch + 8, ), float("nan"), device="cuda", dtype=a.dtype)
    out = torch.as_strided(storage, (m, n), (pitch, step), storage_offset=offset)
    options = []

    def record_launch(*args, **kwargs):
        assert "STATIC_SHAPE" not in kwargs
        options.append((args[5:9], kwargs["STORE_TRANSPOSED"], kwargs["CACHE_A_CG"], kwargs["SPLIT_K"]))

    monkeypatch.setattr(_gfx950._iw_a16w16_8wave, "pre_run_hooks", [record_launch])
    assert tlx_ops.mm(a, weight.T, out=out, space="heuristic") is out
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        tlx_ops.mm(a, weight.T, out=out, space="heuristic")
    a.mul_(0.5)
    weight.neg_()
    expected = torch.matmul(a.float(), weight.T.float()).to(a.dtype)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected, atol=8e-3 * expected.abs().max().item(), rtol=8e-3)
    geometry = (8192, m, k, k) if step == 1 else (m, n, k, k)
    assert options == [(geometry, step == 1, step == 1, 1)] * 2
    body = storage[offset:offset + m * pitch].view(m, pitch)
    if step == 2:
        assert body[:, 1:n * step:2].isnan().all()
    if padding:
        assert body[:, n * step:].isnan().all()
    assert storage[:offset].isnan().all() and storage[-8:].isnan().all()


_BF16_WG_LAYOUT_EXPECTED = {
    (3840, 2304):
    dict(MI_WAVE_TILE_M=6, MI_WAVE_TILE_N=6, DIRECT_CHUNK=128, REFILL_READ_WINDOW=0, DIRECT_A_CACHE_CG=True,
         DOT_MODE="logical", WIDE_EPILOGUE=True, READ_COVER_SIXTEENTHS=11, LOOP_UNROLL_FACTOR=2, SPLIT_K=1),
    (1920, 2304):
    dict(MI_WAVE_TILE_M=6, MI_WAVE_TILE_N=6, DIRECT_CHUNK=128, REFILL_READ_WINDOW=0, SPLIT_K=2),
    (1920, 4096):
    dict(MI_WAVE_TILE_M=8, MI_WAVE_TILE_N=4, WARPS_M=1, WARPS_N=4, BLOCK_K=64, NUM_XCDS=4, WORKGROUP_MAPPING=8,
         SPLIT_K=1, PGR2_OPERANDS=True, DIRECT_TO_LDS=True, LOCAL_STAGES=2, DIRECT_CHUNK=128, READ_COVER_SIXTEENTHS=12,
         LOOP_UNROLL_FACTOR=2, DOT_MODE="logical", DIRECT_B_CACHE_CG=True),
}


def _bf16_wg_layout_operands(m, n, seed, pattern, layout):
    k = 8192
    torch.manual_seed(seed)
    a_padding = 1 if layout in ("a_padding1", "input_padding") else 0
    b_padding = 1 if layout == "b_padding1" else 8 if layout == "input_padding" else 0
    offset = int(layout == "unaligned_inputs")
    a = torch.randn(m * (k + a_padding) + offset, device="cuda", dtype=torch.bfloat16)[offset:]
    b = torch.randn(n * (k + b_padding) + offset, device="cuda", dtype=a.dtype)[offset:]
    a, b = a.view(m, k + a_padding)[:, :k], b.view(n, k + b_padding)[:, :k].T
    if pattern == "scaled":
        a.mul_(0.125)
        b.mul_(32)
    elif pattern == "cancellation":
        a.fill_(1)
        values = torch.tensor([16.0, 1 / 16, -16.0, 1 / 16], device="cuda", dtype=a.dtype).repeat(k // 4)
        b.copy_(values[:, None].expand(k, n))
    return a, b


def _bf16_wg_layout_output(m, n, layout):
    step = 2 if layout == "stride2" else 1
    padding = 1 if layout == "row_padding1" else 8 if layout == "row_padding8" else 0
    offset = int(layout == "offset")
    pitch = n * step + padding
    backing = torch.full((offset + m * pitch + 8, ), float("nan"), device="cuda", dtype=torch.bfloat16)
    out = torch.as_strided(backing, (m, n), (pitch, step), storage_offset=offset)
    body = backing[offset:offset + m * pitch].view(m, pitch)
    gaps = (backing[:offset], backing[offset + m * pitch:])
    if step == 2:
        gaps += (body[:, 1:n * step:2], )
    if padding:
        gaps += (body[:, n * step:], )
    return out, gaps


@pytest.mark.parametrize("m,n,seed,space,pattern,layout", [
    *[(3840, 2304, 101, space, "random", layout)
      for space in ("heuristic", "full")
      for layout in ("contiguous", "offset", "stride2", "row_padding1", "row_padding8")],
    *[(1920, 2304, 107, space, "random", layout)
      for space, layout in (("heuristic", "stride2"), ("full", "stride2"), ("heuristic", "offset"), ("full", "offset"),
                            ("heuristic", "row_padding1"), ("full", "row_padding1"), ("heuristic", "row_padding8"),
                            ("full", "row_padding8"), ("heuristic", "a_padding1"), ("heuristic", "b_padding1"))],
    *[(1920, 4096, 181, space, pattern, layout)
      for space, pattern, layout in (("heuristic", "random", "contiguous"), ("full", "scaled", "contiguous"),
                                     ("heuristic", "cancellation", "contiguous"), ("full", "random", "row_padding8"),
                                     ("heuristic", "random", "stride2"), ("full", "random", "offset"),
                                     ("heuristic", "random", "input_padding"), ("full", "random", "unaligned_inputs"))],
])
def test_mm_bf16_wave_grid_projection_layouts(monkeypatch, m, n, seed, space, pattern, layout):
    k = 8192
    a, b = _bf16_wg_layout_operands(m, n, seed, pattern, layout)
    out, gaps = _bf16_wg_layout_output(m, n, layout)
    expected_options = _BF16_WG_LAYOUT_EXPECTED[(m, n)]
    launches, workspace_pointers, reductions = [], [], []
    generic = _gfx950._launch_wave_grid

    def record_launch(*args, **kwargs):
        assert args[4:7] == (m, n, k)
        if m == 3840:
            assert kwargs.get("regclass_priority_trumps_globalness", True) is True
        if n == 4096:
            assert args[3] is out and args[13:15] == out.stride()
        if expected_options["SPLIT_K"] == 2:
            workspace = args[3]
            assert workspace.dtype == torch.float32 and workspace.shape == (2 * m, n)
            assert workspace.stride() == (n, 1)
            workspace_pointers.append(workspace.data_ptr())
        launches.append({key: kwargs[key] for key in expected_options})

    def record_reduction(*args, **kwargs):
        assert args[0].data_ptr() == workspace_pointers[-1] and args[2] is out
        assert (args[7], args[8]) == out.stride()
        reductions.append((kwargs["SPLIT_K"], kwargs["BLOCK_SIZE_M"], kwargs["BLOCK_SIZE_N"], kwargs["num_warps"]))

    def record_generic(*args, **kwargs):
        launches.append("generic")
        return generic(*args, **kwargs)

    def reject_compiled(*args, **kwargs):
        raise AssertionError("Unaligned operands must use the generic wave-grid launcher")

    # Full search can try different plans and temporary reduction buffers.
    if space == "heuristic":
        if layout == "unaligned_inputs":
            monkeypatch.setattr(_gfx950, "_PRODUCTION_COMPILED_WAVE_GRID_SHAPES",
                                _gfx950._PRODUCTION_COMPILED_WAVE_GRID_SHAPES | {(m, n, k)})
            monkeypatch.setattr(_gfx950, "_production_wave_grid_launcher", reject_compiled)
            monkeypatch.setattr(_gfx950, "_launch_wave_grid", record_generic)
        else:
            monkeypatch.setattr(_gfx950._wg_kernel_regular_mi16_wave_grid, "pre_run_hooks", [record_launch])
        if expected_options["SPLIT_K"] == 2:
            monkeypatch.setattr(_gfx950._reduce_k_kernel, "pre_run_hooks", [record_reduction])
    assert tlx_ops.mm(a, b, out=out, space=space) is out
    expected = torch.matmul(a.float(), b.float()).to(a.dtype)
    torch.testing.assert_close(out, expected, atol=8e-3 * expected.abs().max().item(), rtol=8e-3)
    if space == "heuristic":
        assert launches == ["generic" if layout == "unaligned_inputs" else expected_options]
        if expected_options["SPLIT_K"] == 2:
            assert reductions == [(2, 32, 128, 4)]
    assert all(gap.isnan().all() for gap in gaps)


@pytest.mark.parametrize("padded_operand", ["a", "b"])
@pytest.mark.parametrize("padding", [1, 8])
def test_mm_bf16_qkv_m3840_wave_grid_input_padding(monkeypatch, padded_operand, padding):
    m, n, k = 3840, 2304, 8192
    torch.manual_seed(103)
    a_padding = padding if padded_operand == "a" else 0
    b_padding = padding if padded_operand == "b" else 0
    a = torch.randn((m, k + a_padding), device="cuda", dtype=torch.bfloat16)[:, :k]
    weight = torch.randn((n, k + b_padding), device="cuda", dtype=a.dtype)[:, :k]
    b = weight.T
    out = torch.full((m, n), float("nan"), device="cuda", dtype=a.dtype)
    options = []

    def record_launch(*args, **kwargs):
        options.append(
            (kwargs["MI_WAVE_TILE_M"], kwargs["MI_WAVE_TILE_N"], kwargs["DIRECT_CHUNK"], kwargs["REFILL_READ_WINDOW"],
             kwargs["DIRECT_A_CACHE_CG"], kwargs["DOT_MODE"], kwargs["WIDE_EPILOGUE"], kwargs["READ_COVER_SIXTEENTHS"],
             kwargs["LOOP_UNROLL_FACTOR"], kwargs["SPLIT_K"]))

    monkeypatch.setattr(_gfx950._wg_kernel_regular_mi16_wave_grid, "pre_run_hooks", [record_launch])
    assert tlx_ops.mm(a, b, out=out, space="heuristic") is out
    expected = torch.matmul(a.float(), b.float()).to(a.dtype)
    torch.testing.assert_close(out, expected, atol=8e-3 * expected.abs().max().item(), rtol=8e-3)
    assert options == [(6, 6, 128, 0, True, "logical", True, 11, 2, 1)]


@pytest.mark.parametrize("space", ["heuristic", "full"])
def test_mm_bf16_qkv_m3840_wave_grid_graph_replay_and_rebinding(monkeypatch, space):
    kernel = _gfx950._wg_kernel_regular_mi16_wave_grid
    warmup = kernel.warmup
    options = []

    def record_warmup(*args, **kwargs):
        options.append((kwargs["DIRECT_A_CACHE_CG"], kwargs["DOT_MODE"], kwargs["WIDE_EPILOGUE"],
                        kwargs["READ_COVER_SIXTEENTHS"], kwargs["LOOP_UNROLL_FACTOR"], kwargs["SPLIT_K"]))
        return warmup(*args, **kwargs)

    if space == "heuristic":
        monkeypatch.setattr(kernel, "warmup", record_warmup)
    _gfx950._production_wave_grid_launcher.cache_clear()
    try:
        _assert_graph_replay_and_rebinding(3840, 2304, 8192, seed=173, space=space, padded=True)
        if space == "heuristic":
            assert options == [(True, "logical", True, 11, 2, 1)]
    finally:
        _gfx950._production_wave_grid_launcher.cache_clear()


def test_mm_bf16_qkv_m1920_split_k_preserves_reducer_hook_after_cache_capture(monkeypatch):
    m, n, k = 1920, 2304, 8192
    torch.manual_seed(113)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((n, k), device="cuda", dtype=a.dtype)
    backing = torch.full((m, n * 2), float("nan"), device="cuda", dtype=a.dtype)
    out = backing[:, ::2]
    reductions = []

    def record_reduction(*args, **kwargs):
        assert args[0].dtype == torch.float32
        assert args[0].shape == (2 * m, n)
        assert args[2] is out
        reductions.append((kwargs["SPLIT_K"], kwargs["BLOCK_SIZE_M"], kwargs["BLOCK_SIZE_N"], kwargs["num_warps"]))

    _gfx950._production_wave_grid_launcher.cache_clear()
    try:
        assert tlx_ops.mm(a, weight.T, out=out, space="heuristic") is out
        assert _gfx950._production_wave_grid_launcher.cache_info().currsize == 1
        monkeypatch.setattr(_gfx950._reduce_k_kernel, "pre_run_hooks", [record_reduction])
        a.mul_(0.5)
        weight.neg_()
        out.fill_(float("nan"))
        assert tlx_ops.mm(a, weight.T, out=out, space="heuristic") is out
        expected = torch.matmul(a, weight.T)
        torch.testing.assert_close(out, expected, atol=8e-3 * expected.abs().max().item(), rtol=8e-3)
        assert reductions == [(2, 32, 128, 4)]
        assert backing[:, 1::2].isnan().all()
    finally:
        _gfx950._production_wave_grid_launcher.cache_clear()


def test_mm_bf16_qkv_m1920_split_k_graph_replay_and_rebinding(monkeypatch):
    m, n, k = 1920, 2304, 8192
    warmup = _gfx950._reduce_k_kernel.warmup
    reductions = []

    def record_warmup(*args, **kwargs):
        assert args[0].dtype == torch.float32
        assert args[0].shape == (2 * m, n)
        reductions.append((kwargs["SPLIT_K"], kwargs["BLOCK_SIZE_M"], kwargs["BLOCK_SIZE_N"], kwargs["num_warps"]))
        return warmup(*args, **kwargs)

    monkeypatch.setattr(_gfx950._reduce_k_kernel, "warmup", record_warmup)
    _gfx950._production_wave_grid_launcher.cache_clear()
    try:
        _assert_graph_replay_and_rebinding(m, n, k, seed=109)
        assert _gfx950._production_wave_grid_launcher.cache_info().currsize == 1
        assert reductions == [(2, 32, 128, 4)]
    finally:
        _gfx950._production_wave_grid_launcher.cache_clear()


def test_mm_bf16_pre_expert_wave_grid_promotion_scope():
    shape = (1920, 4096, 8192)
    promotion = _gfx950._BF16_WAVE_GRID_PROMOTIONS[shape]
    assert _gfx950._dispatch_plan(*shape, torch.bfloat16, 2) == ("wave_grid", promotion)
    assert (promotion["mi_wave_tile_m"], promotion["mi_wave_tile_n"], promotion["warps_m"], promotion["warps_n"],
            promotion["num_xcds"], promotion["workgroup_mapping"], promotion["split_k"]) == (8, 4, 1, 4, 4, 8, 1)
    for dims, dtype in [((1919, 4096, 8192), torch.bfloat16), ((1920, 4095, 8192), torch.bfloat16),
                        ((1920, 4096, 8191), torch.bfloat16), (shape, torch.float16)]:
        assert _gfx950._dispatch_plan(*dims, dtype, 2) != ("wave_grid", promotion)


def test_mm_bf16_pre_expert_wave_grid_graph_replay_and_rebinding():
    _gfx950._WAVE_GRID_COMPILED_CACHE.clear()
    try:
        _assert_graph_replay_and_rebinding(1920, 4096, 8192, seed=191, padded=True, capture_calls=32)
    finally:
        _gfx950._WAVE_GRID_COMPILED_CACHE.clear()


@pytest.mark.parametrize("dim", ["M", "N", "K"])
def test_mm_bf16_pre_expert_wave_grid_promotion_keeps_invariance(dim):
    spec = BitwiseInvarianceSpec.dims(dim)
    promotion = _gfx950._BF16_WAVE_GRID_PROMOTIONS[(1920, 4096, 8192)]
    fixed = _gfx950._dispatch_plan(1920, 4096, 8192, torch.bfloat16, 2, invariance=spec)
    assert fixed != ("wave_grid", promotion)
    run_mm_invariance_case("gfx950", 1920, 4096, 8192, (8192, 1), (1, 8192), "bf16", spec)


@pytest.mark.parametrize("space", ["heuristic", "full"])
@pytest.mark.parametrize("m,n,k,group_m,num_xcds,store_cache", [
    (3840, 4096, 8192, 4, 8, ""),
    (1920, 8192, 4096, 2, 1, ".cs"),
    (3840, 8192, 4096, 2, 1, ".cs"),
])
def test_mm_bf16_expert_iw_numerics_and_strided_output(monkeypatch, space, m, n, k, group_m, num_xcds, store_cache):
    torch.manual_seed(139)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((n, k), device="cuda", dtype=a.dtype)
    backing = torch.full((m, 2 * n), float("nan"), device=a.device, dtype=a.dtype)
    out = backing[:, ::2]
    launches = []

    def record_launch(*args, **kwargs):
        launches.append((kwargs["SPLIT_K"], kwargs["_iw_BLOCK_M"], kwargs["_iw_BLOCK_N"], kwargs["_iw_GROUP_SIZE_M"],
                         kwargs["_iw_NUM_XCDS"], kwargs["STORE_CACHE"]))

    if space == "heuristic":
        monkeypatch.setattr(_gfx950._iw_a16w16_8wave, "pre_run_hooks", [record_launch])
    assert tlx_ops.mm(a, weight.T, out=out, space=space) is out
    expected = torch.matmul(a, weight.T)
    torch.testing.assert_close(out, expected, atol=8e-3 * expected.abs().max().item(), rtol=8e-3)
    if space == "heuristic":
        assert launches
        assert all(launch == (1, 256, 256, group_m, num_xcds, store_cache) for launch in launches)
    assert backing[:, 1::2].isnan().all()


@pytest.mark.parametrize("space", ["heuristic", "full"])
@pytest.mark.parametrize("m,n,k", [(3840, 4096, 8192), (1920, 8192, 4096), (3840, 8192, 4096)])
@pytest.mark.parametrize("a_offset,b_offset,padding", [(1, 0, 0), (0, 1, 0), (0, 0, 1)])
def test_mm_bf16_expert_iw_alignment_and_pitched_storage(space, m, n, k, a_offset, b_offset, padding):
    torch.manual_seed(149)
    a_storage = torch.randn(m * (k + padding) + a_offset, device="cuda", dtype=torch.bfloat16)
    weight_storage = torch.randn(n * (k + padding) + b_offset, device="cuda", dtype=a_storage.dtype)
    a = a_storage[a_offset:].view(m, k + padding)[:, :k]
    weight = weight_storage[b_offset:].view(n, k + padding)[:, :k]
    backing = torch.full((m, 2 * n + 1), float("nan"), device=a.device, dtype=a.dtype)
    out = backing[:, 1::2]

    assert tlx_ops.mm(a, weight.T, out=out, space=space) is out
    expected = torch.matmul(a, weight.T)
    torch.testing.assert_close(out, expected, atol=8e-3 * expected.abs().max().item(), rtol=8e-3)
    assert backing[:, ::2].isnan().all()


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
        (1024, 6144, 4096, "transposed_wave_grid"),
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
    # The full space includes the heuristic plan, so remove it to reach the register fallback.
    for name in ("_optimized_dispatch_for", "_dispatch_for", "_dispatch_plan", "heuristic_config"):
        monkeypatch.setattr(_gfx950, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(_gfx950, "_range_dispatch_candidates", lambda *args, **kwargs: [])
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
        (torch.bfloat16, (1024, 6144, 51584)),
        (torch.float16, (1024, 6144, 20480)),
        (torch.bfloat16, (1024, 6144, 8192)),
    ]:
        m, n, k = shape
        a = torch.empty((m, k), device="meta", dtype=dtype)
        b = torch.empty((k, n), device="meta", dtype=dtype)
        _gfx950._launch_lds(a, b, SPLIT_K=1, TILE=(256, 256))

    assert launches == [False, False, True, True]


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

_SMALL_M_INVARIANCE_FALLBACK_SHAPES = [
    (10304, 20926),
    (224, 20926),
    (80, 80),
    (56768, 64),
    (65462, 96),
]


@pytest.mark.parametrize("n,k", _SMALL_M_INVARIANCE_FALLBACK_SHAPES)
def test_mm_small_m_invariance_fallback_has_one_schedule(n, k):
    invariance = BitwiseInvarianceSpec.dims("M")
    plans = [
        _gfx950.heuristic_config(m, n, k, torch.float16, 2, (k, 1), (1, k), invariance=invariance)
        for m in range(1, 17)
    ]
    assert plans[0][0] == "register"
    assert all(plan == plans[0] for plan in plans)


def test_mm_small_m_invariance_fallback_is_bitwise():
    m, n, k = 16, 56768, 64
    torch.manual_seed(91)
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((n, k), device="cuda", dtype=torch.float16).T
    space = Space("heuristic", invariance=BitwiseInvarianceSpec.dims("M"))
    full = tlx_ops.mm(a, b, space=space)
    for prefix in range(1, m):
        actual = tlx_ops.mm(a[:prefix], b, space=space)
        assert torch.equal(actual.view(torch.int16), full[:prefix].view(torch.int16)), prefix


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


@pytest.mark.parametrize("shape", FOCUS.shapes("gfx950"))
def test_mm_invariance_focus_shapes_have_small_m_dispatch(shape):
    invariance = BitwiseInvarianceSpec.dims("M")
    dtype = torch.float16 if shape.dtype == "fp16" else torch.bfloat16
    for m in range(1, 17):
        dispatch = _gfx950.heuristic_config(
            m,
            shape.n,
            shape.k,
            dtype,
            2,
            shape.a_strides,
            shape.b_strides,
            invariance=invariance,
        )
        _assert_sequential_dispatch(dispatch)


def test_mm_focus_shapes_do_not_use_small_m_invariance_fallback(monkeypatch):

    def fail(*_args):
        raise AssertionError("normal dispatch reached the invariance-only fallback")

    monkeypatch.setattr(_gfx950, "_small_m_invariance_register_config", fail)
    for shape in FOCUS.shapes("gfx950"):
        dtype = torch.float16 if shape.dtype == "fp16" else torch.bfloat16
        _gfx950.heuristic_config(
            shape.m,
            shape.n,
            shape.k,
            dtype,
            2,
            shape.a_strides,
            shape.b_strides,
        )


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


def _runtime_cache_output(m, n, step, dtype):
    backing = torch.full((m, n * step + 16), float("nan"), device="cuda", dtype=dtype)
    return backing, backing[:, :n * step:step]


def _assert_runtime_cache_output_gaps(backing, n, step):
    for gap in range(1, step):
        assert backing[:, gap:n * step:step].isnan().all()
    assert backing[:, n * step:].isnan().all()


def _assert_runtime_cache_reused(compiled):
    assert len(compiled) == 2
    assert compiled[0].hash == compiled[1].hash
    assert compiled[0] is compiled[1]


def _assert_runtime_cache_mm_result(actual, a, b):
    expected = (a.float() @ b.float()).to(actual.dtype)
    torch.testing.assert_close(actual, expected, atol=8e-3 * expected.abs().max().item(), rtol=8e-3)


@pytest.mark.parametrize("packed", [False, True])
def test_mm_gemv_runtime_shape_and_strides_reuse_compilation(packed):
    torch.manual_seed(180 if packed else 179)
    kernel = _gfx950._gemv_packed_dot2_kernel if packed else _gfx950._gemv_kernel
    compiled = []
    for n, k, step in ((64, 2048, 2), (80, 4096, 3)):
        if packed:
            a = torch.randn((1, k), device="cuda", dtype=torch.bfloat16)
            b = torch.randn((n, k), device="cuda", dtype=a.dtype).T
        else:
            a = torch.randn((1, k * step), device="cuda", dtype=torch.bfloat16)[:, ::step]
            b = torch.randn((n, k * step + 16), device="cuda", dtype=a.dtype)[:, :k * step:step].T
        backing, out = _runtime_cache_output(1, n, step, a.dtype)
        strides = (b.stride(1), out.stride(1)) if packed else (a.stride(1), *b.stride(), out.stride(1))
        compiled.append(
            kernel.run(
                a,
                b,
                out,
                n,
                k,
                *strides,
                BLOCK_N=8,
                BLOCK_K=4096,
                CACHE_B_CG=True,
                num_warps=4,
                num_stages=1,
                grid=(triton.cdiv(n, 8), ),
                warmup=False,
            ))
        _assert_runtime_cache_mm_result(out, a, b)
        _assert_runtime_cache_output_gaps(backing, n, step)
    _assert_runtime_cache_reused(compiled)


def test_mm_split_register_runtime_k_and_strides_reuse_compilation():
    torch.manual_seed(181)
    split_k = 8
    producers, reducers = [], []
    for m, n, k, step in ((27, 128, 2048, 2), (59, 192, 4096, 3)):
        a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((n, k), device="cuda", dtype=a.dtype).T
        partials = torch.empty((split_k, m, n), device="cuda", dtype=torch.float32)
        backing, out = _runtime_cache_output(m, n, step, a.dtype)
        producers.append(
            _gfx950._split_register_kernel.run(
                a,
                b,
                partials,
                m,
                n,
                k,
                *a.stride(),
                *b.stride(),
                BLOCK_M=32,
                BLOCK_N=64,
                BLOCK_K=64,
                SPLIT_K=split_k,
                num_warps=4,
                num_stages=3,
                grid=(triton.cdiv(m, 32) * triton.cdiv(n, 64), split_k),
                warmup=False,
            ))
        reducers.append(
            _gfx950._split_register_reduce_kernel.run(
                partials,
                out,
                m,
                n,
                *out.stride(),
                SPLIT_K=split_k,
                BLOCK_M=32,
                BLOCK_N=32,
                num_warps=4,
                num_stages=1,
                grid=(triton.cdiv(m, 32), triton.cdiv(n, 32)),
                warmup=False,
            ))
        _assert_runtime_cache_mm_result(out, a, b)
        _assert_runtime_cache_output_gaps(backing, n, step)
    _assert_runtime_cache_reused(producers)
    _assert_runtime_cache_reused(reducers)


def test_mm_split_register_reducer_runtime_shape_and_strides_reuse_compilation():
    torch.manual_seed(191)
    compiled = []
    for m, n, step in ((16, 48, 2), (32, 80, 3)):
        partials = torch.randn((8, m, n), device="cuda", dtype=torch.float32)
        backing, out = _runtime_cache_output(m, n, step, partials.dtype)
        compiled.append(
            _gfx950._split_register_reduce_kernel.run(
                partials,
                out,
                m,
                n,
                *out.stride(),
                SPLIT_K=8,
                BLOCK_M=32,
                BLOCK_N=32,
                num_warps=4,
                num_stages=1,
                grid=(triton.cdiv(m, 32), triton.cdiv(n, 32)),
                warmup=False,
            ))
        torch.testing.assert_close(out, partials.sum(dim=0), atol=2e-6, rtol=2e-6)
        _assert_runtime_cache_output_gaps(backing, n, step)
    _assert_runtime_cache_reused(compiled)


@pytest.mark.parametrize("k_values,pipe_tiles", [((512, 2048), 0), ((1040, 1120), 0), ((1024, 1024), 0)])
def test_mm_iw_runtime_k_and_strides_reuse_compilation(monkeypatch, k_values, pipe_tiles):
    torch.manual_seed(193)
    compiled = []
    original_run = _gfx950._iw_a16w16_8wave.run

    def record_launch(*args, **kwargs):
        assert kwargs["PIPE_TILES"] == pipe_tiles
        kernel = original_run(*args, **kwargs)
        compiled.append(kernel)
        return kernel

    monkeypatch.setattr(_gfx950._iw_a16w16_8wave, "run", record_launch)
    for k, step in zip(k_values, (2, 3)):
        a = torch.randn((256, k + 16), device="cuda", dtype=torch.bfloat16)[:, :k]
        b = torch.randn((256, k + 32), device="cuda", dtype=a.dtype)[:, :k].T
        backing, out = _runtime_cache_output(256, 256, step, a.dtype)
        assert _gfx950._iw_launch(a, b, out=out, SPLIT_K=1, TILE=(256, 256), GROUP_M_OVERRIDE=1,
                                  NUM_XCDS_OVERRIDE=1) is out
        _assert_runtime_cache_mm_result(out, a, b)
        _assert_runtime_cache_output_gaps(backing, 256, step)
    _assert_runtime_cache_reused(compiled)
