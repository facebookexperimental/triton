"""gfx950 correctness coverage for ``tlx.ops.mm``."""

import time

import pytest
import torch
from triton._internal_testing import is_hip_cdna4
from triton.tlx.ops import InvalidInput, UnsupportedOp
from triton.tlx.ops.kernels.mm._shapes import CORRECTNESS_SHAPES, operand
from triton.tlx.ops.kernels.mm import gfx950 as _gfx950

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
    tolerance = 1e-2 if dtype == torch.bfloat16 else 1e-3
    torch.testing.assert_close(
        out,
        expected,
        atol=tolerance * expected.abs().max().item(),
        rtol=tolerance,
    )


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
