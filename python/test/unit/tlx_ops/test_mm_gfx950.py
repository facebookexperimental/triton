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
FAILED_SHAPES = set()


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


@pytest.mark.parametrize("m", [256, 257])
def test_mm_small_square_register_plan(m):
    from triton.tlx.ops import mm as tlx_mm

    n, k = 257, 4096
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((n, k), device="cuda", dtype=torch.float16).T
    out = tlx_mm(a, b, space="heuristic")
    expected = torch.matmul(a, b)
    torch.testing.assert_close(
        out,
        expected,
        atol=1e-3 * expected.abs().max().item(),
        rtol=1e-3,
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
    from triton.tlx.ops.kernels.mm.gfx950 import mm

    a = torch.randn((7, 2048), device="cuda", dtype=torch.float16)
    b = torch.randn((8192, 2048), device="cuda", dtype=torch.float16).T
    expected = torch.empty((7, 8192), device="cuda", dtype=torch.float16)

    def launch_register(actual_a, actual_b, *, out):
        assert actual_a is a
        assert actual_b is b
        assert out.shape == expected.shape
        return expected

    monkeypatch.setattr(_gfx950, "_launch_register", launch_register)
    assert mm(a, b, space="full") is expected


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
    from triton.tlx.ops.kernels.mm.gfx950 import matmul

    m, n, k = 2048, 512, 2048
    assert _gfx950._dispatch_plan(m, n, k, torch.float16, 2) == (
        "lds",
        (128, 128, 2),
    )
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((n, k), device="cuda", dtype=torch.float16).T
    out = torch.empty((n, m), device="cuda", dtype=torch.float16).T

    actual = matmul(a, b, out=out)
    expected = torch.matmul(a, b)

    assert actual is out
    assert out.stride() == (1, m)
    torch.testing.assert_close(
        actual,
        expected,
        atol=1e-3 * expected.abs().max().item(),
        rtol=1e-3,
    )


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


def test_mm_irregular_shape_policy():
    assert _gfx950._lds_plan_for_shape(677, 4096, 8192) == (256, 256, 4)
    assert _gfx950._strong_lds_plan(677, 4096, 8192) == (192, 256, 4)
    common_square = _gfx950._register_plan_for_shape(
        2041, 2041, 2048, torch.bfloat16
    )
    assert (
        common_square["BLOCK_M"],
        common_square["BLOCK_N"],
        common_square["BLOCK_K"],
        common_square["GROUP_M"],
        common_square["NUM_XCDS"],
        common_square["num_warps"],
    ) == (128, 128, 128, 16, 8, 8)
    common_narrow = _gfx950._register_plan_for_shape(
        2048, 256, 1024
    )
    assert (
        common_narrow["BLOCK_M"],
        common_narrow["BLOCK_N"],
        common_narrow["BLOCK_K"],
        common_narrow["GROUP_M"],
        common_narrow["NUM_XCDS"],
        common_narrow["num_warps"],
    ) == (64, 64, 256, 4, 8, 8)
    for m in (256, 257):
        small_square = _gfx950._register_plan_for_shape(
            m, 257, 4096, torch.float16
        )
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
        plan = _gfx950._register_plan_for_shape(
            *shape, dtype=torch.float16
        )
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
    assert _gfx950._register_plan_for_shape(
        272, 3072, 4608, torch.float16
    )["BLOCK_M"] == 64

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
    from triton.tlx.ops.kernels.mm.gfx950 import matmul

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
        matmul(a, b, out=out)


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
