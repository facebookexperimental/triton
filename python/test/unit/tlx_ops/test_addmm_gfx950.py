import pytest
import torch

from triton._internal_testing import is_hip_cdna4
from triton.tlx.ops.kernels.addmm._shapes import CORRECTNESS_SHAPES, inputs

pytestmark = pytest.mark.skipif(not is_hip_cdna4(), reason="tlx.ops.addmm requires gfx950")


@pytest.mark.parametrize("m,n,k,a_strides,b_strides,bias_strides,dtype_name", CORRECTNESS_SHAPES)
def test_addmm(m, n, k, a_strides, b_strides, bias_strides, dtype_name):
    from triton.tlx.ops import addmm as tlx_addmm

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    torch.manual_seed(m + n + k)
    entry = (m, n, k, a_strides, b_strides, bias_strides, dtype_name)
    bias, a, b = inputs(entry, dtype)
    out = torch.empty((m, n), device="cuda", dtype=dtype)
    actual = tlx_addmm(bias, a, b, out=out)
    expected = torch.addmm(bias, a, b)

    assert actual is out
    precision = {torch.float16: 1e-3, torch.bfloat16: 8e-3}[dtype]
    atol = max(0.05, precision * expected.abs().max().item())
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0.05)

    del bias, a, b, out, actual, expected
    torch.cuda.empty_cache()


@pytest.mark.parametrize("bias_shape", [(37, ), (1, 37), (23, 1), (23, 37)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_generic_addmm_broadcast(bias_shape, dtype):
    from triton.tlx.ops import addmm as tlx_addmm

    torch.manual_seed(0)
    a = torch.randn((23, 19), device="cuda", dtype=dtype)
    b = torch.randn((19, 37), device="cuda", dtype=dtype)
    bias = torch.randn(bias_shape, device="cuda", dtype=dtype)
    actual = tlx_addmm(bias, a, b, space="heuristic")
    expected = torch.addmm(bias, a, b)
    torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.05)


@pytest.mark.parametrize(
    "expected_path,m,n,k,dtype,b_row_major",
    [
        ("cached_b_lds", 819200, 1024, 192, torch.bfloat16, False),
        ("short_k_register", 512, 512, 64, torch.bfloat16, False),
        ("row_major_direct", 2252800, 256, 256, torch.float16, True),
        ("hybrid_n160", 208, 106496, 6144, torch.float16, False),
        ("streamk", 4096, 4352, 8192, torch.float16, False),
        ("ragged_n", 4096, 320, 4096, torch.float16, False),
        ("m192n256", 800, 76800, 512, torch.float16, False),
        ("transposed_wave_grid", 128, 70770, 192, torch.float16, False),
        ("transposed_register", 128, 1536, 2688, torch.float16, False),
        ("fragmented_register", 196608, 80, 80, torch.float16, False),
        ("wave_grid", 128, 9216, 7392, torch.float16, False),
        ("transposed_wave_grid", 1024, 6144, 4096, torch.float16, False),
        ("range_register", 24, 1024, 768, torch.bfloat16, False),
        ("inter_wave", 4096, 4096, 2048, torch.float16, False),
        ("lds", 1024, 6144, 20480, torch.bfloat16, False),
        ("register", 4096, 192, 2048, torch.float16, False),
        ("local_split_u", 7, 8192, 2048, torch.float16, False),
    ],
)
def test_mm_family_fused_bias(expected_path, m, n, k, dtype, b_row_major):
    from triton.tlx.ops.kernels.mm import gfx950 as mm_gfx950

    torch.manual_seed(m + n + k)
    a = torch.randn((m, k), device="cuda", dtype=dtype)
    if b_row_major:
        b = torch.randn((k, n), device="cuda", dtype=dtype)
    else:
        b = torch.randn((n, k), device="cuda", dtype=dtype).T
    bias = torch.randn((n, ), device="cuda", dtype=dtype)
    if expected_path == "row_major_direct":
        dispatch = expected_path, mm_gfx950._ROW_MAJOR_DIRECT_PLANS[(m, n, k)]
    elif expected_path == "ragged_n":
        dispatch = expected_path, None
    else:
        dispatch = mm_gfx950._dispatch_for(a, b)
        assert dispatch is not None and dispatch[0] == expected_path

    actual = torch.empty((m, n), device="cuda", dtype=dtype)
    mm_gfx950._launch_dispatch(a, b, actual, dispatch, bias=bias.reshape(1, n).expand(m, n))
    expected = torch.addmm(bias, a, b)
    precision = 1e-3 if dtype == torch.float16 else 8e-3
    atol = max(0.05, precision * expected.abs().max().item())
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0.05)


def test_addmm_m4096_uses_fused_four_epilogue(monkeypatch):
    from triton.tlx.ops import addmm as tlx_addmm
    from triton.tlx.ops.kernels.mm import gfx950 as mm_gfx950

    captured = {}
    real_launch = mm_gfx950._launch_dispatch

    def spy(*args, **kwargs):
        captured["dispatch"] = args[3] if len(args) > 3 else kwargs["dispatch"]
        return real_launch(*args, **kwargs)

    monkeypatch.setattr(mm_gfx950, "_launch_dispatch", spy)
    torch.manual_seed(4096 + 2048 + 4096)
    bias = torch.randn((2048, ), device="cuda", dtype=torch.float16)
    a = torch.randn((4096, 4096), device="cuda", dtype=torch.float16)
    b = torch.randn((2048, 4096), device="cuda", dtype=torch.float16).T
    out = torch.empty((4096, 2048), device="cuda", dtype=torch.float16)
    actual = tlx_addmm(bias, a, b, out=out)
    assert actual is out
    assert captured["dispatch"][0] == "wave_grid"
    assert captured["dispatch"][1]["fused_group_fragments"] == 4
    expected = torch.addmm(bias, a, b)
    atol = max(0.05, 1e-3 * expected.abs().max().item())
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0.05)


def test_addmm_wave_grid_vector_bias_matches_torch():
    from triton.tlx.ops import addmm as tlx_addmm

    torch.manual_seed(3072 + 2048 + 2048)
    bias = torch.randn((2048, ), device="cuda", dtype=torch.float16)
    a = torch.randn((3072, 2048), device="cuda", dtype=torch.float16)
    b = torch.randn((2048, 2048), device="cuda", dtype=torch.float16).T
    out = torch.empty((3072, 2048), device="cuda", dtype=torch.float16)
    actual = tlx_addmm(bias, a, b, out=out)
    assert actual is out
    expected = torch.addmm(bias, a, b)
    atol = max(0.05, 1e-3 * expected.abs().max().item())
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0.05)


@pytest.mark.parametrize(
    "dtype,m,n,k,expected_path",
    [
        (torch.float16, 819200, 1024, 192, "cached_a"),
        (torch.bfloat16, 819200, 1024, 192, "cached_a"),
        (torch.float16, 3072, 10240, 128, "mt256_k128"),
        (torch.float16, 64000, 256, 256, "mt256"),
        (torch.float16, 2252800, 256, 256, "mt256"),
    ],
)
def test_addmm_shallow_k_vector_bias_uses_specialized_kernel(monkeypatch, dtype, m, n, k, expected_path):
    from triton.tlx.ops.kernels.addmm import gfx950 as addmm_gfx950

    class FakeTensor:

        def __init__(self, shape, strides):
            self.shape = shape
            self.dtype = dtype
            self._strides = strides

        def stride(self, dim=None):
            return self._strides if dim is None else self._strides[dim]

    a = FakeTensor((m, k), (k, 1))
    b = FakeTensor((k, n), (1, k))
    bias = FakeTensor((m, n), (0, 1))
    out = FakeTensor((m, n), (n, 1))
    captured = {}

    def cached_a_spy(actual_a, actual_b, actual_bias, actual_out):
        captured["path"] = "cached_a"
        captured["args"] = (actual_a, actual_b, actual_bias, actual_out)
        return actual_out

    def mt256_spy(actual_a, actual_b, actual_bias, actual_out):
        captured["path"] = "mt256"
        captured["args"] = (actual_a, actual_b, actual_bias, actual_out)
        return actual_out

    def mt256_k128_spy(actual_a, actual_b, actual_bias, actual_out):
        captured["path"] = "mt256_k128"
        captured["args"] = (actual_a, actual_b, actual_bias, actual_out)
        return actual_out

    monkeypatch.setattr(addmm_gfx950, "_launch_shallow_k192_cached_a_bias", cached_a_spy)
    monkeypatch.setattr(addmm_gfx950, "_launch_shallow_k256_mt256_bias", mt256_spy)
    monkeypatch.setattr(addmm_gfx950, "_launch_shallow_k128_mt256_bias", mt256_k128_spy)
    assert addmm_gfx950._launch(a, b, bias, out) is out
    assert captured["path"] == expected_path
    assert captured["args"] == (a, b, bias, out)


def test_addmm_large_shallow_k_plan_matches_rotating_winner():
    from triton.tlx.ops.kernels.addmm import gfx950 as addmm_gfx950

    plan = addmm_gfx950._SHALLOW_K256_MT256_PLAN
    assert (plan["mi_wave_tile_m"], plan["mi_wave_tile_n"]) == (8, 8)
    assert (plan["warps_m"], plan["warps_n"], plan["block_k"]) == (2, 2, 64)
    assert plan["local_stages"] == 2
    assert plan["pgr2_operands"]
    assert not plan["direct_to_lds"]
    assert plan["dot_mode"] == "logical"


@pytest.mark.parametrize(
    "dtype,expected_tiles,expected_prefetch",
    [(torch.float16, 4, True), (torch.bfloat16, 2, False)],
)
def test_addmm_k192_cross_half_b_prefetch(
    monkeypatch,
    dtype,
    expected_tiles,
    expected_prefetch,
):
    from triton.tlx.ops.kernels.addmm import gfx950 as addmm_gfx950

    class FakeTensor:

        def __init__(self, shape, strides):
            self.shape = shape
            self.dtype = dtype
            self._strides = strides

        def stride(self, dim=None):
            return self._strides if dim is None else self._strides[dim]

    captured = {}

    class KernelSpy:

        def __getitem__(self, grid):
            captured["grid"] = grid

            def launch(*args, **kwargs):
                captured["options"] = kwargs

            return launch

    monkeypatch.setattr(
        addmm_gfx950,
        "_shallow_k_4wave_vector_bias_kernel",
        KernelSpy(),
    )
    a = FakeTensor((819_200, 192), (192, 1))
    b = FakeTensor((192, 1_024), (1, 192))
    bias = FakeTensor((1_024, ), (1, ))
    out = FakeTensor((819_200, 1_024), (1_024, 1))
    assert addmm_gfx950._launch_shallow_k192_cached_a_bias(a, b, bias, out) is out
    assert captured["options"]["N_TILES_PER_CTA"] == expected_tiles
    assert captured["options"]["PREFETCH_NEXT_B"] == expected_prefetch


def test_addmm_k128_plan_matches_rotating_winner():
    from triton.tlx.ops.kernels.addmm import gfx950 as addmm_gfx950

    plan = addmm_gfx950._SHALLOW_K128_MT256_PLAN
    assert (plan["mi_wave_tile_m"], plan["mi_wave_tile_n"]) == (8, 4)
    assert (plan["warps_m"], plan["warps_n"], plan["block_k"]) == (2, 2, 32)
    assert plan["local_stages"] == 1
    assert plan["pgr2_operands"]
    assert not plan["direct_to_lds"]
    assert plan["dot_mode"] == "logical"
    assert plan["llvm_fn_attrs"] == ""


def test_addmm_k128_vector_bias_uses_output_precision(monkeypatch):
    from triton.tlx.ops.kernels.addmm import gfx950 as addmm_gfx950

    class FakeTensor:

        def __init__(self, shape, strides):
            self.shape = shape
            self._strides = strides

        def stride(self, dim=None):
            return self._strides if dim is None else self._strides[dim]

    a = FakeTensor((3072, 128), (128, 1))
    b = FakeTensor((128, 10240), (1, 128))
    bias = FakeTensor((3072, 10240), (0, 1))
    captured = {}

    def matmul_spy(*args, **kwargs):
        captured["plan"] = kwargs["_candidate_plan"]
        return kwargs["out"]

    monkeypatch.setattr(addmm_gfx950._mm, "_wg_matmul", matmul_spy)
    out = object()
    assert addmm_gfx950._launch_shallow_k128_mt256_bias(a, b, bias, out) is out
    assert captured["plan"]["output_precision_bias"]
    assert captured["plan"]["dot_mode"] == "persistent"
    assert captured["plan"]["prefetch_bias"]


def test_addmm_k80_plan_matches_rotating_winner():
    from triton.tlx.ops.kernels.addmm import gfx950 as addmm_gfx950

    plan = addmm_gfx950._SHALLOW_K80_FRAGMENTED_PLAN
    assert plan["kind"] == "register_fragmented_m"
    assert plan["m_fragments"] == ((0, 16), (16, 64))
    assert (plan["block_n"], plan["block_k"]) == (256, 128)
    assert plan["num_warps"] == 4


def test_addmm_k80_vector_bias_uses_fragmented_winner(monkeypatch):
    from triton.tlx.ops.kernels.addmm import gfx950 as addmm_gfx950

    class FakeTensor:

        def __init__(self, shape, strides):
            self.shape = shape
            self.dtype = torch.float16
            self._strides = strides

        def stride(self, dim=None):
            return self._strides if dim is None else self._strides[dim]

    a = FakeTensor((4718592, 80), (80, 1))
    b = FakeTensor((80, 80), (1, 80))
    bias = FakeTensor((4718592, 80), (0, 1))
    out = FakeTensor((4718592, 80), (80, 1))
    captured = {}

    def spy(actual_a, actual_b, actual_bias, actual_out):
        captured["args"] = (actual_a, actual_b, actual_bias, actual_out)
        return actual_out

    monkeypatch.setattr(addmm_gfx950, "_launch_shallow_k80_fragmented_bias", spy)
    assert addmm_gfx950._launch(a, b, bias, out) is out
    assert captured["args"] == (a, b, bias, out)


def test_addmm_k359_plan_matches_rotating_winner():
    from triton.tlx.ops.kernels.addmm import gfx950 as addmm_gfx950

    plan = addmm_gfx950._SHALLOW_K359_MT32X64_PLAN
    assert (plan["mi_wave_tile_m"], plan["mi_wave_tile_n"]) == (1, 2)
    assert (plan["warps_m"], plan["warps_n"], plan["block_k"]) == (2, 2, 64)
    assert plan["pgr2_operands"]
    assert plan["local_stages"] == 2
    assert plan["compact_k64_prefetch_before_publish"]
    assert plan["exact_lds"]
    assert plan["dot_mode"] == "logical"
    assert plan["prefetch_bias"]
    assert plan["waves_per_eu"] == 2
    assert plan["fused_group_fragments"] == 2
    assert plan["llvm_fn_attrs"] == (("amdgpu-sched-strategy", "iterative-maxocc"), )
    assert not plan["regclass_priority_trumps_globalness"]
    assert plan["sink_insts_to_avoid_spills"]
    assert plan["disable_unclustered_high_rp_reschedule"]


def test_addmm_k797_plan_matches_rotating_winner():
    from triton.tlx.ops.kernels.addmm import gfx950 as addmm_gfx950

    plan = addmm_gfx950._SHALLOW_K797_MT192X128_PLAN
    assert (plan["mi_wave_tile_m"], plan["mi_wave_tile_n"]) == (6, 4)
    assert (plan["warps_m"], plan["warps_n"], plan["block_k"]) == (2, 2, 64)
    assert plan["pgr2_operands"]
    assert plan["local_stages"] == 2
    assert plan["direct_to_lds"]
    assert plan["direct_chunk"] == 32
    assert plan["pack_direct_chunks"]
    assert plan["loop_unroll_factor"] == 3
    assert plan["read_cover_sixteenths"] == 11
    assert plan["enable_sched_group_barrier_scheduler"]
    assert plan["sched_group_barrier_mfma_per_dwordx4"] == 4
    assert plan["fused_group_fragments"] == 4
    assert plan["store_cache"] == ".cs"
    assert plan["prefetch_bias"]
    assert not plan["exact_lds"]
    assert plan["dot_mode"] == "logical"


def test_addmm_k797_vector_bias_uses_four_wave_winner(monkeypatch):
    from triton.tlx.ops.kernels.addmm import gfx950 as addmm_gfx950

    class FakeTensor:

        def __init__(self, shape, strides):
            self.shape = shape
            self.dtype = torch.float16
            self._strides = strides

        def stride(self, dim=None):
            return self._strides if dim is None else self._strides[dim]

    a = FakeTensor((3_072, 797), (797, 1))
    b = FakeTensor((797, 2_048), (1, 797))
    bias = FakeTensor((3_072, 2_048), (0, 1))
    out = FakeTensor((3_072, 2_048), (2_048, 1))
    captured = {}

    def spy(actual_a, actual_b, actual_bias, actual_out):
        captured["args"] = (actual_a, actual_b, actual_bias, actual_out)
        return actual_out

    monkeypatch.setattr(addmm_gfx950, "_launch_shallow_k797_mt192x128_bias", spy)
    assert addmm_gfx950._launch(a, b, bias, out) is out
    assert captured["args"] == (a, b, bias, out)


@pytest.mark.parametrize("dot_mode,expected", [("logical", True), ("transient", False)])
def test_addmm_fallback_wave_grid_vector_bias_prefetch_follows_dot_mode(monkeypatch, dot_mode, expected):
    from triton.tlx.ops.kernels.addmm import gfx950 as addmm_gfx950

    class FakeTensor:

        def __init__(self, shape, strides):
            self.shape = shape
            self.dtype = torch.float16
            self._strides = strides

        def stride(self, dim=None):
            return self._strides if dim is None else self._strides[dim]

    a = FakeTensor((3_072, 6_640), (6_640, 1))
    b = FakeTensor((6_640, 2_048), (1, 6_640))
    bias = FakeTensor((3_072, 2_048), (0, 1))
    out = FakeTensor((3_072, 2_048), (2_048, 1))
    captured = {}
    monkeypatch.setattr(
        addmm_gfx950._mm,
        "_dispatch_for",
        lambda actual_a, actual_b: ("wave_grid", {"dot_mode": dot_mode}),
    )

    def launch_spy(actual_a, actual_b, actual_out, dispatch, bias):
        captured["dispatch"] = dispatch
        return actual_out

    monkeypatch.setattr(addmm_gfx950._mm, "_launch_dispatch", launch_spy)
    assert addmm_gfx950._launch(a, b, bias, out) is out
    assert captured["dispatch"][1].get("prefetch_bias", False) is expected


def test_addmm_k282_register_plan_matches_rotating_winner():
    from triton.tlx.ops.kernels.addmm import gfx950 as addmm_gfx950

    plan = addmm_gfx950._mm._FP16_TRANSPOSED_REGISTER_PLANS[(256, 282, 282)]
    assert (plan["BLOCK_M"], plan["BLOCK_N"], plan["BLOCK_K"]) == (32, 32, 128)
    assert plan["GROUP_M"] == 8
    assert plan["NUM_XCDS"] == 1
    assert plan["num_warps"] == 4
    assert plan["num_stages"] == 3
    assert plan["waves_per_eu"] == 0
    assert "llvm_fn_attrs" not in plan


def test_addmm_k359_vector_bias_uses_four_wave_winner(monkeypatch):
    from triton.tlx.ops.kernels.addmm import gfx950 as addmm_gfx950

    class FakeTensor:

        def __init__(self, shape, strides):
            self.shape = shape
            self.dtype = torch.float16
            self._strides = strides

        def stride(self, dim=None):
            return self._strides if dim is None else self._strides[dim]

    a = FakeTensor((256, 359), (359, 1))
    b = FakeTensor((359, 2048), (1, 359))
    bias = FakeTensor((256, 2048), (0, 1))
    out = FakeTensor((256, 2048), (2048, 1))
    captured = {}

    def spy(actual_a, actual_b, actual_bias, actual_out):
        captured["args"] = (actual_a, actual_b, actual_bias, actual_out)
        return actual_out

    monkeypatch.setattr(addmm_gfx950, "_launch_shallow_k359_mt32x64_bias", spy)
    assert addmm_gfx950._launch(a, b, bias, out) is out
    assert captured["args"] == (a, b, bias, out)


@pytest.mark.parametrize("n", [19_584, 39_936])
def test_addmm_k768_fp16_matrix_bias_uses_four_wave_pipeline(monkeypatch, n):
    from triton.tlx.ops.kernels.addmm import gfx950 as addmm_gfx950

    class FakeTensor:

        def __init__(self, shape, strides):
            self.shape = shape
            self.dtype = torch.float16
            self._strides = strides

        def stride(self, dim=None):
            return self._strides if dim is None else self._strides[dim]

    a = FakeTensor((3_072, 768), (768, 1))
    b = FakeTensor((768, n), (1, 768))
    bias = FakeTensor((3_072, n), (n, 1))
    out = FakeTensor((3_072, n), (n, 1))
    captured = {}

    def matmul_spy(*args, **kwargs):
        captured["plan"] = kwargs["_candidate_plan"]
        return kwargs["out"]

    monkeypatch.setattr(addmm_gfx950._mm, "_wg_matmul", matmul_spy)
    assert addmm_gfx950._launch(a, b, bias, out) is out
    assert captured["plan"]["num_warps"] == 4
    assert captured["plan"]["direct_to_lds"]
    assert captured["plan"]["loop_unroll_factor"] == 3
    assert captured["plan"]["dot_mode"] == "persistent"
    assert captured["plan"]["prefetch_matrix_bias"]
    assert captured["plan"]["matrix_bias_prefetch_rows"] == 3
    assert captured["plan"]["llvm_fn_attrs"] == (("amdgpu-sched-strategy", "iterative-maxocc"), )


def test_addmm_k768_bf16_matrix_bias_uses_four_wave_pipeline(monkeypatch):
    from triton.tlx.ops.kernels.addmm import gfx950 as addmm_gfx950

    class FakeTensor:

        def __init__(self, shape, strides):
            self.shape = shape
            self.dtype = torch.bfloat16
            self._strides = strides

        def stride(self, dim=None):
            return self._strides if dim is None else self._strides[dim]

    a = FakeTensor((1_024, 768), (768, 1))
    b = FakeTensor((768, 47_232), (1, 768))
    bias = FakeTensor((1_024, 47_232), (47_232, 1))
    out = FakeTensor((1_024, 47_232), (47_232, 1))
    captured = {}

    def matmul_spy(*args, **kwargs):
        captured["plan"] = kwargs["_candidate_plan"]
        return kwargs["out"]

    monkeypatch.setattr(addmm_gfx950._mm, "_wg_matmul", matmul_spy)
    assert addmm_gfx950._launch(a, b, bias, out) is out
    assert captured["plan"]["num_warps"] == 4
    assert captured["plan"]["loop_unroll_factor"] == 3
    assert captured["plan"]["workgroup_mapping"] == 4
    assert not captured["plan"]["pack_direct_chunks"]
    assert captured["plan"]["prefetch_matrix_bias"]
    assert captured["plan"]["matrix_bias_prefetch_rows"] == 2
    assert captured["plan"]["llvm_fn_attrs"] == (("amdgpu-sched-strategy", "iterative-maxocc"), )


def test_addmm_k128_matrix_bias_uses_four_wave_family(monkeypatch):
    from triton.tlx.ops.kernels.addmm import gfx950 as addmm_gfx950

    class FakeTensor:

        def __init__(self, shape, strides):
            self.shape = shape
            self.dtype = torch.float16
            self._strides = strides

        def stride(self, dim=None):
            return self._strides if dim is None else self._strides[dim]

    a = FakeTensor((3072, 128), (128, 1))
    b = FakeTensor((128, 44032), (1, 128))
    bias = FakeTensor((3072, 44032), (44032, 1))
    out = FakeTensor((3072, 44032), (44032, 1))
    captured = {}

    def matmul_spy(*args, **kwargs):
        captured["plan"] = kwargs["_candidate_plan"]
        return kwargs["out"]

    monkeypatch.setattr(addmm_gfx950._mm, "_wg_matmul", matmul_spy)
    assert addmm_gfx950._launch(a, b, bias, out) is out
    plan = captured["plan"]
    assert (plan["mi_wave_tile_m"], plan["mi_wave_tile_n"]) == (6, 4)
    assert (plan["warps_m"], plan["warps_n"], plan["block_k"]) == (2, 2, 64)
    assert plan["local_stages"] == 1
    assert plan["pgr2_operands"]
    assert plan["disable_unclustered_high_rp_reschedule"]
    assert plan["dot_mode"] == "persistent"
    assert plan["prefetch_matrix_bias"]
    assert plan["matrix_bias_prefetch_rows"] == 4


@pytest.mark.parametrize(
    "m,expected_num_xcds,expected_late_reads,expected_waves_per_eu,expected_output_precision_bias",
    [(64000, 8, 1, 0, False), (2252800, 1, 0, 0, False)],
)
def test_addmm_k256_family_selects_validated_pipeline(
    monkeypatch,
    m,
    expected_num_xcds,
    expected_late_reads,
    expected_waves_per_eu,
    expected_output_precision_bias,
):
    from triton.tlx.ops.kernels.addmm import gfx950 as addmm_gfx950

    class FakeTensor:
        shape = (m, 256)

    captured = {}

    def matmul_spy(*args, **kwargs):
        captured["plan"] = kwargs["_candidate_plan"]
        return kwargs["out"]

    monkeypatch.setattr(addmm_gfx950._mm, "_wg_matmul", matmul_spy)
    out = object()
    assert addmm_gfx950._launch_shallow_k256_mt256_bias(FakeTensor(), object(), object(), out) is out
    assert captured["plan"]["num_xcds"] == expected_num_xcds
    assert captured["plan"]["pgr2_late_read_count"] == expected_late_reads
    assert captured["plan"]["waves_per_eu"] == expected_waves_per_eu
    assert captured["plan"].get("output_precision_bias", False) is expected_output_precision_bias
    assert captured["plan"]["prefetch_bias"]


@pytest.mark.parametrize("n", [19584, 19968, 22240])
def test_addmm_k512_matrix_family_uses_direct_four_wave_pipeline(monkeypatch, n):
    from triton.tlx.ops.kernels.addmm import gfx950 as addmm_gfx950

    class FakeTensor:

        def __init__(self, shape, strides):
            self.shape = shape
            self.dtype = torch.float16
            self._strides = strides

        def stride(self, dim=None):
            return self._strides if dim is None else self._strides[dim]

    a = FakeTensor((3072, 512), (512, 1))
    b = FakeTensor((512, n), (1, 512))
    bias = FakeTensor((3072, n), (n, 1))
    out = FakeTensor((3072, n), (n, 1))
    captured = {}

    def matmul_spy(*args, **kwargs):
        captured["plan"] = kwargs["_candidate_plan"]
        return kwargs["out"]

    monkeypatch.setattr(addmm_gfx950._mm, "_wg_matmul", matmul_spy)
    assert addmm_gfx950._launch(a, b, bias, out) is out
    plan = captured["plan"]
    assert plan["num_warps"] == 4
    assert plan["direct_to_lds"]
    assert plan["pack_direct_chunks"]
    assert plan["row_major_b_lds"]
    assert plan["loop_unroll_factor"] == 2
    assert plan["mi_wave_tile_m"] == (6 if n in (19584, 19968) else 8)
    assert plan["dot_mode"] == "persistent"
    assert plan["workgroup_mapping"] == 4
    assert plan["refill_read_window"] == (1 if n == 19968 else 0)
    assert plan["tiles_per_program"] == 1
    assert not plan["linear_tiles"]
    assert plan["num_xcds"] == 1
    assert plan["xcd_group_size"] == 1
    assert plan["prefetch_matrix_bias"]
    assert plan["matrix_bias_prefetch_rows"] == (4 if n == 22240 else 3)
    assert plan["enable_sched_group_barrier_scheduler"] is (n == 19968)
    assert plan["sched_group_barrier_mfma_per_dwordx4"] == (6 if n == 19968 else 4)
    assert plan["llvm_fn_attrs"] == ((("amdgpu-sched-strategy", "iterative-maxocc"), ) if n in (19584, 19968) else
                                     (("amdgpu-sched-strategy", "iterative-ilp"), ))


def test_addmm_bf16_k768_vector_uses_direct_four_wave_pipeline(monkeypatch):
    from triton.tlx.ops.kernels.addmm import gfx950 as addmm_gfx950

    class FakeTensor:

        def __init__(self, shape, strides):
            self.shape = shape
            self.dtype = torch.bfloat16
            self._strides = strides

        def stride(self, dim=None):
            return self._strides if dim is None else self._strides[dim]

    a = FakeTensor((2048, 768), (768, 1))
    b = FakeTensor((768, 40192), (1, 768))
    bias = FakeTensor((2048, 40192), (0, 1))
    out = FakeTensor((2048, 40192), (40192, 1))
    captured = {}

    def matmul_spy(*args, **kwargs):
        captured["plan"] = kwargs["_candidate_plan"]
        return kwargs["out"]

    monkeypatch.setattr(addmm_gfx950._mm, "_wg_matmul", matmul_spy)
    assert addmm_gfx950._launch(a, b, bias, out) is out
    plan = captured["plan"]
    assert plan["num_warps"] == 4
    assert plan["direct_to_lds"]
    assert not plan["pack_direct_chunks"]
    assert plan["row_major_b_lds"]
    assert plan["loop_unroll_factor"] == 3
    assert plan["dot_mode"] == "persistent"
    assert plan["sched_group_barrier_mfma_per_dwordx4"] == 6
    assert plan["prefetch_bias"]


def test_addmm_fp16_k768_vector_uses_direct_four_wave_pipeline(monkeypatch):
    from triton.tlx.ops.kernels.addmm import gfx950 as addmm_gfx950

    class FakeTensor:

        def __init__(self, shape, strides):
            self.shape = shape
            self.dtype = torch.float16
            self._strides = strides

        def stride(self, dim=None):
            return self._strides if dim is None else self._strides[dim]

    a = FakeTensor((3_072, 768), (768, 1))
    b = FakeTensor((768, 65_536), (1, 768))
    bias = FakeTensor((3_072, 65_536), (0, 1))
    out = FakeTensor((3_072, 65_536), (65_536, 1))
    captured = {}

    def matmul_spy(*args, **kwargs):
        captured["plan"] = kwargs["_candidate_plan"]
        return kwargs["out"]

    monkeypatch.setattr(addmm_gfx950._mm, "_wg_matmul", matmul_spy)
    assert addmm_gfx950._launch(a, b, bias, out) is out
    plan = captured["plan"]
    assert plan["num_warps"] == 4
    assert plan["direct_to_lds"]
    assert plan["pack_direct_chunks"]
    assert plan["row_major_b_lds"]
    assert plan["loop_unroll_factor"] == 3
    assert plan["refill_read_window"] == 1
    assert plan["num_xcds"] == 8
    assert plan["workgroup_mapping"] == 8
    assert plan["dot_mode"] == "persistent"
    assert plan["prefetch_bias"]


@pytest.mark.parametrize("n,k,expected_store_cache", [(40440, 256, ".cs"), (44536, 128, None)])
def test_addmm_matrix_bias_register_store_policy(monkeypatch, n, k, expected_store_cache):
    from triton.tlx.ops.kernels.addmm import gfx950 as addmm_gfx950

    class FakeTensor:

        def __init__(self, shape, strides):
            self.shape = shape
            self.dtype = torch.float16
            self._strides = strides

        def stride(self, dim=None):
            return self._strides if dim is None else self._strides[dim]

    a = FakeTensor((3072, k), (k, 1))
    b = FakeTensor((k, n), (1, k))
    bias = FakeTensor((3072, n), (n, 1))
    out = FakeTensor((3072, n), (n, 1))
    captured = {}

    monkeypatch.setattr(addmm_gfx950._mm, "_dispatch_for", lambda actual_a, actual_b: ("register", {}))

    def spy(actual_a, actual_b, actual_out, dispatch, bias):
        captured["dispatch"] = dispatch
        return actual_out

    monkeypatch.setattr(addmm_gfx950._mm, "_launch_dispatch", spy)
    assert addmm_gfx950._launch(a, b, bias, out) is out
    assert captured["dispatch"][1].get("STORE_CACHE") == expected_store_cache


def test_addmm_k797_matrix_bias_prefetches_six_rows(monkeypatch):
    from triton.tlx.ops.kernels.addmm import gfx950 as addmm_gfx950

    class FakeTensor:

        def __init__(self, shape, strides):
            self.shape = shape
            self.dtype = torch.float16
            self._strides = strides

        def stride(self, dim=None):
            return self._strides if dim is None else self._strides[dim]

    a = FakeTensor((3_072, 797), (797, 1))
    b = FakeTensor((797, 4_096), (1, 1_156))
    bias = FakeTensor((3_072, 4_096), (4_096, 1))
    out = FakeTensor((3_072, 4_096), (4_096, 1))
    captured = {}

    monkeypatch.setattr(addmm_gfx950._mm, "_dispatch_for", lambda actual_a, actual_b: ("wave_grid", {}))

    def spy(actual_a, actual_b, actual_out, dispatch, bias):
        captured["dispatch"] = dispatch
        return actual_out

    monkeypatch.setattr(addmm_gfx950._mm, "_launch_dispatch", spy)
    assert addmm_gfx950._launch(a, b, bias, out) is out
    plan = captured["dispatch"][1]
    assert plan["mi_wave_tile_m"] == 6
    assert plan["prefetch_matrix_bias"]
    assert plan["matrix_bias_prefetch_rows"] == 6
    assert not plan["direct_to_lds"]
    assert plan["exact_lds"]
    assert plan["pgr2_late_read_count"] == 10
    assert not plan["pack_direct_chunks"]
    assert plan["dot_mode"] == "logical"
    assert plan["llvm_fn_attrs"] == (("amdgpu-sched-strategy", "iterative-maxocc"), )
    assert not plan["enable_sched_group_barrier_scheduler"]
    assert plan["sched_group_barrier_mfma_per_dwordx4"] == 4


def test_addmm_odd_shallow_k_vector_bias_uses_padded_path(monkeypatch):
    from triton.tlx.ops.kernels.addmm import gfx950 as addmm_gfx950

    class FakeTensor:

        def __init__(self, shape, strides):
            self.shape = shape
            self.dtype = torch.float16
            self._strides = strides

        def stride(self, dim=None):
            return self._strides if dim is None else self._strides[dim]

    a = FakeTensor((3072, 607), (607, 1))
    b = FakeTensor((607, 38848), (1, 607))
    bias = FakeTensor((3072, 38848), (0, 1))
    out = FakeTensor((3072, 38848), (38848, 1))
    captured = {}

    def spy(actual_a, actual_b, actual_bias, actual_out):
        captured["args"] = (actual_a, actual_b, actual_bias, actual_out)
        return actual_out

    monkeypatch.setattr(addmm_gfx950, "_launch_shallow_k607_padded", spy)
    assert addmm_gfx950._launch(a, b, bias, out) is out
    assert captured["args"] == (a, b, bias, out)

    monkeypatch.setattr(addmm_gfx950._mm, "_dispatch_for", lambda actual_a, actual_b: ("inter_wave", {}))
    dispatch = addmm_gfx950._shallow_k607_padded_dispatch(a, b)
    assert dispatch[1]["GROUP_M_OVERRIDE"] == 6
    assert dispatch[1]["VECTOR_BIAS_BROADCAST"]


@pytest.mark.parametrize(
    "entry",
    [
        (3072, 1024, 2048, (2048, 1), (1, 2048), (1, ), "fp16"),
        (16, 512, 80, (80, 1), (1, 80), (1, ), "bf16"),
    ],
)
def test_full_space(entry):
    from triton.tlx.ops import addmm as tlx_addmm

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[entry[-1]]
    bias, a, b = inputs(entry, dtype)
    actual = tlx_addmm(bias, a, b, space="full")
    expected = torch.addmm(bias, a, b)
    torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.05)
