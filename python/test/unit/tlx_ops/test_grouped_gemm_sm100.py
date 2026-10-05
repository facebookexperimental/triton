"""Public ``tlx.ops.grouped_gemm`` coverage for sm100."""

import inspect
import subprocess
import sys

import pytest
import torch
from triton._internal_testing import is_blackwell

pytestmark = pytest.mark.skipif(not is_blackwell(), reason="Requires sm100")


def _make_groups(shape_spec, *, device="cuda", requires_grad=False):
    group_a, group_b = [], []
    for m, n, k in shape_spec:
        a = torch.randn((m, k), device=device, dtype=torch.float16)
        b_storage = torch.randn((n, k), device=device, dtype=torch.float16)
        group_a.append(a.requires_grad_(requires_grad))
        group_b.append(b_storage.T.requires_grad_(requires_grad))
    return group_a, group_b


def _assert_correct(outputs, group_a, group_b):
    assert len(outputs) == len(group_a)
    for output, a, b in zip(outputs, group_a, group_b):
        assert output.shape == (a.shape[0], b.shape[1])
        assert output.dtype == a.dtype
        assert output.device == a.device
        torch.testing.assert_close(output, a @ b, atol=2e-2, rtol=1e-2)


def _sm100_device_indices():
    return [index for index in range(torch.cuda.device_count()) if torch.cuda.get_device_capability(index)[0] == 10]


def test_grouped_gemm_catalog_entry():
    from triton.tlx.ops._catalog import CATALOG, InvalidInput, check_inputs

    specs = [spec for spec in CATALOG if (spec.op, spec.arch) == ("grouped_gemm", "sm100")]
    assert len(specs) == 1
    spec = specs[0]
    assert spec.impl == "triton.tlx.ops.kernels.grouped_gemm.sm100:grouped_gemm"
    assert spec.dtypes == frozenset({"float16"})
    assert spec.requires == frozenset({"tma", "tmem"})
    assert not spec.supports_backward

    check_inputs(spec, dtype=torch.float16, row_strides=(72, 72, 264), base_ptrs=(0, 16), elem_bytes=2)
    with pytest.raises(InvalidInput, match="does not support bfloat16"):
        check_inputs(spec, dtype=torch.bfloat16, row_strides=(72, 72, 264), base_ptrs=(0, 16), elem_bytes=2)
    with pytest.raises(InvalidInput, match="does not support these inputs"):
        check_inputs(spec, dtype=torch.float16, row_strides=(70, 70, 264), base_ptrs=(0, 16), elem_bytes=2)
    with pytest.raises(InvalidInput, match="does not support these inputs"):
        check_inputs(spec, dtype=torch.float16, row_strides=(72, 72, 264), base_ptrs=(2, 16), elem_bytes=2)


def test_grouped_gemm_public_signature_and_lazy_import():
    script = """
import inspect
import sys
from triton.tlx import ops
assert 'triton.tlx.ops.kernels.grouped_gemm.sm100' not in sys.modules
assert str(inspect.signature(ops.grouped_gemm)) == '(group_a, group_b)'
assert 'grouped_gemm' in ops.__all__
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_grouped_gemm_catalog_resolves_implementation():
    from triton.tlx.ops._catalog import impl_for
    from triton.tlx.ops.kernels.grouped_gemm.sm100 import grouped_gemm

    implementation, spec = impl_for("grouped_gemm", arch="sm100")
    assert implementation is grouped_gemm
    assert not spec.supports_backward


def test_grouped_gemm_config_table_is_legal():
    from triton.tlx.ops.kernels.grouped_gemm import sm100

    assert sm100._CONFIGS["fallback"] is sm100._CONFIG
    for config in sm100._CONFIGS.values():
        assert sm100._config_error(config, num_sms=148) is None
        assert sm100._estimate_smem_bytes(config) + sm100._SMEM_SAFETY_MARGIN_BYTES <= sm100._MAX_SMEM_BYTES
        assert sm100._estimate_tmem_bytes(config) <= sm100._MAX_TMEM_BYTES


def test_grouped_gemm_config_validation_rejects_invalid_resources():
    from triton.tlx.ops.kernels.grouped_gemm import sm100

    invalid_ctas = dict(sm100._CONFIG)
    invalid_ctas["NUM_CTAS"] = 3
    assert sm100._config_error(invalid_ctas) == "NUM_CTAS must be 1 or 2"

    invalid_m = dict(sm100._CONFIG)
    invalid_m["BLOCK_SIZE_M"] = 64
    assert sm100._config_error(invalid_m) == "BLOCK_SIZE_M must be 128"

    oversized_smem = sm100._make_config(
        block_n=256,
        block_k=64,
        num_smem_buffers=4,
        epilogue_subtile=1,
        num_ctas=1,
    )
    assert sm100._config_error(oversized_smem) == "configuration exceeds the shared-memory budget"
    assert sm100._config_error(sm100._CONFIGS["n256_2cta"], num_sms=1) == "needs at least 2 SMs"


def test_grouped_gemm_group_tile_stats():
    from triton.tlx.ops.kernels.grouped_gemm import sm100

    stats = sm100._group_tile_stats(((129, 256, 64), (257, 512, 128)), block_n=256, num_ctas=2, num_sms=149)
    assert stats == {
        "real_tiles": 8,
        "scheduled_tiles": 10,
        "virtual_tiles": 2,
        "grid": 148,
        "waves": 1,
    }


@pytest.mark.parametrize(
    "shapes,num_sms,expected",
    [
        ([(512, 128, 4096)] * 8, 148, "n128_2cta"),
        ([(513, 128, 4096)] * 8, 148, "n128_1cta"),
        ([(4096, 4096, 4096)] * 16, 148, "n256_2cta"),
        ([(512, 8192, 4096)] * 16, 148, "n256_1cta_4stage"),
        ([(4096, 16384, 6144)] * 8, 148, "n256_1cta_3stage"),
        ([(513, 4096, 4096)] * 16, 148, "n256_1cta_4stage"),
        ([(2049, 4096, 4096)] * 16, 148, "n256_1cta_4stage"),
        ([(2561, 4096, 4096)] * 16, 149, "n256_2cta"),
        ([(4096, 4096, 4096)] * 4, 1, "n256_1cta_4stage"),
    ],
)
def test_grouped_gemm_shape_heuristic(shapes, num_sms, expected):
    from triton.tlx.ops.kernels.grouped_gemm import sm100

    assert sm100._pick_config(shapes, num_sms) is sm100._CONFIGS[expected]


def test_grouped_gemm_rejects_unaligned_descriptor_rows():
    from triton.tlx.ops import InvalidInput, grouped_gemm

    bad_k_a, bad_k_b = _make_groups([(17, 24, 70)])
    with pytest.raises(InvalidInput, match="does not support these inputs"):
        grouped_gemm(bad_k_a, bad_k_b)

    bad_n_a, bad_n_b = _make_groups([(17, 130, 72)])
    with pytest.raises(InvalidInput, match="does not support these inputs"):
        grouped_gemm(bad_n_a, bad_n_b)


def test_grouped_gemm_rejects_unaligned_base_pointers():
    from triton.tlx.ops import InvalidInput, grouped_gemm

    m, n, k = 17, 24, 72
    canonical_a, canonical_b = _make_groups([(m, n, k)])
    offset_a = torch.randn(m * k + 1, device="cuda", dtype=torch.float16)[1:].view(m, k)
    offset_b_storage = torch.randn(n * k + 1, device="cuda", dtype=torch.float16)[1:].view(n, k)
    assert offset_a.stride() == (k, 1)
    assert offset_b_storage.T.stride() == (1, k)

    with pytest.raises(InvalidInput, match="does not support these inputs"):
        grouped_gemm([offset_a], canonical_b)
    with pytest.raises(InvalidInput, match="does not support these inputs"):
        grouped_gemm(canonical_a, [offset_b_storage.T])


def test_grouped_gemm_non_square_ragged_and_virtual_tiles():
    from triton.tlx.ops import grouped_gemm

    # Includes tile tails, one real M tile plus a virtual 2-CTA partner, and two
    # real M tiles. N and K remain 16-byte aligned for TMA descriptors.
    shapes = [
        (137, 264, 72),
        (257, 392, 200),
        (1, 8, 8),
        (129, 128, 64),
    ]
    group_a, group_b = _make_groups(shapes)
    outputs = grouped_gemm(group_a, group_b)
    _assert_correct(outputs, group_a, group_b)


def test_grouped_gemm_persistent_grid_and_descriptor_ring():
    from triton.tlx.ops import grouped_gemm

    # The heuristic selects the four-stage one-CTA path. Every problem exceeds
    # one B200 grid, and six transitions wrap its five-slot descriptor ring.
    group_a, group_b = _make_groups([(8192, 1024, 64)] * 6)
    outputs = grouped_gemm(group_a, group_b)
    _assert_correct(outputs, group_a, group_b)


def test_grouped_gemm_two_cta_descriptor_ring():
    from triton.tlx.ops import grouped_gemm
    from triton.tlx.ops.kernels.grouped_gemm import sm100

    # The selected three-stage two-CTA path has four descriptor slots. Each
    # problem exceeds one B200 grid, and the fifth transition wraps the ring.
    shapes = [(8192, 1024, 1024)] * 5
    assert sm100._pick_config(shapes, num_sms=148) is sm100._CONFIGS["n256_2cta"]
    group_a, group_b = _make_groups(shapes)
    outputs = grouped_gemm(group_a, group_b)
    _assert_correct(outputs, group_a, group_b)


@pytest.mark.parametrize(
    "config_name",
    [
        "fallback",
        "n128_1cta",
        "n128_2cta",
        "n256_1cta_3stage",
        "n256_1cta_4stage",
        "n256_2cta",
    ],
)
def test_grouped_gemm_private_config_paths(config_name):
    from triton.tlx.ops.kernels.grouped_gemm import sm100

    group_a, group_b = _make_groups([(33, 136, 72), (257, 264, 200)])
    args = sm100._make_grouped_gemm_args(group_a, group_b)
    d_a, d_b, d_c, sizes, lds, group_size, outputs = args
    sm100._launch_grouped_gemm(d_a, d_b, d_c, sizes, lds, group_size, sm100._CONFIGS[config_name])
    _assert_correct(outputs, group_a, group_b)


def test_grouped_gemm_backward_policy():
    from triton.tlx.ops import UnsupportedBackward, grouped_gemm

    group_a, group_b = _make_groups([(32, 48, 64)], requires_grad=True)
    with pytest.raises(UnsupportedBackward, match="does not support backward on sm100"):
        grouped_gemm(group_a, group_b)
    with torch.no_grad():
        outputs = grouped_gemm(group_a, group_b)
    _assert_correct(outputs, group_a, group_b)


@pytest.mark.skipif(len(_sm100_device_indices()) < 2, reason="Requires two sm100 GPUs")
def test_grouped_gemm_uses_input_device_and_restores_current_device():
    from triton.tlx.ops import grouped_gemm

    current_device, input_device = _sm100_device_indices()[:2]
    torch.cuda.set_device(current_device)
    group_a, group_b = _make_groups([(64, 96, 80)], device=f"cuda:{input_device}")
    outputs = grouped_gemm(group_a, group_b)
    assert torch.cuda.current_device() == current_device
    _assert_correct(outputs, group_a, group_b)


def test_public_signature_matches_contract():
    from triton.tlx import ops

    assert str(inspect.signature(ops.grouped_gemm)) == "(group_a, group_b)"
