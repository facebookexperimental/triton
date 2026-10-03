"""Public ``tlx.ops.grouped_gemm`` coverage for gfx950."""

import os
import subprocess
import sys
from unittest import mock

import pytest
import torch

from triton._internal_testing import is_hip_cdna4
from triton.tlx.ops.kernels.grouped_gemm._shapes import CORRECTNESS_SHAPES


def _gfx950_device_indices():
    return [
        index for index in range(torch.cuda.device_count())
        if getattr(torch.cuda.get_device_properties(index), "gcnArchName", "").startswith("gfx950")
    ]


def test_gfx950_device_indices_support_nonconsecutive_ordinals():
    properties = (
        mock.Mock(gcnArchName="gfx942"),
        mock.Mock(gcnArchName="gfx950:sramecc+:xnack-"),
        mock.Mock(gcnArchName="gfx942:sramecc+:xnack-"),
        mock.Mock(gcnArchName="gfx950"),
    )
    with (
            mock.patch.object(torch.cuda, "device_count", return_value=len(properties)),
            mock.patch.object(torch.cuda, "get_device_properties", side_effect=properties.__getitem__),
    ):
        assert _gfx950_device_indices() == [1, 3]


def _make_groups(shape_spec, *, device="cuda", requires_grad=False, padded=False):
    group_a, group_b = [], []
    for M, N, K in shape_spec:
        if padded:
            a = torch.randn((M, K + 8), device=device, dtype=torch.float16)[:, :K]
            b_storage = torch.randn((N, K + 8), device=device, dtype=torch.float16)[:, :K]
        else:
            a = torch.randn((M, K), device=device, dtype=torch.float16)
            b_storage = torch.randn((N, K), device=device, dtype=torch.float16)
        group_a.append(a.requires_grad_(requires_grad))
        group_b.append(b_storage.T.requires_grad_(requires_grad))
    return group_a, group_b


def test_grouped_gemm_catalog_entry():
    from triton.tlx.ops._catalog import CATALOG, InvalidInput, check_inputs

    specs = [spec for spec in CATALOG if (spec.op, spec.arch) == ("grouped_gemm", "gfx950")]
    assert len(specs) == 1
    spec = specs[0]
    assert spec.impl == "triton.tlx.ops.kernels.grouped_gemm.gfx950:grouped_gemm"
    assert spec.dtypes == frozenset({"float16"})
    assert not spec.supports_backward

    check_inputs(spec, dtype=torch.float16)
    with pytest.raises(InvalidInput, match="does not support bfloat16"):
        check_inputs(spec, dtype=torch.bfloat16)


def test_grouped_gemm_public_signature_and_lazy_import():
    script = """
import inspect
import os
import sys
from triton.tlx import ops
assert 'triton.tlx.ops.kernels.grouped_gemm.gfx950' not in sys.modules
assert str(inspect.signature(ops.grouped_gemm)) == '(group_a, group_b)'
assert 'grouped_gemm' in ops.__all__
os.environ.pop('TRITON_DISABLE_POST_MISCHED', None)
from triton.tlx.ops.kernels.grouped_gemm import gfx950
assert gfx950 is not None
assert 'TRITON_DISABLE_POST_MISCHED' not in os.environ
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_grouped_gemm_validation_before_dispatch():
    from triton.tlx.ops import InvalidInput, grouped_gemm

    with pytest.raises(InvalidInput, match="lists or tuples"):
        grouped_gemm(torch.empty(1), [])
    with pytest.raises(InvalidInput, match="same length"):
        grouped_gemm([torch.empty((2, 2))], [])
    with pytest.raises(InvalidInput, match="at least one"):
        grouped_gemm([], [])
    with pytest.raises(InvalidInput, match="tensor elements"):
        grouped_gemm([object()], [object()])

    good_a = torch.empty((2, 4), dtype=torch.float16)
    good_b = torch.empty((3, 4), dtype=torch.float16).T
    with pytest.raises(InvalidInput, match="rank-2"):
        grouped_gemm([good_a[0]], [good_b])
    with pytest.raises(InvalidInput, match="dimensions must be positive"):
        grouped_gemm([torch.empty((0, 4), dtype=torch.float16)], [good_b])
    with pytest.raises(InvalidInput, match="reduction dimensions"):
        grouped_gemm([torch.empty((2, 5), dtype=torch.float16)], [good_b])
    with pytest.raises(InvalidInput, match="row-major A"):
        grouped_gemm([torch.empty((4, 2), dtype=torch.float16).T], [good_b])
    with pytest.raises(InvalidInput, match="column-major B"):
        grouped_gemm([good_a], [torch.empty((4, 3), dtype=torch.float16)])
    with pytest.raises(InvalidInput, match="share a dtype and device"):
        grouped_gemm([good_a], [good_b.to(torch.bfloat16)])


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950")
def test_grouped_gemm_catalog_resolves_implementation():
    from triton.tlx.ops._catalog import impl_for
    from triton.tlx.ops.kernels.grouped_gemm.gfx950 import grouped_gemm

    implementation, spec = impl_for("grouped_gemm", arch="gfx950")
    assert implementation is grouped_gemm
    assert not spec.supports_backward


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950")
def test_grouped_gemm_heuristic_reaches_both_tile_modes():
    from triton.tlx.ops.kernels.grouped_gemm.gfx950 import _pick_config

    device = torch.device("cuda")
    quadrant = _pick_config([(4096, 4096, 4096)] * 16, device)
    generic = _pick_config([(16, 4096, 4096)] * 8, device)
    assert quadrant["TILE_MODE"] == 0
    assert generic["TILE_MODE"] == 1


def test_grouped_gemm_rejects_padded_leading_strides():
    from triton.tlx.ops import InvalidInput, grouped_gemm

    padded_a, padded_b = _make_groups([(33, 65, 50)], device="cpu", padded=True)
    canonical_a, canonical_b = _make_groups([(33, 65, 50)], device="cpu")
    with pytest.raises(InvalidInput, match="row-major A"):
        grouped_gemm(padded_a, canonical_b)
    with pytest.raises(InvalidInput, match="column-major B"):
        grouped_gemm(canonical_a, padded_b)


def test_grouped_gemm_sets_post_misched_during_launch(monkeypatch):
    from triton.tlx.ops.kernels.grouped_gemm import gfx950

    observed = []

    class FakeKernel:

        def __getitem__(self, grid):
            assert grid == (1, )

            def launch(*args, **kwargs):
                observed.append(os.environ.get("TRITON_DISABLE_POST_MISCHED"))

            return launch

    metadata = torch.empty(1)
    cfg = {
        "BLOCK_SIZE_M": 1,
        "BLOCK_SIZE_N": 1,
        "BLOCK_SIZE_K": 1,
        "GROUP_SIZE_M": 1,
        "XCD_CHUNK": 1,
        "NUM_BUFFERS": 1,
        "TILE_MODE": 1,
        "NUM_STAGES": 1,
        "HAS_K_TAIL": False,
        "num_warps": 1,
    }
    monkeypatch.delenv("TRITON_DISABLE_POST_MISCHED", raising=False)
    with (
            mock.patch.object(gfx950, "grouped_gemm_kernel", FakeKernel()),
            mock.patch.object(gfx950, "_num_sms", return_value=1),
            mock.patch.object(torch.cuda, "device", return_value=mock.MagicMock()),
    ):
        gfx950._perf_fn(metadata, metadata, metadata, metadata, metadata, 1, cfg)
    assert observed == ["1"]
    assert "TRITON_DISABLE_POST_MISCHED" not in os.environ


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950")
def test_grouped_gemm_restores_post_misched_environment(monkeypatch):
    from triton.tlx.ops import grouped_gemm

    monkeypatch.delenv("TRITON_DISABLE_POST_MISCHED", raising=False)
    group_a, group_b = _make_groups([(16, 16, 16)])
    grouped_gemm(group_a, group_b)
    assert "TRITON_DISABLE_POST_MISCHED" not in os.environ


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950")
def test_grouped_gemm_backward_policy():
    from triton.tlx.ops import UnsupportedBackward, grouped_gemm

    group_a, group_b = _make_groups([(32, 48, 64)], requires_grad=True)
    with pytest.raises(UnsupportedBackward, match="does not support backward"):
        grouped_gemm(group_a, group_b)
    with torch.no_grad():
        outputs = grouped_gemm(group_a, group_b)
    torch.testing.assert_close(outputs[0], group_a[0] @ group_b[0], atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950")
@pytest.mark.parametrize("label,shape_spec", CORRECTNESS_SHAPES, ids=[case.label for case in CORRECTNESS_SHAPES])
def test_grouped_gemm(label, shape_spec):
    del label
    from triton.tlx.ops import grouped_gemm

    group_a, group_b = _make_groups(shape_spec)
    outputs = grouped_gemm(group_a, group_b)
    assert len(outputs) == len(shape_spec)
    for output, a, b in zip(outputs, group_a, group_b):
        assert output.shape == (a.shape[0], b.shape[1])
        assert output.dtype == a.dtype
        assert output.device == a.device
        torch.testing.assert_close(output, a @ b, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(len(_gfx950_device_indices()) < 2, reason="Requires two gfx950 GPUs")
def test_grouped_gemm_uses_input_device_and_restores_current_device():
    from triton.tlx.ops import grouped_gemm

    gfx950_devices = _gfx950_device_indices()
    current_device, input_device = gfx950_devices[:2]
    torch.cuda.set_device(current_device)
    group_a, group_b = _make_groups([(64, 96, 80)], device=f"cuda:{input_device}")
    outputs = grouped_gemm(group_a, group_b)
    assert torch.cuda.current_device() == current_device
    assert outputs[0].device.index == input_device
    torch.testing.assert_close(outputs[0], group_a[0] @ group_b[0], atol=1e-2, rtol=1e-2)
