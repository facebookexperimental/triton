"""Authoritative coverage for ``tlx.ops.mm_mxfp8``.

The explicit-config cases were migrated from the frozen
``tutorials/blackwell_gemm_ws_mxfp8.py`` tests in ``test_correctness.py``.
"""

from __future__ import annotations

import inspect
import subprocess
import sys

import pytest
import torch
from triton._internal_testing import swizzle_scale_to_5d


def _is_sm100():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
requires_sm100 = pytest.mark.skipif(not _is_sm100(), reason="Requires SM100")

CONFIG_2CTA = {
    "BLOCK_SIZE_N": 256,
    "GROUP_SIZE_M": 2,
    "NUM_TMEM_BUFFERS": 1,
    "NUM_CTAS": 2,
}

OVERLAP = {
    "BLOCK_SIZE_N": 256,
    "GROUP_SIZE_M": 4,
    "NUM_SMEM_BUFFERS": 4,
    "NUM_TMEM_BUFFERS": 1,
    "EPILOGUE_SUBTILE": 4,
    "OVERLAP_ACC": True,
}

COLLAB_2CTA_OVERLAP = {"NUM_CTAS": 2, "COLLAB_2CTA": True, "GROUP_SIZE_M": 2, "NUM_SMEM_BUFFERS": 6}

# B fully resident in SMEM (N = K = 384): collaborative 2-CTA, whole 128 x N row per tile.
B_RESIDENT_FUSED = {
    "BLOCK_SIZE_N": 128, "GROUP_SIZE_M": 2, "NUM_TMEM_BUFFERS": 1, "NUM_CTAS": 2, "COLLAB_2CTA": True,
    "EPILOGUE_SUBTILE": 1, "NUM_SMEM_BUFFERS": 5, "B_RESIDENT": True, "B_RES_N_TILES": 3, "B_RES_K_TILES": 3,
    "FUSE_N_TILES": 3, "ASYNC_STORE": True
}


def _call(inputs, **kwargs):
    from triton.tlx.ops import mm_mxfp8

    return mm_mxfp8(*inputs, **kwargs)


def _replace(inputs, index, value):
    updated = list(inputs)
    updated[index] = value
    return tuple(updated)


def _e8m0(shape, *, device, low=124, high=129, seed=0):
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randint(low, high, shape, device=device, dtype=torch.uint8,
                         generator=generator).view(torch.float8_e8m0fnu)


def _dequantize(data, scale):
    decoded = torch.exp2(scale.view(torch.uint8).to(torch.float32) - 127)
    return data.to(torch.float32) * decoded.repeat_interleave(32, dim=-1)


def _to_blocked(scale):
    rows, k_groups = scale.shape
    return swizzle_scale_to_5d(scale.view(torch.uint8).reshape(1, rows, k_groups), rows // 128,
                               k_groups // 4).view(torch.float8_e8m0fnu).reshape(-1, 512)


def _dummy_inputs(*, device="cpu", m=128, n=128, k=128):
    a = torch.zeros((m, k), device=device, dtype=torch.float8_e4m3fn)
    b = torch.zeros((n, k), device=device, dtype=torch.float8_e4m3fn)
    a_scale = torch.zeros((m, k // 32), device=device, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    b_scale = torch.zeros((n, k // 32), device=device, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    return a, a_scale, b, b_scale


def _make_inputs(m, n, k, *, sf_layout="natural", device="cuda"):
    generator = torch.Generator(device=device).manual_seed(0)
    a = (torch.randn((m, k), device=device, generator=generator) * 0.5).to(torch.float8_e4m3fn)
    b = (torch.randn((n, k), device=device, generator=generator) * 0.5).to(torch.float8_e4m3fn)
    a_scale = _e8m0((m, k // 32), device=device, seed=1)
    b_scale = _e8m0((n, k // 32), device=device, seed=2)
    reference = _dequantize(a, a_scale) @ _dequantize(b, b_scale).T
    if sf_layout == "cublas_blocked":
        a_scale, b_scale = _to_blocked(a_scale), _to_blocked(b_scale)
    return (a, a_scale, b, b_scale), reference


def _assert_correct(actual, reference):
    assert actual.shape == reference.shape
    assert actual.dtype == torch.bfloat16
    assert actual.is_contiguous()
    torch.testing.assert_close(actual.float(), reference, atol=1e-1, rtol=1e-2)


def _assert_invalid_before_resolution(monkeypatch, inputs, match, **kwargs):
    from triton.tlx import ops

    def unexpected_resolution(*args, **resolution_kwargs):
        pytest.fail(f"implementation resolution reached with args={args}, kwargs={resolution_kwargs}")

    monkeypatch.setattr(ops, "impl_for", unexpected_resolution)
    with pytest.raises(ops.InvalidInput, match=match):
        _call(inputs, **kwargs)


def test_mm_mxfp8_catalog_entry():
    from triton.tlx.ops._catalog import CATALOG, InvalidInput, check_inputs, has_impl

    specs = [spec for spec in CATALOG if spec.op == "mm_mxfp8"]
    assert len(specs) == 1
    spec = specs[0]
    assert spec.arch == "sm100"
    assert spec.impl == "triton.tlx.ops.kernels.mm_mxfp8.sm100:mm_mxfp8"
    assert spec.dtypes == frozenset({"float8_e4m3fn"})
    assert spec.requires == frozenset({"tma", "tmem"})
    assert not spec.supports_backward
    assert has_impl("mm_mxfp8", "sm100")
    assert not has_impl("mm_mxfp8", "sm90")

    check_inputs(spec, dtype=torch.float8_e4m3fn, base_ptrs=(0, 16, 32, 48))
    with pytest.raises(InvalidInput, match="does not support bfloat16"):
        check_inputs(spec, dtype=torch.bfloat16, base_ptrs=(0, 16, 32, 48))
    with pytest.raises(InvalidInput, match="does not support these inputs"):
        check_inputs(spec, dtype=torch.float8_e4m3fn, base_ptrs=(0, 17, 32, 48))


def test_mm_mxfp8_public_signature_export_and_lazy_import():
    script = """
import inspect
import sys
from triton.tlx import ops
assert 'triton.tlx.ops.kernels.mm_mxfp8.sm100' not in sys.modules
assert str(inspect.signature(ops.mm_mxfp8)) == "(a, a_scale, b, b_scale, *, out=None, sf_layout='natural', space='full')"
assert 'mm_mxfp8' in ops.__all__
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_mm_mxfp8_catalog_resolves_implementation():
    from triton.tlx.ops._catalog import impl_for
    from triton.tlx.ops.kernels.mm_mxfp8.sm100 import mm_mxfp8

    implementation, _ = impl_for("mm_mxfp8", arch="sm100")
    assert implementation is mm_mxfp8
    assert str(
        inspect.signature(implementation)) == "(a, a_scale, b, b_scale, *, out=None, sf_layout='natural', space='full')"


def test_mm_mxfp8_import_builds_no_autotune_space():
    from triton.runtime.autotuner import Autotuner
    from triton.tlx.ops.kernels.mm_mxfp8 import sm100

    assert not isinstance(sm100._gemm_mxfp8_ws_kernel, Autotuner)


@pytest.mark.parametrize(
    "shape",
    [(128, 128, 128), (256, 256, 256), (128, 128, 4096), (8192, 128, 1024), (128, 8192, 16384), (4096, 4096, 4096)],
)
def test_mm_mxfp8_heuristic_config_is_legal(shape):
    from triton.tlx.ops.kernels.mm_mxfp8 import sm100

    for num_sms in (148, 152):
        assert sm100._config_error(sm100.heuristic_config(*shape, num_sms), *shape) is None


def test_mm_mxfp8_b_resident_family_only_when_b_fits():
    from triton.tlx.ops.kernels.mm_mxfp8 import sm100

    assert len(sm100._b_resident_configs(3159936, 384, 384)) == 4
    assert sm100._b_resident_configs(4096, 4096, 4096) == []
    for config in sm100._b_resident_configs(3159936, 384, 384):
        assert sm100._config_error({**sm100.DEFAULT_CONFIG, **config.kwargs}, 3159936, 384, 384) is None


@pytest.mark.parametrize("shape", [(128, 128, 128), (2048, 128, 1024), (128, 8192, 4096), (4096, 4096, 4096)])
def test_mm_mxfp8_full_space_prunes_to_legal_configs(shape):
    from triton.tlx.ops.kernels.mm_mxfp8 import sm100

    M, N, K = shape
    pruned = sm100.preprocess_configs(sm100.get_cuda_autotune_config(), {"M": M, "N": N, "K": K}, NUM_SMS=148)
    assert pruned
    assert all(sm100._config_error(config.kwargs, M, N, K) is None for config in pruned)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"BLOCK_SIZE_M": 256}, "BLOCK_SIZE_M=128"),
        ({"NUM_CTAS": 4}, "NUM_CTAS must be 1 or 2"),
        ({"NUM_CTAS": 2, "BLOCK_SIZE_N": 128, "GROUP_SIZE_M": 3}, "multiple of NUM_CTAS"),
        ({"NUM_CTAS": 2, "GROUP_SIZE_M": 2, "NUM_TMEM_BUFFERS": 2}, "one TMEM buffer"),
        ({"SPLIT_K": 2}, "SPLIT_K"),
        ({**OVERLAP, "NUM_CTAS": 2, "GROUP_SIZE_M": 2}, "OVERLAP_ACC requires"),
        ({"COLLAB_2CTA": True}, "COLLAB_2CTA requires"),
        ({
            "B_RESIDENT": True, "B_RES_N_TILES": 1, "B_RES_K_TILES": 1, "SPLIT_K": 1, "NUM_CTAS": 2, "GROUP_SIZE_M": 2,
            "NUM_TMEM_BUFFERS": 1
        }, "B_RESIDENT requires"),
        ({"B_RESIDENT": True, "B_RES_N_TILES": 2, "B_RES_K_TILES": 1}, "must cover N and K"),
        ({"FUSE_N_TILES": 2}, "FUSE_N_TILES requires B_RESIDENT"),
        ({"N_INNER": True, "GROUP_SIZE_M": 8}, "N_INNER requires"),
        ({**OVERLAP, "EPILOGUE_SUBTILE": 2}, "OVERLAP_ACC requires"),
        ({**OVERLAP, "PEELED_FIRST_K": True}, "does not peel"),
    ],
)
def test_mm_mxfp8_config_rejects_unsupported_variants(override, message):
    from triton.tlx.ops.kernels.mm_mxfp8 import sm100

    assert message in sm100._config_error({**sm100.DEFAULT_CONFIG, **override}, 128, 128, 128)


def test_mm_mxfp8_natural_scale_packing_matches_cublas_swizzle():
    from triton.tlx.ops.kernels.mm_mxfp8 import sm100

    scale = _e8m0((256, 16), device="cpu", low=0, high=255)
    packed = sm100._scale_5d(scale, 256, 512, "natural")
    blocked = sm100._scale_5d(_to_blocked(scale), 256, 512, "cublas_blocked")
    assert packed.shape == blocked.shape == (1, 2, 4, 2, 256)
    assert torch.equal(packed.view(torch.uint8), blocked.view(torch.uint8))


def test_mm_mxfp8_rejects_types_ranks_and_shapes_before_resolution(monkeypatch):
    inputs = _dummy_inputs()
    _assert_invalid_before_resolution(monkeypatch, _replace(inputs, 0, object()), "tensor inputs")
    _assert_invalid_before_resolution(monkeypatch, _replace(inputs, 0, inputs[0].unsqueeze(0)), "rank-2")
    _assert_invalid_before_resolution(monkeypatch, _replace(inputs, 2, inputs[2][:, :0]), "dimensions must match")
    _assert_invalid_before_resolution(monkeypatch, _dummy_inputs(m=64), "divisible by 128")
    _assert_invalid_before_resolution(monkeypatch, _dummy_inputs(n=192), "divisible by 128")


def test_mm_mxfp8_rejects_dtypes_layouts_and_devices_before_resolution(monkeypatch):
    inputs = _dummy_inputs()
    _assert_invalid_before_resolution(monkeypatch, _replace(inputs, 0, inputs[0].to(torch.bfloat16)), "E4M3")
    _assert_invalid_before_resolution(monkeypatch, _replace(inputs, 1, inputs[1].view(torch.uint8)), "E8M0")
    noncontiguous_b = torch.zeros((128, 256), dtype=torch.float8_e4m3fn)[:, ::2]
    _assert_invalid_before_resolution(monkeypatch, _replace(inputs, 2, noncontiguous_b), "contiguous")
    _assert_invalid_before_resolution(monkeypatch, inputs, "same CUDA device")


@requires_cuda
def test_mm_mxfp8_rejects_scales_out_and_space_before_resolution(monkeypatch):
    inputs = _dummy_inputs(device="cuda", m=256, k=256)
    _assert_invalid_before_resolution(monkeypatch, inputs, "sf_layout", sf_layout="blocked")
    _assert_invalid_before_resolution(monkeypatch, _replace(inputs, 1, inputs[1][:128]), "exact shapes")
    blocked = _replace(_replace(inputs, 1, _to_blocked(inputs[1])), 3, _to_blocked(inputs[3]))
    _assert_invalid_before_resolution(monkeypatch, _replace(blocked, 3, blocked[3][:-1]), "must contain exactly",
                                      sf_layout="cublas_blocked")
    _assert_invalid_before_resolution(monkeypatch, inputs, "tensor or None", out=object())
    _assert_invalid_before_resolution(monkeypatch, inputs, "contiguous BF16", out=torch.empty((128, 128), device="cuda",
                                                                                              dtype=torch.bfloat16))
    overlapping = inputs[0].view(torch.bfloat16).reshape(-1)[:256 * 128].view(256, 128)
    _assert_invalid_before_resolution(monkeypatch, inputs, "must not overlap", out=overlapping)
    _assert_invalid_before_resolution(monkeypatch, inputs, "space='smoke'", space="smoke")


@requires_cuda
def test_mm_mxfp8_rejects_unaligned_base_before_launch(monkeypatch):
    from triton.tlx import ops
    from triton.tlx.ops._catalog import CATALOG

    spec = next(spec for spec in CATALOG if (spec.op, spec.arch) == ("mm_mxfp8", "sm100"))
    monkeypatch.setattr(ops, "impl_for", lambda *args, **kwargs: (pytest.fail, spec))
    inputs = _dummy_inputs(device="cuda")
    storage = torch.zeros(inputs[0].numel() + 1, device="cuda", dtype=torch.float8_e4m3fn)
    with pytest.raises(ops.InvalidInput, match="does not support these inputs"):
        _call(_replace(inputs, 0, storage[1:].view(inputs[0].shape)))


@requires_sm100
@pytest.mark.parametrize("shape", [(128, 128, 128), (256, 256, 256), (384, 256, 512)], ids=str)
@pytest.mark.parametrize("sf_layout", ["natural", "cublas_blocked"])
def test_mm_mxfp8_heuristic(shape, sf_layout):
    inputs, reference = _make_inputs(*shape, sf_layout=sf_layout)
    _assert_correct(_call(inputs, sf_layout=sf_layout, space="heuristic"), reference)


@requires_sm100
def test_mm_mxfp8_heuristic_split_k():
    from triton.tlx.ops.kernels.mm_mxfp8 import sm100

    assert sm100.heuristic_config(128, 128, 4096, 148)["SPLIT_K"] == 8
    inputs, reference = _make_inputs(128, 128, 4096)
    _assert_correct(_call(inputs, space="heuristic"), reference)


@requires_sm100
def test_mm_mxfp8_natural_and_blocked_are_bitwise_equal():
    natural_inputs, _ = _make_inputs(256, 384, 512)
    blocked_inputs, _ = _make_inputs(256, 384, 512, sf_layout="cublas_blocked")
    assert torch.equal(_call(natural_inputs, space="heuristic"),
                       _call(blocked_inputs, sf_layout="cublas_blocked", space="heuristic"))


@requires_sm100
def test_mm_mxfp8_default_full_space():
    inputs, reference = _make_inputs(128, 128, 128)
    _assert_correct(_call(inputs), reference)


@requires_sm100
def test_mm_mxfp8_out_identity_and_backward_policy():
    from triton.tlx.ops import UnsupportedBackward

    inputs, reference = _make_inputs(256, 256, 256)
    out = torch.full(reference.shape, float("nan"), device="cuda", dtype=torch.bfloat16)
    assert _call(inputs, out=out, space="heuristic") is out
    _assert_correct(out, reference)

    grad_out = torch.empty_like(out, requires_grad=True)
    with pytest.raises(UnsupportedBackward, match="does not support backward on sm100"):
        _call(inputs, out=grad_out, space="heuristic")


# CONFIG_2CTA is BLOCK_SIZE_N=256 over 2 CTAs, i.e. 128 columns each. Halving it
# to 64 per CTA is the narrow-tile split (EPILOGUE_SUBTILE=4 then cuts 32-column
# subtiles), and an odd M-tile count pads by a whole tile row.
@requires_sm100
@pytest.mark.parametrize(
    ("shape", "config"),
    [
        ((256, 256, 256), CONFIG_2CTA),
        ((256, 384, 256),
         {"BLOCK_SIZE_N": 256, "GROUP_SIZE_M": 4, "NUM_SMEM_BUFFERS": 2, "NUM_TMEM_BUFFERS": 1, "EPILOGUE_SUBTILE": 1}),
        ((256, 256, 256), {"GROUP_SIZE_M": 4, "NUM_SMEM_BUFFERS": 4, "NUM_TMEM_BUFFERS": 1, "EPILOGUE_SUBTILE": 1}),
        ((256, 128, 256), {**CONFIG_2CTA, "BLOCK_SIZE_N": 128}),
        ((384, 128, 256), {**CONFIG_2CTA, "BLOCK_SIZE_N": 128}),
        ((384, 256, 256), CONFIG_2CTA),
        ((512, 256, 256), {**CONFIG_2CTA, "GROUP_SIZE_M": 4, "NUM_SMEM_BUFFERS": 4, "EPILOGUE_SUBTILE": 1}),
        # Split-K against block scales: the failure mode is silently wrong rows.
        ((128, 128, 640), {"SPLIT_K": 4, "NUM_SMEM_BUFFERS": 4}),
        ((128, 128, 2048), {"SPLIT_K": 4, "NUM_SMEM_BUFFERS": 4}),
        ((256, 256, 640), {**CONFIG_2CTA, "SPLIT_K": 4, "NUM_SMEM_BUFFERS": 4}),
        # The padding M tile's partial must not overwrite the next split's rows.
        ((384, 256, 1024), {**CONFIG_2CTA, "SPLIT_K": 4, "NUM_SMEM_BUFFERS": 4}),
        ((256, 256, 512), {"PEELED_FIRST_K": True}),
        # Overlapped accumulators: alternate slots across tiles, odd tile counts
        # per CTA, split-K FP32 staging, and BK=256.
        ((4096, 1024, 4096), OVERLAP),
        ((384, 512, 768), OVERLAP),
        ((128, 4096, 512), {**OVERLAP, "NUM_SMEM_BUFFERS": 3, "SPLIT_K": 4}),
        ((512, 1024, 1024), {**OVERLAP, "BLOCK_SIZE_K": 256, "NUM_SMEM_BUFFERS": 2}),
        # Collaborative 2-CTA: leader-only MMA, cta_group::2 loads, multicast B scales.
        ((384, 512, 256), {**CONFIG_2CTA, "COLLAB_2CTA": True}),
        ((384, 256, 1024), {**CONFIG_2CTA, "COLLAB_2CTA": True, "SPLIT_K": 4, "NUM_SMEM_BUFFERS": 4}),
        ((2048, 1024, 2048), {**OVERLAP, **COLLAB_2CTA_OVERLAP}),
        ((384, 512, 768), {**OVERLAP, **COLLAB_2CTA_OVERLAP}),
        ((384, 512, 2048), {**OVERLAP, **COLLAB_2CTA_OVERLAP, "NUM_SMEM_BUFFERS": 5, "SPLIT_K": 4}),
        # Resident B, fused N tiles, async TMA stores; odd and multi-wave M tile counts.
        ((896, 384, 384), B_RESIDENT_FUSED),
        ((128 * 401, 384, 384), B_RESIDENT_FUSED),
        ((1024, 384, 384), {**B_RESIDENT_FUSED, "FUSE_N_TILES": 1, "ASYNC_STORE": False, "NUM_SMEM_BUFFERS": 6}),
        ((1024, 256, 256), {**B_RESIDENT_FUSED, "B_RES_N_TILES": 2, "B_RES_K_TILES": 2, "FUSE_N_TILES": 2}),
        ((1024, 384, 512),
         {"GROUP_SIZE_M": 1, "N_INNER": True, "NUM_SMEM_BUFFERS": 4, "ASYNC_STORE": True, "EPILOGUE_SUBTILE": 2}),
    ],
    ids=[
        "2cta",
        "bn256_1cta",
        "lean_pipeline",
        "2cta_64_columns_per_cta",
        "2cta_64_columns_odd_m_tiles",
        "2cta_odd_m_tiles",
        "2cta_tall_short_k",
        "split_k",
        "deep_k_split_k",
        "2cta_uneven_split_k",
        "2cta_odd_m_tiles_split_k",
        "peeled_first_k",
        "overlap_acc",
        "overlap_acc_odd_tiles",
        "overlap_acc_split_k",
        "overlap_acc_bk256",
        "collab_2cta_odd_m_tiles",
        "collab_2cta_split_k",
        "collab_2cta_overlap_acc",
        "collab_2cta_overlap_acc_odd_tiles",
        "collab_2cta_overlap_acc_split_k",
        "b_resident_fused_odd_m_tiles",
        "b_resident_fused_multi_wave",
        "b_resident_unfused",
        "b_resident_fused_n256",
        "n_inner_async_store",
    ],
)
def test_mm_mxfp8_explicit_config(shape, config):
    from triton.tlx.ops.kernels.mm_mxfp8 import sm100

    inputs, reference = _make_inputs(*shape)
    _assert_correct(sm100._run_config(*inputs, config), reference)
