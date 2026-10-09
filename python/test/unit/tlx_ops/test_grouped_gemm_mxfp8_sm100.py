"""Authoritative public-contract coverage for ``tlx.ops.grouped_gemm_mxfp8``."""

from __future__ import annotations

import inspect
import math
import subprocess
import sys

import pytest
import torch
from triton._internal_testing import run_in_process, swizzle_scale_to_5d


def _is_sm100():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
requires_sm100 = pytest.mark.skipif(not _is_sm100(), reason="Requires SM100")


def _replace(inputs, index, value):
    updated = list(inputs)
    updated[index] = value
    return tuple(updated)


def _call(inputs, *, out=None, num_sms=None, sf_layout="natural"):
    from triton.tlx.ops import grouped_gemm_mxfp8

    return grouped_gemm_mxfp8(
        *inputs,
        out=out,
        num_sms=num_sms,
        sf_layout=sf_layout,
    )


def _dummy_inputs(
    *,
    device="cpu",
    gm=128,
    groups=1,
    n=256,
    k=256,
    weight_rank=3,
    sf_layout="natural",
):
    scale_k = k // 32
    x = torch.empty((gm, k), device=device, dtype=torch.float8_e4m3fn)
    w_3d = torch.empty((groups, n, k), device=device, dtype=torch.float8_e4m3fn)
    x_scale = torch.zeros((gm, scale_k), device=device, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    w_scale_3d = torch.zeros((groups, n, scale_k), device=device, dtype=torch.uint8).view(torch.float8_e8m0fnu)

    if sf_layout == "cublas_blocked":
        x_scale = _to_blocked_scale(x_scale.unsqueeze(0), rows=gm).reshape(-1, 512)
        w_scale = _to_blocked_scale(w_scale_3d, rows=n).reshape(-1, 512)
    elif weight_rank == 2:
        w_scale = w_scale_3d.reshape(groups * n, scale_k)
    else:
        w_scale = w_scale_3d

    w = w_3d if weight_rank == 3 else w_3d.reshape(groups * n, k)
    if gm < 128 * (groups - 1):
        raise ValueError("dummy GM cannot provide aligned prefixes for every group")
    split_values = [128] * (groups - 1) + [gm - 128 * (groups - 1)]
    split_sizes = torch.tensor(split_values, device=device, dtype=torch.int32)
    return x, x_scale, w, w_scale, split_sizes


def _assert_invalid_before_resolution(monkeypatch, inputs, match, **kwargs):
    from triton.tlx import ops

    def unexpected_resolution(*args, **resolution_kwargs):
        pytest.fail(f"implementation resolution reached with args={args}, kwargs={resolution_kwargs}")

    monkeypatch.setattr(ops, "impl_for", unexpected_resolution)
    with pytest.raises(ops.InvalidInput, match=match):
        _call(inputs, **kwargs)


def _offset_tensor_like(tensor):
    storage = torch.empty(tensor.numel() + 1, device=tensor.device, dtype=tensor.dtype)
    return storage[1:].view(tensor.shape)


def _patterned_e4m3(shape, *, salt, device):
    values = torch.arange(math.prod(shape), device=device, dtype=torch.int64)
    values = (((values * (salt * 2 + 1)) % 17) - 8).to(torch.float32) / 4
    return values.reshape(shape).to(torch.float8_e4m3fn)


def _patterned_e8m0(shape, *, salt, device):
    rows = torch.arange(shape[-2], device=device, dtype=torch.int64).unsqueeze(1)
    cols = torch.arange(shape[-1], device=device, dtype=torch.int64).unsqueeze(0)
    base = 124 + ((rows * 3 + cols * 2 + salt) % 5)
    if len(shape) == 3:
        groups = torch.arange(shape[0], device=device, dtype=torch.int64).view(-1, 1, 1)
        base = 124 + ((base.unsqueeze(0) + groups * 2 - 124) % 5)
    return base.to(torch.uint8).view(torch.float8_e8m0fnu)


def _to_blocked_scale(scale, *, rows):
    scale_bytes = scale.view(torch.uint8)
    k_groups = scale.shape[-1]
    outer_chunks = (rows + 127) // 128
    padded_rows = outer_chunks * 128
    if rows != padded_rows:
        scale_bytes = torch.nn.functional.pad(scale_bytes, (0, 0, 0, padded_rows - rows))
    return swizzle_scale_to_5d(
        scale_bytes.reshape(-1, padded_rows, k_groups),
        outer_chunks,
        (k_groups + 3) // 4,
    ).view(torch.float8_e8m0fnu)


def _e8m0_to_float32(scale):
    encoded = scale.view(torch.uint8)
    decoded = torch.exp2(encoded.to(torch.float32) - 127)
    return torch.where(encoded == 255, float("nan"), decoded)


def _dequantize_mxfp8(data, natural_scale):
    scale = _e8m0_to_float32(natural_scale).repeat_interleave(32, dim=-1)
    return data.to(torch.float32) * scale


def _make_numerical_inputs(
    split_sizes,
    *,
    n=256,
    k=256,
    weight_rank=3,
    sf_layout="natural",
    device="cuda",
):
    gm = sum(split_sizes)
    groups = len(split_sizes)
    scale_k = k // 32
    x = _patterned_e4m3((gm, k), salt=1, device=device)
    w_3d = _patterned_e4m3((groups, n, k), salt=2, device=device)
    x_scale_natural = _patterned_e8m0((gm, scale_k), salt=1, device=device)
    w_scale_natural = _patterned_e8m0((groups, n, scale_k), salt=2, device=device)

    references = []
    row = 0
    x_dequantized = _dequantize_mxfp8(x, x_scale_natural)
    for group, group_rows in enumerate(split_sizes):
        if group_rows:
            w_dequantized = _dequantize_mxfp8(w_3d[group], w_scale_natural[group])
            references.append(x_dequantized[row:row + group_rows] @ w_dequantized.T)
        row += group_rows
    reference = torch.cat(references, dim=0)

    if sf_layout == "cublas_blocked":
        x_scale = _to_blocked_scale(x_scale_natural.unsqueeze(0), rows=gm).reshape(-1, 512)
        w_scale = _to_blocked_scale(w_scale_natural, rows=n).reshape(-1, 512)
    elif weight_rank == 2:
        x_scale = x_scale_natural
        w_scale = w_scale_natural.reshape(groups * n, scale_k)
    else:
        x_scale = x_scale_natural
        w_scale = w_scale_natural

    w = w_3d if weight_rank == 3 else w_3d.reshape(groups * n, k)
    splits = torch.tensor(split_sizes, device=device, dtype=torch.int32)
    return (x, x_scale, w, w_scale, splits), reference


def _assert_correct(actual, reference):
    assert actual.shape == reference.shape
    assert actual.dtype == torch.bfloat16
    assert actual.device == reference.device
    assert actual.is_contiguous()
    torch.testing.assert_close(actual.float(), reference, atol=1.0, rtol=2e-2)


def _run_invalid_split_values(split_sizes):
    inputs, _ = _make_numerical_inputs((128, 0))
    invalid = torch.tensor(split_sizes, device="cuda", dtype=torch.int32)
    _call(_replace(inputs, 4, invalid), num_sms=1)
    torch.cuda.synchronize()


def test_e8m0_reference_decodes_extreme_encodings():
    encoded = torch.tensor([0, 127, 254, 255], dtype=torch.uint8).view(torch.float8_e8m0fnu)
    decoded = _e8m0_to_float32(encoded)
    assert decoded[0] == 2.0**-127
    assert decoded[1] == 1.0
    assert decoded[2] == 2.0**127
    assert torch.isnan(decoded[3])


def test_grouped_gemm_mxfp8_catalog_entry():
    from triton.tlx.ops._catalog import CATALOG, InvalidInput, check_inputs, has_impl

    specs = [spec for spec in CATALOG if spec.op == "grouped_gemm_mxfp8" and spec.arch == "sm100"]
    assert len(specs) == 1
    spec = specs[0]
    assert spec.variant == "ws_persistent_mxfp8"
    assert spec.impl == "triton.tlx.ops.kernels.grouped_gemm_mxfp8.sm100:grouped_gemm_mxfp8"
    assert spec.dtypes == frozenset({"float8_e4m3fn"})
    assert spec.requires == frozenset({"tma", "tmem"})
    assert not spec.supports_backward
    assert has_impl("grouped_gemm_mxfp8", "sm100")
    assert not has_impl("grouped_gemm_mxfp8", "sm90")
    assert has_impl("grouped_gemm_mxfp8", "gfx950")

    check_inputs(
        spec,
        dtype=torch.float8_e4m3fn,
        row_bytes=(256, 256, 256),
        base_ptrs=(0, 16, 32, 48),
    )
    with pytest.raises(InvalidInput, match="does not support bfloat16"):
        check_inputs(
            spec,
            dtype=torch.bfloat16,
            row_bytes=(256, 256, 256),
            base_ptrs=(0, 16, 32, 48),
        )
    with pytest.raises(InvalidInput, match="does not support these inputs"):
        check_inputs(
            spec,
            dtype=torch.float8_e4m3fn,
            row_bytes=(255, 256, 256),
            base_ptrs=(0, 16, 32, 48),
        )


def test_grouped_gemm_mxfp8_public_signature_export_and_lazy_import():
    script = """
import inspect
import sys
from triton.tlx import ops
assert 'triton.tlx.ops.kernels.grouped_gemm_mxfp8.sm100' not in sys.modules
assert str(inspect.signature(ops.grouped_gemm_mxfp8)) == "(x, x_scale, w, w_scale, split_sizes, *, out=None, num_sms=None, sf_layout='natural')"
assert 'grouped_gemm_mxfp8' in ops.__all__
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_grouped_gemm_mxfp8_catalog_resolves_fixed_implementation():
    from triton.tlx.ops._catalog import impl_for
    from triton.tlx.ops.kernels.grouped_gemm_mxfp8.sm100 import grouped_gemm_mxfp8

    implementation, spec = impl_for("grouped_gemm_mxfp8", arch="sm100")
    assert implementation is grouped_gemm_mxfp8
    assert not spec.supports_backward
    assert str(
        inspect.signature(implementation)) == ("(x: 'torch.Tensor', x_scale: 'torch.Tensor', w: 'torch.Tensor', "
                                               "w_scale: 'torch.Tensor', split_sizes: 'torch.Tensor', *, "
                                               "out: 'torch.Tensor | None' = None, num_sms: 'int | None' = None, "
                                               "sf_layout: 'str' = 'natural') -> 'torch.Tensor'")


def test_grouped_gemm_mxfp8_fixed_config_resources_are_legal():
    from triton.tlx.ops.kernels.grouped_gemm_mxfp8 import sm100

    assert sm100._CONFIG_SPEC == {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 128,
        "NUM_DATA_BUFFERS": 4,
        "NUM_SCALE_BUFFERS": 4,
        "NUM_TMEM_BUFFERS": 1,
        "NUM_TILE_BUFFERS": 3,
        "EPILOGUE_SUBTILE": 4,
        "OVERLAP_ACC": True,
        "NUM_CTAS": 1,
    }
    assert sm100._CONFIG_2CTA_SPEC == {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 128,
        "NUM_DATA_BUFFERS": 6,
        "NUM_SCALE_BUFFERS": 6,
        "NUM_TMEM_BUFFERS": 1,
        "NUM_TILE_BUFFERS": 3,
        "EPILOGUE_SUBTILE": 4,
        "OVERLAP_ACC": True,
        "NUM_CTAS": 2,
    }
    for config in (sm100._CONFIG_SPEC, sm100._CONFIG_2CTA_SPEC):
        assert sm100._config_error(config) is None
        assert sm100._estimate_smem_bytes(config) == (sm100._estimate_operand_smem_bytes(config) +
                                                      sm100._estimate_scale_smem_bytes(config) +
                                                      sm100._estimate_epilogue_smem_bytes(config) +
                                                      sm100._estimate_tile_id_smem_bytes(config) +
                                                      sm100._estimate_barrier_smem_bytes(config))
        assert (sm100._estimate_smem_bytes(config) + sm100._SMEM_SAFETY_MARGIN_BYTES <= sm100._SM100_SMEM_BYTES)
        assert sm100._estimate_tmem_columns(config) <= sm100._SM100_TMEM_COLUMNS
    assert sm100._estimate_tile_id_smem_bytes(sm100._CONFIG_SPEC) == 12
    assert sm100._estimate_tile_id_smem_bytes(sm100._CONFIG_2CTA_SPEC) == 0
    assert sm100._estimate_barrier_smem_bytes(sm100._CONFIG_SPEC) == 128
    assert sm100._estimate_barrier_smem_bytes(sm100._CONFIG_2CTA_SPEC) == 112
    assert sm100._estimate_smem_bytes(sm100._CONFIG_SPEC) == 219276
    assert sm100._estimate_smem_bytes(sm100._CONFIG_2CTA_SPEC) == 222320
    # Two overlapped 256-column accumulators share one 64-column subtile.
    assert sm100._accumulator_tmem_columns(sm100._CONFIG_SPEC) == 448
    assert sm100._accumulator_tmem_columns(sm100._CONFIG_2CTA_SPEC) == 448
    # The TMEM alias also holds the explicitly staged scales past column 448.
    assert sm100._estimate_tmem_columns(sm100._CONFIG_SPEC) == 512
    assert sm100._estimate_tmem_columns(sm100._CONFIG_2CTA_SPEC) == 512

    from triton.runtime.autotuner import Autotuner

    assert not isinstance(sm100._mxfp8_grouped_gemm_kernel, Autotuner)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"NUM_CTAS": 3}, "NUM_CTAS must be 1 or 2"),
        ({"NUM_DATA_BUFFERS": 3}, "NUM_DATA_BUFFERS must be 4"),
        ({"NUM_SCALE_BUFFERS": 3}, "NUM_SCALE_BUFFERS must be 4"),
        ({"NUM_TILE_BUFFERS": 2}, "NUM_TILE_BUFFERS must be 3"),
    ],
)
def test_grouped_gemm_mxfp8_config_rejects_unsupported_variants(override, message):
    from triton.tlx.ops.kernels.grouped_gemm_mxfp8 import sm100

    assert sm100._config_error(dict(sm100._CONFIG_SPEC, **override)) == message


def test_grouped_gemm_mxfp8_rejects_types_ranks_and_shapes_before_resolution(monkeypatch):
    inputs = _dummy_inputs()
    _assert_invalid_before_resolution(monkeypatch, _replace(inputs, 0, object()), "tensor inputs")
    _assert_invalid_before_resolution(
        monkeypatch,
        _replace(inputs, 0, inputs[0].unsqueeze(0)),
        "ranks 2/\\(2 or 3\\)/1",
    )
    _assert_invalid_before_resolution(monkeypatch, _replace(inputs, 2, inputs[2].unsqueeze(0)), "ranks")
    _assert_invalid_before_resolution(monkeypatch, _replace(inputs, 4, inputs[4].unsqueeze(0)), "ranks")
    _assert_invalid_before_resolution(monkeypatch, _replace(inputs, 0, inputs[0][:0]), "positive GM")
    _assert_invalid_before_resolution(monkeypatch, _replace(inputs, 0, inputs[0][:64]), "divisible by 128")
    _assert_invalid_before_resolution(monkeypatch, _replace(inputs, 2, inputs[2][..., :128]), "dimensions must match")

    rank2 = _dummy_inputs(groups=2, weight_rank=2)
    bad_rank2_w = rank2[2][:-1]
    _assert_invalid_before_resolution(monkeypatch, _replace(rank2, 2, bad_rank2_w), "divisible by G")


def test_grouped_gemm_mxfp8_rejects_dtypes_layouts_and_devices_before_resolution(monkeypatch):
    inputs = _dummy_inputs()
    _assert_invalid_before_resolution(monkeypatch, _replace(inputs, 0, inputs[0].to(torch.bfloat16)), "E4M3")
    _assert_invalid_before_resolution(monkeypatch, _replace(inputs, 1, inputs[1].view(torch.uint8)), "E8M0")
    _assert_invalid_before_resolution(monkeypatch, _replace(inputs, 4, inputs[4].to(torch.int64)), "torch.int32")

    noncontiguous_x = torch.empty((128, 512), dtype=torch.float8_e4m3fn)[:, ::2]
    assert noncontiguous_x.shape == inputs[0].shape and not noncontiguous_x.is_contiguous()
    _assert_invalid_before_resolution(monkeypatch, _replace(inputs, 0, noncontiguous_x), "contiguous")
    _assert_invalid_before_resolution(monkeypatch, inputs, "same CUDA device")


@requires_cuda
def test_grouped_gemm_mxfp8_rejects_scale_contract_num_sms_and_out_before_resolution(monkeypatch):
    inputs = _dummy_inputs(device="cuda")
    _assert_invalid_before_resolution(monkeypatch, inputs, "sf_layout", sf_layout="blocked")
    _assert_invalid_before_resolution(
        monkeypatch,
        _replace(inputs, 1, inputs[1][:-1]),
        "natural scales must have exact shapes",
    )
    blocked = _dummy_inputs(device="cuda", sf_layout="cublas_blocked")
    _assert_invalid_before_resolution(
        monkeypatch,
        _replace(blocked, 3, blocked[3][:-1]),
        "must contain exactly",
        sf_layout="cublas_blocked",
    )

    for invalid_num_sms in (0, -1, True, 1.5):
        _assert_invalid_before_resolution(monkeypatch, inputs, "num_sms must be an int", num_sms=invalid_num_sms)
    device_sms = torch.cuda.get_device_properties(inputs[0].device).multi_processor_count
    _assert_invalid_before_resolution(monkeypatch, inputs, "num_sms must be an int", num_sms=device_sms + 1)

    _assert_invalid_before_resolution(monkeypatch, inputs, "tensor or None", out=object())
    _assert_invalid_before_resolution(
        monkeypatch,
        inputs,
        "contiguous BF16",
        out=torch.empty((127, 256), device="cuda", dtype=torch.bfloat16),
    )
    overlapping_out = inputs[2].view(torch.bfloat16).reshape(-1)[:128 * 256].view(128, 256)
    _assert_invalid_before_resolution(monkeypatch, inputs, "must not overlap", out=overlapping_out)


@requires_cuda
def test_grouped_gemm_mxfp8_lazy_dispatch_forwards_public_keywords(monkeypatch):
    from triton.tlx import ops
    from triton.tlx.ops._catalog import CATALOG

    inputs = _dummy_inputs(device="cuda", groups=2, weight_rank=2)
    spec = next(spec for spec in CATALOG if (spec.op, spec.arch) == ("grouped_gemm_mxfp8", "sm100"))
    observed = {}

    def fake_implementation(x, x_scale, w, w_scale, split_sizes, *, out, num_sms, sf_layout):
        observed.update(
            out=out,
            num_sms=num_sms,
            sf_layout=sf_layout,
            w_rank=w.ndim,
            scale_ranks=(x_scale.ndim, w_scale.ndim),
        )
        return torch.empty((x.shape[0], w.shape[0] // split_sizes.shape[0]), device=x.device, dtype=torch.bfloat16)

    monkeypatch.setattr(ops, "impl_for", lambda *args, **kwargs: (fake_implementation, spec))
    result = _call(inputs, num_sms=1, sf_layout="natural")
    assert observed == {
        "out": None,
        "num_sms": 1,
        "sf_layout": "natural",
        "w_rank": 2,
        "scale_ranks": (2, 2),
    }
    assert result.shape == (128, 256)


@requires_cuda
def test_grouped_gemm_mxfp8_rejects_tma_alignment_before_kernel_launch(monkeypatch):
    from triton.tlx import ops
    from triton.tlx.ops._catalog import CATALOG

    spec = next(spec for spec in CATALOG if (spec.op, spec.arch) == ("grouped_gemm_mxfp8", "sm100"))

    def unexpected_kernel(*args, **kwargs):
        pytest.fail(f"kernel launch reached with args={args}, kwargs={kwargs}")

    monkeypatch.setattr(ops, "impl_for", lambda *args, **kwargs: (unexpected_kernel, spec))
    inputs = _dummy_inputs(device="cuda")
    for index in range(4):
        with pytest.raises(ops.InvalidInput, match="does not support these inputs"):
            _call(_replace(inputs, index, _offset_tensor_like(inputs[index])))

    unaligned_output_rows = _dummy_inputs(device="cuda", n=250)
    with pytest.raises(ops.InvalidInput, match="does not support these inputs"):
        _call(unaligned_output_rows)


@requires_sm100
@pytest.mark.parametrize(
    "split_sizes",
    [(-128, 256), (64, 64), (128, 128)],
    ids=["negative", "unaligned_prefix", "wrong_sum"],
)
def test_grouped_gemm_mxfp8_invalid_split_values_trap_in_subprocess(split_sizes):
    result = run_in_process(
        _run_invalid_split_values,
        (split_sizes, ),
        env={"CUDA_LAUNCH_BLOCKING": "1"},
    )
    assert isinstance(
        result.exc,
        RuntimeError), (f"expected a CUDA trap, got {result.exc!r}; driver stderr:\n{result.driver_stderr_output}")
    error = f"{result.exc}\n{result.driver_stderr_output}"
    assert any(message in error
               for message in ("device-side assert", "unspecified launch failure", "illegal instruction")), error


@requires_sm100
def test_grouped_gemm_mxfp8_historical_g2_m128_n256_k256():
    inputs, reference = _make_numerical_inputs((128, 128), n=256, k=256)
    _assert_correct(_call(inputs), reference)


@requires_sm100
def test_grouped_gemm_mxfp8_native_2cta_selected_path():
    from triton.tlx.ops.kernels.grouped_gemm_mxfp8 import sm100

    assert sm100._select_num_ctas(gm=512, n=4096, k=2048, launch_sms=2) == 2
    inputs, reference = _make_numerical_inputs((256, 256), n=4096, k=2048)
    _assert_correct(_call(inputs, num_sms=2), reference)


@requires_sm100
def test_grouped_gemm_mxfp8_native_2cta_n3072():
    from triton.tlx.ops.kernels.grouped_gemm_mxfp8 import sm100

    assert sm100._select_num_ctas(gm=512, n=3072, k=2048, launch_sms=2) == 2
    assert sm100._select_num_ctas(gm=512, n=2048, k=2048, launch_sms=2) == 1
    inputs, reference = _make_numerical_inputs((256, 256), n=3072, k=2048)
    _assert_correct(_call(inputs, num_sms=2), reference)


@requires_sm100
def test_grouped_gemm_mxfp8_native_2cta_odd_partner():
    from triton.tlx.ops.kernels.grouped_gemm_mxfp8 import sm100

    assert sm100._select_num_ctas(gm=384, n=4096, k=2048, launch_sms=2) == 2
    inputs, reference = _make_numerical_inputs((128, 256), n=4096, k=2048)
    _assert_correct(_call(inputs, num_sms=2), reference)


@requires_sm100
def test_grouped_gemm_mxfp8_static_2cta_crosses_groups_and_tile_orders():
    from triton.tlx.ops.kernels.grouped_gemm_mxfp8 import sm100

    split_sizes = (4096, 128)
    assert sm100._select_num_ctas(gm=sum(split_sizes), n=4096, k=2048, launch_sms=2) == 2
    inputs, reference = _make_numerical_inputs(split_sizes, n=4096, k=2048)
    _assert_correct(_call(inputs, num_sms=2), reference)


@requires_sm100
def test_grouped_gemm_mxfp8_static_2cta_zero_and_uneven_splits():
    from triton.tlx.ops.kernels.grouped_gemm_mxfp8 import sm100

    split_sizes = (0, 128, 0, 256, 0)
    assert sm100._select_num_ctas(gm=sum(split_sizes), n=4096, k=2048, launch_sms=2) == 2
    inputs, reference = _make_numerical_inputs(split_sizes, n=4096, k=2048)
    _assert_correct(_call(inputs, num_sms=2), reference)


@requires_sm100
def test_grouped_gemm_mxfp8_static_2cta_multiple_waves_are_deterministic():
    from triton.tlx.ops.kernels.grouped_gemm_mxfp8 import sm100

    split_sizes = (512, 384, 256)
    assert sm100._select_num_ctas(gm=sum(split_sizes), n=4096, k=2048, launch_sms=2) == 2
    inputs, reference = _make_numerical_inputs(split_sizes, n=4096, k=2048)
    out = torch.empty(reference.shape, device="cuda", dtype=torch.bfloat16)
    baseline = _call(inputs, out=out, num_sms=2).clone()
    _assert_correct(baseline, reference)

    for _ in range(3):
        out.fill_(float("nan"))
        returned = _call(inputs, out=out, num_sms=2)
        assert returned is out
        assert torch.equal(returned, baseline)


@requires_sm100
def test_grouped_gemm_mxfp8_static_2cta_cuda_graph_replay():
    from triton.tlx.ops.kernels.grouped_gemm_mxfp8 import sm100

    split_sizes = (256, 256)
    assert sm100._select_num_ctas(gm=sum(split_sizes), n=4096, k=2048, launch_sms=2) == 2
    inputs, reference = _make_numerical_inputs(
        split_sizes,
        n=4096,
        k=2048,
        sf_layout="cublas_blocked",
    )
    out = torch.empty(reference.shape, device="cuda", dtype=torch.bfloat16)

    _call(inputs, out=out, num_sms=2, sf_layout="cublas_blocked")
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _call(inputs, out=out, num_sms=2, sf_layout="cublas_blocked")
    assert captured is out

    for _ in range(3):
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.isfinite(out).all()
        _assert_correct(out, reference)


@requires_sm100
@pytest.mark.parametrize(
    ("weight_rank", "n"),
    [(2, 256), (3, 256), (3, 136)],
    ids=["packed_rank2", "grouped_rank3", "blocked_n_padding"],
)
def test_grouped_gemm_mxfp8_rank_and_natural_blocked_parity(weight_rank, n):
    natural_inputs, reference = _make_numerical_inputs((128, 256), weight_rank=weight_rank, n=n)
    blocked_inputs, blocked_reference = _make_numerical_inputs(
        (128, 256),
        weight_rank=weight_rank,
        n=n,
        sf_layout="cublas_blocked",
    )
    natural = _call(natural_inputs)
    blocked = _call(blocked_inputs, sf_layout="cublas_blocked")

    torch.testing.assert_close(reference, blocked_reference, atol=0, rtol=0)
    _assert_correct(natural, reference)
    _assert_correct(blocked, reference)
    torch.testing.assert_close(blocked, natural, atol=0, rtol=0)


@requires_sm100
@pytest.mark.parametrize(
    "split_sizes",
    [
        (128, 128, 128, 128),
        (128, 384, 128, 256),
        (0, 128, 0, 256, 0),
    ],
    ids=["balanced", "aligned_skew", "zero_expert_positions"],
)
def test_grouped_gemm_mxfp8_split_distributions_out_identity_and_num_sms(split_sizes):
    inputs, reference = _make_numerical_inputs(split_sizes)
    out = torch.full(reference.shape, float("nan"), device="cuda", dtype=torch.bfloat16)
    returned = _call(inputs, out=out, num_sms=2)
    assert returned is out
    _assert_correct(returned, reference)


@requires_sm100
def test_grouped_gemm_mxfp8_permutation_sensitive_scale_exponents():
    inputs, reference = _make_numerical_inputs((128, 256), sf_layout="cublas_blocked")
    x_scale_bytes = _patterned_e8m0((384, 8), salt=1, device="cuda").view(torch.uint8)
    assert torch.unique(x_scale_bytes).tolist() == [124, 125, 126, 127, 128]
    assert not torch.equal(x_scale_bytes[0], x_scale_bytes[1])
    _assert_correct(_call(inputs, sf_layout="cublas_blocked"), reference)


@requires_sm100
def test_grouped_gemm_mxfp8_multiple_waves_are_deterministic():
    inputs, reference = _make_numerical_inputs((512, 384, 256), n=512)
    out = torch.empty(reference.shape, device="cuda", dtype=torch.bfloat16)
    baseline = _call(inputs, out=out, num_sms=2).clone()
    _assert_correct(baseline, reference)

    for _ in range(3):
        out.fill_(float("nan"))
        returned = _call(inputs, out=out, num_sms=2)
        assert returned is out
        assert torch.equal(returned, baseline)


@requires_sm100
def test_grouped_gemm_mxfp8_cuda_graph_replay_resets_dynamic_counter():
    inputs, reference = _make_numerical_inputs((512, 384), n=512, sf_layout="cublas_blocked")
    out = torch.empty(reference.shape, device="cuda", dtype=torch.bfloat16)

    _call(inputs, out=out, num_sms=2, sf_layout="cublas_blocked")
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _call(inputs, out=out, num_sms=2, sf_layout="cublas_blocked")
    assert captured is out

    for _ in range(3):
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.isfinite(out).all()
        _assert_correct(out, reference)


@requires_sm100
def test_grouped_gemm_mxfp8_backward_policy():
    from triton.tlx.ops import UnsupportedBackward

    inputs, reference = _make_numerical_inputs((128, 128))
    out = torch.empty(reference.shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    with pytest.raises(UnsupportedBackward, match="does not support backward on sm100"):
        _call(inputs, out=out)

    with torch.no_grad():
        returned = _call(inputs, out=out)
    assert returned is out
    _assert_correct(returned, reference)
