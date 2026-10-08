"""gfx950 coverage for ``tlx.ops.grouped_gemm_mxfp8``.

Architecture-neutral input validation lives in test_grouped_gemm_mxfp8_sm100.py;
this file covers the gfx950 catalog entry and the gfx950 kernel numerics.
"""

from __future__ import annotations

import math

import pytest
import torch
from triton._internal_testing import is_hip_cdna4, run_in_process, swizzle_scale_to_5d

pytestmark = pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950")

_QUAD = {
    "BLOCK_SIZE_M": 256,
    "BLOCK_SIZE_N": 256,
    "GROUP_SIZE_M": 4,
    "XCD_CHUNK": 8,
    "TILE_MODE": 0,
    "NUM_STAGES": 1,
}
_GENERIC = {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "GROUP_SIZE_M": 8,
    "XCD_CHUNK": 8,
    "TILE_MODE": 1,
    "NUM_STAGES": 2,
}
_ENGINES = pytest.mark.parametrize("config", [_QUAD, _GENERIC], ids=["quad", "generic"])


def _call(inputs, *, out=None, num_sms=None, sf_layout="natural"):
    from triton.tlx.ops import grouped_gemm_mxfp8

    return grouped_gemm_mxfp8(*inputs, out=out, num_sms=num_sms, sf_layout=sf_layout)


def _call_backend(inputs, config, *, out=None, num_sms=None, sf_layout="natural"):
    from triton.tlx.ops.kernels.grouped_gemm_mxfp8.gfx950 import grouped_gemm_mxfp8

    return grouped_gemm_mxfp8(*inputs, out=out, num_sms=num_sms, sf_layout=sf_layout, config=config)


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


def _dequantize_mxfp8(data, natural_scale):
    scale = torch.exp2(natural_scale.view(torch.uint8).to(torch.float32) - 127).repeat_interleave(32, dim=-1)
    return data.to(torch.float32) * scale


def _make_numerical_inputs(split_sizes, *, n=256, k=256, weight_rank=3, sf_layout="natural", device="cuda"):
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
    assert actual.is_contiguous()
    torch.testing.assert_close(actual.float(), reference, atol=1.0, rtol=2e-2)


def _run_invalid_split_values(split_sizes):
    inputs, _ = _make_numerical_inputs((128, 0))
    invalid = torch.tensor(split_sizes, device="cuda", dtype=torch.int32)
    inputs = inputs[:4] + (invalid, )
    _call(inputs, num_sms=1)
    torch.cuda.synchronize()


def test_grouped_gemm_mxfp8_gfx950_catalog_entry():
    from triton.tlx.ops._catalog import CATALOG, impl_for
    from triton.tlx.ops.kernels.grouped_gemm_mxfp8.gfx950 import grouped_gemm_mxfp8

    specs = [spec for spec in CATALOG if spec.op == "grouped_gemm_mxfp8" and spec.arch == "gfx950"]
    assert len(specs) == 1
    assert specs[0].dtypes == frozenset({"float8_e4m3fn"})
    assert not specs[0].supports_backward
    implementation, _ = impl_for("grouped_gemm_mxfp8", arch="gfx950")
    assert implementation is grouped_gemm_mxfp8


def test_grouped_gemm_mxfp8_gfx950_public_dispatch():
    inputs, reference = _make_numerical_inputs((128, 256), n=256, k=512)
    _assert_correct(_call(inputs), reference)


@_ENGINES
@pytest.mark.parametrize(
    ("split_sizes", "n", "k"),
    [
        ((256, 384), 256, 512),
        ((128, 0, 640, 256), 136, 384),
        ((512, 512, 512, 512), 1024, 1024),
        ((128, ), 512, 128),
        ((0, 128, 0, 256, 0), 256, 768),
    ],
    ids=["two_groups", "zero_and_n_tail_odd_k", "multi_wave", "single_k_tile", "zero_expert_positions"],
)
def test_grouped_gemm_mxfp8_gfx950_numerics(config, split_sizes, n, k):
    inputs, reference = _make_numerical_inputs(split_sizes, n=n, k=k)
    _assert_correct(_call_backend(inputs, config), reference)


@pytest.mark.parametrize("operand", ["a", "b"])
def test_grouped_gemm_mxfp8_gfx950_generic_large_row_offsets(operand):
    from triton.tlx.ops.kernels.grouped_gemm_mxfp8._scales import prepare_scales
    from triton.tlx.ops.kernels.grouped_gemm_mxfp8.gfx950 import _generic_tile

    block = 128
    # Only the final tile executes. Its first row*K offset is exactly 2**31.
    outer = 32768 + block
    k = 65536
    m, n = (outer, 8) if operand == "a" else (block, outer)
    pid_m, pid_n = (outer // block - 1, 0) if operand == "a" else (0, outer // block - 1)
    x_bytes = torch.empty((m, k), device="cuda", dtype=torch.uint8)
    w_bytes = torch.empty((1, n, k), device="cuda", dtype=torch.uint8)
    # E4M3 1.0 is encoded as 0x38; only accessed rows need initialization.
    if operand == "a":
        x_bytes[-block:].fill_(0x38)
        w_bytes.fill_(0x38)
    else:
        x_bytes.fill_(0x38)
        w_bytes[:, -block:].fill_(0x38)
    x = x_bytes.view(torch.float8_e4m3fn)
    w = w_bytes.view(torch.float8_e4m3fn)
    x_scale = torch.full((m, k // 32), 127, device="cuda", dtype=torch.uint8).view(torch.float8_e8m0fnu)
    w_scale = torch.full((1, n, k // 32), 127, device="cuda", dtype=torch.uint8).view(torch.float8_e8m0fnu)
    k_major = n % 128 == 0
    xs, ws = prepare_scales(x_scale, w_scale, gm=m, g=1, n=n, k=k, sf_layout="natural", k_major_chunks=k_major)
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

    _generic_tile[(1, )](
        pid_m,
        pid_n,
        x,
        w,
        out,
        xs,
        ws,
        0,
        m,
        n,
        k,
        k // block,
        BLOCK_SIZE_M=block,
        BLOCK_SIZE_N=block,
        BLOCK_SIZE_K=block,
        NUM_STAGES=2,
        K_MAJOR=k_major,
        num_warps=8,
    )
    tile = out[-block:] if operand == "a" else out[:, -block:]
    torch.testing.assert_close(tile, torch.full_like(tile, k), atol=0, rtol=0)


@_ENGINES
@pytest.mark.parametrize(
    ("weight_rank", "n"),
    [(2, 256), (3, 256), (3, 136)],
    ids=["packed_rank2", "grouped_rank3", "blocked_n_padding"],
)
def test_grouped_gemm_mxfp8_gfx950_rank_and_natural_blocked_parity(config, weight_rank, n):
    natural_inputs, reference = _make_numerical_inputs((128, 256), weight_rank=weight_rank, n=n)
    blocked_inputs, _ = _make_numerical_inputs((128, 256), weight_rank=weight_rank, n=n, sf_layout="cublas_blocked")
    natural = _call_backend(natural_inputs, config)
    blocked = _call_backend(blocked_inputs, config, sf_layout="cublas_blocked")
    _assert_correct(natural, reference)
    torch.testing.assert_close(blocked, natural, atol=0, rtol=0)


@_ENGINES
def test_grouped_gemm_mxfp8_gfx950_deterministic_out_identity_and_num_sms(config):
    inputs, reference = _make_numerical_inputs((512, 384, 256), n=512, k=1024)
    out = torch.empty(reference.shape, device="cuda", dtype=torch.bfloat16)
    baseline = _call_backend(inputs, config, out=out, num_sms=3).clone()
    _assert_correct(baseline, reference)
    for _ in range(3):
        out.fill_(float("nan"))
        returned = _call_backend(inputs, config, out=out, num_sms=3)
        assert returned is out
        assert torch.equal(returned, baseline)


def test_grouped_gemm_mxfp8_gfx950_graph_replay():
    inputs, reference = _make_numerical_inputs((512, 384), n=512, k=512, sf_layout="cublas_blocked")
    out = torch.empty(reference.shape, device="cuda", dtype=torch.bfloat16)
    _call(inputs, out=out, sf_layout="cublas_blocked")
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _call(inputs, out=out, sf_layout="cublas_blocked")
    assert captured is out
    for _ in range(3):
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        _assert_correct(out, reference)


@pytest.mark.parametrize(
    "split_sizes",
    [(-128, 256), (64, 64), (128, 128)],
    ids=["negative", "unaligned_prefix", "wrong_sum"],
)
def test_grouped_gemm_mxfp8_gfx950_invalid_split_values_trap_in_subprocess(split_sizes):
    # s_trap usually aborts the HSA queue, killing the child before it reports.
    try:
        result = run_in_process(_run_invalid_split_values, (split_sizes, ))
    except RuntimeError as error:
        assert "exited with code" in str(error)
        return
    assert result.exc is not None, f"expected a device trap; driver stderr:\n{result.driver_stderr_output}"
