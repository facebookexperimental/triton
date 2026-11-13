"""Public ``tlx.ops.grouped_gemm`` coverage for gfx1250."""

import subprocess
import sys

import pytest
import torch

from triton._internal_testing import is_hip_gfx1250

requires_gfx1250 = pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")


def _make_groups(m_list, n, k, *, device="cuda"):
    group_a = [torch.randn((m, k), device=device, dtype=torch.float16) for m in m_list]
    group_b = [torch.randn((n, k), device=device, dtype=torch.float16).T for _ in m_list]
    return group_a, group_b


def _check(outputs, group_a, group_b):
    assert len(outputs) == len(group_a)
    for output, a, b in zip(outputs, group_a, group_b):
        assert output.shape == (a.shape[0], b.shape[1])
        assert output.dtype == torch.float16 and output.is_contiguous()
        torch.testing.assert_close(output.float(), a.float() @ b.float(), atol=5e-2, rtol=1e-2)


def test_grouped_gemm_gfx1250_catalog_entry():
    from triton.tlx.ops._catalog import CATALOG, InvalidInput, check_inputs

    specs = [spec for spec in CATALOG if (spec.op, spec.arch) == ("grouped_gemm", "gfx1250")]
    assert len(specs) == 1
    spec = specs[0]
    assert spec.impl == "triton.tlx.ops.kernels.grouped_gemm.gfx1250:grouped_gemm"
    assert spec.dtypes == frozenset({"float16"})
    assert not spec.supports_backward
    check_inputs(spec, dtype=torch.float16, base_ptrs=(0, 16))
    with pytest.raises(InvalidInput, match="does not support bfloat16"):
        check_inputs(spec, dtype=torch.bfloat16, base_ptrs=(0, ))
    with pytest.raises(InvalidInput, match="does not support these inputs"):
        check_inputs(spec, dtype=torch.float16, base_ptrs=(0, 8))


def test_grouped_gemm_gfx1250_lazy_import():
    script = """
import sys
from triton.tlx import ops
assert 'triton.tlx.ops.kernels.grouped_gemm.gfx1250' not in sys.modules
from triton.tlx.ops.kernels.grouped_gemm import gfx1250
assert gfx1250.grouped_gemm is not None
"""
    subprocess.run([sys.executable, "-c", script], check=True)


@requires_gfx1250
@pytest.mark.parametrize(
    "m_list,n,k",
    [
        ([256, 512], 256, 256),  # full tiles
        ([300, 17, 1], 384, 512),  # ragged M with M and N tails
        ([256, 768], 512, 384),  # odd K blocks: no cross-tile prefetch
        ([1024] * 4, 512, 1024),
    ],
)
def test_grouped_gemm_gfx1250(m_list, n, k):
    from triton.tlx.ops import grouped_gemm

    group_a, group_b = _make_groups(m_list, n, k)
    _check(grouped_gemm(group_a, group_b), group_a, group_b)


@requires_gfx1250
def test_grouped_gemm_gfx1250_multicast_cluster():
    from triton.tlx.ops.kernels.grouped_gemm.gfx1250 import _make_grouped_gemm_args, _perf_fn

    group_a, group_b = _make_groups([2048, 2048], 1024, 256)
    *args, cfg, outputs = _make_grouped_gemm_args(group_a, group_b, config={"NUM_SM": 16})
    assert cfg["CLUSTER_SIZE"] == 4 and not cfg["ALLOW_MN_TAILS"]
    _perf_fn(*args, cfg)
    _check(outputs, group_a, group_b)


@requires_gfx1250
@pytest.mark.parametrize(
    "shapes,match",
    [
        ([(256, 256, 256), (256, 384, 256)], "same N and K"),
        ([(256, 256, 256), (256, 256, 384)], "same N and K"),
        ([(256, 256, 200)], "divisible by 128"),
        ([(256, 256, 128)], "enough blocks"),
    ],
)
def test_grouped_gemm_gfx1250_rejects_unsupported_shapes(shapes, match):
    from triton.tlx.ops import InvalidInput, grouped_gemm

    group_a = [torch.randn((m, k), device="cuda", dtype=torch.float16) for m, n, k in shapes]
    group_b = [torch.randn((n, k), device="cuda", dtype=torch.float16).T for m, n, k in shapes]
    with pytest.raises(InvalidInput, match=match):
        grouped_gemm(group_a, group_b)


@requires_gfx1250
def test_grouped_gemm_gfx1250_backward_policy():
    from triton.tlx.ops import UnsupportedBackward, grouped_gemm

    group_a, group_b = _make_groups([256], 256, 256)
    group_a[0].requires_grad_(True)
    with pytest.raises(UnsupportedBackward, match="does not support backward"):
        grouped_gemm(group_a, group_b)
    with torch.no_grad():
        _check(grouped_gemm(group_a, group_b), group_a, group_b)
