"""Compile and hardware coverage for reduce-marker multi-CTA collectives."""

import importlib.util
from pathlib import Path
import sys

import pytest
import torch

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

RUN_MULTI_CTA_REDUCTION_TESTS = False
pytestmark = pytest.mark.skipif(
    not RUN_MULTI_CTA_REDUCTION_TESTS,
    reason="Multi-CTA reduction tests are re-enabled in D123952662 after migration to reduce markers",
)


@triton.jit
def _marked_sum_kernel(
    x,
    y,
    n,
    BLOCK_SIZE: tl.constexpr,
    NUM_CTAS: tl.constexpr,
):
    local_rank = tl.program_id(1) % NUM_CTAS
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in tl.range(
            local_rank * BLOCK_SIZE,
            n,
            NUM_CTAS * BLOCK_SIZE,
    ):
        cols = off + tl.arange(0, BLOCK_SIZE)
        values = tl.load(x + cols, mask=cols < n, other=0.0).to(tl.float32)
        acc += values

    result = tl.sum(acc, axis=0, multi_cta=True)
    for off in tl.range(
            local_rank * BLOCK_SIZE,
            n,
            NUM_CTAS * BLOCK_SIZE,
    ):
        tl.store(y + off, result, mask=off < n)


@triton.jit
def _marked_sum_axis_kernel(
    x,
    y,
    n,
    BLOCK_SIZE: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    CLUSTER_AXIS: tl.constexpr,
):
    local_rank = tl.program_id(CLUSTER_AXIS) % NUM_CTAS
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in tl.range(
            local_rank * BLOCK_SIZE,
            n,
            NUM_CTAS * BLOCK_SIZE,
    ):
        cols = off + tl.arange(0, BLOCK_SIZE)
        acc += tl.load(x + cols, mask=cols < n, other=0.0)
    result = tl.sum(acc, axis=0, multi_cta=True)
    for off in tl.range(
            local_rank * BLOCK_SIZE,
            n,
            NUM_CTAS * BLOCK_SIZE,
    ):
        tl.store(y + off, result, mask=off < n)


@triton.jit
def _unmarked_sum_kernel(
    x,
    y,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    values = tl.load(x + offsets, mask=offsets < n, other=0.0).to(tl.float32)
    result = tl.sum(values, axis=0)
    tl.store(y, result)


@triton.jit
def _marked_max_kernel(
    x,
    y,
    n,
    BLOCK_SIZE: tl.constexpr,
    NUM_CTAS: tl.constexpr,
):
    local_rank = tl.program_id(1) % NUM_CTAS
    acc = tl.full([BLOCK_SIZE], -float("inf"), dtype=tl.float32)
    for off in tl.range(
            local_rank * BLOCK_SIZE,
            n,
            NUM_CTAS * BLOCK_SIZE,
    ):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        values = tl.load(x + offsets, mask=offsets < n, other=-float("inf"))
        acc = tl.maximum(acc, values)

    result = tl.max(acc, axis=0, multi_cta=True)
    for off in tl.range(
            local_rank * BLOCK_SIZE,
            n,
            NUM_CTAS * BLOCK_SIZE,
    ):
        tl.store(y + off, result, mask=off < n)


@triton.jit
def _warp_specialized_marker_kernel(
    x,
    y,
    n,
    BLOCK_SIZE: tl.constexpr,
    NUM_CTAS: tl.constexpr,
):
    local_rank = tl.program_id(1) % NUM_CTAS
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in tl.range(
            local_rank * BLOCK_SIZE,
            n,
            NUM_CTAS * BLOCK_SIZE,
            warp_specialize=True,
    ):
        cols = off + tl.arange(0, BLOCK_SIZE)
        values = tl.load(x + cols, mask=cols < n, other=0.0).to(tl.float32)
        acc += values

    result = tl.sum(acc, axis=0, multi_cta=True)
    for off in tl.range(
            local_rank * BLOCK_SIZE,
            n,
            NUM_CTAS * BLOCK_SIZE,
    ):
        tl.store(y + off, result, mask=off < n)


@triton.jit
def _rank_zero_store_marker_kernel(
    x,
    y,
    n,
    BLOCK_SIZE: tl.constexpr,
    NUM_CTAS: tl.constexpr,
):
    local_rank = tl.program_id(1) % NUM_CTAS
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in tl.range(
            local_rank * BLOCK_SIZE,
            n,
            NUM_CTAS * BLOCK_SIZE,
    ):
        cols = off + tl.arange(0, BLOCK_SIZE)
        acc += tl.load(x + cols, mask=cols < n, other=0.0)
    result = tl.sum(acc, axis=0, multi_cta=True)
    tl.store(y, result, mask=local_rank == 0)


@triton.jit
def _overlapping_store_marker_kernel(
    x,
    y,
    n,
    BLOCK_SIZE: tl.constexpr,
    NUM_CTAS: tl.constexpr,
):
    local_rank = tl.program_id(1) % NUM_CTAS
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in tl.range(
            local_rank * BLOCK_SIZE,
            n,
            NUM_CTAS * BLOCK_SIZE,
    ):
        cols = off + tl.arange(0, BLOCK_SIZE)
        values = tl.load(x + cols, mask=cols < n, other=0.0).to(tl.float32)
        acc += values

    result = tl.sum(acc, axis=0, multi_cta=True)
    tl.store(y, result)


@triton.jit
def _non_unit_stride_store_marker_kernel(
    x,
    y,
    n,
    BLOCK_SIZE: tl.constexpr,
    NUM_CTAS: tl.constexpr,
):
    local_rank = tl.program_id(1) % NUM_CTAS
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in tl.range(
            local_rank * BLOCK_SIZE,
            n,
            NUM_CTAS * BLOCK_SIZE,
    ):
        cols = off + tl.arange(0, BLOCK_SIZE)
        acc += tl.load(x + cols, mask=cols < n, other=0.0)
    result = tl.sum(acc, axis=0, multi_cta=True)
    for off in tl.range(
            local_rank * BLOCK_SIZE,
            n,
            NUM_CTAS * BLOCK_SIZE,
    ):
        aliased = off % BLOCK_SIZE
        tl.store(y + aliased, result, mask=aliased < n)


@triton.jit
def _overlapping_descriptor_store_marker_kernel(
    x,
    y,
    n,
    BLOCK_SIZE: tl.constexpr,
    NUM_CTAS: tl.constexpr,
):
    local_rank = tl.program_id(1) % NUM_CTAS
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in tl.range(
            local_rank * BLOCK_SIZE,
            n,
            NUM_CTAS * BLOCK_SIZE,
    ):
        cols = off + tl.arange(0, BLOCK_SIZE)
        values = tl.load(x + cols, mask=cols < n, other=0.0).to(tl.float32)
        acc += values

    result = tl.sum(acc, axis=0, multi_cta=True)
    output = tl.make_tensor_descriptor(
        y,
        shape=[n],
        strides=[1],
        block_shape=[BLOCK_SIZE],
    )
    output.store([0], acc + result)


@triton.jit
def _atomic_store_marker_kernel(
    x,
    y,
    n,
    BLOCK_SIZE: tl.constexpr,
    NUM_CTAS: tl.constexpr,
):
    local_rank = tl.program_id(1) % NUM_CTAS
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in tl.range(
            local_rank * BLOCK_SIZE,
            n,
            NUM_CTAS * BLOCK_SIZE,
    ):
        cols = off + tl.arange(0, BLOCK_SIZE)
        values = tl.load(x + cols, mask=cols < n, other=0.0).to(tl.float32)
        acc += values

    result = tl.sum(acc, axis=0, multi_cta=True)
    tl.atomic_add(y + local_rank, result)


@triton.jit
def _divergent_loop_marker_kernel(
    x,
    y,
    n,
    BLOCK_SIZE: tl.constexpr,
    NUM_CTAS: tl.constexpr,
):
    local_rank = tl.program_id(1) % NUM_CTAS
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in tl.range(
            local_rank * BLOCK_SIZE,
            n,
            NUM_CTAS * BLOCK_SIZE,
    ):
        cols = off + tl.arange(0, BLOCK_SIZE)
        values = tl.load(x + cols, mask=cols < n, other=0.0).to(tl.float32)
        acc += values

    result = tl.sum(acc, axis=0)
    for i in tl.range(0, local_rank + 1):
        result = tl.sum(acc + i.to(tl.float32), axis=0, multi_cta=True)
    for off in tl.range(
            local_rank * BLOCK_SIZE,
            n,
            NUM_CTAS * BLOCK_SIZE,
    ):
        tl.store(y + off, result, mask=off < n)


@triton.jit
def _raw_cluster_pid_kernel(
    x,
    y,
    n,
    BLOCK_SIZE: tl.constexpr,
    NUM_CTAS: tl.constexpr,
):
    rank = tl.program_id(1)
    cols = rank * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    values = tl.load(x + cols, mask=cols < n, other=0.0)
    result = tl.sum(values, axis=0, multi_cta=True)
    tl.store(y + cols, result, mask=cols < n)


@triton.jit
def _multiply(lhs, rhs):
    return lhs * rhs


@triton.jit
def _unsupported_combiner_kernel(
    x,
    y,
    n,
    BLOCK_SIZE: tl.constexpr,
    NUM_CTAS: tl.constexpr,
):
    local_rank = tl.program_id(1) % NUM_CTAS
    acc = tl.full([BLOCK_SIZE], 1.0, dtype=tl.float32)
    for off in tl.range(
            local_rank * BLOCK_SIZE,
            n,
            NUM_CTAS * BLOCK_SIZE,
    ):
        cols = off + tl.arange(0, BLOCK_SIZE)
        values = tl.load(x + cols, mask=cols < n, other=1.0)
        acc *= values
    result = tl.reduce(acc, axis=0, combine_fn=_multiply, multi_cta=True)
    for off in tl.range(
            local_rank * BLOCK_SIZE,
            n,
            NUM_CTAS * BLOCK_SIZE,
    ):
        tl.store(y + off, result, mask=off < n)


@triton.jit
def _non_bool_marker_kernel(
    x,
    y,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    values = tl.load(x + offsets)
    result = tl.sum(values, axis=0, multi_cta=1)
    tl.store(y, result)


def _compile(fn, constexprs, cluster_dims=None, capability=100):
    options = {"num_warps": 4}
    if cluster_dims is not None:
        options["cluster_dims"] = cluster_dims
    return triton.compile(
        ASTSource(
            fn=fn,
            signature={"x": "*fp32", "y": "*fp32", "n": "i32"} if "n" in fn.arg_names else {"x": "*fp32", "y": "*fp32"},
            constexprs=constexprs,
        ),
        target=GPUTarget("cuda", capability, 32),
        options=options,
    )


def test_reduce_marker_emitted_and_lowered():
    compiled = _compile(
        _marked_sum_kernel,
        {"BLOCK_SIZE": 1024, "NUM_CTAS": 4},
        cluster_dims=(1, 4, 1),
    )
    assert "tt.multi_cta" in compiled.asm["ttir"]
    assert "async_remote_shmem_store" in compiled.asm["ttgir"]
    assert "ttng.cluster_reduce" not in compiled.asm["ttgir"]


@pytest.mark.parametrize("capability", [90, 100, 103])
def test_reduce_marker_supported_compute_capability(capability):
    compiled = _compile(
        _marked_sum_kernel,
        {"BLOCK_SIZE": 1024, "NUM_CTAS": 4},
        cluster_dims=(1, 4, 1),
        capability=capability,
    )
    assert "async_remote_shmem_store" in compiled.asm["ttgir"]


@pytest.mark.parametrize("capability", [80, 89, 110, 120])
def test_reduce_marker_rejects_unsupported_compute_capability(capability):
    with pytest.raises(Exception) as exc_info:
        _compile(
            _marked_sum_kernel,
            {"BLOCK_SIZE": 1024, "NUM_CTAS": 4},
            cluster_dims=(1, 4, 1),
            capability=capability,
        )
    assert "multi_cta requires NVIDIA compute capability 90, 100, or 103" in str(exc_info.value)


@pytest.mark.parametrize("cluster_dims", [(1, 3, 1), (1, 8, 1), (2, 2, 1)])
def test_reduce_marker_rejects_unsupported_cluster_shape(cluster_dims):
    with pytest.raises(Exception) as exc_info:
        _compile(
            _marked_sum_kernel,
            {"BLOCK_SIZE": 1024, "NUM_CTAS": 4},
            cluster_dims=cluster_dims,
        )
    assert "multi_cta requires a one-dimensional physical cluster of size 2 or 4" in str(exc_info.value)


def test_reduce_marker_rejects_mismatched_rank_divisor():
    with pytest.raises(Exception) as exc_info:
        _compile(
            _marked_sum_kernel,
            {"BLOCK_SIZE": 1024, "NUM_CTAS": 2},
            cluster_dims=(1, 4, 1),
        )
    assert "local-rank divisor must equal physical cluster size 4" in str(exc_info.value)


def test_reduce_marker_rejects_raw_cluster_pid():
    with pytest.raises(Exception) as exc_info:
        _compile(
            _raw_cluster_pid_kernel,
            {"BLOCK_SIZE": 1024, "NUM_CTAS": 4},
            cluster_dims=(1, 4, 1),
        )
    assert "cluster-axis program ID may only be used as local_rank" in str(exc_info.value)


def test_reduce_marker_rejects_unsupported_combiner():
    with pytest.raises(Exception) as exc_info:
        _compile(
            _unsupported_combiner_kernel,
            {"BLOCK_SIZE": 1024, "NUM_CTAS": 4},
            cluster_dims=(1, 4, 1),
        )
    assert "multi_cta requires an unordered f32 add, max, or min reduction" in str(exc_info.value)


@pytest.mark.parametrize(
    "axis,cluster_dims",
    [(0, (2, 1, 1)), (1, (1, 2, 1)), (2, (1, 1, 2))],
)
def test_reduce_marker_supports_each_physical_axis(axis, cluster_dims):
    compiled = _compile(
        _marked_sum_axis_kernel,
        {"BLOCK_SIZE": 1024, "NUM_CTAS": 2, "CLUSTER_AXIS": axis},
        cluster_dims=cluster_dims,
    )
    assert "async_remote_shmem_store" in compiled.asm["ttgir"]


def test_unmarked_reduce_has_no_collective():
    compiled = _compile(_unmarked_sum_kernel, {"BLOCK_SIZE": 1024})
    assert "tt.multi_cta" not in compiled.asm["ttir"]
    assert "async_remote_shmem_store" not in compiled.asm["ttgir"]


def test_reduce_marker_single_cta_fallback():
    compiled = _compile(
        _marked_sum_kernel,
        {"BLOCK_SIZE": 1024, "NUM_CTAS": 1},
    )
    assert "tt.multi_cta" in compiled.asm["ttir"]
    assert "tt.multi_cta" not in compiled.asm["ttgir"]
    assert "async_remote_shmem_store" not in compiled.asm["ttgir"]


def test_marked_max_is_supported():
    compiled = _compile(
        _marked_max_kernel,
        {"BLOCK_SIZE": 1024, "NUM_CTAS": 4},
        cluster_dims=(1, 4, 1),
    )
    assert "async_remote_shmem_store" in compiled.asm["ttgir"]


def test_reduce_marker_rejects_warp_specialization():
    with pytest.raises(Exception) as exc_info:
        _compile(
            _warp_specialized_marker_kernel,
            {"BLOCK_SIZE": 1024, "NUM_CTAS": 4},
            cluster_dims=(1, 4, 1),
        )
    assert "multi_cta does not support warp specialization in v1" in str(exc_info.value)


def test_reduce_marker_rejects_warp_specialization_sm90_without_meta_ws():
    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = False
        with pytest.raises(Exception) as exc_info:
            triton.compile(
                ASTSource(
                    fn=_warp_specialized_marker_kernel,
                    signature={"x": "*fp32", "y": "*fp32", "n": "i32"},
                    constexprs={"BLOCK_SIZE": 1024, "NUM_CTAS": 4},
                ),
                target=GPUTarget("cuda", 90, 32),
                options={"num_warps": 4, "cluster_dims": (1, 4, 1)},
            )
    assert "multi_cta does not support warp specialization in v1" in str(exc_info.value)


def test_reduce_marker_accepts_rank_zero_store():
    compiled = _compile(
        _rank_zero_store_marker_kernel,
        {"BLOCK_SIZE": 1024, "NUM_CTAS": 4},
        cluster_dims=(1, 4, 1),
    )
    assert "async_remote_shmem_store" in compiled.asm["ttgir"]


def test_reduce_marker_rejects_overlapping_store():
    with pytest.raises(Exception) as exc_info:
        _compile(
            _overlapping_store_marker_kernel,
            {"BLOCK_SIZE": 1024, "NUM_CTAS": 4},
            cluster_dims=(1, 4, 1),
        )
    assert "multi_cta stores must be rank-strided or guarded by local_rank == 0" in str(exc_info.value)


def test_reduce_marker_rejects_non_unit_stride_store():
    with pytest.raises(Exception) as exc_info:
        _compile(
            _non_unit_stride_store_marker_kernel,
            {"BLOCK_SIZE": 1024, "NUM_CTAS": 4},
            cluster_dims=(1, 4, 1),
        )
    assert "must be a unit-stride projection" in str(exc_info.value)


def test_reduce_marker_rejects_overlapping_descriptor_store():
    with pytest.raises(Exception) as exc_info:
        _compile(
            _overlapping_descriptor_store_marker_kernel,
            {"BLOCK_SIZE": 128, "NUM_CTAS": 4},
            cluster_dims=(1, 4, 1),
        )
    assert "multi_cta descriptor store requires exactly one rank * block index" in str(exc_info.value)


def test_reduce_marker_rejects_atomic_store():
    with pytest.raises(Exception) as exc_info:
        _compile(
            _atomic_store_marker_kernel,
            {"BLOCK_SIZE": 1024, "NUM_CTAS": 4},
            cluster_dims=(1, 4, 1),
        )
    assert "multi_cta does not support atomic or reducing memory writes" in str(exc_info.value)


def test_reduce_marker_rejects_rank_divergent_loop():
    with pytest.raises(Exception) as exc_info:
        _compile(
            _divergent_loop_marker_kernel,
            {"BLOCK_SIZE": 1024, "NUM_CTAS": 4},
            cluster_dims=(1, 4, 1),
        )
    assert "multi_cta loop bounds and step must not depend on cluster rank" in str(exc_info.value)


def test_reduce_marker_rejects_automatic_warp_specialization():
    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True
        with pytest.raises(Exception) as exc_info:
            _compile(
                _marked_sum_kernel,
                {"BLOCK_SIZE": 1024, "NUM_CTAS": 4},
                cluster_dims=(1, 4, 1),
            )
    assert "does not support automatic warp specialization" in str(exc_info.value)


def test_reduce_marker_requires_bool():
    with pytest.raises(Exception) as exc_info:
        _compile(_non_bool_marker_kernel, {"BLOCK_SIZE": 1024})
    cause = exc_info.value
    while cause is not None:
        if "multi_cta must be a bool" in str(cause):
            break
        cause = cause.__cause__
    assert cause is not None


@pytest.fixture
def multi_cta_tutorial_loader():

    def load(filename):
        path = Path(__file__).resolve().parents[3] / "tutorials" / filename
        name = "_test_" + path.stem.replace("-", "_")
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    return load


@pytest.mark.parametrize("num_ctas", [1, 2, 4])
@pytest.mark.parametrize("n", [1023, 65537])
@pytest.mark.parametrize("block_rows", [0, 4, 8])
def test_multi_cta_layernorm_ragged_hardware(num_ctas, n, block_rows, multi_cta_tutorial_loader):
    if not torch.cuda.is_available():
        pytest.skip("requires an NVIDIA GPU")
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "cuda" or target.arch not in (90, 100, 103):
        pytest.skip("requires NVIDIA SM90, SM100, or SM103")
    tutorial = multi_cta_tutorial_loader("15-multi-cta-layer-norm.py")
    torch.manual_seed(0)
    x = torch.randn((17, n), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    if block_rows:
        actual, mean, rstd = tutorial.multi_cta_layernorm_2d(x, weight, bias, NUM_CTAS=num_ctas,
                                                             BLOCK_SIZE_M=block_rows)
    else:
        actual, mean, rstd = tutorial.multi_cta_layernorm(x, weight, bias, NUM_CTAS=num_ctas)
    x_ref = x.double()
    mean_ref = x_ref.mean(dim=1)
    centered = x_ref - mean_ref[:, None]
    rstd_ref = torch.rsqrt((centered * centered).mean(dim=1) + 1e-5)
    y_ref = (centered * rstd_ref[:, None] * weight.double() + bias.double()).to(x.dtype)
    torch.testing.assert_close(actual, y_ref, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(mean, mean_ref.float(), rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(rstd, rstd_ref.float(), rtol=1e-5, atol=1e-6)
