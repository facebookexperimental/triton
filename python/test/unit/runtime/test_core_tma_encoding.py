"""Validates the shared-core recipe TMA encoder (launch.h
triton_construct_tma_desc) produces byte-identical CUtensorMap descriptors to
the proven host-TMA encoder (driver.c fillTMADescriptorTiled).

This is the correctness check for converging TMA construction into the shared
core: the recipe path (used by TritonCC / AOT-T and auto-TMA) must encode
exactly like the battle-tested JIT path (dim reversal, derived outer stride,
L2_128B promotion, small-tensor driver workaround).
"""

import pytest
import torch

import triton
from triton._internal_testing import is_cuda


def _has_hopper():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


pytestmark = pytest.mark.skipif(not _has_hopper(), reason="Requires Hopper or newer (sm90+)")

# (shape, block_shape, elem_size_bytes, host_tma_dtype)
# host_tma_dtype: CUtensorMapDataType — 6=FLOAT16, 7=FLOAT32, 3=INT32, 5=INT64
_CASES = [
    ((1024, ), (256, ), 4, 7),
    ((256, 512), (64, 64), 2, 6),
    ((128, 256), (32, 64), 4, 7),
    ((64, 128, 256), (16, 32, 64), 2, 6),
]


@pytest.mark.parametrize("shape,block,elem_size,host_dtype", _CASES)
def test_core_recipe_matches_fill_tma_tiled(shape, block, elem_size, host_dtype):
    mod = triton.runtime.driver.active.utils
    t = torch.empty(shape, device="cuda", dtype=torch.int8)
    ndim = len(shape)
    swizzle = 0  # CU_TENSOR_MAP_SWIZZLE_NONE — always valid regardless of block

    proven = mod.fill_tma_descriptor_tiled(
        t.data_ptr(),
        swizzle,
        elem_size,
        host_dtype,
        list(block),
        list(t.shape),
        list(t.stride()),
        0,
    )
    proven_bytes = mod._tma_desc_bytes(proven)

    core_bytes = mod._test_construct_tma_desc(
        t.data_ptr(),
        ndim,
        host_dtype,
        elem_size,
        swizzle,
        list(block),
        list(t.shape),
        list(t.stride()),
        0,
        0,
    )

    assert len(core_bytes) == 128
    if core_bytes != proven_bytes:
        diffs = [i for i in range(128) if proven_bytes[i] != core_bytes[i]]
        pytest.fail(f"diverged at {len(diffs)} of 128 bytes; first offsets={diffs[:12]}\n"
                    f"proven[:32]={proven_bytes[:32].hex()}\n"
                    f"core[:32]  ={core_bytes[:32].hex()}")


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
                    reason="Requires SM100 or SM103")
@pytest.mark.parametrize("width", [64, 128])
@pytest.mark.parametrize("rank", [2, 3, 4, 5])
def test_core_recipe_fp6_host_map(width, rank):
    mod = triton.runtime.driver.active.utils
    outer_shape = [4] * (rank - 1)
    storage = torch.empty((*outer_shape, width * 3 // 4), device="cuda", dtype=torch.uint8)
    block, shape, strides = [*([2] * (rank - 1)), width], [*outer_shape, width], list(storage.stride())
    host_dtype = 15  # CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B

    def host_map():
        return mod.fill_tma_descriptor_tiled(storage.data_ptr(), 0, 1, host_dtype, block, shape, strides, 0)

    def core_map():
        return mod._test_construct_tma_desc(storage.data_ptr(), rank, host_dtype, 1, 0, block, shape, strides, 0, 0)

    if width == 64 and torch.cuda.get_device_capability() != (10, 3):
        with pytest.raises(ValueError, match="64-value FP6 TMA stores require SM103"):
            host_map()
        with pytest.raises(RuntimeError, match="triton_construct_tma_desc failed"):
            core_map()
        return

    assert core_map() == mod._tma_desc_bytes(host_map())


@pytest.mark.parametrize("rank", [0, 6])
def test_host_tensor_map_rank_bounds(rank):
    mod = triton.runtime.driver.active.utils
    storage = torch.empty(64, device="cuda", dtype=torch.uint8)
    with pytest.raises(ValueError, match="tensor map rank must be between 1 and 5"):
        mod.fill_tma_descriptor_tiled(storage.data_ptr(), 0, 1, 15, [64] * rank, [64] * rank, [1] * rank, 0)
