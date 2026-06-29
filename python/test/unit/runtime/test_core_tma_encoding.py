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


# (shape, block_shape, elem_size_bytes, host_tma_dtype)
# host_tma_dtype: CUtensorMapDataType — 6=FLOAT16, 7=FLOAT32, 4=INT32, 5=INT64
_CASES = [
    ((1024, ), (256, ), 4, 7),
    ((256, 512), (64, 64), 2, 6),
    ((128, 256), (32, 64), 4, 7),
    ((64, 128, 256), (16, 32, 64), 2, 6),
]


@pytest.mark.skipif(not _has_hopper(), reason="cuTensorMapEncodeTiled requires Hopper+")
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
