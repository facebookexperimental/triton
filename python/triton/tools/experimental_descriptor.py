from dataclasses import dataclass
from typing import List, Any

import torch
import triton


def _fill_desc(desc, ptr, dims, block_dims, element_size):
    assert len(dims) == len(block_dims)
    assert 1 <= len(dims) <= 2
    assert desc.tma_desc_cpu_ptr() % 64 == 0

    if len(dims) == 1:
        triton.runtime.driver.active.utils.fill_1d_tma_descriptor(ptr, dims[0], block_dims[0], element_size,
                                                                  desc.tma_desc_cpu_ptr())
    else:
        triton.runtime.driver.active.utils.fill_2d_tma_descriptor(ptr, dims[0], dims[1], block_dims[0], block_dims[1],
                                                                  element_size, desc.tma_desc_cpu_ptr())


def create_1d_tma_descriptor(ptr, dim, block_dim, element_size):
    desc = triton.runtime.driver.active.utils.TmaDescKernelParam()
    _fill_desc(desc, ptr, [dim], [block_dim], element_size)
    return desc


def create_2d_tma_descriptor(ptr, dim1, dim0, block_dim1, block_dim0, element_size):
    desc = triton.runtime.driver.active.utils.TmaDescKernelParam()
    _fill_desc(desc, ptr, [dim1, dim0], [block_dim1, block_dim0], element_size)
    return desc


@dataclass
class TensorDescriptor:
    base: Any
    shape: List[int]
    strides: List[int]
    block_shape: List[int]

    def from_tensor(tensor: Any, block_shape: List[int]):
        return TensorDescriptor(
            tensor,
            tensor.shape,
            tensor.stride(),
            block_shape,
        )


class TmaDescKernelParamType:
    TMA_DESC_SIZE = 128

    def __init__(self, ptr, dims, block_dims, dtype):
        self.desc = torch.empty(self.TMA_DESC_SIZE, dtype=torch.uint8, device="cpu")
        assert len(dims) == len(block_dims)
        assert 1 <= len(dims) <= 2
        assert self.desc.data_ptr() % 64 == 0

        if len(dims) == 1:
            triton.runtime.driver.active.utils.fill_1d_tma_descriptor_type(ptr, dims[0], block_dims[0], dtype,
                                                                           self.desc.data_ptr())
        else:
            triton.runtime.driver.active.utils.fill_2d_tma_descriptor_type(
                ptr,
                dims[0],
                dims[1],
                block_dims[0],
                block_dims[1],
                dtype,
                self.desc.data_ptr(),
            )

    # Return a CUtensorMap* pointer in host memory
    def tma_desc_cpu_ptr(self):
        return self.desc.data_ptr()


def create_1d_tma_descriptor_type(ptr, dim, block_dim, dtype):
    return TmaDescKernelParamType(ptr, [dim], [block_dim], dtype)


def create_2d_tma_descriptor_type(ptr, dim1, dim0, block_dim1, block_dim0, dtype):
    return TmaDescKernelParamType(ptr, [dim1, dim0], [block_dim1, block_dim0], dtype)


def enable_in_pytorch():
    return False
