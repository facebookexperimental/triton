from typing import Sequence

from triton.language import core
from triton._C.libtriton import ir

__all__ = [
    "experimental_device_tensormap_create1d",
    "experimental_device_tensormap_create2d",
    "experimental_tensormap_fenceproxy_acquire",
]

# https://docs.nvidia.com/cuda/parallel-thread-execution/#tensormap-new-val-validity
def _determine_elem_type(element_ty: core.dtype):
    if element_ty.primitive_bitwidth == 8:
        return 0
    elif element_ty.is_uint16():
        return 1
    elif element_ty.is_uint32():
        return 2
    elif element_ty.is_int32():
        return 3
    elif element_ty.is_uint64():
        return 4
    elif element_ty.is_int64():
        return 5
    elif element_ty.is_fp16():
        return 6
    elif element_ty.is_fp32():
        return 7
    elif element_ty.is_fp64():
        return 9
    elif element_ty.is_bf16():
        return 10
    else:
        raise ValueError("element_ty is only supported for above types")


@core.builtin
def experimental_device_tensormap_create1d(
    desc_ptr: core.tensor,
    global_address: core.tensor,
    load_size: core.tensor,
    global_size: core.tensor,
    element_ty: core.dtype,
    _semantic=None,
):
    load_size = core._unwrap_if_constexpr(load_size)
    global_size = _semantic.to_tensor(global_size)
    element_ty = core._unwrap_if_constexpr(element_ty)
    element_stride = [core.full([], 1, core.int32, _semantic=_semantic)]

    _semantic.tensormap_create(
        desc_ptr=desc_ptr,
        global_address=global_address,
        box_dim=[_semantic.to_tensor(load_size)],
        global_dim=[global_size],
        global_stride=[],
        element_stride=element_stride,
        elem_type=_determine_elem_type(element_ty),
        interleave_layout=0,
        swizzle_mode=0,
        fill_mode=0,
    )


@core.builtin
def experimental_device_tensormap_create2d(
    desc_ptr: core.tensor,
    global_address: core.tensor,
    load_size: Sequence[core.constexpr],
    global_size: Sequence[core.tensor],
    element_ty: core.dtype,
    _semantic=None,
):
    assert len(load_size) == 2
    assert len(global_size) == 2
    load_size = [core._unwrap_if_constexpr(x) for x in load_size]
    global_size = [_semantic.to_tensor(x) for x in global_size]

    element_size = element_ty.primitive_bitwidth // 8
    element_size_t = core.full([], element_size, core.int64, _semantic=_semantic)
    global_stride = _semantic.mul(element_size_t, global_size[-1], True)

    contig_dim_size_in_bytes = element_size * load_size[-1]
    if contig_dim_size_in_bytes > 128:
        load_size[-1] = 128 // element_size

    elem_stride = core.full([], 1, core.int32, _semantic=_semantic)

    _semantic.tensormap_create(
        desc_ptr=desc_ptr,
        global_address=global_address,
        box_dim=[_semantic.to_tensor(x) for x in load_size[::-1]],
        global_dim=global_size[::-1],
        global_stride=[global_stride],
        element_stride=[elem_stride, elem_stride],
        elem_type=_determine_elem_type(element_ty),
        interleave_layout=0,
        swizzle_mode=_determine_swizzle_mode_2d(contig_dim_size_in_bytes, load_size),
        fill_mode=0,
    )


def _determine_swizzle_mode_2d(contig_dim_size_in_bytes, load_size):
    if contig_dim_size_in_bytes >= 128:
        return 3
    elif contig_dim_size_in_bytes >= 64:
        return 2
    elif contig_dim_size_in_bytes >= 32:
        return 1
    else:
        raise ValueError("block size too small")


@core.builtin
def experimental_tensormap_fenceproxy_acquire(desc_ptr: core.tensor, _semantic = None):
    _semantic.tensormap_fenceproxy_acquire(desc_ptr)
