# AOT ID: ['149_backward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_binbao/o3/co3wat6tiwap7yhb2rgnq5vh3wdhv5iotvitvfuuvphtkoljhflg.py
# Topologically Sorted Source Nodes: [max_pool2d], Original ATen: [aten.max_pool2d_with_indices, aten.max_pool2d_with_indices_backward]
# Source node to ATen node mapping:
#   max_pool2d => _low_memory_max_pool_offsets_to_indices
# Graph fragment:
#   %getitem_1 : Tensor "i8[1, 2, 2, 4][16, 1, 8, 2]cuda:0" = PlaceHolder[target=getitem_1]
#   %tangents_1 : Tensor "f32[1, 2, 2, 4][16, 1, 8, 2]cuda:0" = PlaceHolder[target=tangents_1]
#   %_low_memory_max_pool_offsets_to_indices : Tensor "i64[1, 2, 2, 4][16, 1, 8, 2]cuda:0"[num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_offsets_to_indices.default](args = (%getitem_1, [3, 3], [3, 6], [2, 2], [1, 1], [1, 1]), kwargs = {})
#   %max_pool2d_with_indices_backward : Tensor "f32[1, 2, 3, 6][36, 1, 12, 2]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.max_pool2d_with_indices_backward.default](args = (%tangents_1, %primals_1, [3, 3], [2, 2], [1, 1], [1, 1], True, %_low_memory_max_pool_offsets_to_indices), kwargs = {})
#   return %max_pool2d_with_indices_backward
triton_kernel = async_compile.triton(
    "triton_kernel",
    """
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i8', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_kernel', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 8, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '43BC6A92279F7119AB953E3C1B0818031BDC768AC57E822187055CE52B5863C4', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_kernel(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2)
    x1 = ((xindex // 2) % 6)
    x2 = xindex // 12
    x3 = xindex // 2
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 2*((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + ((4) * ((4) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (4))))) + ((-1) + ((4) * ((4) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (4)))) * (((-1) + ((4) * ((4) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (4)))) < (((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))))) + 8*((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) * ((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) <= ((-1) + ((2) * ((2) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (2))))) + ((-1) + ((2) * ((2) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (2)))) * (((-1) + ((2) * ((2) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (2)))) < (((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0))))))), xmask)
    tl.static_assert(tmp0.dtype == tl.int8)
    tmp6 = tl.load(in_ptr1 + (x0 + 2*((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + ((4) * ((4) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (4))))) + ((-1) + ((4) * ((4) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (4)))) * (((-1) + ((4) * ((4) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (4)))) < (((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))))) + 8*((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) * ((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) <= ((-1) + ((2) * ((2) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (2))))) + ((-1) + ((2) * ((2) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (2)))) * (((-1) + ((2) * ((2) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (2)))) < (((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0))))))), xmask)
    tl.static_assert(tmp6.dtype == tl.float32)
    tmp12 = tl.load(in_ptr0 + (x0 + 2*((1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + ((4) * ((4) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (4))))) + ((-1) + ((4) * ((4) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (4)))) * (((-1) + ((4) * ((4) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (4)))) < (1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))))) + 8*((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) * ((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) <= ((-1) + ((2) * ((2) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (2))))) + ((-1) + ((2) * ((2) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (2)))) * (((-1) + ((2) * ((2) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (2)))) < (((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0))))))), xmask)
    tl.static_assert(tmp12.dtype == tl.int8)
    tmp17 = tl.load(in_ptr1 + (x0 + 2*((1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + ((4) * ((4) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (4))))) + ((-1) + ((4) * ((4) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (4)))) * (((-1) + ((4) * ((4) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (4)))) < (1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))))) + 8*((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) * ((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) <= ((-1) + ((2) * ((2) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (2))))) + ((-1) + ((2) * ((2) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (2)))) * (((-1) + ((2) * ((2) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (2)))) < (((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0))))))), xmask)
    tl.static_assert(tmp17.dtype == tl.float32)
    tmp30 = tl.load(in_ptr0 + (x0 + 2*((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + ((4) * ((4) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (4))))) + ((-1) + ((4) * ((4) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (4)))) * (((-1) + ((4) * ((4) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (4)))) < (((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))))) + 8*((1 + ((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) * ((1 + ((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) <= ((-1) + ((2) * ((2) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (2))))) + ((-1) + ((2) * ((2) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (2)))) * (((-1) + ((2) * ((2) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (2)))) < (1 + ((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0))))))), xmask)
    tl.static_assert(tmp30.dtype == tl.int8)
    tmp35 = tl.load(in_ptr1 + (x0 + 2*((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + ((4) * ((4) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (4))))) + ((-1) + ((4) * ((4) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (4)))) * (((-1) + ((4) * ((4) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (4)))) < (((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))))) + 8*((1 + ((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) * ((1 + ((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) <= ((-1) + ((2) * ((2) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (2))))) + ((-1) + ((2) * ((2) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (2)))) * (((-1) + ((2) * ((2) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (2)))) < (1 + ((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0))))))), xmask)
    tl.static_assert(tmp35.dtype == tl.float32)
    tmp46 = tl.load(in_ptr0 + (x0 + 2*((1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + ((4) * ((4) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (4))))) + ((-1) + ((4) * ((4) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (4)))) * (((-1) + ((4) * ((4) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (4)))) < (1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))))) + 8*((1 + ((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) * ((1 + ((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) <= ((-1) + ((2) * ((2) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (2))))) + ((-1) + ((2) * ((2) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (2)))) * (((-1) + ((2) * ((2) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (2)))) < (1 + ((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0))))))), xmask)
    tl.static_assert(tmp46.dtype == tl.int8)
    tmp51 = tl.load(in_ptr1 + (x0 + 2*((1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + ((4) * ((4) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (4))))) + ((-1) + ((4) * ((4) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (4)))) * (((-1) + ((4) * ((4) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (4)))) < (1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))))) + 8*((1 + ((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) * ((1 + ((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) <= ((-1) + ((2) * ((2) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (2))))) + ((-1) + ((2) * ((2) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (2)))) * (((-1) + ((2) * ((2) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (2)))) < (1 + ((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0))))))), xmask)
    tl.static_assert(tmp51.dtype == tl.float32)
    tmp1 = tl.full([XBLOCK], 9, tl.int32)
    tl.static_assert(tmp1.dtype == tl.int32)
    tmp2 = tmp0 + tmp1
    tl.static_assert(tmp2.dtype == tl.int32)
    tl.static_assert(tmp2.dtype == tl.int32)
    tmp3 = tmp0 < 0
    tl.static_assert(tmp3.dtype == tl.int1)
    tl.static_assert(tmp3.dtype == tl.int1)
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.static_assert(tmp4.dtype == tl.int32)
    tl.static_assert(tmp4.dtype == tl.int32)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 9)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 9")
    tmp7 = (-7) + tmp4 + 2*((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + ((4) * ((4) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (4))))) + ((-1) + ((4) * ((4) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (4)))) * (((-1) + ((4) * ((4) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (4)))) < (((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))))) + 3*(tmp4 // 3) + 12*((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) * ((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) <= ((-1) + ((2) * ((2) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (2))))) + ((-1) + ((2) * ((2) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (2)))) * (((-1) + ((2) * ((2) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (2)))) < (((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0))))))
    tl.static_assert(tmp7.dtype == tl.int32)
    tmp8 = x3
    tl.static_assert(tmp8.dtype == tl.int32)
    tmp9 = tmp7 == tmp8
    tl.static_assert(tmp9.dtype == tl.int1)
    tl.static_assert(tmp9.dtype == tl.int1)
    tmp10 = 0.0
    tl.static_assert(tmp10.dtype == tl.float32)
    tl.static_assert(tmp10.dtype == tl.float32)
    tmp11 = tl.where(tmp9, tmp6, tmp10)
    tl.static_assert(tmp11.dtype == tl.float32)
    tl.static_assert(tmp11.dtype == tl.float32)
    tl.static_assert(tmp1.dtype == tl.int32)
    tmp13 = tmp12 + tmp1
    tl.static_assert(tmp13.dtype == tl.int32)
    tl.static_assert(tmp13.dtype == tl.int32)
    tmp14 = tmp12 < 0
    tl.static_assert(tmp14.dtype == tl.int1)
    tl.static_assert(tmp14.dtype == tl.int1)
    tmp15 = tl.where(tmp14, tmp13, tmp12)
    tl.static_assert(tmp15.dtype == tl.int32)
    tl.static_assert(tmp15.dtype == tl.int32)
    tl.device_assert(((0 <= tmp15) & (tmp15 < 9)) | ~(xmask), "index out of bounds: 0 <= tmp15 < 9")
    tmp18 = (-7) + tmp15 + 2*((1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + ((4) * ((4) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (4))))) + ((-1) + ((4) * ((4) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (4)))) * (((-1) + ((4) * ((4) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (4)))) < (1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))))) + 3*(tmp15 // 3) + 12*((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) * ((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) <= ((-1) + ((2) * ((2) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (2))))) + ((-1) + ((2) * ((2) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (2)))) * (((-1) + ((2) * ((2) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (2)))) < (((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0))))))
    tl.static_assert(tmp18.dtype == tl.int32)
    tl.static_assert(tmp8.dtype == tl.int32)
    tmp19 = tmp18 == tmp8
    tl.static_assert(tmp19.dtype == tl.int1)
    tl.static_assert(tmp19.dtype == tl.int1)
    tmp20 = ((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))
    tl.static_assert(tmp20.dtype == tl.int32)
    tmp21 = ((2) * ((2) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (2)))
    tl.static_assert(tmp21.dtype == tl.int32)
    tmp22 = tmp20 < tmp21
    tl.static_assert(tmp22.dtype == tl.int1)
    tl.static_assert(tmp22.dtype == tl.int1)
    tmp23 = 1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))
    tl.static_assert(tmp23.dtype == tl.int32)
    tmp24 = ((4) * ((4) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (4)))
    tl.static_assert(tmp24.dtype == tl.int32)
    tmp25 = tmp23 < tmp24
    tl.static_assert(tmp25.dtype == tl.int1)
    tl.static_assert(tmp25.dtype == tl.int1)
    tmp26 = tmp22 & tmp25
    tl.static_assert(tmp26.dtype == tl.int1)
    tl.static_assert(tmp26.dtype == tl.int1)
    tmp27 = tmp26 & tmp19
    tl.static_assert(tmp27.dtype == tl.int1)
    tl.static_assert(tmp27.dtype == tl.int1)
    tmp28 = tmp11 + tmp17
    tl.static_assert(tmp28.dtype == tl.float32)
    tl.static_assert(tmp28.dtype == tl.float32)
    tmp29 = tl.where(tmp27, tmp28, tmp11)
    tl.static_assert(tmp29.dtype == tl.float32)
    tl.static_assert(tmp29.dtype == tl.float32)
    tl.static_assert(tmp1.dtype == tl.int32)
    tmp31 = tmp30 + tmp1
    tl.static_assert(tmp31.dtype == tl.int32)
    tl.static_assert(tmp31.dtype == tl.int32)
    tmp32 = tmp30 < 0
    tl.static_assert(tmp32.dtype == tl.int1)
    tl.static_assert(tmp32.dtype == tl.int1)
    tmp33 = tl.where(tmp32, tmp31, tmp30)
    tl.static_assert(tmp33.dtype == tl.int32)
    tl.static_assert(tmp33.dtype == tl.int32)
    tl.device_assert(((0 <= tmp33) & (tmp33 < 9)) | ~(xmask), "index out of bounds: 0 <= tmp33 < 9")
    tmp36 = (-7) + tmp33 + 2*((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + ((4) * ((4) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (4))))) + ((-1) + ((4) * ((4) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (4)))) * (((-1) + ((4) * ((4) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (4)))) < (((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))))) + 3*(tmp33 // 3) + 12*((1 + ((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) * ((1 + ((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) <= ((-1) + ((2) * ((2) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (2))))) + ((-1) + ((2) * ((2) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (2)))) * (((-1) + ((2) * ((2) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (2)))) < (1 + ((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0))))))
    tl.static_assert(tmp36.dtype == tl.int32)
    tl.static_assert(tmp8.dtype == tl.int32)
    tmp37 = tmp36 == tmp8
    tl.static_assert(tmp37.dtype == tl.int1)
    tl.static_assert(tmp37.dtype == tl.int1)
    tmp38 = 1 + ((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))
    tl.static_assert(tmp38.dtype == tl.int32)
    tl.static_assert(tmp21.dtype == tl.int32)
    tmp39 = tmp38 < tmp21
    tl.static_assert(tmp39.dtype == tl.int1)
    tl.static_assert(tmp39.dtype == tl.int1)
    tmp40 = ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))
    tl.static_assert(tmp40.dtype == tl.int32)
    tl.static_assert(tmp24.dtype == tl.int32)
    tmp41 = tmp40 < tmp24
    tl.static_assert(tmp41.dtype == tl.int1)
    tl.static_assert(tmp41.dtype == tl.int1)
    tmp42 = tmp39 & tmp41
    tl.static_assert(tmp42.dtype == tl.int1)
    tl.static_assert(tmp42.dtype == tl.int1)
    tmp43 = tmp42 & tmp37
    tl.static_assert(tmp43.dtype == tl.int1)
    tl.static_assert(tmp43.dtype == tl.int1)
    tmp44 = tmp29 + tmp35
    tl.static_assert(tmp44.dtype == tl.float32)
    tl.static_assert(tmp44.dtype == tl.float32)
    tmp45 = tl.where(tmp43, tmp44, tmp29)
    tl.static_assert(tmp45.dtype == tl.float32)
    tl.static_assert(tmp45.dtype == tl.float32)
    tl.static_assert(tmp1.dtype == tl.int32)
    tmp47 = tmp46 + tmp1
    tl.static_assert(tmp47.dtype == tl.int32)
    tl.static_assert(tmp47.dtype == tl.int32)
    tmp48 = tmp46 < 0
    tl.static_assert(tmp48.dtype == tl.int1)
    tl.static_assert(tmp48.dtype == tl.int1)
    tmp49 = tl.where(tmp48, tmp47, tmp46)
    tl.static_assert(tmp49.dtype == tl.int32)
    tl.static_assert(tmp49.dtype == tl.int32)
    tl.device_assert(((0 <= tmp49) & (tmp49 < 9)) | ~(xmask), "index out of bounds: 0 <= tmp49 < 9")
    tmp52 = (-7) + tmp49 + 2*((1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + ((4) * ((4) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (4))))) + ((-1) + ((4) * ((4) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (4)))) * (((-1) + ((4) * ((4) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (4)))) < (1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))))) + 3*(tmp49 // 3) + 12*((1 + ((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) * ((1 + ((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) <= ((-1) + ((2) * ((2) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (2))))) + ((-1) + ((2) * ((2) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (2)))) * (((-1) + ((2) * ((2) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (2)))) < (1 + ((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0))))))
    tl.static_assert(tmp52.dtype == tl.int32)
    tl.static_assert(tmp8.dtype == tl.int32)
    tmp53 = tmp52 == tmp8
    tl.static_assert(tmp53.dtype == tl.int1)
    tl.static_assert(tmp53.dtype == tl.int1)
    tl.static_assert(tmp38.dtype == tl.int32)
    tl.static_assert(tmp21.dtype == tl.int32)
    tl.static_assert(tmp39.dtype == tl.int1)
    tl.static_assert(tmp23.dtype == tl.int32)
    tl.static_assert(tmp24.dtype == tl.int32)
    tl.static_assert(tmp25.dtype == tl.int1)
    tmp54 = tmp39 & tmp25
    tl.static_assert(tmp54.dtype == tl.int1)
    tl.static_assert(tmp54.dtype == tl.int1)
    tmp55 = tmp54 & tmp53
    tl.static_assert(tmp55.dtype == tl.int1)
    tl.static_assert(tmp55.dtype == tl.int1)
    tmp56 = tmp45 + tmp51
    tl.static_assert(tmp56.dtype == tl.float32)
    tl.static_assert(tmp56.dtype == tl.float32)
    tmp57 = tl.where(tmp55, tmp56, tmp45)
    tl.static_assert(tmp57.dtype == tl.float32)
    tl.static_assert(tmp57.dtype == tl.float32)
    tl.store(out_ptr0 + (x5), tmp57, xmask)
""",
    device_str="cuda",
)


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        primals_1, getitem_1, tangents_1 = args
        args.clear()
        assert_size_stride(primals_1, (1, 2, 3, 6), (36, 1, 12, 2))
        assert_size_stride(getitem_1, (1, 2, 2, 4), (16, 1, 8, 2))
        assert_size_stride(tangents_1, (1, 2, 2, 4), (16, 1, 8, 2))
        primals_1 = torch.load("primals.pt", map_location="cuda")
        getitem_1 = torch.load("getitem.pt", map_location="cuda")
        tangents_1 = torch.load("tangents.pt", map_location="cuda")
        # torch.save(primals_1, "primals.pt")
        # torch.save(getitem_1, "getitem.pt")
        # torch.save(tangents_1, "tangents.pt")
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((1, 2, 3, 6), (36, 1, 12, 2), torch.float32)
            # Topologically Sorted Source Nodes: [max_pool2d], Original ATen: [aten.max_pool2d_with_indices, aten.max_pool2d_with_indices_backward]
            # [Provenance debug handles] triton_kernel:229
            stream0 = get_raw_stream(0)
            triton_kernel.run(getitem_1, tangents_1, buf0, 36, stream=stream0)
            del getitem_1
            del tangents_1
        # torch.save(buf0, "out.pt")
        out = torch.load("out.pt", map_location="cuda")
        allclose = torch.allclose(out, buf0, atol=1e-4, rtol=1e-4)
        if allclose:
            print("✅ match")
        else:
            print("❌ differ")
            print(f"Out: {out}")
            print(f"Buf0: {buf0}")

        return (buf0, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=1, repeat=1):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 2, 3, 6), (36, 1, 12, 2), device='cuda:0', dtype=torch.float32)
    getitem_1 = rand_strided((1, 2, 2, 4), (16, 1, 8, 2), device='cuda:0', dtype=torch.int8)
    tangents_1 = rand_strided((1, 2, 2, 4), (16, 1, 8, 2), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, getitem_1, tangents_1])
    return print_performance(fn, times=1, repeat=1)


from torch._inductor.wrapper_benchmark import compiled_module_main
compiled_module_main('None', benchmark_compiled_module)
