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
# empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
# empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool

empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/tmpdx_p7g8o/lf/clfxn6pbsyz4nez6cbgvnpuenbxe5mq22pww7ejjpgzs6bnkkfmh.py
# Topologically Sorted Source Nodes: [max_pool2d_with_indices_backward], Original ATen: [aten.max_pool2d_with_indices_backward]
# Source node to ATen node mapping:
#   max_pool2d_with_indices_backward => max_pool2d_with_indices_backward
# Graph fragment:
#   %arg2_1 : Tensor "i64[2, 4, 21, 29][2436, 609, 29, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg0_1 : Tensor "f32[2, 4, 21, 29][2436, 609, 29, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %max_pool2d_with_indices_backward : Tensor "f32[2, 4, 40, 56][8960, 2240, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.max_pool2d_with_indices_backward.default](args = (%arg0_1, %arg1_1, [3, 3], [2, 2], [1, 1], [1, 1], True, %arg2_1), kwargs = {})
#   return %max_pool2d_with_indices_backward
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_backward_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 17920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 56)
    x1 = ((xindex // 56) % 40)
    x2 = xindex // 2240
    x3 = (xindex % 2240)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (29*((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + ((21) * ((21) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (21))))) + ((-1) + ((21) * ((21) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (21)))) * (((-1) + ((21) * ((21) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (21)))) < (((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))))) + 609*x2 + ((((0) * ((0) >= (x0 // 2)) + (x0 // 2) * ((x0 // 2) > (0)))) * ((((0) * ((0) >= (x0 // 2)) + (x0 // 2) * ((x0 // 2) > (0)))) <= ((-1) + ((29) * ((29) <= (1 + ((1 + x0) // 2))) + (1 + ((1 + x0) // 2)) * ((1 + ((1 + x0) // 2)) < (29))))) + ((-1) + ((29) * ((29) <= (1 + ((1 + x0) // 2))) + (1 + ((1 + x0) // 2)) * ((1 + ((1 + x0) // 2)) < (29)))) * (((-1) + ((29) * ((29) <= (1 + ((1 + x0) // 2))) + (1 + ((1 + x0) // 2)) * ((1 + ((1 + x0) // 2)) < (29)))) < (((0) * ((0) >= (x0 // 2)) + (x0 // 2) * ((x0 // 2) > (0))))))), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (29*((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + ((21) * ((21) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (21))))) + ((-1) + ((21) * ((21) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (21)))) * (((-1) + ((21) * ((21) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (21)))) < (((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))))) + 609*x2 + ((((0) * ((0) >= (x0 // 2)) + (x0 // 2) * ((x0 // 2) > (0)))) * ((((0) * ((0) >= (x0 // 2)) + (x0 // 2) * ((x0 // 2) > (0)))) <= ((-1) + ((29) * ((29) <= (1 + ((1 + x0) // 2))) + (1 + ((1 + x0) // 2)) * ((1 + ((1 + x0) // 2)) < (29))))) + ((-1) + ((29) * ((29) <= (1 + ((1 + x0) // 2))) + (1 + ((1 + x0) // 2)) * ((1 + ((1 + x0) // 2)) < (29)))) * (((-1) + ((29) * ((29) <= (1 + ((1 + x0) // 2))) + (1 + ((1 + x0) // 2)) * ((1 + ((1 + x0) // 2)) < (29)))) < (((0) * ((0) >= (x0 // 2)) + (x0 // 2) * ((x0 // 2) > (0))))))), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (29*((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + ((21) * ((21) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (21))))) + ((-1) + ((21) * ((21) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (21)))) * (((-1) + ((21) * ((21) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (21)))) < (((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))))) + 609*x2 + ((1 + ((0) * ((0) >= (x0 // 2)) + (x0 // 2) * ((x0 // 2) > (0)))) * ((1 + ((0) * ((0) >= (x0 // 2)) + (x0 // 2) * ((x0 // 2) > (0)))) <= ((-1) + ((29) * ((29) <= (1 + ((1 + x0) // 2))) + (1 + ((1 + x0) // 2)) * ((1 + ((1 + x0) // 2)) < (29))))) + ((-1) + ((29) * ((29) <= (1 + ((1 + x0) // 2))) + (1 + ((1 + x0) // 2)) * ((1 + ((1 + x0) // 2)) < (29)))) * (((-1) + ((29) * ((29) <= (1 + ((1 + x0) // 2))) + (1 + ((1 + x0) // 2)) * ((1 + ((1 + x0) // 2)) < (29)))) < (1 + ((0) * ((0) >= (x0 // 2)) + (x0 // 2) * ((x0 // 2) > (0))))))), xmask)
    tmp7 = tl.load(in_ptr1 + (29*((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + ((21) * ((21) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (21))))) + ((-1) + ((21) * ((21) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (21)))) * (((-1) + ((21) * ((21) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (21)))) < (((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))))) + 609*x2 + ((1 + ((0) * ((0) >= (x0 // 2)) + (x0 // 2) * ((x0 // 2) > (0)))) * ((1 + ((0) * ((0) >= (x0 // 2)) + (x0 // 2) * ((x0 // 2) > (0)))) <= ((-1) + ((29) * ((29) <= (1 + ((1 + x0) // 2))) + (1 + ((1 + x0) // 2)) * ((1 + ((1 + x0) // 2)) < (29))))) + ((-1) + ((29) * ((29) <= (1 + ((1 + x0) // 2))) + (1 + ((1 + x0) // 2)) * ((1 + ((1 + x0) // 2)) < (29)))) * (((-1) + ((29) * ((29) <= (1 + ((1 + x0) // 2))) + (1 + ((1 + x0) // 2)) * ((1 + ((1 + x0) // 2)) < (29)))) < (1 + ((0) * ((0) >= (x0 // 2)) + (x0 // 2) * ((x0 // 2) > (0))))))), xmask)
    tmp19 = tl.load(in_ptr0 + (29*((1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + ((21) * ((21) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (21))))) + ((-1) + ((21) * ((21) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (21)))) * (((-1) + ((21) * ((21) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (21)))) < (1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))))) + 609*x2 + ((((0) * ((0) >= (x0 // 2)) + (x0 // 2) * ((x0 // 2) > (0)))) * ((((0) * ((0) >= (x0 // 2)) + (x0 // 2) * ((x0 // 2) > (0)))) <= ((-1) + ((29) * ((29) <= (1 + ((1 + x0) // 2))) + (1 + ((1 + x0) // 2)) * ((1 + ((1 + x0) // 2)) < (29))))) + ((-1) + ((29) * ((29) <= (1 + ((1 + x0) // 2))) + (1 + ((1 + x0) // 2)) * ((1 + ((1 + x0) // 2)) < (29)))) * (((-1) + ((29) * ((29) <= (1 + ((1 + x0) // 2))) + (1 + ((1 + x0) // 2)) * ((1 + ((1 + x0) // 2)) < (29)))) < (((0) * ((0) >= (x0 // 2)) + (x0 // 2) * ((x0 // 2) > (0))))))), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr1 + (29*((1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + ((21) * ((21) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (21))))) + ((-1) + ((21) * ((21) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (21)))) * (((-1) + ((21) * ((21) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (21)))) < (1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))))) + 609*x2 + ((((0) * ((0) >= (x0 // 2)) + (x0 // 2) * ((x0 // 2) > (0)))) * ((((0) * ((0) >= (x0 // 2)) + (x0 // 2) * ((x0 // 2) > (0)))) <= ((-1) + ((29) * ((29) <= (1 + ((1 + x0) // 2))) + (1 + ((1 + x0) // 2)) * ((1 + ((1 + x0) // 2)) < (29))))) + ((-1) + ((29) * ((29) <= (1 + ((1 + x0) // 2))) + (1 + ((1 + x0) // 2)) * ((1 + ((1 + x0) // 2)) < (29)))) * (((-1) + ((29) * ((29) <= (1 + ((1 + x0) // 2))) + (1 + ((1 + x0) // 2)) * ((1 + ((1 + x0) // 2)) < (29)))) < (((0) * ((0) >= (x0 // 2)) + (x0 // 2) * ((x0 // 2) > (0))))))), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr0 + (29*((1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + ((21) * ((21) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (21))))) + ((-1) + ((21) * ((21) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (21)))) * (((-1) + ((21) * ((21) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (21)))) < (1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))))) + 609*x2 + ((1 + ((0) * ((0) >= (x0 // 2)) + (x0 // 2) * ((x0 // 2) > (0)))) * ((1 + ((0) * ((0) >= (x0 // 2)) + (x0 // 2) * ((x0 // 2) > (0)))) <= ((-1) + ((29) * ((29) <= (1 + ((1 + x0) // 2))) + (1 + ((1 + x0) // 2)) * ((1 + ((1 + x0) // 2)) < (29))))) + ((-1) + ((29) * ((29) <= (1 + ((1 + x0) // 2))) + (1 + ((1 + x0) // 2)) * ((1 + ((1 + x0) // 2)) < (29)))) * (((-1) + ((29) * ((29) <= (1 + ((1 + x0) // 2))) + (1 + ((1 + x0) // 2)) * ((1 + ((1 + x0) // 2)) < (29)))) < (1 + ((0) * ((0) >= (x0 // 2)) + (x0 // 2) * ((x0 // 2) > (0))))))), xmask)
    tmp31 = tl.load(in_ptr1 + (29*((1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + ((21) * ((21) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (21))))) + ((-1) + ((21) * ((21) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (21)))) * (((-1) + ((21) * ((21) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (21)))) < (1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))))) + 609*x2 + ((1 + ((0) * ((0) >= (x0 // 2)) + (x0 // 2) * ((x0 // 2) > (0)))) * ((1 + ((0) * ((0) >= (x0 // 2)) + (x0 // 2) * ((x0 // 2) > (0)))) <= ((-1) + ((29) * ((29) <= (1 + ((1 + x0) // 2))) + (1 + ((1 + x0) // 2)) * ((1 + ((1 + x0) // 2)) < (29))))) + ((-1) + ((29) * ((29) <= (1 + ((1 + x0) // 2))) + (1 + ((1 + x0) // 2)) * ((1 + ((1 + x0) // 2)) < (29)))) * (((-1) + ((29) * ((29) <= (1 + ((1 + x0) // 2))) + (1 + ((1 + x0) // 2)) * ((1 + ((1 + x0) // 2)) < (29)))) < (1 + ((0) * ((0) >= (x0 // 2)) + (x0 // 2) * ((x0 // 2) > (0))))))), xmask)
    tmp2 = x3
    tmp3 = tmp0 == tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp3, tmp1, tmp4)
    tmp8 = tmp6 == tmp2
    tmp9 = ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))
    tmp10 = ((21) * ((21) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (21)))
    tmp11 = tmp9 < tmp10
    tmp12 = 1 + ((0) * ((0) >= (x0 // 2)) + (x0 // 2) * ((x0 // 2) > (0)))
    tmp13 = ((29) * ((29) <= (1 + ((1 + x0) // 2))) + (1 + ((1 + x0) // 2)) * ((1 + ((1 + x0) // 2)) < (29)))
    tmp14 = tmp12 < tmp13
    tmp15 = tmp11 & tmp14
    tmp16 = tmp15 & tmp8
    tmp17 = tmp5 + tmp7
    tmp18 = tl.where(tmp16, tmp17, tmp5)
    tmp21 = tmp19 == tmp2
    tmp22 = 1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))
    tmp23 = tmp22 < tmp10
    tmp24 = ((0) * ((0) >= (x0 // 2)) + (x0 // 2) * ((x0 // 2) > (0)))
    tmp25 = tmp24 < tmp13
    tmp26 = tmp23 & tmp25
    tmp27 = tmp26 & tmp21
    tmp28 = tmp18 + tmp20
    tmp29 = tl.where(tmp27, tmp28, tmp18)
    tmp32 = tmp30 == tmp2
    tmp33 = tmp23 & tmp14
    tmp34 = tmp33 & tmp32
    tmp35 = tmp29 + tmp31
    tmp36 = tl.where(tmp34, tmp35, tmp29)
    tl.store(out_ptr0 + (x5), tmp36, xmask)






if __name__ == "__main__":

    # from torch._dynamo.testing import rand_strided

    # arg0_1 = rand_strided((2, 4, 21, 29), (2436, 609, 29, 1), device='cuda:0', dtype=torch.float32)
    # arg2_1 = rand_strided((2, 4, 21, 29), (2436, 609, 29, 1), device='cuda:0', dtype=torch.int64)

    # torch.save(arg0_1, "arg0_1.pt")
    # torch.save(arg2_1, "arg2_1.pt")
    
    arg0_1 = torch.load("arg0_1.pt", map_location="cuda")
    arg2_1 = torch.load("arg2_1.pt", map_location="cuda")

    assert_size_stride(arg0_1, (2, 4, 21, 29), (2436, 609, 29, 1))
    assert_size_stride(arg2_1, (2, 4, 21, 29), (2436, 609, 29, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2, 4, 40, 56), (8960, 2240, 56, 1), torch.float32)
        # Topologically Sorted Source Nodes: [max_pool2d_with_indices_backward], Original ATen: [aten.max_pool2d_with_indices_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_backward_0[(70, 1, 1)](arg2_1, arg0_1, buf0, 17920, 256, num_warps=4, num_stages=1)
        del arg0_1
        del arg2_1

    # Store buf0 at a bad commit
    # torch.save(buf0, "buf0_bad_commit.pt")
 
    buf0_bad_commit = torch.load("buf0_bad_commit.pt", map_location="cuda")
    allclose = torch.allclose(buf0_bad_commit, buf0, atol=1e-4, rtol=1e-4)
    if allclose:
        # NOTE. reference buf0 is collected at a bad commit
        # so we fail the test when allclose is True
        print("❌ Failed")  
    else:
        print("✅ Passed")
