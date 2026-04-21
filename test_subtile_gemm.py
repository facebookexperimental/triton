"""
Run the autoWS addmm (beta GEMM) kernel with subtiled region generation
and verify numerical correctness against a torch reference.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python/test/unit/language"))
from test_autows_addmm import addmm_kernel_tma_persistent_ws, _compute_pid

M, N, K = 512, 512, 256
BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 256
BLOCK_SIZE_K = 64
EPILOGUE_SUBTILE = 2
DATA_PARTITION_FACTOR = 1
FLATTEN = True
num_stages = 2
num_warps = 4
GROUP_SIZE_M = 8
SMEM_ALLOC_ALGO = 0

dtype = torch.float16
device = "cuda"
NUM_SMS = torch.cuda.get_device_properties(device).multi_processor_count

torch.manual_seed(42)
A = torch.randn((M, K), dtype=dtype, device=device)
B = torch.randn((N, K), dtype=dtype, device=device)
bias = torch.randn((M, N), dtype=dtype, device=device)
C = torch.empty((M, N), dtype=dtype, device=device)


def alloc_fn(size, align, stream):
    return torch.empty(size, dtype=torch.int8, device="cuda")


triton.set_allocator(alloc_fn)

a_desc = TensorDescriptor(A, [M, K], [K, 1], [BLOCK_SIZE_M, BLOCK_SIZE_K])
b_desc = TensorDescriptor(B, [N, K], [K, 1], [BLOCK_SIZE_N, BLOCK_SIZE_K])
c_desc = TensorDescriptor(C, C.shape, C.stride(), [BLOCK_SIZE_M, BLOCK_SIZE_N // EPILOGUE_SUBTILE])
bias_desc = TensorDescriptor(bias, [M, N], [N, 1], [BLOCK_SIZE_M, BLOCK_SIZE_N // EPILOGUE_SUBTILE])

grid = lambda META: (min(
    NUM_SMS,
    triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
), )

print(f"Running addmm kernel: M={M}, N={N}, K={K}, "
      f"BLOCK_M={BLOCK_SIZE_M}, BLOCK_N={BLOCK_SIZE_N}")
print(f"  EPILOGUE_SUBTILE={EPILOGUE_SUBTILE}, "
      f"DATA_PARTITION_FACTOR={DATA_PARTITION_FACTOR}")
print("  early_tma_store_lowering=True, generate_subtiled_region=True")

with triton.knobs.nvidia.scope():
    triton.knobs.nvidia.use_meta_ws = True
    triton.knobs.nvidia.use_meta_partition = True

    addmm_kernel_tma_persistent_ws[grid](
        a_desc,
        b_desc,
        c_desc,
        bias_desc,
        M,
        N,
        K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        EPILOGUE_SUBTILE=EPILOGUE_SUBTILE,
        NUM_SMS=NUM_SMS,
        FLATTEN=FLATTEN,
        A_COL_MAJOR=False,
        B_COL_MAJOR=False,
        DATA_PARTITION_FACTOR=DATA_PARTITION_FACTOR,
        SMEM_ALLOC_ALGO=SMEM_ALLOC_ALGO,
        num_stages=num_stages,
        num_warps=num_warps,
        early_tma_store_lowering=True,
        generate_subtiled_region=True,
    )

ref = torch.matmul(A.float(), B.T.float()).to(dtype) + bias
torch.testing.assert_close(C, ref, atol=0.03, rtol=0.03)
print("PASS: numerical results match reference")
