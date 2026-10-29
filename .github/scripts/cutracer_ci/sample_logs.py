"""Compute Sanitizer log shapes shared by the tests."""

# Real compute-sanitizer output shapes, taken from CUTracer's parser fixtures
# (python/tests/example_inputs/racecheck_*.txt in facebookexperimental/CUTracer).
RACECHECK_HAZARD = """\
========= COMPUTE-SANITIZER
========= Error: Race reported between Write access at kernel_a+0x100 in kernels.py:42
child process output may be interleaved
=========     and Read access at kernel_a+0x200 in kernels.py:57 [3 hazards]
=========     and Read access at kernel_a+0x240 in kernels.py:58 [5 hazards]
=========
========= Warning: Maximum number of hazards reached.
=========
========= Warning: Race reported between Read access at kernel_b+0x300 in other.py:9
=========     and Write access at kernel_b+0x340 in other.py:12 [2 hazards]
=========
1 failed, 1 warning in 12.34s
========= Target application returned an error
========= RACECHECK SUMMARY: 7 hazards displayed (6 errors, 1 warning)
"""
RACECHECK_CLEAN = """\
========= COMPUTE-SANITIZER
7 passed in 5.01s
========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)
"""
SYNCCHECK_ERROR = """\
========= COMPUTE-SANITIZER
========= Barrier error detected. Missing wait.
=========     at _attn_fwd_ws+0x1a40 in sm100.py:812
=========
========= ERROR SUMMARY: 1 error
"""
SYNCCHECK_CLEAN = "========= COMPUTE-SANITIZER\n========= ERROR SUMMARY: 0 errors\n"
# memcheck on the B200 runner (tlx.ops flash-attention forward): Triton's
# launcher probing cuFuncSetAttribute, which the test survives.
MEMCHECK_API_ONLY = """\
========= COMPUTE-SANITIZER
========= Program hit CUDA_ERROR_INVALID_VALUE (error 1) due to "invalid argument" on CUDA API call to cuFuncSetAttribute.
=========     Saved host backtrace up to driver entry point at error
=========         Host Frame: loadBinary [0x6f43] in cuda_utils.cpython-312-x86_64-linux-gnu.so
=========
========= Program hit CUDA_ERROR_INVALID_VALUE (error 1) due to "invalid argument" on CUDA API call to cuFuncSetAttribute.
=========     Saved host backtrace up to driver entry point at error
=========
========= ERROR SUMMARY: 2 errors
"""
MEMCHECK_MIXED = MEMCHECK_API_ONLY.replace(
    "========= ERROR SUMMARY: 2 errors",
    "========= Invalid __global__ read of size 4 bytes\n"
    "=========     at kernel+0x10 in k.py:3\n"
    "========= ERROR SUMMARY: 3 errors",
)
# Errors outside the kernel under test on the B200 runner: a cuBLAS launch
# synccheck could not track (from a test's torch reference matmul), and a
# runtime-API out-of-memory error under memcheck.
OUTSIDE_THE_KERNEL = """\
========= COMPUTE-SANITIZER
========= Internal Sanitizer Error: The Sanitizer encountered an error while launching nvjet_sm100_hsh_64x8_64x16_1x4_h_bz_NNT and didn't track the launch. Errors might go undetected. (One or more of the parameters is invalid)
=========
========= Program hit cudaErrorMemoryAllocation (error 2) due to "out of memory" on CUDA API call to cudaMalloc.
=========     Saved host backtrace up to driver entry point at error
=========         Host Frame: c10::cuda::CUDACachingAllocator::Native::DeviceCachingAllocator::malloc [0x519b0] in libc10_cuda.so
=========
========= ERROR SUMMARY: 2 errors
"""

# A 2-CTA TMA load in CTA 1 signalling the pair's mbarrier in CTA 0, past the
# print limit.
PEER_BARRIER = """\
========= COMPUTE-SANITIZER
========= Invalid __remote_shared__ write of size 8 bytes
=========     at matmul_kernel_tma_ws_blackwell+0xbb0 in sm100.py:327
=========     by thread (0,0,0) in block (1,0,0)
=========     Barrier at 0x4400 in CTA 0 is not located in destination CTA
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame: triton_launch_kernel [0xbff9] in cuda_utils.cpython-312-x86_64-linux-gnu.so
=========
========= ERROR SUMMARY: 2370 errors
========= ERROR SUMMARY: 2369 errors were not printed. Use --print-limit option to adjust the number of printed errors
"""
