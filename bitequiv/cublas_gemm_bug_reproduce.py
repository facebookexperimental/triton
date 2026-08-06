#!/usr/bin/env python3
r"""Two fp16 cuBLASLt matrix multiplies that each leave the last 8 values of k out of the sum.

A and B are all ones, so every element of C must be exactly K:

    C[m][n] = sum over k = 0 .. K-1 of 1 * 1 = K

                K = 8648            N = 8                N = 8
            +-------------+      +---------+       +---------------+
     M = 1  | 1 1 ... 1 1 | x    | 1 ... 1 |  =    | 8648 ... 8648 |  M = 1
            +-------------+      | 1 ... 1 |       +---------------+
                   A             |   ...   |               C
                                 | 1 ... 1 |  K = 8648
                                 +---------+
                                      B

Nothing rounds. A product of two ones is exact in fp32, the accumulator is fp32, and the
output is fp32, which represents every whole number up to 2^24 exactly. So the number that
comes back *is* the count of k values that were summed. It comes back short by a whole number,
which is that many missing terms of `1 * 1` rather than a rounding error of any size.

The three parts below are the same code with a different K. Point `CUBLASLT` at one library,
run once, and every part runs: the ones that match your GPU and library come back short and the
rest come back right.

    part one   K =  8648  -> 8640    sm_100 or sm_103, cuBLASLt 13.1.1 / 13.2.2
    part two   K = 57608  -> 57600   sm_103 or sm_90,  cuBLASLt 12.8.5
    part three K = 11528  -> 11520   sm_90,            cuBLASLt 12.8.5 AND 13.1.1

Measured, `.` meaning correct and `-8` meaning eight values of k left out:

                       sm_100     sm_103     sm_103     sm_90      sm_90
                       13.1.1     13.1.x     12.8.5     13.1.1     12.8.5
    part one   8648      -8         -8          .          .          .
    part two  57608       ?          .         -8          .         -8
    part three 11528      ?          ?          ?         -8         -8

WHY NO ONE SHAPE COVERS EVERYTHING. It is not that some versions are fixed. The loss happens
only where cuBLASLt decides to split K across threadblocks, and that decision moves with both
the architecture and the library. On GB300 the two libraries were nearly disjoint -- over 12,282
shapes, 13.1.1 met the condition on 193 and 12.8.5 on 309, with **zero** overlap; at K = 8648
12.8.5 returns SPLITK_NUM = 1 and never splits, and at K = 57608 the roles swap. On H100 they
almost coincide instead: over 1,542,555 real GEMMs each, 13.1.1 loses k on 2,714 shapes and
12.8.5 on 2,995, the smallest is K = 11528 for both, and they only diverge past about K = 50000.
So the shape has to be chosen per architecture and per library, which is what the three parts
are.

THE CONDITION, the same everywhere. `block_k` is the k step of one threadblock, 64 for this
fp16 kernel family; `q = K // block_k`; `tail = K % block_k`; `s = SPLITK_NUM`, the number of
splits cuBLASLt picks. The tail is lost exactly when

    ALGO_ID == 66   and   tail != 0   and   q % s == 0   and   s > tail

    part one   q = 135, tail = 8, s =  9    135 %  9 == 0,  9 > 8
    part two   q = 900, tail = 8, s = 45    900 % 45 == 0, 45 > 8   (on sm_90; s = 9 on sm_103)
    part three q = 180, tail = 8, s =  9    180 %  9 == 0,  9 > 8

The `ALGO_ID` part is not decoration. On sm_90, `1 x 2 x 1032` satisfies all three arithmetic
parts and loses nothing, because it lands on `ALGO_ID 23`, a CUTLASS
`cutlass_80_wmma_tensorop_s161616gemm_f16_32x32_64x2_nn_align2` kernel rather than on `nvjet`.
Over 1,542,555 real GEMMs per library on H100 the four-part condition predicted every loss with
**0 false positives and 0 false negatives**; dropping the `ALGO_ID` part misclassified 3,557
shapes under each library. Every one of the 5,709 losses was exactly `K % block_k`.

WHERE THE PIECE IS LOST, drawn for part one. K is handed out to the 9 threadblocks in whole
blocks of 64. 8648 holds 135 whole blocks, and 135 / 9 = 15 exactly, so each takes 15 * 64 = 960:

    k=0       960     1920    2880    3840    4800    5760    6720    7680    8640 8648
      |-------|-------|-------|-------|-------|-------|-------|-------|-------|xxxx|
        CTA 0   CTA 1   CTA 2   CTA 3   CTA 4   CTA 5   CTA 6   CTA 7   CTA 8    ^
                                                                                 |
                                                        these 8 belong to no CTA

Every threadblock is full. The last one covers [7680, 8640), which is 15 whole blocks of 64 with
nothing short about the last one:

    CTA 8:      7680   7744           8512   8576   8640   8648
                |--64--|--64--|  ...  |--64--|--64--|xxxxxx|
                   t0     t1            t13    t14     ^
                                                       |
                                            outside every CTA

And the reduction adds all 9 partial results; not one is skipped:

    partial 0   partial 1   partial 2   ...   partial 8      (9 results, M x N each)
          \          |           |             /
           \_________|___________|____________/
                       |
              splitKreduce_kernel adds all 9
                       |
                    C[m][n]

Nothing computed is thrown away, and nothing is added twice. The 9 ranges simply do not cover K:
9 * 960 = 8640, and K is 8648. Part two is the same picture with 9 * 6400 = 57600 against 57608.

THE ONE THING YOU MUST NOT DROP is the workspace. Split-K writes its partial results into it,
so with a zero-size workspace cuBLASLt never splits and both shapes come back right -- measured,
0 bytes gives 8648 and 57608. The script passes no algorithm at all: `cublasLtMatmul` takes NULL
for `algo` and cuBLASLt chooses for itself. It picks the same split either way, so the heuristic
query is no part of the bug and is not in the code; the `SPLITK_NUM` and the kernel names quoted
here were read with a separate query.

WHICH KERNELS RUN. Two, which is what split-K means: one GEMM that splits K and writes one
partial result per threadblock, then one reduction that adds those partials. One launch each
(names from `torch.profiler`):

    13.1.1 / 13.2.2 on sm_103   nvjet_sm103_hsh_64x8_64x16_1x1_h_bz_splitK_NNT
    13.1.1          on sm_100   nvjet_sm100_hsh_64x8_64x16_1x1_h_bz_splitK_NNT
    13.1.1          on sm_90    nvjet_sm90_hss_64x8_64x16_1x1_h_bz_splitK_NNT
    12.8.5          on sm_103   nvjet_hsh_64x8_64x16_1x1_v_bz_splitK_NNT
    12.8.5          on sm_90    nvjet_hss_64x8_64x16_1x1_v_bz_splitK_NNT
    all of them                 cublasLt::splitKreduce_kernel<32, 16, int, float, ...>

All of them are the same tile family `64x8_64x16_1x1`, same `bz`, `splitK`, `NNT`. The 13.x
names differ from each other only in the architecture tag -- the GEMM kernel is a separate
binary per architecture, since nvjet is built at run time, yet the tile shape, the block of k,
the cluster and the layouts are the same, so they do the same arithmetic in the same order and
lose the same values. 12.8.5's name carries **no** architecture tag at all on either GPU, and
rasters `v` where 13.x rasters `h`. The third letter follows the output type, `hss` for the
fp32 output used here and `hsh` for fp16.

Reading the `64x16` group as <block of k>x<stages> gives the block of 64. Do not take that from
the `cublasLtMatmulStages_t` enum -- its `block_k = 16 << ((id-1)//6)` rule covers ids 1..24 and
`STAGES_ID` here is 35. On sm_90 the 64 was confirmed three independent ways: the kernel name;
two discriminating measurements (`K = 57664`, a multiple of 64, loses nothing, which rules out
128; `K = 57632`, a multiple of 32, loses 32, which rules out 32); and every one of the 5,709
observed losses being exactly `K % 64`.

WHY THE OUTPUT IS fp32. An fp16 output cannot be trusted at part two's depth. Above 16384 the
fp16 step is 16, so a K of the form `q * 64 + 8` sits exactly halfway between two fp16 values
and rounds down to `q * 64` on its own. The control is K = 57616, a shape that is **correct**:
with an fp16 output it reads 57600 and looks 16 short; with an fp32 output it reads 57616.
Asking for fp32 does not change what the heuristic picks -- checked on 600 random shapes, all
600 returned a bit-identical nine-field config either way, and at both shapes here.

IT IS ALSO NOT THREE HAND-PICKED SHAPES. Over 496,906 random shapes on a GB300 under 13.1.1,
1,062 came back not bit-identical to a full-K reference, and 1,059 of those meet the condition
above. Under 12.8.5 on the same GPU, 179 shapes were flagged and all 179 lose exactly `tail`
values of k, while 9,525,600 GEMMs at smaller K lose nothing. On H100, 2,714 shapes under 13.1.1
and 2,995 under 12.8.5 lose k out of 1,542,555 real GEMMs each. It is not confined to skinny
gemv-like shapes either: on H100, 363 of 885 (M, N) pairs under 13.1.1 and 416 of 885 under
12.8.5 lose k somewhere, and 32 x 32, 64 x 64 and 128 x 128 all lose 8 at K = 11528.

`torch.matmul` does not show any of this -- its path never picks this algorithm, so the call has
to go to cuBLASLt directly. Checked on torch 2.9.1 over eight shapes, four operand layouts,
`matmul` / `addmm` / `linear`, and with `preferred_blas_library("cublaslt")`.

Point `CUBLASLT` at one library and run. Every part runs every time, so the printed lines are
themselves the table above for your machine: the parts that match your GPU and library come back
short, the rest come back right.

    export CUBLASLT=/usr/local/cuda-13.0/lib64/libcublasLt.so   # 13.1.1 or 13.2.2
    export CUBLASLT=/usr/local/cuda-12.8/lib64/libcublasLt.so   # 12.8.5
    python cublas_gemm_bug_reproduce.py

Needs a CUDA GPU, and PyTorch for device memory and nothing else.
`cublaslt_splitk_tail_repro.py` next to this file is a longer version with a sweep.
"""

import ctypes
import os
from ctypes import byref, c_float, c_int, c_int64, c_size_t, c_uint64, c_void_p

import torch

# From cublasLt.h.
CUDA_R_16F, CUDA_R_32F, CUBLAS_COMPUTE_32F = 2, 0, 68
CUBLASLT_MATRIX_LAYOUT_ORDER, CUBLASLT_ORDER_ROW = 1, 1
WORKSPACE_BYTES = 32 << 20

# ========================================================================== #
# PART ONE.  cuBLASLt 13.1.1 or 13.2.2, on sm_100 (GB200) or sm_103 (GB300).
#            K = 8648: q = 135, tail = 8, SPLITK_NUM = 9.
# ========================================================================== #

# 1. cuBLASLt setup. Nothing here depends on the matrices.
lib = ctypes.CDLL(os.environ["CUBLASLT"])
handle, desc = c_void_p(), c_void_p()
assert lib.cublasLtCreate(byref(handle)) == 0
assert lib.cublasLtMatmulDescCreate(byref(desc), c_int(CUBLAS_COMPUTE_32F),
                                    c_int(CUDA_R_32F)) == 0            # fp32 accumulate
workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device="cuda")

# 2. The input. A and B all ones, so C must be exactly K. The output is fp32 because an fp16
#    output cannot represent 8648 exactly and would round down on its own.
M, N, K = 1, 8, 8648
a = torch.ones(M, K, dtype=torch.float16, device="cuda")
b = torch.ones(K, N, dtype=torch.float16, device="cuda")
c = torch.empty(M, N, dtype=torch.float32, device="cuda")
layouts = []
for rows, cols, dtype in ((M, K, CUDA_R_16F), (K, N, CUDA_R_16F), (M, N, CUDA_R_32F)):
    layout = c_void_p()
    assert lib.cublasLtMatrixLayoutCreate(byref(layout), c_int(dtype), c_uint64(rows),
                                          c_uint64(cols), c_int64(cols)) == 0
    order = c_int(CUBLASLT_ORDER_ROW)        # cuBLASLt reads column-major unless told this
    assert lib.cublasLtMatrixLayoutSetAttribute(
        layout, c_int(CUBLASLT_MATRIX_LAYOUT_ORDER), byref(order), c_size_t(4)) == 0
    layouts.append(layout)
a_layout, b_layout, c_layout = layouts

# 3. One matrix multiply. `algo` is NULL, so cuBLASLt picks for itself.
alpha, beta = c_float(1.0), c_float(0.0)
assert lib.cublasLtMatmul(handle, desc, byref(alpha), c_void_p(a.data_ptr()), a_layout,
                          c_void_p(b.data_ptr()), b_layout, byref(beta),
                          c_void_p(c.data_ptr()), c_layout, c_void_p(c.data_ptr()), c_layout,
                          None, c_void_p(workspace.data_ptr()), c_size_t(WORKSPACE_BYTES),
                          c_void_p(torch.cuda.current_stream().cuda_stream)) == 0
torch.cuda.synchronize()

# 4. C reads back how many k values were summed. It is short by a whole number.
part_one = c[0, 0].item()
print(f"part one  K = {K}  C[0,0] = {part_one:.0f}  short by {K - part_one:.0f}")

# ========================================================================== #
# PART TWO.  cuBLASLt 12.8.5, on sm_103 (GB300). The same code again, with a different
#            library and a different K = 57608: q = 900, tail = 8, SPLITK_NUM = 9.
# ========================================================================== #

# 1. cuBLASLt setup. Nothing here depends on the matrices.
lib = ctypes.CDLL(os.environ["CUBLASLT"])
handle, desc = c_void_p(), c_void_p()
assert lib.cublasLtCreate(byref(handle)) == 0
assert lib.cublasLtMatmulDescCreate(byref(desc), c_int(CUBLAS_COMPUTE_32F),
                                    c_int(CUDA_R_32F)) == 0            # fp32 accumulate
workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device="cuda")

# 2. The input. A and B all ones, so C must be exactly K. The output is fp32 because an fp16
#    output cannot represent 57608 exactly and would round down on its own.
M, N, K = 1, 8, 57608
a = torch.ones(M, K, dtype=torch.float16, device="cuda")
b = torch.ones(K, N, dtype=torch.float16, device="cuda")
c = torch.empty(M, N, dtype=torch.float32, device="cuda")
layouts = []
for rows, cols, dtype in ((M, K, CUDA_R_16F), (K, N, CUDA_R_16F), (M, N, CUDA_R_32F)):
    layout = c_void_p()
    assert lib.cublasLtMatrixLayoutCreate(byref(layout), c_int(dtype), c_uint64(rows),
                                          c_uint64(cols), c_int64(cols)) == 0
    order = c_int(CUBLASLT_ORDER_ROW)        # cuBLASLt reads column-major unless told this
    assert lib.cublasLtMatrixLayoutSetAttribute(
        layout, c_int(CUBLASLT_MATRIX_LAYOUT_ORDER), byref(order), c_size_t(4)) == 0
    layouts.append(layout)
a_layout, b_layout, c_layout = layouts

# 3. One matrix multiply. `algo` is NULL, so cuBLASLt picks for itself.
alpha, beta = c_float(1.0), c_float(0.0)
assert lib.cublasLtMatmul(handle, desc, byref(alpha), c_void_p(a.data_ptr()), a_layout,
                          c_void_p(b.data_ptr()), b_layout, byref(beta),
                          c_void_p(c.data_ptr()), c_layout, c_void_p(c.data_ptr()), c_layout,
                          None, c_void_p(workspace.data_ptr()), c_size_t(WORKSPACE_BYTES),
                          c_void_p(torch.cuda.current_stream().cuda_stream)) == 0
torch.cuda.synchronize()

# 4. C reads back how many k values were summed. It is short by a whole number.
part_two = c[0, 0].item()
print(f"part two  K = {K}  C[0,0] = {part_two:.0f}  short by {K - part_two:.0f}")

# ========================================================================== #
# PART THREE. cuBLASLt 12.8.5 AND 13.1.1, on sm_90 (H100). The same code again with
#             K = 11528: q = 180, tail = 8, SPLITK_NUM = 9. On H100 the two libraries
#             agree, so this one shape covers both.
# ========================================================================== #

# 1. cuBLASLt setup. Nothing here depends on the matrices.
lib = ctypes.CDLL(os.environ["CUBLASLT"])
handle, desc = c_void_p(), c_void_p()
assert lib.cublasLtCreate(byref(handle)) == 0
assert lib.cublasLtMatmulDescCreate(byref(desc), c_int(CUBLAS_COMPUTE_32F),
                                    c_int(CUDA_R_32F)) == 0            # fp32 accumulate
workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device="cuda")

# 2. The input. A and B all ones, so C must be exactly K. The output is fp32 because an fp16
#    output cannot represent 11528 exactly and would round down on its own.
M, N, K = 1, 8, 11528
a = torch.ones(M, K, dtype=torch.float16, device="cuda")
b = torch.ones(K, N, dtype=torch.float16, device="cuda")
c = torch.empty(M, N, dtype=torch.float32, device="cuda")
layouts = []
for rows, cols, dtype in ((M, K, CUDA_R_16F), (K, N, CUDA_R_16F), (M, N, CUDA_R_32F)):
    layout = c_void_p()
    assert lib.cublasLtMatrixLayoutCreate(byref(layout), c_int(dtype), c_uint64(rows),
                                          c_uint64(cols), c_int64(cols)) == 0
    order = c_int(CUBLASLT_ORDER_ROW)        # cuBLASLt reads column-major unless told this
    assert lib.cublasLtMatrixLayoutSetAttribute(
        layout, c_int(CUBLASLT_MATRIX_LAYOUT_ORDER), byref(order), c_size_t(4)) == 0
    layouts.append(layout)
a_layout, b_layout, c_layout = layouts

# 3. One matrix multiply. `algo` is NULL, so cuBLASLt picks for itself.
alpha, beta = c_float(1.0), c_float(0.0)
assert lib.cublasLtMatmul(handle, desc, byref(alpha), c_void_p(a.data_ptr()), a_layout,
                          c_void_p(b.data_ptr()), b_layout, byref(beta),
                          c_void_p(c.data_ptr()), c_layout, c_void_p(c.data_ptr()), c_layout,
                          None, c_void_p(workspace.data_ptr()), c_size_t(WORKSPACE_BYTES),
                          c_void_p(torch.cuda.current_stream().cuda_stream)) == 0
torch.cuda.synchronize()

# 4. C reads back how many k values were summed. It is short by a whole number.
part_three = c[0, 0].item()
print(f"part three  K = {K}  C[0,0] = {part_three:.0f}  short by {K - part_three:.0f}")

# ========================================================================== #
# Every element of C must be exactly K.
# ========================================================================== #
assert part_one == 8648, f"part one: cuBLASLt summed {part_one:.0f} of 8648"
assert part_two == 57608, f"part two: cuBLASLt summed {part_two:.0f} of 57608"
assert part_three == 11528, f"part three: cuBLASLt summed {part_three:.0f} of 11528"
