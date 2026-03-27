#!/usr/bin/env python3
"""
SkinnyGemm: tinygemm-inspired split-K matmul in stock Triton.

Four data points:
  1. cuBLAS         — torch.matmul
  2. stock triton   — standard Triton matmul (no split-K)
  3. skinny_atomic  — split-K with atomic fp16 reduction
  4. skinny_twopass — split-K with TwoPass: fp32 scratch + reduction kernel

Tinygemm ideas (D89012710, Jeff Johnson):
  - Target multiple waves of SMs via aggressive split-K
  - TwoPass reduction (no atomics) for clean accumulation
  - Small-ish tiles for high occupancy on skinny shapes
"""
import math
import torch
import triton
import triton.language as tl

DEVICE = "cuda"
NUM_SMS = torch.cuda.get_device_properties(DEVICE).multi_processor_count

# Shared tile config list
_TILE_CONFIGS = [
    # (BM, BN, BK, stages, warps)
    (32, 32, 64, 4, 4),
    (32, 32, 128, 2, 4),
    (64, 32, 64, 4, 4),
    (32, 64, 64, 4, 4),
    (64, 64, 64, 4, 4),
    (64, 64, 128, 2, 4),
    (128, 32, 64, 4, 4),
    (32, 128, 64, 4, 4),
    (128, 64, 64, 4, 8),
    (128, 128, 64, 3, 8),
    (128, 128, 128, 2, 8),
]


def _compute_split_k(M, N, K, target_waves=4):
    tiles = math.ceil(M / 64) * math.ceil(N / 64)
    split_k = 1
    if tiles < NUM_SMS * target_waves:
        target_sk = max(1, (NUM_SMS * target_waves) // tiles)
        for sk in [256, 128, 64, 32, 16, 8, 4, 2]:
            if sk <= target_sk and K // sk >= 128:
                split_k = sk
                break
    return split_k


# =========================================================================== #
# Stock Triton matmul (no split-K)
# =========================================================================== #


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BK, "GROUP_SIZE_M": 8},
            num_stages=s,
            num_warps=w,
        ) for BM in [128] for BN in [128, 256] for BK in [64, 128] for s in [2, 3, 4] for w in [4, 8]
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _stock_triton_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = acc.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def stock_triton_matmul(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]), )
    _stock_triton_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )
    return c


# =========================================================================== #
# SkinnyGemm ATOMIC: split-K with atomic fp16 reduction
# =========================================================================== #


def _atomic_pre_hook(nargs):
    if nargs["SPLIT_K"] > 1:
        nargs["c_ptr"].zero_()


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_SIZE_M": 8},
            num_stages=s,
            num_warps=w,
            pre_hook=_atomic_pre_hook,
        ) for bm, bn, bk, s, w in _TILE_CONFIGS
    ],
    key=["M", "N", "K_PER_SPLIT"],
)
@triton.jit
def _skinny_atomic_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    K_PER_SPLIT,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    k_start = pid_k * K_PER_SPLIT
    k_end = min(k_start + K_PER_SPLIT, K)
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + (k_start + offs_k[None, :]) * stride_ak
    b_ptrs = b_ptr + (k_start + offs_k[:, None]) * stride_bk + offs_bn[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K_PER_SPLIT, BLOCK_K)):
        k_remaining = k_end - (k_start + k * BLOCK_K)
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = acc.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=c_mask)


def skinny_atomic_matmul(a, b):
    M, K = a.shape
    _, N = b.shape
    split_k = _compute_split_k(M, N, K)
    k_per_split = (K + split_k - 1) // split_k
    if split_k > 1:
        c = torch.zeros((M, N), device=a.device, dtype=torch.float16)
    else:
        c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        split_k,
    )
    _skinny_atomic_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        k_per_split,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        SPLIT_K=split_k,
    )
    return c


# =========================================================================== #
# SkinnyGemm TWOPASS: split-K with fp32 scratch buffer + reduction kernel
# =========================================================================== #

# --- Pass 1: Compute partial results into fp32 scratch buffer ---
# scratch layout: [split_k, M, N] in fp32


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_SIZE_M": 8},
            num_stages=s,
            num_warps=w,
        ) for bm, bn, bk, s, w in _TILE_CONFIGS
    ],
    key=["M", "N", "K_PER_SPLIT"],
)
@triton.jit
def _twopass_compute_kernel(
    a_ptr,
    b_ptr,
    scratch_ptr,
    M,
    N,
    K,
    K_PER_SPLIT,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_sm,  # scratch stride for M dim (within one split-k slice)
    stride_sn,  # scratch stride for N dim
    stride_sk,  # scratch stride between split-k slices (= M * N)
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    k_start = pid_k * K_PER_SPLIT
    k_end = min(k_start + K_PER_SPLIT, K)
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + (k_start + offs_k[None, :]) * stride_ak
    b_ptrs = b_ptr + (k_start + offs_k[:, None]) * stride_bk + offs_bn[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K_PER_SPLIT, BLOCK_K)):
        k_remaining = k_end - (k_start + k * BLOCK_K)
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    # Store fp32 partial result into scratch[pid_k, :, :]
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    scratch_ptrs = scratch_ptr + pid_k * stride_sk + offs_cm[:, None] * stride_sm + offs_cn[None, :] * stride_sn
    tl.store(scratch_ptrs, acc, mask=c_mask)


# --- Pass 2: Reduce scratch[split_k, M, N] -> output[M, N] in fp16 ---


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=w)
        for bm in [32, 64, 128]
        for bn in [32, 64, 128]
        for w in [4, 8]
        if bm * bn <= 128 * 128
    ],
    key=["M", "N", "SPLIT_K"],
)
@triton.jit
def _twopass_reduce_kernel(
    scratch_ptr,
    c_ptr,
    M,
    N,
    stride_sm,
    stride_sn,
    stride_sk,
    stride_cm,
    stride_cn,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    # Sum across split-K slices
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for sk in range(SPLIT_K):
        s_ptrs = scratch_ptr + sk * stride_sk + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn
        partial = tl.load(s_ptrs, mask=mask, other=0.0)
        acc += partial
    # Store as fp16
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask)


def skinny_twopass_matmul(a, b):
    M, K = a.shape
    _, N = b.shape
    split_k = _compute_split_k(M, N, K)
    k_per_split = (K + split_k - 1) // split_k

    if split_k <= 1:
        # No split-K needed, just use a simple matmul (reuse atomic kernel with SPLIT_K=1)
        c = torch.empty((M, N), device=a.device, dtype=torch.float16)
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
            1,
        )
        _skinny_atomic_kernel[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            SPLIT_K=1,
        )
        return c

    # Pass 1: compute partials into fp32 scratch buffer [split_k, M, N]
    scratch = torch.empty((split_k, M, N), device=a.device, dtype=torch.float32)
    grid1 = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        split_k,
    )
    _twopass_compute_kernel[grid1](
        a,
        b,
        scratch,
        M,
        N,
        K,
        k_per_split,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        scratch.stride(1),
        scratch.stride(2),
        scratch.stride(0),
        SPLIT_K=split_k,
    )

    # Pass 2: reduce across split_k -> fp16 output
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid2 = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]), )
    _twopass_reduce_kernel[grid2](
        scratch,
        c,
        M,
        N,
        scratch.stride(1),
        scratch.stride(2),
        scratch.stride(0),
        c.stride(0),
        c.stride(1),
        SPLIT_K=split_k,
    )
    return c


# =========================================================================== #
# Benchmark
# =========================================================================== #

SKINNY_SHAPES = [
    (256, 256, 589824),
    (256, 128, 589824),
    (768, 256, 73728),
    (1024, 256, 73728),
    (192, 448, 294912),
    (192, 448, 147456),
    (256, 256, 294912),
    (192, 448, 442368),
    (589824, 256, 128),
]

LARGE_SHAPES = [
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
]


def check_correctness(fn, a, b, name):
    out = fn(a, b)
    ref = torch.matmul(a, b)
    max_err = (out.float() - ref.float()).abs().max().item()
    ref_max = ref.float().abs().max().item()
    rel_err = max_err / ref_max if ref_max > 0 else 0
    if rel_err > 0.02:
        print(f"  WARN {name}: rel_err={rel_err:.4f}")


def main():
    gpu_name = torch.cuda.get_device_name()
    cc = torch.cuda.get_device_capability()
    print(f"GPU: {gpu_name} (sm_{cc[0]}{cc[1]})")
    print(f"SMs: {NUM_SMS}")
    print()

    all_shapes = SKINNY_SHAPES + LARGE_SHAPES

    providers = [
        ("cuBLAS", lambda a, b: torch.matmul(a, b)),
        ("triton", stock_triton_matmul),
        ("atomic", skinny_atomic_matmul),
        ("twopass", skinny_twopass_matmul),
    ]
    pnames = [p[0] for p in providers]

    results = []
    for M, N, K in all_shapes:
        shape_str = f"{M}x{N}x{K}"
        a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
        b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)

        sk = _compute_split_k(M, N, K)
        print(f"Shape {shape_str:>28s}  split_k={sk}")

        row = {"shape": shape_str, "M": M, "N": N, "K": K, "split_k": sk}

        for name, fn in providers:
            print(f"  {name:>10s}...", end="", flush=True)
            try:
                check_correctness(fn, a, b, name)
                ms = triton.testing.do_bench(lambda fn=fn: fn(a, b), warmup=200, rep=500)
                row[name] = ms
                print(f" {ms:.3f} ms")
            except Exception as e:
                row[name] = None
                print(f" FAIL ({e})")

        results.append(row)
        del a, b
        torch.cuda.empty_cache()

    # Results table
    print(f"\n{'='*120}")
    hdr = f"{'Shape':>28s}  {'sk':>3s}  {'cuBLAS':>7s}"
    for p in pnames[1:]:
        hdr += f"  {p:>8s} {'spd':>5s}"
    print(hdr)
    print("-" * 120)

    geos = {p: [] for p in pnames[1:]}
    n_skinny = len(SKINNY_SHAPES)

    for row in results:
        cu = row.get("cuBLAS")
        line = f"{row['shape']:>28s}  {row['split_k']:>3d}"
        line += f"  {cu:7.3f}" if cu else f"  {'N/A':>7s}"
        for p in pnames[1:]:
            ms = row.get(p)
            if ms is not None and cu is not None:
                spd = cu / ms
                line += f"  {ms:8.3f} {spd:4.2f}x"
                geos[p].append(spd)
            elif ms is not None:
                line += f"  {ms:8.3f}  {'':>5s}"
            else:
                line += f"  {'FAIL':>8s}  {'':>5s}"
        print(line)

    def geo(vals):
        return math.exp(sum(math.log(x) for x in vals) / len(vals)) if vals else 0

    print("-" * 120)
    geo_line = f"{'All geo':>28s}  {'':>3s}  {'':>7s}"
    for p in pnames[1:]:
        geo_line += f"  {'':>8s} {geo(geos[p]):4.2f}x"
    print(geo_line)

    geo_line2 = f"{'Skinny geo':>28s}  {'':>3s}  {'':>7s}"
    for p in pnames[1:]:
        s = geos[p][:n_skinny]
        geo_line2 += f"  {'':>8s} {geo(s):4.2f}x"
    print(geo_line2)

    # Wins
    for p in pnames[1:]:
        w = sum(1 for x in geos[p] if x >= 1.0)
        print(f"  {p} vs cuBLAS: {w}/{len(geos[p])} wins")


if __name__ == "__main__":
    main()
