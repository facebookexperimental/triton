import math

import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from typing import Optional
from triton.tools.tensor_descriptor import TensorDescriptor

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def alloc_fn(size: int, align: int, stream: Optional[int]):
    assert align == 128
    assert stream == 0
    return torch.empty(size, dtype=torch.int8, device=DEVICE)


@triton.jit
def _get_bufidx_phase(accum_cnt, NUM_BUFFERS):
    bufIdx = accum_cnt % NUM_BUFFERS
    phase = (accum_cnt // NUM_BUFFERS) & 1
    return bufIdx, phase


def matmul_tma_set_block_size_hook(nargs):
    BLOCK_M = nargs["BM"]
    BLOCK_N = nargs["BN"]
    BLOCK_K = nargs["BK"]
    NUM_MMA_GROUPS = nargs["NUM_MMA_GROUPS"]
    BLOCK_M_SPLIT = BLOCK_M // NUM_MMA_GROUPS
    NUM_CTAS = nargs.get("NUM_CTAS", 1)
    # For column-major inputs, TMA descriptor block shape matches the transposed view
    if nargs.get("A_ROW_MAJOR", True):
        nargs["a_desc"].block_shape = [BLOCK_M_SPLIT, BLOCK_K]
    else:
        nargs["a_desc"].block_shape = [BLOCK_K, BLOCK_M_SPLIT]
    if nargs.get("B_ROW_MAJOR", True):
        nargs["b_desc"].block_shape = [BLOCK_K, BLOCK_N // NUM_CTAS]
    else:
        nargs["b_desc"].block_shape = [BLOCK_N // NUM_CTAS, BLOCK_K]
    EPILOGUE_SUBTILE = nargs.get("EPILOGUE_SUBTILE", False)
    if EPILOGUE_SUBTILE:
        nargs["c_desc"].block_shape = [BLOCK_M_SPLIT, BLOCK_N // 2]
    else:
        nargs["c_desc"].block_shape = [BLOCK_M_SPLIT, BLOCK_N]
    # Add NUM_SMS
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    nargs["NUM_SMS"] = NUM_SMS


def _skinny_zero_c_hook(nargs):
    pass


def _get_skinny_autotune_configs():
    configs = []
    for bm, bn, bk in [
        (128, 64, 64),
        (128, 128, 64),
        (128, 256, 64),
        (64, 256, 64),
        (128, 256, 128),
        (128, 64, 128),
        (128, 128, 128),
        (64, 256, 128),
        (64, 64, 64),
        (64, 128, 64),
        (64, 64, 128),
        (64, 128, 128),
    ]:
        for s in [2, 3, 4]:
            if bk == 128 and s > 2:
                continue
            for gsm in [4, 8]:
                for nw in [4, 8]:
                    configs.append(
                        triton.Config(
                            {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_SIZE_M": gsm, "NUM_STAGES": s},
                            num_warps=nw,
                            num_stages=1,
                            pre_hook=_skinny_zero_c_hook,
                        ))
    return configs


@triton.autotune(
    configs=_get_skinny_autotune_configs(),
    key=["M", "N", "K_LEN"],
)
@triton.jit
def _skinny_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K_LEN,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_ck,
    stride_cm,
    stride_cn,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_STAGES: tl.constexpr,
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

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    k_start = pid_k * K_LEN
    a_ptr += k_start * stride_ak
    b_ptr += k_start * stride_bk

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    buffers_A = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_ptr), NUM_STAGES)
    buffers_B = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(b_ptr), NUM_STAGES)

    for i in tl.range(0, NUM_STAGES - 1, loop_unroll_factor=NUM_STAGES - 1):
        a_buf = tlx.local_view(buffers_A, i)
        b_buf = tlx.local_view(buffers_B, i)
        token_a = tlx.async_load(a_ptrs, a_buf, mask=offs_k[None, :] < K_LEN - i * BLOCK_K)
        token_b = tlx.async_load(b_ptrs, b_buf, mask=offs_k[:, None] < K_LEN - i * BLOCK_K)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        tlx.async_load_commit_group([token_a, token_b])

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K_LEN, BLOCK_K), num_stages=0):
        buf = k % NUM_STAGES
        a_k = tlx.local_view(buffers_A, buf)
        b_k = tlx.local_view(buffers_B, buf)

        tlx.async_load_wait_group(NUM_STAGES - 2)
        acc = tlx.async_dot(a_k, b_k, acc)

        i = k + NUM_STAGES - 1
        a_next = tlx.local_view(buffers_A, i % NUM_STAGES)
        b_next = tlx.local_view(buffers_B, i % NUM_STAGES)
        acc = tlx.async_dot_wait(1, acc)
        token_a = tlx.async_load(a_ptrs, a_next, mask=offs_k[None, :] < K_LEN - i * BLOCK_K)
        token_b = tlx.async_load(b_ptrs, b_next, mask=offs_k[:, None] < K_LEN - i * BLOCK_K)
        tlx.async_load_commit_group([token_a, token_b])
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    acc = tlx.async_dot_wait(0, acc)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    c = acc.to(tl.float16)
    if SPLIT_K > 1:
        c_ptrs = c_ptr + pid_k * stride_ck + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    else:
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=c_mask)


def _skinny_matmul(a, b, M, N, K):
    triton.set_allocator(alloc_fn)
    NUM_SMS = torch.cuda.get_device_properties(a.device).multi_processor_count

    tiles = math.ceil(M / 128) * math.ceil(N / 64)

    split_k = 1
    k_blocks = K // 64
    target_sk = max(1, 2 * NUM_SMS // tiles)
    for sk in [128, 64, 32, 16, 8, 4, 2]:
        if sk <= target_sk and k_blocks // sk >= 16 and tiles * sk >= 8:
            split_k = sk
            break

    k_per_split = K // split_k

    if split_k > 1:
        c = torch.empty((split_k, M, N), dtype=torch.float16, device=a.device)
        stride_ck = M * N
    else:
        c = torch.empty((M, N), dtype=torch.float16, device=a.device)
        stride_ck = 0

    grid = lambda META: (  # noqa: E731
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        split_k,
    )
    _skinny_matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        k_per_split,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        stride_ck,
        c.stride(-2),
        c.stride(-1),
        SPLIT_K=split_k,
    )

    if split_k > 1:
        c = c.sum(dim=0)
    return c


def _skinny_tma_set_block_hook(nargs):
    BM = nargs["BLOCK_M"]
    BN = nargs["BLOCK_N"]
    BK = nargs["BLOCK_K"]
    nargs["a_desc"].block_shape = [BM, BK]
    nargs["b_desc"].block_shape = [BK, BN]


def _get_skinny_tma_configs():
    configs = []
    for bm, bn, bk in [
        (128, 64, 64),
        (128, 128, 64),
        (128, 256, 64),
        (64, 256, 64),
        (64, 64, 64),
        (64, 128, 64),
        (128, 64, 128),
        (128, 128, 128),
        (64, 64, 128),
        (64, 128, 128),
    ]:
        for s in [2, 3, 4]:
            if bk == 128 and s > 2:
                continue
            for gsm in [4, 8]:
                for nw in [4, 8]:
                    configs.append(
                        triton.Config(
                            {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_SIZE_M": gsm, "NUM_STAGES": s},
                            num_warps=nw,
                            num_stages=1,
                            pre_hook=_skinny_tma_set_block_hook,
                        ))
    return configs


@triton.autotune(
    configs=_get_skinny_tma_configs(),
    key=["M", "N", "K_LEN"],
)
@triton.jit
def _skinny_tma_kernel(
    a_desc,
    b_desc,
    c_ptr,
    M,
    N,
    K_LEN,
    stride_ck,
    stride_cm,
    stride_cn,
    K_START,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_STAGES: tl.constexpr,
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

    k_start = pid_k * K_LEN + K_START
    offset_am = pid_m * BLOCK_M
    offset_bn = pid_n * BLOCK_N

    buffers_A = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_desc), NUM_STAGES)
    buffers_B = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(b_desc), NUM_STAGES)

    bars_full_a = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
    bars_full_b = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)

    num_k_iters = tl.cdiv(K_LEN, BLOCK_K)

    for i in tl.range(0, NUM_STAGES - 1, loop_unroll_factor=NUM_STAGES - 1):
        buf_a = tlx.local_view(buffers_A, i)
        buf_b = tlx.local_view(buffers_B, i)
        bar_a = tlx.local_view(bars_full_a, i)
        bar_b = tlx.local_view(bars_full_b, i)

        tlx.barrier_expect_bytes(bar_a, BLOCK_M * BLOCK_K * tlx.size_of(tlx.dtype_of(a_desc)))
        tlx.barrier_expect_bytes(bar_b, BLOCK_K * BLOCK_N * tlx.size_of(tlx.dtype_of(b_desc)))

        offset_k = k_start + i * BLOCK_K
        tlx.async_descriptor_load(a_desc, buf_a, [offset_am, offset_k], bar_a)
        tlx.async_descriptor_load(b_desc, buf_b, [offset_k, offset_bn], bar_b)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, num_k_iters, num_stages=0):
        buf = k % NUM_STAGES
        phase = (k // NUM_STAGES) & 1

        bar_a = tlx.local_view(bars_full_a, buf)
        bar_b = tlx.local_view(bars_full_b, buf)
        tlx.barrier_wait(bar=bar_a, phase=phase)
        tlx.barrier_wait(bar=bar_b, phase=phase)

        a_k = tlx.local_view(buffers_A, buf)
        b_k = tlx.local_view(buffers_B, buf)
        acc = tlx.async_dot(a_k, b_k, acc)

        acc = tlx.async_dot_wait(0, acc)

        next_i = k + NUM_STAGES - 1
        if next_i < num_k_iters:
            next_buf = next_i % NUM_STAGES
            buf_a_next = tlx.local_view(buffers_A, next_buf)
            buf_b_next = tlx.local_view(buffers_B, next_buf)
            bar_a_next = tlx.local_view(bars_full_a, next_buf)
            bar_b_next = tlx.local_view(bars_full_b, next_buf)

            tlx.barrier_expect_bytes(bar_a_next, BLOCK_M * BLOCK_K * tlx.size_of(tlx.dtype_of(a_desc)))
            tlx.barrier_expect_bytes(bar_b_next, BLOCK_K * BLOCK_N * tlx.size_of(tlx.dtype_of(b_desc)))

            offset_k = k_start + next_i * BLOCK_K
            tlx.async_descriptor_load(a_desc, buf_a_next, [offset_am, offset_k], bar_a_next)
            tlx.async_descriptor_load(b_desc, buf_b_next, [offset_k, offset_bn], bar_b_next)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    c = acc.to(tl.float16)
    if SPLIT_K > 1:
        c_ptrs = c_ptr + pid_k * stride_ck + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    else:
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=c_mask)


def _skinny_matmul_tma(a, b, M, N, K):
    triton.set_allocator(alloc_fn)
    NUM_SMS = torch.cuda.get_device_properties(a.device).multi_processor_count

    tiles = math.ceil(M / 128) * math.ceil(N / 64)

    split_k = 1
    k_blocks = K // 64
    target_sk = max(1, 2 * NUM_SMS // tiles)
    for sk in [128, 64, 32, 16, 8, 4, 2]:
        if sk <= target_sk and k_blocks // sk >= 16 and tiles * sk >= 8:
            split_k = sk
            break

    k_per_split = K // split_k

    if split_k > 1:
        c = torch.empty((split_k, M, N), dtype=torch.float16, device=a.device)
        stride_ck = M * N
    else:
        c = torch.empty((M, N), dtype=torch.float16, device=a.device)
        stride_ck = 0

    dummy_block = [1, 1]
    desc_a = TensorDescriptor(a, shape=[M, K], strides=[K, 1], block_shape=dummy_block)
    desc_b = TensorDescriptor(b, shape=[K, N], strides=[N, 1], block_shape=dummy_block)

    grid = lambda META: (  # noqa: E731
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        split_k,
    )
    _skinny_tma_kernel[grid](
        desc_a,
        desc_b,
        c,
        M,
        N,
        k_per_split,
        stride_ck,
        c.stride(-2),
        c.stride(-1),
        K_START=0,
        SPLIT_K=split_k,
    )

    if split_k > 1:
        c = c.sum(dim=0)
    return c


def preprocess_configs(configs, named_args, **kwargs):
    M = named_args["M"]
    N = named_args["N"]
    K = named_args["K"]

    k_iters = K // 64
    if k_iters <= 6:
        filtered = [c for c in configs if c.kwargs.get("NUM_STAGES", 3) <= 2]
        if filtered:
            configs = filtered
    elif k_iters <= 12:
        filtered = [c for c in configs if c.kwargs.get("NUM_STAGES", 3) <= 3]
        if filtered:
            configs = filtered

    min_bm = min(c.kwargs["BM"] for c in configs)
    min_bn = min(c.kwargs["BN"] for c in configs)
    max_tiles = math.ceil(M / min_bm) * math.ceil(N / min_bn)
    if max_tiles < 8:
        filtered = [c for c in configs if c.kwargs.get("NUM_CTAS", 1) == 1]
        if filtered:
            configs = filtered

    IMBALANCE_THRESHOLD = 10
    if M > N * IMBALANCE_THRESHOLD:
        # M >> N: keep only small GROUP_SIZE_M to sweep M, reuse B
        configs = [c for c in configs if c.kwargs["GROUP_SIZE_M"] == 1]
    elif N > M * IMBALANCE_THRESHOLD:
        # N >> M: keep only large GROUP_SIZE_M to sweep N, reuse A
        configs = [c for c in configs if c.kwargs["GROUP_SIZE_M"] >= 32]
    else:
        # Balanced: keep moderate GROUP_SIZE_M for L2 locality
        configs = [c for c in configs if c.kwargs["GROUP_SIZE_M"] == 8]

    return configs


def get_autotune_configs():
    return [
        triton.Config(
            {
                "BM": BM,
                "BN": BN,
                "BK": BK,
                "GROUP_SIZE_M": g,
                "NUM_STAGES": s,
                "NUM_MMA_WARPS": 8,
                "NUM_MMA_GROUPS": 2,
                "EPILOGUE_SUBTILE": epilogue,
                "NUM_CTAS": num_ctas,
                "USE_WARP_BARRIER": uwb,
            },
            num_stages=1,
            num_warps=4,
            pre_hook=matmul_tma_set_block_size_hook,
            ctas_per_cga=(num_ctas, 1, 1),
        )
        for BM in [128, 256]
        for BN in [128, 256]
        for BK in [64]
        for s in [2, 3, 4]
        for epilogue in [True, False]
        for g in [1, 8, 64]
        for num_ctas in [1, 2]
        for uwb in [False, True]
    ]


@triton.autotune(
    configs=get_autotune_configs(),
    key=["M", "N", "K"],
    use_cuda_graph=True,
    prune_configs_by={"early_config_prune": preprocess_configs},
)
@triton.jit
def matmul_kernel_tlx_ws(a_desc, b_desc, c_desc,  #
                         M, N, K,  #
                         BM: tl.constexpr,  #
                         BN: tl.constexpr,  #
                         BK: tl.constexpr,  #
                         GROUP_SIZE_M: tl.constexpr,  #
                         NUM_STAGES: tl.constexpr,  #
                         NUM_MMA_WARPS: tl.constexpr,  #
                         NUM_MMA_GROUPS: tl.constexpr,  #
                         EPILOGUE_SUBTILE: tl.constexpr,  #
                         NUM_CTAS: tl.constexpr,  #
                         NUM_SMS: tl.constexpr,  #
                         USE_WARP_BARRIER: tl.constexpr = False,  #
                         A_ROW_MAJOR: tl.constexpr = True,  #
                         B_ROW_MAJOR: tl.constexpr = True,  #
                         ):
    # Descriptor
    BLOCK_M_SPLIT: tl.constexpr = BM // NUM_MMA_GROUPS

    # Need NUM_STAGES sets of SMEM buffers for A and B
    # where each set contains two for A and one for B.
    # Split A into two in M-dimension to have two consumer tasks for wgmma
    if not A_ROW_MAJOR:
        a = tlx.local_alloc((BK, BLOCK_M_SPLIT), tlx.dtype_of(a_desc), NUM_STAGES * NUM_MMA_GROUPS)
    else:
        a = tlx.local_alloc((BLOCK_M_SPLIT, BK), tlx.dtype_of(a_desc), NUM_STAGES * NUM_MMA_GROUPS)
    if not B_ROW_MAJOR:
        b = tlx.local_alloc((BN, BK), tlx.dtype_of(b_desc), NUM_STAGES)
    else:
        b = tlx.local_alloc((BK, BN), tlx.dtype_of(b_desc), NUM_STAGES)

    # Need NUM_STAGES sets of mbarriers for A and B
    # where each set contains two for A and one for B.
    # Do the above for both empty states and full states respectively.
    if USE_WARP_BARRIER:
        bars_empty_a = tlx.alloc_warp_barrier(num_barriers=NUM_STAGES * NUM_MMA_GROUPS, num_warps=4)
        bars_empty_b = tlx.alloc_warp_barrier(num_barriers=NUM_STAGES, num_warps=4, num_arrivals=NUM_MMA_GROUPS)
    else:
        bars_empty_a = tlx.alloc_barriers(num_barriers=NUM_STAGES * NUM_MMA_GROUPS, arrive_count=1)
        bars_empty_b = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=NUM_MMA_GROUPS)
    bars_full_a = tlx.alloc_barriers(num_barriers=NUM_STAGES * NUM_MMA_GROUPS, arrive_count=1)
    bars_full_b = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)

    # Barriers for cross-CTA synchronization before multicast TMA loads
    if NUM_CTAS == 2:
        cta_bars = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=2)

    # Warp specilization
    with tlx.async_tasks():
        # Producer (async load)
        with tlx.async_task("default"):
            sm_id = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BM)
            num_pid_n = tl.cdiv(N, BN)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            num_tiles = num_pid_m * num_pid_n

            # Persistent loop - each SM processes tiles with stride NUM_SMS
            tile_id = sm_id
            smem_accum_cnt = 0
            while tile_id < num_tiles:
                # Convert tile_id to pid_m and pid_n
                pid = tile_id
                group_id = pid // num_pid_in_group
                first_pid_m = group_id * GROUP_SIZE_M
                group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
                pid_m = first_pid_m + (pid % group_size_m)
                pid_n = (pid % num_pid_in_group) // group_size_m
                offset_am = pid_m * BM
                offset_bn = pid_n * BN

                for k in range(0, tl.cdiv(K, BK)):
                    buf, p = _get_bufidx_phase(smem_accum_cnt, NUM_STAGES)
                    offset_k = k * BK

                    # Async load to a[buf]
                    empty_a_1st = tlx.local_view(bars_empty_a, buf)  # mbar
                    full_a_1st = tlx.local_view(bars_full_a, buf)  # mbar
                    tlx.barrier_wait(bar=empty_a_1st, phase=p ^ 1)  # EmptyBar A1 wait
                    tlx.barrier_expect_bytes(full_a_1st, BLOCK_M_SPLIT * BK * tlx.size_of(tlx.dtype_of(a_desc)))
                    data_a_1st = tlx.local_view(a, buf)  # smem data
                    if not A_ROW_MAJOR:
                        tlx.async_descriptor_load(a_desc, data_a_1st, [offset_k, offset_am], full_a_1st)
                    else:
                        tlx.async_descriptor_load(a_desc, data_a_1st, [offset_am, offset_k], full_a_1st)

                    # Async load to b[buf]
                    empty_b = tlx.local_view(bars_empty_b, buf)
                    full_b = tlx.local_view(bars_full_b, buf)
                    tlx.barrier_wait(bar=empty_b, phase=p ^ 1)
                    tlx.barrier_expect_bytes(full_b, BN * BK * tlx.size_of(tlx.dtype_of(a_desc)))
                    data_b = tlx.local_view(b, buf)

                    if NUM_CTAS == 2:
                        # Sync cluster: ensure both CTAs' buffers are ready for multicast
                        cta_id = tlx.cluster_cta_rank()
                        cta_bar = tlx.local_view(cta_bars, buf)
                        tlx.barrier_arrive(cta_bar, 1)
                        tlx.barrier_arrive(cta_bar, 1, remote_cta_rank=cta_id ^ 1)
                        tlx.barrier_wait(cta_bar, p)

                        # Each CTA loads half of B and multicasts to both CTAs
                        if not B_ROW_MAJOR:
                            if cta_id == 0:
                                buf_b_slice = tlx.local_slice(data_b, [0, 0], [BN // 2, BK])
                            else:
                                buf_b_slice = tlx.local_slice(data_b, [BN // 2, 0], [BN // 2, BK])
                            tlx.async_descriptor_load(
                                b_desc,
                                buf_b_slice,
                                [offset_bn + cta_id * BN // 2, offset_k],
                                full_b,
                                multicast_targets=[cta_id, cta_id ^ 1],
                            )
                        else:
                            if cta_id == 0:
                                buf_b_slice = tlx.local_slice(data_b, [0, 0], [BK, BN // 2])
                            else:
                                buf_b_slice = tlx.local_slice(data_b, [0, BN // 2], [BK, BN // 2])
                            tlx.async_descriptor_load(
                                b_desc,
                                buf_b_slice,
                                [offset_k, offset_bn + cta_id * BN // 2],
                                full_b,
                                multicast_targets=[cta_id, cta_id ^ 1],
                            )
                    else:
                        if not B_ROW_MAJOR:
                            tlx.async_descriptor_load(b_desc, data_b, [offset_bn, offset_k], full_b)
                        else:
                            tlx.async_descriptor_load(b_desc, data_b, [offset_k, offset_bn], full_b)

                    # Async load to a[buf+NUM_STAGES]
                    empty_a_2nd = tlx.local_view(bars_empty_a, buf + NUM_STAGES)
                    full_a_2nd = tlx.local_view(bars_full_a, buf + NUM_STAGES)
                    tlx.barrier_wait(bar=empty_a_2nd, phase=p ^ 1)
                    tlx.barrier_expect_bytes(bar=full_a_2nd,
                                             size=BLOCK_M_SPLIT * BK * tlx.size_of(tlx.dtype_of(a_desc)))
                    data_a_2nd = tlx.local_view(a, buf + NUM_STAGES)  # smem data
                    if not A_ROW_MAJOR:
                        tlx.async_descriptor_load(a_desc, data_a_2nd, [offset_k, offset_am + BLOCK_M_SPLIT], full_a_2nd)
                    else:
                        tlx.async_descriptor_load(a_desc, data_a_2nd, [offset_am + BLOCK_M_SPLIT, offset_k], full_a_2nd)

                    smem_accum_cnt += 1

                # Move to next tile with stride NUM_SMS
                tile_id += NUM_SMS

        # consumers (wgmma + async store)
        with tlx.async_task(num_warps=4, replicate=2):
            sm_id = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BM)
            num_pid_n = tl.cdiv(N, BN)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            num_tiles = num_pid_m * num_pid_n

            # Persistent loop - each SM processes tiles with stride NUM_SMS
            tile_id = sm_id
            smem_accum_cnt = 0
            while tile_id < num_tiles:
                # Convert tile_id to pid_m and pid_n
                pid = tile_id
                group_id = pid // num_pid_in_group
                first_pid_m = group_id * GROUP_SIZE_M
                group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
                pid_m = first_pid_m + (pid % group_size_m)
                pid_n = (pid % num_pid_in_group) // group_size_m
                offset_am = pid_m * BM
                offset_bn = pid_n * BN

                acc = tl.zeros([BM // 2, BN], dtype=tl.float32)
                for k in range(0, tl.cdiv(K, BK)):
                    buf, p = _get_bufidx_phase(smem_accum_cnt, NUM_STAGES)

                    # Wait for TMA load
                    full_a = tlx.local_view(bars_full_a, buf + NUM_STAGES * tlx.async_task_replica_id())  # noqa
                    full_b = tlx.local_view(bars_full_b, buf)
                    tlx.barrier_wait(bar=full_a, phase=p)
                    tlx.barrier_wait(bar=full_b, phase=p)

                    # async_dot
                    data_a = tlx.local_view(a, buf + NUM_STAGES * tlx.async_task_replica_id())  # noqa
                    data_b = tlx.local_view(b, buf)
                    # Transpose SMEM buffers if inputs were column-major
                    a_operand = tlx.local_trans(data_a) if not A_ROW_MAJOR else data_a
                    b_operand = tlx.local_trans(data_b) if not B_ROW_MAJOR else data_b
                    acc = tlx.async_dot(
                        a_operand,
                        b_operand,
                        acc,
                    )
                    # async_wait
                    acc = tlx.async_dot_wait(tl.constexpr(0), acc)

                    # Release buffers
                    empty_a = tlx.local_view(bars_empty_a, buf + NUM_STAGES * tlx.async_task_replica_id())  # noqa
                    empty_b = tlx.local_view(bars_empty_b, buf)
                    tlx.barrier_arrive(empty_a)  # EmptyBar A1 arrive
                    tlx.barrier_arrive(empty_b)

                    smem_accum_cnt += 1

                offset_cm = offset_am + BLOCK_M_SPLIT * tlx.async_task_replica_id()
                if EPILOGUE_SUBTILE:
                    acc = tl.reshape(acc, (BLOCK_M_SPLIT, 2, BN // 2))
                    acc = tl.permute(acc, (0, 2, 1))
                    acc0, acc1 = tl.split(acc)
                    c0 = acc0.to(tlx.dtype_of(c_desc))
                    c_desc.store([offset_cm, offset_bn], c0)
                    c1 = acc1.to(tlx.dtype_of(c_desc))
                    c_desc.store([offset_cm, offset_bn + BN // 2], c1)
                else:
                    c_desc.store([offset_cm, offset_bn], acc.to(tlx.dtype_of(c_desc)))  # noqa

                # Move to next tile with stride NUM_SMS
                tile_id += NUM_SMS


def matmul(a, b, config=None):
    """Matrix multiplication using TLX GEMM kernel."""
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Illegal dimensions of input operands"
    M, K = a.shape
    K, N = b.shape

    triton.set_allocator(alloc_fn)

    (M, N, K) = (a.shape[0], b.shape[1], a.shape[1])

    if config is None and min(M, N) <= 256:
        NUM_SMS = torch.cuda.get_device_properties(DEVICE).multi_processor_count
        ws_tiles = math.ceil(M / 128) * math.ceil(N / 128)
        if ws_tiles < NUM_SMS:
            if K <= 8192:
                return _skinny_matmul_tma(a, b, M, N, K)
            else:
                return _skinny_matmul(a, b, M, N, K)

    # Allocates output.
    c = torch.empty(
        (M, N),
        dtype=torch.float16,
        device=DEVICE,
    )

    # Detect column-major inputs.
    # A column-major (M, K) tensor has strides (1, M); its .T is row-major (K, M).
    a_row_major = a.is_contiguous()
    b_row_major = b.is_contiguous()

    # Get number of SMs
    NUM_SMS = torch.cuda.get_device_properties(DEVICE).multi_processor_count

    dummy_block = [1, 1]
    if not a_row_major:
        a_t = a.T  # (K, M) with strides (M, 1) — row-major
        desc_in_1 = TensorDescriptor(a_t, a_t.shape, a_t.stride(), dummy_block)
    else:
        desc_in_1 = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    if not b_row_major:
        b_t = b.T  # (N, K) with strides (K, 1) — row-major
        desc_in_2 = TensorDescriptor(b_t, b_t.shape, b_t.stride(), dummy_block)
    else:
        desc_in_2 = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    desc_out = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    if config is not None:
        # Set descriptor block shapes according to config
        NUM_MMA_GROUPS = config["NUM_MMA_GROUPS"]
        BLOCK_M_SPLIT = config["BM"] // NUM_MMA_GROUPS
        NUM_CTAS = config.get("NUM_CTAS", 1)
        if a_row_major:
            desc_in_1.block_shape = [BLOCK_M_SPLIT, config["BK"]]
        else:
            desc_in_1.block_shape = [config["BK"], BLOCK_M_SPLIT]
        if b_row_major:
            desc_in_2.block_shape = [config["BK"], config["BN"] // NUM_CTAS]
        else:
            desc_in_2.block_shape = [config["BN"] // NUM_CTAS, config["BK"]]
        if config.get("EPILOGUE_SUBTILE", False):
            desc_out.block_shape = [BLOCK_M_SPLIT, config["BN"] // 2]
        else:
            desc_out.block_shape = [BLOCK_M_SPLIT, config["BN"]]

        # Use persistent kernel with min(NUM_SMS, total_tiles) blocks
        num_pid_m = triton.cdiv(M, config["BM"])
        num_pid_n = triton.cdiv(N, config["BN"])
        total_tiles = num_pid_m * num_pid_n
        grid = (min(NUM_SMS, total_tiles), )
        matmul_kernel_tlx_ws.fn[grid](
            desc_in_1,
            desc_in_2,
            desc_out,
            M,
            N,
            K,
            A_ROW_MAJOR=a_row_major,
            B_ROW_MAJOR=b_row_major,
            NUM_SMS=NUM_SMS,
            **config,
        )
    else:
        # Use persistent kernel with min(NUM_SMS, total_tiles) blocks
        grid = lambda META: (min(NUM_SMS, triton.cdiv(M, META["BM"]) * triton.cdiv(N, META["BN"])), )  # noqa: E731
        matmul_kernel_tlx_ws[grid](
            desc_in_1,
            desc_in_2,
            desc_out,
            M,
            N,
            K,
            A_ROW_MAJOR=a_row_major,
            B_ROW_MAJOR=b_row_major,
            NUM_SMS=NUM_SMS,
        )
    return c
