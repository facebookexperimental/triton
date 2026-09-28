# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-ignore-all-errors
# Adapted from GEO's 3836160x256x256 production kernel.

"""TLX Blackwell-optimized fused matmul, sigmoid, and multiply kernel.

Computes:
    out[m, n] = 2 * x[m, n] * sigmoid(sum_k x[m, k] * w[k, n])

Also stores the intermediate sigmoid(x @ w) tensor for backward use.

Inputs:
    x: [M, K] bfloat16 (also reused as the gate operand in the epilogue, K must equal N)
    w: [K, N] bfloat16, K == N (square weight)

Outputs:
    out: [M, N] bfloat16 — gated output (2 * x * sigmoid(x @ w))
    s:   [M, N] bfloat16 — saved sigmoid(x @ w) for backward

Optimizations applied (matmul-template style, autotuned per shape):
    S2  Warp specialization (3-way: Producer / MMA / Epilogue)
    S3  CLC persistent scheduling
    S5  Epilogue subtiling along N
    S6  2-CTA cooperative B loading (autotuned, BN=256)
    S7  f32x2 packed math
    S8  SFU sigmoid (tanh.approx)
    S10 TMA epilogue (async TMA stores via SMEM bounce buffers)
    S11 NUM_MMA_GROUPS (split BLOCK_M into M-dim sub-tiles for MMA pipelining)
    S13 INTERLEAVE_EPILOGUE (alternating stores across 2 MMA groups)
"""

import functools
import math
from typing import Any, Dict

import torch
import triton  # @manual=//triton:triton
import triton.language as tl  # @manual=//triton:triton
import triton.language.extra.tlx as tlx  # @manual=//triton:triton
from triton.tools.tensor_descriptor import TensorDescriptor  # @manual=//triton:triton


@triton.jit
def _mul_f32x2(a, b):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b64 ra, rb, rc;
            mov.b64 ra, { $2, $3 };
            mov.b64 rb, { $4, $5 };
            mul.f32x2 rc, ra, rb;
            mov.b64 { $0, $1 }, rc;
        }
        """,
        "=r,=r,r,r,r,r",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=2,
    )


@triton.jit
def _fma_f32x2(a, b, c):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b64 ra, rb, rc, rd;
            mov.b64 ra, { $2, $3 };
            mov.b64 rb, { $4, $5 };
            mov.b64 rc, { $6, $7 };
            fma.rn.f32x2 rd, ra, rb, rc;
            mov.b64 { $0, $1 }, rd;
        }
        """,
        "=r,=r,r,r,r,r,r,r",
        [a, b, c],
        dtype=tl.float32,
        is_pure=True,
        pack=2,
    )


@triton.jit
def tanh_approx_fp32(x):
    output = tl.inline_asm_elementwise(
        asm="""
            tanh.approx.f32 $0, $1;
            """,
        constraints="=r,r",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    return output


@triton.jit
def sigmoid_approx_fp32(x):
    output = _fma_f32x2(0.5, tanh_approx_fp32(_mul_f32x2(0.5, x)), 0.5)
    return output


# ---- Helpers ----


@functools.lru_cache(maxsize=1)
def _get_num_sms():
    return torch.cuda.get_device_properties("cuda").multi_processor_count


@triton.jit
def _get_bufidx_phase(accum_cnt, NUM_BUFFERS):
    bufIdx = accum_cnt % NUM_BUFFERS
    phase = (accum_cnt // NUM_BUFFERS) & 1
    return bufIdx, phase


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


# ---- Autotune ----


def get_autotune_configs():
    """Autotune config space sweeping S5/S6/S11/S13 axes."""
    configs = []
    # NUM_CTAS=1: BM in {128, 256}, BN=128
    for BM, NMG in [(128, 1), (256, 2)]:
        for ep in [1, 2]:
            for g in [1, 4, 8]:
                for interleave in [0, 1]:
                    if interleave and NMG != 2:
                        continue
                    configs.append(
                        triton.Config(
                            {
                                "BLOCK_SIZE_M": BM,
                                "BLOCK_SIZE_K": 128,
                                "BLOCK_SIZE_N": 128,
                                "GROUP_SIZE_M": g,
                                "NUM_SMEM_BUFFERS": 2,
                                "NUM_TMEM_BUFFERS": 2,
                                "NUM_MMA_GROUPS": NMG,
                                "EPILOGUE_SUBTILE": ep,
                                "NUM_CTAS": 1,
                                "INTERLEAVE_EPILOGUE": interleave,
                                "USE_CLC": True,
                            },
                            num_warps=8,
                            num_stages=1,
                            pre_hook=matmul_tma_set_block_size_hook,
                        )
                    )
    # NUM_CTAS=2 (PAIR_CTA): BN=256, GROUP_SIZE_M >= 2
    for BM, NMG in [(128, 1), (256, 2)]:
        for ep in [1, 2]:
            for g in [2, 4, 8]:
                for interleave in [0, 1]:
                    if interleave and NMG != 2:
                        continue
                    configs.append(
                        triton.Config(
                            {
                                "BLOCK_SIZE_M": BM,
                                "BLOCK_SIZE_K": 128,
                                "BLOCK_SIZE_N": 256,
                                "GROUP_SIZE_M": g,
                                "NUM_SMEM_BUFFERS": 2,
                                "NUM_TMEM_BUFFERS": 2,
                                "NUM_MMA_GROUPS": NMG,
                                "EPILOGUE_SUBTILE": ep,
                                "NUM_CTAS": 2,
                                "INTERLEAVE_EPILOGUE": interleave,
                                "USE_CLC": True,
                            },
                            num_warps=8,
                            num_stages=1,
                            pre_hook=matmul_tma_set_block_size_hook,
                            ctas_per_cga=(2, 1, 1),
                        )
                    )
    return configs


def preprocess_configs(configs, named_args, **kwargs):
    """Prune autotune configs against B200 SMEM/TMEM budgets and shape mismatches."""
    MAX_SMEM = 232 * 1024
    MAX_TMEM = 256 * 1024
    M = named_args["M"]
    N = named_args["N"]
    pruned = []
    for conf in configs:
        BM = conf.kwargs["BLOCK_SIZE_M"]
        BN = conf.kwargs["BLOCK_SIZE_N"]
        BK = conf.kwargs["BLOCK_SIZE_K"]
        NUM_SMEM_BUFFERS = conf.kwargs["NUM_SMEM_BUFFERS"]
        NUM_TMEM_BUFFERS = conf.kwargs["NUM_TMEM_BUFFERS"]
        NUM_MMA_GROUPS = conf.kwargs["NUM_MMA_GROUPS"]
        NUM_CTAS = conf.kwargs["NUM_CTAS"]
        EPILOGUE_SUBTILE = conf.kwargs["EPILOGUE_SUBTILE"]
        INTERLEAVE = conf.kwargs["INTERLEAVE_EPILOGUE"]
        BM_SPLIT = BM // NUM_MMA_GROUPS

        # Hardware MMA: max 128 rows per MMA, 2-CTA needs >= 128
        if BM_SPLIT > 128 or BM_SPLIT < 64:
            continue
        if NUM_CTAS == 2 and BM_SPLIT == 64:
            continue
        if INTERLEAVE and (NUM_MMA_GROUPS != 2 or EPILOGUE_SUBTILE < 2):
            continue
        if NUM_CTAS == 2:
            num_tiles = math.ceil(M / BM) * math.ceil(N / BN)
            if num_tiles % 2 != 0:
                continue

        # SMEM budget
        smem_a = BM_SPLIT * BK * 2 * NUM_SMEM_BUFFERS * NUM_MMA_GROUPS
        smem_b = BK * (BN // NUM_CTAS) * 2 * NUM_SMEM_BUFFERS
        smem_x = BM * BN * 2  # full aux x tile
        # Epilogue async store buffers
        NUM_EPILOGUE_SMEM_BUFFERS = 2 if EPILOGUE_SUBTILE > 1 else 1
        SLICE_SIZE = BN // EPILOGUE_SUBTILE
        smem_c_store = BM_SPLIT * SLICE_SIZE * 2 * NUM_EPILOGUE_SMEM_BUFFERS
        smem_s_store = BM_SPLIT * SLICE_SIZE * 2 * NUM_EPILOGUE_SMEM_BUFFERS
        smem_bars = (
            (NUM_SMEM_BUFFERS * NUM_MMA_GROUPS * 2)  # A full + empty
            + NUM_SMEM_BUFFERS  # B full
            + 2  # x full + empty
            + (NUM_TMEM_BUFFERS * NUM_MMA_GROUPS * 2)  # tmem full + empty
            + (NUM_SMEM_BUFFERS * NUM_MMA_GROUPS if NUM_CTAS == 2 else 0)
        ) * 8  # mbarrier size
        total_smem = (
            smem_a + smem_b + smem_x + smem_c_store + smem_s_store + smem_bars + 256
        )
        if total_smem > MAX_SMEM:
            continue

        # TMEM budget
        total_tmem = BM_SPLIT * BN * 4 * NUM_TMEM_BUFFERS * NUM_MMA_GROUPS
        if total_tmem > MAX_TMEM:
            continue

        pruned.append(conf)
    return pruned


def matmul_tma_set_block_size_hook(nargs: Dict[str, Any]) -> None:
    BLOCK_M = nargs["BLOCK_SIZE_M"]
    BLOCK_N = nargs["BLOCK_SIZE_N"]
    BLOCK_K = nargs["BLOCK_SIZE_K"]
    NUM_MMA_GROUPS = nargs.get("NUM_MMA_GROUPS", 1)
    NUM_CTAS = nargs.get("NUM_CTAS", 1)
    EPILOGUE_SUBTILE = nargs.get("EPILOGUE_SUBTILE", 1)
    BM_SPLIT = BLOCK_M // NUM_MMA_GROUPS
    nargs["a_desc"].block_shape = [BM_SPLIT, BLOCK_K]
    nargs["b_desc"].block_shape = [BLOCK_K, BLOCK_N // NUM_CTAS]
    nargs["x_desc"].block_shape = [BLOCK_M, BLOCK_N]
    nargs["c_desc"].block_shape = [BM_SPLIT, BLOCK_N // EPILOGUE_SUBTILE]
    nargs["s_desc"].block_shape = [BM_SPLIT, BLOCK_N // EPILOGUE_SUBTILE]


# ---- Kernel ----


@triton.autotune(
    configs=get_autotune_configs(),
    key=["M", "N", "K"],
    prune_configs_by={"early_config_prune": preprocess_configs},
)
@triton.jit
def matmul_sigmoid_mul_kernel(  # noqa: C901
    a_desc,
    b_desc,
    x_desc,
    c_desc,
    s_desc,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMEM_BUFFERS: tl.constexpr,
    NUM_TMEM_BUFFERS: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    INTERLEAVE_EPILOGUE: tl.constexpr,
    NUM_SMS: tl.constexpr,
    USE_CLC: tl.constexpr,
) -> None:
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_SIZE_M // NUM_MMA_GROUPS
    SLICE_SIZE: tl.constexpr = BLOCK_SIZE_N // EPILOGUE_SUBTILE

    # ── SMEM allocations ──
    buffers_A = tlx.local_alloc(
        (BLOCK_M_SPLIT, BLOCK_SIZE_K),
        tl.bfloat16,
        NUM_SMEM_BUFFERS * NUM_MMA_GROUPS,
    )
    buffers_B = tlx.local_alloc(
        (BLOCK_SIZE_K, BLOCK_SIZE_N // NUM_CTAS),
        tl.bfloat16,
        NUM_SMEM_BUFFERS,
    )
    buffers_X = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.bfloat16, 1)
    tmem_buffers = tlx.local_alloc(
        (BLOCK_M_SPLIT, BLOCK_SIZE_N),
        tl.float32,
        NUM_TMEM_BUFFERS * NUM_MMA_GROUPS,
        tlx.storage_kind.tmem,
    )

    # ── SMEM buffers for async TMA epilogue stores ──
    # For EPILOGUE_SUBTILE=2: double-buffer (2 buffers per output)
    # For EPILOGUE_SUBTILE=1: single-buffer (1 buffer per output)
    NUM_EPILOGUE_SMEM_BUFFERS: tl.constexpr = 2 if EPILOGUE_SUBTILE > 1 else 1
    c_smem_buffers = tlx.local_alloc(
        (BLOCK_M_SPLIT, SLICE_SIZE),
        tl.bfloat16,
        NUM_EPILOGUE_SMEM_BUFFERS,
    )
    s_smem_buffers = tlx.local_alloc(
        (BLOCK_M_SPLIT, SLICE_SIZE),
        tl.bfloat16,
        NUM_EPILOGUE_SMEM_BUFFERS,
    )

    # ── Barriers ──
    A_smem_full_bars = tlx.alloc_barriers(
        num_barriers=NUM_SMEM_BUFFERS * NUM_MMA_GROUPS, arrive_count=1
    )
    A_smem_empty_bars = tlx.alloc_barriers(
        num_barriers=NUM_SMEM_BUFFERS * NUM_MMA_GROUPS, arrive_count=1
    )
    B_smem_full_bars = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    x_full_bars = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    x_empty_bars = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    tmem_full_bars = tlx.alloc_barriers(
        num_barriers=NUM_TMEM_BUFFERS * NUM_MMA_GROUPS, arrive_count=1
    )
    tmem_empty_bars = tlx.alloc_barriers(
        num_barriers=NUM_TMEM_BUFFERS * NUM_MMA_GROUPS,
        arrive_count=EPILOGUE_SUBTILE,
    )

    if NUM_CTAS == 2:
        cluster_cta_rank = tlx.cluster_cta_rank()
        pred_leader_cta = cluster_cta_rank % 2 == 0
        cta_bars = tlx.alloc_barriers(
            num_barriers=NUM_SMEM_BUFFERS * NUM_MMA_GROUPS, arrive_count=2
        )
    else:
        cluster_cta_rank = 0
        pred_leader_cta = False

    if USE_CLC:
        clc_context = tlx.clc_create_context(num_consumers=6 if NUM_CTAS == 2 else 3)

    DSIZE: tl.constexpr = 2  # bf16 = 2 bytes

    with tlx.async_tasks():
        # ───────── Epilogue (default async task) ─────────
        with tlx.async_task("default"):
            start_pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
            num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
            num_pid_m = (num_pid_m + NUM_CTAS - 1) // NUM_CTAS * NUM_CTAS
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            num_tiles = num_pid_m * num_pid_n

            tmem_read_phase = 0
            cur_tmem_buf = 0
            x_phase = 0
            tile_id = start_pid
            if USE_CLC:
                clc_phase_producer = 1
                clc_phase_consumer = 0
            while (tile_id != -1) if USE_CLC else (tile_id < num_tiles):
                if USE_CLC:
                    tlx.clc_producer(
                        clc_context, clc_phase_producer, multi_ctas=NUM_CTAS == 2
                    )
                    clc_phase_producer = clc_phase_producer ^ 1
                pid_m, pid_n = _compute_pid(
                    tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M
                )
                offs_bn = pid_n * BLOCK_SIZE_N

                # Load aux x ([BM, BN]) to registers
                tlx.barrier_wait(x_full_bars[0], x_phase)
                x_full = tlx.local_load(buffers_X[0])
                tlx.fence_async_shared()
                tlx.barrier_arrive(x_empty_bars[0], 1)

                # Split x along M into NUM_MMA_GROUPS slices
                if NUM_MMA_GROUPS > 1:
                    x_per_group = x_full.reshape(
                        NUM_MMA_GROUPS, BLOCK_M_SPLIT, BLOCK_SIZE_N
                    ).split()

                if INTERLEAVE_EPILOGUE:
                    # Interleaved 2-group epilogue (S13) with async TMA stores
                    buf_idx_0 = 0 * NUM_TMEM_BUFFERS + cur_tmem_buf
                    buf_idx_1 = 1 * NUM_TMEM_BUFFERS + cur_tmem_buf
                    acc_tmem_0 = tmem_buffers[buf_idx_0]
                    acc_tmem_1 = tmem_buffers[buf_idx_1]
                    offs_am_0 = pid_m * BLOCK_SIZE_M
                    offs_am_1 = pid_m * BLOCK_SIZE_M + BLOCK_M_SPLIT
                    x_g0 = x_per_group[0]
                    x_g1 = x_per_group[1]

                    tlx.barrier_wait(tmem_full_bars[buf_idx_0], tmem_read_phase)
                    if EPILOGUE_SUBTILE > 1:
                        x_g0_pieces = tl.trans(
                            x_g0.reshape(BLOCK_M_SPLIT, EPILOGUE_SUBTILE, SLICE_SIZE),
                            (0, 2, 1),
                        ).split()
                        x_g1_pieces = tl.trans(
                            x_g1.reshape(BLOCK_M_SPLIT, EPILOGUE_SUBTILE, SLICE_SIZE),
                            (0, 2, 1),
                        ).split()
                    acc_sub = tlx.subslice(acc_tmem_0, 0, SLICE_SIZE)
                    result = tlx.local_load(acc_sub)
                    tlx.barrier_arrive(tmem_empty_bars[buf_idx_0], 1)
                    sig = sigmoid_approx_fp32(result)
                    sig_bf16 = sig.to(tl.bfloat16)
                    if EPILOGUE_SUBTILE > 1:
                        x_chunk = x_g0_pieces[0]
                    else:
                        x_chunk = x_g0
                    y = 2.0 * _mul_f32x2(x_chunk.to(tl.float32), sig)
                    # Async TMA store via SMEM bounce
                    c_smem = c_smem_buffers[0]
                    s_smem = s_smem_buffers[0]
                    if NUM_EPILOGUE_SMEM_BUFFERS > 1:
                        tlx.async_descriptor_store_wait(1)
                    else:
                        tlx.async_descriptor_store_wait(0)
                    tlx.local_store(c_smem, y.to(tl.bfloat16))
                    tlx.local_store(s_smem, sig_bf16)
                    tlx.fence_async_shared()
                    tlx.async_descriptor_store(
                        c_desc,
                        c_smem,
                        [offs_am_0, offs_bn],
                        eviction_policy="evict_first",
                    )
                    tlx.async_descriptor_store(
                        s_desc,
                        s_smem,
                        [offs_am_0, offs_bn],
                        eviction_policy="evict_first",
                    )

                    tlx.barrier_wait(tmem_full_bars[buf_idx_1], tmem_read_phase)
                    acc_sub = tlx.subslice(acc_tmem_1, 0, SLICE_SIZE)
                    result = tlx.local_load(acc_sub)
                    tlx.barrier_arrive(tmem_empty_bars[buf_idx_1], 1)
                    sig = sigmoid_approx_fp32(result)
                    sig_bf16 = sig.to(tl.bfloat16)
                    if EPILOGUE_SUBTILE > 1:
                        x_chunk = x_g1_pieces[0]
                    else:
                        x_chunk = x_g1
                    y = 2.0 * _mul_f32x2(x_chunk.to(tl.float32), sig)
                    # Async TMA store via SMEM bounce
                    buf_sel = 1 if NUM_EPILOGUE_SMEM_BUFFERS > 1 else 0
                    c_smem = c_smem_buffers[buf_sel]
                    s_smem = s_smem_buffers[buf_sel]
                    if NUM_EPILOGUE_SMEM_BUFFERS > 1:
                        tlx.async_descriptor_store_wait(1)
                    else:
                        tlx.async_descriptor_store_wait(0)
                    tlx.local_store(c_smem, y.to(tl.bfloat16))
                    tlx.local_store(s_smem, sig_bf16)
                    tlx.fence_async_shared()
                    tlx.async_descriptor_store(
                        c_desc,
                        c_smem,
                        [offs_am_1, offs_bn],
                        eviction_policy="evict_first",
                    )
                    tlx.async_descriptor_store(
                        s_desc,
                        s_smem,
                        [offs_am_1, offs_bn],
                        eviction_policy="evict_first",
                    )

                    for slice_id in tl.static_range(1, EPILOGUE_SUBTILE):
                        # group 0
                        acc_sub = tlx.subslice(
                            acc_tmem_0, slice_id * SLICE_SIZE, SLICE_SIZE
                        )
                        result = tlx.local_load(acc_sub)
                        tlx.barrier_arrive(tmem_empty_bars[buf_idx_0], 1)
                        sig = sigmoid_approx_fp32(result)
                        sig_bf16 = sig.to(tl.bfloat16)
                        x_chunk = x_g0_pieces[slice_id]
                        y = 2.0 * _mul_f32x2(x_chunk.to(tl.float32), sig)
                        # Async TMA store via SMEM bounce (alternate buffers)
                        buf_sel = slice_id % NUM_EPILOGUE_SMEM_BUFFERS
                        c_smem = c_smem_buffers[buf_sel]
                        s_smem = s_smem_buffers[buf_sel]
                        tlx.async_descriptor_store_wait(1)
                        tlx.local_store(c_smem, y.to(tl.bfloat16))
                        tlx.local_store(s_smem, sig_bf16)
                        tlx.fence_async_shared()
                        tlx.async_descriptor_store(
                            c_desc,
                            c_smem,
                            [offs_am_0, offs_bn + slice_id * SLICE_SIZE],
                            eviction_policy="evict_first",
                        )
                        tlx.async_descriptor_store(
                            s_desc,
                            s_smem,
                            [offs_am_0, offs_bn + slice_id * SLICE_SIZE],
                            eviction_policy="evict_first",
                        )
                        # group 1
                        acc_sub = tlx.subslice(
                            acc_tmem_1, slice_id * SLICE_SIZE, SLICE_SIZE
                        )
                        result = tlx.local_load(acc_sub)
                        tlx.barrier_arrive(tmem_empty_bars[buf_idx_1], 1)
                        sig = sigmoid_approx_fp32(result)
                        sig_bf16 = sig.to(tl.bfloat16)
                        x_chunk = x_g1_pieces[slice_id]
                        y = 2.0 * _mul_f32x2(x_chunk.to(tl.float32), sig)
                        # Async TMA store via SMEM bounce (alternate buffers)
                        buf_sel = (slice_id + 1) % NUM_EPILOGUE_SMEM_BUFFERS
                        c_smem = c_smem_buffers[buf_sel]
                        s_smem = s_smem_buffers[buf_sel]
                        tlx.async_descriptor_store_wait(1)
                        tlx.local_store(c_smem, y.to(tl.bfloat16))
                        tlx.local_store(s_smem, sig_bf16)
                        tlx.fence_async_shared()
                        tlx.async_descriptor_store(
                            c_desc,
                            c_smem,
                            [offs_am_1, offs_bn + slice_id * SLICE_SIZE],
                            eviction_policy="evict_first",
                        )
                        tlx.async_descriptor_store(
                            s_desc,
                            s_smem,
                            [offs_am_1, offs_bn + slice_id * SLICE_SIZE],
                            eviction_policy="evict_first",
                        )
                else:
                    # Standard sequential per-group epilogue with async TMA stores
                    for group_id in tl.static_range(NUM_MMA_GROUPS):
                        buf_idx = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf
                        tlx.barrier_wait(tmem_full_bars[buf_idx], tmem_read_phase)
                        acc_tmem = tmem_buffers[buf_idx]
                        offs_am = pid_m * BLOCK_SIZE_M + group_id * BLOCK_M_SPLIT
                        if NUM_MMA_GROUPS > 1:
                            x_grp = x_per_group[group_id]
                        else:
                            x_grp = x_full
                        if EPILOGUE_SUBTILE > 1:
                            x_grp_pieces = tl.trans(
                                x_grp.reshape(
                                    BLOCK_M_SPLIT, EPILOGUE_SUBTILE, SLICE_SIZE
                                ),
                                (0, 2, 1),
                            ).split()
                        for slice_id in tl.static_range(EPILOGUE_SUBTILE):
                            acc_sub = tlx.subslice(
                                acc_tmem, slice_id * SLICE_SIZE, SLICE_SIZE
                            )
                            result = tlx.local_load(acc_sub)
                            tlx.barrier_arrive(tmem_empty_bars[buf_idx], 1)
                            sig = sigmoid_approx_fp32(result)
                            sig_bf16 = sig.to(tl.bfloat16)
                            if EPILOGUE_SUBTILE > 1:
                                x_chunk = x_grp_pieces[slice_id]
                            else:
                                x_chunk = x_grp
                            y = 2.0 * _mul_f32x2(x_chunk.to(tl.float32), sig)
                            # Async TMA store via SMEM bounce
                            buf_sel = slice_id % NUM_EPILOGUE_SMEM_BUFFERS
                            c_smem = c_smem_buffers[buf_sel]
                            s_smem = s_smem_buffers[buf_sel]
                            if NUM_EPILOGUE_SMEM_BUFFERS > 1:
                                tlx.async_descriptor_store_wait(1)
                            else:
                                tlx.async_descriptor_store_wait(0)
                            tlx.local_store(c_smem, y.to(tl.bfloat16))
                            tlx.local_store(s_smem, sig_bf16)
                            tlx.fence_async_shared()
                            tlx.async_descriptor_store(
                                c_desc,
                                c_smem,
                                [offs_am, offs_bn + slice_id * SLICE_SIZE],
                                eviction_policy="evict_first",
                            )
                            tlx.async_descriptor_store(
                                s_desc,
                                s_smem,
                                [offs_am, offs_bn + slice_id * SLICE_SIZE],
                                eviction_policy="evict_first",
                            )

                cur_tmem_buf = (cur_tmem_buf + 1) % NUM_TMEM_BUFFERS
                tmem_read_phase = tmem_read_phase ^ (cur_tmem_buf == 0)
                x_phase = x_phase ^ 1
                if USE_CLC:
                    tile_id = tlx.clc_consumer(
                        clc_context, clc_phase_consumer, multi_ctas=NUM_CTAS == 2
                    )
                    clc_phase_consumer = clc_phase_consumer ^ 1
                else:
                    tile_id += NUM_SMS

            # Wait for all async TMA stores to complete before epilogue exits
            tlx.async_descriptor_store_wait(0)

        # ───────── MMA consumer ─────────
        with tlx.async_task(num_warps=1, registers=24):
            start_pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
            num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
            num_pid_m = (num_pid_m + NUM_CTAS - 1) // NUM_CTAS * NUM_CTAS
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            num_tiles = num_pid_m * num_pid_n
            k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

            cur_tmem_buf = 0
            tmem_write_phase = 0
            smem_accum_cnt = 0
            tile_id = start_pid
            if USE_CLC:
                clc_phase_consumer = 0
            while (tile_id != -1) if USE_CLC else (tile_id < num_tiles):
                # Peeled first K iter
                buf, phase = _get_bufidx_phase(smem_accum_cnt, NUM_SMEM_BUFFERS)
                tlx.barrier_wait(B_smem_full_bars[buf], phase)
                for group_id in tl.static_range(NUM_MMA_GROUPS):
                    a_buf = group_id * NUM_SMEM_BUFFERS + buf
                    acc_buf = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf
                    tlx.barrier_wait(A_smem_full_bars[a_buf], phase)
                    tlx.barrier_wait(tmem_empty_bars[acc_buf], tmem_write_phase ^ 1)
                    if NUM_CTAS == 2:
                        tlx.barrier_arrive(
                            cta_bars[a_buf],
                            arrive_count=1,
                            remote_cta_rank=cluster_cta_rank & ~1,
                        )
                        tlx.barrier_wait(
                            cta_bars[a_buf], phase=phase, pred=pred_leader_cta
                        )
                    tlx.async_dot(
                        buffers_A[a_buf],
                        buffers_B[buf],
                        tmem_buffers[acc_buf],
                        use_acc=False,
                        mBarriers=[A_smem_empty_bars[a_buf]],
                        two_ctas=NUM_CTAS == 2,
                        out_dtype=tl.float32,
                    )
                smem_accum_cnt += 1

                # Remaining K iters
                for _ in range(1, k_tiles):
                    buf, phase = _get_bufidx_phase(smem_accum_cnt, NUM_SMEM_BUFFERS)
                    tlx.barrier_wait(B_smem_full_bars[buf], phase)
                    for group_id in tl.static_range(NUM_MMA_GROUPS):
                        a_buf = group_id * NUM_SMEM_BUFFERS + buf
                        acc_buf = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf
                        tlx.barrier_wait(A_smem_full_bars[a_buf], phase)
                        if NUM_CTAS == 2:
                            tlx.barrier_arrive(
                                cta_bars[a_buf],
                                arrive_count=1,
                                remote_cta_rank=cluster_cta_rank & ~1,
                            )
                            tlx.barrier_wait(
                                cta_bars[a_buf], phase=phase, pred=pred_leader_cta
                            )
                        tlx.async_dot(
                            buffers_A[a_buf],
                            buffers_B[buf],
                            tmem_buffers[acc_buf],
                            use_acc=True,
                            mBarriers=[A_smem_empty_bars[a_buf]],
                            two_ctas=NUM_CTAS == 2,
                            out_dtype=tl.float32,
                        )
                    smem_accum_cnt += 1

                # After K-loop: wait last A_empty for each group, signal tmem_full
                last_buf, last_phase = _get_bufidx_phase(
                    smem_accum_cnt - 1, NUM_SMEM_BUFFERS
                )
                for group_id in tl.static_range(NUM_MMA_GROUPS):
                    a_buf = group_id * NUM_SMEM_BUFFERS + last_buf
                    tlx.barrier_wait(A_smem_empty_bars[a_buf], last_phase)
                    acc_buf = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf
                    tlx.barrier_arrive(tmem_full_bars[acc_buf], 1)

                cur_tmem_buf = (cur_tmem_buf + 1) % NUM_TMEM_BUFFERS
                tmem_write_phase = tmem_write_phase ^ (cur_tmem_buf == 0)
                if USE_CLC:
                    tile_id = tlx.clc_consumer(
                        clc_context, clc_phase_consumer, multi_ctas=NUM_CTAS == 2
                    )
                    clc_phase_consumer = clc_phase_consumer ^ 1
                else:
                    tile_id += NUM_SMS

        # ───────── Producer (TMA load) ─────────
        with tlx.async_task(num_warps=1, registers=24):
            start_pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
            num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
            num_pid_m = (num_pid_m + NUM_CTAS - 1) // NUM_CTAS * NUM_CTAS
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            num_tiles = num_pid_m * num_pid_n
            k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

            smem_accum_cnt = 0
            x_phase = 0
            tile_id = start_pid
            if USE_CLC:
                clc_phase_consumer = 0
            while (tile_id != -1) if USE_CLC else (tile_id < num_tiles):
                pid_m, pid_n = _compute_pid(
                    tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M
                )
                offs_am = pid_m * BLOCK_SIZE_M
                offs_bn_full = pid_n * BLOCK_SIZE_N
                if NUM_CTAS == 2:
                    offs_bn = offs_bn_full + cluster_cta_rank * (BLOCK_SIZE_N // 2)
                else:
                    offs_bn = offs_bn_full

                for k_idx in range(0, k_tiles):
                    buf, phase = _get_bufidx_phase(smem_accum_cnt, NUM_SMEM_BUFFERS)
                    offs_k = k_idx * BLOCK_SIZE_K

                    # Load A for group 0
                    a_buf_0 = 0 * NUM_SMEM_BUFFERS + buf
                    tlx.barrier_wait(A_smem_empty_bars[a_buf_0], phase ^ 1)
                    tlx.barrier_expect_bytes(
                        A_smem_full_bars[a_buf_0],
                        DSIZE * BLOCK_M_SPLIT * BLOCK_SIZE_K,
                    )
                    tlx.async_descriptor_load(
                        a_desc,
                        buffers_A[a_buf_0],
                        [offs_am, offs_k],
                        A_smem_full_bars[a_buf_0],
                        eviction_policy="evict_last",
                    )

                    # Load B (gated on LAST group's A_empty)
                    last_a_buf = (NUM_MMA_GROUPS - 1) * NUM_SMEM_BUFFERS + buf
                    tlx.barrier_wait(A_smem_empty_bars[last_a_buf], phase ^ 1)
                    tlx.barrier_expect_bytes(
                        B_smem_full_bars[buf],
                        DSIZE * BLOCK_SIZE_K * (BLOCK_SIZE_N // NUM_CTAS),
                    )
                    tlx.async_descriptor_load(
                        b_desc,
                        buffers_B[buf],
                        [offs_k, offs_bn],
                        B_smem_full_bars[buf],
                        eviction_policy="evict_last",
                    )

                    # Load A for remaining groups
                    for group_id in tl.static_range(1, NUM_MMA_GROUPS):
                        a_buf = group_id * NUM_SMEM_BUFFERS + buf
                        tlx.barrier_wait(A_smem_empty_bars[a_buf], phase ^ 1)
                        offs_am2 = offs_am + group_id * BLOCK_M_SPLIT
                        tlx.barrier_expect_bytes(
                            A_smem_full_bars[a_buf],
                            DSIZE * BLOCK_M_SPLIT * BLOCK_SIZE_K,
                        )
                        tlx.async_descriptor_load(
                            a_desc,
                            buffers_A[a_buf],
                            [offs_am2, offs_k],
                            A_smem_full_bars[a_buf],
                            eviction_policy="evict_last",
                        )
                    smem_accum_cnt += 1

                # After K-loop: load aux x for elementwise (full [BM, BN])
                tlx.barrier_wait(x_empty_bars[0], x_phase ^ 1)
                tlx.barrier_expect_bytes(
                    x_full_bars[0], DSIZE * BLOCK_SIZE_M * BLOCK_SIZE_N
                )
                tlx.async_descriptor_load(
                    x_desc,
                    buffers_X[0],
                    [offs_am, offs_bn_full],
                    x_full_bars[0],
                    eviction_policy="evict_first",
                )
                x_phase = x_phase ^ 1

                if USE_CLC:
                    tile_id = tlx.clc_consumer(
                        clc_context, clc_phase_consumer, multi_ctas=NUM_CTAS == 2
                    )
                    clc_phase_consumer = clc_phase_consumer ^ 1
                else:
                    tile_id += NUM_SMS


# ---- Launch ----


def alloc_fn(size: int, alignment: int, _):
    return torch.empty(size, dtype=torch.int8, device="cuda")


@torch.library.custom_op(
    "torch_tlx::sm100_02_fused_kernel", mutates_args=()
)
def mm_sigmoid_mul_mul__returns_sigmoid(
    x: torch.Tensor, w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused matrix multiplication, sigmoid, and multiply: out = 2 * x * sigmoid(x @ w). Also returns sigmoid.

    Args:
        x: [M, K] bfloat16, contiguous. K must equal N (square w).
        w: [K, N] bfloat16, contiguous. K == N.

    Returns:
        out: [M, N] bfloat16 — gated SE output.
        s:   [M, N] bfloat16 — saved sigmoid(x @ w) for backward.
    """
    assert x.shape[1] == w.shape[0], "Incompatible dimensions"
    assert w.shape[0] == w.shape[1], "Weight must be square (K == N)"
    assert x.is_contiguous(), "x must be contiguous"
    assert w.is_contiguous(), "w must be contiguous"
    M, K = x.shape
    _, N = w.shape

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    s = torch.empty((M, N), device=x.device, dtype=x.dtype)

    triton.set_allocator(alloc_fn)

    NUM_SMS = _get_num_sms()

    dummy_block = [1, 1]
    a_desc = TensorDescriptor(x, x.shape, x.stride(), dummy_block)
    b_desc = TensorDescriptor(w, w.shape, w.stride(), dummy_block)
    # x is reused as the gate operand; same tensor, separate descriptor for the
    # epilogue load (block shape differs from a_desc's BLOCK_M_SPLIT × BLOCK_K).
    x_desc = TensorDescriptor(x, x.shape, x.stride(), dummy_block)
    c_desc = TensorDescriptor(out, out.shape, out.stride(), dummy_block)
    s_desc = TensorDescriptor(s, s.shape, s.stride(), dummy_block)

    def grid(META):
        NUM_CTAS = META["NUM_CTAS"]
        USE_CLC = META["USE_CLC"]
        num_pid_m = triton.cdiv(M, META["BLOCK_SIZE_M"])
        num_pid_n = triton.cdiv(N, META["BLOCK_SIZE_N"])
        num_pid_m = (num_pid_m + NUM_CTAS - 1) // NUM_CTAS * NUM_CTAS
        total_tiles = num_pid_m * num_pid_n
        # CLC needs the full tile count as the grid (it dispatches up to grid
        # size virtual tiles); software-persistent caps at NUM_SMS.
        if USE_CLC:
            return (total_tiles,)
        return (min(NUM_SMS, total_tiles),)

    matmul_sigmoid_mul_kernel[grid](
        a_desc,
        b_desc,
        x_desc,
        c_desc,
        s_desc,
        M,
        N,
        K,
        NUM_SMS=NUM_SMS,
    )
    return out, s


@mm_sigmoid_mul_mul__returns_sigmoid.register_fake
def _(x: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    M, K = x.shape
    _, N = w.shape
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    s = torch.empty((M, N), device=x.device, dtype=x.dtype)
    return out, s


def _aten_sm100_02(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    accumulator = x.float() @ weight.float()
    sigmoid = torch.sigmoid(accumulator)
    output = (2.0 * x.float() * sigmoid).to(torch.bfloat16)
    return output, sigmoid.to(torch.bfloat16)


@torch.library.custom_op("torch_tlx::sm100_02_semantic", mutates_args=())
def sm100_02_semantic(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _aten_sm100_02(x, weight)


@sm100_02_semantic.register_fake
def _fake_sm100_02_semantic(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (x.shape[0], weight.shape[1])
    return (
        torch.empty(shape, device=x.device, dtype=torch.bfloat16),
        torch.empty(shape, device=x.device, dtype=torch.bfloat16),
    )


def _fused_sm100_02(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return mm_sigmoid_mul_mul__returns_sigmoid(x, weight)


def _eligible_sm100_02(match) -> bool:
    from torch._inductor import config

    from ..hw.target import current_target

    if config.triton.tlx_mode not in ("allow", "force"):
        return False
    if not current_target().is_blackwell:
        return False
    x = match.kwargs["x"].meta.get("val")
    weight = match.kwargs["weight"].meta.get("val")
    if not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor):
        return False
    return bool(
        x.shape == (3836160, 256)
        and weight.shape == (256, 256)
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.device == weight.device
        and tuple(x.stride()) == (256, 1)
        and tuple(weight.stride()) == (256, 1)
    )


@functools.cache
def register_sm100_02_pattern() -> None:
    from torch._inductor.fx_passes.post_grad import pass_patterns
    from torch._inductor.kernel.custom_op import CustomOpConfig
    from torch._inductor.pattern_matcher import fwd_only, register_replacement

    from .subgraph import register_tlx_subgraph_autotuning

    register_tlx_subgraph_autotuning(
        sm100_02_semantic,
        name="tlx_sm100_02_matmul_sigmoid_mul",
        tlx_configs=[CustomOpConfig(_fused_sm100_02)],
        aten_impl=_aten_sm100_02,
    )

    example_inputs = (
        torch.empty((2, 64), dtype=torch.bfloat16),
        torch.empty((64, 64), dtype=torch.bfloat16),
    )

    def pattern(x, weight):
        return _aten_sm100_02(x, weight)

    def replacement(x, weight):
        return sm100_02_semantic(x, weight)

    register_replacement(
        pattern,
        replacement,
        example_inputs,
        fwd_only,
        pass_patterns[0],
        extra_check=_eligible_sm100_02,
        pattern_name="tlx_sm100_02_matmul_sigmoid_mul",
    )
