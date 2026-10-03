"""Blackwell (sm100) grouped GEMM for ``tlx.ops.grouped_gemm``.

The public op supplies row-major A and column-major B tensors. B is described
through its zero-copy row-major transpose view and transposed back in shared
memory before MMA.
"""

import functools
from typing import Optional

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.language.extra.tlx.warp_spec import get_bufidx_phase

_MAX_SMEM_BYTES = 232448
_SMEM_SAFETY_MARGIN_BYTES = 1024
_MAX_TMEM_BYTES = 256 * 1024


def _make_config(block_n, block_k, num_smem_buffers, epilogue_subtile, num_ctas):
    return {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": block_k,
        "NUM_SMEM_BUFFERS": num_smem_buffers,
        "NUM_TMEM_BUFFERS": 2,
        "EPILOGUE_SUBTILE": epilogue_subtile,
        "NUM_CTAS": num_ctas,
        "num_warps": 4,
    }


# The original promotion config remains the fallback and benchmark baseline.
_CONFIG = _make_config(block_n=128, block_k=64, num_smem_buffers=2, epilogue_subtile=4, num_ctas=2)

# Retuned offline on B200 over the tutorial search space. At runtime, this
# implementation selects a config analytically from the group shapes and SM
# count; it does not benchmark candidates or invoke Triton's autotuner.
_CONFIGS = {
    "fallback": _CONFIG,
    "n128_1cta": _make_config(block_n=128, block_k=128, num_smem_buffers=3, epilogue_subtile=1, num_ctas=1),
    "n128_2cta": _make_config(block_n=128, block_k=128, num_smem_buffers=4, epilogue_subtile=1, num_ctas=2),
    "n256_1cta_3stage": _make_config(block_n=256, block_k=64, num_smem_buffers=3, epilogue_subtile=4, num_ctas=1),
    "n256_1cta_4stage": _make_config(block_n=256, block_k=64, num_smem_buffers=4, epilogue_subtile=4, num_ctas=1),
    "n256_2cta": _make_config(block_n=256, block_k=128, num_smem_buffers=3, epilogue_subtile=4, num_ctas=2),
}


def _estimate_smem_bytes(config):
    block_m = config["BLOCK_SIZE_M"]
    block_n = config["BLOCK_SIZE_N"]
    block_k = config["BLOCK_SIZE_K"]
    num_ctas = config["NUM_CTAS"]
    num_smem_buffers = config["NUM_SMEM_BUFFERS"]
    num_tmem_buffers = config["NUM_TMEM_BUFFERS"]
    epilogue_subtile = config["EPILOGUE_SUBTILE"]

    operand_bytes = num_smem_buffers * 2 * block_k * (block_m + block_n // num_ctas)
    # local_load lowers through a shared staging tile before the descriptor store.
    epilogue_bytes = block_m * (block_n // epilogue_subtile) * 2
    barrier_count = 2 * num_smem_buffers + 2 * num_tmem_buffers
    if num_ctas == 2:
        barrier_count += num_smem_buffers
    return operand_bytes + epilogue_bytes + barrier_count * 8


def _estimate_tmem_bytes(config):
    return (config["BLOCK_SIZE_M"] * config["BLOCK_SIZE_N"] * 4 * config["NUM_TMEM_BUFFERS"])


def _config_error(config, num_sms=None):
    block_m = config["BLOCK_SIZE_M"]
    block_n = config["BLOCK_SIZE_N"]
    block_k = config["BLOCK_SIZE_K"]
    num_ctas = config["NUM_CTAS"]
    epilogue_subtile = config["EPILOGUE_SUBTILE"]

    if block_m != 128:
        return "BLOCK_SIZE_M must be 128"
    if block_n not in (128, 256):
        return "BLOCK_SIZE_N must be 128 or 256"
    if block_k not in (64, 128):
        return "BLOCK_SIZE_K must be 64 or 128"
    if num_ctas not in (1, 2):
        return "NUM_CTAS must be 1 or 2"
    if num_sms is not None and num_sms < num_ctas:
        return f"needs at least {num_ctas} SMs"
    if block_n % num_ctas:
        return "BLOCK_SIZE_N must be divisible by NUM_CTAS"
    if epilogue_subtile not in (1, 2, 4) or block_n % epilogue_subtile:
        return "EPILOGUE_SUBTILE must divide BLOCK_SIZE_N"
    if config["NUM_SMEM_BUFFERS"] not in (2, 3, 4):
        return "NUM_SMEM_BUFFERS must be 2, 3, or 4"
    if config["NUM_TMEM_BUFFERS"] != 2:
        return "NUM_TMEM_BUFFERS must be 2"
    if config["num_warps"] != 4:
        return "num_warps must be 4"
    if _estimate_smem_bytes(config) + _SMEM_SAFETY_MARGIN_BYTES > _MAX_SMEM_BYTES:
        return "configuration exceeds the shared-memory budget"
    if _estimate_tmem_bytes(config) > _MAX_TMEM_BYTES:
        return "configuration exceeds the tensor-memory budget"
    return None


for _config_name, _candidate_config in _CONFIGS.items():
    if error := _config_error(_candidate_config):
        raise RuntimeError(f"invalid sm100 grouped GEMM config {_config_name!r}: {error}")


def _cdiv(x, y):
    return (x + y - 1) // y


def _group_tile_stats(shapes, block_n, num_ctas, num_sms):
    real_tiles = 0
    scheduled_tiles = 0
    for m, n, _ in shapes:
        m_tiles = _cdiv(m, 128)
        n_tiles = _cdiv(n, block_n)
        real_tiles += m_tiles * n_tiles
        if num_ctas == 2:
            m_tiles = (m_tiles + 1) & ~1
        scheduled_tiles += m_tiles * n_tiles

    grid = num_sms - num_sms % num_ctas
    waves = _cdiv(scheduled_tiles, grid) if grid and scheduled_tiles else 0
    return {
        "real_tiles": real_tiles,
        "scheduled_tiles": scheduled_tiles,
        "virtual_tiles": scheduled_tiles - real_tiles,
        "grid": grid,
        "waves": waves,
    }


def _pick_config(shapes, num_sms):
    if not shapes:
        raise ValueError("grouped GEMM needs at least one shape")

    narrow_n = max(n for _, n, _ in shapes) <= 128
    block_n = 128 if narrow_n else 256
    paired = _group_tile_stats(shapes, block_n=block_n, num_ctas=2, num_sms=num_sms)
    excessive_pair_padding = paired["virtual_tiles"] * 20 >= paired["real_tiles"]

    if narrow_n:
        name = "n128_1cta" if num_sms < 2 or excessive_pair_padding else "n128_2cta"
        return _CONFIGS[name]

    useful_work = sum(m * n * k for m, n, k in shapes)
    wide_work = sum(m * n * k for m, n, k in shapes if n >= 2 * k)
    deep_wide_work = sum(m * n * k for m, n, k in shapes if n >= 2 * k and k > 4096)

    if num_sms < 2 or excessive_pair_padding:
        return _CONFIGS["n256_1cta_4stage"]
    if wide_work * 2 >= useful_work:
        name = "n256_1cta_3stage" if deep_wide_work * 2 >= wide_work else "n256_1cta_4stage"
        return _CONFIGS[name]
    return _CONFIGS["n256_2cta"]


def _alloc_tma_descriptor_buffer(size: int, _alignment: int, _stream: Optional[int]):
    device = torch.device("cuda", torch.cuda.current_device())
    return torch.empty(size, device=device, dtype=torch.int8)


@functools.lru_cache(maxsize=None)
def _get_num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@triton.jit
def grouped_gemm_kernel(
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    group_gemm_sizes,
    group_lds,
    group_size,
    NUM_SM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_SMEM_BUFFERS: tl.constexpr,
    NUM_TMEM_BUFFERS: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_CTAS: tl.constexpr,
):
    dtype = tl.float16

    if NUM_CTAS == 2:
        cluster_cta_rank = tlx.cluster_cta_rank()
        pred_cta0 = cluster_cta_rank == 0
        cta_bars = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=2)
    else:
        cluster_cta_rank = 0

    buffers_a = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype, NUM_SMEM_BUFFERS)
    # The public B is column-major [K, N]. Load its row-major [N, K]
    # transpose view and recover [K, N] with local_trans before MMA.
    buffers_b = tlx.local_alloc((BLOCK_SIZE_N // NUM_CTAS, BLOCK_SIZE_K), dtype, NUM_SMEM_BUFFERS)
    tmem_buffers = tlx.local_alloc(
        (BLOCK_SIZE_M, BLOCK_SIZE_N),
        tl.float32,
        NUM_TMEM_BUFFERS,
        tlx.storage_kind.tmem,
    )

    smem_empty_bars = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    smem_full_bars = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    tmem_full_bars = tlx.alloc_barriers(num_barriers=NUM_TMEM_BUFFERS, arrive_count=1)
    tmem_empty_bars = tlx.alloc_barriers(num_barriers=NUM_TMEM_BUFFERS, arrive_count=1)

    with tlx.async_tasks():
        with tlx.async_task("default"):
            tile_idx = tl.program_id(0)
            last_problem_end = 0
            accum_cnt_tmem = 0
            for g in range(group_size):
                gm = tl.load(group_gemm_sizes + g * 3)
                gn = tl.load(group_gemm_sizes + g * 3 + 1)
                num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
                if NUM_CTAS == 2:
                    num_m_tiles = (num_m_tiles + 1) & ~1
                num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
                num_tiles = num_m_tiles * num_n_tiles

                if tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
                    ldc = tl.load(group_lds + g * 3 + 2)
                    c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(dtype))
                    c_desc = tl.make_tensor_descriptor(
                        c_ptr,
                        shape=[gm, gn],
                        strides=[ldc, 1],
                        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N // EPILOGUE_SUBTILE],
                    )

                    while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
                        tile_idx_in_gemm = tile_idx - last_problem_end
                        tile_m_idx = tile_idx_in_gemm % num_m_tiles
                        tile_n_idx = tile_idx_in_gemm // num_m_tiles

                        tmem_buf, tmem_phase = get_bufidx_phase(accum_cnt_tmem, NUM_TMEM_BUFFERS)
                        tlx.barrier_wait(tmem_full_bars[tmem_buf], tmem_phase)

                        offs_cm = tile_m_idx * BLOCK_SIZE_M
                        offs_cn = tile_n_idx * BLOCK_SIZE_N
                        slice_size: tl.constexpr = BLOCK_SIZE_N // EPILOGUE_SUBTILE
                        for slice_id in tl.static_range(EPILOGUE_SUBTILE):
                            acc_slice = tlx.local_slice(
                                tmem_buffers[tmem_buf],
                                [0, slice_id * slice_size],
                                [BLOCK_SIZE_M, slice_size],
                            )
                            result = tlx.local_load(acc_slice).to(dtype)
                            c_desc.store([offs_cm, offs_cn + slice_id * slice_size], result)

                        tlx.barrier_arrive(tmem_empty_bars[tmem_buf], 1)
                        accum_cnt_tmem += 1
                        tile_idx += NUM_SM

                last_problem_end += num_tiles

        with tlx.async_task(num_warps=1, num_regs=48):
            tile_idx = tl.program_id(0)
            last_problem_end = 0
            accum_cnt_smem = 0
            accum_cnt_tmem = 0
            for g in range(group_size):
                gm = tl.load(group_gemm_sizes + g * 3)
                gn = tl.load(group_gemm_sizes + g * 3 + 1)
                gk = tl.load(group_gemm_sizes + g * 3 + 2)
                num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
                if NUM_CTAS == 2:
                    num_m_tiles = (num_m_tiles + 1) & ~1
                num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
                num_tiles = num_m_tiles * num_n_tiles

                if tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
                    while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
                        tmem_buf, tmem_phase = get_bufidx_phase(accum_cnt_tmem, NUM_TMEM_BUFFERS)
                        tlx.barrier_wait(tmem_empty_bars[tmem_buf], tmem_phase ^ 1)

                        for kk in range(0, tl.cdiv(gk, BLOCK_SIZE_K)):
                            smem_buf, smem_phase = get_bufidx_phase(accum_cnt_smem, NUM_SMEM_BUFFERS)
                            tlx.barrier_wait(smem_full_bars[smem_buf], smem_phase)
                            if NUM_CTAS == 2:
                                tlx.barrier_arrive(cta_bars[smem_buf], 1, remote_cta_rank=0)
                                tlx.barrier_wait(cta_bars[smem_buf], phase=smem_phase, pred=pred_cta0)

                            tlx.async_dot(
                                buffers_a[smem_buf],
                                tlx.local_trans(buffers_b[smem_buf]),
                                tmem_buffers[tmem_buf],
                                use_acc=kk > 0,
                                mBarriers=[smem_empty_bars[smem_buf]],
                                two_ctas=NUM_CTAS == 2,
                                out_dtype=tl.float32,
                            )
                            accum_cnt_smem += 1

                        tlx.tcgen05_commit(tmem_full_bars[tmem_buf], two_ctas=NUM_CTAS == 2)
                        accum_cnt_tmem += 1
                        tile_idx += NUM_SM

                last_problem_end += num_tiles

        with tlx.async_task(num_warps=1, num_regs=48):
            tile_idx = tl.program_id(0)
            last_problem_end = 0
            accum_cnt = 0
            accum_cnt_outer = 0
            desc_a_ptrs = tlx.allocate_tensor_descriptor(num=NUM_SMEM_BUFFERS + 1)
            desc_b_ptrs = tlx.allocate_tensor_descriptor(num=NUM_SMEM_BUFFERS + 1)

            for g in range(group_size):
                gm = tl.load(group_gemm_sizes + g * 3)
                gn = tl.load(group_gemm_sizes + g * 3 + 1)
                gk = tl.load(group_gemm_sizes + g * 3 + 2)
                num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
                if NUM_CTAS == 2:
                    num_m_tiles = (num_m_tiles + 1) & ~1
                num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
                num_k_tiles = tl.cdiv(gk, BLOCK_SIZE_K)
                num_tiles = num_m_tiles * num_n_tiles

                if tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
                    lda = tl.load(group_lds + g * 3)
                    ldb = tl.load(group_lds + g * 3 + 1)
                    a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(dtype))
                    b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(dtype))
                    desc_buf, _ = get_bufidx_phase(accum_cnt_outer, NUM_SMEM_BUFFERS + 1)

                    tlx.make_tensor_descriptor(
                        desc_ptr=desc_a_ptrs[desc_buf],
                        base=a_ptr,
                        shape=[gm, gk],
                        strides=[lda, 1],
                        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
                    )
                    tlx.make_tensor_descriptor(
                        desc_ptr=desc_b_ptrs[desc_buf],
                        base=b_ptr,
                        shape=[gn, gk],
                        strides=[ldb, 1],
                        block_shape=[BLOCK_SIZE_N // NUM_CTAS, BLOCK_SIZE_K],
                    )

                    while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
                        tile_idx_in_gemm = tile_idx - last_problem_end
                        tile_m_idx = tile_idx_in_gemm % num_m_tiles
                        tile_n_idx = tile_idx_in_gemm // num_m_tiles

                        a_desc = tlx.reinterpret_tensor_descriptor(
                            desc_ptr=desc_a_ptrs[desc_buf],
                            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
                            dtype=dtype,
                        )
                        b_desc = tlx.reinterpret_tensor_descriptor(
                            desc_ptr=desc_b_ptrs[desc_buf],
                            block_shape=[BLOCK_SIZE_N // NUM_CTAS, BLOCK_SIZE_K],
                            dtype=dtype,
                        )

                        offs_am = tile_m_idx * BLOCK_SIZE_M
                        offs_bn = tile_n_idx * BLOCK_SIZE_N
                        if NUM_CTAS == 2:
                            offs_bn += cluster_cta_rank * (BLOCK_SIZE_N // 2)

                        for kk in range(0, num_k_tiles):
                            buf, phase = get_bufidx_phase(accum_cnt, NUM_SMEM_BUFFERS)
                            tlx.barrier_wait(smem_empty_bars[buf], phase ^ 1)
                            tlx.barrier_expect_bytes(
                                smem_full_bars[buf],
                                tlx.size_of(dtype) * (BLOCK_SIZE_M + BLOCK_SIZE_N // NUM_CTAS) * BLOCK_SIZE_K,
                            )
                            tlx.async_descriptor_load(
                                a_desc,
                                buffers_a[buf],
                                [offs_am, kk * BLOCK_SIZE_K],
                                smem_full_bars[buf],
                            )
                            tlx.async_descriptor_load(
                                b_desc,
                                buffers_b[buf],
                                [offs_bn, kk * BLOCK_SIZE_K],
                                smem_full_bars[buf],
                            )
                            accum_cnt += 1

                        tile_idx += NUM_SM

                    accum_cnt_outer += 1
                last_problem_end += num_tiles


def _make_grouped_gemm_args(group_a, group_b):
    device = group_a[0].device
    a_addrs, b_addrs, c_addrs = [], [], []
    group_sizes, group_lds = [], []
    outputs = []

    with torch.cuda.device(device):
        for a, b in zip(group_a, group_b):
            m, k = a.shape
            kb, n = b.shape
            assert k == kb
            c = torch.empty((m, n), device=device, dtype=a.dtype)
            outputs.append(c)
            a_addrs.append(a.data_ptr())
            b_addrs.append(b.data_ptr())
            c_addrs.append(c.data_ptr())
            group_sizes += [m, n, k]
            group_lds += [a.stride(0), b.stride(1), c.stride(0)]

        d_a_ptrs = torch.tensor(a_addrs, dtype=torch.int64, device=device)
        d_b_ptrs = torch.tensor(b_addrs, dtype=torch.int64, device=device)
        d_c_ptrs = torch.tensor(c_addrs, dtype=torch.int64, device=device)
        d_group_sizes = torch.tensor(group_sizes, dtype=torch.int32, device=device)
        d_group_lds = torch.tensor(group_lds, dtype=torch.int32, device=device)

    return d_a_ptrs, d_b_ptrs, d_c_ptrs, d_group_sizes, d_group_lds, len(group_a), outputs


def _launch_grouped_gemm(d_a_ptrs, d_b_ptrs, d_c_ptrs, group_sizes, group_lds, group_size, config):
    device = d_a_ptrs.device
    num_ctas = config["NUM_CTAS"]
    with torch.cuda.device(device):
        triton.set_allocator(_alloc_tma_descriptor_buffer)
        device_index = torch.cuda.current_device()
        num_sms = _get_num_sms(device_index)
        num_sms -= num_sms % num_ctas
        if num_sms == 0:
            raise RuntimeError(f"sm100 grouped GEMM needs at least {num_ctas} SMs")

        grouped_gemm_kernel[(num_sms, )](
            d_a_ptrs,
            d_b_ptrs,
            d_c_ptrs,
            group_sizes,
            group_lds,
            group_size,
            NUM_SM=num_sms,
            BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
            NUM_SMEM_BUFFERS=config["NUM_SMEM_BUFFERS"],
            NUM_TMEM_BUFFERS=config["NUM_TMEM_BUFFERS"],
            EPILOGUE_SUBTILE=config["EPILOGUE_SUBTILE"],
            NUM_CTAS=num_ctas,
            num_warps=config["num_warps"],
            num_stages=1,
            ctas_per_cga=(num_ctas, 1, 1) if num_ctas > 1 else None,
        )


def grouped_gemm(group_a, group_b):
    """Run a ragged group of FP16 matrix multiplications on Blackwell."""
    device = group_a[0].device
    shapes = tuple((a.shape[0], b.shape[1], a.shape[1]) for a, b in zip(group_a, group_b))
    with torch.cuda.device(device):
        num_sms = _get_num_sms(torch.cuda.current_device())
    config = _pick_config(shapes, num_sms)

    args = _make_grouped_gemm_args(group_a, group_b)
    d_a_ptrs, d_b_ptrs, d_c_ptrs, group_sizes, group_lds, group_size, outputs = args
    _launch_grouped_gemm(d_a_ptrs, d_b_ptrs, d_c_ptrs, group_sizes, group_lds, group_size, config)
    return outputs


__all__ = ["grouped_gemm"]
