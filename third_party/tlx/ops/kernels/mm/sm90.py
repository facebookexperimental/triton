"""Hopper (sm90) cooperative persistent GEMM for ``tlx.ops.mm``.

Promoted from ``tutorials/hopper-persistent-gemm-ws-cooperative.py``. The
producer loads two private A tiles and one shared B tile for two replicated
WGMMA consumer tasks.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.tools.tensor_descriptor import TensorDescriptor

from ..._catalog import InvalidInput
from ._shapes import SM90_FOCUS

PERF_SHAPES = SM90_FOCUS


@functools.lru_cache(maxsize=1)
def _get_num_sms():
    return torch.cuda.get_device_properties("cuda").multi_processor_count


def matmul_tma_set_block_size_hook(nargs):
    block_m = nargs["BM"]
    block_n = nargs["BN"]
    block_k = nargs["BK"]
    block_m_split = block_m // nargs["NUM_MMA_GROUPS"]

    if nargs.get("A_ROW_MAJOR", True):
        nargs["a_desc"].block_shape = [block_m_split, block_k]
    else:
        nargs["a_desc"].block_shape = [block_k, block_m_split]
    if nargs.get("B_ROW_MAJOR", True):
        nargs["b_desc"].block_shape = [block_k, block_n]
    else:
        nargs["b_desc"].block_shape = [block_n, block_k]
    nargs["c_desc"].block_shape = [block_m_split, block_n // 2]


def _config(block_m, block_n, num_stages):
    return triton.Config(
        {
            "BM": block_m,
            "BN": block_n,
            "BK": 64,
            "GROUP_SIZE_M": 8,
            "NUM_STAGES": num_stages,
            "NUM_MMA_GROUPS": 2,
            "EPILOGUE_SUBTILE": True,
        },
        num_stages=1,
        num_warps=4,
        pre_hook=matmul_tma_set_block_size_hook,
    )


def _full_configs():
    return [
        _config(block_m, block_n, num_stages) for block_m, block_n in ((128, 256), (256, 128)) for num_stages in (3, 4)
    ]


CONFIGS = _full_configs


def _smoke_configs():
    return [_config(128, 256, 3)]


SMOKE_CONFIGS = _smoke_configs


def heuristic_config(M, N, K):
    del K
    block_m, block_n = (256, 128) if M >= N else (128, 256)
    return [_config(block_m, block_n, 3)]


@triton.jit
# Triton TR001: `_tuned` applies caller-selected autotune spaces lazily.
# triton-lint: assume NUM_STAGES=4, NUM_MMA_GROUPS=2
def matmul_kernel_tma_ws_hopper_cooperative(  # noqa: TR001
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    NUM_SMS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    A_ROW_MAJOR: tl.constexpr,
    B_ROW_MAJOR: tl.constexpr,
):
    block_m_split: tl.constexpr = BM // NUM_MMA_GROUPS

    if A_ROW_MAJOR:
        a = tlx.local_alloc(
            (block_m_split, BK),
            tlx.dtype_of(a_desc),
            NUM_STAGES * NUM_MMA_GROUPS,
        )
    else:
        a = tlx.local_alloc(
            (BK, block_m_split),
            tlx.dtype_of(a_desc),
            NUM_STAGES * NUM_MMA_GROUPS,
        )
    if B_ROW_MAJOR:
        b = tlx.local_alloc((BK, BN), tlx.dtype_of(b_desc), NUM_STAGES)
    else:
        b = tlx.local_alloc((BN, BK), tlx.dtype_of(b_desc), NUM_STAGES)

    bars_empty_a = tlx.alloc_barriers(
        num_barriers=NUM_STAGES * NUM_MMA_GROUPS,
        arrive_count=1,
    )
    bars_full_a = tlx.alloc_barriers(
        num_barriers=NUM_STAGES * NUM_MMA_GROUPS,
        arrive_count=1,
    )
    bars_empty_b = tlx.alloc_barriers(
        num_barriers=NUM_STAGES,
        arrive_count=NUM_MMA_GROUPS,
    )
    bars_full_b = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)

    with tlx.async_tasks():
        with tlx.async_task("default"):
            start_pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BM)
            num_pid_n = tl.cdiv(N, BN)
            num_tiles = num_pid_m * num_pid_n
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            phase = 1
            buf = 0

            for tile_id in range(start_pid, num_tiles, NUM_SMS):
                group_id = tile_id // num_pid_in_group
                first_pid_m = group_id * GROUP_SIZE_M
                group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
                pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
                pid_n = (tile_id % num_pid_in_group) // group_size_m
                offset_am = pid_m * BM
                offset_bn = pid_n * BN

                for k in range(0, tl.cdiv(K, BK)):
                    offset_k = k * BK

                    empty_a_0 = tlx.local_view(bars_empty_a, buf)
                    full_a_0 = tlx.local_view(bars_full_a, buf)
                    # Triton TR051: bounded unrolling misses the matching release
                    # after the nested persistent loop revisits this ring slot.
                    tlx.barrier_wait(empty_a_0, phase)  # noqa: TR051
                    tlx.barrier_expect_bytes(
                        full_a_0,
                        block_m_split * BK * tlx.size_of(tlx.dtype_of(a_desc)),
                    )
                    data_a_0 = tlx.local_view(a, buf)
                    if A_ROW_MAJOR:
                        tlx.async_descriptor_load(
                            a_desc,
                            data_a_0,
                            [offset_am, offset_k],
                            full_a_0,
                        )
                    else:
                        tlx.async_descriptor_load(
                            a_desc,
                            data_a_0,
                            [offset_k, offset_am],
                            full_a_0,
                        )

                    empty_b = tlx.local_view(bars_empty_b, buf)
                    full_b = tlx.local_view(bars_full_b, buf)
                    tlx.barrier_wait(empty_b, phase)
                    tlx.barrier_expect_bytes(
                        full_b,
                        BN * BK * tlx.size_of(tlx.dtype_of(b_desc)),
                    )
                    data_b = tlx.local_view(b, buf)
                    if B_ROW_MAJOR:
                        tlx.async_descriptor_load(
                            b_desc,
                            data_b,
                            [offset_k, offset_bn],
                            full_b,
                        )
                    else:
                        tlx.async_descriptor_load(
                            b_desc,
                            data_b,
                            [offset_bn, offset_k],
                            full_b,
                        )

                    a_1_index = buf + NUM_STAGES
                    empty_a_1 = tlx.local_view(bars_empty_a, a_1_index)
                    full_a_1 = tlx.local_view(bars_full_a, a_1_index)
                    tlx.barrier_wait(empty_a_1, phase)
                    tlx.barrier_expect_bytes(
                        full_a_1,
                        block_m_split * BK * tlx.size_of(tlx.dtype_of(a_desc)),
                    )
                    data_a_1 = tlx.local_view(a, a_1_index)
                    if A_ROW_MAJOR:
                        tlx.async_descriptor_load(
                            a_desc,
                            data_a_1,
                            [offset_am + block_m_split, offset_k],
                            full_a_1,
                        )
                    else:
                        tlx.async_descriptor_load(
                            a_desc,
                            data_a_1,
                            [offset_k, offset_am + block_m_split],
                            full_a_1,
                        )

                    phase = phase ^ (buf == NUM_STAGES - 1)
                    buf = (buf + 1) % NUM_STAGES

        with tlx.async_task(num_warps=4, replicate=2, registers=232):
            start_pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BM)
            num_pid_n = tl.cdiv(N, BN)
            num_tiles = num_pid_m * num_pid_n
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            consumer_id: tl.constexpr = tlx.async_task_replica_id()
            phase = 0
            buf = 0

            for tile_id in range(start_pid, num_tiles, NUM_SMS):
                group_id = tile_id // num_pid_in_group
                first_pid_m = group_id * GROUP_SIZE_M
                group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
                pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
                pid_n = (tile_id % num_pid_in_group) // group_size_m
                offset_am = pid_m * BM
                offset_bn = pid_n * BN

                last_buf = buf
                a_index = buf + NUM_STAGES * consumer_id
                full_a = tlx.local_view(bars_full_a, a_index)
                full_b = tlx.local_view(bars_full_b, buf)
                tlx.barrier_wait(full_a, phase)
                tlx.barrier_wait(full_b, phase)

                data_a = tlx.local_view(a, a_index)
                data_b = tlx.local_view(b, buf)
                a_operand = data_a if A_ROW_MAJOR else tlx.local_trans(data_a)
                b_operand = data_b if B_ROW_MAJOR else tlx.local_trans(data_b)
                acc = tlx.async_dot(a_operand, b_operand)

                phase = phase ^ (buf == NUM_STAGES - 1)
                buf = (buf + 1) % NUM_STAGES

                for _ in range(1, tl.cdiv(K, BK)):
                    a_index = buf + NUM_STAGES * consumer_id
                    full_a = tlx.local_view(bars_full_a, a_index)
                    full_b = tlx.local_view(bars_full_b, buf)
                    tlx.barrier_wait(full_a, phase)
                    # Triton TR051: bounded unrolling misses the producer's TMA
                    # completion after the shared B ring wraps.
                    tlx.barrier_wait(full_b, phase)  # noqa: TR051

                    data_a = tlx.local_view(a, a_index)
                    data_b = tlx.local_view(b, buf)
                    a_operand = data_a if A_ROW_MAJOR else tlx.local_trans(data_a)
                    b_operand = data_b if B_ROW_MAJOR else tlx.local_trans(data_b)
                    acc = tlx.async_dot(a_operand, b_operand, acc)
                    acc = tlx.async_dot_wait(1, acc)

                    empty_a = tlx.local_view(
                        bars_empty_a,
                        last_buf + NUM_STAGES * consumer_id,
                    )
                    empty_b = tlx.local_view(bars_empty_b, last_buf)
                    tlx.barrier_arrive(empty_a)
                    tlx.barrier_arrive(empty_b)

                    last_buf = buf
                    phase = phase ^ (buf == NUM_STAGES - 1)
                    buf = (buf + 1) % NUM_STAGES

                acc = tlx.async_dot_wait(0, acc)
                empty_a = tlx.local_view(
                    bars_empty_a,
                    last_buf + NUM_STAGES * consumer_id,
                )
                empty_b = tlx.local_view(bars_empty_b, last_buf)
                tlx.barrier_arrive(empty_a)
                tlx.barrier_arrive(empty_b)

                offset_cm = offset_am + block_m_split * consumer_id
                if EPILOGUE_SUBTILE:
                    acc = tl.reshape(acc, (block_m_split, 2, BN // 2))
                    acc = tl.permute(acc, (0, 2, 1))
                    acc_0, acc_1 = tl.split(acc)
                    c_desc.store(
                        [offset_cm, offset_bn],
                        acc_0.to(tlx.dtype_of(c_desc)),
                    )
                    c_desc.store(
                        [offset_cm, offset_bn + BN // 2],
                        acc_1.to(tlx.dtype_of(c_desc)),
                    )
                else:
                    c_desc.store(
                        [offset_cm, offset_bn],
                        acc.to(tlx.dtype_of(c_desc)),
                    )


@functools.lru_cache(maxsize=None)
def _tuned(space, shape=None):
    if space == "heuristic":
        configs = heuristic_config(*shape)
    elif space == "full":
        configs = CONFIGS()
    elif space == "smoke":
        configs = SMOKE_CONFIGS()
    else:
        raise InvalidInput(f"sm90 tlx.ops.mm does not provide search space {space!r}; "
                           "expected 'heuristic', 'full', or 'smoke'")
    return triton.autotune(
        configs=configs,
        key=["M", "N", "K"],
    )(matmul_kernel_tma_ws_hopper_cooperative)


def mm(a, b, *, out=None, space="full"):
    """Compute ``a @ b`` with the Hopper cooperative persistent GEMM."""
    M, K = a.shape
    _, N = b.shape

    if min(M, N, K) <= 0:
        raise InvalidInput("sm90 tlx.ops.mm requires positive M, N, and K")
    if a.data_ptr() % 16 != 0 or b.data_ptr() % 16 != 0:
        raise InvalidInput("sm90 tlx.ops.mm requires 16-byte-aligned operands")
    if not (a.is_contiguous() or a.T.is_contiguous()):
        raise InvalidInput("sm90 tlx.ops.mm requires A to be row- or column-major")
    if not (b.is_contiguous() or b.T.is_contiguous()):
        raise InvalidInput("sm90 tlx.ops.mm requires B to be row- or column-major")

    if out is not None:
        if (out.shape != (M, N) or out.device != a.device or out.dtype != a.dtype or not out.is_contiguous()):
            raise InvalidInput(f"out must be a contiguous {a.dtype} tensor with shape "
                               f"({M}, {N}) on A's device")
        c = out
    else:
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    if c.data_ptr() % 16 != 0:
        raise InvalidInput("sm90 tlx.ops.mm requires a 16-byte-aligned output")

    a_row_major = a.is_contiguous()
    b_row_major = b.is_contiguous()
    a_src = a if a_row_major else a.T
    b_src = b if b_row_major else b.T

    dummy_block = [1, 1]
    a_desc = TensorDescriptor(a_src, a_src.shape, a_src.stride(), dummy_block)
    b_desc = TensorDescriptor(b_src, b_src.shape, b_src.stride(), dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    num_sms = _get_num_sms()

    def grid(meta):
        num_tiles = triton.cdiv(M, meta["BM"]) * triton.cdiv(N, meta["BN"])
        return (min(num_sms, num_tiles), )

    kernel = _tuned(space, (M, N, K) if space == "heuristic" else None)
    kernel[grid](
        a_desc,
        b_desc,
        c_desc,
        M,
        N,
        K,
        NUM_SMS=num_sms,
        A_ROW_MAJOR=a_row_major,
        B_ROW_MAJOR=b_row_major,
    )
    return c
