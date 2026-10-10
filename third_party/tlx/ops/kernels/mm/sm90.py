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
from ._sm90_core import _allocate_buffers, _consume_tile, _run_producer, _split_acc, _store_subtile

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
    nargs["c_desc"].block_shape = [block_m_split, block_n // 4 if nargs["EPILOGUE_SUBTILE"] else block_n]


def _config(block_m, block_n, num_stages, group_size_m=8):
    return triton.Config(
        {
            "BM": block_m,
            "BN": block_n,
            "BK": 64,
            "GROUP_SIZE_M": group_size_m,
            "NUM_STAGES": num_stages,
            "NUM_MMA_GROUPS": 2,
            "EPILOGUE_SUBTILE": True,
        },
        num_stages=1,
        num_warps=4,
        pre_hook=matmul_tma_set_block_size_hook,
    )


def _full_configs():
    # GROUP_SIZE_M=1 walks tiles row-major so all N-tiles of a narrow-N, tall-M
    # product share the A panel in L2; square shapes prefer the grouped order.
    return [
        _config(block_m, block_n, num_stages, group_size_m)
        for block_m, block_n in ((128, 256), (256, 128))
        for num_stages in (3, 4)
        for group_size_m in (1, 8)
    ]


CONFIGS = _full_configs


def _prune_full_configs(configs, nargs, **kwargs):
    # With N <= 512 only 2-4 tiles span N, and grouped order (G=8) is bimodal on
    # tall shapes (e.g. 2Mx512x512 runs at 420 or ~480 TFLOPS), so autotune can
    # lock in a slow config from one lucky sample. Row-major order is stable.
    if nargs["N"] > 512:
        return configs
    return [c for c in configs if c.kwargs["GROUP_SIZE_M"] == 1] or configs


def _smoke_configs():
    return [_config(128, 256, 3)]


SMOKE_CONFIGS = _smoke_configs


def heuristic_config(M, N, K):
    # Square shapes favor 128x256 tiles. The 4-stage ring (the SMEM max) wins at
    # large K but regresses some tall shapes with K <= 512.
    block_m, block_n = (256, 128) if M > N else (128, 256)
    num_stages = 4 if K >= 1024 else 3
    return [_config(block_m, block_n, num_stages)]


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
    A_EVICTION: tl.constexpr,
):
    block_m_split: tl.constexpr = BM // NUM_MMA_GROUPS
    a, b, bars_empty_a, bars_full_a, bars_empty_b, bars_full_b = _allocate_buffers(
        a_desc,
        b_desc,
        BM,
        BN,
        BK,
        NUM_STAGES,
        NUM_MMA_GROUPS,
        A_ROW_MAJOR,
        B_ROW_MAJOR,
    )
    if EPILOGUE_SUBTILE:
        c_smem = tlx.local_alloc((block_m_split, BN // 4), tlx.dtype_of(c_desc), 2 * NUM_MMA_GROUPS)

    with tlx.async_tasks():
        with tlx.async_task("default"):
            _run_producer(
                M,
                N,
                K,
                a_desc,
                b_desc,
                a,
                b,
                bars_empty_a,
                bars_full_a,
                bars_empty_b,
                bars_full_b,
                NUM_SMS,
                BM,
                BN,
                BK,
                GROUP_SIZE_M,
                NUM_STAGES,
                NUM_MMA_GROUPS,
                A_ROW_MAJOR,
                B_ROW_MAJOR,
                A_EVICTION,
            )

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

                acc, buf, phase = _consume_tile(
                    K,
                    a,
                    b,
                    bars_empty_a,
                    bars_full_a,
                    bars_empty_b,
                    bars_full_b,
                    buf,
                    phase,
                    consumer_id,
                    BK,
                    NUM_STAGES,
                    A_ROW_MAJOR,
                    B_ROW_MAJOR,
                )

                offset_cm = offset_am + block_m_split * consumer_id
                if EPILOGUE_SUBTILE:
                    acc_0, acc_1, acc_2, acc_3 = _split_acc(acc, block_m_split, BN)
                    slot_0 = tlx.local_view(c_smem, 2 * consumer_id)
                    slot_1 = tlx.local_view(c_smem, 2 * consumer_id + 1)
                    _store_subtile(c_desc, slot_0, acc_0, offset_cm, offset_bn)
                    _store_subtile(c_desc, slot_1, acc_1, offset_cm, offset_bn + BN // 4)
                    _store_subtile(c_desc, slot_0, acc_2, offset_cm, offset_bn + BN // 2)
                    _store_subtile(c_desc, slot_1, acc_3, offset_cm, offset_bn + 3 * (BN // 4))
                else:
                    c_desc.store(
                        [offset_cm, offset_bn],
                        acc.to(tlx.dtype_of(c_desc)),
                    )

            if EPILOGUE_SUBTILE:
                tlx.async_descriptor_store_wait(0)


@functools.lru_cache(maxsize=None)
def _tuned(space, shape=None):
    prune_configs_by = None
    if space == "heuristic":
        configs = heuristic_config(*shape)
    elif space == "full":
        configs = CONFIGS()
        prune_configs_by = {"early_config_prune": _prune_full_configs}
    elif space == "smoke":
        configs = SMOKE_CONFIGS()
    else:
        raise InvalidInput(f"sm90 tlx.ops.mm does not provide search space {space!r}; "
                           "expected 'heuristic', 'full', or 'smoke'")
    return triton.autotune(
        configs=configs,
        key=["M", "N", "K"],
        prune_configs_by=prune_configs_by,
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
        # With N <= 1024 only a few N-tiles read each A panel, back to back;
        # keeping it in L2 serves the repeat reads. Larger N regresses slightly.
        A_EVICTION="evict_last" if N <= 1024 else "",
    )
    return c
