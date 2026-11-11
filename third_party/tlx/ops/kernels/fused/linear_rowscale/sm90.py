"""Linear layer with a per-row output scale, forward, for sm90 (H100). FP16 or BF16.

    out = ((x @ weight.T + bias) * row_scale[:, None]).to(x.dtype)

with ``x`` of shape ``(M, K)``, ``weight`` of shape ``(N, K)`` (``nn.Linear`` layout), an optional ``bias`` of ``N``
elements and an optional FP32 ``row_scale`` of ``M`` elements, accumulated in FP32 and rounded once. It targets wide,
short-K products (``K <= 512``, ``N`` in the tens of thousands, ``M`` up to a few thousand), whose output is much larger
than their inputs: a row scale that a caller would otherwise apply in a separate pass over that output (a deferred
normalization) is applied to the accumulator instead.

The kernel is persistent and warp-specialized. The CTAs are split into one group per 128-row block of ``x``; each CTA
keeps its block of ``x`` (all of K) in shared memory and walks every ``cpm``-th 128-column tile of ``weight``, so the
groups read each weight tile at about the same time and the weight comes from DRAM once. A producer warp streams the
weight tiles through a ring of TMA buffers, and two consumer warpgroups take alternate tiles in a ping-pong: named
barriers order their wgmma phases, so one consumer's epilogue (bias, row scale, rounding and TMA store) overlaps the
other's products. The TMA descriptors are built on the device from the pointers, which also works where a caller
cannot pass host descriptors (AOTInductor in older torch releases).

The schedule balances while the row blocks are few against the SMs, ``cdiv(M, 128) <= SMs // 2``, or close to a
multiple of the SM count. In between, each row block gets one CTA and the other SMs idle: on 132 SMs, ``M`` from 8,449
to 16,896 rows keeps only ``cdiv(M, 128)`` of them busy.

Each output element's accumulation order depends only on K, so a row's result does not depend on ``M`` or on the other
rows.
"""

import functools
from typing import NamedTuple

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from torch._library.triton import wrap_triton
from torch._subclasses.fake_tensor import FakeTensor, is_fake

from ...._catalog import InvalidInput

BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 64
# Shared memory holds K_TILES x-blocks and NUM_STAGES weight tiles of 16 KB each plus 32 KB of output staging; their
# sum stays at 12 to fit the 227 KB per CTA.
SMEM_TILES = 12
MAX_K = 512
PRODUCER_REGISTERS = 40
CONSUMER_REGISTERS = 232
# Per CTA: the x and weight descriptors and one output descriptor per consumer.
DESC_SLOTS = 4
DESC_BYTES = 128


@triton.jit
def _linear_rowscale_fwd(
    x_ptr,
    w_ptr,
    out_ptr,
    desc_slots,
    bias_ptr,
    scale_ptr,
    stride_xm,
    stride_wn,
    stride_om,
    M,
    N,
    K,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    K_TILES: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    PRODUCER_REGISTERS: tl.constexpr,
    CONSUMER_REGISTERS: tl.constexpr,
    DESC_SLOTS: tl.constexpr,
    DESC_BYTES: tl.constexpr,
):
    BN_EPI: tl.constexpr = BN // 2
    x_s = tlx.local_alloc((BM, BK), tlx.dtype_of(x_ptr), K_TILES)
    w_s = tlx.local_alloc((BN, BK), tlx.dtype_of(w_ptr), NUM_STAGES)
    x_full = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    x_empty = tlx.alloc_barriers(num_barriers=1, arrive_count=2)
    w_full = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
    w_empty = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
    out_s = tlx.local_alloc((BM, BN_EPI), tlx.dtype_of(out_ptr), 2)
    slots_base = desc_slots + tl.program_id(0) * DESC_SLOTS * DESC_BYTES
    M32 = tl.cast(M, tl.int32)
    N32 = tl.cast(N, tl.int32)
    K32 = tl.cast(K, tl.int32)
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    num_pid_m = tl.cdiv(M32, BM)
    num_pid_n = tl.cdiv(N32, BN)
    # cpm CTAs per row block; CTAs past m_slots * cpm, and every CTA when M == 0, have no work.
    cpm = tl.maximum(nprog // tl.maximum(num_pid_m, 1), 1)
    m_slots = nprog // cpm
    m_first = pid // cpm
    n_first = pid % cpm

    with tlx.async_tasks():
        with tlx.async_task("default", registers=PRODUCER_REGISTERS):
            slots = tlx.tensor_descriptor_ptr(slots_base.handle, DESC_SLOTS, DESC_BYTES)
            x_desc = tlx.make_tensor_descriptor(slots[0], x_ptr, shape=[M32, K32], strides=[stride_xm, 1],
                                                block_shape=[BM, BK])
            w_desc = tlx.make_tensor_descriptor(slots[1], w_ptr, shape=[N32, K32], strides=[stride_wn, 1],
                                                block_shape=[BN, BK])
            phase = 1
            buf = 0
            x_phase = 1
            for pid_m in range(m_first, num_pid_m, m_slots):
                tlx.barrier_wait(tlx.local_view(x_empty, 0), x_phase)
                xf = tlx.local_view(x_full, 0)
                tlx.barrier_expect_bytes(xf, K_TILES * BM * BK * tlx.size_of(tlx.dtype_of(x_desc)))
                for kt in tl.static_range(K_TILES):
                    # TMA zero-fills the columns past K and the rows past M.
                    tlx.async_descriptor_load(x_desc, tlx.local_view(x_s, kt), [pid_m * BM, kt * BK], xf)
                x_phase = x_phase ^ 1
                for pid_n in range(n_first, num_pid_n, cpm):
                    for kt in tl.static_range(K_TILES):
                        tlx.barrier_wait(tlx.local_view(w_empty, buf), phase)  # noqa: TR051
                        wf = tlx.local_view(w_full, buf)
                        tlx.barrier_expect_bytes(wf, BN * BK * tlx.size_of(tlx.dtype_of(w_desc)))
                        tlx.async_descriptor_load(w_desc, tlx.local_view(w_s, buf), [pid_n * BN, kt * BK], wf)
                        phase = phase ^ (buf == NUM_STAGES - 1)
                        buf = (buf + 1) % NUM_STAGES

        with tlx.async_task(num_warps=4, replicate=2, registers=CONSUMER_REGISTERS):
            cid: tl.constexpr = tlx.async_task_replica_id()
            slots = tlx.tensor_descriptor_ptr(slots_base.handle, DESC_SLOTS, DESC_BYTES)
            out_desc = tlx.make_tensor_descriptor(slots[2 + cid], out_ptr, shape=[M32, N32], strides=[stride_om, 1],
                                                  block_shape=[BM, BN_EPI])
            if cid == 1:
                tlx.named_barrier_arrive(9, 256)
            x_phase = 0
            # Tiles this CTA ran before the current row block: the ring order is global, and consumers alternate tiles.
            base = 0
            for pid_m in range(m_first, num_pid_m, m_slots):
                offset_m = pid_m * BM
                if HAS_SCALE:
                    offs_m = offset_m + tl.arange(0, BM)
                    s = tl.load(scale_ptr + offs_m, mask=offs_m < M, other=0.0)
                tlx.barrier_wait(tlx.local_view(x_full, 0), x_phase)
                x_phase = x_phase ^ 1
                n_count = tl.cdiv(num_pid_n - n_first, cpm)
                for j in range((cid + base) % 2, n_count, 2):
                    offset_n = (n_first + j * cpm) * BN
                    offs_n = offset_n + tl.arange(0, BN)
                    if HAS_BIAS:
                        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
                    k_offset = (base + j) * K_TILES
                    buf = k_offset % NUM_STAGES
                    phase = (k_offset // NUM_STAGES) % 2
                    if cid == 0:
                        tlx.named_barrier_wait(9, 256)
                    else:
                        tlx.named_barrier_wait(10, 256)
                    last_buf = buf
                    tlx.barrier_wait(tlx.local_view(w_full, buf), phase)
                    acc = tlx.async_dot(tlx.local_view(x_s, 0), tlx.local_trans(tlx.local_view(w_s, buf)))
                    phase = phase ^ (buf == NUM_STAGES - 1)
                    buf = (buf + 1) % NUM_STAGES
                    for kt in tl.static_range(1, K_TILES):
                        tlx.barrier_wait(tlx.local_view(w_full, buf), phase)  # noqa: TR051
                        acc = tlx.async_dot(tlx.local_view(x_s, kt), tlx.local_trans(tlx.local_view(w_s, buf)), acc)
                        acc = tlx.async_dot_wait(1, acc)
                        tlx.barrier_arrive(tlx.local_view(w_empty, last_buf))
                        last_buf = buf
                        phase = phase ^ (buf == NUM_STAGES - 1)
                        buf = (buf + 1) % NUM_STAGES
                    if cid == 0:
                        tlx.named_barrier_arrive(10, 256)
                    else:
                        tlx.named_barrier_arrive(9, 256)
                    acc = tlx.async_dot_wait(0, acc)
                    tlx.barrier_arrive(tlx.local_view(w_empty, last_buf))

                    if HAS_BIAS:
                        acc = acc + bias.to(tl.float32)[None, :]
                    if HAS_SCALE:
                        acc = acc * s[:, None]
                    c = tl.permute(tl.reshape(acc.to(tlx.dtype_of(out_ptr)), (BM, 2, BN_EPI)), (0, 2, 1))
                    c0, c1 = tl.split(c)
                    # Two half-tile stores through one buffer per consumer: the wait before each rewrite is the
                    # price of a deeper weight ring.
                    slot = tlx.local_view(out_s, cid)
                    tlx.async_descriptor_store_wait(0)
                    tlx.local_store(slot, c0)
                    tlx.fence("async_shared")
                    tlx.async_descriptor_store(out_desc, slot, [offset_m, offset_n])
                    tlx.async_descriptor_store_wait(0)
                    tlx.local_store(slot, c1)
                    tlx.fence("async_shared")
                    tlx.async_descriptor_store(out_desc, slot, [offset_m, offset_n + BN_EPI])
                tlx.barrier_arrive(tlx.local_view(x_empty, 0))
                base += n_count
            tlx.async_descriptor_store_wait(0)


class KernelLaunch(NamedTuple):
    """``kernel[grid](*args, **kwargs)``; ``kernel`` is a plain ``triton.jit`` function."""
    kernel: object
    grid: tuple
    args: tuple
    kwargs: dict


@functools.lru_cache(maxsize=None)
def _num_sms(index):
    return torch.cuda.get_device_properties(index).multi_processor_count


def _aligned(t):
    """Whether ``t`` starts on a 16-byte boundary.

    A traced tensor has no address, so its storage offset decides: Inductor treats an input as aligned only when the
    traced input was.
    """
    if is_fake(t):
        return t.storage_offset() * t.element_size() % 16 == 0
    return t.data_ptr() % 16 == 0


def _tma_readable(t):
    """Whether TMA can address the 2-D ``t`` in place: unit column stride, disjoint 16-byte rows, aligned base."""
    return t.stride(1) == 1 and t.stride(0) % 8 == 0 and t.stride(0) >= t.shape[1] and _aligned(t)


def _check_operands(x, weight, bias, row_scale, out):
    """Checks the launch operands; returns ``(M, N, K)``."""
    if x.dim() != 2 or weight.dim() != 2 or weight.shape[1] != x.shape[1]:
        raise InvalidInput("sm90 tlx.ops.linear_rowscale expects x of shape (M, K) and weight of shape (N, K); got "
                           f"{tuple(x.shape)} and {tuple(weight.shape)}")
    M, K = x.shape
    N = weight.shape[0]
    dtype, device = x.dtype, x.device
    for name, t in (("x", x), ("weight", weight), ("out", out)):
        if t.dtype != dtype or t.device != device:
            raise InvalidInput(f"{name} must be a {dtype} tensor on {device}")
        if not _tma_readable(t):
            raise InvalidInput(f"{name} must have a unit column stride and a 16-byte-aligned base and row stride")
    if tuple(out.shape) != (M, N):
        raise InvalidInput(f"out must have shape {(M, N)}; got {tuple(out.shape)}")
    if bias is not None and (tuple(bias.shape) != (N, ) or bias.dtype != dtype or bias.device != device
                             or bias.stride(0) != 1):
        raise InvalidInput(f"bias must be a contiguous {dtype} tensor of shape ({N},) on {device}")
    if row_scale is not None and (tuple(row_scale.shape) != (M, ) or row_scale.dtype != torch.float32
                                  or row_scale.device != device or row_scale.stride(0) != 1):
        raise InvalidInput(f"row_scale must be a contiguous float32 tensor of shape ({M},) on {device}")
    return M, N, K


def linear_rowscale_launch(x, weight, bias, row_scale, *, out):
    """The launch of :func:`linear_rowscale` into ``out`` as a :class:`KernelLaunch`.

    For callers that launch the kernel themselves, such as a ``torch.library.triton_op`` that calls
    ``wrap_triton(kernel)[grid](...)`` so that torch.export / AOTInductor compile the kernel into the artifact. ``x``,
    ``weight`` and ``out`` must be TMA-readable as they are (see :func:`_check_operands`); nothing is copied. The grid
    is one CTA per SM whatever M is, so a symbolic M is fine. The TMA descriptors are built on the device in a small
    scratch tensor that the launch allocates.
    """
    M, N, K = _check_operands(x, weight, bias, row_scale, out)
    num_ctas = _num_sms(x.device.index if x.device.index is not None else torch.cuda.current_device())
    desc_slots = torch.empty(num_ctas * DESC_SLOTS * DESC_BYTES, device=x.device, dtype=torch.int8)
    k_tiles = triton.cdiv(K, BLOCK_K)
    args = (x, weight, out, desc_slots, bias if bias is not None else x, row_scale if row_scale is not None else x,
            x.stride(0), weight.stride(0), out.stride(0), M, N, K)
    kwargs = dict(BM=BLOCK_M, BN=BLOCK_N, BK=BLOCK_K, K_TILES=k_tiles, NUM_STAGES=SMEM_TILES - k_tiles, HAS_BIAS=bias
                  is not None, HAS_SCALE=row_scale is not None, PRODUCER_REGISTERS=PRODUCER_REGISTERS,
                  CONSUMER_REGISTERS=CONSUMER_REGISTERS, DESC_SLOTS=DESC_SLOTS, DESC_BYTES=DESC_BYTES, num_warps=4,
                  num_stages=1)
    return KernelLaunch(_linear_rowscale_fwd, (num_ctas, ), args, kwargs)


def linear_rowscale(x, weight, bias, row_scale):
    """Launch the forward; ``tlx.ops.linear_rowscale`` validates shapes and dtypes.

    ``x`` and ``weight`` that TMA cannot address in place are copied to fresh, contiguous storage first.
    """
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    # A symbolic M is not compared, so that tracing adds no guard; the kernel runs no tile when M == 0.
    if isinstance(x, FakeTensor) or (isinstance(M, int) and M == 0):
        return out
    x = x if _tma_readable(x) else x.clone(memory_format=torch.contiguous_format)
    weight = weight if _tma_readable(weight) else weight.clone(memory_format=torch.contiguous_format)
    bias = bias.contiguous() if bias is not None else None
    row_scale = row_scale.contiguous() if row_scale is not None else None
    launch = linear_rowscale_launch(x, weight, bias, row_scale, out=out)
    # Only tracing needs the wrapper; in eager mode it costs about 0.2 ms of host time per call.
    kernel = wrap_triton(launch.kernel) if torch.compiler.is_compiling() else launch.kernel
    with torch.cuda.device(x.device):
        kernel[launch.grid](*launch.args, **launch.kwargs)
    return out
