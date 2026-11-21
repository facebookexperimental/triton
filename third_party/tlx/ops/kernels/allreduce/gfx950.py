"""Single-node BF16 all-reduce over peer-mapped (IPC) memory, on gfx950.

A TLX port of aiter's custom all-reduce, which SGLang uses for tensor-parallel all-reduce on MI350X.

``one_stage`` ports ``aiter::cross_device_reduce_1stage``: every rank reads the whole message from every peer and
writes the full sum. It is aiter's choice for small messages, where reading ``world_size`` times the message is cheap
and a single pass needs no intermediate buffer.

The kernel brackets its data movement with per-program barriers, as aiter does. A barrier writes a flag into every
peer's signal block and waits until every peer's flag has arrived in its own. The start barrier makes sure every
peer's input is ready; the end barrier keeps a rank from overwriting its input while a peer is still reading it.
Program ``i`` of every rank covers the same elements, which is what lets a barrier between programs ``i`` stand in for
a barrier between kernels.

Note: the flag counter advances by 1 per call here (aiter's ``start_sync``/``end_sync`` each advance ``_flag`` by 1,
i.e. 2 per call), so this signal block cannot be shared with aiter's kernels: sharing it would deadlock or pass
through a barrier early. Like aiter's, the uint32 counter wraps after 2**32 calls, past which the barriers would pass
spuriously; a process never makes that many calls between restarts, so this is documented, not handled.

Peer buffers are passed as tuples of pointers, one per rank, so every load is a 16-byte vector access. The sum is
taken in fp32 in rank order, so every rank produces bitwise identical output.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from ._ipc import DevicePtr, HIP_DEVICE_MALLOC_DEFAULT, HIP_DEVICE_MALLOC_UNCACHED, IpcBuffer

MAX_BLOCKS = 80
MAX_RANKS = 8
# aiter's Signal block, in uint32 words: start[MAX_BLOCKS][MAX_RANKS], end[MAX_BLOCKS][MAX_RANKS], flag[MAX_BLOCKS].
_START_OFF = tl.constexpr(0)
_END_OFF = tl.constexpr(MAX_BLOCKS * MAX_RANKS)
_FLAG_OFF = tl.constexpr(2 * MAX_BLOCKS * MAX_RANKS)
_MAX_RANKS = tl.constexpr(MAX_RANKS)
SIGNAL_BYTES = 64 * 1024
# Tuned one-stage program count (swept at 4 and 8 ranks; see bench.py --tlx-blocks): 16 wins for large messages,
# where fewer programs spend less time in the per-program barriers. Small messages are grid-capped by size either way.
ONE_STAGE_BLOCKS = 16
PACK = 8  # bf16 elements per 16-byte access
THREADS = 512


@triton.jit
def _select(ptrs, idx, NGPUS: tl.constexpr):
    ptr = ptrs[0]
    for p in tl.static_range(1, NGPUS):
        ptr = tl.where(idx == p, ptrs[p], ptr)
    return ptr


@triton.jit
def _signal(sigs, ARRAY_OFF: tl.constexpr, pid, rank, flag, NGPUS: tl.constexpr):
    # One lane per peer, so the NGPUS remote writes are in flight together.
    lanes = tl.arange(0, NGPUS)
    dst = _select(sigs, lanes, NGPUS) + ARRAY_OFF + pid * _MAX_RANKS + rank
    tl.store(dst, tl.full([NGPUS], 0, tl.uint32) + flag)


@triton.jit
def _wait(self_sig, ARRAY_OFF: tl.constexpr, pid, flag, NGPUS: tl.constexpr):
    # volatile: without it the load is hoisted out of the loop, and the loop out of the kernel.
    mine = self_sig + ARRAY_OFF + pid * _MAX_RANKS + tl.arange(0, NGPUS)
    while tl.min(tl.load(mine, cache_modifier=".cv", volatile=True), axis=0) < flag:
        pass


@triton.jit
def _one_stage_kernel(inps, sigs, self_sig, out, rank, N, NGPUS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    flag = tl.load(self_sig + _FLAG_OFF + pid) + 1
    _signal(sigs, _START_OFF, pid, rank, flag, NGPUS)
    _wait(self_sig, _START_OFF, pid, flag, NGPUS)

    cols = tl.arange(0, BLOCK)
    for start in range(pid * BLOCK, N, nprog * BLOCK):
        offs = start + cols
        mask = offs < N
        acc = tl.load(inps[0] + offs, mask=mask, other=0.0).to(tl.float32)
        for p in tl.static_range(1, NGPUS):
            acc += tl.load(inps[p] + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out + offs, acc.to(tl.bfloat16), mask=mask)

    # Every peer must be done reading this rank's input before this rank's next kernel can overwrite it.
    tl.debug_barrier()
    _signal(sigs, _END_OFF, pid, rank, flag, NGPUS)
    _wait(self_sig, _END_OFF, pid, flag, NGPUS)
    tl.store(self_sig + _FLAG_OFF + pid, flag)


class CustomAllReduce:
    """Peer-mapped buffers and launchers for one tensor-parallel group.

    ``input_buffer(numel)`` returns this rank's registered input; an all-reduce reads every rank's registered input,
    so callers write the value to reduce there (or pass any tensor to ``all_reduce`` to have it copied in). Collective
    over ``group``.
    """

    def __init__(self, group, *, max_bytes: int = 8 * 1024 * 1024):
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        if self.world_size not in (2, 4, 8):
            raise ValueError(f"all-reduce supports 2, 4 or 8 ranks; got {self.world_size}")
        self.max_bytes = max_bytes
        self.group = group
        kw = dict(rank=self.rank, world_size=self.world_size, group=group)
        self._signal = IpcBuffer(SIGNAL_BYTES, flags=HIP_DEVICE_MALLOC_UNCACHED, **kw)
        self._input = IpcBuffer(max_bytes, flags=HIP_DEVICE_MALLOC_DEFAULT, **kw)
        self._inps = tuple(DevicePtr(p, torch.bfloat16) for p in self._input.ptrs)
        self._sigs = tuple(DevicePtr(p, torch.uint32) for p in self._signal.ptrs)
        self._self_sig = self._sigs[self.rank]

    def close(self):
        self._signal.close()
        self._input.close()

    def input_buffer(self, numel: int) -> torch.Tensor:
        return self._input.local_tensor(numel, torch.bfloat16)

    def all_reduce(self, x: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
        n = x.numel()
        if x.dtype != torch.bfloat16 or n % PACK != 0 or n * 2 > self.max_bytes:
            raise ValueError("all_reduce expects bf16, numel % 8 == 0 and fits max_bytes")
        if x.data_ptr() != self._input.ptrs[self.rank]:
            self.input_buffer(n).copy_(x.reshape(-1))
        if out is None:
            out = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
        if out.dtype != torch.bfloat16 or out.numel() != n:
            raise ValueError("all_reduce needs out with the same bf16 numel as x")
        if not out.is_contiguous():
            raise ValueError("all_reduce needs a contiguous out; the kernel writes it as a flat buffer")
        self.one_stage(n, out.view(-1))
        return out

    def one_stage(self, n: int, out: torch.Tensor, *, max_blocks: int = ONE_STAGE_BLOCKS):
        """All-reduce the first ``n`` elements of every rank's input buffer into ``out``."""
        if n % PACK != 0 or n * 2 > self.max_bytes:
            raise ValueError("one_stage expects numel % 8 == 0 and fits max_bytes")
        if out.dtype != torch.bfloat16 or out.numel() < n or not out.is_contiguous():
            raise ValueError("one_stage needs a contiguous bf16 out holding at least n elements")
        out_ptr, in_ptr = out.data_ptr(), self._input.ptrs[self.rank]
        if out_ptr < in_ptr + n * 2 and in_ptr < out_ptr + n * 2:
            raise ValueError("one_stage cannot write into this rank's input buffer: "
                             "peers may still be reading it")
        if not 1 <= max_blocks <= MAX_BLOCKS:
            raise ValueError(f"one_stage supports 1 to {MAX_BLOCKS} programs; got {max_blocks}")
        block = THREADS * PACK
        grid = (min(max_blocks, triton.cdiv(n, block)), )
        _one_stage_kernel[grid](self._inps, self._sigs, self._self_sig, out, self.rank, n, NGPUS=self.world_size,
                                BLOCK=block, num_warps=THREADS // 64)
