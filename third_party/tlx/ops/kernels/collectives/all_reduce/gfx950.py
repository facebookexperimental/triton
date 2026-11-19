"""Intra-node BF16 all-reduce over xGMI peer pointers on gfx950.

Every rank reads the peers' buffers directly. Each lane owns 8 contiguous elements of a chunk and loads that 16-byte
slice from all ``W`` peers at once (a pinned TLX layout keeps the ``W`` copies in one lane's registers), then adds them
in FP32 in rank order and rounds to BF16. Every path uses this one sum, so all ranks, both algorithms and the fused
ops in ``collectives/fused`` produce the same bits for an element, independent of the token count.

- One-shot: every rank reduces the whole tensor. Two barriers, one read round, ``W`` times the remote traffic.
- Two-shot: rank ``r`` reduces partition ``r`` into its scratch buffer; after a barrier every rank copies the ``W``
  partitions from the owners. Two barriers: the next call's first barrier already orders the scratch reuse.

Remote reads are latency bound: a CU keeps a limited number of misses in flight, so the all-reduce kernels run one
wave per program and spread up to 128 programs over the CUs. Fewer, wider programs (4 waves on 2048 elements) take
1.5-1.7x as long from 9 to 150 tokens; more programs mostly contend for the links. One-shot programs reduce
512-element chunks (16-byte lane slices). A two-shot partition is small, so its kernel shrinks the chunk to 256 or 128
elements while that keeps one round of programs: from 11 to 32 tokens of 4096 this takes 12-20% off.

A barrier is per program: program ``p`` of each rank publishes a per-program counter into its slot of every peer's
signal buffer and spins on its own slots with plain volatile loads, which disturb the concurrent data traffic less
than atomic polling. The signal and scratch buffers are uncached device memory, like the HIP custom all-reduce's:
with cached memory a peer's flag can stay invisible for milliseconds.
"""

from __future__ import annotations

import ctypes

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

PROGRAMS = 128
MAX_PROGRAMS = 256
CHUNK = 512
ONE_SHOT_MAX_BYTES = 88 * 1024
_BARRIERS_PER_CALL = tl.constexpr(2)
# Row-local sync of the cooperative fused boundary in ``collectives/fused``, in uncached local memory: per row and
# phase, one 64-bit word per program (a call tag and an FP32 partial sum).
ROW_SYNC_ROWS = 32
ROW_SYNC_PARTS = 8
ROW_SYNC_BYTES = ROW_SYNC_ROWS * 2 * ROW_SYNC_PARTS * 8
_HIP_DEVICE_MALLOC_UNCACHED = 3
_HIP_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 1


def pick_all_reduce(numel: int, world: int) -> str:
    # Two-shot partitions are whole lane slices, so a length that is not a multiple of ``world * 8`` takes one-shot at
    # any size, with ``world`` times the remote traffic. Hidden sizes of 4096, 8192 and 16384 never do.
    if numel * 2 < ONE_SHOT_MAX_BYTES or numel % (world * 8):
        return "one_shot"
    return "two_shot"


def two_shot_config(numel: int, world: int) -> tuple[int, int]:
    """(chunk, programs) of the two-shot kernel: the smallest chunk that keeps one round of at most ``PROGRAMS``
    programs, else one round of up to 200 512-element chunks; larger partitions loop over 128-element chunks on
    ``MAX_PROGRAMS`` programs, and the largest over 512-element chunks on ``PROGRAMS``."""
    part = numel // world
    for chunk in (128, 256, 512):
        if part <= PROGRAMS * chunk:
            return chunk, triton.cdiv(part, chunk)
    if part <= 200 * CHUNK:
        return CHUNK, triton.cdiv(part, CHUNK)
    if part <= 2048 * 128:
        return 128, MAX_PROGRAMS
    return CHUNK, PROGRAMS


@triton.jit
def _signals(sig_ptrs, rank, W: tl.constexpr):
    peer_sig = tl.load(sig_ptrs + tl.arange(0, W)).to(tl.pointer_type(tl.int32))
    own = tl.load(sig_ptrs + rank).to(tl.pointer_type(tl.int32))
    return peer_sig, own


@triton.jit
def _barrier(peer_sig, own, rank, slot, value, W: tl.constexpr, RELEASE: tl.constexpr):
    base = (tl.program_id(0) * _BARRIERS_PER_CALL + slot) * W
    tl.debug_barrier()
    values = tl.full((W, ), value, tl.int32)
    if RELEASE:
        tl.atomic_store(peer_sig + base + rank, values, sem="release", scope="sys")
    else:
        tl.atomic_store(peer_sig + base + rank, values, sem="relaxed", scope="sys")
    flags = own + base + tl.arange(0, W)
    seen = tl.load(flags, volatile=True)
    while tl.min(tl.where(seen == value, 1, 0)) == 0:
        seen = tl.load(flags, volatile=True)
    if RELEASE:
        tl.atomic_load(flags, sem="acquire", scope="gpu")
    tl.debug_barrier()


@triton.jit
def _peer_bases(ptrs, W: tl.constexpr):
    base = tl.load(ptrs + tl.arange(0, W)).to(tl.pointer_type(tl.bfloat16))
    return tl.multiple_of(base, [16])[:, None]


@triton.constexpr_function
def _lane_layout(W, BLOCK, NW, VEC):
    return tlx.layout(shape=((64, NW), (VEC, W, BLOCK // (64 * VEC * NW))),
                      stride=((VEC, 64 * VEC), (1, BLOCK, 64 * VEC * NW)))


@triton.jit
def _reduce(peers, offs, mask, W: tl.constexpr, BLOCK: tl.constexpr, NW: tl.constexpr, VEC: tl.constexpr):
    layout: tl.constexpr = _lane_layout(W, BLOCK, NW, VEC)
    ptrs = tl.max_contiguous(tl.multiple_of(tlx.require_layout(peers + offs[None, :], layout), [1, 2 * VEC]), [1, VEC])
    vals = tl.load(ptrs, mask=tlx.require_layout(mask[None, :] & (tl.arange(0, W)[:, None] >= 0), layout))
    return tl.sum(vals.to(tl.float32), axis=0).to(tl.bfloat16)


@triton.jit
def _one_shot_kernel(BUF_PTRS, SIG_PTRS, FLAGS, OUT, numel, rank, W: tl.constexpr, BLOCK: tl.constexpr,
                     NW: tl.constexpr):
    pid = tl.program_id(0)
    count = tl.load(FLAGS + pid)
    peers = _peer_bases(BUF_PTRS, W)
    peer_sig, own = _signals(SIG_PTRS, rank, W)
    _barrier(peer_sig, own, rank, 0, count + 1, W, False)
    for start in range(pid * BLOCK, numel, tl.num_programs(0) * BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < numel
        tl.store(OUT + offs, _reduce(peers, offs, mask, W, BLOCK, NW, 8), mask=mask)
    _barrier(peer_sig, own, rank, 1, count + 2, W, False)
    tl.store(FLAGS + pid, count + 2)


@triton.jit
def _two_shot_kernel(BUF_PTRS, TMP_PTRS, SIG_PTRS, FLAGS, OUT, numel, rank, W: tl.constexpr, BLOCK: tl.constexpr,
                     NW: tl.constexpr, VEC: tl.constexpr):
    pid = tl.program_id(0)
    step = tl.num_programs(0) * BLOCK
    count = tl.load(FLAGS + pid)
    part = numel // W
    peers = _peer_bases(BUF_PTRS, W)
    tmps = _peer_bases(TMP_PTRS, W)
    own_tmp = tl.multiple_of(tl.load(TMP_PTRS + rank).to(tl.pointer_type(tl.bfloat16)), 16)
    peer_sig, own = _signals(SIG_PTRS, rank, W)
    _barrier(peer_sig, own, rank, 0, count + 1, W, False)
    base = rank * part
    for start in range(pid * BLOCK, part, step):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < part
        tl.store(own_tmp + base + offs, _reduce(peers, base + offs, mask, W, BLOCK, NW, VEC), mask=mask)
    _barrier(peer_sig, own, rank, 1, count + 2, W, True)
    # Program p copies the chunks that program p of every owner reduced, so its barrier covers them.
    layout: tl.constexpr = _lane_layout(W, BLOCK, NW, VEC)
    first = tl.arange(0, W)[:, None] * part
    offs = tlx.require_layout(first + tl.arange(0, BLOCK)[None, :], layout)
    for start in range(pid * BLOCK, part, step):
        mask = start + offs - first < part
        src = tl.max_contiguous(tl.multiple_of(tmps + offs + start, [1, 2 * VEC]), [1, VEC])
        dst = tl.max_contiguous(tl.multiple_of(OUT + offs + start, [1, 2 * VEC]), [1, VEC])
        tl.store(dst, tl.load(src, mask=mask), mask=mask)
    tl.store(FLAGS + pid, count + 2)


def _hip():
    with open("/proc/self/maps") as maps:
        path = next((line.split()[-1] for line in maps if "libamdhip64.so" in line), None)
    if path is None:
        raise RuntimeError("tlx.ops all_reduce needs the HIP runtime loaded")
    return ctypes.CDLL(path)


class _IpcHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_byte * 64)]


def _check(status, what):
    if status != 0:
        raise RuntimeError(f"tlx.ops all_reduce: {what} failed with HIP error {status}")


class Workspace:
    """Peer-mapped buffers for one process group: a symmetric BF16 input of ``max_numel`` elements that callers
    write the local partial into, plus uncached scratch and signal buffers.

    Calls must not overlap: issue them from one stream, or order the streams (graph capture on a side stream is
    fine). Two calls in flight at once race on the input buffer, the flags and the signal slots, so give each
    concurrent stream its own workspace.

    Setup is collective. Each rank's outcome is exchanged after the local allocations and again after the peer
    mappings, so a failure on one rank raises on every rank instead of blocking the others in the next collective.
    A failure inside the symmetric-memory rendezvous itself is not covered.
    """

    def __init__(self, max_numel: int, group=None):
        import torch.distributed as dist
        import torch.distributed._symmetric_memory as symm_mem

        self.group = group or dist.group.WORLD
        self.world = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
        self.max_numel = max_numel
        self.device = torch.device("cuda", torch.cuda.current_device())
        self._hip = _hip()
        self._owned, self._opened = [], []
        self._closed = False
        local = error = None
        try:
            self._input = symm_mem.empty(max_numel, dtype=torch.bfloat16, device=self.device)
            tmp = self._uncached_alloc(max_numel * 2)
            sig = self._uncached_alloc(_BARRIERS_PER_CALL.value * MAX_PROGRAMS * self.world * 4)
            self.row_sync_ptr = self._uncached_local(ROW_SYNC_BYTES)
            self.flags = torch.zeros(MAX_PROGRAMS, device=self.device, dtype=torch.int32)
            _check(self._hip.hipDeviceSynchronize(), "hipDeviceSynchronize")
            local = (self._ipc_handle(tmp), self._ipc_handle(sig))
        except Exception as exc:
            error = exc
        handles = self._agree(local, error)
        try:
            self._handle = symm_mem.rendezvous(self._input, group=self.group)
            self.buf_ptrs = torch.tensor(self._handle.buffer_ptrs, device=self.device, dtype=torch.int64)
            self.tmp_ptrs = self._open_peers(tmp, [h[0] for h in handles])
            self.sig_ptrs = self._open_peers(sig, [h[1] for h in handles])
            torch.cuda.synchronize(self.device)
        except Exception as exc:
            error = exc
        self._agree(True, error)

    def _agree(self, value, error):
        """Every rank's ``value``, or an error on every rank when any rank failed."""
        import torch.distributed as dist

        gathered = [None] * self.world
        dist.all_gather_object(gathered, None if error is not None else value, group=self.group)
        failed = [rank for rank, item in enumerate(gathered) if item is None]
        if failed:
            self._free()
            if error is not None:
                raise error
            raise RuntimeError(f"tlx.ops all_reduce_workspace: setup failed on rank(s) {failed}")
        return gathered

    def _uncached_alloc(self, nbytes):
        hip = self._hip
        ptr = ctypes.c_void_p()
        _check(
            hip.hipExtMallocWithFlags(ctypes.byref(ptr), ctypes.c_size_t(nbytes),
                                      ctypes.c_uint(_HIP_DEVICE_MALLOC_UNCACHED)), "hipExtMallocWithFlags")
        self._owned.append(ptr.value)
        _check(hip.hipMemset(ptr, 0, ctypes.c_size_t(nbytes)), "hipMemset")
        return ptr.value

    def _uncached_local(self, nbytes):
        return torch.tensor([self._uncached_alloc(nbytes)], device=self.device, dtype=torch.int64)

    def _ipc_handle(self, ptr):
        handle = _IpcHandle()
        _check(self._hip.hipIpcGetMemHandle(ctypes.byref(handle), ctypes.c_void_p(ptr)), "hipIpcGetMemHandle")
        return bytes(handle)

    def _open_peers(self, ptr, handles):
        ptrs = []
        for peer, raw in enumerate(handles):
            if peer == self.rank:
                ptrs.append(ptr)
                continue
            mapped = ctypes.c_void_p()
            _check(
                self._hip.hipIpcOpenMemHandle(ctypes.byref(mapped), _IpcHandle.from_buffer_copy(raw),
                                              ctypes.c_uint(_HIP_IPC_MEM_LAZY_ENABLE_PEER_ACCESS)),
                "hipIpcOpenMemHandle")
            self._opened.append(mapped.value)
            ptrs.append(mapped.value)
        return torch.tensor(ptrs, device=self.device, dtype=torch.int64)

    @property
    def closed(self) -> bool:
        return self._closed

    def input(self, *shape):
        if self._closed:
            raise ValueError("workspace is closed")
        numel = 1
        for dim in shape:
            numel *= dim
        if numel > self.max_numel:
            raise ValueError(f"workspace holds {self.max_numel} elements, asked for {numel}")
        return self._input[:numel].view(*shape)

    def holds(self, x) -> bool:
        return not self._closed and x.data_ptr() == self._input.data_ptr() and x.numel() <= self.max_numel

    def close(self):
        """Release the buffers. Collective: a rank frees its buffers only after every rank's calls finished, since a
        peer may still read them."""
        import torch.distributed as dist

        torch.cuda.synchronize(self.device)
        dist.barrier(self.group)
        self._free()

    def _free(self):
        torch.cuda.synchronize(self.device)
        for ptr in self._opened:
            self._hip.hipIpcCloseMemHandle(ctypes.c_void_p(ptr))
        for ptr in self._owned:
            self._hip.hipFree(ctypes.c_void_p(ptr))
        self._opened, self._owned = [], []
        self._closed = True


def all_reduce(x, workspace, out):
    numel = x.numel()
    if out is None:
        out = torch.empty_like(x)
    if numel == 0:
        return out
    W = workspace.world
    if pick_all_reduce(numel, W) == "one_shot":
        programs = min(PROGRAMS, triton.cdiv(numel, CHUNK))
        _one_shot_kernel[(programs, )](workspace.buf_ptrs, workspace.sig_ptrs, workspace.flags, out, numel,
                                       workspace.rank, W=W, BLOCK=CHUNK, NW=1, num_warps=1, num_stages=1)
    else:
        chunk, programs = two_shot_config(numel, W)
        _two_shot_kernel[(programs, )](workspace.buf_ptrs, workspace.tmp_ptrs, workspace.sig_ptrs, workspace.flags, out,
                                       numel, workspace.rank, W=W, BLOCK=chunk, NW=1, VEC=chunk // 64, num_warps=1,
                                       num_stages=1)
    return out
