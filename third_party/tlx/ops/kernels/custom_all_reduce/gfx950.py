"""Single-node custom all-reduce over xGMI peer memory for gfx950.

A Triton port of the one-stage and two-stage kernels in ROCm/aiter's
``csrc/include/custom_all_reduce.cuh`` (Apache-2.0, derived from vLLM's custom
all-reduce). The kernels keep aiter's names so they line up in traces:

* ``cross_device_reduce_1stage`` reads all ``WORLD`` inputs and sums them in
  fp32 in rank order 0..WORLD-1, so every rank produces bitwise-identical
  output. aiter uses it below 80 KB at 8 ranks.
* ``cross_device_reduce_2stage`` reduce-scatters (rank ``r`` sums slice ``r``
  into its scratch area), synchronizes, then all-gathers every slice.

Each rank exposes one symmetric (IPC-mapped) buffer holding three regions: an
input staging area, the two-stage scratch area and an int32 signal pad.
Barriers are per program: program ``b`` of rank ``r`` writes a monotonically
increasing flag into slot ``[b][r]`` of every peer's pad, then waits until its
own slots ``[b][*]`` reach it. The flag counter lives in device memory, so a
call is graph-capturable. In the two-stage kernel program ``b`` gathers exactly
the offsets program ``b`` reduced on each peer, so a per-program barrier is
enough; aiter relies on the same invariant.

Peer flags are polled with system-scope atomics. Plain or volatile loads can
spin forever on a stale cached copy: peer writes over xGMI to coarse-grained
memory do not invalidate this GPU's cache. Every spin is bounded by
``max_spin``; a timeout sets the plan's error flag instead of hanging the GPU.
"""

import triton
import triton.language as tl

MAX_BLOCKS = 80  # aiter's kMaxBlocks
# aiter's full-xGMI dispatch: one stage below these sizes, two stages at and above.
_ONE_STAGE_MAX_BYTES = {2: 1 << 62, 4: 160 * 1024, 8: 80 * 1024}
_BLOCK = 1024
_NUM_WARPS = 8


@triton.jit
def _rank_barrier(sig_tab, self_sig, slot_base, rank, flag, err_ptr, max_spin, WORLD: tl.constexpr):
    # Every store this program issued completes before the release announces it.
    tl.debug_barrier()
    # One lane per peer, like aiter's `threadIdx.x < ngpus`: all signals go out at once.
    peers = tl.arange(0, WORLD)
    peer_sig = tl.load(sig_tab + peers).to(tl.pointer_type(tl.int32))
    tl.atomic_xchg(peer_sig + slot_base + rank, tl.full([WORLD], flag, tl.int32), sem="release", scope="sys")
    mine = self_sig + slot_base + peers
    zeros = tl.zeros([WORLD], dtype=tl.int32)
    seen = tl.atomic_add(mine, zeros, sem="acquire", scope="sys")
    spins = 0
    while (tl.min(seen, axis=0) < flag) & (spins < max_spin):
        seen = tl.atomic_add(mine, zeros, sem="acquire", scope="sys")
        spins += 1
    tl.atomic_max(err_ptr, tl.where(tl.min(seen, axis=0) >= flag, 0, 1))
    tl.debug_barrier()


@triton.jit
def cross_device_reduce_1stage(in_tab, sig_tab, self_sig, out_ptr, err_ptr, n, rank, max_spin, WORLD: tl.constexpr,
                               MAXB: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    flag_off = 2 * MAXB * WORLD + pid
    flag = tl.load(self_sig + flag_off) + 1
    dtype = out_ptr.dtype.element_ty

    # Peers' inputs were written by work that finished before their launch of this kernel.
    _rank_barrier(sig_tab, self_sig, pid * WORLD, rank, flag, err_ptr, max_spin, WORLD)
    lanes = tl.arange(0, BLOCK)
    for base in range(pid * BLOCK, n, nprog * BLOCK):
        offs = base + lanes
        mask = offs < n
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for p in tl.static_range(WORLD):  # fixed rank order -> identical bits on every rank
            src = tl.load(in_tab + p).to(tl.pointer_type(dtype))
            acc += tl.load(src + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs, acc.to(dtype), mask=mask)
    # No rank may restage its input (the next call) while a peer is still reading it.
    _rank_barrier(sig_tab, self_sig, MAXB * WORLD + pid * WORLD, rank, flag, err_ptr, max_spin, WORLD)
    tl.store(self_sig + flag_off, flag)


@triton.jit
def cross_device_reduce_2stage(in_tab, tmp_tab, sig_tab, self_sig, self_tmp, out_ptr, err_ptr, n, rank, max_spin,
                               WORLD: tl.constexpr, MAXB: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    flag_off = 2 * MAXB * WORLD + pid
    flag = tl.load(self_sig + flag_off) + 1
    dtype = out_ptr.dtype.element_ty

    part = n // WORLD
    my_start = rank * part
    my_len = tl.where(rank == WORLD - 1, n - (WORLD - 1) * part, part)
    lanes = tl.arange(0, BLOCK)

    _rank_barrier(sig_tab, self_sig, pid * WORLD, rank, flag, err_ptr, max_spin, WORLD)
    # Stage 1: reduce-scatter. This rank owns [my_start, my_start + my_len).
    for base in range(pid * BLOCK, my_len, nprog * BLOCK):
        offs = base + lanes
        mask = offs < my_len
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for p in tl.static_range(WORLD):
            src = tl.load(in_tab + p).to(tl.pointer_type(dtype))
            acc += tl.load(src + my_start + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(self_tmp + offs, acc.to(dtype), mask=mask)
    # Release this program's partial sums and acquire every peer's matching program.
    _rank_barrier(sig_tab, self_sig, MAXB * WORLD + pid * WORLD, rank, flag, err_ptr, max_spin, WORLD)
    # Stage 2: all-gather. Program pid reads the offsets program pid wrote on each peer.
    for p in tl.static_range(WORLD):
        tmp_p = tl.load(tmp_tab + p).to(tl.pointer_type(dtype))
        p_len = n - (WORLD - 1) * part if p == WORLD - 1 else part
        for base in range(pid * BLOCK, p_len, nprog * BLOCK):
            offs = base + lanes
            mask = offs < p_len
            v = tl.load(tmp_p + offs, mask=mask, other=0.0, volatile=True)
            tl.store(out_ptr + p * part + offs, v, mask=mask)
    tl.store(self_sig + flag_off, flag)


class CustomAllReducePlan:
    """Symmetric workspace for one process group; reuse it for every call on that group.

    ``error`` is a device flag set when a barrier times out; ``check()`` raises on it.
    """

    def __init__(self, group, world, rank, max_bytes, max_spin, buf, handle, device):
        import torch

        self.group = group
        self.world = world
        self.rank = rank
        self.max_bytes = max_bytes
        self.max_spin = max_spin
        self.device = device
        self._buf = buf
        self._handle = handle  # keeps the peer mappings alive
        sig_off = 2 * max_bytes
        bases = torch.tensor(list(handle.buffer_ptrs), dtype=torch.int64, device=device)
        self.in_tab = bases
        self.tmp_tab = bases + max_bytes
        self.sig_tab = bases + sig_off
        self.inp = buf[:max_bytes]
        self.tmp = buf[max_bytes:sig_off]
        self.sig = buf[sig_off:].view(torch.int32)
        self.error = torch.zeros(1, dtype=torch.int32, device=device)

    def method_for(self, nbytes):
        return "1stage" if nbytes < _ONE_STAGE_MAX_BYTES[self.world] else "2stage"

    def check(self):
        if int(self.error.item()):
            raise RuntimeError("tlx.ops.custom_all_reduce: a peer never reached the barrier (spin limit hit)")


def prepare_custom_all_reduce(group, max_bytes, max_spin):
    import torch
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem

    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    device = torch.device("cuda", torch.cuda.current_device())
    sig_ints = 2 * MAX_BLOCKS * world + MAX_BLOCKS
    buf = symm_mem.empty(2 * max_bytes + 4 * sig_ints, dtype=torch.uint8, device=device)
    buf.zero_()
    torch.cuda.synchronize(device)
    dist.barrier(group=group)
    handle = symm_mem.rendezvous(buf, group=group)
    plan = CustomAllReducePlan(group, world, rank, max_bytes, max_spin, buf, handle, device)
    dist.barrier(group=group)
    return plan


def custom_all_reduce(x, plan, out, method):
    if not isinstance(plan, CustomAllReducePlan):
        raise TypeError("plan must come from tlx.ops.prepare_custom_all_reduce")
    if x.device != plan.device:
        raise ValueError(f"x is on {x.device} but the plan was prepared on {plan.device}")
    n = x.numel()
    nbytes = n * x.element_size()
    if nbytes > plan.max_bytes:
        raise ValueError(f"{nbytes} bytes exceed the plan's max_bytes={plan.max_bytes}")
    method = method or plan.method_for(nbytes)
    if method not in ("1stage", "2stage"):
        raise ValueError(f"method must be '1stage' or '2stage'; got {method!r}")
    import torch

    out = torch.empty_like(x) if out is None else out
    if n == 0:
        return out
    plan.inp[:nbytes].view(x.dtype).copy_(x.reshape(-1))
    if method == "1stage":
        grid = (min(MAX_BLOCKS, triton.cdiv(n, _BLOCK)), )
        cross_device_reduce_1stage[grid](plan.in_tab, plan.sig_tab, plan.sig, out, plan.error, n, plan.rank,
                                         plan.max_spin, WORLD=plan.world, MAXB=MAX_BLOCKS, BLOCK=_BLOCK,
                                         num_warps=_NUM_WARPS)
    else:
        tmp = plan.tmp[:nbytes].view(x.dtype)
        grid = (min(MAX_BLOCKS, triton.cdiv(triton.cdiv(n, plan.world), _BLOCK)), )
        cross_device_reduce_2stage[grid](plan.in_tab, plan.tmp_tab, plan.sig_tab, plan.sig, tmp, out, plan.error, n,
                                         plan.rank, plan.max_spin, WORLD=plan.world, MAXB=MAX_BLOCKS, BLOCK=_BLOCK,
                                         num_warps=_NUM_WARPS)
    return out
