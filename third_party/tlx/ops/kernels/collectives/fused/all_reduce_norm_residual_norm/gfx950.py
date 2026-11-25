"""Intra-node BF16 all-reduce fused with the residual-stream boundary of ``norm_residual_norm``, on gfx950.

The one-shot reduction of a row (the all-reduce's FP32 sum of the ranks' values in rank order, rounded to BF16) feeds
``norm_residual_norm``'s row math in the same launch, so the fused kernels and the all-reduce followed by the
standalone boundary kernel produce the same bits. Two kernels:

- Row: one 8-wave program per row reduces the whole row and runs the row math. It pays off only at H 4096.
- Cooperative: a row is split over ``COOP_PARTS`` one-wave programs. Program ``w`` reduces exactly the elements that
  wave ``w`` of the standalone 8-wave boundary kernel holds (8 contiguous elements per lane, two 4096-element
  repetitions at H 8192), so the remote reads keep the all-reduce's one-wave parallelism. The two row sums of the
  boundary (post-norm and pre-norm) are exchanged through per-row words in uncached memory (like the all-reduce's
  signals; with cached memory, each program would need GPU-scope release/acquire fences that write back and
  invalidate its XCD's L2): each program publishes its wave's partial sum with the call's tag in one 64-bit store,
  polls the row's other words and adds the ``COOP_PARTS`` partials the way the standalone kernel adds its waves'
  sums, so every program and the standalone kernel get the same sum. All programs of a call are resident at once (at
  most ``ROW_SYNC_ROWS * COOP_PARTS`` one-wave programs), so the row wait cannot deadlock. The two dependent exchanges
  cost about as much as the standalone boundary launch they replace, so the kernel only wins at a few rows.

Outside both row ranges the op runs the all-reduce followed by the standalone boundary kernel.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

from ...all_reduce import gfx950 as comm
from ....fused.norm_residual_norm import gfx950 as boundary

FUSED_ROWS = {4096: (1, 24)}
# Measured crossover against the all-reduce followed by the standalone boundary (8 MI350X, HIP graph): the
# cooperative kernel is 3-9% faster up to 8 rows, even at 9-10 and slower from 12; the row kernel is slower at H 8192.
COOP_ROWS = {8192: (1, 8)}
COOP_PARTS = 8  # the standalone boundary kernel's waves
COOP_SPT = 8  # elements per lane per repetition in the standalone boundary kernel's layout


def pick_kernel(rows: int, H: int):
    """"row", "coop", or None for the all-reduce followed by the standalone boundary kernel."""
    for kernel, ranges in (("row", FUSED_ROWS), ("coop", COOP_ROWS)):
        low, high = ranges.get(H, (1, 0))
        if low <= rows <= high:
            return kernel
    return None


def pick_fused(rows: int, H: int) -> bool:
    return pick_kernel(rows, H) is not None


@triton.jit
def _reduce_row(peers, base, W: tl.constexpr, H: tl.constexpr):
    layout: tl.constexpr = comm._lane_layout(W, H, 8, 8)
    offs = tlx.require_layout(tl.arange(0, W)[:, None] * 0 + tl.arange(0, H)[None, :], layout)
    vals = tl.load(tl.max_contiguous(tl.multiple_of(peers + base + offs, [1, 16]), [1, 8]))
    return tl.sum(vals.to(tl.float32), axis=0).to(tl.bfloat16)


@triton.jit
def _fused_kernel(BUF_PTRS, SIG_PTRS, FLAGS, STREAM, OUTPUT, NEW_STREAM, POST_WEIGHT, PRE_WEIGHT, ALPHA, BETA, rows,
                  rank, post_eps, pre_eps, W: tl.constexpr, H: tl.constexpr, POST_INPUT_SCALE: tl.constexpr,
                  HAS_POST_WEIGHT: tl.constexpr, HAS_PRE_WEIGHT: tl.constexpr, GATE: tl.constexpr,
                  GATE_PER_CHANNEL: tl.constexpr, PRE_NORM_BF16: tl.constexpr):
    pid = tl.program_id(0)
    count = tl.load(FLAGS + pid)
    peers = comm._peer_bases(BUF_PTRS, W)
    peer_sig, own = comm._signals(SIG_PTRS, rank, W)
    comm._barrier(peer_sig, own, rank, 0, count + 1, W, False)
    offs = tl.arange(0, H)
    for row in range(pid, rows, tl.num_programs(0)):
        base = row.to(tl.int64) * H
        branch = _reduce_row(peers, base, W, H)
        stream = tl.load(STREAM + base + offs)
        if GATE_PER_CHANNEL:
            alpha = tl.load(ALPHA + offs)
            beta = tl.load(BETA + offs)
        else:
            alpha = 0.0
            beta = 0.0
        if HAS_PRE_WEIGHT:
            pre_weight = tl.load(PRE_WEIGHT + offs).to(tl.float32)
        else:
            pre_weight = 0.0
        if HAS_POST_WEIGHT:
            post_weight = tl.load(POST_WEIGHT + offs).to(tl.float32)
        else:
            post_weight = 0.0
        boundary._plain_row(branch, stream, alpha, beta, post_weight, pre_weight, base, OUTPUT, NEW_STREAM, ALPHA, BETA,
                            post_eps, pre_eps, H, POST_INPUT_SCALE, HAS_POST_WEIGHT, HAS_PRE_WEIGHT, GATE,
                            GATE_PER_CHANNEL, PRE_NORM_BF16)
    comm._barrier(peer_sig, own, rank, 1, count + 2, W, False)
    tl.store(FLAGS + pid, count + 2)


@triton.jit
def _row_total(slots, tag, part, value, PARTS: tl.constexpr):
    """Publish this program's partial of a row sum, wait for the row's other programs and return the sum of the
    ``PARTS`` partials. Each program writes one 64-bit word, the call's tag in the low half and the partial's bits in
    the high half, so one store publishes both and the readers poll the words themselves."""
    word = (value.to(tl.int32, bitcast=True).to(tl.int64) << 32) | (tag.to(tl.int64) & 0xFFFFFFFF)
    tl.store(slots + part, word)
    words = tl.load(slots + tl.arange(0, PARTS), volatile=True)
    while tl.min(tl.where(words.to(tl.int32) == tag, 1, 0)) == 0:
        words = tl.load(slots + tl.arange(0, PARTS), volatile=True)
    return tl.sum((words >> 32).to(tl.int32).to(tl.float32, bitcast=True))


@triton.jit
def _coop_kernel(BUF_PTRS, SIG_PTRS, FLAGS, SYNC_PTR, STREAM, OUTPUT, NEW_STREAM, POST_WEIGHT, PRE_WEIGHT, ALPHA, BETA,
                 rank, post_eps, pre_eps, W: tl.constexpr, H: tl.constexpr, PARTS: tl.constexpr, SPT: tl.constexpr,
                 POST_INPUT_SCALE: tl.constexpr, HAS_POST_WEIGHT: tl.constexpr, HAS_PRE_WEIGHT: tl.constexpr,
                 GATE: tl.constexpr, GATE_PER_CHANNEL: tl.constexpr, PRE_NORM_BF16: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // PARTS
    part = pid % PARTS
    WAVE_SPAN: tl.constexpr = 64 * SPT
    BLOCK: tl.constexpr = H // PARTS
    i = tl.arange(0, BLOCK)
    offs = (i // WAVE_SPAN) * (WAVE_SPAN * PARTS) + part * WAVE_SPAN + i % WAVE_SPAN
    base = row.to(tl.int64) * H
    # Local operands first: they do not depend on the peers.
    stream = tl.load(STREAM + base + offs)
    if GATE_PER_CHANNEL:
        alpha = tl.load(ALPHA + offs)
        beta = tl.load(BETA + offs)
    if HAS_POST_WEIGHT:
        post_weight = tl.load(POST_WEIGHT + offs).to(tl.float32)
    if HAS_PRE_WEIGHT:
        pre_weight = tl.load(PRE_WEIGHT + offs).to(tl.float32)
    count = tl.load(FLAGS + pid)
    # A row's PARTS words of a phase always change together, so this program's own word holds the previous call's
    # tag of the row; consecutive calls are separate launches, so no program still polls for that tag.
    slots = tl.load(SYNC_PTR).to(tl.pointer_type(tl.int64)) + row * (2 * PARTS)
    tag0 = (tl.load(slots + part, volatile=True) + 1).to(tl.int32)
    tag1 = (tl.load(slots + PARTS + part, volatile=True) + 1).to(tl.int32)
    peers = comm._peer_bases(BUF_PTRS, W)
    peer_sig, own = comm._signals(SIG_PTRS, rank, W)
    comm._barrier(peer_sig, own, rank, 0, count + 1, W, False)
    branch = comm._reduce(peers, base + offs, i < BLOCK, W, BLOCK, 1, SPT)

    post_input = branch.to(tl.float32)
    if HAS_POST_WEIGHT and POST_INPUT_SCALE:
        post_input = (post_input * post_weight).to(tl.bfloat16).to(tl.float32)
    post_sum = _row_total(slots, tag0, part, tl.sum(post_input * post_input), PARTS)
    normed_branch = post_input * tl.rsqrt(post_sum / H + post_eps)
    if HAS_POST_WEIGHT and not POST_INPUT_SCALE:
        normed_branch = normed_branch * post_weight
    normed_branch = normed_branch.to(tl.bfloat16).to(tl.float32)
    if GATE:
        if not GATE_PER_CHANNEL:
            alpha = tl.load(ALPHA)
            beta = tl.load(BETA)
        new_stream = alpha * stream + beta * normed_branch
    else:
        new_stream = stream + normed_branch
    tl.store(NEW_STREAM + base + offs, new_stream)
    pre_input = new_stream
    if PRE_NORM_BF16:
        pre_input = pre_input.to(tl.bfloat16).to(tl.float32)
    pre_sum = _row_total(slots + PARTS, tag1, part, tl.sum(pre_input * pre_input), PARTS)
    normed = pre_input * tl.rsqrt(pre_sum / H + pre_eps)
    if HAS_PRE_WEIGHT:
        normed = normed * pre_weight
    tl.store(OUTPUT + base + offs, normed.to(tl.bfloat16))
    comm._barrier(peer_sig, own, rank, 1, count + 2, W, False)
    tl.store(FLAGS + pid, count + 2)


def all_reduce_norm_residual_norm(stream, branch, workspace, post_weight, post_norm_input_scale, pre_weight, alpha,
                                  beta, post_eps, pre_eps, pre_norm_bf16, out, new_stream):
    M, H = branch.shape
    if out is None:
        out = torch.empty(M, H, device=branch.device, dtype=torch.bfloat16)
    if new_stream is None:
        new_stream = torch.empty(M, H, device=branch.device, dtype=torch.float32)
    if M == 0:
        return out, new_stream
    kernel = pick_kernel(M, H)
    if kernel is None:
        reduced = comm.all_reduce(branch, workspace, None)
        return boundary.norm_residual_norm(stream, reduced, post_weight, post_norm_input_scale, pre_weight, alpha, beta,
                                           post_eps, pre_eps, pre_norm_bf16, out, new_stream)
    operands = (stream, out, new_stream, post_weight if post_weight is not None else stream,
                pre_weight if pre_weight is not None else stream, alpha if alpha is not None else stream,
                beta if beta is not None else stream)
    flags = dict(W=workspace.world, H=H, POST_INPUT_SCALE=post_norm_input_scale, HAS_POST_WEIGHT=post_weight
                 is not None, HAS_PRE_WEIGHT=pre_weight is not None, GATE=alpha is not None,
                 GATE_PER_CHANNEL=alpha is not None and alpha.numel() == H, PRE_NORM_BF16=pre_norm_bf16, num_stages=1)
    if kernel == "coop":
        _coop_kernel[(M * COOP_PARTS, )](workspace.buf_ptrs, workspace.sig_ptrs, workspace.flags,
                                         workspace.row_sync_ptr, *operands, workspace.rank, float(post_eps),
                                         float(pre_eps), PARTS=COOP_PARTS, SPT=COOP_SPT, num_warps=1, **flags)
        return out, new_stream
    _fused_kernel[(min(comm.PROGRAMS, M), )](workspace.buf_ptrs, workspace.sig_ptrs, workspace.flags, *operands, M,
                                             workspace.rank, float(post_eps), float(pre_eps), num_warps=8, **flags)
    return out, new_stream
