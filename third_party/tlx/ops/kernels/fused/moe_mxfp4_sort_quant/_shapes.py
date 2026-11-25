from __future__ import annotations

from typing import NamedTuple

from ..._shape_suites import FocusRegistry, FocusSuite


class MoeMxfp4SortQuantShape(NamedTuple):
    """``tokens`` bf16 rows of width ``hidden`` routed to ``topk`` of ``experts``; this rank holds ``local_experts``
    experts from ``local_start``."""
    tokens: int
    hidden: int
    experts: int
    topk: int
    local_experts: int
    local_start: int


SYNTHETIC: tuple[MoeMxfp4SortQuantShape, ...] = (
    MoeMxfp4SortQuantShape(3, 1024, 64, 4, 16, 16),
    MoeMxfp4SortQuantShape(17, 2048, 128, 8, 64, 0),
    MoeMxfp4SortQuantShape(64, 4096, 256, 8, 32, 224),
)

# Decode batches of 1 to 64 requests: 256 experts of width 4096, top 8, on the third of eight expert-parallel ranks.
GFX950_1 = FocusSuite(name="gfx950_1", op="moe_mxfp4_sort_quant",
                      shapes=tuple(MoeMxfp4SortQuantShape(t, 4096, 256, 8, 32, 64) for t in (1, 2, 4, 8, 16, 32, 64)))
GFX950_ALL = FocusSuite(name="gfx950_all", op="moe_mxfp4_sort_quant", includes=("gfx950_1", ))

FOCUS_SUITES = (GFX950_1, GFX950_ALL)
DEFAULT_SUITES = {"gfx950": ("gfx950_all", )}
FOCUS = FocusRegistry("moe_mxfp4_sort_quant", FOCUS_SUITES, DEFAULT_SUITES)
CORRECTNESS_SHAPES = tuple(dict.fromkeys((*SYNTHETIC, *FOCUS.all_shapes())))
BLOCK_M = 32


def label(shape: MoeMxfp4SortQuantShape) -> str:
    return (f"(({shape.tokens}, {shape.hidden}), {{'experts': '{shape.experts}', 'topk': '{shape.topk}', "
            f"'local': '{shape.local_start}+{shape.local_experts}'}})")


def inputs(shape: MoeMxfp4SortQuantShape, seed: int = 0, device="cuda") -> dict:
    """bf16 activations with an all-zero group, distinct random expert ids per token with one crowded local expert
    (so it spans several blocks), and the keyword arguments of ``tlx.ops.moe_mxfp4_sort_quant``."""
    import torch

    gen = torch.Generator().manual_seed(seed)
    x = (torch.randn(shape.tokens, shape.hidden, generator=gen) * 3).to(torch.bfloat16)
    x[0, :32] = 0
    ids = torch.stack([torch.randperm(shape.experts, generator=gen)[:shape.topk]
                       for _ in range(shape.tokens)]).to(torch.int32)
    crowd = shape.local_start + min(3, shape.local_experts - 1)
    for t in range(shape.tokens):
        if not (ids[t] == crowd).any():
            ids[t, 0] = crowd
    return dict(x=x.to(device), topk_ids=ids.to(device), num_local_experts=shape.local_experts,
                local_expert_start=shape.local_start, num_experts=shape.experts, block_m=BLOCK_M)


def quant_reference(x):
    """AITER ``dynamic_mxfp4_quant`` on the CPU: round-up amax, scale ``2^(floor(log2(amax)) - 2)``, RNE E2M1.
    Returns (packed uint8 (T, K / 2), E8M0 uint8 (T, K / 32))."""
    import torch

    x = x.float().cpu().view(x.shape[0], -1, 32)
    amax = x.abs().amax(-1, keepdim=True)
    amax = ((amax.view(torch.int32) + 0x200000) & 0x7F800000).view(torch.float32)
    unb = torch.clamp(torch.floor(torch.log2(amax)) - 2, -127, 127)
    q = (x * torch.exp2(-unb)).view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    sgn = q & 0x80000000
    q = q ^ sgn
    qf = q.to(torch.int32).view(torch.float32)
    sat, den = qf >= 6.0, (qf < 6.0) & (qf < 1.0)
    den_int = ((127 - 1) + (23 - 1) + 1) << 23
    dx = (((qf + torch.tensor(den_int, dtype=torch.int32).view(torch.float32)).view(torch.int32).to(torch.int64)
           & 0xFFFFFFFF) - den_int) & 0xFF
    nx = (((q + 3240099839 + ((q >> 22) & 1)) & 0xFFFFFFFF) >> 22) & 0xFF
    v = torch.full_like(q, 0x7)
    v = torch.where(~sat & ~den, nx, v)
    v = torch.where(den, dx, v)
    v = (v | (sgn >> 28)).to(torch.uint8).view(x.shape[0], -1, 2)
    return v[..., 0] | (v[..., 1] << 4), (unb.to(torch.int32) + 127).to(torch.uint8).view(x.shape[0], -1)


def sort_reference(topk_ids, local_start, num_local, block_m=BLOCK_M):
    """AITER ``moe_sorting`` with the local-expert mask: (sorted ids up to the padded count, local expert per block)."""
    ids = topk_ids.cpu()
    T, topk = ids.shape
    rows, eids = [], []
    for e in range(num_local):
        slots = [(t, k) for t in range(T) for k in range(topk) if ids[t, k] == local_start + e]
        if not slots:
            continue
        span = -(-len(slots) // block_m) * block_m
        rows += [t | (k << 24) for t, k in slots] + [T | (topk << 24)] * (span - len(slots))
        eids += [e] * (span // block_m)
    return rows, eids


def scale_offset(r, c, groups):
    """Byte offset of the scale of sorted row ``r``, group ``c``, in ``moe_mxfp4_sort``'s [M/32, N/8, 4, 16] uint32
    tiles."""
    word = (r // 32) * (groups // 8 * 64) + (c // 8) * 64 + (c % 4) * 16 + (r % 16)
    return word * 4 + (r // 16) % 2 + 2 * ((c // 4) % 2)


def reference(x, topk_ids, num_local_experts, local_expert_start, num_experts, block_m=BLOCK_M):
    """The op's outputs from the three separate steps, on the CPU; unwritten bytes of the op's outputs are zero here.
    Returns the same tuple as the op."""
    import torch

    T, K = x.shape
    topk = topk_ids.shape[1]
    q, s = quant_reference(x)
    rows, eids = sort_reference(topk_ids, local_expert_start, num_local_experts, block_m)
    cap = T * topk + num_experts * block_m - topk
    sorted_ids = torch.zeros(cap, dtype=torch.int32)
    sorted_ids[:len(rows)] = torch.tensor(rows, dtype=torch.int32)
    sorted_eids = torch.zeros(-(-cap // block_m), dtype=torch.int32)
    sorted_eids[:len(eids)] = torch.tensor(eids, dtype=torch.int32)
    groups = K // 32
    scales = torch.zeros(-(-cap // 32) * 32 * groups, dtype=torch.uint8)
    for r, v in enumerate(rows):
        if (v & 0xFFFFFF) < T:
            off = torch.tensor([scale_offset(r, c, groups) for c in range(groups)])
            scales[off] = s[v & 0xFFFFFF]
    return q, sorted_ids, sorted_eids, torch.tensor([len(rows), T], dtype=torch.int32), scales.view(-1, groups)


def reference_torch(x, topk_ids, num_local_experts, local_expert_start, num_experts, block_m=BLOCK_M):
    """The op's contract in vectorized PyTorch on the device (no host reads, so ``torch.compile`` captures it whole):
    the same bytes as the op wherever the op writes; positions the op leaves unwritten are zero."""
    import struct

    import torch

    dev = x.device
    T, K = x.shape
    topk = topk_ids.shape[1]
    S = T * topk
    groups = K // 32
    # dynamic_mxfp4_quant
    xf = x.float().view(T, groups, 32)
    amax = xf.abs().amax(-1, keepdim=True)
    amax = ((amax.view(torch.int32) + 0x200000) & 0x7F800000).view(torch.float32)
    unb = torch.clamp(torch.floor(torch.log2(amax)) - 2, -127, 127)
    q = (xf * torch.exp2(-unb)).view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    sgn = q & 0x80000000
    q = q ^ sgn
    qf = q.to(torch.int32).view(torch.float32)
    sat, den = qf >= 6.0, (qf < 6.0) & (qf < 1.0)
    den_int = ((127 - 1) + (23 - 1) + 1) << 23
    den_f = struct.unpack("f", struct.pack("i", den_int))[0]  # a Python float: nothing is copied during capture
    dx = (((qf + den_f).view(torch.int32).to(torch.int64) & 0xFFFFFFFF) - den_int) & 0xFF
    nx = (((q + 3240099839 + ((q >> 22) & 1)) & 0xFFFFFFFF) >> 22) & 0xFF
    v = torch.where(~sat & ~den, nx, torch.full_like(q, 0x7))
    v = (torch.where(den, dx, v) | (sgn >> 28)).to(torch.uint8).view(T, -1, 2)
    x_fp4 = v[..., 0] | (v[..., 1] << 4)
    scale = (unb.to(torch.int32) + 127).to(torch.uint8).view(T, groups)
    # moe_sorting with the local-expert mask
    loc = topk_ids.view(-1).long() - local_expert_start
    local = (loc >= 0) & (loc < num_local_experts)
    e = torch.where(local, loc, num_local_experts)
    order = torch.argsort(e * S + torch.arange(S, device=dev))
    es = e[order]
    cnt = (e[:, None] == torch.arange(num_local_experts, device=dev)).sum(0)  # bincount would read its max on the host
    span = (cnt + block_m - 1) // block_m * block_m
    base = torch.cumsum(span, 0) - span
    first = torch.cumsum(cnt, 0) - cnt
    ok = es < num_local_experts
    esc = es.clamp(max=num_local_experts - 1)
    pos = torch.where(ok, base[esc] + torch.arange(S, device=dev) - first[esc], 0)
    total = span.sum()
    cap = S + num_experts * block_m - topk
    p = torch.arange(cap, device=dev)
    sorted_ids = torch.where(p < total, T | (topk << 24), 0).to(torch.int32)
    val = ((order // topk) | ((order % topk) << 24)).to(torch.int32)
    sorted_ids = sorted_ids.scatter(0, torch.where(ok, pos, cap - 1), torch.where(ok, val, sorted_ids[cap - 1]))
    nblk = -(-cap // block_m)
    b = torch.arange(nblk, device=dev) * block_m
    sorted_eids = torch.where(b < total, torch.searchsorted(torch.cumsum(span, 0), b, right=True), 0).to(torch.int32)
    num_valid = torch.stack([total, torch.full_like(total, T)]).to(torch.int32)
    # moe_mxfp4_sort
    rows = -(-cap // 32) * 32
    c = torch.arange(groups, device=dev)
    r = pos[:, None]
    word = (r // 32) * (groups // 8 * 64) + (c // 8) * 64 + (c % 4) * 16 + (r % 16)
    off = word * 4 + (r // 16) % 2 + 2 * ((c // 4) % 2)
    off = torch.where(ok[:, None], off, rows * groups)  # dropped slots write past the end, then cut
    flat = torch.zeros(rows * groups + 1, dtype=torch.uint8, device=dev)
    flat = flat.scatter(0, off.reshape(-1), scale[order // topk].reshape(-1))
    return x_fp4, sorted_ids, sorted_eids, num_valid, flat[:-1].view(rows, groups)
