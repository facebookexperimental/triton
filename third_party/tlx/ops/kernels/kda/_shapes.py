from __future__ import annotations

# Entries are [B, T, H, HEAD_DIM, dtype], where B is the number of packed
# sequences and T the tokens in each; the op takes them as one `[1, B*T, H, D]`
# tensor plus `cu_seqlens`. HEAD_DIM is 128 throughout -- the catalog entry
# accepts nothing else.

#: The L1 shape, from `test_kimi_delta_attention.py::_inputs`.
SYNTHETIC: list[list] = [
    [2, 64, 2, 128, "bf16"],
]

#: TODO: placeholder shapes, not a capture. Awaiting real ones.
SM100_FOCUS: list[list] = [
    [4, 4096, 8, 128, "bf16"],
    [4, 4096, 16, 128, "bf16"],
    [2, 8192, 8, 128, "bf16"],
    [8, 2048, 8, 128, "bf16"],
]

#: Prepared-input gfx950 prefill shapes: total tokens, sequences, H, K, V, dtype.
GFX950_PREFILL_SYNTHETIC: list[list] = [
    [64, 1, 4, 128, 128, "bf16"],
]

GFX950_PREFILL_FOCUS: list[list] = [
    [4096, 1, 4, 128, 128, "bf16"],
    [4096, 4, 4, 128, 128, "bf16"],
    [131072, 1, 4, 128, 128, "bf16"],
    [131072, 8, 4, 128, 128, "bf16"],
    [4096, 1, 12, 128, 128, "bf16"],
    [4096, 4, 12, 128, 128, "bf16"],
    [131072, 1, 12, 128, 128, "bf16"],
    [131072, 8, 12, 128, 128, "bf16"],
]

#: Indexed one-token gfx950 decode shapes: batch, H, K, V, dtype.
GFX950_DECODE_SYNTHETIC: list[list] = [
    [1, 4, 128, 128, "bf16"],
]

GFX950_DECODE_FOCUS: list[list] = [
    [batch, heads, 128, 128, "bf16"]
    for heads in (4, 12)
    for batch in (1, 2, 4, 8, 16, 32)
]

#: Chunk length the kernel works in. Used only by `flops`.
CHUNK = 64

#: Absolute TFLOP/s gate -- the only perf gate available, since KDA has no
#: runnable reference. None = report only, until a clean run exists to seed it.
FLOOR_TFLOPS = None


def inputs(B, T, H, head_dim, dtype, requires_grad=False, device="cuda"):
    """Packed `[1, B*T, H, D]` inputs plus `cu_seqlens`, as the op wants them.

    Mirrors `test_kimi_delta_attention.py::_inputs`. The normalization and the
    negative softplus are load-bearing: the delta rule diverges on plain random
    inputs, which would time arithmetic no real caller produces.
    """
    import torch
    import torch.nn.functional as F

    gen = torch.Generator(device=device).manual_seed(0)

    def rn(*shape):
        return torch.randn(*shape, generator=gen, device=device, dtype=torch.float32)

    total = B * T
    q = F.normalize(rn(1, total, H, head_dim), dim=-1).to(dtype).requires_grad_(requires_grad)
    k = F.normalize(rn(1, total, H, head_dim), dim=-1).to(dtype).requires_grad_(requires_grad)
    v = rn(1, total, H, head_dim).to(dtype).requires_grad_(requires_grad)
    g = (-F.softplus(rn(1, total, H, head_dim))).requires_grad_(requires_grad)
    beta = torch.sigmoid(rn(1, total, H)).requires_grad_(requires_grad)
    cu_seqlens = torch.arange(0, (B + 1) * T, T, device=device, dtype=torch.int64)
    return q, k, v, g, beta, cu_seqlens


def flops(B, T, H, HEAD_DIM, direction="fwd", chunk=CHUNK):
    """Approximate: NOT comparable to mm or attention, only to other KDA runs.

    Per chunk per (sequence, head), the matmuls in `sm100.py`'s docstring are
    four of `2*C*C*D` and three of `2*C*D*D`; the triangular inverse is not
    counted and the backward is taken as 2x for its two passes. The uncounted
    work is a real fraction of the runtime, so `mtokens_per_s` in
    `Result.extra` is the honest rate.
    """
    per_seq_head = 8 * T * chunk * HEAD_DIM + 6 * T * HEAD_DIM * HEAD_DIM
    total = B * H * per_seq_head
    if direction == "bwd":
        total *= 2.0
    return int(total)


def label(B, T, H, HEAD_DIM, dtype, direction="fwd") -> str:
    """The report's input column."""
    return (f"((), {{'dtype': '{dtype}', 'dir': '{direction}', "
            f"'B': '{B}', 'T': '{T}', 'H': '{H}', 'HEAD_DIM': '{HEAD_DIM}'}})")
