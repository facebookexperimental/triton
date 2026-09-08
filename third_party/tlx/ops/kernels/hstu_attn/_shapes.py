from __future__ import annotations

# Entries are [Z, MAX_SEQ_LEN, H, HEAD_DIM, causal, dtype]. The op is
# causal-only, so `causal` is True throughout and is carried for the label
# rather than as a knob.

#: Identical to `test_hstu_attn.py::SM100_SHAPES`.
SYNTHETIC: list[list] = [
    [1, 256, 4, 128, True, "bf16"],
    [2, 512, 4, 128, True, "bf16"],
    [2, 512, 8, 64, True, "bf16"],
    [1, 1024, 4, 64, True, "bf16"],
    [4, 256, 4, 128, True, "bf16"],
    [2, 1024, 4, 128, True, "bf16"],
    [8, 128, 4, 128, True, "bf16"],
    [2, 256, 16, 64, True, "bf16"],
    [1, 2048, 2, 128, True, "bf16"],
]

#: TODO: placeholder shapes, not a capture. The production shape is
#: heads=4 seq=4096 sparsity=0.95 batch=512 (`tutorials/hstu_self_attn/bench_self.py:41`),
#: which needs the sparse length draw below before it can be used here.
SM100_FOCUS: list[list] = [
    [32, 1024, 4, 128, True, "bf16"],
    [64, 512, 4, 128, True, "bf16"],
    [16, 2048, 4, 128, True, "bf16"],
    [32, 1024, 8, 128, True, "bf16"],
    [32, 1024, 4, 128, True, "fp16"],
]

#: Empty: the AMD entry is forward-only and the reference is unvalidated there.
GFX950_FOCUS: list[list] = []

#: How sequence lengths are drawn. The distribution dominates ragged-attention
#: performance, so a number taken under one is not comparable to one taken under
#: another -- hence it is in the label.
#: TODO: only "uniform" exists; production is sparsity 0.95, see
#: `generate_sparse_seq_len` in `tutorials/hstu_self_attn/bench_self.py`.
RAGGED = "uniform"


def inputs(Z, max_seq_len, H, head_dim, dtype, requires_grad=False, device="cuda"):
    """A uniform-length ragged batch: every sequence is exactly max_seq_len."""
    import torch

    offsets = torch.arange(0, (Z + 1) * max_seq_len, max_seq_len, device=device, dtype=torch.int64)
    total = int(offsets[-1])
    q, k, v = (torch.randn(total, H, head_dim, device=device, dtype=dtype).requires_grad_(requires_grad)
               for _ in range(3))
    attn_scale = torch.tensor(1.0 / max_seq_len, device=device, dtype=torch.float32)
    return q, k, v, offsets, attn_scale


def flops(Z, MAX_SEQ_LEN, H, HEAD_DIM, causal, direction="fwd", tokens=None):
    """Same convention as `flash_attn`, so the two ops' TFLOP/s read alike.

    `tokens` is the OBSERVED total. Under the uniform draw it equals
    `Z * MAX_SEQ_LEN`; under any other draw the nominal product overstates the
    work by the padding factor, so the mean actual length is used instead.
    """
    seq = (tokens / Z) if tokens else MAX_SEQ_LEN
    total = 2 * (2.0 * Z * H * seq * seq * HEAD_DIM)
    if causal:
        total *= 0.5
    if direction == "bwd":
        total *= 2.5
    return int(total)


def label(Z, MAX_SEQ_LEN, H, HEAD_DIM, causal, dtype, direction="fwd") -> str:
    """The report's input column."""
    return (f"((), {{'dtype': '{dtype}', 'ragged': '{RAGGED}', 'dir': '{direction}', "
            f"'Z': '{Z}', 'MAX_SEQ_LEN': '{MAX_SEQ_LEN}', 'H': '{H}', 'HEAD_DIM': '{HEAD_DIM}'}})")
