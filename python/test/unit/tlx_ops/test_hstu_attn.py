"""L1 correctness for `tlx.ops.hstu_attn_dev`.

No compile-time cap as in test_mm.py: this kernel has no shape heuristic, so the
correctness caller autotunes a small space and wall-clock says nothing about
compile time.

The op is causal-only, so every shape here is causal and `causal=False` is a
rejection case rather than a numerics case -- see `test_non_causal_rejected`.
"""

import pytest
import torch

from triton._internal_testing import is_blackwell, is_hip_cdna4

torch.manual_seed(0)

SM100_ARCH = "sm100"
GFX950_ARCH = "gfx950"

#  Hardware agnostic testsuite ``[Z, MAX_SEQ_LEN, H, HEAD_DIM, causal]``
SM100_SHAPES = [
    [1, 256, 4, 128, True],
    [2, 512, 4, 128, True],
    [2, 512, 8, 64, True],
    [1, 1024, 4, 64, True],
    # Batch- vs sequence-dominated.
    [4, 256, 4, 128, True],
    [2, 1024, 4, 128, True],
    [8, 128, 4, 128, True],
    [2, 256, 16, 64, True],
    [1, 2048, 2, 128, True],
]

# atol tracks the reference's dynamic range; see test_mm.py.
REL_PRECISION = {torch.float16: 1e-3, torch.bfloat16: 8e-3}

# gfx950 HSTU is the AMD forward implementation promoted from tutorials. Keep
# this small: the full production-size sweeps belong in benchmarks, not L1.
GFX950_SHAPES = [
    (2, 128, 2, 128, 128),
    (4, 256, 4, 128, 128),
]


def _inputs(Z, max_seq_len, H, head_dim, dtype):
    """Uniform-length ragged batch: every sequence is exactly max_seq_len."""
    offsets = torch.arange(0, (Z + 1) * max_seq_len, max_seq_len, device="cuda", dtype=torch.int64)
    total = int(offsets[-1])
    q, k, v = (torch.randn(total, H, head_dim, device="cuda", dtype=dtype) for _ in range(3))
    attn_scale = torch.tensor(1.0 / max_seq_len, device="cuda", dtype=torch.float32)
    return q, k, v, offsets, attn_scale


def _float_ref(q, k, v, offsets, attn_scale, alpha, causal):
    """HSTU is SiLU-scaled, not softmax -- torch has no equivalent to call."""
    qf, kf, vf = q.float(), k.float(), v.float()
    outs = []
    for z in range(offsets.numel() - 1):
        s, e = int(offsets[z]), int(offsets[z + 1])
        qk = torch.einsum("qhd,khd->hqk", qf[s:e], kf[s:e]) * alpha
        sig = qk * torch.sigmoid(qk) * attn_scale.item()
        if causal:
            i = torch.arange(e - s, device=qk.device)
            sig = sig * (i[:, None] >= i[None, :]).float()[None]
        outs.append(torch.einsum("hqk,khd->qhd", sig, vf[s:e]))
    return torch.cat(outs, 0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
@pytest.mark.parametrize("Z, MAX_SEQ_LEN, H, HEAD_DIM, causal", SM100_SHAPES)
def test_hstu_attn_sm100(Z, MAX_SEQ_LEN, H, HEAD_DIM, causal, dtype):
    from triton.tlx.ops import hstu_attn_dev as tlx_hstu_attn

    q, k, v, offsets, attn_scale = _inputs(Z, MAX_SEQ_LEN, H, HEAD_DIM, dtype)
    alpha = 1.0 / HEAD_DIM

    out = tlx_hstu_attn(q, k, v, offsets, MAX_SEQ_LEN, attn_scale, alpha=alpha, causal=causal, arch=SM100_ARCH,
                        space="smoke")

    ref = _float_ref(q, k, v, offsets, attn_scale, alpha, causal).to(out.dtype)
    precision = REL_PRECISION[dtype]
    torch.testing.assert_close(out, ref, atol=precision * ref.abs().max().item(), rtol=precision)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU")
def test_non_causal_rejected():
    """`causal=False` must raise rather than return the causal answer.

    Causality is structural -- the KV block range and the score mask are both
    unconditionally causal, and the flag reaches neither -- so a dropped flag is
    bit-identical to an honoured one and nothing downstream can notice.
    """
    from triton.tlx.ops import InvalidInput
    from triton.tlx.ops import hstu_attn as tlx_hstu_attn

    q, k, v, offsets, attn_scale = _inputs(2, 512, 4, 128, torch.bfloat16)
    with pytest.raises(InvalidInput, match="causal"):
        tlx_hstu_attn(q, k, v, offsets, 512, attn_scale, alpha=1.0 / 128, causal=False, arch=SM100_ARCH,
                      space="smoke")


def _gfx950_inputs(batch_size, max_seq_len, H, attn_dim, hidden_dim, dtype):
    device = torch.device("cuda")
    lengths = torch.linspace(max_seq_len // 2, max_seq_len, batch_size, device=device, dtype=torch.int32)
    offsets = torch.zeros((batch_size + 1, ), dtype=torch.int64, device=device)
    offsets[1:] = torch.cumsum(lengths.to(torch.int64), dim=0)
    total = int(offsets[-1].item())
    x = torch.empty((total, H, attn_dim * 2 + hidden_dim), dtype=dtype, device=device).uniform_(-0.01, 0.01)
    q, k, v = torch.split(x, [attn_dim, attn_dim, hidden_dim], dim=-1)
    num_targets = torch.clamp(lengths // 4, min=1)
    return q.contiguous(), k.contiguous(), v.contiguous(), offsets, num_targets


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
@pytest.mark.parametrize("batch_size, MAX_SEQ_LEN, H, ATTN_DIM, HIDDEN_DIM", GFX950_SHAPES)
def test_hstu_attn_gfx950(batch_size, MAX_SEQ_LEN, H, ATTN_DIM, HIDDEN_DIM):
    from triton.tlx.ops import hstu_attn_dev as tlx_hstu_attn
    from triton.tlx.ops.kernels.hstu_attn.gfx950 import torch_hstu_attention as torch_hstu_attn_ref

    torch.cuda.empty_cache()
    dtype = torch.bfloat16
    alpha = 10000.0 / ATTN_DIM
    q, k, v, offsets, num_targets = _gfx950_inputs(batch_size, MAX_SEQ_LEN, H, ATTN_DIM, HIDDEN_DIM, dtype)

    out = tlx_hstu_attn(q, k, v, offsets, MAX_SEQ_LEN, None, alpha=alpha, causal=True, num_targets=num_targets,
                        arch=GFX950_ARCH, space="smoke")
    ref = torch_hstu_attn_ref(
        MAX_SEQ_LEN,
        alpha,
        q,
        k,
        v,
        offsets,
        causal=True,
        dropout_pr=0.0,
        training=False,
        num_targets=num_targets,
    )

    torch.testing.assert_close(out * MAX_SEQ_LEN, ref * MAX_SEQ_LEN, atol=1e-3, rtol=0)
