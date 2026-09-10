"""tlx.ops.hstu_attn correctness -- gfx950."""
import pytest
import torch
from triton._internal_testing import is_hip_cdna4

GFX950_ARCH = "gfx950"

GFX950_SHAPES = [
    (2, 128, 2, 128, 128),
    (4, 256, 4, 128, 128),
]


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
