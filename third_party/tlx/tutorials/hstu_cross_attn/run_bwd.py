"""Minimal OSS driver: run the HSTU cross-attention backward (reduce_dq) standalone.
Usage:
  HSTU_BWD_VARIANT=triton_redq  ~/.conda/envs/metamain2/bin/python run_bwd.py        # non-WS
  HSTU_BWD_VARIANT=triton_autows TRITON_USE_META_WS=1 ~/.conda/envs/metamain2/bin/python run_bwd.py  # autoWS
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import torch
import triton_bw_cross_attention as xa


def main():
    torch.manual_seed(0)
    dev = "cuda"
    Z, H, D = 2, 2, 128
    L = 256  # per-sequence length (Q == KV here)
    dtype = torch.bfloat16
    total = Z * L
    q = torch.randn(total, H, D, device=dev, dtype=dtype, requires_grad=True)
    k = torch.randn(total, H, D, device=dev, dtype=dtype, requires_grad=True)
    v = torch.randn(total, H, D, device=dev, dtype=dtype, requires_grad=True)
    seq_offsets = torch.tensor([0, L, 2 * L], device=dev, dtype=torch.int64)
    attn_scale = torch.tensor(1.0 / L, device=dev, dtype=torch.float32)

    out = xa.triton_bw_hstu_mha_wrapper(
        max_seq_len=L,
        alpha=1.0 / D,
        q=q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        attn_scale=attn_scale,
        max_q_len=L,
        seq_offsets_q=seq_offsets,
        num_softmax_heads=H,   # all-or-nothing softmax (matches constexpr fix)
        shared_kv=False,
        enable_tma=True,
    )
    print("fwd out:", tuple(out.shape), out.dtype, "finite:", torch.isfinite(out).all().item())
    do = torch.randn_like(out)
    out.backward(do)
    print("dq:", tuple(q.grad.shape), "finite:", torch.isfinite(q.grad).all().item())
    print("dk:", tuple(k.grad.shape), "finite:", torch.isfinite(k.grad).all().item())
    print("dv:", tuple(v.grad.shape), "finite:", torch.isfinite(v.grad).all().item())
    print("OK variant:", xa.get_bwd_variant().name)


if __name__ == "__main__":
    main()
