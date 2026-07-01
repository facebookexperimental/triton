"""Validation driver for the HSTU cross-attn autoWS BACKWARD (reduce_dq else branch).

The stock run_bwd.py crashes in the FORWARD kernel under TRITON_USE_META_WS=1
because the forward autotune configs use num_stages=0 and the WS pass asserts
numStages>=1. That forward crash is unrelated to the backward L6 fix and blocks
reaching the backward. Here we force forward configs to num_stages>=1 (and shrink
the fwd autotune space) so we can actually compile+run the backward reduce_dq
autoWS kernel, then compare grads against the non-WS redq reference.

Usage:
  TRITON_USE_META_WS=1 python run_bwd_validate.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import torch
import triton
import triton_bw_cross_attention as xa


def _patch_fwd_configs_num_stages():
    """Force forward autotune configs to num_stages>=1 and shrink the space so the
    WS-pass numStages>=1 assert doesn't fire on the forward kernel."""
    def _one(cfgs):
        c = cfgs[0]
        c.num_stages = 2
        return [c]

    # Shrink each forward autotuner to a SINGLE config with num_stages>=1 so the
    # WS-pass numStages>=1 assert doesn't fire and autotuning is fast.
    for name in ("_attn_fwd_triton", "_attn_fwd_triton_spec"):
        fn = getattr(xa, name, None)
        if fn is None:
            continue
        if hasattr(fn, "configs") and fn.configs:
            c = fn.configs[0]
            c.num_stages = 2
            fn.configs = [c]

    # Also shrink the backward redq autotuner to a single config for speed
    # (both redq and autows share _hstu_attn_bwd_redq's autotuner).
    for name in ("_hstu_attn_bwd_redq",):
        fn = getattr(xa, name, None)
        if fn is not None and hasattr(fn, "configs") and fn.configs:
            fn.configs = [fn.configs[0]]


def run(bwd_variant, q, k, v, seq_offsets, attn_scale, L, D, H):
    xa.set_bwd_variant(bwd_variant)
    for t in (q, k, v):
        if t.grad is not None:
            t.grad = None
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
        num_softmax_heads=H,  # all-or-nothing softmax
        shared_kv=False,
        enable_tma=True,
    )
    return out


def main():
    torch.manual_seed(0)
    dev = "cuda"
    Z, H, D = 2, 2, 128
    L = 256
    dtype = torch.bfloat16
    total = Z * L

    def mk():
        return torch.randn(total, H, D, device=dev, dtype=dtype, requires_grad=True)

    q, k, v = mk(), mk(), mk()
    seq_offsets = torch.tensor([0, L, 2 * L], device=dev, dtype=torch.int64)
    attn_scale = torch.tensor(1.0 / L, device=dev, dtype=torch.float32)
    do = torch.randn(total, H, D, device=dev, dtype=dtype)

    _patch_fwd_configs_num_stages()
    xa.set_fwd_variant(xa.FwdVariant.TRITON)  # plain fwd (no warp_specialize loops)

    # --- reference: non-WS redq (compile WITHOUT meta-ws so the peeled path is
    #     not force-pipelined by the WS pass) ---
    os.environ["TRITON_USE_META_WS"] = "0"
    out_ref = run(xa.BwdVariant.TRITON_REDQ, q, k, v, seq_offsets, attn_scale, L, D, H)
    out_ref.backward(do)
    dq_ref = q.grad.detach().clone()
    dk_ref = k.grad.detach().clone()
    dv_ref = v.grad.detach().clone()
    print("REF (redq) fwd finite:", torch.isfinite(out_ref).all().item())
    print("REF dq/dk/dv finite:",
          torch.isfinite(dq_ref).all().item(),
          torch.isfinite(dk_ref).all().item(),
          torch.isfinite(dv_ref).all().item())

    # --- test: autoWS (compile WITH meta-ws) ---
    os.environ["TRITON_USE_META_WS"] = "1"
    out_ws = run(xa.BwdVariant.TRITON_AUTOWS, q, k, v, seq_offsets, attn_scale, L, D, H)
    print("WS variant selected:", xa.get_bwd_variant().name)
    print("WS fwd finite:", torch.isfinite(out_ws).all().item())
    out_ws.backward(do)
    dq_ws = q.grad.detach().clone()
    dk_ws = k.grad.detach().clone()
    dv_ws = v.grad.detach().clone()

    def rel_l2(a, b):
        return (torch.linalg.vector_norm((a - b).float()) /
                (torch.linalg.vector_norm(b.float()) + 1e-12)).item()

    def maxabs(a, b):
        return (a - b).abs().max().item()

    print("\n=== autoWS bwd vs redq reference ===")
    for name, a, b in (("dq", dq_ws, dq_ref), ("dk", dk_ws, dk_ref), ("dv", dv_ws, dv_ref)):
        print(f"{name}: rel_l2={rel_l2(a, b):.3e}  max_abs={maxabs(a, b):.3e}  "
              f"ws_finite={torch.isfinite(a).all().item()}")


if __name__ == "__main__":
    main()
