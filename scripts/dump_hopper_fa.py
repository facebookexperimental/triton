#!/usr/bin/env python3
"""Dump TTIR for Hopper FA kernels."""
import os
import sys
import torch

sys.path.insert(0, os.path.expanduser("~/triton-fb/triton"))

os.environ["TRITON_KERNEL_DUMP"] = "1"
os.environ["TRITON_DUMP_DIR"] = os.path.expanduser("~/triton_ir_dump_hopper_fa")
os.environ["TRITON_ALWAYS_COMPILE"] = "1"

B, H, N_CTX, D_HEAD = 1, 1, 256, 64
sm_scale = 1.0 / (D_HEAD ** 0.5)
q = torch.randn(B, H, N_CTX, D_HEAD, device='cuda', dtype=torch.float16)
k = torch.randn(B, H, N_CTX, D_HEAD, device='cuda', dtype=torch.float16)
v = torch.randn(B, H, N_CTX, D_HEAD, device='cuda', dtype=torch.float16)

kernels = [
    ("hopper_fa_ws", "third_party.tlx.tutorials.hopper_fa_ws"),
    ("hopper_fa_ws_pipelined", "third_party.tlx.tutorials.hopper_fa_ws_pipelined"),
    ("hopper_fa_ws_pipelined_pingpong", "third_party.tlx.tutorials.hopper_fa_ws_pipelined_pingpong"),
    ("hopper_fa_ws_pipelined_pingpong_persistent", "third_party.tlx.tutorials.hopper_fa_ws_pipelined_pingpong_persistent"),
]

for name, mod_path in kernels:
    print(f"=== {name} ===")
    try:
        mod = __import__(mod_path, fromlist=["attention"])
        fn = getattr(mod, "attention")
        o = fn(q, k, v, sm_scale)
        print(f"  OK: output shape={o.shape}")
    except Exception as e:
        print(f"  Error: {e}")
