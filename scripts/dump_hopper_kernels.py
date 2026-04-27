"""Compile Hopper TLX kernels and dump IR for deadlock analysis."""
import os
import sys
import torch

DUMP_DIR = os.path.expanduser("~/triton_ir_dump_hopper")
os.environ["TRITON_KERNEL_DUMP"] = "1"
os.environ["TRITON_DUMP_DIR"] = DUMP_DIR
os.environ["TRITON_ALWAYS_COMPILE"] = "1"
os.makedirs(DUMP_DIR, exist_ok=True)

REPO = os.path.expanduser("~/triton-fb/triton")
sys.path.insert(0, os.path.join(REPO, "third_party/tlx/tutorials"))

# hopper_fa_ws
print("=== hopper_fa_ws ===")
try:
    from hopper_fa_ws import attention as fa_ws_attention
    q = torch.randn(1, 16, 64, 64, dtype=torch.float16, device="cuda")
    k = torch.randn(1, 16, 64, 64, dtype=torch.float16, device="cuda")
    v = torch.randn(1, 16, 64, 64, dtype=torch.float16, device="cuda")
    o = fa_ws_attention(q, k, v, 1.0 / (64**0.5))
    print(f"  Done: output shape {o.shape}")
except Exception as e:
    print(f"  Error: {e}")

# hopper_fa_ws_pipelined
print("=== hopper_fa_ws_pipelined ===")
try:
    from hopper_fa_ws_pipelined import attention as fa_ws_pipe_attention
    q = torch.randn(1, 16, 64, 64, dtype=torch.float16, device="cuda")
    k = torch.randn(1, 16, 64, 64, dtype=torch.float16, device="cuda")
    v = torch.randn(1, 16, 64, 64, dtype=torch.float16, device="cuda")
    o = fa_ws_pipe_attention(q, k, v, 1.0 / (64**0.5))
    print(f"  Done: output shape {o.shape}")
except Exception as e:
    print(f"  Error: {e}")

# hopper_fa_ws_pipelined_pingpong
print("=== hopper_fa_ws_pipelined_pingpong ===")
try:
    from hopper_fa_ws_pipelined_pingpong import attention as fa_ws_pp_attention
    q = torch.randn(1, 16, 64, 64, dtype=torch.float16, device="cuda")
    k = torch.randn(1, 16, 64, 64, dtype=torch.float16, device="cuda")
    v = torch.randn(1, 16, 64, 64, dtype=torch.float16, device="cuda")
    o = fa_ws_pp_attention(q, k, v, 1.0 / (64**0.5))
    print(f"  Done: output shape {o.shape}")
except Exception as e:
    print(f"  Error: {e}")

# List dumped files
print("\n=== Dumped TTIR files ===")
for d in sorted(os.listdir(DUMP_DIR)):
    subdir = os.path.join(DUMP_DIR, d)
    if os.path.isdir(subdir):
        for f in os.listdir(subdir):
            if f.endswith(".ttir"):
                print(f"  {f}: {os.path.join(subdir, f)}")
