"""Compile a single TLX Hopper kernel and dump IR for deadlock analysis."""
import os
import sys
import torch

DUMP_DIR = os.path.expanduser("~/triton_ir_dump")
os.environ["TRITON_KERNEL_DUMP"] = "1"
os.environ["TRITON_DUMP_DIR"] = DUMP_DIR
os.environ["TRITON_ALWAYS_COMPILE"] = "1"
os.makedirs(DUMP_DIR, exist_ok=True)

REPO = os.path.expanduser("~/triton-fb/triton")
sys.path.insert(0, os.path.join(REPO, "third_party/tlx/tutorials"))

from hopper_gemm_ws import matmul  # noqa: E402

M, N, K = 256, 256, 256
a = torch.randn(M, K, dtype=torch.float16, device="cuda")
b = torch.randn(K, N, dtype=torch.float16, device="cuda")
c = matmul(a, b)
print(f"Done: output shape {c.shape}")

# Find first ttir file
for d in sorted(os.listdir(DUMP_DIR)):
    subdir = os.path.join(DUMP_DIR, d)
    if os.path.isdir(subdir):
        ttir = os.path.join(subdir, "matmul_kernel_tlx_ws.ttir")
        if os.path.exists(ttir):
            print(f"TTIR: {ttir}")
            break
