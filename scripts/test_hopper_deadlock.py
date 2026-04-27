#!/usr/bin/env python3
"""Dump TTIR for all Hopper TLX kernels and run barrier deadlock detection.

Usage: python3 scripts/test_hopper_deadlock.py
"""

import os
import sys
import glob
import subprocess
import shutil

TRITON_OPT = os.path.expanduser(
    "~/triton-fb/triton/build/cmake.linux-x86_64-cpython-3.12/bin/triton-opt"
)
DUMP_DIR = os.path.expanduser("~/triton_ir_dump_hopper")
Z3_DIR = "/tmp/z3_hopper_scripts"

# Clean and recreate dump dirs
for d in [DUMP_DIR, Z3_DIR]:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)

os.environ["TRITON_KERNEL_DUMP"] = "1"
os.environ["TRITON_DUMP_DIR"] = DUMP_DIR
os.environ["TRITON_ALWAYS_COMPILE"] = "1"

# Hopper kernels and their entry points
KERNELS = {
    "hopper_gemm_ws": {
        "module": "third_party.tlx.tutorials.hopper_gemm_ws",
        "func": "matmul",
        "args": "M=256, N=256, K=256",
    },
    "hopper_gemm_pipelined": {
        "module": "third_party.tlx.tutorials.hopper_gemm_pipelined",
        "func": "matmul",
        "args": "M=256, N=256, K=256",
    },
    "hopper_fa_ws": {
        "module": "third_party.tlx.tutorials.hopper_fa_ws",
        "func": "attention",
        "args": "B=1, H=1, N_CTX=256, D_HEAD=64",
    },
    "hopper_fa_ws_pipelined": {
        "module": "third_party.tlx.tutorials.hopper_fa_ws_pipelined",
        "func": "attention",
        "args": "B=1, H=1, N_CTX=256, D_HEAD=64",
    },
    "hopper_fa_ws_pipelined_pingpong": {
        "module": "third_party.tlx.tutorials.hopper_fa_ws_pipelined_pingpong",
        "func": "attention",
        "args": "B=1, H=1, N_CTX=256, D_HEAD=64",
    },
    "hopper_fa_ws_pipelined_pingpong_persistent": {
        "module": "third_party.tlx.tutorials.hopper_fa_ws_pipelined_pingpong_persistent",
        "func": "attention",
        "args": "B=1, H=1, N_CTX=256, D_HEAD=64",
    },
}

# Step 1: Dump TTIR for each kernel
print("=" * 60)
print("Step 1: Dumping TTIR for Hopper kernels")
print("=" * 60)

for name, info in KERNELS.items():
    print(f"\n--- Dumping {name} ---")
    script = f"""
import sys
sys.path.insert(0, '.')
import torch
from {info['module']} import {info['func']}
{info['args'].replace(', ', '; ')}
"""
    # Build a proper invocation script
    script_path = f"/tmp/dump_{name}.py"
    with open(script_path, "w") as f:
        if "gemm" in name:
            f.write(f"""
import sys, os, torch
sys.path.insert(0, '{os.path.expanduser("~/triton-fb/triton")}')
os.environ['TRITON_KERNEL_DUMP'] = '1'
os.environ['TRITON_DUMP_DIR'] = '{DUMP_DIR}'
os.environ['TRITON_ALWAYS_COMPILE'] = '1'
from {info['module']} import {info['func']}
a = torch.randn(256, 256, device='cuda', dtype=torch.float16)
b = torch.randn(256, 256, device='cuda', dtype=torch.float16)
c = {info['func']}(a, b)
print(f'{name}: done, output shape={{c.shape}}')
""")
        else:
            # Flash attention
            f.write(f"""
import sys, os, torch
sys.path.insert(0, '{os.path.expanduser("~/triton-fb/triton")}')
os.environ['TRITON_KERNEL_DUMP'] = '1'
os.environ['TRITON_DUMP_DIR'] = '{DUMP_DIR}'
os.environ['TRITON_ALWAYS_COMPILE'] = '1'
from {info['module']} import {info['func']}
B, H, N_CTX, D_HEAD = 1, 1, 256, 64
q = torch.randn(B, H, N_CTX, D_HEAD, device='cuda', dtype=torch.float16)
k = torch.randn(B, H, N_CTX, D_HEAD, device='cuda', dtype=torch.float16)
v = torch.randn(B, H, N_CTX, D_HEAD, device='cuda', dtype=torch.float16)
sm_scale = 1.0 / (D_HEAD ** 0.5)
o = {info['func']}(q, k, v, sm_scale)
print(f'{name}: done, output shape={{o.shape}}')
""")
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, timeout=120,
            cwd=os.path.expanduser("~/triton-fb/triton"),
        )
        if result.returncode == 0:
            print(f"  OK: {result.stdout.strip().split(chr(10))[-1]}")
        else:
            print(f"  FAILED: {result.stderr.strip()[-200:]}")
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT")

# Step 2: Find TTIR files and run deadlock detection
print("\n" + "=" * 60)
print("Step 2: Running barrier deadlock detection")
print("=" * 60)

ttir_files = glob.glob(f"{DUMP_DIR}/**/*.ttir", recursive=True)
print(f"Found {len(ttir_files)} TTIR files")

results = {}
for ttir in sorted(ttir_files):
    kernel_name = os.path.basename(ttir).replace(".ttir", "")
    z3_script = f"{Z3_DIR}/{kernel_name}.py"
    print(f"\n--- {kernel_name} ({ttir}) ---")

    # Run triton-opt to generate Z3 script
    cmd = [
        TRITON_OPT, ttir,
        f"--pass-pipeline=builtin.module(tritongpu-barrier-deadlock-detection{{output-path={z3_script}}})",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        print(f"  triton-opt FAILED: {result.stderr.strip()[-200:]}")
        results[kernel_name] = "TRITON-OPT-FAIL"
        continue

    if not os.path.exists(z3_script):
        print(f"  No Z3 script generated (no warp_specialize?)")
        results[kernel_name] = "NO-WS"
        continue

    # Check syntax
    try:
        with open(z3_script) as f:
            compile(f.read(), z3_script, "exec")
    except SyntaxError as e:
        print(f"  Z3 script syntax error: {e}")
        results[kernel_name] = "SYNTAX-ERROR"
        continue

    # Run Z3 solver
    z3_result = subprocess.run(
        [sys.executable, z3_script],
        capture_output=True, text=True, timeout=120,
    )
    output = z3_result.stdout.strip()
    print(f"  {output}")
    if "unsat" in output.lower():
        results[kernel_name] = "UNSAT (safe)"
    elif "sat" in output.lower():
        results[kernel_name] = "SAT (deadlock?!)"
    else:
        results[kernel_name] = f"UNKNOWN: {output[:100]}"

# Summary
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
for k, v in sorted(results.items()):
    status = "PASS" if "UNSAT" in v else ("FALSE POSITIVE!" if "SAT" in v else "?")
    print(f"  {k:50s} {v:20s} [{status}]")
