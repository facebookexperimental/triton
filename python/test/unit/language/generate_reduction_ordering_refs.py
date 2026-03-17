"""Generate reference tensors for reduction ordering bitwise equivalence tests.

Run this script once to produce .pt files in test_data/ that the tests load.
Uses num_warps=1 with INNER_TREE ordering to produce the canonical reference.
A single 32-row input is used for all BLOCK_M tile sizes.
"""

import os
import torch
import triton
import triton.language as tl

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "test_data")

TOTAL_ROWS = 32
BLOCK_N = 1024


@triton.jit
def sum_kernel(X, Z, stride_x, N_ROWS: tl.constexpr, BLOCK_N: tl.constexpr, ORDERING: tl.constexpr):
    offs_m = tl.arange(0, N_ROWS)
    offs_n = tl.arange(0, BLOCK_N)
    x = tl.load(X + offs_m[:, None] * stride_x + offs_n[None, :])
    z = tl.sum(x, axis=1, reduction_ordering=ORDERING)
    tl.store(Z + offs_m, z)


@triton.jit
def mul_combine(a, b):
    return a * b


@triton.jit
def mul_kernel(X, Z, stride_x, N_ROWS: tl.constexpr, BLOCK_N: tl.constexpr, ORDERING: tl.constexpr):
    offs_m = tl.arange(0, N_ROWS)
    offs_n = tl.arange(0, BLOCK_N)
    x = tl.load(X + offs_m[:, None] * stride_x + offs_n[None, :])
    z = tl.reduce(x, axis=1, combine_fn=mul_combine, reduction_ordering=ORDERING)
    tl.store(Z + offs_m, z)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    device = "cuda"

    # Sum reference
    torch.manual_seed(42)
    x_sum = torch.randn((TOTAL_ROWS, BLOCK_N), device=device, dtype=torch.float32)
    torch.save(x_sum.cpu(), os.path.join(DATA_DIR, "reduction_ordering_sum_input.pt"))

    out_sum = torch.empty(TOTAL_ROWS, device=device, dtype=torch.float32)
    sum_kernel[(1, )](x_sum, out_sum, x_sum.stride(0), N_ROWS=TOTAL_ROWS, BLOCK_N=BLOCK_N,
                      ORDERING=tl.ReductionOrdering.INNER_TREE, num_warps=1)
    torch.save(out_sum.cpu(), os.path.join(DATA_DIR, "reduction_ordering_sum_ref.pt"))
    print(f"Saved sum input ({TOTAL_ROWS}x{BLOCK_N}) + sum reference ({TOTAL_ROWS},)")

    # Mul reference (use small uniform values to avoid inf/0)
    x_mul = torch.empty((TOTAL_ROWS, BLOCK_N), device=device, dtype=torch.float32).uniform_(0.99, 1.01)
    torch.save(x_mul.cpu(), os.path.join(DATA_DIR, "reduction_ordering_mul_input.pt"))

    out_mul = torch.empty(TOTAL_ROWS, device=device, dtype=torch.float32)
    mul_kernel[(1, )](x_mul, out_mul, x_mul.stride(0), N_ROWS=TOTAL_ROWS, BLOCK_N=BLOCK_N,
                      ORDERING=tl.ReductionOrdering.INNER_TREE, num_warps=1)
    torch.save(out_mul.cpu(), os.path.join(DATA_DIR, "reduction_ordering_mul_ref.pt"))
    print(f"Saved mul input ({TOTAL_ROWS}x{BLOCK_N}) + mul reference ({TOTAL_ROWS},)")

    print(f"All reference tensors saved to {DATA_DIR}")


if __name__ == "__main__":
    main()
