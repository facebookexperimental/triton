"""Generate reference tensors for reduction ordering bitwise equivalence tests.

Run this script once to produce .pt files in test_data/ that the tests load.
Uses num_warps=1 with INNER_TREE ordering to produce the canonical reference.
"""

import os
import torch
import triton
import triton.language as tl

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "test_data")


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
    BLOCK_N = 1024
    torch.manual_seed(42)

    for N_ROWS in [1, 4, 16, 32]:
        x = torch.randn((N_ROWS, BLOCK_N), device=device, dtype=torch.float32)
        # Save the input so tests use identical data
        torch.save(x.cpu(), os.path.join(DATA_DIR, f"reduction_ordering_input_{N_ROWS}.pt"))

        # Sum reference
        out_sum = torch.empty(N_ROWS, device=device, dtype=torch.float32)
        sum_kernel[(1, )](x, out_sum, x.stride(0), N_ROWS=N_ROWS, BLOCK_N=BLOCK_N,
                          ORDERING=tl.ReductionOrdering.INNER_TREE, num_warps=1)
        torch.save(out_sum.cpu(), os.path.join(DATA_DIR, f"reduction_ordering_sum_ref_{N_ROWS}.pt"))

        # Mul reference (use small uniform values to avoid inf/0)
        x_mul = torch.empty((N_ROWS, BLOCK_N), device=device, dtype=torch.float32).uniform_(0.99, 1.01)
        torch.save(x_mul.cpu(), os.path.join(DATA_DIR, f"reduction_ordering_mul_input_{N_ROWS}.pt"))

        out_mul = torch.empty(N_ROWS, device=device, dtype=torch.float32)
        mul_kernel[(1, )](x_mul, out_mul, x_mul.stride(0), N_ROWS=N_ROWS, BLOCK_N=BLOCK_N,
                          ORDERING=tl.ReductionOrdering.INNER_TREE, num_warps=1)
        torch.save(out_mul.cpu(), os.path.join(DATA_DIR, f"reduction_ordering_mul_ref_{N_ROWS}.pt"))

        print(f"N_ROWS={N_ROWS}: saved input + sum_ref + mul_input + mul_ref")

    print(f"All reference tensors saved to {DATA_DIR}")


if __name__ == "__main__":
    main()
