"""Run hand-written TLX-WS reference kernel for case5."""

from __future__ import annotations

import sys

import handwritten as hw
import torch
import triton
from triton.tools.tensor_descriptor import TensorDescriptor

NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count


def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


def main() -> int:
    triton.set_allocator(alloc_fn)
    torch.manual_seed(0)
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
    NUM_BUFFERS_AB, NUM_BUFFERS_ACC = 2, 2

    shapes = [
        (256, 256, 128),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
        (1024, 1024, 16384),
    ]
    failed = 0
    for M, N, K in shapes:
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)
        bias = torch.randn(M, N, device="cuda", dtype=torch.float16)
        c = torch.full((M, N), float("nan"), device="cuda", dtype=torch.float16)

        a_desc = TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K])
        b_desc = TensorDescriptor.from_tensor(b, [BLOCK_K, BLOCK_N])
        bias_desc = TensorDescriptor.from_tensor(bias, [BLOCK_M, BLOCK_N])
        c_desc = TensorDescriptor.from_tensor(c, [BLOCK_M, BLOCK_N])

        grid = (NUM_SMS, )
        hw.addmm_persistent_2d_bias[grid](
            a_desc,
            b_desc,
            bias_desc,
            c_desc,
            M,
            N,
            K,
            NUM_SMS=NUM_SMS,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            NUM_BUFFERS_AB=NUM_BUFFERS_AB,
            NUM_BUFFERS_ACC=NUM_BUFFERS_ACC,
            num_warps=4,
            num_ctas=1,
            num_stages=2,
        )

        ref = (torch.matmul(a.float(), b.float()) + bias.float()).to(torch.float16)
        nan = torch.isnan(c).sum().item()
        err = (c.float() - ref.float()).abs().max().item()
        rel = err / max(ref.float().abs().max().item(), 1e-9)
        ok = nan == 0 and rel < 5e-3
        marker = "PASS" if ok else "FAIL"
        print(f"[{marker}] M={M} N={N} K={K}  nan={nan}  max abs={err:.3e}  rel={rel:.3e}")
        if not ok:
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
