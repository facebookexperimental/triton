"""Run emitter-generated case5 addmm-bias on B200 vs torch.

NOTE: as of 2026-05-17, sched2tlx does NOT yet emit working code for the
outer-loop epilogue bias TMA load (`_render_descriptor_load` returns
`<tma_load_inline_unsupported>` for in-the-default-partition-body usage).
This runner exists so the regression sweep covers case5 once the emitter
gap is closed. The non-WS baseline (`run_handwritten_nows.py`) and the
hand-written TLX-WS target (`run_handwritten.py`) both pass on all 6
shapes today; they pin the correctness target the generated kernel should
match.
"""

from __future__ import annotations

import sys

import generated
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
        try:
            generated.addmm_persistent_2d_bias[grid](
                a_desc,
                b_desc,
                bias_desc,
                c_desc,
                M,
                N,
                K,
                num_warps=4,
                num_ctas=1,
                num_stages=2,
            )
        except Exception as e:
            print(f"[FAIL-COMPILE] M={M} N={N} K={K}: {type(e).__name__}: {str(e)[:200]}")
            failed += 1
            continue

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
