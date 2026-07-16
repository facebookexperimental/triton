"""Run emitter-generated case9 blockwise fp8 scaled_mm on B200 vs torch.

The generated kernel is the sched2tlx lowering of the non-WS blockwise scaled_mm
(fresh per-128-K-group MMA → tmem_load → rescale by sa[m,g]*sb[nblk,g] → register
running-sum), warp-specialized into LOAD-A / LOAD-B / SFLoad / MMA / PROMO groups.
"""

from __future__ import annotations

import importlib
import sys

import torch
import triton

# Import the sibling generated.py. As a plain script it's a top-level module; in
# a buck par it's the full dotted package (base_module=""), so fall back to that.
try:
    import generated
except ModuleNotFoundError:
    generated = importlib.import_module(
        (__package__ + ".generated") if __package__ else "generated"
    )

NUM_SMS = 148


def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


def _ref(a, b, scale_a, scale_b):
    """fp32 blockwise reference (mirrors blackwell_scaled_mm_ws._ref)."""
    M, K = a.shape
    N = b.shape[0]
    af = a.to(torch.float32)
    bf = b.to(torch.float32)
    out = torch.zeros((M, N), dtype=torch.float32, device=a.device)
    for g in range(K // 128):
        ak = af[:, g * 128 : (g + 1) * 128]
        bk = bf[:, g * 128 : (g + 1) * 128]
        partial = ak @ bk.t()
        sa = scale_a[:, g][:, None]
        sb = scale_b[:, g].repeat_interleave(128)[None, :]
        out += partial * sa * sb
    return out


def main() -> int:
    triton.set_allocator(alloc_fn)
    torch.manual_seed(0)

    shapes = [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ]
    failed = 0
    for M, N, K in shapes:
        a = (torch.randn(M, K, device="cuda") * 0.1).to(torch.float8_e4m3fn)
        b = (torch.randn(N, K, device="cuda") * 0.1).to(torch.float8_e4m3fn)
        # scale_a M-major (outer-dim-major): stride (1, M); scale_b row-major.
        scale_a = (
            torch.rand(M, K // 128, device="cuda", dtype=torch.float32)
            .t()
            .contiguous()
            .t()
        )
        scale_b = torch.rand(N // 128, K // 128, device="cuda", dtype=torch.float32)
        c = torch.full((M, N), float("nan"), device="cuda", dtype=torch.bfloat16)

        grid = (NUM_SMS,)
        try:
            generated._scaled_mm_blockwise[grid](
                a,
                b,
                c,
                scale_a,
                scale_b,
                M,
                N,
                K,
                a.stride(0),  # stride_am (= K)
                b.stride(0),  # stride_bn (= K)
                c.stride(0),  # stride_cm (= N)
                scale_a.stride(1),  # stride_sa_g (= M, M-major)
                scale_b.stride(0),  # stride_sb_n (= K//128)
                num_warps=8,
                num_ctas=1,
                num_stages=2,
            )
        except Exception as e:
            print(f"[FAIL-COMPILE] M={M} N={N} K={K}: {type(e).__name__}: {str(e)[:300]}")
            failed += 1
            continue

        ref = _ref(a, b, scale_a, scale_b)
        nan = torch.isnan(c).sum().item()
        err = (c.float() - ref).abs().max().item()
        rel = err / max(ref.abs().max().item(), 1e-9)
        ok = nan == 0 and rel < 5e-2  # fp8 tolerance
        marker = "PASS" if ok else "FAIL"
        print(f"[{marker}] M={M} N={N} K={K}  nan={nan}  max abs={err:.3e}  rel={rel:.3e}")
        if not ok:
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
