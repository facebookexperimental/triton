"""case6 LayerNorm: perf of modulo-generated vs hand-written TLX (GB/s).

Memory-bound kernel → report effective bandwidth (read x + write y, fp16).
Both run at N=512 for a fair comparison (generated bakes N=512).
"""

from __future__ import annotations

import sys

import generated
import handwritten
import torch
import torch.nn.functional as F
import triton


def alloc_fn(size, alignment, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


def _time(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms


def main() -> int:
    triton.set_allocator(alloc_fn)
    torch.manual_seed(0)
    N = 512
    eps = 1e-5
    NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
    NUM_PERSIST = NUM_SMS * 4
    BLOCK_M = 8

    print(f"{'M':>8} {'N':>5} | {'gen us':>8} {'gen GB/s':>9} | "
          f"{'hw us':>8} {'hw GB/s':>9} | {'gen/hw':>7}")
    print("-" * 70)
    for M in [16384, 65536, 262144]:
        x = torch.randn(M, N, device="cuda", dtype=torch.float16)
        w = torch.randn(N, device="cuda", dtype=torch.float16)
        b = torch.randn(N, device="cuda", dtype=torch.float16)
        yg = torch.empty(M, N, device="cuda", dtype=torch.float16)
        yh = torch.empty(M, N, device="cuda", dtype=torch.float16)
        bytes_moved = M * N * 2 * 2  # read x + write y

        def run_gen():
            # Launch hints emitted by the modulo pass for memory-bound kernels
            # (absent → legacy launch: 1x-SMS grid, register file auto-fill).
            grid = NUM_SMS * getattr(generated, "RECOMMENDED_GRID_MULTIPLIER", 1)
            kw = {"num_warps": 4}
            if (mreg := getattr(generated, "RECOMMENDED_MAXNREG", None)) is not None:
                kw["maxnreg"] = mreg
            generated.layernorm_fwd_nows[(grid, )](x, w, b, yg, M, eps, **kw)

        def run_hw():
            handwritten.layernorm_fwd_tma[(NUM_PERSIST, )](
                x,
                w,
                b,
                yh,
                M,
                eps,
                row_stride=x.stride(0),
                N=N,
                BLOCK_M=BLOCK_M,
                BLOCK_N=512,
                NUM_PERSIST=NUM_PERSIST,
                num_warps=4,
            )

        # correctness guard
        run_gen()
        run_hw()
        torch.cuda.synchronize()
        ref = F.layer_norm(x.float(), (N, ), w.float(), b.float(), eps)
        for name, y in (("gen", yg), ("hw", yh)):
            rel = (y.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-9)
            if rel > 1e-2:
                print(f"  WARN {name} M={M} rel={rel:.2e}")

        gms = _time(run_gen)
        hms = _time(run_hw)
        ggb = bytes_moved / (gms * 1e-3) / 1e9
        hgb = bytes_moved / (hms * 1e-3) / 1e9
        print(f"{M:>8} {N:>5} | {gms*1e3:>8.1f} {ggb:>9.1f} | "
              f"{hms*1e3:>8.1f} {hgb:>9.1f} | {ggb/hgb:>6.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
