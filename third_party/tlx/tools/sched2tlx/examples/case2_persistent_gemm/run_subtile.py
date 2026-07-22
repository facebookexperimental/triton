"""Pass A.7 epilogue-subtile demo for case2 (bm128 persistent GEMM).

Compares the baseline (`generated_base`) against the epilogue-subtiled S=4
kernel (`generated_subtile`) for correctness and perf. Both are emitted from
`pre_modulo_bm128.ttgir`:

    TRITON_OPT=<beta triton-opt>
    $TRITON_OPT pre_modulo_bm128.ttgir --nvgpu-modulo-schedule \
        -o /dev/null   # +TRITON_MODULO_DUMP_SCHEDULE=schedule_graph_base.json
    TRITON_MODULO_EPILOGUE_SUBTILE=4 ... --nvgpu-modulo-schedule \
        -o /dev/null   # +TRITON_MODULO_DUMP_SCHEDULE=schedule_graph_subtile.json
    python3 -m sched2tlx schedule_graph_{base,subtile}.json -o generated_{base,subtile}.py

Run on a B200: `python3 run_subtile.py`.
"""

from __future__ import annotations

import sys

import importlib

import torch
import triton

# Import the sibling generated modules. As plain scripts they are top-level; in a
# buck par (base_module="") they are the full dotted package, so fall back to that.
try:
    import generated_base
    import generated_subtile
except ModuleNotFoundError:
    _pkg = __package__ or ""
    generated_base = importlib.import_module(
        (_pkg + ".generated_base") if _pkg else "generated_base"
    )
    generated_subtile = importlib.import_module(
        (_pkg + ".generated_subtile") if _pkg else "generated_subtile"
    )

NUM_SMS = 148  # baked into the persistent loop step at emit time


def alloc_fn(size, alignment, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


def _launch(mod, a, b, c, M, N, K):
    mod._gemm_persistent[(NUM_SMS,)](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        b.stride(0),
        c.stride(0),
        num_warps=4,
        num_ctas=1,
        num_stages=2,
    )


def check(mod, M, N, K):
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.full((M, N), float("nan"), device="cuda", dtype=torch.float16)
    _launch(mod, a, b, c, M, N, K)
    torch.cuda.synchronize()
    ref = torch.matmul(a.float(), b.float())
    nan = torch.isnan(c).sum().item()
    rel = (c.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-9)
    return nan == 0 and rel < 5e-3, rel


def tflops(mod, M, N, K):
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.empty(M, N, device="cuda", dtype=torch.float16)
    ms = triton.testing.do_bench(
        lambda: _launch(mod, a, b, c, M, N, K), warmup=50, rep=200
    )
    return 2 * M * N * K / (ms * 1e-3) / 1e12


def main() -> int:
    triton.set_allocator(alloc_fn)
    # Epilogue-bound (small K, large M*N) — where subtiling helps — plus a
    # K-heavy neutrality check.
    shapes = [
        (8192, 8192, 64),
        (16384, 16384, 64),
        (8192, 8192, 256),
        (4096, 4096, 4096),
    ]
    failed = 0
    print("=== correctness ===")
    for M, N, K in shapes:
        for name, mod in (("base", generated_base), ("subtile", generated_subtile)):
            ok, rel = check(mod, M, N, K)
            failed += 0 if ok else 1
            print(f"[{'PASS' if ok else 'FAIL'}] {name:8s} {M}x{N}x{K}  rel={rel:.2e}")
    print("\n=== perf (base vs subtile S=4) ===")
    print(f"{'shape':>18} {'base TF':>9} {'sub TF':>9} {'ratio':>7}")
    for M, N, K in shapes:
        btf = tflops(generated_base, M, N, K)
        stf = tflops(generated_subtile, M, N, K)
        print(f"{f'{M}x{N}x{K}':>18} {btf:>9.1f} {stf:>9.1f} {stf / btf:>6.2f}x")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
