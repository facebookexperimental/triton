"""case5 epilogue-subtile demo: baseline vs S=4 subtiled addmm+bias.

Validates the GENERALIZED epilogue subtiler: case5's epilogue chain is
`tmem_load` + bias `descriptor_load` -> `extf` -> `addf` -> `truncf` -> store,
i.e. it has an external tensor operand (the bias) and a non-cast compute op
(`addf`) — neither of which the old truncf/convert-only subtiler could handle.
The bias staging is loaded full once per tile and sub-sliced per sub-tile
(option B). Both kernels are emitted from `addmm_bias_pre_modulo.ttgir`:

    $TRITON_OPT addmm_bias_pre_modulo.ttgir --nvgpu-modulo-schedule -o /dev/null
        # TRITON_MODULO_DUMP_SCHEDULE=schedule_graph.json  (base, no subtile)
    TRITON_MODULO_EPILOGUE_SUBTILE=4 ... --nvgpu-modulo-schedule -o /dev/null
        # TRITON_MODULO_DUMP_SCHEDULE=schedule_graph_subtile.json  (S=4)
    python3 -m sched2tlx schedule_graph{,_subtile}.json -o generated{,_subtile}.py
"""

from __future__ import annotations

import importlib
import sys

import torch
import triton
from triton.tools.tensor_descriptor import TensorDescriptor

NUM_SMS = 148
BM, BN, BK = 128, 128, 64
S = 4  # epilogue subtile factor baked into generated_subtile.py

try:
    import generated
    import generated_subtile
except ModuleNotFoundError:
    _pkg = __package__ or ""
    generated = importlib.import_module((_pkg + ".generated") if _pkg else "generated")
    generated_subtile = importlib.import_module(
        (_pkg + ".generated_subtile") if _pkg else "generated_subtile"
    )


def alloc_fn(size, alignment, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


def _launch(mod, a, b, bias, c, M, N, K, cn):
    a_desc = TensorDescriptor.from_tensor(a, [BM, BK])
    b_desc = TensorDescriptor.from_tensor(b, [BK, BN])
    bias_desc = TensorDescriptor.from_tensor(bias, [BM, BN])
    # subtiled store writes (BM, BN/S) tiles, so its c_desc block is (BM, cn).
    c_desc = TensorDescriptor.from_tensor(c, [BM, cn])
    mod.addmm_persistent_2d_bias[(NUM_SMS,)](
        a_desc, b_desc, bias_desc, c_desc, M, N, K,
        num_warps=4, num_ctas=1, num_stages=2,
    )


def check(mod, M, N, K, cn):
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    bias = torch.randn(M, N, device="cuda", dtype=torch.float16)
    c = torch.full((M, N), float("nan"), device="cuda", dtype=torch.float16)
    _launch(mod, a, b, bias, c, M, N, K, cn)
    torch.cuda.synchronize()
    ref = torch.matmul(a.float(), b.float()) + bias.float()
    nan = torch.isnan(c).sum().item()
    rel = (c.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-9)
    return nan == 0 and rel < 5e-3, rel


def tflops(mod, M, N, K, cn):
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    bias = torch.randn(M, N, device="cuda", dtype=torch.float16)
    c = torch.empty(M, N, device="cuda", dtype=torch.float16)
    ms = triton.testing.do_bench(
        lambda: _launch(mod, a, b, bias, c, M, N, K, cn), warmup=50, rep=200
    )
    return 2 * M * N * K / (ms * 1e-3) / 1e12


def main() -> int:
    triton.set_allocator(alloc_fn)
    # epilogue-bound (small K) where subtiling helps, + a K-heavy neutrality check
    shapes = [(8192, 8192, 64), (16384, 16384, 64), (8192, 8192, 256), (4096, 4096, 4096)]
    failed = 0
    print("=== correctness ===")
    for M, N, K in shapes:
        for name, mod, cn in (("base", generated, BN), ("subtile", generated_subtile, BN // S)):
            ok, rel = check(mod, M, N, K, cn)
            failed += 0 if ok else 1
            print(f"[{'PASS' if ok else 'FAIL'}] {name:8s} {M}x{N}x{K}  rel={rel:.2e}")
    print("\n=== perf (base vs subtile S=4) ===")
    print(f"{'shape':>18} {'base TF':>9} {'sub TF':>9} {'ratio':>7}")
    for M, N, K in shapes:
        btf = tflops(generated, M, N, K, BN)
        stf = tflops(generated_subtile, M, N, K, BN // S)
        print(f"{f'{M}x{N}x{K}':>18} {btf:>9.1f} {stf:>9.1f} {stf / btf:>6.2f}x")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
