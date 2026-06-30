"""n05 — GEMM/MMA: which config knobs change the bits (anticipates M3).

A matmul ``C = A @ B`` via ``tl.dot``. Two findings, both relevant to the M3
"GEMM equivalence" milestone:

1. **Tiling knobs are bitwise-STABLE here.** Sweeping ``num_warps`` (4, 8) and
   ``BLOCK_K`` (32, 64, 128) at fixed precision lands in a SINGLE bit-class. The
   tensor-core MMA accumulates K in fixed instruction-sized chunks regardless of
   how we tile the loop, so the accumulation order -- and the bits -- don't move.
   (This is *why* MMA equivalence is a different, in some ways easier sub-problem
   than reduction equivalence: the tiling doesn't perturb the order.)

2. **The precision knob DOES change the bits.** Sweeping ``input_precision``
   (ieee / tf32 / tf32x3) gives DISTINCT bit-classes -- different MMA precision
   modes round differently. An autotuner that is free to pick the precision mode
   therefore silently picks the numerics, exactly like the reduction examples.

So for GEMM the "config silently sets the bits" problem shows up through the
precision mode (and, in general, through MMA instruction selection that M3
targets) rather than through loop tiling.

We hold ``num_warps=4`` for the MMA: a tensor-core dot needs a full warpgroup,
and on Blackwell num_warps<4 HANGS the dot.

Run:  python bitequiv/examples/numerical-inconsistency/n05_dot_reduce.py
"""
import torch

import triton
import triton.language as tl

from _helpers import (banner, compile_only, group_by_bits, is_cuda, short, show)

M, N, K = 128, 128, 128
NUM_WARPS = 4
# The knob that changes the bits. (tf32x3 is omitted: its 3-pass emulation needs
# more tensor memory than a 128x128 tile allows on this GPU; ieee vs tf32 already
# demonstrate the divergence. Unsupported modes are skipped at runtime anyway.)
PRECISIONS = ["ieee", "tf32"]
TILINGS = [(4, 128), (8, 128), (4, 64), (4, 32)]  # (num_warps, BLOCK_K) -- stable


@triton.jit
def gemm_kernel(a, b, c, M, N, K,
                sa0, sa1, sb0, sb1, sc0, sc1,
                PREC: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    offm = tl.arange(0, BM)
    offn = tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        offk = k0 + tl.arange(0, BK)
        a_tile = tl.load(a + offm[:, None] * sa0 + offk[None, :] * sa1)
        b_tile = tl.load(b + offk[:, None] * sb0 + offn[None, :] * sb1)
        acc = tl.dot(a_tile, b_tile, acc, input_precision=PREC)
    tl.store(c + offm[:, None] * sc0 + offn[None, :] * sc1, acc)


def run_config(a, b, prec, num_warps, block_k):
    c = torch.empty(M, N, device="cuda", dtype=torch.float32)
    gemm_kernel[(1, )](a, b, c, M, N, K,
                       a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                       c.stride(0), c.stride(1),
                       PREC=prec, BM=M, BN=N, BK=block_k,
                       num_warps=num_warps, num_stages=1)
    torch.cuda.synchronize()
    return c


def main():
    g = torch.Generator(device="cpu").manual_seed(0)
    a = torch.randn(M, K, generator=g).to("cuda", torch.float32)
    b = torch.randn(K, N, generator=g).to("cuda", torch.float32)
    ref = a.double() @ b.double()

    def rel_err(c):
        return (c.double() - ref).abs().max().item() / max(1.0, ref.abs().max().item())

    banner("n05 (1) tiling knobs (num_warps, BLOCK_K) at fixed ieee precision")
    tiled = [(f"nw={nw},BK={bk}", run_config(a, b, "ieee", nw, bk)) for nw, bk in TILINGS]
    tiling_classes = group_by_bits(tiled)
    for key, members in tiling_classes.items():
        print(f"    {members}  ->  bitclass {short(key)}")
    print(f"    => {len(tiling_classes)} bit-class(es): the MMA tiling is "
          f"{'STABLE' if len(tiling_classes) == 1 else 'NOT stable'} across these configs.")

    banner("n05 (2) precision knob (input_precision) -- this changes the bits")
    ck = compile_only(gemm_kernel, a, b, torch.empty(M, N, device="cuda"), M, N, K,
                      a.stride(0), a.stride(1), b.stride(0), b.stride(1), M, N,
                      PREC="tf32", BM=M, BN=N, BK=K, num_warps=NUM_WARPS, num_stages=1, grid=(1, ))
    show(ck, "ttgir", grep=["tf32", "InputPrecision", "input_precision"], limit=3)
    prec_outputs = []
    for prec in PRECISIONS:
        try:
            c = run_config(a, b, prec, NUM_WARPS, K)
        except Exception as e:  # noqa: BLE001 - some precision modes may exceed resources
            print(f"    input_precision={prec}:  skipped ({type(e).__name__})")
            continue
        prec_outputs.append((f"input_precision={prec}", c))
        print(f"    input_precision={prec}:  rel err vs fp64={rel_err(c):.3e}")

    classes = group_by_bits(prec_outputs)
    print("\n    bitwise-equivalence classes (by exact bits of the whole C matrix):")
    for key, members in classes.items():
        print(f"        {members}  ->  bitclass {short(key)}")

    assert len(classes) > 1, (
        f"expected the precision knob to change the bits; got {len(classes)} class(es).")
    print(f"\n[OK] for this GEMM the tiling knobs were bitwise-stable ({len(tiling_classes)} class) "
          f"but the precision knob produced {len(classes)} distinct bit-classes — so an autotuner "
          "free to pick MMA precision silently picks the numerics. This is the M3 GEMM-equivalence "
          "problem: prove/force bitwise-equiv MMA across the configs that DO differ.")


if __name__ == "__main__":
    if not is_cuda():
        raise SystemExit("Requires a CUDA GPU.")
    main()
