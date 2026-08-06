"""Example: how to use the cuBLAS-equivalent Triton GEMM API.

Run:  python -m bitequiv.example_cublas_equiv_gemm

Shows the fp16 and fp8 entry points, that the result is bit-identical to cuBLAS,
and how to handle a shape that has no Triton reconstruction.

`enable_runtime_match` is written explicitly at every call below. The default is
`False` (static: reconstruct from the cuBLAS heuristic alone, no GEMM run); pass
`True` to allow a one-time runtime byte-compare against cuBLAS for shapes that are
not statically guaranteed.
"""
import torch

from bitequiv.cublas_equiv_gemm import (
    CublasNeedRuntimeMatch,
    CublasUnsupportedShape,
    cublas_equivalent_gemm,
    cublas_equivalent_scaled_mm,
    cublas_matmul,
)

DEVICE = "cuda"


def _bit_equal(x, y):
    return torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8))


def example_fp16_plain():
    """fp16 GEMM on a large shape -> cuBLAS runs a plain kernel; our Triton GEMM matches."""
    M, N, K = 4096, 4096, 4096
    a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
    b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)

    out = cublas_equivalent_gemm(a, b, enable_runtime_match=False)  # static (default): heuristic only, no GEMM run
    ref = cublas_matmul(a, b)                                       # cuBLAS's own output (the reference)
    print(f"fp16 plain    {M}x{N}x{K}: bit-identical to cuBLAS = {_bit_equal(out, ref)}")


def example_fp16_split_k():
    """fp16 GEMM on a skinny+deep shape -> cuBLAS runs split-K. Split-K is NOT statically
    guaranteed for either dtype (about 1% of split-K shapes are not reproducible at any
    partition, and nothing in the heuristic flags them), so it needs the runtime byte-compare.
    The split count and the cut both come from the heuristic; the runtime step only gates that
    residual, and the verified plan is cached, so later calls are pure Triton."""
    M, N, K = 64, 64, 32768
    a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
    b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)

    try:                                          # static mode (default): split-K declines
        cublas_equivalent_gemm(a, b, enable_runtime_match=False)
        print(f"fp16 split-K  {M}x{N}x{K}: statically reconstructed")
    except CublasNeedRuntimeMatch:
        print(f"fp16 split-K  {M}x{N}x{K}: needs a runtime match (static mode declined)")

    out = cublas_equivalent_gemm(a, b, enable_runtime_match=True)
    ref = cublas_matmul(a, b)
    print(f"fp16 split-K  {M}x{N}x{K}: with enable_runtime_match=True -> "
          f"bit-identical to cuBLAS = {_bit_equal(out, ref)}")


def example_fp8_plain():
    """fp8 (e4m3) GEMM, plain shape. `b` is [K,N] column-major (as produced by a weight
    `w.t()`); scales are scalars, output defaults to fp16. fp8 plain is static-exact."""
    M, N, K = 8192, 8192, 8192
    a = (torch.randn(M, K, device=DEVICE) * 0.2).to(torch.float8_e4m3fn)
    b = (torch.randn(N, K, device=DEVICE) * 0.2).to(torch.float8_e4m3fn).t()   # [K,N] column-major

    out = cublas_equivalent_scaled_mm(a, b, scale_a=1.0, scale_b=1.0, out_dtype=torch.float16,
                                      enable_runtime_match=False)  # static (default): large-output fp8 plain is exact
    ref = cublas_matmul(a, b, out_dtype=torch.float16)
    print(f"fp8  plain    {M}x{N}x{K}: bit-identical to cuBLAS = {_bit_equal(out, ref)}")


def example_fp8_split_k_runtime():
    """fp8 split-K is NOT statically guaranteed (the vertical/cluster kernel is not
    bit-reproducible). By default the API returns need-runtime-match; pass
    enable_runtime_match=True to let it byte-compare against cuBLAS and reconstruct (or raise
    CublasUnsupportedShape for the vertical family)."""
    M, N, K = 64, 64, 65536
    a = (torch.randn(M, K, device=DEVICE) * 0.2).to(torch.float8_e4m3fn)
    b = (torch.randn(N, K, device=DEVICE) * 0.2).to(torch.float8_e4m3fn).t()

    try:                                          # static mode (default): heuristic only, no cuBLAS GEMM run
        cublas_equivalent_scaled_mm(a, b, enable_runtime_match=False)
        print(f"fp8  split-K  {M}x{N}x{K}: statically reconstructed")
    except CublasNeedRuntimeMatch:
        print(f"fp8  split-K  {M}x{N}x{K}: needs a runtime match (static mode declined)")

    try:                                          # runtime mode: allow a one-time byte-compare against cuBLAS
        out = cublas_equivalent_scaled_mm(a, b, enable_runtime_match=True)
        print(f"fp8  split-K  {M}x{N}x{K}: with enable_runtime_match=True -> "
              f"bit-identical = {_bit_equal(out, cublas_matmul(a, b))}")
    except CublasUnsupportedShape:
        out = cublas_matmul(a, b)                 # vertical split-K: fall back to cuBLAS
        print(f"fp8  split-K  {M}x{N}x{K}: UNSUPPORTED -> fell back to cuBLAS")


def example_cannot_match():
    """Shapes where the cuBLAS heuristic's algo has NO bit-identical Triton reconstruction,
    even with enable_runtime_match=True -> CublasUnsupportedShape (never a wrong result).

    Two families cover them:
      1. fp16 non-aligned (odd N or odd K): cuBLAS switches to a CUTLASS kernel that either
         uses an MMA with K=8 (`s1688`) or an odd-K reduction tail. Triton's `tl.dot` emits
         K=16, so the base MMA groups/rounds the products differently.
      2. a split-K residual (~1.1% of aligned fp16 split-K, ~0.4% of fp8) where K is not a
         whole number of k-tiles, so the last slice is a partial one: no K partition
         reproduces these -- a wide sweep of thousands of alternative partitions per shape
         found none. Strongly associated with K%64 != 0, but not decided by it, so it can
         only be caught by the runtime byte-compare."""
    fp16_nonaligned = [(64, 64, 257), (16, 65, 8192), (64, 129, 8192), (512, 513, 4096), (64, 64, 4097),
                       (128, 127, 16384), (33, 33, 15000)]
    fp16_partial_last_slice = [(96, 48, 175248), (64, 64, 116864), (64, 24, 116864), (64, 64, 219648)]

    print("cannot-match (heuristic gives an algo, but no Triton reconstruction is bit-identical):")
    print("  -- fp16 non-aligned -> CUTLASS s1688 (MMA K=8) / odd-K tail --")
    for (M, N, K) in fp16_nonaligned:
        a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
        b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)
        try:
            cublas_equivalent_gemm(a, b, enable_runtime_match=True)
            print(f"  fp16 {M}x{N}x{K}: reconstructed (unexpected)")
        except CublasUnsupportedShape:
            print(f"  fp16 {M}x{N}x{K}: UNSUPPORTED -> caller falls back to cuBLAS")

    print("  -- aligned fp16 split-K, partial last k-slice -> no partition reproduces it --")
    for (M, N, K) in fp16_partial_last_slice:
        a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
        b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)
        try:
            cublas_equivalent_gemm(a, b, enable_runtime_match=True)
            print(f"  fp16 {M}x{N}x{K}: reconstructed (unexpected)")
        except CublasUnsupportedShape:
            print(f"  fp16 {M}x{N}x{K}: UNSUPPORTED -> caller falls back to cuBLAS")


def main():
    if not torch.cuda.is_available():
        print("no CUDA GPU; this example needs one.")
        return
    print(f"device: {torch.cuda.get_device_name()}\n")
    example_fp16_plain()
    example_fp16_split_k()
    example_fp8_plain()
    example_fp8_split_k_runtime()
    print()
    example_cannot_match()


if __name__ == "__main__":
    main()
