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
    cublas_matmul,
    set_cublaslt,
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

    out = cublas_equivalent_gemm(a, b, scale_a=1.0, scale_b=1.0, out_dtype=torch.float16,
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
        cublas_equivalent_gemm(a, b, enable_runtime_match=False)
        print(f"fp8  split-K  {M}x{N}x{K}: statically reconstructed")
    except CublasNeedRuntimeMatch:
        print(f"fp8  split-K  {M}x{N}x{K}: needs a runtime match (static mode declined)")

    try:                                          # runtime mode: allow a one-time byte-compare against cuBLAS
        out = cublas_equivalent_gemm(a, b, enable_runtime_match=True)
        print(f"fp8  split-K  {M}x{N}x{K}: with enable_runtime_match=True -> "
              f"bit-identical = {_bit_equal(out, cublas_matmul(a, b))}")
    except CublasUnsupportedShape:
        out = cublas_matmul(a, b)                 # vertical split-K: fall back to cuBLAS
        print(f"fp8  split-K  {M}x{N}x{K}: UNSUPPORTED -> fell back to cuBLAS")


def example_one_api_two_dtypes():
    """One entry point for both dtypes; `b`'s required layout is the only thing that differs."""
    print("-- one API, fp16 and fp8 --")
    M, N, K = 1024, 1024, 2048

    a16 = torch.randn(M, K, device=DEVICE, dtype=torch.float16)     # [M,K] row-major
    b16 = torch.randn(K, N, device=DEVICE, dtype=torch.float16)     # [K,N] row-major
    out = cublas_equivalent_gemm(a16, b16)
    print(f"  fp16 {M}x{N}x{K}: {'BIT-IDENTICAL' if _bit_equal(out, cublas_matmul(a16, b16)) else 'MISMATCH'}")

    a8 = (torch.randn(M, K, device=DEVICE) / 4).to(torch.float8_e4m3fn)              # [M,K] row-major
    b8 = (torch.randn(N, K, device=DEVICE) / 4).to(torch.float8_e4m3fn).t()          # [K,N] column-major
    # Awkward scales on purpose: a power of two is exact either way and proves nothing.
    for sa, sb in ((1.0, 1.0), (1.3, 0.017)):
        out = cublas_equivalent_gemm(a8, b8, scale_a=sa, scale_b=sb)
        ref = cublas_matmul(a8, b8, torch.float16, scale_a=sa, scale_b=sb)
        print(f"  fp8  {M}x{N}x{K} scales {sa}, {sb}: "
              f"{'BIT-IDENTICAL' if _bit_equal(out, ref) else 'MISMATCH'}")

    # cuBLAS is told one layout per dtype, so the wrong one used to return a quietly
    # non-matching result. It now raises.
    try:
        cublas_equivalent_gemm(a8, (torch.randn(K, N, device=DEVICE) / 4).to(torch.float8_e4m3fn))
    except ValueError as e:
        print(f"  fp8 with a row-major b: refused -- {str(e).split('.')[0]}")


def example_choose_cublas_version():
    """Which cuBLAS we are bit-identical to is a choice, not whatever loaded first.

    A box usually carries several: here CUDA 13.0 (13.1.1) and CUDA 12.8 (12.8.5, the one torch
    is built against). Pass a version prefix or a full path; libraries are cached, so switching
    back and forth costs nothing after the first use of each."""
    print("-- choosing which cuBLAS to match --")
    M, N, K = 2048, 2048, 4096
    a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
    b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)

    for spec in ("12.8", "13.1"):
        out = cublas_equivalent_gemm(a, b, cublaslt=spec)          # per call
        ref = cublas_matmul(a, b, cublaslt=spec)
        print(f"  cuBLAS {spec}: {'BIT-IDENTICAL' if _bit_equal(out, ref) else 'MISMATCH'}")

    set_cublaslt("12.8")                                           # or for the whole process
    out = cublas_equivalent_gemm(a, b)
    print(f"  set_cublaslt(\"12.8\"): {'BIT-IDENTICAL' if _bit_equal(out, cublas_matmul(a, b)) else 'MISMATCH'}")
    set_cublaslt()                                                 # back to the newest installed


def example_cannot_match():
    """Shapes the heuristic answers but no Triton reconstruction matches, so the API declines.

    All of them are matrix-times-vector: M == 1 or N == 1. cuBLAS then leaves its tensor-core
    kernels entirely and runs a SIMT fallback -- `gemv2T_kernel`, `dot_kernel`,
    `reduce_1Block_kernel` -- which is CUDA-core FFMA with a shuffle reduction, an accumulation
    order none of the reconstructions here has. The caller gets an exception, not wrong bits."""
    print("cannot-match (cuBLAS runs a SIMT gemv, not a tensor-core kernel):")
    for M, N, K in [(1, 246, 342349), (94, 1, 175811), (170, 1, 170157), (251, 1, 38346),
                    (21, 1, 65878), (1, 23, 32259), (1, 104, 151171), (161, 1, 391706)]:
        a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
        b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)
        try:
            cublas_equivalent_gemm(a, b, enable_runtime_match=True)
            print(f"  fp16 {M}x{N}x{K}: reconstructed (unexpected)")
        except CublasUnsupportedShape:
            print(f"  fp16 {M}x{N}x{K}: declined -> fall back to cublas_matmul")
        except CublasNeedRuntimeMatch:
            print(f"  fp16 {M}x{N}x{K}: needs a runtime match")
        del a, b

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
    example_one_api_two_dtypes()
    print()
    example_choose_cublas_version()
    print()
    example_cannot_match()


if __name__ == "__main__":
    main()
