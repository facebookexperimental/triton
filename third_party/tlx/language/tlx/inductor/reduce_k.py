"""
Split-K reduction kernel for TLX Blackwell GEMM templates.

When SPLIT_K > 1, the main GEMM kernel writes fp32 partial results to a
workspace of shape (SPLIT_K * M, N).  This module provides:
  - _reduce_k_kernel:  Triton JIT kernel that sums the partials and writes
                        the final output in the target dtype.
  - emit_reduce_k_call: helper that emits wrapper code to launch the
                        reduction kernel after the main GEMM.

Ported from upstream:
  third-party/triton/beta/triton/third_party/tlx/tutorials/blackwell_gemm_ws.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import triton
import triton.language as tl

if TYPE_CHECKING:
    from torch._inductor.codegen.wrapper import WrapperCodeGen


@triton.jit
def _reduce_k_kernel(
    workspace_ptr,
    c_ptr,
    M,
    N,
    SPLIT_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    base_offs = offs_m[:, None] * N + offs_n[None, :]

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for s in range(SPLIT_K):
        ws_offs = base_offs + s * M * N
        partial = tl.load(workspace_ptr + ws_offs, mask=mask, other=0.0)
        acc += partial.to(tl.float32)

    tl.store(c_ptr + base_offs, acc.to(OUTPUT_DTYPE), mask=mask)


def emit_reduce_k_call(
    wrapper: "WrapperCodeGen",
    ws_name: str,
    output_name: str,
    M_expr: str,
    N_expr: str,
    split_k: int,
    output_triton_dtype: str,
) -> None:
    """Emit wrapper code to launch the split-K reduction kernel.

    Args:
        wrapper: The WrapperCodeGen instance to write lines to.
        ws_name: Variable name of the fp32 workspace buffer in wrapper code.
        output_name: Variable name of the output buffer in wrapper code.
        M_expr: Expression string for the M dimension.
        N_expr: Expression string for the N dimension.
        split_k: The SPLIT_K value (compile-time constant).
        output_triton_dtype: Triton dtype string for the output (e.g. "tl.float16").
    """
    wrapper.writeline(
        "from triton.language.extra.tlx.inductor.reduce_k import _reduce_k_kernel"
    )
    wrapper.writeline("import triton")
    wrapper.writeline("import triton.language as tl")
    wrapper.writeline(
        f"_reduce_k_kernel[(triton.cdiv({M_expr}, 32), triton.cdiv({N_expr}, 32))]"
        f"({ws_name}, {output_name}, {M_expr}, {N_expr},"
        f" SPLIT_K={split_k}, BLOCK_SIZE_M=32, BLOCK_SIZE_N=32,"
        f" OUTPUT_DTYPE={output_triton_dtype})"
    )
