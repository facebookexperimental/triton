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

import logging
import textwrap
from typing import Any, TYPE_CHECKING

import sympy
import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from torch._inductor.codegen.wrapper import WrapperCodeGen

log: logging.Logger = logging.getLogger(__name__)


@triton.jit
def _reduce_k_kernel(
    workspace_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    SPLIT_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
    HAS_BIAS: tl.constexpr = False,
    STRIDE_BIAS_M: tl.constexpr = 0,
    STRIDE_BIAS_N: tl.constexpr = 1,
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

    # addmm bias: applied here because the split-K main kernel bypasses store_output
    # (and its bias epilogue). STRIDE_BIAS_M=0 broadcasts a 1D [N] bias over M.
    if HAS_BIAS:
        bias_offs = offs_m[:, None] * STRIDE_BIAS_M + offs_n[None, :] * STRIDE_BIAS_N
        acc += tl.load(bias_ptr + bias_offs, mask=mask, other=0.0).to(tl.float32)

    tl.store(c_ptr + base_offs, acc.to(OUTPUT_DTYPE), mask=mask)


_TRITON_DTYPE = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}

# Above this the partial-results workspace is too big to allocate just for a
# measurement; fall back to the bandwidth model.
_MEASURE_WS_BYTES_LIMIT = 2 * 1024**3

_reduce_k_cost_cache: dict[tuple, float] = {}


def _modelled_cost_ms(M: int, N: int, split_k: int, itemsize: int) -> float:
    """Bandwidth model for the reducer: it is purely streaming."""
    read = split_k * M * N * 4  # fp32 partials
    write = M * N * itemsize
    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        # memory_clock_rate is kHz, bus width is bits; fall back to a gfx950-ish
        # 5 TB/s if the attributes are unavailable.
        bw = 2 * props.memory_clock_rate * 1e3 * props.memory_bus_width / 8
    except Exception:
        bw = 0.0
    if bw <= 0:
        bw = 5.0e12
    # ~70% of peak is what a pure streaming kernel achieves in practice.
    return (read + write) / (0.7 * bw) * 1e3


def reduce_k_cost_ms(
    M: int, N: int, split_k: int, out_dtype: torch.dtype, has_bias: bool
) -> float:
    """Measured wall time of the split-K reducer for this output shape, in ms.

    Autotune scores a template candidate on the GEMM kernel alone. A SPLIT_K > 1
    candidate additionally needs `_reduce_k_kernel` to run before its output is
    valid, so its true cost is main + reducer. Selection compares it against
    SPLIT_K=1 candidates that carry no such tail, which is why split-K used to be
    picked while being net-slower e2e. This returns the missing term.

    Cached on (M, N, SPLIT_K, dtype, has_bias) so the many tile configs that share
    one SPLIT_K pay a single measurement.
    """
    if split_k <= 1:
        return 0.0
    key = (M, N, split_k, out_dtype, has_bias)
    hit = _reduce_k_cost_cache.get(key)
    if hit is not None:
        return hit

    itemsize = torch.empty((), dtype=out_dtype).element_size()
    ws_bytes = split_k * M * N * 4
    cost = None
    if ws_bytes <= _MEASURE_WS_BYTES_LIMIT and out_dtype in _TRITON_DTYPE:
        try:
            from torch._inductor.runtime.benchmarking import benchmarker

            dev = torch.cuda.current_device()
            ws = torch.empty(split_k * M * N, dtype=torch.float32, device=dev)
            c = torch.empty((M, N), dtype=out_dtype, device=dev)
            bias = torch.empty(N, dtype=out_dtype, device=dev) if has_bias else c
            grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))

            def run() -> None:
                _reduce_k_kernel[grid](
                    ws, c, bias, M, N,
                    SPLIT_K=split_k,
                    BLOCK_SIZE_M=32,
                    BLOCK_SIZE_N=32,
                    OUTPUT_DTYPE=_TRITON_DTYPE[out_dtype],
                    HAS_BIAS=has_bias,
                    STRIDE_BIAS_M=0,
                    STRIDE_BIAS_N=1,
                )

            cost = benchmarker.benchmark_gpu(run)
            del ws, c, bias
        except Exception as e:  # OOM, compile failure, API drift
            log.warning(
                "split-K reducer cost measurement failed for M=%s N=%s SPLIT_K=%s "
                "(%s); falling back to the bandwidth model",
                M, N, split_k, e,
            )
            cost = None
    if cost is None:
        cost = _modelled_cost_ms(M, N, split_k, itemsize)

    _reduce_k_cost_cache[key] = cost
    return cost


def emit_reduce_k_call(
    wrapper: "WrapperCodeGen",
    ws_name: str,
    output_name: str,
    M_expr: str,
    N_expr: str,
    split_k: int,
    output_triton_dtype: str,
    bias_name: str | None = None,
    stride_bias_m: int = 0,
    stride_bias_n: int = 1,
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
        bias_name: Variable name of the addmm bias buffer, or None for plain mm.
            When set, _reduce_k_kernel adds it to the reduced result.
        stride_bias_m: Bias row stride (0 to broadcast a 1D [N] bias over M).
        stride_bias_n: Bias column stride.
    """
    wrapper.writeline(
        "from triton.language.extra.tlx.inductor.reduce_k import _reduce_k_kernel"
    )
    wrapper.writeline("import triton")
    wrapper.writeline("import triton.language as tl")
    # No bias (plain mm): pass the output buffer as a dummy ptr; HAS_BIAS=False
    # means it is never dereferenced.
    has_bias = bias_name is not None
    bias_ptr = bias_name if has_bias else output_name
    wrapper.writeline(
        f"_reduce_k_kernel[(triton.cdiv({M_expr}, 32), triton.cdiv({N_expr}, 32))]"
        f"({ws_name}, {output_name}, {bias_ptr}, {M_expr}, {N_expr},"
        f" SPLIT_K={split_k}, BLOCK_SIZE_M=32, BLOCK_SIZE_N=32,"
        f" OUTPUT_DTYPE={output_triton_dtype},"
        f" HAS_BIAS={has_bias}, STRIDE_BIAS_M={stride_bias_m},"
        f" STRIDE_BIAS_N={stride_bias_n})"
    )


def _reduce_k_body_source(
    workspace_name: str,
    output_name: str,
    M_expr: str,
    N_expr: str,
    split_k: int,
    epilogue_code: str,
    bias_name: str | None = None,
    stride_bias_m: int = 0,
    stride_bias_n: int = 1,
    index_dtype: str = "tl.int32",
) -> str:
    # The spliced fused epilogue references INDEX_DTYPE (Inductor's per-kernel index
    # dtype, used for dynamic-shape bounds like `tl.full([], ks0, dtype=INDEX_DTYPE)`).
    # The main template kernel defines it, but this generated reducer is a standalone
    # kernel, so define it here from the template kernel's own index_dtype.
    code = textwrap.dedent(f"""
    BLOCK_SIZE_M: tl.constexpr = 32
    BLOCK_SIZE_N: tl.constexpr = 32
    INDEX_DTYPE: tl.constexpr = {index_dtype}
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_m[:, None] < {M_expr}) & (offs_n[None, :] < {N_expr})
    base_offs = offs_m[:, None] * {N_expr} + offs_n[None, :]
    xnumel = {M_expr} * {N_expr}
    xindex = base_offs
    xmask = mask
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for s in range({split_k}):
        ws_offs = base_offs + s * {M_expr} * {N_expr}
        partial = tl.load({workspace_name} + ws_offs, mask=mask, other=0.0)
        acc += partial.to(tl.float32)
    """)
    if bias_name is not None:
        code += textwrap.dedent(f"""
        bias_offs = offs_m[:, None] * {stride_bias_m} + offs_n[None, :] * {stride_bias_n}
        acc += tl.load({bias_name} + bias_offs, mask=mask, other=0.0).to(tl.float32)
        """)
    epilogue_lines = [
        line.lstrip() for line in textwrap.dedent(epilogue_code).splitlines()
    ]
    epilogue_body = "\n".join(line for line in epilogue_lines if line)
    if "fused_result" not in epilogue_body:
        raise AssertionError("reduce-k epilogue must define fused_result")
    code += "\n" + epilogue_body
    code += f"\ntl.store({output_name} + base_offs, fused_result, mask=mask)\n"
    return code


def emit_aoti_reduce_k_call(
    wrapper: "WrapperCodeGen",
    workspace_arg: Any,
    output_node: Any,
    bias_node: Any | None,
    M: Any,
    N: Any,
    split_k: int,
    output_triton_dtype: str,
    M_kernel_expr: str | None = None,
    N_kernel_expr: str | None = None,
    stride_bias_m: int = 0,
    stride_bias_n: int = 1,
    template_kernel: Any | None = None,
    main_kernel_name: str | None = None,
    epilogue_code: str | None = None,
    final_output_ptr: str | None = None,
    bias_kernel_ptr: str | None = None,
) -> None:
    """Register and launch the split-K reducer through the AOTI C++ wrapper."""
    from torch._inductor import ir
    from torch._inductor.runtime.triton_heuristics import FixedGrid
    from torch._inductor.utils import IndentedBuffer
    from torch._inductor.virtualized import V
    from torch.utils._sympy.functions import CeilDiv

    # Single template-side reducer path: with no fused epilogue, use an identity so the
    # generated reducer still does sum + bias + store. This keeps every split-K reducer
    # on the code-generated define_kernel path (never define_user_defined_triton_kernel),
    # so no WorkspaceArg special-casing is needed in the Inductor wrapper.
    if epilogue_code is None:
        epilogue_code = "fused_result = acc"
    if epilogue_code is not None:
        if template_kernel is None or main_kernel_name is None or final_output_ptr is None:
            raise AssertionError("fused reduce-k requires template kernel metadata")
        from torch._inductor.select_algorithm import Placeholder

        argdefs, call_args, _, arg_types = template_kernel.args.python_argdefs()
        reduce_name = f"{main_kernel_name}_reduce_k"
        source = IndentedBuffer()
        source.splice(template_kernel.gen_common_triton_imports())
        source.splice(template_kernel.jit_lines())
        source.writeline(
            f"def {reduce_name}({', '.join(arg.full_name() for arg in argdefs)}):"
        )
        with source.indent():
            source.splice(_reduce_k_body_source(
                workspace_arg.inner_name,
                final_output_ptr,
                M_kernel_expr or str(M),
                N_kernel_expr or str(N),
                split_k,
                epilogue_code,
                bias_kernel_ptr,
                stride_bias_m,
                stride_bias_n,
                index_dtype=getattr(template_kernel, "index_dtype", "tl.int32"),
            ))
        source_code = source.getvalue().replace(
            str(Placeholder.DESCRIPTIVE_NAME), reduce_name
        ).replace(str(Placeholder.KERNEL_NAME), reduce_name)
        compile_wrapper = IndentedBuffer()
        compile_wrapper.writeline(f"async_compile.triton({reduce_name!r}, '''")
        compile_wrapper.splice(source_code, strip=True)
        device = V.graph.get_current_device_or_throw()
        compile_wrapper.writeline(f"''', device_str='{device.type}')")
        kernel_body = compile_wrapper.getvalue()
        wrapper.src_to_kernel[kernel_body] = reduce_name
        wrapper.define_kernel(reduce_name, kernel_body, "# TLX split-K fused reducer")
        grid_args = [CeilDiv(M, 32), CeilDiv(N, 32), sympy.Integer(1)]
        wrapper.generate_kernel_call(
            reduce_name,
            [*call_args, *grid_args],
            arg_types=[*arg_types, *map(type, grid_args)],
            triton_meta=template_kernel.triton_meta,
            inductor_meta=FixedGrid.setup_grid_as_args(),
            triton=True,
            device=output_node.get_device(),
        )
        return
