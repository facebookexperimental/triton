"""Reduction-aware TorchTLX template epilogues.

A GEMM template materializes its output and emits per-row partial statistics
for every row reduction in the fused epilogue. A generated consumer kernel
combines the partials and evaluates the epilogue's final pointwise expression.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os
from collections.abc import Sequence
from typing import Any

import sympy
import torch
from torch._inductor import config
from torch._inductor.codegen.common import WorkspaceArg, WorkspaceZeroMode
from torch._inductor.codegen.simd import CantSplit
from torch._inductor.codegen.triton import triton_compute_type, triton_type
from torch._inductor.ir import Buffer, ComputedBuffer, MultiTemplateBuffer, Pointwise, Reduction
from torch._inductor.kernel.loop_ir_epilogue_lowering import (
    GemmEpilogueIRAnalysis,
    GemmEpilogueIRExpression,
)
from torch._inductor.scheduler import BaseSchedulerNode
from torch._inductor.select_algorithm import TritonTemplateKernel
from torch._inductor.virtualized import V
from torch.utils._sympy.functions import CeilDiv

log: logging.Logger = torch._logging.getArtifactLogger(__name__, "fusion")

_SUPPORTED_REDUCTIONS = ("sum", "welford_reduce")
# welford_reduce outputs (mean, m2, weight); the workspace keeps mean and m2.
_STAT_SLOTS = {"sum": 1, "welford_reduce": 2}
_BINARY_OPS = {
    "add": "({} + {})",
    "sub": "({} - {})",
    "mul": "({} * {})",
    "truediv": "({} / {})",
    "maximum": "tl.maximum({}, {})",
    "minimum": "tl.minimum({}, {})",
}
_UNARY_OPS = {
    "neg": "(-{})",
    "reciprocal": "(1.0 / {})",
    "square": "({0} * {0})",
    "rsqrt": "tl.rsqrt({})",
    "sqrt": "tl.sqrt({})",
    "exp": "tl.exp({})",
    "log": "tl.log({})",
    "abs": "tl.abs({})",
    "sigmoid": "tl.sigmoid({})",
    "relu": "tl.maximum({}, 0.0)",
}
_CONSUMER_MAX_BLOCK_N = 4096
_CONSUMER_MAX_BLOCK_STATS = 1024
_INSTALLED = False


class RowPartialsUnsupported(CantSplit):
    """The selected template cannot emit row partials for a planned epilogue.

    Subclasses CantSplit so fusion benchmarking skips the candidate.
    """

    def __init__(self, message: str) -> None:
        super().__init__(None, None)
        self.message = message

    def __str__(self) -> str:
        return self.message


@dataclasses.dataclass(frozen=True)
class RowReduction:
    reduction_type: str
    source: GemmEpilogueIRExpression


@dataclasses.dataclass(frozen=True)
class TlxReductionEpiloguePlan:
    """GEMM materialization followed by row reductions and a pointwise output."""

    raw_output: str
    final_output: str
    final_value: GemmEpilogueIRExpression
    reductions: tuple[RowReduction, ...]
    columns: int


@dataclasses.dataclass(frozen=True)
class _Context:
    row: sympy.Expr
    column: sympy.Expr | None


class _Reject(Exception):
    pass


def _buffer_context(buffer: ComputedBuffer, rows: sympy.Expr, columns: int) -> _Context:
    sizevars = V.graph.sizevars
    size = buffer.get_size()
    args = buffer.data.inner_fn_args()
    if not sizevars.statically_known_equals(size[0], rows):
        raise _Reject("row count differs from the GEMM")
    trailing_one = all(sizevars.statically_known_equals(s, 1) for s in size[1:])
    if isinstance(buffer.data, Reduction):
        ranges = buffer.data.get_reduction_size()
        if len(ranges) != 1 or not sizevars.statically_known_equals(ranges[0], columns):
            raise _Reject("reduction is not over all GEMM columns")
        if not trailing_one or buffer.data.reduction_type not in _SUPPORTED_REDUCTIONS:
            raise _Reject("unsupported reduction buffer")
        return _Context(args[0][0], args[1][0])
    if not isinstance(buffer.data, Pointwise):
        raise _Reject("unsupported epilogue node")
    if len(size) == 2 and sizevars.statically_known_equals(size[1], columns):
        return _Context(args[0][0], args[0][1])
    if trailing_one:
        return _Context(args[0][0], None)
    raise _Reject("unsupported epilogue buffer shape")


def _is_row_buffer(buffer: Buffer) -> bool:
    return all(V.graph.sizevars.statically_known_equals(s, 1) for s in buffer.get_size()[1:])


def _is_row_major(buffer: Buffer, columns: int) -> bool:
    size, stride = buffer.get_size(), buffer.get_stride()
    sizevars = V.graph.sizevars
    return (
        len(size) == 2
        and sizevars.statically_known_equals(size[1], columns)
        and sizevars.statically_known_equals(stride[1], 1)
        and sizevars.statically_known_equals(stride[0], columns)
    )


class _Validator:
    """Check that an epilogue is GEMM tile + [N] vectors + full-row reductions."""

    def __init__(self, gemm: Buffer, buffers: dict[str, ComputedBuffer], contexts: dict[str, _Context], columns: int) -> None:
        self.gemm = gemm
        self.buffers = buffers
        self.contexts = contexts
        self.columns = columns
        self.reductions: dict[RowReduction, None] = {}
        self.memo: dict[tuple[int, int, bool], bool] = {}

    def _check_index(self, index: sympy.Expr, expected: sympy.Expr) -> None:
        if not V.graph.sizevars.statically_known_equals(index - expected, 0):
            raise _Reject("index is not the row-major tile index")

    def visit(self, expr: object, ctx: _Context, *, in_reduction: bool) -> bool:
        """Return whether the subtree reads the GEMM tile."""
        key = (id(expr), id(ctx), in_reduction)
        if key not in self.memo:
            self.memo[key] = self._visit(expr, ctx, in_reduction)
        return self.memo[key]

    def _visit(self, expr: object, ctx: _Context, in_reduction: bool) -> bool:
        if not isinstance(expr, GemmEpilogueIRExpression):
            if isinstance(expr, (tuple, list)):
                raise _Reject("unexpected tuple operand")
            return False
        op, args = expr.op, expr.args
        if op == "load":
            name, index, stored = args
            if name == self.gemm.get_name():
                stride = self.gemm.get_stride()
                if ctx.column is None:
                    raise _Reject("GEMM tile read from a row context")
                self._check_index(index, ctx.row * stride[0] + ctx.column * stride[1])
                return True
            if name in self.buffers:
                inner = self.contexts[name]
                stride = self.buffers[name].get_stride()
                if _is_row_buffer(self.buffers[name]):
                    if in_reduction:
                        raise _Reject("row statistic read inside a reduction")
                    self._check_index(index, ctx.row * stride[0])
                else:
                    if ctx.column is None:
                        raise _Reject("tile read from a row context")
                    self._check_index(index, ctx.row * stride[0] + ctx.column * stride[1])
                return self.visit(stored, inner, in_reduction=in_reduction)
            buffer = V.graph.try_get_buffer(name)
            if buffer is None or ctx.column is None:
                raise _Reject("unsupported external load")
            size, stride = buffer.get_size(), buffer.get_stride()
            if len(size) != 1 or not V.graph.sizevars.statically_known_equals(size[0], self.columns):
                raise _Reject("external input is not an [N] vector")
            self._check_index(index, ctx.column * stride[0])
            return False
        if op == "reduction":
            if in_reduction:
                raise _Reject("nested reduction")
            reduction_type, source = args[2], args[3]
            if reduction_type not in _SUPPORTED_REDUCTIONS:
                raise _Reject("unsupported reduction type")
            if reduction_type == "welford_reduce" and args[4] not in (0, 1, 2):
                raise _Reject("unexpected welford output")
            if not self.visit(source, ctx, in_reduction=True):
                raise _Reject("reduction source does not read the GEMM tile")
            self.reductions[RowReduction(reduction_type, source)] = None
            return False
        if op == "constant":
            if isinstance(args[0], bool) or not math.isfinite(float(args[0])):
                raise _Reject("non-finite or boolean constant")
            return False
        if op == "index_expr":
            if not isinstance(args[0], (int, sympy.Integer)):
                raise _Reject("non-constant index_expr")
            return False
        if op == "to_dtype":
            if not args[1].is_floating_point:
                raise _Reject("non-float cast")
            return self.visit(args[0], ctx, in_reduction=in_reduction)
        if op in _BINARY_OPS:
            left = self.visit(args[0], ctx, in_reduction=in_reduction)
            right = self.visit(args[1], ctx, in_reduction=in_reduction)
            return left or right
        if op in _UNARY_OPS:
            return self.visit(args[0], ctx, in_reduction=in_reduction)
        raise _Reject(f"unsupported op {op}")


def _external_users(name: str, group: set[str]) -> bool:
    buf = V.graph.scheduler.name_to_buf.get(name)
    if buf is None:
        return True
    return any(user.get_name() not in group for user in buf.users)


def plan_reduction_epilogue(
    gemm: Buffer,
    nodes: Sequence[BaseSchedulerNode],
) -> TlxReductionEpiloguePlan | None:
    """Recognize GEMM -> row reductions -> pointwise output from scheduler Loop IR."""
    if os.environ.get("TORCHINDUCTOR_TLX_REDUCTION_EPILOGUE", "1") == "0":
        return None
    if config.triton.tlx_mode not in ("allow", "force") or not isinstance(gemm, Buffer):
        return None
    if len(gemm.get_size()) != 2:
        return None
    rows, columns = gemm.get_size()
    if not isinstance(columns, (int, sympy.Integer)) or not _is_row_major(gemm, int(columns)):
        return None
    columns = int(columns)
    children = [child for node in nodes for child in node.get_nodes()]
    buffers = [child.node for child in children]
    if not buffers or not all(isinstance(buffer, ComputedBuffer) for buffer in buffers):
        return None
    group = {child.get_name() for child in children}
    if _external_users(gemm.get_name(), group):
        return None
    try:
        contexts = {buffer.get_name(): _buffer_context(buffer, rows, columns) for buffer in buffers}
        outputs = [buffer for buffer in buffers if _external_users(buffer.get_name(), group)]
        if len(outputs) != 1:
            raise _Reject("epilogue must have exactly one externally used output")
        (final,) = outputs
        if contexts[final.get_name()].column is None or isinstance(final.data, Reduction):
            raise _Reject("externally used output is not a full tile")
        if not _is_row_major(final, columns):
            raise _Reject("output is not row-major contiguous")
        analysis = GemmEpilogueIRAnalysis.from_buffers(buffers)
        store = analysis.store(final.get_name())
        if store is None:
            raise _Reject("output store not captured")
        ctx = contexts[final.get_name()]
        stride = final.get_stride()
        if not V.graph.sizevars.statically_known_equals(store.index - (ctx.row * stride[0] + ctx.column * stride[1]), 0):
            raise _Reject("output store index is not row-major")
        validator = _Validator(gemm, {buffer.get_name(): buffer for buffer in buffers}, contexts, columns)
        validator.visit(store.value, ctx, in_reduction=False)
    except _Reject as e:
        log.debug("reduction epilogue rejected: %r", e)
        return None
    if not validator.reductions:
        return None
    return TlxReductionEpiloguePlan(
        raw_output=gemm.get_name(),
        final_output=final.get_name(),
        final_value=store.value,
        reductions=tuple(validator.reductions),
        columns=columns,
    )


def _template_supports_row_partials(template: object) -> bool:
    if isinstance(template, MultiTemplateBuffer):
        caller = getattr(template, "_render_caller", None)
        if caller is not None:
            return getattr(caller, "_tlx_row_partials", False)
        return any(getattr(choice, "_tlx_row_partials", False) for choice in template.choices)
    return False


def can_fuse_reduction_epilogue(
    node1: BaseSchedulerNode,
    node2: BaseSchedulerNode,
) -> bool:
    template = node1.get_template_node()
    if not isinstance(template, Buffer) or not _template_supports_row_partials(template):
        return False
    epilogue_nodes = [node for node in node1.get_nodes() if node.get_template_node() is None]
    return plan_reduction_epilogue(template, (*epilogue_nodes, node2)) is not None


class _Emitter:
    """Emit SSA Triton lines for a validated epilogue expression."""

    def __init__(self, prefix: str, bind_load, bind_reduction) -> None:  # type: ignore[no-untyped-def]
        self.prefix = prefix
        self.bind_load = bind_load
        self.bind_reduction = bind_reduction
        self.lines: list[str] = []
        self.memo: dict[int, str] = {}

    def emit(self, expr: object) -> str:
        if not isinstance(expr, GemmEpilogueIRExpression):
            raise AssertionError(f"unexpected epilogue operand {expr!r}")
        if id(expr) in self.memo:
            return self.memo[id(expr)]
        op, args = expr.op, expr.args
        if op == "load":
            name, _, stored = args
            value = self.emit(stored) if stored is not None else self.bind_load(name)
        elif op == "reduction":
            value = self.bind_reduction(RowReduction(args[2], args[3]), args[4] if len(args) > 4 else 0)
        elif op in ("constant", "index_expr"):
            value = repr(float(args[0])) if args[1].is_floating_point else str(int(args[0]))
        elif op == "to_dtype":
            value = f"{self.emit(args[0])}.to({triton_compute_type(args[1])})"
        elif op in _BINARY_OPS:
            value = _BINARY_OPS[op].format(self.emit(args[0]), self.emit(args[1]))
        else:
            value = _UNARY_OPS[op].format(self.emit(args[0]))
        name = f"{self.prefix}{len(self.memo)}"
        self.lines.append(f"{name} = {value}")
        self.memo[id(expr)] = name
        return name


def _stat_slots(plan: TlxReductionEpiloguePlan) -> list[int]:
    slots, offset = [], 0
    for reduction in plan.reductions:
        slots.append(offset)
        offset += _STAT_SLOTS[reduction.reduction_type]
    return slots


def _workspace(kernel: TritonTemplateKernel, plan: TlxReductionEpiloguePlan, partial_columns: int) -> WorkspaceArg:
    existing = getattr(kernel, "_tlx_reduction_workspace", None)
    if existing is not None:
        if kernel._tlx_reduction_partial_columns != partial_columns:
            raise AssertionError("one reduction epilogue must use one partial width")
        return existing
    nblocks = CeilDiv(plan.columns, partial_columns)
    rows = kernel.output_node.get_size()[0]
    workspace = WorkspaceArg(
        count=sum(_STAT_SLOTS[r.reduction_type] for r in plan.reductions) * rows * nblocks,
        zero_mode=WorkspaceZeroMode.UNINITIALIZED,
        device=kernel.output_node.get_device(),
        outer_name=WorkspaceArg.unique_name("row_stats_ws_"),
        inner_name="row_stats_ws",
        dtype=torch.float32,
    )
    kernel.args.workspace_args.append(workspace)
    kernel._tlx_reduction_workspace = workspace
    kernel._tlx_reduction_partial_columns = partial_columns
    kernel._tlx_reduction_nblocks = int(nblocks)
    kernel._tlx_reduction_consumer = _consumer_body(kernel, plan, partial_columns)
    return workspace


def _vector_load(kernel: TritonTemplateKernel, name: str, columns: str, mask: str) -> str:
    stride = kernel.kexpr(kernel.rename_indexing(V.graph.get_buffer(name).get_stride()[0]))
    return f"tl.load({kernel.args.input(name)} + ({columns}) * ({stride}), mask={mask}, other=0.0).to(tl.float32)"


def _partial_stat_lines(
    kernel: TritonTemplateKernel,
    plan: TlxReductionEpiloguePlan,
    value: str,
    indices: tuple[str, str],
    mask: str,
    partial_index: str,
) -> list[str]:
    row_index, column_index = indices
    rows, columns = kernel.size(None, 0), kernel.size(None, 1)
    nblocks = kernel._tlx_reduction_nblocks
    out_dtype = triton_type(kernel.output_node.get_dtype())
    gemm = plan.raw_output
    column_mask = f"({column_index} < {columns})"

    def bind_load(name: str) -> str:
        return "tlx_raw" if name == gemm else _vector_load(kernel, name, column_index, column_mask)

    lines = [
        f"tlx_raw = {value}.to({out_dtype}).to(tl.float32)",
        f"tlx_rows = tl.min({row_index}, axis=1)",
        f"tlx_row_mask = tlx_rows < {rows}",
        f"tlx_stat_offset = tlx_rows * {nblocks} + ({partial_index})",
    ]
    for k, (reduction, slot) in enumerate(zip(plan.reductions, _stat_slots(plan))):
        emitter = _Emitter(f"tlx_src{k}_", bind_load, None)
        source = emitter.emit(reduction.source)
        lines += emitter.lines
        lines.append(f"tlx_sum{k} = tl.sum(tl.where({mask}, {source}, 0.0), axis=1)")
        ws = f"row_stats_ws + {slot} * {rows} * {nblocks} + tlx_stat_offset"
        if reduction.reduction_type == "sum":
            lines.append(f"tl.store({ws}, tlx_sum{k}, mask=tlx_row_mask)")
            continue
        lines += [
            f"tlx_count{k} = tl.sum({mask}.to(tl.float32), axis=1)",
            f"tlx_mean{k} = tlx_sum{k} / tl.maximum(tlx_count{k}, 1.0)",
            f"tlx_dev{k} = tl.where({mask}, {source} - tlx_mean{k}[:, None], 0.0)",
            f"tl.store({ws}, tlx_mean{k}, mask=tlx_row_mask)",
            f"tl.store({ws} + {rows} * {nblocks}, tl.sum(tlx_dev{k} * tlx_dev{k}, axis=1), mask=tlx_row_mask)",
        ]
    return lines


def _consumer_body(kernel: TritonTemplateKernel, plan: TlxReductionEpiloguePlan, partial_columns: int) -> str:
    """Combine row partials and evaluate the final epilogue expression per row."""
    n = plan.columns
    nblocks = kernel._tlx_reduction_nblocks
    block_n = min(_CONSUMER_MAX_BLOCK_N, 1 << (n - 1).bit_length())
    block_stats = min(_CONSUMER_MAX_BLOCK_STATS, 1 << max(nblocks - 1, 0).bit_length())
    rows = kernel.kexpr(kernel.rename_indexing(kernel.output_node.get_size()[0]))
    raw = kernel.args.output(plan.raw_output)
    final = kernel.args.output(plan.final_output)
    out_dtype = triton_type(V.graph.get_buffer(plan.final_output).get_dtype())
    lines = [
        f"INDEX_DTYPE: tl.constexpr = {getattr(kernel, 'index_dtype', 'tl.int32')}",
        "tlx_row = tl.program_id(0).to(tl.int64)",
        f"tlx_stats = tl.arange(0, {block_stats})",
    ]
    values: dict[tuple[RowReduction, int], str] = {}
    for k, (reduction, slot) in enumerate(zip(plan.reductions, _stat_slots(plan))):
        base = f"row_stats_ws + ({slot} * {rows} + tlx_row) * {nblocks}"
        loop = [
            f"for tlx_p0 in range(0, {nblocks}, {block_stats}):",
            "    tlx_p = tlx_p0 + tlx_stats",
            f"    tlx_pm = tlx_p < {nblocks}",
            f"    tlx_part = tl.load({base} + tlx_p, mask=tlx_pm, other=0.0)",
        ]
        if reduction.reduction_type == "sum":
            lines += [f"tlx_acc{k} = tl.zeros([{block_stats}], dtype=tl.float32)", *loop, f"    tlx_acc{k} += tlx_part", f"tlx_red{k}_0 = tl.sum(tlx_acc{k}, axis=0)"]
            values[(reduction, 0)] = f"tlx_red{k}_0"
            continue
        # Chan's parallel combine: partial counts are the valid columns per block.
        count = f"tl.where(tlx_pm, tl.minimum({partial_columns}, {n} - tlx_p * {partial_columns}), 0).to(tl.float32)"
        m2 = f"tl.load({base} + {rows} * {nblocks} + tlx_p, mask=tlx_pm, other=0.0)"
        lines += [f"tlx_acc{k} = tl.zeros([{block_stats}], dtype=tl.float32)", *loop, f"    tlx_acc{k} += {count} * tlx_part", f"tlx_red{k}_0 = tl.sum(tlx_acc{k}, axis=0) / {float(n)!r}"]
        lines += [f"tlx_acc{k} = tl.zeros([{block_stats}], dtype=tl.float32)", *loop, f"    tlx_dm = tlx_part - tlx_red{k}_0", f"    tlx_acc{k} += {m2} + {count} * tlx_dm * tlx_dm", f"tlx_red{k}_1 = tl.sum(tlx_acc{k}, axis=0)"]
        values[(reduction, 0)] = f"tlx_red{k}_0"
        values[(reduction, 1)] = f"tlx_red{k}_1"
        values[(reduction, 2)] = repr(float(n))

    def bind_load(name: str) -> str:
        if name == plan.raw_output:
            return f"tl.load({raw} + tlx_row * {n} + tlx_cols, mask=tlx_cm, other=0.0).to(tl.float32)"
        return _vector_load(kernel, name, "tlx_cols", "tlx_cm")

    emitter = _Emitter("tlx_v", bind_load, lambda reduction, index: values[(reduction, index)])
    result = emitter.emit(plan.final_value)
    lines += [
        f"for tlx_c0 in range(0, {n}, {block_n}):",
        f"    tlx_cols = tlx_c0 + tl.arange(0, {block_n})",
        f"    tlx_cm = tlx_cols < {n}",
        *(f"    {line}" for line in emitter.lines),
        f"    tl.store({final} + tlx_row * {n} + tlx_cols, {result}.to({out_dtype}), mask=tlx_cm)",
    ]
    return "\n".join(lines) + "\n"


def _resolve_columns(kernel: TritonTemplateKernel, expression: int | str) -> int:
    meta = {k: sympy.Integer(v) for k, v in kernel.meta.items() if isinstance(v, int)}
    try:
        return int(sympy.sympify(expression, locals=meta))
    except (TypeError, ValueError, sympy.SympifyError) as exc:
        raise RowPartialsUnsupported("partial width must resolve to an integer") from exc


def reduction_store_output(
    kernel: TritonTemplateKernel,
    indices: tuple[str, str],
    value: str,
    mask: str,
    *,
    partial_index: str,
    partial_columns: str,
    **store_kwargs: Any,
) -> str:
    """Store the template output and, under a plan, emit its row partials.

    Partial p must cover the contiguous columns [p * width, (p + 1) * width).
    `partial_index` is p, `partial_columns` is width, and `mask` excludes tail
    columns and rows. The consumer derives each partial's count from this layout.
    """
    plan = getattr(kernel, "_tlx_reduction_epilogue_plan", None)
    if plan is None:
        return kernel.store_output(indices, value, mask, **store_kwargs)
    if getattr(kernel, "_tlx_split_k", 1) != 1:
        raise RowPartialsUnsupported("row-partial epilogues require SPLIT_K=1")
    width = _resolve_columns(kernel, partial_columns)
    _workspace(kernel, plan, width)
    kernel._tlx_row_partials_emitted = True

    raw_name = plan.raw_output
    original_store = kernel.store
    store_was_overridden = "store" in vars(kernel)

    def store_with_partials(name, index, store_value, mode=None):  # type: ignore[no-untyped-def]
        result = original_store(name, index, store_value, mode)
        if name == raw_name:
            for line in _partial_stat_lines(kernel, plan, str(store_value), indices, mask, partial_index):
                kernel.compute.writeline(line)
        return result

    kernel.store = store_with_partials  # type: ignore[method-assign]
    try:
        return kernel.store_output(indices, value, mask, **store_kwargs)
    finally:
        if store_was_overridden:
            kernel.store = original_store  # type: ignore[method-assign]
        else:
            del kernel.store


def emit_post_kernel(kernel: TritonTemplateKernel, wrapper: Any, kernel_name: str) -> None:
    """Define and launch the generated row consumer next to the GEMM kernel."""
    plan = getattr(kernel, "_tlx_reduction_epilogue_plan", None)
    body = getattr(kernel, "_tlx_reduction_consumer", None)
    if plan is None or body is None:
        return
    from torch._inductor.runtime.triton_heuristics import FixedGrid
    from torch._inductor.select_algorithm import Placeholder
    from torch._inductor.utils import IndentedBuffer

    argdefs, call_args, _, arg_types = kernel.args.python_argdefs()
    name = f"{kernel_name}_row_consumer"
    source = IndentedBuffer()
    source.splice(kernel.gen_common_triton_imports())
    source.splice(kernel.jit_lines())
    source.writeline(f"def {name}({', '.join(arg.full_name() for arg in argdefs)}):")
    with source.indent():
        source.splice(body)
    source_code = (
        source.getvalue()
        .replace(str(Placeholder.DESCRIPTIVE_NAME), name)
        .replace(str(Placeholder.KERNEL_NAME), name)
    )
    compile_wrapper = IndentedBuffer()
    compile_wrapper.writeline(f"async_compile.triton({name!r}, '''")
    compile_wrapper.splice(source_code, strip=True)
    device = V.graph.get_current_device_or_throw()
    compile_wrapper.writeline(f"''', device_str='{device.type}')")
    kernel_body = compile_wrapper.getvalue()
    wrapper.src_to_kernel[kernel_body] = name
    wrapper.define_kernel(name, kernel_body, "# TorchTLX reduction epilogue consumer")
    grid_args = [kernel.output_node.get_size()[0], sympy.Integer(1), sympy.Integer(1)]
    wrapper.generate_kernel_call(
        name,
        [*call_args, *grid_args],
        arg_types=[*arg_types, *map(type, grid_args)],
        triton_meta=kernel.triton_meta,
        inductor_meta=FixedGrid.setup_grid_as_args(),
        triton=True,
        device=kernel.output_node.get_device(),
    )


def _selected_caller_supports_row_partials(template_node: BaseSchedulerNode) -> bool:
    buffer = template_node.node
    caller = getattr(buffer, "_render_caller", None)
    return caller is not None and getattr(caller, "_tlx_row_partials", False)


def install() -> None:
    """Install the Inductor hooks used by reduction-aware TLX templates."""
    global _INSTALLED
    if _INSTALLED:
        return
    from torch._inductor.codegen.cuda_combined_scheduling import CUDACombinedScheduling
    from torch._inductor.codegen.triton import TritonScheduling

    def install_can_fuse_hook(scheduling_cls: type[Any]) -> None:
        original_can_fuse = getattr(scheduling_cls, "can_fuse_reduction_epilogue", None)

        def patched_can_fuse(self, node1, node2):  # type: ignore[no-untyped-def]
            return can_fuse_reduction_epilogue(node1, node2) or (
                original_can_fuse is not None and original_can_fuse(self, node1, node2)
            )

        scheduling_cls.can_fuse_reduction_epilogue = patched_can_fuse

    # ROCm uses CUDACombinedScheduling as the device-level dispatcher.
    install_can_fuse_hook(CUDACombinedScheduling)
    install_can_fuse_hook(TritonScheduling)

    from torch._inductor.scheduler import FusionResult, Scheduler

    original_speedup_by_fusion = Scheduler.speedup_by_fusion

    def patched_speedup_by_fusion(self, node1, node2):  # type: ignore[no-untyped-def]
        if not can_fuse_reduction_epilogue(node1, node2):
            return original_speedup_by_fusion(self, node1, node2)
        if os.environ.get("TORCHINDUCTOR_TLX_REDUCTION_EPILOGUE") == "force":
            # Skip profitability; take the fastest unfused choice that can emit partials.
            template = node1.get_template_node()
            timings = template.choice_timings()
            choice = min(
                (c for c in timings if getattr(c, "_tlx_row_partials", False) and getattr(c, "_tlx_split_k", 1) == 1),
                key=timings.__getitem__,
                default=None,
            )
            if choice is None:
                return FusionResult.fuse(False)
            template.finalize_as_triton_caller(choice)
            return FusionResult.fuse(True)
        # Candidates without row-partial support raise RowPartialsUnsupported
        # while compiling and drop out of the fused benchmark.
        # Newer nightly builds use the epilogue benchmark switch for templates.
        benchmark_option = (
            "benchmark_template_fusion"
            if hasattr(config, "benchmark_template_fusion")
            else "benchmark_epilogue_fusion"
        )
        with config.patch({benchmark_option: True}):
            return original_speedup_by_fusion(self, node1, node2)

    Scheduler.speedup_by_fusion = patched_speedup_by_fusion

    original_codegen = TritonTemplateKernel.codegen_template_body

    def patched_codegen(  # type: ignore[no-untyped-def]
        self,
        scheduling,
        template_node,
        epilogue_nodes,
        prologue_nodes,
        buf_name_to_prologue_group,
        prologue_preserves_zero_mask_fn,
        render,
    ):
        plan = plan_reduction_epilogue(self.output_node, epilogue_nodes) if epilogue_nodes else None
        self._tlx_reduction_epilogue_plan = plan
        self._tlx_row_partials_emitted = False
        if plan is not None:
            if not _selected_caller_supports_row_partials(template_node):
                raise RowPartialsUnsupported("selected template does not emit row partials")
            self.must_keep_buffers.add(plan.raw_output)
            self.must_keep_buffers.add(plan.final_output)

        def checked_render():  # type: ignore[no-untyped-def]
            result = render()
            if plan is not None and not self._tlx_row_partials_emitted:
                raise RowPartialsUnsupported("template did not call reduction_store_output")
            return result

        return original_codegen(
            self,
            scheduling,
            template_node,
            epilogue_nodes,
            prologue_nodes,
            buf_name_to_prologue_group,
            prologue_preserves_zero_mask_fn,
            checked_render,
        )

    TritonTemplateKernel.codegen_template_body = patched_codegen

    original_metadata = TritonTemplateKernel._compute_fusion_metadata

    def patched_metadata(self, scheduling, epilogue_nodes, prologue_nodes, groups):  # type: ignore[no-untyped-def]
        original_metadata(self, scheduling, epilogue_nodes, prologue_nodes, groups)
        if getattr(self, "_tlx_reduction_epilogue_plan", None) is not None:
            from collections import defaultdict

            self._epilogue_nodes_by_subgraph = defaultdict(list)
            self._unfused_epilogues = []

    TritonTemplateKernel._compute_fusion_metadata = patched_metadata

    original_render = TritonTemplateKernel.render

    def patched_render(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        if not any(fn.__name__ == "reduction_store_output" for fn in self.extra_template_env_fns):
            fn = lambda *a, **kw: reduction_store_output(self, *a, **kw)  # noqa: E731
            fn.__name__ = "reduction_store_output"
            self._register_extra_template_env_fns(fn)
        return original_render(self, *args, **kwargs)

    TritonTemplateKernel.render = patched_render

    original_post = TritonTemplateKernel._emit_post_kernel_code

    def patched_post(self, wrapper, kernel_name):  # type: ignore[no-untyped-def]
        original_post(self, wrapper, kernel_name)
        emit_post_kernel(self, wrapper, kernel_name)

    TritonTemplateKernel._emit_post_kernel_code = patched_post

    original_call_kernel = TritonTemplateKernel.call_kernel

    def patched_call_kernel(self, name, node=None, deallocate_ws=True):  # type: ignore[no-untyped-def]
        workspace = getattr(self, "_tlx_reduction_workspace", None)
        if workspace is None or self.workspace_arg is not None:
            return original_call_kernel(self, name, node, deallocate_ws)
        # The row-stat workspace is sized by the rendered tile, so it is not
        # known to the template choice. Keep it alive across the producer and
        # the consumer emitted by _emit_post_kernel_code.
        wrapper = V.graph.wrapper_code
        wrapper.generate_workspace_allocation(workspace)
        try:
            return original_call_kernel(self, name, node, deallocate_ws=False)
        finally:
            if deallocate_ws:
                wrapper.generate_workspace_deallocation(workspace)

    TritonTemplateKernel.call_kernel = patched_call_kernel
    _INSTALLED = True
