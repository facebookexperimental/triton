# pyre-strict
"""Emit TLX Python source from a v0.1 schedule_graph.json.

Pure mechanical lowering — no optimization, no heuristics. Walks the JSON
graph and produces a single-file TLX kernel.

High-level structure of the generated source:

    @triton.jit
    def <kernel_name>(<args>, <constexprs>):
        # preamble: function-scope ops up to (but not including) scf.for
        # except allocs (hoisted) and the loop's tmem init (uses use_acc=False on MMA)

        # buffer allocs (SMEM + TMEM) from loop.buffers + function-scope tmem_alloc
        # mbarriers from cross-warp-group edges

        with tlx.async_tasks():
            with tlx.async_task("default"):
                # epilogue: tmem_load → descriptor_store
            with tlx.async_task(num_warps=1, num_regs=24):  # one per warp group
                # in-loop ops with that warp_group, sorted by (stage, cluster)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

from .schedule_graph import (
    ArgRef,
    Buffer,
    ConstRef,
    IterArgRef,
    IvRef,
    Loop,
    Node,
    Op,
    OperandRef,
    OpRef,
    ScheduleGraph,
    WarpGroup,
)
from .semaphore_ir import AccessKind, AsyncKind, build_sem_set_for_graph, SemSet


def _use_semaphore_ir() -> bool:
    """SemIR-based barrier emission is the default. Set `TLX_EMIT_LEGACY=1`
    to fall back to the legacy ad-hoc barrier emission (kept around for
    A/B comparison while SemIR matures).
    """
    return os.environ.get("TLX_EMIT_LEGACY", "0") != "1"


# ===========================================================================
# Type / dtype helpers
# ===========================================================================

_DTYPE_FROM_BITS_F = {16: "tl.float16", 32: "tl.float32", 64: "tl.float64"}
_DTYPE_FROM_BITS_I = {8: "tl.int8", 16: "tl.int16", 32: "tl.int32", 64: "tl.int64"}


def _bytes_per_elem_bits(bits: int) -> int:
    return bits // 8


def _bits_to_tl_dtype(bits: int, is_float: bool = True) -> str:
    if is_float:
        return _DTYPE_FROM_BITS_F.get(bits, "tl.float32")
    return _DTYPE_FROM_BITS_I.get(bits, "tl.int32")


# Parse `tensor<128x64xf16, ...>` → ([128, 64], "f16")
# Restrict dtype to the small set of MLIR scalar types we care about so the
# greedy match doesn't swallow the last dimension.
_DTYPE_ALT = r"(?:bf16|f8e4m3|f8e5m2|f16|f32|f64|i1|i8|i16|i32|i64)"
_TENSOR_TYPE_RE = re.compile(rf"tensor<([0-9x]+)x({_DTYPE_ALT})\b")
_DESC_TYPE_RE = re.compile(rf"!tt\.tensordesc<tensor<([0-9x]+)x({_DTYPE_ALT})\b")
# `!ttg.memdesc<128x128xbf16, ...>` — used for hoisted SMEM/TMEM allocs.
_MEMDESC_TYPE_RE = re.compile(rf"!ttg\.memdesc<([0-9x]+)x({_DTYPE_ALT})\b")


def _parse_tensor_shape(type_str: str) -> tuple[list[int], str] | None:
    """Extract [shape, dtype] from a printed `tensor<...>` MLIR type, or
    from a `!ttg.memdesc<...>` (hoisted SMEM/TMEM alloc result type)."""
    m = _TENSOR_TYPE_RE.search(type_str)
    if not m:
        m = _MEMDESC_TYPE_RE.search(type_str)
    if not m:
        return None
    dims_str, dtype = m.group(1), m.group(2)
    dims = [int(d) for d in dims_str.split("x") if d.isdigit()]
    return dims, dtype


def _parse_desc_block_shape(type_str: str) -> tuple[list[int], str] | None:
    """Extract block shape from `!tt.tensordesc<tensor<128x64xf16,...>>`."""
    m = _DESC_TYPE_RE.search(type_str)
    if not m:
        return None
    dims_str, dtype = m.group(1), m.group(2)
    dims = [int(d) for d in dims_str.split("x") if d.isdigit()]
    return dims, dtype


def _dtype_str_to_tl(dtype: str) -> str:
    if dtype.startswith("f"):
        bits = int(dtype[1:])
        return _bits_to_tl_dtype(bits, is_float=True)
    if dtype.startswith("i"):
        bits = int(dtype[1:])
        return _bits_to_tl_dtype(bits, is_float=False)
    if dtype.startswith("bf"):
        return "tl.bfloat16"
    return f"tl.{dtype}"


# ===========================================================================
# _Lines: indented source builder
# ===========================================================================


class _Lines:

    def __init__(self) -> None:
        self.buf: list[str] = []
        self.indent: int = 0

    def __iadd__(self, line: str) -> "_Lines":
        self.buf.append(("    " * self.indent + line) if line else "")
        return self

    def block(self, header: str):
        outer = self

        class Ctx:

            def __enter__(self_inner) -> None:
                outer.__iadd__(header)
                outer.indent += 1

            def __exit__(self_inner, *a: object) -> None:
                outer.indent -= 1

        return Ctx()

    def render(self) -> str:
        return "\n".join(self.buf) + "\n"


# ===========================================================================
# Renderer: op → Python expression
# ===========================================================================


@dataclass
class RenderCtx:
    """Per-emission state shared across renderers."""

    graph: ScheduleGraph
    # Map op_id → Python variable name once an op is emitted as `name = expr`.
    # If absent, the op is inlined every time (typical for trivial ops).
    op_var: dict[str, str]
    # Map (loop_id, buffer.id) → the Python variable name of its
    # tlx.local_alloc. Buffer IDs are local to each loop and collide across
    # loops, so we key globally by (loop_id, buf_id).
    buffer_var: dict[tuple[int, int], str]
    # Map alloc op_id → the Python variable name (used for ops referencing
    # an alloc by SSA op_id rather than by buffer index).
    alloc_op_var: dict[str, str]
    # Map loop id → induction-variable Python name.
    loop_iv: dict[int, str]
    # Renderer recursion depth (for safety).
    depth: int = 0
    # TMEM ring-buffer depth (count). Set by `_emit_buffers` after hoisting
    # the outer-loop's TMEM accumulator. =1 → no ring buffer indexing.
    tmem_count: int = 1
    # (loop_id, iter_arg_idx) → Python variable name. Populated by
    # `_emit_inner_loop_in_outer` per-UWG before/inside the loop so
    # `_render_operand(IterArgRef)` can produce a real name. Mirrors Meta's
    # WSSpecialize::SpecializeForOp where each per-task ForOp keeps only the
    # iter_args its ops consume; outside that scope, iter_args are unbound.
    iter_arg_var: dict[tuple[int, int], str] = field(default_factory=dict)
    # All cross-WG channels (smem + tmem). Set by the top-level emit() right
    # after derivation; consumed by in-loop emitters (MMA / tmem_load /
    # tmem_alloc bridge) to inject barrier_wait/arrive at the right places.
    channels: list[Any] = field(default_factory=list)
    # Cross-loop iter_arg result channels: dicts with keys {bufname, shape,
    # dtype, loop_id, idx, producer_wg, var_name}. The default partition
    # reads `var_name` after barrier_wait + local_load; the producer WG
    # writes via local_store + barrier_arrive after its inner loop ends.
    crossloop_channels: list[Any] = field(default_factory=list)
    # SemIR-based barrier emission. Built once in emit_kernel; consumed by
    # in-loop / per-WG emitters when SemIR is enabled (default). None when the
    # legacy Channel-based path is in use.
    sem_set: SemSet | None = None
    # Function-scope per-tile-resident loads: list of dicts with
    #   {alloc_var, alloc_op_id, load_op_id, load_op}
    # These are TMA loads (descriptor_load) that feed a function-scope SMEM
    # alloc (e.g., the Q tile in non-persistent FA). Loaded ONCE per kernel
    # invocation (not in the inner K-loop), need their own mbarrier.
    fn_scope_loads: list[Any] = field(default_factory=list)
    # Outer-loop intra-WG TMA loads (case5 bias): map from outer-loop
    # `tt.descriptor_load` node id → {bufname, count, n_bytes, var_name}.
    # `_emit_outer_op` reads this to emit the TMA-load sequence
    # (barrier_expect_bytes + async_descriptor_load + barrier_wait +
    # local_load) and bind the result to `var_name`.
    outer_load_bindings: dict[int, dict[str, Any]] = field(default_factory=dict)
    # Pass A.5 data partitioning: (loop_id, buf_id) → list of per-group var
    # names (e.g., [acc_tmem_g0, acc_tmem_g1]) for buffers that have
    # partition_count > 1. When non-empty, the buffer was emitted as N
    # separate `tlx.local_alloc`s of the per-group shape (mSize, X). Emitters
    # for descriptor_load / MMA / epilogue check this and fan out N parallel
    # ops with per-group M-offsets. The legacy `buffer_var[key]` still maps
    # to the FIRST group's name so single-group code paths keep working.
    partition_buffer_names: dict[tuple[int, int], list[str]] = field(default_factory=dict)
    # Mirror of partition_buffer_names keyed by def_op id (the SSA alloc).
    # Used when an emitter resolves a buffer via alloc_op_var instead of
    # (loop_id, buf_id).
    partition_alloc_names: dict[str, list[str]] = field(default_factory=dict)
    # Monotonic, globally-unique counter for auto-named variables. Every
    # auto-named op draws a fresh index via `fresh_idx()`, so names minted in
    # different scopes (preamble, each per-WG outer-loop body, epilogue,
    # infra-deps, ...) can NEVER collide. The previous code used a separate
    # counter per scope that each restarted at 0, which silently produced
    # duplicate names like `div_4`; inside a loop a reassigned name is
    # loop-carried, so an in-loop op auto-named `div_4 = tile_id // div_4`
    # clobbered the preamble's `div_4 = cdiv(N, 128)` and corrupted the tile
    # address arithmetic on every iteration after the first.
    _var_seq: int = 0

    def fresh_idx(self) -> int:
        """Return the next globally-unique auto-name index."""
        self._var_seq += 1
        return self._var_seq


def _render_operand(ref: OperandRef, rctx: RenderCtx) -> str:
    if isinstance(ref, ArgRef):
        return ref.name
    if isinstance(ref, IvRef):
        return rctx.loop_iv.get(ref.loop_id, f"iv_{ref.loop_id}")
    if isinstance(ref, IterArgRef):
        # Inside a per-UWG specialized loop body, the iter_arg is bound to a
        # Python variable (mirror of WSSpecialize::SpecializeForOp's
        # newForOp.getRegionIterArgs() mapping).
        nm = rctx.iter_arg_var.get((ref.loop_id, ref.idx))
        if nm is not None:
            return nm
        # Heuristic for OUTER persistent loops with no explicit binding:
        # tile_id_c = start_pid - step yielded as tile_id_c + step each iter,
        # so iter_arg == iv - step. Subsequent `iter_arg + step` simplifies to
        # `iv` (the current tile_id).
        loop = next((L for L in rctx.graph.loops if L.loop_id == ref.loop_id), None)
        if loop is not None and loop.is_outer:
            iv = rctx.loop_iv.get(ref.loop_id, "iv")
            step = _render_operand(loop.schedule.step, rctx)
            return f"({iv} - {step})"
        return f"<inner_iter_arg_{ref.loop_id}_{ref.idx}>"
    if isinstance(ref, ConstRef):
        return _render_const(ref)
    if isinstance(ref, OpRef):
        op = rctx.graph.ops.get(ref.op_id)
        # Reference to an scf.for result == post-loop value of the iter_arg
        # at index `result_idx`. Resolve via the emitted iter_arg variable
        # name (whatever the emitter bound for that loop+idx); same Python
        # name still holds after the for-loop in @triton.jit semantics.
        if op is not None and op.kind == "scf.for":
            target = next(
                (L for L in rctx.graph.loops if _find_loop_for(rctx.graph, L) is op),
                None,
            )
            if target is not None:
                nm = rctx.iter_arg_var.get((target.loop_id, ref.result_idx))
                if nm is not None:
                    return nm
                # Fallback: derive the same name we'd have used at binding.
                specs = _loop_iter_args(rctx.graph, target)
                for idx, init, _yld in specs:
                    if idx == ref.result_idx:
                        return _iter_arg_python_name(target.loop_id, idx, init)
        # Already-named op → use its var. Otherwise inline-render.
        if ref.op_id in rctx.op_var:
            return rctx.op_var[ref.op_id]
        # Reference to an alloc (smem/tmem) → resolve to the hoisted alloc
        # variable name (e.g., `acc_tmem`, `L0_smem_0`). The alloc's slot is
        # always index 0 since we always count=1 unless ring-buffered, and
        # consumers using a ring-indexed slot pass an explicit index.
        if ref.op_id in rctx.alloc_op_var:
            return f"{rctx.alloc_op_var[ref.op_id]}[0]"
        if op is None:
            return f"<missing:{ref.op_id}>"
        return _render_op_expr(op, rctx)
    return "?"


def _render_const(ref: ConstRef) -> str:
    v = ref.value
    if v is None:
        return "0"
    # Non-finite floats are dumped as JSON strings ("inf"/"-inf"/"nan").
    scalar = (f"float('{v}')" if isinstance(v, str) and v in ("inf", "-inf", "nan") else
              ("True" if v != 0 else "False") if ref.type == "i1" else repr(v))
    # Tensor splat (e.g., dense<-inf> in tensor<128xf32>) — wrap with
    # tl.full so the loop-carried iter_arg keeps its tensor type. Without
    # this, init `m_i = float('-inf')` is a Python scalar that mismatches
    # the recurrence's tensor type each iteration.
    if ref.type and "tensor<" in ref.type:
        sd = _parse_tensor_shape(ref.type)
        if sd:
            shape, dtype = sd
            shape_str = (str(shape[0]) + "," if len(shape) == 1 else ", ".join(str(d) for d in shape))
            tl_dtype = _dtype_str_to_tl(dtype)
            return f"tl.full(({shape_str}), {scalar}, {tl_dtype})"
    return scalar


def _render_op_expr(op: Op, rctx: RenderCtx) -> str:
    """Render an op's *expression* (RHS only; no assignment)."""
    if rctx.depth > 50:
        return "<RECURSION>"
    rctx.depth += 1
    try:
        rendered = _RENDERERS.get(op.kind, _render_unknown)(op, rctx)
    finally:
        rctx.depth -= 1
    return rendered


def _render_unknown(op: Op, rctx: RenderCtx) -> str:
    return f"<unhandled:{op.kind}>"


def _render_constant(op: Op, rctx: RenderCtx) -> str:
    val = op.attributes.get("value")
    rt = op.result_types[0] if op.result_types else ""
    if isinstance(val, int):
        if rt == "i1":
            return "True" if val != 0 else "False"
        return str(val)
    if isinstance(val, float):
        return repr(val)
    if isinstance(val, str):
        # Tensor constant like "dense<0.000000e+00> : tensor<...xf32, ...>"
        if "dense<0" in val and "tensor<" in val:
            shape_dt = _parse_tensor_shape(val)
            if shape_dt:
                shape, dtype = shape_dt
                shape_str = ", ".join(str(d) for d in shape)
                return f"tl.zeros(({shape_str}), {_dtype_str_to_tl(dtype)})"
        return f"# const-string: {val[:40]}..."
    return "0"


def _render_get_program_id(op: Op, rctx: RenderCtx) -> str:
    axis = op.attributes.get("axis", 0)
    return f"tl.program_id({axis})"


def _render_get_num_programs(op: Op, rctx: RenderCtx) -> str:
    axis = op.attributes.get("axis", 0)
    return f"tl.num_programs({axis})"


def _render_make_range(op: Op, rctx: RenderCtx) -> str:
    start = op.attributes.get("start", 0)
    end = op.attributes.get("end", 0)
    return f"tl.arange({start}, {end})"


def _render_extsi(op: Op, rctx: RenderCtx) -> str:
    # i32 → i64 cast: in Python/Triton, just pass through.
    return _render_operand(op.operands[0], rctx)


def _render_extf(op: Op, rctx: RenderCtx) -> str:
    # f16/bf16 → f32 cast: emit `.to(target_dtype)` so the receiver sees
    # the wider type. Without this, an addmul-style chain like
    # `(acc_f32 + bias_f16_loaded_as_f16)` would do mixed-dtype arithmetic
    # whose semantics aren't well-defined in Triton.
    inner = _render_operand(op.operands[0], rctx)
    rt = op.result_types[0] if op.result_types else ""
    sd = _parse_tensor_shape(rt)
    target = "tl.float32"
    if sd:
        target = _dtype_str_to_tl(sd[1])
    return f"{inner}.to({target})"


def _render_truncf(op: Op, rctx: RenderCtx) -> str:
    inner = _render_operand(op.operands[0], rctx)
    rt = op.result_types[0] if op.result_types else ""
    sd = _parse_tensor_shape(rt)
    target = "tl.float16"
    if sd:
        target = _dtype_str_to_tl(sd[1])
    return f"{inner}.to({target})"


def _make_binop_renderer(pyop: str):
    """Factory for `arith.muli/addi/subi/divsi/remsi/...` renderers."""

    def render(op: Op, rctx: RenderCtx) -> str:
        a = _render_operand(op.operands[0], rctx)
        b = _render_operand(op.operands[1], rctx)
        return f"({a} {pyop} {b})"

    return render


def _render_convert_layout(op: Op, rctx: RenderCtx) -> str:
    # Layout conversions are implicit in the Triton DSL.
    return _render_operand(op.operands[0], rctx)


def _render_memdesc_trans(op: Op, rctx: RenderCtx) -> str:
    # ttg.memdesc_trans on an SMEM buffer becomes tlx.local_trans.
    inner = _render_operand(op.operands[0], rctx)
    return f"tlx.local_trans({inner})"


def _render_make_tensor_descriptor(op: Op, rctx: RenderCtx) -> str:
    # operandSegmentSizes: [ptr_count, shape_count, stride_count, padding_count]
    seg = op.attributes.get("operandSegmentSizes", [1, 2, 2, 0])
    if isinstance(seg, list):
        n_ptr, n_shape, n_stride, _ = seg + [0] * (4 - len(seg))
    else:
        n_ptr, n_shape, n_stride = 1, 2, 2

    ops = op.operands
    ptr = _render_operand(ops[0], rctx)
    shape = [_render_operand(o, rctx) for o in ops[n_ptr:n_ptr + n_shape]]
    strides = [_render_operand(o, rctx) for o in ops[n_ptr + n_shape:n_ptr + n_shape + n_stride]]
    rt = op.result_types[0] if op.result_types else ""
    block_info = _parse_desc_block_shape(rt)
    if block_info is None:
        return f"tl.make_tensor_descriptor({ptr}, [{', '.join(shape)}], [{', '.join(strides)}], [...])"
    block_dims, _ = block_info
    block_str = ", ".join(str(d) for d in block_dims)
    return (f"tl.make_tensor_descriptor({ptr}, [{', '.join(shape)}], "
            f"[{', '.join(strides)}], [{block_str}])")


def _render_descriptor_load(op: Op, rctx: RenderCtx) -> str:
    # A TMA load consumed directly in registers (no downstream ttg.local_alloc
    # staging it into SMEM — e.g. LayerNorm's `load → tl.sum`) lowers to the
    # plain descriptor `.load()` that returns a register tensor. Loads that DO
    # feed a local_alloc are staged through SMEM by the in-loop buffer path and
    # never reach this renderer.
    desc = _render_operand(op.operands[0], rctx)
    offs = ", ".join(_render_operand(o, rctx) for o in op.operands[1:])
    return f"{desc}.load([{offs}])"


# ----- M3.1+: math + tensor-manip renderers (added for FA forward) -----


def _render_maxnumf(op: Op, rctx: RenderCtx) -> str:
    a = _render_operand(op.operands[0], rctx)
    b = _render_operand(op.operands[1], rctx)
    return f"tl.maximum({a}, {b})"


def _render_minnumf(op: Op, rctx: RenderCtx) -> str:
    a = _render_operand(op.operands[0], rctx)
    b = _render_operand(op.operands[1], rctx)
    return f"tl.minimum({a}, {b})"


def _make_unary_call(fn: str):

    def render(op: Op, rctx: RenderCtx) -> str:
        return f"{fn}({_render_operand(op.operands[0], rctx)})"

    return render


def _render_expand_dims(op: Op, rctx: RenderCtx) -> str:
    # Most common pattern in FA: axis=1 → x[:, None]; axis=0 → x[None, :].
    axis = op.attributes.get("axis", -1)
    inner = _render_operand(op.operands[0], rctx)
    if axis == 0:
        return f"{inner}[None, :]"
    if axis == 1:
        return f"{inner}[:, None]"
    return f"tl.expand_dims({inner}, {axis})"


def _render_broadcast(op: Op, rctx: RenderCtx) -> str:
    # Triton handles broadcast implicitly when operands of an arith op have
    # compatible shapes — passthrough; the binop downstream will broadcast.
    return _render_operand(op.operands[0], rctx)


def _render_splat(op: Op, rctx: RenderCtx) -> str:
    # Scalar → tensor of given shape (broadcast scalar). Inferred at use site.
    return _render_operand(op.operands[0], rctx)


def _render_reshape(op: Op, rctx: RenderCtx) -> str:
    inner = _render_operand(op.operands[0], rctx)
    rt = op.result_types[0] if op.result_types else ""
    sd = _parse_tensor_shape(rt)
    if sd:
        shape = ", ".join(str(d) for d in sd[0])
        return f"tl.reshape({inner}, [{shape}])"
    return f"tl.reshape({inner}, [...])"


def _render_trans(op: Op, rctx: RenderCtx) -> str:
    inner = _render_operand(op.operands[0], rctx)
    order = op.attributes.get("order")
    if order is not None and isinstance(order, list):
        # Default 2-D: just .trans(); explicit reorder via tl.trans for >2D.
        if len(order) == 2 and order == [1, 0]:
            return f"{inner}.trans()"
        return f"tl.trans({inner}, ({', '.join(str(o) for o in order)}))"
    return f"{inner}.trans()"


def _render_split(op: Op, rctx: RenderCtx) -> str:
    # tt.split returns 2 tensors. The emitter emits this as an assignment
    # of a tuple (handled in caller); here we render the RHS as `tl.split(x)`.
    return f"tl.split({_render_operand(op.operands[0], rctx)})"


def _render_join(op: Op, rctx: RenderCtx) -> str:
    a = _render_operand(op.operands[0], rctx)
    b = _render_operand(op.operands[1], rctx)
    return f"tl.join({a}, {b})"


def _reduce_kind(op: Op, rctx: RenderCtx) -> str:
    """Heuristic: determine the reduction kind. The dumper doesn't (yet) emit
    the body op kind, so we walk forward from this reduce through the
    arith chain. If we hit `arith.maxnumf` along the way, it's a max-reduce
    (typical FA: row-max → scale → maximum); otherwise sum.

    Walks at most 4 hops through pure arith ops (mul/add/extf/truncf/etc)
    that don't change the reduce semantics."""
    PASSTHROUGH = {
        "arith.mulf",
        "arith.addf",
        "arith.subf",
        "arith.divf",
        "arith.extf",
        "arith.truncf",
        "ttg.convert_layout",
        "tt.expand_dims",
        "tt.broadcast",
    }
    visited: set[str] = set()
    frontier = [op.op_id]
    for _ in range(4):
        next_frontier = []
        for src_id in frontier:
            if src_id in visited:
                continue
            visited.add(src_id)
            for cand in rctx.graph.ops.values():
                for o in cand.operands:
                    if isinstance(o, OpRef) and o.op_id == src_id:
                        if cand.kind == "arith.maxnumf":
                            return "max"
                        if cand.kind == "arith.minnumf":
                            return "min"
                        if cand.kind in PASSTHROUGH:
                            next_frontier.append(cand.op_id)
        frontier = next_frontier
        if not frontier:
            break
    return "sum"


def _render_reduce(op: Op, rctx: RenderCtx) -> str:
    axis = op.attributes.get("axis", 0)
    inner = _render_operand(op.operands[0], rctx)
    kind = _reduce_kind(op, rctx)
    return f"tl.{kind}({inner}, {axis})"


def _render_addptr(op: Op, rctx: RenderCtx) -> str:
    a = _render_operand(op.operands[0], rctx)
    b = _render_operand(op.operands[1], rctx)
    return f"({a} + {b})"


def _render_tt_load(op: Op, rctx: RenderCtx) -> str:
    # Pointer load `tl.load(ptr[, mask, other])`. operandSegmentSizes =
    # [n_ptr, n_mask, n_other]; w/b loads have no mask/other.
    seg = op.attributes.get("operandSegmentSizes", [1, 0, 0])
    if not isinstance(seg, list) or not seg:
        seg = [1, 0, 0]
    n_ptr = seg[0]
    n_mask = seg[1] if len(seg) > 1 else 0
    n_other = seg[2] if len(seg) > 2 else 0
    args = [_render_operand(op.operands[0], rctx)]
    idx = n_ptr
    if n_mask:
        args.append(f"mask={_render_operand(op.operands[idx], rctx)}")
        idx += n_mask
    if n_other:
        args.append(f"other={_render_operand(op.operands[idx], rctx)}")
    return f"tl.load({', '.join(args)})"


def _render_tmem_load(op: Op, rctx: RenderCtx) -> str:
    # In-loop tmem_load (returns tensor). The TMEM buffer is the operand.
    return f"tlx.local_load({_render_operand(op.operands[0], rctx)})"


# MLIR `arith.cmpi` predicate codes (mlir/Dialect/Arith/IR/ArithBase.td).
_CMPI_OP = {
    0: "==",
    1: "!=",
    2: "<",
    3: "<=",
    4: ">",
    5: ">=",
    6: "<",
    7: "<=",
    8: ">",
    9: ">=",
}
_CMPF_OP = {
    1: "==",
    2: ">",
    3: ">=",
    4: "<",
    5: "<=",
    6: "!=",
    8: "==",
    9: ">",
    10: ">=",
    11: "<",
    12: "<=",
    13: "!=",
}


def _render_cmp(op: Op, rctx: RenderCtx) -> str:
    pred = op.attributes.get("predicate", 0)
    table = _CMPI_OP if op.kind == "arith.cmpi" else _CMPF_OP
    sym = table.get(pred, "==")
    a = _render_operand(op.operands[0], rctx)
    b = _render_operand(op.operands[1], rctx)
    return f"({a} {sym} {b})"


def _render_scf_if(op: Op, rctx: RenderCtx) -> str:
    # Render scf.if as a Python `then if cond else else` ternary, picking the
    # yield matching the OpRef's result_idx via context. Only single-result
    # ifs render cleanly here (multi-result requires a tuple); pick yield 0.
    if not op.then_yields:
        return "<scf.if(no_then_yield)>"
    cond = _render_operand(op.operands[0], rctx)
    then_v = _render_operand(op.then_yields[0], rctx)
    else_v = _render_operand(op.else_yields[0], rctx) if op.else_yields else "0"
    return f"({then_v} if {cond} else {else_v})"


def _render_tmem_alloc(op: Op, rctx: RenderCtx) -> str:
    # In-loop tmem_alloc(tensor) — allocates and stores in one op. Used
    # for things like the P matrix in FA. We render as a placeholder; the
    # actual hoisted alloc + local_store sequence is emitted at function
    # scope. If we see this as an inline expression, fall through.
    if op.operands:
        return _render_operand(op.operands[0], rctx)
    return "<tmem_alloc_no_init>"


_RENDERERS: dict[str, Any] = {
    # arith integer
    "arith.constant": _render_constant,
    "arith.muli": _make_binop_renderer("*"),
    "arith.addi": _make_binop_renderer("+"),
    "arith.subi": _make_binop_renderer("-"),
    "arith.divsi": _make_binop_renderer("//"),
    "arith.divui": _make_binop_renderer("//"),
    "arith.remsi": _make_binop_renderer("%"),
    "arith.remui": _make_binop_renderer("%"),
    "arith.andi": _make_binop_renderer("&"),
    "arith.ori": _make_binop_renderer("|"),
    "arith.xori": _make_binop_renderer("^"),
    # arith float (added for FA)
    "arith.addf": _make_binop_renderer("+"),
    "arith.subf": _make_binop_renderer("-"),
    "arith.mulf": _make_binop_renderer("*"),
    "arith.divf": _make_binop_renderer("/"),
    "arith.maxnumf": _render_maxnumf,
    "arith.minnumf": _render_minnumf,
    # arith casts
    "arith.extsi": _render_extsi,
    "arith.extf": _render_extf,
    "arith.truncf": _render_truncf,
    # math (added for FA)
    "math.exp2": _make_unary_call("tl.math.exp2"),
    "math.log2": _make_unary_call("tl.math.log2"),
    "math.exp": _make_unary_call("tl.math.exp"),
    "math.log": _make_unary_call("tl.math.log"),
    "math.sqrt": _make_unary_call("tl.math.sqrt"),
    "math.rsqrt": _make_unary_call("tl.math.rsqrt"),
    # tensor manipulation (added for FA)
    "tt.expand_dims": _render_expand_dims,
    "tt.broadcast": _render_broadcast,
    "tt.splat": _render_splat,
    "tt.reshape": _render_reshape,
    "tt.trans": _render_trans,
    "tt.split": _render_split,
    "tt.join": _render_join,
    "tt.reduce": _render_reduce,
    # tt
    "tt.get_program_id": _render_get_program_id,
    "tt.get_num_programs": _render_get_num_programs,
    "tt.make_range": _render_make_range,
    "tt.make_tensor_descriptor": _render_make_tensor_descriptor,
    "tt.descriptor_load": _render_descriptor_load,
    "tt.load": _render_tt_load,
    "tt.addptr": _render_addptr,
    # ttg
    "ttg.convert_layout": _render_convert_layout,
    "ttg.memdesc_trans": _render_memdesc_trans,
    # ttng (in-loop)
    "ttng.tmem_load": _render_tmem_load,
    "ttng.tmem_alloc": _render_tmem_alloc,
    "scf.if": _render_scf_if,
    "arith.cmpi": _render_cmp,
    "arith.cmpf": _render_cmp,
}

# ===========================================================================
# Op classification helpers
# ===========================================================================

# Ops that produce a value worth naming as a Python variable in the preamble.
_NAMED_FUNCTION_OPS = {
    "tt.make_tensor_descriptor",
    "tt.get_program_id",
    "tt.get_num_programs",
    "tt.make_range",
    "arith.muli",
    "arith.addi",
    "arith.subi",
    "arith.divsi",
    "arith.divui",
    "arith.remsi",
    "arith.remui",
    "arith.extsi",
    "arith.extf",
}

# Ops worth assigning to a Python variable when emitted in a loop body
# (vs inlining at use-site). Anything that returns a tensor and is used by
# multiple downstream ops should be named to avoid duplicate Python source.
_IN_LOOP_NAMED_OPS = _NAMED_FUNCTION_OPS | {
    "arith.addf",
    "arith.subf",
    "arith.mulf",
    "arith.divf",
    "arith.maxnumf",
    "arith.minnumf",
    "arith.truncf",
    "math.exp2",
    "math.log2",
    "math.exp",
    "math.log",
    "math.sqrt",
    "math.rsqrt",
    "tt.expand_dims",
    "tt.broadcast",
    "tt.splat",
    "tt.reshape",
    "tt.trans",
    "tt.join",
    "tt.reduce",
    "tt.addptr",
    "ttg.convert_layout",
    "ttg.memdesc_trans",
    "ttng.tmem_load",
    # Register-consumed TMA load (no SMEM staging buffer); the in-loop
    # buffer path handles loads that feed a local_alloc and never reaches
    # the generic named-op path.
    "tt.descriptor_load",
}

# Ops we skip emission for entirely at function scope (handled elsewhere or no-op).
_SKIP_FUNCTION_SCOPE = {
    "scf.for", "scf.yield", "tt.return", "ttg.local_alloc",  # hoisted to top-of-kernel allocs
    "ttng.tmem_alloc",  # hoisted to top-of-kernel allocs
    "ttng.tmem_store",  # init-zero handled by use_acc=False on first MMA
    "arith.constant",  # inlined
}


def _is_in_loop(op: Op, loop: Loop) -> bool:
    return op.scope == f"loop:{loop.loop_id}"


def _function_scope_ops_in_order(graph: ScheduleGraph) -> list[Op]:
    """Iterate ops table in insertion order, function scope only."""
    return [op for op in graph.ops.values() if op.scope == "function"]


def _ops_before_loop(graph: ScheduleGraph, loop: Loop) -> list[Op]:
    """Function-scope ops that come strictly before the loop (preamble)."""
    out: list[Op] = []
    for op in graph.ops.values():
        if op.scope == "function":
            if op.kind == "scf.for":
                break
            out.append(op)
    return out


def _ops_after_loop(graph: ScheduleGraph, loop: Loop) -> list[Op]:
    """Function-scope ops that come after the loop (epilogue)."""
    out: list[Op] = []
    seen_loop = False
    for op in graph.ops.values():
        if op.scope == "function":
            if op.kind == "scf.for":
                seen_loop = True
                continue
            if seen_loop:
                out.append(op)
    return out


# ===========================================================================
# Buffer / barrier naming
# ===========================================================================


def _buffer_var_name(b: Buffer) -> str:
    if b.kind == "tmem":
        return f"acc_tmem_{b.id}"
    if b.kind == "barrier":
        return f"bar_{b.id}"
    return f"smem_{b.id}"


def _bar_full(buffer_var: str) -> str:
    return f"{buffer_var}_full"


def _bar_empty(buffer_var: str) -> str:
    return f"{buffer_var}_empty"


# ──────────────────────────────────────────────────────────────────────────
# SemIR barrier-emission helpers (used by in-loop emitters when the flag is on)
# ──────────────────────────────────────────────────────────────────────────


def _semir_consumer_waits(loop_id: int, node_id: int, rctx: RenderCtx) -> list[str]:
    """`tlx.barrier_wait(...)` lines a consumer node emits before reading."""
    if not _use_semaphore_ir() or rctx.sem_set is None:
        return []
    return rctx.sem_set.consumer_waits(loop_id, node_id)


def _semir_consumer_arrives(loop_id: int, node_id: int, rctx: RenderCtx) -> list[str]:
    """SW-recycle `tlx.barrier_arrive(...)` lines a consumer node emits after read.
    Empty when the consumer is HW-issued (e.g., MMA reading SMEM) — recycle
    happens via the consumer's mBarriers slot instead."""
    if not _use_semaphore_ir() or rctx.sem_set is None:
        return []
    return rctx.sem_set.consumer_arrives(loop_id, node_id)


def _semir_consumer_mbarriers(loop_id: int, node_id: int, rctx: RenderCtx) -> list[str]:
    """Slot exprs for an MMA consumer's mBarriers list — operand-recycle side."""
    if not _use_semaphore_ir() or rctx.sem_set is None:
        return []
    return rctx.sem_set.consumer_mbarriers(loop_id, node_id)


def _semir_producer_waits(loop_id: int, node_id: int, rctx: RenderCtx) -> list[str]:
    """`tlx.barrier_wait(...)` for a producer waiting empty before write."""
    if not _use_semaphore_ir() or rctx.sem_set is None:
        return []
    return rctx.sem_set.producer_waits(loop_id, node_id)


def _semir_producer_arrives(loop_id: int, node_id: int, rctx: RenderCtx) -> list[str]:
    """SW `tlx.barrier_arrive(...)` after a NONE-async producer's write."""
    if not _use_semaphore_ir() or rctx.sem_set is None:
        return []
    return rctx.sem_set.producer_arrives(loop_id, node_id)


def _semir_producer_expect_bytes(loop_id: int, node_id: int, rctx: RenderCtx) -> list[str]:
    """`tlx.barrier_expect_bytes(...)` lines for a TMA producer, before the load."""
    if not _use_semaphore_ir() or rctx.sem_set is None:
        return []
    return rctx.sem_set.expect_bytes_for(loop_id, node_id)


def _semir_producer_mbarriers(loop_id: int, node_id: int, rctx: RenderCtx) -> list[str]:
    """Slot exprs for a producer's mBarriers list (MMA / TMA / TMEM_COPY)."""
    if not _use_semaphore_ir() or rctx.sem_set is None:
        return []
    return rctx.sem_set.mbarriers_for(loop_id, node_id)


def _semir_producer_barrier_for_tma(loop_id: int, node_id: int, rctx: RenderCtx) -> str | None:
    """Single barrier slot expression a TMA load passes as its mbarrier arg.
    The TMA producer should have exactly one outgoing semaphore."""
    slots = _semir_producer_mbarriers(loop_id, node_id, rctx)
    return slots[0] if len(slots) == 1 else None


def sem_set_slot_for_buffer(rctx: RenderCtx, loop_id: int, buffer_id: int) -> str:
    """Slot subscript expression to use when indexing a buffer ring under SemIR.
    All semaphores guarding this buffer share the same slot expression."""
    if rctx.sem_set is None:
        return "0"
    for ls in rctx.sem_set.lowered:
        b = ls.sem.buffer
        if b is not None and b.loop_id == loop_id and b.buffer_id == buffer_id:
            return ls.slot_expr
    return "0"


def _semir_pre_arrives_for_wg(loop_id: int, wg: int, rctx: RenderCtx) -> list[str]:
    """`tlx.barrier_arrive(...)` lines for is_released loop-carry semaphores
    whose consumer lives in this WG."""
    if not _use_semaphore_ir() or rctx.sem_set is None:
        return []
    return rctx.sem_set.pre_arrives_for_wg(loop_id, wg)


def _semir_emit_consumer_block(n: Node, g: ScheduleGraph, loop: Loop, rctx: RenderCtx, lines: "_Lines") -> None:
    """Emit consumer-side waits + local_loads + recycle arrives for every
    semaphore where this node is a SW consumer. UNKNOWN-access (MMA-like)
    consumers are skipped — the MMA emit branch handles them via mBarriers.
    Binds each producer's op_id to the loaded variable so downstream renders
    see the cross-WG value."""
    if rctx.sem_set is None:
        return
    nodes_by_id = {nn.id: nn for nn in loop.schedule.nodes}
    for ls in rctx.sem_set.by_consumer.get((loop.loop_id, n.id), []):
        sem = ls.sem
        wait = ls.consumer_wait_at.get(n.id)
        if not wait:
            continue
        cons = next((c for c in sem.consumers if c.node.node_id == n.id), None)
        if cons is None or cons.access_kind == AccessKind.UNKNOWN:
            # MMA-like consumer: handled by the MMA emit (wait + mBarriers).
            continue
        if sem.buffer is None:
            # Signal-only semaphore: just the wait, no local_load.
            lines += f"{wait}  # {sem.note}"
            continue
        # Has a buffer — materialize the cross-WG value via local_load.
        buf_var = rctx.buffer_var.get((sem.buffer.loop_id, sem.buffer.buffer_id))
        if buf_var is None:
            lines += f"# (buffer {sem.buffer.buffer_id} unresolved for {ls.name})"
            continue
        # A TMA descriptor_store consumer can read its source STRAIGHT from the
        # channel SMEM — no register round-trip (which would spill a full tile
        # into a 1-warp store group, the case6 store-WG slowdown). Emit only the
        # wait here and hand the channel buffer to the descriptor_store handler;
        # the empty-recycle arrive fires AFTER the store drains.
        if n.op_kind == "tt.descriptor_store":
            lines += wait
            rctx._store_from_channel = {
                "buf_var": buf_var,
                "slot": ls.slot_expr,
                "arrive": ls.consumer_arrive_at.get(n.id),
            }
            continue
        seed = int(rctx.op_var.get("__inloop_var_seed__", "0"))
        chan_var = f"chan_{ls.name}_{seed}"
        rctx.op_var["__inloop_var_seed__"] = str(seed + 1)
        lines += wait
        lines += f"{chan_var} = tlx.local_load({buf_var}[{ls.slot_expr}])"
        if a := ls.consumer_arrive_at.get(n.id):
            lines += a
        # Bind each producer's op_id → chan_var so cross-WG consumers
        # resolve to the materialized variable, not a recursive render.
        for prod in sem.producers:
            prod_node = nodes_by_id.get(prod.node.node_id)
            if prod_node and prod_node.op_ref:
                rctx.op_var[prod_node.op_ref] = chan_var


def _semir_mma_operand_waits_and_mbarriers(op: Op, g: ScheduleGraph, loop: Loop,
                                           rctx: RenderCtx) -> tuple[list[str], list[str]]:
    """For an MMA op, look up SemIR semaphores guarding its operand buffers.
    The schedule's cross_wg_barriers list names the local_alloc node as
    consumer (not the MMA), so we look up consumer-waits on that alloc node.
    Returns (wait_lines, mbarrier_slots)."""
    if rctx.sem_set is None:
        return [], []
    waits: list[str] = []
    mbar: list[str] = []
    seen_sems: set[int] = set()
    nodes_by_op = {n.op_ref: n for n in loop.schedule.nodes if n.op_ref}
    for operand in op.operands[:2]:  # A, B (potentially through memdesc_trans)
        if not isinstance(operand, OpRef):
            continue
        opd = g.ops.get(operand.op_id)
        if opd is None:
            continue
        # Walk through memdesc_trans to find the underlying alloc.
        alloc_op_id = operand.op_id
        if opd.kind == "ttg.memdesc_trans" and opd.operands:
            inner = opd.operands[0]
            if isinstance(inner, OpRef):
                alloc_op_id = inner.op_id
        # Find the schedule node for the local_alloc — it's the SemIR consumer.
        alloc_node = nodes_by_op.get(alloc_op_id)
        if alloc_node is None:
            continue
        cons_sems = rctx.sem_set.by_consumer.get((loop.loop_id, alloc_node.id), [])
        if cons_sems:
            for ls in cons_sems:
                if ls.sem_id in seen_sems:
                    continue
                seen_sems.add(ls.sem_id)
                wait = ls.consumer_wait_at.get(alloc_node.id)
                if wait:
                    waits.append(wait)
                # HW recycle: MMA reading the SMEM operand signals the EMPTY
                # barrier when reads complete (lets producer overwrite).
                if ls.alloc_empty_stmt is not None:
                    mbar.append(f"{ls.empty_name}[{ls.slot_expr}]")
            continue
        # If a TMEM bridge channel covers this operand, skip the intra-WG
        # fallback — the bridge's own emit path (downstream in async_dot)
        # adds the wait + empty mBarrier. Without this skip we'd emit
        # duplicate barrier_waits and double-recycle the empty barrier.
        if any(c.kind == "tmem" and c.alloc_op_id == alloc_op_id for c in rctx.channels):
            continue
        # Intra-WG fallback: no SemIR cross-WG semaphore for this operand
        # (producer + consumer share a WG). Use the legacy `<buf>_full`/
        # `<buf>_empty` pair allocated in the `extra` carve-out. Per-buf
        # slot/phase use the buffer's ring count (matches the load).
        buf = next((b for b in loop.schedule.buffers if b.def_op == alloc_op_id), None)
        if buf is None:
            continue
        buf_var = rctx.buffer_var.get((loop.loop_id, buf.id))
        if buf_var is None:
            continue
        idx = "0" if buf.count == 1 else "buf"
        ph = "(_it & 1)" if buf.count == 1 else "phase"
        waits.append(f"tlx.barrier_wait({_bar_full(buf_var)}[{idx}], {ph})")
        mbar.append(f"{_bar_empty(buf_var)}[{idx}]")
    return waits, mbar


def _semir_emit_producer_block(
    n: Node,
    value_var: str,
    g: ScheduleGraph,
    loop: Loop,
    rctx: RenderCtx,
    lines: "_Lines",
) -> None:
    """Emit producer-side wait-empty + local_store + arrive-full for every
    semaphore where this node is a producer of a buffered value (SW path).

    `value_var` is the Python variable holding the freshly produced value
    (i.e., the LHS of `name = expr` for the just-emitted compute op).

    HW-issued producers (TMA / MMA / TMEM_COPY) are NOT emitted here — their
    producer-arrive lands in the load/MMA's mBarriers/expect arg via the
    op-specific renderers (descriptor_load, async_dot).

    TMEM bridge: when this producer's value also feeds a TMEM bridge op
    (`ttng.tmem_alloc(value)` in another WG), additionally store the value
    into the TMEM buffer with its own full/empty barriers — the consumer MMA
    reads from the TMEM buffer, not from the SMEM staging."""
    if rctx.sem_set is None:
        return
    op = g.ops.get(n.op_ref) if n.op_ref else None
    for ls in rctx.sem_set.by_producer.get((loop.loop_id, n.id), []):
        sem = ls.sem
        # Skip HW-issued releases — descriptor_load / async_dot handles them.
        prod = next((p for p in sem.producers if p.node.node_id == n.id), None)
        if prod is None or prod.async_kind != AsyncKind.NONE:
            continue
        # SW producer: wait empty (unless is_released), store, arrive full.
        if w := ls.producer_wait_at.get(n.id):
            lines += w
        if sem.buffer is not None:
            buf_var = rctx.buffer_var.get((sem.buffer.loop_id, sem.buffer.buffer_id))
            if buf_var is not None:
                lines += f"tlx.local_store({buf_var}[{ls.slot_expr}], {value_var})"
                # If a consumer reads this channel via the async proxy (a TMA
                # descriptor_store reads SMEM directly), the producer's
                # generic-proxy write must be fenced before the full-arrive so
                # the TMA sees it. Register consumers don't need it, but it's
                # only emitted when a TMA store consumes the channel.
                if _channel_has_tma_store_consumer(sem, loop):
                    lines += "tlx.fence_async_shared()"
        if a := ls.producer_arrive_at.get(n.id):
            lines += a

    # TMEM bridge handover: if this op produces a value that's wrapped by a
    # ttng.tmem_alloc(value) bridge in another WG (cross_wg_barriers won't
    # capture the SMEM→TMEM data-routing), emit a direct local_store into the
    # TMEM buffer + full/empty barriers (legacy carve-out style).
    # NOTE: use plain barrier_arrive, NOT tcgen05_commit. local_store on TMEM
    # is synchronous from the warp's POV. tcgen05_commit ties the barrier to
    # async tcgen5.mma completion; with no async op pending it becomes a no-op
    # and the barrier never fires — deadlocks the consumer (caught at scale:
    # case3 FA fwd, 2048+ CTAs, large N_CTX).
    if op is None:
        return
    for c in rctx.channels:
        if c.kind != "tmem" or not c.bridge_op_id:
            continue
        bridge_op = g.ops.get(c.bridge_op_id)
        if (bridge_op and bridge_op.operands and isinstance(bridge_op.operands[0], OpRef)
                and bridge_op.operands[0].op_id == op.op_id):
            lines += (f"tlx.barrier_wait({_bar_empty(c.name)}[0], "
                      f"(_it & 1) ^ 1)  # TMEM bridge")
            lines += f"tlx.local_store({c.name}[0], {value_var})"
            lines += f"tlx.barrier_arrive({_bar_full(c.name)}[0], 1)"


# ===========================================================================
# Function signature
# ===========================================================================


def _kernel_sig_lines(g: ScheduleGraph, lines: _Lines) -> None:
    lines += "@triton.jit"
    lines += f"def {g.kernel.name}("
    # TensorDescriptor args are flattened to 5 fields each (ptr, 2x shape,
    # 2x stride) sharing a loc name. The dumper deduped them as
    # `<name>, <name>_0, <name>_1, <name>_2, <name>_3`. For the emitted
    # kernel signature, keep ONLY the bare name — Triton auto-flattens when
    # passed a TensorDescriptor at the call site.
    arg_names = [a.name for a in g.kernel.args]
    name_set = set(arg_names)
    kept: list[str] = []
    for a in g.kernel.args:
        m = re.match(r"^(.+)_\d+$", a.name)
        if m and m.group(1) in name_set:
            continue  # dropped flattened sibling
        kept.append(a.name)
    n = len(kept)
    for i, name in enumerate(kept):
        sep = "," if i + 1 < n else ""
        lines += f"    {name}{sep}"
    lines += "):"


# ===========================================================================
# Preamble (function-scope ops before the loop)
# ===========================================================================


def _emit_preamble(g: ScheduleGraph, loop: Loop, rctx: RenderCtx, lines: _Lines) -> None:
    lines += "# ── Preamble (function-scope ops before the loop) ──"
    pre_ops = _ops_before_loop(g, loop)
    for op in pre_ops:
        if op.kind in _SKIP_FUNCTION_SCOPE:
            continue
        if op.kind not in _NAMED_FUNCTION_OPS:
            continue
        # Auto-name based on op kind (descriptors get nice names).
        name = _auto_name(op, rctx.fresh_idx())
        rctx.op_var[op.op_id] = name
        rhs = _render_op_expr(op, rctx)
        lines += f"{name} = {rhs}"
    lines += ""


_OP_KIND_NAME_PREFIX = {
    "arith.muli": "mul",
    "arith.addi": "add",
    "arith.subi": "sub",
    "arith.divsi": "div",
    "arith.divui": "div",
    "arith.remsi": "rem",
    "arith.remui": "rem",
    "arith.extsi": "ext",
    "arith.extf": "ext",
    "arith.addf": "addf",
    "arith.subf": "subf",
    "arith.mulf": "mulf",
    "arith.divf": "divf",
    "arith.maxnumf": "maxf",
    "arith.minnumf": "minf",
    "arith.truncf": "trunc",
    "math.exp2": "exp2",
    "math.log2": "log2",
    "math.exp": "exp",
    "math.log": "log",
    "math.sqrt": "sqrt",
    "math.rsqrt": "rsqrt",
    "tt.expand_dims": "expand",
    "tt.broadcast": "bcast",
    "tt.splat": "splat",
    "tt.reshape": "reshape",
    "tt.trans": "trans",
    "tt.join": "join",
    "tt.reduce": "red",
    "tt.addptr": "addptr",
    "ttg.convert_layout": "cvt",
    "ttg.memdesc_trans": "mdt",
    "ttng.tmem_load": "tmload",
}


def _auto_name(op: Op, idx: int) -> str:
    """Generate a Python variable name for an op based on its kind."""
    if op.kind == "tt.make_tensor_descriptor":
        # Use the underlying ptr arg name if available: A → a_desc.
        if op.operands and isinstance(op.operands[0], ArgRef):
            base = op.operands[0].name.lower()
            # Strip any numeric suffix that was appended for arg dedup
            # (a_desc_0/_1/... all alias back to a_desc).
            base = re.sub(r"_\d+$", "", base)
            # If the base already ends with "_desc" (e.g., the kernel arg is
            # named "a_desc"), don't double-suffix.
            return base if base.endswith("_desc") else f"{base}_desc"
        return f"desc_{idx}"
    if op.kind == "tt.get_program_id":
        axis = op.attributes.get("axis", 0)
        return f"pid_{axis}"
    if op.kind == "tt.get_num_programs":
        axis = op.attributes.get("axis", 0)
        return f"nprog_{axis}"
    if op.kind == "tt.make_range":
        return f"range_{idx}"
    if op.kind in _OP_KIND_NAME_PREFIX:
        return f"{_OP_KIND_NAME_PREFIX[op.kind]}_{idx}"
    return f"v_{idx}"


# ===========================================================================
# Buffer allocs at top-of-kernel
# ===========================================================================


def _emit_buffers(loop: Loop, g: ScheduleGraph, rctx: RenderCtx, lines: _Lines) -> None:
    lines += "# ── Multi-buffered allocations (from modulo's lifetime analysis) ──"
    loop_tag = "inner" if not loop.is_outer else "outer"
    # Track the FIRST allocated variable for each merge_group_id so subsequent
    # buffers in the same group emit `reuse=<first_var>` (Step 4.5 says they
    # have disjoint lifetimes — same physical bytes, different time slots).
    merge_group_owner: dict[int, str] = {}
    for b in loop.schedule.buffers:
        # Per-loop unique name to avoid id collisions between inner/outer.
        var = f"L{loop.loop_id}_{_buffer_var_name(b)}"
        rctx.buffer_var[(loop.loop_id, b.id)] = var
        if b.def_op:
            rctx.alloc_op_var[b.def_op] = var
        if b.kind == "smem":
            # 1D shapes need trailing comma in Python tuple syntax
            # (`(256,)` not `(256)`); 2D+ are fine as-is.
            shape = (str(b.shape[0]) + "," if len(b.shape) == 1 else ", ".join(str(d) for d in b.shape))
            # Prefer the def_op's actual MLIR dtype string (preserves bf16
            # vs f16, which `element_bits` collapses).
            dtype = _bits_to_tl_dtype(b.element_bits, is_float=True)
            if b.def_op:
                def_op = g.ops.get(b.def_op)
                if def_op and def_op.result_types:
                    sd = _parse_tensor_shape(def_op.result_types[0])
                    if sd:
                        dtype = _dtype_str_to_tl(sd[1])
            else:
                # Synthesized cross-WG channel buffer (no def_op). The
                # producer op's result type carries the actual dtype.
                for cb in loop.schedule.cross_wg_barriers:
                    if cb.paired_buffer_id != b.id:
                        continue
                    prod_node = next(
                        (n for n in loop.schedule.nodes if n.id == cb.producer_node),
                        None,
                    )
                    if prod_node and prod_node.op_ref:
                        prod_op = g.ops.get(prod_node.op_ref)
                        if prod_op and prod_op.result_types:
                            sd = _parse_tensor_shape(prod_op.result_types[0])
                            if sd:
                                dtype = _dtype_str_to_tl(sd[1])
                                break
            origin = ("channel for cross-WG hand-off"
                      if b.def_op is None else f"modulo lifetime [{b.live_start}..{b.live_end}], "
                      f"II={loop.schedule.II}")
            mgid = b.merge_group_id
            reuse = ""
            if mgid is not None and mgid in merge_group_owner:
                reuse = f", reuse={merge_group_owner[mgid]}"
                origin += f"; reuses {merge_group_owner[mgid]} (group {mgid})"
            lines += f"# {loop_tag}-loop buf {b.id}: SMEM count={b.count} ({origin})"
            if b.partition_count > 1:
                # Pass A.5: emit N SMEM allocs, each (mSize, *trailing).
                # Per-group shape replaces the partition_dim (M=0) with mSize.
                pshape_dims = list(b.shape)
                pshape_dims[b.partition_dim] = b.m_size
                pshape_str = (str(pshape_dims[0]) + "," if len(pshape_dims) == 1 else ", ".join(
                    str(d) for d in pshape_dims))
                names = []
                for gi in range(b.partition_count):
                    gvar = f"{var}_g{gi}"
                    names.append(gvar)
                    lines += (f"{gvar} = tlx.local_alloc(({pshape_str}), {dtype}, "
                              f"{b.count}{reuse})")
                rctx.partition_buffer_names[(loop.loop_id, b.id)] = names
                if b.def_op:
                    rctx.partition_alloc_names[b.def_op] = names
                # `var` (legacy name) aliases group 0 — keep so legacy code
                # that references `buffer_var[key]` resolves to a real alloc.
                lines += f"{var} = {names[0]}"
            else:
                lines += (f"{var} = tlx.local_alloc(({shape}), {dtype}, {b.count}{reuse})")
            if mgid is not None and mgid not in merge_group_owner:
                merge_group_owner[mgid] = var
        elif b.kind == "tmem":
            shape = (str(b.shape[0]) + "," if len(b.shape) == 1 else ", ".join(str(d) for d in b.shape))
            # TMEM dtype: prefer the def_op's MLIR type (P_tmem is bf16,
            # accumulator is fp32) — element_bits alone collapses bf16/f16.
            dtype = "tl.float32"
            if b.def_op:
                def_op = g.ops.get(b.def_op)
                if def_op and def_op.result_types:
                    sd = _parse_tensor_shape(def_op.result_types[0])
                    if sd:
                        dtype = _dtype_str_to_tl(sd[1])
            mgid = b.merge_group_id
            reuse = ""
            origin_suffix = ""
            if mgid is not None and mgid in merge_group_owner:
                reuse = f", reuse={merge_group_owner[mgid]}"
                origin_suffix = f"; reuses {merge_group_owner[mgid]} (group {mgid})"
            lines += (f"# {loop_tag}-loop buf {b.id}: TMEM count={b.count} "
                      f"(producer→consumer pipelining across iters{origin_suffix})")
            lines += (f"{var} = tlx.local_alloc(({shape}), {dtype}, "
                      f"{b.count}, tlx.storage_kind.tmem{reuse})")
            if mgid is not None and mgid not in merge_group_owner:
                merge_group_owner[mgid] = var
        elif b.kind == "barrier":
            # Barriers are emitted later (paired with their data buffer).
            continue
    # Function-scope allocs (e.g., the accumulator TMEM for case1, or the
    # per-tile-resident Q SMEM for non-persistent FA) live in the preamble.
    # Hoist them up here so MMAs can reference them by name. Pre-sort the
    # ttng.tmem_alloc ops by size descending so the LARGEST gets the
    # canonical `acc_tmem` name (the running accumulator the epilogue
    # reads); smaller TMEM ops (e.g., qk scratch) get suffixed names.
    fn_ops = list(g.ops.values())

    def _tmem_size(op):
        if op.kind != "ttng.tmem_alloc" or not op.result_types:
            return 0
        sd = _parse_tensor_shape(op.result_types[0])
        if not sd:
            return 0
        prod = 1
        for d in sd[0]:
            prod *= d
        return prod

    fn_ops.sort(key=lambda o: (-_tmem_size(o), o.op_id))
    for op in fn_ops:
        if op.scope != "function":
            continue
        if op.kind == "ttg.local_alloc":
            if op.op_id in rctx.alloc_op_var:
                continue
            shape_dt = (_parse_tensor_shape(op.result_types[0]) if op.result_types else None)
            if not shape_dt:
                continue
            shape, dtype_str = shape_dt
            shape_str = ", ".join(str(d) for d in shape) + ("," if len(shape) == 1 else "")
            dtype = _dtype_str_to_tl(dtype_str)
            name = f"q_smem_{len(rctx.alloc_op_var)}"
            rctx.alloc_op_var[op.op_id] = name
            lines += (f"# {name}: function-scope SMEM alloc (e.g., per-tile "
                      f"resident Q tile in non-persistent FA)")
            lines += f"{name} = tlx.local_alloc(({shape_str}), {dtype}, 1)"
            # If this alloc is fed by a function-scope tt.descriptor_load,
            # track it: we need to emit the TMA load (in MEM-role WG) +
            # consumer-side wait (in MMA-role WG) outside the K-loop.
            if op.operands and isinstance(op.operands[0], OpRef):
                load_op = g.ops.get(op.operands[0].op_id)
                if (load_op and load_op.kind == "tt.descriptor_load" and load_op.scope == "function"):
                    rctx.fn_scope_loads.append({
                        "alloc_var": name,
                        "alloc_op_id": op.op_id,
                        "load_op_id": load_op.op_id,
                        "load_op": load_op,
                    })
            continue
        if op.kind == "ttng.tmem_alloc":
            if op.op_id in rctx.alloc_op_var:
                continue  # already emitted from loop.buffers list
            shape_dt = (_parse_tensor_shape(op.result_types[0]) if op.result_types else None)
            shape = shape_dt[0] if shape_dt else [128, 128]
            dtype = _dtype_str_to_tl(shape_dt[1]) if shape_dt else "tl.float32"
            shape_str = ", ".join(str(d) for d in shape)
            # Reserve `acc_tmem` for the LARGEST function-scope tmem_alloc
            # (the running output accumulator); secondary ones (e.g., QK
            # tmem in non-persistent FA) get `acc_tmem_<idx>` to avoid
            # name collision.
            existing_acc = any(v == "acc_tmem" for v in rctx.alloc_op_var.values())
            name = f"acc_tmem_{len(rctx.alloc_op_var)}" if existing_acc else "acc_tmem"
            # Bind via alloc_op_var only — `_render_operand` walks alloc_op_var
            # and appends `[0]` to make it a slot reference. If we ALSO bind
            # via op_var, the bare name without `[0]` wins and we get
            # `local_load(acc_tmem)` instead of `local_load(acc_tmem[0])`.
            rctx.alloc_op_var[op.op_id] = name
            lines += (f"{name} = tlx.local_alloc(({shape_str}), {dtype}, 1, "
                      f"tlx.storage_kind.tmem)")
    # Epilogue staging SMEM (for the descriptor_store) — derived from the
    # store op's source tensor shape.
    epi_store = next(
        (op for op in g.ops.values() if op.scope == "function" and op.kind == "tt.descriptor_store"),
        None,
    )
    if epi_store and len(epi_store.operands) >= 2:
        # Derive shape AND dtype from the descriptor's block type so the
        # epilogue staging matches the actual descriptor element type
        # (case3 nows uses bf16, case2 uses f16).
        shape: list[int] = [128, 128]
        dtype = "tl.float16"
        if isinstance(epi_store.operands[0], OpRef):
            desc_op = g.ops.get(epi_store.operands[0].op_id)
            if desc_op:
                bs = _parse_desc_block_shape(desc_op.result_types[0] if desc_op.result_types else "")
                if bs:
                    shape, dtype = bs[0], _dtype_str_to_tl(bs[1])
        elif isinstance(epi_store.operands[0], ArgRef):
            arg = next((a for a in g.kernel.args if a.name == epi_store.operands[0].name), None)
            if arg:
                bs = _parse_desc_block_shape(arg.type)
                if bs:
                    shape, dtype = bs[0], _dtype_str_to_tl(bs[1])
        # Pass A.7: if the descriptor_store node was marked as subtiled
        # (subtile_count > 1), shrink the SMEM staging buffer to BN/S. The
        # sub-stores reuse this single buffer in series — safe because each
        # async_descriptor_store_wait drains before the next iter writes.
        sub_count = 1
        sub_n_size = 0
        for lp in g.loops:
            for nd in lp.schedule.nodes:
                if nd.op_ref == epi_store.op_id and nd.subtile_count > 1:
                    sub_count = nd.subtile_count
                    sub_n_size = nd.n_size
                    break
            if sub_count > 1:
                break
        if sub_count > 1 and sub_n_size > 0 and len(shape) >= 2:
            shape = [shape[0], sub_n_size]
        shape_str = ", ".join(str(d) for d in shape)
        lines += f"c_smem = tlx.local_alloc(({shape_str}), {dtype}, 1)"
    lines += ""


# ===========================================================================
# Mbarrier emission
# ===========================================================================


@dataclass
class Channel:
    """Cross-warp-group data hand-off needing full+empty mbarriers."""

    name: str  # python identifier (e.g., "smem_0")
    depth: int
    producer_wg: int
    consumer_wg: int
    kind: str = "smem"  # "smem" | "tmem"
    # The alloc op id (TMEM channels only) — used by emitters to detect when
    # an op writes to / reads from this channel and inject the right barriers.
    alloc_op_id: str | None = None
    # The "bridge" op id (TMEM only): the op in the consumer-side WG that
    # writes the value (e.g., a tmem_alloc(value) where value comes from the
    # producer WG). Emitter relocates this op to the producer WG's body.
    bridge_op_id: str | None = None
    # Schedule-driven channels from cross_wg_barriers: producer/consumer
    # node ids (within the loop's schedule.nodes) and the buffer slot.
    producer_node: int | None = None
    consumer_node: int | None = None
    buffer_id: int | None = None
    loop_id: int | None = None  # which loop owns the buffer


def _derive_crossloop_result_channels(g: ScheduleGraph, rctx: RenderCtx) -> list[dict[str, Any]]:
    """Detect references to scf.for results (= final iter_arg values) from
    function-scope ops (the default partition's epilogue). The value lives
    in the WG that wrote the last iter_arg yield — if that WG isn't the
    default partition, we need to stage through SMEM:

      producer WG (after inner loop): local_store + barrier_arrive
      default partition (before use):  barrier_wait + local_load

    Returns one descriptor per (loop, idx) pair needing staging.
    """
    out: list[dict[str, Any]] = []
    for loop in g.loops:
        if loop.is_outer:
            continue  # only inner-loop iter_arg results need this staging
        specs = _loop_iter_args(g, loop)
        if not specs:
            continue
        # Build wg_of for this loop's nodes.
        wg_of_op: dict[str, int] = {n.op_ref: n.warp_group for n in loop.schedule.nodes if n.op_ref}
        for idx, init, yld in specs:
            # Find the producer WG of the yield value.
            if not isinstance(yld, OpRef):
                continue
            prod_wg = wg_of_op.get(yld.op_id)
            if prod_wg is None:
                continue
            # Default partition is wg=-1 / -2 typically; treat anything other
            # than the producer's WG as cross-WG when reading from epilogue.
            # Determine if any function-scope op references scf.for.result[idx].
            referenced_by_epi = False
            for op in g.ops.values():
                if op.scope != "function":
                    continue
                for o in op.operands:
                    if (isinstance(o, OpRef) and o.result_idx == idx and (find := g.ops.get(o.op_id))
                            and find.kind == "scf.for"):
                        referenced_by_epi = True
                        break
                if referenced_by_epi:
                    break
            if not referenced_by_epi:
                continue
            # Resolve type / shape from the init.
            shape = [128]
            dtype = "tl.float32"
            if isinstance(init, ConstRef) and init.type:
                sd = _parse_tensor_shape(init.type)
                if sd:
                    shape, dt = sd
                    dtype = _dtype_str_to_tl(dt)
            var_name = _iter_arg_python_name(loop.loop_id, idx, init)
            out.append({
                "bufname": f"epi_{var_name}_smem",
                "shape": shape,
                "dtype": dtype,
                "loop_id": loop.loop_id,
                "idx": idx,
                "producer_wg": prod_wg,
                "var_name": var_name,
            })
    return out


def _alias_predecessors(channel_name: str, rctx: RenderCtx, graph: ScheduleGraph) -> list[str]:
    """For a channel buffer that reuses another buffer's bytes, return the
    list of buffer NAMES whose `_empty` must be signaled before this
    channel's producer overwrites the storage.

    Layer 2 of the buffer-aliasing safety: even though Step 4.5 verified
    disjoint lifetimes, the actual CONSUMER must finish reading the
    aliased buffer before the next producer writes. We enforce this by
    waiting on the predecessor's `_empty` barrier.

    Predecessors = other members of the same merge_group_id that are
    EARLIER in lifetime order (lower live_start cycle).
    """
    # Find which buffer corresponds to this channel name and what its
    # merge group / lifetime is.
    target_loop = None
    target_buf = None
    for L in graph.loops:
        for b in L.schedule.buffers:
            buf_name = rctx.buffer_var.get((L.loop_id, b.id))
            if buf_name == channel_name:
                target_loop = L
                target_buf = b
                break
        if target_buf:
            break
    if not target_buf or target_buf.merge_group_id is None:
        return []
    mgid = target_buf.merge_group_id
    self_start = target_buf.live_start
    # Build set of buffer ids paired to backward-edge (loop-carry release)
    # channels. These buffers carry no real data — the emitter treats them
    # as signal-only — so they can't be sensibly waited-on as predecessors.
    # Their `_empty` barrier never gets signaled (we skip the empty arrive
    # for backward channels), so any alias-predecessor wait on them would
    # deadlock.
    cycle_of = {n.id: n.schedule_cycle for n in target_loop.schedule.nodes}
    signal_only_buf_ids: set[int] = set()
    for cb in target_loop.schedule.cross_wg_barriers:
        if cb.paired_buffer_id is None:
            continue
        pc = cycle_of.get(cb.producer_node)
        cc = cycle_of.get(cb.consumer_node)
        if pc is not None and cc is not None and pc > cc:
            signal_only_buf_ids.add(cb.paired_buffer_id)
    preds: list[str] = []
    for b2 in target_loop.schedule.buffers:
        if b2.id == target_buf.id:
            continue
        if b2.merge_group_id != mgid:
            continue
        if b2.live_start >= self_start:
            continue  # not a predecessor
        if b2.id in signal_only_buf_ids:
            continue  # signal-only buffer — no real consumer to wait on
        nm = rctx.buffer_var.get((target_loop.loop_id, b2.id))
        if nm:
            preds.append(nm)
    return preds


def _derive_tmem_channels(g: ScheduleGraph, inner: Loop) -> list[Channel]:
    """TMEM cross-WG channels can't be detected from the DDG (which only
    records register-level edges). We find them by walking TMEM allocs and
    inspecting their producer/consumer ops:

      * Producer WG = WG of the op that WRITES to the alloc (tc_gen5_mma's
        operand[2], or tmem_alloc(value) / tmem_store).
      * Consumer WG = WG of the op that READS from the alloc (tmem_load,
        or another tc_gen5_mma reading the alloc as operand[0]/[1]).

    For tmem_alloc(value) where the value originates in another WG, the
    "bridge" op (the alloc itself) needs to be relocated to the value's WG
    so that local_store + barrier_arrive happen at the value producer side.
    """
    # Build inner-loop op_id → wg map (only inner ops have warp_group).
    wg_of: dict[str, int] = {}
    for n in inner.schedule.nodes:
        if n.op_ref:
            wg_of[n.op_ref] = n.warp_group

    out: list[Channel] = []
    seen_alloc: set[str] = set()

    # Build the full candidate list of TMEM allocs to check for cross-WG flow:
    # (1) loop-scope buffers from each schedule_loop.buffers list, AND
    # (2) function-scope ttng.tmem_alloc ops (e.g., the per-tile QK scratch
    #     and the running output accumulator in non-persistent FA — both at
    #     function scope, NOT in any loop's buffer list, but written/read
    #     by inner-loop ops in different WGs).
    @dataclass
    class _AllocCand:
        def_op: str
        kind: str  # buffer kind for naming

    candidates: list[_AllocCand] = []
    for L in g.loops:
        for b in L.schedule.buffers:
            if b.kind == "tmem" and b.def_op:
                candidates.append(_AllocCand(def_op=b.def_op, kind="tmem"))
    # Add function-scope ttng.tmem_alloc ops not already covered.
    for oid, op in g.ops.items():
        if op.scope != "function" or op.kind != "ttng.tmem_alloc":
            continue
        if any(c.def_op == oid for c in candidates):
            continue
        candidates.append(_AllocCand(def_op=oid, kind="tmem"))

    for cand in candidates:
        b_def_op = cand.def_op
        if True:
            if b_def_op in seen_alloc:
                continue
            # Find producer/consumer ops touching this alloc.
            producer_wgs: set[int] = set()
            consumer_wgs: set[int] = set()
            bridge_op_id: str | None = None
            for oid, op in g.ops.items():
                wg = wg_of.get(oid)
                if wg is None:
                    continue
                # tc_gen5_mma writes operand[2] (acc).
                if op.kind == "ttng.tc_gen5_mma" and len(op.operands) >= 3:
                    if (isinstance(op.operands[2], OpRef) and op.operands[2].op_id == b_def_op):
                        producer_wgs.add(wg)
                    if (isinstance(op.operands[0], OpRef) and op.operands[0].op_id == b_def_op):
                        consumer_wgs.add(wg)
                    if (isinstance(op.operands[1], OpRef) and op.operands[1].op_id == b_def_op):
                        consumer_wgs.add(wg)
                # tmem_load reads operand[0].
                if op.kind == "ttng.tmem_load" and op.operands:
                    if (isinstance(op.operands[0], OpRef) and op.operands[0].op_id == b_def_op):
                        consumer_wgs.add(wg)
                # tmem_store: MLIR layout is [dest, token, value, pred] —
                # operand[0] is the destination buffer.
                if op.kind == "ttng.tmem_store" and len(op.operands) >= 1:
                    if (isinstance(op.operands[0], OpRef) and op.operands[0].op_id == b_def_op):
                        producer_wgs.add(wg)
                # tmem_alloc(value) is itself the alloc — when it carries a
                # value operand, it both *is* the buffer AND stores to it.
                # If oid == b_def_op AND it has a value operand, this is the
                # bridge. The producer-side WG is wherever the value comes
                # from (chase the OpRef).
                if op.kind == "ttng.tmem_alloc" and oid == b_def_op and op.operands:
                    bridge_op_id = oid
                    val_ref = op.operands[0]
                    val_wg = (wg_of.get(val_ref.op_id) if isinstance(val_ref, OpRef) else None)
                    # Actual producer is the value's WG; the "consumer" is
                    # the WG that owns the alloc op (which will read it).
                    if val_wg is not None:
                        producer_wgs.add(val_wg)
                        consumer_wgs.add(wg)
            if not producer_wgs or not consumer_wgs:
                continue
            if producer_wgs == consumer_wgs:
                continue
            seen_alloc.add(b_def_op)
            # Pick a representative producer/consumer wg for naming.
            pwg = next(iter(producer_wgs - consumer_wgs), next(iter(producer_wgs)))
            cwg = next(iter(consumer_wgs - producer_wgs), next(iter(consumer_wgs)))
            # Use a stable placeholder; the emitter resolves to the bound alloc name.
            out.append(
                Channel(
                    name=b_def_op,  # placeholder; resolved later via alloc_op_var
                    depth=1,  # function-scope allocs are typically count=1
                    producer_wg=pwg,
                    consumer_wg=cwg,
                    kind="tmem",
                    alloc_op_id=b_def_op,
                    bridge_op_id=bridge_op_id,
                ))
    return out


def _derive_channels(loop: Loop, rctx: RenderCtx) -> list[Channel]:
    """One Channel per cross-WG handoff that carries data through a buffer.

    Uses the schedule's `cross_wg_barriers` as the authoritative source —
    each entry already records which buffer (if any) is the data payload
    via `paired_buffer_id`. Entries without a paired buffer are pure
    handshake signals (named barriers) and don't need a Channel."""
    seen: set[int] = set()
    out: list[Channel] = []
    for cb in loop.schedule.cross_wg_barriers:
        buf_id = cb.paired_buffer_id
        if buf_id is None or buf_id in seen:
            continue
        buf = next((b for b in loop.schedule.buffers if b.id == buf_id), None)
        if buf is None:
            continue
        seen.add(buf_id)
        out.append(
            Channel(
                name=rctx.buffer_var.get((loop.loop_id, buf.id), f"L{loop.loop_id}_{_buffer_var_name(buf)}"),
                depth=buf.count,
                producer_wg=cb.producer_wg,
                consumer_wg=cb.consumer_wg,
            ))
    return out


def _emit_mbarriers(
    channels: list[Channel],
    lines: _Lines,
    tmem_count: int = 1,
    have_separate_tmem_handoff: bool = True,
    extra_buffers: list[tuple[str, int]] | None = None,
) -> None:
    """Single source of truth for mbarrier allocation.

    Every producer-consumer SMEM/TMEM buffer needs a `_full` + `_empty`
    barrier pair. There are three sources:

      1. Cross-WG channels (producer WG ≠ consumer WG) — `channels` list.
      2. Intra-WG async producers (TMA loads → consumer in same WG) —
         `extra_buffers` list. Triton still requires explicit barriers
         since loads are async.
      3. The outer-loop TMEM handoff (TC writes → default reads), special-
         cased because it's outside the inner loop's edge graph.
    """
    lines += "# ── Mbarriers (one full+empty pair per channel) ──"
    seen: set[str] = set()
    for ch in channels:
        if ch.name in seen:
            continue
        seen.add(ch.name)
        lines += (f"# {ch.name}: wg{ch.producer_wg} → wg{ch.consumer_wg}, "
                  f"depth={ch.depth} (matches buffer ring count)")
        lines += (f"{_bar_full(ch.name)} = tlx.alloc_barriers"
                  f"(num_barriers={ch.depth}, arrive_count=1)")
        lines += (f"{_bar_empty(ch.name)} = tlx.alloc_barriers"
                  f"(num_barriers={ch.depth}, arrive_count=1)")
    # The outer-loop TMEM buffer is a cross-WG channel: TC partition writes
    # it (via MMA), default partition reads it (via tmem_load). Producer and
    # consumer are in different warp groups → needs full/empty mbarrier
    # pairs, exactly like the SMEM channels above. Depth and bank count
    # come from outer-loop schedule_loop.buffers[tmem].count.
    if have_separate_tmem_handoff and "acc_tmem" not in seen:
        lines += (f"# acc_tmem: cross-WG channel TC → default "
                  f"(outer-loop TMEM buf, depth={tmem_count})")
        lines += (f"acc_tmem_full = tlx.alloc_barriers"
                  f"(num_barriers={tmem_count}, arrive_count=1)")
        lines += (f"acc_tmem_empty = tlx.alloc_barriers"
                  f"(num_barriers={tmem_count}, arrive_count=1)")
        seen.add("acc_tmem")
    # Extra buffers: load → MMA pipelines that aren't cross-WG channels but
    # still need barriers (TMA loads are async).
    for name, depth in extra_buffers or []:
        if name in seen:
            continue
        seen.add(name)
        lines += f"# {name}: TMA-load → consumer barrier (intra-WG async)"
        lines += (f"{_bar_full(name)} = tlx.alloc_barriers"
                  f"(num_barriers={depth}, arrive_count=1)")
        lines += (f"{_bar_empty(name)} = tlx.alloc_barriers"
                  f"(num_barriers={depth}, arrive_count=1)")
    lines += ""


# ===========================================================================
# Async-task emission
# ===========================================================================


def _nodes_in_warp_group(loop: Loop, wg: int) -> list[Node]:
    return sorted(
        [n for n in loop.schedule.nodes if n.warp_group == wg],
        key=lambda n: (n.schedule_stage, n.schedule_cluster, n.id),
    )


def _channel_for_buffer(channels: list[Channel], buffer_var: str) -> Channel | None:
    for ch in channels:
        if ch.name == buffer_var:
            return ch
    return None


def _trip_count_expr(loop: Loop, rctx: RenderCtx) -> str:
    sched = loop.schedule
    lo = _render_operand(sched.lower_bound, rctx)
    hi = _render_operand(sched.upper_bound, rctx)
    step = _render_operand(sched.step, rctx)
    return f"tl.cdiv({hi} - {lo}, {step})"


def _loop_range_expr(loop: Loop, rctx: RenderCtx) -> str:
    """Emit `range(lb, ub, step)` so the IV preserves its semantic value
    (matches the MLIR scf.for IV which steps by `step`, not by 1)."""
    sched = loop.schedule
    lo = _render_operand(sched.lower_bound, rctx)
    hi = _render_operand(sched.upper_bound, rctx)
    step = _render_operand(sched.step, rctx)
    return f"range({lo}, {hi}, {step})"


def _emit_default_partition(g: ScheduleGraph, loop: Loop, rctx: RenderCtx, lines: _Lines) -> None:
    # The caller (`_emit_uwg_body_impl`) has already opened the
    # `with tlx.async_task("default"):` block — emit body directly here.
    epi_ops = _ops_after_loop(g, loop)
    if True:
        # acc_tmem is a legacy carve-out under SemIR — full+empty pair.
        lines += "tlx.barrier_wait(acc_tmem_full[0], 0)"
        # Cross-loop iter_arg result channels: pull each cross-WG iter_arg
        # final value from its SMEM channel and bind to the iter_arg var.
        for ch in rctx.crossloop_channels:
            lines += f"tlx.barrier_wait({_bar_full(ch['bufname'])}[0], 0)"
            lines += f"{ch['var_name']} = tlx.local_load({ch['bufname']}[0])"
        # Find the tmem_load → arith.truncf → ttg.convert_layout → tt.descriptor_store chain.
        # Render each non-skipped epilogue op.
        for op in epi_ops:
            if op.kind in _SKIP_FUNCTION_SCOPE:
                continue
            if op.kind == "ttng.tmem_load":
                lines += "acc = tlx.local_load(acc_tmem[0])"
                # acc_tmem is a legacy carve-out under SemIR — full+empty.
                lines += "tlx.barrier_arrive(acc_tmem_empty[0], 1)"
                rctx.op_var[op.op_id] = "acc"
                continue
            if op.kind == "arith.truncf":
                inner = _render_operand(op.operands[0], rctx)
                # Render dtype from result_type.
                rt = op.result_types[0] if op.result_types else ""
                sd = _parse_tensor_shape(rt)
                target = _dtype_str_to_tl(sd[1]) if sd else "tl.float16"
                name = f"trunc_{rctx.fresh_idx()}"
                rctx.op_var[op.op_id] = name
                lines += f"{name} = {inner}.to({target})"
                continue
            if op.kind == "ttg.convert_layout":
                # Layout conversions are implicit; alias to operand 0.
                inner = _render_operand(op.operands[0], rctx)
                rctx.op_var[op.op_id] = inner
                continue
            if op.kind == "tt.descriptor_store":
                desc = _render_operand(op.operands[0], rctx)
                value_expr = _render_operand(op.operands[1], rctx)
                offsets = [_render_operand(o, rctx) for o in op.operands[2:]]
                offs_str = ", ".join(offsets)
                lines += f"tlx.local_store(c_smem[0], {value_expr})"
                lines += "tlx.fence_async_shared()"
                lines += f"tlx.async_descriptor_store({desc}, c_smem[0], [{offs_str}])"
                lines += "tlx.async_descriptor_store_wait(0)"
                continue
            if op.kind in _NAMED_FUNCTION_OPS:
                name = _auto_name(op, rctx.fresh_idx())
                rctx.op_var[op.op_id] = name
                lines += f"{name} = {_render_op_expr(op, rctx)}"


def _op_depends_on_iv(op_id: str, g: ScheduleGraph, lid: int, cache: dict) -> bool:
    """True if op `op_id` (transitively) references loop `lid`'s induction var."""
    if op_id in cache:
        return cache[op_id]
    cache[op_id] = False  # guard against cycles
    op = g.ops.get(op_id)
    res = False
    if op is not None:
        for o in op.operands:
            if isinstance(o, IvRef) and o.loop_id == lid:
                res = True
                break
            if isinstance(o, OpRef) and _op_depends_on_iv(o.op_id, g, lid, cache):
                res = True
                break
    cache[op_id] = res
    return res


def _iv_dep_op_subtree(load_op: Op, g: ScheduleGraph, lid: int) -> set[str]:
    """Collect all op_ids in the load's offset-operand subtrees that depend on
    the IV — these must bypass the var cache when re-rendering at a shifted IV."""
    cache: dict = {}
    out: set[str] = set()
    stack = [o for o in load_op.operands[1:] if isinstance(o, OpRef)]
    while stack:
        ref = stack.pop()
        if ref.op_id in out:
            continue
        if not _op_depends_on_iv(ref.op_id, g, lid, cache):
            continue
        out.add(ref.op_id)
        op = g.ops.get(ref.op_id)
        if op is not None:
            for o in op.operands:
                if isinstance(o, OpRef):
                    stack.append(o)
    return out


def _render_load_offsets_at(load_op: Op, g: ScheduleGraph, rctx: RenderCtx, lid: int, iv_expr: str) -> list[str]:
    """Render the load's offset operands as-if the induction var = `iv_expr`
    (e.g. `(tile_id + nprog)` for the next-tile prefetch). Bypasses the var
    cache for IV-dependent offset ops so they inline-render with the shift."""
    saved_iv = rctx.loop_iv.get(lid)
    saved_op_var = rctx.op_var
    bypass = _iv_dep_op_subtree(load_op, g, lid)
    rctx.loop_iv[lid] = iv_expr
    rctx.op_var = {k: v for k, v in saved_op_var.items() if k not in bypass}
    try:
        offs = [_render_operand(o, rctx) for o in load_op.operands[1:]]
    finally:
        rctx.op_var = saved_op_var
        if saved_iv is not None:
            rctx.loop_iv[lid] = saved_iv
        else:
            rctx.loop_iv.pop(lid, None)
    return offs


def _register_consumed_loads(loop: Loop, g: ScheduleGraph):
    """In-loop tt.descriptor_load nodes whose result is consumed directly in
    registers (no downstream ttg.local_alloc) → candidates for prefetch
    conversion (async double-buffered SMEM ring)."""
    out = []
    for n in loop.schedule.nodes:
        if n.op_kind != "tt.descriptor_load" or not n.op_ref:
            continue
        consumed_by_alloc = any(op.kind == "ttg.local_alloc" and op.operands and isinstance(op.operands[0], OpRef)
                                and op.operands[0].op_id == n.op_ref for op in g.ops.values())
        if not consumed_by_alloc:
            out.append(n)
    return out


def _emit_warp_group(
    g: ScheduleGraph,
    loop: Loop,
    wg: WarpGroup,
    channels: list[Channel],
    rctx: RenderCtx,
    lines: _Lines,
) -> None:
    nodes = _nodes_in_warp_group(loop, wg.id)
    if not nodes:
        return

    # Filter to ops that emit something. Allocs are SSA-only.
    emit_nodes = [n for n in nodes if n.op_kind not in ("ttg.local_alloc", "ttng.tmem_alloc", "scf.yield")]
    if not emit_nodes:
        return

    # Pick a depth for the K-loop ring buffer index: max depth among
    # buffers this WG produces or consumes, OR cross-WG channels involving it.
    touched_buf_ids: set[int] = set()
    for n in nodes:
        if n.produces_buffer is not None:
            touched_buf_ids.add(n.produces_buffer)
        for b in n.consumes_buffers:
            touched_buf_ids.add(b)
    # Also include channels where this WG is producer or consumer — for MEM
    # WGs this catches the load → alloc chain (the load doesn't directly
    # `produces_buffer` since the buffer is created by the consumer alloc).
    for ch in channels:
        if ch.producer_wg == wg.id or ch.consumer_wg == wg.id:
            for b in loop.schedule.buffers:
                if b.id in (None, ):
                    continue
                if (rctx.buffer_var.get((loop.loop_id, b.id), f"L{loop.loop_id}_{_buffer_var_name(b)}") == ch.name):
                    touched_buf_ids.add(b.id)
    # For a MEM-only WG (descriptor_load split out from the alloc/MMA WG),
    # follow each load → downstream local_alloc to find the actual data
    # buffer's count. Without this the rep_depth falls back to the synthesized
    # routing buffer's depth=1 and the load writes always land on slot 0,
    # corrupting depth>1 ring buffers (e.g., V tile in FA).
    for n in nodes:
        if n.op_kind != "tt.descriptor_load" or not n.op_ref:
            continue
        for oid, op in g.ops.items():
            if op.kind != "ttg.local_alloc" or not op.operands:
                continue
            opd0 = op.operands[0]
            if not (isinstance(opd0, OpRef) and opd0.op_id == n.op_ref):
                continue
            for b in loop.schedule.buffers:
                if b.def_op == oid:
                    touched_buf_ids.add(b.id)
    depths = [
        b.count for b in loop.schedule.buffers if b.id in touched_buf_ids and b.kind in ("smem", "tmem") and b.count > 1
    ]
    rep_depth = max(depths) if depths else 1
    iv = loop.schedule.induction_var_name
    # Preserve the IV's MLIR semantic value (= byte offset into K), so the
    # in-loop offsets that reference iv don't need a `* step` multiplier.
    step_expr = _render_operand(loop.schedule.step, rctx)

    # Determine if this WG owns the PRIMARY accumulator — i.e., contains an
    # MMA whose destination operand resolves to the function-scope `acc_tmem`
    # (the running output the epilogue reads). Multiple WGs may have MMAs
    # (case3 FA: QK in wg1 → acc_tmem_4 (scratch); PV in wg3 → acc_tmem),
    # but only the one writing to `acc_tmem` should signal acc_tmem_full
    # at end-of-loop. Otherwise the default partition wakes early.
    has_mma = False
    for n in nodes:
        if n.op_kind != "ttng.tc_gen5_mma" or not n.op_ref:
            continue
        mma_op = g.ops.get(n.op_ref)
        if mma_op is None or len(mma_op.operands) < 3:
            continue
        dest = mma_op.operands[2]
        if isinstance(dest, OpRef) and rctx.alloc_op_var.get(dest.op_id) == "acc_tmem":
            has_mma = True
            break

    # Caller has already opened the per-WG `with tlx.async_task(...):` —
    # body emitted directly here.
    # Function-scope per-tile-resident loads (e.g. FA's Q tile): emit the
    # TMA load in the MEM-role WG (= the WG that has TMA descriptor_load
    # nodes), and emit the consumer-side wait in any WG whose MMA reads the
    # corresponding alloc. Done BEFORE the K-loop body.
    # Function-scope TMA loads (e.g. Q tile in non-persistent FA) are
    # OUTER-scope ops, not in any inner-loop WG's nodes. Each such load
    # must be emitted EXACTLY ONCE across all MEM WGs — emitting in
    # every MEM WG (the old `is_mem_wg = any(descriptor_load in nodes)`
    # check) duplicates the TMA write to the same SMEM and deadlocks
    # the kernel. Track emission state on rctx so the first MEM WG
    # claims the load and later MEM WGs skip it.
    if not hasattr(rctx, "_fn_load_emitted"):
        rctx._fn_load_emitted = set()
    is_mem_wg = any(n.op_kind == "tt.descriptor_load" for n in nodes)
    mma_alloc_op_ids: set[str] = set()
    for n in nodes:
        if n.op_kind != "ttng.tc_gen5_mma" or not n.op_ref:
            continue
        mma_op = g.ops.get(n.op_ref)
        if mma_op is None:
            continue
        for src_idx in (0, 1):
            if len(mma_op.operands) > src_idx and isinstance(mma_op.operands[src_idx], OpRef):
                mma_alloc_op_ids.add(mma_op.operands[src_idx].op_id)
    for fl in rctx.fn_scope_loads:
        load_op = fl["load_op"]
        already_emitted = fl["load_op_id"] in rctx._fn_load_emitted
        if is_mem_wg and not already_emitted:
            rctx._fn_load_emitted.add(fl["load_op_id"])
            # Emit the TMA load in the WG that owns it.
            desc = _render_operand(load_op.operands[0], rctx)
            offsets = [_render_operand(o, rctx) for o in load_op.operands[1:]]
            offs_str = ", ".join(offsets)
            shape_dt = (_parse_tensor_shape(load_op.result_types[0]) if load_op.result_types else None)
            shape = shape_dt[0] if shape_dt else [0]
            dtype = shape_dt[1] if shape_dt else "f16"
            n_bytes = 1
            for d in shape:
                n_bytes *= int(d)
            n_bytes *= _bytes_per_elem_bits(16 if dtype in ("f16", "bf16") else 32)
            lines += "# load Q tile (per-tile resident)"
            lines += f"tlx.barrier_expect_bytes({fl['alloc_var']}_full[0], {n_bytes})"
            lines += (f"tlx.async_descriptor_load({desc}, {fl['alloc_var']}[0], "
                      f"[{offs_str}], {fl['alloc_var']}_full[0])")
        if fl["alloc_op_id"] in mma_alloc_op_ids:
            lines += (f"tlx.barrier_wait({fl['alloc_var']}_full[0], 0)  # wait Q tile loaded")
    if has_mma:
        # acc_tmem is a legacy carve-out under SemIR (cross-region edge,
        # not in the inner-loop's cross_wg_barriers). Still uses
        # full+empty pair convention.
        lines += "tlx.barrier_wait(acc_tmem_empty[0], 1)"
    if _use_semaphore_ir():
        # SemIR: loop-carry pre-arrives looked up by (loop, wg).
        for line in _semir_pre_arrives_for_wg(loop.loop_id, wg.id, rctx):
            lines += line
    else:
        # Loop-carry pre-arrive (partition-agnostic): for any cross-WG channel
        # consumed by this WG where producer_cycle > consumer_cycle within an
        # iteration, the dep is loop-carried (producer iter K → consumer iter
        # K+1). Iter 0's consumer would deadlock waiting for a producer that
        # hasn't run yet. Pre-arrive on `_full` once before the loop so iter 0
        # passes immediately. Lets the emitter handle ANY partition shape with
        # backward-edge channels (e.g., split-MMA partitions where qk-release
        # signals the next iter's Q@K slot).
        cycle_of = {n.id: n.schedule_cycle for n in loop.schedule.nodes}
        for c in channels:
            if c.loop_id is not None and c.loop_id != loop.loop_id:
                continue
            if c.consumer_wg != wg.id:
                continue
            if c.producer_node is None or c.consumer_node is None:
                continue
            prod_cyc = cycle_of.get(c.producer_node)
            cons_cyc = cycle_of.get(c.consumer_node)
            if prod_cyc is None or cons_cyc is None:
                continue
            if prod_cyc > cons_cyc:
                lines += (f"tlx.barrier_arrive({_bar_full(c.name)}[0], 1)  "
                          f"# loop-carry pre-arrive (producer cyc={prod_cyc} > "
                          f"consumer cyc={cons_cyc})")

    # Per-WG iter_arg trim (mirror of _emit_inner_loop_in_outer): keep only
    # iter_args this WG's ops actually consume. Init each before the loop;
    # reassign at end of body from the yield expression.
    iter_specs = _loop_iter_args(g, loop)
    wg_op_ids = {n.op_ref for n in nodes if n.op_ref}
    used_idxs: set[int] = set()
    for oid in wg_op_ids:
        op_obj = g.ops.get(oid)
        if op_obj is None:
            continue
        for o in op_obj.operands:
            if isinstance(o, IterArgRef) and o.loop_id == loop.loop_id:
                used_idxs.add(o.idx)
    kept = [(idx, init, yld, _iter_arg_python_name(loop.loop_id, idx, init))
            for (idx, init, yld) in iter_specs
            if idx in used_idxs]
    saved_iter_arg_var = dict(rctx.iter_arg_var)
    for idx, init, _yld, name in kept:
        rctx.iter_arg_var[(loop.loop_id, idx)] = name
        lines += f"{name} = {_render_operand(init, rctx)}"

    # M2 (load prefetch): in the single-WG path, convert register-consumed TMA
    # loads (load → reduce, no SMEM alloc) into async double-buffered SMEM rings:
    # prologue prefetches tile 0, the loop waits/consumes tile i and prefetches
    # tile i+1. Hides the load round-trip behind compute. Mirrors handwritten.py.
    prefetch_loads: list[dict] = []
    hi_b = None
    if getattr(rctx, "defer_inloop_store", False):
        NB = 2
        lo_b = _render_operand(loop.schedule.lower_bound, rctx)
        hi_b = _render_operand(loop.schedule.upper_bound, rctx)
        for i, ln in enumerate(_register_consumed_loads(loop, g)):
            lop = g.ops.get(ln.op_ref)
            if lop is None or not lop.result_types:
                continue
            sd = _parse_tensor_shape(lop.result_types[0])
            if not sd:
                continue
            shape, dtype = sd
            tl_dt = _dtype_str_to_tl(dtype)
            nbytes = 1
            for d in shape:
                nbytes *= int(d)
            nbytes *= _bytes_per_elem_bits(16 if dtype in ("f16", "bf16") else 32)
            shp = ", ".join(str(d) for d in shape)
            ring, bar = f"_pf{i}_buf", f"_pf{i}_bar"
            desc = _render_operand(lop.operands[0], rctx)
            offs0 = ", ".join(_render_load_offsets_at(lop, g, rctx, loop.loop_id, lo_b))
            lines += f"{ring} = tlx.local_alloc(({shp}), {tl_dt}, {NB})"
            lines += f"{bar} = tlx.alloc_barriers(num_barriers={NB})"
            with lines.block(f"if {lo_b} < {hi_b}:"):
                lines += f"tlx.barrier_expect_bytes({bar}[0], {nbytes})"
                lines += (f"tlx.async_descriptor_load({desc}, {ring}[0], [{offs0}], {bar}[0])")
            prefetch_loads.append({
                "node": ln,
                "op": lop,
                "ring": ring,
                "bar": bar,
                "nbytes": nbytes,
                "desc": desc,
                "NB": NB,
                "ld_var": f"_pf{i}_val",
            })
    prefetch_node_ids = {p["node"].op_ref for p in prefetch_loads}

    if True:
        with lines.block(f"for {iv} in {_loop_range_expr(loop, rctx)}:"):
            # Iteration count = (iv - lb) // step. Use it for ring-buffer index.
            lo = _render_operand(loop.schedule.lower_bound, rctx)
            lines += f"_it = ({iv} - {lo}) // {step_expr}"
            # `phase` MUST toggle per iteration even when ring depth=1 —
            # it's the parity that mbarriers use to detect the next phase.
            # For depth=N, phase advances every N iters (same buf slot revisits).
            lines += f"buf = _it % {rep_depth}"
            lines += f"phase = (_it // {rep_depth}) & 1"
            # Skip bridge ops (cross-WG TMEM tmem_alloc(value)): emitted as
            # local_store + barrier_arrive in the value producer's WG.
            bridge_op_ids = {c.bridge_op_id for c in rctx.channels if c.kind == "tmem" and c.bridge_op_id}
            # M2 (load prefetch): wait the current tile's load, bind its SSA var
            # to the local_load, and prefetch the next tile into the alternate
            # ring slot. The blocking descriptor_load node is skipped below.
            for p in prefetch_loads:
                nbv = p["NB"]
                lines += f"_pf_slot = _it % {nbv}"
                lines += f"_pf_phase = (_it // {nbv}) & 1"
                lines += f"tlx.barrier_wait({p['bar']}[_pf_slot], _pf_phase)"
                lines += f"{p['ld_var']} = tlx.local_load({p['ring']}[_pf_slot])"
                rctx.op_var[p["op"].op_id] = p["ld_var"]
                next_iv = f"({iv} + {step_expr})"
                offs_n = ", ".join(_render_load_offsets_at(p["op"], g, rctx, loop.loop_id, next_iv))
                lines += f"_pf_nslot = (_it + 1) % {nbv}"
                with lines.block(f"if {next_iv} < {hi_b}:"):
                    lines += f"tlx.barrier_expect_bytes({p['bar']}[_pf_nslot], {p['nbytes']})"
                    lines += (f"tlx.async_descriptor_load({p['desc']}, {p['ring']}[_pf_nslot], "
                              f"[{offs_n}], {p['bar']}[_pf_nslot])")
            for n in emit_nodes:
                if n.op_ref in bridge_op_ids or n.op_ref in prefetch_node_ids:
                    continue
                _emit_in_loop_node(n, g, loop, channels, rctx, lines)
            # Recurrence: reassign iter_args from yields (Triton folds
            # these into iter_args automatically inside @triton.jit).
            for idx, _init, yld, name in kept:
                lines += f"{name} = {_render_operand(yld, rctx)}"

        # M1 (store deferral): drain the last in-loop TMA store issued with the
        # deferred-wait pattern (wait moved to the top of the next iteration).
        if getattr(rctx, "_needs_store_drain", False):
            lines += "tlx.async_descriptor_store_wait(0)  # drain final deferred store"
            rctx._needs_store_drain = False

        # Cross-loop iter_arg result channel: this WG owns one or more
        # iter_args whose final values the default partition's epilogue
        # reads. Stage them through SMEM with a barrier_arrive after the
        # loop ends.
        for ch in rctx.crossloop_channels:
            if ch["loop_id"] != loop.loop_id:
                continue
            if ch["producer_wg"] != wg.id:
                continue
            lines += f"tlx.local_store({ch['bufname']}[0], {ch['var_name']})"
            lines += f"tlx.barrier_arrive({_bar_full(ch['bufname'])}[0], 1)"

        rctx.iter_arg_var = saved_iter_arg_var
        if has_mma:
            # acc_tmem is a legacy carve-out under SemIR — full+empty pair.
            # tcgen05_commit makes acc_tmem_full track completion of all
            # prior async tcgen5 ops in this WG (the PV MMAs). Without it,
            # a plain barrier_arrive races with the still-pending HW MMA.
            lines += "tlx.tcgen05_commit(acc_tmem_full[0])"


def _use_acc_expr(op: Op, loop: Loop, rctx: RenderCtx) -> str:
    """Render the MMA's use_acc expression.

    The MMA's useD operand (operands[4]) is the static IR value — but for
    running-accumulator MMAs, the IR-level useD is a constant True (or an
    iter_arg) and the loop-init semantics are implicit. We need to emit
    `(iv > lb)` so the FIRST iter initializes the accumulator, regardless.

    Only when useD is statically False (scratch MMA, e.g. FA's QK that
    overwrites its destination every iter) do we honor the constant.
    """
    if len(op.operands) > 4:
        useD = op.operands[4]
        if isinstance(useD, ConstRef) and useD.value in (0, False, None, "0"):
            return "False"  # scratch MMA: always overwrite (no accumulation)
    iv = loop.schedule.induction_var_name
    lo = _render_operand(loop.schedule.lower_bound, rctx)
    return f"({iv} > {lo})"


def _emit_in_loop_node(
    n: Node,
    g: ScheduleGraph,
    loop: Loop,
    channels: list[Channel],
    rctx: RenderCtx,
    lines: _Lines,
) -> None:
    op = g.ops.get(n.op_ref) if n.op_ref else None
    if op is None:
        return

    if _use_semaphore_ir():
        # SemIR-driven consumer block: emit waits + local_loads + recycle
        # arrives for every semaphore where this node is a consumer.
        _semir_emit_consumer_block(n, g, loop, rctx, lines)
    else:
        # Schedule-driven cross-WG channels (Pass B Step 2): inject barrier_wait
        # + local_load BEFORE this node when it's a consumer; bind the producer's
        # op_id to the loaded variable so any operand referencing the producer
        # (across the WG boundary) resolves to the loaded value, not a recursive
        # render of the producer's chain.
        # Filter to channels owned by THIS loop (outer-loop channels would
        # otherwise spuriously fire inside the inner body).
        for c in rctx.channels:
            if c.loop_id is not None and c.loop_id != loop.loop_id:
                continue
            if c.consumer_node == n.id and c.producer_node is not None:
                prod_op = next(
                    (nn.op_ref for nn in loop.schedule.nodes if nn.id == c.producer_node),
                    None,
                )
                if prod_op is None:
                    continue
                # If a TMEM bridge already handles this producer (the TMEM
                # tmem_alloc(value) absorbs the same SSA value), skip the SMEM
                # channel — TMEM is the natural staging for MMA-bound values.
                already_tmem = any(
                    tc.kind == "tmem" and tc.bridge_op_id and ((lambda bop: bop and bop.operands and isinstance(
                        bop.operands[0], OpRef) and bop.operands[0].op_id == prod_op)(g.ops.get(tc.bridge_op_id)))
                    for tc in rctx.channels)
                if already_tmem:
                    continue
                # Detect backward-edge / loop-carry channels (producer_cycle >
                # consumer_cycle within an iter): these are release SIGNALS
                # not data transfers — emit barrier-only, no load/store.
                prod_cyc = next(
                    (nn.schedule_cycle for nn in loop.schedule.nodes if nn.id == c.producer_node),
                    None,
                )
                cons_cyc = next(
                    (nn.schedule_cycle for nn in loop.schedule.nodes if nn.id == c.consumer_node),
                    None,
                )
                is_loop_carry = (prod_cyc is not None and cons_cyc is not None and prod_cyc > cons_cyc)
                if c.kind == "named" or is_loop_carry:
                    # Signal-only: just wait for producer signal. No buffer
                    # load (the synthesized buffer for loop-carry channels is
                    # an artifact — it may even alias another buffer's storage,
                    # which would CORRUPT memory if we wrote/read it).
                    kind_note = ("named-channel" if c.kind == "named" else "loop-carry release")
                    lines += (f"tlx.barrier_wait({_bar_full(c.name)}[0], _it & 1)  "
                              f"# {kind_note} (n{c.producer_node}→n{c.consumer_node})")
                    continue
                seed = int(rctx.op_var.get("__inloop_var_seed__", "0"))
                chan_var = f"chan_{c.name}_{seed}"
                rctx.op_var["__inloop_var_seed__"] = str(seed + 1)
                # Cross-WG channel barriers are count=1 → toggle every iter
                # (use `_it & 1`), NOT the ring-buffer `phase` which only
                # toggles every `depth` iters.
                lines += f"tlx.barrier_wait({_bar_full(c.name)}[0], _it & 1)"
                lines += f"{chan_var} = tlx.local_load({c.name}[0])"
                lines += f"tlx.barrier_arrive({_bar_empty(c.name)}[0], 1)"
                rctx.op_var[prod_op] = chan_var

    _load_buf = (_find_load_target_buffer(n, op, g, loop) if n.op_kind == "tt.descriptor_load" else None)
    if _load_buf is None and n.op_kind == "tt.descriptor_load":
        # No original staging buffer — the load result is a register tile that
        # the schedule routes to a consumer in ANOTHER warp group (LayerNorm:
        # load → compute). The C++ partition synthesized an SMEM channel for
        # that cross-WG flow; stage the load INTO it so the consumer WG can
        # local_load it. Without this the load would fall through to the
        # register-returning `.load()` and never signal the channel (deadlock).
        _load_buf = _find_channel_producer_buffer(n, loop)
    if n.op_kind == "tt.descriptor_load" and _load_buf is not None:
        # Load that feeds an SMEM buffer (original staging for GEMM/FA, or a
        # synthesized cross-WG channel for register-consumed loads): stage it
        # through SMEM here, signalling the buffer's full barrier.
        buf = _load_buf
        buf_var = rctx.buffer_var.get((loop.loop_id, buf.id), f"L{loop.loop_id}_{_buffer_var_name(buf)}")
        desc = _render_operand(op.operands[0], rctx)
        offsets = [_render_operand(o, rctx) for o in op.operands[1:]]
        offs_str = ", ".join(offsets)
        if _use_semaphore_ir():
            # SemIR path: producer wait empty (if any), expect_bytes, then
            # async_descriptor_load with the semaphore as the mbarrier arg.
            # Buffer slot uses the BUFFER's ring count (buf.count); the
            # barrier slot uses the semaphore depth (often "0" for depth=1).
            for w in _semir_producer_waits(loop.loop_id, n.id, rctx):
                lines += w
            for e in _semir_producer_expect_bytes(loop.loop_id, n.id, rctx):
                lines += e
            bar_arg = _semir_producer_barrier_for_tma(loop.loop_id, n.id, rctx)
            data_slot = "buf" if buf is not None and buf.count > 1 else "0"
            lines += f"# load → {buf_var}"
            if bar_arg is None:
                # No cross-WG semaphore for this TMA (producer + consumer in
                # the same WG). The intra-WG `<buf>_full` barrier was
                # allocated in the `extra` carve-out — use it: wait empty,
                # expect bytes, load with full as the mbarrier arg.
                ph = "(_it & 1)" if buf.count == 1 else "phase"
                idx = "0" if buf.count == 1 else "buf"
                nbytes = ((buf.shape[0] * buf.shape[1] *
                           _bytes_per_elem_bits(buf.element_bits)) if len(buf.shape) >= 2 else buf.size_bytes)
                full = _bar_full(buf_var)
                empty = _bar_empty(buf_var)
                lines += f"tlx.barrier_wait({empty}[{idx}], {ph} ^ 1)"
                lines += f"tlx.barrier_expect_bytes({full}[{idx}], {nbytes})"
                bar_arg = f"{full}[{idx}]"

            # Pass A.5: TMA load is NOT split — the descriptor's block shape
            # is the full (BM, BK), so the SMEM destination must match. The
            # MMA emit takes per-group `tlx.local_slice` views to feed the
            # N async_dot calls.
            lines += (f"tlx.async_descriptor_load({desc}, {buf_var}[{data_slot}], "
                      f"[{offs_str}], {bar_arg})")
            return
        # Legacy: per-buffer index AND phase. The loop's `buf` and `phase` are sized
        # for the loop's max-depth ring. A count=1 buffer needs:
        #   index = [0]  (single slot, not [buf] which can overrun)
        #   phase = `_it & 1`  (barrier flips every iter, not every `depth` iters)
        # Without this, the consumer-signaled empty barrier appears never to
        # release on subsequent iters → producer wait blocks → kernel hang
        # OR illegal barrier op if the state goes inconsistent.
        if buf.count == 1:
            idx = "0"
            ph = "(_it & 1)"
        else:
            idx = "buf"
            ph = "phase"
        nbytes = ((buf.shape[0] * buf.shape[1] *
                   _bytes_per_elem_bits(buf.element_bits)) if len(buf.shape) >= 2 else buf.size_bytes)
        full = _bar_full(buf_var)
        empty = _bar_empty(buf_var)
        lines += f"# load → {buf_var}"
        lines += f"tlx.barrier_wait({empty}[{idx}], {ph} ^ 1)"
        lines += f"tlx.barrier_expect_bytes({full}[{idx}], {nbytes})"
        lines += (f"tlx.async_descriptor_load({desc}, {buf_var}[{idx}], "
                  f"[{offs_str}], {full}[{idx}])")
        return

    if n.op_kind == "ttng.tc_gen5_mma":
        # operands[0,1] = SMEM allocs (a, b); operands[2] = TMEM acc; etc.
        # B may go through ttg.memdesc_trans — walk through it.
        a_buf, a_loop = _resolve_alloc_to_buffer(op.operands[0], g, loop)
        b_buf, b_loop = _resolve_alloc_to_buffer(op.operands[1], g, loop)
        b_via_trans = False
        if b_buf is None and isinstance(op.operands[1], OpRef):
            mid_op = g.ops.get(op.operands[1].op_id)
            if mid_op and mid_op.kind == "ttg.memdesc_trans" and mid_op.operands:
                b_buf, b_loop = _resolve_alloc_to_buffer(mid_op.operands[0], g, loop)
                b_via_trans = True

        # Outer-loop SMEM (e.g. case3 Q tile) is per-tile resident — index
        # is always 0; no `[buf]` ring slot. Inner-loop SMEM uses the ring
        # counter `buf` (set up at top of inner-loop body) UNLESS the buffer
        # has count=1 (then [0] always, since the ring has only one slot).
        # Phase MUST match: count=1 → `_it & 1` (barrier flips every iter);
        # count=N → loop-wide `phase = (_it // N) & 1`.
        def _ring_idx_phase(buf, lp):
            if lp is not loop:
                return "0", "0"
            if buf is not None and buf.count == 1:
                return "0", "(_it & 1)"
            return "buf", "phase"

        a_idx, a_ph = _ring_idx_phase(a_buf, a_loop)
        b_idx, b_ph = _ring_idx_phase(b_buf, b_loop)

        # Resolve operand names; fallback to alloc_op_var for ops whose
        # alloc lives at function scope (no loop owns it — e.g. case3 nows
        # Q SMEM hoisted as `q_smem_*`).
        def _resolve_alloc_var(operand: OperandRef, buf, lp) -> str:
            if buf is not None and lp is not None:
                return rctx.buffer_var.get((lp.loop_id, buf.id), f"L{lp.loop_id}_{_buffer_var_name(buf)}")
            if isinstance(operand, OpRef) and operand.op_id in rctx.alloc_op_var:
                return rctx.alloc_op_var[operand.op_id]
            return "<a?>"

        a_var = _resolve_alloc_var(op.operands[0], a_buf, a_loop)
        b_var = _resolve_alloc_var(op.operands[1], b_buf, b_loop)
        if b_via_trans and isinstance(op.operands[1], OpRef):
            mid_op = g.ops.get(op.operands[1].op_id)
            if mid_op and mid_op.kind == "ttg.memdesc_trans" and mid_op.operands:
                b_var = _resolve_alloc_var(mid_op.operands[0], b_buf, b_loop)
        lines += "# MMA"
        # Determine destination TMEM acc and name, regardless of barrier path.
        b_expr_pre = (f"tlx.local_trans({b_var}[{b_idx}])" if b_via_trans else f"{b_var}[{b_idx}]")
        acc_idx = "tmem_buf" if rctx.tmem_count > 1 else "0"
        dest_var = "acc_tmem"
        dest_op_id = None
        if len(op.operands) >= 3 and isinstance(op.operands[2], OpRef):
            dest_op_id = op.operands[2].op_id
            dest_var = rctx.alloc_op_var.get(dest_op_id, dest_var)

        if _use_semaphore_ir():
            # SemIR: consumer waits + mBarriers cover four sources —
            #  (1) operand SMEM buffers (cross_wg_barriers names the
            #      local_alloc as consumer, MMA is the actual data reader),
            #  (2) cross-WG signal semaphores where THIS MMA is consumer
            #      (e.g., named TMEM-handoff semaphores from softmax),
            #  (3) TMEM channel operands (e.g., PV MMA reading P_tmem written
            #      by softmax via the bridge) — full+empty pair, legacy carve.
            #  (4) the MMA's own producer-side semaphores (acc TMEM signals).
            opnd_waits, opnd_mbar = _semir_mma_operand_waits_and_mbarriers(op, g, loop, rctx)
            for w in opnd_waits:
                lines += w
            # TMEM bridge operand: wait the bridge's full barrier; the empty
            # side goes into the MMA's mBarriers list for HW recycle.
            tmem_chan_for_recycle: list[str] = []
            for src_idx in (0, 1):
                if not (len(op.operands) > src_idx and isinstance(op.operands[src_idx], OpRef)):
                    continue
                src_id = op.operands[src_idx].op_id
                for c in rctx.channels:
                    if c.kind == "tmem" and c.alloc_op_id == src_id:
                        lines += (f"tlx.barrier_wait({_bar_full(c.name)}[0], "
                                  f"_it & 1)  # TMEM bridge operand")
                        tmem_chan_for_recycle.append(f"{_bar_empty(c.name)}[0]")
            for w in _semir_consumer_waits(loop.loop_id, n.id, rctx):
                lines += w
            lines += f"use_acc = {_use_acc_expr(op, loop, rctx)}"
            mbar_list: list[str] = []
            mbar_list.extend(opnd_mbar)
            mbar_list.extend(tmem_chan_for_recycle)
            mbar_list.extend(_semir_consumer_mbarriers(loop.loop_id, n.id, rctx))
            mbar_list.extend(_semir_producer_mbarriers(loop.loop_id, n.id, rctx))

            # Pass A.5: when the MMA is partitioned, fan out to N async_dot
            # calls. Each call takes a `tlx.local_slice` view of the shared
            # A SMEM (M-axis offset = gi*m_size) and writes its own TMEM
            # accumulator. The same mBarriers list (including SMEM _empty)
            # is attached to every call — the empty barrier's arrive_count
            # was set to N at allocation time.
            if n.partition_count > 1:
                dest_names = (rctx.partition_alloc_names.get(dest_op_id) if dest_op_id else None)
                N = n.partition_count
                m_size = n.m_size
                if dest_names is None or len(dest_names) != N or m_size <= 0:
                    lines += (f"# WARNING: partition_count={N} but TMEM per-group "
                              f"names missing; falling back to single MMA")
                    if mbar_list:
                        mb = ", ".join(mbar_list)
                        lines += (f"tlx.async_dot({a_var}[{a_idx}], {b_expr_pre}, "
                                  f"{dest_var}[{acc_idx}], use_acc=use_acc, "
                                  f"mBarriers=[{mb}])")
                    else:
                        lines += (f"tlx.async_dot({a_var}[{a_idx}], {b_expr_pre}, "
                                  f"{dest_var}[{acc_idx}], use_acc=use_acc)")
                    return
                # A SMEM is shared (full BM tile); slice along M for each group.
                bk = (a_buf.shape[1] if (a_buf is not None and len(a_buf.shape) >= 2) else 0)
                for gi in range(N):
                    dg = dest_names[gi]
                    a_view = (f"tlx.local_slice({a_var}[{a_idx}], "
                              f"[{gi * m_size}, 0], [{m_size}, {bk}])")
                    if mbar_list:
                        mb = ", ".join(mbar_list)
                        lines += (f"tlx.async_dot({a_view}, {b_expr_pre}, "
                                  f"{dg}[{acc_idx}], use_acc=use_acc, "
                                  f"mBarriers=[{mb}])")
                    else:
                        lines += (f"tlx.async_dot({a_view}, {b_expr_pre}, "
                                  f"{dg}[{acc_idx}], use_acc=use_acc)")
                return

            if mbar_list:
                mb = ", ".join(mbar_list)
                lines += (f"tlx.async_dot({a_var}[{a_idx}], {b_expr_pre}, "
                          f"{dest_var}[{acc_idx}], use_acc=use_acc, mBarriers=[{mb}])")
            else:
                lines += (f"tlx.async_dot({a_var}[{a_idx}], {b_expr_pre}, "
                          f"{dest_var}[{acc_idx}], use_acc=use_acc)")
            return

        # ── Legacy channel-based path ─────────────────────────────────────
        # Only inner-loop SMEM has a per-iter producer/consumer barrier
        # (load → MMA in same K-iter). Outer-loop SMEM (Q tile) is resident
        # — there's a one-time q_full barrier separate from the K-iter ring.
        if a_buf and a_loop is loop:
            lines += f"tlx.barrier_wait({_bar_full(a_var)}[{a_idx}], {a_ph})"
        if b_buf and b_loop is loop and (a_buf is None or b_buf.id != a_buf.id):
            lines += f"tlx.barrier_wait({_bar_full(b_var)}[{b_idx}], {b_ph})"
        # Cross-WG TMEM consumer: this MMA reads a TMEM operand whose value
        # was produced in another WG via the channel's `_full` mbarrier.
        # Skip allocs already in this loop's buffers — the SMEM-style
        # barrier above already covers them with `[buf]` indexing.
        local_buf_def_ops = {b.def_op for b in loop.schedule.buffers}
        for src_idx in (0, 1):
            if not (len(op.operands) > src_idx and isinstance(op.operands[src_idx], OpRef)):
                continue
            src_id = op.operands[src_idx].op_id
            if src_id in local_buf_def_ops:
                continue
            for c in rctx.channels:
                if c.kind == "tmem" and c.alloc_op_id == src_id:
                    # Channel barriers are count=1 → toggle every iter.
                    lines += f"tlx.barrier_wait({_bar_full(c.name)}[0], _it & 1)"
        lines += f"use_acc = {_use_acc_expr(op, loop, rctx)}"
        b_expr = b_expr_pre
        # Build the MMA's `mBarriers=` list — these are HARDWARE-issued
        # arrivals that fire when the tensor core actually completes. We
        # include:
        #   (1) TMEM `_full` for cross-WG-consumed produced acc — reader WG
        #       knows when result lands.
        #   (2) SMEM `_empty` for each consumed operand (A, B) — producer
        #       WG knows when the MMA hardware is DONE reading and the
        #       buffer can be overwritten.
        #   (3) Named-channel `_full` for each cross-WG signal where THIS
        #       MMA is the producer node — consumer WG knows when MMA
        #       result is available (e.g., qk_tmem ready for softmax).
        # All three must be on mBarriers (not separate `barrier_arrive`
        # lines) because async_dot is fire-and-forget: a software arrive
        # after the call would fire ~immediately, before the MMA hardware
        # actually finishes — racing the consumer/producer to a buffer
        # → "Illegal barrier arrive" or data corruption. Hand-written FA
        # uses this exact `mBarriers=[acc_full, kv_empties[k_buf]]` idiom.
        mbar_list = []
        # (1) Cross-WG TMEM producer barriers.
        producer_chans = [c for c in rctx.channels if c.kind == "tmem" and c.alloc_op_id == dest_op_id]
        for c in producer_chans:
            mbar_list.append(f"{_bar_full(c.name)}[0]")
        # (2) SMEM operand `_empty` barriers — released by HW when reads
        # complete. Skip when operand is outer-loop / function-scope
        # resident (no per-iter ring barrier).
        if a_buf and a_loop is loop:
            mbar_list.append(f"{_bar_empty(a_var)}[{a_idx}]")
        if b_buf and b_loop is loop and (a_buf is None or b_buf.id != a_buf.id):
            mbar_list.append(f"{_bar_empty(b_var)}[{b_idx}]")
        # (3) Named-channel signals where THIS MMA is the producer node.
        # Match by node id (which the schedule pass populated from the
        # cross_wg_barriers entry). Includes loop-carry signals — the
        # consumer's WG preamble pre-arrives so iter 0 doesn't deadlock.
        for c in rctx.channels:
            if c.kind != "named":
                continue
            if c.loop_id is not None and c.loop_id != loop.loop_id:
                continue
            if c.producer_node == n.id:
                mbar_list.append(f"{_bar_full(c.name)}[0]")
        if mbar_list:
            mb = ", ".join(mbar_list)
            lines += (f"tlx.async_dot({a_var}[{a_idx}], {b_expr}, {dest_var}[{acc_idx}], "
                      f"use_acc=use_acc, mBarriers=[{mb}])")
        else:
            lines += f"tlx.async_dot({a_var}[{a_idx}], {b_expr}, {dest_var}[{acc_idx}], use_acc=use_acc)"
        # NOTE: no explicit `barrier_arrive` for the SMEM operand `_empty`
        # barriers — the MMA hardware signals them via mBarriers above.
        return

    # Side-effect ops (no result, write to memory).
    if n.op_kind == "ttng.tmem_store":
        # MLIR: `ttng.tmem_store %dest, %token, %value, %pred`. Operand[2]
        # is the value; we ignore the token (handled by SSA) and predicate.
        dest = _render_operand(op.operands[0], rctx)
        value = _render_operand(op.operands[2], rctx) if len(op.operands) > 2 else "?"
        if _use_semaphore_ir():
            # SemIR: producer waits empty, writes to TMEM, signals consumer.
            # NOTE: use plain barrier_arrive, NOT tcgen05_commit. local_store
            # on TMEM (tcgen5.st) is synchronous from the issuing warp's POV;
            # tcgen05_commit ties the barrier to in-flight async tcgen5.mma
            # completion, but with no async op pending it becomes a no-op and
            # the barrier never fires — deadlocks the consumer. (Caught at
            # scale: case3 FA fwd, 2048+ CTAs, large N_CTX.)
            for w in _semir_producer_waits(loop.loop_id, n.id, rctx):
                lines += w
            lines += f"tlx.local_store({dest}, {value})"
            for a in _semir_producer_arrives(loop.loop_id, n.id, rctx):
                lines += a
            return
        lines += f"tlx.local_store({dest}, {value})"
        # Side-effect ops don't go through the named-value producer-channel
        # scan (which only fires for ops emitting a Python variable). We
        # explicitly handle the named-channel signal here for tmem_store
        # producers (e.g., case3's acc-store-token signal to the next-WG
        # P@V MMA).
        for c in rctx.channels:
            if c.kind != "named":
                continue
            if c.loop_id is not None and c.loop_id != loop.loop_id:
                continue
            if c.producer_node == n.id:
                lines += (f"tlx.barrier_arrive({_bar_full(c.name)}[0], 1)  "
                          f"# named-channel signal (n{c.producer_node}→n{c.consumer_node})")
        return
    if n.op_kind == "tt.store":
        ptr = _render_operand(op.operands[0], rctx)
        value = _render_operand(op.operands[1], rctx)
        lines += f"tl.store({ptr}, {value})"
        return
    if n.op_kind == "tt.descriptor_store":
        # Cross-WG channel store: the producer WG already wrote the value into
        # the channel SMEM (and fenced it for the async proxy). TMA-store
        # straight from that buffer — no local_store, no register round-trip,
        # no separate staging buffer — then recycle the channel after drain.
        sc = getattr(rctx, "_store_from_channel", None)
        if sc is not None:
            rctx._store_from_channel = None
            desc = _render_operand(op.operands[0], rctx)
            offs_str = ", ".join(_render_operand(o, rctx) for o in op.operands[2:])
            lines += (f"tlx.async_descriptor_store({desc}, "
                      f"{sc['buf_var']}[{sc['slot']}], [{offs_str}])")
            lines += "tlx.async_descriptor_store_wait(0)"
            if sc["arrive"]:
                lines += sc["arrive"]
            return
        # In-loop TMA store: stage the register value into its SMEM buffer,
        # then async_descriptor_store. operands = [desc, value, off0, off1...].
        # (Outer-loop epilogue stores are handled separately in the
        # outer-epilogue emitter; this is the in-persistent-loop path.)
        desc = _render_operand(op.operands[0], rctx)
        value_expr = _render_operand(op.operands[1], rctx)
        offs_str = ", ".join(_render_operand(o, rctx) for o in op.operands[2:])
        bid = op.attributes.get("buffer.id")
        buf = next((b for b in loop.schedule.buffers if b.id == bid), None)
        if buf is not None:
            buf_var = rctx.buffer_var.get((loop.loop_id, buf.id), f"L{loop.loop_id}_{_buffer_var_name(buf)}")
            slot = "buf" if buf.count > 1 else "0"
        else:
            buf_var, slot = "c_smem", "0"
        if getattr(rctx, "defer_inloop_store", False):
            # M1 (store deferral): don't block on this store. Wait on the
            # PREVIOUS iteration's store right before reusing the staging buffer
            # — that wait overlaps this iteration's load+compute, taking the
            # store latency off the critical path. The final store is drained
            # after the loop (see _emit_warp_group). Iter 0's wait is a no-op
            # (0 stores in flight). Single-buffer-safe (depth-1).
            lines += "tlx.async_descriptor_store_wait(0)  # drain prev store (deferred)"
            lines += f"tlx.local_store({buf_var}[{slot}], {value_expr})"
            lines += "tlx.fence_async_shared()"
            lines += (f"tlx.async_descriptor_store({desc}, {buf_var}[{slot}], [{offs_str}])")
            rctx._needs_store_drain = True
            return
        lines += f"tlx.local_store({buf_var}[{slot}], {value_expr})"
        lines += "tlx.fence_async_shared()"
        lines += f"tlx.async_descriptor_store({desc}, {buf_var}[{slot}], [{offs_str}])"
        lines += "tlx.async_descriptor_store_wait(0)"
        return
    # Multi-result ops.
    if n.op_kind == "tt.split":
        # Returns (left, right) — assign as tuple.
        if op.op_id in rctx.op_var:
            return
        seed = int(rctx.op_var.get("__inloop_var_seed__", "0"))
        nm_a = f"split_a_{100 + seed}"
        nm_b = f"split_b_{100 + seed}"
        rctx.op_var["__inloop_var_seed__"] = str(seed + 1)
        # The two results are referenced by index — for now just produce
        # `nm_a` for op_id; downstream consumers may index. This is a
        # simplification that works when split's second result is unused.
        rctx.op_var[op.op_id] = nm_a
        lines += f"{nm_a}, {nm_b} = {_render_op_expr(op, rctx)}"
        return

    # Cross-WG TMEM consumer: tmem_load reading a channel needs a
    # barrier_wait before the load. Same for the regular named-op path —
    # we just inject the wait first then fall through.
    if n.op_kind == "ttng.tmem_load" and op.operands:
        if _use_semaphore_ir():
            # The consumer block at the top of this function already emitted
            # the wait (tmem_load is a SemIR consumer of any cross-WG TMEM
            # semaphore). Nothing extra here.
            pass
        elif isinstance(op.operands[0], OpRef):
            src_id = op.operands[0].op_id
            for c in rctx.channels:
                if c.kind == "tmem" and c.alloc_op_id == src_id:
                    # count=1 cross-WG channel → toggle every iter.
                    lines += f"tlx.barrier_wait({_bar_full(c.name)}[0], _it & 1)"

    # Compute ops with a known renderer get assigned to a Python variable
    # so downstream ops can reference them by name (rather than re-rendering
    # the whole expression at each use). Skipping arith.constant — always inline.
    if n.op_kind in _IN_LOOP_NAMED_OPS and n.op_kind != "arith.constant":
        if op.op_id in rctx.op_var:
            return  # already named earlier in this scope
        name = _auto_name(op, rctx.fresh_idx())
        rctx.op_var[op.op_id] = name
        lines += f"{name} = {_render_op_expr(op, rctx)}"
        if _use_semaphore_ir():
            # SemIR: producer wait empty + (if buffer) local_store + arrive full.
            _semir_emit_producer_block(n, name, g, loop, rctx, lines)
            return
        # Schedule-driven channel producer side: if this node is a producer
        # in a cross-WG barrier, stage the value through the channel's SMEM
        # buffer and signal the reader. Guard with the same TMEM-bridge
        # check as the consumer side so we don't emit a redundant SMEM
        # store when an MMA already consumes the value via TMEM.
        for c in rctx.channels:
            if c.loop_id is not None and c.loop_id != loop.loop_id:
                continue
            if c.producer_node == n.id and c.consumer_node is not None:
                already_tmem = any(
                    tc.kind == "tmem" and tc.bridge_op_id and ((lambda bop: bop and bop.operands and isinstance(
                        bop.operands[0], OpRef) and bop.operands[0].op_id == op.op_id)(g.ops.get(tc.bridge_op_id)))
                    for tc in rctx.channels)
                if already_tmem:
                    continue
                # Loop-carry detection: same as consumer side. Signal-only.
                prod_cyc = next(
                    (nn.schedule_cycle for nn in loop.schedule.nodes if nn.id == c.producer_node),
                    None,
                )
                cons_cyc = next(
                    (nn.schedule_cycle for nn in loop.schedule.nodes if nn.id == c.consumer_node),
                    None,
                )
                is_loop_carry = (prod_cyc is not None and cons_cyc is not None and prod_cyc > cons_cyc)
                if c.kind == "named" or is_loop_carry:
                    kind_note = ("named-channel" if c.kind == "named" else "loop-carry release")
                    lines += (f"tlx.barrier_arrive({_bar_full(c.name)}[0], 1)  "
                              f"# {kind_note} signal (n{c.producer_node}→n{c.consumer_node})")
                    continue
                # Layer 1: wait for slot free before overwriting.
                lines += f"tlx.barrier_wait({_bar_empty(c.name)}[0], (_it & 1) ^ 1)"
                # Layer 2: if this buffer reuses another's bytes (merge
                # group alias), the preceding alias members' consumers
                # must also have finished before we overwrite.
                for pred_name in _alias_predecessors(c.name, rctx, g):
                    lines += (f"tlx.barrier_wait({_bar_empty(pred_name)}[0], "
                              f"(_it & 1) ^ 1)  # alias predecessor")
                lines += f"tlx.local_store({c.name}[0], {name})"
                lines += f"tlx.barrier_arrive({_bar_full(c.name)}[0], 1)"
        # Cross-WG TMEM bridge producer: emits the local_store + barrier
        # signal for a TMEM channel where this op produces the value
        # consumed by the bridge_op (a tmem_alloc(value) in the consumer
        # WG). PREVIOUSLY we skipped this when a parallel SMEM channel
        # also targeted the same op (the SMEM-side `already_tmem` check
        # already deferred to us). Without firing here, the consumer WG's
        # `barrier_wait(<tmem>_full)` would block forever — exactly the
        # case3 truncf → P_tmem deadlock.
        for c in rctx.channels:
            if c.kind != "tmem" or not c.bridge_op_id:
                continue
            bridge_op = g.ops.get(c.bridge_op_id)
            if (bridge_op and bridge_op.operands and isinstance(bridge_op.operands[0], OpRef)
                    and bridge_op.operands[0].op_id == op.op_id):
                # Same per-iter parity as the schedule-channel path.
                lines += f"tlx.barrier_wait({_bar_empty(c.name)}[0], (_it & 1) ^ 1)"
                lines += f"tlx.local_store({c.name}[0], {name})"
                lines += f"tlx.barrier_arrive({_bar_full(c.name)}[0], 1)"
        return
    if n.op_kind in ("arith.constant", "scf.yield"):
        return
    lines += f"# (unhandled in-loop op: {n.op_kind})"


def _find_load_target_buffer(node: Node, op: Op, g: ScheduleGraph, loop: Loop) -> Buffer | None:
    """Find the SMEM buffer that consumes this descriptor_load's result."""
    # Walk g.ops for an alloc whose first operand is `op`.
    for cand in g.ops.values():
        if cand.kind != "ttg.local_alloc":
            continue
        if not cand.operands:
            continue
        first = cand.operands[0]
        if isinstance(first, OpRef) and first.op_id == op.op_id:
            # Find the buffer with def_op == cand.op_id.
            for b in loop.schedule.buffers:
                if b.def_op == cand.op_id:
                    return b
    return None


def _channel_has_tma_store_consumer(sem, loop: Loop) -> bool:
    """True if any consumer of this semaphore is a TMA `tt.descriptor_store`
    (which reads the channel SMEM via the async proxy, so the SW producer must
    fence_async_shared() before signalling full)."""
    kinds = {nn.id: nn.op_kind for nn in loop.schedule.nodes}
    for c in sem.consumers:
        if kinds.get(c.node.node_id) == "tt.descriptor_store":
            return True
    return False


def _find_channel_producer_buffer(node: Node, loop: Loop) -> Buffer | None:
    """If `node` is the producer of a synthesized cross-WG SMEM channel
    (recorded in cross_wg_barriers with a paired_buffer_id), return that
    channel buffer.

    Used for TMA loads whose result is consumed in a DIFFERENT warp group
    with NO original staging buffer (e.g. LayerNorm: load → compute, where the
    load result is a register tile, not a `load → local_alloc → MMA` SMEM
    staging). The C++ partition synthesizes an SMEM channel to carry that tile
    across the WG boundary; the load must stage INTO it (async_descriptor_load
    + signal) rather than fall through to the register-returning `.load()`."""
    for cb in loop.schedule.cross_wg_barriers:
        if cb.producer_node == node.id and cb.paired_buffer_id is not None:
            for b in loop.schedule.buffers:
                if b.id == cb.paired_buffer_id:
                    return b
    return None


def _resolve_alloc_to_buffer(operand: OperandRef, g: ScheduleGraph, loop: Loop) -> tuple[Buffer | None, Loop | None]:
    """Find the (buffer, owning_loop) for an operand that is an alloc OpRef.
    Searches the given loop first, then any other loop in the graph (so an
    inner-loop MMA can find a buffer that's owned by the outer-loop schedule,
    e.g., case3's per-tile-resident Q SMEM)."""
    if not isinstance(operand, OpRef):
        return None, None
    for b in loop.schedule.buffers:
        if b.def_op == operand.op_id:
            return b, loop
    for L in g.loops:
        if L is loop:
            continue
        for b in L.schedule.buffers:
            if b.def_op == operand.op_id:
                return b, L
    return None, None


# ===========================================================================
# Top-level
# ===========================================================================

# ===========================================================================
# Topology detection + unified warp groups (M3)
# ===========================================================================


def _find_outer_loop(graph: ScheduleGraph) -> Loop | None:
    """Outermost loop = is_outer=True if present, else the only loop."""
    for L in graph.loops:
        if L.is_outer:
            return L
    return graph.loops[0] if graph.loops else None


def _find_super_node(loop: Loop) -> Node | None:
    """Find the super-node in `loop` (representing a nested inner loop)."""
    for n in loop.schedule.nodes:
        if n.child_pipeline_id is not None:
            return n
    return None


def _find_inner_loop(graph: ScheduleGraph, outer: Loop) -> Loop | None:
    """Inner loop referenced by outer's super-node (if any).

    NOTE: super_node.child_pipeline_id is an index into the outer's local
    ScheduleGraph.loops (not into the top-level JSON loops). For a simple
    2-level nest we identify the inner by `is_outer=False` and the outer by
    `is_outer=True`.
    """
    sn = _find_super_node(outer)
    if sn is None:
        return None
    if outer is None:
        return None
    for L in graph.loops:
        if L is not outer and not L.is_outer:
            return L
    return None


@dataclass
class UnifiedWG:
    """One async_task in the emitted kernel.

    `outer_wg` and `inner_wg` map to the warp_group ids in the outer / inner
    loop's local numbering. None means this UWG owns no ops at that level
    (e.g., default partition has no inner_wg; loads partition has no outer_wg
    epilogue ops, only outer infra it computes per-tile).
    """

    name: str
    role: str  # "default" | "TMA" | "TC" | other
    outer_wg: int | None
    inner_wg: int | None
    num_warps: int = 4
    num_regs: int | None = None
    is_default: bool = False


def _unified_warp_groups(graph: ScheduleGraph, outer: Loop | None, inner: Loop | None) -> list[UnifiedWG]:
    """Build the unified warp-group list for the whole kernel.

    Mirrors auto-WS's task-id propagation: each (role, scope) pair becomes
    one unified WG, which becomes one async_task in the emission.
    """
    if outer is None:
        return []

    # Single-loop kernel (case1, case6): every WG in the only loop = one UWG.
    if inner is None:
        out: list[UnifiedWG] = []
        # Does this kernel have a function-scope epilogue (ops AFTER the loop,
        # e.g. case1 GEMM's tmem_load → truncf → descriptor_store)? If so it
        # needs a dedicated "default" task to own that epilogue. If NOT (case6
        # LayerNorm: load+compute+store all happen IN the loop), emitting a
        # separate empty default would deadlock on the acc_tmem carve-out — so
        # instead promote the COMPUTE warp group (most warps, not a pure TMA
        # producer) to be the "default" task.
        epi_ops = [o for o in _ops_after_loop(graph, outer) if o.kind not in _SKIP_FUNCTION_SCOPE]
        has_epilogue = len(epi_ops) > 0
        default_wg_id: int | None = None
        if not has_epilogue:
            candidates = [w for w in outer.warp_groups if "TMA" not in w.pipelines and "TC" not in w.pipelines
                          ] or [w for w in outer.warp_groups if "TMA" not in w.pipelines]
            if candidates:
                default_wg_id = max(candidates, key=lambda w: w.num_warps).id
        if default_wg_id is None:
            out.append(
                UnifiedWG(
                    name="default",
                    role="default",
                    outer_wg=None,
                    inner_wg=None,
                    num_warps=4,
                    is_default=True,
                ))
        for wg in outer.warp_groups:
            roles = "+".join(wg.pipelines)
            primary = ("TC" if "TC" in wg.pipelines else ("TMA" if "TMA" in wg.pipelines else roles))
            # Layer B: trust the schedule pass's per-WG num_warps decision
            # (= max minWarps over the WG's ops, snapped to {1,2,4,8}).
            # The previous binary "has TMEM ops? → 4w/152r else 1w/24r"
            # rule mis-sized WGs whose SFU/CUDA work demanded 4 warps
            # without touching TMEM.
            num_warps = wg.num_warps
            num_regs = 152 if num_warps >= 4 else 24
            is_def = wg.id == default_wg_id
            out.append(
                UnifiedWG(
                    name="default" if is_def else f"wg{wg.id}_{primary}",
                    role="default" if is_def else primary,
                    outer_wg=None,
                    inner_wg=wg.id,
                    num_warps=num_warps,
                    num_regs=(None if is_def else num_regs),
                    is_default=is_def,
                ))
        return out

    # Persistent / nested-loop kernel (case2):
    # - One UWG per inner WG (each inner partition wraps the outer loop and
    #   does its inner ops at the super-node position).
    # - One "default" UWG owning the outer-loop epilogue (whichever outer wg
    #   contains the descriptor_store / tmem_load).
    out: list[UnifiedWG] = []

    # Find which outer wg owns the descriptor_store (the epilogue partition).
    epi_outer_wg: int | None = None
    for n in outer.schedule.nodes:
        if n.op_kind in ("tt.descriptor_store", "ttng.tmem_load"):
            epi_outer_wg = n.warp_group
            break
    out.append(
        UnifiedWG(
            name="default",
            role="default",
            outer_wg=epi_outer_wg,
            inner_wg=None,
            num_warps=4,
            is_default=True,
        ))

    # One UWG per inner warp group. Trust the schedule pass's num_warps
    # decision (Layer B); see the case1 branch above for the rationale.
    for wg in inner.warp_groups:
        roles = "+".join(wg.pipelines)
        primary = ("TC" if "TC" in wg.pipelines else ("TMA" if "TMA" in wg.pipelines else roles))
        num_warps = wg.num_warps
        num_regs = 152 if num_warps >= 4 else 24
        out.append(
            UnifiedWG(
                name=f"inner_wg{wg.id}_{primary}",
                role=primary,
                outer_wg=None,
                inner_wg=wg.id,
                num_warps=num_warps,
                num_regs=num_regs,
            ))
    return out


def _task_header(uwg: UnifiedWG) -> str:
    if uwg.is_default:
        return 'with tlx.async_task("default"):'
    parts: list[str] = []
    # Always emit num_warps explicitly — TLX's hardware checks (e.g.,
    # "TMEM load/store requires numWarps == 4 or 8") trigger on the
    # task-level value, not the kernel default.
    parts.append(f"num_warps={uwg.num_warps}")
    if uwg.num_regs is not None:
        parts.append(f"num_regs={uwg.num_regs}")
    return f"with tlx.async_task({', '.join(parts)}):"


# ===========================================================================
# Per-UWG body emission (M3 — port of WSSpecialize::SpecializeForOp)
# ===========================================================================


def _outer_nodes_for_uwg(outer: Loop, uwg: UnifiedWG) -> list[Node]:
    """Outer-loop nodes this UWG owns: nodes with matching warp_group, plus
    super-node (which is wg=-1 but always entered by inner-WG UWGs)."""
    out: list[Node] = []
    for n in outer.schedule.nodes:
        # Super-node: only inner-WG UWGs enter it.
        if n.child_pipeline_id is not None:
            if uwg.inner_wg is not None:
                out.append(n)
            continue
        # Other nodes: own them if warp_group matches.
        if uwg.outer_wg is not None and n.warp_group == uwg.outer_wg:
            out.append(n)
    return sorted(out, key=lambda n: (n.schedule_stage, n.schedule_cluster, n.id))


def _inner_nodes_for_uwg(inner: Loop, uwg: UnifiedWG) -> list[Node]:
    out = [n for n in inner.schedule.nodes if n.warp_group == uwg.inner_wg]
    return sorted(out, key=lambda n: (n.schedule_stage, n.schedule_cluster, n.id))


def _depends_on_iv_or_iter_arg(g: ScheduleGraph, op_id: str, visited: set[str] | None = None) -> bool:
    """True if op_id (transitively) references a loop IV or iter_arg.
    Such ops must be emitted INSIDE the loop body, not at task entry."""
    if visited is None:
        visited = set()
    if op_id in visited:
        return False
    visited.add(op_id)
    op = g.ops.get(op_id)
    if op is None:
        return False
    for o in op.operands:
        if isinstance(o, (IvRef, IterArgRef)):
            return True
        if isinstance(o, OpRef) and _depends_on_iv_or_iter_arg(g, o.op_id, visited):
            return True
    return False


def _collect_infra_deps_recursive(
    g: ScheduleGraph,
    op_id: str,
    visited: set[str],
    rctx: RenderCtx,
    out: list[Op],
    pre_loop: bool = True,
) -> None:
    """Walk op_id's transitive deps; append any `_NAMED_FUNCTION_OPS` op
    not already emitted to `out` in toposort order.

    `pre_loop=True` means we're emitting at task entry — skip ops with IV
    or iter_arg deps (those need to be computed inside the loop body and
    are handled separately in `_emit_uwg_body`)."""
    if op_id in visited:
        return
    visited.add(op_id)
    if op_id in rctx.op_var:
        return
    op = g.ops.get(op_id)
    if op is None or op.kind not in _NAMED_FUNCTION_OPS:
        return
    if pre_loop and _depends_on_iv_or_iter_arg(g, op_id):
        return  # leave for in-loop emission
    for o in op.operands:
        if isinstance(o, OpRef):
            _collect_infra_deps_recursive(g, o.op_id, visited, rctx, out, pre_loop)
    out.append(op)


def _replicate_infra_deps(
    g: ScheduleGraph,
    target_nodes: list[Node],
    container_loop: Loop,
    rctx: RenderCtx,
    lines: _Lines,
) -> None:
    """For each owned node, walk its operand DAG and rematerialize any cheap
    arith / index ops INSIDE this task body. Mirrors auto-WS's constant
    rematerialization: cheap deps cloned per region rather than captured."""
    visited: set[str] = set()
    to_emit: list[Op] = []
    for n in target_nodes:
        if not n.op_ref:
            continue
        target_op = g.ops.get(n.op_ref)
        if target_op is None:
            continue
        for o in target_op.operands:
            if isinstance(o, OpRef):
                _collect_infra_deps_recursive(g, o.op_id, visited, rctx, to_emit)

    for op in to_emit:
        name = _auto_name(op, rctx.fresh_idx())
        rctx.op_var[op.op_id] = name
        lines += f"{name} = {_render_op_expr(op, rctx)}"


# ---------------------------------------------------------------------------
# Iter-arg threading (port of WSSpecialize::collectBlockArgsForTask +
# SpecializeForOp). Each per-UWG specialized loop keeps only the iter_args
# its ops actually consume — same trim as Meta's WS path.
# ---------------------------------------------------------------------------


def _find_loop_yield(g: ScheduleGraph, loop_id: int) -> Op | None:
    """The scf.yield in the body of loop `loop_id` (scope == `loop:<id>`)."""
    for op in g.ops.values():
        if op.kind == "scf.yield" and op.scope == f"loop:{loop_id}":
            return op
    return None


def _find_loop_for(g: ScheduleGraph, loop: Loop) -> Op | None:
    """The scf.for op for this Loop. Matches by lo/hi/step operands against
    the schedule's bounds — robust to the IV never appearing as IvRef
    (common in modulo'd loops where all uses go through iter_args)."""
    target = (loop.schedule.lower_bound, loop.schedule.upper_bound, loop.schedule.step)
    for op in g.ops.values():
        if op.kind != "scf.for" or len(op.operands) < 3:
            continue
        if (op.operands[0], op.operands[1], op.operands[2]) == target:
            return op
    return None


def _loop_iter_args(g: ScheduleGraph, loop: Loop) -> list[tuple[int, OperandRef, OperandRef]]:
    """Return [(idx, init_ref, yield_ref), ...] for `loop`'s iter_args.

    init_ref comes from scf.for operand[3+idx]; yield_ref from scf.yield
    operand[idx] in the body scope. Filters out non-value iter_args (TMEM
    handles, async tokens) — those represent memory state threaded through
    the loop, not actual register values, and have no Python equivalent."""
    for_op = _find_loop_for(g, loop)
    yield_op = _find_loop_yield(g, loop.loop_id)
    if for_op is None or yield_op is None:
        return []
    inits = for_op.operands[3:]  # operand[0..2] are lo/hi/step
    yields = yield_op.operands
    n = min(len(inits), len(yields))
    out: list[tuple[int, OperandRef, OperandRef]] = []
    for i in range(n):
        init = inits[i]
        # Skip iter_args whose init is an OpRef to a memory-handle or token
        # alloc (TMEM/SMEM allocs threaded as scf.yield tokens). These have
        # no value-level recurrence — the buffer itself is the state.
        if isinstance(init, OpRef):
            init_op = g.ops.get(init.op_id)
            if init_op is not None and init_op.kind in (
                    "ttng.tmem_alloc",
                    "ttg.local_alloc",
                    "ttng.tmem_store",
                    "ub.poison",
            ):
                continue
        # Skip iter_args whose type is a memory descriptor or async token.
        if isinstance(init, ConstRef) and init.type:
            t = init.type
            if "memdesc" in t or "async.token" in t or "tensor_memory" in t:
                continue
        out.append((i, init, yields[i]))
    return out


def _iter_args_used_by_inner_uwg(
    g: ScheduleGraph,
    inner: Loop,
    uwg: UnifiedWG,
) -> list[int]:
    """Mirror WSSpecialize::collectBlockArgsForTask for the inner loop +
    one UWG. An iter_arg is "kept" iff some op in this UWG (i.e., scope ==
    `loop:<inner_id>` AND warp_group == uwg.inner_wg) references it."""
    if uwg.inner_wg is None:
        return []
    wg_op_ids = {n.op_ref for n in inner.schedule.nodes if n.warp_group == uwg.inner_wg and n.op_ref}
    used: set[int] = set()
    for oid in wg_op_ids:
        op = g.ops.get(oid)
        if op is None:
            continue
        for o in op.operands:
            if isinstance(o, IterArgRef) and o.loop_id == inner.loop_id:
                used.add(o.idx)
    return sorted(used)


def _iter_arg_python_name(loop_id: int, idx: int, init: OperandRef) -> str:
    """Stable per-loop iter_arg name. Use semantic hints from the init when
    the pattern is unmistakable (FA softmax: -inf → m_i, 1.0 → l_i); fall
    back to `i{loop}_{idx}` otherwise."""
    if isinstance(init, ConstRef):
        if init.value == "-inf":
            return f"m_i_{loop_id}"
        if init.value == 1 and "tensor" in (init.type or ""):
            return f"l_i_{loop_id}"
    return f"i{loop_id}_{idx}"


def _has_smem_ring_buffer_inner(inner: Loop, uwg: UnifiedWG) -> bool:
    """Does this UWG touch any SMEM buffer in the inner loop?

    Returns True for any partition that reads or writes a SMEM buf, even
    when count=1 — the emitter still uses `smem_accum` as a per-iter parity
    counter for barrier phase calculation (`_it & 1`), independent of the
    ring depth.
    """
    if inner is None or uwg.inner_wg is None:
        return False
    nodes = _inner_nodes_for_uwg(inner, uwg)
    for n in nodes:
        for bid in n.consumes_buffers + ([n.produces_buffer] if n.produces_buffer is not None else []):
            buf = next((b for b in inner.schedule.buffers if b.id == bid), None)
            if buf and buf.kind == "smem":
                return True
    # MEM partition: nodes don't have produces_buffer set on the load itself,
    # but the load feeds an alloc that's a SMEM buffer. Use heuristic: any
    # descriptor_load in the WG implies SMEM ring usage.
    if any(n.op_kind == "tt.descriptor_load" for n in nodes):
        return True
    return False


def _has_tmem_handoff(inner: Loop, uwg: UnifiedWG) -> bool:
    """Does this UWG participate in a TMEM hand-off (TC produces, default
    consumes)? True for TC partition and default in persistent kernels."""
    if uwg.is_default:
        return True
    if inner is not None and uwg.inner_wg is not None:
        nodes = _inner_nodes_for_uwg(inner, uwg)
        if any(n.op_kind == "ttng.tc_gen5_mma" for n in nodes):
            return True
    return False


def _smem_depth_for_uwg(inner: Loop, uwg: UnifiedWG) -> int:
    nodes = _inner_nodes_for_uwg(inner, uwg)
    # Use produces_buffer / consumes_buffers to find inner SMEM depth.
    for n in nodes:
        for bid in n.consumes_buffers + ([n.produces_buffer] if n.produces_buffer is not None else []):
            buf = next((b for b in inner.schedule.buffers if b.id == bid), None)
            if buf and buf.kind == "smem" and buf.count > 1:
                return buf.count
    # Fallback: max smem depth in the loop.
    return max((b.count for b in inner.schedule.buffers if b.kind == "smem"), default=1)


def _emit_inner_loop_in_outer(
    g: ScheduleGraph,
    inner: Loop,
    uwg: UnifiedWG,
    channels: list[Channel],
    rctx: RenderCtx,
    lines: _Lines,
    persistent: bool,
) -> None:
    """Emit the inner K-loop trimmed to this UWG's inner partition. For
    persistent kernels, uses `smem_accum` (declared outside the outer loop)
    as the ring-buffer counter so it persists across tiles."""
    iv = inner.schedule.induction_var_name
    lo = _render_operand(inner.schedule.lower_bound, rctx)
    hi = _render_operand(inner.schedule.upper_bound, rctx)
    step = _render_operand(inner.schedule.step, rctx)
    depth = _smem_depth_for_uwg(inner, uwg)

    # TC partition: wait for epilogue release before starting next tile's K-loop.
    # Uses tmem_buf / tmem_phase computed at top of per-tile body for ring
    # buffer indexing (depth = tmem_count).
    if persistent and uwg.role == "TC":
        lines += "tlx.barrier_wait(acc_tmem_empty[tmem_buf], tmem_phase ^ 1)"

    # Loop-carry pre-arrive: under SemIR, looked up from is_released
    # semaphores keyed by (loop, wg). Legacy path uses cycle comparison.
    if _use_semaphore_ir():
        for line in _semir_pre_arrives_for_wg(inner.loop_id, uwg.inner_wg, rctx):
            lines += line
    else:
        cycle_of = {n.id: n.schedule_cycle for n in inner.schedule.nodes}
        for c in rctx.channels:
            if c.loop_id is not None and c.loop_id != inner.loop_id:
                continue
            if c.consumer_wg != uwg.inner_wg:
                continue
            if c.producer_node is None or c.consumer_node is None:
                continue
            prod_cyc = cycle_of.get(c.producer_node)
            cons_cyc = cycle_of.get(c.consumer_node)
            if prod_cyc is None or cons_cyc is None:
                continue
            if prod_cyc > cons_cyc:
                lines += (f"tlx.barrier_arrive({_bar_full(c.name)}[0], 1)  "
                          f"# loop-carry pre-arrive (producer cyc={prod_cyc} > "
                          f"consumer cyc={cons_cyc})")

    # Per-UWG iter_arg trim (mirror of WSSpecialize::SpecializeForOp): only
    # keep iter_args this UWG's inner ops actually consume. Init each kept
    # one before the loop; reassign at end of body from the yield expression.
    iter_specs = _loop_iter_args(g, inner)
    used_idxs = _iter_args_used_by_inner_uwg(g, inner, uwg)
    kept = [(idx, init, yld, _iter_arg_python_name(inner.loop_id, idx, init))
            for (idx, init, yld) in iter_specs
            if idx in used_idxs]
    saved_iter_arg_var = dict(rctx.iter_arg_var)
    for idx, init, _yld, name in kept:
        rctx.iter_arg_var[(inner.loop_id, idx)] = name
        lines += f"{name} = {_render_operand(init, rctx)}"

    lines += (f"# Inner K-loop (loop {inner.loop_id}, II={inner.schedule.II}). "
              f"SMEM ring depth={depth}; smem_accum persists across outer tiles.")
    with lines.block(f"for {iv} in range({lo}, {hi}, {step}):"):
        if persistent:
            # accum_cnt persists across tiles
            # SemIR's phase/slot formulas reference `_it`; persistent inner
            # loops use `smem_accum` as their iteration counter, so alias.
            lines += "_it = smem_accum"
            lines += f"buf = smem_accum % {depth}"
            lines += f"phase = (smem_accum // {depth}) & 1"
        else:
            # `phase` toggles per `depth` iters — REQUIRED by mbarriers.
            # Even depth=1 needs `_it & 1` (every iter is its own phase).
            lines += f"_it = ({iv} - {lo}) // {step}"
            lines += f"buf = _it % {depth}"
            lines += f"phase = (_it // {depth}) & 1"
        # Skip bridge ops (cross-WG TMEM tmem_alloc(value)): they're emitted
        # as local_store + barrier_arrive in the value producer's WG body.
        bridge_op_ids = {c.bridge_op_id for c in rctx.channels if c.kind == "tmem" and c.bridge_op_id}
        for n in _inner_nodes_for_uwg(inner, uwg):
            if n.op_kind in (
                    "ttg.local_alloc",
                    "ttng.tmem_alloc",
                    "scf.yield",
                    "ttg.memdesc_trans",
            ):
                continue  # SSA wrappers — not emitted as side effects
            if n.op_ref in bridge_op_ids:
                continue  # relocated to the producer WG
            _emit_in_loop_node(n, g, inner, channels, rctx, lines)
        if persistent:
            lines += "smem_accum += 1"
        # Recurrence: reassign each kept iter_arg from its yield producer
        # (matches scf.yield semantics — Triton folds these into iter_args).
        for idx, _init, yld, name in kept:
            lines += f"{name} = {_render_operand(yld, rctx)}"

    rctx.iter_arg_var = saved_iter_arg_var

    # TC partition: signal epilogue that TMEM is full. Use `tcgen05_commit`
    # rather than `barrier_arrive` so the barrier fires only AFTER all prior
    # async `tcgen05.mma` ops complete. With plain `barrier_arrive` the
    # epilogue's `tmem_load(acc_tmem)` can race the in-flight MMA — for
    # large K-loops the natural drain time hides it, but for small K (e.g.
    # K=128 / 2 inner iters) the read happens before the last MMA lands
    # and the epilogue reads stale TMEM. Manifests as ~1/4 of each m-tile's
    # rows being wrong (one warp's share of the 4-warp tmem_load layout).
    if persistent and uwg.role == "TC":
        lines += "tlx.tcgen05_commit(acc_tmem_full[tmem_buf])"


def _find_partition_chain(
    outer_nodes: list[Node],
    g: ScheduleGraph,
    outer_loop: Loop,
) -> tuple[int, int, int, int] | None:
    """Detect Pass A.5 partition chain in outer-loop nodes.

    Returns (chain_start_idx, chain_end_idx_exclusive, partition_count, m_size)
    when the epilogue's tmem_load resolves to a TMEM buffer with
    partition_count > 1; the chain spans from the tmem_load through the
    next descriptor_store. Returns None otherwise.
    """
    # Find the partition_count on any function-scope TMEM buf in the outer
    # loop. There's at most one accumulator per persistent kernel.
    pcount = 0
    msize = 0
    for b in outer_loop.schedule.buffers:
        if b.kind == "tmem" and b.partition_count > 1:
            pcount = b.partition_count
            msize = b.m_size
            break
    if pcount <= 1:
        return None
    # Locate tmem_load → ... → descriptor_store in outer_nodes.
    chain_start = -1
    for i, n in enumerate(outer_nodes):
        if n.op_kind == "ttng.tmem_load":
            chain_start = i
            break
    if chain_start < 0:
        return None
    chain_end = chain_start + 1
    for j in range(chain_start, len(outer_nodes)):
        chain_end = j + 1
        if outer_nodes[j].op_kind == "tt.descriptor_store":
            break
    return chain_start, chain_end, pcount, msize


def _emit_outer_epilogue_partitioned(
    chain_nodes: list[Node],
    N: int,
    m_size: int,
    g: ScheduleGraph,
    rctx: RenderCtx,
    lines: _Lines,
) -> None:
    """Emit Pass A.5 partitioned epilogue chain inside the outer loop body.

    The kernel's c_desc is still (BM, BN), but the TMEM accumulator was
    split into N (m_size, BN) groups. We load each group, cast to the
    store dtype, `tl.cat` them back along M into a (BM, BN) tile, then
    issue a single descriptor_store against the full tile.

        tlx.barrier_wait(acc_tmem_full[tmem_buf], tmem_phase)
        acc_g0 = tlx.local_load(acc_tmem_g0[tmem_buf]); trunc_g0 = ...
        acc_g1 = tlx.local_load(acc_tmem_g1[tmem_buf]); trunc_g1 = ...
        tlx.barrier_arrive(acc_tmem_empty[tmem_buf], 1)
        c_full = tl.cat([trunc_g0, trunc_g1], dim=0)  # (BM, BN)
        tlx.local_store(c_smem[0], c_full)
        tlx.fence_async_shared()
        tlx.async_descriptor_store(c_desc, c_smem[0], [m, n])
        tlx.async_descriptor_store_wait(0)

    This keeps c_desc and c_smem at the full (BM, BN) shape so the
    launcher's `c_desc.block_shape = (BM, BN)` contract is unchanged.
    """
    lines += f"# Pass A.5 partitioned epilogue (N={N}, m_size={m_size})"
    lines += "tlx.barrier_wait(acc_tmem_full[tmem_buf], tmem_phase)"
    # Per-group: load → cast → local_store into c_smem at per-group M-slice.
    # Defer the single descriptor_store + drain until after all groups have
    # written into c_smem. Avoids a register-level cat.
    store_desc: str | None = None
    store_offsets: list[str] = []
    bn = 0
    for gi in range(N):
        for n in chain_nodes:
            op = g.ops.get(n.op_ref) if n.op_ref else None
            if op is None or op.kind in _SKIP_FUNCTION_SCOPE:
                continue
            if op.kind == "ttng.tmem_load":
                dest_op_id = (op.operands[0].op_id if op.operands and isinstance(op.operands[0], OpRef) else None)
                pnames = (rctx.partition_alloc_names.get(dest_op_id) if dest_op_id else None)
                tmem_name = pnames[gi] if pnames else f"acc_tmem_g{gi}"
                acc_var = f"acc_g{gi}"
                lines += f"{acc_var} = tlx.local_load({tmem_name}[tmem_buf])"
                rctx.op_var[op.op_id] = acc_var
                continue
            if op.kind == "arith.truncf":
                inner = _render_operand(op.operands[0], rctx)
                rt = op.result_types[0] if op.result_types else ""
                sd = _parse_tensor_shape(rt)
                target = _dtype_str_to_tl(sd[1]) if sd else "tl.float16"
                name = f"trunc_g{gi}_{rctx.fresh_idx()}"
                rctx.op_var[op.op_id] = name
                lines += f"{name} = {inner}.to({target})"
                continue
            if op.kind == "ttg.convert_layout":
                rctx.op_var[op.op_id] = _render_operand(op.operands[0], rctx)
                continue
            if op.kind == "tt.descriptor_store":
                if store_desc is None:
                    store_desc = _render_operand(op.operands[0], rctx)
                    store_offsets = [_render_operand(o, rctx) for o in op.operands[2:]]
                    # BN from descriptor block shape (second op operand).
                    if isinstance(op.operands[0], OpRef):
                        desc_op = g.ops.get(op.operands[0].op_id)
                        if desc_op and desc_op.result_types:
                            bs = _parse_desc_block_shape(desc_op.result_types[0])
                            if bs:
                                bn = bs[0][1]
                    elif isinstance(op.operands[0], ArgRef):
                        arg = next(
                            (a for a in g.kernel.args if a.name == op.operands[0].name),
                            None,
                        )
                        if arg:
                            bs = _parse_desc_block_shape(arg.type)
                            if bs:
                                bn = bs[0][1]
                value_expr = _render_operand(op.operands[1], rctx)
                lines += (f"tlx.local_store(tlx.local_slice(c_smem[0], "
                          f"[{gi * m_size}, 0], [{m_size}, {bn}]), {value_expr})")
                continue
    # Release TMEM as soon as both loads are done — store happens against
    # registers + c_smem now, no longer reads acc_tmem.
    lines += "tlx.barrier_arrive(acc_tmem_empty[tmem_buf], 1)"
    if store_desc is not None:
        offs_str = ", ".join(store_offsets)
        lines += "tlx.fence_async_shared()"
        lines += f"tlx.async_descriptor_store({store_desc}, c_smem[0], [{offs_str}])"
        lines += "tlx.async_descriptor_store_wait(0)"


def _find_subtile_chain(outer_nodes: list[Node]) -> tuple[int, int, int, int] | None:
    """Detect Pass A.7 subtile chain in outer-loop nodes.

    Returns (chain_start_idx, chain_end_idx_exclusive, subtile_count, n_size)
    when any node has subtile_count > 1; the chain spans from the first such
    node through (and including) the next descriptor_store. Returns None if
    no subtiled chain is present (default = no-op).
    """
    chain_start = -1
    S = 0
    sub_size = 0
    for i, n in enumerate(outer_nodes):
        if n.subtile_count > 1:
            chain_start = i
            S = n.subtile_count
            sub_size = n.n_size
            break
    if chain_start < 0:
        return None
    chain_end = chain_start + 1
    for j in range(chain_start, len(outer_nodes)):
        chain_end = j + 1
        if outer_nodes[j].op_kind == "tt.descriptor_store":
            break
    return chain_start, chain_end, S, sub_size


def _emit_outer_epilogue_subtiled(
    chain_nodes: list[Node],
    S: int,
    sub_size: int,
    g: ScheduleGraph,
    rctx: RenderCtx,
    lines: _Lines,
) -> None:
    """Emit a Pass A.7 subtiled epilogue chain inside the outer loop body.

    Pattern (single SMEM staging buffer, subsliced per sub-tile):

        tlx.barrier_wait(acc_tmem_full[tmem_buf], tmem_phase)
        for sub_n in range(S):
            n_off = sub_n * sub_size
            acc_sub = tlx.subslice(acc_tmem[tmem_buf], n_off, sub_size)
            acc = tlx.local_load(acc_sub)
            ... cast / convert chain ...
            c_sub = tlx.subslice(c_smem[0], n_off, sub_size)
            tlx.local_store(c_sub, c)
            tlx.fence_async_shared()
            tlx.async_descriptor_store(c_desc, c_sub, [m, n + n_off])
            tlx.async_descriptor_store_wait(0)
        tlx.barrier_arrive(acc_tmem_empty[tmem_buf], 1)
    """
    lines += "# Pass A.7 epilogue subtile (S=%d, sub_size=%d)" % (S, sub_size)
    lines += "tlx.barrier_wait(acc_tmem_full[tmem_buf], tmem_phase)"
    # tlx.subslice requires constexpr offsets, so we unroll the subtile loop
    # in Python at emit time. c_smem is double-buffered (count=2 set in
    # _emit_buffers) — `wait(1)` BEFORE every local_store gates buffer
    # reuse (ring depth=2). TMEM is released right after the last
    # local_load — TMA stores read c_smem, not acc_tmem, so cross-tile
    # MMA → epilogue overlap is unblocked. No per-tile drain; the final
    # `wait(0)` is emitted by the caller AFTER the persistent loop ends.
    # Pattern matches blackwell_gemm_ws tutorial.
    BUF_RING = 2  # matches buf.count set in extractBufferShape / _emit_buffers
    for sub_n in range(S):
        n_off = sub_n * sub_size
        smem_slot = sub_n % BUF_RING
        is_last_sub = sub_n == S - 1
        for n in chain_nodes:
            op = g.ops.get(n.op_ref) if n.op_ref else None
            if op is None or op.kind in _SKIP_FUNCTION_SCOPE:
                continue
            if op.kind == "ttng.tmem_load":
                acc_var = f"acc_{sub_n}"
                lines += (f"acc_sub_{sub_n} = tlx.subslice(acc_tmem[tmem_buf], "
                          f"{n_off}, {sub_size})")
                lines += f"{acc_var} = tlx.local_load(acc_sub_{sub_n})"
                rctx.op_var[op.op_id] = acc_var
                # Release TMEM the moment the final load completes — TMA
                # stores from here on read c_smem only, so MMA WG can
                # start the next tile's MMA immediately.
                if is_last_sub:
                    lines += "tlx.barrier_arrive(acc_tmem_empty[tmem_buf], 1)"
                continue
            if op.kind == "arith.truncf":
                inner = _render_operand(op.operands[0], rctx)
                rt = op.result_types[0] if op.result_types else ""
                sd = _parse_tensor_shape(rt)
                target = _dtype_str_to_tl(sd[1]) if sd else "tl.float16"
                name = f"trunc_{sub_n}_{rctx.fresh_idx()}"
                rctx.op_var[op.op_id] = name
                lines += f"{name} = {inner}.to({target})"
                continue
            if op.kind == "ttg.convert_layout":
                rctx.op_var[op.op_id] = _render_operand(op.operands[0], rctx)
                continue
            if op.kind == "tt.descriptor_store":
                desc = _render_operand(op.operands[0], rctx)
                value_expr = _render_operand(op.operands[1], rctx)
                offsets = [_render_operand(o, rctx) for o in op.operands[2:]]
                if len(offsets) >= 2:
                    offsets[-1] = f"({offsets[-1]} + {n_off})"
                offs_str = ", ".join(offsets)
                # wait(1) gates c_smem[smem_slot] reuse — at most 1 store
                # in flight when we arrive here, and the OLDEST one (which
                # is on the slot we're about to write) gets drained. The
                # very first call across the kernel is a no-op (0 in flight).
                lines += f"tlx.async_descriptor_store_wait({BUF_RING - 1})"
                lines += f"tlx.local_store(c_smem[{smem_slot}], {value_expr})"
                lines += "tlx.fence_async_shared()"
                # eviction_policy="evict_first" matches the tutorial — hints
                # L2 to drop the staging buffer after the TMA store, freeing
                # cache capacity for actual data traffic. Should help memory
                # throughput on large shapes.
                lines += (f"tlx.async_descriptor_store({desc}, c_smem[{smem_slot}], "
                          f'[{offs_str}], eviction_policy="evict_first")')
                continue
    # NOTE: no per-tile wait(0). The caller emits a single wait(0) AFTER
    # the persistent for-loop to drain remaining in-flight stores.


def _emit_outer_op(n: Node, g: ScheduleGraph, rctx: RenderCtx, lines: _Lines) -> None:
    """Emit one outer-loop op (computational or epilogue side-effect)."""
    op = g.ops.get(n.op_ref) if n.op_ref else None
    if op is None or op.kind in _SKIP_FUNCTION_SCOPE:
        return
    if op.op_id in rctx.op_var:
        return  # already emitted (e.g., in preamble)

    if op.kind == "ttng.tmem_load":
        # In the default (epilogue) partition: wait for MMA fill, load, signal release.
        # Uses tmem_buf / tmem_phase computed at top of per-iter body.
        lines += "tlx.barrier_wait(acc_tmem_full[tmem_buf], tmem_phase)"
        lines += "acc = tlx.local_load(acc_tmem[tmem_buf])"
        lines += "tlx.barrier_arrive(acc_tmem_empty[tmem_buf], 1)"
        rctx.op_var[op.op_id] = "acc"
        return
    if op.kind == "tt.descriptor_load":
        # Outer-loop intra-WG TMA load (case5 bias). The TMA target SMEM and
        # full/empty barriers were registered in emit() during the outer SMEM
        # alloc loop. Emit the standard async-load sequence + local_load.
        binding = rctx.outer_load_bindings.get(n.id)
        if binding is None:
            # No matching SMEM buf found upstream; fall through to the
            # generic renderer (will emit `<tma_load_inline_unsupported>`).
            return
        bufname = binding["bufname"]
        n_bytes = binding["n_bytes"]
        var_name = binding["var_name"]
        desc = _render_operand(op.operands[0], rctx)
        offsets = [_render_operand(o, rctx) for o in op.operands[1:]]
        offs_str = ", ".join(offsets)
        # Per-tile cycling: each TMA arrive flips the barrier's phase.
        # `tmem_accum_cnt` is already the per-tile counter declared by the
        # default partition; reuse its parity for the wait phase.
        full_bar = _bar_full(bufname)
        empty_bar = _bar_empty(bufname)
        # Empty-side wait gates reuse of the SMEM across tiles (count is
        # typically 1, so phase flips every tile).
        lines += f"tlx.barrier_wait({empty_bar}[0], (tmem_accum_cnt & 1) ^ 1)"
        lines += f"tlx.barrier_expect_bytes({full_bar}[0], {n_bytes})"
        lines += (f"tlx.async_descriptor_load({desc}, {bufname}[0], "
                  f"[{offs_str}], {full_bar}[0])")
        lines += f"tlx.barrier_wait({full_bar}[0], tmem_accum_cnt & 1)"
        lines += f"{var_name} = tlx.local_load({bufname}[0])"
        lines += f"tlx.barrier_arrive({empty_bar}[0], 1)"
        rctx.op_var[op.op_id] = var_name
        return
    if op.kind == "arith.truncf":
        inner = _render_operand(op.operands[0], rctx)
        rt = op.result_types[0] if op.result_types else ""
        sd = _parse_tensor_shape(rt)
        target = _dtype_str_to_tl(sd[1]) if sd else "tl.float16"
        name = f"trunc_{rctx.fresh_idx()}"
        rctx.op_var[op.op_id] = name
        lines += f"{name} = {inner}.to({target})"
        return
    if op.kind == "ttg.convert_layout":
        rctx.op_var[op.op_id] = _render_operand(op.operands[0], rctx)
        return
    if op.kind == "tt.descriptor_store":
        desc = _render_operand(op.operands[0], rctx)
        value_expr = _render_operand(op.operands[1], rctx)
        offsets = [_render_operand(o, rctx) for o in op.operands[2:]]
        offs_str = ", ".join(offsets)
        lines += f"tlx.local_store(c_smem[0], {value_expr})"
        lines += "tlx.fence_async_shared()"
        lines += f"tlx.async_descriptor_store({desc}, c_smem[0], [{offs_str}])"
        lines += "tlx.async_descriptor_store_wait(0)"
        return
    if op.kind in _NAMED_FUNCTION_OPS:
        name = _auto_name(op, rctx.fresh_idx())
        rctx.op_var[op.op_id] = name
        lines += f"{name} = {_render_op_expr(op, rctx)}"
        return


def _emit_uwg_body(
    g: ScheduleGraph,
    outer: Loop,
    inner: Loop | None,
    uwg: UnifiedWG,
    channels: list[Channel],
    rctx: RenderCtx,
    lines: _Lines,
) -> None:
    """Emit ONE async_task body. Mirror of WSSpecialize::SpecializeForOp.

    Per-task variable scoping: snapshot `rctx.op_var` on entry and restore on
    exit so per-task rematerialized ops (pid_m_c, offs_am, etc.) don't leak
    into other tasks' rendering.
    """
    op_var_snapshot = dict(rctx.op_var)
    try:
        _emit_uwg_body_impl(g, outer, inner, uwg, channels, rctx, lines)
    finally:
        rctx.op_var = op_var_snapshot


def _emit_uwg_body_impl(
    g: ScheduleGraph,
    outer: Loop,
    inner: Loop | None,
    uwg: UnifiedWG,
    channels: list[Channel],
    rctx: RenderCtx,
    lines: _Lines,
) -> None:
    persistent = inner is not None  # has nested loop pattern

    if not persistent:
        # case1 path: just emit single inner loop body for this WG.
        # A default task with NO inner_wg owns the function-scope epilogue
        # (case1 GEMM). A default task WITH an inner_wg is a compute group
        # promoted to "default" (case6 LayerNorm: no function-scope epilogue)
        # — emit its in-loop WG body like any other warp group.
        if uwg.is_default and uwg.inner_wg is None:
            _emit_default_partition(g, outer, rctx, lines)
        else:
            wg_obj = next((w for w in outer.warp_groups if w.id == uwg.inner_wg), None)
            if wg_obj is not None:
                _emit_warp_group(g, outer, wg_obj, channels, rctx, lines)
        return

    # Persistent path: outer for-loop wrapping per-WG body.
    # Counter declarations BEFORE the outer loop.
    if _has_smem_ring_buffer_inner(inner, uwg):
        # smem_accum: K-loop ring index (persists across persistent tiles
        # so the SMEM ring keeps rotating through the outer iterations).
        lines += "smem_accum = 0"
    has_tmem = _has_tmem_handoff(inner, uwg)
    if has_tmem:
        # tmem_accum_cnt: outer-loop ring index for TMEM hand-off
        # (count from outer-loop schedule.buffers; >1 → cross-tile pipelining).
        lines += "tmem_accum_cnt = 0"

    # Replicate infra deps that the outer-loop ops need.
    outer_nodes = _outer_nodes_for_uwg(outer, uwg)
    _replicate_infra_deps(g, outer_nodes, outer, rctx, lines)

    # Outer for-loop scaffolding.
    out_iv = outer.schedule.induction_var_name
    out_lo = _render_operand(outer.schedule.lower_bound, rctx)
    out_hi = _render_operand(outer.schedule.upper_bound, rctx)
    out_step = _render_operand(outer.schedule.step, rctx)

    lines += (f"# Outer persistent loop (loop {outer.loop_id}, II={outer.schedule.II}). "
              f"Each task replays it; body trimmed to this WG's ops.")
    with lines.block(f"for {out_iv} in range({out_lo}, {out_hi}, {out_step}):"):
        # Per-tile TMEM ring-buffer indexing.
        if has_tmem:
            tc = rctx.tmem_count
            lines += f"tmem_buf = tmem_accum_cnt % {tc}"
            lines += f"tmem_phase = (tmem_accum_cnt // {tc}) & 1"
        # Per-iter infra: ops with IV/iter_arg deps that this body needs
        # (e.g., pid_m, pid_n, offs_am, offs_bn for the in-loop accesses).
        in_loop_infra_visited: set[str] = set()
        in_loop_infra: list[Op] = []
        for n in outer_nodes:
            if not n.op_ref:
                continue
            target_op = g.ops.get(n.op_ref)
            if target_op is None:
                continue
            for o in target_op.operands:
                if isinstance(o, OpRef):
                    _collect_infra_deps_recursive(
                        g,
                        o.op_id,
                        in_loop_infra_visited,
                        rctx,
                        in_loop_infra,
                        pre_loop=False,
                    )
        # Filter out infra deps that transitively consume an outer-load
        # value (case5 bias: `arith.extf` reads the bias TMA load). Those
        # need the load's result variable, which only exists AFTER the
        # load is emitted in the main outer_nodes loop. Let the main loop
        # emit them at the right point.
        outer_load_op_ids: set[str] = set()
        for n in outer_nodes:
            if n.id in rctx.outer_load_bindings and n.op_ref:
                outer_load_op_ids.add(n.op_ref)
        if outer_load_op_ids:
            dep_cache: dict[str, bool] = {}

            def _depends_on_outer_load(op_id: str) -> bool:
                if op_id in dep_cache:
                    return dep_cache[op_id]
                if op_id in outer_load_op_ids:
                    dep_cache[op_id] = True
                    return True
                op = g.ops.get(op_id)
                result = False
                if op is not None:
                    for o in op.operands:
                        if isinstance(o, OpRef) and _depends_on_outer_load(o.op_id):
                            result = True
                            break
                dep_cache[op_id] = result
                return result

            in_loop_infra = [op for op in in_loop_infra if not _depends_on_outer_load(op.op_id)]
        for op in in_loop_infra:
            name = _auto_name(op, rctx.fresh_idx())
            rctx.op_var[op.op_id] = name
            lines += f"{name} = {_render_op_expr(op, rctx)}"

        # Emit outer-loop `tt.descriptor_load` nodes (case5 bias) AFTER
        # infra deps so the load's offset operands (`mul_3, mul_4`) are
        # already bound. The load body uses the SMEM + barriers
        # registered in emit() at function scope.
        for n in outer_nodes:
            if n.op_kind != "tt.descriptor_load":
                continue
            if n.id not in rctx.outer_load_bindings:
                continue
            op = g.ops.get(n.op_ref) if n.op_ref else None
            if op is None or op.op_id in rctx.op_var:
                continue
            _emit_outer_op(n, g, rctx, lines)

        # Pass A.7: detect a subtiled epilogue chain. When present, the chain
        # nodes get emitted as a single `for sub_n` loop instead of per-op.
        sub_info = _find_subtile_chain(outer_nodes)
        subtile_start = sub_info[0] if sub_info else -1
        subtile_end = sub_info[1] if sub_info else -1
        # Pass A.5: detect partitioned epilogue chain (mutually exclusive with
        # A.7 for now — A.7+A.5 interaction is future work).
        part_info = (_find_partition_chain(outer_nodes, g, outer) if not sub_info else None)
        part_start = part_info[0] if part_info else -1
        part_end = part_info[1] if part_info else -1
        for i, n in enumerate(outer_nodes):
            if n.child_pipeline_id is not None:
                # Super-node → emit the inner K-loop here.
                _emit_inner_loop_in_outer(g, inner, uwg, channels, rctx, lines, persistent=True)
                continue
            if sub_info and i == subtile_start:
                chain_nodes = outer_nodes[subtile_start:subtile_end]
                _emit_outer_epilogue_subtiled(chain_nodes, sub_info[2], sub_info[3], g, rctx, lines)
                continue
            if sub_info and subtile_start < i < subtile_end:
                continue  # already emitted inside the subtile loop
            if part_info and i == part_start:
                chain_nodes = outer_nodes[part_start:part_end]
                _emit_outer_epilogue_partitioned(
                    chain_nodes,
                    part_info[2],
                    part_info[3],
                    g,
                    rctx,
                    lines,
                )
                continue
            if part_info and part_start < i < part_end:
                continue  # already emitted inside the partition loop
            _emit_outer_op(n, g, rctx, lines)

        # Advance the TMEM ring counter at end of each tile (both default
        # and TC partitions, so tmem_buf/tmem_phase stay in sync).
        if has_tmem:
            lines += "tmem_accum_cnt += 1"

    # Pass A.7: when a subtiled epilogue chain is active, the per-tile body
    # omits the trailing wait(0) so cross-tile TMA store overlap is possible.
    # Drain any remaining in-flight TMA stores AFTER the persistent loop.
    if _find_subtile_chain(outer_nodes) is not None:
        lines += "tlx.async_descriptor_store_wait(0)  # A.7: drain remaining TMA stores"


# ===========================================================================
# Top-level
# ===========================================================================


def emit(graph: ScheduleGraph) -> str:
    lines = _Lines()
    lines += "# AUTO-GENERATED by sched2tlx — do not edit by hand."
    lines += f"# Source: schedule_graph for kernel `{graph.kernel.name}`"
    if graph.loops:
        wgs = ", ".join(f"wg{w.id}=[{'+'.join(w.pipelines)}]" for w in graph.loops[0].warp_groups)
        lines += f"# Warp groups (loop 0): {wgs}"
    lines += "import torch"
    lines += "import triton"
    lines += "import triton.language as tl"
    lines += "import triton.language.extra.tlx as tlx"
    lines += ""
    _kernel_sig_lines(graph, lines)

    outer_loop = _find_outer_loop(graph)
    inner_loop = _find_inner_loop(graph, outer_loop) if outer_loop else None

    # Disambiguate IV names. MLIR loc carries no semantic name → "iv" by
    # default. Rename outer IV to "tile_id" and inner IV to "k" so they
    # don't shadow each other in the per-task body.
    iv_names: dict[int, str] = {}
    for L in graph.loops:
        base = L.schedule.induction_var_name
        if outer_loop and L is outer_loop and base == "iv":
            iv_names[L.loop_id] = "tile_id"
        elif inner_loop and L is inner_loop and base == "iv":
            iv_names[L.loop_id] = "k"
        else:
            iv_names[L.loop_id] = base
    # Mutate the schedule_loop's IV name in place so downstream rendering
    # picks up the new name (the operand renderer reads from rctx.loop_iv).
    for L in graph.loops:
        L.schedule.induction_var_name = iv_names[L.loop_id]

    rctx = RenderCtx(
        graph=graph,
        op_var={},
        buffer_var={},
        alloc_op_var={},
        loop_iv=iv_names,
    )

    lines.indent = 1
    if outer_loop is None:
        return lines.render()

    # Preamble (function-scope ops before the outermost loop).
    _emit_preamble(graph, outer_loop, rctx, lines)

    # Buffer allocs from BOTH outer and inner loops, hoisted to function scope.
    if inner_loop is not None:
        _emit_buffers(inner_loop, graph, rctx, lines)
        # Outer-loop TMEM buffer(s) hoisted to function scope. Counts
        # come from modulo's outer-loop schedule (>1 enables cross-tile
        # TC↔default pipelining via tmem_accum_cnt ring indexing).
        # Outer-loop SMEM buffers (e.g. case3's Q-tile resident through the
        # K-loop). Each gets a unique hoisted alloc + def_op binding so
        # downstream MMA renderers resolve `<a?>`/`<b?>` correctly.
        #
        # Skip the descriptor_store's buf — the dedicated c_smem emission
        # below handles it. Previously this loop also emitted a duplicate
        # `L*_smem_*` for the store buf and the bias-load binding heuristic
        # paired it with any descriptor_load whose shape happened to match.
        # That dual-use silently broke under A.7 subtile (which shrinks the
        # store buf's N dim) — the bias load's actual shape no longer
        # matched the shrunk store buf. The new flow:
        #   1. Emit only non-store outer SMEM bufs here.
        #   2. Separately walk outer descriptor_load nodes and allocate a
        #      dedicated SMEM staging buffer per load (sized to the load's
        #      result type), so bias-style loads never depend on shape
        #      coincidence with the store buf.
        for buf in outer_loop.schedule.buffers:
            if buf.kind != "smem":
                continue
            if buf.def_op:
                def_op = graph.ops.get(buf.def_op)
                if def_op and def_op.kind == "tt.descriptor_store":
                    # c_smem alloc below handles this — don't duplicate it.
                    continue
            shape = ", ".join(str(d) for d in buf.shape)
            dtype = _bits_to_tl_dtype(buf.element_bits, is_float=True)
            name = f"L{outer_loop.loop_id}_smem_{buf.id}"
            rctx.buffer_var[(outer_loop.loop_id, buf.id)] = name
            if buf.def_op:
                rctx.alloc_op_var[buf.def_op] = name
            lines += (f"# {name}: outer-loop SMEM buf {buf.id}, count={buf.count} "
                      f"(per-tile resident, e.g. Q tile through K-loop)")
            lines += f"{name} = tlx.local_alloc(({shape}), {dtype}, {buf.count})"

        # Dedicated SMEM staging per outer-loop `tt.descriptor_load` (e.g.
        # case5 bias). Each load gets its own buffer sized to its result
        # type — decoupled from the store buf so A.7 subtile doesn't
        # accidentally shrink it. `_emit_outer_op` emits the TMA-load
        # sequence using the bufname registered here.
        for node in outer_loop.schedule.nodes:
            if node.op_kind != "tt.descriptor_load":
                continue
            if node.warp_group < 0:
                continue
            if node.id in rctx.outer_load_bindings:
                continue
            op = graph.ops.get(node.op_ref) if node.op_ref else None
            if op is None:
                continue
            rt = op.result_types[0] if op.result_types else ""
            rt_shape = _parse_tensor_shape(rt)
            if not rt_shape:
                continue
            shape_dims, dtype_str = rt_shape
            shape_str = ", ".join(str(d) for d in shape_dims)
            dtype_tl = _dtype_str_to_tl(dtype_str)
            # bits from dtype prefix (f16 → 16, bf16 → 16, f32 → 32).
            if dtype_str.startswith("bf"):
                bits = 16
            elif dtype_str[0] in "fi":
                bits = int(dtype_str[1:])
            else:
                bits = 16
            n_elems = 1
            for d in shape_dims:
                n_elems *= d
            n_bytes = n_elems * bits // 8
            bufname = f"outer_load_{node.id}_smem"
            lines += (f"# {bufname}: dedicated staging for outer descriptor_load "
                      f"N{node.id} (shape ({shape_str}))")
            lines += f"{bufname} = tlx.local_alloc(({shape_str}), {dtype_tl}, 1)"
            rctx.outer_load_bindings[node.id] = {
                "bufname": bufname,
                "count": 1,
                "n_bytes": n_bytes,
                "var_name": f"load_{node.id}",
            }
        # When multiple TMEM buffers exist (e.g. case3 FA has qk + acc),
        # name them per-id and reserve `acc_tmem` for the LARGEST one
        # (the final accumulator that the epilogue reads).
        tmem_bufs = [b for b in outer_loop.schedule.buffers if b.kind == "tmem"]
        # Largest by total bytes is the final accumulator.
        primary = max(tmem_bufs, key=lambda b: b.total_bytes) if tmem_bufs else None
        for buf in tmem_bufs:
            shape = ", ".join(str(d) for d in buf.shape)
            name = "acc_tmem" if buf is primary else f"tmem_{buf.id}"
            rctx.buffer_var[(outer_loop.loop_id, buf.id)] = name
            if buf.def_op:
                rctx.alloc_op_var[buf.def_op] = name
            # Stash the primary's count for the TC↔default ring depth.
            if buf is primary:
                rctx.tmem_count = buf.count
            if buf.partition_count > 1:
                # Pass A.5: emit N separate TMEM allocs each (m_size, *trailing).
                # Each MMA partition writes to its own acc_tmem_g{i}; the
                # epilogue loads each and stores at M-offset.
                pshape_dims = list(buf.shape)
                pshape_dims[buf.partition_dim] = buf.m_size
                pshape_str = ", ".join(str(d) for d in pshape_dims)
                names = []
                for gi in range(buf.partition_count):
                    gname = f"{name}_g{gi}"
                    names.append(gname)
                    lines += (f"# {gname}: Pass A.5 group {gi}/{buf.partition_count}"
                              f" of partitioned TMEM acc buf {buf.id} "
                              f"(per-group shape ({pshape_str}))")
                    lines += (f"{gname} = tlx.local_alloc(({pshape_str}), "
                              f"tl.float32, {buf.count}, tlx.storage_kind.tmem)")
                rctx.partition_buffer_names[(outer_loop.loop_id, buf.id)] = names
                if buf.def_op:
                    rctx.partition_alloc_names[buf.def_op] = names
                # `name` (legacy `acc_tmem`) aliases group 0 so any path that
                # references it without partition awareness still resolves.
                lines += f"{name} = {names[0]}"
            else:
                lines += (f"# {name}: outer-loop buf {buf.id}, count={buf.count} "
                          f"(TC writes / default reads across {buf.count} tiles)")
                lines += (f"{name} = tlx.local_alloc(({shape}), tl.float32, "
                          f"{buf.count}, tlx.storage_kind.tmem)")
        # Epilogue staging SMEM (for descriptor_store) — derive from the
        # store's descriptor operand. The descriptor may be a kernel arg
        # (case2) or a make_tensor_descriptor op (case1).
        epi_store = next((op for op in graph.ops.values() if op.kind == "tt.descriptor_store"), None)
        if epi_store and epi_store.operands:
            shape = [128, 128]
            dtype = "tl.float16"
            op0 = epi_store.operands[0]
            if isinstance(op0, OpRef):
                desc_op = graph.ops.get(op0.op_id)
                if desc_op:
                    bs = _parse_desc_block_shape(desc_op.result_types[0] if desc_op.result_types else "")
                    if bs:
                        shape, dtype = bs[0], _dtype_str_to_tl(bs[1])
            elif isinstance(op0, ArgRef):
                arg = next((a for a in graph.kernel.args if a.name == op0.name), None)
                if arg:
                    bs = _parse_desc_block_shape(arg.type)
                    if bs:
                        shape, dtype = bs[0], _dtype_str_to_tl(bs[1])
            # Pass A.7: shrink to (BM, BN/S) when the descriptor_store node
            # was marked as subtiled. Sub-stores share this single buffer.
            sub_count = 1
            sub_n_size = 0
            for lp in graph.loops:
                for nd in lp.schedule.nodes:
                    if nd.op_ref == epi_store.op_id and nd.subtile_count > 1:
                        sub_count = nd.subtile_count
                        sub_n_size = nd.n_size
                        break
                if sub_count > 1:
                    break
            if sub_count > 1 and sub_n_size > 0 and len(shape) >= 2:
                shape = [shape[0], sub_n_size]
            # Pass A.7: when subtiling, double-buffer the staging SMEM so the
            # emitter can overlap the next local_store with the prior TMA
            # store (matches blackwell_gemm_ws tutorial). For S=2 this gives
            # same total SMEM as the un-subtiled baseline but unlocks overlap.
            buf_count = 2 if sub_count > 1 else 1
            shape_str = ", ".join(str(d) for d in shape)
            lines += f"c_smem = tlx.local_alloc(({shape_str}), {dtype}, {buf_count})"
        lines += ""
    else:
        _emit_buffers(outer_loop, graph, rctx, lines)

    # Channels + mbarriers from the inner loop (or only loop).
    deriving_loop = inner_loop if inner_loop is not None else outer_loop
    channels = _derive_channels(deriving_loop, rctx)
    # Schedule-pass-provided cross-WG barriers (Pass B Step 2). Each entry
    # carries producer/consumer node ids + the SMEM/TMEM buffer slot used
    # to ferry the value across WG boundary. This is the authoritative
    # source for register-typed cross-WG flows (alpha, l, m in FA softmax)
    # which the schedule pass synthesizes buffers for.
    sched_channels: list[Channel] = []
    for L in graph.loops:
        for cb in L.schedule.cross_wg_barriers:
            if cb.paired_buffer_id is None:
                # 'named' / signal-only channel: no data buffer, just a
                # producer→consumer sync barrier. Synthesize a Channel with
                # no associated buffer; the emitter allocates the barrier
                # pair and inserts arrive/wait via the cross-WG channel
                # path. Used for MMA→consumer or store→consumer signals
                # where no SMEM staging is needed (the data lives in TMEM
                # or is implicit, just need ordering).
                sync_name = f"sync_n{cb.producer_node}_to_n{cb.consumer_node}"
                sched_channels.append(
                    Channel(
                        name=sync_name,
                        depth=cb.depth,
                        producer_wg=cb.producer_wg,
                        consumer_wg=cb.consumer_wg,
                        kind="named",  # signal-only, no buffer
                        producer_node=cb.producer_node,
                        consumer_node=cb.consumer_node,
                        buffer_id=None,
                        loop_id=L.loop_id,
                    ))
                continue
            buf_var = rctx.buffer_var.get((L.loop_id, cb.paired_buffer_id))
            if buf_var is None:
                continue  # buffer wasn't emitted (skipped kind?)
            sched_channels.append(
                Channel(
                    name=buf_var,
                    depth=cb.depth,
                    producer_wg=cb.producer_wg,
                    consumer_wg=cb.consumer_wg,
                    kind="smem",  # all schedule-synthesized are SMEM-staged
                    producer_node=cb.producer_node,
                    consumer_node=cb.consumer_node,
                    buffer_id=cb.paired_buffer_id,
                    loop_id=L.loop_id,
                ))
    # TMEM cross-WG channels (e.g. case3 qk_tmem TC→softmax, P_tmem
    # softmax→TC). Detected by walking alloc op_id ↔ writer/reader WGs.
    # For non-persistent kernels (only one loop, treated as outer), the
    # detector still needs to run on that loop's nodes — pass it as `inner`.
    detect_loop = inner_loop if inner_loop is not None else outer_loop
    if detect_loop is not None:
        tmem_channels = _derive_tmem_channels(graph, detect_loop)
        # Resolve channel name from the alloc binding we set up earlier.
        for ch in tmem_channels:
            if ch.alloc_op_id and ch.alloc_op_id in rctx.alloc_op_var:
                ch.name = rctx.alloc_op_var[ch.alloc_op_id]
        # Index existing channels by name. A TMEM channel detected here may
        # match a channel already produced by the SMEM (DDG) detector (same
        # buffer, different evidence). Merge our bridge_op_id / kind / alloc
        # metadata onto the existing entry so downstream emitters see them.
        by_name = {c.name: c for c in channels}
        # Reserved: acc_tmem is the outer-loop TC↔default channel allocated
        # specially below; don't double-emit.
        for tc in tmem_channels:
            if tc.name == "acc_tmem":
                continue
            if tc.name in by_name:
                ex = by_name[tc.name]
                ex.kind = "tmem"
                ex.alloc_op_id = ex.alloc_op_id or tc.alloc_op_id
                ex.bridge_op_id = ex.bridge_op_id or tc.bridge_op_id
            else:
                channels.append(tc)
                by_name[tc.name] = tc
    # Merge schedule-driven channels (cross_wg_barriers) — these are
    # authoritative for register-typed cross-WG flows. Skip if a channel
    # for the same buffer name already exists (e.g., DDG-detected SMEM
    # ring), but back-fill the producer_node/consumer_node info onto it.
    #
    # Also skip a SMEM-staged sched_channel whose consumer is a
    # tmem_alloc(value) bridge: the same data is already routed through
    # the TMEM bridge channel by direct cross-WG TMEM store, so the SMEM
    # staging buffer (and its full+empty barrier pair) is dead. Dropping
    # it removes one barrier_wait + barrier_arrive per iter from each side.
    bridge_op_ids = {c.bridge_op_id for c in channels if c.kind == "tmem" and c.bridge_op_id}
    node_op_ref: dict[tuple[int, int], str] = {}
    for L in graph.loops:
        for n in L.schedule.nodes:
            if n.op_ref:
                node_op_ref[(L.loop_id, n.id)] = n.op_ref
    by_name = {c.name: c for c in channels}
    for sc in sched_channels:
        if sc.kind == "smem" and sc.consumer_node is not None:
            cons_op = node_op_ref.get((sc.loop_id, sc.consumer_node))
            if cons_op and cons_op in bridge_op_ids:
                continue  # redundant SMEM staging — TMEM bridge covers it
        if sc.name in by_name:
            ex = by_name[sc.name]
            ex.producer_node = ex.producer_node or sc.producer_node
            ex.consumer_node = ex.consumer_node or sc.consumer_node
            ex.buffer_id = ex.buffer_id or sc.buffer_id
            ex.loop_id = ex.loop_id if ex.loop_id is not None else sc.loop_id
        else:
            channels.append(sc)
            by_name[sc.name] = sc
    # Extra: any SMEM/TMEM loop-buffer with a TMA load producer (async)
    # needs barriers even if intra-WG. Walk all loops; collect names not
    # already covered by `channels`.
    chan_names = {c.name for c in channels}
    extra: list[tuple[str, int]] = []
    for L in graph.loops:
        for b in L.schedule.buffers:
            if b.kind not in ("smem", "tmem"):
                continue
            if not b.def_op:
                continue
            name = rctx.buffer_var.get((L.loop_id, b.id))
            if name is None or name in chan_names:
                continue
            # Only allocate barriers for buffers fed by an async load
            # (TMA descriptor_load) — others are pure register-staging.
            has_load = any(op.kind in ("ttg.local_alloc", ) and op.operands and (isinstance(op.operands[0], OpRef) and (
                (g_op := graph.ops.get(op.operands[0].op_id)) and g_op.kind == "tt.descriptor_load"))
                           for oid, op in graph.ops.items()
                           if oid == b.def_op)
            if has_load:
                extra.append((name, b.count))
    # Outer-loop intra-WG TMA loads (case5 bias): their SMEM bufs need
    # full+empty barriers even though the load + consumer are on the same WG.
    for binding in rctx.outer_load_bindings.values():
        bufname = binding["bufname"]
        if bufname in chan_names:
            continue
        if any(name == bufname for name, _ in extra):
            continue
        extra.append((bufname, binding["count"]))

    # Cross-loop iter_arg result channels: the default partition's
    # epilogue may reference scf.for results (= final iter_arg values),
    # but those values live in the WG that wrote the last iter_arg yield.
    # Stage each such cross-WG result through a SMEM buffer.
    rctx.crossloop_channels = _derive_crossloop_result_channels(graph, rctx)
    for ch in rctx.crossloop_channels:
        extra.append((ch["bufname"], 1))
        # Hoist alloc here (no def_op — synthesized).
        shape = ", ".join(str(d) for d in ch["shape"]) + ("," if len(ch["shape"]) == 1 else "")
        lines += (f"# {ch['bufname']}: cross-loop iter_arg result channel "
                  f"(loop {ch['loop_id']} iter_arg {ch['idx']} → epilogue)")
        lines += f"{ch['bufname']} = tlx.local_alloc(({shape}), {ch['dtype']}, 1)"

    # ── No warp specialization (single warp group) ──────────────────────────
    # When the schedule assigns every op to one warp group and there are no
    # cross-WG / cross-loop channels, there is nothing to specialize. Emit the
    # loop directly at kernel scope — a plain @triton.jit body, no
    # `tlx.async_tasks()` wrapper, no cross-WG mbarriers, no default-partition
    # task. A TC-free / memory-bound loop (e.g. LayerNorm) gets its load/compute
    # overlap from async TMA + double-buffering within the single warp group;
    # warp specialization would only add barriers.
    real_wg_ids = sorted(
        {n.warp_group
         for L in graph.loops
         for n in L.schedule.nodes
         if n.warp_group is not None and n.warp_group >= 0})
    if inner_loop is None and not rctx.crossloop_channels and len(real_wg_ids) <= 1:
        rctx.channels = []
        rctx.sem_set = None
        # M1 (store deferral): in the single-WG path, in-loop TMA stores use the
        # deferred-wait pattern so the store latency overlaps the next iteration's
        # load+compute instead of blocking. Scoped here so the WS path is intact.
        rctx.defer_inloop_store = True
        wg_obj = (next((w for w in outer_loop.warp_groups if w.id == real_wg_ids[0]), None) if real_wg_ids else None)
        if wg_obj is not None:
            _emit_warp_group(graph, outer_loop, wg_obj, [], rctx, lines)
        return lines.render()

    # Build the SemIR view of cross-WG synchronization. Always built — even
    # when the legacy emit path is used — so we can dump for inspection.
    # The actual emit decision is gated by `_use_semaphore_ir()`.
    wg_of_node: dict[tuple[int, int], int] = {}
    for L in graph.loops:
        for n in L.schedule.nodes:
            wg_of_node[(L.loop_id, n.id)] = n.warp_group
    rctx.sem_set = build_sem_set_for_graph(graph, wg_of_node=wg_of_node)

    if _use_semaphore_ir():
        # SemIR-driven mbarrier emission: one Semaphore = one full+empty pair
        # (NVWS protocol). Signal-only Semaphores (no buffer) only allocate
        # the full barrier — there's no recycle direction.
        lines += "# ── Mbarriers (SemIR: full+empty pair per semaphore) ──"
        for ls in rctx.sem_set.lowered:
            sem = ls.sem
            note = sem.note or ""
            lines += f"# {ls.name}: {note}"
            lines += ls.alloc_full_stmt
            if ls.alloc_empty_stmt is not None:
                lines += ls.alloc_empty_stmt
        # Cross-loop / cross-region edges that aren't in any loop's
        # cross_wg_barriers (e.g., the acc_tmem TC-loop → default-epilogue
        # hand-off, and per-loop iter_arg result channels). Keep the legacy
        # full+empty pair convention for these — they're emitted/consumed by
        # the legacy code paths in _emit_default_partition / _emit_warp_group.
        if "acc_tmem" not in {c.name for c in channels}:
            lines += (f"# acc_tmem: cross-region TC-loop → default-epilogue "
                      f"hand-off, depth={rctx.tmem_count} (legacy carve-out)")
            lines += (f"acc_tmem_full = tlx.alloc_barriers"
                      f"(num_barriers={rctx.tmem_count}, arrive_count=1)")
            lines += (f"acc_tmem_empty = tlx.alloc_barriers"
                      f"(num_barriers={rctx.tmem_count}, arrive_count=1)")
        # TMEM bridge channels (e.g., FA's softmax → P_tmem TMEM that the
        # PV MMA reads). Their full/empty barriers get emitted by the SW
        # producer block + consumer-side wait.
        for c in channels:
            if c.kind != "tmem" or not c.bridge_op_id:
                continue
            lines += (f"# {c.name}: TMEM bridge channel "
                      f"(SW producer → MMA consumer via TMEM, depth={c.depth})")
            lines += (f"{_bar_full(c.name)} = tlx.alloc_barriers"
                      f"(num_barriers={c.depth}, arrive_count=1)")
            lines += (f"{_bar_empty(c.name)} = tlx.alloc_barriers"
                      f"(num_barriers={c.depth}, arrive_count=1)")
        # Function-scope per-tile-resident loads (e.g. FA's Q tile): one
        # mbarrier per load. Emitted by MEM-role WG, consumed by the WG
        # whose MMA reads the alloc.
        for fl in rctx.fn_scope_loads:
            lines += f"# {fl['alloc_var']}_full: per-tile resident load barrier"
            lines += (f"{fl['alloc_var']}_full = tlx.alloc_barriers"
                      f"(num_barriers=1, arrive_count=1)")
        for name, depth in extra or []:
            lines += (f"# {name}: legacy carve-out (cross-loop iter_arg or "
                      f"intra-WG async load)")
            lines += (f"{_bar_full(name)} = tlx.alloc_barriers"
                      f"(num_barriers={depth}, arrive_count=1)")
            lines += (f"{_bar_empty(name)} = tlx.alloc_barriers"
                      f"(num_barriers={depth}, arrive_count=1)")
    else:
        # Function-scope per-tile-resident loads (e.g. FA's Q tile): one
        # mbarrier per load. Routed through `extra` so the legacy
        # `_emit_mbarriers` allocates the full+empty pair.
        extra_with_fnscope = list(extra)
        for fl in rctx.fn_scope_loads:
            extra_with_fnscope.append((fl["alloc_var"], 1))
        _emit_mbarriers(
            channels,
            lines,
            tmem_count=rctx.tmem_count,
            extra_buffers=extra_with_fnscope,
        )
    rctx.channels = channels

    # Async tasks. One per unified warp group; each runs the outer
    # persistent loop (if any) replicated and trimmed to its ops.
    uwgs = _unified_warp_groups(graph, outer_loop, inner_loop)
    with lines.block("with tlx.async_tasks():"):
        for uwg in uwgs:
            # Reference Phase 4's plan so the role attribution is visible.
            origin = (f"outer wg{uwg.outer_wg}" if uwg.outer_wg is not None else
                      (f"inner wg{uwg.inner_wg}" if uwg.inner_wg is not None else "(none)"))
            lines += f"# Async task: role={uwg.role} ← {origin} (Phase 4 plan)"
            with lines.block(_task_header(uwg)):
                _emit_uwg_body(graph, outer_loop, inner_loop, uwg, channels, rctx, lines)

    return lines.render()
