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
import sys
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
# fp8 uses MLIR's canonical spelling (f8E4M3FN, f8E5M2, ...); list the longer
# FNUZ variants first so the alternation doesn't stop at the shorter prefix.
_DTYPE_ALT = (
    r"(?:bf16|f8E4M3FNUZ|f8E5M2FNUZ|f8E4M3FN|f8E5M2|f8E4M3|f8E8M0FNU"
    r"|f8e4m3|f8e5m2|f16|f32|f64|i1|i8|i16|i32|i64)"
)
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


# MLIR fp8 type name → tl dtype. e4m3fn is the NVIDIA/OCP "nv" variant.
_FP8_TO_TL = {
    "f8E4M3FN": "tl.float8e4nv",
    "f8E4M3": "tl.float8e4nv",
    "f8E4M3FNUZ": "tl.float8e4b8",
    "f8E5M2": "tl.float8e5",
    "f8E5M2FNUZ": "tl.float8e5b16",
}


def _dtype_str_to_tl(dtype: str) -> str:
    if dtype in _FP8_TO_TL:
        return _FP8_TO_TL[dtype]
    if dtype.startswith("bf"):
        return "tl.bfloat16"
    if dtype.startswith("f"):
        bits = int(dtype[1:])
        return _bits_to_tl_dtype(bits, is_float=True)
    if dtype.startswith("i"):
        bits = int(dtype[1:])
        return _bits_to_tl_dtype(bits, is_float=False)
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
    partition_buffer_names: dict[tuple[int, int], list[str]] = field(
        default_factory=dict
    )
    # True iff the kernel has an MMA → default-partition `acc_tmem` hand-off.
    # When False (e.g. case6 LayerNorm — no MMA), the emitter must NOT allocate
    # `acc_tmem` barriers nor emit the default task's `barrier_wait(acc_tmem_full)`:
    # with no MMA there is no `tcgen05_commit` to arrive it, so the wait would
    # deadlock. Set in `emit()`.
    has_acc_tmem_handoff: bool = True
    # Mirror of partition_buffer_names keyed by def_op id (the SSA alloc).
    # Used when an emitter resolves a buffer via alloc_op_var instead of
    # (loop_id, buf_id).
    partition_alloc_names: dict[str, list[str]] = field(default_factory=dict)
    # Intra-WG stage-skew (emitter software pipelining). Computed once in
    # emit() by _compute_skew_plan. skew_plan: (loop_id, wg_id) → plan dict
    # {group_of: {node_id: group}, n_groups, ring_edges}. skew_ring_by_op /
    # skew_ring: the async producer's destination buffer becomes a
    # full/empty ring of depth (skew gap + 1); keyed by alloc op_id before
    # name binding, mirrored by alloc var name after. skew_ring_consumers:
    # (loop_id, node_id) → list of ring entries this node reads (SW wait
    # full / arrive empty around its emission).
    skew_plan: dict[tuple[int, int], dict] = field(default_factory=dict)
    skew_ring_by_op: dict[str, dict] = field(default_factory=dict)
    skew_ring: dict[str, dict] = field(default_factory=dict)
    skew_ring_consumers: dict[tuple[int, int], list] = field(default_factory=dict)
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
            var = rctx.alloc_op_var[ref.op_id]
            # Intra-WG skew ring: consumers index the producer's logical
            # iteration slot instead of the fixed [0].
            ring = rctx.skew_ring.get(var)
            if ring is not None:
                return f"{var}[{ring['slot']}]"
            return f"{var}[0]"
        if op is None:
            return f"<missing:{ref.op_id}>"
        return _render_op_expr(op, rctx)
    return "?"


def _render_const(ref: ConstRef) -> str:
    v = ref.value
    if v is None:
        return "0"
    # Non-finite floats are dumped as JSON strings ("inf"/"-inf"/"nan").
    scalar = (
        f"float('{v}')"
        if isinstance(v, str) and v in ("inf", "-inf", "nan")
        else ("True" if v != 0 else "False") if ref.type == "i1" else repr(v)
    )
    # Tensor splat (e.g., dense<-inf> in tensor<128xf32>) — wrap with
    # tl.full so the loop-carried iter_arg keeps its tensor type. Without
    # this, init `m_i = float('-inf')` is a Python scalar that mismatches
    # the recurrence's tensor type each iteration.
    if ref.type and "tensor<" in ref.type:
        sd = _parse_tensor_shape(ref.type)
        if sd:
            shape, dtype = sd
            shape_str = (
                str(shape[0]) + ","
                if len(shape) == 1
                else ", ".join(str(d) for d in shape)
            )
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
    shape = [_render_operand(o, rctx) for o in ops[n_ptr : n_ptr + n_shape]]
    strides = [
        _render_operand(o, rctx)
        for o in ops[n_ptr + n_shape : n_ptr + n_shape + n_stride]
    ]
    rt = op.result_types[0] if op.result_types else ""
    block_info = _parse_desc_block_shape(rt)
    if block_info is None:
        return f"tl.make_tensor_descriptor({ptr}, [{', '.join(shape)}], [{', '.join(strides)}], [...])"
    block_dims, _ = block_info
    block_str = ", ".join(str(d) for d in block_dims)
    return (
        f"tl.make_tensor_descriptor({ptr}, [{', '.join(shape)}], "
        f"[{', '.join(strides)}], [{block_str}])"
    )


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
    # Plain pointer load (m/D row vectors in FA-bwd). Same-WG consumers used
    # to inline-render it at the use site, but a CROSS-WG consumer needs the
    # named-op path so the producer block fires (load into a register, store
    # to the synthesized channel, arrive full) — otherwise the load is never
    # issued and the consumer WG deadlocks on the channel's full barrier.
    "tt.load",
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
    "scf.for",
    "scf.yield",
    "tt.return",
    "ttg.local_alloc",  # hoisted to top-of-kernel allocs
    "ttng.tmem_alloc",  # hoisted to top-of-kernel allocs
    "ttng.tmem_store",  # init-zero handled by use_acc=False on first MMA
    "arith.constant",  # inlined
}


def _is_in_loop(op: Op, loop: Loop) -> bool:
    return op.scope == f"loop:{loop.loop_id}"


def _function_scope_ops_in_order(graph: ScheduleGraph) -> list[Op]:
    """Iterate ops table in insertion order, function scope only."""
    return [op for op in graph.ops.values() if op.scope == "function"]


def _descriptor_reduce_desc_ops(graph: ScheduleGraph) -> set[str]:
    """op_ids of make_tensor_descriptor ops feeding a tt.descriptor_reduce —
    these are re-materialized inside the consuming task, not the preamble."""
    out: set[str] = set()
    for op in graph.ops.values():
        if op.kind != "tt.descriptor_reduce" or not op.operands:
            continue
        if isinstance(op.operands[0], OpRef):
            out.add(op.operands[0].op_id)
    return out


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


def _skw(buffer_var: str) -> str:
    """Barrier-name namespace for intra-WG stage-skew rings. The ring's DATA
    var is the accumulator/alloc var itself, but its barriers must not share
    names with that var's other barrier pairs: the FA-bwd epilogue handoff
    allocates `{var}_full/_empty` for the same accumulator, and in merged
    (all-MMA-in-one-WG) layouts both pairs coexist — the later skew-ring
    binding silently shadowed the handoff pair (case4 merged draw,
    2026-07-11), mistargeting the epilogue wait and post-loop commit."""
    return f"{buffer_var}_skw"


def _ring_exprs(count: int, rctx: "RenderCtx") -> tuple[str, str]:
    """(index, phase) for subscripting a SPECIFIC ring buffer in the current
    WG body.

    The WG-shared `buf`/`phase` counters advance modulo the WG's
    REPRESENTATIVE (max) ring depth — correct only for buffers whose own
    count equals it. A shallower ring in the same WG must use its OWN
    modulus, or its subscript drifts off the slots its producer and
    barriers use (case4 v2 draw, 2026-07-10: a count-2 dS ring in a
    rep_depth-3 WG was read at [_it % 3] but written at [_it % 2] — dQ/dK
    one tile off and index 2 out of bounds, dV clean).
    """
    if count == 1:
        return "0", "(_it & 1)"
    rep = getattr(rctx, "_wg_rep_depth", None)
    if rep is not None and count != rep:
        return f"(_it % {count})", f"((_it // {count}) & 1)"
    return "buf", "phase"


# ── Merge-group alias safety (Step 4.5 storage reuse) ──────────────────────
# HW-issued producer kinds: the write happens asynchronously (TMA/MMA engine)
# and its wait-empty lives in op-specific renderers where alias waits are not
# injected (yet) — a merge group containing one cannot be guarded by the SW
# producer path below, so its storage reuse is dropped instead.
_HW_PRODUCER_KINDS = (
    "tt.descriptor_load",
    "tt.descriptor_gather",
    "ttng.tc_gen5_mma",
    "ttng.tc_gen5_mma_scaled",
    "ttng.warp_group_dot",
    "ttng.tmem_copy",
    "ttg.async_copy_global_to_local",
)


def _alias_group_safety(loop: Loop) -> dict[int, bool]:
    """merge_group_id → can the SW producer path synchronize the alias?

    Step 4.5's lifetime check proves merged buffers occupy disjoint cycle
    windows, but cycles are a MODEL — on hardware only barriers order the
    aliased writes against the other members' readers. A group is guardable
    iff every member is a synthesized cross-WG channel (has a paired
    cross_wg_barrier) whose producer is SW-issued and whose channel is
    forward (a backward/loop-carry channel is signal-only: its empty never
    arrives, so waiting on it deadlocks). Guardable groups get alias waits
    at their producers (_alias_wait_stmts); every other multi-member SMEM
    group has its reuse DROPPED at alloc time — separate bytes are always
    correct, merely less thrifty. Single-member groups are trivially safe.
    """
    cycle_of = {n.id: n.schedule_cycle for n in loop.schedule.nodes}
    kind_of = {n.id: n.op_kind for n in loop.schedule.nodes}
    cbs_by_buf: dict[int, list] = {}
    for cb in loop.schedule.cross_wg_barriers:
        if cb.paired_buffer_id is not None:
            cbs_by_buf.setdefault(cb.paired_buffer_id, []).append(cb)
    groups: dict[int, list] = {}
    for b in loop.schedule.buffers:
        if b.merge_group_id is not None and b.kind == "smem":
            groups.setdefault(b.merge_group_id, []).append(b)
    safety: dict[int, bool] = {}
    for mgid, members in groups.items():
        if len(members) < 2:
            safety[mgid] = True
            continue
        ok = True
        for m in members:
            cbs = cbs_by_buf.get(m.id, [])
            if not cbs:
                ok = False
                break
            for cb in cbs:
                pc = cycle_of.get(cb.producer_node)
                cc = cycle_of.get(cb.consumer_node)
                if (
                    pc is None
                    or cc is None
                    or pc > cc
                    or kind_of.get(cb.producer_node) in _HW_PRODUCER_KINDS
                ):
                    ok = False
                    break
            if not ok:
                break
        safety[mgid] = ok
    return safety


def _alias_wait_stmts(loop: Loop, rctx: "RenderCtx", buf_id: int) -> list[str]:
    """Alias-safety waits for a channel producer whose buffer shares bytes
    with other merge-group members (Step 4.5 storage reuse).

    The group is one physical slot with k producer/consumer pairs per
    iteration in lifetime order: p1→c1→…→pk→ck→p1(next iter). Enforcement:
      * a LATER member's producer waits each EARLIER member's empty at the
        CONSUMER phase (that member's reader of THIS iteration has drained);
      * the EARLIEST member's producer waits every other member's empty at
        the PRODUCER phase (their readers of the PREVIOUS iteration — passes
        immediately on iteration 0), closing the ring.
    Where a same-WG data dependence already orders the aliased accesses the
    wait is redundant but cheap; it exists for WG topologies with no such
    chain (case4 FA-bwd flake, 2026-07-10: two D-load channels merged, the
    second WG's local_store raced the first channel's reader — dQ/dK/dV
    corrupt by 1e3 while every cycle-model check passed).
    """
    if rctx.sem_set is None:
        return []
    safety = getattr(rctx, "_alias_group_safety", {}).get(loop.loop_id, {})
    own = next((b for b in loop.schedule.buffers if b.id == buf_id), None)
    if own is None or own.merge_group_id is None or own.kind != "smem":
        return []
    mgid = own.merge_group_id
    if not safety.get(mgid, False):
        return []  # reuse was dropped at alloc time — bytes are private
    members = sorted(
        (
            b
            for b in loop.schedule.buffers
            if b.merge_group_id == mgid and b.kind == "smem"
        ),
        key=lambda b: (b.live_start, b.id),
    )
    if len(members) < 2:
        return []
    idx = getattr(rctx, "_ls_by_buffer", None)
    if idx is None:
        idx = {}
        for ls in rctx.sem_set.lowered:
            if ls.sem.buffer is not None:
                idx.setdefault(
                    (ls.sem.buffer.loop_id, ls.sem.buffer.buffer_id), []
                ).append(ls)
        rctx._ls_by_buffer = idx
    own_key = (own.live_start, own.id)
    is_earliest = own_key == (members[0].live_start, members[0].id)
    targets = (
        [m for m in members if m.id != own.id]
        if is_earliest
        else [m for m in members if (m.live_start, m.id) < own_key]
    )
    out: list[str] = []
    for m in targets:
        seen: set[str] = set()
        for ls in idx.get((loop.loop_id, m.id), []):
            if ls.alloc_empty_stmt is None or ls.empty_name in seen:
                continue
            seen.add(ls.empty_name)
            if is_earliest:
                phase = f"({ls.phase_expr} ^ 1)"
                note = "alias wrap"
            else:
                phase = ls.phase_expr
                note = "alias predecessor"
            out.append(
                f"tlx.barrier_wait({ls.empty_name}[{ls.slot_expr}], {phase})"
                f"  # {note} (merge group {mgid})"
            )
    return out


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


def _semir_producer_expect_bytes(
    loop_id: int, node_id: int, rctx: RenderCtx
) -> list[str]:
    """`tlx.barrier_expect_bytes(...)` lines for a TMA producer, before the load."""
    if not _use_semaphore_ir() or rctx.sem_set is None:
        return []
    return rctx.sem_set.expect_bytes_for(loop_id, node_id)


def _semir_producer_mbarriers(loop_id: int, node_id: int, rctx: RenderCtx) -> list[str]:
    """Slot exprs for a producer's mBarriers list (MMA / TMA / TMEM_COPY)."""
    if not _use_semaphore_ir() or rctx.sem_set is None:
        return []
    return rctx.sem_set.mbarriers_for(loop_id, node_id)


def _semir_producer_barrier_for_tma(
    loop_id: int, node_id: int, rctx: RenderCtx
) -> str | None:
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


def _semir_emit_consumer_block(
    n: Node, g: ScheduleGraph, loop: Loop, rctx: RenderCtx, lines: "_Lines"
) -> None:
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
        if n.op_kind == "ttg.memdesc_trans":
            # A memdesc_trans is an address/layout calc for a downstream MMA
            # that reads the channel SMEM directly — never a register consumer.
            # Materializing it here (local_load → local_trans of a register)
            # would crash. Its handshake is owned by the MMA operand path, which
            # looks up this trans node.
            continue
        if sem.buffer is None:
            # Signal-only semaphore: just the wait, no local_load.
            # Same-stream dedup: when one semaphore fans out to several
            # consumers in the SAME warp group (e.g. loop-carry MMA release
            # read twice per iter), the producer arrives ONCE per iteration —
            # a second wait on the same slot+phase races the hardware arrive
            # and can livelock the device. Program order makes the first wait
            # cover all later same-stream consumers.
            seen = getattr(rctx, "_emitted_full_waits", None)
            if seen is not None:
                if wait in seen:
                    continue
                seen.add(wait)
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


def _semir_mma_operand_waits_and_mbarriers(
    op: Op, g: ScheduleGraph, loop: Loop, rctx: RenderCtx, node: Node | None = None
) -> tuple[list[str], list[str]]:
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
        # The cross-WG consumer recorded in the schedule may be the alloc node
        # (operand buffer lives in the MMA WG) OR — when modulo co-locates the
        # operand alloc with its producer load — the memdesc_trans node that
        # reads the SMEM operand into the MMA. Look up both so the MMA itself
        # owns the handshake (wait on full + recycle empty), instead of a
        # register-materializing consumer block (which would `local_trans` a
        # register and crash).
        cand_ids = {alloc_node.id}
        trans_node = nodes_by_op.get(operand.op_id)
        if trans_node is not None:
            cand_ids.add(trans_node.id)
        cons_sems = [
            ls
            for cid in cand_ids
            for ls in rctx.sem_set.by_consumer.get((loop.loop_id, cid), [])
        ]
        if cons_sems:
            for ls in cons_sems:
                if ls.sem_id in seen_sems:
                    continue
                seen_sems.add(ls.sem_id)
                # Use the semaphore's own recorded consumer node for the wait.
                cnode = (
                    ls.sem.consumers[0].node.node_id
                    if ls.sem.consumers
                    else alloc_node.id
                )
                wait = ls.consumer_wait_at.get(cnode) or ls.consumer_wait_at.get(
                    alloc_node.id
                )
                # Multi-consumer dedup: when >1 MMA in this WG reads the same
                # single-buffered operand (FA-bwd dO → dpT + dV), its `_full`
                # completes once per load — wait it only on the FIRST consumer
                # (a 2nd wait on the same phase deadlocks); its `_empty`
                # (arrive_count=1) must be signalled only by the LAST consumer
                # (else it frees early and the producer overwrites mid-read).
                # Key the dedup on the OPERAND buffer (the alloc this MMA reads),
                # matching _opbuf_users — the semaphore's own buffer_id may be the
                # cross-WG channel's paired buffer (e.g. dsT: operand buf3 vs
                # channel buf4), which wouldn't match.
                obuf = next(
                    (b for b in loop.schedule.buffers if b.def_op == alloc_op_id),
                    None,
                )
                bufid = (
                    obuf.id
                    if obuf is not None
                    else (
                        ls.sem.buffer.buffer_id if ls.sem.buffer is not None else None
                    )
                )
                users = (
                    (getattr(rctx, "_opbuf_users", {}) or {}).get(bufid, [])
                    if bufid is not None
                    else []
                )
                is_first = node is None or not users or node.id == users[0]
                if wait and is_first:
                    waits.append(wait)
                # HW recycle: EVERY consuming MMA signals EMPTY. The empty's
                # arrive_count = #consumers (SemIR sums the per-consumer
                # cross_wg_barriers), so all N arrives are required — do NOT
                # dedup (that would leave the producer's empty-wait short → hang).
                # (The FULL wait above IS deduped: a load's full completes once.)
                if ls.alloc_empty_stmt is not None:
                    mbar.append(f"{ls.empty_name}[{ls.slot_expr}]")
            continue
        # If a TMEM bridge channel covers this operand, skip the intra-WG
        # fallback — the bridge's own emit path (downstream in async_dot)
        # adds the wait + empty mBarrier. Without this skip we'd emit
        # duplicate barrier_waits and double-recycle the empty barrier.
        if any(
            c.kind == "tmem" and c.alloc_op_id == alloc_op_id for c in rctx.channels
        ):
            continue
        # Intra-WG fallback: no SemIR cross-WG semaphore for this operand
        # (producer + consumer share a WG). Use the legacy `<buf>_full`/
        # `<buf>_empty` pair allocated in the `extra` carve-out. Per-buf
        # slot/phase use the buffer's ring count (matches the load).
        buf = next((b for b in loop.schedule.buffers if b.def_op == alloc_op_id), None)
        if buf is None:
            continue
        # If a TMA-fed cross-WG SemIR semaphore already covers this buffer, its
        # consumer handshake is emitted by the MMA-node SemIR path (consumer
        # waits + mBarriers, keyed on the MMA node). Adding the legacy `[buf]`
        # handshake here too would double-wait/double-arrive the same barrier.
        # (Intra-WG TMA buffers never reach this fallback — their consumer is
        # the alloc node, handled by the cons_sems branch above.)
        if rctx.sem_set is not None and any(
            ls.sem.buffer is not None
            and ls.sem.buffer.buffer_id == buf.id
            and ls.sem.producers
            and ls.sem.producers[0].async_kind == AsyncKind.TMA_LOAD
            for ls in rctx.sem_set.lowered
        ):
            continue
        buf_var = rctx.buffer_var.get((loop.loop_id, buf.id))
        if buf_var is None:
            continue
        idx, ph = _ring_exprs(buf.count, rctx)
        # Multi-consumer dedup: when several MMAs in this WG read the same
        # single-buffered operand (e.g. FA-bwd dO feeds both dpT and dV), its
        # `_full` completes ONCE per load — wait it only on the FIRST consumer
        # (a second wait on the same phase would block forever) and recycle the
        # `_empty` only on the LAST consumer (so the producer can't overwrite
        # while an earlier-issued MMA still reads). Matches the hand-written
        # kernel's single do_fulls wait + single do_empties arrive.
        users = (getattr(rctx, "_opbuf_users", {}) or {}).get(buf.id, [])
        is_first = node is None or not users or node.id == users[0]
        if is_first:
            waits.append(f"tlx.barrier_wait({_bar_full(buf_var)}[{idx}], {ph})")
        # Every consuming MMA recycles EMPTY (arrive_count=#consumers in SemIR).
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
    # Producer-side data-channel triples (wait-empty → store → arrive-full)
    # sink to the END of the loop body when the caller provides a deferral
    # list (see _emit_warp_group). The stored value has no same-WG reader —
    # only the cross-WG consumer needs it — so the handshake carries no
    # ordering constraint within this body, while its EMPTY wait can stall
    # the stream for the whole downstream round-trip. Emitting the triple
    # after the body's independent compute hides that stall (measured on
    # FA-fwd at II=1325: the row-sum reduce trapped below the alpha-channel
    # wait cost 8.7% at (1,32,8192); sunk triples restore baseline).
    # Signal-only arrives stay inline: loop-carry release signals (e.g. the
    # acc-store → PV-MMA edge) sit ON the recurrence critical path.
    deferred = getattr(rctx, "_deferred_producer_triples", None)

    def _put(stmt: str, data_channel: bool) -> None:
        nonlocal lines
        if data_channel and deferred is not None:
            deferred.append(stmt)
        else:
            lines += stmt

    op = g.ops.get(n.op_ref) if n.op_ref else None
    for ls in rctx.sem_set.by_producer.get((loop.loop_id, n.id), []):
        sem = ls.sem
        # Skip HW-issued releases — descriptor_load / async_dot handles them.
        prod = next((p for p in sem.producers if p.node.node_id == n.id), None)
        if prod is None or prod.async_kind != AsyncKind.NONE:
            continue
        is_data = sem.buffer is not None
        # SW producer: wait empty (unless is_released), store, arrive full.
        if w := ls.producer_wait_at.get(n.id):
            _put(w, is_data)
            # Alias safety: when this buffer shares bytes with other
            # merge-group members (Step 4.5 reuse), also wait until the
            # aliased members' readers have drained — cycle-disjoint
            # lifetimes are a model property, only barriers order the
            # hardware (case4 FA-bwd flake, 2026-07-10).
            if sem.buffer is not None:
                for aw in _alias_wait_stmts(loop, rctx, sem.buffer.buffer_id):
                    _put(aw, is_data)
        if sem.buffer is not None:
            buf_var = rctx.buffer_var.get((sem.buffer.loop_id, sem.buffer.buffer_id))
            if buf_var is not None:
                _put(f"tlx.local_store({buf_var}[{ls.slot_expr}], {value_var})", True)
                # If a consumer reads this channel via the async proxy (a TMA
                # descriptor_store reads SMEM directly), the producer's
                # generic-proxy write must be fenced before the full-arrive so
                # the TMA sees it. Register consumers don't need it, but it's
                # only emitted when a TMA store consumes the channel.
                if _channel_has_tma_store_consumer(sem, loop):
                    _put("tlx.fence_async_shared()", True)
        if a := ls.producer_arrive_at.get(n.id):
            _put(a, is_data)

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
        if (
            bridge_op
            and bridge_op.operands
            and isinstance(bridge_op.operands[0], OpRef)
            and bridge_op.operands[0].op_id == op.op_id
        ):
            _put(
                f"tlx.barrier_wait({_bar_empty(c.name)}[0], "
                f"(_it & 1) ^ 1)  # TMEM bridge",
                True,
            )
            _put(f"tlx.local_store({c.name}[0], {value_var})", True)
            _put(f"tlx.barrier_arrive({_bar_full(c.name)}[0], 1)", True)


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


def _emit_preamble(
    g: ScheduleGraph, loop: Loop, rctx: RenderCtx, lines: _Lines
) -> None:
    lines += "# ── Preamble (function-scope ops before the loop) ──"
    pre_ops = _ops_before_loop(g, loop)
    # Descriptors consumed by an in-loop tt.descriptor_reduce (FA-bwd dQ) are
    # re-materialized inside the consuming task, not captured from the preamble:
    # an in-task async_descriptor_store infers an nvmma_shared tensordesc type
    # that mismatches the bare type at the warp_specialize capture boundary.
    reduce_descs = _descriptor_reduce_desc_ops(g)
    for op in pre_ops:
        if op.kind in _SKIP_FUNCTION_SCOPE:
            continue
        if op.kind not in _NAMED_FUNCTION_OPS:
            continue
        if op.op_id in reduce_descs:
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


def _signal_only_buffer_ids(loop: Loop) -> set[int]:
    """Buffer ids that carry NO data — paired only with a backward (loop-carry
    release) cross-WG barrier. Such a barrier is a slot-free SIGNAL (e.g. the
    acc_tmem release in blockwise scaled_mm: the promotion tells the MMA the TMEM
    slot is free; the real data lives in TMEM). Materializing a SMEM data buffer
    for it is pure waste — a 128x128 fp32 phantom is 64 KB. A buffer paired by any
    FORWARD barrier, or produced/consumed by a node, carries real data and is
    kept (e.g. the sa channel, whose store/load the channel path emits)."""
    cyc = {n.id: n.schedule_cycle for n in loop.schedule.nodes}
    data_used: set[int] = set()
    for n in loop.schedule.nodes:
        if n.produces_buffer is not None:
            data_used.add(n.produces_buffer)
        data_used.update(n.consumes_buffers or [])
    fwd_paired: set[int] = set()
    bwd_paired: set[int] = set()
    for cb in loop.schedule.cross_wg_barriers:
        if cb.paired_buffer_id is None:
            continue
        pc, cc = cyc.get(cb.producer_node), cyc.get(cb.consumer_node)
        if pc is not None and cc is not None and pc > cc:
            bwd_paired.add(cb.paired_buffer_id)
        else:
            fwd_paired.add(cb.paired_buffer_id)
    return {b for b in bwd_paired if b not in fwd_paired and b not in data_used}


def _emit_buffers(loop: Loop, g: ScheduleGraph, rctx: RenderCtx, lines: _Lines) -> None:
    lines += "# ── Multi-buffered allocations (from modulo's lifetime analysis) ──"
    loop_tag = "inner" if not loop.is_outer else "outer"
    signal_only = _signal_only_buffer_ids(loop)
    # Track the FIRST allocated variable for each merge_group_id so subsequent
    # buffers in the same group emit `reuse=<first_var>` (Step 4.5 says they
    # have disjoint lifetimes — same physical bytes, different time slots).
    merge_group_owner: dict[int, str] = {}
    # Alias-safety verdict per SMEM merge group: guardable groups get alias
    # waits at their producers (_alias_wait_stmts); unguardable multi-member
    # groups have their reuse dropped right here. Stashed on rctx for the
    # producer-side emitters.
    alias_safety = _alias_group_safety(loop)
    if not hasattr(rctx, "_alias_group_safety"):
        rctx._alias_group_safety = {}
    rctx._alias_group_safety[loop.loop_id] = alias_safety
    for b in loop.schedule.buffers:
        # Signal-only loop-carry-release buffer: no data, so no alloc (the
        # handshake uses its own named barrier, not this buffer). Skips the 64 KB
        # phantom acc_tmem-release buffer in blockwise scaled_mm.
        if b.id in signal_only:
            continue
        # Per-loop unique name to avoid id collisions between inner/outer.
        var = f"L{loop.loop_id}_{_buffer_var_name(b)}"
        rctx.buffer_var[(loop.loop_id, b.id)] = var
        # A double-buffered buffer and its paired barrier share a def_op; the
        # barrier must not overwrite the data buffer's alloc_op_var mapping (else
        # a data operand renders as the barrier var, e.g. `L0_bar_4`).
        if b.def_op and b.kind != "barrier":
            rctx.alloc_op_var[b.def_op] = var
        if b.kind == "smem":
            # 1D shapes need trailing comma in Python tuple syntax
            # (`(256,)` not `(256)`); 2D+ are fine as-is.
            shape = (
                str(b.shape[0]) + ","
                if len(b.shape) == 1
                else ", ".join(str(d) for d in b.shape)
            )
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
            origin = (
                "channel for cross-WG hand-off"
                if b.def_op is None
                else f"modulo lifetime [{b.live_start}..{b.live_end}], "
                f"II={loop.schedule.II}"
            )
            mgid = b.merge_group_id
            if mgid is not None and not alias_safety.get(mgid, True):
                # Unguardable alias (HW producer / signal-only / non-channel
                # member): allocate private bytes instead of racing on shared
                # ones. Costs SMEM, never correctness.
                origin += f"; merge group {mgid} reuse DROPPED (unsynchronizable alias)"
                mgid = None
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
                pshape_str = (
                    str(pshape_dims[0]) + ","
                    if len(pshape_dims) == 1
                    else ", ".join(str(d) for d in pshape_dims)
                )
                names = []
                for gi in range(b.partition_count):
                    gvar = f"{var}_g{gi}"
                    names.append(gvar)
                    lines += (
                        f"{gvar} = tlx.local_alloc(({pshape_str}), {dtype}, "
                        f"{b.count}{reuse})"
                    )
                rctx.partition_buffer_names[(loop.loop_id, b.id)] = names
                if b.def_op:
                    rctx.partition_alloc_names[b.def_op] = names
                # `var` (legacy name) aliases group 0 — keep so legacy code
                # that references `buffer_var[key]` resolves to a real alloc.
                lines += f"{var} = {names[0]}"
            else:
                lines += (
                    f"{var} = tlx.local_alloc(({shape}), {dtype}, {b.count}{reuse})"
                )
            if mgid is not None and mgid not in merge_group_owner:
                merge_group_owner[mgid] = var
        elif b.kind == "tmem":
            # Emitter capability (EmitterCaps.kMaxTMEMBlockM in
            # ModuloSchedulePass.cpp): TMEM accumulators support
            # blockM <= 128 — no MMA splitting for larger tiles yet.
            # Fail clearly instead of emitting a kernel that traps
            # (case2's 256-blockM pre_modulo is the known instance).
            if len(b.shape) >= 1 and b.shape[0] > 128:
                raise NotImplementedError(
                    f"TMEM buffer {b.id} has blockM={b.shape[0]} > 128; "
                    "the emitter cannot split MMAs for tiles beyond the "
                    "TMEM row limit (EmitterCaps.kMaxTMEMBlockM). "
                    "Regenerate the schedule with blockM <= 128.")
            shape = (
                str(b.shape[0]) + ","
                if len(b.shape) == 1
                else ", ".join(str(d) for d in b.shape)
            )
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
            count = b.count
            ring = rctx.skew_ring_by_op.get(b.def_op) if b.def_op else None
            if ring is not None and ring["depth"] > count:
                count = ring["depth"]
                origin_suffix += f"; intra-WG skew ring depth={count}"
            lines += (
                f"# {loop_tag}-loop buf {b.id}: TMEM count={count} "
                f"(producer→consumer pipelining across iters{origin_suffix})"
            )
            lines += (
                f"{var} = tlx.local_alloc(({shape}), {dtype}, "
                f"{count}, tlx.storage_kind.tmem{reuse})"
            )
            if ring is not None:
                ring["var"] = var
                rctx.skew_ring[var] = ring
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
    # TMEM interval-coloring: disjoint-lifetime accumulators share a
    # storage_alias_spec (FA-bwd: qkT and dQ → one slot), so the function-scope
    # accs fit the 512-col TMEM budget. {alloc_op_id: color} for aliased accs.
    tmem_alias = _tmem_alias_groups(g)
    tmem_spec_var: dict[int, str] = {}  # color → emitted spec var name

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
            shape_dt = (
                _parse_tensor_shape(op.result_types[0]) if op.result_types else None
            )
            if not shape_dt:
                continue
            shape, dtype_str = shape_dt
            shape_str = ", ".join(str(d) for d in shape) + (
                "," if len(shape) == 1 else ""
            )
            dtype = _dtype_str_to_tl(dtype_str)
            name = f"q_smem_{len(rctx.alloc_op_var)}"
            rctx.alloc_op_var[op.op_id] = name
            lines += (
                f"# {name}: function-scope SMEM alloc (e.g., per-tile "
                f"resident Q tile in non-persistent FA)"
            )
            lines += f"{name} = tlx.local_alloc(({shape_str}), {dtype}, 1)"
            # If this alloc is fed by a function-scope tt.descriptor_load,
            # track it: we need to emit the TMA load (in MEM-role WG) +
            # consumer-side wait (in MMA-role WG) outside the K-loop.
            if op.operands and isinstance(op.operands[0], OpRef):
                load_op = g.ops.get(op.operands[0].op_id)
                if (
                    load_op
                    and load_op.kind == "tt.descriptor_load"
                    and load_op.scope == "function"
                ):
                    rctx.fn_scope_loads.append(
                        {
                            "alloc_var": name,
                            "alloc_op_id": op.op_id,
                            "load_op_id": load_op.op_id,
                            "load_op": load_op,
                        }
                    )
            continue
        if op.kind == "ttng.tmem_alloc":
            if op.op_id in rctx.alloc_op_var:
                continue  # already emitted from loop.buffers list
            shape_dt = (
                _parse_tensor_shape(op.result_types[0]) if op.result_types else None
            )
            shape = shape_dt[0] if shape_dt else [128, 128]
            dtype = _dtype_str_to_tl(shape_dt[1]) if shape_dt else "tl.float32"
            # Emitter capability (EmitterCaps.kMaxTMEMBlockM): TMEM
            # accumulators support blockM <= 128 — fail clearly instead of
            # emitting a kernel that traps (case2's 256-blockM pre_modulo
            # is the known instance).
            if shape and shape[0] > 128:
                raise NotImplementedError(
                    f"function-scope TMEM alloc has blockM={shape[0]} > 128; "
                    "the emitter cannot split MMAs for tiles beyond the TMEM "
                    "row limit (EmitterCaps.kMaxTMEMBlockM). Regenerate the "
                    "schedule with blockM <= 128.")
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
            # Intra-WG skew ring: the async producer's destination needs
            # (skew gap + 1) slots so issue overlaps the consumer's stage.
            ring = rctx.skew_ring_by_op.get(op.op_id)
            count = ring["depth"] if ring is not None else 1
            if ring is not None:
                ring["var"] = name
                rctx.skew_ring[name] = ring
            # If this acc is in an aliased color group, route it through a
            # shared storage_alias_spec (no set_buffer_overlap → all members
            # overlap at offset 0, size=max; safe since lifetimes are disjoint).
            color = tmem_alias.get(op.op_id)
            if color is not None:
                spec = tmem_spec_var.get(color)
                if spec is None:
                    spec = f"tmem_alias_{color}"
                    tmem_spec_var[color] = spec
                    lines += (
                        f"{spec} = tlx.storage_alias_spec("
                        f"storage=tlx.storage_kind.tmem)"
                    )
                lines += (
                    f"{name} = tlx.local_alloc(({shape_str}), {dtype}, {count}, "
                    f"tlx.storage_kind.tmem, reuse={spec})"
                )
            else:
                lines += (
                    f"{name} = tlx.local_alloc(({shape_str}), {dtype}, {count}, "
                    f"tlx.storage_kind.tmem)"
                )
    # Epilogue staging SMEM (for the descriptor_store) — derived from the
    # store op's source tensor shape.
    epi_store = next(
        (
            op
            for op in g.ops.values()
            if op.scope == "function" and op.kind == "tt.descriptor_store"
        ),
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
                bs = _parse_desc_block_shape(
                    desc_op.result_types[0] if desc_op.result_types else ""
                )
                if bs:
                    shape, dtype = bs[0], _dtype_str_to_tl(bs[1])
        elif isinstance(epi_store.operands[0], ArgRef):
            arg = next(
                (a for a in g.kernel.args if a.name == epi_store.operands[0].name), None
            )
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
        # Pass A.5: a data-partitioned accumulator stores one (m_size, BN) group
        # at a time reusing a single staging buffer — shrink c_smem to
        # (m_size, BN). Halves the epilogue SMEM so deeper operand rings fit.
        part_m = 0
        for lp in g.loops:
            for b in lp.schedule.buffers:
                if b.kind == "tmem" and b.partition_count > 1:
                    part_m = b.m_size
                    break
            if part_m:
                break
        if part_m and len(shape) >= 2:
            # A.5 and A.7 both reshape this staging buffer; the epilogue path
            # treats them as mutually exclusive (`_find_partition_chain` runs
            # only `if not sub_info`). Assert that here so the shrink is a
            # load-bearing invariant rather than an incidental one.
            assert sub_count <= 1, (
                "Pass A.5 (data partition) and Pass A.7 (epilogue subtile) "
                "both reshaped c_smem; they are mutually exclusive by design"
            )
            shape = [part_m, shape[1]]
        shape_str = ", ".join(str(d) for d in shape)
        lines += f"c_smem = tlx.local_alloc(({shape_str}), {dtype}, 1)"
    # Dedicated staging SMEM for an in-loop TMA reduce (e.g. FA-bwd dQ
    # atomic-add). Separate from c_smem because the reduce runs inside the loop
    # in its own warp group while the dK/dV epilogue stores reuse c_smem
    # post-loop in another warp group — sharing would race across groups.
    redu_op = next(
        (op for op in g.ops.values() if op.kind == "tt.descriptor_reduce"),
        None,
    )
    if redu_op and redu_op.operands:
        rshape: list[int] = [128, 128]
        rdtype = "tl.float16"
        if isinstance(redu_op.operands[0], OpRef):
            desc_op = g.ops.get(redu_op.operands[0].op_id)
            if desc_op:
                bs = _parse_desc_block_shape(
                    desc_op.result_types[0] if desc_op.result_types else ""
                )
                if bs:
                    rshape, rdtype = bs[0], _dtype_str_to_tl(bs[1])
        elif isinstance(redu_op.operands[0], ArgRef):
            arg = next(
                (a for a in g.kernel.args if a.name == redu_op.operands[0].name), None
            )
            if arg:
                bs = _parse_desc_block_shape(arg.type)
                if bs:
                    rshape, rdtype = bs[0], _dtype_str_to_tl(bs[1])
        rshape_str = ", ".join(str(d) for d in rshape)
        lines += f"dq_smem = tlx.local_alloc(({rshape_str}), {rdtype}, 1)"
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
    # Number of distinct consumer WGs reading this buffer. >1 when one buffer
    # feeds multiple consumers (e.g. case7 dout → MMA + bias-reduce); the empty
    # barrier's arrive_count must equal this so the producer waits for ALL
    # consumers to release the slot before recycling it.
    num_consumers: int = 1


def _result_feeds_descriptor_store(
    g: ScheduleGraph, for_op_id: str, idx: int, epi_scopes: set[str]
) -> bool:
    """Forward-walk from scf.for result[idx]: True iff it reaches a
    `tt.descriptor_store` (the default epilogue's TMA store) before any pointer
    `tt.store`. Distinguishes a genuine cross-WG epilogue value (case9 blockwise
    scaled_mm: running-sum → truncf → descriptor_store) from a fused reduction
    (case7 bias db: reduce → convert_layout → tt.store), which is emitted in the
    producing WG by `_emit_outer_reduction_stores` and needs no staging channel.
    """
    seeds = [
        oid
        for oid, o in g.ops.items()
        if o.scope in epi_scopes
        and any(
            isinstance(x, OpRef) and x.op_id == for_op_id and x.result_idx == idx
            for x in o.operands
        )
    ]
    seen = set(seeds)
    stack = list(seeds)
    while stack:
        cur = stack.pop()
        op = g.ops.get(cur)
        if op is None:
            continue
        if op.kind == "tt.descriptor_store":
            return True
        if op.kind == "tt.store":
            continue  # pointer-store reduction terminal — emitted in producer WG
        for oid, o in g.ops.items():
            if oid in seen:
                continue
            if any(isinstance(x, OpRef) and x.op_id == cur for x in o.operands):
                seen.add(oid)
                stack.append(oid)
    return False


def _epilogue_colocation_wg(g: ScheduleGraph) -> int | None:
    """The single inner warp group that produces the outer-loop epilogue's
    register input — the WG the epilogue can be CO-LOCATED into (promotion +
    store in one task, like the hand-written kernel), dropping the cross-WG SMEM
    staging entirely. Returns None when there are zero or multiple such producers
    (multi-producer, e.g. FA-bwd dK/dV, keeps the SMEM fallback) or the kernel is
    non-persistent (no separate outer-epilogue task to merge)."""
    outer_scopes = {f"loop:{L.loop_id}" for L in g.loops if L.is_outer}
    if not outer_scopes:
        return None
    producers: set[int] = set()
    for loop in g.loops:
        if loop.is_outer:
            continue
        for_op = _find_loop_for(g, loop)
        if for_op is None:
            continue
        wg_of_op = {n.op_ref: n.warp_group for n in loop.schedule.nodes if n.op_ref}
        for idx, init, yld in _loop_iter_args(g, loop):
            if not isinstance(yld, OpRef):
                continue
            pw = wg_of_op.get(yld.op_id)
            if pw is None:
                continue
            if not (
                isinstance(init, ConstRef)
                and init.type
                and _TENSOR_TYPE_RE.search(init.type)
            ):
                continue
            if _result_feeds_descriptor_store(g, for_op.op_id, idx, outer_scopes):
                producers.add(pw)
    return next(iter(producers)) if len(producers) == 1 else None


def _derive_crossloop_result_channels(
    g: ScheduleGraph, rctx: RenderCtx
) -> list[dict[str, Any]]:
    """Detect references to scf.for results (= final iter_arg values) from
    function-scope ops (the default partition's epilogue). The value lives
    in the WG that wrote the last iter_arg yield — if that WG isn't the
    default partition, we need to stage through SMEM:

      producer WG (after inner loop): local_store + barrier_arrive
      default partition (before use):  barrier_wait + local_load

    Returns one descriptor per (loop, idx) pair needing staging.
    """
    out: list[dict[str, Any]] = []
    # Epilogue consumers of an inner-loop result live either at function scope
    # (non-persistent: ops after the single loop) or inside the OUTER loop body
    # (persistent: the per-tile epilogue). The outer scf.for carries no results
    # in these kernels, so any scf.for-result reference is to the inner loop.
    epi_scopes = {"function"} | {f"loop:{L.loop_id}" for L in g.loops if L.is_outer}
    for loop in g.loops:
        if loop.is_outer:
            continue  # only inner-loop iter_arg results need this staging
        specs = _loop_iter_args(g, loop)
        if not specs:
            continue
        # Build wg_of for this loop's nodes.
        wg_of_op: dict[str, int] = {
            n.op_ref: n.warp_group for n in loop.schedule.nodes if n.op_ref
        }
        for_op = _find_loop_for(g, loop)
        if for_op is None:
            continue
        outer_scopes = {f"loop:{L.loop_id}" for L in g.loops if L.is_outer}
        for idx, init, yld in specs:
            # Find the producer WG of the yield value.
            if not isinstance(yld, OpRef):
                continue
            prod_wg = wg_of_op.get(yld.op_id)
            if prod_wg is None:
                continue
            # Only stage a real register tensor value. Async tokens / memdesc
            # iter_args (e.g. the tmem_alloc token threaded through case7's inner
            # loop) are not registers and must not get a channel.
            if not (
                isinstance(init, ConstRef)
                and init.type
                and _TENSOR_TYPE_RE.search(init.type)
            ):
                continue
            # Function-scope epilogue consumer (non-persistent, e.g. case3 FA
            # m_i): the original rule — any non-`tt.store` reference to
            # result[idx].
            func_ref = any(
                op.scope == "function"
                and op.kind != "tt.store"
                and any(
                    isinstance(o, OpRef)
                    and o.result_idx == idx
                    and (f := g.ops.get(o.op_id))
                    and f.kind == "scf.for"
                    for o in op.operands
                )
                for op in g.ops.values()
            )
            # Outer-loop-scope epilogue consumer (persistent): only when the
            # value flows to the default TMA store (case9 blockwise running-sum),
            # NOT a producer-WG pointer-store reduction (case7 bias db).
            outer_ref = _result_feeds_descriptor_store(
                g, for_op.op_id, idx, outer_scopes
            )
            # Lever #2: if this producer WG will own the epilogue itself
            # (co-location), the register value never crosses a WG boundary — no
            # SMEM staging channel is needed. Only the SMEM fallback path (no
            # co-location, or a function-scope consumer) still stages.
            if outer_ref and prod_wg == _epilogue_colocation_wg(g):
                outer_ref = False
            if not (func_ref or outer_ref):
                continue
            # Resolve type / shape from the init.
            shape, dt = _parse_tensor_shape(init.type)
            dtype = _dtype_str_to_tl(dt)
            var_name = _iter_arg_python_name(loop.loop_id, idx, init)
            out.append(
                {
                    "bufname": f"epi_{var_name}_smem",
                    "shape": shape,
                    "dtype": dtype,
                    "loop_id": loop.loop_id,
                    "idx": idx,
                    "producer_wg": prod_wg,
                    "var_name": var_name,
                }
            )
    return out


def _alias_predecessors(
    channel_name: str, rctx: RenderCtx, graph: ScheduleGraph
) -> list[str]:
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
                    if (
                        isinstance(op.operands[2], OpRef)
                        and op.operands[2].op_id == b_def_op
                    ):
                        producer_wgs.add(wg)
                    if (
                        isinstance(op.operands[0], OpRef)
                        and op.operands[0].op_id == b_def_op
                    ):
                        consumer_wgs.add(wg)
                    if (
                        isinstance(op.operands[1], OpRef)
                        and op.operands[1].op_id == b_def_op
                    ):
                        consumer_wgs.add(wg)
                # tmem_load reads operand[0].
                if op.kind == "ttng.tmem_load" and op.operands:
                    if (
                        isinstance(op.operands[0], OpRef)
                        and op.operands[0].op_id == b_def_op
                    ):
                        consumer_wgs.add(wg)
                # tmem_store: MLIR layout is [dest, token, value, pred] —
                # operand[0] is the destination buffer.
                if op.kind == "ttng.tmem_store" and len(op.operands) >= 1:
                    if (
                        isinstance(op.operands[0], OpRef)
                        and op.operands[0].op_id == b_def_op
                    ):
                        producer_wgs.add(wg)
                # tmem_alloc(value) is itself the alloc — when it carries a
                # value operand, it both *is* the buffer AND stores to it.
                # If oid == b_def_op AND it has a value operand, this is the
                # bridge. The producer-side WG is wherever the value comes
                # from (chase the OpRef).
                if op.kind == "ttng.tmem_alloc" and oid == b_def_op and op.operands:
                    bridge_op_id = oid
                    val_ref = op.operands[0]
                    val_wg = (
                        wg_of.get(val_ref.op_id) if isinstance(val_ref, OpRef) else None
                    )
                    # Actual producer is the value's WG; the "consumer" is
                    # the WG that owns the alloc op (which will read it).
                    if val_wg is not None:
                        producer_wgs.add(val_wg)
                        consumer_wgs.add(wg)
            if not producer_wgs or not consumer_wgs:
                continue
            # Same-WG: skip the SMEM-style channel UNLESS it's a tmem_alloc(value)
            # bridge (bridge_op_id set). An intra-WG bridge (FA-bwd dsT/pT at
            # small BLOCK_M, where the value-store and the consuming MMA land in
            # one WG) still needs the producer local_store + full/empty handshake
            # emitted (the async MMA reads TMEM); routing it through the bridge
            # path provides exactly that, and prevents the broken intra-WG
            # fallback (which references undeclared `<buf>_full`).
            if producer_wgs == consumer_wgs and bridge_op_id is None:
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
                )
            )
    return out


def _derive_smem_bridge_channels(g: ScheduleGraph, inner: Loop) -> list[Channel]:
    """Intra-WG SMEM `local_alloc(value)` → MMA bridges — the SMEM analogue of
    the intra-WG `tmem_alloc(value)` bridge in `_derive_tmem_channels`.

    A register value staged to SMEM via `ttg.local_alloc(%val)` and consumed by
    MMA operand(s) in the SAME warp group. `cross_wg_barriers` only cover
    cross-WG staging, so an intra-WG such buffer is otherwise emitted with NO
    store into it and NO barriers (FA-bwd `dsT` when softmax + dK/dQ MMAs share
    a WG — the all-MMA-in-one-WG partition variants). Emit it as a bridge
    channel; `kind="tmem"` reuses the bridge store+handshake emission path
    verbatim (local_store + full/empty, no fence — MMA consumer). Storage is
    unaffected: the buffer stays SMEM (allocated by the normal buffer emission);
    the bridge only adds the value store and the completion barriers.
    """
    wg_of = {n.op_ref: n.warp_group for n in inner.schedule.nodes if n.op_ref}

    def _operand_alloc(op: Op, si: int) -> str | None:
        if len(op.operands) <= si or not isinstance(op.operands[si], OpRef):
            return None
        aid = op.operands[si].op_id
        mid = g.ops.get(aid)
        if (
            mid is not None
            and mid.kind == "ttg.memdesc_trans"
            and mid.operands
            and isinstance(mid.operands[0], OpRef)
        ):
            aid = mid.operands[0].op_id
        return aid

    out: list[Channel] = []
    for L in g.loops:
        for b in L.schedule.buffers:
            if b.kind != "smem" or not b.def_op:
                continue
            alloc = g.ops.get(b.def_op)
            if (
                alloc is None
                or alloc.kind != "ttg.local_alloc"
                or not alloc.operands
                or not isinstance(alloc.operands[0], OpRef)
            ):
                continue  # plain alloc (no staged value) — a load ring, not a bridge
            valop = g.ops.get(alloc.operands[0].op_id)
            if valop is not None and valop.kind == "tt.descriptor_load":
                continue  # TMA-fed ring — handled by the async-load path, not here
            val_wg = wg_of.get(alloc.operands[0].op_id)
            if val_wg is None:
                continue
            # MMA operand consumers of this buffer (count occurrences for the
            # empty barrier's arrive_count — each consuming MMA arrives once).
            cons_wgs: set[int] = set()
            n_uses = 0
            for oid, op in g.ops.items():
                if op.kind not in ("ttng.tc_gen5_mma", "ttng.tc_gen5_mma_scaled"):
                    continue
                w = wg_of.get(oid)
                if w is None:
                    continue
                for si in (0, 1):
                    if _operand_alloc(op, si) == b.def_op:
                        cons_wgs.add(w)
                        n_uses += 1
            if not cons_wgs or cons_wgs != {val_wg}:
                continue  # cross-WG (cross_wg_barriers cover it) or no MMA consumer
            if any(cb.paired_buffer_id == b.id for cb in L.schedule.cross_wg_barriers):
                continue  # already staged by a synthesized cross-WG channel
            # The bridge store/handshake protocol is DEPTH-1 by construction:
            # the producer store, its full/empty waits, and the operand-side
            # full wait are all emitted at slot [0] with (_it & 1) parity.
            # A buffer arriving here with count > 1 (schedule lifetime/II+1
            # depth) would be written at [0] but read at the WG ring counter
            # by the consuming MMAs — 2 of every `count` iterations read a
            # never-written slot (case4 merged-partition draw, 2026-07-11:
            # dsT depth-3 bridge fed dK/dQ garbage, errors growing with N;
            # dV clean). Clamp the buffer to the protocol's depth so the
            # data alloc, the barrier ring, and _ring_exprs all agree; the
            # depth-1 handshake already serialized the pipeline, so no
            # overlap is lost.
            b.count = 1
            out.append(
                Channel(
                    name=b.def_op,  # resolved to the buffer var by the caller
                    depth=1,
                    producer_wg=val_wg,
                    consumer_wg=val_wg,
                    kind="tmem",  # reuse bridge store+handshake emission
                    alloc_op_id=b.def_op,
                    bridge_op_id=b.def_op,
                    loop_id=L.loop_id,
                    buffer_id=b.id,
                    num_consumers=max(1, n_uses),
                )
            )
    return out


def _derive_channels(loop: Loop, rctx: RenderCtx) -> list[Channel]:
    """One Channel per cross-WG handoff that carries data through a buffer.

    Uses the schedule's `cross_wg_barriers` as the authoritative source —
    each entry already records which buffer (if any) is the data payload
    via `paired_buffer_id`. Entries without a paired buffer are pure
    handshake signals (named barriers) and don't need a Channel."""
    # Distinct consumer WGs per buffer — a buffer shared by >1 consumer WG
    # (e.g. case7 dout feeding both the MMA and the bias-reduce) needs its empty
    # barrier's arrive_count = #consumers (else the producer recycles the slot
    # after only one consumer releases it → mid-read overwrite).
    consumers_by_buf: dict[int, set[int]] = {}
    for cb in loop.schedule.cross_wg_barriers:
        if cb.paired_buffer_id is not None:
            consumers_by_buf.setdefault(cb.paired_buffer_id, set()).add(cb.consumer_wg)
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
                name=rctx.buffer_var.get(
                    (loop.loop_id, buf.id), f"L{loop.loop_id}_{_buffer_var_name(buf)}"
                ),
                depth=buf.count,
                producer_wg=cb.producer_wg,
                consumer_wg=cb.consumer_wg,
                buffer_id=buf_id,
                num_consumers=len(consumers_by_buf.get(buf_id, {cb.consumer_wg})),
            )
        )
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
        lines += (
            f"# {ch.name}: wg{ch.producer_wg} → wg{ch.consumer_wg}, "
            f"depth={ch.depth} (matches buffer ring count)"
        )
        lines += (
            f"{_bar_full(ch.name)} = tlx.alloc_barriers"
            f"(num_barriers={ch.depth}, arrive_count=1)"
        )
        lines += (
            f"{_bar_empty(ch.name)} = tlx.alloc_barriers"
            f"(num_barriers={ch.depth}, arrive_count={ch.num_consumers})"
        )
    # The outer-loop TMEM buffer is a cross-WG channel: TC partition writes
    # it (via MMA), default partition reads it (via tmem_load). Producer and
    # consumer are in different warp groups → needs full/empty mbarrier
    # pairs, exactly like the SMEM channels above. Depth and bank count
    # come from outer-loop schedule_loop.buffers[tmem].count.
    if have_separate_tmem_handoff and "acc_tmem" not in seen:
        lines += (
            f"# acc_tmem: cross-WG channel TC → default "
            f"(outer-loop TMEM buf, depth={tmem_count})"
        )
        lines += (
            f"acc_tmem_full = tlx.alloc_barriers"
            f"(num_barriers={tmem_count}, arrive_count=1)"
        )
        lines += (
            f"acc_tmem_empty = tlx.alloc_barriers"
            f"(num_barriers={tmem_count}, arrive_count=1)"
        )
        seen.add("acc_tmem")
    # Extra buffers: load → MMA pipelines that aren't cross-WG channels but
    # still need barriers (TMA loads are async).
    for name, depth in extra_buffers or []:
        if name in seen:
            continue
        seen.add(name)
        lines += f"# {name}: TMA-load → consumer barrier (intra-WG async)"
        lines += (
            f"{_bar_full(name)} = tlx.alloc_barriers"
            f"(num_barriers={depth}, arrive_count=1)"
        )
        lines += (
            f"{_bar_empty(name)} = tlx.alloc_barriers"
            f"(num_barriers={depth}, arrive_count=1)"
        )
    lines += ""


# ===========================================================================
# Async-task emission
# ===========================================================================


def _reassign_orphan_nodes(graph: ScheduleGraph, loop: Loop) -> None:
    """The partition pass leaves pipeline=NONE sink ops (e.g. the FA-bwd dQ
    `tt.descriptor_reduce`) at warp_group=-1 — no compute pipeline of their
    own. Fold each into the warp group that produces its data, so it executes
    in the same task that computed the value (matching how a hand-written WS
    kernel puts the dQ atomic-add in the reduction task next to its tmem_load).

    Walks the op's SSA operands transitively to the nearest producing node with
    a real warp group. Idempotent — repeats to a fixpoint so orphan chains
    resolve."""
    node_by_op = {n.op_ref: n for n in loop.schedule.nodes if n.op_ref}
    changed = True
    while changed:
        changed = False
        for n in loop.schedule.nodes:
            if n.warp_group != -1 or not n.op_ref:
                continue
            op = graph.ops.get(n.op_ref)
            if op is None:
                continue
            for operand in op.operands:
                src_op = getattr(operand, "op_id", None)
                if src_op is None:
                    continue
                producer = node_by_op.get(src_op)
                if producer is not None and producer.warp_group != -1:
                    n.warp_group = producer.warp_group
                    changed = True
                    break


def _epilogue_acc_wg(g: ScheduleGraph, rctx: RenderCtx) -> dict[str, int]:
    """Map each non-canonical epilogue-output TMEM accumulator (var name) to the
    warp group whose in-loop MMA produces it.

    The legacy single-accumulator handoff carves out one canonical `acc_tmem`.
    FA-bwd writes dK and dV into two *additional* distinct TMEM accumulators
    that an epilogue `tmem_load` reads; each needs its own full barrier that its
    producer WG commits after the loop. Result is cached on rctx."""
    cached = getattr(rctx, "_epilogue_acc_wg_cache", None)
    if cached is not None:
        return cached
    epi_vars: set[str] = set()
    for op in g.ops.values():
        if op.scope != "function" or op.kind != "ttng.tmem_load":
            continue
        if op.operands and isinstance(op.operands[0], OpRef):
            v = rctx.alloc_op_var.get(op.operands[0].op_id)
            if v and v != "acc_tmem":
                epi_vars.add(v)
    out: dict[str, int] = {}
    for L in g.loops:
        for n in L.schedule.nodes:
            if not n.op_ref or "mma" not in n.op_kind.lower():
                continue
            op = g.ops.get(n.op_ref)
            if op is None or len(op.operands) < 3:
                continue
            if not isinstance(op.operands[2], OpRef):
                continue
            v = rctx.alloc_op_var.get(op.operands[2].op_id)
            if v in epi_vars and n.warp_group != -1:
                out[v] = n.warp_group
    rctx._epilogue_acc_wg_cache = out
    return out


def _buf_mma_consumer_counts(g: ScheduleGraph) -> dict[str, int]:
    """{buffer_var_name: number of in-loop MMA nodes consuming it}. An operand
    buffer read by N MMAs in a WG has its EMPTY barrier arrived N times (each
    async_dot recycles via mBarriers), so its arrive_count must be N. Cross-WG
    channels get this via the SemIR per-consumer sum; intra-WG load buffers
    (the `extra` carve-out) are hardcoded to 1 and need this override."""
    out: dict[str, int] = {}
    for loop in g.loops:
        by_def: dict[str, int] = {}
        for n in loop.schedule.nodes:
            if not n.op_ref or "mma" not in n.op_kind.lower():
                continue
            op = g.ops.get(n.op_ref)
            if op is None:
                continue
            for o in op.operands[:2]:
                aid = getattr(o, "op_id", None)
                if aid is None:
                    continue
                mid = g.ops.get(aid)
                if mid is not None and mid.kind == "ttg.memdesc_trans" and mid.operands:
                    aid = getattr(mid.operands[0], "op_id", aid)
                by_def[aid] = by_def.get(aid, 0) + 1
        for b in loop.schedule.buffers:
            if b.def_op and by_def.get(b.def_op, 0) > 1:
                cnt = by_def[b.def_op]
                var = f"L{loop.loop_id}_{_buffer_var_name(b)}"
                out[var] = cnt
                out[b.id] = cnt  # by buffer id for semaphore-side lookup
                # Propagate across the merge group: a cross-WG channel's
                # synthesized data buffer (def_op=None) aliases the real alloc
                # the MMAs read (same merge_group_id) and is what the SemIR
                # semaphore references (e.g. dsT: MMAs read buf3, channel=buf4,
                # both mgid=3).
                if b.merge_group_id is not None:
                    for o in loop.schedule.buffers:
                        if o.merge_group_id == b.merge_group_id:
                            out[o.id] = cnt
                            out[f"L{loop.loop_id}_{_buffer_var_name(o)}"] = cnt
    return out


def _tmem_cyclic_occupies(lo: int, hi: int, ii: int) -> list[tuple[int, int]]:
    """Project an ABSOLUTE-issue-time lifetime [lo, hi] onto the cyclic [0, II)
    footprint it occupies. `lo`/`hi` are in absolute issue time (stage*II +
    within-II cycle), so the caller guarantees hi >= lo. A span >= II covers the
    whole loop; otherwise it is one [s, e) window that splits into two pieces
    when it crosses the II boundary. (Do NOT pass a raw within-II [lo, hi] where
    a prior-iteration consumer can make hi < lo — that inversion is exactly the
    bug this absolute-time formulation avoids.)"""
    length = hi - lo
    if length >= ii:
        return [(0, ii)]
    s = lo % ii
    e = s + length
    if e <= ii:
        return [(s, e)]
    return [(s, ii), (0, e - ii)]


def _tmem_lifetimes_conflict(
    la: tuple[int, int] | None, lb: tuple[int, int] | None, ii: int
) -> bool:
    """Two TMEM-accumulator lifetimes conflict iff their cyclic footprints
    intersect. `None` = whole-loop (an accumulator read only in the epilogue),
    which conflicts with everything."""
    if la is None or lb is None:
        return True
    oa = _tmem_cyclic_occupies(la[0], la[1], ii)
    ob = _tmem_cyclic_occupies(lb[0], lb[1], ii)
    return any(s1 < e2 and s2 < e1 for (s1, e1) in oa for (s2, e2) in ob)


def _tmem_alias_groups(g: ScheduleGraph) -> dict[str, int]:
    """Interval-color the function-scope TMEM accumulators so disjoint-lifetime
    accs share storage (FA-bwd: qkT and dQ never overlap → one slot). Returns
    {tmem_alloc_op_id: color_id}; only colors with ≥2 members get aliased.

    Lifetime of each acc, modulo the loop II:
      - accumulator (read only in the post-loop epilogue, i.e. dV/dK) → live the
        whole loop, conflicts with everything;
      - per-iter (in-loop consumer) → [producer_cycle .. last_consumer_cycle],
        which may wrap past II.
    Two per-iter lifetimes conflict iff their cyclic projections intersect.
    Cross-WG/iteration ordering is already enforced by the schedule's dataflow
    barriers (same guarantee the hand-written kernel's qk_p/dp_dq reuse relies
    on), so sharing a slot needs no new barrier."""
    loops = [L for L in g.loops if not L.is_outer] or g.loops
    if not loops:
        return {}
    loop = loops[0]
    II = max(loop.schedule.II, 1)

    # Only alias when TMEM would otherwise overflow the 512-column budget.
    # Aliasing forces two accumulators to share one physical slot, but the
    # emitter gives each its OWN recycle barrier (the shared slot's two readers
    # are not jointly gated), which is unsafe unless space pressure demands it.
    # When everything already fits (e.g. small BLOCK_M/HEAD_DIM), don't alias.
    def _tmem_cols(dims: list[int], bits: int) -> int:
        if len(dims) < 2:
            return 0
        return max(1, (dims[1] * bits + 31) // 32)

    total_cols = 0
    for b in loop.schedule.buffers:
        if b.kind == "tmem":
            total_cols += _tmem_cols(b.shape, b.element_bits)
    for op in g.ops.values():
        if op.scope == "function" and op.kind == "ttng.tmem_alloc" and op.result_types:
            sd = _parse_tensor_shape(op.result_types[0])
            if sd:
                bits = 16 if sd[1] in ("f16", "bf16") else 32
                total_cols += _tmem_cols(sd[0], bits)
    if total_cols <= 512:
        return {}

    def _is_mma(n: Node) -> bool:
        return bool(n.op_ref) and "mma" in n.op_kind.lower()

    # Epilogue (function-scope) tmem_load reads → accumulator marker.
    epi_read: set[str] = set()
    for op in g.ops.values():
        if op.scope == "function" and op.kind == "ttng.tmem_load" and op.operands:
            if isinstance(op.operands[0], OpRef):
                epi_read.add(op.operands[0].op_id)

    # Per-alloc producer + in-loop consumer cycles.
    accs: dict[str, dict] = {}
    for n in loop.schedule.nodes:
        if not n.op_ref:
            continue
        op = g.ops.get(n.op_ref)
        if op is None:
            continue
        # Absolute issue time = stage*II + within-II cycle. Using absolute time
        # (not the raw within-II cycle) is what keeps a lifetime's hi >= lo when
        # a consumer reads a value produced in an EARLIER iteration (multi-stage
        # / skewed schedules) — otherwise max(cons_cycle) < min(prod_cycle)
        # inverts the interval and the cyclic-overlap test silently misses real
        # conflicts, mis-coloring overlapping accumulators onto one slot.
        abs_t = n.schedule_stage * II + n.schedule_cycle
        if _is_mma(n) and len(op.operands) >= 3 and isinstance(op.operands[2], OpRef):
            aid = op.operands[2].op_id
            if g.ops.get(aid) and g.ops[aid].kind == "ttng.tmem_alloc":
                accs.setdefault(aid, {"prod": [], "cons": []})["prod"].append(abs_t)
        if (
            op.kind == "ttng.tmem_load"
            and op.operands
            and isinstance(op.operands[0], OpRef)
        ):
            aid = op.operands[0].op_id
            if g.ops.get(aid) and g.ops[aid].kind == "ttng.tmem_alloc":
                accs.setdefault(aid, {"prod": [], "cons": []})["cons"].append(abs_t)

    # Build lifetimes. whole=None means whole-loop (conflicts all).
    life: dict[str, tuple | None] = {}
    for aid, d in accs.items():
        if not d["prod"]:
            continue
        if aid in epi_read or not d["cons"]:
            life[aid] = None  # accumulator → whole loop
        else:
            life[aid] = (min(d["prod"]), max(d["cons"]))

    # Greedy coloring (largest-first for stable packing). Conflict = cyclic
    # lifetime overlap; see _tmem_lifetimes_conflict / _tmem_cyclic_occupies.
    order = sorted(life.keys(), key=lambda a: (life[a] is not None, a))
    color: dict[str, int] = {}
    colors: list[list[str]] = []
    for aid in order:
        placed = False
        for ci, members in enumerate(colors):
            if all(
                not _tmem_lifetimes_conflict(life[aid], life[m], II) for m in members
            ):
                members.append(aid)
                color[aid] = ci
                placed = True
                break
        if not placed:
            color[aid] = len(colors)
            colors.append([aid])
    # Only colors with ≥2 members are worth aliasing.
    multi = {ci for ci, m in enumerate(colors) if len(m) > 1}
    return {aid: ci for aid, ci in color.items() if ci in multi}


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


def _emit_default_partition(
    g: ScheduleGraph, loop: Loop, rctx: RenderCtx, lines: _Lines
) -> None:
    # The caller (`_emit_uwg_body_impl`) has already opened the
    # `with tlx.async_task("default"):` block — emit body directly here.
    epi_ops = _ops_after_loop(g, loop)
    if True:
        _n0 = len(lines.buf)  # detect an empty default-task body below
        # acc_tmem is a legacy carve-out under SemIR — full+empty pair. Only
        # wait when an MMA actually produces it (else nothing arrives → hang;
        # e.g. case6 LayerNorm has no MMA).
        if rctx.has_acc_tmem_handoff:
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
                # Resolve the source TMEM alloc → its emitted buffer var so a
                # multi-output epilogue (FA-bwd: dV from acc_tmem_6, dK from
                # acc_tmem_8) reads the correct accumulator instead of all
                # aliasing acc_tmem[0]. Each non-canonical accumulator gets its
                # own full/empty handoff (see _emit_epilogue_acc_handoff).
                src_var = "acc_tmem"
                if op.operands and isinstance(op.operands[0], OpRef):
                    src_var = rctx.alloc_op_var.get(op.operands[0].op_id, "acc_tmem")
                if src_var == "acc_tmem":
                    name = "acc"
                    lines += f"{name} = tlx.local_load({src_var}[0])"
                    # acc_tmem is a legacy carve-out under SemIR — full+empty.
                    lines += "tlx.barrier_arrive(acc_tmem_empty[0], 1)"
                else:
                    name = f"acc_{rctx.fresh_idx()}"
                    lines += f"tlx.barrier_wait({src_var}_full[0], 0)"
                    lines += f"{name} = tlx.local_load({src_var}[0])"
                    lines += f"tlx.barrier_arrive({src_var}_empty[0], 1)"
                rctx.op_var[op.op_id] = name
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
        if len(lines.buf) == _n0:
            # No work landed in the default task (e.g. case6 with no acc_tmem
            # hand-off and the store living in a compute WG). Keep the
            # `with tlx.async_task("default"):` block valid Python.
            lines += "pass"


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


def _render_load_offsets_at(
    load_op: Op, g: ScheduleGraph, rctx: RenderCtx, lid: int, iv_expr: str
) -> list[str]:
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
        consumed_by_alloc = any(
            op.kind == "ttg.local_alloc"
            and op.operands
            and isinstance(op.operands[0], OpRef)
            and op.operands[0].op_id == n.op_ref
            for op in g.ops.values()
        )
        if not consumed_by_alloc:
            out.append(n)
    return out


# Async producers whose completion can be skewed across the stage boundary:
# their result is memory-resident by construction (tcgen05 MMA writes TMEM),
# so no register value needs to survive the skew.
_SKEW_ASYNC_PRODUCER_KINDS = ("ttng.tc_gen5_mma", "ttng.tc_gen5_mma_scaled")


def _skew_tmem_budget_ok(g: ScheduleGraph, loop: Loop, extra_by_op: dict[str, int]) -> bool:
    """Estimate total TMEM columns (512 budget, 32-bit cols) with the ring
    depth bumps applied. Conservative: ignores storage-alias reuse."""
    cols = 0

    def _alloc_cols(shape: list[int], bits: int, count: int) -> int:
        if not shape:
            return 0
        n = shape[-1]
        per_slot = -(-(n * max(bits, 1)) // 32)  # ceil(n*bits/32)
        return per_slot * max(count, 1)

    for op in g.ops.values():
        if op.kind != "ttng.tmem_alloc" or op.scope != "function":
            continue
        sd = _parse_tensor_shape(op.result_types[0]) if op.result_types else None
        if not sd:
            continue
        shape, dtype = sd
        bits = 16 if dtype in ("f16", "bf16") else 32
        cols += _alloc_cols(shape, bits, extra_by_op.get(op.op_id, 1))
    for b in loop.schedule.buffers:
        if b.kind != "tmem":
            continue
        cols += _alloc_cols(b.shape, b.element_bits, max(b.count, extra_by_op.get(b.def_op or "", 1)))
    return cols <= 512


def _compute_skew_plan(g: ScheduleGraph, loop: Loop, rctx: RenderCtx) -> set:
    """Intra-WG stage-skew plan (emitter software pipelining).

    A warp group whose schedule spans several stages AND contains an async
    producer (MMA) consumed at a LATER stage in the same WG was, until now,
    emitted linearly with an inline completion wait — serializing the tensor
    core against the WG's own compute. This computes, per WG, the classic
    modulo grouping: union-find over all distance-0 intra-WG edges EXCEPT the
    skewable ones (async producer → later-stage consumer), then a longest-path
    skew index over the component DAG formed by the skewable edges. Group g
    processes logical iteration (_it − g); the producer's destination buffer
    becomes a full/empty ring of depth (gap + 1).

    Register values never cross a group boundary by construction: every
    non-skewable d=0 edge (which includes all register-carried SSA deps) is
    unioned into one component. Returns the set of (loop_id, producer_node,
    consumer_node) pairs the SemIR derivation must skip (the ring replaces
    their signal-only handshake).
    """
    skip_pairs: set = set()
    nodes_by_id = {n.id: n for n in loop.schedule.nodes}
    for wg in loop.warp_groups:
        nodes = [n for n in loop.schedule.nodes if n.warp_group == wg.id]
        ids = {n.id for n in nodes}
        if len({n.schedule_stage for n in nodes}) <= 1:
            continue
        parent = {i: i for i in ids}

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(a: int, b: int) -> None:
            ra, rb = _find(a), _find(b)
            if ra != rb:
                parent[ra] = rb

        skewable: list[tuple[int, int]] = []
        for e in loop.schedule.edges:
            if e.distance != 0 or e.src not in ids or e.dst not in ids:
                continue
            s, d = nodes_by_id[e.src], nodes_by_id[e.dst]
            if (
                s.op_kind in _SKEW_ASYNC_PRODUCER_KINDS
                and d.schedule_stage > s.schedule_stage
            ):
                skewable.append((e.src, e.dst))
            else:
                _union(e.src, e.dst)
        # Fan-out constraint: all same-WG consumers of one async producer
        # must land in the same group (a shared completion signal cannot be
        # waited at two different skew offsets).
        cons_by_prod: dict[int, list[int]] = {}
        for src, dst in skewable:
            cons_by_prod.setdefault(src, []).append(dst)
        for cons in cons_by_prod.values():
            for other in cons[1:]:
                _union(cons[0], other)
        cross = [(s, d) for s, d in skewable if _find(s) != _find(d)]
        if not cross:
            continue
        # Longest-path skew index over components (skewable edges only —
        # every other d=0 edge is intra-component by construction).
        comp = {i: _find(i) for i in ids}
        skew: dict[int, int] = {c: 0 for c in set(comp.values())}
        for _ in range(len(cross) + 1):
            changed = False
            for s, d in cross:
                if skew[comp[d]] < skew[comp[s]] + 1:
                    skew[comp[d]] = skew[comp[s]] + 1
                    changed = True
            if not changed:
                break
        group_of = {i: skew[comp[i]] for i in ids}
        n_groups = max(group_of.values()) + 1

        # Ring info per producer destination buffer. Bail out (serial
        # fallback) if any producer's destination can't be resolved to a
        # plain alloc — never mis-lower silently.
        ring_by_op: dict[str, dict] = {}
        ok = True
        for s, d in cross:
            prod = nodes_by_id[s]
            op = g.ops.get(prod.op_ref) if prod.op_ref else None
            dest_id = None
            if op is not None and len(op.operands) >= 3 and isinstance(op.operands[2], OpRef):
                dest_id = op.operands[2].op_id
            if dest_id is None or prod.partition_count > 1:
                ok = False
                break
            gap = group_of[d] - group_of[s]
            entry = ring_by_op.setdefault(
                dest_id,
                {
                    "depth": 1,
                    "producer_node": s,
                    "consumer_nodes": [],
                    "loop_id": loop.loop_id,
                    "wg_id": wg.id,
                },
            )
            entry["depth"] = max(entry["depth"], gap + 1)
            if d not in entry["consumer_nodes"]:
                entry["consumer_nodes"].append(d)
        if not ok:
            print(
                f"# sched2tlx: skew plan for loop{loop.loop_id} wg{wg.id} "
                f"dropped (unresolvable ring destination) — serial fallback",
                file=sys.stderr,
            )
            continue
        depth_by_op = {oid: e["depth"] for oid, e in ring_by_op.items()}
        if not _skew_tmem_budget_ok(g, loop, depth_by_op):
            print(
                f"# sched2tlx: skew plan for loop{loop.loop_id} wg{wg.id} "
                f"dropped (TMEM budget exceeded with ring depths "
                f"{depth_by_op}) — serial fallback",
                file=sys.stderr,
            )
            continue

        for oid, entry in ring_by_op.items():
            d = entry["depth"]
            entry["slot"] = "0" if d == 1 else f"(_it % {d})"
            entry["phase"] = "(_it & 1)" if d == 1 else f"((_it // {d}) & 1)"
            # Split consumers by handshake style, in EMISSION order (the
            # _nodes_in_warp_group sort): SW consumers wait full at the first
            # site and recycle empty at the last; each MMA consumer does both
            # through its own emit branch (wait + mBarriers).
            emit_key = lambda nid: (  # noqa: E731
                nodes_by_id[nid].schedule_stage,
                nodes_by_id[nid].schedule_cluster,
                nid,
            )
            entry["consumer_nodes"].sort(key=emit_key)
            entry["mma_consumers"] = [
                c
                for c in entry["consumer_nodes"]
                if nodes_by_id[c].op_kind in _SKEW_ASYNC_PRODUCER_KINDS
            ]
            entry["sw_consumers"] = [
                c
                for c in entry["consumer_nodes"]
                if c not in entry["mma_consumers"]
            ]
            rctx.skew_ring_by_op[oid] = entry
            for c in entry["consumer_nodes"]:
                rctx.skew_ring_consumers.setdefault((loop.loop_id, c), []).append(entry)
                skip_pairs.add((loop.loop_id, entry["producer_node"], c))
        rctx.skew_plan[(loop.loop_id, wg.id)] = {
            "group_of": group_of,
            "n_groups": n_groups,
        }
    return skip_pairs


def _node_emission_may_wait(n: Node, loop: Loop, rctx: RenderCtx) -> bool:
    """True when emitting `n` can produce a barrier wait. Deferred producer
    triples must flush BEFORE such a node: sinking a channel hand-off below
    another wait can close a cross-WG cycle (observed on an FA-bwd partition:
    the m/D channel stores sank below a wait on an MMA whose inputs depend
    on those very channels — instant deadlock)."""
    if n.op_kind in (
        "ttng.tc_gen5_mma",
        "ttng.tc_gen5_mma_scaled",
        "tt.descriptor_load",
        "tt.descriptor_store",
        "tt.descriptor_reduce",
        "ttng.tmem_store",
    ):
        return True
    if rctx.sem_set is not None and rctx.sem_set.by_consumer.get(
        (loop.loop_id, n.id)
    ):
        return True
    if rctx.skew_ring_consumers.get((loop.loop_id, n.id)):
        return True
    return False


def _flush_deferred_producer_triples(rctx: RenderCtx, lines: _Lines) -> None:
    deferred = getattr(rctx, "_deferred_producer_triples", None)
    if deferred:
        for stmt in deferred:
            lines += stmt
        del deferred[:]


def _emit_skew_ring_consumer_wait(
    n: Node, loop: Loop, rctx: RenderCtx, lines: _Lines
) -> None:
    """SW consumer of an intra-WG skew ring: wait the producer's full slot
    before reading. Deduped so only the first same-stream consumer waits."""
    for entry in rctx.skew_ring_consumers.get((loop.loop_id, n.id), []):
        var = entry.get("var")
        if var is None or n.id in entry.get("mma_consumers", []):
            continue  # MMA consumers handshake inside the MMA emit branch
        bv = entry.get("bar_var") or _skw(var)
        w = (
            f"tlx.barrier_wait({_bar_full(bv)}[{entry['slot']}], "
            f"{entry['phase']})  # intra-WG skew ring (async result ready)"
        )
        seen = getattr(rctx, "_emitted_full_waits", None)
        if seen is not None:
            if w in seen:
                continue
            seen.add(w)
        lines += w


def _emit_skew_ring_consumer_arrive(
    n: Node, loop: Loop, rctx: RenderCtx, lines: _Lines
) -> None:
    """SW recycle of an intra-WG skew ring slot — emitted after the LAST
    same-stream consumer's read."""
    for entry in rctx.skew_ring_consumers.get((loop.loop_id, n.id), []):
        var = entry.get("var")
        if var is None or n.id in entry.get("mma_consumers", []):
            continue
        sw = entry.get("sw_consumers", [])
        if sw and sw[-1] == n.id:
            bv = entry.get("bar_var") or _skw(var)
            lines += (
                f"tlx.barrier_arrive({_bar_empty(bv)}[{entry['slot']}], 1)"
                f"  # intra-WG skew ring recycle"
            )


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

    # Per-operand-buffer MMA user lists (emission order) for the multi-consumer
    # intra-WG fallback: an operand read by >1 MMA in this WG must wait its
    # `_full` once (first user) and recycle `_empty` once (last user).
    opbuf_users: dict[int, list[int]] = {}
    for nd in nodes:
        if nd.op_kind != "ttng.tc_gen5_mma" or not nd.op_ref:
            continue
        mop = g.ops.get(nd.op_ref)
        if mop is None:
            continue
        for src_idx in (0, 1):
            if len(mop.operands) <= src_idx or not isinstance(
                mop.operands[src_idx], OpRef
            ):
                continue
            aid = mop.operands[src_idx].op_id
            mid = g.ops.get(aid)
            if (
                mid is not None
                and mid.kind == "ttg.memdesc_trans"
                and mid.operands
                and isinstance(mid.operands[0], OpRef)
            ):
                aid = mid.operands[0].op_id
            buf = next(
                (
                    b
                    for b in loop.schedule.buffers
                    if b.def_op == aid and b.kind != "barrier"
                ),
                None,
            )
            if buf is not None and nd.id not in opbuf_users.setdefault(buf.id, []):
                opbuf_users[buf.id].append(nd.id)
    rctx._opbuf_users = opbuf_users
    # Per-WG-body dedup of identical `_full` waits: a single-buffered operand
    # read by >1 MMA in this WG completes its full ONCE per load, so a repeated
    # `barrier_wait(<buf>_full[0], <same phase>)` would block forever. The
    # strings are identical for count=1 operands (same slot+phase), so exact-
    # match dedup is safe (count=2 buffers differ in slot/phase and are kept).
    rctx._emitted_full_waits = set()

    # Filter to ops that emit something. Allocs are SSA-only.
    emit_nodes = [
        n
        for n in nodes
        if n.op_kind not in ("ttg.local_alloc", "ttng.tmem_alloc", "scf.yield")
    ]
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
                if b.id in (None,):
                    continue
                if (
                    rctx.buffer_var.get(
                        (loop.loop_id, b.id), f"L{loop.loop_id}_{_buffer_var_name(b)}"
                    )
                    == ch.name
                ):
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
        b.count
        for b in loop.schedule.buffers
        if b.id in touched_buf_ids and b.kind in ("smem", "tmem") and b.count > 1
    ]
    rep_depth = max(depths) if depths else 1
    # Buffers whose count differs from rep_depth must NOT be subscripted
    # with the shared `buf`/`phase` — _ring_exprs consults this to emit
    # per-buffer moduli for them.
    rctx._wg_rep_depth = rep_depth
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
            if len(mma_op.operands) > src_idx and isinstance(
                mma_op.operands[src_idx], OpRef
            ):
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
            shape_dt = (
                _parse_tensor_shape(load_op.result_types[0])
                if load_op.result_types
                else None
            )
            shape = shape_dt[0] if shape_dt else [0]
            dtype = shape_dt[1] if shape_dt else "f16"
            n_bytes = 1
            for d in shape:
                n_bytes *= int(d)
            n_bytes *= _bytes_per_elem_bits(16 if dtype in ("f16", "bf16") else 32)
            lines += "# load Q tile (per-tile resident)"
            lines += f"tlx.barrier_expect_bytes({fl['alloc_var']}_full[0], {n_bytes})"
            lines += (
                f"tlx.async_descriptor_load({desc}, {fl['alloc_var']}[0], "
                f"[{offs_str}], {fl['alloc_var']}_full[0])"
            )
        if fl["alloc_op_id"] in mma_alloc_op_ids:
            lines += (
                f"tlx.barrier_wait({fl['alloc_var']}_full[0], 0)  # wait Q tile loaded"
            )
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
                lines += (
                    f"tlx.barrier_arrive({_bar_full(c.name)}[0], 1)  "
                    f"# loop-carry pre-arrive (producer cyc={prod_cyc} > "
                    f"consumer cyc={cons_cyc})"
                )

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
    kept = [
        (idx, init, yld, _iter_arg_python_name(loop.loop_id, idx, init))
        for (idx, init, yld) in iter_specs
        if idx in used_idxs
    ]
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
                lines += (
                    f"tlx.async_descriptor_load({desc}, {ring}[0], [{offs0}], {bar}[0])"
                )
            prefetch_loads.append(
                {
                    "node": ln,
                    "op": lop,
                    "ring": ring,
                    "bar": bar,
                    "nbytes": nbytes,
                    "desc": desc,
                    "NB": NB,
                    "ld_var": f"_pf{i}_val",
                }
            )
    prefetch_node_ids = {p["node"].op_ref for p in prefetch_loads}

    # A non-default async_task cannot capture a RankedTensorType from the
    # enclosing (function) scope. Re-materialize any function-scope register
    # tensors this WG consumes (e.g. case6 v2's W/B loads) with task-local
    # names; restore the global bindings after the body. No-op if none.
    _reg_loc_saved = _localize_captured_reg_tensors(g, [loop], emit_nodes, rctx, lines)

    if True:
        lo = _render_operand(loop.schedule.lower_bound, rctx)
        # Skip bridge ops (cross-WG TMEM tmem_alloc(value)): emitted as
        # local_store + barrier_arrive in the value producer's WG.
        bridge_op_ids = {
            c.bridge_op_id
            for c in rctx.channels
            if c.kind == "tmem" and c.bridge_op_id
        }
        skewp = (
            rctx.skew_plan.get((loop.loop_id, wg.id)) if not prefetch_loads else None
        )
        if skewp is not None:
            # ── Intra-WG stage-skew (software pipelining) ────────────────
            # Group g processes logical iteration (_it − g): async producers
            # issue for the newest iteration while later stages consume the
            # results of earlier ones. The physical loop is extended by G−1
            # iterations to drain; each group re-binds the induction var to
            # its logical value so every existing rendering (offsets, ring
            # slots, phases, use_acc) stays correct verbatim.
            G = skewp["n_groups"]
            group_of = skewp["group_of"]
            hi = _render_operand(loop.schedule.upper_bound, rctx)
            piv = f"_skew_{iv}"
            node_by_opref = {
                nn.op_ref: nn for nn in loop.schedule.nodes if nn.op_ref
            }

            def _yield_group(yld: OperandRef) -> int:
                if isinstance(yld, OpRef):
                    nd = node_by_opref.get(yld.op_id)
                    if nd is not None and nd.id in group_of:
                        return group_of[nd.id]
                return G - 1

            lines += (
                f"# Intra-WG stage-skew: {G} groups; group g runs logical "
                f"iteration (_it - g); {G - 1} drain iteration(s) appended."
            )
            hdr = (
                f"for {piv} in range({lo}, ({hi}) + {G - 1} * ({step_expr}), "
                f"{step_expr}):"
            )
            with lines.block(hdr):
                for gi in range(G):
                    conds = []
                    if gi > 0:
                        conds.append(f"{piv} >= ({lo}) + {gi} * ({step_expr})")
                    if gi < G - 1:
                        conds.append(f"{piv} < ({hi}) + {gi} * ({step_expr})")
                    with lines.block(f"if {' and '.join(conds)}:"):
                        if gi == 0:
                            lines += f"{iv} = {piv}"
                        else:
                            lines += f"{iv} = {piv} - {gi} * ({step_expr})"
                        lines += f"_it = ({iv} - {lo}) // {step_expr}"
                        lines += f"buf = _it % {rep_depth}"
                        lines += f"phase = (_it // {rep_depth}) & 1"
                        # Wait-dedup scope = one group block (each block has
                        # its own logical `_it` binding).
                        rctx._emitted_full_waits = set()
                        rctx._deferred_producer_triples = []
                        for n in emit_nodes:
                            if group_of.get(n.id, 0) != gi:
                                continue
                            if n.op_ref in bridge_op_ids:
                                continue
                            if _node_emission_may_wait(n, loop, rctx):
                                _flush_deferred_producer_triples(rctx, lines)
                            _emit_skew_ring_consumer_wait(n, loop, rctx, lines)
                            _emit_in_loop_node(n, g, loop, channels, rctx, lines)
                            _emit_skew_ring_consumer_arrive(n, loop, rctx, lines)
                        # Sunk producer-side channel triples (see
                        # _semir_emit_producer_block).
                        _flush_deferred_producer_triples(rctx, lines)
                        rctx._deferred_producer_triples = None
                        # Recurrence: reassign iter_args whose yield value is
                        # produced by THIS group (the var only exists here).
                        for idx, _init, yld, name in kept:
                            if _yield_group(yld) != gi:
                                continue
                            lines += f"{name} = {_render_operand(yld, rctx)}"
        elif True:
            with lines.block(f"for {iv} in {_loop_range_expr(loop, rctx)}:"):
                # Iteration count = (iv - lb) // step; ring-buffer index.
                lines += f"_it = ({iv} - {lo}) // {step_expr}"
                # `phase` MUST toggle per iteration even when ring depth=1 —
                # it's the parity that mbarriers use to detect the next phase.
                # For depth=N, phase advances every N iters.
                lines += f"buf = _it % {rep_depth}"
                lines += f"phase = (_it // {rep_depth}) & 1"
                # M2 (load prefetch): wait the current tile's load, bind its
                # SSA var to the local_load, and prefetch the next tile into
                # the alternate ring slot. The blocking descriptor_load node
                # is skipped below.
                for p in prefetch_loads:
                    nbv = p["NB"]
                    lines += f"_pf_slot = _it % {nbv}"
                    lines += f"_pf_phase = (_it // {nbv}) & 1"
                    lines += f"tlx.barrier_wait({p['bar']}[_pf_slot], _pf_phase)"
                    lines += f"{p['ld_var']} = tlx.local_load({p['ring']}[_pf_slot])"
                    rctx.op_var[p["op"].op_id] = p["ld_var"]
                    next_iv = f"({iv} + {step_expr})"
                    offs_n = ", ".join(
                        _render_load_offsets_at(p["op"], g, rctx, loop.loop_id, next_iv)
                    )
                    lines += f"_pf_nslot = (_it + 1) % {nbv}"
                    with lines.block(f"if {next_iv} < {hi_b}:"):
                        lines += f"tlx.barrier_expect_bytes({p['bar']}[_pf_nslot], {p['nbytes']})"
                        lines += (
                            f"tlx.async_descriptor_load({p['desc']}, {p['ring']}[_pf_nslot], "
                            f"[{offs_n}], {p['bar']}[_pf_nslot])"
                        )
                rctx._deferred_producer_triples = []
                for n in emit_nodes:
                    if n.op_ref in bridge_op_ids or n.op_ref in prefetch_node_ids:
                        continue
                    if _node_emission_may_wait(n, loop, rctx):
                        _flush_deferred_producer_triples(rctx, lines)
                    _emit_in_loop_node(n, g, loop, channels, rctx, lines)
                # Sunk producer-side channel triples (see
                # _semir_emit_producer_block): the handshakes go after the
                # body's independent compute so their EMPTY waits are covered
                # by useful work instead of stalling the stream.
                _flush_deferred_producer_triples(rctx, lines)
                rctx._deferred_producer_triples = None
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
        # Non-canonical epilogue accumulators this WG produces (FA-bwd: dK in
        # the dK-MMA's WG, dV in the dV-MMA's WG). Independent of has_mma (which
        # only flags the canonical acc_tmem producer) — commit so the default
        # partition's post-loop tmem_load sees the completed accumulator.
        for var, pwg in _epilogue_acc_wg(g, rctx).items():
            if pwg == wg.id:
                lines += f"tlx.tcgen05_commit({var}_full[0])"

    # Restore global (preamble) names shadowed by the per-WG localization above.
    for _oid, _prev in _reg_loc_saved.items():
        if _prev is None:
            rctx.op_var.pop(_oid, None)
        else:
            rctx.op_var[_oid] = _prev


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
                    (
                        nn.op_ref
                        for nn in loop.schedule.nodes
                        if nn.id == c.producer_node
                    ),
                    None,
                )
                if prod_op is None:
                    continue
                # If a TMEM bridge already handles this producer (the TMEM
                # tmem_alloc(value) absorbs the same SSA value), skip the SMEM
                # channel — TMEM is the natural staging for MMA-bound values.
                already_tmem = any(
                    tc.kind == "tmem"
                    and tc.bridge_op_id
                    and (
                        (
                            lambda bop: bop
                            and bop.operands
                            and isinstance(bop.operands[0], OpRef)
                            and bop.operands[0].op_id == prod_op
                        )(g.ops.get(tc.bridge_op_id))
                    )
                    for tc in rctx.channels
                )
                if already_tmem:
                    continue
                # Detect backward-edge / loop-carry channels (producer_cycle >
                # consumer_cycle within an iter): these are release SIGNALS
                # not data transfers — emit barrier-only, no load/store.
                prod_cyc = next(
                    (
                        nn.schedule_cycle
                        for nn in loop.schedule.nodes
                        if nn.id == c.producer_node
                    ),
                    None,
                )
                cons_cyc = next(
                    (
                        nn.schedule_cycle
                        for nn in loop.schedule.nodes
                        if nn.id == c.consumer_node
                    ),
                    None,
                )
                is_loop_carry = (
                    prod_cyc is not None
                    and cons_cyc is not None
                    and prod_cyc > cons_cyc
                )
                if c.kind == "named" or is_loop_carry:
                    # Signal-only: just wait for producer signal. No buffer
                    # load (the synthesized buffer for loop-carry channels is
                    # an artifact — it may even alias another buffer's storage,
                    # which would CORRUPT memory if we wrote/read it).
                    kind_note = (
                        "named-channel" if c.kind == "named" else "loop-carry release"
                    )
                    lines += (
                        f"tlx.barrier_wait({_bar_full(c.name)}[0], _it & 1)  "
                        f"# {kind_note} (n{c.producer_node}→n{c.consumer_node})"
                    )
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

    _load_buf = (
        _find_load_target_buffer(n, op, g, loop)
        if n.op_kind == "tt.descriptor_load"
        else None
    )
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
        buf_var = rctx.buffer_var.get(
            (loop.loop_id, buf.id), f"L{loop.loop_id}_{_buffer_var_name(buf)}"
        )
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
            data_slot = _ring_exprs(buf.count, rctx)[0] if buf is not None else "0"
            lines += f"# load → {buf_var}"
            if bar_arg is None:
                # No cross-WG semaphore for this TMA (producer + consumer in
                # the same WG). The intra-WG `<buf>_full` barrier was
                # allocated in the `extra` carve-out — use it: wait empty,
                # expect bytes, load with full as the mbarrier arg.
                idx, ph = _ring_exprs(buf.count, rctx)
                nbytes = (
                    (
                        buf.shape[0]
                        * buf.shape[1]
                        * _bytes_per_elem_bits(buf.element_bits)
                    )
                    if len(buf.shape) >= 2
                    else buf.size_bytes
                )
                full = _bar_full(buf_var)
                empty = _bar_empty(buf_var)
                lines += f"tlx.barrier_wait({empty}[{idx}], {ph} ^ 1)"
                lines += f"tlx.barrier_expect_bytes({full}[{idx}], {nbytes})"
                bar_arg = f"{full}[{idx}]"

            # Pass A.5: TMA load is NOT split — the descriptor's block shape
            # is the full (BM, BK), so the SMEM destination must match. The
            # MMA emit takes per-group `tlx.local_slice` views to feed the
            # N async_dot calls.
            lines += (
                f"tlx.async_descriptor_load({desc}, {buf_var}[{data_slot}], "
                f"[{offs_str}], {bar_arg})"
            )
            return
        # Legacy: per-buffer index AND phase. The loop's `buf` and `phase` are sized
        # for the loop's max-depth ring. A count=1 buffer needs:
        #   index = [0]  (single slot, not [buf] which can overrun)
        #   phase = `_it & 1`  (barrier flips every iter, not every `depth` iters)
        # Without this, the consumer-signaled empty barrier appears never to
        # release on subsequent iters → producer wait blocks → kernel hang
        # OR illegal barrier op if the state goes inconsistent.
        idx, ph = _ring_exprs(buf.count, rctx)
        nbytes = (
            (buf.shape[0] * buf.shape[1] * _bytes_per_elem_bits(buf.element_bits))
            if len(buf.shape) >= 2
            else buf.size_bytes
        )
        full = _bar_full(buf_var)
        empty = _bar_empty(buf_var)
        lines += f"# load → {buf_var}"
        lines += f"tlx.barrier_wait({empty}[{idx}], {ph} ^ 1)"
        lines += f"tlx.barrier_expect_bytes({full}[{idx}], {nbytes})"
        lines += (
            f"tlx.async_descriptor_load({desc}, {buf_var}[{idx}], "
            f"[{offs_str}], {full}[{idx}])"
        )
        return

    if n.op_kind == "ttng.tc_gen5_mma":
        # operands[0,1] = SMEM allocs (a, b); operands[2] = TMEM acc; etc.
        # B may go through ttg.memdesc_trans — walk through it.
        a_buf, a_loop = _resolve_alloc_to_buffer(op.operands[0], g, loop)
        b_buf, b_loop = _resolve_alloc_to_buffer(op.operands[1], g, loop)
        # Either operand may go through ttg.memdesc_trans: wgrad transposes A
        # (tl.trans(dout) → doutᵀ @ act), FA-style transposes B. Walk through
        # it on whichever side, and render that operand via tlx.local_trans.
        a_via_trans = False
        if a_buf is None and isinstance(op.operands[0], OpRef):
            mid_a = g.ops.get(op.operands[0].op_id)
            if mid_a and mid_a.kind == "ttg.memdesc_trans" and mid_a.operands:
                a_buf, a_loop = _resolve_alloc_to_buffer(mid_a.operands[0], g, loop)
                a_via_trans = True
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
            if buf is None:
                return "buf", "phase"
            return _ring_exprs(buf.count, rctx)

        a_idx, a_ph = _ring_idx_phase(a_buf, a_loop)
        b_idx, b_ph = _ring_idx_phase(b_buf, b_loop)

        # Resolve operand names; fallback to alloc_op_var for ops whose
        # alloc lives at function scope (no loop owns it — e.g. case3 nows
        # Q SMEM hoisted as `q_smem_*`).
        def _resolve_alloc_var(operand: OperandRef, buf, lp) -> str:
            if buf is not None and lp is not None:
                return rctx.buffer_var.get(
                    (lp.loop_id, buf.id), f"L{lp.loop_id}_{_buffer_var_name(buf)}"
                )
            if isinstance(operand, OpRef) and operand.op_id in rctx.alloc_op_var:
                return rctx.alloc_op_var[operand.op_id]
            return "<a?>"

        a_var = _resolve_alloc_var(op.operands[0], a_buf, a_loop)
        b_var = _resolve_alloc_var(op.operands[1], b_buf, b_loop)
        if a_via_trans and isinstance(op.operands[0], OpRef):
            mid_a = g.ops.get(op.operands[0].op_id)
            if mid_a and mid_a.kind == "ttg.memdesc_trans" and mid_a.operands:
                a_var = _resolve_alloc_var(mid_a.operands[0], a_buf, a_loop)
        if b_via_trans and isinstance(op.operands[1], OpRef):
            mid_op = g.ops.get(op.operands[1].op_id)
            if mid_op and mid_op.kind == "ttg.memdesc_trans" and mid_op.operands:
                b_var = _resolve_alloc_var(mid_op.operands[0], b_buf, b_loop)
        lines += "# MMA"
        # Determine destination TMEM acc and name, regardless of barrier path.
        b_expr_pre = (
            f"tlx.local_trans({b_var}[{b_idx}])" if b_via_trans else f"{b_var}[{b_idx}]"
        )
        a_expr_pre = (
            f"tlx.local_trans({a_var}[{a_idx}])" if a_via_trans else f"{a_var}[{a_idx}]"
        )
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
            opnd_waits, opnd_mbar = _semir_mma_operand_waits_and_mbarriers(
                op, g, loop, rctx, n
            )
            _seen_fw = getattr(rctx, "_emitted_full_waits", None)
            for w in opnd_waits:
                # Skip a duplicate `_full` wait already emitted in this WG body
                # (multi-consumer single-buffered operand — see _emit_warp_group).
                # _seen_fw is only set on the _emit_warp_group path; other paths
                # (e.g. case2 persistent inner loop) never multi-consume, so skip.
                if _seen_fw is not None and "_full[" in w and "barrier_wait" in w:
                    if w in _seen_fw:
                        continue
                    _seen_fw.add(w)
                lines += w
            # TMEM bridge operand: wait the bridge's full barrier; the empty
            # side goes into the MMA's mBarriers list for HW recycle.
            tmem_chan_for_recycle: list[str] = []
            _seen_fw_br = getattr(rctx, "_emitted_full_waits", None)
            for src_idx in (0, 1):
                if not (
                    len(op.operands) > src_idx
                    and isinstance(op.operands[src_idx], OpRef)
                ):
                    continue
                # Match the operand's alloc directly, or through one
                # memdesc_trans hop (a bridge operand read transposed, e.g.
                # FA-bwd dK reads trans(dsT)). Direct-match cases are unchanged.
                cand_ids = [op.operands[src_idx].op_id]
                mid = g.ops.get(cand_ids[0])
                if (
                    mid is not None
                    and mid.kind == "ttg.memdesc_trans"
                    and mid.operands
                    and isinstance(mid.operands[0], OpRef)
                ):
                    cand_ids.append(mid.operands[0].op_id)
                for c in rctx.channels:
                    if c.kind == "tmem" and c.alloc_op_id in cand_ids:
                        fw = (
                            f"tlx.barrier_wait({_bar_full(c.name)}[0], "
                            f"_it & 1)  # TMEM bridge operand"
                        )
                        # Multi-consumer dedup: a bridge read by >1 MMA in this
                        # WG (FA-bwd dsT → dK+dQ) has its `_full` completed once
                        # per store — wait it only on the first consumer. Each
                        # consumer still arrives `_empty` (arrive_count=#uses).
                        if _seen_fw_br is None or fw not in _seen_fw_br:
                            lines += fw
                            if _seen_fw_br is not None:
                                _seen_fw_br.add(fw)
                        tmem_chan_for_recycle.append(f"{_bar_empty(c.name)}[0]")
            for w in _semir_consumer_waits(loop.loop_id, n.id, rctx):
                if _seen_fw is not None and "_full[" in w and "barrier_wait" in w:
                    if w in _seen_fw:
                        continue
                    _seen_fw.add(w)
                lines += w
            # Intra-WG skew ring, consumer side: this MMA reads a ring slot
            # written by a skewed async producer in the same WG — wait its
            # full and recycle empty via mBarriers (HW, the read is async).
            ring_consumer_empties: list[str] = []
            for side, opnd_var in (("a", a_var), ("b", b_var)):
                rc = rctx.skew_ring.get(opnd_var)
                if rc is not None and n.id in rc["consumer_nodes"]:
                    rc_bv = rc.get("bar_var") or _skw(opnd_var)
                    w = (
                        f"tlx.barrier_wait({_bar_full(rc_bv)}[{rc['slot']}], "
                        f"{rc['phase']})  # intra-WG skew ring operand"
                    )
                    if _seen_fw is None or w not in _seen_fw:
                        lines += w
                        if _seen_fw is not None:
                            _seen_fw.add(w)
                    ring_consumer_empties.append(
                        f"{_bar_empty(rc_bv)}[{rc['slot']}]"
                    )
                    # Re-slot the operand expression onto the ring index.
                    if side == "a":
                        base = f"{opnd_var}[{rc['slot']}]"
                        a_expr_pre = (
                            f"tlx.local_trans({base})" if a_via_trans else base
                        )
                    else:
                        base = f"{opnd_var}[{rc['slot']}]"
                        b_expr_pre = (
                            f"tlx.local_trans({base})" if b_via_trans else base
                        )
            # Intra-WG skew ring, producer side: don't overwrite a slot the
            # consumer group (skew iterations behind) hasn't drained yet.
            ring_prod = rctx.skew_ring.get(dest_var)
            if ring_prod is not None and n.id == ring_prod["producer_node"]:
                acc_idx = ring_prod["slot"]
                rp_bv = ring_prod.get("bar_var") or _skw(dest_var)
                lines += (
                    f"tlx.barrier_wait({_bar_empty(rp_bv)}[{ring_prod['slot']}], "
                    f"{ring_prod['phase']} ^ 1)  # intra-WG skew ring slot free"
                )
            lines += f"use_acc = {_use_acc_expr(op, loop, rctx)}"
            mbar_list: list[str] = []
            mbar_list.extend(opnd_mbar)
            mbar_list.extend(tmem_chan_for_recycle)
            mbar_list.extend(ring_consumer_empties)
            mbar_list.extend(_semir_consumer_mbarriers(loop.loop_id, n.id, rctx))
            mbar_list.extend(_semir_producer_mbarriers(loop.loop_id, n.id, rctx))
            if ring_prod is not None and n.id == ring_prod["producer_node"]:
                rp_bv = ring_prod.get("bar_var") or _skw(dest_var)
                mbar_list.append(f"{_bar_full(rp_bv)}[{ring_prod['slot']}]")

            # Pass A.5: when the MMA is partitioned, fan out to N async_dot
            # calls. Each call takes a `tlx.local_slice` view of the shared
            # A SMEM (M-axis offset = gi*m_size) and writes its own TMEM
            # accumulator. The same mBarriers list (including SMEM _empty)
            # is attached to every call — the empty barrier's arrive_count
            # was set to N at allocation time.
            if n.partition_count > 1:
                dest_names = (
                    rctx.partition_alloc_names.get(dest_op_id) if dest_op_id else None
                )
                N = n.partition_count
                m_size = n.m_size
                if dest_names is None or len(dest_names) != N or m_size <= 0:
                    lines += (
                        f"# WARNING: partition_count={N} but TMEM per-group "
                        f"names missing; falling back to single MMA"
                    )
                    if mbar_list:
                        mb = ", ".join(mbar_list)
                        lines += (
                            f"tlx.async_dot({a_expr_pre}, {b_expr_pre}, "
                            f"{dest_var}[{acc_idx}], use_acc=use_acc, "
                            f"mBarriers=[{mb}])"
                        )
                    else:
                        lines += (
                            f"tlx.async_dot({a_expr_pre}, {b_expr_pre}, "
                            f"{dest_var}[{acc_idx}], use_acc=use_acc)"
                        )
                    return
                # A SMEM is shared (full BM tile); slice along M for each group.
                bk = (
                    a_buf.shape[1]
                    if (a_buf is not None and len(a_buf.shape) >= 2)
                    else 0
                )
                for gi in range(N):
                    dg = dest_names[gi]
                    a_view = (
                        f"tlx.local_slice({a_var}[{a_idx}], "
                        f"[{gi * m_size}, 0], [{m_size}, {bk}])"
                    )
                    if mbar_list:
                        mb = ", ".join(mbar_list)
                        lines += (
                            f"tlx.async_dot({a_view}, {b_expr_pre}, "
                            f"{dg}[{acc_idx}], use_acc=use_acc, "
                            f"mBarriers=[{mb}])"
                        )
                    else:
                        lines += (
                            f"tlx.async_dot({a_view}, {b_expr_pre}, "
                            f"{dg}[{acc_idx}], use_acc=use_acc)"
                        )
                return

            if mbar_list:
                mb = ", ".join(mbar_list)
                lines += (
                    f"tlx.async_dot({a_expr_pre}, {b_expr_pre}, "
                    f"{dest_var}[{acc_idx}], use_acc=use_acc, mBarriers=[{mb}])"
                )
            else:
                lines += (
                    f"tlx.async_dot({a_expr_pre}, {b_expr_pre}, "
                    f"{dest_var}[{acc_idx}], use_acc=use_acc)"
                )
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
            if not (
                len(op.operands) > src_idx and isinstance(op.operands[src_idx], OpRef)
            ):
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
        producer_chans = [
            c for c in rctx.channels if c.kind == "tmem" and c.alloc_op_id == dest_op_id
        ]
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
            lines += (
                f"tlx.async_dot({a_expr_pre}, {b_expr}, {dest_var}[{acc_idx}], "
                f"use_acc=use_acc, mBarriers=[{mb}])"
            )
        else:
            lines += f"tlx.async_dot({a_expr_pre}, {b_expr}, {dest_var}[{acc_idx}], use_acc=use_acc)"
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
                lines += (
                    f"tlx.barrier_arrive({_bar_full(c.name)}[0], 1)  "
                    f"# named-channel signal (n{c.producer_node}→n{c.consumer_node})"
                )
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
            lines += (
                f"tlx.async_descriptor_store({desc}, "
                f"{sc['buf_var']}[{sc['slot']}], [{offs_str}])"
            )
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
            buf_var = rctx.buffer_var.get(
                (loop.loop_id, buf.id), f"L{loop.loop_id}_{_buffer_var_name(buf)}"
            )
            slot = _ring_exprs(buf.count, rctx)[0]
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
            lines += (
                f"tlx.async_descriptor_store({desc}, {buf_var}[{slot}], [{offs_str}])"
            )
            rctx._needs_store_drain = True
            return
        lines += f"tlx.local_store({buf_var}[{slot}], {value_expr})"
        lines += "tlx.fence_async_shared()"
        lines += f"tlx.async_descriptor_store({desc}, {buf_var}[{slot}], [{offs_str}])"
        lines += "tlx.async_descriptor_store_wait(0)"
        return
    if n.op_kind == "tt.descriptor_reduce":
        # Atomic-add store (e.g. FA-bwd dQ: many K-tiles accumulate into the same
        # Q rows). Like the in-loop store but with store_reduce="add".
        # operands = [desc, value, off0, off1, ...].
        # Re-materialize the descriptor here (task-local) rather than capturing
        # it from the preamble: an in-task async_descriptor_store infers an
        # nvmma_shared tensordesc type that mismatches the bare captured type.
        if isinstance(op.operands[0], OpRef):
            desc_op = g.ops.get(op.operands[0].op_id)
            if desc_op is not None and desc_op.op_id not in rctx.op_var:
                dname = _auto_name(desc_op, rctx.fresh_idx())
                rctx.op_var[desc_op.op_id] = dname
                lines += f"{dname} = {_render_op_expr(desc_op, rctx)}"
        desc = _render_operand(op.operands[0], rctx)
        value_expr = _render_operand(op.operands[1], rctx)
        offs_str = ", ".join(_render_operand(o, rctx) for o in op.operands[2:])
        bid = op.attributes.get("buffer.id")
        buf = next((b for b in loop.schedule.buffers if b.id == bid), None)
        if buf is not None:
            buf_var = rctx.buffer_var.get(
                (loop.loop_id, buf.id), f"L{loop.loop_id}_{_buffer_var_name(buf)}"
            )
            slot = _ring_exprs(buf.count, rctx)[0]
        else:
            buf_var, slot = "dq_smem", "0"
        lines += f"tlx.local_store({buf_var}[{slot}], {value_expr})"
        lines += "tlx.fence_async_shared()"
        lines += (
            f"tlx.async_descriptor_store({desc}, {buf_var}[{slot}], "
            f'[{offs_str}], store_reduce="add")'
        )
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
                    tc.kind == "tmem"
                    and tc.bridge_op_id
                    and (
                        (
                            lambda bop: bop
                            and bop.operands
                            and isinstance(bop.operands[0], OpRef)
                            and bop.operands[0].op_id == op.op_id
                        )(g.ops.get(tc.bridge_op_id))
                    )
                    for tc in rctx.channels
                )
                if already_tmem:
                    continue
                # Loop-carry detection: same as consumer side. Signal-only.
                prod_cyc = next(
                    (
                        nn.schedule_cycle
                        for nn in loop.schedule.nodes
                        if nn.id == c.producer_node
                    ),
                    None,
                )
                cons_cyc = next(
                    (
                        nn.schedule_cycle
                        for nn in loop.schedule.nodes
                        if nn.id == c.consumer_node
                    ),
                    None,
                )
                is_loop_carry = (
                    prod_cyc is not None
                    and cons_cyc is not None
                    and prod_cyc > cons_cyc
                )
                if c.kind == "named" or is_loop_carry:
                    kind_note = (
                        "named-channel" if c.kind == "named" else "loop-carry release"
                    )
                    lines += (
                        f"tlx.barrier_arrive({_bar_full(c.name)}[0], 1)  "
                        f"# {kind_note} signal (n{c.producer_node}→n{c.consumer_node})"
                    )
                    continue
                # Layer 1: wait for slot free before overwriting.
                lines += f"tlx.barrier_wait({_bar_empty(c.name)}[0], (_it & 1) ^ 1)"
                # Layer 2: if this buffer reuses another's bytes (merge
                # group alias), the preceding alias members' consumers
                # must also have finished before we overwrite.
                for pred_name in _alias_predecessors(c.name, rctx, g):
                    lines += (
                        f"tlx.barrier_wait({_bar_empty(pred_name)}[0], "
                        f"(_it & 1) ^ 1)  # alias predecessor"
                    )
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
            if (
                bridge_op
                and bridge_op.operands
                and isinstance(bridge_op.operands[0], OpRef)
                and bridge_op.operands[0].op_id == op.op_id
            ):
                # Same per-iter parity as the schedule-channel path.
                lines += f"tlx.barrier_wait({_bar_empty(c.name)}[0], (_it & 1) ^ 1)"
                lines += f"tlx.local_store({c.name}[0], {name})"
                lines += f"tlx.barrier_arrive({_bar_full(c.name)}[0], 1)"
        return
    if n.op_kind in ("arith.constant", "scf.yield"):
        return
    lines += f"# (unhandled in-loop op: {n.op_kind})"


def _find_load_target_buffer(
    node: Node, op: Op, g: ScheduleGraph, loop: Loop
) -> Buffer | None:
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


def _resolve_alloc_to_buffer(
    operand: OperandRef, g: ScheduleGraph, loop: Loop
) -> tuple[Buffer | None, Loop | None]:
    """Find the (buffer, owning_loop) for an operand that is an alloc OpRef.
    Searches the given loop first, then any other loop in the graph (so an
    inner-loop MMA can find a buffer that's owned by the outer-loop schedule,
    e.g., case3's per-tile-resident Q SMEM)."""
    if not isinstance(operand, OpRef):
        return None, None
    # A data operand (MMA/trans input) resolves to the SMEM/TMEM data buffer,
    # never its paired barrier — a double-buffered buffer and its barrier share
    # the same def_op, so skip barrier-kind entries to avoid mis-resolving to
    # the barrier (which would emit `L<lp>_bar_<id>` as an MMA operand).
    for b in loop.schedule.buffers:
        if b.def_op == operand.op_id and b.kind != "barrier":
            return b, loop
    for L in g.loops:
        if L is loop:
            continue
        for b in L.schedule.buffers:
            if b.def_op == operand.op_id and b.kind != "barrier":
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


def _unified_warp_groups(
    graph: ScheduleGraph, outer: Loop | None, inner: Loop | None
) -> list[UnifiedWG]:
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
        epi_ops = [
            o
            for o in _ops_after_loop(graph, outer)
            if o.kind not in _SKIP_FUNCTION_SCOPE
        ]
        has_epilogue = len(epi_ops) > 0
        default_wg_id: int | None = None
        if not has_epilogue:
            candidates = [
                w
                for w in outer.warp_groups
                if "TMA" not in w.pipelines and "TC" not in w.pipelines
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
                )
            )
        for wg in outer.warp_groups:
            roles = "+".join(wg.pipelines)
            primary = (
                "TC"
                if "TC" in wg.pipelines
                else ("TMA" if "TMA" in wg.pipelines else roles)
            )
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
                )
            )
        return _rescale_task_regs(out)

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
    # Lever #2: when the epilogue's register input comes from a single inner WG,
    # CO-LOCATE the epilogue into that WG (it becomes the default task owning both
    # its inner loop and the outer epilogue) instead of a separate default task
    # fed by a cross-WG SMEM channel. Matches the hand-written PROMO=default and
    # drops the epi_* staging buffer + round-trip.
    colo_wg = _epilogue_colocation_wg(graph)
    if colo_wg is None:
        out.append(
            UnifiedWG(
                name="default",
                role="default",
                outer_wg=epi_outer_wg,
                inner_wg=None,
                num_warps=4,
                is_default=True,
            )
        )

    # One UWG per inner warp group. Trust the schedule pass's num_warps
    # decision (Layer B); see the case1 branch above for the rationale.
    for wg in inner.warp_groups:
        roles = "+".join(wg.pipelines)
        primary = (
            "TC"
            if "TC" in wg.pipelines
            else ("TMA" if "TMA" in wg.pipelines else roles)
        )
        num_warps = wg.num_warps
        num_regs = 152 if num_warps >= 4 else 24
        # Lever #2: the co-located WG becomes the default task and also owns the
        # outer epilogue (outer_wg=epi_outer_wg); as the default task it gets the
        # leftover (largest) register budget — right for the register-heavy
        # promotion + store.
        is_colo = wg.id == colo_wg
        out.append(
            UnifiedWG(
                name="default" if is_colo else f"inner_wg{wg.id}_{primary}",
                role="default" if is_colo else primary,
                outer_wg=epi_outer_wg if is_colo else None,
                inner_wg=wg.id,
                num_warps=num_warps,
                num_regs=(None if is_colo else num_regs),
                is_default=is_colo,
            )
        )
    # Multi-WG OUTER bodies: any outer warp group beyond the default's gets
    # its own task (e.g. an epilogue store split from the drain compute).
    # Without these the ops of such a group had no owning task — the silent
    # drop the task-coverage check now refuses.
    for wg in outer.warp_groups:
        if wg.id == epi_outer_wg:
            continue
        owns_any = any(
            n.warp_group == wg.id
            and n.child_pipeline_id is None
            and n.op_kind not in _COVERAGE_EXEMPT_KINDS
            for n in outer.schedule.nodes
        )
        if not owns_any:
            continue
        roles = "+".join(wg.pipelines)
        primary = (
            "TC"
            if "TC" in wg.pipelines
            else ("TMA" if "TMA" in wg.pipelines else roles)
        )
        out.append(
            UnifiedWG(
                name=f"outer_wg{wg.id}_{primary}",
                role=primary,
                outer_wg=wg.id,
                inner_wg=None,
                num_warps=wg.num_warps,
                num_regs=152 if wg.num_warps >= 4 else 24,
            )
        )
    return _rescale_task_regs(out)


# Register-file budget for explicit task requests (Blackwell: 64K 32-bit
# registers per SM; setmaxnreg is per-thread, a multiple of 8). The default
# task can be trimmed to ~80 regs/thread by AllocateWarpGroups before compute
# WGs start losing registers — reserve that floor for it.
_SM_REG_FILE = 65536
_DEFAULT_TASK_RESERVE = 4 * 32 * 80


def _rescale_task_regs(uwgs: list[UnifiedWG]) -> list[UnifiedWG]:
    """Scale down over-budget register requests so the kernel still compiles.

    The per-WG request rule (152 regs/thread for a 4-warp task) over-asks when
    a partition carries several 4-warp compute groups: AllocateWarpGroups then
    trims someone below what its code needs and ptxas aborts (C7602 —
    observed on an 8-WG FA-bwd partition). Fit the requests to the
    register file instead: when the total fits, every task keeps its request
    (committed kernels are byte-identical); otherwise the >24-reg tasks share
    the remainder equally, floored to a multiple of 8.
    """
    explicit = [u for u in uwgs if not u.is_default and u.num_regs]
    total = sum(u.num_warps * 32 * u.num_regs for u in explicit)
    if total + _DEFAULT_TASK_RESERVE <= _SM_REG_FILE:
        return uwgs
    small = [u for u in explicit if u.num_regs <= 24]
    big = [u for u in explicit if u.num_regs > 24]
    threads = sum(u.num_warps * 32 for u in big)
    if not big or threads <= 0:
        return uwgs
    budget = (
        _SM_REG_FILE
        - _DEFAULT_TASK_RESERVE
        - sum(u.num_warps * 32 * u.num_regs for u in small)
    )
    per = max(24, (budget // threads) // 8 * 8)
    print(
        f"# sched2tlx: register request over budget "
        f"({total + _DEFAULT_TASK_RESERVE} > {_SM_REG_FILE}); scaling "
        f"{len(big)} compute task(s) from num_regs="
        f"{sorted({u.num_regs for u in big})} to {per}",
        file=sys.stderr,
    )
    for u in big:
        u.num_regs = per
    return uwgs


# Node kinds that never emit task-body code (hoisted allocs / SSA glue), so a
# task-less warp group containing ONLY these is not a lowering hole.
_COVERAGE_EXEMPT_KINDS = {
    "ttg.local_alloc",
    "ttng.tmem_alloc",
    "scf.yield",
    "arith.constant",
}


def _check_task_coverage(
    g: ScheduleGraph, outer: Loop, inner: Loop | None, uwgs: list["UnifiedWG"]
) -> None:
    """Refuse to emit a kernel with orphaned scheduled ops.

    Every emittable node the schedule assigned to a warp group must be owned
    by exactly the task that will render it. The historic failure mode this
    guards against is SILENT: a warp group with no owning async task keeps
    its barrier declarations (emitted from the semaphore set) while its body
    ops simply vanish — the kernel compiles, launches, and produces garbage
    (observed: an outer-loop epilogue split into its own WG dropped its
    descriptor_store and returned NaN). A named generation-time error turns
    that into a visible emitter-capability gap instead.
    """
    covered_inner = {u.inner_wg for u in uwgs if u.inner_wg is not None}
    covered_outer = {u.outer_wg for u in uwgs if u.outer_wg is not None}

    def _orphans(loop: Loop, covered: set) -> list[Node]:
        out = []
        for n in loop.schedule.nodes:
            if n.child_pipeline_id is not None:
                continue  # super-node: replayed by every inner-WG task
            if n.warp_group is None or n.warp_group < 0:
                continue  # infra/replicated: attached at emission sites
            if n.op_kind in _COVERAGE_EXEMPT_KINDS:
                continue
            if n.warp_group not in covered:
                out.append(n)
        return out

    problems: list[tuple[int, Node]] = []
    if inner is not None:
        # Persistent kernel: outer-loop nodes are only reachable through a
        # task with a matching outer_wg (the per-inner-WG tasks replay the
        # outer loop but own none of its non-super-node ops).
        problems += [(outer.loop_id, n) for n in _orphans(outer, covered_outer)]
        problems += [(inner.loop_id, n) for n in _orphans(inner, covered_inner)]
    else:
        problems += [(outer.loop_id, n) for n in _orphans(outer, covered_inner)]
    if problems:
        detail = ", ".join(
            f"loop{lid} wg{n.warp_group} N{n.id} {n.op_kind}" for lid, n in problems
        )
        raise RuntimeError(
            "sched2tlx task-coverage error: scheduled ops with no owning "
            f"async task ({detail}). This partition shape is not lowerable "
            "by the current emitter (e.g. a multi-WG OUTER loop body); "
            "emitting it would silently drop these ops."
        )


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


def _depends_on_iv_or_iter_arg(
    g: ScheduleGraph, op_id: str, visited: set[str] | None = None
) -> bool:
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


def _result_is_register_tensor(op: Op) -> bool:
    """True if op's first result is a register tensor (RankedTensorType).
    Such values are ILLEGAL to capture across a WarpSpecializeOp boundary
    ("WarpSpecializeOp should not capture RankedTensorType"); scalars,
    descriptors (!tt.tensordesc) and memdescs (!ttg.memdesc) are fine."""
    return bool(op.result_types) and op.result_types[0].lstrip().startswith("tensor<")


def _localize_captured_reg_tensors(
    g: ScheduleGraph,
    loops: list[Loop],
    target_nodes: list[Node],
    rctx: RenderCtx,
    lines: _Lines,
    descend_iv: bool = False,
) -> dict[str, str | None]:
    """Re-materialize, with task-local names, any function-scope register-tensor
    op a non-default warp group consumes that is currently bound to a global
    (preamble) variable. A non-default `tlx.async_task` may not capture a
    RankedTensorType from the enclosing scope (e.g. case6 v2: the W/B loads
    `ext_7`/`ext_8` are used inside a non-default compute WG), so each such task
    gets its own copy of the (cheap, loop-invariant) value instead of capturing
    it. Returns {op_id: prev_var} so the caller can restore the global bindings
    after this WG's body. No-op when the WG captures nothing (the common case)."""
    order: list[Op] = []
    seen: set[str] = set()

    # Values produced by scheduled (non-infra) nodes are bound to task-local
    # names or channel payloads during node emission — the renderer uses the
    # binding and never inline-expands past them. Only un-scheduled
    # (inline-rendered) expression trees can smuggle a function-scope register
    # tensor into this task's rendered code, so the walk mirrors rendering:
    # stop at scheduled nodes, recurse through everything else — including
    # IV/iter-arg-dependent inline ops, whose subtrees can still reference
    # IV-INVARIANT preamble tensors (case4 v2:
    # `tl.load(D + (tile_id + range_12))` capturing the preamble `tl.arange`).
    bound_ids = {
        n.op_ref
        for lp in loops
        for n in lp.schedule.nodes
        if n.op_ref and n.warp_group >= 0
    }

    def walk(op_id: str) -> None:
        if op_id in seen:
            return
        seen.add(op_id)
        if op_id in bound_ids:
            return
        op = g.ops.get(op_id)
        if op is None:
            return
        # By default prune at IV / iter-arg-dependent ops (they're rematerialized
        # per-tile by the infra path). With descend_iv=True (persistent per-UWG
        # localization) keep descending so we reach IV-INDEPENDENT captured
        # register tensors nested inside them — e.g. a tt.make_range buried in a
        # per-tile scale-offset addptr (blockwise scaled_mm). Only IV-independent
        # leaves are localized either way.
        if not descend_iv and _depends_on_iv_or_iter_arg(g, op_id):
            return
        for o in op.operands:
            if isinstance(o, OpRef):
                walk(o.op_id)
        # An IV/iter-arg-dependent value can't be hoisted as a task-local
        # copy (it isn't loop-invariant) — but its operands, walked above,
        # still can.
        if _depends_on_iv_or_iter_arg(g, op_id):
            return
        # Candidate: a register-tensor value already named at function scope
        # (preamble). Re-emit it here so the task doesn't capture it.
        if (
            not _depends_on_iv_or_iter_arg(g, op_id)
            and op.op_id in rctx.op_var
            and _result_is_register_tensor(op)
        ):
            order.append(op)

    for n in target_nodes:
        if not n.op_ref:
            continue
        top = g.ops.get(n.op_ref)
        if top is None:
            continue
        for o in top.operands:
            if isinstance(o, OpRef):
                walk(o.op_id)

    if not order:
        return {}
    saved: dict[str, str | None] = {op.op_id: rctx.op_var.get(op.op_id) for op in order}
    lines += "# Re-materialize function-scope register tensors locally (a "
    lines += "# non-default warp group cannot capture RankedTensorType)."
    vidx = 600
    for op in order:  # deps-before-dependents (toposort from walk)
        name = f"_wgloc{vidx}"
        vidx += 1
        rhs = _render_op_expr(op, rctx)  # operands already rebound to locals
        rctx.op_var[op.op_id] = name
        lines += f"{name} = {rhs}"
    return saved


def _localize_rendered_captures(
    g: ScheduleGraph, rctx: RenderCtx, lines: _Lines, body_start: int
) -> None:
    """Safety net behind _localize_captured_reg_tensors. The pre-walk predicts
    which function-scope register tensors a WG body will reference by walking
    the graph, but a few render paths expand expressions the walk cannot see
    (e.g. the prologue/skew re-materialization of a channel producer's address
    chain — case4 v2's `tl.load(D + (tile_id + range_12))` inlines past a
    channel-bound node). This pass checks the property the TTIR verifier
    actually enforces ("WarpSpecializeOp should not capture RankedTensorType"):
    scan the RENDERED body for function-scope register-tensor names, and
    re-materialize + substitute any found at the top of the task body. No-op
    when the pre-walk got everything — which keeps committed-kernel parity
    byte-exact — and correct by construction when it didn't."""
    # Every function-scope register tensor currently bound to a preamble name.
    fn_names: dict[str, Op] = {}
    for op_id, name in rctx.op_var.items():
        if not name:
            continue
        op = g.ops.get(op_id)
        if op is not None and op.scope == "function" and _result_is_register_tensor(op):
            fn_names[name] = op
    if not fn_names:
        return

    def names_in(text: str) -> set[str]:
        return {n for n in fn_names if re.search(rf"\b{re.escape(n)}\b", text)}

    body_text = "\n".join(lines.buf[body_start:])
    needed = names_in(body_text)
    if not needed:
        return
    # A re-materialized expression can itself reference other function-scope
    # tensors (case6's arange → W/B-load chain) — chase to a fixpoint.
    exprs: dict[str, str] = {}
    work = list(needed)
    while work:
        name = work.pop()
        if name in exprs:
            continue
        exprs[name] = _render_op_expr(fn_names[name], rctx)
        work.extend(n for n in names_in(exprs[name]) if n not in exprs)

    # Topological order by textual dependency (SSA — no cycles).
    ordered: list[str] = []
    emitted: set[str] = set()

    def visit(name: str) -> None:
        if name in emitted:
            return
        emitted.add(name)
        for dep in names_in(exprs[name]):
            if dep != name:
                visit(dep)
        ordered.append(name)

    for name in sorted(exprs):
        visit(name)

    rebound = {name: f"_wgcap{700 + i}" for i, name in enumerate(ordered)}
    pattern = re.compile("|".join(rf"\b{re.escape(n)}\b" for n in rebound))

    def sub(text: str) -> str:
        return pattern.sub(lambda m: rebound[m.group(0)], text)

    indent = "    " * lines.indent
    remat = [
        f"{indent}# Re-materialize captured function-scope register tensors",
        f"{indent}# (render-level safety net; see _localize_rendered_captures).",
    ]
    remat += [f"{indent}{rebound[name]} = {sub(exprs[name])}" for name in ordered]
    for i in range(body_start, len(lines.buf)):
        lines.buf[i] = sub(lines.buf[i])
    lines.buf[body_start:body_start] = remat


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


def _loop_iter_args(
    g: ScheduleGraph, loop: Loop
) -> list[tuple[int, OperandRef, OperandRef]]:
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
    wg_op_ids = {
        n.op_ref
        for n in inner.schedule.nodes
        if n.warp_group == uwg.inner_wg and n.op_ref
    }
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


def _has_smem_ring_buffer_inner(
    inner: Loop, uwg: UnifiedWG, channels: list[Any] | None = None
) -> bool:
    """Does this UWG touch any SMEM buffer in the inner loop?

    Returns True for any partition that reads or writes a SMEM buf, even
    when count=1 — the emitter still uses `smem_accum` as a per-iter parity
    counter for barrier phase calculation (`_it & 1`), independent of the
    ring depth.
    """
    if inner is None or uwg.inner_wg is None:
        return False
    # A UWG that consumes (or produces) a cross-WG SMEM channel ring-indexes it
    # with `smem_accum` too — e.g. the case7 bias-reduce WG reads the shared
    # dout SMEM via local_load. Detect this from the raw cross_wg_barriers
    # (authoritative): the deduped `channels` list can merge a shared buffer's
    # second consumer away, but SemIR still emits its ring-indexed local_load.
    for cb in getattr(inner.schedule, "cross_wg_barriers", []):
        if uwg.inner_wg not in (cb.consumer_wg, cb.producer_wg):
            continue
        buf = next(
            (b for b in inner.schedule.buffers if b.id == cb.paired_buffer_id), None
        )
        if buf is not None and buf.kind == "smem":
            return True
    nodes = _inner_nodes_for_uwg(inner, uwg)
    for n in nodes:
        for bid in n.consumes_buffers + (
            [n.produces_buffer] if n.produces_buffer is not None else []
        ):
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
        for bid in n.consumes_buffers + (
            [n.produces_buffer] if n.produces_buffer is not None else []
        ):
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
    if persistent and uwg.role == "TC" and rctx.has_acc_tmem_handoff:
        lines += "tlx.barrier_wait(acc_tmem_empty[tmem_buf], tmem_phase ^ 1)"

    # Loop-carry pre-arrive: under SemIR, looked up from is_released
    # semaphores keyed by (loop, wg). Legacy path uses cycle comparison.
    # For a PERSISTENT kernel the inner-loop carry is continuous across tiles
    # (smem_accum/_it persists), so the prime is hoisted to ONCE before the outer
    # loop by the caller — re-priming here per tile would add an unmatched arrive
    # and drift the barrier phase → cross-tile deadlock (case9 sem4).
    if _use_semaphore_ir() and not persistent:
        for line in _semir_pre_arrives_for_wg(inner.loop_id, uwg.inner_wg, rctx):
            lines += line
    elif not _use_semaphore_ir():
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
                lines += (
                    f"tlx.barrier_arrive({_bar_full(c.name)}[0], 1)  "
                    f"# loop-carry pre-arrive (producer cyc={prod_cyc} > "
                    f"consumer cyc={cons_cyc})"
                )

    # Per-UWG iter_arg trim (mirror of WSSpecialize::SpecializeForOp): only
    # keep iter_args this UWG's inner ops actually consume. Init each kept
    # one before the loop; reassign at end of body from the yield expression.
    iter_specs = _loop_iter_args(g, inner)
    used_idxs = _iter_args_used_by_inner_uwg(g, inner, uwg)
    kept = [
        (idx, init, yld, _iter_arg_python_name(inner.loop_id, idx, init))
        for (idx, init, yld) in iter_specs
        if idx in used_idxs
    ]
    saved_iter_arg_var = dict(rctx.iter_arg_var)
    for idx, init, _yld, name in kept:
        rctx.iter_arg_var[(inner.loop_id, idx)] = name
        lines += f"{name} = {_render_operand(init, rctx)}"

    lines += (
        f"# Inner K-loop (loop {inner.loop_id}, II={inner.schedule.II}). "
        f"SMEM ring depth={depth}; smem_accum persists across outer tiles."
    )
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
        bridge_op_ids = {
            c.bridge_op_id for c in rctx.channels if c.kind == "tmem" and c.bridge_op_id
        }
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
    if persistent and uwg.role == "TC" and rctx.has_acc_tmem_handoff:
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
    # The chain is tmem_load → (truncf | convert_layout)* → descriptor_store.
    # If the store is present in this WG's nodes, end the chain just past it.
    # If it isn't (the scheduler placed the store in a different WG), bound
    # chain_end at the last recognized cast/convert instead of running to the
    # end of the WG — otherwise unrelated trailing nodes get swallowed into the
    # chain and silently dropped. The caller splices the cross-WG store back in.
    cast_kinds = ("ttng.tmem_load", "arith.truncf", "ttg.convert_layout")
    store_idx = -1
    last_cast_idx = chain_start
    for j in range(chain_start, len(outer_nodes)):
        kind = outer_nodes[j].op_kind
        if kind == "tt.descriptor_store":
            store_idx = j
            break
        if kind in cast_kinds:
            last_cast_idx = j
    chain_end = (store_idx + 1) if store_idx >= 0 else (last_cast_idx + 1)
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
    split into N (m_size, BN) groups. Each group is handled independently:
    load its (m_size, BN) accumulator, cast to the store dtype, stage it
    into a single reused (m_size, BN) c_smem[0], and TMA-store it at the
    group's M row offset (base_row + gi*m_size). Reusing one c_smem buffer
    is safe because async_descriptor_store_wait drains the prior group's
    store before the next group overwrites it.

        tlx.barrier_wait(acc_tmem_full[tmem_buf], tmem_phase)
        for gi in range(N):                       # unrolled below
            acc_gi   = tlx.local_load(acc_tmem_g{gi}[tmem_buf])
            trunc_gi = acc_gi.to(store_dtype)
            tlx.local_store(c_smem[0], trunc_gi)
            tlx.fence_async_shared()
            tlx.async_descriptor_store(
                c_desc, c_smem[0], [row + gi*m_size, col])
            tlx.async_descriptor_store_wait(0)
        tlx.barrier_arrive(acc_tmem_empty[tmem_buf], 1)

    c_desc stays (BM, BN) so the launcher's `c_desc.block_shape = (BM, BN)`
    contract is unchanged; c_smem is shrunk to (m_size, BN) at its alloc
    site since only one group's tile is staged at a time.
    """
    lines += (
        f"# Pass A.5 partitioned epilogue (N={N}, m_size={m_size}, per-group c_smem)"
    )
    lines += "tlx.barrier_wait(acc_tmem_full[tmem_buf], tmem_phase)"
    # Each group loads its (m_size, BN) accumulator, casts, writes a single
    # reused (m_size, BN) c_smem buffer, and TMA-stores it at the group's M row
    # offset (base + gi*m_size). Reusing one c_smem is safe: the
    # async_descriptor_store_wait drains the prior group before the next writes.
    for gi in range(N):
        for n in chain_nodes:
            op = g.ops.get(n.op_ref) if n.op_ref else None
            if op is None or op.kind in _SKIP_FUNCTION_SCOPE:
                continue
            if op.kind == "ttng.tmem_load":
                dest_op_id = (
                    op.operands[0].op_id
                    if op.operands and isinstance(op.operands[0], OpRef)
                    else None
                )
                pnames = (
                    rctx.partition_alloc_names.get(dest_op_id) if dest_op_id else None
                )
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
                store_desc = _render_operand(op.operands[0], rctx)
                offs = [_render_operand(o, rctx) for o in op.operands[2:]]
                value_expr = _render_operand(op.operands[1], rctx)
                # Group gi's tile goes to M row offset (base_row + gi*m_size).
                row = offs[0] if gi == 0 else f"({offs[0]} + {gi * m_size})"
                col = offs[1] if len(offs) > 1 else "0"
                lines += f"tlx.local_store(c_smem[0], {value_expr})"
                lines += "tlx.fence_async_shared()"
                lines += (
                    f"tlx.async_descriptor_store({store_desc}, c_smem[0], "
                    f"[{row}, {col}])"
                )
                lines += "tlx.async_descriptor_store_wait(0)"
                continue
            # Safety net: an unrecognized op kind landed inside the partition
            # chain (e.g. a scheduler-interleaved side effect between the cast
            # and the store). Emit it once — via the normal outer-op path on the
            # first group — rather than silently dropping it.
            if gi == 0:
                _emit_outer_op(n, g, rctx, lines)
    # All groups' TMEM read + stored — release the accumulator for the next tile.
    lines += "tlx.barrier_arrive(acc_tmem_empty[tmem_buf], 1)"


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
                lines += (
                    f"acc_sub_{sub_n} = tlx.subslice(acc_tmem[tmem_buf], "
                    f"{n_off}, {sub_size})"
                )
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
                lines += (
                    f"tlx.async_descriptor_store({desc}, c_smem[{smem_slot}], "
                    f'[{offs_str}], eviction_policy="evict_first")'
                )
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
        lines += (
            f"tlx.async_descriptor_load({desc}, {bufname}[0], "
            f"[{offs_str}], {full_bar}[0])"
        )
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
        offsets = [_render_operand(o, rctx) for o in op.operands[2:]]
        offs_str = ", ".join(offsets)
        # Cross-WG channel store (multi-WG outer body): the producer WG
        # already staged the tile into the channel SMEM; TMA straight from
        # it and recycle after the drain — no register round-trip.
        sc = getattr(rctx, "_store_from_channel", None)
        if sc is not None:
            rctx._store_from_channel = None
            lines += (
                f"tlx.async_descriptor_store({desc}, "
                f"{sc['buf_var']}[{sc['slot']}], [{offs_str}])"
            )
            lines += "tlx.async_descriptor_store_wait(0)"
            if sc["arrive"]:
                lines += sc["arrive"]
            return
        value_expr = _render_operand(op.operands[1], rctx)
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


def _emit_outer_reduction_stores(
    g: ScheduleGraph,
    outer: Loop,
    inner: Loop | None,
    uwg: UnifiedWG,
    rctx: RenderCtx,
    lines: _Lines,
) -> None:
    """Emit outer-scope `tt.store` ops (e.g. case7's fused bias gradient db)
    whose value is produced by a `tt.reduce` in THIS uwg's inner loop.

    The scheduler leaves such pointer-typed stores unassigned (warp_group ==
    -1) and wraps them in an idempotent `if pid == 0` guard — the reduced
    value is identical across the guarded-away tiles. We emit them
    unconditionally right after this uwg's inner loop, where the loop-carried
    accumulator is still live, mirroring the hand-written
    `tl.store(bias_out + offs, acc)` idiom. No-op for every uwg that doesn't
    own such a reduction (i.e. all the other cases)."""
    if inner is None or uwg.inner_wg is None:
        return
    reduce_ids = {
        n.op_ref
        for n in inner.schedule.nodes
        if n.warp_group == uwg.inner_wg
        and n.op_ref
        and g.ops.get(n.op_ref) is not None
        and g.ops[n.op_ref].kind == "tt.reduce"
    }
    if not reduce_ids:
        return
    # The store value comes through the inner loop's RESULT (the post-loop value
    # of the reduction accumulator), so connect the store to this uwg by walking
    # the value back to the inner loop's `scf.for` op — NOT to the reduce op
    # directly (the reduce is yielded inside the loop body, never an operand of
    # the loop). `reduce_ids` above already gates to the reduction-owning uwg.
    inner_for = _find_loop_for(g, inner)
    inner_for_id = inner_for.op_id if inner_for is not None else None
    if inner_for_id is None:
        return

    def _from_inner_loop(op_id: str, seen: set[str]) -> bool:
        if op_id in seen:
            return False
        seen.add(op_id)
        if op_id == inner_for_id:
            return True
        o = g.ops.get(op_id)
        if o is None:
            return False
        return any(
            isinstance(x, OpRef) and _from_inner_loop(x.op_id, seen) for x in o.operands
        )

    outer_scope = f"loop:{outer.loop_id}"
    for op_id, op in g.ops.items():
        if op.kind != "tt.store" or op.scope != outer_scope:
            continue
        if len(op.operands) < 2 or not isinstance(op.operands[1], OpRef):
            continue
        if not _from_inner_loop(op.operands[1].op_id, set()):
            continue
        ptr_expr = _render_operand(op.operands[0], rctx)
        val_expr = _render_operand(op.operands[1], rctx)
        lines += "# fused reduction store (e.g. bias gradient db): the reduced value is"
        lines += "# identical across the guarded-away tiles, so store unconditionally."
        lines += f"tl.store({ptr_expr}, {val_expr})"


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
    if _has_smem_ring_buffer_inner(inner, uwg, channels):
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

    # OUTER-loop cross-WG semaphores (multi-WG outer bodies): their lowered
    # phase expressions are `_it`-based, so a task touching them needs an
    # outer iteration counter. Inner K-loop bodies rebind `_it` inside their
    # own loop, so a task owning BOTH an inner loop and outer channels would
    # clobber it — refuse loudly rather than emit racy phases.
    has_outer_sems = False
    if rctx.sem_set is not None:
        for n in outer_nodes:
            if n.child_pipeline_id is not None:
                continue
            if rctx.sem_set.by_consumer.get(
                (outer.loop_id, n.id)
            ) or rctx.sem_set.by_producer.get((outer.loop_id, n.id)):
                has_outer_sems = True
                break
    if has_outer_sems and uwg.inner_wg is not None:
        raise RuntimeError(
            "sched2tlx: outer-loop cross-WG channel in a task that also owns "
            f"an inner loop (task {uwg.name}) — the outer iteration counter "
            "would collide with the inner `_it`; this partition shape is not "
            "lowerable yet."
        )
    if has_outer_sems:
        lines += "_oit = 0"

    # Cross-loop register-result channels (persistent kernels): the producing
    # inner WG stages its final iter_arg value through SMEM once per tile; the
    # default (epilogue) task drains it. Registers can't cross warp groups, so
    # this is the SMEM analogue of the acc_tmem TC→default hand-off. Per-tile
    # phase via a dedicated counter, mirroring tmem_accum_cnt.
    cl_prod = [
        ch
        for ch in rctx.crossloop_channels
        if inner is not None
        and uwg.inner_wg is not None
        and ch["loop_id"] == inner.loop_id
        and ch["producer_wg"] == uwg.inner_wg
    ]
    cl_cons = list(rctx.crossloop_channels) if uwg.is_default else []
    for ch in {c["bufname"]: c for c in cl_prod + cl_cons}.values():
        lines += f"{ch['bufname']}_cnt = 0"

    # A non-default async_task cannot capture a RankedTensorType from function
    # scope. Re-materialize any function-scope register tensors this task's ops
    # consume (e.g. blockwise scaled_mm's scale-offset tt.make_range) with
    # task-local names. The per-task op_var snapshot in _emit_uwg_body restores
    # the global bindings afterward. No-op when the task captures nothing.
    _loc_nodes = list(_outer_nodes_for_uwg(outer, uwg))
    if inner is not None and uwg.inner_wg is not None:
        _loc_nodes += _inner_nodes_for_uwg(inner, uwg)
    # No scheduled-node stop-set on this walk (loops=[]): the per-UWG
    # localization descends the full operand tree like the walk that emitted
    # the committed fixtures; the stop-set is scoped to the per-loop WG
    # emission site, and anything missed there is caught by the render-level
    # capture safety net.
    _localize_captured_reg_tensors(g, [], _loc_nodes, rctx, lines, descend_iv=True)

    # Loop-carry pre-arrives (is_released semaphores, e.g. case9 sem4 acc_tmem
    # release): PRIME ONCE here, before the persistent loop. The inner-loop carry
    # is continuous across tiles (smem_accum/_it persists), so priming per tile
    # would add an unmatched arrive and drift the phase → cross-tile deadlock.
    if _use_semaphore_ir() and inner is not None and uwg.inner_wg is not None:
        for line in _semir_pre_arrives_for_wg(inner.loop_id, uwg.inner_wg, rctx):
            lines += line

    # Outer for-loop scaffolding.
    out_iv = outer.schedule.induction_var_name
    out_lo = _render_operand(outer.schedule.lower_bound, rctx)
    out_hi = _render_operand(outer.schedule.upper_bound, rctx)
    out_step = _render_operand(outer.schedule.step, rctx)

    lines += (
        f"# Outer persistent loop (loop {outer.loop_id}, II={outer.schedule.II}). "
        f"Each task replays it; body trimmed to this WG's ops."
    )
    with lines.block(f"for {out_iv} in range({out_lo}, {out_hi}, {out_step}):"):
        # Per-tile TMEM ring-buffer indexing.
        if has_tmem:
            tc = rctx.tmem_count
            lines += f"tmem_buf = tmem_accum_cnt % {tc}"
            lines += f"tmem_phase = (tmem_accum_cnt // {tc}) & 1"
        if has_outer_sems:
            # Bind the lowered semaphores' `_it`-based slot/phase expressions
            # to the per-tile counter.
            lines += "_it = _oit"
        # Cross-loop result channel (consumer side): drain the producing WG's
        # staged final iter_arg value and bind its var so the epilogue reads it.
        for ch in cl_cons:
            _ph = f"({ch['bufname']}_cnt & 1)"
            lines += f"tlx.barrier_wait({_bar_full(ch['bufname'])}[0], {_ph})"
            lines += f"{ch['var_name']} = tlx.local_load({ch['bufname']}[0])"
            lines += f"tlx.barrier_arrive({_bar_empty(ch['bufname'])}[0], 1)"
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

            in_loop_infra = [
                op for op in in_loop_infra if not _depends_on_outer_load(op.op_id)
            ]
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
        part_info = (
            _find_partition_chain(outer_nodes, g, outer) if not sub_info else None
        )
        part_start = part_info[0] if part_info else -1
        part_end = part_info[1] if part_info else -1
        # Pass A.5: the partition scheduler may assign the epilogue
        # `tt.descriptor_store` to a different warp group than its tmem_load →
        # trunc → convert chain (the chain lands in the TMEM-reading WG, the
        # store in the default/epilogue WG). `outer_nodes` is per-WG, so neither
        # WG sees the whole chain. Locate the store globally: the partitioned
        # epilogue (emitted in the tmem_load's WG, where the cast values are
        # live) issues it, and the orphan store in the other WG is skipped.
        part_acc = any(
            b.kind == "tmem" and b.partition_count > 1 for b in outer.schedule.buffers
        )
        global_store = (
            next(
                (n for n in outer.schedule.nodes if n.op_kind == "tt.descriptor_store"),
                None,
            )
            if part_acc
            else None
        )
        for i, n in enumerate(outer_nodes):
            if n.child_pipeline_id is not None:
                # Super-node → emit the inner K-loop here.
                _emit_inner_loop_in_outer(
                    g, inner, uwg, channels, rctx, lines, persistent=True
                )
                # Fused post-loop reduction store (e.g. case7 bias gradient db),
                # emitted in the WG that owns the reduction accumulator.
                _emit_outer_reduction_stores(g, outer, inner, uwg, rctx, lines)
                # Cross-loop result channel (producer side): stage this WG's
                # final iter_arg value through SMEM for the epilogue task.
                for ch in cl_prod:
                    _ph = f"({ch['bufname']}_cnt & 1)"
                    lines += (
                        f"tlx.barrier_wait({_bar_empty(ch['bufname'])}[0], {_ph} ^ 1)"
                    )
                    lines += f"tlx.local_store({ch['bufname']}[0], {ch['var_name']})"
                    lines += f"tlx.barrier_arrive({_bar_full(ch['bufname'])}[0], 1)"
                continue
            if sub_info and i == subtile_start:
                chain_nodes = outer_nodes[subtile_start:subtile_end]
                _emit_outer_epilogue_subtiled(
                    chain_nodes, sub_info[2], sub_info[3], g, rctx, lines
                )
                continue
            if sub_info and subtile_start < i < subtile_end:
                continue  # already emitted inside the subtile loop
            if part_info and i == part_start:
                chain_nodes = list(outer_nodes[part_start:part_end])
                # Splice in the cross-WG epilogue store if the chain didn't
                # already capture it (store assigned to a different WG).
                if global_store is not None and all(
                    cn.id != global_store.id for cn in chain_nodes
                ):
                    chain_nodes.append(global_store)
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
            if (
                global_store is not None
                and part_info is None
                and n.id == global_store.id
            ):
                continue  # epilogue store emitted by the partitioned epilogue in the tmem_load's WG
            # OUTER cross-WG handshakes: consumer waits/loads before the op
            # (a descriptor_store consumer instead TMA-stores straight from
            # the channel buffer via _store_from_channel), producer
            # wait-empty/store/arrive-full after the value is materialized.
            if has_outer_sems:
                _semir_emit_consumer_block(n, g, outer, rctx, lines)
            _emit_outer_op(n, g, rctx, lines)
            if has_outer_sems and n.op_ref:
                opv = rctx.op_var.get(n.op_ref)
                if opv is not None:
                    _semir_emit_producer_block(n, opv, g, outer, rctx, lines)

        # Advance the cross-loop result-channel counter(s) at end of each tile
        # (producer and consumer stay in phase lockstep).
        for ch in {c["bufname"]: c for c in cl_prod + cl_cons}.values():
            lines += f"{ch['bufname']}_cnt += 1"
        # Advance the TMEM ring counter at end of each tile (both default
        # and TC partitions, so tmem_buf/tmem_phase stay in sync).
        if has_tmem:
            lines += "tmem_accum_cnt += 1"
        if has_outer_sems:
            lines += "_oit += 1"

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
    # Emit a generated-code marker so linters/formatters (arc f, black) skip the
    # output. Token is split here so this emitter source isn't itself flagged.
    lines += "# @" + "generated by sched2tlx — do not edit by hand."
    lines += f"# Source: schedule_graph for kernel `{graph.kernel.name}`"
    if graph.loops:
        wgs = ", ".join(
            f"wg{w.id}=[{'+'.join(w.pipelines)}]" for w in graph.loops[0].warp_groups
        )
        lines += f"# Warp groups (loop 0): {wgs}"
    lines += "import torch"
    lines += "import triton"
    lines += "import triton.language as tl"
    lines += "import triton.language.extra.tlx as tlx"
    lines += ""
    _kernel_sig_lines(graph, lines)

    outer_loop = _find_outer_loop(graph)
    inner_loop = _find_inner_loop(graph, outer_loop) if outer_loop else None

    # Pre-pass: clamp intra-WG SMEM→MMA bridge buffers to the bridge
    # protocol's depth-1 BEFORE buffer allocs render — the same derivation
    # runs again later for channel construction and its clamp side effect
    # is idempotent, but by then the alloc lines (count=N) are already out.
    _bridge_detect_loop = inner_loop if inner_loop is not None else outer_loop
    if _bridge_detect_loop is not None:
        _derive_smem_bridge_channels(graph, _bridge_detect_loop)

    # Fold pipeline=NONE sink ops (warp_group=-1) into their producer's group.
    for L in graph.loops:
        _reassign_orphan_nodes(graph, L)

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
    # Per-buffer MMA-consumer counts → intra-WG `extra` empties need
    # arrive_count = N consumers (each consuming async_dot recycles EMPTY).
    rctx._buf_consumer_count = _buf_mma_consumer_counts(graph)
    # The acc_tmem TC→default-epilogue carve-out only applies when the epilogue
    # actually reads the accumulator via a tmem_load OUTSIDE the inner loop
    # (cases 2/5: MMA accumulates across K into acc_tmem, epilogue reads it once
    # per tile). Two cases where it must NOT fire, else the TC's acc_tmem_empty
    # wait is never arrived and the kernel hangs / the default waits a full that
    # never comes: (a) no MMA at all (case6 LayerNorm); (b) the only tmem_load is
    # INSIDE the inner loop (blockwise scaled_mm: a fresh per-group MMA partial
    # the promotion drains via its own sem channels — acc_tmem is intra-loop
    # scratch, not a per-tile accumulator).
    _epi_scopes = {"function"} | {f"loop:{L.loop_id}" for L in graph.loops if L.is_outer}
    _has_mma = any(
        op.kind in ("ttng.tc_gen5_mma", "ttng.tc_gen5_mma_scaled")
        for op in graph.ops.values()
    )
    _epi_tmem_read = any(
        op.kind == "ttng.tmem_load" and op.scope in _epi_scopes
        for op in graph.ops.values()
    )
    rctx.has_acc_tmem_handoff = _has_mma and _epi_tmem_read

    lines.indent = 1
    if outer_loop is None:
        return lines.render()

    # Intra-WG stage-skew plan (emitter software pipelining). Non-persistent
    # kernels only for now — the persistent per-tile body has its own inner
    # loop emitter; a WG needing skew there falls back to serial emission.
    skew_skip_pairs: set = set()
    if _use_semaphore_ir() and inner_loop is None:
        skew_skip_pairs = _compute_skew_plan(graph, outer_loop, rctx)

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
            lines += (
                f"# {name}: outer-loop SMEM buf {buf.id}, count={buf.count} "
                f"(per-tile resident, e.g. Q tile through K-loop)"
            )
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
            lines += (
                f"# {bufname}: dedicated staging for outer descriptor_load "
                f"N{node.id} (shape ({shape_str}))"
            )
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
            # Stash the primary's count for the TC↔default ring depth. When
            # acc_tmem is intra-loop scratch (no epilogue read — blockwise), the
            # MMA↔promotion sem channels are single-slot, so force depth-1 and
            # the MMA writes acc_tmem[0] where the in-loop promotion reads.
            if buf is primary:
                rctx.tmem_count = buf.count if rctx.has_acc_tmem_handoff else 1
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
                    lines += (
                        f"# {gname}: Pass A.5 group {gi}/{buf.partition_count}"
                        f" of partitioned TMEM acc buf {buf.id} "
                        f"(per-group shape ({pshape_str}))"
                    )
                    lines += (
                        f"{gname} = tlx.local_alloc(({pshape_str}), "
                        f"tl.float32, {buf.count}, tlx.storage_kind.tmem)"
                    )
                rctx.partition_buffer_names[(outer_loop.loop_id, buf.id)] = names
                if buf.def_op:
                    rctx.partition_alloc_names[buf.def_op] = names
                # `name` (legacy `acc_tmem`) aliases group 0 so any path that
                # references it without partition awareness still resolves.
                lines += f"{name} = {names[0]}"
            else:
                # Emitter capability (EmitterCaps.kMaxTMEMBlockM): TMEM
                # accumulators support blockM <= 128 — fail clearly instead
                # of emitting a kernel that traps (case2's 256-blockM
                # pre_modulo is the known instance; the unpartitioned
                # carve-out path is where it lands).
                if buf.shape and buf.shape[0] > 128:
                    raise NotImplementedError(
                        f"TMEM accumulator buf {buf.id} has "
                        f"blockM={buf.shape[0]} > 128; the emitter cannot "
                        "split MMAs for tiles beyond the TMEM row limit "
                        "(EmitterCaps.kMaxTMEMBlockM). Regenerate the "
                        "schedule with blockM <= 128.")
                lines += (
                    f"# {name}: outer-loop buf {buf.id}, count={buf.count} "
                    f"(TC writes / default reads across {buf.count} tiles)"
                )
                lines += (
                    f"{name} = tlx.local_alloc(({shape}), tl.float32, "
                    f"{buf.count}, tlx.storage_kind.tmem)"
                )
        # Epilogue staging SMEM (for descriptor_store) — derive from the
        # store's descriptor operand. The descriptor may be a kernel arg
        # (case2) or a make_tensor_descriptor op (case1).
        epi_store = next(
            (op for op in graph.ops.values() if op.kind == "tt.descriptor_store"), None
        )
        if epi_store and epi_store.operands:
            shape = [128, 128]
            dtype = "tl.float16"
            op0 = epi_store.operands[0]
            if isinstance(op0, OpRef):
                desc_op = graph.ops.get(op0.op_id)
                if desc_op:
                    bs = _parse_desc_block_shape(
                        desc_op.result_types[0] if desc_op.result_types else ""
                    )
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
            # Pass A.5: a data-partitioned accumulator stores one (m_size, BN)
            # group at a time reusing this buffer — shrink c_smem to (m_size, BN)
            # so deeper operand rings fit in SMEM.
            part_m = 0
            for lp in graph.loops:
                for b in lp.schedule.buffers:
                    if b.kind == "tmem" and b.partition_count > 1:
                        part_m = b.m_size
                        break
                if part_m:
                    break
            if part_m and len(shape) >= 2:
                # A.5 and A.7 both reshape this staging buffer; the epilogue
                # path treats them as mutually exclusive (`_find_partition_chain`
                # runs only `if not sub_info`). Assert that here so the shrink is
                # a load-bearing invariant rather than an incidental one.
                assert sub_count <= 1, (
                    "Pass A.5 (data partition) and Pass A.7 (epilogue subtile) "
                    "both reshaped c_smem; they are mutually exclusive by design"
                )
                shape = [part_m, shape[1]]
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
                    )
                )
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
                )
            )
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
        # Intra-WG SMEM local_alloc(value) → MMA bridges (v3/v4-style
        # all-MMA-in-one-WG partitions): the SMEM analogue of the intra-WG
        # TMEM bridge, otherwise emitted with no store + no barriers.
        for sb in _derive_smem_bridge_channels(graph, detect_loop):
            sb.name = rctx.buffer_var.get((sb.loop_id, sb.buffer_id), sb.name)
            if sb.name in by_name:
                continue  # already covered by a TMEM/cross-WG channel
            channels.append(sb)
            by_name[sb.name] = sb
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
    bridge_op_ids = {
        c.bridge_op_id for c in channels if c.kind == "tmem" and c.bridge_op_id
    }
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
            has_load = any(
                op.kind in ("ttg.local_alloc",)
                and op.operands
                and (
                    isinstance(op.operands[0], OpRef)
                    and (
                        (g_op := graph.ops.get(op.operands[0].op_id))
                        and g_op.kind == "tt.descriptor_load"
                    )
                )
                for oid, op in graph.ops.items()
                if oid == b.def_op
            )
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
        shape = ", ".join(str(d) for d in ch["shape"]) + (
            "," if len(ch["shape"]) == 1 else ""
        )
        lines += (
            f"# {ch['bufname']}: cross-loop iter_arg result channel "
            f"(loop {ch['loop_id']} iter_arg {ch['idx']} → epilogue)"
        )
        lines += f"{ch['bufname']} = tlx.local_alloc(({shape}), {ch['dtype']}, 1)"

    # ── No warp specialization (single warp group) ──────────────────────────
    # When the schedule assigns every op to one warp group and there are no
    # cross-WG / cross-loop channels, there is nothing to specialize. Emit the
    # loop directly at kernel scope — a plain @triton.jit body, no
    # `tlx.async_tasks()` wrapper, no cross-WG mbarriers, no default-partition
    # task. A TC-free / memory-bound loop (e.g. LayerNorm) gets its load/compute
    # overlap from async TMA + double-buffering within the single warp group;
    # warp specialization would only add barriers.
    #
    # This fast path is TC-FREE ONLY. A single-WG partition that still contains
    # an MMA (TC) / TMEM accumulator needs the full WS path: the MMA→epilogue
    # TMEM hand-off barriers and the in-WG async-load completion barriers must
    # be allocated, and the default partition must emit the epilogue store.
    # (Such a partition — load+MMA serialized in one WG — is the worst-ranked
    # candidate, but the autotuner still needs it to lower correctly.)
    real_wg_ids = sorted(
        {
            n.warp_group
            for L in graph.loops
            for n in L.schedule.nodes
            if n.warp_group is not None and n.warp_group >= 0
        }
    )

    def _is_tc_node(n: Node) -> bool:
        op = graph.ops.get(n.op_ref) if n.op_ref else None
        return op is not None and op.kind in (
            "ttng.tc_gen5_mma",
            "ttng.tc_gen5_mma_scaled",
            "ttng.warp_group_dot",
            "tt.dot",
        )

    has_tc = any(_is_tc_node(n) for L in graph.loops for n in L.schedule.nodes)
    if (
        inner_loop is None
        and not rctx.crossloop_channels
        and len(real_wg_ids) <= 1
        and not has_tc
    ):
        rctx.channels = []
        rctx.sem_set = None
        # M1 (store deferral): in the single-WG path, in-loop TMA stores use the
        # deferred-wait pattern so the store latency overlaps the next iteration's
        # load+compute instead of blocking. Scoped here so the WS path is intact.
        rctx.defer_inloop_store = True
        wg_obj = (
            next((w for w in outer_loop.warp_groups if w.id == real_wg_ids[0]), None)
            if real_wg_ids
            else None
        )
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
    rctx.sem_set = build_sem_set_for_graph(
        graph, wg_of_node=wg_of_node, intra_wg_skip_pairs=skew_skip_pairs
    )

    if _use_semaphore_ir():
        # SemIR-driven mbarrier emission: one Semaphore = one full+empty pair
        # (NVWS protocol). Signal-only Semaphores (no buffer) only allocate
        # the full barrier — there's no recycle direction.
        lines += "# ── Mbarriers (SemIR: full+empty pair per semaphore) ──"
        # A buffer with multiple cross-WG consumers (e.g. wgrad dout feeding
        # both the MMA and the bias-reduce) yields several semaphores that share
        # one buffer-named barrier pair. They must allocate ONCE, and the empty
        # barrier's arrive_count = sum over consumers (each arrives empty once);
        # otherwise the producer recycles the slot after one consumer releases →
        # mid-read overwrite. Dedup by name and sum the empty arrive_counts.
        empty_total: dict[str, int] = {}
        for ls in rctx.sem_set.lowered:
            if ls.alloc_empty_stmt is not None:
                m = re.search(r"arrive_count=(\d+)", ls.alloc_empty_stmt)
                empty_total[ls.empty_name] = empty_total.get(ls.empty_name, 0) + (
                    int(m.group(1)) if m else 1
                )
        # Floor each empty's arrive_count by the buffer's true #MMA-consumers.
        # The cb-sum above undercounts when several MMAs read one channel buffer
        # but the schedule recorded a single cross_wg_barrier (FA-bwd dsT feeds
        # both dK and dQ → 1 cb entry, but 2 async_dots recycle EMPTY).
        bcc = getattr(rctx, "_buf_consumer_count", {}) or {}
        for ls in rctx.sem_set.lowered:
            if ls.alloc_empty_stmt is None or ls.sem.buffer is None:
                continue
            cnt = bcc.get(ls.sem.buffer.buffer_id, 0)
            if cnt > empty_total.get(ls.empty_name, 1):
                empty_total[ls.empty_name] = cnt
        seen_alloc: set[str] = set()
        for ls in rctx.sem_set.lowered:
            sem = ls.sem
            if ls.full_name in seen_alloc:
                continue
            seen_alloc.add(ls.full_name)
            lines += f"# {ls.name}: {sem.note or ''}"
            lines += ls.alloc_full_stmt
            if ls.alloc_empty_stmt is not None:
                lines += re.sub(
                    r"arrive_count=\d+",
                    f"arrive_count={empty_total[ls.empty_name]}",
                    ls.alloc_empty_stmt,
                )
        # Cross-loop / cross-region edges that aren't in any loop's
        # cross_wg_barriers (e.g., the acc_tmem TC-loop → default-epilogue
        # hand-off, and per-loop iter_arg result channels). Keep the legacy
        # full+empty pair convention for these — they're emitted/consumed by
        # the legacy code paths in _emit_default_partition / _emit_warp_group.
        if rctx.has_acc_tmem_handoff and "acc_tmem" not in {c.name for c in channels}:
            lines += (
                f"# acc_tmem: cross-region TC-loop → default-epilogue "
                f"hand-off, depth={rctx.tmem_count} (legacy carve-out)"
            )
            lines += (
                f"acc_tmem_full = tlx.alloc_barriers"
                f"(num_barriers={rctx.tmem_count}, arrive_count=1)"
            )
            lines += (
                f"acc_tmem_empty = tlx.alloc_barriers"
                f"(num_barriers={rctx.tmem_count}, arrive_count=1)"
            )
        # TMEM bridge channels (e.g., FA's softmax → P_tmem TMEM that the
        # PV MMA reads). Their full/empty barriers get emitted by the SW
        # producer block + consumer-side wait.
        for c in channels:
            if c.kind != "tmem" or not c.bridge_op_id:
                continue
            lines += (
                f"# {c.name}: TMEM bridge channel "
                f"(SW producer → MMA consumer via TMEM, depth={c.depth})"
            )
            lines += (
                f"{_bar_full(c.name)} = tlx.alloc_barriers"
                f"(num_barriers={c.depth}, arrive_count=1)"
            )
            # Empty arrive_count = #MMA consumers: each consuming MMA arrives
            # `_empty` once via mBarriers, and the producer's empty-wait must
            # see ALL of them before overwriting the slot. 1-consumer bridges
            # (num_consumers defaults to 1) are unchanged.
            lines += (
                f"{_bar_empty(c.name)} = tlx.alloc_barriers"
                f"(num_barriers={c.depth}, arrive_count={c.num_consumers})"
            )
        # Function-scope per-tile-resident loads (e.g. FA's Q tile): one
        # mbarrier per load. Emitted by MEM-role WG, consumed by the WG
        # whose MMA reads the alloc.
        for fl in rctx.fn_scope_loads:
            lines += f"# {fl['alloc_var']}_full: per-tile resident load barrier"
            lines += (
                f"{fl['alloc_var']}_full = tlx.alloc_barriers"
                f"(num_barriers=1, arrive_count=1)"
            )
        # Non-canonical epilogue accumulators (FA-bwd dK/dV): full+empty pair
        # each, committed post-loop by the producing WG, read by the default
        # partition. (The else/legacy branch routes these through extra_buffers.)
        for var in _epilogue_acc_wg(graph, rctx):
            lines += f"# {var}_full: epilogue accumulator handoff (TC → default)"
            lines += f"{var}_full = tlx.alloc_barriers(num_barriers=1, arrive_count=1)"
            lines += f"{var}_empty = tlx.alloc_barriers(num_barriers=1, arrive_count=1)"
        for name, depth in extra or []:
            lines += (
                f"# {name}: legacy carve-out (cross-loop iter_arg or "
                f"intra-WG async load)"
            )
            # EMPTY arrive_count = #MMA consumers (a load read by N MMAs in the
            # WG is recycled by N async_dots); FULL stays 1 (single producer).
            eac = (getattr(rctx, "_buf_consumer_count", {}) or {}).get(name, 1)
            lines += (
                f"{_bar_full(name)} = tlx.alloc_barriers"
                f"(num_barriers={depth}, arrive_count=1)"
            )
            lines += (
                f"{_bar_empty(name)} = tlx.alloc_barriers"
                f"(num_barriers={depth}, arrive_count={eac})"
            )
        # Intra-WG stage-skew rings: full/empty pair on the skewed async
        # producer's destination buffer. Depth = skew gap + 1; the producer
        # waits empty (phase ^ 1, no pre-arrive — first `depth` waits pass on
        # the fresh barrier), consumers wait full and recycle empty (SW at the
        # last same-stream site; each MMA consumer via its own mBarriers).
        skew_alloc_names = {c.name for c in channels} | {nm for nm, _ in extra or []}
        for var, entry in rctx.skew_ring.items():
            if var in skew_alloc_names or _bar_full(var) in seen_alloc:
                # Delegated: an existing channel/extra pair already owns
                # `{var}_full/_empty`; the ring's waits reuse those names.
                entry["bar_var"] = var
                continue
            entry["bar_var"] = _skw(var)
            skew_alloc_names.add(var)
            eac = len(entry.get("mma_consumers", [])) + (
                1 if entry.get("sw_consumers") else 0
            )
            lines += (
                f"# {var}: intra-WG stage-skew ring (depth={entry['depth']}; "
                f"producer N{entry['producer_node']} issues "
                f"{entry['depth'] - 1} iter(s) ahead of its consumers)"
            )
            lines += (
                f"{_bar_full(entry['bar_var'])} = tlx.alloc_barriers"
                f"(num_barriers={entry['depth']}, arrive_count=1)"
            )
            lines += (
                f"{_bar_empty(entry['bar_var'])} = tlx.alloc_barriers"
                f"(num_barriers={entry['depth']}, arrive_count={max(eac, 1)})"
            )
    else:
        # Function-scope per-tile-resident loads (e.g. FA's Q tile): one
        # mbarrier per load. Routed through `extra` so the legacy
        # `_emit_mbarriers` allocates the full+empty pair.
        extra_with_fnscope = list(extra)
        for fl in rctx.fn_scope_loads:
            extra_with_fnscope.append((fl["alloc_var"], 1))
        # Non-canonical epilogue accumulators (FA-bwd dK/dV): one full+empty
        # pair each, committed by the producing WG after its loop.
        for var in _epilogue_acc_wg(graph, rctx):
            extra_with_fnscope.append((var, 1))
        _emit_mbarriers(
            channels,
            lines,
            tmem_count=rctx.tmem_count,
            have_separate_tmem_handoff=rctx.has_acc_tmem_handoff,
            extra_buffers=extra_with_fnscope,
        )
    rctx.channels = channels

    # Async tasks. One per unified warp group; each runs the outer
    # persistent loop (if any) replicated and trimmed to its ops.
    uwgs = _unified_warp_groups(graph, outer_loop, inner_loop)
    _check_task_coverage(graph, outer_loop, inner_loop, uwgs)
    with lines.block("with tlx.async_tasks():"):
        for uwg in uwgs:
            # Reference Phase 4's plan so the role attribution is visible.
            origin = (
                f"outer wg{uwg.outer_wg}"
                if uwg.outer_wg is not None
                else (
                    f"inner wg{uwg.inner_wg}" if uwg.inner_wg is not None else "(none)"
                )
            )
            lines += f"# Async task: role={uwg.role} ← {origin} (Phase 4 plan)"
            with lines.block(_task_header(uwg)):
                body_start = len(lines.buf)
                _emit_uwg_body(
                    graph, outer_loop, inner_loop, uwg, channels, rctx, lines
                )
                if not uwg.is_default:
                    _localize_rendered_captures(graph, rctx, lines, body_start)

    return lines.render()
