# pyre-strict
"""Typed view over modulo's schedule_graph.json (schema v0.1).

The dump is produced by `dumpScheduleGraphAsJSON` in the modulo pass and is
self-sufficient — no companion file needed. See SCHEMA.md for the contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Operand refs
# ---------------------------------------------------------------------------


@dataclass
class OpRef:
    op_id: str
    result_idx: int = 0  # which result of a multi-result op (0 if single)


@dataclass
class ArgRef:
    name: str


@dataclass
class IvRef:
    loop_id: int


@dataclass
class IterArgRef:
    loop_id: int
    idx: int


@dataclass
class ConstRef:
    value: Any
    type: str


@dataclass
class UnknownBlockArg:
    pass


OperandRef = OpRef | ArgRef | IvRef | IterArgRef | ConstRef | UnknownBlockArg


def _parse_operand(d: dict[str, Any]) -> OperandRef:
    if "op" in d:
        return OpRef(op_id=d["op"], result_idx=d.get("result", 0))
    if "arg" in d:
        return ArgRef(name=d["arg"])
    if "iv" in d:
        return IvRef(loop_id=d["iv"])
    if "iter_arg" in d:
        return IterArgRef(loop_id=d["iter_arg"]["loop"], idx=d["iter_arg"]["idx"])
    if "const" in d:
        return ConstRef(value=d["const"], type=d.get("type", ""))
    return UnknownBlockArg()


# ---------------------------------------------------------------------------
# Top-level entities
# ---------------------------------------------------------------------------


@dataclass
class KernelArg:
    name: str
    type: str  # "*f16", "i32", ...


@dataclass
class Op:
    """An entry in the top-level ops table — any op in the kernel."""

    op_id: str
    kind: str  # MLIR op name (e.g., "tt.descriptor_load")
    scope: str  # "function" or "loop:<id>"
    operands: list[OperandRef]
    result_types: list[str]
    attributes: dict[str, Any]
    # For scf.if: yielded values from each region (one per result).
    then_yields: list[OperandRef] = field(default_factory=list)
    else_yields: list[OperandRef] = field(default_factory=list)


@dataclass
class Buffer:
    """A multi-buffered allocation inside a loop."""

    id: int
    kind: str  # "smem" | "tmem" | "barrier" | "register"
    shape: list[int]
    element_bits: int
    count: int  # depth (modulo's lifetime-aware count)
    size_bytes: int
    total_bytes: int
    merge_group_id: int | None
    paired_buffer_id: int | None  # for SMEM buf, this is the BARRIER buf
    live_start: int
    live_end: int
    def_op: str | None  # op_id of the original alloc op
    # Pass A.5: when partition_count > 1, this buffer is the TMEM
    # accumulator for an MMA that A.5 splits into N groups; the emitter
    # allocates N separate TMEM bufs each of size (m_size, BN).
    partition_count: int = 1
    partition_dim: int = 0
    m_size: int = 0


@dataclass
class Node:
    """A scheduled op in a loop's graph."""

    id: int  # stable within this loop
    op_ref: str | None  # key into top-level ops table
    op_kind: str  # MLIR op name (also reachable via op_ref → ops table)
    pipeline: str  # "TMA" | "TC" | "CUDA" | "SFU" | "NONE"
    warp_group: int  # index into loop.warp_groups
    latency: int
    self_latency: int
    frequency_multiplier: int
    schedule_cycle: int
    schedule_stage: int
    schedule_cluster: int
    produces_buffer: int | None  # buffer.id this node fills
    consumes_buffers: list[int]
    child_pipeline_id: int | None = None
    prologue_latency: int = 0
    # Pass A.7 epilogue subtiling. When subtile_count > 1, this op is one of
    # S sibling ops produced by splitting an epilogue chain along N; emitter
    # groups siblings into a `for sub in range(S)` loop. Defaults mean "not
    # subtiled" (op operates on the full BN tile).
    subtile_index: int = -1
    subtile_count: int = 1
    n_offset: int = 0
    n_size: int = 0  # 0 = full BN

    # Pass A.5 data partitioning. When partition_count > 1, the op (and its
    # companion chain — A-side load+alloc and the TMEM accumulator buf for
    # an MMA) should be emitted N times in parallel (NUM_MMA_GROUPS style).
    # `m_size` is the per-group tile width along the partition dim.
    partition_index: int = -1
    partition_count: int = 1
    partition_dim: int = 0  # 0 = M, 1 = N
    m_size: int = 0


@dataclass
class Edge:
    src: int  # node id
    dst: int  # node id
    kind: str  # "data" | "anti" | ...
    distance: int  # 0 = intra-iter, 1+ = loop-carried
    latency: int


@dataclass
class CrossWGBarrier:
    """Cross-warp-group barrier from Pass B Step 2. Pairs producer node with
    consumer node and the SMEM/TMEM buffer that ferries the value across."""

    producer_node: int
    consumer_node: int
    producer_wg: int
    consumer_wg: int
    kind: str  # "mbarrier" | "named"
    depth: int
    paired_buffer_id: int | None
    expect_bytes: int


@dataclass
class WarpGroup:
    id: int
    pipelines: list[str]
    # Chosen by the schedule pass (Layer B): max minWarps over the WG's ops,
    # snapped to {1,2,4,8}. Defaults to 4 for backward compatibility with
    # older JSON dumps that did not record num_warps.
    num_warps: int = 4


@dataclass
class ScheduleLoop:
    """The schedule for one scf.for."""

    id: int
    II: int
    max_stage: int
    prologue_latency: int
    trip_count: int
    trip_count_estimated: bool
    induction_var_name: str
    induction_var_type: str
    lower_bound: OperandRef
    upper_bound: OperandRef
    step: OperandRef
    buffers: list[Buffer]
    nodes: list[Node]
    edges: list[Edge]
    cross_wg_barriers: list[CrossWGBarrier] = field(default_factory=list)


@dataclass
class Loop:
    """A loop entry under graph.loops."""

    loop_id: int
    is_outer: bool
    warp_groups: list[WarpGroup]
    schedule: ScheduleLoop


@dataclass
class Kernel:
    name: str
    args: list[KernelArg]


@dataclass
class ScheduleGraph:
    schema_version: str
    kernel: Kernel
    ops: dict[str, Op]
    loops: list[Loop]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _to_op(op_id: str, d: dict[str, Any]) -> Op:
    op = Op(
        op_id=op_id,
        kind=d["kind"],
        scope=d["scope"],
        operands=[_parse_operand(o) for o in d.get("operands", [])],
        result_types=d.get("result_types", []),
        attributes=d.get("attributes", {}),
    )
    # scf.if regions: then/else yielded values exposed as parsed OperandRefs
    # in the same shape as `operands`. Stored on the Op object for renderers.
    if "then_yields" in d:
        op.then_yields = [_parse_operand(o) for o in d["then_yields"]]
    if "else_yields" in d:
        op.else_yields = [_parse_operand(o) for o in d["else_yields"]]
    return op


def _to_buffer(d: dict[str, Any]) -> Buffer:
    return Buffer(
        id=d["id"],
        kind=d["kind"],
        shape=d["shape"],
        element_bits=d["element_bits"],
        count=d["count"],
        size_bytes=d["size_bytes"],
        total_bytes=d["total_bytes"],
        merge_group_id=d.get("merge_group_id"),
        partition_count=d.get("partition_count", 1),
        partition_dim=d.get("partition_dim", 0),
        m_size=d.get("m_size", 0),
        paired_buffer_id=d.get("paired_buffer_id"),
        live_start=d["live_start"],
        live_end=d["live_end"],
        def_op=d.get("def_op"),
    )


def _to_node(d: dict[str, Any]) -> Node:
    sched = d.get("schedule", {})
    return Node(
        id=d["id"],
        op_ref=d.get("op_ref"),
        op_kind=d.get("op_kind", ""),
        pipeline=d.get("pipeline", "NONE"),
        warp_group=d.get("warp_group", -1),
        latency=d.get("latency", 0),
        self_latency=d.get("self_latency", 0),
        frequency_multiplier=d.get("frequency_multiplier", 1),
        schedule_cycle=sched.get("cycle", -1),
        schedule_stage=sched.get("stage", -1),
        schedule_cluster=sched.get("cluster", -1),
        produces_buffer=d.get("produces_buffer"),
        consumes_buffers=d.get("consumes_buffers", []),
        child_pipeline_id=d.get("child_pipeline_id"),
        prologue_latency=d.get("prologue_latency", 0),
        subtile_index=d.get("subtile_index", -1),
        subtile_count=d.get("subtile_count", 1),
        n_offset=d.get("n_offset", 0),
        n_size=d.get("n_size", 0),
        partition_index=d.get("partition_index", -1),
        partition_count=d.get("partition_count", 1),
        partition_dim=d.get("partition_dim", 0),
        m_size=d.get("m_size", 0),
    )


def _to_schedule_loop(d: dict[str, Any]) -> ScheduleLoop:
    iv = d.get("induction_var", {})
    g = d.get("graph", {})
    return ScheduleLoop(
        id=d["id"],
        II=d["II"],
        max_stage=d["max_stage"],
        prologue_latency=d.get("prologue_latency", 0),
        trip_count=d.get("trip_count", 0),
        trip_count_estimated=d.get("trip_count_estimated", False),
        induction_var_name=iv.get("name", "iv"),
        induction_var_type=iv.get("type", "i32"),
        lower_bound=_parse_operand(d["lower_bound"]),
        upper_bound=_parse_operand(d["upper_bound"]),
        step=_parse_operand(d["step"]),
        buffers=[_to_buffer(b) for b in d.get("buffers", [])],
        nodes=[_to_node(n) for n in g.get("nodes", [])],
        edges=[
            Edge(
                src=e["src"],
                dst=e["dst"],
                kind=e["kind"],
                distance=e["distance"],
                latency=e["latency"],
            ) for e in g.get("edges", [])
        ],
        cross_wg_barriers=[
            CrossWGBarrier(
                producer_node=b["producer_node"],
                consumer_node=b["consumer_node"],
                producer_wg=b["producer_wg"],
                consumer_wg=b["consumer_wg"],
                kind=b["kind"],
                depth=b["depth"],
                paired_buffer_id=b.get("paired_buffer_id"),
                expect_bytes=b["expect_bytes"],
            ) for b in g.get("cross_wg_barriers", [])
        ],
    )


def load_graph(path: str | Path) -> ScheduleGraph:
    with open(path) as f:
        data = json.load(f)
    return ScheduleGraph(
        schema_version=data.get("schema_version", "0.0"),
        kernel=Kernel(
            name=data["kernel"]["name"],
            args=[KernelArg(**a) for a in data["kernel"]["args"]],
        ),
        ops={op_id: _to_op(op_id, op_data)
             for op_id, op_data in data.get("ops", {}).items()},
        loops=[
            Loop(
                loop_id=L["loop_id"],
                is_outer=L.get("is_outer", False),
                warp_groups=[WarpGroup(**w) for w in L.get("warp_groups", [])],
                schedule=_to_schedule_loop(L["schedule_loop"]),
            ) for L in data.get("loops", [])
        ],
    )
