"""Unified in-memory model of a kernel's schedule graph for the solver.

Primary input is modulo's ``schedule_graph.json`` (the emitter's input, which
embeds the full DDG + the committed schedule). When the pre-schedule
``ddg.json`` sits next to it we merge two fields it uniquely carries — per-node
``occupancy`` (true resource hold-time) and per-loop ``min_ii``/``res_mii``/
``rec_mii`` — by matching node ``id`` within each loop. When ``ddg.json`` is
absent (e.g. case7) we fall back to estimates derived from the schedule graph.

The raw JSON dict is retained verbatim so ``graph_writer`` can rewrite only the
fields the emitter consumes and pass everything else through unchanged.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import MachineModel


@dataclass
class SolverNode:
    id: int
    op_kind: str
    pipeline: str  # TMA | TC | CUDA | SFU | NONE
    latency: int
    self_latency: int
    occupancy: int  # resource hold-time (from ddg.json, else estimated)
    min_warps: int
    is_super_node: bool
    produces_buffer: int | None
    consumes_buffers: list[int]
    warp_group: int  # baseline assignment (from schedule_graph.json)
    result_type: str  # first result MLIR type (for regs derivation), or ""


@dataclass
class SolverEdge:
    src: int
    dst: int
    distance: int  # 0 = intra-iteration, >=1 = loop-carried
    latency: int


@dataclass
class SolverBuffer:
    id: int
    kind: str  # smem | tmem | barrier | register
    shape: list[int]
    element_bits: int
    count: int  # baseline depth
    size_bytes: int
    total_bytes: int
    paired_buffer_id: int | None
    merge_group_id: int | None


@dataclass
class LoopModel:
    loop_id: int
    is_outer: bool
    nodes: list[SolverNode]
    edges: list[SolverEdge]
    buffers: list[SolverBuffer]
    baseline_ii: int
    min_ii: int
    res_mii: int
    rec_mii: int
    # Baseline warp-group definitions [{id, pipelines, num_warps}].
    warp_groups: list[dict[str, Any]]
    # Baseline cross-WG barriers (verbatim dicts).
    cross_wg_barriers: list[dict[str, Any]]

    def node_by_id(self, nid: int) -> SolverNode:
        for n in self.nodes:
            if n.id == nid:
                return n
        raise KeyError(f"node id {nid} not in loop {self.loop_id}")

    def buffer_by_id(self, bid: int) -> SolverBuffer | None:
        for b in self.buffers:
            if b.id == bid:
                return b
        return None


@dataclass
class Model:
    kernel_name: str
    loops: list[LoopModel]
    machine: MachineModel
    raw: dict[str, Any]  # verbatim schedule_graph.json dict
    schedule_graph_path: Path
    ddg_path: Path | None = field(default=None)

    def raw_copy(self) -> dict[str, Any]:
        return copy.deepcopy(self.raw)


# ---------------------------------------------------------------------------
# occupancy fallback (when ddg.json is missing)
# ---------------------------------------------------------------------------


def _estimate_occupancy(pipeline: str, latency: int, self_latency: int) -> int:
    """Conservative occupancy estimate matching NVLatencyModel semantics:
    TC serializes (occ == latency); TMA has a small engine hold-time; others
    hold their self_latency.
    """
    if pipeline == "TC":
        return latency
    if pipeline == "TMA":
        # NVLatencyModel: occupancy = max(issue=30, 6*KB); latency includes the
        # ~460 fixed transfer wait. Without KB we use self_latency (issue cost),
        # which is a safe lower bound for the resource table.
        return max(self_latency, 1)
    if pipeline == "NONE":
        return 0
    return max(self_latency, 1)


# ---------------------------------------------------------------------------
# Pass A.5 data-partition SMEM accounting
# ---------------------------------------------------------------------------


def _apply_dp_smem_shrink(buffers: list[SolverBuffer],
                          raw_buffers: list[dict[str, Any]]) -> None:
    """Mirror the emitter's A.5 c_smem shrink in the solver's accounting.

    When a TMEM accumulator is partitioned (partition_count > 1), the emitter
    stages the epilogue through a single per-group (m_size, BN) c_smem instead
    of the full (BM, BN) — see emitter.py Pass A.5. The modulo pass uses the
    shrunk size in ITS budget math (that is how a partitioned dump affords
    deeper operand rings) but dumps the un-shrunk shape. Scale the matching
    SMEM buffer here so the solver's SMEM budget sees the memory the kernel
    actually allocates; graph_writer never writes sizes back, so this stays
    an in-memory accounting view.
    """
    dp = [(rb.get("m_size", 0), list(rb.get("shape") or []))
          for rb in raw_buffers
          if rb.get("kind") == "tmem" and rb.get("partition_count", 1) > 1]
    if not dp:
        return
    m_size, acc_shape = dp[0]
    if not acc_shape or m_size <= 0 or m_size >= acc_shape[0]:
        return
    for b in buffers:
        # Only the epilogue staging buffer — the SMEM buf with the
        # accumulator's full (BM, BN) shape. Operand rings (BM, BK) keep
        # their size: the emitter slices them per group, it does not shrink
        # the allocation.
        if b.kind == "smem" and list(b.shape) == acc_shape:
            b.shape = [m_size] + list(b.shape[1:])
            b.size_bytes = b.size_bytes * m_size // acc_shape[0]
            b.total_bytes = b.size_bytes * b.count


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def _first_result_type(op: dict[str, Any] | None) -> str:
    if not op:
        return ""
    rts = op.get("result_types", [])
    return rts[0] if rts else ""


def load(schedule_graph_path: str | Path,
         ddg_path: str | Path | None = None) -> Model:
    sg_path = Path(schedule_graph_path)
    with open(sg_path) as f:
        sg = json.load(f)

    # Auto-discover a sibling ddg.json if not given.
    if ddg_path is None:
        cand = sg_path.parent / "ddg.json"
        ddg_path = cand if cand.exists() else None
    ddg = None
    if ddg_path is not None:
        with open(ddg_path) as f:
            ddg = json.load(f)

    machine = MachineModel()
    if ddg is not None:
        cfg = ddg.get("config", {})
        machine = MachineModel(
            smem_budget_bytes=cfg.get("smem_budget_bytes",
                                      machine.smem_budget_bytes),
            tmem_budget_bytes=cfg.get("tmem_budget_bytes",
                                      machine.tmem_budget_bytes),
        )

    ops_table = sg.get("ops", {})

    # Index ddg.json loops by loop_id for the occupancy/MII merge.
    ddg_loops: dict[int, dict[str, Any]] = {}
    if ddg is not None:
        for dl in ddg.get("loops", []):
            ddg_loops[dl.get("loop_id")] = dl

    loops: list[LoopModel] = []
    for L in sg.get("loops", []):
        loop_id = L["loop_id"]
        sched = L["schedule_loop"]
        g = sched.get("graph", {})

        dl = ddg_loops.get(loop_id)
        # node id -> occupancy from ddg.json
        occ_by_id: dict[int, int] = {}
        super_by_id: dict[int, bool] = {}
        if dl is not None:
            for dn in dl.get("ddg", {}).get("nodes", []):
                occ_by_id[dn["id"]] = dn.get("occupancy", 0)
                super_by_id[dn["id"]] = dn.get("is_super_node", False)

        nodes: list[SolverNode] = []
        for n in g.get("nodes", []):
            nid = n["id"]
            pipeline = n.get("pipeline", "NONE")
            latency = n.get("latency", 0)
            self_lat = n.get("self_latency", 0)
            occ = occ_by_id.get(
                nid, _estimate_occupancy(pipeline, latency, self_lat))
            is_super = super_by_id.get(
                nid, n.get("child_pipeline_id") is not None)
            nodes.append(SolverNode(
                id=nid,
                op_kind=n.get("op_kind", ""),
                pipeline=pipeline,
                latency=latency,
                self_latency=self_lat,
                occupancy=occ,
                min_warps=n.get("min_warps", 1),
                is_super_node=is_super,
                produces_buffer=n.get("produces_buffer"),
                consumes_buffers=n.get("consumes_buffers", []),
                warp_group=n.get("warp_group", -1),
                result_type=_first_result_type(ops_table.get(n.get("op_ref"))),
            ))

        edges = [SolverEdge(src=e["src"], dst=e["dst"],
                            distance=e["distance"], latency=e["latency"])
                 for e in g.get("edges", [])]

        buffers = [SolverBuffer(
            id=b["id"], kind=b["kind"], shape=b.get("shape", []),
            element_bits=b.get("element_bits", 0), count=b["count"],
            size_bytes=b.get("size_bytes", 0),
            total_bytes=b.get("total_bytes", 0),
            paired_buffer_id=b.get("paired_buffer_id"),
            merge_group_id=b.get("merge_group_id"),
        ) for b in sched.get("buffers", [])]
        _apply_dp_smem_shrink(buffers, sched.get("buffers", []))

        min_ii = res_mii = rec_mii = 0
        if dl is not None:
            min_ii = dl.get("min_ii", 0)
            res_mii = dl.get("res_mii", 0)
            rec_mii = dl.get("rec_mii", 0)

        loops.append(LoopModel(
            loop_id=loop_id,
            is_outer=L.get("is_outer", False),
            nodes=nodes,
            edges=edges,
            buffers=buffers,
            baseline_ii=sched.get("II", 0),
            min_ii=min_ii,
            res_mii=res_mii,
            rec_mii=rec_mii,
            warp_groups=L.get("warp_groups", []),
            cross_wg_barriers=g.get("cross_wg_barriers", []),
        ))

    return Model(
        kernel_name=sg.get("kernel", {}).get("name", "kernel"),
        loops=loops,
        machine=machine,
        raw=sg,
        schedule_graph_path=sg_path,
        ddg_path=Path(ddg_path) if ddg_path is not None else None,
    )
