"""ScheduleBinder — extract the bindable parameter subset (SKC step 2).

From a joint solution it derives, in the solver's normalized time frame:
  * NUM_MMA_GROUPS       — number of softmax chains
  * NUM_BUFFERS_KV/QK    — ring depths from schedule liveness (span/II)
  * chain order          — which chain the mma issuer serves first (ping-pong)
  * issue-order check    — solver mma order vs skeleton program order
  * register quotas      — R2 role table (validated against the reg file)

Cycle-level detail the skeleton cannot express (exact sub-II offsets, the
solver's intra-warp placement of zero-latency ops) is explicitly dropped and
listed in the audit — the same information loss the paper accepts when its
schedule is handed to a human implementer.
"""

import re
from dataclasses import dataclass, field

from .roles import RoleMap

_TENSOR_SHAPE_RE = re.compile(r"tensor<(\d+)x(\d+)x")

MMA_KIND = "ttng.tc_gen5_mma"

# Zero-latency plumbing the liveness walk sees through: the value they carry
# is the producer's buffer, not a new one.
_PASSTHROUGH = {"ttg.local_alloc", "ttg.memdesc_trans", "ttng.tmem_alloc",
                "tt.expand_dims", "tt.broadcast"}


@dataclass
class Binding:
    params: dict
    audit: dict = field(default_factory=dict)


def _succs(prob):
    out = {}
    for e in prob.edges:
        out.setdefault(e.src, []).append(e)
    return out


def _t(sol, v):
    return sol["cycles"][str(v)]


def _classify_mmas(prob, roles: RoleMap, sol):
    """Split tc_gen5 nodes into QK (feeds a softmax chain) and PV (accumulates)."""
    softmax_warps = set(roles.softmax)
    warp_of = {int(i): w for i, w in sol["warp"].items()}
    preds = {}
    for e in prob.edges:
        preds.setdefault(e.dst, []).append(e.src)
    qk, pv = [], []
    for v, n in prob.nodes.items():
        if n.op_kind != MMA_KIND:
            continue
        if any(warp_of.get(p) in softmax_warps for p in preds.get(v, [])):
            pv.append(v)
        else:
            qk.append(v)
    return qk, pv


def _chain_of_qk(prob, roles: RoleMap, sol, qk_mmas):
    """Map each QK mma to the softmax warp that consumes its result."""
    warp_of = {int(i): w for i, w in sol["warp"].items()}
    succs = _succs(prob)
    chain = {}
    for v in qk_mmas:
        stack, seen = [v], set()
        while stack:
            u = stack.pop()
            for e in succs.get(u, []):
                c = e.dst
                if c in seen:
                    continue
                seen.add(c)
                w = warp_of.get(c)
                if w in roles.softmax:
                    chain[v] = w
                    stack.clear()
                    break
                # pass through infra ops (tmem_load on the mma warp etc.)
                stack.append(c)
    return chain


def _real_consumers(prob, producer):
    """Consumers of producer's buffer, seen through zero-latency plumbing.

    Yields (consumer, cumulative_distance) pairs; pass-through nodes
    (local_alloc / trans / tmem_alloc / expand / broadcast) forward the
    buffer instead of consuming it.
    """
    succs = _succs(prob)
    out = []
    stack = [(producer, 0)]
    seen = set()
    while stack:
        u, d = stack.pop()
        for e in succs.get(u, []):
            key = (e.dst, d + e.distance)
            if key in seen or e.dst == producer:
                continue
            seen.add(key)
            if prob.nodes[e.dst].op_kind in _PASSTHROUGH:
                stack.append((e.dst, d + e.distance))
            else:
                out.append((e.dst, d + e.distance))
    return out


def _copies(prob, sol, producer, ii):
    """Concurrent live copies of producer's value under the modulo schedule.

    Value born at t(p); consumer c reached with cumulative distance d reads
    it at t(c) + II*d in the producer's iteration frame and holds the buffer
    until its own completion (t + lat).  copies = floor(span / II) + 1.
    """
    t_p = _t(sol, producer)
    death = t_p
    consumers = []
    for c, d in _real_consumers(prob, producer):
        t_c = _t(sol, c) + ii * d + prob.lat.get(c, 0)
        consumers.append((c, d, t_c))
        death = max(death, t_c)
    span = max(0, death - t_p)
    return span // ii + 1, {"birth": t_p, "death": death, "span": span,
                            "consumers": consumers}


def _qk_tile_shape(prob, ddg_raw, qk_mmas):
    """(SUB_M, BLOCK_N) of the solved problem, read off the qk tmem_load."""
    if not ddg_raw:
        return None
    ops = ddg_raw.get("ops", {})
    for v in qk_mmas:
        for c, _ in _real_consumers(prob, v):
            n = prob.nodes[c]
            if n.op_kind != "ttng.tmem_load":
                continue
            for rt in ops.get(n.op_ref, {}).get("result_types", []):
                m = _TENSOR_SHAPE_RE.search(rt)
                if m:
                    return int(m.group(1)), int(m.group(2))
    return None


def bind(prob, sol, roles: RoleMap, *, quotas=None, ddg_raw=None) -> Binding:
    ii = sol["ii"]
    quotas = quotas or {"REGS_SOFTMAX": 152, "REGS_MMA": 24, "REGS_LOAD": 24}

    qk_mmas, pv_mmas = _classify_mmas(prob, roles, sol)
    chain = _chain_of_qk(prob, roles, sol, qk_mmas)
    audit: dict = {"ii": ii, "length": sol["length"], "copies": sol["copies"],
                   "qk_mmas": qk_mmas, "pv_mmas": pv_mmas,
                   "protocol_overrides": roles.protocol_overrides}

    # ── group count ──
    num_groups = len(roles.softmax)

    # ── chain order (ping-pong interleave) ──
    # Skeleton cid order == order the mma issuer emits the QK dots; bind cid 0
    # to the chain whose QK dot is issued first.
    qk_sorted = sorted(qk_mmas, key=lambda v: _t(sol, v))
    chain_order = []
    for v in qk_sorted:
        w = chain.get(v)
        if w is not None and w not in chain_order:
            chain_order.append(w)
    for w in roles.softmax:  # chains the walk failed to reach keep solver order
        if w not in chain_order:
            chain_order.append(w)
    audit["chain_order"] = chain_order
    if len(chain_order) >= 2:
        # informational: normalized stagger between the two chains' QK issues
        audit["pingpong_offset_norm"] = _t(sol, qk_sorted[1]) - _t(sol, qk_sorted[0])

    # ── issue-order / skew (bindable: program order of the mma role) ──
    # The frame order is always all-QK-then-all-PV here, but the *steady
    # state* interleave is what the modulo schedule pins down: PV of
    # iteration i issuing >= II after QK of iteration i means the static
    # loop must issue QK(i) before PV(i-skew).
    pv_sorted = sorted(pv_mmas, key=lambda v: _t(sol, v))
    skew = 0
    if qk_sorted and pv_sorted:
        skew = (_t(sol, pv_sorted[0]) - _t(sol, qk_sorted[0])) // ii
    steady = sorted(qk_mmas + pv_mmas, key=lambda v: _t(sol, v) % ii)
    audit["mma_issue_order"] = {
        "frame": [(v, _t(sol, v)) for v in sorted(qk_mmas + pv_mmas, key=lambda v: _t(sol, v))],
        "steady_state_mod_ii": [(v, _t(sol, v) % ii) for v in steady],
        "pv_skew_stages": skew,
    }
    if skew > 1:
        audit.setdefault("dropped", []).append(
            f"solver PV skew is {skew} stages; skeleton expresses at most 1 "
            "— clamped")
        skew = 1

    # ── ring depths from liveness ──
    tma_loads = [v for v, n in prob.nodes.items()
                 if sol["warp"][str(v)] == roles.load and n.pipeline == "TMA"]
    k_loads = [v for v in tma_loads
               if any(c in qk_mmas for c, _ in _real_consumers(prob, v))]
    v_loads = [v for v in tma_loads
               if any(c in pv_mmas for c, _ in _real_consumers(prob, v))]
    unmapped = [v for v in tma_loads if v not in k_loads and v not in v_loads]
    if unmapped:
        audit.setdefault("dropped", []).append(
            f"TMA loads {unmapped} reach no tensor-core consumer; "
            "excluded from KV depth")
    depth_detail = {}
    kv_depth = 0
    for name, loads in (("K", k_loads), ("V", v_loads)):
        c = 1
        for p in loads:
            ci, det = _copies(prob, sol, p, ii)
            depth_detail[f"{name}:{p}"] = det | {"copies": ci}
            c = max(c, ci)
        kv_depth += c
    qk_depth = 1
    for p in qk_mmas:
        ci, det = _copies(prob, sol, p, ii)
        depth_detail[f"QK:{p}"] = det | {"copies": ci}
        qk_depth = max(qk_depth, ci)
    audit["depth_detail"] = depth_detail

    # ── geometry ──
    # The schedule was solved (II, liveness, TMEM capacity) at the DDG's tile
    # geometry; realize it at that geometry.  BLOCK_M = chains * SUB_M.
    geom = {}
    shape = _qk_tile_shape(prob, ddg_raw, qk_mmas)
    if shape:
        sub_m, block_n = shape
        audit["geometry"] = {"sub_m": sub_m, "block_n": block_n,
                             "source": "qk tmem_load result type"}
        # Build constraint: this TLX build's TMEM allocator only pads
        # sub-128-row tiles for scales, so each chain's row block must be
        # >= 128.  Scaling the row block is schedule-preserving (each chain's
        # ops scale together; issue order, skew and ring depths are
        # unchanged); BLOCK_N — the per-iteration work unit II is measured
        # over — is kept as solved.
        split = sub_m
        if split < 128:
            split = 128
            audit["geometry"]["realized_split"] = split
            audit.setdefault("dropped", []).append(
                f"solver row block SUB_M={sub_m} < 128-row TMEM tile minimum "
                f"of this build; row block realized at {split} per chain "
                "(schedule-preserving scale-up)")
        geom = {"BLOCK_M": num_groups * split, "BLOCK_N": block_n}
    else:
        audit.setdefault("dropped", []).append(
            "qk tile shape not recoverable from DDG; skeleton default "
            "geometry kept")
    audit.setdefault("dropped", []).append(
        "solver cycle values are kept only as issue order / chain order / "
        "liveness depths; exact normalized offsets are not representable in "
        "a static-program skeleton")

    if skew >= qk_depth:
        audit.setdefault("dropped", []).append(
            f"PV skew {skew} needs QK depth {skew + 1}; liveness gave "
            f"{qk_depth} — depth raised to match")
        qk_depth = skew + 1

    params = {
        **geom,
        "NUM_MMA_GROUPS": num_groups,
        "NUM_BUFFERS_KV": kv_depth,
        "NUM_BUFFERS_QK": qk_depth,
        "MMA_PV_SKEW": skew,
        **quotas,
    }
    return Binding(params=params, audit=audit)


def bind_bwd(prob, sol, roles, *, quotas=None, ddg_raw=None) -> Binding:
    """Bind a skeleton-family bwd solution onto fa_bwd_dkdv_skc parameters.

    Bindable: Q ring depth (liveness of the Q descriptor load), tile geometry
    (from the qkT tmem_load shape), register quota table.  The 5-dot skewed
    pipeline and single-buffered aliased TMEM are protocol, not parameters;
    solver placements that disagree are audited as dropped.
    """
    ii = sol["ii"]
    quotas = quotas or {"REGS_MMA": 88, "REGS_REDUCTION": 88, "REGS_LOAD": 24}
    audit: dict = {"ii": ii, "length": sol["length"], "copies": sol["copies"],
                   "protocol_overrides": list(roles.protocol_overrides)}

    warp_of = {int(i): w for i, w in sol["warp"].items()}
    # The model's VARIABLELATENCY constraint pins the M/D row loads (tt.load)
    # on the VL warp; the skeleton realizes them as tl.load inside the compute
    # task.  Implementation-layer relocation, recorded like the R1 overrides.
    for v, n in prob.nodes.items():
        if warp_of.get(v) == roles.load and n.op_kind == "tt.load":
            audit["protocol_overrides"].append({
                "node": v, "op_kind": n.op_kind, "solver_warp": roles.load,
                "rule": "skeleton loads M/D via tl.load in the compute task "
                        "(model keeps them on the VL warp)"})
    mma_nodes = [v for v, n in prob.nodes.items()
                 if n.op_kind == MMA_KIND and warp_of.get(v) == roles.mma]

    # Q ring depth: liveness of the Q tile (descriptor load consumed by the
    # QK dot in-frame and by the dK dot one frame later under the skew).
    tma_loads = [v for v, n in prob.nodes.items()
                 if warp_of.get(v) == roles.load and n.op_kind == "tt.descriptor_load"]
    q_depth = None
    depth_detail = {}
    for p in tma_loads:
        cons = [c for c, _ in _real_consumers(prob, p)]
        if any(c in mma_nodes for c in cons):
            copies, det = _copies(prob, sol, p, ii)
            n_mma = sum(1 for c in cons if c in mma_nodes)
            depth_detail[f"load:{p}"] = det | {"copies": copies,
                                               "mma_consumers": n_mma}
            # Q feeds two dots (QK and dK); dO feeds two as well but its ring
            # is fixed at 1 by the protocol — take the deeper-lived load as Q.
            if q_depth is None or copies > q_depth:
                q_depth = copies
    audit["depth_detail"] = depth_detail
    if q_depth is None:
        audit.setdefault("dropped", []).append(
            "no TMA load reaches a tensor-core consumer; Q ring depth kept "
            "at skeleton default")

    # Geometry.  The fixture's tmem_loads are sub-tile reads (qkT is read in
    # 64-column halves), so their column counts are not the tile geometry.
    # Rows are not split: BLOCK_M = rows of the Q descriptor load's tile,
    # BLOCK_N = rows of the qkT read.
    geom = {}
    ops = (ddg_raw or {}).get("ops", {})

    def _rows(op_ref):
        for rt in ops.get(op_ref, {}).get("result_types", []):
            m = re.search(r"<(\d+)x(\d+)x", rt)
            if m:
                return int(m.group(1))
        return None

    block_m = block_n = None
    for p in tma_loads:
        if any(c in mma_nodes for c, _ in _real_consumers(prob, p)):
            r = _rows(prob.nodes[p].op_ref)
            if r:
                block_m = max(block_m or 0, r)
    qkT_loads = [c for v in mma_nodes for c, _ in _real_consumers(prob, v)
                 if prob.nodes[c].op_kind == "ttng.tmem_load"]
    for c in qkT_loads:
        r = _rows(prob.nodes[c].op_ref)
        if r:
            block_n = max(block_n or 0, r)
    if block_m and block_n:
        audit["geometry"] = {"block_m": block_m, "block_n": block_n,
                             "source": "Q descriptor rows / qkT read rows"}
        # Build constraint (same as fwd): the dq TMEM tile is BLOCK_M rows
        # and this build only pads sub-128-row TMEM tiles for scales.
        # Scaling the M block is schedule-preserving (per-iteration ops scale
        # together; issue order and ring structure unchanged).
        realized_bm = block_m
        if realized_bm < 128:
            realized_bm = 128
            audit["geometry"]["realized_block_m"] = realized_bm
            audit.setdefault("dropped", []).append(
                f"solver M block {block_m} < 128-row TMEM tile minimum of "
                f"this build; realized at {realized_bm} "
                "(schedule-preserving scale-up)")
        geom = {"BLOCK_M": realized_bm, "BLOCK_N": block_n}
    else:
        audit.setdefault("dropped", []).append(
            "tile geometry not recoverable from DDG; skeleton defaults kept")

    # Skew audit: the dK/dQ dots must lag the QK dot by >= 1 II frame (the
    # skeleton's constructed 1-stage skew); anything else is unexpressible.
    qk_dot = min(mma_nodes, key=lambda v: _t(sol, v)) if mma_nodes else None
    if qk_dot is not None:
        lags = {v: (_t(sol, v) - _t(sol, qk_dot)) // ii for v in mma_nodes}
        audit["mma_frame_lags"] = lags

    params = {**geom, **quotas}
    if q_depth is not None:
        bound_q = max(2, q_depth)
        if q_depth < 2:
            audit.setdefault("dropped", []).append(
                f"liveness Q copies={q_depth} but the skewed pipeline needs "
                "the ring >= 2 (dK reads Q one frame late) — raised to 2")
        # Build constraint: the model's SMEM budget has no line item for the
        # skeleton's barrier/padding overhead (~17 KB measured), so its
        # liveness can ask for a ring the kernel cannot allocate.  Clamp to
        # the skeleton's real SMEM accounting (Q=3 is a measured OOM at
        # 128x128xHD128).
        bm = geom.get("BLOCK_M", 128)
        bn = geom.get("BLOCK_N", 128)
        hd = 128
        fixed = 2 * (2 * bn * hd + bm * hd + bn * bm)  # K+V, dO, ds (fp16)
        overhead = 17072
        q_bytes = 2 * bm * hd
        max_q = (232448 - fixed - overhead) // q_bytes
        if bound_q > max_q:
            audit.setdefault("dropped", []).append(
                f"liveness Q copies={q_depth} exceeds the skeleton's SMEM "
                f"accounting (max ring {max_q} after barrier/padding "
                "overhead the model does not price) — clamped")
            bound_q = max_q
        params["NUM_BUFFERS_Q"] = bound_q
    return Binding(params=params, audit=audit)
