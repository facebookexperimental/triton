"""Write a solver solution back into a schedule_graph.json for sched2tlx.

Rewrites a BASELINE schedule_graph.json (dumped by the modulo-scheduling pass
from the same TTGIR as the ddg, so node ids match) with the solver's modulo
schedule + warp assignment:

  * per-node schedule.cycle = M*(v) (absolute steady-state position),
    schedule.stage = cycle // II, schedule.cluster = dense rank of cycle
    within the stage (the emitter's intra-stage ordering key);
  * per-node warp_group, with solver warp ids densely remapped onto a
    rebuilt warp_groups[] list (pipelines = union of member pipelines,
    num_warps = max member min_warps snapped to {1,2,4,8});
  * cross_wg_barriers producer_wg/consumer_wg remapped, depth refreshed
    from the paired buffer's count (loop-carry vs forward direction is
    re-derived by the emitter from the new cycles);
  * buffers[].count = ceil(live-span / II) from the new producer/consumer
    cycles (min 1, capped at baseline count + 2); barrier buffers mirror
    their paired data buffer;
  * NEW cross-WG register channels synthesized for intra-iteration register
    dataflow the solver's partition routes across warp groups where the
    baseline had none (see _synthesize_register_channels). No-op when the
    partition introduces no new cross-WG register flows.

Everything else (ops table, kernel signature, buffer ids/shapes) is kept
untouched.
"""

import json
import re
import sys
from pathlib import Path

_SNAP_WARPS = (1, 2, 4, 8)
_WIDE_PIPELINES = ("CUDA", "SFU")

_MMA_KINDS = ("ttng.tc_gen5_mma", "ttng.tc_gen5_mma_scaled",
              "ttng.warp_group_dot")
# Producers whose cross-WG flow is already buffer-mediated (SMEM/TMEM allocs
# and stores, TMA loads, MMAs) — the baseline barrier machinery covers them.
_NON_CHANNEL_PRODUCERS = _MMA_KINDS + (
    "tt.descriptor_load", "ttg.local_alloc", "ttng.tmem_alloc",
    "ttng.tmem_store", "ttg.local_store", "tt.descriptor_store",
    "tt.descriptor_reduce",
)
# Consumer node kinds whose emission never register-rematerializes operand
# chains (operands resolve to buffers/descriptors) — skip walking them.
_NON_WALK_CONSUMERS = _MMA_KINDS + ("tt.descriptor_load",)
# Consumer node kinds where the emitter's SemIR consumer block does NOT bind
# the producer value to a loaded variable (MMAs recycle via mBarriers,
# memdesc_trans is an address calc, allocs never emit, descriptor_store
# reads the channel SMEM directly).
_NON_BINDING_CONSUMERS = _MMA_KINDS + (
    "ttg.memdesc_trans", "ttg.local_alloc", "ttng.tmem_alloc",
    "tt.descriptor_store", "scf.yield",
)
# Node kinds with observable effects — seeds for per-WG liveness.
_SIDE_EFFECT_KINDS = _MMA_KINDS + (
    "ttng.tmem_store", "ttg.local_store", "tt.store",
    "tt.descriptor_store", "tt.descriptor_reduce", "tt.descriptor_load",
)
# A synthesized channel buffer never exceeds this footprint (SMEM budget):
# large staging tiles stay single-slot instead of multi-buffering.
_MAX_CHANNEL_BYTES = 16384


def _sched2tlx_modules():
    root = Path(__file__).resolve().parents[2] / "sched2tlx"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from sched2tlx import emitter, schedule_graph, semaphore_ir
    return schedule_graph, semaphore_ir, emitter


def _target_loop(data):
    loops = [L for L in data["loops"] if not L.get("is_outer", False)]
    if len(loops) != 1:
        raise ValueError(
            f"expected exactly one non-outer loop, found {len(loops)}")
    return loops[0]


def _snap_num_warps(n):
    for w in _SNAP_WARPS:
        if w >= n:
            return w
    return _SNAP_WARPS[-1]


def _node_min_warps(node):
    mw = node.get("min_warps")
    if mw is not None:
        return mw
    return 4 if node.get("pipeline") in _WIDE_PIPELINES else 1


def _assign_clusters(nodes):
    """schedule.cluster = dense rank of schedule.cycle within the stage,
    computed globally across warp groups (ties share a rank) — mirrors the
    modulo pass's dump and satisfies the emitter's (stage, cluster, id) sort."""
    by_stage = {}
    for n in nodes:
        sched = n.get("schedule")
        if sched is None or sched.get("cycle", -1) < 0:
            continue
        by_stage.setdefault(sched["stage"], []).append(n)
    for members in by_stage.values():
        rank = {c: i for i, c in
                enumerate(sorted({n["schedule"]["cycle"] for n in members}))}
        for n in members:
            n["schedule"]["cluster"] = rank[n["schedule"]["cycle"]]


def _recompute_buffers(sl, ii, ops):
    nodes = {n["id"]: n for n in sl["graph"]["nodes"]}
    succ = {}
    for e in sl["graph"]["edges"]:
        if e["kind"] == "data" and e["distance"] == 0:
            succ.setdefault(e["src"], []).append(e["dst"])
    cbs = sl["graph"].get("cross_wg_barriers", [])
    buffers = sl["buffers"]
    for b in buffers:
        if b["kind"] == "barrier":
            continue
        starts, ends = [], []
        direct = [n for n in nodes.values()
                  if b["id"] in (n.get("consumes_buffers") or [])]
        for n in nodes.values():
            if n.get("produces_buffer") == b["id"]:
                starts.append(n["schedule"]["cycle"])
        # Real HW readers: direct consumers plus, through zero-latency
        # wrapper nodes (memdesc_trans etc.), their downstream consumers.
        seen = {n["id"] for n in direct}
        work = list(direct)
        while work:
            n = work.pop()
            ends.append(n["schedule"]["cycle"] + n["latency"])
            if n["latency"] == 0:
                for s in succ.get(n["id"], []):
                    if s not in seen:
                        seen.add(s)
                        work.append(nodes[s])
        # Synthesized channel buffers: producer/consumer via cross_wg_barriers.
        # Only FORWARD pairings carry data through the buffer. A buffer paired
        # solely by backward (loop-carry) entries is a release-signal ghost:
        # its semaphore protocol (single slot-0 pre-arrive, per-iter wait +
        # arrive on slot _it%depth) only works at depth 1 — inflating its
        # count makes iteration 1 wait a slot nobody has arrived (deadlock).
        for cb in cbs:
            if cb.get("paired_buffer_id") != b["id"]:
                continue
            p = nodes.get(cb["producer_node"])
            c = nodes.get(cb["consumer_node"])
            if p is None or c is None:
                continue
            pc, cc = p["schedule"]["cycle"], c["schedule"]["cycle"]
            if pc < 0 or cc < 0 or pc > cc:
                continue  # backward (loop-carry release) — no data flow
            starts.append(pc)
            ends.append(cc + c["latency"])
        if not starts and not ends:
            continue  # no schedule refs — keep baseline count/lifetime
        live_start = min(starts) if starts else b["live_start"]
        live_end = max(ends) if ends else live_start
        span = live_end - live_start
        count = max(1, min(-(-span // ii), b["count"] + 2))
        # TMEM bridge buffers (tmem_alloc wrapping a register value) are
        # emitted at a fixed slot with a depth-1 handshake — inflating their
        # count only wastes TMEM budget. Keep the baseline depth.
        def_op = (ops or {}).get(b.get("def_op") or "") or {}
        if (b["kind"] == "tmem" and def_op.get("kind") == "ttng.tmem_alloc"
                and def_op.get("operands")):
            count = b["count"]
        b.update(live_start=live_start, live_end=live_end, count=count,
                 total_bytes=b["size_bytes"] * count)
    by_id = {b["id"]: b for b in buffers}
    for b in buffers:
        if b["kind"] != "barrier":
            continue
        paired = by_id.get(b.get("paired_buffer_id"))
        if paired is None:
            continue
        b.update(live_start=paired["live_start"], live_end=paired["live_end"],
                 count=paired["count"],
                 total_bytes=b["size_bytes"] * paired["count"])


def _parse_tensor_type(rt):
    """'tensor<64x64xf32, #layout>' → (shape list, element_bits) or None."""
    m = re.match(r"tensor<((?:\d+x)*\d+)x([a-z]+)(\d+)[,>]", rt or "")
    if not m:
        return None
    return [int(d) for d in m.group(1).split("x")], int(m.group(3))


def _operand_refs(op):
    return (list(op.get("operands", [])) + list(op.get("then_yields", []))
            + list(op.get("else_yields", [])))


def _emit_key(n):
    sched = n.get("schedule") or {}
    return (sched.get("stage", -1), sched.get("cluster", -1), n["id"])


def _synthesize_register_channels(data, loop):
    """Synthesize cross-WG register channels for the solver's partition.

    The solver may place a register value's producer (a CUDA/SFU node) and
    its consumers in different warp groups where the baseline partition kept
    them together. The emitter re-materializes such a cross-WG operand chain
    inline in the consumer group; when the chain bottoms out in a loop-carried
    iter_arg the consumer group does not keep, the render degrades to an
    `<inner_iter_arg_*>` placeholder. Mirror the schedule pass's synthesized
    staging-buffer routing (the baseline's def_op-None channel buffers): add
    an SMEM buffers[] entry sized from the producer's register result type
    plus mbarrier cross_wg_barriers[] entries pairing the producer with the
    first needy consumer per group. The emitter's SemIR path lowers each into
    a full/empty semaphore whose consumer block does wait + local_load and
    binds the producer's op to the loaded variable — flipping the render from
    rematerialize to channel-load. Semaphore depth = data-buffer count, as in
    the baseline's synthesized entries.

    Channels are synthesized where the consumer-side render would otherwise:
      * reach an iter_arg the consumer group does not keep (the placeholder
        case) — cut at the innermost CUDA/SFU register producer whose
        rematerialization cannot be repaired deeper; or
      * re-read TMEM through a cross-WG `ttng.tmem_load` with no completion
        handshake in the consumer's group (a race) — only for values that
        are live in the consumer group.

    No-op when the partition introduces no new cross-WG register flows.
    """
    sl = loop["schedule_loop"]
    ii = sl["II"]
    loop_id = sl["id"]
    ops = data["ops"]
    nodes = sl["graph"]["nodes"]
    cwbs = sl["graph"].setdefault("cross_wg_barriers", [])
    buffers = sl["buffers"]
    by_id = {n["id"]: n for n in nodes}
    node_by_opref = {}
    for n in nodes:
        if n.get("op_ref") and n["op_ref"] not in node_by_opref:
            node_by_opref[n["op_ref"]] = n

    def wg_of(n):
        return n.get("warp_group", -1)

    real_wgs = sorted({wg_of(n) for n in nodes if wg_of(n) >= 0})

    # VALUE iter_args (mirror of the emitter's _loop_iter_args): filter out
    # memory-handle / async-token iter_args — those thread memory state, not
    # register values, and the renderer never materializes them.
    bounds = (sl.get("lower_bound"), sl.get("upper_bound"), sl.get("step"))
    for_op = next(
        (op for op in ops.values()
         if op.get("kind") == "scf.for" and len(op.get("operands", [])) >= 3
         and tuple(op["operands"][:3]) == bounds),
        None)
    yield_op = next(
        (op for op in ops.values()
         if op.get("kind") == "scf.yield"
         and op.get("scope") == f"loop:{loop_id}"),
        None)
    inits = (for_op or {}).get("operands", [])[3:]
    yield_refs = list((yield_op or {}).get("operands", []))
    value_idxs = set()
    for i in range(min(len(inits), len(yield_refs))):
        init = inits[i]
        if isinstance(init, dict) and "op" in init:
            init_op = ops.get(init["op"])
            if init_op is not None and init_op.get("kind") in (
                    "ttng.tmem_alloc", "ttg.local_alloc", "ttng.tmem_store",
                    "ub.poison"):
                continue
        if isinstance(init, dict) and "const" in init:
            t = init.get("type", "")
            if "memdesc" in t or "async.token" in t or "tensor_memory" in t:
                continue
        value_idxs.add(i)

    # Value iter_arg idxs each WG keeps: idxs directly referenced by the WG's
    # ops (mirror of the emitter's per-WG iter_arg trim).
    kept = {B: set() for B in real_wgs}
    for n in nodes:
        B = wg_of(n)
        if B < 0 or not n.get("op_ref"):
            continue
        op = ops.get(n["op_ref"])
        for ref in _operand_refs(op or {}):
            if isinstance(ref, dict) and "iter_arg" in ref:
                ia = ref["iter_arg"]
                if ia.get("loop") == loop_id and ia.get("idx") in value_idxs:
                    kept[B].add(ia.get("idx"))

    def _is_forward(cb):
        p, c = by_id.get(cb["producer_node"]), by_id.get(cb["consumer_node"])
        if p is None or c is None:
            return False
        return (p["schedule"]["cycle"] >= 0 and c["schedule"]["cycle"] >= 0
                and p["schedule"]["cycle"] <= c["schedule"]["cycle"])

    # op_refs whose value resolves without rematerialization in WG B:
    # nodes owned by B (or emitter infra, wg < 0), plus producers of
    # existing forward data channels whose consumer block binds in B.
    bound = {B: set() for B in real_wgs}
    for n in nodes:
        if n.get("op_ref") and wg_of(n) < 0:
            for B in real_wgs:
                bound[B].add(n["op_ref"])
        elif n.get("op_ref") and wg_of(n) in bound:
            bound[wg_of(n)].add(n["op_ref"])
    bufs_by_id = {b["id"]: b for b in buffers}
    for cb in cwbs:
        buf = bufs_by_id.get(cb.get("paired_buffer_id"))
        c = by_id.get(cb["consumer_node"])
        p = by_id.get(cb["producer_node"])
        if (buf is None or buf["kind"] == "barrier" or c is None or p is None
                or not _is_forward(cb)
                or c["op_kind"] in _NON_BINDING_CONSUMERS
                or wg_of(c) not in bound or not p.get("op_ref")):
            continue
        bound[wg_of(c)].add(p["op_ref"])

    # Per-WG liveness (only gates the tmem_load race rule): a node is live
    # when its value reaches a side effect, a kept iter_arg yield, an
    # existing data-channel store, or a TMEM bridge in its own group.
    chan_producer_refs = {
        by_id[cb["producer_node"]].get("op_ref")
        for cb in cwbs
        if cb.get("paired_buffer_id") in bufs_by_id
        and bufs_by_id[cb["paired_buffer_id"]]["kind"] != "barrier"
        and cb["producer_node"] in by_id and _is_forward(cb)
    }
    bridge_value_refs = set()
    for op in ops.values():
        if op.get("kind") == "ttng.tmem_alloc":
            for ref in op.get("operands", []):
                if isinstance(ref, dict) and "op" in ref:
                    bridge_value_refs.add(ref["op"])
    def _live_set(B):
        owned = [n for n in nodes if wg_of(n) == B and n.get("op_ref")]
        live = set()
        for n in owned:
            if (n["op_kind"] in _SIDE_EFFECT_KINDS
                    or n["op_ref"] in chan_producer_refs
                    or n["op_ref"] in bridge_value_refs):
                live.add(n["id"])
        for idx in kept[B]:
            if idx < len(yield_refs) and isinstance(yield_refs[idx], dict):
                ref = yield_refs[idx]
                if "op" in ref:
                    yn = node_by_opref.get(ref["op"])
                    if yn is not None and wg_of(yn) == B:
                        live.add(yn["id"])
        changed = True
        while changed:
            changed = False
            for n in owned:
                if n["id"] not in live:
                    continue
                op = ops.get(n["op_ref"]) or {}
                for ref in _operand_refs(op):
                    if isinstance(ref, dict) and "op" in ref:
                        m = node_by_opref.get(ref["op"])
                        if (m is not None and wg_of(m) == B
                                and m["id"] not in live):
                            live.add(m["id"])
                            changed = True
        return live

    def _eligible(p):
        """A producer whose register value can ferry through an SMEM channel:
        a CUDA/SFU node with a register-tensor result, not buffer-mediated."""
        if p.get("pipeline") not in _WIDE_PIPELINES:
            return False
        if p["op_kind"] in _NON_CHANNEL_PRODUCERS:
            return False
        if p.get("produces_buffer") is not None:
            return False
        op = ops.get(p.get("op_ref") or "")
        if not op or not op.get("result_types"):
            return False
        return _parse_tensor_type(op["result_types"][0]) is not None

    def _tmem_read_safe(p, B):
        """True when re-rendering cross-WG tmem_load `p` inside WG B reads
        TMEM that B already holds a completion handshake for: every writer of
        the source TMEM alloc either lives in B or produces an existing
        cross_wg_barrier consumed in B."""
        op = ops.get(p.get("op_ref") or "") or {}
        srcs = [r["op"] for r in op.get("operands", [])
                if isinstance(r, dict) and "op" in r]
        if not srcs:
            return True
        alloc_id = srcs[0]
        writers = []
        for n in nodes:
            wop = ops.get(n.get("op_ref") or "")
            if not wop:
                continue
            refs = wop.get("operands", [])
            if n["op_kind"] in _MMA_KINDS and len(refs) >= 3:
                acc = refs[2]
                if isinstance(acc, dict) and acc.get("op") == alloc_id:
                    writers.append(n)
            elif n["op_kind"] == "ttng.tmem_store" and refs:
                dst = refs[0]
                if isinstance(dst, dict) and dst.get("op") == alloc_id:
                    writers.append(n)
        for w in writers:
            if wg_of(w) == B:
                continue
            if any(cb["producer_node"] == w["id"]
                   and by_id.get(cb["consumer_node"]) is not None
                   and wg_of(by_id[cb["consumer_node"]]) == B
                   for cb in cwbs):
                continue
            return False
        return True

    next_buf_id = max((b["id"] for b in buffers), default=-1) + 1
    new_buf_by_producer = {}   # producer node id -> buffer dict
    new_entries = []           # synthesized cross_wg_barriers dicts
    uses_by_producer = {}      # producer node id -> [consumer node dicts]

    def _channel(p, consumer, B):
        nonlocal next_buf_id
        buf = new_buf_by_producer.get(p["id"])
        if buf is None:
            op = ops[p["op_ref"]]
            shape, bits = _parse_tensor_type(op["result_types"][0])
            size = (bits // 8)
            for d in shape:
                size *= d
            buf = {
                "id": next_buf_id,
                "kind": "smem",
                "shape": shape,
                "element_bits": bits,
                "count": 1,
                "size_bytes": size,
                "total_bytes": size,
                "merge_group_id": None,
                "paired_buffer_id": None,
                "live_start": p["schedule"]["cycle"],
                "live_end": p["schedule"]["cycle"],
                "def_op": None,
            }
            next_buf_id += 1
            new_buf_by_producer[p["id"]] = buf
        if not any(e["producer_node"] == p["id"] and e["consumer_wg"] == B
                   for e in new_entries):
            new_entries.append({
                "producer_node": p["id"],
                "consumer_node": consumer["id"],
                "producer_wg": wg_of(p),
                "consumer_wg": B,
                "kind": "mbarrier",
                "depth": 1,
                "paired_buffer_id": buf["id"],
                "expect_bytes": buf["size_bytes"],
            })
        uses_by_producer.setdefault(p["id"], []).append(consumer)
        bound[B].add(p["op_ref"])

    def _visit(ref, B, consumer, consumer_live, stack):
        """True iff `ref` renders in WG B without placeholders, synthesizing
        channels along the way. False bubbles up so an eligible ancestor can
        cut the chain instead."""
        if not isinstance(ref, dict):
            return True
        if "iter_arg" in ref:
            ia = ref["iter_arg"]
            if ia.get("loop") != loop_id or ia.get("idx") not in value_idxs:
                return True  # other loop, or token/handle (never rendered)
            return ia.get("idx") in kept[B]
        if "op" not in ref:
            return True  # kernel arg / iv / const
        oid = ref["op"]
        if oid in bound[B] or oid in stack:
            return True
        op = ops.get(oid)
        if op is None:
            return True
        rts = op.get("result_types", [])
        ridx = ref.get("result", 0)
        if ridx < len(rts) and "async.token" in rts[ridx]:
            return True  # token result — the renderer never materializes it
        p = node_by_opref.get(oid)
        stack.add(oid)
        try:
            children = _operand_refs(op)
            if p is None or wg_of(p) < 0 or wg_of(p) == B:
                # Function-scope op, emitter infra, or same group: renders
                # locally; its own operands must still resolve.
                ok = True
                for c in children:
                    ok = _visit(c, B, consumer, consumer_live, stack) and ok
                return ok
            if consumer["op_kind"] == "tt.descriptor_store":
                # The emitter feeds a descriptor_store consumer straight from
                # the channel SMEM — only the store's direct value operand may
                # be channeled.
                cons_operands = (ops.get(consumer.get("op_ref") or "")
                                 or {}).get("operands", [])
                store_ok = len(cons_operands) >= 2 and ref is cons_operands[1]
            else:
                store_ok = True
            can_channel = _eligible(p) and store_ok
            # A producer directly consuming an unkept iter_arg cannot be
            # repaired deeper — cut here.
            if can_channel and any(
                    isinstance(c, dict) and "iter_arg" in c
                    and c["iter_arg"].get("loop") == loop_id
                    and c["iter_arg"].get("idx") in value_idxs
                    and c["iter_arg"].get("idx") not in kept[B]
                    for c in children):
                _channel(p, consumer, B)
                return True
            ok = True
            for c in children:
                ok = _visit(c, B, consumer, consumer_live, stack) and ok
            if ok:
                if (p["op_kind"] == "ttng.tmem_load" and consumer_live
                        and can_channel and not _tmem_read_safe(p, B)):
                    _channel(p, consumer, B)
                return True
            if can_channel:
                _channel(p, consumer, B)
                return True
            return False
        finally:
            stack.discard(oid)

    for B in real_wgs:
        live = _live_set(B)
        owned = sorted(
            (n for n in nodes if wg_of(n) == B and n.get("op_ref")),
            key=_emit_key)
        last = owned[-1] if owned else None
        for n in owned:
            if (n["op_kind"] in _NON_WALK_CONSUMERS
                    or n["op_kind"] in ("ttg.local_alloc", "ttng.tmem_alloc",
                                        "scf.yield")
                    or n.get("schedule", {}).get("cycle", -1) < 0):
                continue
            op = ops.get(n["op_ref"]) or {}
            for ref in _operand_refs(op):
                _visit(ref, B, n, n["id"] in live, set())
        # Kept iter_args are reassigned from their yield refs at end of body;
        # those renders must resolve too. Bind at the last emitted node so
        # the channel load precedes the yield reassignment.
        if last is not None:
            for idx in sorted(kept[B]):
                if idx < len(yield_refs):
                    _visit(yield_refs[idx], B, last, True, set())

    if not new_entries:
        return

    # Ring depth from the value's live span (baseline convention:
    # ceil(span / II), min 1), capped so a synthesized channel never exceeds
    # the SMEM footprint budget.
    for pid, buf in new_buf_by_producer.items():
        p = by_id[pid]
        start = p["schedule"]["cycle"]
        end = start
        for c in uses_by_producer.get(pid, []):
            end = max(end, c["schedule"]["cycle"] + c.get("latency", 0))
        count = max(1, -(-(end - start) // ii))
        while count > 1 and buf["size_bytes"] * count > _MAX_CHANNEL_BYTES:
            count -= 1
        buf.update(live_start=start, live_end=end, count=count,
                   total_bytes=buf["size_bytes"] * count)
        buffers.append(buf)
    for e in new_entries:
        e["depth"] = new_buf_by_producer[e["producer_node"]]["count"]
    cwbs.extend(new_entries)


def rewrite_schedule_graph(baseline_path, out_path, *, ii, cycles, warp,
                           length):
    if ii < 1:
        raise ValueError(f"ii must be >= 1, got {ii}")
    with open(baseline_path) as f:
        data = json.load(f)
    loop = _target_loop(data)
    sl = loop["schedule_loop"]
    nodes = sl["graph"]["nodes"]
    by_id = {n["id"]: n for n in nodes}

    unknown = (set(cycles) | set(warp)) - set(by_id)
    if unknown:
        raise ValueError(f"solution references unknown node ids: "
                         f"{sorted(unknown)}")
    # Zero-latency ops may legally sit at cycle == length (COMPLETION allows
    # t <= T - lat), so the bound is inclusive.
    bad = [(v, c) for v, c in cycles.items() if not 0 <= c <= length]
    if bad:
        raise ValueError(f"cycles outside [0, {length}]: {sorted(bad)}")

    for nid, c in cycles.items():
        by_id[nid]["schedule"]["cycle"] = c
    for n in nodes:
        sched = n.get("schedule")
        if sched is not None and sched.get("cycle", -1) >= 0:
            sched["stage"] = sched["cycle"] // ii
    _assign_clusters(nodes)

    # Remap solver warp ids onto a dense warp_groups[] rebuild. Nodes with a
    # negative baseline warp_group are emitter infra (replicated/folded at
    # emission sites) and keep it; nodes the solver did not assign keep their
    # baseline group as the solver id.
    solver_wg = {}
    for n in nodes:
        if n.get("warp_group", -1) >= 0:
            solver_wg[n["id"]] = warp.get(n["id"], n["warp_group"])
    remap = {w: i for i, w in enumerate(sorted(set(solver_wg.values())))}
    for nid, w in solver_wg.items():
        by_id[nid]["warp_group"] = remap[w]

    groups = []
    for w, new_id in sorted(remap.items(), key=lambda kv: kv[1]):
        members = [by_id[nid] for nid, sw in solver_wg.items() if sw == w]
        groups.append({
            "id": new_id,
            "num_warps": _snap_num_warps(
                max(_node_min_warps(n) for n in members)),
            "pipelines": sorted({n["pipeline"] for n in members}),
        })
    loop["warp_groups"] = groups

    sl["II"] = ii
    stages = [n["schedule"]["stage"] for n in nodes
              if n.get("schedule", {}).get("cycle", -1) >= 0]
    sl["max_stage"] = max(stages, default=0)

    _recompute_buffers(sl, ii, data.get("ops"))

    bufs = {b["id"]: b for b in sl["buffers"]}
    for cb in sl["graph"].get("cross_wg_barriers", []):
        p = by_id.get(cb["producer_node"])
        c = by_id.get(cb["consumer_node"])
        if p is not None:
            cb["producer_wg"] = p["warp_group"]
        if c is not None:
            cb["consumer_wg"] = c["warp_group"]
        paired = bufs.get(cb.get("paired_buffer_id"))
        if paired is not None and paired["kind"] != "barrier":
            cb["depth"] = paired["count"]

    _synthesize_register_channels(data, loop)

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def validate(out_path):
    """Offline emit-check of a written schedule_graph.json. Returns a list of
    problem strings; empty means sched2tlx can load, derive semaphores for,
    and emit the graph."""
    sg, si, em = _sched2tlx_modules()
    try:
        g = sg.load_graph(out_path)
    except Exception as e:
        return [f"load_graph failed: {e!r}"]

    problems = []
    for loop in g.loops:
        sched = loop.schedule
        tag = f"loop{loop.loop_id}"
        wg_ids = {w.id for w in loop.warp_groups}
        if sched.II < 1:
            problems.append(f"{tag}: II={sched.II} < 1")
            continue
        per_group = {}
        for n in sched.nodes:
            if n.warp_group >= 0 and n.warp_group not in wg_ids:
                problems.append(
                    f"{tag} N{n.id}: warp_group {n.warp_group} not in "
                    f"warp_groups {sorted(wg_ids)}")
            if n.schedule_cycle < 0:
                continue
            if n.schedule_cycle // sched.II != n.schedule_stage:
                problems.append(
                    f"{tag} N{n.id}: stage {n.schedule_stage} != "
                    f"cycle {n.schedule_cycle} // II {sched.II}")
            if n.schedule_stage > sched.max_stage:
                problems.append(
                    f"{tag} N{n.id}: stage {n.schedule_stage} > "
                    f"max_stage {sched.max_stage}")
            per_group.setdefault(
                (n.warp_group, n.schedule_stage), []).append(n)
        for (wg, stage), members in per_group.items():
            for a in members:
                for b in members:
                    if a.schedule_cycle < b.schedule_cycle and not (
                            a.schedule_cluster < b.schedule_cluster):
                        problems.append(
                            f"{tag} wg{wg} stage{stage}: cluster order "
                            f"contradicts cycle order (N{a.id} vs N{b.id})")
        bufs = {b.id: b for b in sched.buffers}
        for b in sched.buffers:
            if b.count < 1:
                problems.append(f"{tag} buf{b.id}: count {b.count} < 1")
            if b.kind == "barrier" and b.paired_buffer_id in bufs:
                paired = bufs[b.paired_buffer_id]
                if b.count != paired.count:
                    problems.append(
                        f"{tag} buf{b.id}: barrier count {b.count} != "
                        f"paired buf{paired.id} count {paired.count}")
        node_wg = {n.id: n.warp_group for n in sched.nodes}
        for cb in sched.cross_wg_barriers:
            for role, nid, wg in (("producer", cb.producer_node, cb.producer_wg),
                                  ("consumer", cb.consumer_node, cb.consumer_wg)):
                if nid not in node_wg:
                    problems.append(
                        f"{tag}: cross_wg_barrier {role} node {nid} missing")
                elif node_wg[nid] != wg:
                    problems.append(
                        f"{tag}: cross_wg_barrier {role}_wg {wg} != "
                        f"N{nid}'s warp_group {node_wg[nid]}")
    if problems:
        return problems

    wg_of_node = {(L.loop_id, n.id): n.warp_group
                  for L in g.loops for n in L.schedule.nodes}
    try:
        si.build_sem_set_for_graph(g, wg_of_node=wg_of_node)
    except Exception as e:
        problems.append(f"semaphore derivation failed: {e!r}")
    try:
        em.emit(sg.load_graph(out_path))  # emit mutates — use a fresh load
    except Exception as e:
        problems.append(f"emit failed: {e!r}")
    return problems
