"""Rau-style iterative modulo scheduling used as a CBC warm start.

The ILP in modulo_ilp.py is the authority (Algorithm 1's
OPTIMAL-MODULO-SCHEDULE); this heuristic only supplies a feasible MIP start
so CBC's "feasible at this II" verdict does not depend on it discovering a
saturated modulo packing from scratch.
"""

from .ddg import Problem


def _heights(prob: Problem) -> dict[int, int]:
    height = {v: prob.lat[v] for v in prob.nodes}
    for _ in range(len(prob.nodes)):
        changed = False
        for e in prob.edges:
            if e.distance:
                continue
            d = prob.edge_latency(e)
            h = d + height[e.dst]
            if h > height[e.src]:
                height[e.src] = h
                changed = True
        if not changed:
            break
    return height


def greedy_modulo(prob: Problem, ii: int,
                  max_ops_scheduled: int | None = None) -> dict[int, int] | None:
    """Iterative modulo scheduling (Rau 1994, simplified).  Returns cycles
    M(v) respecting dependences and the modulo reservation table, or None."""
    ids = list(prob.nodes)
    height = _heights(prob)
    order = sorted(ids, key=lambda v: -height[v])
    budget = max_ops_scheduled or 30 * len(ids)
    placed: dict[int, int] = {}
    # rrt[(functional unit, residue)] -> {occupant op: integer demand}
    rrt: dict[tuple[str, int], dict[int, int]] = {}
    in_edges: dict[int, list] = {}
    for e in prob.edges:
        in_edges.setdefault(e.dst, []).append(e)
    out_edges: dict[int, list] = {}
    for e in prob.edges:
        out_edges.setdefault(e.src, []).append(e)

    def earliest(v: int) -> int:
        t = 0
        for e in in_edges.get(v, ()):  # noqa: B023
            if e.src in placed:
                d = prob.edge_latency(e)
                t = max(t, placed[e.src] + d - e.distance * ii)
        return t

    def footprint(v: int, start: int) -> dict[tuple[str, int], int]:
        result = {}
        for functional_unit, cycle, amount in prob.reservations(v):
            key = (functional_unit, (start + cycle) % ii)
            result[key] = result.get(key, 0) + amount
        return result

    def fits(candidate: dict[tuple[str, int], int]) -> bool:
        return all(
            sum(rrt.get(key, {}).values()) + amount
            <= prob.machine.cap(key[0])
            for key, amount in candidate.items()
        )

    def unplace(v: int) -> None:
        candidate = footprint(v, placed[v])
        for key in candidate:
            occupants = rrt[key]
            del occupants[v]
            if not occupants:
                del rrt[key]
        del placed[v]

    last_cycle: dict[int, int] = {}
    worklist = list(order)
    steps = 0
    while worklist:
        steps += 1
        if steps > budget:
            return None
        v = worklist.pop(0)
        start = max(earliest(v), last_cycle.get(v, -1) + 1)
        chosen = None
        for t in range(start, start + ii):
            if fits(footprint(v, t)):
                chosen = t
                break
        if chosen is None:
            # Force-place at `start`, ejecting conflicting occupants (Rau).
            chosen = start
            candidate = footprint(v, chosen)
            if any(
                amount > prob.machine.cap(key[0])
                for key, amount in candidate.items()
            ):
                return None
            victims = {
                occupant
                for key, amount in candidate.items()
                if sum(rrt.get(key, {}).values()) + amount
                > prob.machine.cap(key[0])
                for occupant in rrt.get(key, {})
            }
            for w in victims:
                unplace(w)
                worklist.append(w)
        placed[v] = chosen
        last_cycle[v] = chosen
        for key, amount in footprint(v, chosen).items():
            rrt.setdefault(key, {})[v] = amount
        # Evict successors whose dependence is now violated.
        for e in out_edges.get(v, ()):
            if e.dst in placed and e.dst != v:
                d = prob.edge_latency(e)
                if placed[e.dst] < chosen + d - e.distance * ii:
                    unplace(e.dst)
                    worklist.append(e.dst)
        # And predecessors constrained backwards (loop-carried into v).
        for e in in_edges.get(v, ()):
            if e.src in placed:
                d = prob.edge_latency(e)
                if chosen < placed[e.src] + d - e.distance * ii:
                    unplace(e.src)
                    worklist.append(e.src)
    # Final validation.
    for e in prob.edges:
        d = prob.edge_latency(e)
        if placed[e.dst] < placed[e.src] + d - e.distance * ii:
            return None
    if any(
        sum(occupants.values()) > prob.machine.cap(functional_unit)
        for (functional_unit, _), occupants in rrt.items()
    ):
        return None
    return placed
