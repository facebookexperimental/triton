"""Classify a joint solution's WS strategy against the reference structure
the paper reports for Blackwell forward FMHA (its Fig 9):

  * one variable-latency group holding the TMA loads;
  * one Tensor-Core group holding all tcgen05 MMAs;
  * TWO distinct softmax groups (one per M-sub-tile exp2 chain);
  * a separate accumulator-rescale group (TMEM traffic + rescale mults);
  * softmax chains anti-phased (ping-pong) in the steady state.

Usage: python -m paper_joint_solver.strategy_report <ddg.json> <solution.json>
"""

import json
import sys
from collections import Counter, defaultdict

from .ddg import Problem, load_problem


def classify(prob: Problem, warp: dict[int, int],
             cycles: dict[int, int] | None = None) -> dict:
    groups = defaultdict(list)
    for v, w in warp.items():
        groups[w].append(prob.nodes[v])

    def group_of(pred):
        return {w for w, members in groups.items()
                if any(pred(n) for n in members)}

    tma_groups = group_of(lambda n: n.pipeline == "TMA" and "load" in n.op_kind)
    mma_groups = group_of(lambda n: "mma" in n.op_kind)
    exp_nodes = [n for n in prob.nodes.values() if "exp2" in n.op_kind]
    exp_groups = {warp[n.id] for n in exp_nodes}
    tmem_groups = group_of(lambda n: n.pipeline == "TMEM")

    # Sub-tile chains: connected components over non-shared nodes (mirrors
    # the DDG construction — chains meet only at K/V staging).
    report = {
        "num_groups_used": len({w for w, m in groups.items() if m}),
        "group_sizes": {w: len(m) for w, m in sorted(groups.items())},
        "tma_isolated": tma_groups and all(
            n.pipeline == "TMA" or n.occupancy == 0
            for w in tma_groups for n in groups[w]),
        "tma_groups": sorted(tma_groups),
        "mma_groups": sorted(mma_groups),
        "mma_single_group": len(mma_groups) == 1,
        "exp_groups": sorted(exp_groups),
        "softmax_two_groups": len(exp_groups) == 2,
        "tmem_groups": sorted(tmem_groups),
        "rescale_separate": bool(tmem_groups - exp_groups - mma_groups),
    }
    if cycles and len(exp_nodes) >= 2:
        by_group = defaultdict(list)
        for n in exp_nodes:
            by_group[warp[n.id]].append(cycles[n.id])
        report["softmax_phase_offsets"] = {
            str(w): sorted(v) for w, v in by_group.items()}
    report["fa4_like"] = bool(report["tma_isolated"]
                              and report["mma_single_group"]
                              and report["softmax_two_groups"]
                              and report["rescale_separate"])
    return report


def main(argv=None):
    argv = argv or sys.argv[1:]
    ddg_path, sol_path = argv[0], argv[1]
    prob = load_problem(ddg_path)
    sol = json.loads(open(sol_path).read())
    warp = {int(k): int(v) for k, v in sol["warp"].items()}
    cycles = {int(k): int(v) for k, v in sol.get("cycles", {}).items()}
    rep = classify(prob, warp, cycles or None)
    rep["ii"] = sol.get("ii")
    rep["length"] = sol.get("length")
    rep["wall_s"] = sol.get("wall_s")
    print(json.dumps(rep, indent=1))
    per_group = defaultdict(Counter)
    for v, w in warp.items():
        per_group[w][prob.nodes[v].op_kind.split(".")[-1]] += 1
    for w in sorted(per_group):
        print(f"group {w}: {dict(per_group[w])}")


if __name__ == "__main__":
    main()
