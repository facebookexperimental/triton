"""RoleClassifier — map solver warp groups onto skeleton roles (SKC step 1).

Fingerprints operate on op_kind/pipeline sets per warp, the same signals
strategy_report.classify uses.  Classification is strict: every warp must
land in exactly one role and the role multiset must match the skeleton
(1 load, 1 mma, k>=1 softmax, 1 correction) or RoleClassificationError is
raised — no silent degradation (R0).
"""

from dataclasses import dataclass, field


class RoleClassificationError(Exception):
    pass


@dataclass
class RoleMap:
    load: int
    mma: int
    softmax: list[int]  # chain order as solver warp ids (binder orders them)
    correction: int
    # Solver placements the skeleton protocol overrides (R1): TMEM traffic
    # the solution parked on the mma/other warps, relocated by the skeleton's
    # fixed intra-role placement.  Recorded, never silently applied.
    protocol_overrides: list[dict] = field(default_factory=list)
    fingerprints: dict = field(default_factory=dict)


def _warp_nodes(prob, sol):
    warps = {}
    for idx_str, w in sol["warp"].items():
        warps.setdefault(w, []).append(prob.nodes[int(idx_str)])
    return warps


def _fingerprint(nodes):
    kinds = {n.op_kind for n in nodes}
    pipes = {n.pipeline for n in nodes}
    return {
        "kinds": kinds,
        "pipes": pipes,
        "has_tc": any(n.op_kind == "ttng.tc_gen5_mma" for n in nodes),
        "has_tma": any(n.pipeline == "TMA" for n in nodes),
        # Vector exp2 (occupancy > 0) is the softmax-chain signature; the
        # scalar alpha exp2 has occupancy 0 and appears in several places.
        "has_vec_exp2": any(n.op_kind == "math.exp2" and n.occupancy > 0 for n in nodes),
        "has_tmem": any(n.pipeline == "TMEM" for n in nodes),
        "n": len(nodes),
    }


def classify(prob, sol) -> RoleMap:
    warps = _warp_nodes(prob, sol)
    fps = {w: _fingerprint(nodes) for w, nodes in warps.items()}

    roles: dict[int, str] = {}
    for w, fp in fps.items():
        if fp["has_tc"]:
            roles[w] = "mma"
        elif fp["has_vec_exp2"]:
            roles[w] = "softmax"
        elif fp["has_tma"] and not fp["has_vec_exp2"]:
            roles[w] = "load"
        else:
            # Correction: TMEM acc reads and/or vector mulf rescale, no
            # tensor-core, no TMA, no softmax chain.
            roles[w] = "correction"

    by_role: dict[str, list[int]] = {}
    for w, r in roles.items():
        by_role.setdefault(r, []).append(w)

    problems = []
    if len(by_role.get("mma", [])) != 1:
        problems.append(f"expected exactly 1 mma warp, got {by_role.get('mma', [])}")
    if len(by_role.get("load", [])) != 1:
        problems.append(f"expected exactly 1 load warp, got {by_role.get('load', [])}")
    if len(by_role.get("softmax", [])) < 1:
        problems.append("no softmax warp found")
    if len(by_role.get("correction", [])) != 1:
        problems.append(f"expected exactly 1 correction warp, got {by_role.get('correction', [])}")
    if problems:
        raise RoleClassificationError(
            "solution does not map onto the fwd skeleton roles: "
            + "; ".join(problems) + f" (fingerprints: {fps})")

    mma_w = by_role["mma"][0]
    overrides = []
    for n in warps[mma_w]:
        if n.pipeline == "TMEM":
            overrides.append({
                "node": n.id, "op_kind": n.op_kind, "solver_warp": mma_w,
                "rule": "R1: mma issuer only issues tcgen05+barriers; "
                        "TMEM traffic follows the skeleton protocol "
                        "(softmax reads qk / stores p, correction rescales acc)",
            })
    for w in by_role["correction"] + by_role["softmax"]:
        pass  # softmax/correction TMEM traffic matches the protocol already

    return RoleMap(
        load=by_role["load"][0],
        mma=mma_w,
        softmax=sorted(by_role["softmax"]),
        correction=by_role["correction"][0],
        protocol_overrides=overrides,
        fingerprints={w: {"kinds": sorted(fp["kinds"]), "pipes": sorted(fp["pipes"])}
                      for w, fp in fps.items()},
    )


@dataclass
class BwdRoleMap:
    load: int
    mma: int
    compute: int  # softmax pT + dS chain (+ M/D loads per skeleton protocol)
    reduction: int  # dQ offload chain
    protocol_overrides: list[dict] = field(default_factory=list)
    fingerprints: dict = field(default_factory=dict)


def classify_bwd(prob, sol) -> BwdRoleMap:
    """Map a bwd solution's warps onto the fa_bwd_dkdv skeleton roles."""
    warps = _warp_nodes(prob, sol)
    fps = {w: _fingerprint(nodes) for w, nodes in warps.items()}

    roles: dict[int, str] = {}
    for w, nodes in warps.items():
        fp = fps[w]
        kinds = fp["kinds"]
        if fp["has_tc"]:
            roles[w] = "mma"
        elif "math.exp2" in kinds:
            roles[w] = "compute"
        elif "tt.descriptor_reduce" in kinds or (
                fp["has_tmem"] and "tt.descriptor_load" not in kinds):
            roles[w] = "reduction"
        elif fp["has_tma"]:
            roles[w] = "load"
        else:
            roles[w] = "unmapped"

    by_role: dict[str, list[int]] = {}
    for w, r in roles.items():
        by_role.setdefault(r, []).append(w)

    problems = []
    for r in ("mma", "compute", "reduction", "load"):
        if len(by_role.get(r, [])) != 1:
            problems.append(f"expected exactly 1 {r} warp, got {by_role.get(r, [])}")
    if by_role.get("unmapped"):
        problems.append(f"unmapped warps {by_role['unmapped']}")
    if problems:
        raise RoleClassificationError(
            "bwd solution does not map onto the skeleton roles: "
            + "; ".join(problems) + f" (fingerprints: {fps})")

    mma_w = by_role["mma"][0]
    overrides = [{
        "node": n.id, "op_kind": n.op_kind, "solver_warp": mma_w,
        "rule": "R1: mma issuer only issues tcgen05+barriers",
    } for n in warps[mma_w] if n.pipeline == "TMEM"]

    return BwdRoleMap(
        load=by_role["load"][0],
        mma=mma_w,
        compute=by_role["compute"][0],
        reduction=by_role["reduction"][0],
        protocol_overrides=overrides,
        fingerprints={w: {"kinds": sorted(fp["kinds"]), "pipes": sorted(fp["pipes"])}
                      for w, fp in fps.items()},
    )
