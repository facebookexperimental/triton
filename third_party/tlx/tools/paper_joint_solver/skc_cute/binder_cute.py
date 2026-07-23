"""CuTe binder — map frozen solver solutions onto FA4's parameter surface.

Emits a binding JSON with four-state classification per parameter:
  BIND     — value written into the shim (register quotas, kv_stage down-clamp,
             split_P_arrive)
  VERIFY   — solver-derived value asserted equal to FA4's own (E1 convergence
             evidence: kv_stage, s_stage, q_stage, dedicated issuer, tiles)
  DROPPED  — solver information the skeleton cannot express (exact cycles)
  FROZEN   — protocol declared out of scope with source-line justification
             (warp tuples, MMA issue order, producer_tail)

Pure Python — no flash_attn import; runs in the main venv.
"""

import json
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent  # paper_joint_solver/

# E1 expectations: FA4's independently-tuned 1-CTA hd128 fp16 non-causal
# operating point, which the solver must have re-derived for convergence.
FA4_1CTA_EXPECT = {
    "kv_stage": 3,       # SMEM formula flash_fwd_sm100.py:360 at 1-CTA hd128
    "s_stage": 2,        # hardcoded :366
    "q_stage": 2,        # interface.py:559 for seqlen > 256
    "tile_m": 128, "tile_n": 128,
    "split_P_arrive": 96,  # n_block//4*3 at :164-167
}
# Upstream's untuned 1-CTA nc hd128 register fallback (_TUNING_CONFIG has no
# (False, False, 128, False) key -> elif branch :332-335): softmax 192,
# correction 80, other = 512 - 2*192 - 80 = 48.
FA4_1CTA_DEFAULT_REGS = {"num_regs_softmax": 192, "num_regs_correction": 80}
# The 2-CTA-tuned key (True, False, 128, False): 176/88 — transfer candidate.
FA4_2CTA_TUNED_REGS = {"num_regs_softmax": 176, "num_regs_correction": 88}

FROZEN = [
    {"param": "warp role tuples", "reason":
     "named-barrier index arithmetic hardcodes 4-warp groups "
     "(flash_fwd_sm100.py:1015-1017, named_barrier.py:15-25)"},
    {"param": "MMA issue order", "reason":
     "fused tri-role mbarrier ring + deliberately elided O-full commits are "
     "justified by the static order (flash_fwd_sm100.py:966-977, 1760-1764)"},
    {"param": "producer_tail workarounds", "reason":
     "upstream comments 'commented out because it hangs' "
     "(flash_bwd_sm100.py:2758-2762) — inherit verbatim, never fix"},
]


def _check_fwd_invariants(regs, kv_stage, split_p):
    sm, corr = regs["num_regs_softmax"], regs["num_regs_correction"]
    other = 512 - 2 * sm - corr
    problems = []
    if sm % 8 or corr % 8 or other % 8:
        problems.append(f"register quotas not multiples of 8: {sm}/{corr}/{other}")
    if other < 24:
        problems.append(f"num_regs_other={other} < 24 floor")
    if not (1 <= kv_stage <= 3):
        problems.append(f"kv_stage={kv_stage} outside [1,3] (3 = 1-CTA hd128 max)")
    if split_p % 32 or not (0 <= split_p < 128):
        problems.append(f"split_P_arrive={split_p} violates %32 / <n_block")
    if problems:
        raise ValueError("fwd binding invariant violations: " + "; ".join(problems))
    return other


# Ops whose results are register-resident in the skeleton (memdesc producers
# and zero-cost aliases excluded).
_REG_KINDS = ("arith.", "math.", "tt.reduce", "ttg.convert_layout",
              "ttng.tmem_load")
# FA4's correction WG streams the TMEM accumulator through registers in
# corr_tile_size=16-column chunks (flash_fwd_sm100.py:2743) — full-tile
# liveness on the correction warp is divided by 128/16.
_CORR_STREAM_DIV = 8
_GEOM_SCALE = 4          # solved 64x64 qk tile -> realized 128x128
_WG_THREADS = 128
_ADDR_FLOOR = 24         # addressing/misc registers per thread


def _peak_live_units(prob, sol, warp):
    """Per-cycle peak of concurrently live register units on one solver warp
    under the modulo schedule (interval overlap incl. cross-iteration copies).
    """
    import math
    ii = sol["ii"]
    succs = {}
    for e in prob.edges:
        succs.setdefault(e.src, []).append(e)
    intervals = []
    for vs, w in sol["warp"].items():
        if w != warp:
            continue
        v = int(vs)
        n = prob.nodes[v]
        if not any(n.op_kind.startswith(k) for k in _REG_KINDS):
            continue
        r = prob.regs.get(v, 0)
        if not r:
            continue
        birth = sol["cycles"][vs]
        death = birth + 1
        for e in succs.get(v, []):
            if e.dst != v:
                death = max(death,
                            sol["cycles"][str(e.dst)] + ii * e.distance)
        intervals.append((birth, death, r))
    peak = 0
    for tau in range(ii):
        live = 0
        for b, d, r in intervals:
            lo = math.ceil((b - tau) / ii)
            hi = math.floor((d - 1 - tau) / ii)
            live += r * max(0, hi - lo + 1)
        peak = max(peak, live)
    return peak


def _solver_reg_quota_fwd(sol):
    """Solver-liveness-derived quotas from the solved schedule itself.

    Softmax: peak concurrent register units on each softmax chain warp,
    scaled to realized geometry, per thread, + addressing floor.
    Correction: same, divided by the skeleton's TMEM streaming factor.
    Roles per the FA4-exact solution: load=0, softmax=1,2, mma=3, correction=4.
    """
    from paper_joint_solver.ddg import load_problem
    prob = load_problem(str(PKG.parent / "sched2tlx" / "examples"
                            / "case3_FA_fp16_subtiled" / "ddg.json"))

    def to_quota(units, div=1):
        per_thread = units * _GEOM_SCALE / _WG_THREADS / div + _ADDR_FLOOR
        return min(int(-(-per_thread // 8) * 8), 256)

    sm_peak = max(_peak_live_units(prob, sol, 1),
                  _peak_live_units(prob, sol, 2))
    corr_peak = _peak_live_units(prob, sol, 4)
    sm = to_quota(sm_peak)
    corr = to_quota(corr_peak, div=_CORR_STREAM_DIV)
    while 2 * sm + corr > 512 - 24:   # keep num_regs_other >= 24
        sm -= 8
    return {"num_regs_softmax": sm, "num_regs_correction": corr,
            "_derivation": {"sm_peak_units": sm_peak,
                            "corr_peak_units": corr_peak,
                            "geom_scale": _GEOM_SCALE,
                            "corr_stream_div": _CORR_STREAM_DIV,
                            "addr_floor": _ADDR_FLOOR}}


def bind_fwd(solution_path=None, out_path=None):
    solution_path = Path(solution_path or PKG / "subtiled_fa4exact_solution.json")
    sol = json.loads(Path(solution_path).read_text())

    solver_regs = _solver_reg_quota_fwd(sol)
    candidates = {
        "solver_liveness": solver_regs,
        "transfer_2cta_tuned": dict(FA4_2CTA_TUNED_REGS),
        "upstream_default": dict(FA4_1CTA_DEFAULT_REGS),
    }
    for name, regs in candidates.items():
        _check_fwd_invariants(
            {k: v for k, v in regs.items() if not k.startswith("_")},
            FA4_1CTA_EXPECT["kv_stage"], FA4_1CTA_EXPECT["split_P_arrive"])

    binding = {
        "kind": "fwd", "solution": Path(solution_path).name,
        "solver_point": {"ii": sol["ii"], "length": sol["length"],
                         "num_warps": sol["stats"].get("num_warps")},
        "verifies": {
            # E1: these must equal what FA4 derives on its own at 1-CTA hd128.
            "kv_stage": {"expect": FA4_1CTA_EXPECT["kv_stage"],
                         "solver_source": "K liveness 2 copies + V 1 (Phase A audit)"},
            "s_stage": {"expect": FA4_1CTA_EXPECT["s_stage"],
                        "solver_source": "QK span 66 > II=66 -> depth 2"},
            "q_stage": {"expect": FA4_1CTA_EXPECT["q_stage"],
                        "solver_source": "two softmax chains (W=5 partition)"},
            "m_block_size": {"expect": 128, "solver_source":
                             "SUB_M=64 realized at 128 (Phase A TMEM-tile clamp)"},
            "n_block_size": {"expect": 128, "solver_source":
                             "BN=64 solved; FA4 tile fixed — VERIFY records the "
                             "geometry delta, not an equality claim"},
            "dedicated_mma_issuer": {"expect": True,
                                     "solver_source": "FA4-exact colocate probe SAT"},
        },
        "overrides": {
            # E2: the only live lever — quota candidates onto the untuned key.
            "reg_candidates": candidates,
            "split_P_arrive": {"default": 96, "sweep": [96, 64, 32, 0],
                               "solver_source":
                               "MMA_PV_SKEW=1 (iteration-level) ~ sub-iteration "
                               "partial-P overlap; semantic approximation, audited"},
            "kv_stage_sweep": [3, 2, 1],
        },
        "dropped": [
            "exact normalized cycles (II=66, L=148 offsets) — same handoff "
            "information loss Phase A audited",
            "solver geometry BN=64 (FA4 tile is protocol; delta recorded in "
            "verifies.n_block_size)",
        ],
        "frozen": FROZEN,
    }
    out_path = Path(out_path or PKG / "skc_cute" / "bindings" / "fwd_1cta.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(binding, indent=1))
    return binding


def bind_bwd(solution_path=None, out_path=None):
    solution_path = Path(solution_path or PKG / "bwd_skc_solution.json")
    sol = json.loads(Path(solution_path).read_text())

    # bwd invariant: reduce + 2*compute + max(load, mma) <= 512 (:229-234).
    candidates = {
        "transfer_2cta_tuned": {"num_regs_reduce": 136, "num_regs_compute": 136,
                                "num_regs_load": 104, "num_regs_mma": 104},
        "upstream_default_1cta": {"num_regs_reduce": 152, "num_regs_compute": 136,
                                  "num_regs_load": 88, "num_regs_mma": 88},
    }
    for name, r in candidates.items():
        total = r["num_regs_reduce"] + 2 * r["num_regs_compute"] + max(
            r["num_regs_load"], r["num_regs_mma"])
        if total > 512:
            raise ValueError(f"bwd quota candidate {name} breaks 512 invariant: {total}")
        if any(v % 8 for v in r.values()):
            raise ValueError(f"bwd quota candidate {name} not multiples of 8")

    binding = {
        "kind": "bwd", "solution": Path(solution_path).name,
        "solver_point": {"ii": sol["ii"], "length": sol["length"]},
        "verifies": {
            "Q_stage": {"expect": 2, "solver_source":
                        "Q liveness 3 clamped to 2 by SMEM accounting (M4 audit); "
                        "FA4 1-CTA sets Q_stage=2 (flash_bwd_sm100.py:238)"},
            "tile_m": {"expect": 128}, "tile_n": {"expect": 128},
            "dedicated_mma_issuer": {"expect": True, "solver_source":
                                     "R1 pin SAT at free optimum (II=95,L=273)"},
            "compute_split_2wg": {"expect": True, "solver_source":
                                  "monolithic compute colocate genuinely UNSAT"},
        },
        "overrides": {"reg_candidates": candidates,
                      "Q_stage_sweep": [2, 1]},
        "dropped": ["exact cycles", "5-dot issue order (FROZEN protocol)"],
        "frozen": FROZEN,
        "risk_R1": "bwd 1-CTA path is never exercised upstream (no CUDA-12 "
                   "fallback for bwd) — M3 correctness gate is load-bearing",
    }
    out_path = Path(out_path or PKG / "skc_cute" / "bindings" / "bwd_1cta.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(binding, indent=1))
    return binding


if __name__ == "__main__":
    print(json.dumps(bind_fwd()["overrides"]["reg_candidates"], indent=1))
    bind_bwd()
    print("bindings written")
