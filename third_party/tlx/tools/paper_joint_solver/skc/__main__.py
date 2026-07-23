"""CLI: python -m skc --solution sol.json --ddg ddg.json --out instance.py

--default-params skips solution binding and emits the skeleton defaults
(M1 baseline instance).  --set K=V overrides individual parameters for A/B
runs (e.g. --set BLOCK_M=128 to adopt the solver fixture's tile geometry).
"""

import argparse
import json
from pathlib import Path

from paper_joint_solver.ddg import load_problem

from .binder import bind, bind_bwd
from .instantiate import render
from .roles import classify, classify_bwd
from .skeleton_bwd import DEFAULT_PARAMS_BWD, check_register_budget_bwd
from .skeleton_fwd import DEFAULT_PARAMS, check_register_budget


def main(argv=None):
    ap = argparse.ArgumentParser(prog="skc")
    ap.add_argument("--solution")
    ap.add_argument("--ddg")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skeleton", choices=["fwd", "bwd"], default="fwd")
    ap.add_argument("--default-params", action="store_true")
    ap.add_argument("--set", action="append", default=[],
                    metavar="K=V", help="override a bound parameter")
    ap.add_argument("--u", type=int, default=300)
    args = ap.parse_args(argv)

    defaults = DEFAULT_PARAMS if args.skeleton == "fwd" else DEFAULT_PARAMS_BWD
    budget_check = (check_register_budget if args.skeleton == "fwd"
                    else check_register_budget_bwd)

    if args.default_params:
        params = dict(defaults)
        audit = {"mode": "default-params (M1 baseline; no solution bound)"}
        sol_path = ddg_path = None
    else:
        if not (args.solution and args.ddg):
            ap.error("--solution and --ddg are required unless --default-params")
        prob = load_problem(args.ddg, u=args.u)
        sol = json.loads(Path(args.solution).read_text())
        ddg_raw = json.loads(Path(args.ddg).read_text())
        if args.skeleton == "fwd":
            roles = classify(prob, sol)
            binding = bind(prob, sol, roles, ddg_raw=ddg_raw)
            role_summary = {"load": roles.load, "mma": roles.mma,
                            "softmax": roles.softmax,
                            "correction": roles.correction}
        else:
            roles = classify_bwd(prob, sol)
            binding = bind_bwd(prob, sol, roles, ddg_raw=ddg_raw)
            role_summary = {"load": roles.load, "mma": roles.mma,
                            "compute": roles.compute,
                            "reduction": roles.reduction}
        params = dict(defaults)
        params.update(binding.params)
        audit = {"mode": "solution-bound",
                 "roles": {**role_summary, "fingerprints": roles.fingerprints},
                 **binding.audit}
        sol_path, ddg_path = args.solution, args.ddg

    for kv in args.set:
        k, v = kv.split("=", 1)
        if k not in params:
            ap.error(f"unknown parameter {k}; known: {sorted(params)}")
        params[k] = int(v)
        audit.setdefault("manual_overrides", []).append(kv)

    total = budget_check(params)
    audit["register_budget_check"] = f"{total} <= 65536 OK"

    render(params, audit, solution_path=sol_path, ddg_path=ddg_path,
           out_path=args.out, skeleton=args.skeleton)
    print(f"wrote {args.out}")
    print(json.dumps(params, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
