"""First-window probe: a published execution protocol, not solver logic.

Algorithm 1 does not terminate on a structurally unsatisfiable model — it
keeps raising I forever.  Batch runs that need bounded evidence for such an
ablation use this verb instead: it locates the smallest I with a modulo
schedule, probes that I's entire L window, and reports the verdict at every
point.  The canonical search path (``python -m paper_joint_solver`` ->
``run_search``) stays literal and unbounded; nothing here feeds back into it.

The exit code is always 0: completing the protocol is the success condition,
and the verdicts are evidence inside the JSON, never a process status.
"""

import argparse
import hashlib
import json
from pathlib import Path

from .__main__ import (
    add_experiment_inputs,
    configure_machine,
    recorded_experiment_inputs,
)
from .ddg import load_problem
from .joint_smt import solve_joint
from .machine import MachineModel
from .modulo_ilp import solve_modulo
from .schedule_plan import PAPER_MODEL_PROFILE, solution_provenance

PROBE_SCHEMA = "paper-joint-probe-v1"
EXECUTION_MODE = "first-window-probe"


def probe_sources_sha256() -> str:
    """This module's own source digest.

    A producer's provenance has to cover exactly the code that can change the
    artifact's meaning -- no less, or edits go unnoticed, and no more, or an
    unrelated edit invalidates artifacts it cannot affect.  This file chooses
    which (I, L) points are probed and how the verdicts are summarized, and
    the shared solver core it calls is covered separately by
    ``solver_sources_sha256``.  Same shape as the curation stage's
    ``curator_sources_sha256``.
    """
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def probe_first_window(prob, num_warps_override: int | None = None,
                       allow_cross_warp: bool = True) -> dict:
    """Probe every L in the first ceiling window of the smallest feasible I."""
    ii = 0
    while True:
        ii += 1
        m = solve_modulo(prob, ii)
        if m is not None:
            break
    length = m.length
    copies = -(-length // ii)
    window = []
    while -(-length // ii) == copies:
        sol, verdict = solve_joint(
            prob, ii, length,
            num_warps_override=num_warps_override,
            allow_cross_warp=allow_cross_warp,
        )
        del sol  # the verdict is the evidence; the model is not recorded
        print("[probe]", {"ii": ii, "L": length, "result": verdict},
              flush=True)
        window.append({"L": length, "verdict": verdict})
        length += 1
    return {
        "ii": ii,
        "window": window,
        "window_size": len(window),
        "unsat_count": sum(1 for point in window
                           if point["verdict"] == "unsat"),
        "sat_found": any(point["verdict"] == "sat" for point in window),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(prog="paper_joint_solver.probe")
    ap.add_argument("ddg", help="curated_ddg.json (see curate_ddg)")
    ap.add_argument("-o", "--out", required=True)
    add_experiment_inputs(ap)
    args = ap.parse_args(argv)

    machine = MachineModel()
    configure_machine(ap, args, machine)
    prob = load_problem(args.ddg, machine=machine, u=args.normalization_u)
    result = probe_first_window(
        prob,
        num_warps_override=args.num_warps_override,
        allow_cross_warp=not args.no_cross_warp,
    )
    provenance = solution_provenance(
        args.ddg,
        machine,
        args.normalization_u,
        model=PAPER_MODEL_PROFILE,
        inputs=recorded_experiment_inputs(args),
    )
    # Appended here rather than inside solution_provenance(): that helper lives
    # in the solver core, so widening it would re-stamp every artifact.
    provenance["probe_sources_sha256"] = probe_sources_sha256()
    payload = {
        "schema_version": PROBE_SCHEMA,
        "execution_mode": EXECUTION_MODE,
        "provenance": provenance,
        **result,
    }
    Path(args.out).write_text(json.dumps(payload, indent=1) + "\n")
    print(f"[probe] I={result['ii']} "
          f"unsat {result['unsat_count']}/{result['window_size']} "
          f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
