"""EVO engine smoke test (search-only, no GPU / no compile).

Confirms the bundled Racket core (`_evo_manager`) ingests a real ptxas `.config`
search space and runs one generation end-to-end, returning an `EvoResult`.

A fast sanity check of the engine wiring before the full PTX-direct e2e
(`run_e2e_cuda12{8,30}.sh`). EVO takes a `.config` search space (Racket DNA),
passed as `EvoSearch(search_space=...)`.

Must run in the `evo` conda env (Python 3.10, evo_nda). Pass a `.config` search
space, or set SS_CONFIG / PTXAS_KNOBS:
    conda run -n evo python evo_smoke.py <search_space.config>
"""
import os
import pathlib
import sys

from evo_solar.config.types import EvoConfiguration, ProblemType, WorkerTypes
from evo_solar.evo import EvoSearch


def objective(*args, **kwargs):
    """No-op objective: constant non-zero score. We only prove the search loop
    turns over the .config; real compile+benchmark scoring is the cross-env M4 work.
    Keep qualitative=True below: qualitative=False turns on score normalization
    (score / baseline), which divides by a meaningless baseline and stamps every
    candidate INVALID, so the search never converges."""
    return 1.0


def _default_ss():
    if os.environ.get("SS_CONFIG"):
        return os.environ["SS_CONFIG"]
    knobs = os.environ.get("PTXAS_KNOBS")
    return os.path.join(knobs, "cuda-13.0-ptxas-p0.config") if knobs else None


def main() -> int:
    arg = sys.argv[1] if len(sys.argv) > 1 else _default_ss()
    if not arg:
        print("[evo-smoke] FAIL: pass a .config path, or set SS_CONFIG / PTXAS_KNOBS")
        return 2
    ss = pathlib.Path(arg)
    if not ss.exists():
        print(f"[evo-smoke] FAIL: search space not found: {ss}")
        return 2
    print(f"[evo-smoke] search_space={ss} size={ss.stat().st_size}")

    cfg = EvoConfiguration(
        problem_type=ProblemType.MIN,
        num_objectives=1,
        qualitative=True,  # no normalization -> constant objective is valid
        generations=1,
        pool_size=6,
        cull_size=2,
        mutate_rate=0.5,
        enable_db=False,
    )
    search = EvoSearch(
        objective_function=objective,
        search_space=ss,
        evo_config=cfg,
        worker_type=WorkerTypes.DEFAULT,
        debug=True,
    )
    result = search.start(num_workers=1)
    ok = result is not None and type(result).__name__ == "EvoResult"
    print(f"[evo-smoke] {'PASS' if ok else 'FAIL'} result_type={type(result).__name__}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
