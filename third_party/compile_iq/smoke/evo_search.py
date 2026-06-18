"""Constrained EVO end-to-end: a REAL ptxas-ACF search driven by the EVO engine.

EVO (evo_solar, Python 3.10 conda env `evo`) orchestrates an evolutionary search over a
constrained ptxas knob space (a `.config` tier from the manifold drop -- EVO-format, which
the older CIQ engine cannot parse). Because the evo env has no triton/torch, each candidate
is benchmarked in base 3.13 through the bench_one.py bridge (cross-env via subprocess).

This is the EVO counterpart to CIQ's run_e2e.sh factory stage. Unlike the full
`ptxas13.3.bin` (whose random candidates wedge the GPU), the constrained p-tier should yield
mostly valid candidates so the search converges.

Run:  conda run -n evo python evo_search.py <task_dir> <search_space.config> [per_cand_timeout_s]
"""
import os
import pathlib
import subprocess
import sys
import tempfile
import time

from evo_solar.config.const import INVALID_SCORE
from evo_solar.config.types import EvoConfiguration, ProblemType, WorkerTypes
from evo_solar.evo import EvoSearch

# Base-env (3.13) interpreter that runs bench_one.py. EVO runs under its own 3.10 env,
# so this must be passed in (run_e2e_search.sh exports BASE_PY); falls back to PATH `python`.
BASE_PY = os.environ.get("BASE_PY", "python")
HERE = pathlib.Path(__file__).resolve().parent
BENCH_ONE = str(HERE / "bench_one.py")

# Set in main(); read by the (forked) objective workers.
TASK_DIR = None
PER_CAND_TIMEOUT = 30


def _log(msg):
    # Per-candidate visibility (the search culls wedged candidates) -> stderr.
    print(f"[evo-cand] {msg}", file=sys.stderr, flush=True)


def objective(acf, *args, **kwargs):
    """EVO hands us a hex-encoded ACF string; bench it in base 3.13 via the bridge.
    Returns the candidate runtime (ms, minimize) or EVO's INVALID_SCORE sentinel."""
    with tempfile.NamedTemporaryFile("w", suffix=".hex", delete=False) as f:
        f.write(acf if isinstance(acf, str) else str(acf))
        hex_path = f.name
    t = time.time()
    try:
        out = subprocess.run(
            [BASE_PY, BENCH_ONE, TASK_DIR, hex_path, str(PER_CAND_TIMEOUT)],
            capture_output=True,
            text=True,
            timeout=PER_CAND_TIMEOUT + 90,
        )
    except subprocess.TimeoutExpired:
        os.unlink(hex_path)
        _log(f"cand wall={time.time()-t:.1f}s -> BRIDGE-TIMEOUT -> INVALID")
        return INVALID_SCORE
    os.unlink(hex_path)
    for line in out.stdout.splitlines():
        if line.startswith("MS "):
            ms = float(line.split()[1])
            _log(f"cand wall={time.time()-t:.1f}s -> ms={ms} VALID")
            return ms
    _log(f"cand wall={time.time()-t:.1f}s -> INVALID (stderr: {out.stderr.strip()[:160]})")
    return INVALID_SCORE


def main():
    global TASK_DIR, PER_CAND_TIMEOUT
    TASK_DIR = sys.argv[1]
    ss = pathlib.Path(sys.argv[2])
    if len(sys.argv) > 3:
        PER_CAND_TIMEOUT = int(sys.argv[3])
    print(f"[evo-search] task={TASK_DIR} space={ss.name} per_cand_timeout={PER_CAND_TIMEOUT}s", flush=True)

    cfg = EvoConfiguration(
        problem_type=ProblemType.MIN,
        num_objectives=1,
        qualitative=True,
        generations=1,
        pool_size=6,
        cull_size=2,
        mutate_rate=0.5,
        enable_db=False,
    )
    search = EvoSearch(objective_function=objective, search_space=ss, evo_config=cfg, worker_type=WorkerTypes.DEFAULT,
                       debug=False)
    result = search.start(num_workers=1)

    df = result.get_results()
    n = len(df) if df is not None else 0
    best = result.get_best_result()
    best_ms = best.get("score_1", best.get("score"))
    print(f"[evo-search] DONE evaluated={n} best_ms={best_ms}", flush=True)

    # Emit the best candidate's hex ACF so the base-env store step can persist it.
    best_out = os.environ.get("EVO_BEST_OUT", os.path.join(TASK_DIR, "best.acf.hex"))
    with open(best_out, "w") as f:
        f.write(best["params"])
    print(f"[evo-search] wrote best ACF hex -> {best_out}", flush=True)
    print("EVO_SEARCH_OK")


if __name__ == "__main__":
    main()
