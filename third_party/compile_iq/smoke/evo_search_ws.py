"""Constrained EVO search for WS kernels -- REAL tuning (non-smoke).

The warp-specialized counterpart of evo_search.py. EVO (3.10 env) drives a search over a .config
space; the baseline and every candidate are evaluated in base 3.13 via bench_one_ws.py (real
matmul launch + ACF via PTXAS_OPTIONS). NON-SMOKE: the best ACF is emitted only if it beats the
no-ACF baseline (so consume HITs only on a genuine speedup; otherwise nothing is stored).

Run: conda run -n evo python evo_search_ws.py <task_dir> <search_space.config> [per_cand_timeout_s]
Env: BASE_PY (base interpreter), WS_GEMM_SIZE, EVO_GENERATIONS (default 1), EVO_POOL (default 8).
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

BASE_PY = os.environ.get("BASE_PY", "python")
HERE = pathlib.Path(__file__).resolve().parent
BENCH = str(HERE / "bench_one_ws.py")
PER_CAND = 300
GENERATIONS = int(os.environ.get("EVO_GENERATIONS", "1"))
POOL = int(os.environ.get("EVO_POOL", "8"))


def _bench(acf_arg):
    """acf_arg = hex-file path, or 'NONE' for the baseline. Returns ms or None."""
    try:
        out = subprocess.run([BASE_PY, BENCH, acf_arg], capture_output=True, text=True,
                             timeout=PER_CAND + 120)
    except subprocess.TimeoutExpired:
        return None
    for line in out.stdout.splitlines():
        if line.startswith("MS "):
            return float(line.split()[1])
    return None


def objective(acf, *args, **kwargs):
    with tempfile.NamedTemporaryFile("w", suffix=".hex", delete=False) as f:
        f.write(acf if isinstance(acf, str) else str(acf))
        hex_path = f.name
    t = time.time()
    ms = _bench(hex_path)
    os.unlink(hex_path)
    print(f"[evo-cand] wall={time.time()-t:.1f}s -> {'ms='+repr(ms) if ms is not None else 'INVALID'}",
          file=sys.stderr, flush=True)
    return ms if ms is not None else INVALID_SCORE


def main():
    global PER_CAND
    task_dir = sys.argv[1]
    ss = pathlib.Path(sys.argv[2])
    if len(sys.argv) > 3:
        PER_CAND = int(sys.argv[3])
    print(f"[evo-ws] task={task_dir} space={ss.name} gens={GENERATIONS} pool={POOL} "
          f"per_cand={PER_CAND}s size={os.environ.get('WS_GEMM_SIZE','2048')}", flush=True)

    base_ms = _bench("NONE")
    if base_ms is None:
        print("[evo-ws] FAIL: baseline (no-ACF) did not run")
        return 1
    print(f"[evo-ws] baseline_ms={base_ms}", flush=True)

    cfg = EvoConfiguration(problem_type=ProblemType.MIN, num_objectives=1, qualitative=True,
                           generations=GENERATIONS, pool_size=POOL, cull_size=2, mutate_rate=0.5,
                           enable_db=False)
    result = EvoSearch(objective_function=objective, search_space=ss, evo_config=cfg,
                       worker_type=WorkerTypes.DEFAULT, debug=False).start(num_workers=1)

    df = result.get_results()
    n = len(df) if df is not None else 0
    best = result.get_best_result()
    best_ms = best.get("score_1", best.get("score"))
    print(f"[evo-ws] DONE evaluated={n} baseline_ms={base_ms} best_ms={best_ms}", flush=True)

    # NON-SMOKE: store only if the best validated candidate beats baseline.
    if isinstance(best_ms, (int, float)) and best_ms < base_ms:
        out = os.path.join(task_dir, "best.acf.hex")
        with open(out, "w") as f:
            f.write(best["params"])
        print(f"[evo-ws] SPEEDUP {100*(base_ms-best_ms)/base_ms:.2f}% -> wrote {out}", flush=True)
        print("EVO_WS_OK stored")
    else:
        print(f"[evo-ws] no candidate beat baseline ({best_ms} vs {base_ms}) -> storing nothing", flush=True)
        print("EVO_WS_OK nostore")


if __name__ == "__main__":
    main()
