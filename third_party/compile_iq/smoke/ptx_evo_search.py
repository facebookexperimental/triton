"""PTX-direct EVO search: mint a ptxas ACF for a FIXED kernel.ptx using the EVO engine, scoring each
candidate via the PTX-direct driver-launch bridge (ptx_bench_one.py) in base Python.

This is the PTX-direct counterpart of evo_search.py: identical cross-env structure (EVO runs in its
own 3.10 conda env with no triton/torch, so each candidate is benchmarked in base 3.13 through a
subprocess bridge), but the bridge is ptx_bench_one.py -- which ptxas-assembles the fixed PTX and
launches the cubin via the CUDA driver, instead of recompiling the kernel from source.

Run (matching versions REQUIRED: a cuda-X.Y .config -> ACFs only ptxas X.Y accepts):
  conda run -n evo python ptx_evo_search.py <kernel.ptx> <spec.json> <search_space.config> [timeout_s]
Emits the best candidate's hex ACF to <spec_dir>/best.acf.hex (override via EVO_BEST_OUT).
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

BASE_PY = os.environ.get("BASE_PY", "python")  # base 3.13 (fbtriton + torch) that runs ptx_bench_one
HERE = pathlib.Path(__file__).resolve().parent
BENCH_ONE = str(HERE / "ptx_bench_one.py")

PTX_FILE = None
SPEC_FILE = None
PER_CAND_TIMEOUT = 30


def _log(msg):
    print(f"[ptx-evo-cand] {msg}", file=sys.stderr, flush=True)


def objective(acf, *args, **kwargs):
    """EVO hands a hex-encoded ACF; materialize it and bench the FIXED PTX with it applied, via the
    base-env PTX-direct driver-launch bridge. Returns ms (minimize) or EVO's INVALID_SCORE."""
    acf_hex = acf if isinstance(acf, str) else str(acf)
    with tempfile.NamedTemporaryFile(suffix=".acf", delete=False) as f:
        f.write(bytes.fromhex(acf_hex))
        acf_path = f.name
    t = time.time()
    try:
        # PER_CAND_TIMEOUT is the real budget; +5s only covers process spawn + torch import. A valid
        # PTX-direct candidate finishes in ~2-3s (ptxas + driver launch, no Triton recompile), so a
        # wedged ACF is killed at ~PER_CAND_TIMEOUT instead of paying a large margin.
        out = subprocess.run([BASE_PY, BENCH_ONE, PTX_FILE, SPEC_FILE, acf_path, "50", "100"], capture_output=True,
                             text=True, timeout=PER_CAND_TIMEOUT + 5)
    except subprocess.TimeoutExpired:
        os.unlink(acf_path)
        _log(f"cand wall={time.time()-t:.1f}s -> BRIDGE-TIMEOUT -> INVALID")
        return INVALID_SCORE
    os.unlink(acf_path)
    for line in out.stdout.splitlines():
        if line.startswith("MS "):
            ms = float(line.split()[1])
            _log(f"cand wall={time.time()-t:.1f}s -> ms={ms} VALID")
            return ms
    _log(f"cand wall={time.time()-t:.1f}s -> INVALID (stderr: {out.stderr.strip()[:160]})")
    return INVALID_SCORE


def main():
    global PTX_FILE, SPEC_FILE, PER_CAND_TIMEOUT
    PTX_FILE = sys.argv[1]
    SPEC_FILE = sys.argv[2]
    ss = pathlib.Path(sys.argv[3])
    if len(sys.argv) > 4:
        PER_CAND_TIMEOUT = int(sys.argv[4])
    print(f"[ptx-evo] ptx={PTX_FILE} space={ss.name} per_cand_timeout={PER_CAND_TIMEOUT}s", flush=True)

    cfg = EvoConfiguration(problem_type=ProblemType.MIN, num_objectives=1, qualitative=True, generations=1, pool_size=6,
                           cull_size=2, mutate_rate=0.5, enable_db=False)
    search = EvoSearch(objective_function=objective, search_space=ss, evo_config=cfg, worker_type=WorkerTypes.DEFAULT,
                       debug=False)
    result = search.start(num_workers=1)

    df = result.get_results()
    n = len(df) if df is not None else 0
    best = result.get_best_result()
    best_ms = best.get("score_1", best.get("score"))
    print(f"[ptx-evo] DONE evaluated={n} best_ms={best_ms}", flush=True)
    if not isinstance(best_ms, (int, float)):
        print("[ptx-evo] no valid candidate -- nothing to emit.", flush=True)
        return

    best_out = os.environ.get("EVO_BEST_OUT", os.path.join(os.path.dirname(SPEC_FILE), "best.acf.hex"))
    with open(best_out, "w") as f:
        f.write(best["params"])
    print(f"[ptx-evo] wrote best ACF hex -> {best_out}", flush=True)
    print("PTX_EVO_SEARCH_OK")


if __name__ == "__main__":
    main()
