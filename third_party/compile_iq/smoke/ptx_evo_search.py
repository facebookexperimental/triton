"""PTX-direct EVO search: mint a ptxas ACF for a FIXED kernel.ptx using the EVO engine, scoring each
candidate via the PTX-direct driver-launch bridge (ptx_bench_one.py) in base Python.

This is the PTX-direct counterpart of evo_search.py: identical cross-env structure (EVO runs in its
own 3.10 conda env with no triton/torch, so each candidate is benchmarked in base 3.13 through a
subprocess bridge), but the bridge is ptx_bench_one.py -- which ptxas-assembles the fixed PTX and
launches the cubin via the CUDA driver, instead of recompiling the kernel from source.

Run (matching versions REQUIRED: a cuda-X.Y .config -> ACFs only ptxas X.Y accepts):
  conda run -n evo python ptx_evo_search.py <kernel.ptx> <spec.json> <search_space.config> [timeout_s]
Emits the best candidate's hex ACF to EVO_BEST_OUT (default: a fresh scratch dir, so the task dir
stays input-only) and logs the (noisy) search-time win vs the no-ACF baseline.
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
    # Search depth (env-overridable). generations=1 is pure random sampling of `pool_size` ACFs;
    # generations>=2 enables EVO's cull/mutate/breed loop -- the actual optimization. Larger spaces
    # (p1/p2) and real perf searches want more of both (and pay more wedge cost). pool must be > cull.
    gens = int(os.environ.get("EVO_GENERATIONS", "1"))
    pool = int(os.environ.get("EVO_POOL", "6"))
    cull = int(os.environ.get("EVO_CULL", "2"))
    if cull % 2:  # EVO requires an even cull_size (pydantic: "multiple of 2")
        cull = max(2, cull - 1)
    print(
        f"[ptx-evo] ptx={PTX_FILE} space={ss.name} per_cand_timeout={PER_CAND_TIMEOUT}s "
        f"generations={gens} pool_size={pool} cull_size={cull}", flush=True)

    cfg = EvoConfiguration(problem_type=ProblemType.MIN, num_objectives=1, qualitative=True, generations=gens,
                           pool_size=pool, cull_size=cull, mutate_rate=0.5, enable_db=False)
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

    # Transient handoff to the (base-env) store step. Default to a fresh scratch dir, NOT the task dir
    # -- the task dir stays input-only (kernel.ptx + spec.json). Callers set EVO_BEST_OUT explicitly.
    best_out = os.environ.get("EVO_BEST_OUT") or os.path.join(tempfile.mkdtemp(prefix="ciq_evo_"), "best.acf.hex")
    with open(best_out, "w") as f:
        f.write(best["params"])
    # Search-time win vs the no-ACF baseline (same cheap per-candidate bench -> NOISY; the trustworthy
    # locked-clock A/B is at consumption). Best-effort: skip the win if baseline can't be measured.
    # TODO(compile_iq perf, item 2): `best_ms` is the MIN over noisy candidates (winner's curse ->
    # biased low) compared to a single un-selected baseline, so this win is inflated and unstable
    # run-to-run. Replace with a per-candidate interleaved A/B (base vs ACF, median over rounds, locked
    # clocks) so the reported win predicts consume.
    base_ms = None
    try:
        bo = subprocess.run([BASE_PY, BENCH_ONE, PTX_FILE, SPEC_FILE, "NONE", "50", "100"], capture_output=True,
                            text=True, timeout=PER_CAND_TIMEOUT + 60)
        for line in bo.stdout.splitlines():
            if line.startswith("MS "):
                base_ms = float(line.split()[1])
    except Exception:
        pass
    win = f"{(base_ms - best_ms) / base_ms * 100:+.2f}%" if base_ms else "n/a"
    print(
        f"[ptx-evo] best ACF search-time win={win} (best_ms={best_ms} vs no-ACF baseline={base_ms}; "
        "NOISY -- validate at consume)", flush=True)
    print(f"[ptx-evo] wrote best ACF hex -> {best_out}", flush=True)
    print("PTX_EVO_SEARCH_OK")


if __name__ == "__main__":
    main()
