"""Stage 2 — ACF factory (basic, generic-replay). Reads a collected task, drives a CompileIQ
search over ptxas Advanced Control Files (ACFs) by replaying the kernel per candidate, and writes
the best ACF to the store. Sufficient for simple (non-warp-specialized) kernels like the naive
matmul -- completing the collect -> factory -> consume loop. Warp-specialized / TMA kernels need
the real-launch + parity + consume-faithful machinery added in the next diff.

    python -m compile_iq.factory <task_dir> [--generations N] [--pool-size N]
                                            [--search-space-bin PATH]

Correctness is checked by SELF-CONSISTENCY (an ACF must not change results vs the no-ACF run),
so it is op-agnostic. Each candidate is applied via the ptx_options launch kwarg and benchmarked
IN AN ISOLATED SPAWN SUBPROCESS: some ACFs wedge the GPU (e.g. on a driver older than the ptxas
that assembled them), and the only reliable way to reclaim a wedged context is to kill the process
holding it. A candidate that wedges (timeout), crashes (IMA), or diverges scores INVALID and the
search keeps moving -- it never hangs the factory. Only an ACF that beats the no-ACF baseline is
stored -- on a miss the consume hook falls back to plain compile.
"""

import argparse
import os
import tempfile

from . import replay, store

# PTXAS search-space bin: a separate NVIDIA CompileIQ artifact (not vendored here).
DEFAULT_SS_BIN = os.environ.get("COMPILE_IQ_SEARCH_SPACE_BIN")
REL_TOL = 1e-2
# Per-candidate wall-clock budget (spawn + import + compile + bench of one ACF). A candidate that
# wedges the GPU never returns; its subprocess is killed at this timeout and scored INVALID so the
# search keeps moving instead of hanging the whole factory.
GENERIC_TIMEOUT = int(os.environ.get("COMPILE_IQ_GENERIC_TIMEOUT", "90"))


def _eval_target(task_dir, acf_path, warmup, rep, q):
    """Run in an isolated spawn subprocess: load the task, replay no-ACF (reference) then with the
    ACF applied via the ptx_options launch kwarg, check self-consistency, and benchmark. Puts the
    runtime ms on the queue (or None if the output diverges). A wedged GPU never returns -- the
    parent kills this process at the timeout; an IMA crash exits nonzero. Either way the candidate
    scores INVALID without poisoning the factory's own CUDA context."""
    try:
        task = replay.load_task(task_dir)
        replay._set_ptxas(task)
        kernel = replay.load_kernel(task_dir, task)
        args, tensors = replay.build_args(task)
        replay.run_once(kernel, task, args, None)  # no-ACF reference (self-consistency)
        ref = [t.detach().clone() for t in tensors]
        replay.run_once(kernel, task, args, acf_path)
        for r, cur in zip(ref, tensors):
            denom = max(r.float().abs().max().item(), 1e-9)
            rel = (cur.float() - r.float()).abs().max().item() / denom
            if not (rel == rel and rel <= REL_TOL):  # NaN-safe
                q.put(None)
                return
        q.put(replay.benchmark(kernel, task, args, acf_path, warmup=warmup, rep=rep))
    except Exception:
        try:
            q.put(None)
        except Exception:
            pass


def _isolated_bench(task_dir, acf_path, timeout, warmup=25, rep=100):
    """Returns the candidate's ms (float), or None if it diverged / crashed / wedged (timeout).
    Every candidate is evaluated in its own spawn subprocess so a bad ACF can only kill its child."""
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_eval_target, args=(task_dir, acf_path, warmup, rep, q))
    p.start()
    p.join(timeout)
    if p.is_alive():  # wedged on the GPU -> kill it
        p.terminate()
        p.join()
        return None
    if p.exitcode != 0:  # crashed (e.g. IMA corrupts the context -> nonzero exit)
        return None
    try:
        return q.get_nowait()
    except Exception:
        return None


def run_factory(task_dir, generations=2, pool_size=8, ss_bin=DEFAULT_SS_BIN):
    # The CompileIQ engine is proprietary and not vendored; fail fast with how to get it.
    try:
        from compileiq.ciq import Search
        from compileiq.search_spaces.compilers import LocalSearchSpaceBin
        from compileiq.types import INVALID_SCORE, SearchConfiguration
        from compileiq.utils.gpu import gpu_benchmark_mode
        from compileiq.utils.helpers import save_compiler_config
    except ImportError as e:
        raise RuntimeError("compile_iq factory needs the CompileIQ engine (the proprietary `compileiq` "
                           "package), not vendored here. Install the internal Evo wheel "
                           "(`pip install <compileiq-evo-wheel>`) or build from a CompileIQ checkout "
                           f"(`git lfs install && git lfs pull && pip install .`). Import error: {e}") from e

    task = replay.load_task(task_dir)
    replay._set_ptxas(task)

    ptxas = replay.find_ptxas()
    if not ptxas or not replay._ptxas_ge_133(ptxas):
        raise RuntimeError(f"compile_iq factory needs ptxas >= 13.3 for --apply-controls, found: {ptxas or 'none'}. "
                           "Fix: `pip install nvidia-cuda-nvcc` (auto-discovered), or set "
                           "TRITON_PTXAS_BLACKWELL_PATH to a 13.3+ ptxas.")
    if not ss_bin or not os.path.exists(ss_bin):
        raise FileNotFoundError(
            f"PTXAS search-space bin not set/found: {ss_bin}. Set COMPILE_IQ_SEARCH_SPACE_BIN "
            "(or --search-space-bin) to the ptxas13.3 search-space .bin (a separate NVIDIA artifact).")
    print(f"[factory] ptxas: {ptxas}")
    # ptxas >= 13.3 *assembles* ACF cubins, but a GPU driver older than CUDA 13.3 cannot *run* them
    # -- such candidates wedge at launch. We don't block on this (the per-candidate isolation reaps
    # them as INVALID and the search still completes), but warn so an all-INVALID result is expected.
    drv = replay.driver_cuda_version()
    if drv is not None and not replay.driver_supports_acf():
        print(f"[factory] WARNING: GPU driver supports only CUDA {replay.fmt_cuda_version(drv)} (< 13.3, which is "
              "required to RUN --apply-controls cubins). ACF candidates will likely wedge and be reaped as "
              "INVALID; the search will complete but probably store nothing. Use a >= 13.3 driver for a real HIT.")

    base_ms = _isolated_bench(task_dir, None, max(GENERIC_TIMEOUT, 180))
    if base_ms is None:
        raise RuntimeError("baseline (no-ACF) failed to benchmark in isolation -- check the task replays correctly.")
    print(f"[factory] task={task['ptx_sha256'][:16]} arch={task['arch']} baseline={base_ms:.4f} ms "
          f"(isolated, per-candidate timeout={GENERIC_TIMEOUT}s)")

    def objective(acf: str) -> float:
        with tempfile.NamedTemporaryFile(suffix=".acf", delete=True) as f:
            save_compiler_config(f.name, acf)
            ms = _isolated_bench(task_dir, f.name, GENERIC_TIMEOUT)
            return ms if ms is not None else INVALID_SCORE

    cfg = SearchConfiguration(problem_type="min", generations=generations, pool_size=pool_size)
    # exit_on_failure=False: if every candidate scores INVALID (e.g. no ACF is safe on this driver),
    # return an all-INVALID result instead of raising -- we then report "no valid candidate" and
    # store nothing, rather than crashing the factory.
    tuner = Search(objective_function=objective, search_space=LocalSearchSpaceBin(ss_bin), search_config=cfg,
                   exit_on_failure=False)
    with gpu_benchmark_mode(clock_mhz=1965, raise_on_failure=False):
        results = tuner.start()

    df = results.get_results()
    n = len(df) if df is not None else 0
    # On an all-INVALID run the result set has no valid best -- get_best_result() may return None or
    # a sentinel score ("*"). Handle both so the outcome is always reported and we never try to store
    # a non-numeric "best".
    try:
        best = results.get_best_result()
        best_ms = best.get("score_1", best.get("score")) if best else None
    except Exception:
        best, best_ms = None, None
    if not isinstance(best_ms, (int, float)):
        print(f"[factory] no valid candidate: all {n} candidate(s) unsafe/failed on this driver -- not storing.")
        return None
    speedup = (base_ms / best_ms - 1) * 100 if base_ms and best_ms else None
    print(f"[factory] radius={n} best={best_ms} ms "
          f"({f'{speedup:+.2f}%' if speedup is not None else '?'} vs baseline {base_ms:.4f})")

    # Only publish an ACF that actually beats the no-ACF baseline -- never store a regression
    # (on a miss the consume hook falls back to plain compile, which is faster).
    if best_ms >= base_ms:
        print(f"[factory] no improvement over baseline "
              f"({f'{speedup:+.2f}%' if speedup is not None else '?'}) -- not storing.")
        return None
    with tempfile.NamedTemporaryFile(suffix=".acf", delete=False) as f:
        save_compiler_config(f.name, best["params"])
        acf_bytes = open(f.name, "rb").read()
    os.unlink(f.name)
    meta = {
        "ptx_sha256": task["ptx_sha256"], "arch": task["arch"], "baseline_ms": base_ms, "best_ms": best_ms,
        "speedup_pct": speedup, "radius": int(n)
    }
    p = store.write_acf(task["ptx_sha256"], task["arch"], acf_bytes, meta)
    print(f"[factory] wrote ACF ({speedup:+.2f}%) -> {p}")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dir")
    ap.add_argument("--generations", type=int, default=2)
    ap.add_argument("--pool-size", type=int, default=8)
    ap.add_argument("--search-space-bin", default=DEFAULT_SS_BIN)
    a = ap.parse_args()
    run_factory(a.task_dir, a.generations, a.pool_size, a.search_space_bin)


if __name__ == "__main__":
    main()
