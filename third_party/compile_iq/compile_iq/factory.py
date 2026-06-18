"""Stage 2 — ACF factory. Reads a collected task, drives a CompileIQ search over ptxas Advanced
Control Files (ACFs) by replaying the kernel per candidate, and writes the best ACF to the store.
Naive (non-warp-specialized) kernels only; WS/TMA kernels are a later diff.

    python -m compile_iq.factory <task_dir> [--generations N] [--pool-size N]
                                            [--search-space-bin PATH] [--smoke-test] [--force]

--smoke-test runs a tiny bounded search and stores the best (still validated) candidate REGARDLESS of
speedup, so the downstream consume hook has an ACF to apply end-to-end (NOT a validated speedup).
Reruns keep the best: an existing stored ACF is not overwritten unless this run's best is strictly
faster (or --force), so a smaller/unluckier rerun can't regress the store.

Correctness is checked by SELF-CONSISTENCY (an ACF must not change results vs the no-ACF run), so it
is op-agnostic. Each candidate's ACF is applied via the `ptx_options` launch kwarg (-> opt.ptx_options
-> ptxas), the SAME path the consume hook uses, so what is benchmarked is byte-identical to what
consume produces. Every candidate is evaluated IN AN ISOLATED SPAWN SUBPROCESS with a timeout: a bad
ACF can wedge the GPU (cuda.synchronize never returns), and CompileIQ evaluates the objective in an
in-process thread pool where a wedged thread can't be killed -- only killing the child process
reclaims the context. A candidate that wedges (timeout), crashes, or diverges scores INVALID and the
search keeps moving; only a validated ACF that beats the no-ACF baseline is stored.
"""

import argparse
import os
import tempfile

from . import replay, store

# PTXAS search-space bin: a separate NVIDIA CompileIQ artifact (not vendored here).
DEFAULT_SS_BIN = os.environ.get("COMPILE_IQ_SEARCH_SPACE_BIN")
REL_TOL = 1e-2
# Per-candidate wall-clock budget for the isolated subprocess (spawn + import + compile + bench of one
# ACF). A candidate whose ACF wedges the GPU never returns; its subprocess is killed at this timeout
# and scored INVALID so the search keeps moving instead of hanging.
GENERIC_TIMEOUT = int(os.environ.get("COMPILE_IQ_GENERIC_TIMEOUT", "90"))


def _eval_target(task_dir, acf_path, warmup, rep, q):
    """Run in an isolated spawn subprocess: load the task, replay no-ACF (reference) then with the ACF
    applied via the ptx_options launch kwarg, check self-consistency, and benchmark. Puts the runtime
    ms on the queue (or None if the output diverges). A wedged GPU never returns -- the parent kills
    this process at the timeout; a crash exits nonzero. Either way the candidate scores INVALID
    without poisoning the factory's own CUDA context."""
    try:
        task = replay.load_task(task_dir)
        replay._set_ptxas(task)
        kernel = replay.load_kernel(task_dir, task)
        args, tensors = replay.build_args(task)
        replay.run_once(kernel, task, args, None)  # no-ACF reference (self-consistency)
        ref = [t.detach().clone() for t in tensors]
        replay.run_once(kernel, task, args, acf_path)  # ACF applied via ptx_options kwarg
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
    """Returns the candidate's ms (float), or None if it diverged / crashed / wedged (timeout). Each
    candidate runs in its own spawn subprocess so a bad ACF can only kill its child."""
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


def run_factory(task_dir, generations=2, pool_size=8, ss_bin=DEFAULT_SS_BIN, smoke_test=False, force=False):
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

    if smoke_test:
        generations, pool_size = 1, 6  # small bound (pool must be > 5)
        print("[factory] *** SMOKE TEST MODE ***: small bounded search (generations=1 pool-size=6); "
              "will store the best validated candidate REGARDLESS of speedup, only to exercise consume.")

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
    # exit_on_failure=False: if EVERY candidate scores INVALID (e.g. all ACFs wedge), return an
    # all-INVALID result instead of raising -- we then report "no valid candidate" and store nothing.
    tuner = Search(objective_function=objective, search_space=LocalSearchSpaceBin(ss_bin), search_config=cfg,
                   exit_on_failure=False)
    with gpu_benchmark_mode(clock_mhz=1965, raise_on_failure=False):
        results = tuner.start()

    df = results.get_results()
    n = len(df) if df is not None else 0
    # On an all-INVALID run there is no valid best -- get_best_result() may return None or a sentinel
    # score; handle both so we always report and never try to store a non-numeric "best".
    try:
        best = results.get_best_result()
        best_ms = best.get("score_1", best.get("score")) if best else None
    except Exception:
        best, best_ms = None, None
    if not isinstance(best_ms, (int, float)):
        print(f"[factory] no valid candidate among {n} -- every ACF wedged/diverged, nothing stored. "
              "If ALL candidates wedge, the GPU driver is most likely < CUDA 13.3 (cannot RUN "
              "--apply-controls cubins) -- try a >= 13.3-driver box; or the search-space bin is "
              "incompatible with this kernel.")
        return None
    speedup = (base_ms / best_ms - 1) * 100
    print(f"[factory] radius={n} best={best_ms:.4f} ms ({speedup:+.2f}% vs baseline {base_ms:.4f})")

    # Normally publish only an ACF that actually beats the baseline -- on a miss the consume hook falls
    # back to plain compile, which is faster than a regression. Under --smoke-test we store the best
    # (still validated) candidate UNCONDITIONALLY so the consume side has something to apply.
    if smoke_test:
        print(f"[factory] *** SMOKE TEST ***: storing best candidate REGARDLESS of speedup "
              f"({speedup:+.2f}%) -- validated (self-consistent, non-wedging) but NOT necessarily "
              "faster; it exists only so the consume hook can be exercised. Not a tuning result.")
    elif best_ms >= base_ms:
        print(f"[factory] no improvement ({speedup:+.2f}%) -- not storing.")
        return None

    # Keep-the-best: never regress an already-stored ACF that is as fast or faster than this run's best
    # (best_ms measured under locked clocks, so comparable across runs). Pass --force to override.
    prev = store.read_meta(task["ptx_sha256"], task["arch"])
    prev_ms = prev.get("best_ms") if prev else None
    if isinstance(prev_ms, (int, float)) and not force and prev_ms <= best_ms:
        print(f"[factory] kept existing stored ACF (best={prev_ms:.4f} ms) -- this run's best "
              f"{best_ms:.4f} ms is not faster; pass --force to overwrite.")
        return None

    with tempfile.NamedTemporaryFile(suffix=".acf", delete=False) as f:
        save_compiler_config(f.name, best["params"])
        acf_bytes = open(f.name, "rb").read()
    os.unlink(f.name)
    meta = {
        "ptx_sha256": task["ptx_sha256"], "arch": task["arch"], "baseline_ms": base_ms, "best_ms": best_ms,
        "speedup_pct": speedup, "radius": int(n), "smoke_test": smoke_test
    }
    p = store.write_acf(task["ptx_sha256"], task["arch"], acf_bytes, meta)
    verb = "overwrote" if prev_ms is not None else "wrote"
    print(f"[factory] {'SMOKE-TEST ' if smoke_test else ''}{verb} ACF ({speedup:+.2f}%) -> {p}")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dir")
    ap.add_argument("--generations", type=int, default=2)
    ap.add_argument("--pool-size", type=int, default=8)
    ap.add_argument("--search-space-bin", default=DEFAULT_SS_BIN)
    ap.add_argument("--smoke-test", action="store_true",
                    help="small bounded search; store the best validated ACF regardless of speedup to exercise consume")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing stored ACF even if it is as fast or faster (default: keep best)")
    a = ap.parse_args()
    run_factory(a.task_dir, a.generations, a.pool_size, a.search_space_bin, smoke_test=a.smoke_test, force=a.force)


if __name__ == "__main__":
    main()
