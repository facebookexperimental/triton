"""Stage 2 — ACF factory (basic, generic-replay). Reads a collected task, drives a CompileIQ
search over ptxas Advanced Control Files (ACFs) by replaying the kernel per candidate, and writes
the best ACF to the store. Sufficient for simple (non-warp-specialized) kernels like the naive
matmul -- completing the collect -> factory -> consume loop. Warp-specialized / TMA kernels need
the real-launch + parity + consume-faithful machinery added in the next diff.

    python -m compile_iq.factory <task_dir> [--generations N] [--pool-size N]
                                            [--search-space-bin PATH]

Correctness is checked by SELF-CONSISTENCY (an ACF must not change results vs the no-ACF run),
so it is op-agnostic. Each candidate is applied via the ptx_options launch kwarg and benchmarked;
a mismatch or compile failure scores INVALID. Only an ACF that beats the no-ACF baseline is
stored -- on a miss the consume hook falls back to plain compile.
"""

import argparse
import os
import tempfile

from . import replay, store

# PTXAS search-space bin: a separate NVIDIA CompileIQ artifact (not vendored here).
DEFAULT_SS_BIN = os.environ.get("COMPILE_IQ_SEARCH_SPACE_BIN")
REL_TOL = 1e-2


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

    kernel = replay.load_kernel(task_dir, task)

    def _bench(acf_path):
        # Self-consistency: an ACF must not change results vs the no-ACF run (op-agnostic).
        # Returns runtime ms, or None if the output diverges.
        args, tensors = replay.build_args(task)
        replay.run_once(kernel, task, args, None)
        ref = [t.detach().clone() for t in tensors]
        replay.run_once(kernel, task, args, acf_path)
        for r, cur in zip(ref, tensors):
            denom = max(r.float().abs().max().item(), 1e-9)
            rel = (cur.float() - r.float()).abs().max().item() / denom
            if not (rel == rel and rel <= REL_TOL):  # NaN-safe
                return None
        return replay.benchmark(kernel, task, args, acf_path)

    base_ms = _bench(None)
    print(f"[factory] task={task['ptx_sha256'][:16]} arch={task['arch']} baseline={base_ms:.4f} ms")

    def objective(acf: str) -> float:
        with tempfile.NamedTemporaryFile(suffix=".acf", delete=True) as f:
            save_compiler_config(f.name, acf)
            try:
                ms = _bench(f.name)
                return ms if ms is not None else INVALID_SCORE
            except Exception as e:
                print(f"[factory] candidate failed: {type(e).__name__}: {e}")
                return INVALID_SCORE

    cfg = SearchConfiguration(problem_type="min", generations=generations, pool_size=pool_size)
    tuner = Search(objective_function=objective, search_space=LocalSearchSpaceBin(ss_bin), search_config=cfg)
    with gpu_benchmark_mode(clock_mhz=1965, raise_on_failure=False):
        results = tuner.start()

    df = results.get_results()
    best = results.get_best_result()
    best_ms = best.get("score_1", best.get("score"))
    speedup = (base_ms / best_ms - 1) * 100 if isinstance(best_ms, (int, float)) and best_ms else None
    print(f"[factory] radius={len(df)} best={best_ms} ms "
          f"({f'{speedup:+.2f}%' if speedup is not None else 'no valid candidate'} vs baseline {base_ms:.4f})")

    # Only publish an ACF that actually beats the no-ACF baseline -- never store a regression
    # (on a miss the consume hook falls back to plain compile, which is faster).
    if not isinstance(best_ms, (int, float)) or best_ms >= base_ms:
        print(f"[factory] no improvement over baseline "
              f"({f'{speedup:+.2f}%' if speedup is not None else 'no valid candidate'}) -- NOT storing.")
        return None
    with tempfile.NamedTemporaryFile(suffix=".acf", delete=False) as f:
        save_compiler_config(f.name, best["params"])
        acf_bytes = open(f.name, "rb").read()
    os.unlink(f.name)
    meta = {"ptx_sha256": task["ptx_sha256"], "arch": task["arch"], "baseline_ms": base_ms,
            "best_ms": best_ms, "speedup_pct": speedup, "radius": int(len(df))}
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
