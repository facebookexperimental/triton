"""Stage 2b — ACF factory. Offline binary: read a compileIQ task, drive a CompileIQ
search over ptxas ACFs (replaying the kernel per candidate), and write the best ACF
to the ACF store keyed by sha256(PTX).

    python -m compile_iq.factory <task_dir> [--generations N] [--pool-size N]
                                            [--search-space-bin PATH]

Correctness: each candidate's outputs are compared against the no-ACF run (an ACF
must not change numerics); a mismatch or compile failure scores INVALID.
"""

import argparse
import os
import tempfile

import torch

from . import replay, store

# PTXAS search-space bin: a separate NVIDIA CompileIQ artifact (not vendored here).
# Provide it via COMPILE_IQ_SEARCH_SPACE_BIN or --search-space-bin. No path baked in.
DEFAULT_SS_BIN = os.environ.get("COMPILE_IQ_SEARCH_SPACE_BIN")
REL_TOL = 1e-2


def _snapshot(args, tensor_idx):
    return [args[i].detach().clone() for i in tensor_idx]


def _matches(ref, args, tensor_idx):
    for r, i in zip(ref, tensor_idx):
        cur = args[i].float()
        denom = r.float().abs().max()
        rel = (cur - r.float()).abs().max() / (denom if denom > 0 else 1.0)
        if not torch.isfinite(rel) or rel.item() > REL_TOL:
            return False
    return True


def run_factory(task_dir, generations=2, pool_size=8, ss_bin=DEFAULT_SS_BIN):
    # The CompileIQ engine is proprietary and not vendored; fail fast with how to get it.
    try:
        from compileiq.ciq import Search
        from compileiq.search_spaces.compilers import LocalSearchSpaceBin
        from compileiq.types import INVALID_SCORE, SearchConfiguration
        from compileiq.utils.gpu import gpu_benchmark_mode
        from compileiq.utils.helpers import save_compiler_config
    except ImportError as e:
        raise RuntimeError("compile_iq factory needs the CompileIQ engine (the `compileiq` package), which is "
                           "proprietary and not vendored here. Install it via the internal Evo wheel "
                           "(`pip install <compileiq-evo-wheel>`), or from a CompileIQ source checkout "
                           "(`git lfs install && git lfs pull && pip install .`). "
                           f"Original import error: {e}") from e

    task = replay.load_task(task_dir)
    replay._set_ptxas(task)

    # Fail fast with an actionable message if no ACF-capable ptxas is available.
    ptxas = replay.find_ptxas()
    if not ptxas or not replay._ptxas_ge_133(ptxas):
        raise RuntimeError(f"compile_iq factory needs ptxas >= 13.3 for --apply-controls, found: {ptxas or 'none'}. "
                           "Fix: `pip install nvidia-cuda-nvcc` (auto-discovered), or set TRITON_PTXAS_BLACKWELL_PATH "
                           "to a 13.3+ ptxas.")
    if not ss_bin or not os.path.exists(ss_bin):
        raise FileNotFoundError(
            f"PTXAS search-space bin not set/found: {ss_bin}. "
            "Set COMPILE_IQ_SEARCH_SPACE_BIN (or --search-space-bin) to the ptxas13.3 search-space .bin "
            "(a separate NVIDIA CompileIQ artifact).")
    print(f"[factory] ptxas: {ptxas}")

    kernel = replay.load_kernel(task_dir, task)
    args, tensor_idx = replay.build_args(task)

    # reference (no ACF) output + baseline
    replay.run_once(kernel, task, args, None)
    ref = _snapshot(args, tensor_idx)
    base_ms = replay.benchmark(kernel, task, args, None)
    print(f"[factory] task={os.path.basename(task_dir)} arch={task['arch']} "
          f"ptx_sha={task['ptx_sha256'][:16]} baseline={base_ms:.4f} ms")

    def objective(acf: str) -> float:
        with tempfile.NamedTemporaryFile(suffix=".acf", delete=True) as f:
            save_compiler_config(f.name, acf)
            try:
                replay.run_once(kernel, task, args, f.name)
                if not _matches(ref, args, tensor_idx):
                    return INVALID_SCORE
                return replay.benchmark(kernel, task, args, f.name)
            except Exception as e:
                print(f"[factory] candidate failed: {type(e).__name__}: {e}")
                return INVALID_SCORE

    cfg = SearchConfiguration(problem_type="min", generations=generations, pool_size=pool_size)
    tuner = Search(objective_function=objective, search_space=LocalSearchSpaceBin(ss_bin), search_config=cfg)
    with gpu_benchmark_mode(clock_mhz=1965, raise_on_failure=False):
        results = tuner.start()

    best = results.get_best_result()
    best_ms = best.get("score_1", best.get("score"))
    with tempfile.NamedTemporaryFile(suffix=".acf", delete=False) as f:
        save_compiler_config(f.name, best["params"])
        acf_bytes = open(f.name, "rb").read()
    os.unlink(f.name)

    meta = {
        "kernel_name": task["kernel_name"],
        "ptx_sha256": task["ptx_sha256"],
        "arch": task["arch"],
        "ptxas_path": task.get("ptxas_path"),
        "baseline_ms": base_ms,
        "best_ms": best_ms,
        "speedup_pct": (base_ms / best_ms - 1) * 100 if best_ms else None,
        "generations": generations,
        "pool_size": pool_size,
        "task_dir": task_dir,
    }
    p = store.write_acf(task["ptx_sha256"], task["arch"], acf_bytes, meta)
    print(f"[factory] best={best_ms:.4f} ms (baseline {base_ms:.4f}, {meta['speedup_pct']:+.2f}%)")
    print(f"[factory] wrote ACF -> {p}")
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
