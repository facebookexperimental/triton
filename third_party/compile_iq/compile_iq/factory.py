"""Stage 2b — ACF factory. Offline binary: read a compileIQ task, drive a CompileIQ
search over ptxas ACFs, and write the best ACF to the ACF store keyed by sha256(PTX).

    python -m compile_iq.factory <task_dir> [--generations N] [--pool-size N]
                                            [--search-space-bin PATH]
                                            [--store-only-if-faster]

Two tuning modes:
  * REAL-LAUNCH (default, used when the task's source module exposes a `matmul`
    wrapper): tune by driving the *actual* production entrypoint, so the PTX/SASS
    being tuned is byte-identical to what `consume` will apply the ACF to. This is
    required for warp-specialized / TMA kernels, whose compiled SASS depends on
    launch state (resolved num_warps, descriptor pre-hooks, ...) that generic replay
    cannot reconstruct -- tuning the reconstructed launch produces a *different* PTX
    and the resulting ACF, filed under the production key, corrupts the real kernel
    (illegal memory access). See replay.py for the generic fallback.
  * GENERIC REPLAY (fallback): reconstruct the launch from the task. Guarded by a
    PARITY check -- the factory refuses to store an ACF unless the PTX it actually
    tuned hashes to the task key, so a divergent replay can never mislabel an ACF.

Correctness is checked by SELF-CONSISTENCY (an ACF must not change results vs the
no-ACF run), not by any kernel-specific reference -- so it is op-agnostic. Each
candidate is benchmarked at production-ish rep counts with a post-bench correctness
re-check, so unsafe ACFs crash/mis-compute DURING tuning and score INVALID.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import tempfile

import torch

from . import replay, store

# PTXAS search-space bin: a separate NVIDIA CompileIQ artifact (not vendored here).
DEFAULT_SS_BIN = os.environ.get("COMPILE_IQ_SEARCH_SPACE_BIN")
REL_TOL = 1e-2
# Admission gate intensity (env-overridable). Production-ish so load-dependent hazards
# surface during tuning; the deterministic ones surface on the very first launch.
GATE_WARMUP = int(os.environ.get("COMPILE_IQ_GATE_WARMUP", "1000"))
GATE_REP = int(os.environ.get("COMPILE_IQ_GATE_REP", "1000"))
# Number of heavy validation rounds per candidate. A single 4000-launch pass can get
# LUCKY on a flaky (race-y) ACF; repeating it catches probabilistic hazards. >1 is the
# difference between "looked safe once" and "robustly safe".
GATE_ROUNDS = int(os.environ.get("COMPILE_IQ_GATE_ROUNDS", "3"))
# Final consume-faithful validation (the real safety gate): heavier, at production conditions.
# The in-search gate only ranks candidates; this one decides admission.
VAL_WARMUP = int(os.environ.get("COMPILE_IQ_VAL_WARMUP", "2000"))
VAL_REP = int(os.environ.get("COMPILE_IQ_VAL_REP", "2000"))
VAL_ROUNDS = int(os.environ.get("COMPILE_IQ_VAL_ROUNDS", "2"))

# Env channel to hand the task dir to IsoMultiProcessWorker subprocesses (spawned, so
# they re-import this module and rebuild state from the task rather than unpickling it).
_TASK_ENV = "COMPILE_IQ_FACTORY_TASK"
# Shared dir where each candidate's EXACT serialized ACF bytes are cached (keyed by its
# params), so we store the bytes that were actually validated -- not a re-serialization of
# best["params"], which may not round-trip to the same bytes.
_ACF_CACHE_ENV = "COMPILE_IQ_ACF_CACHE"


def _params_key(params):
    try:
        s = json.dumps(params, sort_keys=True, default=str)
    except Exception:
        s = repr(params)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:32]


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------
def _restore_env(task):
    """Re-apply the kernel-selecting env captured at collection (e.g. TLX_GEMM_USE_HEURISTIC),
    so the factory compiles the SAME single config -- fast, and PTX-key-faithful."""
    for k, v in (task.get("env") or {}).items():
        os.environ[k] = v


def _import_source(task_dir, task):
    src = os.path.join(task_dir, task.get("source_file", "source.py"))
    spec = importlib.util.spec_from_file_location("ciq_factory_src", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_module(task_dir, task):
    """Prefer the SAME installed module the kernel was defined in (Triton bakes module
    identity into the PTX, so importing the source.py copy yields a different SASS/key).
    Fall back to the source copy only if the module isn't importable here."""
    name = task.get("module")
    if name:
        try:
            return importlib.import_module(name)
        except Exception:
            pass
    return _import_source(task_dir, task)


def _cache_ptx_shas(cache_dir):
    """sha256 of every .ptx Triton wrote under cache_dir (robust across Triton versions)."""
    import glob
    shas = set()
    for p in glob.glob(os.path.join(cache_dir, "**", "*.ptx"), recursive=True):
        try:
            with open(p) as f:
                shas.add(store.ptx_sha256(f.read()))
        except Exception:
            pass
    return shas


def _real_launch_ptx_shas(st):
    """Compile the real wrapper into a FRESH temp Triton cache (cache enabled so the PTX is
    written to disk) and return the sha(s) of the emitted PTX -- the production SASS the ACF
    will actually be applied to."""
    import tempfile
    cache_dir = tempfile.mkdtemp(prefix="ciq_parity_")
    prev_cache = os.environ.get("TRITON_CACHE_DIR")
    prev_ac = os.environ.pop("TRITON_ALWAYS_COMPILE", None)  # let it write to the cache
    os.environ["TRITON_CACHE_DIR"] = cache_dir
    try:
        os.environ.pop("PTXAS_OPTIONS", None)
        st["wrapper"](st["a"], st["b"])
        torch.cuda.synchronize()
        return _cache_ptx_shas(cache_dir)
    finally:
        if prev_cache is not None:
            os.environ["TRITON_CACHE_DIR"] = prev_cache
        else:
            os.environ.pop("TRITON_CACHE_DIR", None)
        if prev_ac is not None:
            os.environ["TRITON_ALWAYS_COMPILE"] = prev_ac


def _gemm_inputs(task, device="cuda"):
    """Reconstruct (a, b) for a `matmul(a, b)` wrapper from the task's M/N/K + dtype."""
    scalars = {a["name"]: a["value"] for a in task["args"] if a["kind"] == "scalar"}
    M, N, K = int(scalars["M"]), int(scalars["N"]), int(scalars["K"])
    dtype = None
    for a in task["args"]:
        if a["kind"] == "tensor_descriptor":
            dtype = replay._DT[a["base_dtype"]]
            break
        if a["kind"] == "tensor":
            dtype = replay._DT[a["dtype"]]
            break
    dtype = dtype or torch.float16
    torch.manual_seed(0)
    a = torch.randn((M, K), device=device, dtype=dtype)
    b = torch.randn((K, N), device=device, dtype=dtype)
    return a, b


# --------------------------------------------------------------------------------------
# Real-launch path (module-level so it survives IsoMultiProcessWorker spawn)
# --------------------------------------------------------------------------------------
_RL = {}  # per-process lazy cache


def _rl_setup():
    if _RL:
        return _RL
    task_dir = os.environ[_TASK_ENV]
    task = replay.load_task(task_dir)
    replay._set_ptxas(task)  # also sets TRITON_ALWAYS_COMPILE=1
    _restore_env(task)  # single-config launch in the worker subprocess too
    module = _load_module(task_dir, task)
    wrapper = getattr(module, "matmul")
    a, b = _gemm_inputs(task)
    # cuBLAS reference (instant; no extra Triton compile per worker). The real-launch path
    # already assumes a GEMM `matmul(a, b)` wrapper, so a@b is the right correctness oracle
    # (an ACF that changes numerics beyond tolerance is rejected).
    ref = torch.matmul(a, b).detach()
    torch.cuda.synchronize()
    _RL.update(task=task, module=module, wrapper=wrapper, a=a, b=b, ref=ref)
    return _RL


def _rl_objective(acf):
    """Tune against the REAL launch: apply the ACF, run the production wrapper, require
    self-consistency vs the no-ACF output before AND after a heavy benchmark."""
    from compileiq.types import INVALID_SCORE
    from compileiq.utils.helpers import save_compiler_config

    import triton

    st = _rl_setup()
    wrapper, a, b, ref = st["wrapper"], st["a"], st["b"], st["ref"]

    def _ok(c):
        denom = max(ref.float().abs().max().item(), 1e-9)
        rel = (c.float() - ref.float()).abs().max().item() / denom
        return rel == rel and rel <= REL_TOL  # NaN-safe

    with tempfile.NamedTemporaryFile(suffix=".acf", delete=True) as f:
        save_compiler_config(f.name, acf)
        # Cache the EXACT bytes this candidate is validated with, keyed by its params, so
        # the winner is stored verbatim (not re-serialized from best["params"]).
        _cache = os.environ.get(_ACF_CACHE_ENV)
        if _cache:
            try:
                os.makedirs(_cache, exist_ok=True)
                with open(f.name, "rb") as _r:
                    _bytes = _r.read()
                with open(os.path.join(_cache, _params_key(acf) + ".acf"), "wb") as _w:
                    _w.write(_bytes)
            except Exception:
                pass
        os.environ["PTXAS_OPTIONS"] = f"--apply-controls={f.name}"
        try:
            # STRESS gate: multiple heavy rounds, each with a post-bench correctness re-check,
            # so a probabilistic (race-y) ACF crashes/mis-computes during tuning -> rejected.
            ms = None
            for _ in range(GATE_ROUNDS):
                c = wrapper(a, b)
                torch.cuda.synchronize()
                if not _ok(c):
                    return INVALID_SCORE
                m = triton.testing.do_bench(lambda: wrapper(a, b), warmup=GATE_WARMUP, rep=GATE_REP,
                                            return_mode="mean")
                ms = m if ms is None else min(ms, m)
                c2 = wrapper(a, b)  # post-bench re-check
                torch.cuda.synchronize()
                if not _ok(c2):
                    return INVALID_SCORE
            return ms
        except Exception as e:
            print(f"[factory] candidate failed: {type(e).__name__}: {e}")
            return INVALID_SCORE
        finally:
            os.environ.pop("PTXAS_OPTIONS", None)


def _validate_target(acf_path, task_dir, warmup, rep, rounds, q):
    """Run in an isolated spawn subprocess under CONSUME-FAITHFUL conditions: production
    clocks (NO gpu_benchmark_mode), plain process, production reps. Puts the measured ms if
    the ACF survives `rounds` heavy rounds with correct output, else None. A crash (IMA)
    kills this process with a nonzero exit code -- handled by the parent."""
    try:
        os.environ[_TASK_ENV] = task_dir
        _RL.clear()
        st = _rl_setup()
        import torch
        import triton
        wrapper, a, b, ref = st["wrapper"], st["a"], st["b"], st["ref"]

        def _ok(c):
            denom = max(ref.float().abs().max().item(), 1e-9)
            rel = (c.float() - ref.float()).abs().max().item() / denom
            return rel == rel and rel <= REL_TOL

        # NOTE: PTXAS_OPTIONS is read by Triton's knobs at IMPORT time, so it must already be
        # in the env when this fresh subprocess imports triton. The parent sets it BEFORE spawn
        # (see _consume_validate). Do NOT set it here -- that would be too late (no effect).
        if "--apply-controls" not in (os.environ.get("PTXAS_OPTIONS") or ""):
            q.put(None)  # ACF not actually applied -> refuse to validate (would be a false pass)
            return
        ms = None
        for _ in range(rounds):
            c = wrapper(a, b)
            torch.cuda.synchronize()
            if not _ok(c):
                q.put(None)
                return
            m = triton.testing.do_bench(lambda: wrapper(a, b), warmup=warmup, rep=rep, return_mode="mean")
            ms = m if ms is None else min(ms, m)
            c2 = wrapper(a, b)
            torch.cuda.synchronize()
            if not _ok(c2):
                q.put(None)
                return
        q.put(ms)
    except Exception:
        try:
            q.put(None)
        except Exception:
            pass


def _consume_validate(acf_path, task_dir, warmup, rep, rounds, timeout=150):
    """Returns the survived ms (float) or None. Isolated so an IMA can't poison the factory.
    Sets PTXAS_OPTIONS in the env BEFORE spawning so the child applies the ACF at its fresh
    triton import (knobs read PTXAS_OPTIONS once, at import)."""
    import multiprocessing as mp
    prev = os.environ.get("PTXAS_OPTIONS")
    os.environ["PTXAS_OPTIONS"] = f"--apply-controls={acf_path}"
    try:
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(target=_validate_target, args=(acf_path, task_dir, warmup, rep, rounds, q))
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            p.join()
            return None
        if p.exitcode != 0:  # crashed (IMA corrupts the context -> nonzero exit)
            return None
        try:
            return q.get_nowait()
        except Exception:
            return None
    finally:
        if prev is not None:
            os.environ["PTXAS_OPTIONS"] = prev
        else:
            os.environ.pop("PTXAS_OPTIONS", None)


def _run_real_launch(task_dir, task, generations, pool_size, ss_bin, store_only_if_faster):
    from compileiq.ciq import Search
    from compileiq.search_spaces.compilers import LocalSearchSpaceBin
    from compileiq.types import SearchConfiguration
    from compileiq.utils.gpu import gpu_benchmark_mode
    from compileiq.utils.helpers import save_compiler_config
    from compileiq.worker import IsoMultiProcessWorker

    import triton

    os.environ[_TASK_ENV] = task_dir
    os.environ.setdefault("CIQ_PROCESS_MODE", "spawn")

    st = _rl_setup()  # builds inputs + cuBLAS reference in THIS process
    os.environ.pop("PTXAS_OPTIONS", None)

    # PARITY GUARD: the real launch must compile to the task key, else storing under it
    # would mislabel the ACF (-> illegal memory access at consume).
    shas = _real_launch_ptx_shas(st)
    if not shas:
        raise RuntimeError("parity check could not recover the real-launch PTX -- refusing to store "
                           "(cannot guarantee the ACF is keyed to the production SASS).")
    if task["ptx_sha256"] not in shas:
        raise RuntimeError(
            f"parity check failed: real launch produced {[s[:16] for s in shas]} != task "
            f"{task['ptx_sha256'][:16]}. The task was likely collected against a stale Triton cache "
            "or a different ptxas/env. Re-collect with a clean cache + the same ptxas>=13.3, then re-run.")
    print(f"[factory] real-launch parity: OK {task['ptx_sha256'][:16]}")

    base_ms = triton.testing.do_bench(lambda: st["wrapper"](st["a"], st["b"]), warmup=GATE_WARMUP,
                                      rep=GATE_REP, return_mode="mean")
    print(f"[factory] task={task['ptx_sha256'][:16]} arch={task['arch']} baseline={base_ms:.4f} ms "
          f"(real launch, gate warmup/rep={GATE_WARMUP}/{GATE_REP})")

    import tempfile as _tf
    acf_cache = _tf.mkdtemp(prefix="ciq_acfcache_")
    os.environ[_ACF_CACHE_ENV] = acf_cache  # inherited by spawned workers

    cfg = SearchConfiguration(problem_type="min", generations=generations, pool_size=pool_size)
    tuner = Search(objective_function=_rl_objective, search_space=LocalSearchSpaceBin(ss_bin),
                   search_config=cfg, worker_type=IsoMultiProcessWorker)
    with gpu_benchmark_mode(clock_mhz=1965, raise_on_failure=False):
        results = tuner.start(num_workers=1, task_timeout=240)

    df = results.get_results()
    best = results.get_best_result()
    best_ms = best.get("score_1", best.get("score"))
    speedup = (base_ms / best_ms - 1) * 100 if isinstance(best_ms, (int, float)) and best_ms else None
    print(f"[factory] radius={len(df)} best={best_ms} ms "
          f"({f'{speedup:+.2f}%' if speedup is not None else 'no valid candidate'} vs baseline {base_ms:.4f})")

    if not isinstance(best_ms, (int, float)):
        print("[factory] no candidate produced a valid in-search score -- storing nothing.")
        return None

    # FINAL admission gate: re-validate candidates under CONSUME-FAITHFUL conditions -- a
    # plain process at production clocks (NO gpu_benchmark_mode), in an ISOLATED subprocess.
    # The in-search gate runs under gpu_benchmark_mode inside a worker, which we found can
    # pass an ACF that then crashes a plain consume run. Store the FASTEST candidate that
    # survives this; if none survive, store NOTHING (consume -> safe MISS -> baseline).
    try:
        records = df.to_dict("records")
    except Exception:
        records = []

    def _rec_score(r):
        return r.get("score_1", r.get("score"))

    cand_paths = []
    for r in records:
        params = r.get("params")
        if params is None:
            continue
        path = os.path.join(acf_cache, _params_key(params) + ".acf")
        if os.path.exists(path):
            s = _rec_score(r)
            cand_paths.append((s if isinstance(s, (int, float)) else float("inf"), path))
    if not cand_paths:  # fallback: validate whatever ACFs we cached (scores unknown)
        import glob
        cand_paths = [(float("inf"), p) for p in sorted(glob.glob(os.path.join(acf_cache, "*.acf")))]
    cand_paths.sort(key=lambda x: x[0])

    print(f"[factory] consume-faithful final validation of {len(cand_paths)} candidate(s) "
          f"(isolated, no benchmark-mode, {VAL_WARMUP}/{VAL_REP} x {VAL_ROUNDS} rounds)...")
    survivors = []
    for score, path in cand_paths:
        ms2 = _consume_validate(path, task_dir, VAL_WARMUP, VAL_REP, VAL_ROUNDS)
        s_str = f"{score:.4f}" if score != float("inf") else "?"
        print(f"[factory]   cand in-search~{s_str} ms: "
              f"{f'SURVIVED {ms2:.4f} ms' if ms2 else 'REJECTED (crash/incorrect under consume conditions)'}",
              flush=True)
        if ms2 is not None:
            survivors.append((ms2, path))
    if not survivors:
        print("[factory] NO ACF survived consume-faithful validation -- storing NOTHING "
              "(safe: consume will MISS -> baseline). This kernel has no robustly-safe ACF in the search space.")
        return None
    survivors.sort()
    ms2, path = survivors[0]
    if store_only_if_faster and ms2 >= base_ms:
        print(f"[factory] fastest SAFE ACF ({(base_ms / ms2 - 1) * 100:+.2f}%) does not beat baseline "
              "-- NOT storing (--store-only-if-faster).")
        return None
    acf_bytes = open(path, "rb").read()
    speedup2 = (base_ms / ms2 - 1) * 100
    meta = {"ptx_sha256": task["ptx_sha256"], "arch": task["arch"], "mode": "real_launch",
            "baseline_ms": base_ms, "validated_ms": ms2, "speedup_pct": speedup2,
            "radius": int(len(df)), "gate_warmup": GATE_WARMUP, "gate_rep": GATE_REP,
            "gate_rounds": GATE_ROUNDS, "consume_validated": True}
    p = store.write_acf(task["ptx_sha256"], task["arch"], acf_bytes, meta)
    print(f"[factory] wrote CONSUME-VALIDATED ACF ({speedup2:+.2f}% vs baseline) -> {p}")
    return p


# --------------------------------------------------------------------------------------
# Generic replay path (fallback) -- now self-consistency + parity-guarded
# --------------------------------------------------------------------------------------
def _run_generic_replay(task_dir, task, generations, pool_size, ss_bin, store_only_if_faster):
    from compileiq.ciq import Search
    from compileiq.search_spaces.compilers import LocalSearchSpaceBin
    from compileiq.types import INVALID_SCORE, SearchConfiguration
    from compileiq.utils.gpu import gpu_benchmark_mode
    from compileiq.utils.helpers import save_compiler_config

    kernel = replay.load_kernel(task_dir, task)

    # PARITY GUARD: does replay recompile to the task key? If not, generic replay tunes
    # different SASS than production and must NOT store (would mislabel the ACF).
    a0, _ = replay.build_args(task)
    replay.run_once(kernel, task, a0, None)
    # Recover the recompiled PTX via the kernel's own cache (robust across Autotuner/JITFunction).
    rk = replay._raw_jit(kernel)
    ptx = None
    cache = getattr(rk, "cache", None) or {}
    for dev in (cache.values() if isinstance(cache, dict) else []):
        for ck in (dev.values() if isinstance(dev, dict) else []):
            asm = getattr(ck, "asm", None)
            if isinstance(asm, dict) and "ptx" in asm:
                ptx = asm["ptx"]
    replay_sha = store.ptx_sha256(ptx) if ptx else None
    if replay_sha and replay_sha != task["ptx_sha256"]:
        raise RuntimeError(
            f"parity check failed: generic-replay PTX {replay_sha[:16]} != task {task['ptx_sha256'][:16]}. "
            "Generic replay does not reproduce production SASS for this kernel (common for warp-spec/TMA "
            "kernels). Refusing to store -- use the real-launch path (expose a `matmul` wrapper).")
    print(f"[factory] generic-replay parity: {'OK ' + replay_sha[:16] if replay_sha else 'unverified'}")

    def _bench(acf_path):
        args, tensors = replay.build_args(task)
        replay.run_once(kernel, task, args, None)  # no-ACF reference (self-consistency)
        ref = [t.detach().clone() for t in tensors]
        replay.run_once(kernel, task, args, acf_path)
        for r, cur in zip(ref, tensors):
            denom = max(r.float().abs().max().item(), 1e-9)
            rel = (cur.float() - r.float()).abs().max().item() / denom
            if not (rel == rel and rel <= REL_TOL):
                return None
        return replay.benchmark(kernel, task, args, acf_path, warmup=GATE_WARMUP, rep=GATE_REP)

    base_ms = _bench(None)
    print(f"[factory] task={task['ptx_sha256'][:16]} arch={task['arch']} baseline={base_ms:.4f} ms (generic replay)")

    def objective(acf):
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

    if not isinstance(best_ms, (int, float)) or (store_only_if_faster and (base_ms is None or best_ms >= base_ms)):
        print("[factory] not storing (no safe improvement / --store-only-if-faster).")
        return None
    with tempfile.NamedTemporaryFile(suffix=".acf", delete=False) as f:
        save_compiler_config(f.name, best["params"])
        acf_bytes = open(f.name, "rb").read()
    os.unlink(f.name)
    meta = {"ptx_sha256": task["ptx_sha256"], "arch": task["arch"], "mode": "generic_replay",
            "baseline_ms": base_ms, "best_ms": best_ms, "speedup_pct": speedup, "radius": int(len(df))}
    p = store.write_acf(task["ptx_sha256"], task["arch"], acf_bytes, meta)
    print(f"[factory] wrote ACF ({speedup:+.2f}%) -> {p}")
    return p


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def run_factory(task_dir, generations=2, pool_size=8, ss_bin=DEFAULT_SS_BIN, store_only_if_faster=False):
    # The CompileIQ engine is proprietary and not vendored; fail fast with how to get it.
    try:
        import compileiq.ciq  # noqa: F401
    except ImportError as e:
        raise RuntimeError("compile_iq factory needs the CompileIQ engine (the `compileiq` package), which is "
                           "proprietary and not vendored here. Install it via the internal Evo wheel "
                           "(`pip install <compileiq-evo-wheel>`), or from a CompileIQ source checkout "
                           "(`git lfs install && git lfs pull && pip install .`). "
                           f"Original import error: {e}") from e

    task = replay.load_task(task_dir)
    replay._set_ptxas(task)
    _restore_env(task)  # reproduce the collected single-config launch (TLX_GEMM_USE_HEURISTIC, ...)

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

    # Prefer the real-launch path when the task's source exposes a `matmul` wrapper:
    # it is the only path that guarantees PTX/SASS parity with production.
    module = _load_module(task_dir, task)
    if hasattr(module, "matmul"):
        print("[factory] mode: real-launch (production wrapper found)")
        return _run_real_launch(task_dir, task, generations, pool_size, ss_bin, store_only_if_faster)
    print("[factory] mode: generic-replay (no wrapper found; parity-guarded)")
    return _run_generic_replay(task_dir, task, generations, pool_size, ss_bin, store_only_if_faster)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dir")
    ap.add_argument("--generations", type=int, default=2)
    ap.add_argument("--pool-size", type=int, default=8)
    ap.add_argument("--search-space-bin", default=DEFAULT_SS_BIN)
    ap.add_argument("--store-only-if-faster", action="store_true",
                    help="only store an ACF that beats baseline (default: store the best SAFE ACF, "
                         "so consume HITs and you can verify the SASS changes even if it is not faster)")
    a = ap.parse_args()
    run_factory(a.task_dir, a.generations, a.pool_size, a.search_space_bin, a.store_only_if_faster)


if __name__ == "__main__":
    main()
