from __future__ import annotations

import builtins
import math
import time
import inspect
import hashlib
import json
import statistics
from collections import deque
from functools import cached_property
from typing import Dict, Tuple, List, Optional

from .. import knobs
from .jit import KernelInterface, JITFunction
from .errors import OutOfResources, PTXASError, AutotunerError
from .driver import driver
from .cache import get_cache_manager, triton_key

try:
    from triton._C.libtriton import native_create_autotune_proxy, native_autotune_proxy_insert, native_autotune_proxy_set_grid
except ImportError:
    native_create_autotune_proxy = None
    native_autotune_proxy_insert = None
    native_autotune_proxy_set_grid = None
from triton._C.libtriton import get_cache_invalidating_env_vars


class _OnlineLinearRegression:

    def __init__(self, window_size: int = 299) -> None:
        self.window_size = window_size
        self._values: deque[float] = deque(maxlen=window_size)
        self._sum_x: float = 0.0
        self._sum_y: float = 0.0
        self._sum_xy: float = 0.0
        self._sum_x2: float = 0.0
        self._sum_y2: float = 0.0
        self._count: int = 0

    def reset(self) -> None:
        self._values.clear()
        self._sum_x = 0.0
        self._sum_y = 0.0
        self._sum_xy = 0.0
        self._sum_x2 = 0.0
        self._sum_y2 = 0.0
        self._count = 0

    def add_value(self, y: float) -> None:
        if len(self._values) == self.window_size:
            y_out = self._values[0]
            self._sum_y += y - y_out
            self._sum_y2 += y * y - y_out * y_out
            sum_remaining_ys = self._sum_y - y
            self._sum_xy -= sum_remaining_ys
            self._sum_xy += (float(self._count) - 1.0) * y
        else:
            x = float(self._count)
            self._sum_x += x
            self._sum_y += y
            self._sum_xy += x * y
            self._sum_x2 += x * x
            self._sum_y2 += y * y
            self._count += 1
        self._values.append(y)

    def get_slope_degrees(self) -> float:
        if self._count < 2:
            return float("nan")
        n = float(self._count)
        mean_x = self._sum_x / n
        mean_y = self._sum_y / n
        denom = (self._sum_x2 / n) - mean_x * mean_x
        if abs(denom) < 1e-12:
            return float("nan")
        slope = ((self._sum_xy / n) - mean_x * mean_y) / denom
        if not math.isfinite(slope):
            return float("nan")
        return math.degrees(math.atan(slope))

    def r_squared(self) -> float:
        if self._count < 2:
            return float("nan")
        n = float(self._count)
        mean_x = self._sum_x / n
        mean_y = self._sum_y / n
        ss_tot = (self._sum_y2 / n) - mean_y * mean_y
        if ss_tot < 1e-12:
            return 1.0
        denom = (self._sum_x2 / n) - mean_x * mean_x
        if abs(denom) < 1e-12:
            return float("nan")
        slope = ((self._sum_xy / n) - mean_x * mean_y) / denom
        intercept = mean_y - slope * mean_x
        if not math.isfinite(slope) or not math.isfinite(intercept):
            return float("nan")
        mean_xy = self._sum_xy / n
        mean_xx = self._sum_x2 / n
        ss_tot_m_res = (slope * ((mean_xy - slope * mean_xx) + (mean_xy - intercept * mean_x)) + intercept *
                        (mean_y - slope * mean_x - intercept) + mean_y * (intercept - mean_y))
        return min(max(ss_tot_m_res / ss_tot, 0.0), 1.0)

    def __len__(self) -> int:
        return self._count


class _EntropyCriterion:

    def __init__(
        self,
        max_angle: float = 0.048,
        min_r2: float = 0.36,
        window_size: int = 299,
        min_warmup_samples: int = 20,
        entropy_window_size: int = 500,
    ) -> None:
        self.max_angle = max_angle
        self.min_r2 = min_r2
        self.min_warmup_samples = min_warmup_samples
        self.entropy_window_size = entropy_window_size
        self.total_samples = 0
        self.measurement_window: deque[float] = deque(maxlen=entropy_window_size)
        self.freq_tracker: Dict[float, int] = {}
        self._sum_count_log_count = 0.0
        self._regression = _OnlineLinearRegression(window_size=window_size)

    def reset(self) -> None:
        self.total_samples = 0
        self.measurement_window.clear()
        self.freq_tracker.clear()
        self._sum_count_log_count = 0.0
        self._regression.reset()

    def add_measurement(self, measurement: float) -> None:
        self.total_samples += 1
        if len(self.measurement_window) == self.entropy_window_size:
            old_value = self.measurement_window[0]
            old_count = self.freq_tracker[old_value]
            self._update_entropy_sum(old_count, old_count - 1)
            self.freq_tracker[old_value] -= 1
            if self.freq_tracker[old_value] == 0:
                del self.freq_tracker[old_value]
        old_count = self.freq_tracker.get(measurement, 0)
        self._update_entropy_sum(old_count, old_count + 1)
        self.freq_tracker[measurement] = old_count + 1
        self.measurement_window.append(measurement)
        n = len(self.measurement_window)
        entropy = max(0.0, math.log2(n) - (self._sum_count_log_count / n)) if n > 0 else 0.0
        self._regression.add_value(entropy)

    def is_finished(self) -> bool:
        if self.total_samples < self.min_warmup_samples:
            return False
        if len(self._regression) < 2:
            return False
        if self.total_samples % 2 != 0:
            return False
        slope_deg = self._regression.get_slope_degrees()
        r2 = self._regression.r_squared()
        if not math.isfinite(slope_deg) or not math.isfinite(r2):
            return False
        return slope_deg <= self.max_angle and r2 >= self.min_r2

    def unique_measurements(self) -> int:
        return len(self.freq_tracker)

    def _update_entropy_sum(self, old_count: int, new_count: int) -> None:
        if old_count > 0 and new_count > 0:
            delta = new_count - old_count
            self._sum_count_log_count += new_count * math.log2(1 + delta / old_count) + delta * math.log2(old_count)
        else:
            if old_count > 0:
                self._sum_count_log_count -= old_count * math.log2(old_count)
            if new_count > 0:
                self._sum_count_log_count += new_count * math.log2(new_count)


def _entropy_warmup(kernel_call, clear_cache, torch, entropy_window_size=500, regr_window_size=299, max_samples=10000):
    """Adaptive warmup using entropy convergence. Returns (n_samples, avg_ms)."""
    crit = _EntropyCriterion(
        max_angle=0.048,
        min_r2=0.36,
        window_size=regr_window_size,
        min_warmup_samples=20,
        entropy_window_size=entropy_window_size,
    )
    rounding_factor = 3
    BATCH_SIZE = 50
    last_batch = [0.0] * BATCH_SIZE
    n_written = 0
    counter = 0
    converged = False
    precision_increase = False

    while True:
        batch_size = min(BATCH_SIZE, max_samples - counter)
        start_ev = [torch.cuda.Event(enable_timing=True) for _ in range(batch_size)]
        end_ev = [torch.cuda.Event(enable_timing=True) for _ in range(batch_size)]
        for i in range(batch_size):
            clear_cache()
            start_ev[i].record()
            kernel_call()
            end_ev[i].record()
        n_written = 0
        for i in range(batch_size):
            end_ev[i].synchronize()
            v = round(start_ev[i].elapsed_time(end_ev[i]), rounding_factor)
            last_batch[i] = v
            n_written = i + 1
            crit.add_measurement(v)
            if crit.is_finished():
                converged = True
                break
        counter += n_written
        if converged or counter >= max_samples:
            break
        if counter >= 200 and not precision_increase:
            if crit.unique_measurements() < 20:
                rounding_factor = 4
                crit.entropy_window_size = min(1000, entropy_window_size * 2)
                crit.measurement_window = deque(maxlen=crit.entropy_window_size)
                crit.reset()
                precision_increase = True

    avg_ms = statistics.fmean(last_batch[:n_written]) if n_written > 0 else 0.0
    return counter, avg_ms


def _timed_measurement(kernel_call, clear_cache, n_repeat, torch):
    """Run n_repeat timed iterations and return a float tensor of times in ms."""
    start_ev = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    end_ev = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    for i in range(n_repeat):
        clear_cache()
        start_ev[i].record()
        kernel_call()
        end_ev[i].record()
    torch.cuda.synchronize()
    return torch.tensor([s.elapsed_time(e) for s, e in zip(start_ev, end_ev)], dtype=torch.float)


class Autotuner(KernelInterface):

    def __init__(self, fn, arg_names, configs, key, reset_to_zero, restore_value, pre_hook=None, post_hook=None,
                 prune_configs_by: Optional[Dict] = None, warmup=None, rep=None, use_cuda_graph=False, do_bench=None,
                 cache_results=False, correctness_fn=None, correctness_prune=True):
        """
        :param prune_configs_by: a dict of functions that are used to prune configs, fields:
            'perf_model': performance model used to predicate running time with different configs, returns running time
            'top_k': number of configs to bench
            'early_config_prune': a function used to prune configs. It should have the signature
                `prune_configs_by( configs: List[triton.Config], named_args: Dict[str, Any], **kwargs: Dict[str, Any]) -> List[triton.Config]:`
                and return pruned configs. It should return at least one config.
            'artifact_config_prune': a function used to prune configs by inspecting each config's
                *compiled artifact* (TTGIR/PTX). Unlike 'early_config_prune', which runs before any
                compilation, this hook runs after compiling each config (via run(warmup=True), no
                kernel launch) so it can filter on the generated IR. It should have the signature
                `artifact_config_prune(config: triton.Config, asm: Dict[str, str], metadata) -> bool`
                and return True to KEEP the config. `asm` is the CompiledKernel.asm dict (keys such
                as 'ttir', 'ttgir', 'llir', 'ptx'); `metadata` is CompiledKernel.metadata.
            'equivalence_fn': a function for STATIC bitwise-equivalence pruning. Like
                'artifact_config_prune' it runs after each config is compiled (via run(warmup=True),
                no launch) and inspects the generated TTGIR, but instead of a per-config bool it
                returns a hashable *equivalence key* (e.g. a reduction-order signature derived from
                the reduce op's data layout). The autotuner keeps only configs whose key matches the
                FIRST config's key (the reference order) and prunes the rest, so the surviving set is
                bitwise-equivalent by construction — no kernel launch and no reference output needed.
                Signature: `equivalence_fn(config: triton.Config, asm: Dict[str, str], metadata) -> Hashable`.
                Results are exposed on `self.equivalence_classes` and `self.pruned_by_equivalence`.
            'equivalence_level' + 'equivalence_checkers': the level-selectable form of the above. Set
                'equivalence_level' to "ttgir", "ptx", "both" (or an ordered list of level names) to
                choose the IR level(s) the equivalence check runs at, and pass 'equivalence_checkers'
                as a {level_name: equivalence_fn} registry (e.g. `bitequiv.equivalence.CHECKERS`).
                Triton core stays decoupled — the checkers are injected, not imported. Multiple levels
                run as a two-stage pipeline (e.g. "both" = TTGIR pre-filter, then PTX backstop on the
                survivors). A requested level missing from the registry, or whose checker raises
                NotImplementedError, raises a clear AutotunerError (a requested level is never silently
                skipped). `equivalence_classes` becomes {level: {key: [Config, ...]}}.
        :param correctness_fn: an optional callable used to validate each config's output before it
            is benchmarked. It has the signature `correctness_fn(named_args: Dict[str, Any]) -> bool`
            where `named_args` is the full set of (kernel args + meta-params) AFTER running the config
            once; it should return True if the output is acceptable. Output buffers are
            snapshotted/restored around the check using the same restore_value/reset_to_zero plumbing
            used for benchmarking, and the check runs separately from `do_bench` so its cost never
            affects timing.
        :param correctness_prune: if True (default), configs that fail `correctness_fn` are excluded
            from selection; if False, results are recorded (see `self.correctness_results`) but all
            configs are still benchmarked.
        """
        if not configs:
            self.configs = [Config({}, num_warps=4, num_stages=3, num_ctas=1)]
        else:
            self.configs = configs
        self.keys = key
        self.cache: Dict[Tuple, Config] = {}
        self.arg_names = arg_names
        self.cache_results = (cache_results or knobs.autotuning.cache) and not knobs.runtime.interpret

        # Reset to zero or restore values
        self.reset_to_zero = []
        if reset_to_zero is not None:
            self.reset_to_zero = list(reset_to_zero)
        self.restore_value = []
        if restore_value is not None:
            self.restore_value = list(restore_value)

        # Hook to reset or restore for required tensors
        self.pre_hook = lambda kwargs, reset_only=False: 0
        self.post_hook = lambda kwargs, exception: 0
        self.user_defined_pre_hook = False
        self.user_defined_post_hook = False
        if pre_hook:
            self.pre_hook = pre_hook
            self.user_defined_pre_hook = True
        elif (len(self.reset_to_zero) > 0 or len(self.restore_value) > 0):

            def _pre_hook(kwargs, reset_only=False):
                for name in self.reset_to_zero:
                    kwargs[name].zero_()
                if not reset_only:
                    self.restore_copies = {name: kwargs[name].clone() for name in self.restore_value}

            self.pre_hook = _pre_hook

        if post_hook:
            self.post_hook = post_hook
            self.user_defined_post_hook = True
        elif len(self.restore_value) > 0:

            def _post_hook(kwargs, exception):
                for name in self.restore_value:
                    kwargs[name].copy_(self.restore_copies[name])
                self.restore_copies = {}

            self.post_hook = _post_hook

        self.perf_model = None
        self.configs_top_k = 1.0
        self.early_config_prune = None
        self.artifact_config_prune = None
        self.equivalence_fn = None
        self.equivalence_level = None
        self.equivalence_checkers = None
        if prune_configs_by:
            self.perf_model = prune_configs_by.get("perf_model", self.perf_model)
            self.configs_top_k = prune_configs_by.get("top_k", self.configs_top_k)
            self.early_config_prune = prune_configs_by.get("early_config_prune", self.early_config_prune)
            self.artifact_config_prune = prune_configs_by.get("artifact_config_prune", self.artifact_config_prune)
            self.equivalence_fn = prune_configs_by.get("equivalence_fn", self.equivalence_fn)
            self.equivalence_level = prune_configs_by.get("equivalence_level", self.equivalence_level)
            self.equivalence_checkers = prune_configs_by.get("equivalence_checkers", self.equivalence_checkers)

        # Correctness checking (T3): optional per-config output validation vs a user reference.
        self.correctness_fn = correctness_fn
        self.correctness_prune = correctness_prune
        # {Config: bool} populated each tuning run when correctness_fn is set (success-rate inspection).
        self.correctness_results: Dict[Config, bool] = {}
        # {Config: reason} for configs dropped by artifact_config_prune (T4).
        self.pruned_by_artifact: Dict[Config, str] = {}
        # Static TTGIR bitwise-equivalence pruning (M1): configs dropped for not matching the
        # reference reduction order, and the {equivalence-key: [Config, ...]} classes seen.
        self.pruned_by_equivalence: Dict[Config, str] = {}
        self.equivalence_classes: Dict = {}

        self.fn = fn
        self.base_fn = fn
        while not inspect.isfunction(self.base_fn):
            self.base_fn = self.base_fn.fn

        self._do_bench = do_bench
        self.num_warmups = warmup
        self.num_reps = rep
        self.use_cuda_graph = use_cuda_graph

        # If we got explicitly called via the old interface, raise a warning
        # and proceed with the old behavior.
        if warmup is not None or rep is not None or use_cuda_graph:
            import warnings
            warnings.warn(("warmup, rep, and use_cuda_graph parameters are deprecated. See "
                           "https://github.com/triton-lang/triton/pull/4496 for details."), DeprecationWarning,
                          stacklevel=1)
            if use_cuda_graph:
                from ..testing import do_bench_cudagraph
                self._do_bench = lambda kernel_call, quantiles: do_bench_cudagraph(
                    kernel_call,
                    rep=rep if rep is not None else knobs.autotuning.rep,
                    quantiles=quantiles,
                )
                return

            import triton.testing
            self._do_bench = lambda kernel_call, quantiles: triton.testing.do_bench(
                kernel_call,
                warmup=warmup if warmup is not None else knobs.autotuning.warmup,
                rep=rep if rep is not None else knobs.autotuning.rep,
                quantiles=quantiles,
            )
            return

    @cached_property
    def do_bench(self):
        if self._do_bench is None:
            if knobs.autotuning.use_entropy:
                entropy_bench = self._make_entropy_benchmarker()
                if entropy_bench is not None:
                    return entropy_bench
            benchmarker = driver.active.get_benchmarker()
            warmup = knobs.autotuning.warmup
            rep = knobs.autotuning.rep
            if warmup != 25 or rep != 100:
                print(f"Autotuning benchmarker using warmup={warmup}ms, rep={rep}ms")
            return lambda kernel_call, quantiles: benchmarker(
                kernel_call,
                warmup=warmup,
                rep=rep,
                quantiles=quantiles,
            )
        return self._do_bench

    def _make_entropy_benchmarker(self):
        import torch

        if not torch.cuda.is_available():
            return None

        rep = knobs.autotuning.rep
        _WARMUP_BUDGET_MS = 250

        def entropy_benchmarker(kernel_call, quantiles):
            cache = driver.active.get_empty_cache_for_benchmark()
            clear = lambda: driver.active.clear_cache(cache)

            # Probe kernel time to scale window sizes
            kernel_call()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            clear()
            start.record()
            kernel_call()
            end.record()
            end.synchronize()
            probe_ms = start.elapsed_time(end)

            # Scale windows so wall-clock warmup stays within budget
            probe_ms = max(probe_ms, 0.001)
            entropy_window = min(500, max(50, int(_WARMUP_BUDGET_MS / probe_ms)))
            regr_window = max(20, int(entropy_window * 0.6))

            n_warmup = _entropy_warmup(
                kernel_call,
                clear,
                torch,
                entropy_window_size=entropy_window,
                regr_window_size=regr_window,
            )
            avg_ms = n_warmup[1]
            n_repeat = max(10, int(rep / avg_ms)) if avg_ms > 0 else 100
            times = _timed_measurement(kernel_call, clear, n_repeat, torch)

            if quantiles is not None:
                q = quantiles if isinstance(quantiles, (list, tuple)) else [quantiles]
                ret = torch.quantile(times, torch.tensor(q, dtype=torch.float)).tolist()
                if len(ret) == 1:
                    ret = ret[0]
                return ret
            return times.median().item()

        return entropy_benchmarker

    def _bench(self, *args, config, **meta):
        from ..compiler.errors import CompileTimeAssertionFailure

        verbose = knobs.autotuning.print
        if verbose:
            print(f"Autotuning kernel {self.base_fn.__name__} with config {config}")

        # check for conflicts, i.e. meta-parameters both provided
        # as kwargs and by the autotuner
        conflicts = meta.keys() & config.kwargs.keys()
        if conflicts:
            raise ValueError(f"Conflicting meta-parameters: {', '.join(conflicts)}."
                             " Make sure that you don't re-define auto-tuned symbols.")
        # augment meta-parameters with tunable ones
        current = dict(meta, **config.all_kwargs())
        full_nargs = {**self.nargs, **current}

        def kernel_call():
            if config.pre_hook:
                config.pre_hook(full_nargs)
            self.pre_hook(full_nargs)
            try:
                self.fn.run(
                    *args,
                    **current,
                )
            except Exception as e:
                try:
                    self.post_hook(full_nargs, exception=e)
                finally:
                    # Throw exception raised by `self.fn.run`
                    raise

            self.post_hook(full_nargs, exception=None)

        try:
            return self.do_bench(kernel_call, quantiles=(0.5, 0.2, 0.8))
        except (OutOfResources, CompileTimeAssertionFailure, PTXASError) as e:
            if verbose:
                print(f"Autotuning failed with {e}")
            return [float("inf"), float("inf"), float("inf")]

    def _check_correctness(self, *args, config, **meta):
        """Run one config once (untimed) and validate its output via ``correctness_fn``.

        Returns True if the config's output passes (or if no ``correctness_fn`` is set). The kernel
        is launched a single time; output buffers are snapshotted/restored using the same
        ``pre_hook``/``post_hook`` (``restore_value``/``reset_to_zero``) plumbing used for
        benchmarking, so neither the timing runs nor the final winner launch observe mutated tensors.
        This is intentionally separate from ``do_bench`` so the comparison cost never affects timing.
        """
        from ..compiler.errors import CompileTimeAssertionFailure

        if self.correctness_fn is None:
            return True
        current = dict(meta, **config.all_kwargs())
        full_nargs = {**self.nargs, **current}
        if config.pre_hook:
            config.pre_hook(full_nargs)
        self.pre_hook(full_nargs)
        passed = False
        try:
            self.fn.run(*args, **current)
            passed = bool(self.correctness_fn(full_nargs))
        except (OutOfResources, CompileTimeAssertionFailure, PTXASError) as e:
            # A config that cannot even compile/run is treated as failing the check.
            if knobs.autotuning.print:
                print(f"[autotune] correctness check could not run config {config}: {e}", flush=True)
            passed = False
        finally:
            self.post_hook(full_nargs, exception=None)
        return passed

    def check_disk_cache(self, tuning_key, configs, bench_fn):
        # We can't serialize prehooks, so just give up and run the benchmarks.
        if not tuning_key or any(cfg.pre_hook for cfg in configs):
            bench_fn()
            return False

        from triton.compiler.compiler import make_backend

        fn = self.fn
        while not isinstance(fn, JITFunction):
            fn = fn.fn

        env_vars = get_cache_invalidating_env_vars()
        cache_key = [
            triton_key(),
            make_backend(driver.active.get_current_target()).hash(),
            fn.cache_key,
            str(sorted(env_vars.items())),
            str(tuning_key),
        ] + [str(c) for c in configs]
        cache_key = hashlib.sha256("-".join(cache_key).encode("utf-8")).hexdigest()
        cache = get_cache_manager(cache_key)
        file_name = f"{fn.__name__[:150]}.autotune.json"
        path = cache.get_file(file_name)
        if path:
            with open(path, "r") as cached_configs:
                timings = json.load(cached_configs)["configs_timings"]
                timings = {Config(**config): timing for config, timing in timings}
                self.cache[tuning_key] = builtins.min(timings, key=timings.get)
                self.configs_timings = timings
            return True

        bench_fn()
        cache.put(
            json.dumps({
                "key":
                tuning_key,
                "configs_timings":
                [(config.__dict__, timings) for config, timings in self.configs_timings.items() if not config.pre_hook],
            }), file_name, binary=False)
        return False

    def __getitem__(self, grid):
        """Return C-level AutotuneCacheProxy for fast dispatch if available."""
        # Check if we can use the C-level autotune proxy
        if (native_create_autotune_proxy is not None and getattr(self.fn, 'c_cache', False)
                and knobs.nvidia.use_autotune_c_cache and knobs.nvidia.use_triton_dispatcher and len(self.configs) > 1):
            proxy = getattr(self, '_autotune_proxy', None)
            if proxy is None:
                # Compute key_indices: positions in arg_names for autotuner key fields
                key_indices = []
                for k in self.keys:
                    if k in self.arg_names:
                        key_indices.append(self.arg_names.index(k))
                # Compute dtype_indices: positions of non-constexpr args that could be tensors
                dtype_indices = [i for i, p in enumerate(self.fn.params) if not p.is_constexpr]

                # fallback_run uses self._proxy_grid (set below) so grid is
                # always current when the C proxy falls back to Python.
                proxy = native_create_autotune_proxy(
                    self.fn,
                    key_indices,
                    dtype_indices,
                    self.fn.params,
                    len(self.fn.params),
                    driver.active.get_current_stream,
                    driver.active.get_current_device,
                    self._proxy_fallback,
                )
                if proxy is not None:
                    self._autotune_proxy = proxy
            if proxy is not None:
                self._proxy_grid = grid
                native_autotune_proxy_set_grid(proxy, grid)
                return proxy

        # Fallback: Python dispatch
        return lambda *args, **kwargs: self.run(*args, grid=grid, warmup=False, **kwargs)

    def _proxy_fallback(self, *a, **kw):
        """Fallback called from C proxy when autotune table misses."""
        return self.run(*a, grid=self._proxy_grid, warmup=False, **kw)

    def _seed_autotune_proxy(self, key, config):
        """Insert a key→config mapping into the C autotune proxy table."""
        proxy = getattr(self, '_autotune_proxy', None)
        if proxy is None or native_autotune_proxy_insert is None:
            return

        # Build constexpr mapping: config values that fill into full_args
        config_kwargs = config.all_kwargs()
        fn_arg_names = self.fn.arg_names
        constexpr_vals = []
        constexpr_positions = []
        for i, param in enumerate(self.fn.params):
            if param.is_constexpr and param.name in config_kwargs:
                constexpr_vals.append(config_kwargs[param.name])
                constexpr_positions.append(i)

        # Compute options_hash matching _try_fast_path's logic exactly
        fn_arg_name_set = set(fn_arg_names)
        _meta = {k: v for k, v in config_kwargs.items() if k not in fn_arg_name_set}
        _meta_opts = {k: v for k, v in _meta.items() if k not in getattr(self.fn, '_param_name_to_idx', {})}
        if _meta_opts:
            options_hash = hash(tuple(sorted(_meta_opts.items()))) & 0xFFFFFFFFFFFFFFFF
        else:
            options_hash = getattr(self.fn, '_fc_options_hash', 0)

        # Build key values matching what C vectorcall extracts:
        # - key field values (actual arg values at key_indices)
        # - dtype objects (not strings!) from tensor args
        full_nargs = getattr(self, '_full_nargs', self.nargs)
        full_args_list = []
        for name in fn_arg_names:
            if name in full_nargs:
                full_args_list.append(full_nargs[name])
            elif name in config_kwargs:
                full_args_list.append(config_kwargs[name])
            else:
                full_args_list.append(None)

        # Build key values matching what C vectorcall extracts:
        # 1. key field values at key_indices positions
        key_vals = []
        for k in self.keys:
            if k in self.arg_names:
                idx = self.arg_names.index(k)
                if idx < len(full_args_list):
                    key_vals.append(full_args_list[idx])

        # 2. dtype from args at dtype_indices positions (all non-constexpr params)
        for i, param in enumerate(self.fn.params):
            if not param.is_constexpr:
                arg = full_args_list[i] if i < len(full_args_list) else None
                if arg is not None and hasattr(arg, 'dtype'):
                    key_vals.append(arg.dtype)

        native_autotune_proxy_insert(proxy, key_vals, constexpr_vals, constexpr_positions, options_hash,
                                     config.pre_hook)

    def _try_fast_path(self, args, kwargs, config):
        """Attempt C fast cache dispatch; return kernel or None to fall back.

        Uses JITCacheProxy (hash=0) for subsequent calls after seeding.
        On the first call per autotuner key, seeds the C cache by calling
        run() with correct meta-params (kernel_cache HIT, no recompile)
        and inserting the result under hash=0.

        Returns None when preconditions aren't met (no c_cache, callable
        grid that can't be evaluated, extra kwargs, etc.).
        """
        input_grid = kwargs.get('grid')
        if input_grid is None or not getattr(self.fn, 'c_cache', False):
            return None

        # Build full positional args: user positional + config constexprs.
        config_kwargs = config.all_kwargs()
        fn_arg_names = self.fn.arg_names
        _arg_name_set = set(fn_arg_names)
        # Fall back if kwargs has extra keys (beyond 'grid'/'warmup') not in arg_names,
        # since those would be silently dropped in the fast path.
        _extra_keys = [k for k in kwargs if k not in {'grid', 'warmup'} and k not in _arg_name_set]
        if _extra_keys:
            return None

        full_args = list(args)
        for name in fn_arg_names[len(args):]:
            if name in config_kwargs:
                full_args.append(config_kwargs[name])
            elif name in kwargs:
                full_args.append(kwargs[name])
            else:
                return None  # Can't resolve all args.

        # Evaluate callable grid using resolved args.
        if callable(input_grid):
            _meta_dict = dict(zip(fn_arg_names, full_args))
            evaluated_grid = input_grid(_meta_dict)
        else:
            evaluated_grid = input_grid

        # Separate meta-params (num_warps, etc.) from kernel args.
        _meta = {k: v for k, v in config_kwargs.items() if k not in _arg_name_set}

        # Seed C cache with hash=0 for native_fast_dispatch.
        # During autotuning, run() stored the kernel with a non-zero hash.
        # We use hash=0 as the canonical key for autotuned steady-state dispatch.
        if not hasattr(self, '_fc_seeded'):
            self._fc_seeded = set()
        # Use autotuner key to distinguish specializations that may select
        # different winning configs (different meta-params).
        _seed_key = getattr(self, '_last_key', None)
        if _seed_key not in self._fc_seeded:
            try:
                from triton._C.libtriton import native_fast_dispatch_insert
            except (ImportError, AttributeError):
                native_fast_dispatch_insert = None
            kernel = self.fn.run(*full_args, grid=evaluated_grid, warmup=False, **_meta)
            # Update _fc_options_hash so C proxy and JIT.run fast path lookups
            # use the same hash that JIT.run's insertion used (includes meta-params
            # like ctas_per_cga that affect compilation options).
            if _meta:
                _meta_opts = {k: v for k, v in _meta.items() if k not in getattr(self.fn, '_param_name_to_idx', {})}
                if _meta_opts:
                    self.fn._fc_options_hash = hash(tuple(sorted(_meta_opts.items()))) & 0xFFFFFFFFFFFFFFFF
                # Store meta kwargs for C proxy fallback forwarding.
                self.fn._fc_meta_kwargs = _meta
                # Invalidate proxy cache so next __getitem__ creates a new proxy
                # with the updated options_hash and meta_kwargs.
                self.fn._jit_proxy_cache = {}
            if native_fast_dispatch_insert is not None:
                _disp = getattr(kernel, '_dispatcher', None)
                if _disp is not None:
                    _padded = tuple(full_args)
                    if len(_padded) < len(self.fn.params):
                        _padded = _padded + (None, ) * (len(self.fn.params) - len(_padded))
                    native_fast_dispatch_insert(self.fn, _padded, self.fn.params, self.fn._fc_options_hash, kernel,
                                                _disp, getattr(kernel, '_dispatch_arg_indices', None))
            self._fc_seeded.add(_seed_key)
            return kernel

        # Steady-state: dispatch via JITCacheProxy (fastest path).
        # The C cache stores dispatch_arg_indices per entry so it correctly
        # selects only the args the dispatcher expects (handles None ptr args).
        return self.fn[evaluated_grid](*full_args)

    def run(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        used_cached_result = True
        if len(self.configs) > 1:
            all_args = {**self.nargs, **kwargs}
            _args = {k: v for (k, v) in all_args.items() if k in self.arg_names}
            # Keep self.nargs as positional-only named args (the contract
            # relied on by prune_configs/early_config_prune/perf_model). Expose
            # the full merged arg set (including kwargs like key fields) via a
            # separate attribute for _seed_autotune_proxy, which must match what
            # the C proxy extracts after merging kwargs into positional.
            self._full_nargs = _args
            key = [_args[key] for key in self.keys if key in _args]
            for _, arg in _args.items():
                if hasattr(arg, "dtype"):
                    key.append(str(arg.dtype))
            key = tuple(key)
            if key not in self.cache:
                used_cached_result = False
                pruned_configs = self.prune_configs(kwargs)

                def benchmark():
                    # facebook begin
                    import importlib
                    if importlib.util.find_spec("torch.monitor") is not None:
                        from torch.monitor import _WaitCounter
                        waitcounter = _WaitCounter("pytorch.triton.benchmark").guard()
                        waitcounter.__enter__()

                    # facebook end
                    bench_start = time.time()
                    # Correctness gating (T3): validate each config's output before timing it.
                    configs_to_bench = pruned_configs
                    if self.correctness_fn is not None:
                        self.correctness_results = {}
                        valid_configs = []
                        for config in pruned_configs:
                            ok = self._check_correctness(*args, config=config, **kwargs)
                            self.correctness_results[config] = ok
                            if ok or not self.correctness_prune:
                                valid_configs.append(config)
                        if knobs.autotuning.print:
                            n_pass = sum(1 for v in self.correctness_results.values() if v)
                            n_tot = len(self.correctness_results)
                            rate = (n_pass / n_tot) if n_tot else 0.0
                            print(
                                f"[autotune] correctness: {n_pass}/{n_tot} configs passed "
                                f"(success rate {rate:.1%})", flush=True)
                        if self.correctness_prune:
                            if not valid_configs:
                                raise AutotunerError("No autotuner configs passed the correctness check. "
                                                     "Relax `correctness_fn` or pass correctness_prune=False.")
                            configs_to_bench = valid_configs
                    timings = {config: self._bench(*args, config=config, **kwargs) for config in configs_to_bench}
                    bench_end = time.time()
                    self.bench_time = bench_end - bench_start
                    # facebook begin T203283446
                    if importlib.util.find_spec("torch.monitor") is not None:
                        waitcounter.__exit__()
                    if knobs.autotuning.print:
                        print(
                            f'\nPrinting ALL Multiple Triton autotuning Configs with timings in sorted order for kernel {self.fn}:',
                            flush=True)
                        sorted_configs = builtins.sorted(timings, key=timings.get)
                        for config in sorted_configs:
                            print(f'Triton autotune config: [{config}]; Triton autotune timing: {timings[config]}',
                                  flush=True)
                    # facebook end T203283446
                    self.cache[key] = builtins.min(timings, key=timings.get)
                    full_nargs = {**self.nargs, **kwargs, **self.cache[key].all_kwargs()}
                    self.pre_hook(full_nargs, reset_only=True)
                    self.configs_timings = timings

                if self.cache_results:
                    used_cached_result = self.check_disk_cache(key, pruned_configs, benchmark)
                else:
                    benchmark()

            config = self.cache[key]
            self._last_key = key
            # Seed the C-level autotune proxy with this key→config mapping
            if not used_cached_result or not hasattr(self, '_at_proxy_seeded'):
                self._at_proxy_seeded = getattr(self, '_at_proxy_seeded', set())
            if key not in getattr(self, '_at_proxy_seeded', set()):
                self._seed_autotune_proxy(key, config)
                if not hasattr(self, '_at_proxy_seeded'):
                    self._at_proxy_seeded = set()
                self._at_proxy_seeded.add(key)
        else:
            config = self.configs[0]
        self.best_config = config
        if knobs.autotuning.print and not used_cached_result:
            print(f"Triton autotuning for function {self.base_fn.__name__},\nwith key as {key},\n"
                  f"finished after {self.bench_time:.2f}s,\nbest config selected: {self.best_config};")
        if config.pre_hook is not None:
            full_nargs = {**self.nargs, **kwargs, **config.all_kwargs()}
            config.pre_hook(full_nargs)
        # Enable IR dumping for best config if requested
        dump_best = knobs.autotuning.dump_best_config_ir
        if dump_best:
            original_dump_ir = knobs.compilation.dump_ir
            original_always_compile = knobs.compilation.always_compile
            knobs.compilation.dump_ir = True
            knobs.compilation.always_compile = True
            # Clear the JIT cache for this kernel to force recompilation
            # so IR can be dumped
            if hasattr(self.fn, 'device_caches'):
                for device_cache in self.fn.device_caches.values():
                    if isinstance(device_cache, tuple) and len(device_cache) >= 1:
                        device_cache[0].clear()
        try:
            ret = None if dump_best else self._try_fast_path(args, kwargs, config)
            if ret is None:
                ret = self.fn.run(*args, **kwargs, **config.all_kwargs())
        finally:
            if dump_best:
                knobs.compilation.dump_ir = original_dump_ir
                knobs.compilation.always_compile = original_always_compile
        self.nargs = None
        self._full_nargs = None
        return ret

    def _artifact_prune_configs(self, configs: List[Config], kwargs: Dict) -> List[Config]:
        """Keep only configs whose compiled artifact satisfies ``artifact_config_prune``.

        Each config is compiled via ``run(warmup=True)`` (real-arg specialization, so the inspected
        TTGIR/PTX matches what the benchmarked/launched kernel will use; no kernel is launched). The
        compiled kernel is cached, so the subsequent benchmark reuses it rather than recompiling.
        The predicate ``artifact_config_prune(config, asm, metadata) -> bool`` returns True to KEEP.
        Configs that fail to compile are dropped (they could not win anyway) and recorded.
        """
        self.pruned_by_artifact = {}
        pos_args = list(self.nargs.values())
        kept: List[Config] = []
        for config in configs:
            run_kwargs = dict(kwargs)
            run_kwargs.update(config.all_kwargs())
            run_kwargs["warmup"] = True  # compile only; do not launch
            try:
                kernel = self.fn.run(*pos_args, **run_kwargs)
            except Exception as e:  # noqa: BLE001 - a config that cannot compile cannot win
                self.pruned_by_artifact[config] = f"compile-error: {type(e).__name__}: {e}"
                continue
            asm = getattr(kernel, "asm", {}) or {}
            metadata = getattr(kernel, "metadata", None)
            try:
                keep = bool(self.artifact_config_prune(config, asm, metadata))
            except Exception as e:
                raise AutotunerError(
                    f"`artifact_config_prune` raised on config {config}: {type(e).__name__}: {e}") from e
            if keep:
                kept.append(config)
            else:
                self.pruned_by_artifact[config] = "artifact-prune"
        if knobs.autotuning.print and self.pruned_by_artifact:
            print(f"[autotune] artifact_config_prune dropped {len(self.pruned_by_artifact)}/{len(configs)} configs",
                  flush=True)
        return kept

    def _resolve_equivalence_pipeline(self):
        """Resolve (equivalence_level, equivalence_checkers, equivalence_fn) into an ordered list of
        (level_name, checker_fn) stages. ``equivalence_level`` may be "ttgir"/"ptx"/"both" or an
        ordered list of level names, looked up in the injected ``equivalence_checkers`` registry; a
        plain ``equivalence_fn`` is appended as a final "custom" stage. Raises AutotunerError if a
        requested level is not in the registry."""
        pipeline = []
        if self.equivalence_level is not None:
            registry = self.equivalence_checkers or {}
            level = self.equivalence_level
            names = ["ttgir", "ptx"] if level == "both" else ([level] if isinstance(level, str) else list(level))
            for name in names:
                if name not in registry:
                    raise AutotunerError(f"equivalence_level {name!r} requested but not provided in "
                                         f"prune_configs_by['equivalence_checkers'] (available: {sorted(registry)}). "
                                         f"Pass a {{level: fn}} registry, e.g. bitequiv.equivalence.CHECKERS.")
                pipeline.append((name, registry[name]))
        if self.equivalence_fn is not None:
            pipeline.append(("custom", self.equivalence_fn))
        return pipeline

    def _equivalence_prune_configs(self, configs: List[Config], kwargs: Dict) -> List[Config]:
        """Keep only configs bitwise-equivalent to the reference (first) config, statically.

        No launch: each config is compiled ONCE (``run(warmup=True)``), then the equivalence pipeline
        (``_resolve_equivalence_pipeline`` -> ordered [(level, checker)] stages, e.g. TTGIR then PTX)
        filters the survivors at each level. A checker maps a config's artifact to a hashable key; at
        each stage only configs whose key matches the first survivor's (the reference order) are kept,
        so a later stage (PTX) checks only configs that already passed the earlier one (TTGIR). The
        surviving set is bitwise-equivalent by construction — without running the kernel or comparing
        any output. Populates ``self.equivalence_classes`` ({level: {key: [Config, ...]}}) and
        ``self.pruned_by_equivalence`` ({Config: "level: reason"}).
        """
        self.pruned_by_equivalence = {}
        self.equivalence_classes = {}
        pipeline = self._resolve_equivalence_pipeline()
        pos_args = list(self.nargs.values())
        # Compile each config once (cached for the later benchmark); collect artifacts.
        compiled: Dict[Config, tuple] = {}  # config -> (asm, metadata)
        for config in configs:
            run_kwargs = dict(kwargs)
            run_kwargs.update(config.all_kwargs())
            run_kwargs["warmup"] = True  # compile only; do not launch
            try:
                kernel = self.fn.run(*pos_args, **run_kwargs)
            except Exception as e:  # noqa: BLE001 - a config that cannot compile cannot win
                self.pruned_by_equivalence[config] = f"compile-error: {type(e).__name__}: {e}"
                continue
            compiled[config] = (getattr(kernel, "asm", {}) or {}, getattr(kernel, "metadata", None))
        survivors = [c for c in configs if c in compiled]
        # Apply each level's checker in order; keep configs matching the reference (first survivor).
        for level, checker in pipeline:
            if not survivors:
                break
            keys: Dict[Config, object] = {}
            for config in survivors:
                asm, metadata = compiled[config]
                try:
                    keys[config] = checker(config, asm, metadata)
                except NotImplementedError as e:
                    raise AutotunerError(f"equivalence_level {level!r} is not available: {e}") from e
                except Exception as e:
                    raise AutotunerError(f"equivalence checker for level {level!r} raised on config "
                                         f"{config}: {type(e).__name__}: {e}") from e
            by_key: Dict[object, List[Config]] = {}
            for config in survivors:
                by_key.setdefault(keys[config], []).append(config)
            self.equivalence_classes[level] = by_key
            reference_key = keys[survivors[0]]
            kept = [c for c in survivors if keys[c] == reference_key]
            for config in survivors:
                if config not in kept:
                    self.pruned_by_equivalence[config] = f"{level}: not-equivalent-to-reference"
            if knobs.autotuning.print:
                print(
                    f"[autotune] equivalence[{level}] kept {len(kept)}/{len(survivors)} configs "
                    f"({len(by_key)} class(es))", flush=True)
            survivors = kept
        return survivors

    def prune_configs(self, kwargs: Dict) -> List[Config]:
        pruned_configs = self.configs
        if self.early_config_prune:
            pruned_configs = self.early_config_prune(self.configs, self.nargs, **kwargs)
            if not pruned_configs:
                raise AutotunerError(
                    "No valid autotuner configs after pruning. `early_config_prune` should return at least one config.")
        # Artifact-based pruning (T4): inspect each config's compiled TTGIR/PTX. This must run after
        # compilation (early_config_prune runs before it), so we compile each config here.
        if self.artifact_config_prune:
            pruned_configs = self._artifact_prune_configs(pruned_configs, kwargs)
            if not pruned_configs:
                raise AutotunerError("No valid autotuner configs after artifact pruning. "
                                     "`artifact_config_prune` should keep at least one config.")
        # Static bitwise-equivalence pruning (M1): keep only configs whose compiled IR matches the
        # reference (first) config at the chosen level(s) (TTGIR / PTX). No launch, no reference output.
        if self.equivalence_level is not None or self.equivalence_fn is not None:
            pruned_configs = self._equivalence_prune_configs(pruned_configs, kwargs)
            if not pruned_configs:
                raise AutotunerError("No valid autotuner configs after equivalence pruning. "
                                     "The equivalence check should keep at least the reference config.")
        if self.perf_model:
            top_k = self.configs_top_k
            if isinstance(top_k, float) and top_k <= 1.0:
                top_k = int(len(self.configs) * top_k)
            elif not isinstance(top_k, int):
                # Slice index must be an integer
                raise TypeError("Error while pruning configs, top_k must be either 1) a float <= 1.0 or 2) an int")

            if len(pruned_configs) > top_k:
                est_timing = {
                    config: self.perf_model(
                        **self.nargs,
                        **kwargs,
                        **config.all_kwargs(),
                    )
                    for config in pruned_configs
                }
                pruned_configs = sorted(est_timing.keys(), key=lambda x: est_timing[x])[:top_k]
        return pruned_configs

    def warmup(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        ret = []
        for autotune_config in self.prune_configs(kwargs):
            ret.append(self.fn.warmup(
                *args,
                **kwargs,
                **autotune_config.all_kwargs(),
            ))
        self.nargs = None
        return ret


class Config:
    """
    An object that represents a possible kernel configuration for the auto-tuner to try.

    :ivar kwargs: a dictionary of meta-parameters to pass to the kernel as keyword arguments.
    :type kwargs: dict[Str, Any]
    :ivar num_warps: the number of warps to use for the kernel when compiled for GPUs. For example, if
                      `num_warps=8`, then each kernel instance will be automatically parallelized to
                      cooperatively execute using `8 * 32 = 256` threads.
    :type num_warps: int
    :ivar num_stages: the number of stages that the compiler should use when software-pipelining loops.
                       Mostly useful for matrix multiplication workloads on SM80+ GPUs.
    :type num_stages: int
    :ivar num_ctas: number of blocks in a block cluster. SM90+ only.
    :type num_ctas: int
    :type maxnreg: Optional[int]
    :ivar maxnreg: maximum number of registers one thread can use.  Corresponds
                       to ptx .maxnreg directive.  Not supported on all platforms.
    :ivar pre_hook: a function that will be called before the kernel is called. Parameters of this
                    function are args.
    :ivar ir_override: filename of a user-defined IR (*.{ttgir|llir|ptx|amdgcn}).
    :ivar ctas_per_cga: number of CTAs per Cooperative Grid Array (cluster) for CUDA Thread Block Clusters. SM90+ only.
        Unlike cluster_dims which spawns new CTAs, ctas_per_cga regroups existing grid CTAs into clusters.
        This matches CUDA's cuLaunchKernelEx CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION semantics.
    :type ctas_per_cga: tuple[int, int, int]
    :ivar preferred_ctas_per_cga: preferred number of CTAs per cluster. Unlike ctas_per_cga which is
        required, this is a hint: the driver may use a smaller cluster if resources are constrained.
        Maps to CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION. The per dim grid size must be divisible by this per dim cluster size.
    :type preferred_ctas_per_cga: tuple[int, int, int]
    """

    @staticmethod
    def _check_reg_auto_ws_alignment(name, value):
        if value is not None and value % 8 != 0:
            raise ValueError(f"{name} must be divisible by 8, got {value}")

    def __init__(
        self,
        kwargs,
        num_warps=4,
        num_stages=3,
        num_ctas=1,
        maxnreg=None,
        pre_hook=None,
        ir_override=None,
        minRegAutoWS=None,
        maxRegAutoWS=None,
        pingpongAutoWS=None,
        num_buffers_warp_spec=0,
        num_consumer_groups=0,
        reg_dec_producer=0,
        reg_inc_consumer=0,
        ctas_per_cga=None,
        early_tma_store_lowering=None,
        generate_subtiled_region=None,
        preferred_ctas_per_cga=None,
    ):
        self.kwargs = kwargs
        self.num_warps = num_warps
        self.num_ctas = num_ctas
        self.num_stages = num_stages
        self.maxnreg = maxnreg
        self.pre_hook = pre_hook
        self.ir_override = ir_override
        self._check_reg_auto_ws_alignment("minRegAutoWS", minRegAutoWS)
        self._check_reg_auto_ws_alignment("maxRegAutoWS", maxRegAutoWS)
        self.minRegAutoWS = minRegAutoWS
        self.maxRegAutoWS = maxRegAutoWS
        self.pingpongAutoWS = pingpongAutoWS
        self.ctas_per_cga = ctas_per_cga
        self.early_tma_store_lowering = early_tma_store_lowering
        self.generate_subtiled_region = generate_subtiled_region
        self.preferred_ctas_per_cga = preferred_ctas_per_cga

    def __setstate__(self, state):
        self.kwargs = state.get("kwargs", {})
        self.num_warps = state.get("num_warps", 4)
        self.num_stages = state.get("num_stages", 3)
        self.num_ctas = state.get("num_ctas", 1)
        self.maxnreg = state.get("maxnreg", None)
        self.pre_hook = state.get("pre_hook", None)
        self.ir_override = state.get("ir_override", None)
        self.minRegAutoWS = state.get("minRegAutoWS", None)
        self.maxRegAutoWS = state.get("maxRegAutoWS", None)
        self.pingpongAutoWS = state.get("pingpongAutoWS", None)
        self.ctas_per_cga = state.get("ctas_per_cga", None)
        self.early_tma_store_lowering = state.get("early_tma_store_lowering", None)
        self.generate_subtiled_region = state.get("generate_subtiled_region", None)
        self.preferred_ctas_per_cga = state.get("preferred_ctas_per_cga", None)

    def all_kwargs(self):
        return {
            **self.kwargs,
            **{
                k: v
                for (k, v) in (
                    ("num_warps", self.num_warps),
                    ("num_ctas", self.num_ctas),
                    ("num_stages", self.num_stages),
                    ("maxnreg", self.maxnreg),
                    ("ir_override", self.ir_override),
                    ("minRegAutoWS", self.minRegAutoWS),
                    ("maxRegAutoWS", self.maxRegAutoWS),
                    ("pingpongAutoWS", self.pingpongAutoWS),
                    ("ctas_per_cga", self.ctas_per_cga),
                    ("early_tma_store_lowering", self.early_tma_store_lowering),
                    ("generate_subtiled_region", self.generate_subtiled_region),
                    ("preferred_ctas_per_cga", self.preferred_ctas_per_cga),
                ) if v is not None
            },
        }

    def __str__(self):
        res = []
        for k, v in self.kwargs.items():
            res.append(f"{k}: {v}")
        res.append(f"num_warps: {self.num_warps}")
        res.append(f"num_ctas: {self.num_ctas}")
        res.append(f"num_stages: {self.num_stages}")
        res.append(f"maxnreg: {self.maxnreg}")
        res.append(f"minRegAutoWS: {self.minRegAutoWS}")
        res.append(f"maxRegAutoWS: {self.maxRegAutoWS}")
        res.append(f"pingpongAutoWS: {self.pingpongAutoWS}")
        res.append(f"ctas_per_cga: {self.ctas_per_cga}")
        res.append(f"early_tma_store_lowering: {self.early_tma_store_lowering}")
        res.append(f"generate_subtiled_region: {self.generate_subtiled_region}")
        res.append(f"preferred_ctas_per_cga: {self.preferred_ctas_per_cga}")
        return ", ".join(res)

    def __hash__(self):
        return hash((*self.all_kwargs().items(), self.pre_hook))

    def __eq__(self, other):
        self_tuple = tuple((
            *self.all_kwargs().items(),
            self.pre_hook,
        ))
        other_tuple = tuple((
            *other.all_kwargs().items(),
            other.pre_hook,
        ))
        return self_tuple == other_tuple


def autotune(configs, key, prune_configs_by=None, reset_to_zero=None, restore_value=None, pre_hook=None, post_hook=None,
             warmup=None, rep=None, use_cuda_graph=False, do_bench=None, cache_results=False, correctness_fn=None,
             correctness_prune=True):
    """
    Decorator for auto-tuning a :code:`triton.jit`'d function.

    .. highlight:: python
    .. code-block:: python

        @triton.autotune(configs=[
            triton.Config(kwargs={'BLOCK_SIZE': 128}, num_warps=4),
            triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=8),
          ],
          key=['x_size'] # the two above configs will be evaluated anytime
                         # the value of x_size changes
        )
        @triton.jit
        def kernel(x_ptr, x_size, BLOCK_SIZE: tl.constexpr):
            ...
    :note: When all the configurations are evaluated, the kernel will run multiple times.
           This means that whatever value the kernel updates will be updated multiple times.
           To avoid this undesired behavior, you can use the `reset_to_zero` argument, which
           resets the value of the provided tensor to `zero` before running any configuration.

    If the environment variable :code:`TRITON_PRINT_AUTOTUNING` is set to
    :code:`"1"`, Triton will print a message to stdout after autotuning each
    kernel, including the time spent autotuning and the best configuration.

    :param configs: a list of :code:`triton.Config` objects
    :type configs: list[triton.Config]
    :param key: a list of argument names whose change in value will trigger the evaluation of all provided configs.
    :type key: list[str]
    :param prune_configs_by: a dict of functions that are used to prune configs, fields:
        'perf_model': performance model used to predicate running time with different configs, returns running time
        'top_k': number of configs to bench
        'early_config_prune': a function used to prune configs. It should have the signature
                `prune_configs_by( configs: List[triton.Config], named_args: Dict[str, Any], **kwargs: Dict[str, Any]) -> List[triton.Config]:`
                and return pruned configs. It should return at least one config.
    :param reset_to_zero: a list of argument names whose value will be reset to zero before evaluating any configs.
    :type reset_to_zero: list[str]
    :param restore_value: a list of argument names whose value will be restored after evaluating any configs.
    :type restore_value: list[str]
    :param pre_hook: a function that will be called before the kernel is called.
        This overrides the default pre_hook used for 'reset_to_zero' and 'restore_value'.
        'kwargs': a dict of all arguments passed to the kernel.
        'reset_only': a boolean indicating whether the pre_hook is called to reset the values only, without a corresponding post_hook.
    :type pre_hook: lambda args, reset_only
    :param post_hook: a function that will be called after the kernel is called.
        This overrides the default post_hook used for 'restore_value'.
        'kwargs': a dict of all arguments passed to the kernel.
        'exception': the exception raised by the kernel in case of a compilation or runtime error.
    :type post_hook: lambda args, exception
    :param warmup: warmup time (in ms) to pass to benchmarking (deprecated).
    :type warmup: int
    :param rep: repetition time (in ms) to pass to benchmarking (deprecated).
    :type rep: int
    :param do_bench: a benchmark function to measure the time of each run.
    :type do_bench: lambda fn, quantiles
    :param cache_results: whether to cache autotune timings to disk.  Defaults to False.
    "type cache_results: bool
    :param correctness_fn: optional callable `correctness_fn(named_args) -> bool` validating each
        config's output (after one untimed run) against a user-defined reference. See
        :class:`Autotuner` for details.
    :type correctness_fn: Optional[Callable[[dict], bool]]
    :param correctness_prune: if True (default), configs failing `correctness_fn` are excluded from
        selection; if False, results are only recorded.
    :type correctness_prune: bool
    """

    def decorator(fn):
        return Autotuner(fn, fn.arg_names, configs, key, reset_to_zero, restore_value, pre_hook=pre_hook,
                         post_hook=post_hook, prune_configs_by=prune_configs_by, warmup=warmup, rep=rep,
                         use_cuda_graph=use_cuda_graph, do_bench=do_bench, cache_results=cache_results,
                         correctness_fn=correctness_fn, correctness_prune=correctness_prune)

    return decorator


class Heuristics(KernelInterface):

    def __init__(self, fn, arg_names, values) -> None:
        self.fn = fn
        self.values = values
        self.arg_names = arg_names

    def run(self, *args, **kwargs):
        for v, heur in self.values.items():
            kwargs[v] = heur({**dict(zip(self.arg_names, args)), **kwargs})
        return self.fn.run(*args, **kwargs)


def heuristics(values):
    """
    Decorator for specifying how the values of certain meta-parameters may be computed.
    This is useful for cases where auto-tuning is prohibitively expensive, or just not applicable.

    .. highlight:: python
    .. code-block:: python

        # smallest power-of-two >= x_size
        @triton.heuristics(values={'BLOCK_SIZE': lambda args: triton.next_power_of_2(args['x_size'])})
        @triton.jit
        def kernel(x_ptr, x_size, BLOCK_SIZE: tl.constexpr):
            ...
    :param values: a dictionary of meta-parameter names and functions that compute the value of the meta-parameter.
                   each such function takes a list of positional arguments as input.
    :type values: dict[str, Callable[[dict[str, Any]], Any]]
    """

    def decorator(fn):
        return Heuristics(fn, fn.arg_names, values)

    return decorator
