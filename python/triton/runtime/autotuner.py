from __future__ import annotations

import builtins
import copy
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
from .jit import KernelInterface, JITFunction, _compile_iq_suppress_competition
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
                 cache_results=False, include_npot=False):
        """
        :param prune_configs_by: a dict of functions that are used to prune configs, fields:
            'perf_model': performance model used to predicate running time with different configs, returns running time
            'top_k': number of configs to bench
            'early_config_prune': a function used to prune configs. It should have the signature
                `prune_configs_by( configs: List[triton.Config], named_args: Dict[str, Any], **kwargs: Dict[str, Any]) -> List[triton.Config]:`
                and return pruned configs. It should return at least one config.
            'ir_config_prune': a function used to prune configs by inspecting each config's
                *compiled artifact* (TTGIR/PTX). Unlike 'early_config_prune', which runs before any
                compilation, this hook runs *after each config has been benchmarked*, reusing the
                CompiledKernel the benchmark already produced (no extra compilation), so it can
                filter on the generated IR. A config the predicate rejects is pruned by marking its
                timing invalid (it can no longer be selected). Signature:
                `ir_config_prune(config: triton.Config, asm: Dict[str, str], metadata) -> bool`
                returning True to KEEP the config. `asm` is the CompiledKernel.asm dict (keys such
                as 'ttir', 'ttgir', 'llir', 'ptx'); `metadata` is CompiledKernel.metadata. The
                predicate MAY take an optional 4th argument
                `ir_config_prune(config, asm, metadata, reference)` where `reference` is the
                `(config, asm, metadata)` of the reference config (by default the first surviving
                config); equivalence-style checks compare each config against it instead of relying
                on call order.

                This is the single IR-based pruning hook. Static bitwise-equivalence pruning (keep
                only configs whose compiled IR matches a reference order, at TTGIR and/or PTX level)
                can be layered on top of it by adapting a per-config equivalence *key* into an
                `ir_config_prune` predicate. Triton core stays decoupled — such equivalence
                checkers live outside core, not here.
        """
        if not configs:
            self.configs = [Config({}, num_warps=4, num_stages=3, num_ctas=1)]
        else:
            self.configs = configs
        self.keys = key
        self.include_npot = include_npot
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
        self.ir_config_prune = None
        if prune_configs_by:
            self.perf_model = prune_configs_by.get("perf_model", self.perf_model)
            self.configs_top_k = prune_configs_by.get("top_k", self.configs_top_k)
            self.early_config_prune = prune_configs_by.get("early_config_prune", self.early_config_prune)
            self.ir_config_prune = prune_configs_by.get("ir_config_prune", self.ir_config_prune)

        # {Config: reason} for configs dropped by ir_config_prune (IR-based pruning, incl.
        # static bitwise-equivalence pruning built on top of it in bitequiv). The number of
        # configs pruned on the last tuning run is `len(<autotuned_kernel>.pruned_by_ir)`.
        self.pruned_by_ir: Dict[Config, str] = {}
        # The CompiledKernel produced by the most recent `_bench` launch, captured so the
        # post-bench IR prune can inspect each config's artifact without recompiling.
        self._last_compiled_kernel = None

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
        """Benchmark one config and return its timing ``[median, p20, p80]`` (``inf`` on
        failure). As a side effect, records the config's ``CompiledKernel`` on
        ``self._last_compiled_kernel`` (``None`` if it could not be compiled/launched) so the
        post-bench IR prune can read its artifacts without recompiling. That kernel is the return
        value of ``self.fn.run`` (jit.py fuses compile + launch), the only point a config's
        compiled artifact surfaces — see the DESIGN NOTE in ``run``'s ``benchmark`` closure.
        Return value unchanged.
        """
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

        # Capture the CompiledKernel the launch returns (run(...) returns it even on a normal
        # launch). `_bench` calls self.fn.run directly, so this is a CompiledKernel with `.asm`.
        captured = []

        def kernel_call():
            if config.pre_hook:
                config.pre_hook(full_nargs)
            self.pre_hook(full_nargs)
            try:
                kernel = self.fn.run(
                    *args,
                    **current,
                )
            except Exception as e:
                try:
                    self.post_hook(full_nargs, exception=e)
                finally:
                    # Throw exception raised by `self.fn.run`
                    raise

            if kernel is not None:
                captured.append(kernel)
            self.post_hook(full_nargs, exception=None)

        self._last_compiled_kernel = None
        try:
            # compile_iq free-win: suppress the launch-time plain-vs-ACF competition so a pending ACF
            # A/B never fires mid-do_bench and corrupts this config's timing (autotune flow tunes on
            # plain timing; ACF candidate-expansion for autotuned kernels is handled separately).
            with _compile_iq_suppress_competition():
                timing = self.do_bench(kernel_call, quantiles=(0.5, 0.2, 0.8))
        except (OutOfResources, CompileTimeAssertionFailure, PTXASError) as e:
            if verbose:
                print(f"Autotuning failed with {e}")
            return [float("inf"), float("inf"), float("inf")]
        except RuntimeError as e:
            # Prune an NPOT candidate to inf if it hit a device compile/launch failure so the sweep
            # continues; pow2 configs (and non-device errors) re-raise.
            if _npot_runtime_error_prunable(config, e, knobs.language.allow_npot):
                if verbose:
                    print(f"Autotuning pruned NPOT config after runtime error: {e}")
                return [float("inf"), float("inf"), float("inf")]
            raise
        self._last_compiled_kernel = captured[-1] if captured else None
        return timing

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
                if getattr(self, "include_npot", False) and knobs.language.allow_npot:
                    # Prune the NPOT-augmented list via the helper (keeps prune_configs 1-arg).
                    configs = self._add_wave_quant_npot_configs(self.configs, _args)
                    pruned_configs = self._prune_configs(kwargs, configs)
                else:
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
                    timings = {}
                    compiled = {}  # config -> CompiledKernel (captured during the bench launch)
                    for config in pruned_configs:
                        timings[config] = self._bench(*args, config=config, **kwargs)
                        compiled[config] = self._last_compiled_kernel
                    # IR-based pruning runs here, AFTER benchmarking, reusing each config's
                    # already-compiled artifact (no extra compile); a rejected config is pruned
                    # by marking its timing invalid (inf) so it cannot win.
                    #
                    # DESIGN NOTE — why this is a post-bench pass and not an inline per-config
                    # prune (raised in review D107928110): compilation is not a discrete step the
                    # autotuner controls. A config's compiled artifact (CompiledKernel.asm) only
                    # becomes available as the return value of `self.fn.run(...)` in jit.py, which
                    # *fuses* compile + launch (JITFunction.run: compile on cache miss, then
                    # launch, then return the kernel). `_bench` captures it as a side effect on
                    # `self._last_compiled_kernel`. So a config's IR exists only once it has been
                    # compiled AND benchmarked; pruning it *before* timing would require a separate
                    # compile pass that re-implements jit.py's run pipeline (deliberately avoided).
                    # Folding this pass into the loop above (per-config inline) is a pure code-org
                    # change and is doable, but must still thread the captured kernel + reference
                    # config through the loop and preserve the "first finite-time config =
                    # reference" and "at least one survivor" semantics — left as a separate pass on
                    # purpose; touch with care.
                    if self.ir_config_prune is not None:
                        self._ir_prune_after_bench(pruned_configs, timings, compiled)
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

    def _add_wave_quant_npot_configs(self, configs, named_args):
        """Append wave-quant NPOT BLOCK_M/N candidates at tune time. No-op on missing shape/SMs or exception."""
        try:
            # Spatial grid dims are M and N; the reduction dim K is a loop, not a grid dim.
            problem_dims = {d: named_args.get(d) for d in ("M", "N") if isinstance(named_args.get(d), int)}
            if len(problem_dims) < 2:
                return configs
            # SM count on NVIDIA / CU count on AMD; None on backends that don't report it, which
            # makes _generate_wave_quant_candidates a no-op. ctas_per_sm is estimated as 1.
            return _generate_wave_quant_candidates(configs, problem_dims, _device_num_sms())
        except Exception:
            return configs

    def _ir_prune_after_bench(self, configs: List[Config], timings: Dict, compiled: Dict) -> None:
        """Apply ``ir_config_prune`` after benchmarking, reusing each config's already-compiled
        artifact — no recompilation, no separate run-pipeline pass.

        Inspects only configs that compiled and benchmarked with a finite time (``compiled[c]``
        is a CompiledKernel and ``timings[c]`` is not ``inf``); a config the predicate rejects is
        pruned by setting ``timings[c] = [inf, inf, inf]`` so it cannot win, and recorded in
        ``self.pruned_by_ir``.

        The predicate is ``ir_config_prune(config, asm, metadata) -> bool`` (True KEEPs). It may
        also take an optional 4th argument ``reference`` — the ``(config, asm, metadata)`` of the
        reference config (by default the first surviving config) — so equivalence-style checks can
        compare each config against a fixed reference instead of relying on call order.

        The pruned configs are recorded in ``self.pruned_by_ir``; the prune count for the run is
        ``len(self.pruned_by_ir)``.
        """
        self.pruned_by_ir = {}
        prune = self.ir_config_prune
        inf = float("inf")
        # Only configs that compiled and timed finitely have an artifact worth checking.
        items = [(c, compiled[c].asm, compiled[c].metadata)
                 for c in configs
                 if compiled.get(c) is not None and timings[c][0] != inf]
        if not items:
            return
        reference = items[0]  # default reference = first surviving (finite-time) config
        try:
            accepts_reference = len(inspect.signature(prune).parameters) >= 4
        except (TypeError, ValueError):
            accepts_reference = False

        for config, asm, metadata in items:
            try:
                keep = bool(
                    prune(config, asm, metadata, reference) if accepts_reference else prune(config, asm, metadata))
            except Exception as e:
                raise AutotunerError(f"`ir_config_prune` raised on config {config}: {type(e).__name__}: {e}") from e
            if not keep:
                timings[config] = [inf, inf, inf]  # mark invalid so it can't be selected
                self.pruned_by_ir[config] = "ir-prune"
        if all(t[0] == inf for t in timings.values()):
            raise AutotunerError("No valid autotuner configs after IR pruning. "
                                 "`ir_config_prune` should keep at least one config.")

    def prune_configs(self, kwargs: Dict) -> List[Config]:
        # Keep 1-arg: subclasses (e.g. hammer, fast_moe) override this; NPOT prunes via _prune_configs.
        return self._prune_configs(kwargs, self.configs)

    def _prune_configs(self, kwargs: Dict, configs: List[Config]) -> List[Config]:
        pruned_configs = configs
        if self.early_config_prune:
            pruned_configs = self.early_config_prune(configs, self.nargs, **kwargs)
            if not pruned_configs:
                raise AutotunerError(
                    "No valid autotuner configs after pruning. `early_config_prune` should return at least one config.")
        if self.perf_model:
            top_k = self.configs_top_k
            if isinstance(top_k, float) and top_k <= 1.0:
                top_k = int(len(configs) * top_k)
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
        auto_tma=None,
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
        # Per-config auto-TMA toggle. None -> defer to the global TRITON_AUTO_TMA
        # knob; True/False lets the autotuner A/B auto-TMA per shape.
        self.auto_tma = auto_tma

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
        self.auto_tma = state.get("auto_tma", None)

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
                    ("auto_tma", self.auto_tma),
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
        res.append(f"auto_tma: {self.auto_tma}")
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


def _is_pow2(x):
    return x > 0 and (x & (x - 1)) == 0


def _cdiv(a, b):
    return -(-a // b)


def _clone_config(config, **kwargs_override):
    """Clone a Config with kwargs overrides; copy.copy carries future Config fields, replace kwargs to avoid alias."""
    new_config = copy.copy(config)
    new_config.kwargs = {**config.kwargs, **kwargs_override}
    return new_config


def _config_has_npot_block(config):
    """True if any of the config's BLOCK_* tile sizes is non-power-of-2."""
    return any(isinstance(v, int) and not _is_pow2(v) for k, v in config.kwargs.items() if 'BLOCK' in k.upper())


# Markers of a device compile/launch failure (vs a logic bug), matched case-insensitively. The
# NVIDIA and AMD drivers tag every device RuntimeError with "Triton Error [CUDA]"/"[HIP]"; the rest
# cover common HW failure signatures (including torch-surfaced wordings).
_DEVICE_ERROR_MARKERS = (
    "triton error [cuda]",
    "triton error [hip]",
    "out of memory",
    "misaligned",
    "illegal memory access",
    "illegal instruction",
    "device-side assert",
)


def _npot_runtime_error_prunable(config, err, allow_npot):
    """Prune only an NPOT config, under the flag, on a device error; else re-raise."""
    if not (allow_npot and _config_has_npot_block(config)):
        return False
    text = str(err).lower()
    return any(marker in text for marker in _DEVICE_ERROR_MARKERS)


def _wave_efficiency(num_tiles, num_units):
    """Useful work fraction = tiles / (waves * units). 1.0 = full waves."""
    if num_tiles <= 0 or num_units <= 0:
        return 1.0
    waves = _cdiv(num_tiles, num_units)
    return num_tiles / (waves * num_units)


def _legal_npot_blocks(base, legal_multiple):
    """Legal NPOT multiples of ``legal_multiple`` in [base/2, base*2], excluding base and pow2."""
    lo = max(legal_multiple, base // 2)
    hi = base * 2
    v = lo + ((legal_multiple - lo % legal_multiple) % legal_multiple)  # first legal multiple >= lo
    out = []
    while v <= hi:
        if v != base and not _is_pow2(v):
            out.append(v)
        v += legal_multiple
    return out


# Only intervene when the pow2 tiling spans more than one wave and wastes more than (1 - threshold)
# of capacity; a single wave cannot be improved by re-tiling.
_WAVE_EFFICIENCY_THRESHOLD = 0.9

# Per-config cap on proposed NPOT tiles (the top-ranked few). This bounds candidate GENERATION; the
# overall sweep size is then governed by the autotuner's normal pruning (early_config_prune/top_k).
_MAX_NPOT_CANDIDATES = 4


def _wave_quant_npot_candidates(base_m, base_n, problem_m, problem_n, num_units, legal_m=16, legal_n=16):
    """Up to _MAX_NPOT_CANDIDATES legal NPOT (BLOCK_M, BLOCK_N) improving wave count then efficiency
    over the pow2 base; [] if pow2 is already efficient (>1 wave and >= threshold)."""
    if not (base_m and base_n and problem_m and problem_n and num_units):
        return []
    pow2_tiles = _cdiv(problem_m, base_m) * _cdiv(problem_n, base_n)
    pow2_waves = _cdiv(pow2_tiles, num_units)
    pow2_eff = pow2_tiles / (pow2_waves * num_units)
    if pow2_waves <= 1 or pow2_eff >= _WAVE_EFFICIENCY_THRESHOLD:
        return []
    scored = []
    for bm in [base_m] + _legal_npot_blocks(base_m, legal_m):
        for bn in [base_n] + _legal_npot_blocks(base_n, legal_n):
            if bm == base_m and bn == base_n:
                continue
            tiles = _cdiv(problem_m, bm) * _cdiv(problem_n, bn)
            waves = _cdiv(tiles, num_units)
            eff = tiles / (waves * num_units)
            if waves < pow2_waves or (waves == pow2_waves and eff > pow2_eff):
                scored.append(((waves, -eff), (bm, bn)))
    scored.sort(key=lambda x: x[0])
    out, seen = [], set()
    for _, pair in scored:
        if pair in seen:
            continue
        seen.add(pair)
        out.append(pair)
        if len(out) >= _MAX_NPOT_CANDIDATES:
            break
    return out


def _generate_wave_quant_candidates(configs, problem_dims, num_units):
    """Append wave-quant NPOT BLOCK_M/N candidates for each pow2 config; no-op if flag off or M/N shape/units unknown.

    ``problem_dims`` = {'M': int, 'N': int} (spatial grid dims; K is a reduction loop)."""
    if not knobs.language.allow_npot or not num_units:
        return configs
    pm = problem_dims.get('M')
    pn = problem_dims.get('N')
    if not (isinstance(pm, int) and isinstance(pn, int)):
        return configs
    expanded = list(configs)
    for config in configs:
        bm = config.kwargs.get('BLOCK_M')
        bn = config.kwargs.get('BLOCK_N')
        if not (isinstance(bm, int) and isinstance(bn, int)):
            continue
        for (nbm, nbn) in _wave_quant_npot_candidates(bm, bn, pm, pn, num_units):
            new_config = _clone_config(config, BLOCK_M=nbm, BLOCK_N=nbn)
            if new_config not in expanded:
                expanded.append(new_config)
    return expanded


def _device_num_sms():
    """Best-effort device SM count (from the Triton driver) for the wave-quant model; None ->
    wave-quant is skipped."""
    try:
        dev = driver.active.get_current_device()
        n = driver.active.utils.get_device_properties(dev).get("multiprocessor_count")
        return int(n) if n else None
    except Exception:
        return None


def autotune(configs, key, prune_configs_by=None, reset_to_zero=None, restore_value=None, pre_hook=None, post_hook=None,
             warmup=None, rep=None, use_cuda_graph=False, do_bench=None, cache_results=False, include_npot=False):
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
    :param include_npot: when True and TRITON_ALLOW_NPOT=1, the autotuner adds wave-quant-targeted
        NPOT BLOCK_M/BLOCK_N candidates at tune time, but ONLY when the pow2 tiling has a real
        wave-quantization tail. No-op otherwise, so pow2 autotuning is unchanged. Defaults to False.
    :type include_npot: bool
    """

    def decorator(fn):
        return Autotuner(fn, fn.arg_names, configs, key, reset_to_zero, restore_value, pre_hook=pre_hook,
                         post_hook=post_hook, prune_configs_by=prune_configs_by, warmup=warmup, rep=rep,
                         use_cuda_graph=use_cuda_graph, do_bench=do_bench, cache_results=cache_results,
                         include_npot=include_npot)

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
