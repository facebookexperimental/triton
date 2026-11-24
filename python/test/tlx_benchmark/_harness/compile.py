from __future__ import annotations

import contextlib
import dataclasses
import os
import tempfile
import time
from typing import Callable, Optional

import triton

#: Absolute ceiling on a first call. A breach is a real defect: the shape is
#: not shippable at the space it needs, and per this suite's convention it is
#: commented out of the shared shape list with a TODO carrying the measurement.
COLD_COMPILE_CAP_S = 120.0


@dataclasses.dataclass(frozen=True)
class CompileStat:
    t_cold_s: float
    n_compiles: Optional[int] = None
    n_configs: Optional[int] = None
    cap_s: float = COLD_COMPILE_CAP_S

    @property
    def over_cap(self) -> bool:
        return self.t_cold_s > self.cap_s

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["over_cap"] = self.over_cap
        return d


@contextlib.contextmanager
def fresh_triton_cache():
    previous_knob = triton.knobs.cache.dir
    previous_env = os.environ.get("TRITON_CACHE_DIR")
    with tempfile.TemporaryDirectory(prefix="tlx-bench-cache-") as tmp:
        triton.knobs.cache.dir = tmp
        os.environ["TRITON_CACHE_DIR"] = tmp
        try:
            yield tmp
        finally:
            triton.knobs.cache.dir = previous_knob
            if previous_env is None:
                os.environ.pop("TRITON_CACHE_DIR", None)
            else:
                os.environ["TRITON_CACHE_DIR"] = previous_env


@triton.jit
def _nop_kernel():
    pass


def prewarm():
    _nop_kernel[(1, )]()


class _Counters:

    def __init__(self):
        self.compiles = 0
        self.configs = 0
        self._prev_compilation = None
        self._prev_autotuning = None

    def _on_compile(self, **kwargs):
        if not kwargs.get("cache_hit", False):
            self.compiles += 1

    def _on_autotune(self, **kwargs):
        timings = kwargs.get("configs_timings") or {}
        self.configs += len(timings)

    def __enter__(self):
        self._prev_compilation = triton.knobs.compilation.listener
        self._prev_autotuning = triton.knobs.autotuning.listener
        triton.knobs.compilation.listener = self._on_compile
        triton.knobs.autotuning.listener = self._on_autotune
        return self

    def __exit__(self, *exc):
        triton.knobs.compilation.listener = self._prev_compilation
        triton.knobs.autotuning.listener = self._prev_autotuning
        return False


def cold_compile(fn: Callable, *, cap_s: float = COLD_COMPILE_CAP_S, do_prewarm: bool = True) -> CompileStat:
    import torch

    with fresh_triton_cache():
        if do_prewarm:
            prewarm()
        torch.cuda.synchronize()
        with _Counters() as counters:
            started = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
    return CompileStat(
        t_cold_s=elapsed,
        n_compiles=counters.compiles,
        n_configs=counters.configs or None,
        cap_s=cap_s,
    )
