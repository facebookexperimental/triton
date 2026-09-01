"""Cold-start compile cost.

The user-visible cost of a TLX op is not only how fast the kernel runs -- it is
also how long the first call takes, which for an autotuned op means pruning,
compiling and benchmarking a search space. ``tlx.ops.mm``'s raw space is about
1.16M configs before ``early_config_prune`` runs.

The guard on this is an **absolute cap**, not a comparison against a baseline.
cuBLAS and rocBLAS have no compile step, so there is no ratio to be worse than;
the only meaningful question is whether a user waits an unacceptable time. A
relative gate would also happily ratchet: three 15% regressions in a row pass
individually and double the wait.

Measurement is deliberately a separate pass from steady-state latency:
``t_cold`` needs a cold cache and latency needs a warm one, so no single call
can yield both.

Two op-agnostic signals come free via Triton's listener knobs, and they are
what make a breach actionable -- a slow first call is nearly always "pruning
left too many configs", not "the compiler got slower":

* ``n_compiles``  -- distinct kernel compilations (compilation listener)
* ``n_configs``   -- configs the autotuner actually benchmarked (autotune listener)
"""

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
    """Run with an empty Triton cache, restoring the real one afterwards.

    Both the knob and the environment variable are set: the knob is what the
    running process reads, and the variable covers anything that re-reads the
    environment or spawns a subprocess.
    """
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
    """Absorb one-time init so it is not billed to the op under test.

    The first Triton compilation in a process pays for driver setup, backend
    import and CUDA context creation. On a cold cache that lands entirely on
    whichever kernel happens to be first, which would otherwise be the op we
    are trying to measure.
    """
    _nop_kernel[(1, )]()


class _Counters:
    """Counts compilations and benchmarked configs via Triton's listener knobs.

    Op-agnostic on purpose: reaching into a kernel module's autotuner would tie
    the harness to one op's internals, and these hooks see any of them.
    """

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
    """Time one call of ``fn`` from a cold Triton cache.

    ``fn`` takes no arguments and must perform the whole user-visible first
    call -- for an autotuned op that includes pruning, compiling and
    benchmarking the search space, because that is what the user waits for.
    """
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
