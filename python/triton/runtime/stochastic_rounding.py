"""Experimental annotated stochastic rounding and CUDA graph seed policy."""

from __future__ import annotations

import ast
import copy
import contextvars
import inspect
import types
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass

from triton.language import bfloat16, float32, dtype
from .. import knobs
from .autotuner import Autotuner, Heuristics
from .jit import JITFunction, KernelInterface

_SEED_ARGS = ("__sr_dynamic_seed", "__sr_base_seed", "__sr_invocation_offset")
_PHILOX_ROUNDS = {"philox4x32-7": 7, "philox4x32-10": 10}


@dataclass(frozen=True)
class SRStoreSpec:
    name: str
    source_dtype: object = float32
    destination_dtype: object = bfloat16
    site_id: int = 0

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or any(c in self.name for c in ",="):
            raise ValueError("SR sites require nonempty names without comma or equals")
        if type(self.site_id) is not int or not 0 <= self.site_id < 1 << 31:
            raise ValueError("SR site IDs must be nonnegative i32 integers")
        if any(not isinstance(value, dtype) or not value.is_floating()
               for value in (self.source_dtype, self.destination_dtype)):
            raise ValueError("SR store dtype declarations must be floating point")


@dataclass(frozen=True)
class StochasticRoundingSpec:
    site: SRStoreSpec | None = None
    version: int = 1
    prng: str = "philox4x32-10"
    sites: tuple[SRStoreSpec, ...] = ()
    ownership: str = "checked"
    overflow: str = "satfinite"
    entropy: str = "element"

    def __post_init__(self):
        sites = (self.site, ) if self.site is not None else tuple(self.sites)
        if self.site is not None and self.sites:
            raise ValueError("specify site or sites, not both")
        if not sites or any(not isinstance(site, SRStoreSpec) for site in sites) or self.version != 1:
            raise ValueError("requires SRStoreSpec declarations and ABI version one")
        if len({site.name for site in sites}) != len(sites) or len({site.site_id for site in sites}) != len(sites):
            raise ValueError("SR site names and numeric IDs must be unique")
        object.__setattr__(self, "sites", () if self.site is not None else sites)
        if self.overflow not in ("ieee", "satfinite"):
            raise ValueError("SR overflow must be ieee or satfinite")
        if self.entropy not in ("element", "packed8"):
            raise ValueError("SR entropy must be element or packed8")
        if self.ownership not in ("checked", "declared"):
            raise ValueError("SR ownership must be checked or declared")
        if not isinstance(self.prng, str) or self.prng not in _PHILOX_ROUNDS:
            raise ValueError("ABI v1 supports philox4x32-7 or philox4x32-10")

    @property
    def store_specs(self):
        return (self.site, ) if self.site is not None else self.sites


@dataclass(frozen=True)
class StochasticRoundingState:
    """Runtime policy and RNG identity for one logical invocation.

    The runtime owns the device seed tensor and reuses invocation_offset for
    recomputation. Graph replay reads the tensor without calling the provider.
    """

    enabled: bool = False
    base_seed: int = 0
    dynamic_seed: object = None
    invocation_offset: int = 0


_state: contextvars.ContextVar[StochasticRoundingState | None] = contextvars.ContextVar(
    "triton_stochastic_rounding", default=None)
_provider: Callable[[StochasticRoundingKernel], StochasticRoundingState] | None = None
_interpreter_launch = contextvars.ContextVar("sr_interpreter_launch", default=None)


def set_stochastic_rounding_provider(
    provider: Callable[[StochasticRoundingKernel], StochasticRoundingState] | None,
) -> Callable[[StochasticRoundingKernel], StochasticRoundingState] | None:
    """Register a process-wide provider; return the old one.

    The provider receives the decorated kernel (including its fn and spec) on
    each launch, warmup or preload and returns current policy and RNG state.
    Explicit stochastic_rounding contexts take priority. None unregisters it.
    """
    if provider is not None and not callable(provider):
        raise TypeError("SR provider must be callable or None")
    global _provider
    previous = _provider
    _provider = provider
    return previous


def _validate_state(base_seed, dynamic_seed, invocation_offset):
    for value in (base_seed, invocation_offset):
        if type(value) is not int or not 0 <= value < 1 << 64:
            raise ValueError("SR base seed and invocation offset must be uint64 integers")
    if (dynamic_seed is None or (not knobs.runtime.interpret and not getattr(dynamic_seed, "is_cuda", False))
            or str(dynamic_seed.dtype) not in ("torch.int64", "torch.uint64") or dynamic_seed.numel() != 1
            or not dynamic_seed.is_contiguous()):
        raise ValueError("SR dynamic seed must be one contiguous CUDA int64/uint64 element")


def _validate_policy(state: StochasticRoundingState) -> None:
    if not isinstance(state, StochasticRoundingState):
        raise TypeError("SR provider must return StochasticRoundingState")
    if type(state.enabled) is not bool:
        raise ValueError("SR policy enabled must be a Python bool")
    if state.enabled:
        _validate_state(state.base_seed, state.dynamic_seed, state.invocation_offset)


def _resolve_state(kernel: StochasticRoundingKernel) -> StochasticRoundingState:
    state = _state.get()
    if state is not None:
        return state
    provider = _provider
    if provider is None:
        return StochasticRoundingState()
    state = provider(kernel)
    _validate_policy(state)
    return state


@contextmanager
def stochastic_rounding(*, enabled=True, base_seed=0, dynamic_seed=None, invocation_offset=0):
    """Override the registered provider within the current Python context.

    Recompute repeats a stream when all four state fields and logical indices
    are reused. The caller owns the seed tensor and its graph lifetime. This
    context does not update the seed, synchronize streams, or allocate memory.
    """
    state = StochasticRoundingState(enabled, base_seed, dynamic_seed, invocation_offset)
    _validate_policy(state)
    token = _state.set(state)
    try:
        yield
    finally:
        _state.reset(token)


def _clone_jit(original, enabled):
    function = types.FunctionType(original.fn.__code__, original.fn.__globals__, original.fn.__name__,
                                  original.fn.__defaults__, original.fn.__closure__)
    function.__kwdefaults__ = original.fn.__kwdefaults__
    function.__annotations__ = dict(original.fn.__annotations__)
    function.__qualname__ = original.fn.__qualname__ + ("__sr_enabled" if enabled else "__sr_plain")
    signature = original.signature
    if any(name.startswith("__sr_") for name in signature.parameters):
        raise ValueError("kernel argument names beginning with __sr_ are reserved")
    if enabled:
        seeds = [
            inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation="*u64" if i == 0 else "u64")
            for i, name in enumerate(_SEED_ARGS)
        ]
        signature = signature.replace(parameters=seeds + list(signature.parameters.values()))
    function.__signature__ = signature
    launch_metadata = original.launch_metadata
    if launch_metadata is not None:

        def launch_metadata(grid, metadata, args):
            return original.launch_metadata(grid, metadata, {k: v for k, v in args.items() if k not in _SEED_ARGS})

    child = JITFunction(
        function, version=original.version,
        do_not_specialize=([p.name for p in original.params if p.do_not_specialize] +
                           (list(_SEED_ARGS) if enabled else [])),
        do_not_specialize_on_alignment=[p.name
                                        for p in original.params
                                        if p.do_not_specialize_on_alignment], debug=original.debug,
        noinline=original.noinline, repr=original._repr, launch_metadata=launch_metadata, c_cache=original.c_cache)
    source = original.src
    if enabled:
        definition = ast.parse(source).body[0]
        opening = source.index("(", source.index("def " + definition.name)) + 1
        source = source[:opening] + ", ".join(name + (": '*u64'" if i == 0 else ": 'u64'")
                                              for i, name in enumerate(_SEED_ARGS)) + ", " + source[opening:]
    child._unsafe_update_src(source)

    def adapt_hook(hook):

        def invoke(*args, **kwargs):
            return hook(*args[3:] if enabled else args, **{k: v for k, v in kwargs.items() if not k.startswith("sr_")})

        return invoke

    child.pre_run_hooks = [adapt_hook(hook) for hook in original.pre_run_hooks]
    return child


def _adapt_callback(callback):
    if callback is None:
        return None

    def clean(mapping):
        return {k: v for k, v in mapping.items() if k not in _SEED_ARGS and not k.startswith("sr_")}

    def invoke(*args, **kwargs):
        return callback(*(clean(arg) if isinstance(arg, dict) else arg for arg in args), **clean(kwargs))

    return invoke


def _clone_kernel(kernel, enabled):
    if isinstance(kernel, JITFunction):
        return _clone_jit(kernel, enabled)
    if knobs.runtime.interpret:
        from .interpreter import InterpretedFunction
        if isinstance(kernel, InterpretedFunction):
            return kernel
    if isinstance(kernel, Heuristics):
        child = _clone_kernel(kernel.fn, enabled)
        return Heuristics(child, child.arg_names, {k: _adapt_callback(v) for k, v in kernel.values.items()})
    if isinstance(kernel, Autotuner):
        child = _clone_kernel(kernel.fn, enabled)
        prune = {
            "perf_model": _adapt_callback(kernel.perf_model), "top_k": kernel.configs_top_k, "early_config_prune":
            _adapt_callback(kernel.early_config_prune), "ir_config_prune": kernel.ir_config_prune
        }
        configs = [copy.copy(config) for config in kernel.configs]
        for config in configs:
            config.pre_hook = _adapt_callback(config.pre_hook)
        return Autotuner(child, child.arg_names, configs, kernel.keys, kernel.reset_to_zero, kernel.restore_value,
                         pre_hook=_adapt_callback(kernel.pre_hook) if kernel.user_defined_pre_hook else None,
                         post_hook=_adapt_callback(kernel.post_hook) if kernel.user_defined_post_hook else None,
                         prune_configs_by=prune, do_bench=kernel._do_bench, cache_results=kernel.cache_results,
                         include_npot=kernel.include_npot)
    raise TypeError("SR decorator supports JITFunction, autotune and heuristics wrappers")


def _validate_buffers(args, kwargs):
    tensors = [value for value in (*args, *kwargs.values()) if hasattr(value, "data_ptr")]
    if any(not getattr(tensor, "is_cuda", False) for tensor in tensors):
        raise ValueError("SR ABI v1 requires CUDA tensor arguments")
    if any(tensor.device != tensors[0].device for tensor in tensors):
        raise ValueError("SR tensor arguments and seed must use the same CUDA device")


def _checked_grid(grid):

    def validate(meta=None):
        result = grid({k: v for k, v in meta.items() if k not in _SEED_ARGS}) if callable(grid) else grid
        if not 1 <= len(result) <= 3 or any(value != 1 for value in result[1:]):
            raise ValueError("SR ABI v1 requires a one-dimensional launch grid")
        return result

    return validate if callable(grid) else validate()


class StochasticRoundingKernel(KernelInterface):

    def __init__(self, kernel, spec):
        self.fn = kernel
        self.spec = spec
        self.arg_names = kernel.arg_names
        self._plain = _clone_kernel(kernel, False)
        self._enabled = _clone_kernel(kernel, True)

    def _launch(self, grid, args, kwargs, warmup):
        state = _resolve_state(self)
        child = self._enabled if state.enabled else self._plain
        if knobs.runtime.interpret:
            token = _interpreter_launch.set((self.spec, state, set()))
            try:
                return child.run(*args, grid=grid, warmup=warmup, **kwargs)
            finally:
                _interpreter_launch.reset(token)
        kwargs = dict(kwargs)
        if any(name.startswith("sr_") or name in _SEED_ARGS for name in kwargs):
            raise ValueError("SR variant options and seed arguments are owned by the proxy")
        kwargs.update(sr_mode="enabled" if state.enabled else "plain",
                      sr_site=",".join(f"{site.name}={site.site_id}" for site in self.spec.store_specs),
                      sr_rounds=_PHILOX_ROUNDS[self.spec.prng], sr_ownership=self.spec.ownership,
                      sr_overflow=self.spec.overflow, sr_entropy=self.spec.entropy)
        if state.enabled:
            values = dict(zip(self.arg_names, args))
            values.update(kwargs)
            kwargs["sr_launch"] = tuple(
                sorted((name, value)
                       for name, value in values.items()
                       if name in self.arg_names and type(value) is int and -(1 << 63) <= value < (1 << 63)))
            kwargs["sr_grid"] = 0 if callable(grid) else grid[0]
            args = (state.dynamic_seed, state.base_seed, state.invocation_offset, *args)
        if state.enabled and not warmup:
            _validate_buffers(args, kwargs)
        checked_grid = _checked_grid(grid) if state.enabled else grid
        if warmup:
            return child.warmup(*args, grid=checked_grid, **kwargs)
        return child[checked_grid](*args, **kwargs)

    def __getitem__(self, grid):
        # Read policy at invocation, including when a launcher is reused.
        return lambda *args, **kwargs: self._launch(grid, args, kwargs, False)

    def run(self, *args, grid, warmup=False, **kwargs):
        return self._launch(grid, args, kwargs, warmup)

    def warmup(self, *args, grid, **kwargs):
        return self._launch(grid, args, kwargs, True)

    def preload(self, specialization_data):
        state = _resolve_state(self)
        child = self._enabled if state.enabled else self._plain
        if not isinstance(child, JITFunction):
            raise TypeError("preload requires an undecorated JIT child")
        return child.preload(specialization_data)


def triton_stochastic_rounding(spec):
    """Build independent plain/SR children. Apply outside autotune/heuristics."""
    if not isinstance(spec, StochasticRoundingSpec):
        raise TypeError("expected StochasticRoundingSpec")
    return lambda kernel: StochasticRoundingKernel(kernel, spec)
