"""Op catalog and dispatch."""

from __future__ import annotations

import dataclasses
import functools
import importlib
from typing import Any, Callable, Mapping, Optional


class UnsupportedOp(RuntimeError):
    """No catalog entry for this op on the current target."""


class UnsupportedBackward(UnsupportedOp):
    """The selected op implementation has no autograd support."""


class InvalidInput(ValueError):
    """An entry exists but cannot handle these inputs."""


@dataclasses.dataclass(frozen=True)
class OpSpec:
    op: str
    arch: str  # must equal Target.key
    variant: str  # label only; not a dispatch input, not user visible
    impl: str  # "absolute.module:attr", imported on first use
    dtypes: frozenset = frozenset()  # bare torch names, so the table needs no torch
    accepts: Optional[Callable[[Mapping[str, Any]], bool]] = None
    requires: frozenset = frozenset()
    supports_backward: bool = False

    def __str__(self) -> str:
        return f"{self.op}/{self.arch} ({self.variant})"


_FP16 = frozenset({"float16", "bfloat16"})
_BF16 = frozenset({"bfloat16"})

# A static table, not decorator self-registration: `impl` stays a string so
# `import triton.tlx` never imports a kernel module or builds autotune configs.
# A missing (op, arch) is deferred, not an error -- the op raises on that arch.
CATALOG: tuple[OpSpec, ...] = (
    OpSpec(
        op="mm",
        arch="sm100",
        variant="ws",
        impl="triton.tlx.ops.kernels.mm.sm100:mm",
        dtypes=_FP16,
        # TMA needs 16-byte-aligned descriptor row strides. Checked against the
        # real strides, not M/N/K: a column-major operand is fed to its
        # descriptor transposed, which moves the constraint to another dim.
        accepts=lambda d: all(s * d["elem_bytes"] % 16 == 0 for s in d["row_strides"]),
        requires=frozenset({"tma", "tmem"}),
    ),
    OpSpec(
        op="mm",
        arch="gfx942",
        variant="direct_load",
        impl="triton.tlx.ops.kernels.mm.gfx942:mm",
        dtypes=_FP16,
        # No `accepts`: operands are read through explicit strides rather than a
        # descriptor, so there is no alignment rule to fail. This arch therefore
        # admits shapes sm100 declines -- see kernels/mm/_shapes.py.
        requires=frozenset(),
    ),
    OpSpec(
        op="addmm",
        arch="gfx942",
        variant="fused_gemm",
        impl="triton.tlx.ops.kernels.addmm.gfx942:addmm",
        dtypes=_FP16,
        requires=frozenset(),
    ),
    OpSpec(
        op="mm",
        arch="gfx950",
        variant="heuristic",
        impl="triton.tlx.ops.kernels.mm.gfx950:mm",
        # LocalSplitU and persistent plans remain FP16-only; the register and
        # general LDS paths also support BF16 and validate their own plans.
        dtypes=_FP16,
        requires=frozenset(),
    ),
    OpSpec(
        # torchTLX: the same mm through torch.compile. Benchmark-only, so it has
        # no `tlx.ops` wrapper; the entry exists so the perf suite can gate on it.
        op="mm_torchtlx",
        arch="sm100",
        variant="inductor_blackwell_gemm_ws",
        impl="triton.language.extra.tlx.inductor.sm100_torch:mm",
        dtypes=_FP16,
        accepts=lambda d: all(s * d["elem_bytes"] % 16 == 0 for s in d["row_strides"]),
        requires=frozenset({"tma", "tmem"}),
    ),
    OpSpec(
        op="mm_torchtlx",
        arch="gfx950",
        variant="inductor_gfx950_mm",
        impl="triton.language.extra.tlx.inductor.gfx950_torch:mm",
        dtypes=_FP16,
        requires=frozenset(),
    ),
    OpSpec(
        # TorchTLX providers are benchmark/catalog entries rather than public
        # wrappers: their API is torch.addmm, with TLX selected by Inductor.
        op="addmm_torchtlx",
        arch="gfx950",
        variant="inductor_gfx950_addmm",
        impl="triton.tlx.ops.kernels.addmm.gfx950_torch:addmm",
        dtypes=_FP16,
        requires=frozenset(),
    ),
    OpSpec(
        op="bmm_torchtlx",
        arch="gfx950",
        variant="inductor_gfx950_bmm",
        impl="triton.tlx.ops.kernels.bmm.gfx950_torch:bmm",
        dtypes=_FP16,
        requires=frozenset(),
    ),
    OpSpec(
        op="flash_attn",
        arch="sm90",
        variant="ws_pipelined_pingpong",
        impl="triton.tlx.ops.kernels.flash_attn.sm90:flash_attn",
        dtypes=_FP16,
        accepts=lambda d: d.get("HEAD_DIM") in (64, 128),
        requires=frozenset({"tma"}),
        supports_backward=True,
    ),
    OpSpec(
        op="flash_attn",
        arch="sm100",
        variant="ws_pipelined_persistent",
        impl="triton.tlx.ops.kernels.flash_attn.sm100:flash_attn",
        dtypes=_FP16,
        accepts=lambda d: d.get("HEAD_DIM") in (64, 128),
        requires=frozenset({"tma", "tmem"}),
        supports_backward=True,
    ),
    OpSpec(
        op="flash_attn_mxfp8",
        arch="sm100",
        variant="ws_pipelined_persistent_mxfp8",
        impl="triton.tlx.ops.kernels.flash_attn_mxfp8.sm100:flash_attn_mxfp8",
        dtypes=_BF16,
        accepts=lambda d: d.get("HEAD_DIM") == 128 and d.get("N_CTX", 0) % 256 == 0,
        requires=frozenset({"tma", "tmem"}),
        supports_backward=True,
    ),
    OpSpec(
        op="hstu_attn_dev",
        arch="sm100",
        variant="ws",
        impl="triton.tlx.ops.kernels.hstu_attn.sm100:hstu_attn",
        dtypes=_FP16,
        # Causal-only, non-causal is not supported yet
        accepts=lambda d: bool(d.get("causal", True)),
        requires=frozenset({"tma", "tmem"}),
        supports_backward=True,
    ),
    OpSpec(
        op="hstu_attn_dev",
        arch="gfx950",
        variant="tlx",
        impl="triton.tlx.ops.kernels.hstu_attn.gfx950:hstu_attn",
        dtypes=_FP16,
        # Causal-only, non-causal is not supported yet.
        accepts=lambda d: bool(d.get("causal", True)),
        supports_backward=True,
    ),
    OpSpec(
        op="kimi_delta_attention",
        arch="sm100",
        variant="ws",
        impl="triton.tlx.ops.kernels.kda.sm100:kimi_delta_attention",
        dtypes=_FP16,
        accepts=lambda d: d.get("HEAD_DIM") == 128,
        requires=frozenset({"tma", "tmem"}),
        supports_backward=True,
    ),
    OpSpec(
        op="kda_paged_prefill",
        arch="gfx950",
        variant="tlx",
        impl="triton.tlx.ops.kernels.kda.gfx950_prefill:kda_paged_prefill",
        dtypes=_BF16,
        accepts=lambda d: d.get("KEY_DIM") == 128 and d.get("VALUE_DIM") == 128,
    ),
    OpSpec(
        op="kda_recurrent_decode",
        arch="gfx950",
        variant="tlx",
        impl="triton.tlx.ops.kernels.kda.gfx950_decode:kda_recurrent_decode",
        dtypes=_BF16,
        accepts=lambda d: 1 <= d.get("KEY_DIM", 0) <= 128 and 1 <= d.get("VALUE_DIM", 0) <= 128,
    ),
)

_BY_KEY = {(s.op, s.arch): s for s in CATALOG}
assert len(_BY_KEY) == len(CATALOG), "duplicate (op, arch) in CATALOG"


def _target(device=None):
    # Lazy: hw.target imports torch.
    from triton.language.extra.tlx.hw.target import current_target, target_for_device

    return current_target() if device is None else target_for_device(device)


def _capabilities(target) -> frozenset:
    if target.spec is None:
        return frozenset()
    caps = set()
    if target.is_cuda and target.capability is not None and target.capability[0] >= 9:
        caps |= {"tma", "cluster"}
    if getattr(target, "has_tmem", False):
        caps.add("tmem")
    return frozenset(caps)


@functools.lru_cache(maxsize=None)
def _load(impl: str) -> Callable[..., Any]:
    mod, _, attr = impl.partition(":")
    return getattr(importlib.import_module(mod), attr)


@functools.lru_cache(maxsize=None)
def _arches_for(op: str) -> tuple[str, ...]:
    return tuple(sorted(s.arch for s in CATALOG if s.op == op))


def has_impl(op: str, arch: str) -> bool:
    """Is there a catalog entry for this pair, without importing the kernel?

    A table lookup, not a capability check: the benchmark suite uses it to skip
    an op cleanly on an arch it was never written for, rather than running every
    shape and reporting each one as an error.
    """
    return (op, arch) in _BY_KEY


@functools.lru_cache(maxsize=None)
def _impl_for_arch(op: str, arch: str) -> tuple[Callable[..., Any], OpSpec]:
    spec = _BY_KEY.get((op, arch))
    if spec is None:
        available = ", ".join(_arches_for(op)) or "(nothing yet)"
        raise UnsupportedOp(f"tlx.ops.{op} has no implementation for arch={arch!r}. Available on: {available}")
    return _load(spec.impl), spec


def impl_for(op: str, arch: Optional[str] = None, *, device=None) -> tuple[Callable[..., Any], OpSpec]:
    """The blessed callable for `op`, plus its spec.

    Raises rather than falling back: a silent fallback turns "TLX is not
    running here" into an unexplained performance cliff.

    Public op wrappers pass the input tensor's `device`, so dispatch follows
    the device that will execute the kernel rather than an ambient current
    device. An explicit `arch` is retained for private catalog tests; it pins
    the entry and skips the capability check.
    """
    if arch is not None and device is not None:
        raise ValueError("impl_for accepts either arch or device, not both")
    if arch is None:
        target = _target(device)
        if not target.key:
            available = ", ".join(_arches_for(op)) or "(nothing yet)"
            location = f" for device={device}" if device is not None else ""
            raise UnsupportedOp(f"tlx.ops.{op}: could not determine a GPU architecture{location}. "
                                f"Available on: {available}")
        spec = _BY_KEY.get((op, target.key))
        if spec is None:
            available = ", ".join(_arches_for(op)) or "(nothing yet)"
            raise UnsupportedOp(f"tlx.ops.{op} has no implementation for {target.key}. "
                                f"Available on: {available}")
        missing = spec.requires - _capabilities(target)
        if missing:
            raise UnsupportedOp(f"{spec} needs {sorted(missing)}, which {target.key} does not report")
        return _load(spec.impl), spec

    return _impl_for_arch(op, arch)


def check_inputs(spec: OpSpec, dtype=None, **dims) -> None:
    if dtype is not None and spec.dtypes:
        name = str(dtype).removeprefix("torch.")
        if name not in spec.dtypes:
            raise InvalidInput(f"{spec} does not support {name}; supported: {sorted(spec.dtypes)}")
    if spec.accepts is not None and not spec.accepts(dims):
        raise InvalidInput(f"{spec} does not support these inputs: {dims}")


def check_backward(spec: OpSpec, *inputs) -> None:
    """Fail before launch when autograd is requested but unavailable."""
    if spec.supports_backward or not any(getattr(tensor, "requires_grad", False) for tensor in inputs):
        return

    import torch

    if torch.is_grad_enabled():
        raise UnsupportedBackward(f"tlx.ops.{spec.op} does not support backward on {spec.arch}; "
                                  "use tensors with requires_grad=False or call it under torch.no_grad()")
