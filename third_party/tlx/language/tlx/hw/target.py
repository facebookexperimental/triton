"""Which GPU are we generating for?

Detection and resolution only -- the hardware facts themselves live in the arch
classes in :mod:`.resources`. Every architecture query in the torchTLX Inductor
integration goes through here rather than a fresh ``torch.version.hip`` /
``gcnArchName`` sniff at the call site.

Arch ids follow the vocabulary in ``triton._internal_testing`` -- capability
strings on NVIDIA (``sm90``, ``sm100``, ``sm107``) and gfx names on AMD
(``gfx942``, ``gfx950``, ``gfx1250``) -- so the two agree on what "the same
target" means. Note the ordering hazard that module documents: ``gfx1250`` must
be matched before any ``gfx12`` prefix rule, or an RDNA4 check swallows it.

Two entry points, deliberately split by cost:

- :func:`is_rocm` only reads ``torch.version.hip``. It is safe at import time
  (heuristic registration needs it) and never initializes a CUDA context.
- :func:`current_target` queries the device, cached and lazy. With no GPU
  visible it reports no arch at all, so a build host can never be mistaken for
  real hardware; see :data:`REFERENCE_ARCH`.
"""

from __future__ import annotations

import dataclasses
import functools

import torch

from .resources import ARCH_SPECS, Sm100, has_tmem

#: Supplies numbers -- and only numbers -- when no GPU is visible, so the
#: pure-Python heuristics stay callable and deterministic on a build host.
#:
#: Deliberately not presented as the current target: ``current_target()`` there
#: returns ``spec=None``, so every ``is_*`` predicate is False. Reading a
#: *number* off a device-less target gives Blackwell's; asking *what arch this
#: is* gives nothing.
REFERENCE_ARCH = Sm100


def is_rocm() -> bool:
    """True on a ROCm build. Cheap: no device query, safe at import time."""
    return torch.version.hip is not None


def _spec_for_cuda(capability: tuple[int, int]) -> type | None:
    """Arch class for a CUDA compute capability, or None if unrecognized.

    A table walk, like its ROCm counterpart: each class decides whether a
    capability is its own, and ARCH_SPECS order breaks the overlap between a
    pinned entry and the broader one it sits inside.
    """
    for spec in ARCH_SPECS.values():
        if spec.vendor == "nvidia" and spec.matches_capability(capability):
            return spec
    return None


def _spec_for_rocm(gcn_arch_name: str) -> type | None:
    """Arch class for a ``gcnArchName``, or None if unrecognized."""
    for spec in ARCH_SPECS.values():
        if spec.vendor == "amd" and spec.matches_arch(gcn_arch_name):
            return spec
    return None


@dataclasses.dataclass(frozen=True)
class Target:
    """Resolved properties of the GPU torchTLX is generating code for."""

    #: The resolved arch class, or None when no GPU is visible. None makes
    #: every arch predicate below False, so a build host is never mistaken for
    #: hardware; the numeric fields still carry REFERENCE_ARCH values so the
    #: pure-Python heuristics stay callable there.
    spec: type | None
    is_rocm: bool
    #: ``gcnArchName`` on ROCm; empty on CUDA and on CPU-only hosts.
    arch: str
    #: CUDA compute capability; ``None`` on ROCm and CPU-only hosts.
    capability: tuple[int, int] | None
    #: Processor count as the driver reports it (``multi_processor_count``):
    #: SMs on NVIDIA, CUs on AMD. See :attr:`num_cus` for the AMD spelling.
    num_sms: int
    #: Fast on-chip memory per block/workgroup as the driver reports it: SMEM
    #: on NVIDIA, LDS on AMD. See :attr:`lds_bytes` for the AMD spelling.
    smem_bytes: int
    #: TMEM columns; None on every arch without tensor memory.
    tmem_columns: int | None

    @property
    def lds_bytes(self) -> int:
        """AMD spelling of :attr:`smem_bytes`.

        Shared memory and LDS are the same budget under two vendors' names, and
        the driver reports one number, so this is an alias rather than a second
        value that could drift.
        """
        return self.smem_bytes

    @property
    def num_cus(self) -> int:
        """AMD spelling of :attr:`num_sms`.

        The driver reports one number through ``multi_processor_count``, so
        this is the same value under the name the CDNA docs and the AMD
        templates use.
        """
        return self.num_sms

    @property
    def has_device(self) -> bool:
        """Whether a real GPU was visible when this target was resolved."""
        return self.spec is not None

    @property
    def key(self) -> str:
        """Arch id, or "" when no device was visible."""
        return self.spec.key if self.spec is not None else ""

    @property
    def is_cuda(self) -> bool:
        return not self.is_rocm

    @property
    def has_tmem(self) -> bool:
        """Whether the target has tensor memory (Blackwell only).

        Check this before reading :attr:`tmem_columns`, which is None on a
        target without it.
        """
        return self.spec is not None and has_tmem(self.spec)

    @property
    def is_hopper(self) -> bool:
        return self.spec is not None and self.spec.key == "sm90"

    @property
    def is_blackwell(self) -> bool:
        return self.spec is not None and self.spec.key == "sm100"

    @property
    def is_rubin(self) -> bool:
        return self.spec is not None and self.spec.key == "sm107"

    @property
    def is_gfx942(self) -> bool:
        """AMD MI300X (CDNA3)."""
        return self.spec is not None and self.spec.key == "gfx942"

    @property
    def is_gfx950(self) -> bool:
        """AMD MI350X (CDNA4)."""
        return self.spec is not None and self.spec.key == "gfx950"

    @property
    def is_gfx1250(self) -> bool:
        """AMD MI450X."""
        return self.spec is not None and self.spec.key == "gfx1250"

    @property
    def num_xcds(self) -> int:
        """XCDs (chiplets) on this part, for the AMD L2 swizzle.

        AttributeError on a non-AMD target: NVIDIA arch classes do not define
        ``num_xcds`` at all, so a caller that forgot to gate on
        :attr:`is_rocm` fails where the bug is.
        """
        return self.spec.num_xcds

    def default_kpack(self, block_k: int = 16) -> int:
        """Mirror of ``torch._inductor.utils.get_default_kpack``.

        Used as a fallback on torch nightlies that do not export it.
        """
        if not self.is_rocm:
            return 0
        return 1 if (self.is_gfx942 and block_k <= 16) else 2


def _device_smem_bytes(props, rocm: bool) -> int | None:
    """Per-block SMEM/LDS reported by the driver, or None if unavailable."""
    for attr in ("shared_memory_per_block_optin", "shared_memory_per_block"):
        value = getattr(props, attr, None)
        if value:
            return int(value)
    return None


@functools.lru_cache(maxsize=1)
def current_target() -> Target:
    """The current GPU target, cached.

    Call ``current_target.cache_clear()`` to re-resolve (tests that monkeypatch
    device properties need this).
    """
    rocm = is_rocm()

    try:
        props = torch.cuda.get_device_properties(0)
    except Exception:
        # No visible device (CPU-only build host, or a driver that failed to
        # initialize). Fall back to the reference arch so the pure-Python
        # heuristics stay callable and deterministic.
        props = None

    if props is None:
        # No arch: every predicate stays False. The numbers come from
        # REFERENCE_ARCH purely so the pure-Python heuristics remain callable.
        return Target(
            spec=None,
            is_rocm=rocm,
            arch="",
            capability=None,
            num_sms=REFERENCE_ARCH.processor_count(),
            smem_bytes=REFERENCE_ARCH.on_chip_bytes(),
            tmem_columns=None,
        )

    arch = ""
    capability: tuple[int, int] | None = None
    if rocm:
        arch = getattr(props, "gcnArchName", "")
        spec = _spec_for_rocm(arch)
        if spec is None:
            # Do NOT fall back to a neighbouring part: inheriting gfx950's
            # 160KB LDS and 8 XCDs on an unrecognized arch silently produces
            # configs that may not fit and a swizzle that does not match.
            raise ValueError(f"unrecognized AMD arch {arch!r}. torchTLX targets "
                             f"{sorted(k for k, s in ARCH_SPECS.items() if s.vendor == 'amd')}; "
                             f"add a row to ARCH_SPECS to support it")
    else:
        capability = (props.major, props.minor)
        spec = _spec_for_cuda(capability)
        if spec is None:
            # Do NOT fall through to Blackwell. A consumer part (major 12) has
            # a far smaller SMEM budget, so claiming it is a B200 would hand
            # out configs that cannot launch.
            raise ValueError(
                f"unrecognized CUDA compute capability {capability}. torchTLX "
                f"targets "
                f"{sorted(k for k, s in ARCH_SPECS.items() if s.vendor == 'nvidia')}; "
                f"add a class to ARCH_SPECS to support it"
            )

    # Prefer what the driver reports; the table is the fallback. This is what
    # lets a part we have not tuned for yet report its own limits.
    smem_bytes = _device_smem_bytes(props, rocm) or spec.on_chip_bytes()
    if not smem_bytes:
        raise ValueError(f"could not determine the SMEM/LDS budget for {spec.key}: the driver "
                         f"reported none and ARCH_SPECS has no static value")

    return Target(
        spec=spec,
        is_rocm=rocm,
        arch=arch,
        capability=capability,
        num_sms=props.multi_processor_count,
        smem_bytes=smem_bytes,
        tmem_columns=spec.tmem_columns() if has_tmem(spec) else None,
    )
