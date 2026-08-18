"""Single source of truth for GPU target detection.

Every architecture query in the torchTLX Inductor integration goes through this
module.  Adding an arch should be one edit here rather than a new
``torch.version.hip`` / ``gcnArchName`` sniff at the call site.

Two entry points, deliberately split by cost:

- :func:`is_rocm` only reads ``torch.version.hip``.  It is safe at import time
  (heuristic registration needs it) and never initializes a CUDA context.
- :func:`current_target` queries the device.  It is cached and lazy, and falls
  back to Blackwell B200 reference values when no device is visible so the
  CPU-only heuristic tests stay deterministic on a build host.
"""

from __future__ import annotations

import dataclasses
import functools

import torch

# Blackwell B200 (sm_100a) reference values, also the no-device fallbacks.
B200_NUM_SMS = 148
# Upstream uses 232*1024 as a loose estimate; this is the real hardware limit.
B200_SMEM_BYTES = 232448
# TMEM is 128 lanes x 512 columns of 32 bits.
B200_TMEM_COLUMNS = 512

# AMD parts with 8 XCDs (chiplets), for the L2 chiplet swizzle.
_XCD8_ARCHES = ("gfx942", "gfx950")


def is_rocm() -> bool:
    """True on a ROCm build.  Cheap: no device query, safe at import time."""
    return torch.version.hip is not None


@dataclasses.dataclass(frozen=True)
class Target:
    """Resolved properties of the GPU torchTLX is generating code for."""

    is_rocm: bool
    #: ``gcnArchName`` on ROCm; empty on CUDA and on CPU-only hosts.
    arch: str
    #: CUDA compute capability; ``None`` on ROCm and CPU-only hosts.
    capability: tuple[int, int] | None
    num_sms: int
    smem_bytes: int
    tmem_columns: int

    @property
    def is_cuda(self) -> bool:
        return not self.is_rocm

    @property
    def is_hopper(self) -> bool:
        return self.capability is not None and self.capability[0] == 9

    @property
    def is_blackwell(self) -> bool:
        return self.capability is not None and self.capability[0] == 10

    @property
    def is_gfx942(self) -> bool:
        """AMD MI300X (CDNA3)."""
        return "gfx942" in self.arch

    @property
    def is_gfx950(self) -> bool:
        """AMD MI350X (CDNA4)."""
        return "gfx95" in self.arch

    @property
    def num_xcds(self) -> int:
        """XCDs (chiplets) on this part.

        Returns 1 -- which makes the L2 chiplet swizzle an identity -- for
        non-HIP parts and for arches with no known XCD count, since the AMD
        templates are registered for all of ROCm, not just the 8-XCD parts.
        """
        return 8 if any(a in self.arch for a in _XCD8_ARCHES) else 1

    def default_kpack(self, block_k: int = 16) -> int:
        """Mirror of ``torch._inductor.utils.get_default_kpack``.

        Used as a fallback on torch nightlies that do not export it.
        """
        if not self.is_rocm:
            return 0
        return 1 if (self.is_gfx942 and block_k <= 16) else 2


@functools.lru_cache(maxsize=1)
def current_target() -> Target:
    """The current GPU target, cached.

    Call ``current_target.cache_clear()`` to re-resolve (tests that monkeypatch
    device properties need this).
    """
    rocm = is_rocm()
    arch = ""
    capability: tuple[int, int] | None = None
    num_sms = B200_NUM_SMS

    try:
        props = torch.cuda.get_device_properties(0)
    except Exception:
        # No visible device (CPU-only build host, or a driver that failed to
        # initialize).  Keep the B200 reference values so the pure-Python
        # heuristics stay callable and deterministic.
        props = None

    if props is not None:
        num_sms = props.multi_processor_count
        if rocm:
            arch = getattr(props, "gcnArchName", "")
        else:
            capability = (props.major, props.minor)

    # SMEM/TMEM stay at the B200 figures for now: the template's SMEM estimate
    # is calibrated against them, and the Blackwell GEMM heuristic is the only
    # consumer.  ``shared_memory_per_block_optin`` reports 232448 on both H100
    # and B200, so querying it would not change any decision today -- wire it
    # up alongside the first non-Blackwell CUDA target.
    return Target(
        is_rocm=rocm,
        arch=arch,
        capability=capability,
        num_sms=num_sms,
        smem_bytes=B200_SMEM_BYTES,
        tmem_columns=B200_TMEM_COLUMNS,
    )
