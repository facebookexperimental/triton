"""Hardware facts per architecture, and whether a tile config fits on them.

Arch classes, with no shared parent across vendors -- NVIDIA and AMD have no
common set of facts worth abstracting over. Within a lineage a newer part is
its predecessor plus what changed::

    NVIDIA                  AMD
    ------                  ---
    Sm90  (H100)            _AmdArch (accessors only, no facts)
      |                       |
    Sm100 (B200)              +-- Gfx942 (MI300X) --- Gfx950 (MI350X)
      |                       |
    Sm107 (Rubin)             +-- Gfx1250 (MI450X)

Sm107 and Gfx1250 are placeholders: their facts are not measured yet. Note the
AMD shape -- Gfx1250 hangs off the accessor base, NOT off Gfx950.

"""

from __future__ import annotations

import dataclasses
from typing import Any

#: Bytes per mbarrier.
MBARRIER_BYTES = 8
#: Bytes per element of the A/B operand buffers (fp16/bf16).
OPERAND_ELEM_BYTES = 2
#: torchTLX stores split-K partials as fp32 (upstream uses the output dtype).
SPLIT_K_PARTIAL_BYTES = 4

# ---------------------------------------------------------------------------
# NVIDIA architectures
# ---------------------------------------------------------------------------
#
# These are namespaces, never instantiated: the facts are class attributes and
# a subclass overrides only what its generation changed. Only verified numbers
# are pinned; an unmeasured fact is left undeclared so the device query
# supplies it and a stale constant cannot masquerade as a measurement.


class Sm90:
    """H100 (Hopper)."""

    key = "sm90"
    vendor = "nvidia"
    product = "H100 (Hopper)"
    #: Compute-capability majors this class covers, and an exact minor when the
    #: class is pinned to one. Resolution walks ARCH_SPECS in order and takes
    #: the first match, so a pinned entry must precede the broader one it
    #: overlaps (Sm107 before Sm100).
    cuda_majors: tuple[int, ...] = (9,)
    cuda_minor: int | None = None
    #: 228KB SMEM per SM; 227KB is the max opt-in per block.
    smem_bytes: int = 232448
    #: H100 SXM5. The PCIe part has 114 SMs, so this is only a fallback for a
    #: host with no visible device; the driver value always wins.
    num_sms: int = 132

    #: NVIDIA parts have no chiplets, so ``num_xcds`` is deliberately absent --
    #: reaching for it is an AttributeError at the buggy call site.

    @classmethod
    def matches_capability(cls, capability: tuple[int, int]) -> bool:
        """Whether a CUDA compute capability is this architecture."""
        major, minor = capability
        return major in cls.cuda_majors and cls.cuda_minor in (None, minor)

    @classmethod
    def on_chip_bytes(cls) -> int:
        """Fast on-chip memory one block may allocate."""
        return cls.smem_bytes

    @classmethod
    def processor_count(cls) -> int:
        """SMs. Fallback only -- ``multi_processor_count`` wins when present."""
        return cls.num_sms


class Sm100(Sm90):
    """B200 (Blackwell). Hopper plus tensor memory.

    ``tmem_columns`` is introduced here: no earlier NVIDIA part and no AMD part
    defines it, so its absence is how callers detect "this arch has no TMEM".
    """

    key = "sm100"
    product = "B200 (Blackwell)"
    #: Majors 10 and 11, mirroring triton._internal_testing.is_blackwell.
    cuda_majors = (10, 11)
    #: Verified on a B200: shared_memory_per_block_optin == 232448. Upstream
    #: uses 232*1024 as a loose estimate; this is the real hardware limit.
    smem_bytes = 232448
    num_sms = 148

    @classmethod
    def tmem_columns(cls) -> int:
        """TMEM is 128 lanes x 512 columns of 32 bits."""
        return 512


class Sm107(Sm100):
    """Rubin. Placeholder for future bring-up.

    Empty on purpose: every fact is inherited from Blackwell and is therefore
    PROVISIONAL, not measured. Replace each one as it is characterized, the way
    Sm100 overrides Sm90.
    """

    key = "sm107"
    product = "Rubin"
    #: Pinned to (10, 7), so this must precede Sm100 in ARCH_SPECS -- major 10
    #: alone would otherwise match Blackwell first.
    cuda_majors = (10,)
    cuda_minor = 7


# ---------------------------------------------------------------------------
# AMD architectures
# ---------------------------------------------------------------------------


class _AmdArch:
    """Accessor shape shared by AMD parts. Declares no hardware facts itself.

    Exists because the AMD parts are two separate lineages, so there is no one
    oldest part for the others to descend from the way Sm100 descends from
    Sm90.
    """

    vendor = "amd"

    @classmethod
    def matches_arch(cls, gcn_arch_name: str) -> bool:
        """Whether a ``gcnArchName`` is this architecture.

        Substring, because the driver appends feature suffixes
        (``gfx950:sramecc+:xnack-``). Resolution walks ARCH_SPECS in order and
        takes the first match, so a longer name must precede any shorter one it
        contains.
        """
        return cls.key in gcn_arch_name

    @classmethod
    def on_chip_bytes(cls) -> int:
        """Fast on-chip memory one workgroup may allocate."""
        return cls.lds_bytes

    @classmethod
    def processor_count(cls) -> int:
        """CUs. Fallback only -- ``multi_processor_count`` wins when present."""
        return cls.num_cus


class Gfx942(_AmdArch):
    """MI300X (CDNA3)."""

    key = "gfx942"
    product = "MI300X (CDNA3)"
    #: 64KB per workgroup on CDNA3 -- a quarter of CDNA4's, which is why a
    #: gfx950-tuned warp-pipe config does not transfer unchanged.
    lds_bytes: int = 65536
    num_cus: int = 304
    #: XCDs (chiplets), for the L2 swizzle. Consistent with the CU count:
    #: 304 = 8 x 38.
    #:
    #: Caveat: gfx942 is the ISA name shared by MI300A/MI300X/MI325X, and
    #: MI300A has 6 XCDs, not 8. This class is MI300X; an MI300A would swizzle
    #: wrongly. Keyed off the gfx name because that is all gcnArchName gives
    #: us -- revisit if MI300A support is ever needed.
    num_xcds: int = 8


class Gfx950(Gfx942):
    """MI350X (CDNA4). CDNA3 with a larger LDS and fewer, wider CUs."""

    key = "gfx950"
    product = "MI350X (CDNA4)"
    #: 160KB. Sourced from the hand-verified budget on the warp-pipe config
    #: table (registry.py WARPPIPE_CONFIGS), not from a datasheet, so the
    #: device query still takes precedence.
    lds_bytes = 163840
    #: 256 = 8 XCDs x 32 CUs, so the inherited XCD count holds.
    num_cus = 256


class Gfx1250(_AmdArch):
    """MI450X. Placeholder for future bring-up.

    Derives from _AmdArch, not Gfx950: gfx942/gfx950 are gfx9 (CDNA) while
    gfx1250 is a different ISA family with hardware CDNA lacks (TDM), so
    inheriting CDNA4's numbers would state them as fact.

    ``lds_bytes`` and ``num_cus`` therefore stay undeclared until measured; the
    device query supplies them at runtime and reading them here raises.
    """

    key = "gfx1250"
    product = "MI450X"
    #: 8, per AMD's own gfx1250 examples in this repo --
    #: third_party/amd/python/examples/gluon/f16_gemm_gfx1250.py passes
    #: num_xcds=8, and f16_gemm_streamk_gfx1250.py comments "8 XCDs".
    num_xcds: int = 8


#: Every architecture torchTLX targets, keyed by the id triton._internal_testing
#: uses.
#:
#: ORDER IS SIGNIFICANT. Resolution walks this dict and takes the first class
#: whose matcher accepts the device, so any entry that overlaps a broader one
#: must come first: Sm107 before Sm100 (both accept major 10), and a longer gfx
#: name before any shorter one it contains.
ARCH_SPECS: dict[str, type] = {
    "sm90": Sm90,
    "sm107": Sm107,
    "sm100": Sm100,
    "gfx1250": Gfx1250,
    "gfx942": Gfx942,
    "gfx950": Gfx950,
}


def has_tmem(spec: type) -> bool:
    """Whether an arch class declares tensor memory.

    TMEM is introduced at Blackwell, so the attribute simply does not exist on
    Hopper or on any AMD part -- absence is the signal, and reaching for
    ``tmem_columns`` there is an AttributeError rather than a plausible 0.
    """
    return hasattr(spec, "tmem_columns")


@dataclasses.dataclass(frozen=True)
class DeviceLimits:
    """On-chip resource limits for one target.

    ``tmem_columns`` is None on every architecture without tensor memory --
    not 0. A model that needs TMEM must reject such a target outright rather
    than compare against a budget that does not exist.
    """

    #: Fast on-chip memory one block/workgroup may allocate: SMEM on NVIDIA,
    #: LDS on AMD. Named for the concept, not either vendor's word for it,
    #: since a budget is all a resource model needs.
    on_chip_bytes: int
    tmem_columns: int | None = None

    @classmethod
    def current(cls) -> DeviceLimits:
        """Limits resolved from the live device.

        Prefer :data:`BLACKWELL_LIMITS` on the heuristic path: going through
        ``current_target`` would initialize a CUDA context inside what is
        otherwise pure, CPU-testable code.

        The import is deferred because target.py imports this module for the
        arch classes; at module scope it would be a cycle.
        """
        from .target import current_target

        target = current_target()
        return cls(on_chip_bytes=target.smem_bytes, tmem_columns=target.tmem_columns)

    @classmethod
    def for_arch(cls, key: str) -> DeviceLimits:
        """Limits for a named arch, for tests and cross-arch validation."""
        spec = ARCH_SPECS[key]
        # AttributeError here means the arch declares no static on-chip size;
        # resolve it from the device via DeviceLimits.current() instead.
        return cls(
            on_chip_bytes=spec.on_chip_bytes(),
            tmem_columns=spec.tmem_columns() if has_tmem(spec) else None,
        )


#: Blackwell limits as import-time constants, with no device query. Used where
#: the code must stay pure (the CPU-testable heuristic path). Named for its arch
#: on purpose: it is the default only for the Blackwell model, and a
#: non-Blackwell model must not silently inherit its budget and TMEM columns.
BLACKWELL_LIMITS = DeviceLimits.for_arch("sm100")

# ---------------------------------------------------------------------------
# Blackwell warp-specialized GEMM
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class BlackwellWSGemmConfig:
    """Tile-shape knobs of the Blackwell WS GEMM template.

    Not a generic tile description: ``num_tmem_buffers`` presumes tensor
    memory, and ``num_ctas`` presumes pair-CTA MMA. Both are Blackwell-only, so
    a Hopper or CDNA template needs its own config type rather than reusing
    this one with fields set to 1 (see :class:`AmdWarpPipeConfig`).
    """

    block_m: int
    block_n: int
    block_k: int
    num_smem_buffers: int
    num_tmem_buffers: int
    num_mma_groups: int
    num_ctas: int
    epilogue_subtile: int

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> BlackwellWSGemmConfig:
        """Build from the ``BLOCK_SIZE_M``-style dicts used by the heuristics."""
        return cls(
            block_m=config["BLOCK_SIZE_M"],
            block_n=config["BLOCK_SIZE_N"],
            block_k=config["BLOCK_SIZE_K"],
            num_smem_buffers=config["NUM_SMEM_BUFFERS"],
            num_tmem_buffers=config["NUM_TMEM_BUFFERS"],
            num_mma_groups=config["NUM_MMA_GROUPS"],
            num_ctas=config["NUM_CTAS"],
            epilogue_subtile=config["EPILOGUE_SUBTILE"],
        )

    @property
    def block_m_per_group(self) -> int:
        return self.block_m // self.num_mma_groups

    @property
    def slice_size(self) -> int:
        return self.block_n // self.epilogue_subtile


class BlackwellWSGemmModel:
    """Resource model for ``blackwell_gemm_ws.py.jinja``."""

    #: Arches this model applies to.
    arches = ("sm100", )

    @staticmethod
    def check_tile_rules(tile: BlackwellWSGemmConfig) -> bool:
        """Structural validity of a tile, independent of memory footprint.

        Rule 1:  each MMA group's tile must fit the hardware MMA (M <= 128).
        Rule 1b: pair-CTA MMA requires exactly M=128 per MMA group.
        Rule 2:  ``EPILOGUE_SUBTILE`` must evenly divide ``BLOCK_N``.
        """
        if tile.block_m_per_group > 128:
            return False
        if tile.num_ctas == 2 and tile.block_m_per_group < 128:
            return False
        if tile.block_n % tile.epilogue_subtile != 0:
            return False
        return True

    @staticmethod
    def estimate_smem(tile: BlackwellWSGemmConfig, *, charge_epilogue: bool, split_k: int = 1) -> int:
        """Bytes of shared memory a tile config needs.

        Matches upstream ``estimate_smem`` for the operand and barrier terms.

        ``charge_epilogue`` controls the epilogue staging buffer. The TMA
        epilogue store path needs it; the ``tl.store`` path does not. Callers
        that validate before the store path is known (the candidate scorer, the
        autotuning-pool prune) charge it unconditionally, which is the
        conservative choice.

        ``split_k > 1`` adds the fp32 partials workspace, which is torchTLX
        specific -- upstream stages split-K partials in the output dtype.
        """
        smem_a = (tile.block_m * tile.block_k * OPERAND_ELEM_BYTES * tile.num_smem_buffers)
        smem_b = (tile.block_k * (tile.block_n // tile.num_ctas) * OPERAND_ELEM_BYTES * tile.num_smem_buffers)
        smem_epilog = (tile.block_m * tile.slice_size * OPERAND_ELEM_BYTES if charge_epilogue else 0)
        # In 2-CTA mode each CTA allocates its own copy of the barriers.
        smem_barriers = (tile.num_smem_buffers * tile.num_mma_groups * MBARRIER_BYTES *
                         (2 if tile.num_ctas == 2 else 1))
        total = smem_a + smem_b + smem_epilog + smem_barriers

        if split_k > 1:
            num_epilogue_smem_buffers = max(tile.num_mma_groups, 2)
            total += (tile.block_m_per_group * tile.slice_size * SPLIT_K_PARTIAL_BYTES * num_epilogue_smem_buffers)
        return total

    @staticmethod
    def estimate_tmem_columns(tile: BlackwellWSGemmConfig) -> int:
        """TMEM columns needed (128-lane granular, so BLOCK_M drops out)."""
        return tile.block_n * tile.num_tmem_buffers * tile.num_mma_groups

    @classmethod
    def validate(
        cls,
        tile: BlackwellWSGemmConfig,
        *,
        charge_epilogue: bool,
        split_k: int = 1,
        smem_margin: int = 0,
        check_rules: bool = True,
        check_tmem: bool = True,
        limits: DeviceLimits | None = None,
    ) -> bool:
        """Whether a tile config is valid on the target GPU.

        ``smem_margin`` is added to the estimate before comparing against the
        limit, to cover epilogue-fusion overhead (alignment padding, extra
        barriers) that the static formula does not capture.
        """
        if limits is None:
            limits = BLACKWELL_LIMITS
        if check_rules and not cls.check_tile_rules(tile):
            return False
        smem = cls.estimate_smem(tile, charge_epilogue=charge_epilogue, split_k=split_k)
        if smem + smem_margin > limits.on_chip_bytes:
            return False
        if check_tmem:
            needed = cls.estimate_tmem_columns(tile)
            # limits.tmem_columns is None on a target with no tensor memory, on
            # which this template cannot run at all.
            if limits.tmem_columns is None or needed > limits.tmem_columns:
                return False
        return True


# ---------------------------------------------------------------------------
# AMD warp-pipelined GEMM (addmm / bmm / persistent addmm)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class AmdWarpPipeConfig:
    """Tile-shape knobs of the AMD warp-pipe templates."""

    block_m: int
    block_n: int
    block_k: int
    num_buffers: int
    #: False on the register path, which allocates no LDS at all.
    use_async: bool = True
    elem_bytes: int = OPERAND_ELEM_BYTES


class AmdWarpPipeModel:
    """Resource model for the ``gfx950_*_warppipe`` templates.

    The kernels allocate exactly two LDS buffers, multi-buffered by
    ``NUM_BUFFERS``::

        smemA = tlx.local_alloc((BLOCK_M, BLOCK_K), dtype_of(A), NUM_BUFFERS)
        smemB = tlx.local_alloc((BLOCK_N, BLOCK_K), dtype_of(B), NUM_BUFFERS)

    (the bmm template spells B as ``(BLOCK_K, BLOCK_N)`` -- same element count),
    so the footprint is ``(BLOCK_M + BLOCK_N) * BLOCK_K * elem * NUM_BUFFERS``.
    There are no mbarriers to charge: the pipeline is ordered by
    ``async_load_commit_group`` / ``async_load_wait_group``, which use no LDS.
    There is no epilogue staging buffer and no tensor memory.

    Derived from the templates and cross-checked against the hand-verified
    budget recorded on ``WARPPIPE_CONFIGS`` in registry.py::

        (128, 256, 64) NB2 -> 96KB     (128, 256, 32) NB3 -> 72KB
        (128, 256, 64) NB3 -> 144KB

    all three of which this formula reproduces exactly. It has not been
    validated against a live gfx950 run; treat a disagreement with the kernel's
    actual ``ttg.shared`` as the formula being wrong, not the kernel.
    """

    arches = ("gfx942", "gfx950", "gfx1250")

    @staticmethod
    def check_tile_rules(cfg: AmdWarpPipeConfig) -> bool:
        """The warp-pipe has no structural tile constraints of its own.

        Correctness needs ``K_ITERS > NUM_BUFFERS``, but that depends on K, not
        on the tile, so the heuristic enforces it where K is known.
        """
        return True

    @staticmethod
    def estimate_smem(cfg: AmdWarpPipeConfig) -> int:
        """Bytes of LDS the warp-pipe needs. Zero on the register path."""
        if not cfg.use_async:
            return 0
        return ((cfg.block_m + cfg.block_n) * cfg.block_k * cfg.elem_bytes * cfg.num_buffers)

    @staticmethod
    def estimate_tmem_columns(cfg: AmdWarpPipeConfig) -> int:
        """The warp-pipe allocates no tensor memory.

        This is a statement about the template's usage, not the hardware's
        capacity -- the arch classes raise for that.
        """
        return 0

    @classmethod
    def validate(
        cls,
        cfg: AmdWarpPipeConfig,
        *,
        smem_margin: int = 0,
        limits: DeviceLimits | None = None,
    ) -> bool:
        if limits is None:
            limits = DeviceLimits.current()
        return cls.estimate_smem(cfg) + smem_margin <= limits.on_chip_bytes


BLACKWELL_WS_GEMM = BlackwellWSGemmModel()
AMD_WARP_PIPE = AmdWarpPipeModel()

#: Template uid -> resource model. Templates absent from this map have no model
#: yet; add one alongside that template's tuning journey.
#:
#: Keyed by uid, not by the Python name. The uids are the wire format -- they
#: end up in generated kernel names -- and follow the same ``tlx_<arch>_<op>``
#: pattern as the Python objects and the .jinja files.
MODELS: dict[str, Any] = {
    "triton::tlx_blackwell_gemm_ws": BLACKWELL_WS_GEMM,
    "triton::tlx_gfx950_addmm_warppipe": AMD_WARP_PIPE,
    "triton::tlx_gfx950_addmm_persistent_warppipe": AMD_WARP_PIPE,
    "triton::tlx_gfx950_bmm_warppipe": AMD_WARP_PIPE,
}

# Module-level aliases for the Blackwell model, which is what the GEMM
# heuristics in registry.py call.
check_tile_rules = BLACKWELL_WS_GEMM.check_tile_rules
estimate_smem = BLACKWELL_WS_GEMM.estimate_smem
estimate_tmem_columns = BLACKWELL_WS_GEMM.estimate_tmem_columns
validate_config = BLACKWELL_WS_GEMM.validate
