"""One shared-memory / tensor-memory model for the TLX Blackwell GEMM template.

This module owns the answer to "does this tile config fit on the GPU?".  It
previously existed as four near-copies inside ``registry.py`` that had drifted
apart -- two different TMEM formulas, and an epilogue-staging term that some
copies charged unconditionally and others gated on TMA epilogue stores.

The only legitimate difference between call sites is *whether the epilogue
staging buffer is charged*, so that is an explicit keyword argument rather than
a difference between four function bodies.

TMEM is modelled in **columns**, not bytes.  Tensor memory is 128 lanes x 512
columns of 32 bits, and an accumulator occupies whole lanes regardless of
``BLOCK_M``, so ``block_n * num_tmem_buffers * num_mma_groups <= 512`` is the
real constraint.  The byte model that used to live in the candidate scorer
(``block_m * block_n * 4 * num_tmem_buffers <= 256KB``) both under-charged
``BLOCK_M < 128`` tiles and omitted ``num_mma_groups`` entirely.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .target import B200_SMEM_BYTES, B200_TMEM_COLUMNS, current_target

#: Bytes per mbarrier.
MBARRIER_BYTES = 8
#: Bytes per element of the A/B operand buffers (fp16/bf16).
OPERAND_ELEM_BYTES = 2
#: torchTLX stores split-K partials as fp32 (upstream uses the output dtype).
SPLIT_K_PARTIAL_BYTES = 4


@dataclasses.dataclass(frozen=True)
class DeviceLimits:
    """Per-SM resource limits for the target GPU."""

    smem_bytes: int
    tmem_columns: int

    @classmethod
    def current(cls) -> DeviceLimits:
        """Limits resolved from the live device.

        Prefer :data:`DEFAULT_LIMITS` on the heuristic path: SMEM and TMEM are
        the same constants for every target torchTLX currently supports, and
        going through :func:`~.target.current_target` would initialize a CUDA
        context inside what is otherwise pure, CPU-testable code. Use this once
        the limits genuinely vary by architecture.
        """
        target = current_target()
        return cls(smem_bytes=target.smem_bytes, tmem_columns=target.tmem_columns)


#: Limits used by default. Import-time constants, no device query.
DEFAULT_LIMITS = DeviceLimits(
    smem_bytes=B200_SMEM_BYTES, tmem_columns=B200_TMEM_COLUMNS
)


@dataclasses.dataclass(frozen=True)
class TileConfig:
    """The tile-shape knobs that determine SMEM/TMEM footprint."""

    block_m: int
    block_n: int
    block_k: int
    num_smem_buffers: int
    num_tmem_buffers: int
    num_mma_groups: int
    num_ctas: int
    epilogue_subtile: int

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> TileConfig:
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


def check_tile_rules(tile: TileConfig) -> bool:
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


def estimate_smem(
    tile: TileConfig, *, charge_epilogue: bool, split_k: int = 1
) -> int:
    """Bytes of shared memory a tile config needs.

    Matches upstream ``estimate_smem`` for the operand and barrier terms.

    ``charge_epilogue`` controls the epilogue staging buffer.  The TMA epilogue
    store path needs it; the ``tl.store`` path does not.  Callers that validate
    a config before the store path is known (the candidate scorer, the
    autotuning-pool prune) charge it unconditionally, which is the conservative
    choice.

    ``split_k > 1`` adds the fp32 partials workspace, which is torchTLX
    specific -- upstream stages split-K partials in the output dtype.
    """
    smem_a = tile.block_m * tile.block_k * OPERAND_ELEM_BYTES * tile.num_smem_buffers
    smem_b = (
        tile.block_k
        * (tile.block_n // tile.num_ctas)
        * OPERAND_ELEM_BYTES
        * tile.num_smem_buffers
    )
    smem_epilog = (
        tile.block_m * tile.slice_size * OPERAND_ELEM_BYTES if charge_epilogue else 0
    )
    # In 2-CTA mode each CTA allocates its own copy of the barriers.
    smem_barriers = (
        tile.num_smem_buffers
        * tile.num_mma_groups
        * MBARRIER_BYTES
        * (2 if tile.num_ctas == 2 else 1)
    )
    total = smem_a + smem_b + smem_epilog + smem_barriers

    if split_k > 1:
        num_epilogue_smem_buffers = max(tile.num_mma_groups, 2)
        total += (
            tile.block_m_per_group
            * tile.slice_size
            * SPLIT_K_PARTIAL_BYTES
            * num_epilogue_smem_buffers
        )
    return total


def estimate_tmem_columns(tile: TileConfig) -> int:
    """TMEM columns a tile config needs (128-lane granular, so M drops out)."""
    return tile.block_n * tile.num_tmem_buffers * tile.num_mma_groups


def validate_config(
    tile: TileConfig,
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
        limits = DEFAULT_LIMITS
    if check_rules and not check_tile_rules(tile):
        return False
    smem = estimate_smem(tile, charge_epilogue=charge_epilogue, split_k=split_k)
    if smem + smem_margin > limits.smem_bytes:
        return False
    if check_tmem and estimate_tmem_columns(tile) > limits.tmem_columns:
        return False
    return True
