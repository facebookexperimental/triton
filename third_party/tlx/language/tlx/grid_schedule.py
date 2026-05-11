"""Host-side grid scheduling utilities for TLX kernels.

Provides the GridSchedule enum and compute_grid function for abstracting
launch grid computation across non-persistent, static persistent, and
CLC (Cooperative Launch Control) scheduling strategies.
"""

import enum
import functools
from typing import Callable, Tuple

import torch


class GridSchedule(enum.Enum):
    """Tile scheduling strategy for TLX kernels.

    NON_PERSISTENT: One CTA per tile. Grid = total_tiles.
        Simplest model. CUDA runtime handles wave scheduling.

    STATIC_PERSISTENT: One CTA per SM, tiles assigned round-robin.
        Grid = min(NUM_SMS, total_tiles). Eliminates wave
        quantization and enables cross-tile pipeline overlap.

    CLC: Cooperative Launch Control (Blackwell+ only).
        Grid = total_tiles. Hardware dynamically assigns tiles
        to CTAs via clusterlaunchcontrol.try_cancel, enabling
        work stealing and dynamic load balancing.
    """

    NON_PERSISTENT = "non_persistent"
    STATIC_PERSISTENT = "static_persistent"
    CLC = "clc"


@functools.lru_cache(maxsize=1)
def _get_max_num_sms() -> int:
    """Cached SM count for the active CUDA device."""
    return torch.cuda.get_device_properties("cuda").multi_processor_count


def compute_grid(
    schedule: GridSchedule,
    total_tiles_fn: Callable[[], int],
    num_sms_fn: Callable[[], int] = _get_max_num_sms,
) -> Tuple[int, ...]:
    """Compute the launch grid for the given schedule type.

    Args:
        schedule: The scheduling strategy.
        total_tiles_fn: A callable that returns the total number of output tiles.
            The caller is responsible for incorporating any kernel-specific
            concerns (cluster padding, Split-K expansion, etc.) into this function.
        num_sms_fn: A callable that returns the number of SMs to use.
            Defaults to the device's full SM count. Override to explicitly
            limit SM usage (e.g., for occupancy tuning or resource partitioning).
            Only called for STATIC_PERSISTENT.

    Returns:
        A 1D grid tuple suitable for passing to kernel[grid](...).
    """
    if schedule == GridSchedule.NON_PERSISTENT:
        return (total_tiles_fn(),)

    if schedule == GridSchedule.STATIC_PERSISTENT:
        return (min(num_sms_fn(), total_tiles_fn()),)

    if schedule == GridSchedule.CLC:
        return (total_tiles_fn(),)

    raise ValueError(f"Unknown schedule: {schedule}")
