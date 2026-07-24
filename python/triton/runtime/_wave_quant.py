from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Optional

from .. import knobs
from .driver import driver


def is_pow2(value):
    return value > 0 and (value & (value - 1)) == 0


def cdiv(lhs, rhs):
    return -(-lhs // rhs)


def clone_config(config, **kwargs_override):
    """Clone a Config while carrying fields added by future Triton versions."""
    new_config = copy.copy(config)
    new_config.kwargs = {**config.kwargs, **kwargs_override}
    return new_config


def config_has_npot_block(config):
    """True if any of the config's BLOCK_* tile sizes is non-power-of-2."""
    return any(
        isinstance(value, int) and not is_pow2(value)
        for name, value in config.kwargs.items()
        if "BLOCK" in name.upper()
    )


# Some backends report an unsupported NPOT candidate as a device RuntimeError or a bare
# RuntimeError from an MLIR pass. Only errors with a known device or NPOT-legality marker are
# prunable; unrelated compiler failures still propagate.
_DEVICE_ERROR_MARKERS = (
    "triton error [cuda]",
    "triton error [hip]",
    "out of memory",
    "misaligned",
    "illegal memory access",
    "illegal instruction",
    "device-side assert",
)
_NPOT_COMPILE_REJECT_MARKERS = (
    "number of elements must be power-of-two",
    "number of elements must be a power of two",
    "npot layout not yet supported",
    "threads per warp, but the module specifies",
    "map::at",
)


def npot_runtime_error_prunable(config, error, allow_npot):
    if not (allow_npot and config_has_npot_block(config)):
        return False
    text = str(error).lower()
    return any(marker in text for marker in _DEVICE_ERROR_MARKERS + _NPOT_COMPILE_REJECT_MARKERS)


def wave_efficiency(num_tiles, num_units):
    """Useful work fraction: tiles / (waves * units)."""
    if num_tiles <= 0 or num_units <= 0:
        return 1.0
    waves = cdiv(num_tiles, num_units)
    return num_tiles / (waves * num_units)


def legal_npot_blocks(base, legal_multiple):
    """Legal NPOT multiples in [base / 2, base * 2], excluding base and powers of two."""
    lo = max(legal_multiple, base // 2)
    hi = base * 2
    value = lo + ((legal_multiple - lo % legal_multiple) % legal_multiple)
    out = []
    while value <= hi:
        if value != base and not is_pow2(value):
            out.append(value)
        value += legal_multiple
    return out


# Only intervene when the base tiling spans more than one wave and wastes at least 10% of the
# available wave slots. Existing seed configs do not count against the generated-candidate cap.
WAVE_EFFICIENCY_THRESHOLD = 0.9
MAX_NPOT_CANDIDATES = 4
MAX_NPOT_VALUES_PER_DIM = 4
MAX_PIPELINE_STAGES = 8
GEMM_SMEM_OVERHEAD = 1024
GEMM_BASE_REGS_PER_THREAD = 16


@dataclass(frozen=True)
class WaveDeviceLimits:
    num_sms: int
    # These two fields are whole-SM/CU budgets used to estimate residency.
    max_shared_mem: int
    max_num_regs: int
    warp_size: int
    max_threads_per_sm: int
    # Per-block limits reject candidates that cannot launch even when the SM/CU has room overall.
    max_shared_mem_per_block: Optional[int] = None
    max_num_regs_per_block: Optional[int] = None
    max_ctas_per_sm: Optional[int] = None


@dataclass(frozen=True)
class WaveFrontierPoint:
    config: Any
    num_tiles: int
    ctas_per_sm: int
    waves: int
    wave_efficiency: float
    shared_mem: int
    registers: int
    estimated_cost: float


def wave_quant_npot_candidates(base_m, base_n, problem_m, problem_n, num_units, legal_m=16, legal_n=16):
    """Return legal NPOT spatial tiles that improve wave count, then wave efficiency."""
    if not (base_m and base_n and problem_m and problem_n and num_units):
        return []
    base_tiles = cdiv(problem_m, base_m) * cdiv(problem_n, base_n)
    base_waves = cdiv(base_tiles, num_units)
    base_efficiency = base_tiles / (base_waves * num_units)
    if base_waves <= 1 or base_efficiency >= WAVE_EFFICIENCY_THRESHOLD:
        return []

    scored = []
    for block_m in [base_m] + legal_npot_blocks(base_m, legal_m):
        for block_n in [base_n] + legal_npot_blocks(base_n, legal_n):
            if block_m == base_m and block_n == base_n:
                continue
            tiles = cdiv(problem_m, block_m) * cdiv(problem_n, block_n)
            waves = cdiv(tiles, num_units)
            efficiency = tiles / (waves * num_units)
            if waves < base_waves or (waves == base_waves and efficiency > base_efficiency):
                scored.append(((waves, -efficiency), (block_m, block_n)))

    scored.sort(key=lambda item: item[0])
    out = []
    for _, pair in scored:
        if pair not in out:
            out.append(pair)
        if len(out) >= MAX_NPOT_CANDIDATES:
            break
    return out


def frontier_block_values(base, legal_multiple, allow_npot):
    pot_values = [
        value
        for value in (base // 2, base * 2)
        if value >= legal_multiple and value % legal_multiple == 0 and is_pow2(value)
    ]
    npot_values = legal_npot_blocks(base, legal_multiple) if allow_npot else []
    npot_values.sort(key=lambda value: (abs(value - base), value))
    return [base] + pot_values + npot_values[:MAX_NPOT_VALUES_PER_DIM]


def estimate_wave_frontier_point(config, problem_dims, limits, element_bytes=2):
    """Estimate GEMM resource use, residency, and wave cost for one config."""
    block_m = config.kwargs.get("BLOCK_M")
    block_n = config.kwargs.get("BLOCK_N")
    block_k = config.kwargs.get("BLOCK_K")
    problem_m = problem_dims.get("M")
    problem_n = problem_dims.get("N")
    problem_k = problem_dims.get("K")
    if not all(isinstance(value, int) and value > 0
               for value in (block_m, block_n, block_k, problem_m, problem_n, problem_k)):
        return None

    num_warps = config.num_warps
    num_stages = config.num_stages
    num_ctas = config.num_ctas
    if not all(isinstance(value, int) and value > 0 for value in (num_warps, num_stages, num_ctas)):
        return None

    threads = num_warps * limits.warp_size
    if threads > limits.max_threads_per_sm:
        return None

    # A/B pipeline buffers do not coexist with the output staging buffer. The accumulator plus a
    # small per-thread baseline dominate registers. Compilation remains the final resource arbiter.
    operand_smem = (block_m * block_k + block_n * block_k) * element_bytes * num_stages
    output_smem = block_m * block_n * element_bytes
    shared_mem = max(operand_smem, output_smem) + GEMM_SMEM_OVERHEAD
    registers = block_m * block_n + GEMM_BASE_REGS_PER_THREAD * threads
    regs_per_thread = cdiv(registers, threads)
    block_shared_limit = limits.max_shared_mem_per_block or limits.max_shared_mem
    block_register_limit = limits.max_num_regs_per_block or limits.max_num_regs
    if shared_mem > block_shared_limit or registers > block_register_limit:
        return None
    if limits.warp_size == 32 and regs_per_thread > 256:
        return None

    residency_limits = [
        limits.max_shared_mem // shared_mem,
        limits.max_num_regs // registers,
        limits.max_threads_per_sm // threads,
    ]
    if limits.max_ctas_per_sm is not None:
        residency_limits.append(limits.max_ctas_per_sm)
    ctas_per_sm = min(residency_limits)
    if ctas_per_sm <= 0:
        return None

    num_tiles = cdiv(problem_m, block_m) * cdiv(problem_n, block_n)
    wave_slots = max(1, limits.num_sms // num_ctas)
    waves = cdiv(num_tiles, wave_slots)
    efficiency = wave_efficiency(num_tiles, wave_slots)

    padded_k = cdiv(problem_k, block_k) * block_k
    padded_work = num_tiles * block_m * block_n * padded_k
    pipeline_gain = 1.0 + 0.06 * min(num_stages - 1, 4)
    residency_gain = 1.0 + 0.04 * min(ctas_per_sm - 1, 3)
    loop_penalty = 1.0 + 0.015 * cdiv(problem_k, block_k)
    tile_scale = min(block_m, 128) / 128 * min(block_n, 128) / 128
    tile_efficiency = 0.75 + 0.25 * tile_scale
    estimated_cost = padded_work * loop_penalty / (
        efficiency * pipeline_gain * residency_gain * tile_efficiency
    )
    return WaveFrontierPoint(
        config,
        num_tiles,
        ctas_per_sm,
        waves,
        efficiency,
        shared_mem,
        registers,
        estimated_cost,
    )


def point_dominates(lhs, rhs):
    lhs_metrics = (lhs.estimated_cost, lhs.shared_mem, lhs.registers, lhs.waves)
    rhs_metrics = (rhs.estimated_cost, rhs.shared_mem, rhs.registers, rhs.waves)
    return all(a <= b for a, b in zip(lhs_metrics, rhs_metrics)) and any(
        a < b for a, b in zip(lhs_metrics, rhs_metrics)
    )


def coupled_wave_frontier(
    config,
    problem_dims,
    limits,
    element_bytes=2,
    legal_m=16,
    legal_n=16,
    legal_k=16,
    allow_npot=False,
):
    """Return a bounded POT/NPOT Pareto set over tiles and pipeline depth."""
    if config_has_npot_block(config):
        return []
    base = estimate_wave_frontier_point(config, problem_dims, limits, element_bytes)
    if base is None or base.wave_efficiency >= WAVE_EFFICIENCY_THRESHOLD:
        return []
    if base.waves <= 1 and base.num_tiles < max(2, limits.num_sms // 2):
        return []

    block_m = config.kwargs["BLOCK_M"]
    block_n = config.kwargs["BLOCK_N"]
    block_k = config.kwargs["BLOCK_K"]
    stages = sorted({
        max(1, config.num_stages - 1),
        config.num_stages,
        min(MAX_PIPELINE_STAGES, config.num_stages + 1),
    })
    points = []
    for next_m in frontier_block_values(block_m, legal_m, allow_npot):
        for next_n in frontier_block_values(block_n, legal_n, allow_npot):
            for next_k in frontier_block_values(block_k, legal_k, allow_npot):
                for num_stages in stages:
                    if (next_m, next_n, next_k, num_stages) == (
                        block_m,
                        block_n,
                        block_k,
                        config.num_stages,
                    ):
                        continue
                    candidate = clone_config(config, BLOCK_M=next_m, BLOCK_N=next_n, BLOCK_K=next_k)
                    candidate.num_stages = num_stages
                    point = estimate_wave_frontier_point(candidate, problem_dims, limits, element_bytes)
                    if point is not None and point.estimated_cost < base.estimated_cost * 0.995:
                        points.append(point)

    frontier = [
        point
        for point in points
        if not any(point_dominates(other, point) for other in points if other is not point)
    ]
    frontier.sort(
        key=lambda point: (
            point.estimated_cost,
            point.waves,
            -point.wave_efficiency,
            -point.config.num_stages,
            point.shared_mem,
            point.registers,
        )
    )
    distinct_tiles = []
    seen_tiles = set()
    for point in frontier:
        tile = tuple(point.config.kwargs[name] for name in ("BLOCK_M", "BLOCK_N", "BLOCK_K"))
        if tile not in seen_tiles:
            distinct_tiles.append(point)
            seen_tiles.add(tile)
    frontier = distinct_tiles

    required = []

    def add_first(predicate):
        point = next((candidate for candidate in frontier if predicate(candidate)), None)
        if point is not None and point not in required:
            required.append(point)

    add_first(
        lambda point: not config_has_npot_block(point.config)
        and (point.config.kwargs["BLOCK_M"], point.config.kwargs["BLOCK_N"]) != (block_m, block_n)
    )
    add_first(lambda point: config_has_npot_block(point.config))
    add_first(lambda point: point.config.num_stages > config.num_stages)

    selected = []
    for point in required + frontier:
        if point not in selected:
            selected.append(point)
        if len(selected) >= MAX_NPOT_CANDIDATES:
            break
    selected.sort(key=lambda point: (point.estimated_cost, point.waves, -point.wave_efficiency))
    return [point.config for point in selected]


def fixed_occupancy_limits(num_units):
    """Compatibility model for phase-one callers that provide only a unit count."""
    unbounded = 1 << 60
    return WaveDeviceLimits(num_units, unbounded, unbounded, 32, unbounded, max_ctas_per_sm=1)


def generate_wave_quant_candidates(configs, problem_dims, device, element_bytes=2):
    """Append at most four resource-constrained POT/NPOT configs across all seeds."""
    if not device:
        return configs
    problem_m = problem_dims.get("M")
    problem_n = problem_dims.get("N")
    if not (isinstance(problem_m, int) and isinstance(problem_n, int)):
        return configs
    limits = fixed_occupancy_limits(device) if isinstance(device, int) else device
    expanded = list(configs)
    generated = []
    for config in configs:
        block_m = config.kwargs.get("BLOCK_M")
        block_n = config.kwargs.get("BLOCK_N")
        if not (isinstance(block_m, int) and isinstance(block_n, int)):
            continue
        if isinstance(config.kwargs.get("BLOCK_K"), int) and isinstance(problem_dims.get("K"), int):
            candidates = coupled_wave_frontier(
                config,
                problem_dims,
                limits,
                element_bytes,
                allow_npot=knobs.language.allow_npot,
            )
        else:
            pairs = (
                wave_quant_npot_candidates(block_m, block_n, problem_m, problem_n, limits.num_sms)
                if knobs.language.allow_npot
                else []
            )
            candidates = [clone_config(config, BLOCK_M=next_m, BLOCK_N=next_n) for next_m, next_n in pairs]
        for candidate in candidates:
            if candidate in expanded or any(candidate == existing for existing, _ in generated):
                continue
            point = estimate_wave_frontier_point(candidate, problem_dims, limits, element_bytes)
            if point is not None:
                rank = (
                    point.estimated_cost,
                    point.waves,
                    -point.wave_efficiency,
                    point.shared_mem,
                    point.registers,
                )
            else:
                candidate_m = candidate.kwargs["BLOCK_M"]
                candidate_n = candidate.kwargs["BLOCK_N"]
                tiles = cdiv(problem_m, candidate_m) * cdiv(problem_n, candidate_n)
                waves = cdiv(tiles, limits.num_sms)
                efficiency = wave_efficiency(tiles, limits.num_sms)
                rank = (tiles * candidate_m * candidate_n / efficiency, waves, -efficiency, 0, 0)
            generated.append((candidate, rank))

    generated.sort(key=lambda item: item[1])
    ordered = [candidate for candidate, _ in generated]
    selected = []

    def add_first(predicate):
        candidate = next((config for config in ordered if predicate(config)), None)
        if candidate is not None and candidate not in selected:
            selected.append(candidate)

    add_first(lambda config: not config_has_npot_block(config))
    if knobs.language.allow_npot:
        add_first(config_has_npot_block)
    for candidate in ordered:
        if config_has_npot_block(candidate) and any(config_has_npot_block(config) for config in selected):
            continue
        if candidate not in selected:
            selected.append(candidate)
        if len(selected) >= MAX_NPOT_CANDIDATES:
            break
    expanded.extend(selected[:MAX_NPOT_CANDIDATES])
    return expanded


def device_wave_limits():
    """Read exact per-block and per-SM/CU resource limits; return None if unavailable."""
    try:
        device = driver.active.get_current_device()
        props = driver.active.utils.get_device_properties(device)
        target = driver.active.get_current_target()
        limits = WaveDeviceLimits(
            num_sms=int(props["multiprocessor_count"]),
            max_shared_mem=int(props["max_shared_mem_per_sm"]),
            max_num_regs=int(props["max_num_regs_per_sm"]),
            warp_size=int(props.get("warpSize") or target.warp_size),
            max_threads_per_sm=int(props["max_threads_per_sm"]),
            max_shared_mem_per_block=int(props["max_shared_mem"]),
            max_num_regs_per_block=int(props["max_num_regs"]),
            max_ctas_per_sm=int(props["max_blocks_per_sm"]),
        )
        values = (
            limits.num_sms,
            limits.max_shared_mem,
            limits.max_num_regs,
            limits.warp_size,
            limits.max_threads_per_sm,
            limits.max_shared_mem_per_block,
            limits.max_num_regs_per_block,
            limits.max_ctas_per_sm,
        )
        return limits if all(value is not None and value > 0 for value in values) else None
    except Exception:
        return None


def problem_element_bytes(named_args):
    """Best-effort input element width; two bytes is the conservative GEMM default."""
    for value in named_args.values():
        element_size = getattr(value, "element_size", None)
        if callable(element_size):
            try:
                size = int(element_size())
                if 0 < size <= 16:
                    return size
            except (TypeError, ValueError):
                pass
    return 2
