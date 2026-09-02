"""What each architecture and cuBLASLt version was measured to do, and how one is chosen.

`ArchProfile` is the whole vocabulary: which `ALGO_ID` belongs to which kernel family, what a
`STAGES_ID` means, which merge scheme a `REDUCTION_SCHEME` is, and the CUDA-core recipe tables.
Every field is measured, and every field says how.  The planner reads a profile and nothing
else, which is why it needs no architecture of its own.

ONE FILE PER (architecture, cuBLASLt major.minor), and the files do not import each other.
sm_103 and sm_90 happen to share most of sm_100's tables, and 13.1 and 12.8 happen to agree
completely -- but those are findings, not rules, so each file states its own tables in full and
records in a header comment how it was read and where it differs.  Deriving one from another
would turn an observation into a structural assumption and make a future divergence awkward to
express.

Adding an architecture, or a cuBLASLt version that turns out to behave differently, is one new
file plus one registry line.  No dispatch code changes.
"""
from __future__ import annotations

import dataclasses
import warnings

import torch

from ..errors import CublasUnsupportedPlatform
from ..ltapi import cublaslt_version

# --------------------------------------------------------------------------- #
# Per-platform strategy
#
# PORTING GUIDE. Everything cuBLAS-specific in this file lives in an ArchProfile below, so
# adding a GPU is filling in one dataclass -- no dispatch code changes. Each field says how to
# measure it, and measuring is the point: sm_103 turned out to share every sm_100 table but one
# gemv row, but that was established by re-reading each kernel family on a GB300, not by
# assuming it. An earlier attempt to carry rules the other way, from sm_103 to sm_100, was
# wrong -- though those rules were an earlier and buggier design, so that failure is weak
# evidence about the architectures and strong evidence about guessing.
#
# The gate is (compute capability, cuBLASLt version). The cuBLASLt library we call through
# ctypes is what decides the kernels -- its name literally carries the arch, e.g.
# `nvjet_sm100_...` -- so it, not the CUDA toolkit torch was built against, is what we pin.
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class ArchProfile:
    """The measured cuBLAS behaviour of one GPU architecture.

    `measured=False` means the entry is a placeholder: the dispatch knows the architecture
    exists but has no rules for it, and the API raises `CublasUnsupportedPlatform`."""

    name: str  # human-readable, e.g. "sm_100 (NVIDIA GB200 / B200)"
    measured: bool = False
    measured_cublaslt: str = ""  # exact version, for the record

    # -- what the heuristic config means on this architecture -----------------------------
    # Every table below is MEASURED. An unknown key declines rather than guessing, so a new
    # cuBLAS or a new GPU loses coverage instead of returning wrong bits.
    #
    # CUBLASLT_ALGO_CONFIG_ID -> which kernel family cuBLAS will launch. Measure: profile the
    # launched kernel name over a shape sweep and cross-tabulate against attr 0.
    algo_family: tuple = ()  # sm_100: see `_SM100` below
    # (family, CUBLASLT_ALGO_CONFIG_STAGES_ID) -> (threadblock block_k, k per dot). This is the
    # pair the reconstruction turns on, and the stages id is the only field that pins it.
    # Measure: same sweep, read `<tbK>x<stages>` and the `s<MMA>gemm` token from the name.
    stages_recipe: tuple = ()
    # (family, CUBLASLT_ALGO_CONFIG_STAGES_ID) keys whose kernel has a SECOND accumulation level
    # at its own block_k: every block_k-long block is summed on its own and the block totals are
    # added forward, rather than one flat accumulator running the whole slice. Empty means every
    # key on this architecture is flat, which is what sm_100 and sm_103 measured. Measure by
    # byte-comparing the two forms as K grows: they agree exactly while K fits ONE block_k step
    # and part company at the first K that needs two, including exact multiples of block_k.
    block_level_keys: tuple = ()
    # Partition grains to try for CUTLASS split-K, largest first; cross-check against `kAlignK`
    # in `cutlass/.../params_universal_base.h`.
    splitk_grains: tuple = ()  # sm_100: (8, 64)
    # CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME -> our merge scheme. Measure: cross-tabulate attr 3
    # against the merge scheme that byte-matches.
    reduction_to_cmode: tuple = ()

    # -- the CUDA-core (SIMT) families ----------------------------------------------------
    # These carry no block_k and no reduction scheme; CUSTOM_OPTION (attr 5) is the whole second
    # degree of freedom. Measure: probe one k at a time with a value large enough to swamp its
    # neighbours (+L at one k, -L at another) and read the grouping straight off the output.
    #
    # (ALGO_ID, CUSTOM_OPTION) -> (S, B) for `_simt_chain_gemm`: S contiguous k chunks, B k per
    # inner accumulator (0 = one accumulator for the whole chunk).
    gemmsn_recipe: tuple = ()
    # (ALGO_ID, CUSTOM_OPTION) -> (V, W, CC, count-down lane tree) for `_gemv_lane`. ALGO_ID 14
    # needs no table: its parameters come from SPLITK_NUM and the operand orientation.
    gemv_recipe: tuple = ()
    # (ALGO_ID, CUSTOM_OPTION) -> (W, C) for `_gemv_cslice`, the gemv whose lane takes a
    # contiguous k slice instead of a strided tile: W lanes per output element and C k per
    # chunk inside the lane. Measure the same way as `gemv_recipe`, with the k probe.
    gemv_cslice_recipe: tuple = ()
    # (ALGO_ID, CUSTOM_OPTION) -> the longest gemv, in output elements, the row above holds for.
    # A longer one declines: cuBLAS keeps the same CUSTOM_OPTION but stops running the same
    # order. Measure by sweeping the output length at a fixed option. The cap "occupancy" means
    # the row holds exactly while the whole launch fits the GPU at once, i.e. up to
    # `sm_count * threads_per_sm // W` output elements.
    gemv_max_elems: tuple = ()
    # How many threads the GPU holds at once, for the "occupancy" cap above. Measure `sm_count`
    # with `torch.cuda.get_device_properties(0).multi_processor_count` (152 on GB200), and take
    # `threads_per_sm` from the architecture's resident-thread limit -- the "Maximum number of
    # resident threads per SM" row of the CUDA C programming guide's compute-capability table,
    # 2048 on sm_100. Cross-check by sweeping the output length: the cap is where the measured
    # order changes (9728 here, and 9728 * 32 == 152 * 2048).
    sm_count: int = 0
    threads_per_sm: int = 0

    # -- Triton-side constraint -----------------------------------------------------------
    # Minimum BM below which Triton stops using the native fp8 tensor-core path. Measure: dump
    # PTX over a (BM, BN, BK, num_warps) grid and look for the MMA instruction changing.
    fp8_min_bm: int = 64


from . import sm90_lt13_1, sm100_lt12_8, sm100_lt13_1, sm103_lt13_1  # noqa: E402

# (compute capability, cuBLASLt (major, minor)) -> the rules measured for exactly that pair.
# A version this table does not name still runs, on the newest profile for the same
# architecture, after warning once -- see `platform`.
_REGISTRY = {
    ((10, 0), (13, 1)): sm100_lt13_1.PROFILE,
    ((10, 0), (12, 8)): sm100_lt12_8.PROFILE,
    ((10, 3), (13, 1)): sm103_lt13_1.PROFILE,
    ((9, 0), (13, 1)): sm90_lt13_1.PROFILE,
}

_PLATFORM_CACHE: dict = {}


def platform():
    """The ArchProfile for the current device and cuBLASLt, or raise CublasUnsupportedPlatform.

    An architecture with no measured profile at all is refused: the rules are architecture-
    specific and must be re-measured, not extrapolated.  An unmeasured *version* of a known
    architecture only warns and runs, because a version bump usually moves nothing, and one
    that renumbered `ALGO_ID` or `STAGES_ID` would fall out of the tables and decline rather
    than answer wrong.  Warned once per (architecture, version), so a fuzzer loop stays quiet.
    """
    cap = torch.cuda.get_device_capability()
    major, minor, patch = cublaslt_version()
    key = (cap, (major, minor))
    if key in _PLATFORM_CACHE:
        return _PLATFORM_CACHE[key]

    prof = _REGISTRY.get(key)
    if prof is None:
        measured = sorted((v, p) for (c, v), p in _REGISTRY.items() if c == cap and p.measured)
        if not measured:
            known = ", ".join(sorted({p.name for p in _REGISTRY.values() if p.measured}))
            raise CublasUnsupportedPlatform(
                f"unsupported on sm_{cap[0]}{cap[1]} ({torch.cuda.get_device_name()}): no "
                f"cuBLAS-equivalence strategy has been measured for this GPU. Supported: {known}. "
                f"The reconstruction rules are architecture-specific and must be re-measured, not "
                f"extrapolated -- see ArchProfile.")
        prof = measured[-1][1]
        want = ", ".join(f"{v[0]}.{v[1]}.x" for v, _ in measured)
        exact = f" (exactly {prof.measured_cublaslt})" if prof.measured_cublaslt else ""
        warnings.warn(
            f"cuBLASLt {major}.{minor}.{patch} on {prof.name} was not measured: the rules come from "
            f"{want}{exact}. Results should still be bit-identical to cuBLAS, but a version bump can "
            f"change which kernel a shape lands on. Re-measure and add a "
            f"(({cap[0]}, {cap[1]}), ({major}, {minor})) entry to `arch._REGISTRY` to silence this.", RuntimeWarning,
            stacklevel=2)
    elif not prof.measured:
        known = ", ".join(sorted({p.name for p in _REGISTRY.values() if p.measured}))
        raise CublasUnsupportedPlatform(
            f"unsupported on sm_{cap[0]}{cap[1]} ({torch.cuda.get_device_name()}): the profile for "
            f"this GPU is a placeholder, not a measurement. Supported: {known}.")

    _PLATFORM_CACHE[key] = prof
    return prof
