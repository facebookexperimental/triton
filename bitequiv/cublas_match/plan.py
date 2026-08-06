"""From a cuBLAS heuristic config to a `CublasGemmPlan`: which kernel, and with what parameters.

`static_plan` is a pure function.  The heuristic config is an *input* -- nothing here queries
cuBLAS, executes a GEMM, or compares bytes -- so the whole planner can be exercised on any
machine, for any architecture, by handing it configs (see `tests/gen_cublas_plan_fixtures.py`).

The recipe comes from five of the nine config fields:

    `ALGO_ID`          -> which kernel family cuBLAS will run, and so which planner
    `STAGES_ID`        -> the threadblock block_k and the k per dot   (tensor-core families)
    `CUSTOM_OPTION`    -> the k split and the lane tree               (CUDA-core families)
    `REDUCTION_SCHEME` -> which of the split-K merge schemes
    `SPLITK_NUM`       -> the partition

what each of those means is measured, and lives in an `ArchProfile`; this module holds only the
logic that reads one.  A config naming something the profile does not carry declines rather than
guessing, which is the safety net for a future cuBLAS or an unfamiliar GPU: coverage is lost
instead of wrong bits being returned.
"""
from __future__ import annotations

import dataclasses

from .ltapi import _CFG_CUSTOM, _CFG_ID, _CFG_REDUCTION, _CFG_SPLITK, _CFG_STAGES


@dataclasses.dataclass(frozen=True)
class CublasGemmPlan:
    """One shape's answer: the kernel to run and the parameters that fix its arithmetic.

    Only the fields a given `mode` needs are set; the rest stay None.  `simt` and `gemv` are
    left as the tuples the `ArchProfile` recipe tables store, rather than spread into named
    fields, so that the meaning of those tuples is defined in exactly one place -- the comments
    on `gemmsn_recipe`, `gemv_recipe` and `gemv_cslice_recipe`.

    Deliberately absent: the accumulate dtype, the reduce dtype and the reduction order.  They
    are the same for every plan (fp32, fp32, forward) and are properties of the kernels in
    `kernels.py`, so a copy here could only ever drift out of agreement with them.  They are
    written down in the README instead.
    """

    mode: str  # plain / k_per_dot / split / split_blocks /
    # splitk_groups / gemmsn / gemv13 / gemv_cslice / gemv14
    algo_id: int
    raw_config: tuple = ()  # the nine heuristic fields, verbatim, for debugging

    # -- tensor-core families -------------------------------------------------------------
    k_chunk: int | None = None  # split-K slice length; the last slice carries the remainder
    block_k: int | None = None  # the threadblock's k step
    k_per_dot: int | None = None  # real k elements per `tl.dot`, i.e. per accumulator update
    leading_group_k: int | None = None  # size of the first group (CUTLASS puts the residue first)
    merge_scheme: int | None = None  # which split-K combine, from REDUCTION_SCHEME

    # -- CUDA-core families ---------------------------------------------------------------
    simt: tuple | None = None  # gemmsn:      (S, B)
    gemv: tuple | None = None  # gemv13:      ((V, W, CC, count-down), SPLITK_NUM)
    # gemv_cslice: (W, C)
    # gemv14:      (n blocks, k per load)


def _chunk_from_ns(K, ns, G):
    total = (K + G - 1) // G
    return ((total + ns - 1) // ns) * G


def _nsplit_of(config):
    """SPLITK_NUM, normalised: cuBLAS leaves it unset (None) on the families that never split."""
    ns = config[_CFG_SPLITK]
    return ns if isinstance(ns, int) and ns >= 1 else 1


def _plan_tensor_core(prof, family, M, N, K, kind, config):
    """The nvjet and CUTLASS families: `tl.dot` with a threadblock block_k from STAGES_ID."""
    nsplit = _nsplit_of(config)
    reduction, stages = config[_CFG_REDUCTION], config[_CFG_STAGES]
    recipe = dict(prof.stages_recipe).get((family, stages))
    if recipe is None:
        return None, f"unsupported STAGES_ID {stages} for {family}"
    block_k, k_per_dot = recipe
    plan = lambda **kw: CublasGemmPlan(algo_id=config[_CFG_ID], raw_config=config, **kw)  # noqa: E731

    if family == "nvjet":  # cuBLAS's own kernels: block_k-grained split-K
        two_level = (family, stages) in prof.block_level_keys
        if nsplit <= 1:
            # One accumulator over the whole K, unless this key's kernel closes an accumulator
            # every block_k -- then the whole K is the single slice and only the block level is
            # left.
            return (plan(mode="split_blocks", k_chunk=K, block_k=block_k) if two_level else plan(mode="plain")), "ok"
        chunk = _chunk_from_ns(K, nsplit, block_k)
        if not block_k <= chunk < K:
            return None, "chunk out of range"
        return (plan(mode="split_blocks", k_chunk=chunk, block_k=block_k) if two_level else plan(
            mode="split", k_chunk=chunk)), "ok"
    if kind == "fp8":  # cuBLAS refuses fp8 unless every dim is a multiple of 16, so it never
        return None, "unsupported fp8 on a CUTLASS kernel"  # leaves nvjet; if it did, unmeasured
    if nsplit <= 1:
        return plan(mode="k_per_dot", k_per_dot=k_per_dot, leading_group_k=K % block_k), "ok"
    cmode = dict(prof.reduction_to_cmode).get(reduction)
    if cmode is None:
        return None, f"unsupported REDUCTION_SCHEME {reduction}"
    # CUTLASS's split-K partition grain: `params_universal_base.h:52-59` takes kAlignK = 64 when
    # K divides evenly by it and falls back to 128 bits / 16 bits = 8 otherwise.
    big, small = max(prof.splitk_grains), min(prof.splitk_grains)
    grain = big if K % big == 0 else small
    chunk = _chunk_from_ns(K, nsplit, grain)
    if not grain <= chunk <= K:
        return None, "chunk out of range"
    return plan(mode="splitk_groups", k_chunk=chunk, k_per_dot=k_per_dot, block_k=block_k, merge_scheme=cmode), "ok"


def _plan_gemmsn(prof, family, M, N, K, kind, config):
    """ALGO_ID 11 (`gemmSN_NN_kernel`) and 16 (`magma_sgemmEx_kernel`): (S, B) from
    (ALGO_ID, CUSTOM_OPTION)."""
    algo, custom = config[_CFG_ID], config[_CFG_CUSTOM]
    if kind == "fp8":
        return None, "unsupported fp8 on a CUDA-core kernel"
    nsplit = _nsplit_of(config)
    if nsplit != 1:
        # Structural: `_triton_gemmsn` has no split-K path, so a split count it cannot honour
        # has to decline. Never observed either -- all 20,057 hits carry SPLITK_NUM 1.
        return None, f"unsupported SPLITK_NUM {nsplit} for ALGO_ID {algo}"
    recipe = dict(prof.gemmsn_recipe).get((algo, custom))
    if recipe is None:
        return None, f"unsupported CUSTOM_OPTION {custom} for ALGO_ID {algo}"
    return CublasGemmPlan(mode="gemmsn", algo_id=algo, raw_config=config, simt=recipe), "ok"


def _plan_gemv(prof, family, M, N, K, kind, config):
    """ALGO_ID 13 (`gemv2T/gemv2N_kernel`) and 14 (`dot_kernel` + `reduce_1Block_kernel`)."""
    algo, custom = config[_CFG_ID], config[_CFG_CUSTOM]
    nsplit = _nsplit_of(config)
    if kind == "fp8":
        return None, "unsupported fp8 on a CUDA-core kernel"
    if M != 1 and N != 1:
        # Structural, not caution: `_gemv_axis` collapses the problem to one output per row or
        # per column, so these kernels cannot express a shape with both dims above 1 at all.
        # Supporting one would mean a new kernel, not a new table row. (It has also never been
        # observed -- 134 hits of these two algos across 12,000 shapes were all vectors.)
        return None, f"unsupported ALGO_ID {algo} on a non-gemv shape (M {M}, N {N})"
    plan = lambda **kw: CublasGemmPlan(algo_id=algo, raw_config=config, **kw)  # noqa: E731
    if algo == 14:
        # k is contiguous in the matrix that carries it when N == 1, and the kernel then loads
        # it two at a time; the other orientation is strided, so one at a time.
        return plan(mode="gemv14", gemv=(nsplit, 2 if N == 1 else 1)), "ok"
    cslice = dict(prof.gemv_cslice_recipe).get((algo, custom))
    if cslice is not None:
        if nsplit != 1 or M != 1:
            # Structural: `_triton_gemv_cslice` has no split-K path and walks the N axis only.
            return None, (f"unsupported CUSTOM_OPTION {custom} for ALGO_ID {algo} at SPLITK_NUM "
                          f"{nsplit} with M {M} (this kernel handles SPLITK_NUM 1, M == 1 only)")
        return plan(mode="gemv_cslice", gemv=cslice), "ok"
    recipe = dict(prof.gemv_recipe).get((algo, custom))
    if recipe is None:
        return None, f"unsupported CUSTOM_OPTION {custom} for ALGO_ID {algo}"
    max_elems = dict(prof.gemv_max_elems).get((algo, custom), 0)
    # A third kind of decline, and the only one of the three that no amount of work here can
    # remove: the config genuinely does not determine the kernel. cuBLAS picks this row's lane
    # width from occupancy, so two shapes with identical 9-field configs run different orders.
    if max_elems == "occupancy":
        max_elems = prof.sm_count * prof.threads_per_sm // recipe[1]
    if max_elems and max(M, N) > max_elems:
        return None, (f"unsupported CUSTOM_OPTION {custom} for ALGO_ID {algo} on a "
                      f"{max(M, N)}-element gemv (measured up to {max_elems})")
    return plan(mode="gemv13", gemv=(recipe, nsplit)), "ok"


# Which planner each kernel family uses. `algo_family` maps ALGO_ID -> family, so adding an
# ALGO_ID to an existing family needs no code, only a table row.
_FAMILY_PLAN = {"nvjet": _plan_tensor_core, "cutlass": _plan_tensor_core, "gemmsn": _plan_gemmsn, "gemv": _plan_gemv}


def static_plan(prof, M, N, K, kind, config):
    """The reconstruction, derived from the cuBLAS heuristic config alone.

    Nothing is executed and nothing is byte-compared: `ALGO_ID` gives the kernel family and
    picks the planner, and inside it `STAGES_ID`, `CUSTOM_OPTION`, `REDUCTION_SCHEME` and
    `SPLITK_NUM` give the recipe. Returns (plan, reason); plan is None when the config falls
    outside what has been measured on this platform, which is a decline, not a guess.

    Measured over 100,692 shapes on sm_100 across six corners (fp16 random / aligned /
    non-aligned / skinny+deep, fp8 random / skinny+deep): a plan is derived for 99.91% of them
    and 99.891% of those are byte-identical to cuBLAS. The residual is 110 shapes, all nvjet
    split-K with very deep K, where the derived partition is wrong AND no partition at all
    reproduces cuBLAS -- a brute-force sweep of every chunk found nothing. Nothing in the
    config, and nothing in the launched kernel name either, separates them from the 25,808
    nvjet split-K shapes that do match, so they are a known residual rather than a case this
    function could decline. See the follow-up note in the README."""
    if config is None:
        return None, "no-heuristic"
    family = dict(prof.algo_family).get(config[_CFG_ID])
    if family is None:
        return None, f"unsupported ALGO_ID {config[_CFG_ID]}"
    return _FAMILY_PLAN[family](prof, family, M, N, K, kind, config)
