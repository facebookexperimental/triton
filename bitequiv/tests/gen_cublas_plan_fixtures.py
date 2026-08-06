"""Generate the cuBLAS-match plan fixtures.

WHAT THIS IS FOR.  `static_plan(M, N, K, kind, config)` is a pure function: the heuristic
config is an *input*, not something it queries.  So the whole planner can be exercised with
no GPU and no cuBLAS at all, by handing it configs we construct ourselves.

We do not sample shapes and hope to hit every table row.  We walk the tables directly --
every `algo_family` entry, every `stages_recipe` key, every `reduction_to_cmode` value, every
`gemmsn_recipe` / `gemv_recipe` / `gemv_cslice_recipe` row -- and build one config per row,
plus a case for each decline branch.  Coverage is therefore complete by construction rather
than by luck, and it holds for all four profiles on any machine, including the three whose
hardware is not in front of us.

The values in those tables were measured on real hardware; replaying them here is not a new
experiment, it is a snapshot of what the planner does with conclusions we already have.

USAGE

    ./.venv/bin/python bitequiv/tests/gen_cublas_plan_fixtures.py

writes `bitequiv/tests/fixtures/cublas_plan_<arch>_<cublaslt>.json`.

The fixtures are generated ONCE from the pre-restructuring code and committed.  The replay
test in `test_cublas_match.py` then has to reproduce them field for field.  That is the
evidence that the restructuring changed no behaviour.
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

from bitequiv.cublas_match import arch  # noqa: E402
from bitequiv.cublas_match.plan import static_plan  # noqa: E402

# The plan -> fixture-record mapping lives in the test, so the generator and the replay cannot
# drift apart: there is one definition, used from both ends.
from test_cublas_match import plan_to_fixture_dict  # noqa: E402

FIXTURE_DIR = os.path.join(_HERE, "fixtures")

# The config is a 9-tuple of algo-config attributes; only these five are read.
_ID, _SPLITK, _REDUCTION, _CUSTOM, _STAGES = 0, 2, 3, 5, 6


def cfg(algo, *, stages=0, custom=0, nsplit=1, reduction=0):
    """A heuristic config tuple, in the same shape `_cfg` collects it."""
    c = [None] * 9
    c[_ID], c[_STAGES], c[_CUSTOM] = algo, stages, custom
    c[_SPLITK], c[_REDUCTION] = nsplit, reduction
    return tuple(c)


# --------------------------------------------------------------------------- #
# Case enumeration, one group per table / per decline branch
# --------------------------------------------------------------------------- #
_KINDS = ("fp16", "bf16", "fp8")
# K values chosen to exercise the chunk arithmetic rather than to be realistic: an exact
# multiple of both grains, a value that is not, a ragged one, and a very deep one.
_KS = (4096, 8192, 3000, 21128)


def _algos_of(prof, family):
    return [a for a, f in prof.algo_family if f == family]


def cases(prof):
    """Every (M, N, K, kind, config) worth planning for this profile, with a label saying
    which table row or which decline branch it is there to cover."""
    out = []

    def add(label, M, N, K, kind, config):
        out.append({"label": label, "M": M, "N": N, "K": K, "kind": kind, "config": config})

    known_algos = [a for a, _ in prof.algo_family]

    # --- algo_family: one minimal case per ALGO_ID, so a family remap is caught ----------
    for algo, family in prof.algo_family:
        if family in ("nvjet", "cutlass"):
            stages = next(s for (f, s), _ in prof.stages_recipe if f == family)
            add(f"algo_family/{algo}", 4096, 4096, 4096, "fp16", cfg(algo, stages=stages))
        elif family == "gemmsn":
            custom = next(c for (a, c), _ in prof.gemmsn_recipe if a == algo)
            add(f"algo_family/{algo}", 8, 4096, 512, "fp16", cfg(algo, custom=custom))
        elif family == "gemv":
            custom = next((c for (a, c), _ in prof.gemv_recipe if a == algo), 0)
            add(f"algo_family/{algo}", 1, 4096, 4096, "fp16", cfg(algo, custom=custom))

    # --- stages_recipe: every key, split and not, at several K and dtypes ----------------
    for (family, stages), _recipe in prof.stages_recipe:
        algo = _algos_of(prof, family)[0]
        for kind in _KINDS:
            for K in _KS:
                add(f"stages/{family}/{stages}/ns1", 4096, 4096, K, kind, cfg(algo, stages=stages, nsplit=1))
                for ns in (2, 8, 592):
                    add(f"stages/{family}/{stages}/ns{ns}", 4096, 4096, K, kind, cfg(algo, stages=stages, nsplit=ns))

    # --- reduction_to_cmode: every scheme, on CUTLASS split-K (the only reader) ----------
    cutlass_algo = _algos_of(prof, "cutlass")[0]
    for reduction, _cmode in prof.reduction_to_cmode:
        for (family, stages), _r in prof.stages_recipe:
            if family != "cutlass":
                continue
            for K in _KS:
                add(f"reduction/{reduction}/{stages}", 4096, 4096, K, "fp16",
                    cfg(cutlass_algo, stages=stages, nsplit=4, reduction=reduction))

    # --- gemmsn_recipe: every row -------------------------------------------------------
    for (algo, custom), _r in prof.gemmsn_recipe:
        for K in (2, 45, 512, 1200):
            add(f"gemmsn/{algo}/{custom}", 8, 4096, K, "fp16", cfg(algo, custom=custom))

    # --- gemv_recipe: every row, both orientations --------------------------------------
    for (algo, custom), _r in prof.gemv_recipe:
        for M, N in ((1, 4096), (4096, 1)):
            for K in (64, 4096, 262144):
                add(f"gemv/{algo}/{custom}", M, N, K, "fp16", cfg(algo, custom=custom))

    # --- gemv_cslice_recipe: every row --------------------------------------------------
    for (algo, custom), _r in prof.gemv_cslice_recipe:
        for K in (64, 4096, 262144):
            add(f"gemv_cslice/{algo}/{custom}", 1, 4096, K, "fp16", cfg(algo, custom=custom))
        # its two structural declines: split-K, and the N == 1 orientation
        add(f"gemv_cslice/{algo}/{custom}/decline-nsplit", 1, 4096, 4096, "fp16", cfg(algo, custom=custom, nsplit=4))
        add(f"gemv_cslice/{algo}/{custom}/decline-orientation", 4096, 1, 4096, "fp16", cfg(algo, custom=custom))

    # --- gemv_max_elems: both sides of every cap ----------------------------------------
    for (algo, custom), cap in prof.gemv_max_elems:
        n = prof.sm_count * prof.threads_per_sm
        if cap == "occupancy":
            recipe = dict(prof.gemv_recipe).get((algo, custom))
            cap = n // recipe[1] if recipe else n
        for elems in (cap, cap + 1):
            add(f"gemv_max_elems/{algo}/{custom}", 1, elems, 4096, "fp16", cfg(algo, custom=custom))

    # --- ALGO_ID 14: no table, parameters come from SPLITK_NUM and the orientation -------
    for algo in _algos_of(prof, "gemv"):
        if algo != 14:
            continue
        for M, N in ((1, 4096), (4096, 1)):
            for ns in (1, 4, 64):
                add(f"gemv14/{ns}", M, N, 262144, "fp16", cfg(algo, nsplit=ns))

    # --- declines -----------------------------------------------------------------------
    unknown_algo = max(known_algos) + 100
    add("decline/algo_id", 4096, 4096, 4096, "fp16", cfg(unknown_algo))
    add("decline/no-heuristic", 4096, 4096, 4096, "fp16", None)

    for family in ("nvjet", "cutlass"):
        algo = _algos_of(prof, family)[0]
        known_stages = {s for (f, s), _ in prof.stages_recipe if f == family}
        add(f"decline/stages/{family}", 4096, 4096, 4096, "fp16", cfg(algo, stages=max(known_stages) + 100))

    add("decline/fp8-on-cutlass", 4096, 4096, 4096, "fp8",
        cfg(cutlass_algo, stages=next(s for (f, s), _ in prof.stages_recipe if f == "cutlass")))

    for family in ("gemmsn", "gemv"):
        for algo in _algos_of(prof, family):
            known_custom = {
                c
                for (a, c), _ in (prof.gemmsn_recipe if family == "gemmsn" else prof.gemv_recipe)
                if a == algo
            }
            if known_custom:
                add(f"decline/custom/{algo}", 1 if family == "gemv" else 8, 4096, 512, "fp16",
                    cfg(algo, custom=max(known_custom) + 100))
            add(f"decline/fp8-on-cuda-core/{algo}", 1 if family == "gemv" else 8, 4096, 512, "fp8",
                cfg(algo, custom=next(iter(known_custom), 0)))

    for algo in _algos_of(prof, "gemv"):
        custom = next((c for (a, c), _ in prof.gemv_recipe if a == algo), 0)
        add(f"decline/gemv-non-vector/{algo}", 4096, 4096, 4096, "fp16", cfg(algo, custom=custom))

    for algo in _algos_of(prof, "gemmsn"):
        custom = next(c for (a, c), _ in prof.gemmsn_recipe if a == algo)
        add(f"decline/gemmsn-splitk/{algo}", 8, 4096, 512, "fp16", cfg(algo, custom=custom, nsplit=4))

    # chunk-out-of-range: a split count far larger than K has k-grains for
    for family in ("nvjet", "cutlass"):
        algo = _algos_of(prof, family)[0]
        stages = next(s for (f, s), _ in prof.stages_recipe if f == family)
        for K in (64, 128, 256):
            add(f"decline/chunk-range/{family}", 4096, 4096, K, "fp16", cfg(algo, stages=stages, nsplit=512,
                                                                            reduction=2))

    return out


def generate(prof, cap, version):
    # No `measured_cublaslt` in the header: which exact patch release a profile records is a
    # label on the profile, not planner output, and the restructuring splits one combined label
    # into two. Only the plans belong in here.
    recs = []
    for c in cases(prof):
        plan, reason = static_plan(prof, c["M"], c["N"], c["K"], c["kind"], c["config"])
        recs.append({
            "label": c["label"],
            "M": c["M"],
            "N": c["N"],
            "K": c["K"],
            "kind": c["kind"],
            "config": list(c["config"]) if c["config"] is not None else None,
            "plan": plan_to_fixture_dict(plan) if plan is not None else None,
            "reason": reason,
        })
    return {
        "profile": prof.name,
        "capability": list(cap),
        "cublaslt": list(version),
        "generated_by": "bitequiv/tests/gen_cublas_plan_fixtures.py",
        "cases": recs,
    }


def _dump(path, data):
    """One case per line.  `json.dump(indent=...)` would put every field on its own line and
    quadruple the file; a single line would make any diff unreadable.  One line per case keeps
    both the size and the reviewability."""
    head = {k: v for k, v in sorted(data.items()) if k != "cases"}
    with open(path, "w") as fh:
        fh.write("{\n")
        for k, v in head.items():
            fh.write(f" {json.dumps(k)}: {json.dumps(v)},\n")
        fh.write(' "cases": [\n')
        for i, rec in enumerate(data["cases"]):
            tail = "" if i == len(data["cases"]) - 1 else ","
            fh.write("  " + json.dumps(rec, sort_keys=True, separators=(",", ":")) + tail + "\n")
        fh.write(" ]\n}\n")


def main():
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    written = []
    for (cap, version), prof in sorted(arch._REGISTRY.items()):
        if prof.measured:
            data = generate(prof, cap, version)
            name = f"cublas_plan_sm{cap[0]}{cap[1]}_lt{version[0]}_{version[1]}.json"
            _dump(os.path.join(FIXTURE_DIR, name), data)
            n_plan = sum(1 for r in data["cases"] if r["plan"] is not None)
            written.append((name, len(data["cases"]), n_plan))

    for name, total, planned in written:
        print(f"  {name:44s} {total:6d} cases, {planned:6d} planned, {total - planned:5d} declined")


if __name__ == "__main__":
    main()
