"""The cublas-match restructuring safety net.

Two layers, with very different reach:

  * `test_plan_replay` -- replays `fixtures/cublas_plan_*.json` through `static_plan`.  No GPU,
    no cuBLAS, no GEMM: the planner is a pure function of (shape, dtype, heuristic config), so
    every architecture and every measured cuBLASLt version is covered from any machine.  The
    fixtures were generated from the code as it stood before the restructuring, so this is the
    statement that the restructuring changed nothing.

  * `test_end_to_end` -- actually runs the reconstruction against cuBLAS on this box and
    compares bytes.  Only reaches the plan modes the local GPU and cuBLASLt happen to route to,
    which is a small subset; it is here to catch a mistake the pure-function layer cannot see
    (a launcher wired to the wrong plan field, say), not to prove coverage.

Regenerate the fixtures with `gen_cublas_plan_fixtures.py`.  They should only ever change when
the planner is deliberately changed, and such a change should be a diff of its own.
"""
from __future__ import annotations

import glob
import json
import os
import warnings

import pytest
import torch

from bitequiv.cublas_match import cublas_equivalent_gemm, cublas_matmul
from bitequiv.cublas_match.arch import _REGISTRY, platform
from bitequiv.cublas_match.ltapi import _cublas_direct
from bitequiv.cublas_match.plan import static_plan

FIXTURES = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "fixtures", "cublas_plan_*.json")))


def _jsonable(x):
    return [_jsonable(v) for v in x] if isinstance(x, (tuple, list)) else x


def plan_to_fixture_dict(plan):
    """A plan in the shape the fixture stores: mode, algo_id, and the fields this mode sets.

    `gen_cublas_plan_fixtures.py` imports this, so the record written and the record compared
    are produced by the same code and cannot drift apart.  `raw_config` is deliberately not in
    here: it is the case's own config and is asserted separately.
    """
    out = {"mode": plan.mode, "algo_id": plan.algo_id}
    for field in ("k_chunk", "block_k", "k_per_dot", "leading_group_k", "merge_scheme", "simt", "gemv"):
        value = getattr(plan, field)
        if value is not None:
            out[field] = _jsonable(value)
    return out


@pytest.mark.parametrize("path", FIXTURES, ids=[os.path.basename(p) for p in FIXTURES])
def test_plan_replay(path):
    """Every recipe-table row and every decline branch, for one (architecture, cuBLASLt)."""
    data = json.load(open(path))
    prof = _REGISTRY[(tuple(data["capability"]), tuple(data["cublaslt"]))]
    assert prof.name == data["profile"]

    for case in data["cases"]:
        config = tuple(case["config"]) if case["config"] is not None else None
        plan, reason = static_plan(prof, case["M"], case["N"], case["K"], case["kind"], config)
        where = f"{data['profile']} {case['label']} {case['M']}x{case['N']}x{case['K']} {case['kind']}"

        assert reason == case["reason"], where
        if case["plan"] is None:
            assert plan is None, f"{where}: expected a decline, got {plan}"
            continue
        assert plan is not None, f"{where}: expected {case['plan']}, got a decline ({reason})"
        assert plan_to_fixture_dict(plan) == case["plan"], where
        assert plan.raw_config == config, where


def test_plan_replay_covers_every_mode():
    """The fixtures are meant to be table-complete; fail loudly if a mode stops being exercised."""
    seen = set()
    for path in FIXTURES:
        for case in json.load(open(path))["cases"]:
            if case["plan"] is not None:
                seen.add(case["plan"]["mode"])
    assert seen == {
        "plain", "k_per_dot", "split", "split_blocks", "splitk_groups", "gemmsn", "gemv13", "gemv_cslice", "gemv14"
    }, sorted(seen)


def test_registry_profiles_are_standalone():
    """Each arch file states its own tables. Nothing empty, nothing accidentally shared."""
    for (cap, version), prof in _REGISTRY.items():
        assert prof.measured, (cap, version)
        for table in ("algo_family", "stages_recipe", "splitk_grains", "reduction_to_cmode", "gemmsn_recipe",
                      "gemv_recipe", "gemv_cslice_recipe"):
            assert getattr(prof, table), (cap, version, table)
        assert prof.sm_count and prof.threads_per_sm, (cap, version)


# --------------------------------------------------------------------------- #
# End to end, on whatever this box is
# --------------------------------------------------------------------------- #
# One shape per plan mode we can actually reach here. Which modes those are depends on the GPU
# and the cuBLASLt version -- the heuristic on a GB300 with 13.2 never returns the gemv algos
# for any of these, for instance -- so a shape whose mode is not the expected one is SKIPPED
# rather than failed. What is not negotiable is that whatever does run matches cuBLAS byte for
# byte.
_E2E = [
    ("plain", "fp16", 4096, 4096, 4096),
    ("plain", "bf16", 2048, 2048, 512),
    ("plain", "fp8", 4096, 4096, 4096),
    ("split", "fp16", 128, 128, 16384),
    ("split", "fp8", 64, 64, 65536),
    ("k_per_dot", "fp16", 2, 4096, 32),
    ("splitk_groups", "fp16", 4096, 4095, 4096),
    ("splitk_groups", "fp16", 1023, 1025, 65536),
    ("gemmsn", "fp16", 2, 4096, 45),
]


def _operands(kind, M, N, K, seed):
    torch.manual_seed(seed)
    if kind == "fp8":
        a = (torch.randn(M, K, device="cuda") / 4).to(torch.float8_e4m3fn)
        b = (torch.randn(N, K, device="cuda") / 4).to(torch.float8_e4m3fn).t()  # [K,N] col-major
        return a, b, torch.float16
    dt = torch.bfloat16 if kind == "bf16" else torch.float16
    return (torch.randn(M, K, device="cuda", dtype=dt), torch.randn(K, N, device="cuda", dtype=dt), dt)


@pytest.mark.parametrize("mode,kind,M,N,K", _E2E, ids=[f"{m}-{k}-{a}x{b}x{c}" for m, k, a, b, c in _E2E])
def test_end_to_end(mode, kind, M, N, K):
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # unmeasured cuBLASLt version
        prof = platform()
        a, b, out_dtype = _operands(kind, M, N, K, seed=0)
        config = _cublas_direct(a, b, kind, out_dtype, execute=False)[1]
        plan, reason = static_plan(prof, M, N, K, kind, config)
        if plan is None or plan.mode != mode:
            got = plan.mode if plan else f"decline ({reason})"
            pytest.skip(f"this GPU/cuBLASLt routes {M}x{N}x{K} {kind} to {got}, not {mode}")

        for seed in range(4):
            a, b, out_dtype = _operands(kind, M, N, K, seed)
            mine = cublas_equivalent_gemm(a, b, out_dtype=out_dtype)
            theirs = cublas_matmul(a, b, out_dtype=out_dtype)
            assert torch.equal(mine.view(torch.uint8), theirs.view(torch.uint8)), \
                f"{mode} {kind} {M}x{N}x{K} seed {seed}: not byte-identical to cuBLAS"
